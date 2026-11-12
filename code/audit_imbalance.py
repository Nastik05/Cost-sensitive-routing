"""Independent replay of every fitted router and every saved test metric."""
import os
for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imbalance_study import ROOT, digest, initialize, choose, write_json
import imbalance_study


def audit_condition(path):
    path = Path(path)
    info = json.loads((path/"complete.json").read_text())
    for name, expected in info["files"].items():
        assert digest(path/name) == expected
    rows = pd.read_csv(path/"metrics.csv")
    pair = imbalance_study.MANIFEST["pairs"][int(rows.pair_id.iloc[0])]
    data = imbalance_study.DATA
    X, W, S = data[f"X__{pair['weak']}"], data[f"correct__{pair['weak']}"], data[f"correct__{pair['strong']}"]
    models = joblib.load(path/"models.joblib")
    selections = pd.read_csv(path/"selections.csv")
    with np.load(path/"scores.npz") as z:
        test, tune, select = z["test_ids"], z["tune_ids"], z["select_ids"]
        assert not set(tune)&set(select) and not set(test)&(set(tune)|set(select))
        np.testing.assert_array_equal(z["W"], W[test])
        np.testing.assert_array_equal(z["S"], S[test])
        np.testing.assert_array_equal(z["row_id"], data["row_id"][test])
        names, scores = z["names"].tolist(), z["scores"]
        index = {name: i for i, name in enumerate(names)}
        selected_scores = {}
        for name, (scaler, model) in models.items():
            arm, method = name.split("__")
            ids = z["fit_ids__"+arm.rsplit("_", 1)[0]]
            assert not set(ids)&(set(test)|set(tune)|set(select))
            assert (data["split"][ids] == "router_fit").all()
            np.testing.assert_allclose(model.predict(scaler.transform(X[test])), scores[index[name]], atol=1e-11, rtol=1e-11)
            selected_scores.setdefault(arm, {"confidence": 1-X[select, 0]})[method] = model.predict(scaler.transform(X[select]))
        np.testing.assert_array_equal(scores[index["random"]], np.zeros(len(test)))
        np.testing.assert_array_equal(scores[index["confidence"]], 1-X[test, 0])
        tie_select = np.random.default_rng(831000+pair["pair_id"]).random(len(select))
        for row in selections.itertuples():
            assert choose(selected_scores[row.arm], W[select], S[select], tie_select, row.fraction) == row.selected_method
        cache, expected_rows = {}, []
        oracle_by_key = rows[rows.method != "selector"].groupby(["arm", "evaluation", "fraction"]).accuracy.max().to_dict()
        confidence_by_key = rows[rows.method == "confidence"].set_index(["evaluation", "fraction"]).accuracy.to_dict()
        for row in rows.itertuples():
            method = row.selected_method if row.method == "selector" else row.method
            name = method if method in ["confidence", "random"] else row.arm+"__"+method
            key = name, row.evaluation, row.fraction
            if key not in cache:
                mask = z["clean"] if row.evaluation == "no_overlap" else np.ones(len(test), bool)
                score, wt, st, tie = scores[index[name]][mask], W[test][mask], S[test][mask], z["tie"][mask]
                order = np.lexsort((tie, -score))
                r = np.zeros(len(score), dtype=int)
                r[order[:int(row.fraction*len(score))]] = 1
                d = st.astype(int)-wt
                plus, minus = d == 1, d == -1
                R, H, k = int(r@plus), int(r@minus), int(r.sum())
                nr, nh = int(plus.sum()), int(minus.sum())
                cache[key] = {"accuracy": np.mean(wt+r*d), "gain_mse": np.mean((score-d)**2),
                              "rescues": R, "harms": H, "calls": k, "missed_rescues": nr-R, "avoided_harms": nh-H,
                              "rescue_precision": R/k, "rescue_recall": R/nr if nr else np.nan,
                              "rescue_f1": 2*R/(k+nr), "harm_rate": H/k,
                              "harm_recall": H/nh if nh else np.nan, "harm_avoidance": 1-H/nh if nh else np.nan}
            expected_rows.append(cache[key])
            if row.method == "selector":
                oracle = max(oracle_by_key[row.arm, row.evaluation, row.fraction], confidence_by_key[row.evaluation, row.fraction])
                assert abs(row.oracle_accuracy-oracle) < 1e-11 and abs(row.regret-(oracle-row.accuracy)) < 1e-11
        expected = pd.DataFrame(expected_rows)
        np.testing.assert_allclose(rows[expected.columns].to_numpy(), expected.to_numpy(), atol=1e-11, rtol=1e-11, equal_nan=True)
    return len(models), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/imbalance_all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    complete = json.loads((args.output/"complete.json").read_text())
    assert digest(args.output/"metrics.csv") == complete["metrics_sha256"]
    assert digest(args.output/"fits.csv") == complete["fits_sha256"]
    paths = sorted(p.parent for p in args.output.glob("p*_s*_n*/complete.json"))
    assert len(paths) == complete["conditions"]
    models, rows = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        futures = [executor.submit(audit_condition, path) for path in paths]
        for i, future in enumerate(as_completed(futures), 1):
            m, r = future.result()
            models += m
            rows += r
            if i % 100 == 0 or i == len(paths):
                print(json.dumps({"conditions_audited": i, "models": models, "metric_rows": rows}), flush=True)
    assert models == complete["models"]
    write_json(args.output/"audit.json", {"status": "passed", "conditions": len(paths), "models_replayed": models,
        "metric_rows_checked": rows, "metrics_sha256": digest(args.output/"metrics.csv"), "audit_code_sha256": digest(__file__)})


if __name__ == "__main__":
    main()
