"""Replay every candidate and independently verify both selection levels."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from common import ROOT, sha, save_json, log, verify
import run_routers as run


def one(path):
    path = Path(path)
    for name, value in json.loads((path/"complete.json").read_text())["files"].items():
        assert sha(path/name) == value
    checkpoint = path/"audit_pass.json"
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous["audit_code_sha256"] == sha(__file__) and previous["condition_sha256"] == sha(path/"complete.json"):
            return previous["models"], previous["rows"]
    rows = pd.read_csv(path/"metrics.csv")
    grid = pd.read_csv(path/"candidates.csv")
    pair = run.PAIRS[int(rows.pair_id.iloc[0])]
    data = run.DATA
    X, W, S = data[f"X__{pair['weak']}"], data[f"correct__{pair['weak']}"], data[f"correct__{pair['strong']}"]
    models = joblib.load(path/"models.joblib")
    with np.load(path/"scores.npz") as archive:
        # NPZ access decompresses the whole array; load once rather than for each head.
        z = {name: archive[name] for name in archive.files}
        test, tune, select = z["test_ids"], z["tune_ids"], z["select_ids"]
        assert not set(test)&(set(tune)|set(select)) and not set(tune)&set(select)
        np.testing.assert_array_equal(z["W"], W[test])
        np.testing.assert_array_equal(z["S"], S[test])
        np.testing.assert_array_equal(z["row_id"], data["row_id"][test])
        names = z["names"].tolist()
        for key, (scaler, model) in models.items():
            idx = names.index(key)
            regime = key.split("__")[0].split("_")[0]
            fit = z[f"fit_ids__{regime}"]
            assert (data["split"][fit] == "router_fit").all()
            np.testing.assert_allclose(scaler.mean_, X[fit].mean(axis=0), atol=1e-12)
            for ids, field in [(test,"scores"), (tune,"tune_scores"), (select,"select_scores")]:
                np.testing.assert_allclose(model.predict(scaler.transform(X[ids])), z[field][idx], atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(z["scores"][0], 1-X[test,0])
        tie_tune = np.random.default_rng(2026111400+pair["pair_id"]).random(len(tune))
        tie_select = np.random.default_rng(2026111400+pair["pair_id"]).random(len(select))
        # Verify grid validation metrics from independently reloaded predictions.
        mse = ((z["tune_scores"][1:]-(S[tune].astype(int)-W[tune]))**2).mean(axis=1)
        np.testing.assert_allclose(grid.tune_mse.to_numpy(), mse, atol=1e-12)
        tune_order = np.lexsort((np.broadcast_to(tie_tune, z["tune_scores"][1:].shape), -z["tune_scores"][1:]), axis=1)
        tune_D = S[tune].astype(int)-W[tune]
        for fraction in run.FRACTIONS:
            expected_accuracy = W[tune].mean()+tune_D[tune_order[:, :int(fraction*len(tune))]].sum(axis=1)/len(tune)
            np.testing.assert_allclose(grid[f"tune_accuracy_{fraction}"], expected_accuracy, atol=1e-12)
        expected_rows, cached = [], {}
        for row in rows.itertuples():
            if row.method in run.METHODS:
                options = grid[(grid.arm == row.arm) & (grid.method == row.method)]
                column, ascending = ("tune_mse",True) if row.criterion == "mse" else (f"tune_accuracy_{row.fraction}", False)
                picked = options.sort_values([column,"strength"], ascending=[ascending,False]).iloc[0]
                assert row.score_index == picked.score_index and row.strength == picked.strength
            key = row.score_index, row.evaluation, row.fraction
            if key not in cached:
                mask = z["clean"] if row.evaluation == "no_overlap" else np.ones(len(test), bool)
                score, wt, st = z["scores"][row.score_index][mask], W[test][mask], S[test][mask]
                D = st.astype(int)-wt
                k = int(row.fraction*len(D))
                selected = np.zeros(len(D), dtype=int)
                selected[np.lexsort((z["tie"][mask], -score))[:k]] = 1
                r, h = int(selected@(D==1)), int(selected@(D==-1))
                nr, nh = int((D==1).sum()), int((D==-1).sum())
                cached[key] = {"accuracy": np.mean(wt+selected*D), "gain_mse": np.mean((score-D)**2),
                    "rescues": r, "harms": h, "calls": k, "rescue_precision": r/k,
                    "rescue_recall": r/nr if nr else np.nan, "rescue_f1": 2*r/(k+nr),
                    "harm_rate": h/k, "harm_recall": h/nh if nh else np.nan,
                    "harm_avoidance": 1-h/nh if nh else np.nan}
            expected_rows.append(cached[key])
        expected = pd.DataFrame(expected_rows)
        np.testing.assert_allclose(rows[expected.columns].to_numpy(), expected.to_numpy(), atol=1e-11, rtol=1e-11, equal_nan=True)
        for row in pd.read_csv(path/"selection.csv").itertuples():
            candidates = rows[(rows.arm == row.arm) & (rows.criterion == row.criterion) & (rows.fraction == row.fraction) & rows.method.isin(run.METHODS)].drop_duplicates("method").set_index("method")
            values = {"confidence": run.metrics(z["select_scores"][0], W[select], S[select], tie_select, row.fraction)["accuracy"]}
            for m in run.METHODS:
                index = int(candidates.loc[m,"score_index"])
                values[m] = run.metrics(z["select_scores"][index], W[select], S[select], tie_select, row.fraction)["accuracy"]
            assert max(run.PREFERENCE, key=lambda m: values[m]) == row.method
            expected_index = 0 if row.method == "confidence" else int(candidates.loc[row.method,"score_index"])
            assert row.score_index == expected_index
            for evaluation in rows.evaluation.unique():
                part = rows[(rows.fraction == row.fraction) & (rows.evaluation == evaluation)]
                relevant = part[((part.arm == row.arm) & (part.criterion == row.criterion) & part.method.isin(run.METHODS)) | (part.method == "confidence")]
                oracle = relevant.accuracy.max()
                winner = part[(part.arm == row.arm) & (part.criterion == row.criterion) & (part.method == "selector")].iloc[0]
                assert abs(winner.regret-(oracle-winner.accuracy)) < 1e-12
    save_json(checkpoint,{"models":len(models),"rows":len(rows),"audit_code_sha256":sha(__file__),"condition_sha256":sha(path/"complete.json")})
    return len(models), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = ROOT/("smoke" if args.smoke else "results")
    complete = json.loads((output/"complete.json").read_text())
    assert complete["metrics_sha256"] == sha(output/"metrics.csv")
    paths = sorted(p.parent for p in output.glob("p*_s*_n*/complete.json"))
    assert len(paths) == complete["conditions"]
    models, rows = 0,0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=run.initialize) as executor:
        futures = [executor.submit(one,p) for p in paths]
        for i,future in enumerate(as_completed(futures),1):
            a,b = future.result()
            models += a
            rows += b
            if i%40 == 0 or i == len(paths):
                log("Replay checkpoint", conditions=i, models=models, rows=rows)
    save_json(output/"audit.json", {"status": "passed", "models_replayed": models, "metrics_checked": rows,
        "metrics_sha256": sha(output/"metrics.csv"), "audit_code_sha256": sha(__file__), "preserved_files": verify()})


if __name__ == "__main__":
    main()
