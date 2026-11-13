"""Full router x imbalance grid on all 21 fixed pairs; resumable CPU workers."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from imbalance_estimators import METHODS, STRENGTHS, fit_estimator

ROOT = Path(__file__).resolve().parent
REGIMES = ["natural", "rescue2", "rescue8", "harm2", "harm8", "discordant8", "neutral8"]
CORRECTED = ["rescue8", "harm8", "discordant8", "neutral8"]
ARMS = [r+"_raw" for r in REGIMES]+[r+"_ipw" for r in CORRECTED]
PREFERENCE = ["confidence", "direct", "weak_error", "marginal", "events", "soft_gain", "ternary", "joint", "hurdle"]
FRACTIONS = [.1, .2, .4]
SEEDS = list(range(20))
SIZES = [200, 500, 1000]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(path)


def probabilities(D, regime):
    a = np.ones(len(D))
    if regime.startswith("rescue"):
        a[D == 1] = int(regime[6:])
    elif regime.startswith("harm"):
        a[D == -1] = int(regime[4:])
    elif regime == "discordant8":
        a[D != 0] = 8
    elif regime == "neutral8":
        a[D == 0] = 8
    elif regime != "natural":
        raise ValueError(regime)
    return a/a.sum()


def draw_ids(pool, D, regime, seed, n):
    p = probabilities(D, regime)
    u = np.random.default_rng(811000+seed).random(n)
    selected = np.searchsorted(np.cumsum(p), u, side="right")
    w = 1/(len(pool)*p[selected])
    nonneutral = D != 0
    scale = float(p[nonneutral].sum()/nonneutral.mean()) if nonneutral.any() else 1.
    return pool[selected], w, scale


def metrics(score, W, S, tie, fraction):
    D = S.astype(int)-W.astype(int)
    n, k = len(D), int(len(D)*fraction)
    picked = np.lexsort((tie, -score))[:k]
    R, H = int((D[picked] == 1).sum()), int((D[picked] == -1).sum())
    total_r, total_h = int((D == 1).sum()), int((D == -1).sum())
    return {"accuracy": float(W.mean()+(R-H)/n), "gain_mse": float(np.mean((score-D)**2)),
            "rescues": R, "harms": H, "missed_rescues": total_r-R, "avoided_harms": total_h-H,
            "rescue_precision": R/k if k else np.nan, "rescue_recall": R/total_r if total_r else np.nan,
            "rescue_f1": 2*R/(k+total_r) if k+total_r else np.nan,
            "harm_rate": H/k if k else np.nan, "harm_recall": H/total_h if total_h else np.nan,
            "harm_avoidance": 1-H/total_h if total_h else np.nan,
            "calls": k, "eval_n": n, "total_rescue": total_r, "total_harm": total_h}


def choose(scores, W, S, tie, fraction):
    return max(PREFERENCE, key=lambda m: metrics(scores[m], W, S, tie, fraction)["accuracy"])


DATA, MANIFEST = None, None


def initialize():
    global DATA, MANIFEST
    MANIFEST = json.loads((ROOT / "data/imbalance_manifest.json").read_text())
    with np.load(ROOT / "data/imbalance_cache.npz", allow_pickle=False) as z:
        DATA = {k: z[k] for k in z.files}


def condition(pair_id, seed, n, output, smoke=False):
    pair = MANIFEST["pairs"][pair_id]
    directory = Path(output) / f"p{pair_id:02d}_s{seed:02d}_n{n}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "complete.json"
    if marker.exists():
        for name, value in json.loads(marker.read_text())["files"].items():
            assert digest(directory/name) == value
        return str(directory)
    start = time.perf_counter()
    X, W, S = DATA[f"X__{pair['weak']}"], DATA[f"correct__{pair['weak']}"], DATA[f"correct__{pair['strong']}"]
    pool, dev, test = [np.flatnonzero(DATA["split"] == split) for split in ["router_fit", "policy_dev", "test"]]
    dev_order = np.random.default_rng(821000+seed).permutation(dev)
    tune, select = np.sort(dev_order[:n//2]), np.sort(dev_order[500:1000])
    if smoke:
        # The preflight uses only previously unused dev objects, never test.
        test = np.sort(dev_order[n//2:500])
    assert not (set(pool)&set(tune) or set(tune)&set(select) or set(test)&set(tune) or set(test)&set(select))
    ties = {name: np.random.default_rng(831000+pair_id).random(len(ids))
            for name, ids in [("train", np.arange(n)), ("tune", tune), ("select", select), ("test", test)]}
    Dpool = S[pool].astype(int)-W[pool]
    arrays = {"row_id": DATA["row_id"][test], "test_ids": test, "tune_ids": tune, "select_ids": select,
              "tie": ties["test"], "W": W[test], "S": S[test], "clean": np.ones(len(test), bool) if smoke else DATA["clean_test"][test]}
    test_scores = {"confidence": 1-X[test, 0], "random": np.zeros(len(test))}
    selector_scores, models, fit_rows, eval_rows, selections, tuning_grids = {}, {}, [], [], [], {}
    metric_cache = {}
    base = {"pair_id": pair_id, "seed": seed, "fit_n": n, "panel": pair["panel"]}
    for regime in REGIMES:
        ids, weights, conditional_scale = draw_ids(pool, Dpool, regime, seed, n)
        assert not set(ids)&(set(tune)|set(select)|set(test))
        scaler = StandardScaler().fit(X[ids])
        tx = {split: scaler.transform(X[idx]) for split, idx in [("train", ids), ("tune", tune), ("select", select), ("test", test)]}
        arrays[f"fit_ids__{regime}"] = ids
        arrays[f"weights__{regime}"] = weights
        unique = np.unique(ids)
        for corrected in ([False, True] if regime in CORRECTED else [False]):
            arm = regime+("_ipw" if corrected else "_raw")
            selector_scores[arm] = {"confidence": 1-X[select, 0]}
            for method in METHODS:
                started = time.perf_counter()
                candidates = []
                for strength in STRENGTHS:
                    model = fit_estimator(method, tx["train"], W[ids], S[ids], strength,
                                          weights if corrected else None, conditional_scale)
                    mse = float(np.mean((model.predict(tx["tune"])-(S[tune].astype(int)-W[tune]))**2))
                    candidates.append((mse, strength, model))
                tune_mse, strength, model = min(candidates, key=lambda c: (c[0], -c[1]))
                name = arm+"__"+method
                predicted = {split: model.predict(a) for split, a in tx.items()}
                assert all(np.isfinite(v).all() for v in predicted.values())
                test_scores[name], selector_scores[arm][method] = predicted["test"], predicted["select"]
                models[name] = (scaler, model)
                tuning_grids[name] = [[c[0], c[1]] for c in candidates]
                info = {**base, "arm": arm, "method": method, "strength": strength, "train_mse": float(np.mean((predicted['train']-(S[ids].astype(int)-W[ids]))**2)),
                        "tune_mse": tune_mse, "select_mse": float(np.mean((predicted['select']-(S[select].astype(int)-W[select]))**2)),
                        "train_accuracy": metrics(predicted["train"], W[ids], S[ids], ties["train"], .2)["accuracy"],
                        "tune_accuracy": metrics(predicted["tune"], W[tune], S[tune], ties["tune"], .2)["accuracy"],
                        "select_accuracy": metrics(predicted["select"], W[select], S[select], ties["select"], .2)["accuracy"],
                        "unique_n": len(unique), "fit_rescue": int((S[ids].astype(int)-W[ids] == 1).sum()),
                        "fit_harm": int((S[ids].astype(int)-W[ids] == -1).sum()),
                        "unique_rescue": int((S[unique].astype(int)-W[unique] == 1).sum()),
                        "unique_harm": int((S[unique].astype(int)-W[unique] == -1).sum()),
                        "ess": float(weights.sum()**2/(weights@weights)) if corrected else float(n),
                        "conditional_scale": conditional_scale if corrected else 1.,
                        "parameter_count": int(model.parameter_count), "fit_seconds": time.perf_counter()-started}
                fit_rows.append(info)
                for evaluation in (["dev_smoke"] if smoke else ["official", "no_overlap"]):
                    mask = arrays["clean"] if evaluation == "no_overlap" else np.ones(len(test), bool)
                    for fraction in FRACTIONS:
                        result = metrics(predicted["test"][mask], W[test][mask], S[test][mask], ties["test"][mask], fraction)
                        metric_cache[arm, method, evaluation, fraction] = result
                        eval_rows.append({**base, "arm": arm, "method": method, "evaluation": evaluation, "fraction": fraction, **result})
    for method in ["confidence", "random"]:
        for evaluation in (["dev_smoke"] if smoke else ["official", "no_overlap"]):
            mask = arrays["clean"] if evaluation == "no_overlap" else np.ones(len(test), bool)
            for fraction in FRACTIONS:
                result = metrics(test_scores[method][mask], W[test][mask], S[test][mask], ties["test"][mask], fraction)
                metric_cache["baseline", method, evaluation, fraction] = result
                eval_rows.append({**base, "arm": "baseline", "method": method, "evaluation": evaluation, "fraction": fraction, **result})
    for arm in ARMS:
        for fraction in FRACTIONS:
            chosen = choose(selector_scores[arm], W[select], S[select], ties["select"], fraction)
            name = chosen if chosen == "confidence" else arm+"__"+chosen
            selections.append({**base, "arm": arm, "fraction": fraction, "selected_method": chosen})
            for evaluation in (["dev_smoke"] if smoke else ["official", "no_overlap"]):
                mask = arrays["clean"] if evaluation == "no_overlap" else np.ones(len(test), bool)
                candidate_metrics = {m: metric_cache["baseline" if m == "confidence" else arm, m, evaluation, fraction] for m in PREFERENCE}
                oracle = max(v["accuracy"] for v in candidate_metrics.values())
                eval_rows.append({**base, "arm": arm, "method": "selector", "evaluation": evaluation, "fraction": fraction,
                                  "selected_method": chosen, "oracle_accuracy": oracle, "regret": oracle-candidate_metrics[chosen]["accuracy"], **candidate_metrics[chosen]})
    arrays["names"] = np.array(list(test_scores))
    arrays["scores"] = np.stack(list(test_scores.values()))
    np.savez_compressed(directory / "scores.npz", **arrays)
    joblib.dump(models, directory / "models.joblib", compress=3)
    pd.DataFrame(eval_rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(directory / "fits.csv", index=False)
    pd.DataFrame(selections).to_csv(directory / "selections.csv", index=False)
    write_json(directory / "tuning.json", tuning_grids)
    write_json(marker, {"seconds": time.perf_counter()-start, "models": len(models), "rows": len(eval_rows),
                       "files": {name: digest(directory/name) for name in ["scores.npz", "models.joblib", "metrics.csv", "fits.csv", "selections.csv", "tuning.json"]}})
    return str(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/imbalance_all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/imbalance_manifest.json").read_text())
    assert digest(ROOT / "data/imbalance_cache.npz") == manifest["cache_sha256"]
    config = {"pairs": manifest["pairs"], "methods": METHODS, "regimes": REGIMES, "arms": ARMS,
              "seeds": SEEDS, "sizes": SIZES, "fractions": FRACTIONS, "smoke": args.smoke,
              "cache_sha256": manifest["cache_sha256"],
              "code": {name: digest(ROOT/name) for name in ["imbalance_study.py", "imbalance_estimators.py", "PROTOCOL_IMBALANCE.md"]},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    if (args.output/"config.json").exists():
        assert json.loads((args.output/"config.json").read_text()) == config, "Changed settings/code: use new output"
    else:
        write_json(args.output/"config.json", config)
    tasks = [(p, s, n) for p in range(21) for s in SEEDS for n in SIZES]
    if args.smoke:
        tasks = [(p, 0, 200) for p in [0, 1, 5, 7, 15]]
    started = time.perf_counter()
    paths = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        futures = {executor.submit(condition, p, s, n, args.output, args.smoke): (p, s, n) for p, s, n in tasks}
        for future in as_completed(futures):
            paths.append(Path(future.result()))
            if len(paths) % 20 == 0 or len(paths) == len(tasks):
                progress = {"completed": len(paths), "total": len(tasks), "elapsed_seconds": round(time.perf_counter()-started, 1), "last": futures[future]}
                write_json(args.output/"progress.json", progress)
                print(json.dumps(progress), flush=True)
    paths.sort()
    for name in ["metrics.csv", "fits.csv", "selections.csv"]:
        pd.concat([pd.read_csv(p/name) for p in paths], ignore_index=True).to_csv(args.output/name, index=False)
    write_json(args.output/"complete.json", {"conditions": len(paths), "models": len(paths)*88,
        "metrics_sha256": digest(args.output/"metrics.csv"), "fits_sha256": digest(args.output/"fits.csv"),
        "elapsed_seconds": time.perf_counter()-started})
    print("All grid conditions completed", flush=True)


if __name__ == "__main__":
    main()
