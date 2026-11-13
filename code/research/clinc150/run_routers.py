"""One candidate grid, two validation criteria; all outputs stay in this new stage."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from common import ROOT, OLD_CODE, sha, save_json, log, verify
from imbalance_estimators import METHODS, STRENGTHS, fit_estimator
from imbalance_study import draw_ids, metrics, PREFERENCE

ARMS = ["natural_raw", "harm8_raw", "harm8_ipw"]
CRITERIA = ["mse", "accuracy"]
FRACTIONS = [.1, .2, .4]
SEEDS = list(range(20))
SIZES = [200, 1000]
DATA, PAIRS = None, None


def initialize():
    global DATA, PAIRS
    with np.load(ROOT/"artifacts/router_cache.npz", allow_pickle=False) as z:
        DATA = {k: z[k] for k in z.files}
    PAIRS = json.loads((ROOT/"artifacts/complete.json").read_text())["pairs"]


def select_lambda(grid, criterion, fraction):
    if criterion == "mse":
        return min(grid, key=lambda c: (c["tune_mse"], -c["strength"]))
    if criterion == "accuracy":
        return min(grid, key=lambda c: (-c[f"tune_accuracy_{fraction}"], -c["strength"]))
    raise ValueError(criterion)


def condition(pair_id, seed, n, output, smoke=False):
    directory = Path(output)/f"p{pair_id}_s{seed:02d}_n{n}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/"complete.json"
    if marker.exists():
        for name, value in json.loads(marker.read_text())["files"].items():
            assert sha(directory/name) == value
        return str(directory)
    started = time.perf_counter()
    pair = PAIRS[pair_id]
    X, W, S = DATA[f"X__{pair['weak']}"], DATA[f"correct__{pair['weak']}"], DATA[f"correct__{pair['strong']}"]
    pool, dev, test = [np.flatnonzero(DATA["split"] == value) for value in ["router_fit", "dev", "test"]]
    order = np.random.default_rng(2026111300+seed).permutation(dev)
    tune, select = np.sort(order[:n//2]), np.sort(order[500:1000])
    if smoke:
        test = np.sort(order[1000:1500])
    assert not set(test)&(set(tune)|set(select)|set(pool)) and not set(tune)&set(select)
    tie = {split: np.random.default_rng(2026111400+pair_id).random(len(ids))
           for split, ids in [("train", np.arange(n)), ("tune", tune), ("select", select), ("test", test)]}
    arrays = {"row_id": DATA["row_id"][test], "test_ids": test, "tune_ids": tune, "select_ids": select,
              "tie": tie["test"], "W": W[test], "S": S[test],
              "clean": np.ones(len(test), bool) if smoke else DATA["clean_test"][test]}
    names = ["confidence"]
    scores = [1-X[test,0]]
    select_scores = {"confidence": 1-X[select,0]}
    tune_scores = {"confidence": 1-X[tune,0]}
    models, grids, fit_rows, rows = {}, {}, [], []
    base = {"pair_id": pair_id, "seed": seed, "fit_n": n}
    samples = {}
    for regime in ["natural", "harm8"]:
        ids, weights, scale = draw_ids(pool, S[pool].astype(int)-W[pool], regime, seed, n)
        samples[regime] = (ids, weights, scale)
        arrays[f"fit_ids__{regime}"] = ids
        arrays[f"weights__{regime}"] = weights
    for arm in ARMS:
        regime = arm.split("_")[0]
        ids, weights, conditional_scale = samples[regime]
        scaler = StandardScaler().fit(X[ids])
        tx = {key: scaler.transform(X[idx]) for key, idx in [("train", ids), ("tune", tune), ("select", select), ("test", test)]}
        for method in METHODS:
            grid = []
            for strength in STRENGTHS:
                start_fit = time.perf_counter()
                model = fit_estimator(method, tx["train"], W[ids], S[ids], strength,
                    weights if arm.endswith("ipw") else None, conditional_scale)
                key = f"{arm}__{method}__{strength}"
                predicted = {split: model.predict(value) for split, value in tx.items()}
                assert all(np.isfinite(v).all() for v in predicted.values())
                item = {**base, "arm": arm, "method": method, "strength": strength, "key": key, "score_index": len(names),
                        "train_mse": float(np.mean((predicted["train"]-(S[ids].astype(int)-W[ids]))**2)),
                        "tune_mse": float(np.mean((predicted["tune"]-(S[tune].astype(int)-W[tune]))**2)),
                        "unique_n": len(np.unique(ids)), "fit_rescue": int((S[ids].astype(int)-W[ids] == 1).sum()),
                        "fit_harm": int((S[ids].astype(int)-W[ids] == -1).sum()),
                        "parameter_count": int(model.parameter_count), "fit_seconds": time.perf_counter()-start_fit}
                for fraction in FRACTIONS:
                    item[f"tune_accuracy_{fraction}"] = metrics(predicted["tune"], W[tune], S[tune], tie["tune"], fraction)["accuracy"]
                grid.append(item)
                fit_rows.append(item)
                models[key] = (scaler, model)
                names.append(key)
                scores.append(predicted["test"])
                select_scores[key], tune_scores[key] = predicted["select"], predicted["tune"]
            grids[arm, method] = grid
    scores = np.stack(scores)
    evaluations = ["dev_smoke"] if smoke else ["official", "no_overlap"]
    cache = {}
    def evaluate(index, evaluation, fraction):
        key = index, evaluation, fraction
        if key not in cache:
            mask = arrays["clean"] if evaluation == "no_overlap" else np.ones(len(test), bool)
            cache[key] = metrics(scores[index][mask], W[test][mask], S[test][mask], tie["test"][mask], fraction)
        return cache[key]
    for fraction in FRACTIONS:
        for evaluation in evaluations:
            rows.append({**base, "arm": "baseline", "criterion": "fixed", "method": "confidence", "strength": np.nan,
                         "evaluation": evaluation, "fraction": fraction, "score_index": 0, **evaluate(0, evaluation, fraction)})
    choices = []
    for arm in ARMS:
        for criterion in CRITERIA:
            for fraction in FRACTIONS:
                selected = {m: select_lambda(grids[arm, m], criterion, fraction) for m in METHODS}
                selector_values = {m: metrics(select_scores[selected[m]["key"]], W[select], S[select], tie["select"], fraction)["accuracy"] for m in METHODS}
                selector_values["confidence"] = metrics(select_scores["confidence"], W[select], S[select], tie["select"], fraction)["accuracy"]
                winner = max(PREFERENCE, key=lambda m: selector_values[m])
                index = 0 if winner == "confidence" else selected[winner]["score_index"]
                choices.append({**base, "arm": arm, "criterion": criterion, "fraction": fraction, "method": winner, "score_index": index})
                for evaluation in evaluations:
                    candidate_values = {"confidence": evaluate(0, evaluation, fraction)["accuracy"]}
                    for method, candidate in selected.items():
                        result = evaluate(candidate["score_index"], evaluation, fraction)
                        candidate_values[method] = result["accuracy"]
                        rows.append({**base, "arm": arm, "criterion": criterion, "method": method, "strength": candidate["strength"],
                            "evaluation": evaluation, "fraction": fraction, "score_index": candidate["score_index"],
                            "tune_mse": candidate["tune_mse"], "tune_accuracy": candidate[f"tune_accuracy_{fraction}"], **result})
                    result = evaluate(index, evaluation, fraction)
                    oracle = max(candidate_values.values())
                    rows.append({**base, "arm": arm, "criterion": criterion, "method": "selector", "selected_method": winner,
                        "strength": np.nan, "evaluation": evaluation, "fraction": fraction, "score_index": index,
                        "oracle_accuracy": oracle, "regret": oracle-result["accuracy"], **result})
    np.savez_compressed(directory/"scores.npz", names=np.array(names), scores=scores,
        select_scores=np.stack([select_scores[name] for name in names]), tune_scores=np.stack([tune_scores[name] for name in names]), **arrays)
    joblib.dump(models, directory/"models.joblib", compress=3)
    pd.DataFrame(rows).to_csv(directory/"metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(directory/"candidates.csv", index=False)
    pd.DataFrame(choices).to_csv(directory/"selection.csv", index=False)
    save_json(marker, {"models": len(models), "metric_rows": len(rows), "seconds": time.perf_counter()-started,
        "files": {name: sha(directory/name) for name in ["scores.npz", "models.joblib", "metrics.csv", "candidates.csv", "selection.csv"]}})
    return str(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = ROOT/("smoke" if args.smoke else "results")
    output.mkdir(exist_ok=True)
    experts = json.loads((ROOT/"artifacts/complete.json").read_text())
    assert sha(ROOT/"artifacts/router_cache.npz") == experts["cache_sha256"]
    config = {"smoke": args.smoke, "pairs": experts["pairs"], "sizes": SIZES, "seeds": SEEDS, "arms": ARMS,
        "criteria": CRITERIA, "fractions": FRACTIONS, "methods": METHODS, "strengths": STRENGTHS,
        "cache_sha256": experts["cache_sha256"], "source_code": {str(p): sha(p) for p in
            [ROOT/"PROTOCOL.md", ROOT/"common.py", Path(__file__), OLD_CODE/"imbalance_estimators.py", OLD_CODE/"imbalance_study.py"]}}
    if (output/"config.json").exists():
        assert json.loads((output/"config.json").read_text()) == config
    else:
        save_json(output/"config.json", config)
    tasks = [(p,s,n) for p in range(6) for s in SEEDS for n in SIZES]
    if args.smoke:
        tasks = [(p,0,200) for p in [0,2,5]]
    started, paths = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        futures = [executor.submit(condition, p,s,n,output,args.smoke) for p,s,n in tasks]
        for future in as_completed(futures):
            paths.append(Path(future.result()))
            if len(paths)%20 == 0 or len(paths)==len(tasks):
                progress = {"complete": len(paths), "total": len(tasks), "seconds": round(time.perf_counter()-started,1)}
                save_json(output/"progress.json", progress)
                log("Router checkpoint", **progress)
    for name in ["metrics.csv", "candidates.csv", "selection.csv"]:
        pd.concat([pd.read_csv(p/name) for p in sorted(paths)], ignore_index=True).to_csv(output/name, index=False)
    save_json(output/"complete.json", {"conditions": len(paths), "candidate_models": len(paths)*96,
        "metrics_sha256": sha(output/"metrics.csv"), "candidates_sha256": sha(output/"candidates.csv"),
        "seconds": time.perf_counter()-started, "preexisting_files_preserved": verify()})
    log("Router grid complete", smoke=args.smoke, conditions=len(paths))


if __name__ == "__main__":
    main()
