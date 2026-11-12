"""Controlled with-replacement sampling; disjoint tuning and method selection."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from experiment import ROOT, CONFIG, digest, fit, predict

REGIMES = ["natural", "rescue_rich", "harm_rich", "discordant_rich"]
PREFERENCE = ["confidence", "direct", "weak_error", "events"]
SETTINGS = {"seeds": list(range(30)), "fit_sizes": [200, 500, 1000],
            "fractions": [.1, .2, .4], "regimes": REGIMES,
            "bootstrap_replicates": 2000, "sampling": "with_replacement"}


def sampling_probabilities(D, regime):
    a = np.ones(len(D), dtype=float)
    if regime == "rescue_rich":
        a[D == 1] = 4
    elif regime == "harm_rich":
        a[D == -1] = 8
    elif regime == "discordant_rich":
        a[D != 0] = 4
    elif regime != "natural":
        raise ValueError(regime)
    return a / a.sum()


def sample_pool(pool, probabilities, seed, n):
    # Common uniforms couple regimes; prefixes couple sizes. Known draw probabilities.
    u = np.random.default_rng(720000 + seed).random(n)
    positions = np.searchsorted(np.cumsum(probabilities), u, side="right")
    weights = 1 / (len(pool) * probabilities[positions])
    return pool[positions], weights


def partition_dev(dev, seed, n):
    order = np.random.default_rng(730000 + seed).permutation(dev)
    return np.sort(order[:n // 2]), np.sort(order[500:1000])


def evaluate(score, W, S, tie, fraction):
    D = S.astype(int) - W.astype(int)
    k = int(fraction * len(D))
    chosen = np.lexsort((tie, -score))[:k]
    rescue, harm = int((D[chosen] == 1).sum()), int((D[chosen] == -1).sum())
    return {"accuracy": float(W.mean() + (rescue-harm)/len(D)),
            "gain_mse": float(np.mean((score-D)**2)), "rescues": rescue,
            "harms": harm, "calls": k, "eval_n": len(D)}


def choose_method(scores, W, S, tie, fraction):
    # Inputs contain selector-dev only; test never enters this function.
    return max(PREFERENCE, key=lambda m: evaluate(scores[m], W, S, tie, fraction)["accuracy"])


def parameter_count(model):
    result = 0
    for head in model[0]:
        fitted = getattr(head, "model", head)
        result += 1 if fitted is None else fitted.coef_.size + np.asarray(fitted.intercept_).size
    return result


def run_condition(data, pair_id, seed, n, directory):
    directory.mkdir(parents=True, exist_ok=True)
    weak, strong = CONFIG["pairs"][pair_id]
    X, W, S = data[f"X__{weak}"], data[f"correct__{weak}"], data[f"correct__{strong}"]
    pool, dev, test = [np.flatnonzero(data["split"] == s) for s in ["router_fit", "policy_dev", "test"]]
    tune, select = partition_dev(dev, seed, n)
    assert not set(tune) & set(select) and not set(pool) & set(dev) and not set(test) & set(dev)
    arrays = {"test_ids": test, "row_id": data["row_id"][test], "tune_ids": tune, "select_ids": select,
              "W": W[test], "S": S[test], "clean": data["clean_test"][test]}
    ties = {split: np.random.default_rng(740000 + pair_id).random(len(ids))
            for split, ids in [("train", np.arange(n)), ("tune", tune), ("select", select), ("official", test)]}
    arrays["tie"] = ties["official"]
    rows, fitted_models, scores_test, names, fit_info = [], {}, [], [], []
    natural_select, natural_test = {}, {}
    base = {"pair_id": pair_id, "weak": weak, "strong": strong, "seed": seed, "fit_n": n,
            "tune_n": len(tune), "select_n": len(select)}
    for regime in REGIMES:
        probability = sampling_probabilities(S[pool].astype(int)-W[pool].astype(int), regime)
        ids, weights = sample_pool(pool, probability, seed, n)
        arrays[f"fit_ids__{regime}"], arrays[f"weights__{regime}"] = ids, weights
        scaler = StandardScaler().fit(X[ids])
        transformed = {s: scaler.transform(X[i]) for s, i in
                       [("train", ids), ("tune", tune), ("select", select), ("official", test)]}
        methods = ["direct", "events", "weak_error", "confidence"] if regime == "natural" else ["direct", "events"]
        for corrected in ([False] if regime == "natural" else [False, True]):
            for method in methods:
                key = f"{regime}__{'ipw' if corrected else 'raw'}__{method}"
                started = time.perf_counter()
                if method == "confidence":
                    score = {s: 1-X[i, 0] for s, i in [("train", ids), ("tune", tune), ("select", select), ("official", test)]}
                    strength, count = np.nan, 0
                else:
                    candidates = []
                    for strength in CONFIG["strengths"]:
                        model = fit(method, transformed["train"], W[ids], S[ids], strength,
                                    weights if corrected else None)
                        mse = np.mean((predict(model, transformed["tune"])-(S[tune].astype(int)-W[tune]))**2)
                        candidates.append((mse, strength, model))
                    _, strength, model = min(candidates, key=lambda c: (c[0], -c[1]))
                    score = {s: predict(model, x) for s, x in transformed.items()}
                    count = parameter_count(model)
                    fitted_models[key] = (scaler, model)
                seconds = time.perf_counter()-started
                info = {**base, "regime": regime, "corrected": corrected, "method": method,
                        "strength": strength, "unique_n": len(np.unique(ids)),
                        "ess": float(weights.sum()**2 / (weights @ weights)) if corrected else float(n),
                        "fit_rescue": int((S[ids].astype(int)-W[ids] == 1).sum()),
                        "fit_harm": int((S[ids].astype(int)-W[ids] == -1).sum()),
                        "parameter_count": count, "fit_seconds": seconds}
                fit_info.append(info)
                for split, idx in [("train", ids), ("tune", tune), ("select", select), ("official", test), ("no_overlap", test)]:
                    source = "official" if split == "no_overlap" else split
                    mask = data["clean_test"][test] if split == "no_overlap" else np.ones(len(idx), bool)
                    for fraction in SETTINGS["fractions"]:
                        rows.append({**info, "evaluation": split, "fraction": fraction,
                                     **evaluate(score[source][mask], W[idx][mask], S[idx][mask], ties[source][mask], fraction)})
                scores_test.append(score["official"])
                names.append(key)
                if regime == "natural":
                    natural_select[method], natural_test[method] = score["select"], score["official"]
    for fraction in SETTINGS["fractions"]:
        selected = choose_method(natural_select, W[select], S[select], ties["select"], fraction)
        arrays[f"selected__{fraction}"] = np.array(selected)
        for evaluation in ["official", "no_overlap"]:
            mask = np.ones(len(test), bool) if evaluation == "official" else data["clean_test"][test]
            metrics = {m: evaluate(s[mask], W[test][mask], S[test][mask], ties["official"][mask], fraction)
                       for m, s in natural_test.items()}
            oracle = max(v["accuracy"] for v in metrics.values())
            rows.append({**base, "regime": "natural", "corrected": False, "method": "selector",
                         "selected_method": selected, "evaluation": evaluation, "fraction": fraction,
                         "oracle_accuracy": oracle, "regret": oracle-metrics[selected]["accuracy"], **metrics[selected]})
    arrays["scores"], arrays["names"] = np.stack(scores_test), np.array(names)
    np.savez_compressed(directory / "scores.npz", **arrays)
    joblib.dump(fitted_models, directory / "models.joblib", compress=3)
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(fit_info).to_csv(directory / "fits.csv", index=False)
    status = {name: digest(directory / name) for name in ["scores.npz", "models.joblib", "metrics.csv", "fits.csv"]}
    (directory / "complete.json").write_text(json.dumps(status, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/followup")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    assert digest(ROOT / "data/banking77.npz") == manifest["cache_sha256"]
    with np.load(ROOT / "data/banking77.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    settings = {**SETTINGS, "cache_sha256": manifest["cache_sha256"],
                "code": {name: digest(ROOT / name) for name in ["followup.py", "experiment.py", "PROTOCOL_FOLLOWUP.md"]},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    config = args.output / "config.json"
    if config.exists():
        assert json.loads(config.read_text()) == settings, "Changed protocol/code/environment: use a fresh output directory"
    else:
        config.write_text(json.dumps(settings, indent=2)+"\n")
    warnings.simplefilter("error", ConvergenceWarning)
    paths = []
    for pair_id in range(2):
        for seed in SETTINGS["seeds"]:
            for n in SETTINGS["fit_sizes"]:
                directory = args.output / f"p{pair_id}_s{seed}_n{n}"
                marker = directory / "complete.json"
                if marker.exists():
                    for name, value in json.loads(marker.read_text()).items():
                        assert digest(directory / name) == value
                else:
                    run_condition(data, pair_id, seed, n, directory)
                paths.append(directory)
            print(f"Completed pair {pair_id}, seed {seed}", flush=True)
    for name in ["metrics.csv", "fits.csv"]:
        pd.concat([pd.read_csv(p / name) for p in paths], ignore_index=True).to_csv(args.output / name, index=False)
    (args.output / "complete.json").write_text(json.dumps({"conditions": len(paths),
        "metrics_sha256": digest(args.output / "metrics.csv"), "fits_sha256": digest(args.output / "fits.csv")}, indent=2)+"\n")


if __name__ == "__main__":
    main()
