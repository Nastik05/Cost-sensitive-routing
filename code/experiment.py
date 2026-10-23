"""Fixed-expert BANKING77 routing experiment."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CONFIG = {"fit_sizes": [200, 500, 1000], "seeds": list(range(10)), "fractions": [.1, .2, .4],
          "strengths": [.001, .01, .1, 1.], "pairs": [("tfidf_lr", "bert_tiny_tuned"), ("tfidf_lr", "minilm_frozen_lr")]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Binary:
    def __init__(self, X, y, strength):
        self.constant = int(y[0]) if np.all(y == y[0]) else None
        self.model = None
        if self.constant is None:
            self.model = LogisticRegression(C=1/(len(y)*strength), solver="lbfgs", max_iter=1000, tol=1e-7).fit(X, y)

    def predict(self, X):
        return np.full(len(X), self.constant, dtype=float) if self.model is None else self.model.predict_proba(X)[:, 1]


def fit(method, X, W, S, strength):
    D = S.astype(int) - W.astype(int)
    if method == "direct":
        return [Ridge(alpha=len(X)*strength).fit(X, D)], 0.
    if method == "weak_error":
        return [Binary(X, 1-W, strength)], float(np.mean(1-S))
    if method == "events":
        return [Binary(X, (D == 1).astype(int), strength), Binary(X, (D == -1).astype(int), strength)], 0.
    raise ValueError(method)


def predict(fitted, X):
    models, offset = fitted
    return models[0].predict(X)-offset if len(models) == 1 else models[0].predict(X)-models[1].predict(X)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    assert digest(ROOT / "data/banking77.npz") == manifest["cache_sha256"]
    assert digest(ROOT / "data/reference.csv.gz") == manifest["reference_sha256"]
    with np.load(ROOT / "data/banking77.npz", allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    pool, dev, test = [np.flatnonzero(data["split"] == split) for split in ["router_fit", "policy_dev", "test"]]
    assert [len(pool), len(dev), len(test)] == [2000, 1000, 3080]
    assert set(pool).isdisjoint(dev) and set(pool).isdisjoint(test) and set(dev).isdisjoint(test)
    reference = pd.read_csv(ROOT / "data/reference.csv.gz").set_index(["pair_id", "fit_n", "seed", "method", "evaluation", "fraction"])
    rows = []
    started = time.perf_counter()
    for pair_id, (weak, strong) in enumerate(CONFIG["pairs"]):
        X = data[f"X__{weak}"]
        W, S = data[f"correct__{weak}"], data[f"correct__{strong}"]
        for seed in CONFIG["seeds"]:
            # Original natural-sampling exponential keys, shared across methods.
            u = np.random.default_rng(41000+seed).uniform(np.finfo(float).eps, 1, len(pool))
            order = np.argsort(-np.log(u), kind="stable")
            dev_order = np.random.default_rng(51000+seed).permutation(dev)
            for n in CONFIG["fit_sizes"]:
                fit_ids, val_ids = np.sort(pool[order[:n]]), np.sort(dev_order[:n//2])
                scaler = StandardScaler().fit(X[fit_ids])
                tr, va, te = [scaler.transform(X[idx]) for idx in [fit_ids, val_ids, test]]
                target = S[val_ids].astype(int)-W[val_ids].astype(int)
                scores, strengths = {}, {}
                for method in ["weak_error", "direct", "events"]:
                    candidates = []
                    for strength in CONFIG["strengths"]:
                        model = fit(method, tr, W[fit_ids], S[fit_ids], strength)
                        candidates.append((np.mean((predict(model, va)-target)**2), strength, model))
                    _, strength, model = min(candidates, key=lambda c: (c[0], -c[1]))
                    scores[method], strengths[method] = predict(model, te), strength
                scores["confidence"] = 1-X[test, 0].astype(np.float32)
                tie = np.random.default_rng(91000+pair_id).random(len(test))
                for evaluation, mask in [("official", np.ones(len(test), bool)), ("no_overlap", data["clean_test"][test])]:
                    wt, st = W[test][mask], S[test][mask]
                    D = st.astype(int)-wt.astype(int)
                    for method, score in scores.items():
                        ranked = np.lexsort((tie[mask], -score[mask]))
                        for fraction in CONFIG["fractions"]:
                            k = int(fraction*len(D))
                            chosen = ranked[:k]
                            rescue, harm = int((D[chosen] == 1).sum()), int((D[chosen] == -1).sum())
                            result = {"pair_id": pair_id, "weak": weak, "strong": strong, "fit_n": n, "dev_n": n//2,
                                      "seed": seed, "method": method, "evaluation": evaluation, "fraction": fraction,
                                      "calls": k, "test_n": len(D), "weak_accuracy": wt.mean(), "strong_accuracy": st.mean(),
                                      "accuracy": wt.mean()+(rescue-harm)/len(D), "rescues": rescue, "harms": harm,
                                      "gain_mse": np.mean((score[mask]-D)**2), "strength": strengths.get(method, np.nan)}
                            ref = reference.loc[(pair_id, n, seed, method, evaluation, fraction)]
                            for key in ["accuracy", "rescues", "harms", "calls", "gain_mse"]:
                                assert np.isclose(result[key], ref[key], atol=1e-10, rtol=1e-10), (pair_id, seed, n, method, key)
                            if method in strengths:
                                assert strengths[method] == ref.strength
                            rows.append(result)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    status = {"config": CONFIG, "reference_rows_reproduced": len(rows), "wall_seconds": time.perf_counter()-started,
              "cache_sha256": manifest["cache_sha256"], "metrics_sha256": digest(output / "metrics.csv"),
              "code_sha256": digest(__file__), "scope": "Routers re-trained on fixed expert cache; same held-out test, no new independent evidence"}
    (output / "run.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    return output / "metrics.csv"
