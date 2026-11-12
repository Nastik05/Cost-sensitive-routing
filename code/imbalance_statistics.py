"""One prespecified simultaneous family: accuracy, interactions and decision profiles."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from imbalance_study import ROOT, SEEDS, METHODS, digest, write_json
from followup_stats import simultaneous_intervals

ARMS = ["natural_raw", "rescue8_raw", "harm8_raw", "rescue8_ipw", "harm8_ipw"]
RULES = METHODS+["confidence", "selector"]
CHANNELS = ["accuracy", "rescue_precision", "harm_rate"]
B = 2000


def wait_for(path, seconds=7200):
    start = time.monotonic()
    while not path.exists():
        if time.monotonic()-start > seconds:
            raise TimeoutError(f"Waiting for completed training checkpoint: {path}")
        time.sleep(5)


def bootstrap_profile(scores, D, tie, counts, fraction=.2, chunk=128):
    """Exact R,H counts in resampled top-k batches, including a partial boundary row."""
    n, k = len(D), int(len(D)*fraction)
    assert counts.ndim == 2 and counts.shape[1] == n and np.all(counts.sum(axis=1) == n)
    order = np.lexsort((tie, -scores))
    # Most resampled top-k boundaries lie near rank k. Use a prefix only after
    # checking it contains at least k copies in every draw; otherwise fall back
    # to the full ranking. This is exact, not a probabilistic truncation.
    prefix = min(n, max(k+64, k+int(8*np.sqrt(k+1))))
    output = np.empty((len(counts), 2))
    for begin in range(0, len(counts), chunk):
        active = order[:prefix]
        c = counts[begin:begin+chunk, active].astype(np.int32)
        if np.any(c.sum(axis=1) < k):
            active = order
            c = counts[begin:begin+chunk, active].astype(np.int32)
        rescue, harm = D[active] == 1, D[active] == -1
        cumulative = np.cumsum(c, axis=1, dtype=np.int32)
        selected = c-np.clip(cumulative-k, 0, c)
        assert np.all(selected.sum(axis=1) == k)
        output[begin:begin+len(c), 0] = selected[:, rescue].sum(axis=1)
        output[begin:begin+len(c), 1] = selected[:, harm].sum(axis=1)
    return output


def pair_statistics(pair, output):
    output = Path(output)
    destination = output/"statistics"/f"pair_{pair:02d}.npz"
    if destination.exists():
        return str(destination)
    with np.load(output/"statistics/resampling.npz") as z:
        counts, seed_weights = z["counts"], z["seed_weights"]
    point = np.empty((len(SEEDS), len(ARMS), len(RULES), 3))
    draws = np.zeros((B, len(ARMS), len(RULES), 3))
    for seed in SEEDS:
        path = output/f"p{pair:02d}_s{seed:02d}_n1000"
        wait_for(path/"complete.json")
        selection = pd.read_csv(path/"selections.csv")
        selection = selection[selection.fraction == .2].set_index("arm").selected_method.to_dict()
        with np.load(path/"scores.npz") as z:
            names = z["names"].tolist()
            D = z["S"].astype(int)-z["W"]
            n, k = len(D), int(.2*len(D))
            needed = ["confidence"]+[arm+"__"+m for arm in ARMS for m in METHODS]
            boot, points = {}, {}
            for name in needed:
                score = z["scores"][names.index(name)]
                rh = bootstrap_profile(score, D, z["tie"], counts)
                boot[name] = np.column_stack([(rh[:, 0]-rh[:, 1])/n, rh[:, 0]/k, rh[:, 1]/k])
                chosen = np.lexsort((z["tie"], -score))[:k]
                R, H = (D[chosen] == 1).sum(), (D[chosen] == -1).sum()
                points[name] = np.array([(R-H)/n, R/k, H/k])
            for ai, arm in enumerate(ARMS):
                for mi, method in enumerate(RULES):
                    selected = selection[arm] if method == "selector" else method
                    name = "confidence" if selected == "confidence" else arm+"__"+selected
                    draws[:, ai, mi] += boot[name]*seed_weights[:, seed, None]/len(SEEDS)
                    point[seed, ai, mi] = points[name]
            row_ids = z["row_id"].copy()
    np.savez_compressed(destination, point=point, draws=draws, row_ids=row_ids,
                        arms=np.array(ARMS), rules=np.array(RULES), channels=np.array(CHANNELS))
    return str(destination)


def build_contrasts(output):
    records, points, columns = [], [], []
    manifest = json.loads((ROOT/"data/imbalance_manifest.json").read_text())
    expected_rows = None
    for pair in manifest["pairs"]:
        with np.load(output/"statistics"/f"pair_{pair['pair_id']:02d}.npz") as z:
            point = z["point"].mean(axis=0)
            draws = z["draws"]
            if expected_rows is None:
                expected_rows = z["row_ids"].copy()
            else:
                np.testing.assert_array_equal(z["row_ids"], expected_rows)

        def add(kind, arm, method, reference, channel=0, change=False, correction=False):
            ai, mi, ri = ARMS.index(arm), RULES.index(method), RULES.index(reference)
            a = point[ai, mi, channel]
            b = point[ai, ri, channel]
            delta = a-b
            boot = draws[:, ai, mi, channel]-draws[:, ai, ri, channel]
            if change:
                delta -= point[0, mi, channel]-point[0, ri, channel]
                boot -= draws[:, 0, mi, channel]-draws[:, 0, ri, channel]
            if correction:
                raw_ai = ARMS.index(arm.replace("_ipw", "_raw"))
                delta = point[ai, mi, channel]-point[raw_ai, mi, channel]
                boot = draws[:, ai, mi, channel]-draws[:, raw_ai, mi, channel]
            records.append({**pair, "kind": kind, "arm": arm, "method": method, "reference": reference,
                            "metric": CHANNELS[channel], "difference_pp": float(100*delta),
                            "gain_only": method in METHODS[1:] and reference in METHODS[1:]})
            points.append(100*delta)
            columns.append(100*boot)

        for method, reference in itertools.combinations(METHODS+["confidence"], 2):
            for arm in ARMS[:3]:
                add("simple", arm, method, reference)
            for arm in ARMS[1:3]:
                add("interaction", arm, method, reference, change=True)
            for channel in [1, 2]:
                add("profile", "natural_raw", method, reference, channel=channel)
        for arm in ARMS[3:]:
            for method in METHODS:
                add("correction", arm, method, method, correction=True)
        for arm in ARMS[:3]:
            add("selector", arm, "selector", "confidence")
    assert len(records) == 5691
    return pd.DataFrame(records), np.array(points), np.stack(columns, axis=1)


def verify_points(table, metrics):
    p = metrics[(metrics.fit_n == 1000) & (metrics.fraction == .2) & (metrics.evaluation == "official")]
    mean = p.groupby(["pair_id", "arm", "method"])[CHANNELS].mean()
    def get(pair, arm, method, metric):
        return mean.loc[(pair, "baseline" if method == "confidence" else arm, method), metric]
    for row in table.itertuples():
        a = get(row.pair_id, row.arm, row.method, row.metric)
        if row.kind == "correction":
            b = get(row.pair_id, row.arm.replace("_ipw", "_raw"), row.method, row.metric)
        else:
            b = get(row.pair_id, row.arm, row.reference, row.metric)
        delta = a-b
        if row.kind == "interaction":
            delta -= get(row.pair_id, "natural_raw", row.method, row.metric)-get(row.pair_id, "natural_raw", row.reference, row.metric)
        assert abs(100*delta-row.difference_pp) < 1e-9, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/imbalance_all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--wait", action="store_true", help="Consume completed pair checkpoints while training continues")
    args = parser.parse_args()
    out = args.output/"statistics"
    out.mkdir(parents=True, exist_ok=True)
    if not args.wait:
        assert (args.output/"complete.json").exists(), "Training is not complete; use --wait to stream checkpoints"
    settings = {"replicates": B, "seed": 2026111209,
                "code_sha256": digest(__file__), "inference_code_sha256": digest(ROOT/"followup_stats.py"),
                "source_config_sha256": digest(args.output/"config.json")}
    if (out/"config.json").exists():
        assert json.loads((out/"config.json").read_text()) == settings, "Changed statistics: use fresh statistics directory"
    else:
        write_json(out/"config.json", settings)
    rng = np.random.default_rng(settings["seed"])
    counts = rng.multinomial(3080, np.full(3080, 1/3080), size=B).astype(np.int16)
    seed_weights = rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1/len(SEEDS)), size=B).astype(np.int16)
    np.savez_compressed(out/"resampling.npz", counts=counts, seed_weights=seed_weights)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(pair_statistics, pair, args.output) for pair in range(21)]
        for i, future in enumerate(as_completed(futures), 1):
            print(json.dumps({"pairs_completed": i, "path": future.result(), "seconds": round(time.perf_counter()-started, 1)}), flush=True)
    table, points, draws = build_contrasts(args.output)
    inference = simultaneous_intervals(points, draws)
    for name in ["se", "low", "high", "adjusted_p"]:
        table[name] = inference[name]
    wait_for(args.output/"complete.json")
    complete = json.loads((args.output/"complete.json").read_text())
    assert complete["conditions"] == 1260
    assert complete["metrics_sha256"] == digest(args.output/"metrics.csv")
    verify_points(table, pd.read_csv(args.output/"metrics.csv"))
    table.to_csv(out/"contrasts.csv", index=False)
    np.savez_compressed(out/"contrasts_bootstrap.npz", point=points, draws=draws)
    write_json(out/"complete.json", {"contrasts": len(table), "critical": inference["critical"], "replicates": B,
        "point_estimates_verified": True, "metrics_sha256": complete["metrics_sha256"],
        "contrasts_sha256": digest(out/"contrasts.csv"), "seconds": time.perf_counter()-started})
    print(table.groupby("kind").adjusted_p.agg(total="size", significant=lambda x: int((x < .05).sum())).to_string(), flush=True)


if __name__ == "__main__":
    main()
