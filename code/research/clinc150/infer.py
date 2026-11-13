"""Prespecified CLINC family; no test-guided selection of reported contrasts."""
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
from common import ROOT, OLD_CODE, sha, save_json, log, verify
from run_routers import ARMS, CRITERIA, METHODS, SEEDS
from imbalance_statistics import bootstrap_profile
from followup_stats import simultaneous_intervals

RULES = METHODS+["confidence","selector"]
REPLICATES = 2000


def specs():
    rows = []
    def add(kind, arm, criterion, m, r, terms):
        rows.append({"kind":kind,"arm":arm,"criterion":criterion,"method":m,"reference":r,"terms":terms})
    def difference(arm, c, m, r):
        return [(arm,c,m,1),(arm,c,r,-1)]
    def interaction(c, m, r):
        return difference("harm8_raw",c,m,r)+[(a,b,d,-w) for a,b,d,w in difference("natural_raw",c,m,r)]
    for c in CRITERIA:
        for m in METHODS[1:]:
            for arm in ARMS:
                add("simple_vs_weak",arm,c,m,"weak_error",difference(arm,c,m,"weak_error"))
            add("interaction_vs_weak","harm8_raw",c,m,"weak_error",interaction(c,m,"weak_error"))
        for m,r in itertools.combinations(METHODS[1:],2):
            for arm in ARMS:
                add("gain_simple",arm,c,m,r,difference(arm,c,m,r))
            add("gain_interaction","harm8_raw",c,m,r,interaction(c,m,r))
        for m in METHODS:
            add("correction","harm8_ipw",c,m,m,[("harm8_ipw",c,m,1),("harm8_raw",c,m,-1)])
        for arm in ARMS:
            for r in ["weak_error","confidence"]:
                add("selector",arm,c,"selector",r,difference(arm,c,"selector",r))
    for arm in ARMS:
        for m in METHODS:
            add("criterion_effect",arm,"accuracy-minus-mse",m,m,[(arm,"accuracy",m,1),(arm,"mse",m,-1)])
    for m in METHODS[1:]:
        add("criterion_interaction","harm8_raw","accuracy-minus-mse",m,"weak_error",
            interaction("accuracy",m,"weak_error")+[(a,b,d,-w) for a,b,d,w in interaction("mse",m,"weak_error")])
    assert len(rows) == 283
    return rows


def one_pair(pair_id, output):
    output = Path(output)
    path = output/f"pair_{pair_id}.npz"
    if path.exists():
        return str(path)
    with np.load(output/"resampling.npz") as z:
        counts, seed_weights = z["counts"], z["seed_weights"]
    points = np.empty((len(SEEDS),len(ARMS),len(CRITERIA),len(RULES)))
    draws = np.zeros((REPLICATES,len(ARMS),len(CRITERIA),len(RULES)))
    expected_rows = None
    for seed in SEEDS:
        condition = ROOT/"results"/f"p{pair_id}_s{seed:02d}_n1000"
        table = pd.read_csv(condition/"metrics.csv")
        table = table[(table.evaluation == "official") & (table.fraction == .2)]
        with np.load(condition/"scores.npz") as z:
            if expected_rows is None:
                expected_rows = z["row_id"].copy()
            else:
                np.testing.assert_array_equal(expected_rows,z["row_id"])
            D = z["S"].astype(int)-z["W"]
            n, k = len(D), int(.2*len(D))
            boot, point = {}, {}
            for index in sorted(table.score_index.unique()):
                score = z["scores"][index]
                rh = bootstrap_profile(score,D,z["tie"],counts,.2)
                boot[index] = (rh[:,0]-rh[:,1])/n
                selected = np.lexsort((z["tie"],-score))[:k]
                point[index] = D[selected].sum()/n
            for ai,arm in enumerate(ARMS):
                for ci,c in enumerate(CRITERIA):
                    for mi,m in enumerate(RULES):
                        row = table[table.method == "confidence"] if m == "confidence" else table[(table.arm == arm)&(table.criterion == c)&(table.method == m)]
                        assert len(row) == 1
                        index = int(row.score_index.iloc[0])
                        points[seed,ai,ci,mi] = point[index]
                        np.testing.assert_allclose(point[index]+z["W"].mean(),row.accuracy.iloc[0],atol=1e-12)
                        draws[:,ai,ci,mi] += boot[index]*seed_weights[:,seed]/len(SEEDS)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,points=points,draws=draws,row_id=expected_rows)
    temporary.replace(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    args = parser.parse_args()
    output = ROOT/"statistics"
    output.mkdir(exist_ok=True)
    run = json.loads((ROOT/"results/complete.json").read_text())
    assert run["conditions"] == 240
    config = {"metrics_sha256":sha(ROOT/"results/metrics.csv"),"source_config_sha256":sha(ROOT/"results/config.json"),
        "replicates":REPLICATES,"seed":2026111213,"code_sha256":sha(__file__),
        "bootstrap_code_sha256":sha(OLD_CODE/"imbalance_statistics.py"),"inference_code_sha256":sha(OLD_CODE/"followup_stats.py")}
    assert config["metrics_sha256"] == run["metrics_sha256"]
    if (output/"config.json").exists():
        assert json.loads((output/"config.json").read_text()) == config
    else:
        save_json(output/"config.json",config)
    with np.load(ROOT/"results/p0_s00_n1000/scores.npz") as z:
        n = len(z["W"])
    rng = np.random.default_rng(config["seed"])
    counts = rng.multinomial(n,np.full(n,1/n),size=REPLICATES).astype(np.int16)
    seed_weights = rng.multinomial(len(SEEDS),np.full(len(SEEDS),1/len(SEEDS)),size=REPLICATES).astype(np.int16)
    np.savez_compressed(output/"resampling.npz",counts=counts,seed_weights=seed_weights)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one_pair,p,output) for p in range(6)]
        for i,future in enumerate(as_completed(futures),1):
            log("Bootstrap pair complete",completed=i,path=future.result(),seconds=round(time.perf_counter()-started,1))
    rows, point_values, columns = [], [], []
    for pair_id in range(6):
        with np.load(output/f"pair_{pair_id}.npz") as z:
            point,boot = z["points"].mean(axis=0),z["draws"]
        for specification in specs():
            value, samples = 0., np.zeros(REPLICATES)
            for arm,criterion,method,weight in specification["terms"]:
                ai,ci,mi = ARMS.index(arm),CRITERIA.index(criterion),RULES.index(method)
                value += weight*point[ai,ci,mi]
                samples += weight*boot[:,ai,ci,mi]
            rows.append({"pair_id":pair_id,**specification,"difference_pp":100*value})
            point_values.append(100*value)
            columns.append(100*samples)
    point_values,columns = np.array(point_values),np.stack(columns,axis=1)
    assert len(rows) == 1698
    inference = simultaneous_intervals(point_values,columns)
    table = pd.DataFrame(rows)
    for key in ["se","low","high","adjusted_p"]:
        table[key] = inference[key]
    # Verify every linear contrast against the independently audited metric table.
    metrics = pd.read_csv(ROOT/"results/metrics.csv")
    metrics = metrics[(metrics.evaluation=="official")&(metrics.fit_n==1000)&(metrics.fraction==.2)]
    means = metrics.groupby(["pair_id","arm","criterion","method"]).accuracy.mean()
    for row in table.itertuples():
        value = 0
        for arm,c,m,w in row.terms:
            key = (row.pair_id,"baseline","fixed","confidence") if m == "confidence" else (row.pair_id,arm,c,m)
            value += w*means.loc[key]
        assert abs(100*value-row.difference_pp)<1e-9
    table["terms"] = table.terms.map(json.dumps)
    table.to_csv(output/"contrasts.csv",index=False)
    np.savez_compressed(output/"contrast_draws.npz",point=point_values,draws=columns)
    save_json(output/"complete.json", {"family":len(table),"replicates":REPLICATES,"critical":inference["critical"],
        "contrast_sha256":sha(output/"contrasts.csv"),"all_points_verified":True,"preserved_files":verify()})
    log("Statistical family complete",contrasts=len(table))


if __name__ == "__main__":
    main()
