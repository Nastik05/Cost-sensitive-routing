"""One larger Monte-Carlo precision check; keep the original inference intact."""
import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import numpy as np
import pandas as pd
from common import ROOT, sha, save_json, log, verify
import infer

B = 10000


def pair_job(pair, output):
    infer.REPLICATES = B
    return infer.one_pair(pair, output)


def main():
    output = ROOT/"statistics_mc10000"
    output.mkdir(exist_ok=True)
    config = {"replicates":B,"seed":2026111214,"family":1698,"code_sha256":sha(__file__),
        "inference_code_sha256":sha(ROOT/"infer.py"),"primary_contrasts_sha256":sha(ROOT/"statistics/contrasts.csv"),
        "precision_plan_sha256":sha(ROOT/"MC_STABILITY.md")}
    if (output/"config.json").exists():
        assert json.loads((output/"config.json").read_text()) == config
    else:
        save_json(output/"config.json",config)
    with np.load(ROOT/"results/p0_s00_n1000/scores.npz") as z:
        n = len(z["W"])
    rng = np.random.default_rng(config["seed"])
    counts = rng.multinomial(n,np.full(n,1/n),size=B).astype(np.int16)
    weights = rng.multinomial(20,np.full(20,.05),size=B).astype(np.int16)
    np.savez_compressed(output/"resampling.npz",counts=counts,seed_weights=weights)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(pair_job,p,output) for p in range(6)]
        for i,future in enumerate(as_completed(futures),1):
            log("Monte-Carlo stability pair",completed=i,path=future.result(),seconds=round(time.perf_counter()-started,1))
    rows, points, columns = [], [], []
    for pair in range(6):
        with np.load(output/f"pair_{pair}.npz") as z:
            point, draws = z["points"].mean(axis=0),z["draws"]
        for spec in infer.specs():
            value, samples = 0.,np.zeros(B)
            for a,c,m,w in spec["terms"]:
                ai,ci,mi = infer.ARMS.index(a),infer.CRITERIA.index(c),infer.RULES.index(m)
                value += w*point[ai,ci,mi]
                samples += w*draws[:,ai,ci,mi]
            rows.append({"pair_id":pair,**spec,"difference_pp":100*value})
            points.append(100*value)
            columns.append(100*samples)
    table = pd.DataFrame(rows)
    points, columns = np.array(points),np.stack(columns,axis=1)
    old = pd.read_csv(ROOT/"statistics/contrasts.csv")
    np.testing.assert_allclose(points,old.difference_pp,atol=1e-12)
    result = infer.simultaneous_intervals(points,columns)
    for name in ["se","low","high","adjusted_p"]:
        table[name] = result[name]
    table["terms"] = table.terms.map(json.dumps)
    table["primary_low"] = old.low
    table["primary_high"] = old.high
    table["primary_p"] = old.adjusted_p
    table["primary_significant"] = (old.low>0)|(old.high<0)
    table["mc_significant"] = (table.low>0)|(table.high<0)
    table["classification_changed"] = table.primary_significant != table.mc_significant
    table.to_csv(output/"contrasts.csv",index=False)
    changed = table[table.classification_changed]
    changed.to_csv(output/"changes.csv",index=False)
    summary = table.groupby(["kind","criterion"])[["primary_significant","mc_significant","classification_changed"]].sum()
    summary.to_csv(output/"summary.csv")
    save_json(output/"complete.json",{"replicates":B,"family":len(table),"critical":result["critical"],
        "changed_classifications":len(changed),"all_point_estimates_unchanged":True,"old_files_preserved":verify(),
        "contrasts_sha256":sha(output/"contrasts.csv")})
    print(summary.to_string(),flush=True)


if __name__ == "__main__":
    main()
