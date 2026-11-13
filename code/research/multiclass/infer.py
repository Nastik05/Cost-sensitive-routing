"""A joint family for accuracy, macro precision and macro F1 on natural data."""
import os
for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from common import ROOT,DATASETS,OLD_CODE,sha,save_json,preserve,from_predictions
from bootstrap import multiclass_bootstrap,prepare_base,stratified_counts
from followup_stats import simultaneous_intervals

METHODS=["weak_error","direct","marginal","events","ternary","joint","hurdle","soft_gain","confidence"]
METRICS=["accuracy","macro_precision","macro_f1"]
B=2000


def one_pair(dataset,pair_id):
    destination=ROOT/"statistics"/f"{dataset}_p{pair_id:02d}.npz"
    if destination.exists():
        return str(destination)
    with np.load(ROOT/"labels"/f"{dataset}.npz") as z:
        labels={key:z[key] for key in z.files}
    manifest=json.loads((ROOT/"labels/manifest.json").read_text())[dataset]
    pair=manifest["pairs"][pair_id]
    expert_names=labels["expert_names"].tolist()
    truth=labels["y_true"]
    first=labels["predictions"][expert_names.index(pair["weak"])]
    second=labels["predictions"][expert_names.index(pair["strong"])]
    k=manifest["classes"]
    with np.load(ROOT/"statistics"/f"{dataset}_resampling.npz") as z:
        counts,weights=z["counts"],z["seed_weights"]
    base=prepare_base(truth,first,counts,k)
    np.testing.assert_array_equal(base["support"],np.broadcast_to(labels["support"],base["support"].shape))
    criteria=["mse"] if dataset=="banking77" else ["mse","accuracy"]
    points=np.empty((20,len(criteria),len(METHODS),3))
    draws=np.zeros((B,len(criteria),len(METHODS),3))
    conf_boot,conf_point=None,None
    for seed in range(20):
        name=f"p{pair_id:02d}_s{seed:02d}_n1000" if dataset=="banking77" else f"p{pair_id}_s{seed:02d}_n1000"
        source=DATASETS[dataset]["results"]/name
        table=pd.read_csv(ROOT/"results"/dataset/name/"primary_rows.csv")
        table=table[(table.arm=="natural_raw")|(table.method=="confidence")]
        with np.load(source/"scores.npz") as archive:
            scores,tie,row_id=archive["scores"],archive["tie"],archive["row_id"]
        np.testing.assert_array_equal(row_id,labels["row_id"])
        selected={}
        for ci,c in enumerate(criteria):
            for mi,m in enumerate(METHODS):
                q=table[table.method==m]
                if m!="confidence":
                    q=q[q.criterion==c]
                assert len(q)==1
                row=q.iloc[0]
                index=int(row.score_index)
                if m=="confidence" and conf_boot is not None:
                    boot,point=conf_boot,conf_point
                else:
                    if index not in selected:
                        score=scores[index]
                        boot=multiclass_bootstrap(score,truth,first,second,tie,counts,k,base=base)
                        prediction=first.copy()
                        chosen=np.lexsort((tie,-score))[:int(.2*len(truth))]
                        prediction[chosen]=second[chosen]
                        values,_,_,_=from_predictions(truth,prediction,k)
                        point=np.array([values[metric] for metric in METRICS])
                        selected[index]=(boot,point)
                    boot,point=selected[index]
                    if m=="confidence":
                        conf_boot,conf_point=boot,point
                np.testing.assert_allclose(point,row[METRICS].to_numpy(dtype=float),atol=1e-12)
                points[seed,ci,mi]=point
                draws[:,ci,mi]+=boot*weights[:,seed,None]/20
    temporary=destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,points=points,draws=draws,criteria=np.array(criteria),methods=np.array(METHODS),metrics=np.array(METRICS))
    temporary.replace(destination)
    return str(destination)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    complete=json.loads((ROOT/"results/complete.json").read_text())
    output=ROOT/"statistics"
    output.mkdir(exist_ok=True)
    config={"replicates":B,"base_seed":2026111307,"scope":"natural fit1000 fraction0.2 official; class-stratified requests and paired seeds",
        "source_summary_sha256":sha(ROOT/"results/complete.json"),"code_sha256":sha(__file__),
        "bootstrap_sha256":sha(ROOT/"bootstrap.py"),"inference_sha256":sha(OLD_CODE/"followup_stats.py")}
    if (output/"config.json").exists():
        assert json.loads((output/"config.json").read_text())==config
    else:
        save_json(output/"config.json",config)
    manifest=json.loads((ROOT/"labels/manifest.json").read_text())
    tasks=[]
    for offset,dataset in enumerate(DATASETS):
        rng=np.random.default_rng(config["base_seed"]+offset)
        with np.load(ROOT/"labels"/f"{dataset}.npz") as z:
            truth=z["y_true"]
        counts=stratified_counts(truth,rng,B)
        weights=rng.multinomial(20,np.full(20,.05),size=B).astype(np.int16)
        np.savez_compressed(output/f"{dataset}_resampling.npz",counts=counts,seed_weights=weights)
        tasks.extend((dataset,p["pair_id"]) for p in manifest[dataset]["pairs"])
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(one_pair,*task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            print({"pair_complete":i,"total":len(tasks),"path":future.result(),"seconds":round(time.perf_counter()-started,1)},flush=True)
    records,points,columns=[],[],[]
    for dataset,pair_id in tasks:
        with np.load(output/f"{dataset}_p{pair_id:02d}.npz") as z:
            point=z["points"].mean(axis=0)
            draws=z["draws"]
            criteria=z["criteria"].tolist()
        pair=manifest[dataset]["pairs"][pair_id]
        for ci,c in enumerate(criteria):
            for mi,ri in itertools.combinations(range(len(METHODS)),2):
                for metric_id,metric in enumerate(METRICS):
                    value=100*(point[ci,mi,metric_id]-point[ci,ri,metric_id])
                    samples=100*(draws[:,ci,mi,metric_id]-draws[:,ci,ri,metric_id])
                    records.append({"dataset":dataset,"pair_id":pair_id,"weak":pair["weak"],"strong":pair["strong"],
                        "criterion":c,"method":METHODS[mi],"reference":METHODS[ri],"metric":metric,"difference_pp":value})
                    points.append(value)
                    columns.append(samples)
    assert len(records)==3564
    points,columns=np.array(points),np.stack(columns,axis=1)
    result=simultaneous_intervals(points,columns)
    table=pd.DataFrame(records)
    for name in ["se","low","high","adjusted_p"]:
        table[name]=result[name]
    table.to_csv(output/"contrasts.csv",index=False)
    np.savez_compressed(output/"contrast_draws.npz",point=points,draws=columns)
    save_json(output/"complete.json",{"family":len(table),"replicates":B,"critical":result["critical"],
        "contrasts_sha256":sha(output/"contrasts.csv"),"preserved_files":preserve(),"seconds":time.perf_counter()-started})
    print(table.groupby("metric").adjusted_p.agg(total="size",significant=lambda x:int((x<.05).sum())).to_string(),flush=True)


if __name__=="__main__":
    main()
