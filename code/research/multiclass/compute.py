"""Reconstruct frozen cascade class labels; no fitting or retuning."""
import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from common import ROOT,DATASETS,sha,save_json,preserve,scalar_metrics,class_metrics

LABELS,MANIFEST=None,None
METRICS=["accuracy","micro_precision","micro_recall","micro_f1","macro_precision","macro_recall","macro_f1","weighted_precision","weighted_recall","weighted_f1"]


def initialize():
    global LABELS,MANIFEST
    MANIFEST=json.loads((ROOT/"labels/manifest.json").read_text())
    LABELS={}
    for dataset in DATASETS:
        path=ROOT/"labels"/f"{dataset}.npz"
        assert sha(path)==MANIFEST[dataset]["labels_sha256"]
        with np.load(path) as z:
            LABELS[dataset]={key:z[key] for key in z.files}


def condition(dataset,directory):
    directory=Path(directory)
    output=ROOT/"results"/dataset/directory.name
    output.mkdir(parents=True,exist_ok=True)
    expected=json.loads((directory/"complete.json").read_text())["files"]
    for name in ["metrics.csv","scores.npz"]:
        assert sha(directory/name)==expected[name]
    if (output/"complete.json").exists():
        old=json.loads((output/"complete.json").read_text())
        for name,value in old["files"].items():
            assert sha(output/name)==value
        return str(output)
    source=pd.read_csv(directory/"metrics.csv")
    pair_id=int(source.pair_id.iloc[0])
    pair=MANIFEST[dataset]["pairs"][pair_id]
    labels=LABELS[dataset]
    names=labels["expert_names"].tolist()
    truth=labels["y_true"]
    first=labels["predictions"][names.index(pair["weak"])]
    second=labels["predictions"][names.index(pair["strong"])]
    K=MANIFEST[dataset]["classes"]
    with np.load(directory/"scores.npz") as archive:
        z={key:archive[key] for key in ["scores","names","row_id","W","S","tie","clean"]}
    np.testing.assert_array_equal(z["row_id"],labels["row_id"])
    np.testing.assert_array_equal(z["W"],first==truth)
    np.testing.assert_array_equal(z["S"],second==truth)
    np.testing.assert_array_equal(z["clean"],labels["clean"])
    indices={name:i for i,name in enumerate(z["names"].tolist())}
    subsets={}
    for evaluation in ["official","no_overlap"]:
        mask=np.ones(len(truth),bool) if evaluation=="official" else z["clean"]
        y,w,s=truth[mask],first[mask],second[mask]
        subsets[evaluation]={"mask":mask,"truth":y,"first":w,"second":s,"support":np.bincount(y,minlength=K),
            "base_tp":np.bincount(y[w==y],minlength=K),"base_pred":np.bincount(w,minlength=K)}
    cache,rankings,checked={}, {}, set()
    records,primary_rows,tp_rows,pred_rows=[],[],[],[]
    for row in source.itertuples():
        method=row.selected_method if row.method=="selector" else row.method
        if dataset=="banking77":
            key=method if method in ["confidence","random"] else row.arm+"__"+method
            score_index=indices[key]
            criterion="fixed" if row.method in ["confidence","random"] else "mse"
        else:
            score_index=int(row.score_index)
            criterion=row.criterion
        split=subsets[row.evaluation]
        y,w,s,support=split["truth"],split["first"],split["second"],split["support"]
        key=score_index,row.evaluation,row.fraction
        if key not in cache:
            rk=score_index,row.evaluation
            if rk not in rankings:
                rankings[rk]=np.lexsort((z["tie"][split["mask"]],-z["scores"][score_index][split["mask"]]))
            k=int(row.fraction*len(y))
            selected=rankings[rk][:k]
            pw,ps,ys=w[selected],s[selected],y[selected]
            pred=split["base_pred"]-np.bincount(pw,minlength=K)+np.bincount(ps,minlength=K)
            tp=split["base_tp"]-np.bincount(ys[pw==ys],minlength=K)+np.bincount(ys[ps==ys],minlength=K)
            assert np.all(tp>=0) and np.all(pred>=0) and np.all(tp<=pred) and np.all(tp<=support)
            values=scalar_metrics(tp,pred,support)
            assert abs(values["weighted_recall"]-values["accuracy"])<1e-12
            if np.all(support==support[0]):
                assert abs(values["macro_recall"]-values["accuracy"])<1e-12
                assert abs(values["weighted_precision"]-values["macro_precision"])<1e-12
                assert abs(values["weighted_f1"]-values["macro_f1"])<1e-12
            values.update(calls=k,rescues=int(((pw!=ys)&(ps==ys)).sum()),harms=int(((pw==ys)&(ps!=ys)).sum()),
                wrong_to_different_wrong=int(((pw!=ys)&(ps!=ys)&(pw!=ps)).sum()),changed_labels=int((pw!=ps).sum()))
            cache[key]=(values,tp,pred,selected)
        values,tp,pred,selected=cache[key]
        assert abs(values["accuracy"]-row.accuracy)<1e-11
        assert values["calls"]==row.calls and values["rescues"]==row.rescues and values["harms"]==row.harms
        record={"dataset":dataset,"pair_id":pair_id,"weak":pair["weak"],"strong":pair["strong"],
            "seed":row.seed,"fit_n":row.fit_n,"arm":row.arm,"criterion":criterion,"method":row.method,
            "evaluation":row.evaluation,"fraction":row.fraction,"score_index":score_index,**values}
        if row.method=="selector":
            record["selected_method"]=row.selected_method
        records.append(record)
        if row.fit_n==1000 and row.evaluation=="official" and row.fraction==.2:
            if key not in checked:
                prediction=w.copy()
                prediction[selected]=s[selected]
                p,r,f,n=precision_recall_fscore_support(y,prediction,labels=np.arange(K),zero_division=0)
                np.testing.assert_allclose(np.stack(class_metrics(tp,pred,support)),np.stack([p,r,f]),atol=1e-12)
                np.testing.assert_array_equal(n,support)
                checked.add(key)
            primary_rows.append(record)
            tp_rows.append(tp)
            pred_rows.append(pred)
    pd.DataFrame(records).to_csv(output/"metrics.csv",index=False)
    files={"metrics.csv":sha(output/"metrics.csv")}
    if primary_rows:
        pd.DataFrame(primary_rows).to_csv(output/"primary_rows.csv",index=False)
        np.savez_compressed(output/"class_counts.npz",tp=np.asarray(tp_rows,dtype=np.uint16),
            predicted=np.asarray(pred_rows,dtype=np.uint16),support=labels["support"])
        files.update({name:sha(output/name) for name in ["primary_rows.csv","class_counts.npz"]})
    save_json(output/"complete.json",{"rows":len(records),"primary_rows":len(primary_rows),"sklearn_checks":len(checked),
        "source_scores_sha256":expected["scores.npz"],"files":files})
    return str(output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    output=ROOT/"results"
    output.mkdir(exist_ok=True)
    config={"protocol_sha256":sha(ROOT/"PROTOCOL.md"),"code_sha256":sha(__file__),"common_sha256":sha(ROOT/"common.py"),
        "label_manifest_sha256":sha(ROOT/"labels/manifest.json")}
    if (output/"config.json").exists():
        assert json.loads((output/"config.json").read_text())==config
    else:
        save_json(output/"config.json",config)
    paths=[]
    tasks=[(dataset,p.parent) for dataset,spec in DATASETS.items() for p in spec["results"].glob("p*_s*_n*/complete.json")]
    assert len(tasks)==1500
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
        futures=[executor.submit(condition,*task) for task in tasks]
        for future in as_completed(futures):
            paths.append(Path(future.result()))
            if len(paths)%100==0 or len(paths)==len(tasks):
                state={"completed":len(paths),"total":len(tasks),"seconds":round(time.perf_counter()-started,1)}
                save_json(output/"progress.json",state)
                print(state,flush=True)
    totals={}
    for dataset in DATASETS:
        own=sorted(p for p in paths if p.parent.name==dataset)
        metrics=pd.concat([pd.read_csv(p/"metrics.csv") for p in own],ignore_index=True)
        metrics.to_csv(output/dataset/"metrics.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        primary=[p for p in own if (p/"primary_rows.csv").exists()]
        pd.concat([pd.read_csv(p/"primary_rows.csv") for p in primary],ignore_index=True).to_csv(output/dataset/"primary_rows.csv",index=False)
        all_tp,all_pred=[],[]
        for p in primary:
            with np.load(p/"class_counts.npz") as z:
                all_tp.append(z["tp"])
                all_pred.append(z["predicted"])
                support=z["support"].copy()
        np.savez_compressed(output/dataset/"class_counts.npz",tp=np.concatenate(all_tp),predicted=np.concatenate(all_pred),support=support)
        totals[dataset]={"rows":len(metrics),"primary_rows":sum(len(x) for x in all_tp),
            "sklearn_checks":sum(json.loads((p/"complete.json").read_text())["sklearn_checks"] for p in own),
            "metrics_sha256":sha(output/dataset/"metrics.csv.gz"),"class_counts_sha256":sha(output/dataset/"class_counts.npz")}
    save_json(output/"complete.json",{"datasets":totals,"seconds":time.perf_counter()-started,"preserved_files":preserve()})
    print(json.dumps(totals,indent=2),flush=True)


if __name__=="__main__":
    main()
