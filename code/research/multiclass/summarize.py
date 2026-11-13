"""Aggregate scalar and classwise metrics without averaging confusion matrices first."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import json
import numpy as np
import pandas as pd
from common import ROOT,DATASETS,sha,save_json,preserve,class_metrics,from_predictions
from compute import METRICS


def main():
    complete=json.loads((ROOT/"results/complete.json").read_text())
    output=ROOT/"summary"
    output.mkdir(exist_ok=True)
    experts,expert_classes=[],[]
    summaries={}
    all_primary=[]
    for dataset in DATASETS:
        source=ROOT/"results"/dataset
        assert sha(source/"metrics.csv.gz")==complete["datasets"][dataset]["metrics_sha256"]
        m=pd.read_csv(source/"metrics.csv.gz")
        group=["dataset","pair_id","weak","strong","fit_n","arm","criterion","method","evaluation","fraction"]
        fields=METRICS+["wrong_to_different_wrong","changed_labels"]
        summary=m.groupby(group)[fields].agg(["mean","std"])
        summary.columns=["_".join(c) for c in summary.columns]
        summary.to_csv(output/f"{dataset}_all_settings.csv.gz",compression={"method":"gzip","mtime":0})
        primary=pd.read_csv(source/"primary_rows.csv")
        with np.load(ROOT/"labels"/f"{dataset}.npz") as z:
            labels={k:z[k] for k in z.files}
        names=labels["class_names"]
        k=len(names)
        with np.load(source/"class_counts.npz") as z:
            tp,predicted,support=z["tp"],z["predicted"],z["support"]
        assert len(tp)==len(primary)
        precision,recall,f1=class_metrics(tp,predicted,support)
        np.testing.assert_allclose(precision.mean(1),primary.macro_precision,atol=1e-12)
        np.testing.assert_allclose(recall.mean(1),primary.macro_recall,atol=1e-12)
        np.testing.assert_allclose(f1.mean(1),primary.macro_f1,atol=1e-12)
        class_rows=[]
        for keys,q in primary.groupby(["pair_id","arm","criterion","method"]):
            idx=q.index.to_numpy()
            base=dict(zip(["pair_id","arm","criterion","method"],keys))
            pmean,rmean,fmean=[a[idx].mean(0) for a in [precision,recall,f1]]
            psd,rsd,fsd=[a[idx].std(0,ddof=1) for a in [precision,recall,f1]]
            for c,name in enumerate(names):
                class_rows.append({"dataset":dataset,**base,"class_id":c,"class_name":str(name),"support":int(support[c]),
                    "precision_mean":pmean[c],"precision_std":psd[c],"recall_mean":rmean[c],"recall_std":rsd[c],
                    "f1_mean":fmean[c],"f1_std":fsd[c],"tp_mean":tp[idx,c].mean(),
                    "fp_mean":(predicted[idx,c].astype(int)-tp[idx,c]).mean(),"fn_mean":(support[c]-tp[idx,c]).mean(),
                    "predicted_support_mean":predicted[idx,c].mean()})
        pd.DataFrame(class_rows).to_csv(output/f"{dataset}_per_class.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        primary_mean=primary.groupby(["dataset","pair_id","weak","strong","arm","criterion","method"])[fields].agg(["mean","std"])
        primary_mean.columns=["_".join(c) for c in primary_mean.columns]
        primary_mean=primary_mean.reset_index()
        primary_mean.to_csv(output/f"{dataset}_primary.csv",index=False)
        all_primary.append(primary_mean)
        for i,name in enumerate(labels["expert_names"]):
            for evaluation,mask in [("official",np.ones(len(labels["y_true"]),bool)),("no_overlap",labels["clean"])]:
                values,t,p,n=from_predictions(labels["y_true"][mask],labels["predictions"][i][mask],k)
                experts.append({"dataset":dataset,"expert":str(name),"evaluation":evaluation,**values})
                pc,rc,fc=class_metrics(t,p,n)
                for c,label in enumerate(names):
                    expert_classes.append({"dataset":dataset,"expert":str(name),"evaluation":evaluation,"class_id":c,
                        "class_name":str(label),"support":int(n[c]),"precision":pc[c],"recall":rc[c],"f1":fc[c]})
        summaries[dataset]={"scalar_rows":len(m),"primary_rows":len(primary),"classwise_summary_rows":len(class_rows),
            "balanced_official":bool(np.all(support==support[0])),"support_per_class":int(support[0]),
            "max_micro_accuracy_difference":float(np.max(np.abs(m.micro_f1-m.accuracy))),
            "max_weighted_recall_accuracy_difference":float(np.max(np.abs(m.weighted_recall-m.accuracy)))}
        print(dataset,summaries[dataset],flush=True)
    pd.concat(all_primary,ignore_index=True).to_csv(output/"primary_all.csv",index=False)
    pd.DataFrame(experts).to_csv(output/"expert_metrics.csv",index=False)
    pd.DataFrame(expert_classes).to_csv(output/"expert_per_class.csv",index=False)
    save_json(output/"complete.json",{"datasets":summaries,"preserved_files":preserve()})


if __name__=="__main__":
    main()
