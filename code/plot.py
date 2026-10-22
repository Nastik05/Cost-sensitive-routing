"""Plot only: consumes saved metrics, never fits models."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def draw(path):
    figdir, generated = ROOT / "paper/figures", ROOT / "paper/generated"
    figdir.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(path)
    official = metrics[(metrics.evaluation == "official") & (metrics.fraction == .2)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    q = official[official.pair_id == 1]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for method, title in [("confidence", "Confidence"), ("weak_error", "Weak-error"), ("direct", "Direct"), ("events", "Rescue/harm")]:
        summary = q[q.method == method].groupby("fit_n").accuracy.agg(["mean", "std"])
        ax.errorbar(summary.index, 100*(1-summary["mean"]), yerr=summary["std"]*100, marker="o", capsize=3, label=title)
    ax.set(xlabel="Примеров обучения роутера (dev = fit/2)", ylabel="Ошибка каскада, %")
    ax.grid(alpha=.2)
    ax.legend(ncol=2)
    for extension in ["pdf", "png"]:
        fig.savefig(figdir / f"error_curve.{extension}", dpi=180)
    plt.close(fig)
    with np.load(ROOT / "code/data/banking77.npz") as z:
        test = z["split"] == "test"
        W, S = z["correct__tfidf_lr"][test], z["correct__bert_tiny_tuned"][test]
        D = S.astype(int)-W.astype(int)
    total_rescue, total_harm = int((D == 1).sum()), int((D == -1).sum())
    count_rows, matrices = [], []
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.65), constrained_layout=True)
    colors = np.array([[[.85,.94,.91]]*2, [[.94,.95,.97]]*2, [[.98,.88,.86]]*2])
    for ax, method, title in zip(axes, ["confidence", "direct", "events"], ["Confidence", "Direct", "Rescue/harm"]):
        q = official[(official.pair_id == 0) & (official.fit_n == 1000) & (official.method == method)]
        r, h = q.rescues.mean(), q.harms.mean()
        neutral_yes = 616-r-h
        neutral_total = int((D == 0).sum())
        values = np.array([[r, total_harm-h], [neutral_yes, neutral_total-neutral_yes], [h, total_rescue-r]])
        assert np.allclose(values.sum(axis=0), [616,2464])
        matrices.append({"method": method, "values": values.tolist()})
        ax.imshow(colors, aspect="auto", extent=(-.5,1.5,2.5,-.5))
        for i, outcome in enumerate([1,0,-1]):
            count_rows.append({"method":method,"outcome":outcome,"escalate_yes":values[i,0],"escalate_no":values[i,1]})
            for j in range(2):
                ax.text(j,i,f"{values[i,j]:.1f}".replace(".",","),ha="center",va="center",fontsize=17,color="#17324d")
        ax.set_xticks([0,1],["Да","Нет"],fontsize=12)
        ax.xaxis.tick_top()
        ax.set_yticks([0,1,2],["+1","0","−1"],fontsize=14)
        ax.set_title(title,fontsize=14,pad=25)
        ax.set_xlabel("Эскалация",fontsize=11,labelpad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([-.5,.5,1.5],minor=True)
        ax.set_yticks([-.5,.5,1.5,2.5],minor=True)
        ax.grid(which="minor",color="white",linewidth=3)
        ax.tick_params(which="minor",length=0)
    for extension in ["pdf","png"]:
        fig.savefig(figdir / f"decision_profiles.{extension}",dpi=180)
    plt.close(fig)
    pd.DataFrame(count_rows).to_csv(generated / "outcomes.csv",index=False)
    (generated / "outcomes.json").write_text(json.dumps(matrices,indent=2)+"\n")
    q = official[(official.pair_id == 1) & (official.fit_n == 1000)]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Способ & Accuracy, \% & SD, п.п. \\", r"\midrule"]
    for method in ["confidence","weak_error","direct","events"]:
        values = q[q.method == method].accuracy*100
        fmt = lambda x: f"{x:.2f}".replace(".","{,}")
        lines.append(method.replace("_",r"\_")+" & "+fmt(values.mean())+" & "+fmt(values.std())+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated / "accuracy_table.tex").write_text("\n".join(lines)+"\n")
    print("Figures and TeX table generated from verified baseline results",flush=True)
