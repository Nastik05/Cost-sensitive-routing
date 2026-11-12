"""Audit saved models/scores, then produce all follow-up statistics and figures."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import ROOT, CONFIG, digest, predict
from followup import SETTINGS, PREFERENCE, choose_method
from followup_stats import bootstrap_gain, simultaneous_intervals

PAPER = ROOT.parent / "paper"
LABELS = {"confidence": "Confidence", "weak_error": "Weak-error", "direct": "Direct",
          "events": "Events", "selector": "Dev-selector"}
PAIRS = ["TF–IDF → BERT-tiny", "TF–IDF → MiniLM"]


def audit(output, metrics):
    with np.load(ROOT / "data/banking77.npz") as z:
        data = {k: z[k] for k in z.files}
    checked_models, checked_rows = 0, 0
    for pair in range(2):
        weak, strong = CONFIG["pairs"][pair]
        X, W, S = data[f"X__{weak}"], data[f"correct__{weak}"], data[f"correct__{strong}"]
        for seed in SETTINGS["seeds"]:
            for n in SETTINGS["fit_sizes"]:
                directory = output / f"p{pair}_s{seed}_n{n}"
                for name, sha in json.loads((directory / "complete.json").read_text()).items():
                    assert digest(directory / name) == sha
                models = joblib.load(directory / "models.joblib")
                with np.load(directory / "scores.npz") as z:
                    names = z["names"].tolist()
                    test, select, tune = z["test_ids"], z["select_ids"], z["tune_ids"]
                    np.testing.assert_array_equal(z["row_id"], data["row_id"][test])
                    assert not set(test) & (set(select) | set(tune))
                    natural_select = {}
                    for i, name in enumerate(names):
                        regime, correction, method = name.split("__")
                        ids = z[f"fit_ids__{regime}"]
                        assert not set(ids) & (set(test) | set(select) | set(tune))
                        assert (data["split"][ids] == "router_fit").all()
                        if method == "confidence":
                            score, select_score = 1-X[test, 0], 1-X[select, 0]
                        else:
                            scaler, model = models[name]
                            score = predict(model, scaler.transform(X[test]))
                            select_score = predict(model, scaler.transform(X[select]))
                            checked_models += 1
                        np.testing.assert_allclose(score, z["scores"][i], rtol=1e-12, atol=1e-12)
                        if regime == "natural":
                            natural_select[method] = select_score
                    tie_select = np.random.default_rng(740000+pair).random(len(select))
                    part = metrics[(metrics.pair_id == pair) & (metrics.seed == seed) & (metrics.fit_n == n)]
                    for fraction in SETTINGS["fractions"]:
                        chosen = choose_method(natural_select, W[select], S[select], tie_select, fraction)
                        assert chosen == str(z[f"selected__{fraction}"])
                    for row in part[part.evaluation.isin(["official", "no_overlap"])].itertuples():
                        method = row.selected_method if row.method == "selector" else row.method
                        name = f"{row.regime}__{'ipw' if row.corrected else 'raw'}__{method}"
                        score = z["scores"][names.index(name)]
                        mask = np.ones(len(test), bool) if row.evaluation == "official" else z["clean"]
                        D = S[test][mask].astype(int)-W[test][mask]
                        ranked = np.lexsort((z["tie"][mask], -score[mask]))
                        r = np.zeros(len(D), dtype=int)
                        r[ranked[:int(row.fraction*len(D))]] = 1
                        expected = np.mean(W[test][mask]+r*D)
                        assert abs(expected-row.accuracy) < 1e-12
                        assert int(r.sum()) == row.calls
                        assert int(((D == 1) & (r == 1)).sum()) == row.rescues
                        assert int(((D == -1) & (r == 1)).sum()) == row.harms
                        assert abs(np.mean((score[mask]-D)**2)-row.gain_mse) < 1e-12
                        checked_rows += 1
    result = {"models_replayed": checked_models, "test_metric_rows_checked": checked_rows,
              "metrics_sha256": digest(output / "metrics.csv"), "status": "passed"}
    (output / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result, flush=True)


def statistics(output, metrics):
    rng = np.random.default_rng(2026111207)
    B, ns = SETTINGS["bootstrap_replicates"], len(SETTINGS["seeds"])
    counts = rng.multinomial(3080, np.full(3080, 1/3080), size=B).astype(np.int16)
    seed_weights = rng.multinomial(ns, np.full(ns, 1/ns), size=B).astype(np.int16)
    np.savez_compressed(output / "resampling.npz", counts=counts, seed_weights=seed_weights)
    columns, records, points = [], [], []
    for pair in range(2):
        contrasts = [(f"{r}__ipw__{m}", f"{r}__raw__{m}", r, m)
                     for r in ["rescue_rich", "harm_rich", "discordant_rich"] for m in ["direct", "events"]]
        contrasts.append(("selector", "natural__raw__confidence", "natural", "selector"))
        draws = np.zeros((B, len(contrasts)))
        point_seeds = np.empty((ns, len(contrasts)))
        for seed in SETTINGS["seeds"]:
            with np.load(output / f"p{pair}_s{seed}_n1000/scores.npz") as z:
                names = z["names"].tolist()
                D = z["S"].astype(int)-z["W"]
                selected = f"natural__raw__{str(z['selected__0.2'])}"
                needed = {selected if name == "selector" else name for c in contrasts for name in c[:2]}
                boot, point = {}, {}
                for name in needed:
                    score = z["scores"][names.index(name)]
                    boot[name] = bootstrap_gain(score, D, z["tie"], counts)
                    picked = np.lexsort((z["tie"], -score))[:616]
                    point[name] = D[picked].sum()/len(D)
                for j, (a, b, _, _) in enumerate(contrasts):
                    a = selected if a == "selector" else a
                    draws[:, j] += (boot[a]-boot[b])*seed_weights[:, seed]/ns
                    point_seeds[seed, j] = point[a]-point[b]
            if seed % 10 == 9:
                print(f"Bootstrap pair {pair}: {seed+1}/{ns} seeds", flush=True)
        for j, (a, b, regime, method) in enumerate(contrasts):
            records.append({"pair_id": pair, "regime": regime, "method": method, "comparison": a+" minus "+b,
                            "difference_pp": 100*point_seeds[:, j].mean(),
                            "seed_sd_pp": 100*point_seeds[:, j].std(ddof=1)})
            points.append(100*point_seeds[:, j].mean())
            columns.append(100*draws[:, j])
    point, draws = np.array(points), np.stack(columns, axis=1)
    inference = simultaneous_intervals(point, draws)
    table = pd.DataFrame(records)
    for name in ["se", "low", "high", "adjusted_p"]:
        table[name] = inference[name]
    assert len(table) == 14
    # Independently verify bootstrap point contrasts against the CSV aggregation.
    primary = metrics[(metrics.fit_n == 1000) & (metrics.evaluation == "official") & (metrics.fraction == .2)]
    for row in table.itertuples():
        p = primary[(primary.pair_id == row.pair_id) & (primary.regime == row.regime)]
        if row.method == "selector":
            delta = p[p.method == "selector"].accuracy.mean()-p[p.method == "confidence"].accuracy.mean()
        else:
            p = p[p.method == row.method]
            delta = p[p.corrected].accuracy.mean()-p[~p.corrected].accuracy.mean()
        assert abs(100*delta-row.difference_pp) < 1e-10
    table.to_csv(output / "contrasts.csv", index=False)
    np.savez_compressed(output / "bootstrap.npz", point=point, draws=draws)
    (output / "statistics.json").write_text(json.dumps({"replicates": B, "contrasts": 14,
        "critical": inference["critical"], "metrics_sha256": digest(output / "metrics.csv"),
        "code_sha256": digest(ROOT / "followup_stats.py")}, indent=2)+"\n")
    return table


def savefig(fig, name):
    for suffix in ["pdf", "png"]:
        fig.savefig(PAPER / "figures" / f"{name}.{suffix}", dpi=190)
    plt.close(fig)


def tex_table(path, header, rows, columns):
    text = [r"\begin{tabular}{"+columns+"}", r"\toprule", header+r" \\", r"\midrule"]
    text += [" & ".join(row)+r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(text)+"\n")


def render(output, metrics, contrasts):
    generated = PAPER / "generated"
    generated.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 10})
    colors = {"direct": "#257c79", "events": "#bd5b35", "confidence": "#6e7583", "weak_error": "#6160a1", "selector": "#15334f"}
    official = metrics[(metrics.evaluation == "official") & (metrics.fraction == .2)]
    natural = metrics[(metrics.regime == "natural") & (metrics.fraction == .2)]
    summary = metrics.groupby(["pair_id", "regime", "corrected", "method", "fit_n", "evaluation", "fraction"])[
        ["accuracy", "gain_mse", "rescues", "harms"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(generated / "followup_summary.csv")
    metrics.to_csv(generated / "followup_metrics.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    contrasts.to_csv(generated / "followup_contrasts.csv", index=False)
    sensitivity = metrics[(metrics.fit_n == 1000) & (metrics.regime == "harm_rich") &
                          metrics.evaluation.isin(["official", "no_overlap"])]
    sensitivity = sensitivity.groupby(["pair_id", "method", "fraction", "evaluation", "corrected"]).accuracy.mean().unstack("corrected")
    sensitivity["difference_pp"] = 100*(sensitivity[True]-sensitivity[False])
    sensitivity.to_csv(generated / "followup_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    for pair, ax in enumerate(axes):
        for method in ["direct", "events"]:
            for split, style, title in [("train", "--", "train"), ("official", "-", "test")]:
                q = natural[(natural.pair_id == pair) & (natural.method == method) & (natural.evaluation == split)]
                s = q.groupby("fit_n").gain_mse.agg(["mean", "std"])
                ax.plot(s.index, s["mean"], style, color=colors[method], marker="o", label=f"{LABELS[method]}: {title}")
                ax.fill_between(s.index, s["mean"]-s["std"], s["mean"]+s["std"], color=colors[method], alpha=.08)
        ax.set(title=PAIRS[pair], xlabel="Число извлечений fit", ylabel="MSE прироста качества")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    savefig(fig, "learning_train_test")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    for pair, ax in enumerate(axes):
        for method in PREFERENCE+["selector"]:
            q = official[(official.pair_id == pair) & (official.regime == "natural") & (official.method == method)]
            s = q.groupby("fit_n").accuracy.agg(["mean", "std"])
            ax.errorbar(s.index, 100*(1-s["mean"]), yerr=100*s["std"], marker="o", capsize=3,
                        color=colors[method], label=LABELS[method], linestyle="--" if method == "selector" else "-")
        ax.set(title=PAIRS[pair], xlabel="Число извлечений fit", ylabel="Ошибка каскада, %")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    savefig(fig, "followup_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    regimes = ["natural", "rescue_rich", "harm_rich", "discordant_rich"]
    for pair, ax in enumerate(axes):
        for method in ["direct", "events"]:
            for corrected, style in [(False, "--"), (True, "-")]:
                means, sd = [], []
                for regime in regimes:
                    q = official[(official.pair_id == pair) & (official.fit_n == 1000) & (official.method == method) &
                                 (official.regime == regime) & (official.corrected == (corrected and regime != "natural"))]
                    means.append(100*q.accuracy.mean())
                    sd.append(100*q.accuracy.std())
                ax.errorbar(np.arange(4), means, yerr=sd, color=colors[method], linestyle=style, marker="o", capsize=3,
                            label=LABELS[method]+(" + IPW" if corrected else ": без коррекции"))
        ax.set_xticks(np.arange(4), ["Natural", "Rescue ×4", "Harm ×8", "Both ×4"])
        ax.set(title=PAIRS[pair], ylabel="Accuracy, %")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    savefig(fig, "sampling_correction")
    # Contrasts are the entire predeclared family, including the selector controls.
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    for i, row in enumerate(contrasts.itertuples()):
        color = "#257c79" if row.low > 0 or row.high < 0 else "#7e8792"
        ax.errorbar(row.difference_pp, i, xerr=[[row.difference_pp-row.low], [row.high-row.difference_pp]],
                    color=color, fmt="o", capsize=3)
    labels = [("BERT" if r.pair_id == 0 else "MiniLM")+" / "+r.regime.replace("_rich", "")+" / "+LABELS[r.method]
              for r in contrasts.itertuples()]
    ax.set_yticks(range(len(contrasts)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Разность accuracy, п.п.; одновременные 95% интервалы")
    ax.grid(axis="x", alpha=.2)
    savefig(fig, "correction_intervals")
    selector = official[(official.regime == "natural") & (official.method == "selector")]
    counts = selector.groupby(["pair_id", "fit_n", "selected_method"]).size().rename("count").reset_index()
    counts.to_csv(generated / "selection_counts.csv", index=False)
    regret = selector.groupby(["pair_id", "fit_n"]).agg(mean_regret=("regret", "mean"),
        sd_regret=("regret", "std"), max_regret=("regret", "max"), oracle_accuracy=("oracle_accuracy", "mean"))
    regret.to_csv(generated / "selection_regret.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    for pair, ax in enumerate(axes):
        bottom = np.zeros(3)
        for method in PREFERENCE:
            vals = [len(selector[(selector.pair_id == pair) & (selector.fit_n == n) & (selector.selected_method == method)]) for n in [200, 500, 1000]]
            ax.bar(np.arange(3), vals, bottom=bottom, label=LABELS[method], color=colors[method])
            bottom += vals
        ax.set_xticks(np.arange(3), ["200", "500", "1000"])
        ax.set(title=PAIRS[pair], xlabel="Число извлечений fit", ylabel="Число выборов из 30")
        ax.legend(fontsize=8, ncol=2)
    savefig(fig, "method_selection")
    fmt = lambda v: f"{v:.2f}".replace(".", "{,}")
    rows = []
    for pair in range(2):
        for method in PREFERENCE+["selector"]:
            q = official[(official.pair_id == pair) & (official.fit_n == 1000) & (official.regime == "natural") & (official.method == method)]
            mse = "---" if method in ["confidence", "selector"] else (
                f"${q.gain_mse.mean():.4f}"+r"\pm"+f"{q.gain_mse.std():.4f}$")
            rows.append(["BERT" if pair == 0 else "MiniLM", LABELS[method],
                         "$"+fmt(100*q.accuracy.mean())+r"\pm"+fmt(100*q.accuracy.std())+"$",
                         mse, fmt(q.rescues.mean()), fmt(q.harms.mean())])
    tex_table(generated / "followup_natural_table.tex", r"Второй & Правило & Accuracy, \% & MSE & Rescue & Harm", rows, "llrrrr")
    rows = []
    for row in contrasts.itertuples():
        title = "Selector" if row.method == "selector" else row.regime.replace("_rich", "")+" / "+LABELS[row.method]
        rows.append(["BERT" if row.pair_id == 0 else "MiniLM", title, fmt(row.difference_pp),
                     "["+fmt(row.low)+"; "+fmt(row.high)+"]", "$<0.001$" if row.adjusted_p < .001 else f"{row.adjusted_p:.3f}"])
    tex_table(generated / "followup_contrasts_table.tex", r"Второй & Контраст & $\Delta$, п.п. & 95\% интервал & $p_{adj}$", rows, "llrrr")
    # A small table for a slide, with both pairs rather than a selected best case.
    rows = []
    for method in ["confidence", "weak_error", "direct", "events", "selector"]:
        vals = []
        for pair in range(2):
            q = official[(official.pair_id == pair) & (official.fit_n == 1000) & (official.regime == "natural") & (official.method == method)]
            vals.append("$"+fmt(100*q.accuracy.mean())+r"\pm"+fmt(100*q.accuracy.std())+"$")
        rows.append([LABELS[method], *vals])
    tex_table(generated / "followup_slide_table.tex", r"Правило & BERT-tiny & MiniLM", rows, "lrr")
    fits = pd.read_csv(output / "fits.csv")
    fits.groupby(["pair_id", "fit_n", "regime", "corrected", "method"])[["unique_n", "ess", "fit_rescue", "fit_harm", "fit_seconds"]].agg(["mean", "std"]).to_csv(generated / "followup_fit_summary.csv")
    report = ["# Контроль отбора и выбора маршрутизатора", "", "## Natural, 1000 извлечений; 20% вызовов", "",
              official[(official.fit_n == 1000) & (official.regime == "natural")].groupby(["pair_id", "method"])[["accuracy", "gain_mse", "rescues", "harms"]].agg(["mean", "std"]).round(5).to_markdown(),
              "", "## Все 14 контрастов общей семьи", "", contrasts.round(5).to_markdown(index=False),
              "", "## Потеря выбора по dev", "", regret.round(5).to_markdown(),
              "", "## Частоты выбора", "", counts.to_markdown(index=False), "",
              "Средние/SD условны на фиксированные пулы и экспертов. Интервалы — exploratory crossed bootstrap с пересчётом top-k и общей max-t поправкой; не независимая проверка на новом датасете."]
    (output / "REPORT.md").write_text("\n".join(report)+"\n")
    (generated / "followup_manifest.json").write_text(json.dumps({"config": json.loads((output / "config.json").read_text()),
        "audit": json.loads((output / "audit.json").read_text()), "statistics": json.loads((output / "statistics.json").read_text()),
        "analysis_code_sha256": digest(__file__)}, indent=2)+"\n")
    print("Generated follow-up tables, figures and report", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/followup")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    metrics = pd.read_csv(args.output / "metrics.csv")
    if not args.plots_only:
        audit(args.output, metrics)
        contrasts = statistics(args.output, metrics)
    else:
        info = json.loads((args.output / "statistics.json").read_text())
        assert info["metrics_sha256"] == digest(args.output / "metrics.csv")
        assert info["code_sha256"] == digest(ROOT / "followup_stats.py")
        contrasts = pd.read_csv(args.output / "contrasts.csv")
    render(args.output, metrics, contrasts)


if __name__ == "__main__":
    main()
