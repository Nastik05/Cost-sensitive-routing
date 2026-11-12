"""Describe the full grid and interpret the prespecified simultaneous contrasts."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from imbalance_study import ROOT, METHODS, PREFERENCE, ARMS, digest, write_json

PAPER = ROOT.parent/"paper"
LABEL = {"weak_error": "Weak-error", "direct": "Direct", "marginal": "Marginal", "events": "Events",
         "ternary": "Ternary", "joint": "Joint", "hurdle": "Hurdle", "soft_gain": "Soft-gain", "confidence": "Confidence", "random": "Random", "selector": "Dev-selector"}
SHORT = {"tfidf_lr": "TFIDF", "bert_tiny_tuned": "BERT", "minilm_frozen_lr": "MiniLM-f", "minilm_tuned": "MiniLM-t",
         "tfidf_1_per_intent": "TF1", "tfidf_3_per_intent": "TF3", "tfidf_10_per_intent": "TF10"}
RULES = METHODS+["confidence"]
COLORS = dict(zip(RULES, ["#6b5ca5", "#217d78", "#ba743b", "#b6444b", "#cf9c31", "#4f78a4", "#699236", "#b169a6", "#4d5661"]))


def savefig(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(PAPER/"figures"/f"imbalance_{name}.{extension}", dpi=180)
    plt.close(fig)


def interpret(contrasts):
    t = contrasts.copy()
    t["significant"] = (t.low > 0) | (t.high < 0)
    counts = t.groupby("kind").agg(total=("significant", "size"), significant=("significant", "sum"))
    gain = t[t.gain_only].groupby("kind").agg(gain_total=("significant", "size"), gain_significant=("significant", "sum"))
    counts = counts.join(gain).fillna(0).astype(int)
    simple = t[t.kind == "simple"].set_index(["pair_id", "arm", "method", "reference"])
    interactions = []
    for row in t[t.kind == "interaction"].itertuples():
        natural = simple.loc[(row.pair_id, "natural_raw", row.method, row.reference)]
        enriched = simple.loc[(row.pair_id, row.arm, row.method, row.reference)]
        record = row._asdict()
        record.update(natural_difference_pp=natural.difference_pp, natural_low=natural.low, natural_high=natural.high,
                      enriched_difference_pp=enriched.difference_pp, enriched_low=enriched.low, enriched_high=enriched.high,
                      point_ranking_flip=natural.difference_pp*enriched.difference_pp < -1e-12,
                      supported_crossover=((natural.low > 0 and enriched.high < 0) or (natural.high < 0 and enriched.low > 0)))
        interactions.append(record)
    interactions = pd.DataFrame(interactions).drop(columns="Index")
    profiles = []
    prof = t[t.kind == "profile"].set_index(["pair_id", "method", "reference", "metric"])
    for (pair, arm, method, reference), row in simple.iterrows():
        if arm != "natural_raw":
            continue
        rescue = prof.loc[(pair, method, reference, "rescue_precision")]
        harm = prof.loc[(pair, method, reference, "harm_rate")]
        profiles.append({"pair_id": pair, "method": method, "reference": reference, "gain_only": bool(row.gain_only),
                         "accuracy_difference_pp": row.difference_pp, "accuracy_low": row.low, "accuracy_high": row.high,
                         "accuracy_significant": bool(row.significant),
                         "rescue_precision_difference_pp": rescue.difference_pp, "rescue_low": rescue.low, "rescue_high": rescue.high,
                         "rescue_significant": bool(rescue.significant),
                         "harm_rate_difference_pp": harm.difference_pp, "harm_low": harm.low, "harm_high": harm.high,
                         "harm_significant": bool(harm.significant),
                         "profile_without_accuracy_detection": not row.significant and (rescue.significant or harm.significant)})
    return counts, interactions, pd.DataFrame(profiles)


def figures(primary, contrasts, mean, pairs):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 10})
    pair_labels = [f"{p['pair_id']:02d}: {SHORT[p['weak']]} → {SHORT[p['strong']]}" for p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for pair, ax in enumerate(axes):
        x = np.arange(len(RULES))
        for metric, offset, color, title in [("rescue_precision", -.18, "#217d78", "Rescue precision"), ("harm_rate", .18, "#b6444b", "Harm rate")]:
            means, sd = [], []
            for m in RULES:
                q = primary[(primary.pair_id == pair) & (primary.arm == ("baseline" if m == "confidence" else "natural_raw")) & (primary.method == m)]
                means.append(100*q[metric].mean())
                sd.append(100*q[metric].std())
            ax.bar(x+offset, means, width=.35, yerr=sd, color=color, alpha=.85, label=title, capsize=2)
        ax.set_xticks(x, [LABEL[m] for m in RULES], rotation=50, ha="right", fontsize=8)
        ax.set(title=pair_labels[pair], ylabel="Доля среди вызовов, %")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
    savefig(fig, "natural_profiles")

    def value(pair, arm, method):
        return mean.loc[(pair, "baseline" if method == "confidence" else arm, method), "accuracy"]

    for regime in ["rescue8", "harm8"]:
        matrix = np.array([[100*(value(p, regime+"_raw", m)-value(p, "natural_raw", m)) for m in RULES] for p in range(21)])
        vmax = max(1., np.max(np.abs(matrix)))
        fig, ax = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
        image = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(RULES)), [LABEL[m] for m in RULES], rotation=35, ha="right")
        ax.set_yticks(range(21), pair_labels, fontsize=8)
        for i in range(21):
            for j in range(len(RULES)):
                ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i,j]) > .6*vmax else "black")
        ax.axhline(5.5, color="black", linewidth=1.5)
        ax.set_title(f"{regime}: изменение accuracy относительно natural, п.п.")
        fig.colorbar(image, ax=ax, shrink=.7)
        savefig(fig, regime+"_effect")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, arm in zip(axes, ["rescue8_raw", "harm8_raw"]):
        mat = np.zeros((9, 9), dtype=int)
        t = contrasts[(contrasts.kind == "interaction") & (contrasts.arm == arm)]
        for m in RULES:
            for r in RULES:
                q = t[((t.method == m) & (t.reference == r)) | ((t.method == r) & (t.reference == m))]
                mat[RULES.index(m), RULES.index(r)] = ((q.low > 0) | (q.high < 0)).sum()
        image = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=21)
        ax.set_xticks(range(9), [LABEL[m] for m in RULES], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(9), [LABEL[m] for m in RULES], fontsize=8)
        for i in range(9):
            for j in range(9):
                if i != j:
                    ax.text(j, i, str(mat[i,j]), ha="center", va="center", fontsize=8,
                            color="white" if mat[i,j] > 11 else "black")
        ax.set_title(arm.replace("_raw", ""))
    fig.colorbar(image, ax=axes, shrink=.7, label="Пар с различимым взаимодействием из 21")
    savefig(fig, "interaction_counts")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    for pair in [0, 1]:
        for col, regime in enumerate(["rescue", "harm"]):
            ax = axes[pair, col]
            for method in RULES:
                y = [100*value(pair, arm, method) for arm in ["natural_raw", regime+"2_raw", regime+"8_raw"]]
                ax.plot([1, 2, 8], y, marker="o", color=COLORS[method], label=LABEL[method], linewidth=1.4)
            ax.set(xscale="log", title=pair_labels[pair]+" / "+regime, xlabel="Множитель отбора", ylabel="Accuracy, %")
            ax.set_xticks([1, 2, 8], ["1", "2", "8"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
    savefig(fig, "dose_response")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, regime in zip(axes, ["rescue8", "harm8"]):
        natural, raw, corrected = [], [], []
        for pair in range(21):
            for dest, arm in [(natural, "natural_raw"), (raw, regime+"_raw"), (corrected, regime+"_ipw")]:
                vals = [100*value(pair, arm, m) for m in METHODS[1:]]
                dest.append(max(vals)-min(vals))
        for i in range(21):
            ax.plot([0,1,2], [natural[i], raw[i], corrected[i]], "o-", color="#217d78" if i < 6 else "#b67947", alpha=.4, markersize=3)
        ax.set_xticks([0,1,2], ["Natural", "Raw", "IPW"])
        ax.set(title=regime, ylabel="Размах средних 7 gain-методов, п.п.")
        ax.grid(alpha=.2)
    savefig(fig, "gain_spread")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/imbalance_all")
    args = parser.parse_args()
    output = args.output
    destination = PAPER/"imbalance_results"
    destination.mkdir(parents=True, exist_ok=True)
    audit = json.loads((output/"audit.json").read_text())
    assert audit["status"] == "passed" and audit["models_replayed"] == 110880
    assert audit["metrics_sha256"] == digest(output/"metrics.csv")
    metrics = pd.read_csv(output/"metrics.csv")
    fits = pd.read_csv(output/"fits.csv")
    contrasts = pd.read_csv(output/"statistics/contrasts.csv")
    pairs = json.loads((ROOT/"data/imbalance_manifest.json").read_text())["pairs"]
    counts, interactions, profiles = interpret(contrasts)
    simple = contrasts[contrasts.kind == "simple"].copy()
    simple["significant"] = (simple.low > 0) | (simple.high < 0)
    simple_counts = simple.groupby("arm").agg(total=("significant", "size"), significant=("significant", "sum"))
    simple_counts = simple_counts.join(simple[simple.gain_only].groupby("arm").agg(gain_total=("significant", "size"), gain_significant=("significant", "sum")))
    simple_counts.to_csv(destination/"simple_counts.csv")
    fields = ["accuracy", "gain_mse", "rescues", "harms", "rescue_precision", "rescue_recall", "rescue_f1", "harm_rate", "harm_recall", "harm_avoidance"]
    summary = metrics.groupby(["pair_id", "arm", "method", "fit_n", "evaluation", "fraction"])[fields].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(destination/"summary.csv.gz", compression={"method": "gzip", "mtime": 0})
    # Per-seed results are also published, including all sensitivity slices.
    metrics.to_csv(destination/"metrics.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    fits.to_csv(destination/"fits.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(pairs).to_csv(destination/"pairs.csv", index=False)
    contrasts.to_csv(destination/"contrasts.csv", index=False)
    counts.to_csv(destination/"family_summary.csv")
    interactions.to_csv(destination/"interactions.csv", index=False)
    profiles.to_csv(destination/"natural_profiles.csv", index=False)
    primary = metrics[(metrics.fit_n == 1000) & (metrics.fraction == .2) & (metrics.evaluation == "official")]
    mean = primary.groupby(["pair_id", "arm", "method"])[fields].mean()
    primary.groupby(["pair_id", "arm", "method"])[fields].agg(["mean", "std"]).to_csv(destination/"primary_summary.csv")
    spread_rows = []
    for pair in range(21):
        for arm in ARMS:
            vals = mean.loc[(pair, arm)].reindex(METHODS[1:]).accuracy*100
            spread_rows.append({"pair_id": pair, "arm": arm, "gain_range_pp": vals.max()-vals.min(),
                                "best_mean_method": vals.idxmax(), "worst_mean_method": vals.idxmin()})
    spreads = pd.DataFrame(spread_rows)
    spreads.to_csv(destination/"gain_method_spread.csv", index=False)
    selector = metrics[metrics.method == "selector"]
    selection_summary = selector.groupby(["pair_id", "arm", "fit_n", "evaluation", "fraction"]).agg(
        accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"), mean_regret=("regret", "mean"), max_regret=("regret", "max"))
    selection_summary.to_csv(destination/"selection_summary.csv")
    chosen = pd.read_csv(output/"selections.csv")
    chosen.groupby(["pair_id", "arm", "fit_n", "fraction", "selected_method"]).size().rename("count").to_csv(destination/"selection_counts.csv")
    significant = interactions[interactions.significant]
    supported_crossovers = interactions[interactions.supported_crossover]
    profile_cases = profiles[profiles.profile_without_accuracy_detection]
    # Confidence is identical across training compositions. These already
    # adjusted interactions are exactly each learned method's raw-minus-natural
    # accuracy change, not an additional post-hoc family.
    changes = interactions[interactions.reference == "confidence"].copy()
    changes["improvement"] = changes.low > 0
    changes["degradation"] = changes.high < 0
    changes.to_csv(destination/"raw_vs_natural.csv", index=False)
    changes_summary = changes.groupby(["arm", "method"])[["improvement", "degradation"]].sum()
    changes_summary.to_csv(destination/"raw_vs_natural_counts.csv")
    facts = {"models": audit["models_replayed"], "metric_rows": len(metrics), "family": counts.to_dict(orient="index"),
             "simple_by_arm": simple_counts.to_dict(orient="index"),
             "interaction_pairs": int(significant.pair_id.nunique()),
             "gain_interaction_pairs": int(significant[significant.gain_only].pair_id.nunique()),
             "supported_crossovers": len(supported_crossovers),
             "gain_supported_crossovers": int(supported_crossovers.gain_only.sum()),
             "profile_without_accuracy_detection": len(profile_cases),
             "gain_profile_without_accuracy_detection": int(profile_cases.gain_only.sum()),
             "raw_improvements_over_natural": int(changes.improvement.sum()),
             "raw_degradations_from_natural": int(changes.degradation.sum()),
             "median_gain_range_pp": spreads.groupby("arm").gain_range_pp.median().to_dict()}
    write_json(destination/"findings.json", facts)
    write_json(destination/"manifest.json", {"run": json.loads((output/"complete.json").read_text()), "audit": audit,
        "statistics": json.loads((output/"statistics/complete.json").read_text()),
        "run_config": json.loads((output/"config.json").read_text()), "analysis_code_sha256": digest(__file__)})
    figures(primary, contrasts, mean, pairs)
    contrast_cols = ["pair_id", "arm", "method", "reference", "difference_pp", "low", "high", "adjusted_p"]
    def md(frame):
        return frame.round(4).to_markdown(index=False) if len(frame) else "Статистически различимых случаев в заданной семье не обнаружено."
    natural_table = []
    for pair in [0, 1, 5, 7]:
        for method in RULES:
            row = mean.loc[(pair, "baseline" if method == "confidence" else "natural_raw", method)]
            natural_table.append({"pair": pair, "method": method, "accuracy_pct": 100*row.accuracy,
                "rescue_precision_pct": 100*row.rescue_precision, "rescue_recall_pct": 100*row.rescue_recall,
                "harm_rate_pct": 100*row.harm_rate})
    means_table = []
    for pair in [0, 1, 5, 7]:
        for method in RULES:
            record = {"pair": pair, "method": method}
            for arm in ["natural_raw", "rescue8_raw", "harm8_raw", "rescue8_ipw", "harm8_ipw"]:
                record[arm] = 100*mean.loc[(pair, "baseline" if method == "confidence" else arm, method), "accuracy"]
            means_table.append(record)
    pd.DataFrame(means_table).to_csv(destination/"example_accuracy.csv", index=False)
    # Compact tables for the manuscript; the full numeric results remain above.
    table = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{TF--IDF $\to$ BERT-tiny} & \multicolumn{3}{c}{TF--IDF $\to$ MiniLM} \\",
             r"Метод & Natural & Rescue8 & Harm8 & Natural & Rescue8 & Harm8 \\", r"\midrule"]
    markdown_table = []
    for method in RULES:
        values = [100*mean.loc[(pair, "baseline" if method == "confidence" else arm, method), "accuracy"]
                  for pair in [0, 1] for arm in ["natural_raw", "rescue8_raw", "harm8_raw"]]
        table.append(LABEL[method]+" & "+" & ".join(f"{v:.2f}" for v in values)+r" \\")
        markdown_table.append([LABEL[method], *[f"{v:.2f}" for v in values]])
    table += [r"\bottomrule", r"\end{tabular}"]
    (PAPER/"generated/imbalance_accuracy_table.tex").write_text("\n".join(table)+"\n")
    (PAPER/"generated/imbalance_accuracy_table.md").write_text(pd.DataFrame(markdown_table,
        columns=["Метод", "BERT Natural", "BERT Rescue8", "BERT Harm8", "MiniLM Natural", "MiniLM Rescue8", "MiniLM Harm8"]).to_markdown(index=False)+"\n")
    example = [r"\begin{tabular}{lrr}", r"\toprule", r"Метод & Natural, \% & Harm8, \% \\", r"\midrule"]
    for method in METHODS[1:]:
        vals = [100*mean.loc[(14, arm, method), "accuracy"] for arm in ["natural_raw", "harm8_raw"]]
        example.append(LABEL[method]+" & "+" & ".join(f"{v:.2f}" for v in vals)+r" \\")
    example += [r"\bottomrule", r"\end{tabular}"]
    (PAPER/"generated/imbalance_gain_example_table.tex").write_text("\n".join(example)+"\n")
    nat_sig = int(simple_counts.loc["natural_raw", "significant"])
    nat_gain_sig = int(simple_counts.loc["natural_raw", "gain_significant"])
    summary_tex = (
        f"После общей поправки в natural обнаружено {nat_sig} из 756 попарных accuracy-различий "
        f"({nat_gain_sig} из 441 между семью gain-оценивателями). "
        f"При этом в {len(profile_cases)} сравнениях различим профиль спасений/вреда без обнаруженного "
        "accuracy-различия; это не доказательство эквивалентности accuracy.\n\n"
        f"Взаимодействие метода и состава поддержано в {int(counts.loc['interaction','significant'])} "
        f"из 1512 контрастов, в том числе в {int(counts.loc['interaction','gain_significant'])} из 882 "
        f"между семью gain-оценивателями. Оно встречается в {facts['interaction_pairs']} из 21 пары "
        f"(gain-only: {facts['gain_interaction_pairs']}). Эти пары зависимы и не являются независимыми задачами.\n\n"
        f"Поддержанных смен знака преимущества, при которых оба простых одновременных интервала "
        f"лежат по разные стороны нуля, получено {facts['supported_crossovers']} "
        f"(gain-only: {facts['gain_supported_crossovers']}). Значимое взаимодействие само по себе "
        "не требует смены победителя и не заменяет проверку простых эффектов.\n\n"
        f"Из 336 изменений отдельных методов от natural при rescue8/harm8 "
        f"{facts['raw_improvements_over_natural']} являются улучшениями и "
        f"{facts['raw_degradations_from_natural']} --- ухудшениями после той же общей поправки. "
        "Эти проверки уже входят в семью как взаимодействия с неизменным confidence.\n")
    (PAPER/"generated/imbalance_summary.tex").write_text(summary_tex)
    report = ["# Выбор маршрутизатора при дисбалансе парных исходов", "",
        "Главный вопрос: меняет ли состав обучения относительное качество разных способов обучения роутера? Коррекция отбора — дополнительный контроль механизма, а не замена этой задачи.", "",
        "## Что проверено", "",
        "Все 8 обучаемых методов (7 gain-оценивателей и weak-error), confidence и random; все 21 прежняя пара семи фиксированных экспертов. 3 размера fit, 7 составов, 4 дополнительных IPW-варианта, 20 seeds, 3 квоты, official/clean test. Обучены 110880 выбранных роутеров после 443520 candidate fits. Пулы, эксперты и тест уже использовались раньше; это exploratory контроль, не независимая репликация.", "",
        "[Протокол до запуска](../../code/PROTOCOL_IMBALANCE.md). [Состав 21 пары](pairs.csv). Основная статистическая область: fit1000, квота20%, official test3080. 2000 crossed paired bootstrap, exact top-k; одна общая max-t поправка на 5691 контраст.", "",
        "## Результаты всей статистической семьи", "", counts.to_markdown(), "",
        "`simple` — попарная accuracy в natural/rescue8/harm8; `interaction` — изменение попарной разности относительно natural; `profile` — rescue precision/harm rate в natural; `correction` — IPW−raw; `selector` — dev-выбор−confidence. `gain_*` исключает weak-error и confidence. Количества относятся к зависимым конфигурациям одного датасета.", "",
        "Попарные accuracy-сравнения по составам:", "", simple_counts.to_markdown(), "",
        f"Различимые взаимодействия обнаружены в {facts['interaction_pairs']} из 21 пары; между семью gain-методами — в {facts['gain_interaction_pairs']}. Поддержанная смена знака преимущества (оба простых одновременных интервала по разные стороны нуля): {facts['supported_crossovers']}, из них gain-only {facts['gain_supported_crossovers']}.", "",
        "![Число взаимодействий](../figures/imbalance_interaction_counts.png)", "",
        "## Близкая accuracy и разные профили решений", "",
        f"В {len(profile_cases)} сравнениях natural различим хотя бы один компонент профиля, тогда как accuracy-контраст не обнаружен; среди gain-only — {facts['gain_profile_without_accuracy_detection']}. Отсутствие значимости accuracy не доказывает её равенства или эквивалентности.", "",
        "Precision спасений — доля спасений среди эскалаций; recall — доля найденных спасений среди всех доступных. При фиксированной квоте эти показатели зависят от одного числа спасений и не являются независимыми подтверждениями. Accuracy определяется спасениями за вычетом вреда. Поэтому анализируется и harm rate.", "",
        "![Профили natural](../figures/imbalance_natural_profiles.png)", "",
        md(pd.DataFrame(natural_table)), "",
        "Полные интервалы профилей всех пар/методов: [natural_profiles.csv](natural_profiles.csv). Четыре приведённые пары: два исходных примера, две близкие MiniLM и low-label TF1→MiniLM-f; это иллюстрации, а не независимые задачи.", "",
        "## Что меняется при дисбалансе", "",
        "![Harm](../figures/imbalance_harm8_effect.png)", "",
        "![Rescue](../figures/imbalance_rescue8_effect.png)", "",
        "Ячейки показывают изменение среднего одного метода от natural, а не самостоятельные тесты его превосходства. Проверка неодинакового влияния на методы — контрасты `interaction`.", "",
        "![Дозы](../figures/imbalance_dose_response.png)", "",
        "Средняя accuracy при множителях 1/2/8 для двух исходных пар. Полные SD, размеры 200/500/1000 и квоты 10/20/40% доступны в численных результатах.", "",
        "### Улучшение или ухудшение относительно natural", "",
        "Confidence не зависит от состава обучения, поэтому interaction каждого метода с confidence в точности равен его изменению accuracy от natural. Ниже число улучшений/ухудшений с той же общей поправкой, из 21 пары для каждой строки. Это позволяет отделить полезную обработку данных от ухудшения чувствительного метода.", "",
        changes_summary.to_markdown(), "",
        "### Наибольшие взаимодействия между gain-методами", "",
        "Ниже приводятся до 12 различимых контрастов, отсортированных после анализа по абсолютному эффекту, если такие обнаружены. Применяется одна общая поправка; полный список опубликован независимо от значимости.", "",
        md(significant[significant.gain_only].assign(magnitude=lambda x: x.difference_pp.abs()).sort_values("magnitude", ascending=False).iloc[:12][contrast_cols]), "",
        "### Простые различия между gain-методами в обогащённых режимах", "",
        md(simple[simple.gain_only & simple.significant][contrast_cols]), "",
        "Значимость разности в одном режиме не равна значимости её изменения относительно natural. Поэтому эти строки не подменяют тест взаимодействия. Пара 14 — TF10→полный TFIDF; её включение в обсуждение обусловлено результатом полной семьи, а не новым подбором тестов.", "",
        "### Поддержанные смены предпочтительности", "",
        md(supported_crossovers.iloc[:16][["pair_id", "arm", "method", "reference", "natural_difference_pp", "natural_low", "natural_high", "enriched_difference_pp", "enriched_low", "enriched_high"]]), "",
        "Различимое взаимодействие может менять величину преимущества без смены победителя. Простая смена порядка средних также не равна статистически поддержанной смене предпочтительности.", "",
        "## Коррекция и выбор метода", "",
        "Медиана размаха средних семи gain-методов по 21 зависимой паре (п.п.; описательная сводка):", "",
        spreads.groupby("arm").gain_range_pp.agg(["median", "min", "max"]).round(4).to_markdown(), "",
        md(pd.DataFrame(means_table)), "",
        "![Размах gain-методов](../figures/imbalance_gain_spread.png)", "",
        "Каждая линия — одна зависимая пара; показан размах средних семи gain-методов. Этот график описательный. IPW компенсирует известный отбор в функции потерь, но не гарантирует уменьшение размаха или превосходство над natural. У hurdle отдельно корректируется условное обучение направления.", "",
        "Selector выбирает среди восьми методов и confidence на отдельных 500 dev-примерах, без test и без выбора состава по test. [Частоты выбора](selection_counts.csv), [accuracy и regret](selection_summary.csv). Test-oracle — ретроспективный ориентир среди тех же fitted candidates, не deployable метод.", "",
        "## Воспроизводимость и границы вывода", "",
        f"Аудит воспроизвёл {audit['models_replayed']} моделей и независимо проверил {audit['metric_rows_checked']} строк метрик. [Хеши и окружение](manifest.json). Все исходные результаты сохранены; новый протокол не переписывает прошлые сетки.", "",
        "- [Все 5691 контраст](contrasts.csv), [взаимодействия с простыми эффектами](interactions.csv).",
        "- [Все per-seed метрики](metrics.csv.gz), [средние/SD всех разрезов](summary.csv.gz), [обучающие характеристики](fits.csv.gz).",
        "- Дополнительные составы neutral/discordant, меньшие объёмы, другие квоты и clean-test — описательные проверки, не новые постфактум семейства значимости.",
        "- Различия параметризаций не отделены от класса функций/регуляризации. Обогащение использует уже размеченный пул и не доказывает экономию acquisition.",
        "- Обнаруженные взаимодействия отвечают на вопрос чувствительности методов к составу. Переносимая рекомендация требует новых датасетов/экспертов и независимой проверки выбора."]
    (destination/"REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps(facts, indent=2), flush=True)


if __name__ == "__main__":
    main()
