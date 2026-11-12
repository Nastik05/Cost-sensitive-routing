# Выбор политики маршрутизации в двухмодельных каскадах: влияние данных, структуры ошибок и вычислительного бюджета

**Автор:** Anastasia Shlopak

**Научный руководитель:** Archil Maysuradze, PhD

**Аннотация**

Исследуется выбор способа обучения маршрутизатора по объёму и составу парных данных, совместным ошибкам экспертов и квоте вызовов. Сравниваются все восемь исходных обучаемых методов — weak-error, direct, marginal, events, ternary, joint, hurdle, soft-gain — и необучаемые контрольные правила. Полная проверка на BANKING77 включает 21 пару семи фиксированных экспертов, три объёма fit, семь составов данных, дополнительные IPW-контроли, 20 повторений и три квоты. Обучены 110880 финальных роутеров. На естественных данных попарные различия accuracy не подтверждены общей поправкой. При обогащении вредными заменами обнаружены различимые взаимодействия метода и состава: gain-оцениватели теряют качество относительно weak-error и confidence. Между самими семью gain-параметризациями такое взаимодействие не установлено, хотя в одном обогащённом режиме различимы четыре простых контраста. Коррекция отбора служит дополнительным контролем смещения обучающей цели. Результаты характеризуют условия сравнительного качества методов; переносимая рекомендация требует независимых датасетов и экспертов.

## Текст

- [Аннотация](paper/paper_sections/аннотация.md)
- [Введение](paper/paper_sections/введение.md)
- [Обзор литературы — Related works](paper/paper_sections/related_works.md)
- [Постановка задачи](paper/paper_sections/постановка%20задачи.md)
- [Метод и теоретические основания](paper/paper_sections/method.md)
- [Вычислительные эксперименты](paper/paper_sections/вычислительные%20эксперименты.md)
- [Полная проверка выбора при дисбалансе](paper/paper_sections/router_comparison.md)
- **[Расширенный отчёт: все методы, метрики и взаимодействия](paper/imbalance_results/REPORT.md)**
- [Заключение](paper/paper_sections/conclusion.md)
- **[Статья — PDF](paper/main.pdf)**
- [Исходник LaTeX](paper/main.tex)

## Литература и иллюстрации

- [Обзор литературы](LINKREVIEW.md)
- [Библиография BibTeX](paper/references.bib)
- [Профили решений](paper/figures/decision_profiles.pdf)
- [Кривая ошибки](paper/figures/error_curve.pdf)
- [Презентация — 10 слайдов и резерв](slides/main.pdf)
- [Однослайдовая версия](slides/one_slide.pdf)
- [Текст доклада, тайминг и ответы на вопросы](slides/README.md)

## Основной результат

![Изменение относительного качества методов](paper/figures/imbalance_interaction_counts.png)

Проверяется **взаимодействие метод × состав**, а не только улучшение отдельного метода. В основной области (fit 1000, квота 20%, official test) после общей поправки на 5691 контраст:

- **56/1512** взаимодействий различимы; они относятся к сравнению gain-оценивателей с weak-error/confidence при harm-обогащении.
- **0/882** взаимодействий между семью gain-оценивателями; **4** простых accuracy-различия между ними обнаружены в harm8 на одной low-label паре.
- Natural-профили численно различаются, но после общей поправки их различия не подтверждены; незначимость accuracy не доказывает её равенства.
- Поддержанной смены знака преимущества с интервалами по разные стороны нуля в обоих режимах не установлено.

[Полный протокол](code/PROTOCOL_IMBALANCE.md), [все контрасты](paper/imbalance_results/contrasts.csv), [метрики по всем повторениям](paper/imbalance_results/metrics.csv.gz). Результаты относятся к повторному анализу фиксированного теста, не к независимой проверке переноса.

## Воспроизведение

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/run.py
```

Запуск повторно обучает роутеры по фиксированному кэшу экспертов и проверяет 1440 строк метрик. Построение графиков отделено от обучения. [Данные, конфигурация и команды сборки документов](code/README.md).

Дополнительный контроль состава и выбора метода:

```bash
.venv/bin/python code/followup.py
.venv/bin/python code/analyze_followup.py
.venv/bin/python code/build_documents.py
```

Обучаются 2700 роутеров; выполняются аудит сохранённых прогнозов, 2000 bootstrap повторов и построение таблиц/графиков. [Полные численные метрики](paper/generated/followup_metrics.csv.gz), [сводки](paper/generated/followup_summary.csv), [проверки и хеши](paper/generated/followup_manifest.json). Исходный baseline и новый контроль сохраняются отдельно.

Полная проверка всех восьми методов:

```bash
.venv/bin/python code/imbalance_study.py --workers 3
.venv/bin/python code/audit_imbalance.py --workers 2
.venv/bin/python code/imbalance_statistics.py --workers 3
.venv/bin/python code/analyze_imbalance.py
```

Сохраняются все модели, вероятности извлечений, IDs, scores и метрики. [Инструкция и checkpoints](code/README.md#полный-набор-методов-и-дисбаланс).

## Лицензия

Код: [MIT](LICENSE). Данные BANKING77 и производный кэш: [CC BY 4.0](code/data/BANKING77_LICENSE), источник — [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), [Casanueva et al., 2020](https://aclanthology.org/2020.nlp4convai-1.5/).
