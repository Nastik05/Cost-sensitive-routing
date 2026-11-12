# Вычислительный эксперимент

Из корня репозитория:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/run.py
```

Один запуск заново обучает weak-error, direct и rescue/harm роутеры и сравнивает их с confidence. Две фиксированные пары: TF-IDF→BERT-tiny и TF-IDF→frozen MiniLM. Размеры fit 200/500/1000, dev=fit/2, 10 подвыборок, квоты 10/20/40%, естественное обучение. Параметры собраны в `experiment.CONFIG`. Все 1440 строк accuracy/rescue/harm/MSE и выбранная регуляризация сверяются с исходным опытом.

Результаты — `code/results/metrics.csv` и `run.json`; повторный запуск воспроизводит и перезаписывает только этот производный output. Вычисления находятся в `experiment.py`, рисунки — в `plot.py`. Для изменения оформления без обучения:

```bash
.venv/bin/python code/run.py --plots-only
```

Рисунки и TeX-таблица генерируются из сохранённого результата в `paper/figures/` и `paper/generated/`. Для сборки PDF после запуска:

```bash
.venv/bin/python code/build_documents.py
```

Нужны LuaLaTeX, BibTeX, fontspec/polyglossia/beamer/TikZ и шрифты Liberation Serif, DejaVu Sans. Pandoc включён в зависимость `pypandoc_binary`. Готовые PDF уже включены. Сборка создаёт `paper/main.pdf`, недостающие Markdown-версии разделов в `paper/paper_sections/`, `slides/main.pdf` и `slides/one_slide.pdf`.

Исходники разделов находятся в `paper/paper_sections/`: `abstract.tex`, `introduction.tex`, `related_works.tex`, `problem_statement.tex`, `method.tex`, `experiments.tex` и `conclusion.tex`. `paper/main.tex` подключает их; обзор идёт после введения, метод — после постановки. Markdown-файлы предназначены для чтения на GitHub. Обычная сборка сохраняет существующие Markdown-файлы. Для явного обновления из TeX используется `python3 code/build_documents.py --refresh-markdown`; перед этим ручные правки нужно синхронизировать с TeX. Можно ограничить обновление, например `--refresh-markdown --section method --section experiments`.

## Контроль состава и выбора метода

Полное описание: [протокол](PROTOCOL_FOLLOWUP.md). Вход — тот же кэш экспертов; новый отбор с возвращением, известные вероятности, 30 seeds и раздельные tuning/selector-dev.

```bash
.venv/bin/python code/followup.py
.venv/bin/python code/analyze_followup.py
.venv/bin/python code/build_documents.py
```

`followup.py` обучает 2700 финальных роутеров, сохраняет выбранные модели, параметры стандартизации, кратности fit-строк, test scores и метрики. Результаты по умолчанию находятся в `code/results/followup/`. Выполненные условия проверяются по хешам и пропускаются при повторном запуске; изменение вычислительного кода, протокола или окружения требует нового каталога через `--output`.

`analyze_followup.py` повторно загружает все модели, сверяет scores и 18360 строк тестовых метрик, затем выполняет 2000 bootstrap повторов и строит всю заранее заданную семью 14 контрастов. Для повторного построения рисунков без обучения и bootstrap: `python3 code/analyze_followup.py --plots-only`. Здесь тоже поддерживается `--output` для выбора каталога исходных результатов.

Публикуемые данные:

- [Полные метрики](../paper/generated/followup_metrics.csv.gz), [средние и SD](../paper/generated/followup_summary.csv).
- [Все контрасты](../paper/generated/followup_contrasts.csv), [частоты выбора](../paper/generated/selection_counts.csv), [потеря выбора](../paper/generated/selection_regret.csv).
- [Чувствительность harm-rich к квоте и clean test](../paper/generated/followup_sensitivity.csv).
- [Уникальные строки, ESS и время](../paper/generated/followup_fit_summary.csv), [хеши, протокол и аудит](../paper/generated/followup_manifest.json).

Сырые модели и scores воспроизводятся командами выше; в репозиторий включены компактные исходные данные и численные результаты. Это анализ фиксированных экспертов и ранее изученного test. Два протокола (исходный без возвращения и новый с возвращением) не смешиваются.

Проверки:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider code/test_experiment.py code/test_followup.py
```

## Полный набор методов и дисбаланс

Расширенный протокол: [PROTOCOL_IMBALANCE.md](PROTOCOL_IMBALANCE.md). Включены все восемь исходных обучаемых методов, confidence/random и все 21 прежняя пара экспертов. Изменяется только обучение роутеров: 7 составов, дополнительные IPW-контроли для 4 сильных обогащений, 3 объёма, 20 повторений, 3 квоты. Всего 110880 выбранных роутеров после обучения 443520 кандидатов регуляризации.

```bash
.venv/bin/python code/imbalance_study.py --workers 3
.venv/bin/python code/audit_imbalance.py --workers 2
.venv/bin/python code/imbalance_statistics.py --workers 3
.venv/bin/python code/analyze_imbalance.py
```

По умолчанию результаты сохраняются в `code/results/imbalance_all/`. На обычном CPU запуск значительно больше двухметодного контроля; все условия имеют checkpoints и проверку хешей. Повторный запуск пропускает готовые условия. Изменённый вычислительный код/протокол требует нового `--output` у всех четырёх команд. Не перезаписывайте прежние результаты.

Для предварительной проверки используются только dev-объекты:

```bash
.venv/bin/python code/imbalance_study.py --smoke --output code/results/imbalance_smoke --workers 3
.venv/bin/python code/audit_imbalance.py --output code/results/imbalance_smoke --workers 2
```

Входной файл `data/imbalance_cache.npz` содержит прежние признаки и правильность всех семи экспертов; SHA, порядок 21 пары и происхождение данных находятся в `data/imbalance_manifest.json`. Исходный `banking77.npz` не меняется. Low-label направления не всегда соответствуют возрастанию стоимости; смотрите `cost_order_holds` в manifest.

Статистика проверяет взаимодействие **метод × состав**, а также rescue precision/harm rate при близкой accuracy. Одна заранее заданная семья из 5691 контраста; все результаты публикуются, независимо от значимости. Bootstrap точно пересчитывает top-k, используя проверяемое сокращение префикса с полным fallback. Это оптимизация вычисления, не аппроксимация квоты.

Тесты расширенного исследования: `python3 -m pytest -q -p no:cacheprovider code/test_imbalance.py`.

[Полный отчёт и численные результаты](../paper/imbalance_results/REPORT.md), [все 5691 контраст](../paper/imbalance_results/contrasts.csv), [сводка взаимодействий](../paper/imbalance_results/interactions.csv). Статья и презентация обновлены по полному сравнению; двухметодный контроль сохраняется как предыдущий этап.

## Источник данных

В `data/banking77.npz` — компактный производный кэш:6080 строк, дешёвые признаки, парная правильность и разбиения. Исходные эксперты фиксированы и заново не обучаются; это повтор именно базового router-эксперимента, а не новый независимый test. Сеть/GPU/API-ключи не нужны. Ни correctness, ни label_id не являются входными признаками роутера.

Источник: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Атрибуция: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI2020, [DOI](https://doi.org/10.18653/v1/2020.nlp4convai-1.5). Данные/кэш: **CC BY4.0**, см. `data/BANKING77_LICENSE`; код — MIT. Параметры экспертов и SHA исходного результата — `data/manifest.json`.

Это воспроизводимая базовая конфигурация исследования. Таблицы профилей описательные; статистическая значимость их компонентов не заявляется. Подробное описание данных, настройки и результатов находится в [разделе экспериментов](../paper/paper_sections/вычислительные%20эксперименты.md).
