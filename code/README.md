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

Нужны LuaLaTeX, BibTeX, fontspec/polyglossia/beamer и шрифты Liberation Serif, DejaVu Sans. Pandoc включён в зависимость `pypandoc_binary`. Готовые PDF уже включены. Сборка создаёт `paper/черновик статьи.pdf`, Markdown-версии четырёх разделов и объединённого текста, а также `slides/main.pdf`.

Тексты редактируются в `paper/abstract.tex`, `introduction.tex`, `problem_statement.tex` и `experiments.tex`. `paper/черновик статьи.tex` подключает эти разделы; Markdown-файлы автоматически обновляются при сборке и предназначены для чтения на GitHub.

## Данные

В `data/banking77.npz` — компактный производный кэш:6080 строк, дешёвые признаки, парная правильность и разбиения. Исходные эксперты фиксированы и заново не обучаются; это повтор именно базового router-эксперимента, а не новый независимый test. Сеть/GPU/API-ключи не нужны. Ни correctness, ни label_id не являются входными признаками роутера.

Источник: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Атрибуция: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI2020, [DOI](https://doi.org/10.18653/v1/2020.nlp4convai-1.5). Данные/кэш: **CC BY4.0**, см. `data/BANKING77_LICENSE`; код — MIT. Параметры экспертов и SHA исходного результата — `data/manifest.json`.

Это воспроизводимая базовая конфигурация исследования. Таблицы профилей описательные; статистическая значимость их компонентов не заявляется. Подробное описание данных, настройки и результатов находится в [разделе экспериментов](../paper/вычислительные%20эксперименты.md).
