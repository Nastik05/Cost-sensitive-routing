# Расширенные эксперименты BANKING77 и CLINC150

В этой папке опубликованы реализации восьми способов обучения, компактные фиксированные кэши экспертов, статистические процедуры и полные таблицы контрастов, использованные в текущей статье. Базовый опыт из `../run.py` сохранён отдельно.

## Быстрая проверка опубликованных чисел

Из корня репозитория:

```bash
python3 code/research/verify_results.py
python3 -m pytest -q code/test_experiment.py code/research/test_imbalance.py
python3 -m pytest -q code/research/clinc150/test_stage.py
python3 -m pytest -q code/research/multiclass/test_metrics.py
```

Тесты этапов запускаются отдельными командами: исторические CLINC- и multiclass-скрипты используют одноимённые модули `common` и `infer`.

Проверка читает все три статистические семьи, сверяет восемь главных эффектов со средними метриками и проверяет SHA-256 входных кэшей. Пересчёт доверительных интервалов требует scores, получаемых при повторном обучении ниже.

## BANKING77: восемь методов и 21 пара

```bash
python3 code/research/imbalance_study.py --smoke --output code/research/results/smoke
python3 code/research/imbalance_study.py --workers 3
python3 code/research/audit_imbalance.py
python3 code/research/imbalance_statistics.py
```

Первый запуск — короткая проверка на dev. Полная сетка: 20 seeds, fit200/500/1000, семь составов и четыре IPW-варианта, 110 880 выбранных роутеров после 443 520 candidate fits. Обучение, replay-аудит и статистика разделены. Полная сетка требует существенно больше времени и места, чем smoke; результаты пишутся в игнорируемую Git папку `results/`.

## CLINC150: два критерия настройки на одной сетке

```bash
python3 code/research/clinc150/run_routers.py --smoke --workers 3
python3 code/research/clinc150/run_routers.py --workers 3
python3 code/research/clinc150/audit.py
python3 code/research/clinc150/infer.py --workers 3
python3 code/research/clinc150/check_mc.py
```

Полная сетка содержит 23 040 кандидатов. MSE и quota-accuracy выбирают регуляризацию из одной сетки, используя раздельные tuning/selector части. Основная семья содержит 1698 контрастов; `check_mc.py` выполняет однократное уточнение с 2000 до 10 000 bootstrap-повторов. Это уточнение точности, а не независимая репликация.

## Многоклассовые метрики

После полного обучения обеих панелей:

```bash
python3 code/research/multiclass/compute.py --workers 3
python3 code/research/multiclass/summarize.py
python3 code/research/multiclass/infer.py --workers 3
```

Замороженные argmax-метки экспертов и class labels включены в `multiclass/labels/`. Они восстановлены из исходных вероятностей и сверены с W/S; генерация новых экспертов не требуется. Вычисляются macro/micro/weighted P/R/F1; семья 3564 контрастов ограничена natural/fit1000/20%/official. Полные per-condition scores и fitted models создаются при обучении, а не загружаются готовым многогигабайтным архивом.

## Опубликованные результаты

- `evidence/bank_contrasts.csv` — все 5691 контраст BANKING77.
- `evidence/clinc_primary.csv`, `clinc_mc10000.csv` — исходные и уточнённые 1698 контрастов CLINC150.
- `evidence/multiclass_contrasts.csv` — все 3564 multiclass-контраста.
- `evidence/primary_metrics.csv` — средние и SD основных режимов всех методов и составов.
- `evidence/article_claims.csv` — восемь выделенных в статье эффектов с указанием ориентации.
- `evidence/class_example.csv`, `class_example_overall.csv` — описательный классовый пример.
- `provenance.json` — хеши и локальные исследовательские источники перенесённых файлов.

Функции обучения, выбора и статистики перенесены из завершённых этапов. Два `common.py` адаптированы к структуре этого репозитория: проверяют включённые кэши вместо файлов других локальных проектов. Историческое поле `preserved_files` в новых запусках этих двух этапов обозначает число проверенных входных кэшей. Исторические протоколы и метаданные сохраняют описания исходного рабочего пространства; для запуска используются команды этого README.

## Данные и атрибуция

Кэши содержат агрегированные признаки первой модели, правильность экспертов, идентификаторы, разбиения и классовые метки; исходные тексты запросов и веса pretrained-моделей сюда не включены.

- **BANKING77:** PolyAI, revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; Casanueva et al., 2020. Производные данные — [CC BY 4.0](../data/BANKING77_LICENSE). Изменения относительно исходного датасета: дедупликация, разбиение, прогнозы фиксированных экспертов и извлечение признаков.
- **CLINC150:** Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019, [DOI](https://doi.org/10.18653/v1/D19-1131). Источник [clinc/oos-eval](https://github.com/clinc/oos-eval), revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`; [исходная лицензия](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Производный кэш: только in-scope, очищенные train/dev, отдельное обучение голов, прогнозы и признаки. Атрибуция не означает одобрения исследования авторами датасета.

Эксперты условно фиксированы: повторяется обучение роутеров, а не независимое обучение всей системы с нуля.
