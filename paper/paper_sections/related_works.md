# Обзор литературы

Исследования маршрутизации рассматривают, какую часть вычислений достаточно выполнить для конкретного запроса и когда дополнительный эксперт способен улучшить ответ. Для нашей задачи существенны три группы подходов: правила на основе уверенности первой модели, сравнительное оценивание полезности перехода и исследования зависимости обучения/выбора оценивателя от данных. Такое разделение позволяет сопоставить используемую информацию, обучающие цели и ограничения.

## Уверенность и сравнительная полезность

Простые confidence-правила не требуют отдельного обучения роутера, но оценивают прежде всего надёжность первой модели. В работе Jitkrittum и др. (2023 г.) показано, когда этого достаточно, и разобраны случаи специализации второго эксперта, шума и сдвига распределения. Там же используется Diff-01 — регрессия разности правильности двух моделей. Поэтому преимущество сравнительного подхода связано не с самим наличием второго эксперта, а с тем, содержат ли доступные признаки информацию о неоднородности его пользы.

Для генеративных моделей уверенность дополнительно зависит от способа агрегации токенных вероятностей и длины ответа. Gupta и др. (2024 г.) показывают ограничения суммы/среднего и исследуют квантили и embeddings. Для accuracy-задач используется rescue-only цель, а для перевода — регрессия разности BLEURT. Эти цели и информационные режимы нельзя отождествлять. Добавление признаков второй модели также меняет стоимость решения.

Post-hoc обучение передачи решения учитывает ошибки и стоимость несовершенного эксперта (Narasimhan и др. 2022 г.). В AutoMix ожидаемая полезность продолжения оценивается по шумному сигналу самопроверки в последовательной модели принятия решений (Aggarwal и др. 2024 г.). Это более общий механизм каскада, но самопроверка требует учёта дополнительных затрат. В unified routing/cascading изучаются условия полезности разных схем в зависимости от качества оценок (Dekoninck и др. 2025 г.). Часть анализа использует искусственный шум истинных quality signals; это отличается от изменения объёма и состава реального обучения конкретного оценивателя.

## Обучающие данные и оценивание разности

В RouteLLM сравнены роутеры на preference data и её дополнениях из gold- и judge-разметки (Ong и др. 2025 г.). Результаты демонстрируют роль соответствия источника обучения задаче. Работа Routing with Generated Data сравнивает query-answer и query-only методы при разном качестве генератора и фильтрует запросы по согласию сильных и расхождению слабых моделей (Niu и др. 2026 г.). Таким образом, целевой отбор информативных примеров уже является предметом исследования. В этих работах выбирается эксперт до получения первого ответа; наш базовый контроль использует post-answer признаки фиксированной первой модели.

В uplift regression сравниваются прямая оценка эффекта и разность отдельных моделей (Rudaś и Jaroszewicz 2018 г.), а исследования meta-learners связывают предпочтительность метода со структурой эффекта и объёмом данных (Künzel и др. 2019 г.). Обычно на объекте известен только один потенциальный исход. При нашей парной разметке оба ответа наблюдаются, поэтому аргументы об эффективности unpaired оценивателей нельзя переносить буквально. Например, на одном $X$ с одинаковой ridge-регуляризацией регрессия $S-W$ тождественна разности двух ridge-прогнозов; различия с logistic-головами возникают из-за класса функций, потерь и регуляризации.

Особенно близка работа Nyberg и Klami (2023 г.): сравниваются sampling schemes, оцениватели и размеры данных, разбираются искажение ранжирования и коррекция вероятностей. Несмотря на пользу большего числа редких событий, изменение состава train может менять саму оцениваемую условную величину. Это объяснение необходимо проверять отдельно от статистической поддержки rescue/harm.

| Подход | Сильная сторона | Ограничение для нашей задачи |
|:---|:---|:---|
| Confidence (Jitkrittum и др. 2023 г.) | Простое правило без отдельной fit-процедуры | Не моделирует неоднородность пользы второго эксперта |
| Post-hoc gain (Jitkrittum и др. 2023 г.; Gupta и др. 2024 г.) | Использует сравнительную разметку и доступные признаки | Зависит от объёма данных, представления и выбранной цели |
| Последовательная utility (Aggarwal и др. 2024 г.; Dekoninck и др. 2025 г.) | Явно сопоставляет качество действий и затраты | Надёжность quality/verifier signal и его цена влияют на результат |
| Данные для routing (Ong и др. 2025 г.; Niu и др. 2026 г.) | Проверяет источники supervision и информативный отбор | Single-shot режим и разметка отличаются от парного post-answer каскада |
| Sampling effect estimation (Nyberg и Klami 2023 г.) | Разделяет дисбаланс, искажение разности и коррекцию | Непарные исходы; прямой отбор индивидуальных rescue/harm недоступен |

Подходы и ограничения, существенные для выбора способа обучения каскадного маршрутизатора.

## Выбор метода и границы текущего исследования

LLMRouterBench показывает близкие результаты ведущих роутеров и конкурентоспособность простых решений при едином протоколе (Li и др. 2026 г.). Эти наблюдения мотивируют проверять, оправданно ли усложнение, но близкие средние сами по себе не доказывают эквивалентность. Отдельно следует оценивать качество выбора метода: такой подход развит для CATE estimator selection (Mahajan и др. 2024 г.). В нашей парной постановке качество политики на dev можно оценить напрямую, без восстановления ненаблюдаемого контрфактического исхода.

Таким образом, существующая литература задаёт целевую полезность, сильные базовые методы и известные механизмы чувствительности к данным. Наше исследование продолжает эту линию контролируемым сравнением способов обучения при фиксированных экспертах и признаках, разном составе парных примеров и одинаковой квоте. Требуется установить, какие различия устойчивы, чем они объясняются и позволяют ли характеристики train/dev рекомендовать метод на новой задаче. Таблицы спасений/вреда используются для анализа механизмов, а не как самостоятельное доказательство новизны или превосходства.

# Литература

Aggarwal, Pranjal, Aman Madaan, Ankit Anand, и др. 2024 г. «AutoMix: Automatically Mixing Language Models». *Advances in Neural Information Processing Systems*. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract-Conference.html>.

Dekoninck, Jasper, Maximilian Baader, и Martin Vechev. 2025 г. «A Unified Approach to Routing and Cascading for LLMs». *Proceedings of the 42nd International Conference on Machine Learning* 267: 12987–3010. <https://proceedings.mlr.press/v267/dekoninck25a.html>.

Gupta, Neha, Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat, Aditya Krishna Menon, и Sanjiv Kumar. 2024 г. «Language Model Cascades: Token-level uncertainty and beyond». *International Conference on Learning Representations*. <https://proceedings.iclr.cc/paper_files/paper/2024/hash/11f5520daf9132775e8604e89f53925a-Abstract-Conference.html>.

Jitkrittum, Wittawat, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, и Sanjiv Kumar. 2023 г. «When Does Confidence-Based Cascade Deferral Suffice?» *Advances in Neural Information Processing Systems*. <https://doi.org/10.52202/075280-0431>.

Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, и Bin Yu. 2019 г. «Metalearners for estimating heterogeneous treatment effects using machine learning». *Proceedings of the National Academy of Sciences* 116 (10): 4156–65. <https://doi.org/10.1073/pnas.1804597116>.

Li, Hao, Yiqun Zhang, Zhaoyan Guo, и др. 2026 г. «LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing». *Findings of the Association for Computational Linguistics: ACL 2026*, 37733–54. <https://doi.org/10.18653/v1/2026.findings-acl.1881>.

Mahajan, Divyat, Ioannis Mitliagkas, Brady Neal, и Vasilis Syrgkanis. 2024 г. «Empirical Analysis of Model Selection for Heterogeneous Causal Effect Estimation». *International Conference on Learning Representations*. <https://openreview.net/forum?id=yuy6cGt3KL>.

Narasimhan, Harikrishna, Wittawat Jitkrittum, Aditya Krishna Menon, Ankit Singh Rawat, и Sanjiv Kumar. 2022 г. «Post-hoc estimators for learning to defer to an expert». *Advances in Neural Information Processing Systems*. <https://doi.org/10.52202/068431-2124>.

Niu, Tianyi, Justin Chen, Genta Indra Winata, и др. 2026 г. «Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection». *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 32441–66. <https://doi.org/10.18653/v1/2026.acl-long.1498>.

Nyberg, Otto, и Arto Klami. 2023 г. «Exploring uplift modeling with high class imbalance». *Data Mining and Knowledge Discovery* 37: 736–66. <https://doi.org/10.1007/s10618-023-00917-9>.

Ong, Isaac, Amjad Almahairi, Vincent Wu, и др. 2025 г. «RouteLLM: Learning to Route LLMs with Preference Data». *International Conference on Learning Representations*. <https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html>.

Rudaś, Krzysztof, и Szymon Jaroszewicz. 2018 г. «Linear regression for uplift modeling». *Data Mining and Knowledge Discovery* 32: 1275–305. <https://doi.org/10.1007/s10618-018-0576-8>.
