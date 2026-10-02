# Выбор политики маршрутизации в двухмодельных каскадах: влияние данных, структуры ошибок и вычислительного бюджета
<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Anastasia Shlopak </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Archil Maysuradze, PhD </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract
Каскадные системы позволяют адаптивно распределять вычислительные ресурсы, передавая часть запросов от дешёвого предиктора более дорогому. Эффективность такой системы зависит от способности маршрутизатора оценить ожидаемое изменение качества ответа: дорогая модель может как исправить ошибку, так и заменить правильный ответ неправильным. Существующие маршрутизаторы используют различные обучающие цели и параметризации, однако для выбора того или иного подхода для новой задачи важно учитывать доступные данные, структуру ошибок моделей и ограничения эксплуатации.
Цель исследования заключается в выявлении условий, при которых различные способы обучения маршрутизатора будут обеспечивать наилучшее соотношение качества и вычислительных затрат. Основная задача -  разработать систему рекомендаций по выбору маршрутизатора на основе характеристик данных и используемых моделей. 
В работе будет рассматривается каскад из двух фиксированных классификаторов и маршрутизатора, получающего на вход признаки объекта и результат дешёвой модели. Для сравнения будут выбраны такие политики маршрутизации как прямая регрессия прироста качества, разность оценок правильности моделей, оценивание вероятностей исправления и ухудшения качества ответа, неуверенность только дешевой модели, сложность признаков входного объекта. Переносимость сформированных рекомендаций планируется проверить на новых наборах данных и парах классификаторов, не использованных при выявлении закономерностей и выборе параметров методов. Итогом работы должны стать экспериментально обоснованные правила выбора способа обучения маршрутизатора, позволяющие определить, когда дополнительная сложность оправданна, а когда сопоставимое качество достигается более простым решением.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
