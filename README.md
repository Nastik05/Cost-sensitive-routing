# Cost-sensitive model routing for Banking Intent Classification

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
        <td> Archil Maysuradze, PhD/DSc </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract
Статья посвящена маршрутизации запросов между специализированным классификатором и большой языковой моделью при ограниченном бюджете вычислений. Использование языковой модели для каждого обращения увеличивает затраты, тогда как передача ей только неуверенных предсказаний не гарантирует улучшения качества: дополнительная модель может как исправить ошибку, так и заменить правильный ответ неправильным. Цель работы - разработать и оценить критерий маршрутизации, учитывающий ожидаемую пользу дополнительного вызова. Предлагаемый подход объединяет вероятности классов, устойчивость предсказаний нескольких дешёвых моделей и статистику ошибок на близких категориях. На основе этих признаков маршрутизатор оценивает вероятность полезного и вредного переключения и выбирает запросы для обработки языковой моделью. Экспериментальную оценку планируется провести на датасете BANKING77, содержащем банковские обращения с разметкой по 77 намерениям.

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
