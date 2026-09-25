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
        <td> Archil Maysuradze, PhD </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract
Каскадная маршрутизация между моделями разного масштаба позволяет существенно снижать вычислительные затраты, сохраняя приемлемое качество ответов. Однако классическая логика таких каскадов (эскалировать запрос к дорогой модели при неуверенности дешёвой) опирается на предположение, что сильная модель заведомо исправит ошибку. Тогда как на практике обе модели могут ошибаться, и эскалация способна не улучшить, а ухудшить результат. Цель данной работы - построить бюджетно-ограниченный маршрутизатор для каскада классификаторов, который принимает решение об обращении к дорогой модели на основе предсказания её реальной пользы. Предлагается обучать маршрутизатор предсказывать знаковую выгоду от эскалации, то есть разницу между вероятностью «спасения» (когда сильная модель исправляет ошибку слабой) и вероятностью «вреда» (когда сильная модель портит правильный ответ слабой). В качестве экспериментальной площадки используется датасет Banking77 с двумя моделями классификации разной стоимости. На вход маршрутизатору будут подаваться эмбеддинг банковского запроса и распределение вероятностей, выданное дешёвым классификатором. Ожидается, что предложенный подход улучшит соотношение качества и числа вызовов дорогой модели по сравнению со случайной эскалацией, а также методами, основанными на предсказании ошибки дешёвой модели или её энтропии, при одинаковом бюджете. Итоговая ценность работы состоит в проверке того, что для эффективных каскадов предсказание инкрементальной пользы эскалации является более корректной целью обучения, чем оценка неопределённости отдельной модели.

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
