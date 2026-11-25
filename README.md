# BioStat

Веб-платформа статистических расчётов для биологии и агрономии.
Референс методов — Welham et al., *Statistical Methods in Biology* (CRC Press, 2015).

Архитектура и план — см. `docs/PROJECT_SPEC.md`.

## Расчётное ядро (Python)

Чистые статистические функции, тестируются отдельно от UI, затем исполняются
в браузере через Pyodide.

Python 3.12 и версии пакетов ровно как в Pyodide (см. `requirements.txt`):
тесты проверяют ядро на тех же библиотеках, на которых оно считает в браузере.

```bash
py -3.12 -m venv bstat.venv --prompt bstat   # Linux/macOS: python3.12 -m venv …
bstat.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Статус

- [x] Фаза 0: окружение, структура, единая модель результата
- [x] Фаза 1: расчётное ядро (64 теста)
  - [x] descriptive.py — описательная статистика, CI для среднего (гл. 2)
  - [x] hypothesis.py — t-тесты, непараметрика (гл. 2)
  - [x] anova.py — однофакторный ANOVA, Tukey HSD, Краскел-Уоллис (гл. 3, 8)
  - [x] regression.py — простая линейная регрессия, предсказание (гл. 5)
  - [x] diagnostics.py — проверка допущений, данные для 4 графиков
- [x] Фаза 2: ядро в браузере (Pyodide) — проверено, числа совпали
- [~] Фаза 3: загрузка и парсинг данных
  - [x] dialect.py — детекция кодировки, разделителя, десятичного знака
  - [x] types.py — типизация колонок, Тир-1 сводка качества
  - [x] parser.py — сквозной разбор файла, подсказка методов
  - [x] UI: загрузка, качество, предпросмотр, выбор метода, результат
  - [x] plotdata.py + графики ECharts (гистограмма, боксплот, scatter, Q-Q, остатки)
  - [x] catalog.py — полная витрина методов (10 методов в 6 группах, параметры)
  - [x] i18n — локализация UI и каталога методов (RU/EN/PT)
- [~] Фаза 4: MVP собран локально, готов к проверке

## Страницы методов

Объяснения методов — `/ru/methods/<id>/` и словарь обозначений `/ru/methods/symbols/`.
Вручную пишется только текст: `content/methods/<id>.json` (по языкам; страница на en или pt
появляется, только когда есть её перевод). Числа примера считает ядро на
`content/examples/field_trial.json` (копия примера сайта), путь по карте и сигналы берутся из
дерева и распознавателей, блок «Проверка» — из `methods.json`.

```
python tools/static_pages.py        # страницы и content/math_tex.json (все формулы)
node web/build-math.mjs             # формулы → MathML (KaTeX) в content/math_cache.json
python tools/static_pages.py        # ещё раз: страницы с формулами
```

Проверка в браузере (ширина 375 px, CSP, кнопка «↑ К списку»), после `npm run build` в `web/`:

```
node tools/check_method_pages.mjs   # нужен Playwright: npm i -g playwright && npx playwright install chromium
node tools/check_method_link.mjs    # калькулятор с методом из адреса (?method=&lang=); без сети — PYODIDE_DIR=<pyodide>
```

## Лицензия

Copyright 2026 Shuhrat Maksumov

Код BioStat распространяется по лицензии Apache License 2.0 — см. `LICENSE` и `NOTICE`.
Сторонние компоненты и их лицензии — `THIRD_PARTY_NOTICES.md`; полные тексты сайт
раздаёт в `/third-party-licenses.txt`.
