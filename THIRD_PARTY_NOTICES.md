# Сторонние компоненты BioStat

Код BioStat — Apache License 2.0 (`LICENSE`, `NOTICE`). Ниже — чужой код, который
получает браузер пользователя. Полные тексты лицензий сайт раздаёт в
`/third-party-licenses.txt`; файл собирает `web/build-licenses.mjs` при сборке из
`node_modules` и `web/licenses/`. Сборка падает, если в зависимостях появился пакет
без записи в его таблице; `tests/test_licenses.py` сверяет этот перечень с ней.

## JavaScript-бандл (раздаёт biostat.app)

Состав проверен по sourcemap сборки `vite build`.

| Компонент | Версия | Лицензия |
|---|---|---|
| echarts | 6.1.0 | Apache-2.0 |
| react | 18.3.1 | MIT |
| react-dom | 18.3.1 | MIT |
| scheduler | 0.23.2 | MIT |
| tslib | 2.3.0 | 0BSD |
| xlsx | 0.18.5 | Apache-2.0 |
| zrender | 6.1.0 | BSD-3-Clause |

- ECharts содержит фрагменты d3 (BSD-3-Clause); их лицензия и `NOTICE` ECharts —
  в `/third-party-licenses.txt`.
- В бандл идёт готовая сборка `xlsx.mjs` (SheetJS). Её npm-зависимости (adler-32, cfb,
  codepage, crc-32, ssf, wmf, word — тоже SheetJS, Apache-2.0) в бандл не входят.
- `loose-envify` и `js-tokens` — трансформация сборки React, в бандл не входят.

## Рантайм Python (раздаёт biostat.app, `/pyodide/v0.26.4/`)

| Компонент | Версия | Лицензия |
|---|---|---|
| pyodide | 0.26.4 | MPL-2.0 |
| CPython | 3.12.1 | PSF-2.0 |

- Лицензия Pyodide — файл `LICENSE` из тега 0.26.4 репозитория
  https://github.com/pyodide/pyodide (`web/licenses/pyodide-0.26.4.LICENSE`). В
  npm-пакете `pyodide` файла лицензии нет, а поле `"license"` его `package.json`
  указывает `Apache-2.0`; принят текст репозитория. Рядом с рантаймом на сайте лежат
  `LICENSE.txt` и `SOURCE.txt` — где взять исходники (MPL-2.0, п. 3.2).
- CPython 3.12.1 (интерпретатор в `pyodide.asm.wasm` и стандартная библиотека в
  `python_stdlib.zip`) — текст `LICENSE` из тега v3.12.1 репозитория
  https://github.com/python/cpython (`web/licenses/cpython-3.12.1.LICENSE`).
- `pyodide.asm.wasm` собран Emscripten 3.1.58 и статически содержит и другие
  библиотеки, нужные CPython. Их перечень здесь не разобран; он есть в репозитории Pyodide.

## Пакеты Python (раздаёт jsDelivr, не biostat.app)

Замыкание зависимостей `REQUIRED_PACKAGES` из `web/src/pyodide/worker.ts` по
lock-файлу Pyodide 0.26.4. Лицензии сверены по `METADATA` и файлам лицензий внутри
колёс; тексты лицензий лежат в самих колёсах. Исключение — openblas: его архив
содержит только `libopenblas.so`; текст лицензии:
https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.26/LICENSE.

| Компонент | Версия | Лицензия |
|---|---|---|
| charset-normalizer | 3.3.2 | MIT |
| numpy | 1.26.4 | BSD-3-Clause |
| openblas | 0.3.26 | BSD-3-Clause |
| packaging | 23.2 | Apache-2.0 OR BSD-2-Clause |
| pandas | 2.2.0 | BSD-3-Clause |
| patsy | 0.5.6 | BSD-2-Clause |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause |
| pytz | 2024.1 | MIT |
| scipy | 1.12.0 | BSD-3-Clause |
| six | 1.16.0 | MIT |
| statsmodels | 0.14.2 | BSD-3-Clause |
