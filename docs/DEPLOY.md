# Деплой на Cloudflare Pages

Сайт статический — всё считается в браузере (Pyodide). Серверного
бэкенда нет, поэтому хостинг бесплатный и простой.

## Почему Cloudflare Pages

Единственный крупный хост с неограниченным бесплатным трафиком для
статики. Pyodide тянет десятки МБ WASM на каждого пользователя — трафик
будет большим, а у Cloudflare за него не платишь. Нет риска «сюрприза в
счёте» при всплеске посещений.

## Ключевое: ядро собирается на Node, не на Python

`core.zip` (расчётное ядро для Pyodide) раньше собирался `build_core.py`
на Python. Cloudflare при сборке запускает только Node. Поэтому
генерация продублирована на Node — `web/build-core.mjs` — и встроена в
`npm run build` через `prebuild`. На сервере Cloudflare Python не нужен.

`build_core.py` остаётся для локальной работы (кто привык), но для
деплоя достаточно `npm run build`.

## Шаги

1. Запушить репозиторий на GitHub (если ещё нет):
   ```
   git push -u origin main
   ```

2. dash.cloudflare.com → Workers & Pages → Create → Pages →
   Connect to Git → выбрать репозиторий.

3. Настройки сборки:
   - Build command:        `cd web && npm install && npm run build`
   - Build output directory: `web/dist`
   - Framework preset:      Vite (или None)

4. Deploy. Первая сборка ~2-3 минуты.

5. Домен: Pages → Custom domains → добавить купленный домен.
   Если домен не на Cloudflare — прописать выданные DNS-записи у
   регистратора. HTTPS/SSL включается автоматически и бесплатно.

## После первого деплоя — проверить

- Открыть сайт, дождаться загрузки Pyodide (первый раз 20-60 сек).
- Открыть консоль браузера (F12) — не должно быть ошибок CSP/CORS про
  WASM или jsdelivr.
- Загрузить тестовый CSV, прогнать расчёт — числа должны совпадать с
  локальными.

Если WASM блокируется — проверить `web/public/_headers` (там намеренно
НЕТ COEP, чтобы не сломать загрузку с CDN).

## Файлы деплоя (в web/public/, копируются в dist/)

- `_headers`   — кеш для /core/*, без cross-origin isolation
- `_redirects` — SPA: любой путь → index.html
