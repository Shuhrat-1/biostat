"""Статичные страницы сайта (конфиденциальность; позже условия, методы) на трёх языках.

Тексты — здесь, HTML собирается в web/public/: /privacy/ (en, x-default),
/ru/privacy/, /pt/privacy/. Страницы без скриптов (CSP сайта), читаются без
JavaScript, с canonical и hreflang. После правки текста:

    python tools/static_pages.py

tests/test_static_pages.py сверяет закоммиченные файлы с генератором.
"""

from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public"
SITE = "https://biostat.app"
LANGS = ("en", "ru", "pt")
LANG_NAMES = {"en": "English", "ru": "Русский", "pt": "Português"}
UPDATED = "2026-11-21"


def page_path(slug: str, lang: str) -> str:
    """Адрес страницы: английская — без префикса языка."""
    return f"/{slug}/" if lang == "en" else f"/{lang}/{slug}/"


PRIVACY: dict[str, dict] = {
    "ru": {
        "title": "Конфиденциальность — BioStat",
        "description": "BioStat считает в браузере: файл не отправляется на сервер. Кто видит IP, что хранится "
                       "в браузере, как обезличить данные. Проверяемые факты.",
        "h1": "Конфиденциальность",
        "updated": "Последнее изменение: 21.11.2026.",
        "sections": [
            ("Данные остаются в браузере", [
                "Файл, который вы открываете в BioStat, читается и считается на вашем компьютере: расчёт идёт "
                "в браузере (Python в Pyodide). Файл и результаты на сервер не отправляются.",
                "Это проверено журналом сетевых запросов собранного сайта: при загрузке файла, расчёте и всех "
                "выгрузках браузер ничего не отправляет — только получает файлы сайта (запросы GET, ни одного "
                "POST).",
            ]),
            ("Кто видит ваш IP-адрес", [
                "Cloudflare — хостинг biostat.app: при открытии сайта браузер получает с него страницу, код, "
                "расчётное ядро и Python (Pyodide).",
                "jsDelivr — сеть доставки пакетов Python: при открытии сайта браузер получает с неё 11 файлов "
                "пакетов (numpy, scipy, pandas, statsmodels и их зависимости). Если браузер сохранил их в своём "
                "кеше, он может взять их оттуда.",
                "Других запросов нет: ни аналитики, ни рекламы, ни шрифтов и картинок с чужих сайтов.",
            ]),
            ("Отчёты об ошибках сети (NEL)", [
                "Cloudflare добавляет к ответам сайта заголовки Report-To и NEL. По ним браузер может отправить "
                "Cloudflare отчёт, если страница не загрузилась из-за сбоя сети. Об успешных загрузках отчётов "
                "нет (success_fraction: 0). Содержимое ваших файлов в такой отчёт не попадает: оно не покидает "
                "браузер.",
            ]),
            ("Что хранится в браузере", [
                "localStorage — только выбранный язык интерфейса (ключ biostat.lang). Cookie нет. Аналитики и "
                "счётчиков нет. Service Worker и кеш приложения не используются.",
            ]),
            ("Выгрузки", [
                "JSON, CSV, Excel, PNG и SVG создаются в браузере и сохраняются на ваш компьютер. Дальше за них "
                "отвечаете вы. В выгрузке — результаты расчёта и имена колонок и групп; у многих методов ещё "
                "значения по наблюдениям (подогнанные значения, остатки), по которым часть данных можно "
                "восстановить. В JSON и Excel — также имя, размер и SHA-256 исходного файла.",
            ]),
            ("Совет: обезличьте данные", [
                "Если для анализа не нужны идентификаторы, уберите их до загрузки: имена, адреса, номера "
                "животных или пациентов и другие сведения, по которым можно узнать человека или хозяйство. Для "
                "расчёта нужны только числа и группы.",
            ]),
            ("Контакт", [
                "Вопросы о данных — privacy@biostat.app. Уязвимости — security@biostat.app "
                "(/.well-known/security.txt).",
            ]),
        ],
        "back": "← Вернуться в BioStat",
        "footer": "Конфиденциальность",
    },
    "en": {
        "title": "Privacy — BioStat",
        "description": "BioStat computes in your browser: your file is never sent to a server. Who sees your IP, "
                       "what the browser stores, how to de-identify data.",
        "h1": "Privacy",
        "updated": "Last updated: 2026-11-21.",
        "sections": [
            ("Your data stays in the browser", [
                "The file you open in BioStat is read and analysed on your computer: the computation runs in the "
                "browser (Python in Pyodide). Neither the file nor the results are sent to a server.",
                "This was checked with a network log of the built site: while loading a file, computing and "
                "exporting, the browser sends nothing — it only fetches the site's own files (GET requests, "
                "not a single POST).",
            ]),
            ("Who sees your IP address", [
                "Cloudflare — the host of biostat.app: when you open the site, the browser fetches the page, "
                "the code, the computing core and Python (Pyodide) from it.",
                "jsDelivr — the network that delivers the Python packages: when you open the site, the browser "
                "fetches 11 package files from it (numpy, scipy, pandas, statsmodels and their dependencies). "
                "If the browser has kept them in its cache, it may take them from there.",
                "There are no other requests: no analytics, no ads, no fonts or images from other sites.",
            ]),
            ("Network error reports (NEL)", [
                "Cloudflare adds the Report-To and NEL headers to the site's responses. With them, the browser "
                "may send Cloudflare a report if a page failed to load because of a network error. Successful "
                "loads are not reported (success_fraction: 0). The contents of your files are never in such a "
                "report: they do not leave the browser.",
            ]),
            ("What the browser stores", [
                "localStorage — only the interface language you chose (key biostat.lang). No cookies. No "
                "analytics or counters. No Service Worker or application cache.",
            ]),
            ("Exports", [
                "JSON, CSV, Excel, PNG and SVG files are created in the browser and saved to your computer. From "
                "then on they are your responsibility. An export contains the results and the column and group "
                "names; for many methods also per-observation values (fitted values, residuals) from which "
                "part of the data can be reconstructed. JSON and Excel also hold the name, size and SHA-256 of "
                "the source file.",
            ]),
            ("Tip: de-identify your data", [
                "If the analysis does not need identifiers, remove them before loading: names, addresses, animal "
                "or patient numbers and anything else that could identify a person or a farm. The computation "
                "needs only numbers and groups.",
            ]),
            ("Contact", [
                "Questions about data — privacy@biostat.app. Vulnerabilities — security@biostat.app "
                "(/.well-known/security.txt).",
            ]),
        ],
        "back": "← Back to BioStat",
        "footer": "Privacy",
    },
    "pt": {
        "title": "Privacidade — BioStat",
        "description": "O BioStat calcula no seu navegador: o ficheiro nunca é enviado a um servidor. Quem vê o "
                       "IP, o que o navegador guarda, como anonimizar dados.",
        "h1": "Privacidade",
        "updated": "Última alteração: 21.11.2026.",
        "sections": [
            ("Os dados ficam no navegador", [
                "O ficheiro que abre no BioStat é lido e analisado no seu computador: o cálculo corre no "
                "navegador (Python no Pyodide). Nem o ficheiro nem os resultados são enviados a um servidor.",
                "Isto foi verificado com um registo de pedidos de rede do site compilado: ao carregar um "
                "ficheiro, calcular e exportar, o navegador não envia nada — apenas obtém os ficheiros do "
                "próprio site (pedidos GET, nenhum POST).",
            ]),
            ("Quem vê o seu endereço IP", [
                "Cloudflare — o alojamento de biostat.app: ao abrir o site, o navegador obtém daí a página, o "
                "código, o núcleo de cálculo e o Python (Pyodide).",
                "jsDelivr — a rede que entrega os pacotes Python: ao abrir o site, o navegador obtém daí 11 "
                "ficheiros de pacotes (numpy, scipy, pandas, statsmodels e as suas dependências). Se o "
                "navegador os guardou na sua cache, pode usá-los a partir daí.",
                "Não há outros pedidos: nem análise de tráfego, nem publicidade, nem tipos de letra ou imagens "
                "de outros sites.",
            ]),
            ("Relatórios de erros de rede (NEL)", [
                "A Cloudflare acrescenta às respostas do site os cabeçalhos Report-To e NEL. Com eles, o "
                "navegador pode enviar à Cloudflare um relatório se uma página não carregou devido a uma falha "
                "de rede. Os carregamentos bem-sucedidos não são comunicados (success_fraction: 0). O conteúdo "
                "dos seus ficheiros nunca está nesse relatório: não sai do navegador.",
            ]),
            ("O que o navegador guarda", [
                "localStorage — apenas o idioma da interface escolhido (chave biostat.lang). Sem cookies. Sem "
                "análise de tráfego nem contadores. Sem Service Worker nem cache da aplicação.",
            ]),
            ("Exportações", [
                "Os ficheiros JSON, CSV, Excel, PNG e SVG são criados no navegador e guardados no seu "
                "computador. A partir daí, a responsabilidade é sua. Uma exportação contém os resultados e os "
                "nomes das colunas e dos grupos; em muitos métodos também valores por observação (valores "
                "ajustados, resíduos), a partir dos quais parte dos dados pode ser reconstruída. Em JSON e "
                "Excel há ainda o nome, o tamanho e o SHA-256 do ficheiro de origem.",
            ]),
            ("Conselho: anonimize os dados", [
                "Se a análise não precisa de identificadores, retire-os antes de carregar: nomes, moradas, "
                "números de animais ou de doentes e tudo o que permita identificar uma pessoa ou uma "
                "exploração. O cálculo só precisa de números e grupos.",
            ]),
            ("Contacto", [
                "Questões sobre dados — privacy@biostat.app. Vulnerabilidades — security@biostat.app "
                "(/.well-known/security.txt).",
            ]),
        ],
        "back": "← Voltar ao BioStat",
        "footer": "Privacidade",
    },
}

STYLE = (
    "body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#18181b;"
    "background:#fafafa;line-height:1.6}main{max-width:44rem;margin:0 auto;padding:1.5rem 1rem 2rem}"
    "h1{font-size:1.6rem;margin:0.6rem 0 0.2rem}h2{font-size:1.1rem;margin:1.6rem 0 0.4rem}"
    "p{margin:0.4rem 0}a{color:#0f766e}.muted{color:#63636b;font-size:0.9rem}"
    "nav{display:flex;flex-wrap:wrap;gap:0.4rem 1rem;justify-content:space-between;align-items:center}"
    "footer{border-top:1px solid #e4e4e7;margin-top:2rem;padding-top:0.8rem;font-size:0.9rem}"
    "code{overflow-wrap:anywhere}"
)


def _links(slug: str, lang: str) -> str:
    alts = "".join(f'    <link rel="alternate" hreflang="{code}" href="{SITE}{page_path(slug, code)}" />\n'
                   for code in LANGS)
    return (f'    <link rel="canonical" href="{SITE}{page_path(slug, lang)}" />\n{alts}'
            f'    <link rel="alternate" hreflang="x-default" href="{SITE}{page_path(slug, "en")}" />\n')


def render(slug: str, texts: dict[str, dict], lang: str) -> str:
    """HTML одной языковой версии страницы."""
    t = texts[lang]
    esc = html.escape
    current = ' aria-current="page"'
    switch = " · ".join(
        f'<a href="{page_path(slug, code)}" hreflang="{code}" lang="{code}"'
        f'{current if code == lang else ""}>{LANG_NAMES[code]}</a>' for code in LANGS)
    body = "".join(f"      <h2>{esc(h)}</h2>\n" + "".join(f"      <p>{esc(p)}</p>\n" for p in ps)
                   for h, ps in t["sections"])
    return (
        f'<!doctype html>\n<html lang="{lang}">\n  <head>\n    <meta charset="UTF-8" />\n'
        f'    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f'    <title>{esc(t["title"])}</title>\n'
        f'    <meta name="description" content="{esc(t["description"])}" />\n'
        f'    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />\n'
        f'{_links(slug, lang)}    <style>{STYLE}</style>\n  </head>\n  <body>\n    <main>\n'
        f'      <nav><a href="/">{esc(t["back"])}</a><span>{switch}</span></nav>\n'
        f'      <h1>{esc(t["h1"])}</h1>\n      <p class="muted">{esc(t["updated"])}</p>\n{body}'
        f'      <footer><a href="{page_path("privacy", lang)}">{esc(PRIVACY[lang]["footer"])}</a></footer>\n'
        f'    </main>\n  </body>\n</html>\n'
    )


def pages() -> dict[Path, str]:
    """Путь в web/public → HTML для всех статичных страниц."""
    return {PUBLIC / page_path("privacy", lang).strip("/") / "index.html": render("privacy", PRIVACY, lang)
            for lang in LANGS}


def main() -> None:
    for path, text in pages().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
