"""Статичные страницы сайта (конфиденциальность, условия, методы) на трёх языках.

Тексты — здесь, HTML собирается в web/public/: /privacy/ и /terms/ (en, x-default),
/methods/ и methods.json — из tools/methods_page.py,
/ru/…, /pt/…. Без встроенных скриптов (CSP сайта), читаются без JavaScript; один внешний
/back.js — возврат к открытому приложению по истории (без него ссылка обычная). С
canonical и hreflang. Ссылка внутри абзаца — [текст](/адрес). После правки текста:

    python tools/static_pages.py

tests/test_static_pages.py сверяет закоммиченные файлы с генератором.
"""

from __future__ import annotations

import html
import re
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
                "В истории вкладки приложение помечает свою запись меткой {biostat: \"app\"}, чтобы ссылка "
                "«← Вернуться в BioStat» вернула к открытому расчёту. Данных из файла в метке нет; она исчезает "
                "вместе с историей вкладки.",
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
                "In the tab history the app marks its own entry with {biostat: \"app\"}, so that the "
                "“← Back to BioStat” link returns you to the open calculation. The mark holds no data from your "
                "file and disappears with the tab history.",
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
                "No histórico do separador, a aplicação marca a sua própria entrada com {biostat: \"app\"}, para "
                "que a ligação «← Voltar ao BioStat» o leve de volta ao cálculo aberto. A marca não contém dados "
                "do seu ficheiro e desaparece com o histórico do separador.",
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
    },
}

WELHAM = ("Welham S. J., Gezan S. A., Clark S. J., Mead A. Statistical Methods in Biology: Design and Analysis "
          "of Experiments and Regression. CRC Press, 2015")
WITTE = "Witte R. S., Witte J. S. Statistics"

TERMS: dict[str, dict] = {
    "ru": {
        "title": "Условия использования — BioStat",
        "description": "Условия BioStat: сервис «как есть», без гарантий; выводы и их применение — ответственность "
                       "пользователя; источники методов; лицензия кода Apache-2.0.",
        "h1": "Условия использования",
        "updated": "Последнее изменение: 21.11.2026.",
        "sections": [
            ("Сервис «как есть»", [
                "BioStat предоставляется «как есть», без гарантий — явных или подразумеваемых, — в том числе "
                "без гарантий точности результатов, пригодности для конкретной цели и бесперебойной работы "
                "сайта.",
            ]),
            ("Выводы — ваша ответственность", [
                "Метод, проверку его допущений и толкование результата выбираете вы. Выводы из расчёта и их "
                "применение — ваша ответственность.",
            ]),
            ("Источники методов", [
                f"Методы реализованы по книгам: {WELHAM}; {WITTE}, 11-е изд. Wiley, 2017. Методы, которых в этих "
                "книгах нет, — по их первоисточникам.",
            ]),
            ("Не профессиональная консультация", [
                "Результаты BioStat не являются медицинской, юридической или иной профессиональной "
                "консультацией.",
            ]),
            ("Лицензия кода", [
                "Код BioStat распространяется по лицензии Apache License 2.0: [текст лицензии](/LICENSE.txt). "
                "Copyright 2026 Shuhrat Maksumov.",
                "Сайт использует сторонние компоненты — React, ECharts, SheetJS (xlsx), Pyodide, Python и пакеты "
                "numpy, scipy, pandas, statsmodels — по их лицензиям: "
                "[перечень и тексты лицензий](/third-party-licenses.txt).",
            ]),
        ],
        "back": "← Вернуться в BioStat",
    },
    "en": {
        "title": "Terms of use — BioStat",
        "description": "BioStat terms: provided as is, without warranties; conclusions and their use are the "
                       "user's responsibility; method sources; Apache-2.0 code license.",
        "h1": "Terms of use",
        "updated": "Last updated: 2026-11-21.",
        "sections": [
            ("Provided as is", [
                "BioStat is provided “as is”, without warranties of any kind, express or implied, including "
                "warranties of accurate results, fitness for a particular purpose and uninterrupted operation of "
                "the site.",
            ]),
            ("Conclusions are your responsibility", [
                "You choose the method, check its assumptions and interpret the result. Conclusions drawn from a "
                "calculation and their use are your responsibility.",
            ]),
            ("Sources of the methods", [
                f"The methods follow two books: {WELHAM}; {WITTE}, 11th ed. Wiley, 2017. Methods not covered by "
                "these books follow their original sources.",
            ]),
            ("Not professional advice", [
                "BioStat results are not medical, legal or other professional advice.",
            ]),
            ("Code license", [
                "BioStat code is distributed under the Apache License 2.0: [license text](/LICENSE.txt). "
                "Copyright 2026 Shuhrat Maksumov.",
                "The site uses third-party components — React, ECharts, SheetJS (xlsx), Pyodide, Python and the "
                "numpy, scipy, pandas and statsmodels packages — under their own licenses: "
                "[list and license texts](/third-party-licenses.txt).",
            ]),
        ],
        "back": "← Back to BioStat",
    },
    "pt": {
        "title": "Termos de utilização — BioStat",
        "description": "Termos do BioStat: serviço fornecido tal como está, sem garantias; as conclusões e o seu uso "
                       "são da responsabilidade do utilizador; licença Apache-2.0.",
        "h1": "Termos de utilização",
        "updated": "Última alteração: 21.11.2026.",
        "sections": [
            ("Fornecido tal como está", [
                "O BioStat é fornecido «tal como está», sem garantias de qualquer tipo, expressas ou implícitas, "
                "incluindo garantias de exatidão dos resultados, adequação a um fim específico e funcionamento "
                "ininterrupto do site.",
            ]),
            ("As conclusões são da sua responsabilidade", [
                "É você quem escolhe o método, verifica os seus pressupostos e interpreta o resultado. As "
                "conclusões tiradas de um cálculo e a sua aplicação são da sua responsabilidade.",
            ]),
            ("Fontes dos métodos", [
                f"Os métodos seguem dois livros: {WELHAM}; {WITTE}, 11.ª ed. Wiley, 2017. Os métodos que não "
                "constam destes livros seguem as suas fontes originais.",
            ]),
            ("Não é aconselhamento profissional", [
                "Os resultados do BioStat não constituem aconselhamento médico, jurídico ou outro aconselhamento "
                "profissional.",
            ]),
            ("Licença do código", [
                "O código do BioStat é distribuído sob a Apache License 2.0: [texto da licença](/LICENSE.txt). "
                "Copyright 2026 Shuhrat Maksumov.",
                "O site usa componentes de terceiros — React, ECharts, SheetJS (xlsx), Pyodide, Python e os pacotes "
                "numpy, scipy, pandas e statsmodels — sob as respetivas licenças: "
                "[lista e textos das licenças](/third-party-licenses.txt).",
            ]),
        ],
        "back": "← Voltar ao BioStat",
    },
}

PAGES: dict[str, dict[str, dict]] = {"privacy": PRIVACY, "terms": TERMS}

STYLE = (
    "body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#18181b;"
    "background:#fafafa;line-height:1.6}main{max-width:44rem;margin:0 auto;padding:1.5rem 1rem 2rem}"
    "h1{font-size:1.6rem;margin:0.6rem 0 0.2rem}h2{font-size:1.1rem;margin:1.6rem 0 0.4rem}"
    "p{margin:0.4rem 0}a{color:#0f766e}.muted{color:#63636b;font-size:0.9rem}"
    "nav{display:flex;flex-wrap:wrap;gap:0.4rem 1rem;justify-content:space-between;align-items:center}"
    "footer{border-top:1px solid #e4e4e7;margin-top:2rem;padding-top:0.8rem;font-size:0.9rem}"
    "code{overflow-wrap:anywhere}h3{font-size:1rem;margin:1.4rem 0 0}h2[id],h3[id]{scroll-margin-top:1rem}ul{margin:0.3rem 0;padding-left:1.2rem}"
    "li{margin:0.15rem 0;overflow-wrap:anywhere}.tag{font-size:0.75rem;font-weight:400;color:#92400e;"
    "border:1px solid #f59e0b;border-radius:4px;padding:0 0.3rem;margin-left:0.4rem;white-space:nowrap}"
)


FOOTER = {
    "ru": ("Конфиденциальность", "Условия", "Методы и проверка", "Лицензия Apache-2.0"),
    "en": ("Privacy", "Terms", "Methods and validation", "Apache-2.0 License"),
    "pt": ("Privacidade", "Termos", "Métodos e verificação", "Licença Apache-2.0"),
}
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _inline(text: str) -> str:
    """Экранированный текст; [текст](/адрес) становится ссылкой."""
    parts, pos = [], 0
    for m in LINK.finditer(text):
        parts += [html.escape(text[pos:m.start()]),
                  f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>']
        pos = m.end()
    return "".join(parts) + html.escape(text[pos:])


def _footer(lang: str) -> str:
    hrefs = (page_path("privacy", lang), page_path("terms", lang), page_path("methods", lang), "/LICENSE.txt")
    links = " · ".join(f'<a href="{h}">{html.escape(t)}</a>' for h, t in zip(hrefs, FOOTER[lang]))
    return f"      <footer>{links}</footer>\n"


def _links(slug: str, lang: str, langs: tuple[str, ...] = LANGS) -> str:
    """canonical и hreflang; x-default — английская версия, а без неё — первая из langs."""
    alts = "".join(f'    <link rel="alternate" hreflang="{code}" href="{SITE}{page_path(slug, code)}" />\n'
                   for code in langs)
    default = "en" if "en" in langs else langs[0]
    return (f'    <link rel="canonical" href="{SITE}{page_path(slug, lang)}" />\n{alts}'
            f'    <link rel="alternate" hreflang="x-default" href="{SITE}{page_path(slug, default)}" />\n')


def shell(slug: str, t: dict, lang: str, body: str, *, langs: tuple[str, ...] = LANGS,
          head: str = "", header: str | None = None, main_class: str = "") -> str:
    """Каркас страницы: head, навигация, заголовок, тело, подвал.

    langs — языки, на которых страница есть; head — доп. теги в <head>; header — свой
    заголовок вместо h1 и строки даты; main_class — класс <main>.
    """
    esc = html.escape
    current = ' aria-current="page"'
    switch = " · ".join(
        f'<a href="{page_path(slug, code)}" hreflang="{code}" lang="{code}"'
        f'{current if code == lang else ""}>{LANG_NAMES[code]}</a>' for code in langs)
    if header is None:
        header = f'      <h1>{esc(t["h1"])}</h1>\n      <p class="muted">{esc(t["updated"])}</p>\n'
    main = f'<main class="{main_class}">' if main_class else "<main>"
    return (
        f'<!doctype html>\n<html lang="{lang}">\n  <head>\n    <meta charset="UTF-8" />\n'
        f'    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f'    <title>{esc(t["title"])}</title>\n'
        f'    <meta name="description" content="{esc(t["description"])}" />\n'
        f'    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />\n'
        f'{_links(slug, lang, langs)}    <style>{STYLE}</style>\n{head}'
        f'    <script src="/back.js" defer></script>\n  </head>\n  <body>\n    {main}\n'
        f'      <nav><a href="/" data-back>{esc(t["back"])}</a><span>{switch}</span></nav>\n'
        f'{header}{body}'
        f'{_footer(lang)}    </main>\n  </body>\n</html>\n'
    )


def render(slug: str, texts: dict[str, dict], lang: str) -> str:
    """HTML одной языковой версии текстовой страницы (разделы из абзацев)."""
    t = texts[lang]
    body = "".join(f"      <h2>{html.escape(h)}</h2>\n" + "".join(f"      <p>{_inline(p)}</p>\n" for p in ps)
                   for h, ps in t["sections"])
    return shell(slug, t, lang, body)


def pages() -> dict[Path, str]:
    """Путь → содержимое всех статичных страниц (и methods.json, content/math_tex.json)."""
    import method_pages  # здесь: оба модуля сами импортируют этот
    import methods_page

    out = {PUBLIC / page_path(slug, lang).strip("/") / "index.html": render(slug, texts, lang)
           for slug, texts in PAGES.items() for lang in LANGS}
    data = methods_page.methods_data()
    return out | methods_page.pages(data) | method_pages.pages(data)


def main() -> None:
    for path, text in pages().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
