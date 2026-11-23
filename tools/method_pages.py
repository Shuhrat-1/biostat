"""Страницы-объяснения методов (шаг M1): /methods/<id>/ и словарь /methods/symbols/.

Вручную пишется только объясняющий текст — content/methods/<id>.json. Остальное
собирается: числа примера — расчётом ядра на content/examples/*.json (копия ROWS из
web/src/exampleData.ts), путь по карте — из decisionTree.json и treeLabels.json, сила
сигналов — из REGISTRY, блок «Проверка» — из methods.json, обозначения — из
content/symbols.json. Формулы — MathML из content/math_cache.json (KaTeX при сборке).

Порядок: python tools/static_pages.py → пишет content/math_tex.json (все TeX после
подстановки чисел); node web/build-math.mjs → дописывает content/math_cache.json;
python tools/static_pages.py → страницы с формулами. Недостающие формулы генератор
называет, tests/test_method_pages.py падает, пока кеш не полон.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import static_pages as sp  # noqa: E402
from method_examples import example_values  # noqa: E402
from method_widgets import groups_split, groups_welch  # noqa: E402

WIDGETS = {"groups-split": groups_split, "groups-welch": groups_welch}

CONTENT = ROOT / "content"
METHODS_DIR = CONTENT / "methods"
SYMBOLS = CONTENT / "symbols.json"
MATH_CACHE = CONTENT / "math_cache.json"
MATH_TEX = CONTENT / "math_tex.json"
TREE = ROOT / "web" / "src" / "decisionTree.json"
TREE_LABELS = ROOT / "web" / "src" / "treeLabels.json"
LANGS = ("ru", "en", "pt")
CSS = '    <link rel="stylesheet" href="/methods/assets/method.css" />\n'
JS = '    <script src="/methods/assets/groups-split.js" defer></script>\n'

PLACEHOLDER = re.compile(r"⟦([\wА-Яа-яЁё]+)(?::([^⟧]+))?⟧")
INLINE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|\[([^\]]+)\]\(([^)\s]+)\)|`([^`]+)`|\$([^$]+)\$")
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contents() -> dict[str, dict]:
    """id метода → его content/methods/<id>.json, по имени файла."""
    return {p.stem: _json(p) for p in sorted(METHODS_DIR.glob("*.json"))}


def available(lang: str) -> list[str]:
    """Методы, у которых есть страница на языке lang (есть текст; на другой язык не откатываемся)."""
    return [mid for mid, c in contents().items() if lang in c]


# --- подстановки чисел и разметка -----------------------------------------------------------

def p_text(p: float) -> str:
    """P по правилу сайта (conclusion.ts, pText): «< 0.001» или три знака."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_value(value: float, spec: str | None, tex: bool) -> str:
    """Число для текста или TeX: формат Python, «sci1» — мантисса·10^порядок,
    «grp» — целое с узкими пробелами между тройками цифр (17 153 136)."""
    if spec == "grp":
        return format(int(value), ",d").replace(",", "\\," if tex else "\u202f")
    if spec == "sci1":
        mantissa, exp = f"{value:.1e}".split("e")
        power = int(exp)
        return f"{mantissa}\\cdot10^{{{power}}}" if tex else f"{mantissa}·10{str(power).translate(SUPERSCRIPT)}"
    if spec:
        return format(value, spec)
    return str(value)


def substitute(text: str, values: dict, tex: bool = False) -> str:
    """⟦имя:формат⟧ → число из расчёта примера; ⟦p⟧ без формата — по правилу сайта."""
    def one(m: re.Match) -> str:
        name, spec = m.group(1), m.group(2)
        if name not in values:
            raise KeyError(f"подстановка ⟦{name}⟧: такого числа у расчёта примера нет")
        if name == "p" and spec is None:
            return p_text(values["p"])
        return fmt_value(values[name], spec, tex)
    return PLACEHOLDER.sub(one, text)


class Math:
    """MathML из кеша; запоминает все TeX страницы (для content/math_tex.json)."""

    def __init__(self) -> None:
        self.cache = _json(MATH_CACHE) if MATH_CACHE.exists() else {"inline": {}, "display": {}}
        self.needed: set[tuple[str, bool]] = set()

    def render(self, tex: str, display: bool = False) -> str:
        self.needed.add((tex, display))
        found = self.cache["display" if display else "inline"].get(tex)
        if found is None:
            return f'<code class="tex-missing">{html.escape(tex)}</code>'
        return found

    def missing(self) -> list[tuple[str, bool]]:
        return sorted(k for k in self.needed if k[0] not in self.cache["display" if k[1] else "inline"])


class Ctx:
    """Всё, что нужно разметке одной страницы: язык, числа примера, формулы, ссылки."""

    def __init__(self, lang: str, values: dict, math: Math) -> None:
        self.lang, self.values, self.math = lang, values, math
        self.detailed = set(available(lang))

    def method_href(self, method: str) -> str:
        """Страница метода, если она есть на этом языке; иначе его карточка в списке."""
        if method in self.detailed:
            return sp.page_path("methods/" + method, self.lang)
        return sp.page_path("methods", self.lang) + "#" + method


def method_link(method: str, inner: str, ctx: Ctx) -> str:
    title = CHROME[ctx.lang]["link_title"]
    return f'<a class="mlink" href="{ctx.method_href(method)}" title="{html.escape(title)}">{inner}</a>'


def inline(text: str, ctx: Ctx) -> str:
    """**жирный**, *курсив*, [текст](url | method:id), `сигнал`, $TeX$; остальное экранируется."""
    text = substitute(text, ctx.values)
    parts, pos = [], 0
    for m in INLINE.finditer(text):
        parts.append(html.escape(text[pos:m.start()]))
        bold, italic, label, href, sig, tex = m.groups()
        if bold is not None:
            parts.append(f"<b>{inline(bold, ctx)}</b>")
        elif italic is not None:
            parts.append(f"<i>{html.escape(italic)}</i>")
        elif label is not None:
            inner = inline(label, ctx)
            parts.append(method_link(href[7:], inner, ctx) if href.startswith("method:")
                         else f'<a href="{html.escape(href)}">{inner}</a>')
        elif sig is not None:
            parts.append(f'<span class="sig">{html.escape(sig)}</span>')
        else:
            parts.append(ctx.math.render(tex))
        pos = m.end()
    return "".join(parts) + html.escape(text[pos:])


def method_links(text: str) -> list[str]:
    """id методов из ссылок [текст](method:id) — для проверки, что они есть в CATALOG."""
    return [href[7:] for _l, href in re.findall(r"\[([^\]]+)\]\((method:[^)\s]+)\)", text)]


# --- данные карты ---------------------------------------------------------------------------

def tree_nodes() -> dict[str, dict]:
    return {n["id"]: n for n in _json(TREE)["nodes"]}


def leaf_path(leaf: str) -> list[str]:
    """Узлы от R0 до листа (у каждого узла дерева один parent — путь единственный)."""
    nodes, out, node = tree_nodes(), [], leaf
    while node:
        out.append(node)
        node = nodes[node].get("parent")
    return out[::-1]


def path_labels(leaf: str, lang: str) -> list[str]:
    """Подписи пути как ответы на рёбрах: часть подписи treeLabels до «:»; R0 не показываем."""
    labels = _json(TREE_LABELS)[lang]["nodes"]
    return [labels[node].split(":")[0].strip() for node in leaf_path(leaf)[1:]]


def signal_strengths() -> dict[str, str]:
    """id сигнала → сила из REGISTRY (strong / hint)."""
    import core.validation.recognizers  # noqa: F401 — регистрирует правила
    import core.validation.recognizers_computed  # noqa: F401
    from core.validation.recognize import REGISTRY

    return {r.id: r.strength for r in REGISTRY}


def signals_html(ids: list[str], ctx: Ctx, strengths: dict[str, str]) -> str:
    """Плашки сигналов через « / »; у подсказки (HINT) — «(подсказка)»."""
    hint = CHROME[ctx.lang]["hint"]
    return " / ".join(f'<span class="sig">{s}</span>' + (f" {html.escape(hint)}" if strengths[s] == "hint" else "")
                      for s in ids)


# --- разделы страницы -----------------------------------------------------------------------

def _section(sid: str, title: str, body: str) -> str:
    return f'  <section id="{sid}">\n    <h2>{html.escape(title)}</h2>\n{body}  </section>\n'


def _ul(items: list[str], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"    <ul{attr}>\n" + "".join(f"      <li>{i}</li>\n" for i in items) + "    </ul>\n"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>\n" for r in rows)
    return f'    <div class="tablewrap"><table>\n      <thead><tr>{th}</tr></thead>\n      <tbody>\n{trs}</tbody>\n    </table></div>\n'


def idea_section(c: dict, t: dict, ctx: Ctx, example: dict) -> str:
    body = "".join(f"    <p>{inline(p, ctx)}</p>\n" for p in t["idea"])
    if c.get("widget", {}).get("type") in WIDGETS:
        body += WIDGETS[c["widget"]["type"]](c["widget"], t["widget"], example, ctx.values)
    return _section("idea", CHROME[ctx.lang]["idea"], body)


def glossary(keys: list[str], ctx: Ctx) -> str:
    """«Все обозначения на странице» (свёрнуто) со ссылкой на общий словарь."""
    ch, symbols = CHROME[ctx.lang], _json(SYMBOLS)
    rows = [[ctx.math.render(symbols[k]["tex"]), inline(symbols[k][ctx.lang], ctx)] for k in keys]
    link = sp.page_path("methods/symbols", ctx.lang)
    return (f'    <details class="glossary" id="symbols">\n      <summary>{html.escape(ch["glossary"])}</summary>\n'
            f'{_table([ch["symbol"], ch["meaning"]], rows)}'
            f'      <p class="note">{html.escape(ch["glossary_note"])} '
            f'<a href="{link}">{html.escape(ch["glossary_link"])}</a>.</p>\n    </details>\n')


def formula_block(f: dict, ctx: Ctx) -> str:
    out = f'    <h3>{inline(f["title"], ctx)}</h3>\n' if f.get("title") else ""
    if f.get("text_before"):
        out += f'    <p>{inline(f["text_before"], ctx)}</p>\n'
    tex = substitute(f["tex"], ctx.values, tex=True)
    out += f'    <div class="math block">{ctx.math.render(tex, display=True)}</div>\n'
    if f.get("text_after"):
        out += f'    <p>{inline(f["text_after"], ctx)}</p>\n'
    return out


def example_table(spec: dict, ctx: Ctx) -> str:
    """«На примере»: таблица ANOVA (parameters.anova_table результата ядра) и строка итогов."""
    ch, fmt = CHROME[ctx.lang], spec.get("format", {})

    def num(v, key: str) -> str:
        if v is None:
            return ""
        s = format(v, fmt[key]) if key in fmt else str(v)
        return s.rstrip("0").rstrip(".") if "." in s else s

    table = ""
    if "labels" in spec:  # таблица ANOVA — только если в тексте заданы подписи её строк
        rows = [[html.escape(spec["labels"][r["term"]]), str(r.get("df", "")), num(r.get("ss"), "ss"),
                 num(r.get("ms"), "ms"), num(r.get("F"), "F"),
                 html.escape(p_text(r["p"])) if r.get("p") is not None else ""] for r in ctx.values["anova_table"]]
        table = _table(ch["anova_head"], rows)
    return (f'    <h3>{html.escape(ch["example"])}</h3>\n{table}'
            f'    <p class="note">{inline(spec["note"], ctx)}</p>\n')


def model_section(t: dict, ctx: Ctx) -> str:
    body = glossary(t["symbols"], ctx) + "".join(formula_block(f, ctx) for f in t["formulas"])
    body += example_table(t["example_table"], ctx)
    body += f'    <p class="src">{inline(t["sources"]["model"], ctx)}</p>\n'
    return _section("model", CHROME[ctx.lang]["model"], body)


def pros_cons_section(t: dict, ctx: Ctx) -> str:
    ch = CHROME[ctx.lang]
    col = lambda cls, title, items: (  # noqa: E731
        f'      <div class="{cls}">\n        <h3>{html.escape(title)}</h3>\n'
        + _ul([inline(i, ctx) for i in items]).replace("    ", "        ", 1) + "      </div>\n")
    body = ('    <div class="cols">\n' + col("plus", ch["pros"], t["pros"]) + col("minus", ch["cons"], t["cons"])
            + "    </div>\n" + f'    <p class="src">{inline(t["sources"]["pros_cons"], ctx)}</p>\n')
    return _section("pros-cons", ch["pros_cons"], body)


def assumptions_section(t: dict, ctx: Ctx) -> str:
    ch = CHROME[ctx.lang]
    rows = [[inline(a["name"], ctx), inline(a["meaning"], ctx), inline(a["biostat"], ctx)] for a in t["assumptions"]]
    body = _table(ch["assumptions_head"], rows) + f'    <p class="src">{inline(t["sources"]["assumptions"], ctx)}</p>\n'
    return _section("assumptions", ch["assumptions"], body)


def map_section(c: dict, t: dict, ctx: Ctx, names: dict[str, str]) -> str:
    """Путь к листу, соседи, проверки листа, что дальше."""
    ch, strengths = CHROME[ctx.lang], signal_strengths()
    labels = path_labels(c["leaf"], ctx.lang)
    path = "›".join(f'<span class="here">{html.escape(x)}</span>' if i == len(labels) - 1
                    else f"<span>{html.escape(x)}</span>" for i, x in enumerate(labels))
    body = (f'    <h3>{html.escape(ch["path"])}</h3>\n    <div class="path" aria-label="{html.escape(ch["path_aria"])}">'
            f"{path}</div>\n" + "".join(f"    <p>{inline(p, ctx)}</p>\n" for p in t["map_intro"]))
    rows = [[method_link(n["method"], html.escape(names[n["method"]]), ctx),
             " ".join(x for x in (signals_html(n["signals"], ctx, strengths), inline(n["when"], ctx)) if x),
             inline(n["advantage"], ctx), inline(n["cost"], ctx)] for n in t["neighbours"]]
    head = [ch["neighbour"], ch["when"], ch["better"].format(short=t["short"]), ch["cost"]]
    body += f'    <h3>{html.escape(ch["neighbours"])}</h3>\n' + _table(head, rows)
    checks = [f'{signals_html(k["signals"], ctx, strengths)} {inline(k["text"], ctx)}' for k in t["checks"]]
    body += f'    <h3>{html.escape(ch["checks"])}</h3>\n' + _ul(checks)
    body += f'    <h3>{html.escape(ch["next"])}</h3>\n    <p>{inline(t["next"], ctx)}</p>\n'
    return _section("map", ch["map"], body)


def check_block(entry: dict, ctx: Ctx) -> str:
    """«Проверка метода» (свёрнуто) — те же факты, что в карточке списка (methods.json)."""
    import methods_page

    items = "".join(f"      <li>{f}</li>\n" for f in methods_page.check_facts(entry, ctx.lang))
    return (f'  <details class="check" id="check">\n    <summary>{html.escape(CHROME[ctx.lang]["check"])}</summary>\n'
            f"    <ul>\n{items}    </ul>\n  </details>\n")


def pager(mid: str, entry_group: str, data: dict, ctx: Ctx) -> str:
    """Предыдущий / следующий метод группы (порядок CATALOG), только среди методов со страницей."""
    ch = CHROME[ctx.lang]
    group = [m for m in data["methods"] if m["group"]["ru"] == entry_group and m["id"] in ctx.detailed]
    ids = [m["id"] for m in group]
    i = ids.index(mid)
    links = []
    if i > 0:
        links.append(f'<a href="{ctx.method_href(ids[i - 1])}"><small>{ch["prev"]}</small>'
                     f'{html.escape(group[i - 1]["name"][ctx.lang])}</a>')
    if i < len(ids) - 1:
        links.append(f'<a class="next" href="{ctx.method_href(ids[i + 1])}"><small>{ch["next_page"]}</small>'
                     f'{html.escape(group[i + 1]["name"][ctx.lang])}</a>')
    return f'  <div class="pager">{"".join(links)}</div>\n' if links else ""


def toc(mid: str, ctx: Ctx) -> str:
    ch = CHROME[ctx.lang]
    back = sp.page_path("methods", ctx.lang) + "#" + mid
    items = "".join(f'  <a href="#{sid}">{html.escape(ch[key])}</a>\n' for sid, key in TOC)
    return (f'<aside class="toc" aria-label="{html.escape(ch["toc_aria"])}">\n  <b>{html.escape(ch["toc"])}</b>\n'
            f'{items}  <a class="toc-back" href="{back}">{html.escape(ch["to_list"])}</a>\n</aside>\n'
            f'<a class="to-list" href="{back}" aria-label="{html.escape(ch["to_list_aria"])}">↑</a>\n')


TOC = [("idea", "idea"), ("model", "model"), ("symbols", "toc_symbols"), ("pros-cons", "pros_cons"),
       ("assumptions", "assumptions"), ("map", "toc_map"), ("check", "check")]


# --- страница целиком -----------------------------------------------------------------------

def crumbs(entry: dict, ctx: Ctx) -> str:
    """BioStat › Методы и проверка › группа (якорь в списке) › метод."""
    import methods_page

    ch, lang = CHROME[ctx.lang], ctx.lang
    group = (f'<a href="{sp.page_path("methods", lang)}#{methods_page.group_id(entry["group"]["ru"])}">'
             f'{html.escape(entry["group"][lang])}</a>')
    return (f'  <nav class="crumbs" aria-label="{html.escape(ch["crumbs_aria"])}"><a href="/">BioStat</a> › '
            f'<a href="{sp.page_path("methods", lang)}">{html.escape(sp.FOOTER[lang][2])}</a> › {group} › '
            f'{html.escape(entry["name"][lang])}</nav>\n')


def method_page(mid: str, c: dict, data: dict, math: Math, lang: str) -> str:
    """HTML страницы метода mid на языке lang."""
    t, entries = c[lang], {m["id"]: m for m in data["methods"]}
    entry = entries[mid]
    values = example_values(mid, c["example"])
    ctx = Ctx(lang, values, math)
    names = {m: e["name"][lang] for m, e in entries.items()}
    article = (crumbs(entry, ctx) + f'  <h1>{html.escape(entry["name"][lang])}</h1>\n'
               f'  <p class="lead">{inline(t["lead"], ctx)}</p>\n'
               + idea_section(c, t, ctx, values) + model_section(t, ctx) + pros_cons_section(t, ctx)
               + assumptions_section(t, ctx) + map_section(c, t, ctx, names) + check_block(entry, ctx)
               + pager(mid, entry["group"]["ru"], data, ctx))
    body = f'<div class="wrap">\n<article>\n{article}</article>\n{toc(mid, ctx)}</div>\n'
    head = CSS + (JS if c.get("widget", {}).get("type") in WIDGETS else "")
    page_t = {"title": f'{entry["name"][lang]} — BioStat', "description": t["description"],
              "back": sp.PRIVACY[lang]["back"]}
    langs = tuple(code for code in LANGS if code in c)
    return sp.shell(f"methods/{mid}", page_t, lang, body, langs=langs, head=head, header="", main_class="mp")


def _tex_name(tex: str) -> str:
    """Ключ сортировки словаря: TeX без «\\», скобок и пробелов, без регистра."""
    return re.sub(r"[\\{}\s]", "", tex).lower()


def symbols_page(lang: str, math: Math, pages_by_key: dict[str, list[tuple[str, str]]]) -> str:
    """Общий словарь /methods/symbols/: все ключи по алфавиту TeX-имени и где они используются."""
    ch, symbols = CHROME[lang], _json(SYMBOLS)
    ctx = Ctx(lang, {}, math)
    rows = []
    for key in sorted(symbols, key=lambda k: _tex_name(symbols[k]["tex"])):
        used = ", ".join(f'<a href="{sp.page_path("methods/" + mid, lang)}#symbols">{html.escape(name)}</a>'
                         for mid, name in pages_by_key.get(key, []))
        rows.append([math.render(symbols[key]["tex"]), inline(symbols[key][lang], ctx), used])
    body = (f'      <p>{html.escape(ch["symbols_intro"])}</p>\n'
            + _table([ch["symbol"], ch["meaning"], ch["used_on"]], rows).replace("    ", "      ", 1))
    t = {"title": ch["symbols_title"], "description": ch["symbols_description"], "h1": ch["symbols_h1"],
         "updated": sp.PRIVACY[lang]["updated"], "back": sp.PRIVACY[lang]["back"]}
    langs = tuple(code for code in LANGS if available(code))
    return sp.shell("methods/symbols", t, lang, body, langs=langs, head=CSS, main_class="mp symbols")


def page_file(slug: str, lang: str) -> Path:
    return sp.PUBLIC / sp.page_path(slug, lang).strip("/") / "index.html"


def pages(data: dict) -> dict[Path, str]:
    """Страницы методов, словарь символов и content/math_tex.json (все TeX для KaTeX)."""
    math, out, by_key = Math(), {}, {}
    for mid, c in contents().items():
        for lang in (code for code in LANGS if code in c):
            out[page_file(f"methods/{mid}", lang)] = method_page(mid, c, data, math, lang)
            name = next(m["name"][lang] for m in data["methods"] if m["id"] == mid)
            for key in c[lang]["symbols"]:
                by_key.setdefault((lang, key), []).append((mid, name))
    for lang in (code for code in LANGS if available(code)):
        keyed = {k: v for (code, k), v in by_key.items() if code == lang}
        out[page_file("methods/symbols", lang)] = symbols_page(lang, math, keyed)
    needed = {"inline": sorted(t for t, d in math.needed if not d), "display": sorted(t for t, d in math.needed if d)}
    out[MATH_TEX] = json.dumps(needed, ensure_ascii=False, indent=1) + "\n"
    if math.missing():
        print(f"нет в content/math_cache.json: {len(math.missing())} формул — запустите node web/build-math.mjs "
              "и повторите python tools/static_pages.py", file=sys.stderr)
    return out


CHROME: dict[str, dict] = {
    "ru": {
        "crumbs_aria": "Навигация", "idea": "Идея", "model": "Модель и формулы", "toc_symbols": "Обозначения",
        "glossary": "Все обозначения на странице", "symbol": "Символ", "meaning": "Значение",
        "glossary_note": "Те же обозначения на всех страницах методов —", "glossary_link": "общий словарь символов",
        "example": "На примере", "anova_head": ["Источник", "df", "SS", "MS", "F", "P"],
        "pros_cons": "Сильные и слабые стороны", "pros": "Сильные стороны", "cons": "Слабые стороны",
        "assumptions": "Допущения", "assumptions_head": ["Допущение", "Что значит", "Как проверяет BioStat"],
        "map": "Как метод выбирает карта", "toc_map": "Как выбирает карта", "path": "Путь к листу",
        "path_aria": "Путь по карте", "neighbours": "Соседи на карте: когда выигрывают они",
        "neighbour": "Сосед", "when": "Когда карта ведёт туда", "better": "Чем сосед лучше здесь",
        "cost": "Чем платит", "checks": "Что карта проверит на этом листе", "next": "Что дальше",
        "check": "Проверка метода", "prev": "← Предыдущий", "next_page": "Следующий →",
        "toc": "На странице", "toc_aria": "На этой странице", "to_list": "↑ К списку методов",
        "to_list_aria": "К списку методов", "hint": "(подсказка)", "link_title": "Открыть описание метода",
        "symbols_title": "Обозначения на страницах методов — BioStat",
        "symbols_description": "Словарь обозначений страниц методов BioStat: символ, его значение и страницы, "
                               "где он используется. Обозначения — по Welham et al. (2015).",
        "symbols_h1": "Обозначения на страницах методов",
        "symbols_intro": "Одно обозначение — одно значение на всех страницах методов. Обозначения — по Welham et "
                         "al. (2015): t — число групп; статистика Стьюдента пишется в контексте.",
        "used_on": "Где используется",
    },
    "en": {
        "crumbs_aria": "Breadcrumbs", "idea": "Idea", "model": "Model and formulas", "toc_symbols": "Notation",
        "glossary": "All symbols on this page", "symbol": "Symbol", "meaning": "Meaning",
        "glossary_note": "The same notation is used on every method page —", "glossary_link": "symbol glossary",
        "example": "On the example", "anova_head": ["Source", "df", "SS", "MS", "F", "P"],
        "pros_cons": "Strengths and weaknesses", "pros": "Strengths", "cons": "Weaknesses",
        "assumptions": "Assumptions", "assumptions_head": ["Assumption", "What it means", "How BioStat checks it"],
        "map": "How the map chooses this method", "toc_map": "How the map chooses", "path": "Path to the leaf",
        "path_aria": "Path on the map", "neighbours": "Neighbours on the map: when they win",
        "neighbour": "Neighbour", "when": "When the map leads there", "better": "Why it beats {short} there",
        "cost": "What it costs", "checks": "What the map checks at this leaf", "next": "What next",
        "check": "Validation of the method", "prev": "← Previous", "next_page": "Next →",
        "toc": "On this page", "toc_aria": "On this page", "to_list": "↑ Back to the method list",
        "to_list_aria": "Back to the method list", "hint": "(hint)", "link_title": "Open the method description",
        "symbols_title": "Notation on the method pages — BioStat",
        "symbols_description": "Glossary of the notation on BioStat method pages: the symbol, its meaning and the "
                               "pages that use it. Notation follows Welham et al. (2015).",
        "symbols_h1": "Notation on the method pages",
        "symbols_intro": "One symbol has one meaning on every method page. Notation follows Welham et al. (2015): "
                         "t is the number of groups; Student's statistic is named in context.",
        "used_on": "Used on",
    },
    "pt": {
        "crumbs_aria": "Navegação", "idea": "Ideia", "model": "Modelo e fórmulas", "toc_symbols": "Notação",
        "glossary": "Todos os símbolos desta página", "symbol": "Símbolo", "meaning": "Significado",
        "glossary_note": "A mesma notação em todas as páginas de métodos —", "glossary_link": "glossário de símbolos",
        "example": "No exemplo", "anova_head": ["Fonte", "gl", "SQ", "QM", "F", "P"],
        "pros_cons": "Pontos fortes e fracos", "pros": "Pontos fortes", "cons": "Pontos fracos",
        "assumptions": "Pressupostos", "assumptions_head": ["Pressuposto", "O que significa", "Como o BioStat verifica"],
        "map": "Como o mapa escolhe este método", "toc_map": "Como o mapa escolhe", "path": "Caminho até à folha",
        "path_aria": "Caminho no mapa", "neighbours": "Vizinhos no mapa: quando ganham eles",
        "neighbour": "Vizinho", "when": "Quando o mapa leva lá", "better": "Em que supera {short} nesse caso",
        "cost": "O que custa", "checks": "O que o mapa verifica nesta folha", "next": "O que vem a seguir",
        "check": "Verificação do método", "prev": "← Anterior", "next_page": "Seguinte →",
        "toc": "Nesta página", "toc_aria": "Nesta página", "to_list": "↑ Voltar à lista de métodos",
        "to_list_aria": "Voltar à lista de métodos", "hint": "(sugestão)", "link_title": "Abrir a descrição do método",
        "symbols_title": "Notação nas páginas de métodos — BioStat",
        "symbols_description": "Glossário da notação das páginas de métodos do BioStat: o símbolo, o seu "
                               "significado e as páginas que o usam. Notação segundo Welham et al. (2015).",
        "symbols_h1": "Notação nas páginas de métodos",
        "symbols_intro": "Um símbolo tem um só significado em todas as páginas de métodos. Notação segundo Welham "
                         "et al. (2015): t é o número de grupos; a estatística de Student é indicada no contexto.",
        "used_on": "Usado em",
    },
}
