"""Шаг M1: страницы-объяснения методов (/ru/methods/<id>/) и словарь обозначений.

Совпадение файлов в web/public и content/math_tex.json с генератором проверяет и
test_static_pages.py::test_committed_pages_match_the_generator; здесь — содержание.
Ширина 375 px — tools/check_method_pages.mjs (Playwright, запуск в README).
"""

from __future__ import annotations

import html as html_lib
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT / "tools"))
import method_pages as mp  # noqa: E402
import methods_page  # noqa: E402
import static_pages  # noqa: E402
from core.validation.catalog import CATALOG  # noqa: E402

CONTENTS = mp.contents()
PAGES = static_pages.pages()
METHOD_IDS = {m.method for m in CATALOG}
WIDGET_JS = WEB / "public" / "methods" / "assets" / "groups-split.js"


def _html(mid: str, lang: str = "ru") -> str:
    return PAGES[mp.page_file(f"methods/{mid}", lang)]


def _text_only(page: str) -> str:
    """Видимый текст: без MathML-аннотаций TeX и тегов."""
    page = re.sub(r"<annotation[^>]*>.*?</annotation>", "", page, flags=re.S)
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", page))


def test_pages_exist_only_for_written_languages() -> None:
    assert {"one_way_anova", "two_sample_t", "welch_anova", "anova_permutation_test"} <= set(CONTENTS)
    for mid, c in CONTENTS.items():
        for lang in mp.LANGS:
            assert (mp.page_file(f"methods/{mid}", lang) in PAGES) == (lang in c), (mid, lang)
    assert mp.page_file("methods/symbols", "ru") in PAGES and mp.page_file("methods/symbols", "en") not in PAGES


@pytest.mark.parametrize("path", [p for p in PAGES if "methods/" in p.as_posix() and p.suffix == ".html"])
def test_committed_page_matches_generator_and_has_no_inline_script(path: Path) -> None:
    page = PAGES[path]
    assert path.read_text(encoding="utf-8") == page
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", page)  # только внешние скрипты
    assert not re.search(r"\son\w+=", page)
    assert set(re.findall(r'<script src="([^"]+)"', page)) <= {"/back.js", "/methods/assets/groups-split.js"}


def test_every_tex_is_in_the_math_cache() -> None:
    needed = json.loads(PAGES[mp.MATH_TEX])
    cache = json.loads(mp.MATH_CACHE.read_text(encoding="utf-8"))
    missing = [t for kind in ("inline", "display") for t in needed[kind] if t not in cache[kind]]
    assert missing == [], "запустите node web/build-math.mjs, затем python tools/static_pages.py"
    assert all("tex-missing" not in p for p in PAGES.values() if isinstance(p, str))


@pytest.mark.parametrize("mid", sorted(CONTENTS))
def test_leaf_neighbours_and_signals_follow_the_map(mid: str) -> None:
    c, nodes, strengths = CONTENTS[mid], mp.tree_nodes(), mp.signal_strengths()
    assert nodes[c["leaf"]]["method"] == mid
    path = set(mp.leaf_path(c["leaf"]))
    for lang in (code for code in mp.LANGS if code in c):
        t = c[lang]
        for n in t["neighbours"]:
            assert n["method"] in METHOD_IDS
            assert all(s in strengths for s in n["signals"])
            if n["signals"]:
                assert all(n["method"] in _suggested_methods(s.split("#")[0]) for s in n["signals"]), n
            else:  # сосед выбирается веткой структуры: его лист — рядом с путём
                assert _near_path(n["method"], path, nodes), n
        assert all(s in strengths for k in t["checks"] for s in k["signals"])
        linked = [m for text in _strings(t) for m in mp.method_links(text)]
        assert set(linked) <= METHOD_IDS


def _suggested_methods(signal: str) -> set[str]:
    """Методы, к которым ведёт сигнал.

    suggests — метод или узел; узел (и слот targets) раскрывается до листьев
    поддерева: pre-узел трансформации ведёт к своему листу. Метод, который правило
    подставляет в params.method (Манн–Уитни при двух группах), — строковый литерал
    в исходнике правила.
    """
    import core.validation.recognizers  # noqa: F401
    from core.validation.recognize import REGISTRY

    nodes, targets = mp.tree_nodes(), json.loads(mp.TREE.read_text(encoding="utf-8"))["targets"]
    rule = next(r for r in REGISTRY if r.id == signal)
    found = {rule.suggests} | {m for nid in targets.get(rule.suggests, [rule.suggests])
                               for m in _leaf_methods(nid, nodes)}
    if rule.fn is not None:
        source = inspect.getsource(inspect.unwrap(rule.fn))
        found |= {m for m in METHOD_IDS if f'"{m}"' in source}
    return found


def _leaf_methods(nid: str, nodes: dict) -> set[str]:
    """Методы листьев поддерева узла nid (сам лист — его метод)."""
    if nid not in nodes:
        return set()
    if nodes[nid].get("method"):
        return {nodes[nid]["method"]}
    children = [k for k, n in nodes.items() if n.get("parent") == nid]
    return {m for k in children for m in _leaf_methods(k, nodes)}


def _near_path(method: str, path: set[str], nodes: dict) -> bool:
    leaves = [n for n in nodes.values() if n.get("method") == method]
    return any(n["parent"] in path or nodes[n["parent"]].get("parent") in path for n in leaves)


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def test_symbols_are_defined_in_three_languages() -> None:
    symbols = json.loads(mp.SYMBOLS.read_text(encoding="utf-8"))
    assert all(set(v) == {"tex", "ru", "en", "pt"} and all(v.values()) for v in symbols.values())
    for c in CONTENTS.values():
        for lang in (code for code in mp.LANGS if code in c):
            assert set(c[lang]["symbols"]) <= set(symbols)


@pytest.mark.parametrize("mid", sorted(CONTENTS))
def test_substitutions_resolved_and_markup_consumed(mid: str) -> None:
    text = _text_only(_html(mid))
    for leftover in ("⟦", "⟧", "$", "**", "](method:"):
        assert leftover not in text, leftover


def test_example_numbers_come_from_the_core() -> None:
    page = _html("one_way_anova")
    tex = json.loads(PAGES[mp.MATH_TEX])["display"]
    assert r"\frac{5.53/2}{0.85/15} = \frac{2.765}{0.0567} = 48.8,\qquad P = 2.7\cdot10^{-7}" in " ".join(tex)
    assert r"\eta^2 = \frac{SS_B}{SS_T} = 0.87" in " ".join(tex) and "= 0.84" in " ".join(tex)
    row = "<tr><td>Между сортами</td><td>2</td><td>5.53</td><td>2.765</td><td>48.8</td><td>&lt; 0.001</td></tr>"
    assert row in page and "<td>Внутри сортов</td><td>15</td><td>0.85</td><td>0.0567</td>" in page
    assert "SE каждой средней 0.097, SED 0.137, LSD (5 %) 0.293; η² = 0.87, ω² = 0.84" in page
    assert "пшеница 5.30, ячмень 4.75, овёс 3.95" in page


def test_example_copy_matches_example_data_ts() -> None:
    src = (WEB / "src" / "exampleData.ts").read_text(encoding="utf-8")
    block = src[src.index("const ROWS"):src.index("];", src.index("const ROWS"))]
    rows = [list(r) for r in re.findall(r'\["([^"]+)", "([^"]+)", "([^"]+)", "([^"]+)"\]', block)]
    header = re.search(r'ru: \{ header: "([^"]+)"', src).group(1).split(";")
    example = json.loads((mp.CONTENT / "examples" / "field_trial.json").read_text(encoding="utf-8"))
    assert example["rows"] == rows and example["columns"] == header


def test_path_uses_tree_labels() -> None:
    labels = json.loads(mp.TREE_LABELS.read_text(encoding="utf-8"))["ru"]["nodes"]
    expected = [labels[n].split(":")[0].strip() for n in mp.leaf_path("B4b.anova")[1:]]
    page = _html("one_way_anova")
    path = re.search(r'<div class="path"[^>]*>(.*?)</div>', page).group(1)
    assert re.findall(r"<span[^>]*>([^<]+)</span>", path) == expected
    assert "Независимые единицы" in expected


def test_hint_marker_comes_from_registry() -> None:
    page, strengths = _html("one_way_anova"), mp.signal_strengths()
    assert strengths["B4.8"] == "hint" and strengths["B4.5"] == "strong"
    assert '<span class="sig">B4.8</span> (подсказка)' in page
    assert '<span class="sig">B4.4</span> (подсказка)' in page
    assert '<span class="sig">B4.5</span> (подсказка)' not in page


def test_neighbour_names_are_quiet_links_to_cards_or_pages() -> None:
    page = _html("one_way_anova")
    welch = '<a class="mlink" href="/ru/methods/welch_anova/" title="Открыть описание метода">'
    assert welch + "ANOVA Уэлча (неравные дисперсии)</a>" in page  # имя — из CATALOG; есть страница
    assert '<a class="mlink" href="/ru/methods/#equivalence_test"' in page  # страницы нет — карточка
    css = (WEB / "public" / "methods" / "assets" / "method.css").read_text(encoding="utf-8")
    assert ".mp a.mlink { color: inherit; text-decoration: none; }" in css
    assert ".mp a.mlink:focus-visible { outline:" in css


def test_navigation_back_to_the_list() -> None:
    page = _html("one_way_anova")
    assert '<a class="toc-back" href="/ru/methods/#one_way_anova">↑ К списку методов</a>' in page
    assert '<a class="to-list" href="/ru/methods/#one_way_anova" aria-label="К списку методов">↑</a>' in page
    assert 'href="/ru/methods/#group-several-groups">Сравнение нескольких групп</a>' in page
    pager = re.search(r'<div class="pager">(.*?)</div>', page, re.S).group(1)
    linked = re.findall(r'href="/ru/methods/([a-z0-9_]+)/"', pager)
    assert linked and set(linked) <= {m for m, c in CONTENTS.items() if "ru" in c}  # только к страницам


def test_list_page_links_detailed_methods_and_groups() -> None:
    ru, en = (PAGES[static_pages.PUBLIC / static_pages.page_path("methods", x).strip("/") / "index.html"]
              for x in ("ru", "en"))
    written = [m for m, c in CONTENTS.items() if "ru" in c]
    assert all(f'<a href="/ru/methods/{m}/">Подробно →</a>' in ru for m in written)
    assert ru.count("Подробно →") == len(written)
    assert "In detail →" not in en  # на en страницы метода нет
    assert set(methods_page.GROUP_IDS) == {m.group for m in CATALOG}
    assert re.findall(r'<h3 id="([a-z0-9_]+)">', ru) == [m.method for m in CATALOG]  # порядок и якоря


def test_ru_only_hreflang_and_sitemap() -> None:
    page = _html("one_way_anova")
    url = "https://biostat.app/ru/methods/one_way_anova/"
    assert re.findall(r'hreflang="([a-z-]+)" href="([^"]+)"', page) == [("ru", url), ("x-default", url)]
    sitemap = (WEB / "public" / "sitemap.xml").read_text(encoding="utf-8")
    assert f"<loc>{url}</loc>" in sitemap and "<loc>https://biostat.app/ru/methods/symbols/</loc>" in sitemap
    assert all(f"<loc>https://biostat.app/ru/methods/{m}/</loc>" in sitemap for m, c in CONTENTS.items() if "ru" in c)
    assert "/en/methods/one_way_anova/" not in sitemap and "https://biostat.app/methods/one_way_anova/" not in sitemap


def test_symbols_page_lists_every_key_with_pages() -> None:
    page = PAGES[mp.page_file("methods/symbols", "ru")]
    symbols = json.loads(mp.SYMBOLS.read_text(encoding="utf-8"))
    assert page.count("<tr><td>") == len(symbols)
    assert '<a href="/ru/methods/one_way_anova/#symbols">Однофакторный ANOVA</a>' in page


def test_katex_is_an_exact_dev_dependency_only() -> None:
    pkg = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
    assert pkg["devDependencies"]["katex"] == "0.16.11" and "katex" not in pkg["dependencies"]


def _node() -> str | None:
    return shutil.which("node")


def _widget(script: str) -> dict:
    """Выполнить groups-split.js в node (vm, как CommonJS) и вернуть JSON из script."""
    runner = ("const fs=require('fs'),vm=require('vm');const ctx={module:{exports:{}}};"
              f"vm.runInNewContext(fs.readFileSync({json.dumps(str(WIDGET_JS))},'utf8'),ctx);"
              f"const api=ctx.module.exports;{script}")
    done = subprocess.run([_node(), "-e", runner], capture_output=True, encoding="utf-8", check=True)
    return json.loads(done.stdout)


def _widget_data(page: str) -> dict:
    attrs = dict(re.findall(r'data-(groups|df|highlight|y|focus)="([^"]+)"', page))
    return {k: json.loads(html_lib.unescape(v)) for k, v in attrs.items()}


@pytest.mark.skipif(_node() is None, reason="нужен node")
def test_widget_p_matches_scipy_at_start() -> None:
    from scipy import stats

    d = _widget_data(_html("one_way_anova"))
    out = _widget(f"const s=api.stats({json.dumps(d['groups'])},{json.dumps(d['df'])},1,1);"
                  "console.log(JSON.stringify({F:s.F,p:s.p}));")
    expected = stats.f.sf(out["F"], *d["df"])
    assert out["p"] == pytest.approx(2.72e-7, rel=0.01) and out["p"] == pytest.approx(expected, rel=0.01)


@pytest.mark.skipif(_node() is None, reason="нужен node")
def test_static_svg_is_the_scripts_start_position() -> None:
    """SVG без JS (генератор) совпадает с раскладкой скрипта при исходных ползунках."""
    page = _html("one_way_anova")
    d = _widget_data(page)
    lay = _widget(f"console.log(JSON.stringify(api.layout({json.dumps(d['groups'])},{json.dumps(d['df'])},"
                  f"{json.dumps(d['y'])},{json.dumps(d['highlight'])},1,1)));")
    dots = re.findall(r'class="gs-dot" data-g="(\d+)" data-j="(\d+)" cx="([\d.]+)" cy="([\d.]+)"', page)
    for g, j, cx, cy in dots:
        ref = lay["dots"][int(g)][int(j)]
        assert float(cx) == pytest.approx(ref["cx"], abs=0.01) and float(cy) == pytest.approx(ref["cy"], abs=0.01)
    grand = re.search(r'class="gs-grand"[^>]* y1="([\d.]+)"', page).group(1)
    assert float(grand) == pytest.approx(lay["grandY"], abs=0.01) and len(dots) == 18


def test_widget_script_has_no_storage_or_network() -> None:
    src = WIDGET_JS.read_text(encoding="utf-8")
    for api in ("localStorage", "sessionStorage", "cookie", "fetch", "XMLHttpRequest", "eval(", "innerHTML"):
        assert api not in src, api
    assert "prefers-reduced-motion" in (WEB / "public" / "methods" / "assets" / "method.css").read_text(encoding="utf-8")



# --- Виджет Уэлча ----------------------------------------------------------------------------

def _welch_start() -> tuple[dict, dict]:
    page = _html("welch_anova")
    d = _widget_data(page)
    n = len(d["groups"][d["focus"]]["y"])
    out = _widget(f"const g={json.dumps(d['groups'])};"
                  f"const s=api.welchStats(g,{d['focus']},1,1,{n});"
                  f"const l=api.welchLayout(g,{d['focus']},{json.dumps(d['y'])},1,1,{n});"
                  "console.log(JSON.stringify({s:s,l:l}));")
    return d, out


@pytest.mark.skipif(_node() is None, reason="нужен node")
def test_welch_widget_starts_at_the_core_result() -> None:
    """Обычный F и F Уэлча скрипта по сводкам групп = ядро (one_way_anova, welch_anova)."""
    import method_examples as me

    _, out = _welch_start()
    v = me.example_values("welch_anova", CONTENTS["welch_anova"]["example"])
    c, w = out["s"]["classic"], out["s"]["welch"]
    assert [c["F"], c["df2"]] == [pytest.approx(v["F_classic"]), v["df_classic_within"]]
    assert c["p"] == pytest.approx(v["p_classic"], rel=1e-3)
    assert [w["F"], w["df2"]] == [pytest.approx(v["F"]), pytest.approx(v["df_within"])]
    assert w["p"] == pytest.approx(v["p"], rel=1e-3)


@pytest.mark.skipif(_node() is None, reason="нужен node")
def test_welch_widget_static_svg_is_the_scripts_start() -> None:
    page = _html("welch_anova")
    _, out = _welch_start()
    lay = out["l"]
    dots = re.findall(r'class="gs-dot" data-g="(\d+)" data-j="(\d+)" cx="([\d.]+)" cy="([\d.]+)"', page)
    assert len(dots) == sum(len(a) for a in lay["dots"])
    for g, j, cx, cy in dots:
        ref = lay["dots"][int(g)][int(j)]
        assert float(cx) == pytest.approx(ref["cx"], abs=0.01) and float(cy) == pytest.approx(ref["cy"], abs=0.01)
    for cls, key in (("gw-grand", "grandY"), ("gw-wmean", "wmeanY")):
        y = re.search(rf'class="{cls}"[^>]* y1="([\d.]+)"', page).group(1)
        assert float(y) == pytest.approx(lay[key], abs=0.01)
    shares = re.findall(r'class="bar gw-share" data-g="(\d+)" style="width:([\d.]+)%', page)
    for g, width in shares:
        assert float(width) == pytest.approx(max(1, 100 * out["s"]["shares"][int(g)]), abs=0.01)


@pytest.mark.skipif(_node() is None, reason="нужен node")
def test_welch_widget_n_slider_keeps_mean_and_sd() -> None:
    """Ползунок n меняет только n группы: среднее и SD те же, бледнеют точки с номера n."""
    d, _ = _welch_start()
    out = _widget(f"const g={json.dumps(d['groups'])};const a=api.welchStats(g,{d['focus']},1,1,10),"
                  f"b=api.welchStats(g,{d['focus']},1,1,3),l=api.welchLayout(g,{d['focus']},[0,40],1,1,3);"
                  "console.log(JSON.stringify({a:a.means,b:b.means,na:a.ns,nb:b.ns,"
                  f"faded:l.dots[{d['focus']}].map(x=>x.faded)}}));")
    assert out["a"] == out["b"] and out["nb"][d["focus"]] == 3 and out["na"][d["focus"]] == 10
    assert out["faded"] == [False] * 3 + [True] * 7


# --- Пример страницы и карта ---------------------------------------------------------------

# Листья по умолчанию своей развилки: положительного сигнала за них нет, карта приводит
# туда, когда ни один сигнал не уводит в сторону.
DEFAULT_METHODS = {"one_way_anova", "two_sample_t"}


def _fired_methods(fired: dict, nodes: dict, targets: dict) -> set[str]:
    """Методы, к которым ведёт сработавший сигнал, в порядке resolveTargets (web/src/mapState.ts)."""
    params = fired.get("message", {}).get("params") or {}
    s = fired["suggests"]
    if params.get("branch"):
        return _leaf_methods(params["branch"], nodes)
    if params.get("method"):
        return {params["method"]}
    if s in nodes:
        return _leaf_methods(s, nodes)
    if s in METHOD_IDS:
        return {s}
    slots = targets.get(f"{s}#{params['variant']}") if params.get("variant") else None
    return {m for nid in (slots if slots is not None else targets.get(s, [])) for m in _leaf_methods(nid, nodes)}


def _example_signals(cfg: dict) -> list[dict]:
    """Сигналы карты на колонках, которые использует страница: отклик, фактор, у RCBD — блок.

    levels — только эти уровни фактора. Роли — как в сценариях карты.
    """
    import method_examples as me
    from core.validation.parser import parse_file
    from core.validation.recognize import recognize

    example = me.load_example(cfg["data"])
    names = [cfg["factor"], cfg["response"]] + ([cfg["block"]] if "block" in cfg else [])
    cols = [me.column(example, n) for n in names]
    cols[1] = [repr(v) for v in me.numbers(cols[1])]
    rows = [r for r in zip(*cols) if r[0] in cfg.get("levels", [r[0]])]
    text = "\n".join(";".join(r) for r in [names, *rows]) + "\n"
    roles = {"value": cfg["response"], "group": cfg["factor"]} | ({"block": cfg["block"]} if "block" in cfg else {})
    out = recognize(parse_file(text.encode("utf-8")), roles)
    assert out["errors"] == []
    return out["fired"]


def _picks(fired: list[dict], nodes: dict, targets: dict) -> dict[str, set[str]]:
    """Сигналы, выбирающие лист или предобработку, → их методы.

    Сигнал, ведущий на развилку (B3.1 — «сколько уровней»), не выбирает метод: в её
    поддереве и лист страницы, и все его соседи.
    """
    params = lambda f: f.get("message", {}).get("params") or {}  # noqa: E731
    forks = {k for k, n in nodes.items() if n.get("kind") in ("test", "choice")}
    return {f["id"]: _fired_methods(f, nodes, targets) for f in fired
            if params(f).get("branch") not in forks and f["suggests"] not in forks}


def _strong_route(mid: str) -> bool:
    """Есть ли в REGISTRY сильный сигнал, ведущий к методу mid."""
    import core.validation.recognizers  # noqa: F401
    from core.validation.recognize import REGISTRY, STRONG

    return any(r.strength == STRONG and mid in _suggested_methods(r.id) for r in REGISTRY)


@pytest.mark.parametrize("mid", sorted(m for m, c in CONTENTS.items() if "example" in c))
def test_example_data_lead_the_map_to_the_method(mid: str) -> None:
    """На данных примера карта выбирает метод страницы; у листа по умолчанию — никуда не уводит.

    Сигнал должен быть сильным, если сильный путь к методу в REGISTRY есть; к Краскелу–Уоллису
    ведёт только подсказка B4.8 (баллы).
    """
    from core.validation.recognize import STRONG

    nodes, targets = mp.tree_nodes(), json.loads(mp.TREE.read_text(encoding="utf-8"))["targets"]
    fired = _example_signals(CONTENTS[mid]["example"])
    picks = _picks(fired, nodes, targets)
    if mid in DEFAULT_METHODS:  # B4.2 у t-теста ведёт на тот же лист (вариант Уэлча) — не уход
        assert all(not ms or mid in ms for ms in picks.values()), picks
        return
    need_strong = _strong_route(mid)
    assert any(mid in picks.get(f["id"], ()) and (f["strength"] == STRONG or not need_strong)
               for f in fired), picks


@pytest.mark.parametrize("sid", ["S07", "S08", "S09", "S12", "S13"])
def test_training_sets_are_copies_of_map_scenarios(sid: str) -> None:
    """content/examples/scenario_<sid>.json — колонки сценария карты без изменений."""
    import method_examples as me
    from tests.map_scenarios import SCENARIOS

    columns = next(s for s in SCENARIOS if s["id"] == sid)["columns"]
    example = me.load_example(f"scenario_{sid.lower()}")
    assert example["columns"] == list(columns)
    for name, values in columns.items():
        got = me.column(example, name)
        numeric = all(isinstance(v, (int, float)) for v in values)
        assert (me.numbers(got) if numeric else got) == ([float(v) for v in values] if numeric else values)


def test_signal_variants_on_pages_exist() -> None:
    """id#variant в соседях и проверках — есть Signal(...) с этим вариантом и текст в treeLabels.json."""
    import signal_variants

    calls = signal_variants.rule_calls()
    labels = json.loads((ROOT / "web" / "src" / "treeLabels.json").read_text(encoding="utf-8"))["ru"]["rec"]
    used = {s for c in CONTENTS.values() if "ru" in c
            for block in (c["ru"].get("neighbours", []), c["ru"].get("checks", [])) for item in block
            for s in item["signals"] if "#" in s}
    assert used, "варианты на страницах не используются — тест ничего не проверяет"
    for key in used:
        assert key in calls and key in labels, key


def test_variant_strength_comes_from_the_signal_call() -> None:
    """Явная strength= у Signal(...) варианта, иначе сила правила (B4.8#blocks)."""
    s = mp.signal_strengths()
    assert [s["B4.5"], s["B4.5#blocks"], s["B4.6"], s["B4.6#blocks"]] == ["strong", "hint", "strong", "hint"]
    assert [s["B4.3"], s["B4.3#three"], s["B4.8#blocks"]] == ["strong", "hint", "hint"]


def test_number_formats() -> None:
    """⟦p⟧ — правило сайта; rel — P со знаком; sci1 — мантисса·10^k; grp — тройки цифр."""
    sub = mp.substitute
    assert sub("⟦p⟧ ⟦p:rel⟧", {"p": 2e-4}) == "< 0.001 < 0.001"
    assert sub("⟦p:rel⟧", {"p": 0.0123}) == "= 0.012"
    assert sub("⟦x:sci1⟧", {"x": 2.72e-7}) == "2.7·10⁻⁷"
    assert sub("⟦x:sci1⟧", {"x": 2.72e-7}, tex=True) == "2.7\\cdot10^{-7}"
    assert sub("⟦n:grp⟧", {"n": 17153136}) == "17 153 136"


def test_no_equals_before_less_than() -> None:
    """«P = < 0.001» — ошибка подстановки: для P ниже порога — ⟦p:rel⟧."""
    for mid, c in CONTENTS.items():
        if "ru" in c:
            text = re.sub(r"<[^>]+>", " ", _html(mid))
            assert not re.search(r"=\s*(&lt;|<)\s*0\.001", text), mid
