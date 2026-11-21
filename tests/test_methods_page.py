"""Шаг 5 (P5): страница «Методы и проверка» собирается из каталога, карты и тестов.

Совпадение закоммиченных страниц и methods.json с генератором проверяет
test_static_pages.py::test_committed_pages_match_the_generator (methods_page входит в pages()).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT / "tools"))
import methods_page  # noqa: E402
import static_pages  # noqa: E402
from core.validation.catalog import CATALOG  # noqa: E402

DATA = json.loads(methods_page.JSON_OUT.read_text(encoding="utf-8"))
IDS = [m.method for m in CATALOG]


def _html(lang: str) -> str:
    path = static_pages.PUBLIC / static_pages.page_path("methods", lang).strip("/") / "index.html"
    return path.read_text(encoding="utf-8")


def _method(method: str) -> dict:
    return next(m for m in DATA["methods"] if m["id"] == method)


@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_every_catalog_method_is_on_the_page(lang: str) -> None:
    html = _html(lang)
    assert re.findall(r'<h3 id="([a-z0-9_]+)">', html) == IDS
    assert DATA["methods_count"] == len(IDS) == len(DATA["methods"])
    assert [m["id"] for m in DATA["methods"]] == IDS


def test_every_method_has_a_source_in_the_map_tree() -> None:
    assert [m for m in IDS if not methods_page.tree_sources()[m]] == []


def test_audit_reference_covers_catalog_methods_only() -> None:
    reference = json.loads(methods_page.REFERENCE.read_text(encoding="utf-8"))
    assert set(reference["methods"]) <= set(IDS) and len(reference["methods"]) == 41
    assert all(set(v) == {"ru", "en", "pt"} and all(v.values()) for v in reference["methods"].values())
    unverified = [m["id"] for m in DATA["methods"] if not m["audit_reference"]]
    assert len(unverified) == len(IDS) - 41
    ru = _html("ru")
    block = ru[ru.index("<h2>Не проверено"):ru.index("<h2>Пометка")]
    assert re.findall(r'<a href="#([a-z0-9_]+)">', block) == unverified


def test_proposed_mark_follows_the_tree() -> None:
    for m in DATA["methods"]:
        assert m["proposed"] == any("предл." in s for s in m["sources"]), m["id"]
    assert _method("rm_anova")["proposed"] and not _method("one_way_anova")["proposed"]


def test_every_biostat_mark_in_the_code_is_listed() -> None:
    """Новый порог [BioStat] в коде без строки на странице — тест падает (и наоборот)."""
    anchors = [a for rule in methods_page.BIOSTAT_RULES for a in rule["anchors"]]
    marked = methods_page.biostat_marked_lines()
    unlisted = [(f, line) for f, line in marked if not any(a in line for a in anchors)]
    stale = [a for a in anchors if not any(a in line for _f, line in marked)]
    assert unlisted == [] and stale == []
    assert all(set(r["text"]) == {"ru", "en", "pt"} for r in methods_page.BIOSTAT_RULES)


def test_book_refs_come_from_tests_that_call_the_method() -> None:
    assert "Table 4.5" in _method("one_way_anova")["book_refs"]["welham"]
    assert "Table 18.10" not in _method("one_way_anova")["book_refs"]["welham"]  # только карта
    assert "Table 8.4" in _method("factorial_anova")["book_refs"]["welham"]
    assert "Table 20.2" in _method("mann_whitney")["book_refs"]["witte"]
    assert "lme4" in _method("glmm_analysis")["external_reference_in_tests"]


def test_called_counts_function_and_bridge_calls_only() -> None:
    import ast

    tree = ast.parse('one_way_anova(a)\nrun("mann_whitney", t)\nx = {"wilcoxon": 1}\ntable("ancova")\n')
    assert methods_page._called(tree) == {"one_way_anova", "mann_whitney"}


def test_sources_are_translated_like_the_map() -> None:
    words = methods_page.source_words()
    assert methods_page.format_source("кн. стр. 74–84", "en", words) == "book p. 74–84"
    assert methods_page.format_source("кн. гл. 14", "pt", words) == "livro cap. 14"
    assert methods_page.format_source("кн. стр. 74–84", "ru", words) == "кн. стр. 74–84"


@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_page_is_static_and_seo_complete(lang: str) -> None:
    html = _html(lang)
    assert "<script" not in html and not re.search(r"\son[a-z]+=", html)
    url = f"https://biostat.app{static_pages.page_path('methods', lang)}"
    assert f'<link rel="canonical" href="{url}" />' in html
    assert sorted(re.findall(r'hreflang="([a-z-]+)" href=', html)) == ["en", "pt", "ru", "x-default"]
    desc = re.search(r'<meta name="description" content="([^"]+)"', html).group(1)
    assert 50 <= len(desc) <= 160
    assert '<a href="/methods.json">' in html


def test_sitemap_and_footers_link_the_page() -> None:
    sitemap = (WEB / "public" / "sitemap.xml").read_text(encoding="utf-8")
    for lang in static_pages.LANGS:
        assert f"<loc>https://biostat.app{static_pages.page_path('methods', lang)}</loc>" in sitemap
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    assert '<a href={pagePath("methods", lang)}' in app
