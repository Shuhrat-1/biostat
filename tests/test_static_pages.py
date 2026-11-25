"""Шаги 3 и 4 (P1 + P3, P2): страницы «Конфиденциальность» и «Условия» — статичный HTML
на трёх языках.

Факты страницы сверены с кодом (тесты ниже) и с журналом сетевых запросов
собранного сайта в Chrome: открытие, пример данных, расчёт, пять выгрузок —
только GET к своему сайту и 11 файлов пакетов с jsDelivr, ни одного POST;
после работы в браузере только localStorage «biostat.lang», cookie нет.
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
import static_pages  # noqa: E402

PAGES = static_pages.pages()


def test_committed_pages_match_the_generator() -> None:
    for path, text in PAGES.items():
        assert path.read_text(encoding="utf-8") == text, f"{path}: запустите python tools/static_pages.py"


SLUGS = tuple(static_pages.PAGES)


def _page(slug: str, lang: str) -> str:
    return PAGES[static_pages.PUBLIC / static_pages.page_path(slug, lang).strip("/") / "index.html"]


def test_privacy_and_terms_pages_exist() -> None:
    assert SLUGS == ("privacy", "terms")


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_page_is_static_and_seo_complete(slug: str, lang: str) -> None:
    html = _page(slug, lang)
    # CSP: без встроенных скриптов и обработчиков; единственный скрипт — внешний /back.js
    assert re.findall(r"<script[^>]*>.*?</script>", html, re.S) == ['<script src="/back.js" defer></script>']
    assert not re.search(r"\son[a-z]+=", html)
    assert f'<html lang="{lang}">' in html and html.count("<h1>") == 1
    url = f"https://biostat.app{static_pages.page_path(slug, lang)}"
    assert f'<link rel="canonical" href="{url}" />' in html
    assert sorted(re.findall(r'hreflang="([a-z-]+)" href=', html)) == ["en", "pt", "ru", "x-default"]
    desc = re.search(r'<meta name="description" content="([^"]+)"', html).group(1)
    assert 50 <= len(desc) <= 160


@pytest.mark.parametrize(("slug", "count"), [("privacy", 7), ("terms", 5)])
def test_same_sections_in_every_language(slug: str, count: int) -> None:
    counts = {lang: len(static_pages.PAGES[slug][lang]["sections"]) for lang in static_pages.LANGS}
    assert set(counts.values()) == {count}


@pytest.mark.parametrize("slug", SLUGS)
def test_sitemap_lists_every_language_version(slug: str) -> None:
    sitemap = (WEB / "public" / "sitemap.xml").read_text(encoding="utf-8")
    for lang in static_pages.LANGS:
        assert f"<loc>https://biostat.app{static_pages.page_path(slug, lang)}</loc>" in sitemap


def test_app_footer_links_to_the_page_in_the_interface_language() -> None:
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    assert 'return lang === "en" ? `/${slug}/` : `/${lang}/${slug}/`;' in app
    for link in ('[pagePath("privacy", lang), t("footer.privacy")]', '[pagePath("terms", lang), t("footer.terms")]',
                 '[pagePath("methods", lang), t("footer.methods")]', '["/LICENSE.txt", t("footer.license")]'):
        assert link in app, link


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_static_footer_links_privacy_terms_and_license(slug: str, lang: str) -> None:
    footer = re.search(r"<footer>(.*)</footer>", _page(slug, lang)).group(1)
    assert re.findall(r'href="([^"]+)"', footer) == [static_pages.page_path("privacy", lang),
                                                     static_pages.page_path("terms", lang),
                                                     static_pages.page_path("methods", lang), "/LICENSE.txt"]


@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_terms_cover_the_required_points(lang: str) -> None:
    html = _page("terms", lang)
    for needle in ("Welham", "Witte", "2015", "Apache License 2.0", "Copyright 2026 Shuhrat Maksumov",
                   '<a href="/LICENSE.txt">', '<a href="/third-party-licenses.txt">'):
        assert needle in html, needle
    assert len(re.findall(r"<h2>", html)) == 5  # как есть, ответственность, источники, не консультация, лицензия


def test_inline_link_markup_is_escaped_outside_links() -> None:
    assert static_pages._inline("a < b [x](/y) & c") == 'a &lt; b <a href="/y">x</a> &amp; c'


# --- факты страницы сверены с кодом -----------------------------------------------------

def _src() -> str:
    return "".join(p.read_text(encoding="utf-8") for p in (WEB / "src").rglob("*.ts*")
                   if "bench" not in p.parts)


def test_only_the_language_is_stored_and_nothing_is_sent() -> None:
    src = _src()
    assert re.findall(r"localStorage\.(?:get|set)Item\(([A-Z_]+)", src) == ["LANG_KEY", "LANG_KEY"]
    assert 'const LANG_KEY = "biostat.lang";' in src
    for api in ("document.cookie", "sessionStorage", "indexedDB", "serviceWorker", "sendBeacon",
                "XMLHttpRequest"):
        assert api not in src, api
    # только свой сайт: core.zip и — при переходе со страницы метода (?method=, L1) — /methods.json
    assert src.count("fetch(") == 2 and "await fetch(coreUrl)" in src and 'await fetch("/methods.json")' in src


def test_eleven_package_files_come_from_jsdelivr() -> None:
    lock_path = WEB / "node_modules" / "pyodide" / "pyodide-lock.json"
    if not lock_path.exists():
        pytest.skip("нет web/node_modules (npm install)")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["packages"]
    worker = (WEB / "src" / "pyodide" / "worker.ts").read_text(encoding="utf-8")
    todo = re.findall(r'"([\w-]+)"', re.search(r"REQUIRED_PACKAGES = \[([^\]]+)\]", worker).group(1))
    seen: set[str] = set()
    while todo:
        name = todo.pop().lower().replace("_", "-")
        if name not in seen:
            seen.add(name)
            todo.extend(lock[name]["depends"])
    assert len(seen) == 11  # столько файлов называет страница
