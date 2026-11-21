"""Шаг 3 (P1 + P3): страница «Конфиденциальность» — статичный HTML на трёх языках.

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


@pytest.mark.parametrize("lang", static_pages.LANGS)
def test_page_is_static_and_seo_complete(lang: str) -> None:
    html = PAGES[static_pages.PUBLIC / static_pages.page_path("privacy", lang).strip("/") / "index.html"]
    assert "<script" not in html and not re.search(r"\son[a-z]+=", html)  # CSP: без скриптов и обработчиков
    assert f'<html lang="{lang}">' in html and html.count("<h1>") == 1
    url = f"https://biostat.app{static_pages.page_path('privacy', lang)}"
    assert f'<link rel="canonical" href="{url}" />' in html
    assert sorted(re.findall(r'hreflang="([a-z-]+)" href=', html)) == ["en", "pt", "ru", "x-default"]
    desc = re.search(r'<meta name="description" content="([^"]+)"', html).group(1)
    assert 50 <= len(desc) <= 160


def test_same_sections_in_every_language() -> None:
    counts = {lang: len(static_pages.PRIVACY[lang]["sections"]) for lang in static_pages.LANGS}
    assert set(counts.values()) == {7}


def test_sitemap_lists_every_language_version() -> None:
    sitemap = (WEB / "public" / "sitemap.xml").read_text(encoding="utf-8")
    for lang in static_pages.LANGS:
        assert f"<loc>https://biostat.app{static_pages.page_path('privacy', lang)}</loc>" in sitemap


def test_app_footer_links_to_the_page_in_the_interface_language() -> None:
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    assert 'return lang === "en" ? `/${slug}/` : `/${lang}/${slug}/`;' in app
    assert '<a href={pagePath("privacy", lang)}' in app


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
    assert src.count("fetch(") == 1 and "await fetch(coreUrl)" in src  # только core.zip со своего сайта


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
