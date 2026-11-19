"""SEO (claude/seo-backlog.md, п. 1–3): web/index.html читается без JavaScript.

Внешний аудит 19.11.2026: H1 0/5 (сканер видел пустой #root), Open Graph 0/5
(og:title разбит на строки), description 212 знаков при норме ≤ 160.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

INDEX = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")


class _Page(HTMLParser):
    """Мета-теги и текст заголовков — как их видит сканер без JS."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: list[dict[str, str]] = []
        self.h1: list[str] = []
        self._in_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "meta":
            self.meta.append({k: v or "" for k, v in attrs})
        self._in_h1 = self._in_h1 or tag == "h1"

    def handle_endtag(self, tag: str) -> None:
        self._in_h1 = self._in_h1 and tag != "h1"

    def handle_data(self, data: str) -> None:
        if self._in_h1 and data.strip():
            self.h1.append(data.strip())


def _page() -> _Page:
    page = _Page()
    page.feed(INDEX)
    return page


def _content(attr: str, value: str) -> list[str]:
    return [m["content"] for m in _page().meta if m.get(attr) == value]


def test_static_h1_inside_root_for_scanners_without_js() -> None:
    assert _page().h1 == ["BioStat — Statistics for biology and agronomy"]
    root = INDEX[INDEX.index('<div id="root">'):INDEX.index("</div>")]
    assert "<h1>" in root and "<p>" in root


def test_description_fits_160_characters() -> None:
    (desc,) = _content("name", "description")
    assert 50 <= len(desc) <= 160


def test_every_meta_tag_is_on_one_line() -> None:
    assert not re.search(r"<meta\s*\n", INDEX)
    assert all(re.search(r"<meta [^\n]*/>", line) for line in INDEX.splitlines() if "<meta" in line)


def test_open_graph_complete_with_full_locales() -> None:
    for prop in ("og:type", "og:url", "og:title", "og:description", "og:image"):
        assert _content("property", prop), prop
    assert _content("property", "og:locale") == ["en_US"]
    assert sorted(_content("property", "og:locale:alternate")) == ["pt_PT", "ru_RU"]
