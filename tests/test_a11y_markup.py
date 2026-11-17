"""Аудит 04-04: разметка для экранного диктора.

`<main>`, связанные подписи полей, роли вкладок и aria-pressed, текстовая
альтернатива графиков со сводкой, ссылка-пропуск «К расчёту». Проверено в Chrome
axe-core на шести экранах: было 3 критических правила (label, select-name) на
трёх экранах, стало 0.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
APP = (WEB / "App.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")
CHARTS = (WEB / "charts.tsx").read_text(encoding="utf-8")
MAP = (WEB / "SensoryMap.tsx").read_text(encoding="utf-8")
I18N = (WEB / "i18n.tsx").read_text(encoding="utf-8")


def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)]


def test_landmarks_header_and_main() -> None:
    assert APP.count("<header") == 1 and APP.count("<main>") == 1 and APP.count("</main>") == 1
    assert APP.index("<SkipLink />") < APP.index("<header") < APP.index("<main>")


def test_skip_link_leads_to_the_form() -> None:
    link = _block(APP, "function SkipLink()", "\nfunction LoadingBar(")
    assert 'href="#calc"' in link and "target.focus();" in link and 't("skip.calc")' in link
    assert "clipPath" in link  # скрыта, пока не в фокусе, но доступна с клавиатуры
    assert COMPONENTS.count('id="calc" tabIndex={-1}') == 2  # форма и пустая панель


def test_method_groups_are_tabs_with_arrow_keys() -> None:
    panel = _block(COMPONENTS, "export function MethodPanel(", "\nfunction optionLabel(")
    for needle in ('role="tablist"', 'role="tab"', "aria-selected={active}", "tabIndex={active ? 0 : -1}",
                   'role="tabpanel"', "aria-labelledby={`${uid}-tab-", "ArrowRight", "ArrowLeft",
                   "Home", "End", "aria-pressed={selected}"):
        assert needle in panel, needle


def test_toggles_announce_their_state() -> None:
    assert "aria-pressed={lang === l.code}" in APP
    assert "aria-pressed={props.context === ctx}" in MAP
    assert 'role="group"' in MAP and "aria-expanded={props.open}" in MAP


def test_every_form_field_has_a_linked_label() -> None:
    panel = _block(COMPONENTS, "export function MethodPanel(", "\nfunction optionLabel(")
    assert "htmlFor={`${uid}-slot-${slot}`}" in panel and "id={`${uid}-slot-${slot}`}" in panel
    field = _block(COMPONENTS, "function ParamField(", "\n}\n")
    assert field.count("htmlFor={id}") == 3 and field.count("id={id}") == 3
    assert 'aria-label={`${param.label}: ${t("param.source")}`}' in field
    assert not re.search(r"<label style=\{s\.label\}>", COMPONENTS)


def test_charts_have_a_text_alternative_with_a_summary() -> None:
    assert '<div role="img" aria-label={summary ? `${title}. ${summary}` : title}>' in CHARTS
    summary = _block(CHARTS, "export function chartSummary(", "\n/** Отрисовать")
    for key in ("boxplot", "bar_means", "means_ci", "histogram", "scatter", "qq", "residual_vs_fitted",
                "group_curves", "paired", "frequency", "differences"):
        assert f'case "{key}"' in summary, key


def test_new_texts_in_three_languages() -> None:
    keys = ["skip.calc", "param.source"] + [f"plots.sum.{k}" for k in (
        "groups", "medians", "means", "obs", "range", "points", "curves", "pairs", "categories",
        "comparisons")]
    for key in keys:
        assert I18N.count(f'"{key}": ') == 3, key
