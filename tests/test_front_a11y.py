"""Аудиты 4 и 5: доступность и первые шаги раскладки (04-01, 04-03, 04-05, 04-06, 04-08,
04-11, 05-04, 05-05). Поведение проверено в Chrome на собранном сайте; здесь — то, что
можно проверить без браузера."""

from __future__ import annotations

from pathlib import Path

from core.validation.catalog import CATALOG

WEB = Path(__file__).resolve().parents[1] / "web" / "src"


def _src(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_04_01_upload_zone_is_a_keyboard_button() -> None:
    """Зона загрузки не получала фокус: файл нельзя было загрузить без мыши (WCAG 2.1.1)."""
    zone = _src("components.tsx").split('type="file"')[0][-1400:]
    assert 'role="button"' in zone and "tabIndex={0}" in zone
    assert 'e.key !== "Enter" && e.key !== " "' in zone and "input.current?.click()" in zone


def test_04_03_result_is_scrolled_into_view_and_focused() -> None:
    app = _src("App.tsx")
    assert 'resultRef.current.scrollIntoView({ block: "start" })' in app
    assert "resultRef.current.focus({ preventScroll: true })" in app
    assert 'ref={resultRef} tabIndex={-1} aria-live="polite"' in app


def test_04_05_severity_is_shown_by_icon_and_word() -> None:
    comp, i18n = _src("components.tsx"), _src("i18n.tsx")
    assert 'SEVERITY_ICON: Record<string, string> = { info: "ℹ", warning: "⚠", error: "✕" }' in comp
    assert "t(`severity.${" in comp
    for level in ("error", "warning", "info"):
        assert i18n.count(f'"severity.{level}": ') == 3
    assert "<strong style={{ fontWeight: 600 }}>{t(`map.strength.${sig.strength}`)}:</strong>" in _src(
        "SensoryMap.tsx")


def test_04_06_axis_names_and_heatmap_numbers_have_contrast() -> None:
    charts = _src("charts.tsx")
    assert "nameTextStyle: { color: chart.axis }" in charts  # #52525b, 7.7:1 (патч 125)
    assert "Math.abs(cell[2]) > 0.55 * limit ? c.panel : c.text" in charts


def test_04_08_signal_source_wraps_under_the_text_on_narrow_screens() -> None:
    row = _src("SensoryMap.tsx").split("signals.map((sig) => (")[1][:400]
    assert 'flexWrap: "wrap"' in row


def test_04_11_map_targets_are_at_least_24_px() -> None:
    assert _src("SensoryMap.tsx").count("minHeight: 24") == 2


def test_05_04_two_groups_tab_opens_on_the_t_test() -> None:
    """Вкладка открывается на первом методе группы: было «Две доли», нужен t-тест."""
    two = [m.method for m in CATALOG if m.group == "Сравнение двух групп"]
    assert two[:2] == ["two_sample_t", "mann_whitney"]
    assert len(CATALOG) == 71 and len({m.method for m in CATALOG}) == 71
