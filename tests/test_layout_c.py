"""Аудит 5: раскладка C (05-01) — список методов свёрнут до выбранного;
05-03 — предпросмотр сворачивается после первого расчёта."""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")
APP = (WEB / "App.tsx").read_text(encoding="utf-8")
I18N = (WEB / "i18n.tsx").read_text(encoding="utf-8")


def _method_panel() -> str:
    start = COMPONENTS.index("export function MethodPanel(")
    return COMPONENTS[start:COMPONENTS.index("\nfunction optionLabel(", start)]


def test_05_01_list_is_collapsed_to_the_selected_card_by_default() -> None:
    panel = _method_panel()
    assert "const [listOpen, setListOpen] = useState(false);" in panel
    assert "listOpen ? groupMethods : groupMethods.filter((m) => m.method === method)" in panel
    assert "{shownMethods.map((m) => {" in panel and "{groupMethods.map(" not in panel


def test_05_01_list_collapses_again_on_card_tab_map_and_new_table() -> None:
    panel = _method_panel()
    # выбор карточки, вкладка, лист карты, новая таблица
    assert panel.count("setListOpen(false);") == 4


def test_05_01_change_method_button_is_full_width_and_announces_state() -> None:
    panel = _method_panel()
    button = panel[panel.index("{others > 0 && ("):panel.index("{/* Колонки и параметры")]
    for needle in ('aria-expanded={listOpen}', 'aria-controls="method-list"', 'width: "100%"',
                   "minHeight: 40", 't("method.change").replace("{n}", String(others))'):
        assert needle in button, needle


def test_05_03_preview_folds_after_the_first_run_unless_user_chose() -> None:
    assert "if (!previewTouched.current) setPreviewFolded(true);" in APP
    # новый файл, пример данных, сброс — снова раскрыт
    assert APP.count("unfoldPreview();") == 3
    assert "folded={previewFolded}" in APP and "previewTouched.current = true;" in APP


def test_05_03_preview_body_hidden_when_folded_with_a_toggle() -> None:
    start = COMPONENTS.index("export function PreviewPanel(")
    panel = COMPONENTS[start:COMPONENTS.index("\nfunction ColumnDetails(", start)]
    assert "{!folded && (" in panel and '<div id="preview-body">' in panel
    assert "aria-expanded={!folded}" in panel and 'aria-controls="preview-body"' in panel


def test_new_texts_exist_in_three_languages() -> None:
    for key in ("method.change", "method.hideList", "preview.fold", "preview.unfold"):
        assert len(re.findall(rf'"{re.escape(key)}": "', I18N)) == 3, key
    assert I18N.count("{n}") >= 3
