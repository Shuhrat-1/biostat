"""Переход на страницы подвала не теряет файл и расчёт; имя файла — в «Качестве данных».

- Подвал приложения открывает страницы в новой вкладке (вкладка с расчётом не трогается).
- «← Вернуться в BioStat» на статичных страницах (web/public/back.js) возвращается по
  истории к уже открытому приложению — браузер восстанавливает его из кеша
  «назад/вперёд» (проверено в Chrome); без Navigation API ссылка обычная.
- «Загрузить другой файл» при загруженном файле — в строке с его именем; в
  планировании опыта (без файла) кнопка остаётся под результатом.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
BACK = WEB / "public" / "back.js"
APP = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "src" / "components.tsx").read_text(encoding="utf-8")

# Обвязка: back.js в node с поддельными document и navigation; печатает, что сделал клик.
HARNESS = r"""
const fs = require("fs"); const vm = require("vm");
const [file, entriesJson, index, eventJson, hasNav] = process.argv.slice(1);
let handler = null; const traversed = [];
const link = { addEventListener: (type, fn) => { if (type === "click") handler = fn; } };
const entries = JSON.parse(entriesJson).map((e) => ({ ...e, getState: () => e.state ?? null }));
const navigation = { entries: () => entries, currentEntry: { index: Number(index) },
  traverseTo: (key) => traversed.push(key) };
const window = hasNav === "1" ? { navigation } : {};
vm.runInNewContext(fs.readFileSync(file, "utf8"),
  { document: { querySelector: (sel) => (sel === "a[data-back]" ? link : null) }, window, URL });
let prevented = false;
if (handler) handler({ button: 0, ...JSON.parse(eventJson), preventDefault: () => { prevented = true; } });
console.log(JSON.stringify({ listened: Boolean(handler), prevented, traversed }));
"""

def _node() -> str | None:
    """Node 22.6+ (флаг --experimental-strip-types), как в test_front_escape.py."""
    exe = shutil.which("node")
    if not exe:
        return None
    version = subprocess.run([exe, "--version"], capture_output=True, encoding="utf-8").stdout.strip()
    major, minor = (int(x) for x in version.lstrip("v").split(".")[:2])
    return exe if (major, minor) >= (22, 6) else None


# url чужих записей Navigation API скрывает (Referrer-Policy: no-referrer) — видно только состояние.
APP_ENTRY = {"url": None, "key": "app", "state": {"biostat": "app"}}
PRIVACY = {"url": None, "key": "p"}
TERMS = {"url": "https://biostat.app/ru/terms/", "key": "t"}


def _click(entries: list[dict], index: int, event: dict | None = None, nav: bool = True) -> dict:
    done = subprocess.run(["node", "-e", HARNESS, str(BACK), json.dumps(entries), str(index),
                           json.dumps(event or {}), "1" if nav else "0"],
                          capture_output=True, encoding="utf-8", check=True)
    return json.loads(done.stdout)


def test_back_link_returns_to_the_open_app_through_history() -> None:
    assert _click([APP_ENTRY, PRIVACY], 1) == {"listened": True, "prevented": True, "traversed": ["app"]}


def test_back_link_skips_other_static_pages_to_reach_the_app() -> None:
    out = _click([APP_ENTRY, PRIVACY, TERMS], 2)
    assert out["traversed"] == ["app"]


@pytest.mark.parametrize("event", [{"ctrlKey": True}, {"metaKey": True}, {"shiftKey": True}, {"button": 1}])
def test_modified_clicks_keep_the_normal_link(event: dict) -> None:
    assert _click([APP_ENTRY, PRIVACY], 1, event)["prevented"] is False


def test_without_the_app_in_history_the_link_is_normal() -> None:
    assert _click([PRIVACY, TERMS], 1) == {"listened": True, "prevented": False, "traversed": []}


def test_without_navigation_api_nothing_is_attached() -> None:
    assert _click([APP_ENTRY, PRIVACY], 1, nav=False)["listened"] is False


def test_app_marks_its_history_entry() -> None:
    main = (WEB / "src" / "main.tsx").read_text(encoding="utf-8")
    mark = (WEB / "src" / "historyMark.ts").read_text(encoding="utf-8")
    assert "markAppHistoryEntry();" in main
    assert 'APP_HISTORY_STATE = { biostat: "app" }' in mark
    assert 'state.biostat === "app"' in BACK.read_text(encoding="utf-8")


def test_back_script_stores_and_sends_nothing() -> None:
    src = BACK.read_text(encoding="utf-8")
    for api in ("localStorage", "sessionStorage", "cookie", "indexedDB", "fetch", "XMLHttpRequest",
                "sendBeacon"):
        assert api not in src, api


def test_footer_links_open_in_a_new_tab_with_a_screen_reader_note() -> None:
    assert '<a href={href} target="_blank" rel="noopener"' in APP
    assert '{t("footer.newTab")}' in APP
    i18n = (WEB / "src" / "i18n.tsx").read_text(encoding="utf-8")
    assert i18n.count('"footer.newTab":') == 3 and i18n.count('"quality.file":') == 3


def test_reset_button_sits_next_to_the_file_name() -> None:
    assert '<QualityPanel table={table} fileName={inputFile?.name ?? ""} onReset={reset} />' in APP
    assert "<FileRow name={fileName} onReset={onReset} />" in COMPONENTS
    row = COMPONENTS[COMPONENTS.index("function FileRow"):]
    assert '{t("quality.file")}' in row and "<strong>{name}</strong>" in row
    assert re.search(r"onClick=\{onReset\}[\s\S]{0,200}\{t\(\"reset\"\)\}", row)


def test_bottom_reset_button_only_without_a_file() -> None:
    """Под результатом кнопка остаётся только для планирования опыта (table = null)."""
    block = APP[APP.index("{/* С файлом кнопка"):]
    assert block.split("\n", 2)[1].strip() == "{!table && ("
    assert APP.count('{t("reset")}') == 1


def test_mark_sets_state_only_where_navigation_api_exists() -> None:
    if _node() is None:
        pytest.skip("нужен Node 22.6+ для запуска web/src/historyMark.ts")
    url = (WEB / "src" / "historyMark.ts").as_uri()
    script = (f'import {{ markAppHistoryEntry }} from "{url}"; const got = [];'
              "const nav = { updateCurrentEntry: (o) => got.push(o.state) };"
              "console.log(JSON.stringify([markAppHistoryEntry({ navigation: nav }), markAppHistoryEntry({}),"
              " markAppHistoryEntry({ navigation: { updateCurrentEntry: () => { throw new Error('x'); } } }), got]));")
    done = subprocess.run([_node(), "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", script],
                          capture_output=True, encoding="utf-8", check=True)
    assert json.loads(done.stdout) == [True, False, False, [{"biostat": "app"}]]
