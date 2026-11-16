"""Аудит 03-05: широкий или огромный файл не замораживает вкладку.

Замер в Chrome на собранном сайте, 5 строк × 50 000 колонок (1.4 МБ): было —
пауза страницы 19 с, 400 тыс. узлов DOM, готовность через 70 с; причина —
таблица на все колонки и 101 МБ JSON из Worker'а, из них 79 МБ одинаковых
списков колонок в 162 слотах методов (json.dumps — 58 с в Pyodide).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.validation.parser import suggest_methods
from tools.bench_methods import template

WEB = Path(__file__).resolve().parents[1] / "web" / "src"


def _wide_csv(ncol: int, nrow: int = 5) -> bytes:
    header = ",".join(f"c{i}" for i in range(ncol))
    rows = (",".join(str((r * i) % 17 + 0.5) for i in range(ncol)) for r in range(nrow))
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _bridge_parse(data: bytes) -> dict:
    """Шаблон parseFile моста, исполненный в CPython; возвращает отправляемый payload."""
    code = template("parseFile").replace("${JSON.stringify(lang)}", '"ru"')
    scope = {"__overrides_json": "{}", "__file_bytes": data}
    exec(code, scope)
    return scope["_payload"]


def _unpack(payload: dict) -> list[dict]:
    """То же, что unpackColumnLists в bridge.ts."""
    lists = payload["column_lists"]
    return [{**s, "needs": {slot: lists[i] for slot, i in s["needs"].items()}}
            for s in payload["suggestions"]]


def test_03_05_column_lists_are_sent_once_and_unpack_to_the_same_suggestions() -> None:
    payload = _bridge_parse(_wide_csv(300))
    assert all(isinstance(i, int) for s in payload["suggestions"] for i in s["needs"].values())
    assert len(payload["column_lists"]) < 10  # 162 слота, единицы уникальных списков
    table = __import__("builtins").__biostat_table  # шаблон кладёт разобранную таблицу сюда
    assert _unpack(payload) == suggest_methods(table.profiles, "ru")


def test_03_05_payload_no_longer_grows_with_the_number_of_slots() -> None:
    payload = _bridge_parse(_wide_csv(5000))
    size = len(json.dumps(payload, ensure_ascii=False)) / 1e6
    assert size < 3.5  # было 9.4 МБ на 5 000 колонках (2.2 МБ профили + 7.2 МБ списки)


def test_03_05_front_unpacks_the_lists() -> None:
    bridge = (WEB / "pyodide" / "bridge.ts").read_text(encoding="utf-8")
    assert "return unpackColumnLists(JSON.parse(raw));" in bridge
    assert re.search(r"needs: Object\.fromEntries\(Object\.entries\(sug\.needs\)\.map\("
                     r"\(\[slot, i\]\) => \[slot, lists\[i\]\]\)\)", bridge)


def test_03_05_preview_and_column_list_show_at_most_200_columns() -> None:
    limits = (WEB / "limits.ts").read_text(encoding="utf-8")
    assert "export const MAX_SHOWN_COLUMNS = 200;" in limits
    assert "export const MAX_FILE_BYTES = 50 * 1024 * 1024;" in limits
    comp = (WEB / "components.tsx").read_text(encoding="utf-8")
    assert "const shownHeader = table.header.slice(0, MAX_SHOWN_COLUMNS);" in comp
    assert comp.count("shownHeader.map(") == 2 and "{table.header.map(" not in comp
    assert ").slice(0, MAX_SHOWN_COLUMNS);" in comp


def test_03_05_oversized_file_is_refused_before_reading() -> None:
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    head = app[app.index("const handleFile = async"):app.index("file.arrayBuffer()")]
    assert "if (file.size > MAX_FILE_BYTES) {" in head and "return;" in head
    i18n = (WEB / "i18n.tsx").read_text(encoding="utf-8")
    for key in ("error.fileOverLimit", "preview.columnsCapped", "preview.detailsCapped"):
        assert i18n.count(f'"{key}": ') == 3, key


def test_03_05_column_lookup_is_not_quadratic() -> None:
    """Сводка по колонкам искала каждую перебором заголовка: 20 000 колонок — 2.9 с в CPython,
    50 000 — 48 с в Pyodide. Проверка не по секундам (машины разные), а по росту: вчетверо
    больше колонок — при линейном поиске около ×4, при квадратичном около ×16."""
    import time

    from core.validation.parser import parse_file

    def best(ncol: int) -> float:
        table = parse_file(_wide_csv(ncol, nrow=1))
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            for name in table.header:  # то, что делает сводка: каждая колонка по имени
                table.column_values(name)
            runs.append(time.perf_counter() - start)
        return min(runs)

    assert best(16000) / best(4000) < 9


def test_03_05_column_lookup_keeps_first_duplicate_and_errors() -> None:
    import pytest

    from core.validation.parser import parse_file

    table = parse_file(b"a;b;a\n1;2;3\n4;5;6\n")
    assert table.header == ["a", "b", "a"] or table.header[:2] == ["a", "b"]
    assert table.column_index("a") == 0 and table.column_values("b") == ["2", "5"]
    with pytest.raises(ValueError, match="Нет колонки: zz"):
        table.column_values("zz")
