"""Единое сообщение о строках, выпавших из расчёта из-за пропусков (аудит 02-08).

Методы получают массивы, а не таблицу, и неполные строки отбрасывают по-своему;
сосчитать их в одном месте можно только по таблице и ролям. Считаются роли,
которые метод использует (включая заданные необязательные слоты и выбранные
переменные регрессии); пропуск в колонке без роли строку не выбрасывает.
Полностью пустые строки убирает парсер — сюда они не попадают.
"""

from __future__ import annotations

import re
from typing import Any

from core.report.schema import Severity, StatResult
from core.validation.types import ColumnType, is_missing, parse_number

# Метод уже объяснил потерю данных точнее (отказ по неполным блокам / субъектам).
_OWN_REPORT = {"rcbd_incomplete", "rm_incomplete_subjects"}
_CODE = "missing_dropped"


def used_columns(table: Any, roles: dict[str, Any], result: StatResult) -> list[str]:
    """Колонки ролей формы и переменные, которые выбрал сам метод, без повторов."""
    names = [v for v in roles.values() if isinstance(v, str)]
    names += [v for v in result.parameters.get("variables") or [] if isinstance(v, str)]
    return [n for n in dict.fromkeys(names) if n in table.header]


def _missing_mask(table: Any, name: str) -> list[bool]:
    """Пропуск по строкам; в числовой колонке пропуском считается и нечисло."""
    profile = next(p for p in table.profiles if p.name == name)
    numeric = profile.detected_type == ColumnType.NUMERIC
    decimal = table.dialect.decimal
    return [is_missing(raw) or (numeric and parse_number(raw, decimal) is None)
            for raw in table.column_values(name)]


def dropped_rows(table: Any, columns: list[str]) -> tuple[int, dict[str, int]]:
    """Число строк с пропуском хотя бы в одной колонке и пропуски по колонкам."""
    masks = {name: _missing_mask(table, name) for name in columns}
    by_column = {name: sum(mask) for name, mask in masks.items() if any(mask)}
    total = sum(any(row) for row in zip(*masks.values())) if masks else 0
    return total, by_column


def note_dropped_rows(result: StatResult, table: Any, roles: dict[str, Any]) -> None:
    """Одно сообщение `missing_dropped` на результат: сколько строк выпало и где пропуски."""
    codes = {w.code for w in result.warnings}
    total, by_column = dropped_rows(table, used_columns(table, roles, result))
    if codes & _OWN_REPORT:
        return
    if total == 0:
        _fill_own_note(result)
        return
    result.warnings = [w for w in result.warnings if w.code != _CODE]
    detail = ", ".join(f"{name} — {n}" for name, n in by_column.items())
    result.add_warning(_CODE, f"Строки с пропусками не вошли в расчёт: {total} "
                       f"(пропуски по колонкам: {detail}).", Severity.INFO,
                       params={"n": total, "detail": detail})


def _fill_own_note(result: StatResult) -> None:
    """Сообщение самого метода без числа в params: взять число из его текста."""
    for w in result.warnings:
        if w.code == _CODE and "n" not in w.params:
            found = re.search(r"(\d+)\D*$", w.message)
            w.params.update({"n": int(found.group(1)) if found else "?", "detail": "—"})
