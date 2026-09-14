"""Разбор загруженного файла: детекция, парсинг, профилирование.

Единая точка входа для интерфейса: на вход сырые байты, на выход —
распознанные параметры, данные для предпросмотра, профили колонок и
числовые ряды, готовые к передаче в статистические методы.

Фронтенд делает один вызов вместо оркестрации трёх модулей.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from core.validation.dialect import DialectInfo, detect_dialect
from core.validation.messages import msg
from core.validation.types import (
    ColumnProfile,
    ColumnType,
    is_missing,
    parse_number,
    profile_table,
)

# Сколько строк показывать в предпросмотре.
PREVIEW_ROWS = 20
# Предел строк для браузерного уровня — дальше нужен другой бэкенд.
BROWSER_ROW_LIMIT = 200_000


@dataclass
class ParsedTable:
    """Результат разбора файла со всем, что нужно интерфейсу."""

    dialect: DialectInfo
    header: list[str]
    preview_rows: list[list[str]]
    profiles: list[ColumnProfile]
    n_rows: int
    warnings: list[dict[str, Any]] = field(default_factory=list)
    _rows: list[list[str]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать для передачи в интерфейс (без полных данных)."""
        return {
            "dialect": self.dialect.to_dict(),
            "header": self.header,
            "preview_rows": self.preview_rows,
            "profiles": [p.to_dict() for p in self.profiles],
            "n_rows": self.n_rows,
            "warnings": self.warnings,
            "quality": self.quality_summary(),
            "summary_values": self.summary_values(),
            "usability": self.usability(),
        }

    def usability(self) -> dict[str, Any]:
        """Годится ли файл для анализа, и если нет — почему.

        Возвращает {ok, reason}. reason — код причины, фронт переводит его
        в понятное объяснение вместо показа пустой таблицы.
        """
        n_numeric = sum(
            1 for p in self.profiles if p.detected_type.value == "numeric"
        )
        if self.n_rows == 0 and not self.header:
            return {"ok": False, "reason": "file_empty"}
        if self.n_rows == 0:
            return {"ok": False, "reason": "no_data_rows"}
        if not self.header:
            return {"ok": False, "reason": "no_columns"}
        if n_numeric == 0:
            return {"ok": False, "reason": "no_numeric"}
        return {"ok": True, "reason": None}

    def summary_values(self) -> dict[str, dict[str, float]]:
        """Ключевые метрики по числовым колонкам для подстановки в параметры.

        Компактнее полной сводки: только те метрики, что осмысленно
        подставлять как параметр (среднее, SD, медиана, границы).
        Структура: колонка -> {метрика: значение}.
        """
        import numpy as np

        result: dict[str, dict[str, float]] = {}
        for name, values in self.numeric_columns_all().items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            metrics = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
            if arr.size >= 2:
                metrics["std"] = float(arr.std(ddof=1))
            result[name] = metrics
        return result

    def quality_summary(self) -> dict[str, Any]:
        """Сводка качества — слой 1 отчёта валидации."""
        total_cells = self.n_rows * len(self.header) if self.header else 0
        n_missing = sum(p.n_missing for p in self.profiles)
        n_mismatch = sum(p.n_type_mismatch for p in self.profiles)
        problem_columns = [p.name for p in self.profiles if p.issues]

        problem_ratio = (n_missing + n_mismatch) / total_cells if total_cells else 0.0
        if problem_ratio > 0.2 or n_mismatch > 0:
            verdict = "problems"
        elif problem_ratio > 0.05 or problem_columns:
            verdict = "warnings"
        else:
            verdict = "ok"

        return {
            "verdict": verdict,
            "n_rows": self.n_rows,
            "n_columns": len(self.header),
            "n_missing": n_missing,
            "n_type_mismatch": n_mismatch,
            "problem_ratio": problem_ratio,
            "problem_columns": problem_columns,
            "top_issues": self._top_issues(),
        }

    def _top_issues(self, limit: int = 5) -> list[dict[str, Any]]:
        """Самые важные проблемы по колонкам.

        Каждое сообщение — структура {code, message, params}; добавляем
        имя колонки, чтобы фронт показал «колонка: перевод».
        """
        issues = []
        for profile in self.profiles:
            for issue in profile.issues:
                issues.append({"column": profile.name, **issue})
        return issues[:limit]

    def rows_slice(self, start: int, count: int) -> list[list[str]]:
        """Вернуть срез строк [start, start+count) для прогрессивного показа.

        Строки нормализуются до ширины заголовка, чтобы предпросмотр не
        разъезжался на неровных строках.
        """
        width = len(self.header)
        result = []
        for row in self._rows[start : start + count]:
            trimmed = row[:width]
            if len(trimmed) < width:
                trimmed = trimmed + [""] * (width - len(trimmed))
            result.append(trimmed)
        return result

    def column_values(self, name: str) -> list[str]:
        """Сырые значения колонки по имени."""
        if name not in self.header:
            raise ValueError(f"Нет колонки: {name}")
        index = self.header.index(name)
        return [row[index] if index < len(row) else "" for row in self._rows]

    def numeric_column(self, name: str) -> list[float]:
        """Числовые значения колонки без пропусков.

        Значения, не разобравшиеся как числа, пропускаются — они уже
        отражены в профиле как несоответствия типу.
        """
        decimal = self.dialect.decimal
        values = []
        for raw in self.column_values(name):
            if is_missing(raw):
                continue
            number = parse_number(raw, decimal)
            if number is not None:
                values.append(number)
        return values

    def paired_columns(
        self, before: str, after: str
    ) -> tuple[list[float], list[float]]:
        """Две колонки как выровненные пары для парных тестов.

        Пара — это одна строка. Строка исключается целиком, если хотя бы
        одно из двух значений пропущено или не число: иначе пары разъедутся
        и тест станет бессмысленным. Возвращает два списка равной длины.
        """
        decimal = self.dialect.decimal
        raw_before = self.column_values(before)
        raw_after = self.column_values(after)
        aligned_before: list[float] = []
        aligned_after: list[float] = []
        for rb, ra in zip(raw_before, raw_after):
            if is_missing(rb) or is_missing(ra):
                continue
            nb = parse_number(rb, decimal)
            na = parse_number(ra, decimal)
            if nb is None or na is None:
                continue
            aligned_before.append(nb)
            aligned_after.append(na)
        return aligned_before, aligned_after

    def numeric_columns_all(self) -> dict[str, list[float]]:
        """Все числовые колонки как словарь имя -> значения без пропусков.

        Для сводной статистики. Берёт только колонки типа numeric —
        ID, категориальные и даты исключены на этапе типизации.
        """
        from core.validation.types import ColumnType as _CT

        return {
            p.name: self.numeric_column(p.name)
            for p in self.profiles
            if p.detected_type == _CT.NUMERIC
        }

    def groups_by(self, value_column: str, group_column: str) -> dict[str, list[float]]:
        """Разбить числовую колонку на группы по категориальной.

        Нужно для ANOVA и сравнения групп: данные в «длинном» формате,
        где одна колонка — значение, другая — метка группы.
        """
        decimal = self.dialect.decimal
        values = self.column_values(value_column)
        labels = self.column_values(group_column)
        groups: dict[str, list[float]] = {}
        for label, raw in zip(labels, values):
            if is_missing(label) or is_missing(raw):
                continue
            number = parse_number(raw, decimal)
            if number is None:
                continue
            groups.setdefault(label.strip(), []).append(number)
        return groups


def parse_file(
    raw: bytes,
    encoding: str | None = None,
    delimiter: str | None = None,
    decimal: str | None = None,
    has_header: bool | None = None,
    type_overrides: dict[str, str] | None = None,
) -> ParsedTable:
    """Разобрать файл: детекция параметров, парсинг, профилирование.

    Любой параметр можно переопределить вручную — автодетекция
    применяется только к тем, что не заданы явно. ``type_overrides``
    задаёт принудительные типы колонок по имени (строкой из ColumnType).
    """
    dialect = detect_dialect(raw)
    if encoding:
        dialect.encoding = encoding
    if delimiter:
        dialect.delimiter = delimiter
    if decimal:
        dialect.decimal = decimal
    if has_header is not None:
        dialect.has_header = has_header

    warnings: list[dict[str, Any]] = list(dialect.notes)
    try:
        text = raw.decode(dialect.encoding, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
        dialect.encoding = "utf-8"
        warnings.append(msg("unknown_encoding_utf8",
                            "Неизвестная кодировка, использована UTF-8"))

    rows = [
        row
        for row in csv.reader(io.StringIO(text), delimiter=dialect.delimiter)
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        return ParsedTable(
            dialect=dialect,
            header=[],
            preview_rows=[],
            profiles=[],
            n_rows=0,
            warnings=warnings + [msg("file_empty",
                                      "Файл пуст или не содержит данных")],
        )

    if dialect.has_header:
        header = [c.strip() or f"col{i + 1}" for i, c in enumerate(rows[0])]
        body = rows[1:]
    else:
        header = [f"col{i + 1}" for i in range(len(rows[0]))]
        body = rows

    dialect.n_columns = len(header)
    if len(body) > BROWSER_ROW_LIMIT:
        warnings.append(
            msg("too_many_rows",
                f"Строк {len(body):,} — для браузера это много. "
                f"Профилирование сделано по первым {BROWSER_ROW_LIMIT:,}.",
                n_rows=len(body), limit=BROWSER_ROW_LIMIT)
        )
        body = body[:BROWSER_ROW_LIMIT]

    overrides_enum = _parse_type_overrides(type_overrides)
    profiles = profile_table(
        header, body, decimal=dialect.decimal, type_overrides=overrides_enum
    )
    return ParsedTable(
        dialect=dialect,
        header=header,
        preview_rows=body[:PREVIEW_ROWS],
        profiles=profiles,
        n_rows=len(body),
        warnings=warnings,
        _rows=body,
    )


def _parse_type_overrides(
    overrides: dict[str, str] | None,
) -> dict[str, ColumnType]:
    """Преобразовать строковые типы из интерфейса в ColumnType.

    Нераспознанные значения игнорируются — лучше вернуться к
    автоопределению, чем упасть на опечатке.
    """
    if not overrides:
        return {}
    result: dict[str, ColumnType] = {}
    for name, type_str in overrides.items():
        try:
            result[name] = ColumnType(type_str)
        except ValueError:
            continue
    return result


def suggest_methods(
    profiles: list[ColumnProfile], lang: str = "ru"
) -> list[dict[str, Any]]:
    """Каталог методов, применимых к данному составу колонок.

    Делегирует в catalog.available_methods — полный список методов с
    группами, параметрами и требованиями к колонкам для витрины.
    """
    from core.validation.catalog import available_methods

    return available_methods(profiles, lang)
