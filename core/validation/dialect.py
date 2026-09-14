"""Детекция параметров CSV-файла перед парсингом.

Определяет кодировку, разделитель, десятичный знак и наличие заголовка.
Каждое решение сопровождается уверенностью — пользователь видит, что
распозналось, и может переопределить вручную.

Кодировка определяется через charset_normalizer: на кириллице в CP1251
он заметно надёжнее chardet (тот даёт уверенность ~0.2 на том же файле).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from charset_normalizer import from_bytes

from core.validation.messages import msg

# Сколько байт читать для детекции — хватает и для больших файлов.
_SNIFF_BYTES = 64 * 1024
# Кандидаты в разделители в порядке приоритета при равенстве.
_DELIMITERS = [",", ";", "\t", "|"]
# Минимум строк, чтобы судить о стабильности числа колонок.
_MIN_ROWS_FOR_STRUCTURE = 2


@dataclass
class DialectInfo:
    """Распознанные параметры файла с оценкой уверенности."""

    encoding: str
    encoding_confidence: float
    delimiter: str
    delimiter_confidence: float
    decimal: str
    has_header: bool
    n_columns: int
    notes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать для передачи в интерфейс."""
        return {
            "encoding": self.encoding,
            "encoding_confidence": self.encoding_confidence,
            "delimiter": self.delimiter,
            "delimiter_confidence": self.delimiter_confidence,
            "decimal": self.decimal,
            "has_header": self.has_header,
            "n_columns": self.n_columns,
            "notes": self.notes,
        }


def detect_encoding(raw: bytes) -> tuple[str, float]:
    """Определить кодировку байтов.

    Возвращает имя кодировки и уверенность (0..1). При неудаче
    откатывается на UTF-8 с нулевой уверенностью.
    """
    if not raw:
        return "utf-8", 0.0

    match = from_bytes(raw[:_SNIFF_BYTES]).best()
    if match is None:
        return "utf-8", 0.0

    # charset_normalizer отдаёт chaos (0 = идеально) — переводим в уверенность.
    confidence = max(0.0, min(1.0, 1.0 - float(match.chaos)))
    return _canonical_encoding(match.encoding or "utf-8"), confidence


def _canonical_encoding(name: str) -> str:
    """Привести имя кодировки к общепринятому виду.

    charset_normalizer отдаёт имена вида ``utf_8``; пользователю и
    Python-кодекам привычнее ``utf-8``. ascii приводим к utf-8 как
    его подмножество.
    """
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "ascii": "utf-8",
        "windows-1251": "cp1251",
        "utf-8-sig": "utf-8-sig",
    }
    return aliases.get(normalized, normalized)


def detect_delimiter(text: str) -> tuple[str, float]:
    """Определить разделитель колонок.

    Считает, какой кандидат даёт наиболее стабильное число колонок
    по строкам. Стабильность и служит мерой уверенности.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()][:50]
    if not lines:
        return ",", 0.0

    best_delim, best_score, best_count = ",", 0.0, 1
    for delim in _DELIMITERS:
        counts = [ln.count(delim) for ln in lines]
        if not any(counts):
            continue
        # Стабильность: доля строк с самым частым числом вхождений.
        mode = max(set(counts), key=counts.count)
        if mode == 0:
            continue
        score = counts.count(mode) / len(counts)
        # При равном счёте предпочитаем разделитель с большим числом колонок.
        if score > best_score or (score == best_score and mode > best_count):
            best_delim, best_score, best_count = delim, score, mode

    return best_delim, best_score


def detect_decimal(text: str, delimiter: str) -> str:
    """Определить десятичный разделитель: точка или запятая.

    Запятая как десятичный знак возможна только если она не занята
    под разделитель колонок. Признак — числа вида ``5,2`` в ячейках.
    """
    if delimiter == ",":
        return "."  # запятая занята под колонки

    comma_decimal = 0
    dot_decimal = 0
    for line in text.splitlines()[:50]:
        for cell in line.split(delimiter):
            cell = cell.strip().strip('"')
            if _looks_numeric(cell, ","):
                comma_decimal += 1
            elif _looks_numeric(cell, "."):
                dot_decimal += 1

    return "," if comma_decimal > dot_decimal else "."


def _looks_numeric(cell: str, decimal: str) -> bool:
    """Похожа ли ячейка на число с указанным десятичным знаком."""
    if not cell:
        return False
    candidate = cell.replace(decimal, ".", 1) if decimal != "." else cell
    if decimal == "," and "," not in cell:
        return False
    if decimal == "." and "." not in cell:
        return False
    try:
        float(candidate)
    except ValueError:
        return False
    return True


def detect_header(text: str, delimiter: str) -> bool:
    """Есть ли строка заголовка.

    Эвристика: первая строка считается заголовком, если её ячейки
    нечисловые, а в следующих строках на тех же местах есть числа.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if len(lines) < _MIN_ROWS_FOR_STRUCTURE:
        return True  # одна строка — безопаснее считать заголовком

    first = [c.strip().strip('"') for c in lines[0].split(delimiter)]
    rest = [
        [c.strip().strip('"') for c in ln.split(delimiter)] for ln in lines[1:]
    ]
    if not first:
        return True

    first_numeric = sum(_is_number(c) for c in first)
    # Доля числовых ячеек в остальных строках на тех же позициях.
    body_numeric = 0
    body_total = 0
    for row in rest:
        for cell in row[: len(first)]:
            body_total += 1
            body_numeric += _is_number(cell)

    if body_total == 0:
        return True
    body_ratio = body_numeric / body_total
    first_ratio = first_numeric / len(first)
    # Заголовок: первая строка заметно менее числовая, чем тело.
    return first_ratio < body_ratio - 0.3 or first_numeric == 0


def _is_number(cell: str) -> bool:
    """Является ли ячейка числом (с точкой или запятой)."""
    if not cell:
        return False
    try:
        float(cell.replace(",", ".", 1))
    except ValueError:
        return False
    return True


def detect_dialect(raw: bytes) -> DialectInfo:
    """Полная детекция параметров файла.

    Главная точка входа: принимает сырые байты файла, возвращает все
    распознанные параметры с уверенностью и замечаниями.
    """
    notes: list[dict[str, Any]] = []
    encoding, enc_conf = detect_encoding(raw)
    if enc_conf < 0.5:
        notes.append(
            msg("low_encoding_confidence",
                "Кодировка определена с низкой уверенностью — проверь предпросмотр")
        )

    try:
        text = raw[:_SNIFF_BYTES].decode(encoding, errors="replace")
    except LookupError:
        encoding, text = "utf-8", raw[:_SNIFF_BYTES].decode("utf-8", errors="replace")
        notes.append(msg("unknown_encoding_utf8",
                          "Неизвестная кодировка, использована UTF-8"))

    if "\ufffd" in text:
        notes.append(msg("unreadable_chars",
                          "В тексте есть нечитаемые символы — вероятно, кодировка неверна"))

    delimiter, delim_conf = detect_delimiter(text)
    if delim_conf < 0.8:
        notes.append(msg("unstable_delimiter",
                          "Разделитель нестабилен по строкам — проверь предпросмотр"))

    decimal = detect_decimal(text, delimiter)
    has_header = detect_header(text, delimiter)
    n_columns = _count_columns(text, delimiter)

    return DialectInfo(
        encoding=encoding,
        encoding_confidence=enc_conf,
        delimiter=delimiter,
        delimiter_confidence=delim_conf,
        decimal=decimal,
        has_header=has_header,
        n_columns=n_columns,
        notes=notes,
    )


def _count_columns(text: str, delimiter: str) -> int:
    """Наиболее частое число колонок по первым строкам."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    counts = []
    for i, row in enumerate(reader):
        if i >= 50:
            break
        if row:
            counts.append(len(row))
    if not counts:
        return 0
    return max(set(counts), key=counts.count)
