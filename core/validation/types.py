"""Определение типов колонок и сводка качества данных (Тир 1).

Ловит классические дефекты экспорта из Excel: числа, ставшие строками,
смешанные форматы дат в одной колонке, неоднозначные даты (03/04/2020 —
март или апрель?), разнородные обозначения пропусков.

Тир 1 работает по выборке и даёт скрининг, а не гарантию: редкий дефект
в сэмпл может не попасть. Полная проверка каждой ячейки — Тир 2, для
больших файлов она принадлежит локальному или облачному бэкенду.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.validation.messages import msg

# Общепринятые обозначения пропуска в биологических данных.
# Русские варианты важны: аудитория пишет «н/д», «нет данных», «б/д».
MISSING_TOKENS = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "-",
        "—",
        "?",
        ".",
        "#н/д",
        "#n/a",
        "н/д",
        "нд",
        "нет данных",
        "б/д",
        "отсутствует",
        "missing",
    }
)

# Форматы дат: шаблон -> человекочитаемое имя.
_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "ISO (ГГГГ-ММ-ДД)"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "слэш (ДД/ММ/ГГГГ или ММ/ДД/ГГГГ)"),
    (re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"), "точки (ДД.ММ.ГГГГ)"),
    (re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"), "дефисы (ДД-ММ-ГГГГ)"),
]

# Доля значений одного типа, при которой колонка считается этого типа.
_TYPE_THRESHOLD = 0.9
# Сколько примеров проблемных значений сохранять.
_MAX_EXAMPLES = 5
# Числовая колонка вероятно категориальна, если уникальных значений мало
# И их сильно меньше числа строк (закодированные группы: 0/1, 1/2/3).
_LIKELY_CATEGORICAL_MAX_UNIQUE = 10
_LIKELY_CATEGORICAL_MIN_ROWS_PER_UNIQUE = 10
# Доля уникальных, при которой колонка похожа на идентификатор.
_LIKELY_ID_UNIQUE_RATIO = 0.95
# Минимум строк, чтобы вообще судить об «айдишности».
_LIKELY_ID_MIN_ROWS = 10


class ColumnType(str, Enum):
    """Определённый тип колонки."""

    NUMERIC = "numeric"
    DATE = "date"
    CATEGORICAL = "categorical"
    ID = "id"
    EMPTY = "empty"


@dataclass
class ColumnProfile:
    """Профиль одной колонки: тип, качество, найденные проблемы."""

    name: str
    detected_type: ColumnType
    confidence: float
    n_total: int
    n_missing: int
    n_type_mismatch: int
    mismatch_examples: list[str] = field(default_factory=list)
    date_formats: dict[str, int] = field(default_factory=dict)
    n_unique: int = 0
    n_duplicates: int = 0
    likely_categorical: bool = False
    likely_id: bool = False
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def missing_ratio(self) -> float:
        """Доля пропусков в колонке."""
        return self.n_missing / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать для интерфейса."""
        return {
            "name": self.name,
            "detected_type": self.detected_type.value,
            "confidence": self.confidence,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "missing_ratio": self.missing_ratio,
            "n_type_mismatch": self.n_type_mismatch,
            "mismatch_examples": self.mismatch_examples,
            "date_formats": self.date_formats,
            "n_unique": self.n_unique,
            "n_duplicates": self.n_duplicates,
            "likely_categorical": self.likely_categorical,
            "likely_id": self.likely_id,
            "issues": self.issues,
        }


def is_missing(value: str) -> bool:
    """Считается ли значение пропуском."""
    return value.strip().lower() in MISSING_TOKENS


def parse_number(value: str, decimal: str = ".") -> float | None:
    """Разобрать число с учётом десятичного знака и разделителей тысяч.

    Понимает пробелы и неразрывные пробелы как разделители тысяч —
    типичный результат экспорта из Excel («1 200,5»).
    """
    text = value.strip().strip('"').replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    if decimal == ",":
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def match_date_format(value: str) -> str | None:
    """Определить формат даты, если значение на неё похоже."""
    text = value.strip().strip('"')
    for pattern, label in _DATE_PATTERNS:
        if pattern.match(text):
            return label
    return None


def profile_column(
    name: str,
    values: list[str],
    decimal: str = ".",
    force_type: ColumnType | None = None,
) -> ColumnProfile:
    """Построить профиль колонки по её значениям.

    Определяет доминирующий тип, считает пропуски и значения, не
    подходящие под тип, собирает примеры проблем.

    ``force_type`` переопределяет автоопределение: пользователь может
    заставить трактовать колонку как числовую или категориальную —
    например, ``pclass`` из чисел 1/2/3, которая по смыслу категория.
    """
    total = len(values)
    present = [v for v in values if not is_missing(v)]
    n_missing = total - len(present)

    if not present:
        return ColumnProfile(
            name=name,
            detected_type=ColumnType.EMPTY,
            confidence=1.0,
            n_total=total,
            n_missing=n_missing,
            n_type_mismatch=0,
            issues=[msg("empty_column", "Колонка пуста")] if total else [],
        )

    numeric_ok = [v for v in present if parse_number(v, decimal) is not None]
    date_formats = Counter(
        fmt for v in present if (fmt := match_date_format(v)) is not None
    )
    n_dates = sum(date_formats.values())

    numeric_ratio = len(numeric_ok) / len(present)
    date_ratio = n_dates / len(present)
    n_unique = len(set(present))
    # Дубликаты среди присутствующих значений (повторы сверх первого вхождения).
    n_duplicates = len(present) - n_unique

    profile = ColumnProfile(
        name=name,
        detected_type=ColumnType.CATEGORICAL,
        confidence=1.0,
        n_total=total,
        n_missing=n_missing,
        n_type_mismatch=0,
        n_unique=n_unique,
        n_duplicates=n_duplicates,
        date_formats=dict(date_formats),
    )

    # Ручное переопределение имеет приоритет над автоопределением.
    if force_type is not None:
        _apply_forced_type(profile, present, force_type, numeric_ratio, decimal)
        _add_missing_issue(profile)
        return profile

    if numeric_ratio >= _TYPE_THRESHOLD:
        _fill_numeric(profile, present, numeric_ratio, decimal)
        # Мало уникальных при большом n — вероятно закодированная категория.
        # Порог по строкам-на-уникальное отсекает просто малые выборки.
        enough_rows = len(present) >= n_unique * _LIKELY_CATEGORICAL_MIN_ROWS_PER_UNIQUE
        if 1 < n_unique <= _LIKELY_CATEGORICAL_MAX_UNIQUE and enough_rows:
            profile.likely_categorical = True
            profile.issues.append(
                msg(
                    "likely_categorical",
                    f"Уникальных значений всего {n_unique} на {len(present)} строк — "
                    f"возможно, это закодированная категория. Смени тип, если нужны "
                    f"группы.",
                    n_unique=n_unique,
                    n_present=len(present),
                )
            )
    elif date_ratio >= _TYPE_THRESHOLD:
        _fill_date(profile, present, date_ratio, date_formats)
    else:
        _fill_categorical(profile, present, numeric_ratio)

    # Подсказка про ID: почти все значения уникальны при достаточном n.
    # Дробные числа исключаем — непрерывные величины (цены, измерения)
    # тоже почти уникальны, но идентификаторами не являются.
    if (
        len(present) >= _LIKELY_ID_MIN_ROWS
        and n_unique / len(present) >= _LIKELY_ID_UNIQUE_RATIO
        and not profile.likely_categorical
        and not _has_fractional(present, decimal)
    ):
        profile.likely_id = True
        profile.issues.append(
            msg(
                "likely_id",
                f"Почти все значения уникальны ({n_unique} из {len(present)}) — "
                f"возможно, это идентификатор. Смени тип на ID, чтобы исключить "
                f"из расчётов.",
                n_unique=n_unique,
                n_present=len(present),
            )
        )

    _add_missing_issue(profile)
    return profile


def _has_fractional(values: list[str], decimal: str) -> bool:
    """Есть ли среди значений дробные числа.

    Целые и строковые ID отличаем от непрерывных величин: наличие дробной
    части — сигнал, что это измерение (цена, вес), а не идентификатор.
    """
    for value in values:
        number = parse_number(value, decimal)
        if number is not None and number != int(number):
            return True
    return False


def _apply_forced_type(
    profile: ColumnProfile,
    present: list[str],
    forced: ColumnType,
    numeric_ratio: float,
    decimal: str,
) -> None:
    """Применить принудительный тип колонки, заданный пользователем."""
    if forced == ColumnType.NUMERIC:
        _fill_numeric(profile, present, numeric_ratio, decimal)
    elif forced == ColumnType.DATE:
        formats = Counter(
            fmt for v in present if (fmt := match_date_format(v)) is not None
        )
        _fill_date(profile, present, len(formats) / len(present), formats)
    elif forced == ColumnType.ID:
        _fill_id(profile, present)
    else:
        profile.detected_type = ColumnType.CATEGORICAL
        profile.confidence = 1.0


def _fill_id(profile: ColumnProfile, present: list[str]) -> None:
    """Заполнить профиль ID-колонки и проверить уникальность.

    Повторяющиеся ID — типичный дефект грязных данных (склейка таблиц,
    ошибки экспорта). Дубликаты выносятся в предупреждение с примерами.
    """
    profile.detected_type = ColumnType.ID
    profile.confidence = 1.0
    counts = Counter(present)
    dups = {value: n for value, n in counts.items() if n > 1}
    if dups:
        examples = ", ".join(
            f"{value!r}×{n}" for value, n in list(dups.items())[:_MAX_EXAMPLES]
        )
        profile.issues.append(
            msg(
                "id_not_unique",
                f"ID не уникален: {len(dups)} повторяющихся значений "
                f"(примеры: {examples}). Проверь целостность данных.",
                n_duplicates=len(dups),
                examples=examples,
            )
        )


def _fill_numeric(
    profile: ColumnProfile, present: list[str], ratio: float, decimal: str
) -> None:
    """Заполнить профиль числовой колонки и отметить нечисловые ячейки."""
    bad = [v for v in present if parse_number(v, decimal) is None]
    profile.detected_type = ColumnType.NUMERIC
    profile.confidence = ratio
    profile.n_type_mismatch = len(bad)
    profile.mismatch_examples = bad[:_MAX_EXAMPLES]
    if bad:
        profile.issues.append(
            msg(
                "numeric_intruders",
                f"В числовой колонке {len(bad)} нечисловых значений "
                f"(примеры: {', '.join(repr(v) for v in bad[:3])})",
                n_bad=len(bad),
                examples=", ".join(repr(v) for v in bad[:3]),
            )
        )


def _fill_date(
    profile: ColumnProfile,
    present: list[str],
    ratio: float,
    formats: Counter[str],
) -> None:
    """Заполнить профиль колонки дат и предупредить о смешении форматов."""
    bad = [v for v in present if match_date_format(v) is None]
    profile.detected_type = ColumnType.DATE
    profile.confidence = ratio
    profile.n_type_mismatch = len(bad)
    profile.mismatch_examples = bad[:_MAX_EXAMPLES]

    if len(formats) > 1:
        listed = ", ".join(f"{k} ({v})" for k, v in formats.items())
        profile.issues.append(
            msg(
                "mixed_date_formats",
                f"Смешаны форматы дат в одной колонке: {listed}. "
                f"Типичный дефект экспорта из Excel.",
                formats=listed,
            )
        )
    if "слэш (ДД/ММ/ГГГГ или ММ/ДД/ГГГГ)" in formats:
        profile.issues.append(
            msg(
                "ambiguous_slash_date",
                "Формат со слэшем неоднозначен: 03/04/2020 — это 3 апреля "
                "или 4 марта? Проверь порядок дня и месяца.",
            )
        )


def _fill_categorical(
    profile: ColumnProfile, present: list[str], numeric_ratio: float
) -> None:
    """Заполнить профиль категориальной колонки."""
    profile.detected_type = ColumnType.CATEGORICAL
    profile.confidence = 1.0 - numeric_ratio
    # Часть значений числовая — возможно, колонка задумана числовой.
    if 0.3 < numeric_ratio < _TYPE_THRESHOLD:
        profile.issues.append(
            msg(
                "mixed_types",
                f"Смешанные типы: {numeric_ratio:.0%} значений числовые. "
                f"Возможно, колонка должна быть числовой.",
                numeric_pct=round(numeric_ratio * 100),
            )
        )


def _add_missing_issue(profile: ColumnProfile) -> None:
    """Добавить замечание о высокой доле пропусков."""
    if profile.missing_ratio > 0.5:
        profile.issues.append(
            msg(
                "missing_high",
                f"Более половины значений пропущено ({profile.missing_ratio:.0%})",
                missing_pct=round(profile.missing_ratio * 100),
            )
        )
    elif profile.missing_ratio > 0.1:
        profile.issues.append(
            msg(
                "missing_notable",
                f"Заметная доля пропусков: {profile.missing_ratio:.0%}",
                missing_pct=round(profile.missing_ratio * 100),
            )
        )


def profile_table(
    header: list[str],
    rows: list[list[str]],
    decimal: str = ".",
    type_overrides: dict[str, ColumnType] | None = None,
) -> list[ColumnProfile]:
    """Построить профили всех колонок таблицы.

    Строки короче заголовка дополняются пропусками, длиннее — обрезаются,
    чтобы неровные строки не ломали профилирование.

    ``type_overrides`` задаёт принудительные типы по имени колонки —
    пользователь может переопределить автоопределение.
    """
    overrides = type_overrides or {}
    n_cols = len(header)
    columns: list[list[str]] = [[] for _ in range(n_cols)]
    for row in rows:
        for i in range(n_cols):
            columns[i].append(row[i] if i < len(row) else "")
    return [
        profile_column(name, values, decimal, force_type=overrides.get(name))
        for name, values in zip(header, columns)
    ]
