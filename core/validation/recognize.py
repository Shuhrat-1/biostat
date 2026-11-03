"""Распознаватели для карты выбора метода (decision-tree-spec v2, раздел 5).

Распознаватель — одно правило «признак в данных → какую ветку дерева он
подсказывает». Здесь каркас: описание правила, контекст с ролями колонок
и функция :func:`recognize`, которую вызывает мост. Сами правила — в
``core/validation/recognizers.py``; реестр (id, сила, источник) — единый
источник правды для карты.

Роли — слоты формы метода (``value``, ``group``, ``block``, ``x`` …):
какую колонку пользователь назначил на какую роль. Без ролей работают
только правила уровня всего файла (имена колонок, состав типов).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable

import numpy as np

from core.validation.messages import msg
from core.validation.types import is_missing, parse_number

STRONG, HINT = "strong", "hint"
STRUCTURAL, COMPUTED = "structural", "computed"

# Слоты формы, которые означают отклик / основной фактор.
_RESPONSE_SLOTS = ("value", "y", "response")
_FACTOR_SLOTS = ("group", "factor_a")


@dataclass(frozen=True)
class Signal:
    """Сработавший распознаватель: что подсказывает и почему."""

    note: str  # русский текст-фолбэк
    params: dict[str, Any] = field(default_factory=dict)


Rule = Callable[["Context"], "Signal | None"]


@dataclass(frozen=True)
class Recognizer:
    """Строка реестра; ``fn`` = None — правило пока только описано."""

    id: str
    node: str
    suggests: str
    strength: str
    cls: str
    source: str
    since: str = "1"
    fn: Rule | None = None

    def meta(self) -> dict[str, Any]:
        """Метаданные для фронта (без функции)."""
        return {k: getattr(self, k) for k in
                ("id", "node", "suggests", "strength", "cls", "source", "since")}


REGISTRY: list[Recognizer] = []


def rule(id: str, node: str, suggests: str, strength: str, source: str,
         cls: str = STRUCTURAL) -> Callable[[Rule], Rule]:
    """Декоратор: зарегистрировать функцию как распознаватель."""

    def wrap(fn: Rule) -> Rule:
        REGISTRY.append(Recognizer(id, node, suggests, strength, cls, source, fn=fn))
        return fn

    return wrap


def declare(id: str, node: str, suggests: str, strength: str, source: str,
            cls: str = COMPUTED) -> None:
    """Описать правило без реализации (расчётные — этап 2 карты)."""
    REGISTRY.append(Recognizer(id, node, suggests, strength, cls, source))


class Context:
    """Таблица и роли колонок с кэшированными выборками значений."""

    def __init__(self, table: Any, roles: dict[str, str] | None = None) -> None:
        self.table = table
        names = set(table.header)
        self.roles = {k: v for k, v in (roles or {}).items() if v in names}
        # Общие для нескольких правил расчёты (подгонка модели): ключ → результат.
        self.cache: dict[str, Any] = {}

    def memo(self, key: str, compute: Callable[[], Any]) -> Any:
        """Посчитать один раз на вызов recognize (модель нужна многим правилам)."""
        if key not in self.cache:
            self.cache[key] = compute()
        return self.cache[key]

    def role(self, *slots: str) -> str | None:
        """Первая назначенная колонка из перечня слотов."""
        return next((self.roles[s] for s in slots if s in self.roles), None)

    @property
    def response(self) -> str | None:
        return self.role(*_RESPONSE_SLOTS)

    @property
    def factor(self) -> str | None:
        return self.role(*_FACTOR_SLOTS)

    def numbers(self, column: str) -> np.ndarray:
        """Числа колонки по строкам; пропуск или нечисло → NaN."""
        return self._numbers[column] if column in self._numbers else self._parse(column)

    def labels(self, column: str) -> list[str]:
        """Метки колонки по строкам; пропуск → "" (разбор один раз на вызов)."""
        return self.memo(f"labels:{column}", lambda: [
            "" if is_missing(v) else v.strip() for v in self.table.column_values(column)])

    def columns_of(self, kind: str) -> list[str]:
        """Имена колонок определённого типа (numeric, categorical …)."""
        return [p.name for p in self.table.profiles if p.detected_type.value == kind]

    @cached_property
    def _numbers(self) -> dict[str, np.ndarray]:
        return {}

    def _parse(self, column: str) -> np.ndarray:
        decimal = self.table.dialect.decimal
        out = []
        for raw in self.table.column_values(column):
            number = None if is_missing(raw) else parse_number(raw, decimal)
            out.append(np.nan if number is None else number)
        arr = np.asarray(out, dtype=float)
        self._numbers[column] = arr
        return arr


def _fill(note: str, params: dict[str, Any]) -> str:
    """Русский фолбэк с подставленными {param}; неизвестные остаются как есть."""
    return re.sub(r"\{(\w+)\}",
                  lambda m: str(params.get(m.group(1), m.group(0))), note)


def recognize(table: Any, roles: dict[str, str] | None = None,
              cls: str | None = None) -> dict[str, Any]:
    """Прогнать реализованные распознаватели (``cls`` — только этого класса).

    Возвращает реестр (метаданные всех правил, в т.ч. ещё не реализованных)
    и список сработавших: id, узел, ветка, сила, сообщение для перевода.
    Ошибка одного правила не роняет остальные — правило просто не срабатывает.
    Карта зовёт структурные сразу, расчётные — вторым вызовом (раздел 8).
    """
    from core.validation import recognizers  # noqa: F401  регистрирует правила

    ctx = Context(table, roles)
    fired = []
    for rec in REGISTRY:
        if rec.fn is None or (cls is not None and rec.cls != cls):
            continue
        try:
            signal = rec.fn(ctx)
        except Exception:  # noqa: BLE001 — сбой правила ≠ сбой карты
            signal = None
        if signal is not None:
            fired.append({**rec.meta(), "message": msg(
                f"rec_{rec.id.replace('.', '_')}", _fill(signal.note, signal.params),
                **signal.params)})
    return {"registry": [r.meta() for r in REGISTRY], "fired": fired}
