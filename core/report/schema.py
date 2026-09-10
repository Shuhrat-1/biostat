"""Единая модель результата статистического расчёта.

Все методы ядра возвращают ``StatResult``. Из него рендерятся все
представления (экран, JSON, отчёт), к нему привязывается качество данных
и предупреждения. Сэмпл-валидация и полная валидация пишут в одну схему.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Уровень серьёзности предупреждения."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Warning_:
    """Предупреждение, привязанное к результату (напр. нарушение допущения).

    ``code`` — стабильный идентификатор для перевода на фронте.
    ``message`` — русский текст-фолбэк (если фронт не знает код).
    ``params`` — значения для подстановки в переведённый шаблон.
    """

    code: str
    message: str
    severity: Severity = Severity.WARNING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Estimate:
    """Точечная оценка с опциональным доверительным интервалом."""

    name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float | None = None
    std_error: float | None = None


@dataclass
class StatResult:
    """Результат любого статистического метода ядра.

    Единая точка для рендеринга, экспорта и привязки качества данных.
    """

    method: str
    estimates: list[Estimate] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)
    p_value: float | None = None
    warnings: list[Warning_] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    n: int | None = None
    seed: int | None = None
    data_quality: dict[str, Any] | None = None

    def add_warning(
        self,
        code: str,
        message: str,
        severity: Severity = Severity.WARNING,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Добавить предупреждение к результату.

        ``code`` — для перевода на фронте, ``message`` — русский фолбэк,
        ``params`` — значения для подстановки в переведённый шаблон.
        """
        self.warnings.append(
            Warning_(
                code=code,
                message=message,
                severity=severity,
                params=params or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в словарь (Enum → строка)."""
        return _enum_safe(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        """Сериализовать в JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _enum_safe(obj: Any) -> Any:
    """Рекурсивно заменить Enum на их значения для JSON-сериализации."""
    if isinstance(obj, dict):
        return {k: _enum_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enum_safe(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj
