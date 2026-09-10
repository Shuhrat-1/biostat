"""Каталог статистических методов — источник правды для витрины.

Описывает все методы ядра декларативно: к какой группе относятся, какие
колонки требуют, какие параметры принимают, при каком составе данных
применимы. Из этого фронтенд собирает форму выбора метода.

Разделение: сами вычисления живут в core/stats, а здесь — метаданные о
том, как метод показать и что ему нужно на вход.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.validation.types import ColumnProfile, ColumnType


@dataclass(frozen=True)
class Param:
    """Параметр метода, задаваемый пользователем в форме.

    ``from_summary`` — имя метрики сводки (mean, std, median...), которую
    можно подставить как значение параметра из любой числовой колонки.
    Пустая строка — только ручной ввод.
    """

    name: str
    label: str
    kind: str  # "number" | "select" | "float"
    default: Any
    options: list[str] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    from_summary: str = ""

    def to_dict(self, lang: str = "ru") -> dict[str, Any]:
        from core.validation.catalog_i18n import LABELS_MISC, tr

        return {
            "name": self.name,
            "label": tr(LABELS_MISC, self.label, lang),
            "kind": self.kind,
            "default": self.default,
            "options": self.options,
            "min": self.min,
            "max": self.max,
            "from_summary": self.from_summary,
        }


@dataclass(frozen=True)
class MethodSpec:
    """Полное описание метода для витрины."""

    method: str
    label: str
    group: str
    # Слоты колонок: имя слота -> какой тип колонки принимает.
    slots: dict[str, ColumnType]
    slot_labels: dict[str, str]
    params: list[Param] = field(default_factory=list)
    # Сколько групп нужно категориальной колонке: None — любое ≥2.
    n_groups: int | None = None
    description: str = ""

    def to_dict(
        self, profiles: list[ColumnProfile], lang: str = "ru"
    ) -> dict[str, Any]:
        """Сериализовать с подстановкой колонок и переводом строк."""
        from core.validation.catalog_i18n import (
            DESCRIPTIONS,
            GROUPS as GROUPS_TR,
            LABELS,
            LABELS_MISC,
            tr,
        )

        by_type = _columns_by_type(profiles)
        needs = {
            slot: by_type.get(col_type, []) for slot, col_type in self.slots.items()
        }
        slot_labels = {
            slot: tr(LABELS_MISC, label, lang)
            for slot, label in self.slot_labels.items()
        }
        return {
            "method": self.method,
            "label": tr(LABELS, self.label, lang),
            "group": tr(GROUPS_TR, self.group, lang),
            "needs": needs,
            "slot_labels": slot_labels,
            "params": [p.to_dict(lang) for p in self.params],
            "n_groups": self.n_groups,
            "description": tr(DESCRIPTIONS, self.description, lang),
        }


# Группы витрины в порядке показа.
GROUPS = [
    "Описательная статистика",
    "Сравнение с эталоном",
    "Сравнение двух групп",
    "Сравнение нескольких групп",
    "Парные измерения",
    "Связь переменных",
    "Проверка допущений",
]

_NUM = ColumnType.NUMERIC
_CAT = ColumnType.CATEGORICAL

_ALTERNATIVE = Param(
    name="alternative",
    label="Гипотеза",
    kind="select",
    default="two-sided",
    options=["two-sided", "less", "greater"],
)
_CI_LEVEL = Param(
    name="level",
    label="Уровень доверия",
    kind="float",
    default=0.95,
    min=0.5,
    max=0.999,
)

# Полный каталог методов.
CATALOG: list[MethodSpec] = [
    MethodSpec(
        method="describe_all",
        label="Сводная описательная статистика",
        group="Описательная статистика",
        slots={},
        slot_labels={},
        description="Все метрики по всем числовым колонкам в одной таблице.",
    ),
    MethodSpec(
        method="mean_ci",
        label="Доверительный интервал среднего",
        group="Описательная статистика",
        slots={"value": _NUM},
        slot_labels={"value": "Числовая колонка"},
        params=[_CI_LEVEL],
        description="Интервальная оценка среднего по t-распределению.",
    ),
    MethodSpec(
        method="normality_test",
        label="Проверка нормальности (Шапиро-Уилк)",
        group="Проверка допущений",
        slots={"value": _NUM},
        slot_labels={"value": "Числовая колонка"},
        description="Проверяет, согласуются ли данные с нормальным распределением.",
    ),
    MethodSpec(
        method="one_sample_t",
        label="Одновыборочный t-тест",
        group="Сравнение с эталоном",
        slots={"value": _NUM},
        slot_labels={"value": "Числовая колонка"},
        params=[
            Param(
                "popmean",
                "Эталонное среднее μ",
                "number",
                0.0,
                from_summary="mean",
            ),
            _ALTERNATIVE,
        ],
        description="Сравнивает среднее выборки с заданным значением.",
    ),
    MethodSpec(
        method="two_sample_t",
        label="Двухвыборочный t-тест",
        group="Сравнение двух групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[_ALTERNATIVE],
        n_groups=2,
        description="Сравнивает средние двух групп (авто-выбор Стьюдент/Уэлч).",
    ),
    MethodSpec(
        method="mann_whitney",
        label="Тест Манна-Уитни (непараметрический)",
        group="Сравнение двух групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[_ALTERNATIVE],
        n_groups=2,
        description="Сравнение двух групп без предположения о нормальности.",
    ),
    MethodSpec(
        method="one_way_anova",
        label="Однофакторный ANOVA",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        description="Сравнивает средние нескольких групп.",
    ),
    MethodSpec(
        method="tukey_hsd",
        label="Пост-хок Tukey HSD",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Все попарные сравнения групп после значимого ANOVA.",
    ),
    MethodSpec(
        method="kruskal_wallis",
        label="Тест Краскела-Уоллиса (непараметрический)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        description="Сравнение нескольких групп без предположения о нормальности.",
    ),
    MethodSpec(
        method="paired_t",
        label="Парный t-тест",
        group="Парные измерения",
        slots={"before": _NUM, "after": _NUM},
        slot_labels={"before": "Колонка «до»", "after": "Колонка «после»"},
        params=[_ALTERNATIVE],
        description="Сравнивает связанные измерения (до/после) на одних объектах.",
    ),
    MethodSpec(
        method="wilcoxon",
        label="Тест Уилкоксона (непараметрический)",
        group="Парные измерения",
        slots={"before": _NUM, "after": _NUM},
        slot_labels={"before": "Колонка «до»", "after": "Колонка «после»"},
        params=[_ALTERNATIVE],
        description="Парное сравнение без предположения о нормальности разностей.",
    ),
    MethodSpec(
        method="simple_linear_regression",
        label="Простая линейная регрессия",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (независимая)", "y": "Y (зависимая)"},
        params=[_CI_LEVEL],
        description="Модель y = b0 + b1·x с оценками и доверительными интервалами.",
    ),
]


def _columns_by_type(profiles: list[ColumnProfile]) -> dict[ColumnType, list[str]]:
    """Сгруппировать имена колонок по определённому типу."""
    result: dict[ColumnType, list[str]] = {}
    for profile in profiles:
        result.setdefault(profile.detected_type, []).append(profile.name)
    return result


def _slot_satisfiable(
    spec: MethodSpec, by_type: dict[ColumnType, list[str]]
) -> bool:
    """Хватает ли колонок нужных типов для метода.

    Для регрессии нужны две числовые колонки (x и y — разные слоты, но
    один тип), поэтому одного столбца мало. Сводная статистика без слотов
    работает по всем числовым — требует хотя бы одну.
    """
    have_numeric = len(by_type.get(_NUM, []))
    if spec.method == "describe_all":
        return have_numeric >= 1
    needed_numeric = sum(1 for t in spec.slots.values() if t == _NUM)
    if have_numeric < needed_numeric:
        return False
    if _CAT in spec.slots.values() and not by_type.get(_CAT):
        return False
    return True


def available_methods(
    profiles: list[ColumnProfile], lang: str = "ru"
) -> list[dict[str, Any]]:
    """Каталог методов, применимых к данному составу колонок.

    Метод показывается, только если есть колонки нужных типов. Порядок —
    по группам из GROUPS. Строки переводятся на указанный язык.
    """
    by_type = _columns_by_type(profiles)
    applicable = [
        spec for spec in CATALOG if _slot_satisfiable(spec, by_type)
    ]
    group_order = {name: i for i, name in enumerate(GROUPS)}
    applicable.sort(key=lambda s: group_order.get(s.group, 999))
    return [spec.to_dict(profiles, lang) for spec in applicable]
