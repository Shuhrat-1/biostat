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
    kind: str  # "number" | "select" | "float" | "text"
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
    # Слоты колонок: имя слота -> тип колонки или кортеж допустимых типов
    # (блоки и делянки часто закодированы числами 1, 2, 3…).
    slots: dict[str, ColumnType | tuple[ColumnType, ...]]
    slot_labels: dict[str, str]
    params: list[Param] = field(default_factory=list)
    # Сколько групп нужно категориальной колонке: None — любое ≥2.
    n_groups: int | None = None
    description: str = ""
    # Необязательные слоты (третий фактор, блок): форма не требует их заполнения.
    optional_slots: tuple[str, ...] = ()

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
            slot: _eligible(by_type, col_type) for slot, col_type in self.slots.items()
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
            "optional_slots": list(self.optional_slots),
        }


# Трансформация отклика для факторных структур: по умолчанию без неё.
_TRANSFORM_NONE = Param(
    "transform_kind",
    "Трансформация",
    "select",
    "none",
    options=["none", "log10", "ln", "sqrt", "logit", "logit_pct"],
)

# Контрасты: «A + C vs B; A vs C» или «F1=0.5, F2=-0.5, …», через «;».
_CONTRASTS = Param("contrasts", "Контрасты (A + C vs B; A vs C)", "text", "")

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
# Структурный фактор (блок, делянка): категориальный или числовой код.
_FACTOR = (ColumnType.CATEGORICAL, ColumnType.NUMERIC)

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
# Регрессия принимает параметр под именем ci_level, а не level.
_REG_CI_LEVEL = Param(
    name="ci_level",
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
        method="treatment_means",
        label="Средние групп с SED и LSD",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Средние с ДИ по общей дисперсии ANOVA, SED, LSD и ДИ разностей.",
    ),
    MethodSpec(
        method="anova_permutation_test",
        label="Перестановочный F-тест (ANOVA)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[
            Param("n_permutations", "Число перестановок", "number", 999, min=99, max=9999),
        ],
        description="F-тест без предположения о нормальности: p по случайным перестановкам.",
    ),
    MethodSpec(
        method="bartlett_test",
        label="Тест Бартлетта (равенство дисперсий)",
        group="Проверка допущений",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        description="Проверяет, равны ли дисперсии групп перед ANOVA.",
    ),
    MethodSpec(
        method="anova_residuals",
        label="Остатки ANOVA и выбросы",
        group="Проверка допущений",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        description="Стандартизованные остатки, графики допущений и кандидаты в выбросы.",
    ),
    MethodSpec(
        method="transformed_means",
        label="ANOVA с трансформацией отклика",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT},
        slot_labels={"value": "Числовая колонка", "group": "Колонка групп"},
        params=[
            Param(
                "transform_kind",
                "Трансформация",
                "select",
                "log10",
                options=["log10", "ln", "sqrt", "logit", "logit_pct"],
            ),
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="ANOVA на шкале log, √ или logit; средние и ДИ переведены обратно.",
    ),
    MethodSpec(
        method="rcbd_anova",
        label="Блочный ANOVA (RCBD)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Каждый вариант по одному разу в каждом блоке; блоки отделяются от фона.",
    ),
    MethodSpec(
        method="rcbd_subsampling_anova",
        label="Блочный ANOVA с повторными измерениями делянки",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _CAT, "block": _FACTOR, "unit": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков",
            "unit": "Колонка делянок",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Несколько измерений на делянку (ряды, растения): варианты против разброса делянок.",
    ),
    MethodSpec(
        method="factorial_anova",
        label="Факторный ANOVA (A×B, A×B×C)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR,
               "factor_c": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "factor_a": "Фактор A",
            "factor_b": "Фактор B",
            "factor_c": "Фактор C (необязательно)",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
            _CONTRASTS,
        ],
        description="Главные эффекты и взаимодействия; средние по членам, отобранным по маргинальности.",
        optional_slots=("factor_c", "block"),
    ),
    MethodSpec(
        method="nested_anova",
        label="Вложенный ANOVA (группа / член, контроль)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "member": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка групп",
            "member": "Колонка членов группы",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Различия между группами и между членами внутри групп (контроль против остальных).",
        optional_slots=("block",),
    ),
    MethodSpec(
        method="treatment_contrasts",
        label="Контрасты вариантов",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            _CONTRASTS,
            Param("adjustment", "Поправка на множественность", "select", "none",
                  options=["none", "bonferroni", "fdr"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Заранее заданные сравнения групп вариантов (A + C vs B) с разбиением SS вариантов.",
        optional_slots=("block",),
    ),
    MethodSpec(
        method="polynomial_contrasts",
        label="Полиномиальный тренд по уровням",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка уровней (дозы, числа)",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("degree", "Степень полинома", "number", 2, min=1, max=4),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Линейный и квадратичный тренд по количественным уровням и проверка остатка.",
        optional_slots=("block",),
    ),
    MethodSpec(
        method="mean_comparisons",
        label="Сравнения средних с поправкой (Тьюки, Даннет, FDR)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("procedure", "Процедура", "select", "tukey",
                  options=["tukey", "dunnett", "bonferroni", "fdr", "lsd", "alsd"]),
            Param("control", "Контрольный уровень (для сравнений с контролем)", "text", ""),
            Param("alternative", "Гипотеза", "select", "two-sided",
                  options=["two-sided", "greater", "less"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Все пары или каждый вариант против контроля с контролем ошибки I рода.",
        optional_slots=("block",),
    ),
    MethodSpec(
        method="latin_square_anova",
        label="Латинский квадрат",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "row": _FACTOR, "column": _FACTOR,
               "factor_b": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "factor_a": "Колонка вариантов (фактор A)",
            "row": "Колонка рядов",
            "column": "Колонка колонок (столбцов)",
            "factor_b": "Фактор B (необязательно)",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Каждый вариант раз в каждом ряду и колонке: два перекрёстных блока.",
        optional_slots=("factor_b",),
    ),
    MethodSpec(
        method="split_plot_anova",
        label="Расщеплённые делянки (split-plot)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR, "block": _FACTOR,
               "whole_plot": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "factor_a": "Фактор целых делянок (A)",
            "factor_b": "Фактор субделянок (B)",
            "block": "Колонка блоков (необязательно)",
            "whole_plot": "Колонка целых делянок (необязательно)",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="A на крупных делянках, B на их частях: у A и B разные ошибки.",
        optional_slots=("block", "whole_plot"),
    ),
    MethodSpec(
        method="bibd_anova",
        label="Неполные блоки (BIBD)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Блок меньше числа вариантов: внутриблочные средние, эффективность EF.",
    ),
    MethodSpec(
        method="replication_power",
        label="Повторность и мощность (по пилоту)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("delta", "Разность, которую нужно обнаружить (δ)", "float", 0.0,
                  min=0.0, max=1e9),
            Param("target_power", "Целевая мощность", "float", 0.8, min=0.5, max=0.99),
            Param("n_max", "Максимум повторностей", "number", 30, min=3, max=200),
            Param("s2_override", "Фоновая дисперсия s² (0 — из данных)", "float", 0.0,
                  min=0.0, max=1e12),
            Param("alternative", "Гипотеза", "select", "two-sided",
                  options=["two-sided", "greater", "less"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Сколько повторностей нужно, чтобы обнаружить разность δ с нужной мощностью.",
        optional_slots=("block",),
    ),
    MethodSpec(
        method="equivalence_test",
        label="Тест эквивалентности (TOST)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "factor_b": _FACTOR, "factor_c": _FACTOR,
               "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов",
            "factor_b": "Фактор B (необязательно)",
            "factor_c": "Фактор C (необязательно)",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("control", "Стандарт (уровень или комбинация)", "text", ""),
            Param("margin", "Граница эквивалентности c", "float", 0.0, min=0.0, max=1e9),
            Param("margin_kind", "Граница в единицах", "select", "absolute",
                  options=["absolute", "percent"]),
            Param("adjustment", "Поправка на множественность", "select", "none",
                  options=["none", "bonferroni"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Доказать, что варианты не отличаются от стандарта больше чем на c.",
        optional_slots=("factor_b", "factor_c", "block"),
    ),
    MethodSpec(
        method="unbalanced_anova",
        label="Неортогональный ANOVA (неравная повторность)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR, "factor_c": _FACTOR,
               "block": _FACTOR, "extraneous": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "factor_a": "Фактор A",
            "factor_b": "Фактор B (необязательно)",
            "factor_c": "Фактор C (необязательно)",
            "block": "Колонка блоков (необязательно)",
            "extraneous": "Посторонний фактор (необязательно)",
        },
        params=[
            Param("model", "Модель", "select", "full", options=["full", "additive"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
            Param("compare_healy_westmacott", "Сравнить с Хили–Уэстмакоттом", "select", "no",
                  options=["no", "yes"]),
        ],
        description="Факторы с неравной повторностью или пустыми ячейками: порядок подгонки, отбор модели.",
        optional_slots=("factor_b", "factor_c", "block", "extraneous"),
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
        params=[_REG_CI_LEVEL],
        description="Модель y = b0 + b1·x с оценками и доверительными интервалами.",
    ),
    MethodSpec(
        method="regression_details",
        label="Регрессия: ANOVA, тесты, предсказания",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("through_origin", "Через начало координат (α = 0)", "select", "no",
                  options=["no", "yes"]),
            Param("slope_null", "Проверяемое значение наклона", "float", 0.0,
                  min=-1e9, max=1e9),
            Param("intercept_null", "Проверяемое значение α", "float", 0.0, min=-1e9, max=1e9),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("scaling", "Параметризация", "select", "none",
                  options=["none", "centred", "standardized"]),
            Param("lack_of_fit", "Проверить неадекватность (повторы x)", "select", "no",
                  options=["no", "yes"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Таблица ANOVA, t-тесты параметров, ДИ предсказаний, R², неадекватность.",
    ),
    MethodSpec(
        method="inverse_prediction",
        label="Калибровка (обратное предсказание)",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (целевая величина)", "y": "Y (замена, измерение)"},
        params=[
            Param("y_new", "Новые значения y (через ;)", "text", ""),
            Param("replicates", "Повторов на новый образец r", "number", 1, min=1, max=100),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Оценить x по новому измерению y с пределами Филлера.",
    ),
    MethodSpec(
        method="regression_diagnostics",
        label="Регрессия: выбросы, рычаги, влияние",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("through_origin", "Через начало координат (α = 0)", "select", "no",
                  options=["no", "yes"]),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
            Param("x_transform", "Трансформация x", "select", "none",
                  options=["none", "log10", "ln", "sqrt", "logit", "logit_pct"]),
            Param("x_offset", "Сдвиг c для x", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Удалённые остатки, рычаги и статистика Кука по наблюдениям с порогами.",
    ),
    MethodSpec(
        method="regression_cross_validation",
        label="Регрессия: кросс-валидация",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("scheme", "Схема", "select", "split", options=["split", "loo", "kfold"]),
            Param("train_fraction", "Доля обучающей выборки", "float", 0.5, min=0.1, max=0.9),
            Param("k_folds", "Число блоков k", "number", 5, min=2, max=100),
            Param("seed", "Зерно случайного разбиения", "number", 1, min=0, max=999999),
            Param("through_origin", "Через начало координат (α = 0)", "select", "no",
                  options=["no", "yes"]),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
            Param("x_transform", "Трансформация x", "select", "none",
                  options=["none", "log10", "ln", "sqrt", "logit", "logit_pct"]),
            Param("x_offset", "Сдвиг c для x", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Предсказательная способность: смещение PB, MAD, RMSE на проверочных данных.",
    ),
    MethodSpec(
        method="x_transform_regression",
        label="Кривая: регрессия по преобразованному x",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("x_transform_kind", "Преобразование x", "select", "log10",
                  options=["log10", "ln", "sqrt", "square", "exp", "reciprocal"]),
            Param("x_offset", "Сдвиг c для x", "float", 0.0, min=0.0, max=1000.0),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Прямая по log x, √x, x², exp x или 1/x: кривая на исходной шкале и сравнение преобразований.",
    ),
    MethodSpec(
        method="polynomial_regression",
        label="Кривая: полиномиальная регрессия",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("degree", "Порядок полинома q", "number", 2, min=1, max=10),
            Param("basis", "Базис степеней", "select", "raw",
                  options=["raw", "centred", "orthogonal"]),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Последовательная ANOVA по степеням x, тест старшего члена, VIF, неадекватность.",
    ),
    MethodSpec(
        method="trigonometric_regression",
        label="Кривая: периодическая (тригонометрическая) регрессия",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (время измерения)", "y": "Y (отклик)"},
        params=[
            Param("period", "Период цикла ω", "float", 12.0, min=1e-6, max=1e9),
            Param("harmonics", "Число гармоник", "number", 1, min=1, max=5),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("estimate_period", "Оценивать период (ω — начальное значение)", "select", "no",
                  options=["no", "yes"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Синус и косинус с известным периодом: амплитуда, фаза, моменты максимума и минимума.",
    ),
    MethodSpec(
        method="nonlinear_regression",
        label="Кривая: нелинейная регрессия",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("model", "Кривая", "select", "exponential",
                  options=["exponential", "inverse_linear", "logistic", "gompertz", "power"]),
            Param("start", "Начальные значения (alpha=…; beta=…; пусто — авто)", "text", ""),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Экспоненциальная, обратная линейная, логистическая, Гомперца: асимптоты, приближённые SE.",
    ),
    MethodSpec(
        method="smooth_curve",
        label="Кривая: сплайны и loess",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("method", "Сглаживатель", "select", "bspline",
                  options=["bspline", "smoothing_spline", "loess"]),
            Param("n_knots", "Внутренних узлов (B-сплайн)", "number", 3, min=0, max=20),
            Param("degree", "Степень B-сплайна", "number", 3, min=1, max=3),
            Param("lam", "Штраф λ сглаживающего сплайна (0 — по GCV)", "float", 0.0,
                  min=0.0, max=1e12),
            Param("frac", "Доля точек в окне loess", "float", 0.667, min=0.05, max=1.0),
            Param("iterations", "Робастных итераций loess", "number", 3, min=0, max=10),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Гибкая кривая без переподгонки: B-сплайн с F против прямой, сглаживающий сплайн (GCV), loess.",
    ),
    MethodSpec(
        method="curve_comparison",
        label="Кривая: сравнение моделей",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "X (объясняющая)", "y": "Y (отклик)"},
        params=[
            Param("models", "Модели (через запятую: linear, log10, poly2, exponential…)", "text",
                  "linear, log10, poly2, poly3, exponential, inverse_linear"),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="ResMS, R²adj, AIC и SBC преобразований, полиномов и нелинейных кривых.",
    ),
    MethodSpec(
        method="multiple_regression",
        label="Множественная регрессия",
        group="Связь переменных",
        slots={"y": _NUM},
        slot_labels={"y": "Y (отклик)"},
        params=[
            Param("variables", "Объясняющие переменные (через запятую; пусто — все)", "text", ""),
            Param("x_predict", "Точки предсказания (x1=…, x2=…; …)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Несколько x: ANOVA типов I и III, параметры, VIF, R², AIC, частные остатки.",
    ),
    MethodSpec(
        method="regression_subsets",
        label="Регрессия: все подмножества переменных",
        group="Связь переменных",
        slots={"y": _NUM},
        slot_labels={"y": "Y (отклик)"},
        params=[
            Param("variables", "Объясняющие переменные (через запятую; пусто — все)", "text", ""),
            Param("always_include", "Обязательные переменные (через запятую)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Все подмодели: R²adj, AIC, SBC, Cp Мэллоуза и лучшая модель по каждому критерию.",
    ),
    MethodSpec(
        method="stepwise_selection",
        label="Регрессия: пошаговый отбор",
        group="Связь переменных",
        slots={"y": _NUM},
        slot_labels={"y": "Y (отклик)"},
        params=[
            Param("variables", "Объясняющие переменные (через запятую; пусто — все)", "text", ""),
            Param("strategy", "Стратегия", "select", "forward_stepwise",
                  options=["forward", "backward", "forward_stepwise", "backward_stepwise"]),
            Param("f_in", "Порог Fin (добавление)", "float", 2.0, min=0.01, max=100.0),
            Param("f_out", "Порог Fout (исключение)", "float", 4.0, min=0.01, max=100.0),
            Param("always_include", "Обязательные переменные (через запятую)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Прямой отбор, обратное исключение и пошаговые варианты по F-тестам.",
    ),
    MethodSpec(
        method="model_cross_validation",
        label="Регрессия: сравнение моделей кросс-валидацией",
        group="Связь переменных",
        slots={"y": _NUM},
        slot_labels={"y": "Y (отклик)"},
        params=[
            Param("models", "Модели (x1 + x2 | x1 | …)", "text", ""),
            Param("scheme", "Схема", "select", "kfold", options=["split", "loo", "kfold"]),
            Param("train_fraction", "Доля обучающей выборки", "float", 0.5, min=0.1, max=0.9),
            Param("k_folds", "Число блоков k", "number", 5, min=2, max=100),
            Param("seed", "Зерно случайного разбиения", "number", 1, min=0, max=999999),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Какая из моделей-кандидатов лучше предсказывает: MSEP, MAD, PB.",
    ),
    MethodSpec(
        method="response_surface",
        label="Поверхность отклика (2–4 переменные)",
        group="Связь переменных",
        slots={"y": _NUM},
        slot_labels={"y": "Y (отклик)"},
        params=[
            Param("variables", "Две–четыре объясняющие переменные (через запятую)", "text", ""),
            Param("order", "Порядок поверхности", "select", "quadratic",
                  options=["linear", "interaction", "quadratic", "cubic"]),
            Param("centre", "Центрировать переменные", "select", "mean", options=["mean", "none"]),
            Param("selection", "Отбор членов (с маргинальностью)", "select", "backward",
                  options=["backward", "none"]),
            Param("error_ms", "ResMS чистой ошибки (0 — по повторам в данных)", "float", 0.0,
                  min=0.0, max=1e12),
            Param("error_df", "df чистой ошибки", "number", 0, min=0, max=100000),
            Param("replicates", "Повторов на строку (если строки — средние)", "number", 1,
                  min=1, max=1000),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Полиномиальная поверхность по двум x: неадекватность против чистой ошибки, отбор членов.",
    ),
    MethodSpec(
        method="regression_with_groups",
        label="Регрессия с группами (параллельные линии)",
        group="Связь переменных",
        slots={"y": _NUM, "group": _FACTOR},
        slot_labels={"y": "Y (отклик)", "group": "Колонка групп"},
        params=[
            Param("variables", "Объясняющие переменные (через запятую; пусто — все)", "text", ""),
            Param("group_first", "Группы первыми в последовательной ANOVA", "select", "no",
                  options=["no", "yes"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Отдельные, параллельные или одна линия по группам: ANOVA и выбор модели.",
    ),
    MethodSpec(
        method="blocked_regression",
        label="Регрессия в блочном опыте (RCBD)",
        group="Связь переменных",
        slots={"value": _NUM, "x": _NUM, "block": _FACTOR},
        slot_labels={"value": "Числовая колонка", "x": "X (доза, количественная обработка)",
                     "block": "Колонка блоков"},
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Прямая по дозе внутри блоков: неадекватность и линия для средних условий.",
    ),
    MethodSpec(
        method="ancova",
        label="Ковариационный анализ (ANCOVA)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "group": _FACTOR, "covariate": _NUM, "factor_b": _FACTOR,
               "block": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "group": "Колонка вариантов (фактор A)",
            "covariate": "Ковариата",
            "factor_b": "Фактор B (необязательно)",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Сравнить варианты с поправкой на ковариату: проверка параллельности, средние.",
        optional_slots=("factor_b", "block"),
    ),
    MethodSpec(
        method="glm_analysis",
        label="GLM: доли (биномиальная) и счёты (Пуассон)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "trials": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR,
               "covariate": _NUM, "block": _FACTOR, "exposure": _NUM},
        slot_labels={
            "value": "Отклик: число успехов или счёт",
            "trials": "Число испытаний m (для долей; необязательно)",
            "factor_a": "Фактор A (необязательно)",
            "factor_b": "Фактор B (необязательно)",
            "covariate": "Ковариата (необязательно)",
            "block": "Колонка блоков (необязательно)",
            "exposure": "Экспозиция: время, площадь, объём (Пуассон; необязательно)",
        },
        params=[
            Param("distribution", "Распределение", "select", "binomial",
                  options=["binomial", "poisson"]),
            Param("link", "Функция связи", "select", "canonical",
                  options=["canonical", "logit", "probit", "cloglog", "log"]),
            Param("n_trials", "Постоянное число испытаний m (если нет колонки)", "number", 0,
                  min=0, max=100000),
            Param("covariate_transform", "Преобразование ковариаты", "select", "none",
                  options=["none", "ln", "log10", "sqrt"]),
            Param("covariate_offset", "Сдвиг c для ковариаты", "float", 0.0, min=0.0, max=1e6),
            Param("interactions", "Взаимодействия", "select", "yes", options=["yes", "no"]),
            Param("lack_of_fit", "Фактор неадекватности по ковариате", "select", "no",
                  options=["no", "yes"]),
            Param("selection", "Отбор членов (маргинальные тесты)", "select", "none",
                  options=["none", "backward"]),
            Param("dispersion_method", "Параметр дисперсии φ", "select", "auto",
                  options=["auto", "fixed", "deviance", "pearson"]),
            Param("x_predict", "Точки предсказания ковариаты (через ;)", "text", ""),
            Param("dose_for_p", "Доза для вероятностей p (LD50: 0.5; через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Логистическая и лог-линейная регрессия: ANODEV, сверхдисперсия, отношения шансов.",
        optional_slots=("trials", "factor_a", "factor_b", "covariate", "block", "exposure"),
    ),
    MethodSpec(
        method="count_regression",
        label="Счёты: отрицательная биномиальная и с нулевым раздуванием",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR, "covariate": _NUM,
               "block": _FACTOR, "exposure": _NUM},
        slot_labels={
            "value": "Счёт",
            "factor_a": "Фактор A (необязательно)",
            "factor_b": "Фактор B (необязательно)",
            "covariate": "Ковариата (необязательно)",
            "block": "Колонка блоков (необязательно)",
            "exposure": "Экспозиция: время, площадь, объём (необязательно)",
        },
        params=[
            Param("model", "Модель счётов", "select", "negbin",
                  options=["negbin", "zip", "zinb", "poisson"]),
            Param("covariate_transform", "Преобразование ковариаты", "select", "none",
                  options=["none", "ln", "log10", "sqrt"]),
            Param("covariate_offset", "Сдвиг c для ковариаты", "float", 0.0, min=0.0, max=1e6),
            Param("interactions", "Взаимодействия", "select", "yes", options=["yes", "no"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Кластеризованные счёты и избыток нулей: α, доля нулей π, сравнение с Пуассоном.",
        optional_slots=("factor_a", "factor_b", "covariate", "block", "exposure"),
    ),
    MethodSpec(
        method="ordinal_regression",
        label="Порядковая регрессия (пропорциональные шансы)",
        group="Сравнение нескольких групп",
        slots={"response": _FACTOR, "factor_a": _FACTOR, "factor_b": _FACTOR,
               "covariate": _NUM, "block": _FACTOR},
        slot_labels={
            "response": "Упорядоченный отклик (категории)",
            "factor_a": "Фактор A (необязательно)",
            "factor_b": "Фактор B (необязательно)",
            "covariate": "Ковариата (необязательно)",
            "block": "Колонка блоков (необязательно)",
        },
        params=[
            Param("levels", "Порядок категорий (через запятую; пусто — естественный)", "text", ""),
            Param("link", "Функция связи", "select", "logit", options=["logit", "probit"]),
            Param("covariate_transform", "Преобразование ковариаты", "select", "none",
                  options=["none", "ln", "log10", "sqrt"]),
            Param("covariate_offset", "Сдвиг c для ковариаты", "float", 0.0, min=0.0, max=1e6),
            Param("interactions", "Взаимодействия", "select", "yes", options=["yes", "no"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Упорядоченные исходы (здоров < слабо < сильно): пороги, отношения шансов, LR-тесты.",
        optional_slots=("factor_a", "factor_b", "covariate", "block"),
    ),
    MethodSpec(
        method="curve_in_blocks",
        label="Кривая в блоках и группах: полином или преобразованный x",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "x": _NUM, "block": _FACTOR, "group": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "x": "X (доза, количественная обработка)",
            "block": "Колонка блоков (необязательно)",
            "group": "Колонка групп (необязательно)",
        },
        params=[
            Param("curve", "Вид кривой", "select", "polynomial",
                  options=["polynomial", "transform"]),
            Param("degree", "Порядок полинома q", "number", 2, min=1, max=6),
            Param("x_transform_kind", "Преобразование x", "select", "log10",
                  options=["log10", "ln", "sqrt", "square", "exp", "reciprocal"]),
            Param("x_offset", "Сдвиг c для x", "float", 0.0, min=0.0, max=1000.0),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Кривая отклика на дозу в RCBD: неадекватность, кривые по группам, ДИ для среднего блока.",
        optional_slots=("block", "group"),
    ),
    MethodSpec(
        method="nonlinear_in_blocks",
        label="Нелинейная кривая в блоках и группах",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "x": _NUM, "block": _FACTOR, "group": _FACTOR},
        slot_labels={
            "value": "Числовая колонка",
            "x": "X (доза, количественная обработка)",
            "block": "Колонка блоков (необязательно)",
            "group": "Колонка групп (необязательно)",
        },
        params=[
            Param("model", "Кривая", "select", "exponential",
                  options=["exponential", "inverse_linear", "logistic", "gompertz", "power"]),
            Param("group_model", "Модель групп", "select", "auto",
                  options=["auto", "common", "alpha", "alpha_beta", "all"]),
            Param("x_predict", "Точки предсказания x (через ;)", "text", ""),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Экспонента, логистическая и др. в RCBD: общие или отдельные параметры групп, неадекватность.",
        optional_slots=("block", "group"),
    ),
    MethodSpec(
        method="mixed_model",
        label="Смешанная модель (REML)",
        group="Сравнение нескольких групп",
        slots={"value": _NUM, "factor_a": _FACTOR, "factor_b": _FACTOR, "factor_c": _FACTOR,
               "stratum_1": _FACTOR, "stratum_2": _FACTOR, "stratum_3": _FACTOR,
               "covariate": _NUM},
        slot_labels={
            "value": "Числовая колонка",
            "factor_a": "Фактор A",
            "factor_b": "Фактор B (необязательно)",
            "factor_c": "Фактор C (необязательно)",
            "stratum_1": "Структура: верхний уровень (блок, ферма)",
            "stratum_2": "Структура: 2-й уровень внутри 1-го (необязательно)",
            "stratum_3": "Структура: 3-й уровень внутри 2-го (необязательно)",
            "covariate": "Ковариата (необязательно)",
        },
        params=[
            Param("covariate_transform", "Трансформация ковариаты", "select", "none",
                  options=["none", "log10", "ln", "sqrt", "logit", "logit_pct"]),
            Param("selection", "Отбор фиксированных членов", "select", "none",
                  options=["none", "backward"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            _TRANSFORM_NONE,
            Param("offset", "Сдвиг c", "float", 0.0, min=0.0, max=1000.0),
        ],
        description="Несбалансированные и иерархические дизайны: компоненты дисперсии, F (Кенвард–Роджер).",
        optional_slots=("factor_b", "factor_c", "stratum_2", "stratum_3", "covariate"),
    ),
    MethodSpec(
        method="contingency_test",
        label="Таблица сопряжённости: χ² и точный тест Фишера",
        group="Связь переменных",
        slots={"factor_a": _FACTOR, "factor_b": _FACTOR, "count": _NUM},
        slot_labels={
            "factor_a": "Классификация строк",
            "factor_b": "Классификация столбцов",
            "count": "Частота (если строки — ячейки таблицы; необязательно)",
        },
        params=[
            Param("correction", "Поправка Йейтса", "select", "auto", options=["auto", "yes", "no"]),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
        ],
        description="Связь двух классификаций: χ² Пирсона, G², Фишер для 2 × 2, V Крамера, остатки.",
        optional_slots=("count",),
    ),
    MethodSpec(
        method="sampling_scheme_power",
        label="План вложенной выборки: SE и мощность по компонентам дисперсии",
        group="Сравнение нескольких групп",
        slots={},
        slot_labels={},
        params=[
            Param("components", "Компоненты дисперсии сверху вниз (Блок=…; Единица=…; …)",
                  "text", ""),
            Param("schemes", "Схемы: n_T n_R n_1 … через пробел, схемы через ;", "text", ""),
            Param("delta", "Разность δ для мощности", "float", 0.0, min=0.0, max=1e9),
            Param("alpha", "Уровень значимости α", "float", 0.05, min=0.001, max=0.2),
            Param("position_terms", "Позиции подуровней — фиксированные члены", "select", "yes",
                  options=["yes", "no"]),
        ],
        description="Сколько единиц брать на каждом уровне: дисперсии страт, SED и мощность схем.",
    ),
    MethodSpec(
        method="pearson_correlation",
        label="Корреляция Пирсона",
        group="Связь переменных",
        slots={"x": _NUM, "y": _NUM},
        slot_labels={"x": "Переменная X", "y": "Переменная Y"},
        params=[_ALTERNATIVE, _REG_CI_LEVEL],
        description="Сила и направление линейной связи двух переменных (r) и тест ρ = 0.",
    ),
]


def _columns_by_type(profiles: list[ColumnProfile]) -> dict[ColumnType, list[str]]:
    """Сгруппировать имена колонок по определённому типу."""
    result: dict[ColumnType, list[str]] = {}
    for profile in profiles:
        result.setdefault(profile.detected_type, []).append(profile.name)
    return result


def _eligible(
    by_type: dict[ColumnType, list[str]], col_type: ColumnType | tuple[ColumnType, ...]
) -> list[str]:
    """Колонки, подходящие слоту с одним или несколькими допустимыми типами."""
    types = col_type if isinstance(col_type, tuple) else (col_type,)
    return [name for t in types for name in by_type.get(t, [])]


_MULTI_VARIATE = frozenset({"multiple_regression", "regression_subsets", "stepwise_selection",
                            "model_cross_validation", "response_surface"})


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
    if spec.method == "sampling_scheme_power":  # планирование без данных (гл. 19)
        return True
    if spec.method in _MULTI_VARIATE:  # отклик и не меньше двух x (гл. 14)
        return have_numeric >= 3
    if spec.method == "regression_with_groups":  # отклик, x и группы (гл. 15)
        return have_numeric >= 2 and (bool(by_type.get(_CAT)) or have_numeric >= 3)
    required = {k: t for k, t in spec.slots.items() if k not in spec.optional_slots}
    needed_numeric = sum(1 for t in required.values() if t == _NUM)
    if have_numeric < needed_numeric:
        return False
    if _CAT in required.values() and not by_type.get(_CAT):
        return False
    # Каждому обязательному слоту нужна своя колонка: хватает ли подходящих в сумме.
    eligible = {name for t in required.values() for name in _eligible(by_type, t)}
    return len(eligible) >= len(required)


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
