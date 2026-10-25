"""Тесты каталога методов: применимость и метаданные."""

from __future__ import annotations

from core.validation.catalog import CATALOG, GROUPS, available_methods
from core.validation.parser import parse_file


def _profiles(data: bytes):
    return parse_file(data).profiles


NUMERIC_ONLY = b"x,y\n1,2\n3,4\n5,6\n7,8\n"
NUM_AND_CAT = (
    "sort,yield\n" + "\n".join(f"{'ab'[i % 2]},{i}" for i in range(20)) + "\n"
).encode("utf-8")
# Блочный дизайн: блоки и делянки закодированы числами, как в файлах книги.
BLOCKED = (
    "block,plot,sort,yield\n"
    + "\n".join(f"{b},{p},{'abcde'[p - 1]},{b * 10 + p}" for b in range(1, 5) for p in range(1, 6))
    + "\n"
).encode("utf-8")


def test_numeric_only_methods():
    """Без категориальной колонки — нет групповых методов."""
    methods = {m["method"] for m in available_methods(_profiles(NUMERIC_ONLY))}
    assert "describe_all" in methods
    assert "mean_ci" in methods
    assert "one_sample_t" in methods
    assert "simple_linear_regression" in methods  # две числовые
    assert "one_way_anova" not in methods
    assert "two_sample_t" not in methods


def test_grouped_methods_appear():
    """С категориальной колонкой появляются сравнения групп."""
    methods = {m["method"] for m in available_methods(_profiles(NUM_AND_CAT))}
    assert "two_sample_t" in methods
    assert "mann_whitney" in methods
    assert "one_way_anova" in methods
    assert "kruskal_wallis" in methods
    assert "tukey_hsd" in methods


def test_single_numeric_no_regression():
    """Одна числовая колонка — регрессия недоступна (нужно две)."""
    methods = {m["method"] for m in available_methods(_profiles(b"x\n1\n2\n3\n"))}
    assert "describe_all" in methods
    assert "simple_linear_regression" not in methods


def test_all_catalog_methods_covered():
    """Каждый метод каталога попадает хоть в какой-то состав данных."""
    num = {m["method"] for m in available_methods(_profiles(NUMERIC_ONLY))}
    cat = {m["method"] for m in available_methods(_profiles(NUM_AND_CAT))}
    blocked = {m["method"] for m in available_methods(_profiles(BLOCKED))}
    covered = num | cat | blocked
    all_methods = {spec.method for spec in CATALOG}
    assert all_methods == covered


def test_params_present():
    """Методы с параметрами отдают их описание."""
    methods = {m["method"]: m for m in available_methods(_profiles(NUMERIC_ONLY))}
    one_sample = methods["one_sample_t"]
    param_names = {p["name"] for p in one_sample["params"]}
    assert "popmean" in param_names
    assert "alternative" in param_names


def test_n_groups_constraint():
    """Двухгрупповые методы помечены n_groups=2."""
    methods = {m["method"]: m for m in available_methods(_profiles(NUM_AND_CAT))}
    assert methods["two_sample_t"]["n_groups"] == 2
    assert methods["mann_whitney"]["n_groups"] == 2
    assert methods["one_way_anova"]["n_groups"] is None


def test_catalog_translations():
    """Метки, группы и описания переводятся на en и pt."""
    profiles = _profiles(NUM_AND_CAT)
    en = {m["method"]: m for m in available_methods(profiles, "en")}
    pt = {m["method"]: m for m in available_methods(profiles, "pt")}
    assert en["one_way_anova"]["label"] == "One-way ANOVA"
    assert en["one_way_anova"]["group"] == "Multi-group comparison"
    assert pt["one_way_anova"]["label"] == "ANOVA de um fator"
    # Русский по умолчанию.
    ru = {m["method"]: m for m in available_methods(profiles)}
    assert ru["one_way_anova"]["label"] == "Однофакторный ANOVA"


def test_catalog_translation_fallback():
    """Неизвестный язык откатывается к русскому ключу без падения."""
    methods = available_methods(_profiles(NUMERIC_ONLY), "xx")
    assert methods  # не пусто, не упало


def test_grouped_by_known_groups():
    """Все группы методов входят в объявленный список GROUPS."""
    for spec in CATALOG:
        assert spec.group in GROUPS


def test_needs_populated():
    """Слоты заполнены реальными именами колонок."""
    methods = {m["method"]: m for m in available_methods(_profiles(NUM_AND_CAT))}
    anova = methods["one_way_anova"]
    assert "sort" in anova["needs"]["group"]
    assert "yield" in anova["needs"]["value"]


def test_catalog_params_match_function_signatures():
    """Каждый параметр каталога существует в сигнатуре своей функции.

    Ловит рассинхрон имён (баг: каталог слал 'level', а регрессия
    принимает 'ci_level').
    """
    import importlib
    import inspect

    # Метод -> модуль ядра (соответствует METHOD_MODULES в мосте).
    modules = {
        "describe_all": "core.stats.descriptive",
        "mean_ci": "core.stats.descriptive",
        "normality_test": "core.stats.diagnostics",
        "one_sample_t": "core.stats.hypothesis",
        "two_sample_t": "core.stats.hypothesis",
        "mann_whitney": "core.stats.hypothesis",
        "paired_t": "core.stats.hypothesis",
        "wilcoxon": "core.stats.hypothesis",
        "one_way_anova": "core.stats.anova",
        "tukey_hsd": "core.stats.anova",
        "kruskal_wallis": "core.stats.anova",
        "treatment_means": "core.stats.anova",
        "anova_permutation_test": "core.stats.anova_checks",
        "bartlett_test": "core.stats.anova_checks",
        "anova_residuals": "core.stats.anova_checks",
        "transformed_means": "core.stats.transforms",
        "rcbd_anova": "core.stats.blocking",
        "rcbd_subsampling_anova": "core.stats.blocking",
        "factorial_anova": "core.stats.factorial",
        "nested_anova": "core.stats.factorial",
        "simple_linear_regression": "core.stats.regression",
        "pearson_correlation": "core.stats.correlation",
    }
    for spec in CATALOG:
        if not spec.params:
            continue
        fn = getattr(importlib.import_module(modules[spec.method]), spec.method)
        sig = set(inspect.signature(fn).parameters)
        for param in spec.params:
            assert param.name in sig, (
                f"{spec.method}: параметр '{param.name}' отсутствует "
                f"в сигнатуре {sorted(sig)}"
            )


def test_blocked_methods_accept_numeric_block_codes():
    """Блоки 1…4 типизируются как числа — слот блока должен их предлагать."""
    methods = {m["method"]: m for m in available_methods(_profiles(BLOCKED))}
    rcbd = methods["rcbd_anova"]
    assert "block" in rcbd["needs"]["block"]
    assert rcbd["needs"]["group"] == ["sort"]
    assert "rcbd_subsampling_anova" in methods


def test_blocked_methods_hidden_without_enough_columns():
    """Двух колонок мало для блочного анализа (нужны отклик, варианты, блоки)."""
    methods = {m["method"] for m in available_methods(_profiles(NUM_AND_CAT))}
    assert "rcbd_anova" not in methods


def test_structured_methods_optional_slots():
    """Третий фактор и блок необязательны: отклик + 2 фактора достаточно."""
    data = b"a,b,y\n" + b"".join(f"{i % 2},{i % 3},{i}\n".encode() for i in range(12))
    methods = {m["method"]: m for m in available_methods(_profiles(data))}
    fac = methods["factorial_anova"]
    assert fac["optional_slots"] == ["factor_c", "block"]
    assert set(fac["needs"]) == {"value", "factor_a", "factor_b", "factor_c", "block"}
    assert methods["nested_anova"]["optional_slots"] == ["block"]
    assert methods["mean_ci"]["optional_slots"] == []


def test_structured_methods_hidden_with_two_columns():
    methods = {m["method"] for m in available_methods(_profiles(NUM_AND_CAT))}
    assert "factorial_anova" not in methods
    assert "nested_anova" not in methods
