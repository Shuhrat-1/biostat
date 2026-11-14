"""Аудит 2: вырожденные данные и единицы измерения (находки 02-01, 02-04, 02-11)."""

from __future__ import annotations

import pytest

from tests.bridge_runner import CASES, errors, p_values, run, scaled, table

# Методы из находки 02-01: отказывали «остаточной вариации нет» на малом масштабе.
SCALE_METHODS = ["factorial_anova", "nested_anova", "treatment_contrasts",
                 "polynomial_contrasts", "mean_comparisons", "latin_square_anova",
                 "split_plot_anova", "bibd_anova", "replication_power", "equivalence_test",
                 "one_way_anova", "unbalanced_anova", "rcbd_anova", "simple_effects"]
# Параметры в единицах отклика масштабируются вместе с ним.
UNIT_KWARGS = ("delta", "margin")


@pytest.mark.parametrize("method", SCALE_METHODS)
@pytest.mark.parametrize("factor", [1e-6, 1e-9])
def test_02_01_p_values_do_not_depend_on_units(method: str, factor: float) -> None:
    """Те же данные в моль/л вместо мкмоль/л: P те же, отказа нет."""
    cols = table(method)
    kwargs = dict(CASES[method][2])
    base = run(method, cols, kwargs=kwargs)
    small_kwargs = {k: v * factor if k in UNIT_KWARGS else v for k, v in kwargs.items()}
    small = run(method, scaled(cols, factor), kwargs=small_kwargs)
    assert errors(base) == [] and errors(small) == []
    assert p_values(base), method
    assert p_values(small) == pytest.approx(p_values(base), rel=1e-6, abs=1e-12)


# --- 02-11. Группа из одного наблюдения ---------------------------------------------------

SINGLE = {"g": ["a"] * 5 + ["b"] * 5 + ["c"],
          "y": [9.1, 10.4, 9.8, 10.9, 9.5, 11.2, 12.1, 11.6, 12.8, 11.9, 14.0]}
G = {"value": "y", "group": "g"}
COMPUTE_WITH_SINGLE = ["one_way_anova", "treatment_means", "anova_residuals",
                       "anova_permutation_test", "transformed_means", "tukey_hsd",
                       "scheffe_test", "kruskal_wallis"]


@pytest.mark.parametrize("method", COMPUTE_WITH_SINGLE)
def test_02_11_single_observation_group_is_computed_with_the_same_warning(method: str) -> None:
    """Группы 5, 5 и 1: расчёт идёт (остаточные df = N − t = 8), предупреждение одно на всех."""
    res = run(method, SINGLE, G, {})
    assert errors(res) == []
    note = [w for w in res["warnings"] if w["code"] == "single_observation_group"]
    assert len(note) == 1 and note[0]["params"] == {"groups": "c"}
    assert note[0]["severity"] == "warning"


def test_02_11_one_way_anova_with_a_single_group_matches_scipy() -> None:
    from scipy import stats

    res = run("one_way_anova", SINGLE, G, {})
    ref = stats.f_oneway(SINGLE["y"][:5], SINGLE["y"][5:10], SINGLE["y"][10:])
    assert res["statistics"]["df_within"] == 8
    assert res["statistics"]["F"] == pytest.approx(ref.statistic)
    assert res["p_value"] == pytest.approx(ref.pvalue)


@pytest.mark.parametrize("method", ["welch_anova", "bartlett_test"])
def test_02_11_methods_that_need_each_group_variance_still_refuse(method: str) -> None:
    assert errors(run(method, SINGLE, G, {})) == ["n_too_small"]


@pytest.mark.parametrize("method", ["one_way_anova", "treatment_means", "anova_residuals"])
def test_02_11_no_residual_df_is_still_refused(method: str) -> None:
    cols = {"g": ["a", "b", "c"], "y": [1.0, 2.0, 4.0]}
    assert errors(run(method, cols, G, {})) == ["n_too_small"]


# --- 02-04. Константный отклик и точная подгонка ------------------------------------------

from tests.bridge_runner import RESPONSES  # noqa: E402

# Методы из находки: падали с ZeroDivisionError / ошибкой scipy или молча давали пустые ячейки.
CONSTANT_METHODS = [
    "normality_test",  # давал W = 1, P = 1.0 — «нормально», хотя проверять нечего
    "regression_details", "inverse_prediction", "regression_cross_validation",
    "x_transform_regression", "polynomial_regression", "trigonometric_regression",
    "nonlinear_regression", "smooth_curve", "curve_comparison", "multiple_regression",
    "regression_subsets", "stepwise_selection", "regression_with_groups",
    "kruskal_wallis", "wilcoxon", "mixed_model",
    "one_way_anova", "two_sample_t", "paired_t", "tukey_hsd", "one_sample_t",
    "simple_linear_regression",
]


def _constant(cols: dict[str, list]) -> dict[str, list]:
    return {k: [5.0] * len(v) if k in RESPONSES else v for k, v in cols.items()}


@pytest.mark.parametrize("method", sorted(CASES))
def test_02_04_constant_response_never_raises(method: str) -> None:
    """Все отклики равны 5: любой метод возвращает результат, а не исключение."""
    run(method, _constant(table(method, 48)))


@pytest.mark.parametrize("method", CONSTANT_METHODS)
def test_02_04_constant_response_is_refused_with_a_code(method: str) -> None:
    res = run(method, _constant(table(method, 48)))
    assert errors(res) == ["no_variation"], errors(res)
    assert res["p_value"] is None and res["estimates"] == []


def test_02_04_describe_all_marks_a_constant_column_without_refusing() -> None:
    """Среднее, n, минимум и максимум осмысленны; асимметрия и эксцесс — прочерк и заметка."""
    import warnings

    cols = {"flat": [5.0] * 8, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scipy не должен считать моменты константы
        res = run("describe_all", cols, {}, {})
    flat, y = res["parameters"]["summary"]["flat"], res["parameters"]["summary"]["y"]
    assert errors(res) == []
    assert (flat["mean"], flat["n"], flat["min"], flat["max"], flat["std"]) == (5.0, 8, 5.0, 5.0, 0)
    assert flat["skewness"] is None and flat["kurtosis"] is None
    assert y["skewness"] is not None and y["kurtosis"] is not None
    (note,) = [w for w in res["warnings"] if w["code"] == "columns_no_variation"]
    assert note["severity"] == "info" and note["params"] == {"columns": "flat"}


EXACT = {"x": list(range(1, 31)), "y": [2 * v + 1 for v in range(1, 31)]}  # y = 2x + 1
EXACT_METHODS = ["simple_linear_regression", "regression_details", "inverse_prediction",
                 "regression_diagnostics", "regression_cross_validation",
                 "polynomial_regression", "x_transform_regression",
                 "trigonometric_regression", "nonlinear_regression", "smooth_curve"]


@pytest.mark.parametrize("method", EXACT_METHODS)
def test_02_04_exact_fit_is_refused_not_reported_as_noise(method: str) -> None:
    """Точная прямая: остаточная SS = 0, F и t — шум округления; отказ с кодом."""
    kwargs = {"y_new": "8"} if method == "inverse_prediction" else {}
    res = run(method, EXACT, {"x": "x", "y": "y"}, kwargs)
    assert errors(res) == ["no_residual_variation"], (errors(res), res["statistics"])


def test_02_04_all_zero_differences_are_refused() -> None:
    cols = {"before": [3.1, 4.2, 5.0, 2.7, 3.9, 4.4], "after": [3.1, 4.2, 5.0, 2.7, 3.9, 4.4]}
    for method in ("paired_t", "wilcoxon"):
        assert errors(run(method, cols, {"before": "before", "after": "after"}, {})) == [
            "no_variation"]


def test_02_04_groups_without_within_variation_are_refused() -> None:
    cols = {"g": ["a"] * 3 + ["b"] * 3 + ["c"] * 3, "y": [1.0] * 3 + [2.0] * 3 + [4.0] * 3}
    for method in ("one_way_anova", "tukey_hsd"):
        assert errors(run(method, cols, G, {})) == ["no_residual_variation"]
    two = {"g": ["a"] * 3 + ["b"] * 3, "y": [1.0] * 3 + [2.0] * 3}
    assert errors(run("two_sample_t", two, G, {})) == ["no_residual_variation"]


# --- 02-07. График transformed_means строится из таблицы результата ----------------------

def test_02_07_transformed_means_chart_shows_the_table_not_arithmetic_means() -> None:
    """Welham гл. 6.3–6.4: на графике — обратно преобразованные средние и асимметричные ДИ."""
    from tests.bridge_runner import plots

    cols = {"g": ["a"] * 5 + ["b"] * 8 + ["c"] * 6,
            "y": [3.1, 4.0, 2.7, 9.5, 3.3, 6.2, 5.1, 14.8, 7.7, 6.9, 5.5, 8.1, 6.0,
                  12.4, 30.2, 11.1, 15.6, 13.3, 12.0]}
    res = run("transformed_means", cols, G, {"transform_kind": "ln"})
    bundle = plots("transformed_means", G)
    table = {e["name"][5:]: e for e in res["estimates"]}
    assert "bar_means" not in bundle and "boxplot" in bundle
    items = bundle["means_ci"]["items"]
    assert [it["label"] for it in items] == ["a", "b", "c"]
    for it in items:
        row = table[it["label"]]
        assert (it["mean"], it["lower"], it["upper"]) == pytest.approx(
            (row["value"], row["ci_lower"], row["ci_upper"]))
        assert it["upper"] - it["mean"] > it["mean"] - it["lower"]  # ДИ асимметричен
    arithmetic = sum(cols["y"][:5]) / 5
    assert items[0]["mean"] < arithmetic  # геометрическое среднее меньше арифметического


# --- 02-08. Строки с пропусками: одно сообщение с числом ----------------------------------

from tests.bridge_runner import codes  # noqa: E402


def _dropped(res: dict) -> list[dict]:
    return [w for w in res["warnings"] if w["code"] == "missing_dropped"]


def _with_gaps() -> dict[str, list]:
    """15 строк: три пустые метки группы («», NA, -), один пустой отклик, пропуски в note."""
    g = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    y = [9.1, 10.4, 9.8, 10.9, 9.5, 11.2, 12.1, 11.6, 12.8, 11.9, 14.0, 13.1, 15.2, 14.4, 13.7]
    g[2], g[7], g[12] = None, "NA", "-"
    y[0] = None
    return {"g": g, "y": y, "note": [None, "ok"] * 7 + [None]}


@pytest.mark.parametrize("method", ["one_way_anova", "kruskal_wallis", "tukey_hsd",
                                    "describe_by_group", "welch_anova"])
def test_02_08_dropped_rows_are_reported_once_with_counts(method: str) -> None:
    res = run(method, _with_gaps(), G, {})
    (note,) = _dropped(res)
    assert note["severity"] == "info"
    assert note["params"] == {"n": 4, "detail": "y — 1, g — 3"}  # колонка note не в счёте
    assert res["n"] == 11


def test_02_08_complete_data_and_trailing_empty_rows_give_no_message() -> None:
    cols = {"g": ["a"] * 4 + ["b"] * 4 + [None, None], "y": [1.0, 2.0, 3.0, 2.5, 4.0, 5.0, 6.0,
                                                               5.5, None, None]}
    res = run("one_way_anova", cols, G, {})
    assert res["n"] == 8 and _dropped(res) == []


def test_02_08_method_own_message_is_not_duplicated() -> None:
    """simple_linear_regression сама сообщает о парах с пропусками: остаётся одно сообщение."""
    cols = {"x": [1, 2, 3, 4, 5, 6, 7, 8], "y": [2.1, None, 6.2, 7.9, None, 12.3, 13.8, 16.1]}
    res = run("simple_linear_regression", cols, {"x": "x", "y": "y"}, {})
    (note,) = _dropped(res)
    assert note["params"] == {"n": 2, "detail": "y — 2"} and res["n"] == 6


def test_02_08_refusals_that_explain_the_gap_are_left_alone() -> None:
    """RCBD с пропуском: метод отказывает `rcbd_incomplete` — второго сообщения нет."""
    cols = table("rcbd_anova", 24)
    cols["y"][3] = None
    res = run("rcbd_anova", cols)
    assert "rcbd_incomplete" in codes(res) and _dropped(res) == []


def test_02_08_only_selected_regression_variables_count() -> None:
    cols = {"y": [3.1, 4.9, 7.2, 8.8, 11.1, 13.2, 14.8, 17.1, 19.0, 21.2],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [2.0, 1.0, 4.0, 3.0, None, 5.0, 8.0, 7.0, 10.0, 9.0],
            "x3": [None, None, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    res = run("multiple_regression", cols, {"y": "y"}, {"variables": "x1, x2"})
    (note,) = _dropped(res)
    assert note["params"] == {"n": 1, "detail": "x2 — 1"} and res["n"] == 9


def test_02_08_paired_methods_count_pairs_with_any_gap() -> None:
    cols = {"before": [3.1, None, 5.0, 2.7, 3.9, 4.4, 5.1],
            "after": [3.6, 4.9, None, 3.1, 4.6, None, 5.9]}
    res = run("paired_t", cols, {"before": "before", "after": "after"}, {})
    (note,) = _dropped(res)
    assert note["params"] == {"n": 3, "detail": "before — 1, after — 2"} and res["n"] == 4


# --- Страховка: новые проверки не ломают обычные данные -----------------------------------

@pytest.mark.parametrize("method", sorted(CASES))
def test_guards_leave_ordinary_data_alone(method: str) -> None:
    """Синтетика `tools/bench_data.py`: ни одного отказа и ни одной пометки о вырожденности."""
    res = run(method, table(method))
    expected = ["surface_no_pure_error"] if method == "response_surface" else []
    assert errors(res) == expected
    assert not {"no_variation", "no_residual_variation", "single_observation_group"} & set(
        codes(res))


def test_02_04_descriptive_table_shows_core_notes() -> None:
    """Заметка `columns_no_variation` видна: таблица описательной статистики выводит сообщения."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "web/src/components.tsx").read_text("utf-8")
    table_src = src[src.index("function SummaryTable("):src.index("function fmtSummary(")]
    assert "<WarningList warnings={result.warnings} />" in table_src
    messages = (Path(__file__).resolve().parents[1] / "web/src/coreMessages.tsx").read_text("utf-8")
    for code in ("columns_no_variation", "no_variation", "single_observation_group"):
        block = messages[messages.index(f"  {code}: {{"):]
        assert all(f"    {lang}: `" in block[:1200] for lang in ("ru", "en", "pt")), code
