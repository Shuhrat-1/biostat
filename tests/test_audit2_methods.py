"""Аудит 2, шаг 2: находки 02-03, 02-05, 02-06, 02-09, 02-10, 02-12 (через шаблоны моста)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from tests.bridge_runner import codes, errors, run

# --- 02-03. nested_anova — вложенные варианты, не иерархическая выборка -------------------


def _leaves_on_plants(seed: int = 7) -> dict[str, list]:
    """2 варианта × 4 растения × 5 листьев; эффекта варианта нет (сценарий `f_nested.py`)."""
    rng = np.random.default_rng(seed)
    plant = np.repeat(np.arange(8), 5)
    y = 10 + rng.normal(0, 1.0, 8)[plant] + rng.normal(0, 0.5, 40)
    return {"y": y.round(3).tolist(), "treatment": [f"t{p // 4}" for p in plant],
            "plant": [f"p{p}" for p in plant]}


def test_02_03_nested_anova_has_no_always_on_warning() -> None:
    """Несколько строк на члена — норма (Welham, гл. 8.4: изолят на нескольких делянках),
    поэтому предупреждения нет; о фиксированных членах говорит описание метода."""
    from core.validation.catalog import CATALOG

    roles = {"value": "y", "group": "treatment", "member": "plant"}
    res = run("nested_anova", _leaves_on_plants(), roles, {})
    assert errors(res) == [] and "nested_members_fixed" not in codes(res)
    spec = next(m for m in CATALOG if m.method == "nested_anova")
    assert ("Оба фактора фиксированы; если члены — единицы выборки (растения, листья), "
            "используйте смешанную модель.") in spec.description


def test_02_03_method_label_says_nested_treatments_in_three_languages() -> None:
    from core.validation.catalog import CATALOG
    from core.validation.catalog_i18n import DESCRIPTIONS, LABELS, tr

    spec = next(m for m in CATALOG if m.method == "nested_anova")
    assert "Вложенные варианты" in spec.label and "Welham, гл. 8.4" in spec.description
    assert tr(LABELS, spec.label, "en").startswith("Nested treatments")
    assert tr(LABELS, spec.label, "pt").startswith("Tratamentos hierárquicos")
    for lang, word in (("en", "mixed model"), ("pt", "modelo misto")):
        text = tr(DESCRIPTIONS, spec.description, lang)
        assert text != spec.description and word in text


# --- 02-05. Таблица 2 × 2: один главный P в двух методах ---------------------------------


def _two_by_two(x1: int, n1: int, x2: int, n2: int) -> dict[str, list]:
    return {"g": ["a"] * n1 + ["b"] * n2,
            "r": ["yes"] * x1 + ["no"] * (n1 - x1) + ["yes"] * x2 + ["no"] * (n2 - x2)}


def test_02_05_two_methods_give_the_same_p_for_the_same_table() -> None:
    """18 из 40 против 9 из 38 (находка): было 0.048 и 0.082, теперь одно значение."""
    cols = _two_by_two(18, 40, 9, 38)
    prop = run("two_proportions", cols, {"response": "r", "group": "g"}, {"success": "yes"})
    cont = run("contingency_test", cols, {"factor_a": "g", "factor_b": "r"}, {})
    obs = [[18, 22], [9, 29]]
    ref = stats.chi2_contingency(obs, correction=False)
    assert cont["p_value"] == pytest.approx(ref.pvalue) == pytest.approx(prop["p_value"])
    assert cont["statistics"]["chi2"] == pytest.approx(prop["statistics"]["z"] ** 2)
    assert cont["statistics"]["yates"] is False


def test_02_05_yates_value_and_fisher_are_shown_in_both_methods() -> None:
    cols = _two_by_two(18, 40, 9, 38)
    obs = [[18, 22], [9, 29]]
    yates = stats.chi2_contingency(obs, correction=True)
    fisher = stats.fisher_exact(obs).pvalue
    prop = run("two_proportions", cols, {"response": "r", "group": "g"}, {"success": "yes"})
    cont = run("contingency_test", cols, {"factor_a": "g", "factor_b": "r"}, {})
    for res in (prop, cont):
        st = res["statistics"]
        assert st["chi2_yates"] == pytest.approx(yates.statistic)
        assert st["p_yates"] == pytest.approx(yates.pvalue)
        assert st["fisher_p"] == pytest.approx(fisher)


def test_02_05_small_expected_frequency_points_to_fisher_in_both_methods() -> None:
    """5 из 6 против 1 из 6: все ожидаемые частоты равны 3."""
    cols = _two_by_two(5, 6, 1, 6)
    prop = run("two_proportions", cols, {"response": "r", "group": "g"}, {"success": "yes"})
    cont = run("contingency_test", cols, {"factor_a": "g", "factor_b": "r"}, {})
    assert "prop_small_expected" in codes(prop) and "contingency_small_expected" in codes(cont)
    note = next(w for w in prop["warnings"] if w["code"] == "prop_small_expected")
    assert note["params"] == {"p": f"{stats.fisher_exact([[5, 1], [1, 5]]).pvalue:.4f}"}
    big = run("two_proportions", _two_by_two(18, 40, 9, 38), {"response": "r", "group": "g"},
              {"success": "yes"})
    assert "prop_small_expected" not in codes(big)


def test_02_05_one_sided_fisher_follows_the_direction_of_the_difference() -> None:
    """greater: доля во второй группе больше (p₂ − p₁ > 0)."""
    cols = _two_by_two(3, 12, 9, 12)
    res = run("two_proportions", cols, {"response": "r", "group": "g"},
              {"success": "yes", "alternative": "greater"})
    ref = stats.fisher_exact([[9, 3], [3, 9]], alternative="greater").pvalue
    assert res["statistics"]["fisher_p"] == pytest.approx(ref)
    assert res["statistics"]["fisher_p"] < 0.05 and "chi2_yates" not in res["statistics"]


# --- 02-06. Нормальность: метод и карта судят по одному правилу --------------------------


def _map_says_nonnormal(cols: dict[str, list]) -> bool:
    from core.validation.parser import parse_file
    from core.validation.recognize import COMPUTED, recognize
    from tests.bridge_runner import to_csv

    out = recognize(parse_file(to_csv(cols)), {"value": "y", "group": "g"}, cls=COMPUTED)
    assert out["errors"] == []
    return any(f["id"] == "B4.1" for f in out["fired"])


def _heavy_tails(seed: int, groups: int = 3, n: int = 150) -> dict[str, list]:
    """Симметричное распределение с тяжёлыми хвостами t(6): F устойчив по ЦПТ."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.standard_t(6, n) + 0.2 * i for i in range(groups)]).round(3)
    return {"g": [f"g{i}" for i in range(groups) for _ in range(n)], "y": y.tolist()}


@pytest.mark.parametrize("method, groups", [("one_way_anova", 3), ("two_sample_t", 2)])
def test_02_06_method_warning_and_map_signal_agree_on_large_samples(method: str,
                                                                    groups: int) -> None:
    """Находка: на 3 × 150 из t(6) метод предупреждал в 36 случаях из 40, карта — в 4."""
    warned = rejected = 0
    for seed in range(20):
        cols = _heavy_tails(seed, groups)
        res = run(method, cols, {"value": "y", "group": "g"}, {})
        by_method = bool({"normality_violated", "residuals_not_normal"} & set(codes(res)))
        assert by_method == _map_says_nonnormal(cols), seed
        warned += by_method
        rejected += res["statistics"]["shapiro_p_residuals"] < 0.05
    assert rejected >= 16 and warned <= 8  # Шапиро отвергает почти всегда, правило — редко


def test_02_06_skewed_large_samples_are_flagged_by_both() -> None:
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.lognormal(1 + 0.2 * i, 0.8, 100) for i in range(3)]).round(3)
    cols = {"g": [f"g{i}" for i in range(3) for _ in range(100)], "y": y.tolist()}
    res = run("one_way_anova", cols, {"value": "y", "group": "g"}, {})
    assert "residuals_not_normal" in codes(res) and _map_says_nonnormal(cols)


def test_02_06_anova_advice_follows_the_book() -> None:
    """Welham стр. 102, 115: перестановочный тест или трансформация, не Краскел–Уоллис."""
    cols = {"g": ["a"] * 8 + ["b"] * 8 + ["c"] * 8,
            "y": [1.0, 1.2, 1.1, 1.3, 1.2, 1.1, 1.0, 9.5, 2.0, 2.2, 2.1, 2.3, 2.2, 2.1, 2.0, 12.5,
                  3.0, 3.2, 3.1, 3.3, 3.2, 3.1, 3.0, 15.5]}
    res = run("one_way_anova", cols, {"value": "y", "group": "g"}, {})
    (note,) = [w for w in res["warnings"] if w["code"] == "residuals_not_normal"]
    assert "перестановочный тест сохраняет F и средние" in note["message"]
    assert "трансформация" in note["message"] and "для баллов и рангов" in note["message"]
    assert note["params"] == {"p": f"{res['statistics']['shapiro_p_residuals']:.4f}"}
    assert "normality_violated" not in codes(res)


def test_02_06_normality_test_itself_reports_shapiro_as_is() -> None:
    """Отдельный тест Шапиро по колонке под правило больших выборок не подпадает."""
    y = np.random.default_rng(0).standard_t(4, 400).round(3).tolist()
    res = run("normality_test", {"y": y}, {"value": "y"}, {})
    assert res["p_value"] == pytest.approx(stats.shapiro(y).pvalue)
    assert res["p_value"] < 0.05 and "normality_violated" in codes(res)


# --- 02-10. Асимметрия и эксцесс — с поправкой на n, формула названа ----------------------

SHAPE = [2.1, 2.4, 1.9, 3.6, 2.8, 2.2, 5.9, 2.6, 3.1, 2.0, 4.4, 2.5, 2.3, 8.2, 2.9]  # n = 15


def _excel_shape(v: list[float]) -> tuple[float, float]:
    """Формулы Excel SKEW и KURT (те же в SPSS и GenStat)."""
    a = np.asarray(v, dtype=float)
    n, z = a.size, (a - a.mean()) / a.std(ddof=1)
    g1 = n / ((n - 1) * (n - 2)) * (z ** 3).sum()
    g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * (z ** 4).sum()
          - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(g1), float(g2)


def test_02_10_describe_all_shows_adjusted_skewness_and_kurtosis() -> None:
    summary = run("describe_all", {"y": SHAPE}, {}, {})["parameters"]["summary"]["y"]
    g1, g2 = _excel_shape(SHAPE)
    assert summary["skewness"] == pytest.approx(g1) and summary["kurtosis"] == pytest.approx(g2)
    assert summary["skewness"] > float(stats.skew(SHAPE))  # g₁ без поправки смещён к нулю


def test_02_10_normality_test_keeps_keys_and_labels_name_the_formula() -> None:
    """Ключи `skewness`, `kurtosis` прежние (экспорт JSON); формула — в подписи фронта."""
    from pathlib import Path

    st = run("normality_test", {"y": SHAPE}, {"value": "y"}, {})["statistics"]
    g1, g2 = _excel_shape(SHAPE)
    assert st["skewness"] == pytest.approx(g1) and st["kurtosis"] == pytest.approx(g2)
    web = Path(__file__).resolve().parents[1] / "web/src"
    labels = (web / "extraLabels.ts").read_text("utf-8").split("const STATS")[1].split("};")[0]
    assert labels.count("skewness: ") == 3 and labels.count("G₁ (") == 3
    assert labels.count("kurtosis: ") == 3 and labels.count("G₂ (") == 3
    assert "statLabel(k, lang)" in (web / "components.tsx").read_text("utf-8")


def test_02_10_labels_name_the_formula_in_three_languages() -> None:
    from pathlib import Path

    i18n = (Path(__file__).resolve().parents[1] / "web/src/i18n.tsx").read_text("utf-8")
    assert i18n.count('"m.skewness": ') == 3 and i18n.count("G₁ (") == 3
    assert i18n.count('"m.kurtosis": ') == 3 and i18n.count("G₂ (") == 3


def test_02_10_map_rules_still_use_unadjusted_skewness() -> None:
    """Пороги R2.7, B4.4 и правила нормальности заданы для g₁ — формула там прежняя."""
    from core.stats import normality_rule
    from core.validation import recognizers_computed as rc

    big = np.asarray(SHAPE * 3, dtype=float)  # n = 45 > 30: решает |g₁| > 0.5
    assert normality_rule.departure_matters(big, 45) == (abs(float(stats.skew(big))) > 0.5)
    assert "bias=False" not in Path_read(rc)


def Path_read(module) -> str:  # noqa: N802 — маленький помощник теста
    from pathlib import Path

    return Path(module.__file__).read_text(encoding="utf-8")


def test_02_10_tiny_samples_get_a_dash_not_a_wrong_number() -> None:
    summary = run("describe_all", {"a": [1.0, 2.0, 4.0], "b": [1.0, 3.0, None]}, {}, {})[
        "parameters"]["summary"]
    assert summary["a"]["skewness"] is not None and summary["a"]["kurtosis"] is None
    assert summary["b"].get("skewness") is None


# --- 02-12. ordinal_regression: сходимость не зависит от порядка строк --------------------


def test_02_12_convergence_warning_does_not_depend_on_row_order() -> None:
    """Синтетика `bench_data.py`, 120 строк: в исходном порядке было предупреждение
    `ordinal_not_converged`, после перестановки строк — нет; оценки при этом совпадали."""
    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    from tests.bridge_runner import table

    cols = table("ordinal_regression")
    order = np.random.default_rng(7).permutation(len(cols["y"]))
    shuffled = {k: [v[i] for i in order] for k, v in cols.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # не должен просочиться наружу
        base, moved = run("ordinal_regression", cols), run("ordinal_regression", shuffled)
    assert "ordinal_not_converged" not in codes(base) + codes(moved)
    assert base["statistics"]["loglik"] == pytest.approx(moved["statistics"]["loglik"], rel=1e-8)
    assert [e["value"] for e in base["estimates"]] == pytest.approx(
        [e["value"] for e in moved["estimates"]], rel=1e-4)


def test_02_12_complete_separation_is_a_refusal() -> None:
    """Категории полностью разделены по x: log L ≈ 0, оценки на бесконечности — отказ."""
    cols = {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "s": ["low"] * 4 + ["mid"] * 4 + ["high"] * 4}
    res = run("ordinal_regression", cols, {"response": "s", "covariate": "x"},
              {"levels": "low, mid, high"})
    assert errors(res) == ["ordinal_not_identifiable"] and res["estimates"] == []


# --- 02-09. Качество данных: регистр меток и нечисловые ячейки ----------------------------


def _issues(values: list[str], **kw) -> dict[str, dict]:
    from core.validation.types import profile_column

    return {i["code"]: i for i in profile_column("c", values, **kw).issues}


def test_02_09_labels_differing_only_by_case_are_named_not_merged() -> None:
    from core.validation.types import profile_column

    values = ["A", "a", "B", "A ", "Контроль", "контроль", "b", "C"]
    profile = profile_column("grp", values)
    issue = {i["code"]: i for i in profile.issues}["labels_differ_by_case"]
    assert issue["params"]["labels"] == "A / a; B / b; Контроль / контроль"
    assert "A / a" in issue["message"]
    assert profile.n_unique == 8  # счёт уникальных прежний: ядро ничего не объединяет


def test_02_09_no_case_note_when_labels_are_distinct() -> None:
    assert "labels_differ_by_case" not in _issues(["A", "B", "C", "A", "B", "C"])


def test_02_09_almost_numeric_column_names_the_guilty_cells() -> None:
    """Одна ячейка «<0.1» из 8 (87.5 % чисел) делает колонку категориальной."""
    values = ["1.2", "0.8", "<0.1", "2.4", "1.9", "", "3.1", "0.7", "1.1"]
    issue = _issues(values)["mixed_types"]
    assert issue["params"]["cells"] == "3: '<0.1'"
    assert "3: '<0.1'" in issue["message"]


def test_02_09_numeric_column_lists_rows_of_non_numeric_cells() -> None:
    values = [str(i) for i in range(1, 31)]
    values[6], values[20] = "3.l", "н/о"
    from core.validation.types import profile_column

    profile = profile_column("y", values)
    assert profile.detected_type.value == "numeric"
    assert profile.mismatch_cells == [{"row": 7, "value": "3.l"}, {"row": 21, "value": "н/о"}]
    assert profile.issues[0]["params"]["cells"] == "7: '3.l'; 21: 'н/о'"


def test_02_09_at_most_five_cells_are_listed() -> None:
    values = [str(i) for i in range(1, 13)] + ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    cells = _issues(values)["mixed_types"]["params"]["cells"]
    assert cells.count(":") == 5 and "x6" not in cells


def test_02_09_row_number_is_the_one_the_user_sees_in_excel() -> None:
    """Заголовок — строка 1, пустые строки считаются: «<0.1» стоит в строке 6 файла."""
    from core.validation.parser import parse_file

    raw = "y;g\n1,2;a\n\n0,8;a\n2,4;b\n<0,1;b\n1,9;a\n3,1;b\n0,7;a\n1,1;b\n".encode()
    profile = parse_file(raw, delimiter=";", decimal=",", has_header=True).profiles[0]
    assert profile.mismatch_cells == [{"row": 6, "value": "<0,1"}]
    assert profile.issues[0]["params"]["cells"] == "6: '<0,1'"


def test_02_09_without_header_first_row_is_row_one() -> None:
    from core.validation.parser import parse_file

    raw = "1.2,a\n0.8,a\n2.4,b\nn/d,b\n1.9,a\n3.1,b\n0.7,a\n1.1,b\n".encode()
    profile = parse_file(raw, delimiter=",", decimal=".", has_header=False).profiles[0]
    assert profile.mismatch_cells == [{"row": 4, "value": "n/d"}]


def test_02_09_messages_have_three_languages() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "web/src/coreMessages.tsx").read_text("utf-8")
    block = text.split("labels_differ_by_case: {")[1].split("\n  },")[0]
    assert all(f"{lang}: `" in block for lang in ("ru", "en", "pt"))
    assert text.count("{cells}") == 6


def _partial(n_a: int, n_b: int, covariate: bool = False, seed: int | None = None) -> dict:
    """Частичное разделение: уровень «a» — только «low», у «b» все три категории поровну."""
    third = n_b // 3
    cols = {"g": ["a"] * n_a + ["b"] * n_b,
            "s": ["low"] * (n_a + third) + ["mid"] * third + ["high"] * (n_b - 2 * third)}
    roles = {"response": "s", "factor_a": "g"}
    if covariate:
        cols["x"] = [float(i % 7) for i in range(n_a + n_b)]
        roles["covariate"] = "x"
    if seed is not None:
        order = np.random.default_rng(seed).permutation(n_a + n_b)
        cols = {k: [v[i] for i in order] for k, v in cols.items()}
    return run("ordinal_regression", cols, roles, {"levels": "low, mid, high"})


@pytest.mark.parametrize(("n_a", "n_b", "covariate"), [(8, 12, False), (16, 24, False),
                                                       (30, 30, True)])
@pytest.mark.parametrize("seed", [None, 1, 2, 3])
def test_02_12_partial_separation_is_a_refusal_whatever_the_row_order(n_a, n_b, covariate,
                                                                     seed) -> None:
    """Было (патч 117): на одних и тех же данных то отказ, то таблица с β ≈ 27 и e^27 —
    по порядку строк и по тому, обратит ли BLAS машины почти вырожденный гессиан.
    Замер на этих 12 вариантах: CPython Linux — 4 отказа из 8 исходных, Pyodide — 3 из 8,
    на Windows у владельца 20 строк дали таблицу. Теперь всегда отказ."""
    res = _partial(n_a, n_b, covariate, seed)
    assert errors(res) == ["ordinal_not_identifiable"] and res["estimates"] == []
    assert "ordinal_not_converged" not in codes(res)


def test_02_12_each_separation_sign_alone_means_separated() -> None:
    """Три признака без данных, не завися от BLAS: log L ≈ 0; гессиан не обратился
    (cov = None — раньше здесь падал statsmodels: «need covariance of parameters»);
    |β| > 10 при SE > 100. Обычная подгонка — не разделение."""
    from core.stats.categorical import _separated

    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    fine = np.array([[0.25]])
    assert _separated(-1e-9, fine, np.array([0.5]), x)
    assert _separated(-20.0, None, np.array([0.5]), x)
    assert _separated(-20.0, np.array([[1e10]]), np.array([27.0]), x)
    assert not _separated(-20.0, fine, np.array([0.5]), x)


def test_02_12_refusal_text_in_three_languages() -> None:
    res = _partial(8, 12)
    (note,) = [w for w in res["warnings"] if w["code"] == "ordinal_not_identifiable"]
    assert note["message"].startswith("Параметры модели не оцениваются: вероятно, есть пустая")
    assert "Объедините соседние категории или уберите фактор" in note["message"]
    web = Path(__file__).resolve().parents[1] / "web/src/coreMessages.tsx"
    block = web.read_text("utf-8").split("ordinal_not_identifiable: {")[1].split("\n  },")[0]
    assert note["message"] in block and all(f"{lang}: `" in block for lang in ("en", "pt"))


def test_02_12_runaway_rule_flags_huge_beta_with_huge_se() -> None:
    """[BioStat] |β| > 10 при SE > 100 на размах столбца; большой β с малой SE — не признак."""
    from core.stats.categorical import _runaway

    dummy = np.array([[0.0], [1.0], [0.0], [1.0]])
    assert _runaway(np.array([44.8]), np.array([[89267.0 ** 2]]), dummy)
    assert not _runaway(np.array([12.0]), np.array([[2.0 ** 2]]), dummy)
    assert not _runaway(np.array([3.0]), np.array([[400.0 ** 2]]), dummy)
    micro = dummy * 1e-6  # ковариата в других единицах: β = 2e6 — это 2 логита на размах
    assert not _runaway(np.array([2e6]), np.array([[5e5 ** 2]]), micro)


def test_02_12_runaway_sign_does_not_depend_on_covariate_units() -> None:
    """Те же данные с ковариатой в единицах ×1e-6: наклон в миллион раз больше, но это
    не разделение — признак считается на размах столбца."""
    from tests.bridge_runner import table

    cols = table("ordinal_regression")
    numeric = [k for k, v in cols.items() if isinstance(v[0], float)]
    small = {k: ([x * 1e-6 for x in v] if k in numeric else v) for k, v in cols.items()}
    res = run("ordinal_regression", small)
    assert numeric and not ({"ordinal_not_converged", "ordinal_not_identifiable"} & set(codes(res)))
