"""Гл. 18.3.4 и 18.4: смещение, таблицы сопряжённости, порядковая регрессия, NB и ZIP.

В книге для этих моделей нет численных примеров (стр. 512–513); проверки — по
определениям, по независимым реализациям (GLM statsmodels, ручные формулы) и по
восстановлению параметров на имитированных данных.
"""

from __future__ import annotations

import warnings
from math import comb

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from core.stats.categorical import (
    adjusted_residuals,
    contingency_test,
    ordinal_regression,
)
from core.stats.counts import count_regression
from core.stats.glm import glm_analysis
from tests.book_ch18_data import APHIDS

warnings.filterwarnings("ignore")

_Y = [r[2] for r in APHIDS]
_FIELD = [str(r[0]) for r in APHIDS]


# --- Смещение log(экспозиция) ------------------------------------------------------


def test_offset_gives_rate_per_unit_exposure():
    """Пуассон с log(e): для одного фактора МП-интенсивность = Σy/Σe в группе."""
    e = [1 + (i % 4) for i in range(45)]
    res = glm_analysis(_Y, factor_a=_FIELD, exposure=[str(v) for v in e], distribution="poisson")
    rates = [p["natural"] for p in res.parameters["factor_predictions"]]
    expected = [sum(y for y, f in zip(_Y, _FIELD) if f == g) /
                sum(v for v, f in zip(e, _FIELD) if f == g) for g in "123"]
    assert rates == pytest.approx(expected)
    assert res.parameters["offset"] == "log(exposure)"


def test_offset_scaling_shifts_intercept_only():
    """Экспозиция × k сдвигает интерсепт на −ln k, остальные параметры не меняются."""
    e = [1 + (i % 3) for i in range(45)]
    a = glm_analysis(_Y, factor_a=_FIELD, exposure=[str(v) for v in e], distribution="poisson")
    b = glm_analysis(_Y, factor_a=_FIELD, exposure=[str(10 * v) for v in e],
                     distribution="poisson")
    pa = [p["estimate"] for p in a.parameters["parameter_tests"]]
    pb = [p["estimate"] for p in b.parameters["parameter_tests"]]
    assert pb[0] == pytest.approx(pa[0] - np.log(10))
    assert pb[1:] == pytest.approx(pa[1:])
    bad = glm_analysis([1, 2, 3, 4], covariate=["1", "2", "3", "4"], n_trials=5,
                       exposure=["1", "1", "1", "1"])
    assert any(w.code == "glm_bad_exposure" for w in bad.warnings)


# --- Таблицы сопряжённости ---------------------------------------------------------


def _rows_from_counts(table: dict[tuple[str, str], int]) -> tuple[list[str], list[str]]:
    a, b = [], []
    for (r, c), n in table.items():
        a += [r] * n
        b += [c] * n
    return a, b


_TAB = {("A", "x"): 20, ("A", "y"): 15, ("A", "z"): 5,
        ("B", "x"): 10, ("B", "y"): 18, ("B", "z"): 22}


def test_chi2_matches_definition_and_counts_input():
    """χ² = Σ(O − E)²/E, E = r·c/n; данные по строкам ≡ колонка частот."""
    a, b = _rows_from_counts(_TAB)
    res = contingency_test(a, b)
    obs = np.array([[20, 15, 5], [10, 18, 22]], float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    assert res.statistics["chi2"] == pytest.approx(((obs - exp) ** 2 / exp).sum())
    assert res.statistics["df"] == 2
    keys = list(_TAB)
    weighted = contingency_test([k[0] for k in keys], [k[1] for k in keys],
                                count=[str(v) for v in _TAB.values()])
    assert weighted.statistics["chi2"] == pytest.approx(res.statistics["chi2"])
    cells = {(c["row"], c["column"]): c for c in res.parameters["cells"]}
    assert cells["A", "x"]["adjusted_residual"] == pytest.approx(
        adjusted_residuals(obs, exp)[0, 0])


def test_g2_equals_log_linear_independence_deviance():
    """G² = отклонение лог-линейной модели независимости A + B (стр. 513)."""
    keys = list(_TAB)
    res = contingency_test([k[0] for k in keys], [k[1] for k in keys],
                           count=[str(v) for v in _TAB.values()])
    glm = glm_analysis(list(_TAB.values()), factor_a=[k[0] for k in keys],
                       factor_b=[k[1] for k in keys], distribution="poisson", interactions="no",
                       dispersion_method="fixed")
    assert res.statistics["g2"] == pytest.approx(glm.statistics["deviance"])


def test_two_by_two_fisher_and_yates():
    """2 × 2: Йейтс по умолчанию, точный P Фишера = сумма гипергеометрических вероятностей."""
    res = contingency_test(["a", "a", "b", "b"], ["x", "y", "x", "y"], count=["8", "2", "1", "5"])
    p_obs = comb(10, 8) * comb(6, 1) / comb(16, 9)
    probs = [comb(10, k) * comb(6, 9 - k) / comb(16, 9) for k in range(3, 10)]
    assert res.statistics["fisher_p"] == pytest.approx(sum(p for p in probs if p <= p_obs * 1.0000001))
    assert res.statistics["yates"]
    assert any(w.code == "contingency_small_expected" for w in res.warnings)
    no_yates = contingency_test(["a", "a", "b", "b"], ["x", "y", "x", "y"],
                                count=["8", "2", "1", "5"], correction="no")
    assert no_yates.statistics["chi2"] > res.statistics["chi2"]


def test_cramers_v_for_perfect_association():
    """Полная связь: V Крамера = 1; одна классификация с одним уровнем — ошибка."""
    res = contingency_test(list("aabbcc"), list("xxyyzz"))
    assert res.statistics["cramers_v"] == pytest.approx(1.0)
    one = contingency_test(list("aaaa"), list("xyxy"))
    assert any(w.code == "contingency_too_small" for w in one.warnings)


# --- Порядковая регрессия ----------------------------------------------------------


def _ordinal_data(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    g = rng.choice(["c", "t"], n)
    x = rng.normal(size=n)
    latent = 0.8 * (g == "t") + 0.5 * x + rng.logistic(size=n)
    cats = np.array(["healthy", "slight", "moderate", "severe"])[np.digitize(latent, [-1, 0.5, 2])]
    return cats.tolist(), g.tolist(), [str(v) for v in x]


def _ordinal(**kw):
    y, g, x = _ordinal_data()
    return ordinal_regression(y, factor_a=g, covariate=x, levels="healthy, slight, moderate, severe",
                              interactions="no", **kw)


def test_ordinal_recovers_parameters():
    """Латентная логистическая модель: β = (0.5, 0.8), пороги (−1, 0.5, 2) восстанавливаются."""
    res = _ordinal()
    coefs = {c["term"]: c["estimate"] for c in res.parameters["coefficients"]}
    assert (coefs["x"], coefs["A t"]) == pytest.approx((0.5, 0.8), abs=0.12)
    assert [t["estimate"] for t in res.parameters["thresholds"]] == pytest.approx(
        [-1, 0.5, 2], abs=0.15)


def test_ordinal_loglik_matches_definition():
    """log L = Σ log[F(θ_y − xβ) − F(θ_(y−1) − xβ)] по возвращённым θ и β; LR модели > 0."""
    y, g, x = _ordinal_data()
    res = _ordinal()
    order = ["healthy", "slight", "moderate", "severe"]
    coefs = {c["term"]: c["estimate"] for c in res.parameters["coefficients"]}
    theta = np.r_[-np.inf, [t["estimate"] for t in res.parameters["thresholds"]], np.inf]
    eta = coefs["x"] * np.array(x, float) + coefs["A t"] * (np.array(g) == "t")
    k = np.array([order.index(v) for v in y])
    ll = np.log(stats.logistic.cdf(theta[k + 1] - eta) - stats.logistic.cdf(theta[k] - eta)).sum()
    assert res.statistics["loglik"] == pytest.approx(ll, abs=1e-4)
    assert res.statistics["lr_model"] > 0 and res.statistics["df_model"] == 2


def test_ordinal_reversed_order_flips_signs_and_probabilities_sum():
    """Обратный порядок категорий меняет знак β; вероятности категорий в сумме 1."""
    y, g, x = _ordinal_data()
    rev = ordinal_regression(y, factor_a=g, covariate=x, levels="severe, moderate, slight, healthy",
                             interactions="no")
    fwd = _ordinal()
    assert [c["estimate"] for c in rev.parameters["coefficients"]] == pytest.approx(
        [-c["estimate"] for c in fwd.parameters["coefficients"]], abs=1e-3)
    for row in fwd.parameters["category_probabilities"]:
        assert sum(row["probabilities"].values()) == pytest.approx(1.0)
    assert len(fwd.parameters["proportionality"]) == 3


def test_ordinal_guards():
    """Две категории — это биномиальная GLM; неизвестная категория — ошибка."""
    res = ordinal_regression(["a", "b", "a", "b"], factor_a=["x", "x", "y", "y"])
    assert any(w.code == "ordinal_two_categories" for w in res.warnings)
    with pytest.raises(ValueError):
        ordinal_regression(["a", "b", "c"], factor_a=["x", "y", "x"], levels="a, b")


# --- Отрицательная биномиальная и нулевое раздувание --------------------------------


def test_negbin_factor_model_reproduces_field_means():
    """NB2 только с фактором: МП-средние групп = выборочные средние (тли, Table 18.10, стр. 504)."""
    res = count_regression(_Y, factor_a=_FIELD)
    means = [p["natural"] for p in res.parameters["factor_predictions"]]
    assert means == pytest.approx([4.6, 15.4, 20 / 3], abs=1e-4)
    assert res.statistics["p_vs_poisson"] < 0.001
    assert res.parameters["nb_alpha"]["estimate"] > 0


def test_negbin_matches_glm_with_fixed_alpha():
    """β NB2 (ML по всем параметрам) = β GLM NegativeBinomial(α = α̂) (IRLS) — две реализации."""
    rng = np.random.default_rng(5)
    n = 800
    g = rng.choice(["a", "b"], n)
    x = rng.normal(size=n)
    mu = np.exp(1 + 0.5 * (g == "b") + 0.3 * x)
    y = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * mu))
    res = count_regression(y.tolist(), factor_a=g.tolist(), covariate=[str(v) for v in x],
                           interactions="no")
    alpha = res.parameters["nb_alpha"]["estimate"]
    design = np.column_stack([np.ones(n), x, (g == "b").astype(float)])
    ref = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha)).fit(tol=1e-12)
    assert [p["estimate"] for p in res.parameters["parameter_tests"]] == pytest.approx(
        ref.params, abs=1e-4)
    assert alpha == pytest.approx(0.6, abs=0.12)


def test_poisson_option_equals_glm():
    """model = poisson совпадает с glm_analysis (оценки и SE при φ = 1)."""
    res = count_regression(_Y, factor_a=_FIELD, model="poisson")
    glm = glm_analysis(_Y, factor_a=_FIELD, distribution="poisson", dispersion_method="fixed")
    for a, b in zip(res.parameters["parameter_tests"], glm.parameters["parameter_tests"]):
        assert (a["estimate"], a["se"]) == pytest.approx((b["estimate"], b["se"]), rel=1e-5)


def test_zip_recovers_inflation_and_zero_counts():
    """ZIP: доля структурных нулей π = 0.3 восстанавливается, ожидаемые нули ≈ наблюдённым."""
    rng = np.random.default_rng(7)
    n = 2000
    g = rng.choice(["a", "b"], n)
    mu = np.exp(1 + 0.5 * (g == "b"))
    y = np.where(rng.random(n) < 0.3, 0, rng.poisson(mu))
    res = count_regression(y.tolist(), factor_a=g.tolist(), model="zip")
    zi = res.parameters["zero_inflation"]
    assert zi["pi"] == pytest.approx(0.3, abs=0.04) and zi["lower"] < 0.3 < zi["upper"]
    assert res.statistics["expected_zeros"] == pytest.approx(res.statistics["observed_zeros"],
                                                             rel=0.02)
    assert [p["estimate"] for p in res.parameters["parameter_tests"]] == pytest.approx(
        [1.0, 0.5], abs=0.08)


def test_zinb_without_inflation_warns():
    """Тли: у ZINB π → 0 — предупреждение «нулевого раздувания нет»."""
    res = count_regression(_Y, factor_a=_FIELD, model="zinb")
    assert any(w.code == "count_no_zero_inflation" for w in res.warnings)
