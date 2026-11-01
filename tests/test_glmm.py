"""GLMM (экспериментальный движок): пять условий приёмки ``glmm-acceptance-criteria.md``.

1. Квадратура с одним узлом = Лаплас.
2. Два внешних эталона: cbpp (биномиальный, logit) и grouseticks (Пуассон, log,
   три скрещённых случайных фактора) — числа lme4 и других пакетов.
3. Предел mixed → fixed: при σ² → 0 GLMM = обычная GLM (``glm_analysis``).
4. Маркер ``glmm_experimental`` (Severity.WARNING) в каждом результате.
5. Расхождения с эталонами — с величиной в докстрингах.

Данные — из R-пакета lme4 через Rdatasets (vincentarelbundock/Rdatasets,
csv/lme4/cbpp.csv и grouseticks.csv), только нужные колонки: tests/data/lme4_*.csv.
В книге (стр. 512–513) GLMM лишь упомянуты, численного примера нет.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from core.report.schema import Severity
from core.stats.glm import glm_analysis
from core.stats.glmm import EXPERIMENTAL, boundary_p, glmm_analysis
from core.stats.glmm_engine import GLMMData, agq, laplace

DATA = Path(__file__).parent / "data"


# --- Данные ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbpp() -> pd.DataFrame:
    return pd.read_csv(DATA / "lme4_cbpp.csv")


@pytest.fixture(scope="module")
def ticks() -> pd.DataFrame:
    return pd.read_csv(DATA / "lme4_grouseticks.csv")


def _indicators(codes: pd.Series) -> np.ndarray:
    return np.column_stack([(codes == lev).to_numpy(float) for lev in sorted(codes.unique())])


def _cbpp_data(c: pd.DataFrame) -> GLMMData:
    x = np.column_stack([np.ones(len(c))] + [(c.period == k).to_numpy(float) for k in (2, 3, 4)])
    return GLMMData(c.incidence.to_numpy(float), c["size"].to_numpy(float), x,
                    _indicators(c.herd), [np.arange(15)], "binomial", "logit")


def _glm_beta(d: GLMMData) -> np.ndarray:
    endog = np.column_stack([d.y, d.m - d.y]) if d.family == "binomial" else d.y
    family = sm.families.Binomial() if d.family == "binomial" else sm.families.Poisson()
    return np.asarray(sm.GLM(endog, d.x, family=family).fit(tol=1e-12).params)


def _cbpp_glmm(c: pd.DataFrame, **kw):
    return glmm_analysis(c.incidence.tolist(), c.herd.astype(str).tolist(),
                         factor_a=c.period.astype(str).tolist(), trials=c["size"].tolist(), **kw)


# --- 1. Один узел квадратуры = Лаплас ---------------------------------------------


@pytest.mark.parametrize("theta, shift", [(0.64, 0.0), (0.2, 0.3), (1.5, -0.4)])
def test_agq_one_node_equals_laplace(cbpp, theta, shift):
    """Условие 1: AGQ с 1 узлом ≡ Лаплас в произвольной точке (θ, β), до 1e-9."""
    d = _cbpp_data(cbpp)
    beta = _glm_beta(d) + shift
    lap = laplace(d, np.array([theta]), beta)["m2ll"]
    assert agq(d, np.array([theta]), beta, 1)["m2ll"] == pytest.approx(lap, abs=1e-9)


def test_agq_one_node_fit_equals_laplace_fit(cbpp):
    """Условие 1 на уровне метода: method=agq, n_agq=1 даёт ту же подгонку, что laplace."""
    lap, one = _cbpp_glmm(cbpp), _cbpp_glmm(cbpp, method="agq", n_agq=1)
    assert one.statistics["m2ll"] == pytest.approx(lap.statistics["m2ll"], abs=1e-8)
    assert [e.value for e in one.estimates] == pytest.approx([e.value for e in lap.estimates],
                                                             abs=1e-6)


def test_agq_converges_in_nodes(cbpp):
    """Квадратура сходится по числу узлов: 15 и 25 узлов совпадают до 1e-7."""
    d = _cbpp_data(cbpp)
    theta, beta = np.array([0.65]), _glm_beta(d)
    assert agq(d, theta, beta, 15)["m2ll"] == pytest.approx(agq(d, theta, beta, 25)["m2ll"],
                                                            abs=1e-7)


# --- 2а. Эталон cbpp (биномиальный) -------------------------------------------------


def test_cbpp_laplace_matches_three_packages(cbpp):
    """cbpp, Лаплас: incidence/size ~ period + (1 | herd).

    Эталоны — lme4 tests/glmer-1.R (lme4 2.0-6-1): coef_m1_glmmML (glmmML), coef_m1_glmmADMB,
    coef_m1_lme4.0; lme4 сам сверяет их с допуском 5e-4. Наше расхождение: с glmmML ≤ 6e-6,
    с glmmADMB ≤ 2e-5, с lme4.0 ≤ 7e-5 (lme4.0 останавливался раньше на пологой поверхности).
    σ²_herd = 0.41250 (summary lme4: 0.4123 — расхождение 5e-4, lme4 печатает оптимум
    BOBYQA с допуском; −2ℓ в нашем оптимуме 184.0525637 меньше, чем в округлённой точке lme4).
    """
    res = _cbpp_glmm(cbpp)
    beta = [e.value for e in res.estimates]
    glmm_ml = [-1.39853234657711, -0.992336901732793, -1.12867036466201, -1.58030977686564]
    admb = [-1.39853810064827, -0.99233330126975, -1.12867317840779, -1.58031150854503]
    lme40 = [-1.39853505102576, -0.992334712470269, -1.12867541092127, -1.58037389566025]
    assert beta == pytest.approx(glmm_ml, abs=1e-5)
    assert beta == pytest.approx(admb, abs=3e-5)
    assert beta == pytest.approx(lme40, abs=1e-4)
    assert res.parameters["variance_components"][0]["variance"] == pytest.approx(0.4123, abs=1e-3)
    assert res.statistics["loglik"] == pytest.approx(-92.0263, abs=1e-3)  # lme4: logLik −92.0
    assert res.statistics["aic"] == pytest.approx(194.1, abs=0.05)  # lme4: AIC 194.1


def test_cbpp_standard_errors_vs_lme4(cbpp):
    """SE фиксированных эффектов против summary(gm1) lme4: 0.2312, 0.3032, 0.3228, 0.4220.

    Наши SE (гессиан −2ℓ по (θ, β), как use.hessian) 0.2325, 0.3066, 0.3266, 0.4274 —
    больше на 0.6–1.3 % (разные численные гессианы: у lme4 — deriv12 в точке останова
    BOBYQA). Допуск 1.5 %; z-тесты и выводы о значимости совпадают (все P < 0.002).
    """
    res = _cbpp_glmm(cbpp)
    se = [r["se"] for r in res.parameters["parameter_tests"]]
    assert se == pytest.approx([0.2312, 0.3032, 0.3228, 0.4220], rel=0.015)
    assert all(r["p"] < 0.002 for r in res.parameters["parameter_tests"])


def test_cbpp_agq25_matches_lme4(cbpp):
    """cbpp, nAGQ = 25: lme4 tests/glmer-1.R, модель m2.

    fixef(m2) = −1.39922533, −0.99140729, −1.12782185, −1.57946627 (допуск lme4 5e-4); наше
    расхождение ≤ 6e-6. −2·logLik(m2) = 100.010030538022: lme4 в квадратуре берёт Σ отклонений
    d_i вместо −2Σ log f, т.е. без константы насыщенной модели (83.9567075 для cbpp); наше
    −2ℓ минус эта константа совпадает до 2e-9.
    """
    res = _cbpp_glmm(cbpp, method="agq", n_agq=25)
    beta = [e.value for e in res.estimates]
    assert beta == pytest.approx([-1.39922533406847, -0.991407294757321, -1.12782184600404,
                                  -1.57946627431248], abs=2e-5)
    y, m = cbpp.incidence.to_numpy(float), cbpp["size"].to_numpy(float)
    saturated = -2 * stats.binom.logpmf(y, m, y / m).sum()
    assert res.statistics["m2ll"] - saturated == pytest.approx(100.010030538022, abs=1e-6)


def test_cbpp_variance_lr_test(cbpp):
    """LR-тест σ²_herd = 0 против GLM: χ² = −2ℓ_GLM − (−2ℓ_GLMM), P = ½·P(χ²₁ ≥ LR) (граница)."""
    res = _cbpp_glmm(cbpp)
    d = _cbpp_data(cbpp)
    glm_m2ll = -2 * stats.binom.logpmf(d.y, d.m, sm.GLM(
        np.column_stack([d.y, d.m - d.y]), d.x, family=sm.families.Binomial()).fit()
        .fittedvalues).sum()
    row = res.parameters["variance_components"][0]
    assert row["lr"] == pytest.approx(glm_m2ll - res.statistics["m2ll"], abs=1e-6)
    assert row["p"] == pytest.approx(0.5 * stats.chi2.sf(row["lr"], 1))
    assert row["levels"] == 15


# --- 2б. Эталон grouseticks (Пуассон, три скрещённых фактора) ------------------------


_ELSTON_BETA = [11.3559080756861, 1.1804105508475, -0.978704335712111, -0.0237607330254979]
_ELSTON_VAR = [0.293232458048324, 0.562551624933584, 0.279548178949372]  # INDEX, BROOD, LOCATION


def _ticks_data(g: pd.DataFrame) -> GLMMData:
    x = np.column_stack([np.ones(len(g)), (g.YEAR == 96).to_numpy(float),
                         (g.YEAR == 97).to_numpy(float), g.HEIGHT.to_numpy(float)])
    zs = [_indicators(g[c]) for c in ("INDEX", "BROOD", "LOCATION")]
    sizes = np.cumsum([0] + [z.shape[1] for z in zs])
    groups = [np.arange(a, b) for a, b in pairwise(sizes)]
    return GLMMData(g.TICKS.to_numpy(float), np.ones(len(g)), x, np.hstack(zs), groups,
                    "poisson", "log")


def test_grouseticks_deviance_function_at_lme4_optimum(ticks):
    """lme4 tests/elston.R: TICKS ~ YEAR + HEIGHT + (1|BROOD) + (1|INDEX) + (1|LOCATION).

    devfun lme4 в напечатанной точке (θ = SD, β) = 1780.5427072; наше −2ℓ по Лапласу —
    1780.54266, расхождение 4e-5 (lme4 печатает параметры с конечным числом знаков).
    """
    d = _ticks_data(ticks)
    m2ll = laplace(d, np.sqrt(_ELSTON_VAR), np.array(_ELSTON_BETA))["m2ll"]
    assert m2ll == pytest.approx(1780.5427072, abs=1e-3)


def test_grouseticks_fit_matches_lme4(ticks):
    """Подгонка grouseticks против lme4 (elston.R): β до 2e-5 отн., σ² до 1e-4.

    Наши β: 11.35589, 1.180413, −0.978695, −0.0237606; σ²: 0.29323, 0.56255, 0.27955.
    Расхождение с lme4: |Δβ| ≤ 2e-5, |Δσ²| ≤ 3e-6; −2ℓ наше 1780.54266 ≤ lme4 1780.54271.
    """
    res = glmm_analysis(ticks.TICKS.tolist(), ticks.INDEX.astype(str).tolist(),
                        factor_a=ticks.YEAR.astype(str).tolist(), covariate=ticks.HEIGHT.tolist(),
                        stratum_2=ticks.BROOD.astype(str).tolist(),
                        stratum_3=ticks.LOCATION.astype(str).tolist(),
                        distribution="poisson", interactions="no")
    est = {e.name: e.value for e in res.estimates}  # порядок членов: ковариата, затем фактор
    ours = [est["[1]"], est["A 96"], est["A 97"], est["x"]]
    assert ours == pytest.approx(_ELSTON_BETA, rel=2e-5)
    var = [r["variance"] for r in res.parameters["variance_components"]]
    assert var == pytest.approx(_ELSTON_VAR, abs=1e-4)
    assert res.statistics["m2ll"] <= 1780.5427072 + 1e-6
    assert [r["levels"] for r in res.parameters["variance_components"]] == [403, 118, 63]


# --- 3. Предел mixed → fixed ---------------------------------------------------------


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_zero_variance_likelihood_is_glm(cbpp, family):
    """При θ = 0 Лаплас и квадратура дают ровно −2ℓ GLM (statsmodels) при её β̂."""
    d = _cbpp_data(cbpp)
    if family == "poisson":
        d = GLMMData(d.y, np.ones_like(d.y), d.x, d.z, d.groups, "poisson", "log")
    beta = _glm_beta(d)
    fitted = 1 / (1 + np.exp(-d.x @ beta)) if family == "binomial" else np.exp(d.x @ beta)
    ll = (stats.binom.logpmf(d.y, d.m, fitted) if family == "binomial"
          else stats.poisson.logpmf(d.y, fitted)).sum()
    for value in (laplace(d, np.zeros(1), beta)["m2ll"], agq(d, np.zeros(1), beta, 7)["m2ll"]):
        assert value == pytest.approx(-2 * ll, abs=1e-8)


def test_profiled_beta_tends_to_glm(cbpp):
    """β̂(θ) — максимум по β при фиксированной θ — монотонно стремится к β̂_GLM при θ → 0."""
    d = _cbpp_data(cbpp)
    glm_beta = _glm_beta(d)
    gaps = []
    for theta in (0.3, 0.1, 0.03, 0.01, 0.001):
        res = optimize.minimize(lambda b, t=theta: laplace(d, np.array([t]), b)["m2ll"],
                                glm_beta, method="BFGS", options={"gtol": 1e-9})
        gaps.append(np.max(np.abs(res.x - glm_beta)))
    assert all(a > b for a, b in pairwise(gaps))
    assert gaps[-1] < 1e-4


def _replicated(distribution: str) -> pd.DataFrame:
    """Одинаковые данные в каждой из 6 групп: оценка σ² = 0 точно (сумма скоров групп = 0)."""
    base = pd.DataFrame({"trt": list("ABCD") * 3, "x": np.repeat([1.0, 2.0, 3.0], 4),
                         "y": [3, 5, 8, 6, 2, 7, 9, 4, 4, 6, 10, 7]})
    frame = pd.concat([base.assign(group=f"g{k}") for k in range(6)], ignore_index=True)
    frame["m"] = 12
    return frame


@pytest.mark.parametrize("distribution", ["binomial", "poisson"])
def test_zero_variance_glmm_equals_glm_analysis(distribution):
    """Условие 3 сквозным путём: σ̂² = 0 ⇒ оценки, SE, предсказания GLMM = glm_analysis.

    Допуски: β 1e-5, SE 1e-3 отн. (SE по численному гессиану против точной информации GLM).
    LR-тест σ² = 0 даёт χ² = 0 и P = 1, есть сообщение о границе.
    """
    f = _replicated(distribution)
    trials = f["m"].tolist() if distribution == "binomial" else None
    common = {"factor_a": f["trt"].tolist(), "covariate": f["x"].tolist(), "trials": trials,
              "distribution": distribution, "interactions": "no"}
    mixed = glmm_analysis(f["y"].tolist(), f["group"].tolist(), **common)
    fixed = glm_analysis(f["y"].tolist(), dispersion_method="fixed", **common)
    assert mixed.parameters["variance_components"][0]["variance"] < 1e-8
    assert [e.value for e in mixed.estimates] == pytest.approx([e.value for e in fixed.estimates],
                                                               abs=1e-5)
    assert [e.std_error for e in mixed.estimates] == pytest.approx(
        [e.std_error for e in fixed.estimates], rel=1e-3)
    nat = [r["natural"] for r in mixed.parameters["factor_predictions"]]
    assert nat == pytest.approx([r["natural"] for r in fixed.parameters["factor_predictions"]],
                                abs=1e-6)
    assert mixed.parameters["variance_components"][0]["p"] == 1.0
    assert "glmm_boundary" in {w.code for w in mixed.warnings}


# --- 4. Маркер «экспериментальный движок» -------------------------------------------


def test_experimental_warning_always_present(cbpp):
    """Условие 4: WARNING glmm_experimental с текстом из критериев — в любом результате."""
    ok = _cbpp_glmm(cbpp)
    failed = glmm_analysis([1, 2, 3], ["a", "a", "a"], trials=[5, 5, 5])  # один уровень
    for res in (ok, failed):
        marker = [w for w in res.warnings if w.code == "glmm_experimental"]
        assert len(marker) == 1 and marker[0].severity == Severity.WARNING
        assert marker[0].message == EXPERIMENTAL
    assert "R/GenStat" in EXPERIMENTAL and "Лаплас/квадратура" in EXPERIMENTAL
    assert any(w.severity == Severity.ERROR for w in failed.warnings)


# --- Прочие свойства ------------------------------------------------------------------


def test_boundary_mixture():
    """Смесь χ̄²: один компонент — P/2; два — ¼χ²₀ + ½χ²₁ + ¼χ²₂; LR = 0 → P = 1."""
    assert boundary_p(3.0, 1) == pytest.approx(0.5 * stats.chi2.sf(3.0, 1))
    assert boundary_p(3.0, 2) == pytest.approx(0.5 * stats.chi2.sf(3.0, 1)
                                               + 0.25 * stats.chi2.sf(3.0, 2))
    assert boundary_p(0.0, 3) == 1.0


def test_nested_structure_and_conditional_modes():
    """Вложенные факторы: члены ферма и ферма.поле; моды b по уровням со средним ≈ 0."""
    rng = np.random.default_rng(3)
    farm = np.repeat([f"F{k}" for k in range(8)], 12)
    field = np.tile(np.repeat(["a", "b", "c"], 4), 8)
    b_farm = np.repeat(rng.normal(0, 0.7, 8), 12)
    trt = np.tile(["T1", "T2", "T3", "T4"], 24)
    eta = -0.5 + b_farm + np.where(trt == "T4", 0.8, 0.0)
    y = rng.binomial(20, 1 / (1 + np.exp(-eta)))
    res = glmm_analysis(y.tolist(), farm.tolist(), factor_a=trt.tolist(),
                        stratum_2=field.tolist(), trials=[20] * 96, random_structure="nested",
                        column_names={"stratum_1": "Farm", "stratum_2": "Field",
                                      "factor_a": "Trt"})
    assert res.parameters["random_terms"] == ["Farm", "Farm.Field"]
    assert [r["levels"] for r in res.parameters["variance_components"]] == [8, 24]
    modes = [r for r in res.parameters["conditional_modes"] if r["term"] == "Farm"]
    assert len(modes) == 8 and abs(np.mean([r["b"] for r in modes])) < 0.2
    assert res.parameters["variance_components"][0]["variance"] > 0.1


def test_agq_falls_back_for_several_factors(ticks):
    """Квадратура с несколькими случайными факторами недоступна: Лаплас и предупреждение."""
    small = ticks.iloc[:120]
    res = glmm_analysis(small.TICKS.tolist(), small.BROOD.astype(str).tolist(),
                        stratum_2=small.LOCATION.astype(str).tolist(), covariate=small.HEIGHT
                        .tolist(), distribution="poisson", method="agq", n_agq=9)
    assert "glmm_agq_single_factor" in {w.code for w in res.warnings}
    assert "n_agq" not in res.parameters


def test_overdispersion_flagged_without_observation_effect(ticks):
    """grouseticks без фактора INDEX (наблюдение): χ² Пирсона/df > 1 и предупреждение."""
    res = glmm_analysis(ticks.TICKS.tolist(), ticks.BROOD.astype(str).tolist(),
                        factor_a=ticks.YEAR.astype(str).tolist(), covariate=ticks.HEIGHT.tolist(),
                        distribution="poisson", interactions="no")
    assert res.parameters["overdispersion"]["ratio"] > 1.5
    assert "glmm_overdispersion" in {w.code for w in res.warnings}
