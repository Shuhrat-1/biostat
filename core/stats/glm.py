"""Обобщённые линейные модели для долей и счётов (гл. 18 книги).

``glm_analysis`` — биномиальная GLM (число успехов из m испытаний; связь logit,
probit или cloglog) и пуассоновская GLM (счёты; связь log). Объясняющие члены:
структурный фактор (блок) первым — внутриблочный анализ (стр. 491, 510), до
двух факторов и ковариата (с преобразованием, например ln(доза + 0.1)).

- ANODEV (Tables 18.2–18.4): модель, остаток, общее отклонение; при оценённой
  дисперсии — отношения отклонений с F, иначе χ².
- Сверхдисперсия: остаточное отклонение против χ²(ResDF); φ̂ = ResDev/ResDF или
  по Пирсону (стр. 489); φ̂ < 1 не используется (стр. 493); для бинарных данных
  φ не оценивается (стр. 501).
- Последовательные ANODEV в двух порядках факторов (Table 18.7): в GLM
  ортогональные в линейной модели члены неортогональны.
- Параметры с SE·√φ (z или t на ResDF), предсказания на шкале предиктора с ДИ
  и обратным преобразованием, средние «до» и «после» обратного преобразования
  (Example 18.2E), отношения (log) и отношения шансов (logit) уровней
  (Example 18.3C), стандартизованные остатки отклонения (стр. 495).

Оценивание — statsmodels GLM (IRLS, максимальное правдоподобие).
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import Terms, _frame, _label
from core.stats.factorial import _new_result
from core.stats.linear_regression import parse_numbers
from core.stats.mixed import _column_label, fixed_design
from core.stats.regression_checks import _transform_column

LINKS = {"binomial": ("logit", "probit", "cloglog"), "poisson": ("log",)}
_DISPERSIONS = ("auto", "fixed", "deviance", "pearson")
_YES_NO = ("yes", "no")
_GRID = 101
_SLOTS = {"blk": "block", "A": "factor_a", "B": "factor_b", "x": "covariate"}
_ALL_FACTORS = ("blk", "A", "B", "lof")
_SELECTIONS = ("none", "backward")


# --- Семейство и подгонка ------------------------------------------------------


def _family(distribution: str, link: str) -> sm.families.Family:
    """Распределение и функция связи statsmodels."""
    links = {"logit": sm.families.links.Logit, "probit": sm.families.links.Probit,
             "cloglog": sm.families.links.CLogLog, "log": sm.families.links.Log}
    family = sm.families.Binomial if distribution == "binomial" else sm.families.Poisson
    return family(link=links[link]())


def glm_fit(frame: pd.DataFrame, x: np.ndarray, family: sm.families.Family) -> dict:
    """МП-оценки, ковариация при φ = 1, отклонение, χ² Пирсона, η и μ."""
    y = frame["y"].to_numpy(float)
    binomial = "m" in frame
    endog = np.column_stack([y, frame["m"] - y]) if binomial else y
    offset = frame["off"].to_numpy(float) if "off" in frame else None
    res = sm.GLM(endog, x, family=family, offset=offset).fit(tol=1e-10, maxiter=200)
    scale = frame["m"].to_numpy(float) if binomial else np.ones_like(y)
    eta = x @ np.asarray(res.params) + (0.0 if offset is None else offset)
    return {"res": res, "beta": np.asarray(res.params), "cov1": np.asarray(res.cov_params()),
            "deviance": float(res.deviance), "pearson": float(res.pearson_chi2),
            "df": len(y) - x.shape[1], "x": x, "eta": eta,
            "mu": np.asarray(res.fittedvalues) * scale, "converged": bool(res.converged)}


class Model:
    """Члены модели: построитель плана, подгонка и отклонение по набору членов."""

    def __init__(self, frame: pd.DataFrame, factors: list[str], variates: list[str],
                 family: sm.families.Family) -> None:
        self.terms = Terms(frame, factors, variates)
        self.frame, self.family = frame, family
        self._cache: dict[tuple, dict] = {}
        self.lof_x = (frame.groupby("lof")["x"].first().to_dict() if "lof" in frame else {})

    def fit(self, terms: list[tuple[str, ...]]) -> dict:
        """Подгонка [1] + члены без вырожденных столбцов (кэш по набору членов)."""
        key = tuple(terms)
        if key not in self._cache:
            x, info = fixed_design(self.terms, terms)
            self._cache[key] = {**glm_fit(self.frame, x, self.family), "info": info,
                                "terms": list(terms)}
        return self._cache[key]


# --- Дисперсия и ANODEV ---------------------------------------------------------


def dispersion(fit: dict, mode: str, binary: bool, alpha: float) -> dict:
    """Проверка сверхдисперсии и выбор φ (стр. 488–489, 492–493, 501)."""
    df = fit["df"]
    est = {"deviance": fit["deviance"] / df, "pearson": fit["pearson"] / df} if df > 0 else {}
    p_over = float(stats.chi2.sf(fit["deviance"], df)) if df > 0 else float("nan")
    if binary or mode == "fixed" or not est:
        phi, source = 1.0, "fixed"
    elif mode == "auto":
        phi, source = (est["deviance"], "deviance") if p_over < alpha else (1.0, "fixed")
    else:
        phi, source = est[mode], mode
    under = phi < 1
    if under:
        phi, source = 1.0, "fixed"
    return {"phi": phi, "estimated": source != "fixed", "source": source, "p_overdispersion":
            p_over, "deviance_estimate": est.get("deviance"), "pearson_estimate":
            est.get("pearson"), "underdispersion": under}


def _test(dev: float, df: int, disp: dict, res_df: int) -> dict:
    """Изменение отклонения: χ²(df) при φ = 1, иначе отношение отклонений с F(df, ResDF)."""
    row = {"df": df, "deviance": dev, "mean_deviance": dev / df}
    if disp["estimated"]:
        f = dev / df / disp["phi"]
        row.update({"F": f, "p": float(stats.f.sf(f, df, res_df))})
    else:
        row["p"] = float(stats.chi2.sf(dev, df))
    return row


def anodev(model: Model, terms: list[tuple[str, ...]], disp: dict) -> list[dict]:
    """Модель, остаток и общее отклонение (Tables 18.3, 18.6, 18.11)."""
    null, full = model.fit([]), model.fit(terms)
    mod_df = null["df"] - full["df"]
    rows = [{"term": "Model", **_test(null["deviance"] - full["deviance"], mod_df, disp,
                                      full["df"])}] if mod_df > 0 else []
    rows.append({"term": "Residual", "df": full["df"], "deviance": full["deviance"],
                 "mean_deviance": full["deviance"] / full["df"] if full["df"] else float("nan")})
    rows.append({"term": "Total", "df": null["df"], "deviance": null["deviance"],
                 "mean_deviance": null["deviance"] / null["df"]})
    return rows


def sequential(model: Model, terms: list[tuple[str, ...]], disp: dict,
               names: dict[str, str]) -> list[dict]:
    """Инкрементные отклонения в заданном порядке против φ полной модели (Table 18.7)."""
    res_df = model.fit(terms)["df"]
    rows, prev = [], model.fit([])
    for k, term in enumerate(terms):
        cur = model.fit(terms[:k + 1])
        df = prev["df"] - cur["df"]
        if df > 0:
            rows.append({"term": "+ " + _label(term, names),
                         **_test(prev["deviance"] - cur["deviance"], df, disp, res_df)})
        prev = cur
    return rows


# --- Параметры и предсказания ---------------------------------------------------


def _scaled_cov(fit: dict, disp: dict) -> np.ndarray:
    """Ковариация оценок с множителем φ (стр. 496)."""
    return fit["cov1"] * disp["phi"]


def _critical(disp: dict, df: int, alpha: float) -> tuple[float, object]:
    """Квантиль и распределение: Normal при φ = 1, t(ResDF) при оценённой φ (стр. 497)."""
    dist = stats.t(df) if disp["estimated"] else stats.norm()
    return float(dist.ppf(1 - alpha / 2)), dist


def parameter_rows(model: Model, fit: dict, disp: dict, names: dict[str, str],
                   alpha: float) -> list[dict]:
    """Оценки first-level-zero с SE, z или t, P и ДИ (Table 18.8)."""
    info, cov = fit["info"], _scaled_cov(fit, disp)
    crit, dist = _critical(disp, fit["df"], alpha)
    labels = ["[1]"] + [_column_label(model.terms, info["owner"][j], j, info, names)
                        for j in info["keep"][1:]]
    rows = []
    for lab, b, v in zip(labels, fit["beta"], np.diag(cov)):
        se = float(np.sqrt(v))
        stat = b / se
        rows.append({"term": lab, "estimate": float(b), "se": se, "statistic": float(stat),
                     "p": float(2 * dist.sf(abs(stat))), "ci_lower": float(b - crit * se),
                     "ci_upper": float(b + crit * se)})
    return rows


def design_row(model: Model, fit: dict, setting: dict[str, object]) -> np.ndarray:
    """Строка плана для уровней факторов и значения ковариаты (столбцы модели)."""
    one = pd.DataFrame({k: [v] for k, v in setting.items()})
    probe = Terms.__new__(Terms)
    probe.frame, probe.levels, probe.variates = one, model.terms.levels, model.terms.variates
    return probe.design(fit["terms"])[0][fit["info"]["keep"]]


def _factors_in(fit: dict, candidates: tuple[str, ...]) -> list[str]:
    """Факторы из candidates, входящие в члены модели."""
    return [f for f in candidates if any(f in t for t in fit["terms"])]


def averaged_rows(model: Model, fit: dict, fixed: dict[str, object]) -> list[np.ndarray]:
    """Строки плана для всех уровней прочих факторов модели (ковариата — среднее)."""
    others = [f for f in _factors_in(fit, _ALL_FACTORS) if f not in fixed]
    base = dict(fixed)
    if "x" in model.terms.variates and "x" not in base:
        base["x"] = float(model.frame["x"].mean())
    rows = []
    for combo in product(*(model.terms.levels[f] for f in others)):
        setting = {**base, **dict(zip(others, combo))}
        if "lof" in setting:  # уровень фактора неадекватности задаёт и значение ковариаты
            setting["x"] = model.lof_x[setting["lof"]]
        rows.append(design_row(model, fit, setting))
    return rows


def _natural(link: sm.families.links.Link, eta: float, grad: np.ndarray, cov: np.ndarray) -> dict:
    """Обратное преобразование и приближённая SE дельта-методом."""
    value = float(link.inverse(eta))
    g = float(link.inverse_deriv(eta)) * grad
    return {"natural": value, "natural_se": float(np.sqrt(g @ cov @ g))}


def prediction(model: Model, fit: dict, disp: dict, rows: list[np.ndarray], alpha: float) -> dict:
    """Среднее по строкам на шкале предиктора с ДИ и оба варианта обратного преобразования."""
    cov, link = _scaled_cov(fit, disp), model.family.link
    crit, _ = _critical(disp, fit["df"], alpha)
    c = np.mean(rows, axis=0)
    eta, se = float(c @ fit["beta"]), float(np.sqrt(c @ cov @ c))
    etas = np.array([r @ fit["beta"] for r in rows])
    grad = np.mean([float(link.inverse_deriv(e)) * r for e, r in zip(etas, rows)], axis=0)
    out = {"eta": eta, "se": se, "lower": eta - crit * se, "upper": eta + crit * se,
           **_natural(link, eta, c, cov),
           "natural_lower": float(link.inverse(eta - crit * se)),
           "natural_upper": float(link.inverse(eta + crit * se)),
           "mean_after": float(np.mean(link.inverse(etas))),
           "mean_after_se": float(np.sqrt(grad @ cov @ grad))}
    lo, hi = sorted((out["natural_lower"], out["natural_upper"]))
    out.update({"natural_lower": lo, "natural_upper": hi})
    return out


def factor_predictions(model: Model, fit: dict, disp: dict, names: dict[str, str],
                       alpha: float) -> list[dict]:
    """Уровни каждого фактора обработок (и их комбинации), усреднённые по блокам (Example 18.2E)."""
    factors = _factors_in(fit, ("A", "B", "lof"))
    sets = [(f,) for f in factors] + ([("A", "B")] if ("A", "B") in fit["terms"] else [])
    out = []
    for group in sets:
        for combo in product(*(model.terms.levels[f] for f in group)):
            rows = averaged_rows(model, fit, dict(zip(group, combo)))
            out.append({"factor": ".".join(names[f] for f in group), "level": " × ".join(combo),
                        "row": np.mean(rows, axis=0), **prediction(model, fit, disp, rows, alpha)})
    return out


def comparisons(preds: list[dict], fit: dict, disp: dict, link: str, alpha: float) -> list[dict]:
    """Разности уровней на шкале предиктора с ДИ; exp — отношение (log) или шансов (logit)."""
    cov = _scaled_cov(fit, disp)
    crit, dist = _critical(disp, fit["df"], alpha)
    out = []
    for a, b in combinations(preds, 2):
        if a["factor"] != b["factor"]:
            continue
        d = b["row"] - a["row"]
        diff, se = float(d @ fit["beta"]), float(np.sqrt(d @ cov @ d))
        row = {"factor": a["factor"], "pair": f"{b['level']} − {a['level']}", "difference": diff,
               "se": se, "statistic": diff / se, "p": float(2 * dist.sf(abs(diff / se))),
               "lower": diff - crit * se, "upper": diff + crit * se}
        if link in ("log", "logit"):
            row.update({"ratio_kind": "ratio" if link == "log" else "odds_ratio",
                        "ratio": float(np.exp(diff)), "ratio_lower": float(np.exp(row["lower"])),
                        "ratio_upper": float(np.exp(row["upper"]))})
        out.append(row)
    return out


def cell_predictions(model: Model, fit: dict) -> list[dict]:
    """Предсказания для всех комбинаций уровней факторов модели (Table 18.9)."""
    factors = _factors_in(fit, _ALL_FACTORS)
    if not factors:
        return []
    out = []
    for combo in product(*(model.terms.levels[f] for f in factors)):
        rows = averaged_rows(model, fit, dict(zip(factors, combo)))
        eta = float(rows[0] @ fit["beta"])
        out.append({**dict(zip(factors, combo)), "eta": eta,
                    "natural": float(model.family.link.inverse(eta))})
    return out


def covariate_predictions(model: Model, fit: dict, disp: dict, points: list[float],
                          transform: tuple[str, float], alpha: float) -> list[dict]:
    """Предсказания при заданных значениях ковариаты (исходная шкала), среднее по факторам."""
    out = []
    for x0 in points:
        w = _transform_value(x0, transform)
        rows = averaged_rows(model, fit, {"x": w})
        out.append({"x": float(x0), "x_model": w, **prediction(model, fit, disp, rows, alpha)})
    return out


def _transform_value(x0: float, transform: tuple[str, float]) -> float:
    """Ковариата на шкале модели: ln, log10 или √ от x + c."""
    kind, c = transform
    funcs = {"none": lambda v: v, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt}
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(funcs[kind](x0 + c)) if kind != "none" else float(x0)


def _curve(model: Model, fit: dict, disp: dict, raw_x: np.ndarray,
           transform: tuple[str, float], alpha: float) -> dict:
    """Кривая на естественной шкале с ДИ по сетке исходной ковариаты (Figs 18.2–18.3)."""
    grid = np.linspace(raw_x.min(), raw_x.max(), _GRID)
    rows = covariate_predictions(model, fit, disp, grid.tolist(), transform, alpha)
    return {"x": grid.tolist(), "fitted": [r["natural"] for r in rows],
            "lower": [r["natural_lower"] for r in rows],
            "upper": [r["natural_upper"] for r in rows], "eta": [r["eta"] for r in rows]}


def residual_plots(fit: dict, disp: dict) -> dict:
    """Стандартизованные остатки отклонения r_D/√(φ(1 − h)) (стр. 495, Figs 18.4, 18.6)."""
    res = fit["res"]
    h = np.asarray(res.get_influence().hat_matrix_diag)
    r = np.asarray(res.resid_deviance) / np.sqrt(disp["phi"] * np.clip(1 - h, 1e-12, None))
    n = r.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return {"residual_vs_fitted": {"fitted": fit["mu"].tolist(), "residuals": r.tolist()},
            "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
            "std_residuals": r.tolist(), "pearson_residuals": np.asarray(res.resid_pearson).tolist(),
            "leverage": h.tolist()}


# --- Подготовка данных ----------------------------------------------------------


def _collect(values: list[float], slots: dict[str, list | None], trials: list | None,
             n_trials: int) -> tuple[pd.DataFrame, int]:
    """Полные строки: успехи/счёты, число испытаний, ковариата и факторы."""
    numeric = {k: v for k, v in (("x", slots["x"]), ("m", trials), ("e", slots.get("e")))
               if v is not None}
    factors = {k: slots[k] for k in ("blk", "A", "B")}
    frame, dropped = _frame(values, numeric, factors)
    if "x" in frame:
        frame["x_raw"] = frame["x"]
    if trials is None and n_trials > 0:
        frame["m"] = float(n_trials)
    if "e" in frame and (frame["e"] > 0).all():
        frame["off"] = np.log(frame["e"])  # смещение log(экспозиция), гл. 18.3 и Exercise 18.10
    return frame, dropped


def _check_exposure(frame: pd.DataFrame, distribution: str, result: StatResult) -> bool:
    """Экспозиция (время, площадь, объём) — только для пуассоновской модели и только > 0."""
    if "e" not in frame:
        return True
    if distribution != "poisson" or "off" not in frame:
        result.add_warning("glm_bad_exposure", "Экспозиция (смещение log e) — только для "
                           "пуассоновской модели и только положительная.", Severity.ERROR)
        return False
    result.parameters["offset"] = "log(exposure)"
    return True


def _check_response(frame: pd.DataFrame, distribution: str, result: StatResult) -> bool:
    """Целые неотрицательные счёты; для биномиального 0 ≤ y ≤ m, m ≥ 1."""
    y = frame["y"].to_numpy(float)
    if distribution == "binomial" and "m" not in frame:
        result.add_warning("glm_no_trials", "Для биномиальной GLM нужна колонка числа испытаний "
                           "или постоянное число испытаний m.", Severity.ERROR)
        return False
    bad = np.any(y < 0) or (distribution == "binomial" and (
        np.any(frame["m"] < 1) or np.any(y > frame["m"])))
    if bad:
        result.add_warning("glm_bad_response", "Отклик должен быть ≥ 0 (и ≤ m для биномиальной).",
                           Severity.ERROR)
        return False
    if np.any(y != np.round(y)):
        result.add_warning("glm_non_integer", "Отклик не целый: GLM рассчитана на счёты.",
                           Severity.WARNING)
    return True


def _check_terms(frame: pd.DataFrame, factors: list[str], variates: list[str],
                 result: StatResult) -> bool:
    """Хотя бы один объясняющий член; у факторов ≥ 2 уровней; ковариата варьирует."""
    if not [f for f in factors if f != "blk"] and not variates:
        result.add_warning("glm_no_terms", "Нужен фактор обработок или ковариата.", Severity.ERROR)
        return False
    if any(frame[f].nunique() < 2 for f in factors) or (variates and frame["x"].nunique() < 2):
        result.add_warning("factor_one_level", "У фактора один уровень или ковариата постоянна.",
                           Severity.ERROR)
        return False
    return True


def model_terms(factors: list[str], variates: list[str], interactions: bool,
                reverse: bool = False) -> list[tuple[str, ...]]:
    """Блок, ковариата, фактор неадекватности, факторы (A, B или B, A), A.B, ковариата ×
    факторы и неадекватность × факторы (стр. 491, 510)."""
    treat = [f for f in factors if f not in ("blk", "lof")]
    treat = treat[::-1] if reverse else treat
    lof = ["lof"] if "lof" in factors else []
    terms = [("blk",)] if "blk" in factors else []
    terms += [(v,) for v in [*variates, *lof]] + [(f,) for f in treat]
    if interactions:
        terms += [tuple(treat)] if len(treat) == 2 else []
        terms += [(v, f) for v in [*variates, *lof] for f in treat]
    return terms


def contains(small: tuple[str, ...], big: tuple[str, ...]) -> bool:
    """small маргинален big; фактор неадекватности включает ковариату (стр. 510, Table 18.13)."""
    def expand(t: tuple[str, ...]) -> set[str]:
        return set(t) | ({"x"} if "lof" in t else set())
    return small != big and expand(small) <= expand(big)


# --- Метод каталога -------------------------------------------------------------


def glm_analysis(
    values: list[float],
    factor_a: list[str] | None = None,
    factor_b: list[str] | None = None,
    covariate: list | None = None,
    block: list[str] | None = None,
    trials: list | None = None,
    exposure: list | None = None,
    distribution: str = "binomial",
    link: str = "canonical",
    n_trials: int = 0,
    covariate_transform: str = "none",
    covariate_offset: float = 0.0,
    interactions: str = "yes",
    lack_of_fit: str = "no",
    selection: str = "none",
    dispersion_method: str = "auto",
    x_predict: str = "",
    dose_for_p: str = "",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Биномиальная или пуассоновская GLM с ANODEV, сверхдисперсией и предсказаниями (гл. 18)."""
    link = _check_options(distribution, link, dispersion_method, interactions, alpha)
    if lack_of_fit not in _YES_NO or selection not in _SELECTIONS:
        raise ValueError(f"lack_of_fit ∈ {_YES_NO}, selection ∈ {_SELECTIONS}")
    slots = {"blk": block, "A": factor_a, "B": factor_b, "x": covariate, "e": exposure}
    frame, dropped = _collect(values, slots, trials, int(n_trials))
    factors = [k for k in ("blk", "A", "B") if slots[k] is not None]
    variates = ["x"] if covariate is not None else []
    names = _names(column_names, factors, variates)
    if variates and lack_of_fit == "yes":
        frame["lof"] = [f"{v:g}" for v in frame["x_raw"]]
        factors.append("lof")
    result = _new_result("glm_analysis", frame, {"alpha": alpha, "transform_kind": "none",
                                                 "offset": 0.0},
                         distribution=distribution, link=link, column_names=names,
                         dispersion_method=dispersion_method, selection=selection)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    ok = (_check_exposure(frame, distribution, result)
          and _check_response(frame, distribution, result)
          and _check_terms(frame, factors, variates, result)
          and (not variates or _transform_column(frame, "x", covariate_transform,
                                                 covariate_offset, result)))
    if ok:
        model = Model(frame, factors, variates, _family(distribution, link))
        spec = {"factors": factors, "variates": variates, "interactions": interactions == "yes",
                "transform": (covariate_transform, covariate_offset), "x_predict": x_predict,
                "raw_x": _raw_covariate(covariate, frame), "dose_for_p": dose_for_p}
        _analyse(model, spec, names, result)
    return result


def _names(column_names: dict[str, str] | None, factors: list[str],
           variates: list[str]) -> dict[str, str]:
    """Имена колонок пользователя; фактор неадекватности — «Fac» + имя ковариаты."""
    cn = column_names or {}
    names = {k: cn.get(_SLOTS[k], k) for k in (*factors, *variates)}
    if variates:
        names["lof"] = "Fac" + names["x"]
    return names


def _check_options(distribution: str, link: str, dispersion_method: str, interactions: str,
                   alpha: float) -> str:
    """Проверка опций; canonical → logit (биномиальное) или log (пуассоновское)."""
    if distribution not in LINKS or dispersion_method not in _DISPERSIONS or \
            interactions not in _YES_NO or not 0 < alpha < 1:
        raise ValueError(f"distribution ∈ {tuple(LINKS)}, dispersion_method ∈ {_DISPERSIONS}")
    link = LINKS[distribution][0] if link == "canonical" else link
    if link not in LINKS[distribution]:
        raise ValueError(f"link для {distribution}: {LINKS[distribution]}")
    return link


def _raw_covariate(covariate: list | None, frame: pd.DataFrame) -> np.ndarray | None:
    """Ковариата до преобразования (для кривой на исходной шкале)."""
    return None if covariate is None else frame["x_raw"].to_numpy(float)


# --- Анализ и отчёт ---------------------------------------------------------------


def _analyse(model: Model, spec: dict, names: dict[str, str], result: StatResult) -> None:
    """Полная модель: ANODEV, дисперсия, последовательные таблицы, затем отчёт."""
    alpha, mode = result.parameters["alpha"], result.parameters["dispersion_method"]
    terms = model_terms(spec["factors"], spec["variates"], spec["interactions"])
    full = model.fit(terms)
    if not full["converged"]:
        result.add_warning("glm_not_converged", "Итерации IRLS не сошлись: возможно полное "
                           "разделение (доли 0 или 1 в целой группе).", Severity.WARNING)
    binary = "m" in model.frame and bool((model.frame["m"] == 1).all())
    disp = dispersion(full, mode, binary, alpha)
    result.parameters.update({
        "terms": [_label(t, names) for t in terms], "anodev": anodev(model, terms, disp),
        "dispersion": disp, "sequential": sequential(model, terms, disp, names)})
    if len([f for f in spec["factors"] if f != "blk"]) == 2:
        rev = model_terms(spec["factors"], spec["variates"], spec["interactions"], reverse=True)
        result.parameters["sequential_reverse"] = sequential(model, rev, disp, names)
    _dispersion_warnings(disp, binary, mode, result)
    final, final_disp = full, disp
    if result.parameters["selection"] == "backward":
        chosen, steps = backward(model, terms, (mode, binary, alpha), names)
        final = model.fit(chosen)
        final_disp = dispersion(final, mode, binary, alpha)
        result.parameters.update({"selection_steps": steps, "final_terms":
                                  [_label(t, names) for t in chosen],
                                  "final_anodev": anodev(model, chosen, final_disp),
                                  "final_dispersion": final_disp})
    result.parameters["marginal"] = marginal_tests(model, final["terms"], final_disp, names)
    report(model, final, final_disp, spec, names, result)
    if spec["dose_for_p"].strip():
        result.parameters["doses"] = effective_doses(model, final, final_disp, spec, names,
                                                     result)


def _dispersion_warnings(disp: dict, binary: bool, mode: str, result: StatResult) -> None:
    """Сообщения о сверх- и недодисперсии и о бинарных данных."""
    p = disp["p_overdispersion"]
    if binary:
        result.add_warning("glm_binary", "Бинарные данные (m = 1): остаточное отклонение не "
                           "проверяет сверхдисперсию, остатки малоинформативны.")
    elif disp["estimated"]:
        result.add_warning("glm_overdispersion", f"Сверхдисперсия: φ = {disp['phi']:.3f}; тесты — "
                           "по F, SE умножены на √φ.", Severity.INFO,
                           params={"phi": round(disp["phi"], 3), "p": round(p, 4)})
    elif mode == "fixed" and p < result.parameters["alpha"]:
        result.add_warning("glm_overdispersion_ignored", "Остаточное отклонение велико "
                           f"(P = {p:.3g}), а φ зафиксирован = 1: тесты слишком оптимистичны.",
                           params={"p": f"{p:.3g}"})
    if disp["underdispersion"]:
        result.add_warning("glm_underdispersion", "φ̂ < 1 (недодисперсия): φ оставлен = 1.",
                           Severity.INFO)


def report(model: Model, fit: dict, disp: dict, spec: dict, names: dict[str, str],
           result: StatResult) -> None:
    """Параметры, предсказания, сравнения, кривая и остатки итоговой модели."""
    alpha = result.parameters["alpha"]
    params = parameter_rows(model, fit, disp, names, alpha)
    preds = factor_predictions(model, fit, disp, names, alpha)
    result.parameters.update({
        "parameter_tests": params, "cells": cell_predictions(model, fit),
        "comparisons": comparisons(preds, fit, disp, result.parameters["link"], alpha),
        "factor_predictions": [{k: v for k, v in p.items() if k != "row"} for p in preds],
        "plots": residual_plots(fit, disp)})
    if any("x" in t for t in fit["terms"]) and not any("lof" in t for t in fit["terms"]):
        points = parse_numbers(spec["x_predict"])
        result.parameters["predictions"] = covariate_predictions(
            model, fit, disp, points, spec["transform"], alpha)
        result.parameters["curve"] = _curve(model, fit, disp, spec["raw_x"], spec["transform"],
                                            alpha)
        y = model.frame["y"] / model.frame.get("m", 1.0)
        result.parameters["points"] = {"x": spec["raw_x"].tolist(), "y": y.tolist()}
    _statistics(model, fit, disp, result)
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))


def _statistics(model: Model, fit: dict, disp: dict, result: StatResult) -> None:
    """Плоские статистики: отклонения, φ, тест модели."""
    null = model.fit([])
    test = _test(null["deviance"] - fit["deviance"], null["df"] - fit["df"], disp, fit["df"])
    result.p_value = test["p"]
    result.statistics.update({
        "deviance": fit["deviance"], "df_resid": fit["df"], "null_deviance": null["deviance"],
        "pearson_chi2": fit["pearson"], "phi": disp["phi"],
        "p_overdispersion": disp["p_overdispersion"], "model_deviance": test["deviance"],
        "model_df": test["df"], "p_model": test["p"], **({"F_model": test["F"]} if "F" in test
                                                        else {})})


# --- Отбор членов -----------------------------------------------------------------


def eligible(terms: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Члены, доступные для проверки: не блок и не подчлен другого члена (Table 18.13)."""
    return [t for t in terms if t != ("blk",) and not any(contains(t, o) for o in terms)]


def marginal_tests(model: Model, terms: list[tuple[str, ...]], disp: dict,
                   names: dict[str, str]) -> list[dict]:
    """Изменение отклонения при исключении каждого допустимого члена (стр. 493)."""
    fit, rows = model.fit(terms), []
    for t in eligible(terms):
        reduced = model.fit([o for o in terms if o != t])
        df = reduced["df"] - fit["df"]
        if df > 0:
            rows.append({"term": "- " + _label(t, names), "key": t,
                         **_test(reduced["deviance"] - fit["deviance"], df, disp, fit["df"])})
    return rows


def backward(model: Model, terms: list[tuple[str, ...]], options: tuple[str, bool, float],
             names: dict[str, str]) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Обратный отбор по маргинальным тестам с переоценкой φ в каждой модели (Example 18.4)."""
    mode, binary, alpha = options
    current, steps = list(terms), []
    while True:
        fit = model.fit(current)
        disp = dispersion(fit, mode, binary, alpha)
        tests = marginal_tests(model, current, disp, names)
        if not tests:
            break
        worst = max(tests, key=lambda r: r["p"])
        drop = worst["p"] > alpha
        steps.append({"step": len(steps) + 1, "terms": [_label(t, names) for t in current],
                      "phi": disp["phi"], "df_resid": fit["df"], "deviance": fit["deviance"],
                      "tests": [{k: v for k, v in r.items() if k != "key"} for r in tests],
                      "dropped": worst["term"][2:] if drop else None})
        if not drop:
            break
        current.remove(worst["key"])
    return current, steps


# --- Доза для заданной вероятности (LD50) --------------------------------------------


def fieller_ratio(num: float, den: float, v_nn: float, v_dd: float, v_nd: float,
                  crit: float) -> tuple[float, float] | None:
    """Пределы Филлера для num/den; None — интервал неограничен (g ≥ 1)."""
    g = crit ** 2 * v_dd / den ** 2
    if g >= 1:
        return None
    rho = num / den
    centre = rho - g * v_nd / v_dd
    half = crit / abs(den) * np.sqrt(max(v_nn - 2 * rho * v_nd + rho ** 2 * v_dd
                                         - g * (v_nn - v_nd ** 2 / v_dd), 0.0))
    return (centre - half) / (1 - g), (centre + half) / (1 - g)


def _inverse_transform(w: float, transform: tuple[str, float]) -> float:
    """Ковариата с шкалы модели обратно на исходную."""
    kind, c = transform
    inverse = {"none": lambda v: v, "ln": np.exp, "log10": lambda v: 10 ** v,
               "sqrt": lambda v: v ** 2 if v >= 0 else float("nan")}
    value = inverse[kind](w)
    return float(value - c) if kind != "none" else float(value)


def _dose_rows(model: Model, fit: dict, disp: dict, rows0: list[np.ndarray],
               rows1: list[np.ndarray], p: float, spec: dict, crit: float) -> dict:
    """η(w) = a + b·w для группы; w₀ = (g(p) − a)/b с пределами Филлера (стр. 502)."""
    cov = _scaled_cov(fit, disp)
    r0, r1 = np.mean(rows0, axis=0), np.mean(rows1, axis=0)
    d = r1 - r0
    a, b = float(r0 @ fit["beta"]), float(d @ fit["beta"])
    target = float(model.family.link(np.array([p]))[0])
    limits = fieller_ratio(target - a, b, float(r0 @ cov @ r0), float(d @ cov @ d),
                           -float(r0 @ cov @ d), crit)
    w0 = (target - a) / b
    out = {"p": p, "dose_model": w0, "dose": _inverse_transform(w0, spec["transform"]),
           "intercept": a, "slope": b}
    if limits:
        lo, hi = sorted(_inverse_transform(v, spec["transform"]) for v in limits)
        out.update({"model_lower": limits[0], "model_upper": limits[1], "lower": lo, "upper": hi})
    return out


def effective_doses(model: Model, fit: dict, disp: dict, spec: dict, names: dict[str, str],
                    result: StatResult) -> list[dict]:
    """Доза для вероятностей p по группам факторов обработок (усреднение по блокам)."""
    terms = fit["terms"]
    if result.parameters["distribution"] != "binomial" or ("x",) not in terms or \
            any("lof" in t for t in terms):
        result.add_warning("glm_dose_unavailable", "Доза для заданной вероятности считается для "
                           "биномиальной модели с ковариатой без фактора неадекватности.")
        return []
    crit, _ = _critical(disp, fit["df"], result.parameters["alpha"])
    groups = _factors_in(fit, ("A", "B"))
    out = []
    for combo in product(*(model.terms.levels[f] for f in groups)):
        fixed = dict(zip(groups, combo))
        rows0 = averaged_rows(model, fit, {**fixed, "x": 0.0})
        rows1 = averaged_rows(model, fit, {**fixed, "x": 1.0})
        for p in parse_numbers(spec["dose_for_p"]):
            if not 0 < p < 1:
                raise ValueError("Вероятности для дозы — в (0, 1)")
            row = _dose_rows(model, fit, disp, rows0, rows1, p, spec, crit)
            out.append({"group": " × ".join(combo) or "all", **row})
    if any(("x", f) in terms for f in groups):
        result.add_warning("glm_dose_nonparallel", "Наклоны по ковариате различаются между "
                           "группами: одна доза не описывает различия (стр. 502).")
    return out
