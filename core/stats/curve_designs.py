"""Кривые в спланированных опытах: блоки, группы, неадекватность (гл. 17 + 15.3).

Упражнения 17.1, 17.4, 17.6 требуют кривой отклика на дозу в RCBD: блок —
структурный член, подгоняемый первым (внутриблочный анализ, разд. 15.3),
неадекватность — фактор уровней дозы после кривой (разд. 12.8, Table 15.13).

- ``curve_in_blocks`` — кривая, линейная по параметрам: полином порядка q или
  преобразование x (log, √, 1/x …). Последовательная ANOVA [1] + Block + степени
  + [Group + степени.Group] + FacX(.Group), предсказания для среднего блока с ДИ.
- ``nonlinear_in_blocks`` — нелинейная кривая (экспоненциальная, обратная
  линейная, логистическая, Гомперца, степенная) с блоками как сдвигами уровня α
  и вложенными моделями для групп: общая кривая → отдельные α → отдельные α, β →
  все параметры отдельные (как гл. 15 для прямых), F-тесты против остатка самой
  общей модели и неадекватность против остатка Block + FacX. Оценивание —
  вариационная проекция: при фиксированных θ линейные параметры по МНК.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import Terms, _frame, _label, parameter_rows, sequential_rows
from core.stats.curves import X_TRANSFORMS, x_transform
from core.stats.factorial import _new_result, _settings
from core.stats.linear_regression import parse_numbers
from core.stats.nonlinear import CURVES, _start
from core.stats.regression_checks import _transform_column

_CURVE_KINDS = ("polynomial", "transform")
GROUP_MODELS = ("common", "alpha", "alpha_beta", "all")
_GRID = 101


# --- Данные -------------------------------------------------------------------


def _collect(values: list[float], x: list, block: list[str] | None,
             group: list[str] | None) -> tuple[pd.DataFrame, int]:
    """Полные строки: отклик, x, блок и группа; FacX — уровни x (и группы)."""
    frame, dropped = _frame(values, {"x": x}, {"block": block, "group": group})
    frame["facx"] = frame["x"].map(lambda v: f"{v:g}")
    return frame, dropped


def _names(column_names: dict[str, str] | None) -> dict[str, str]:
    """Подписи членов по именам колонок пользователя."""
    cn = column_names or {}
    x = cn.get("x", "x")
    return {"block": cn.get("block", "Block"), "group": cn.get("group", "Group"), "x": x,
            "facx": "Fac" + x}


def _guard(frame: pd.DataFrame, n_params: int, result: StatResult) -> bool:
    """Достаточно точек и уровней x; блоки и группы — ≥ 2 уровней."""
    if frame["x"].nunique() < 3 or len(frame) <= n_params + 1:
        result.add_warning("curve_design_too_small", "Нужно ≥ 3 уровней x и наблюдений больше, "
                           "чем параметров.", Severity.ERROR)
        return False
    if any(c in frame and frame[c].nunique() < 2 for c in ("block", "group")):
        result.add_warning("factor_one_level", "У блока или группы один уровень.", Severity.ERROR)
        return False
    return True


def _start_result(method: str, frame: pd.DataFrame, dropped: int, settings: dict,
                  names: dict[str, str], **extra: object) -> StatResult:
    """Результат с настройками и сообщением о пропусках."""
    result = _new_result(method, frame, settings, column_names=names, **extra)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    return result


# --- curve_in_blocks -------------------------------------------------------------


def _power_columns(frame: pd.DataFrame, curve: str, degree: int, kind: str,
                   x_offset: float) -> list[str]:
    """Столбцы w1 … wq: степени центрированного x или f(x + c)."""
    if curve == "transform":
        frame["w1"] = x_transform(frame["x"].to_numpy(float), kind, x_offset)
        return ["w1"]
    centre = float(frame["x"].mean())
    for k in range(1, degree + 1):
        frame[f"w{k}"] = (frame["x"] - centre) ** k
    frame.attrs["centre"] = centre
    return [f"w{k}" for k in range(1, degree + 1)]


def _linear_terms(ws: list[str], has_block: bool, has_group: bool) -> list[tuple[str, ...]]:
    """Block, степени, Group, степени.Group, FacX, FacX.Group."""
    terms = [("block",)] if has_block else []
    terms += [(w,) for w in ws]
    if has_group:
        terms += [("group",)] + [(w, "group") for w in ws]
    terms += [("facx",)] + ([("facx", "group")] if has_group else [])
    return terms


def _w_values(points: np.ndarray, frame: pd.DataFrame, curve: str, kind: str,
              x_offset: float, ws: list[str]) -> np.ndarray:
    """Столбцы w для новых x."""
    if curve == "transform":
        return x_transform(points, kind, x_offset)[:, None]
    centre = frame.attrs["centre"]
    return np.column_stack([(points - centre) ** (k + 1) for k in range(len(ws))])


def _average_rows(model: Terms, terms: list[tuple[str, ...]], w: np.ndarray,
                  group: str | None) -> np.ndarray:
    """Строки плана при w для группы, усреднённые по блокам (Example 15.3)."""
    blocks = model.levels.get("block", [None])
    rows = []
    for b in blocks:
        one = pd.DataFrame({f"w{k + 1}": w[:, k] for k in range(w.shape[1])})
        if b is not None:
            one["block"] = b
        if group is not None:
            one["group"] = group
        probe = Terms.__new__(Terms)
        probe.frame, probe.levels, probe.variates = one, model.levels, model.variates
        rows.append(probe.design(terms))
    return np.mean(rows, axis=0)


def _linear_predictions(model: Terms, terms: list[tuple[str, ...]], fit: dict, w: np.ndarray,
                        points: np.ndarray, group: str | None, alpha: float) -> list[dict]:
    """μ̂ для среднего блока с ДИ среднего и нового наблюдения."""
    rows = _average_rows(model, terms, w, group)
    cov = fit["s2"] * np.linalg.pinv(fit["design"].T @ fit["design"])
    mean = rows @ fit["beta"]
    se = np.sqrt(np.einsum("ij,jk,ik->i", rows, cov, rows))
    crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    return [{"x": float(points[i]), "group": group, "fitted": float(mean[i]),
             "se_mean": float(se[i]), "mean_lower": float(mean[i] - crit * se[i]),
             "mean_upper": float(mean[i] + crit * se[i]),
             "new_lower": float(mean[i] - crit * np.hypot(se[i], np.sqrt(fit["s2"]))),
             "new_upper": float(mean[i] + crit * np.hypot(se[i], np.sqrt(fit["s2"])))}
            for i in range(points.size)]


def _curve_terms(seq: list[dict], ws: list[str], has_block: bool, has_group: bool,
                 alpha: float) -> list[tuple[str, ...]]:
    """Модель предсказания: отдельные кривые, если степени.Group значимы; параллельные — если
    значима Group; иначе общая (стр. 396–399)."""
    base = ([("block",)] if has_block else []) + [(w,) for w in ws]
    if not has_group:
        return base
    by_term = {r["term"]: r for r in seq}
    inter = [by_term.get(f"+ {w}.group", {}).get("p", 1.0) for w in ws]
    if min(inter) < alpha:
        return base + [("group",)] + [(w, "group") for w in ws]
    return base + [("group",)] if by_term.get("+ group", {}).get("p", 1.0) < alpha else base


def curve_in_blocks(
    values: list[float],
    x: list,
    block: list[str] | None = None,
    group: list[str] | None = None,
    curve: str = "polynomial",
    degree: int = 2,
    x_transform_kind: str = "log10",
    x_offset: float = 0.0,
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Полином или преобразованный x в блоках и группах с неадекватностью (Exercises 17.1a, 17.4)."""
    degree = int(degree)
    if curve not in _CURVE_KINDS or not 1 <= degree <= 6 or x_transform_kind not in X_TRANSFORMS:
        raise ValueError(f"curve ∈ {_CURVE_KINDS}, degree ∈ [1, 6], x_transform_kind ∈ "
                         f"{X_TRANSFORMS}")
    settings = _settings(alpha, transform_kind, offset)
    frame, dropped = _collect(values, x, block, group)
    names = _names(column_names)
    result = _start_result("curve_in_blocks", frame, dropped, settings, names, curve=curve,
                           degree=degree)
    if not (_transform_column(frame, "y", transform_kind, offset, result)
            and _guard(frame, degree + 2, result)):
        return result
    try:
        ws = _power_columns(frame, curve, degree, x_transform_kind, x_offset)
    except ValueError as exc:
        result.add_warning("x_transform_domain", f"x: {exc}", Severity.ERROR)
        return result
    _fill_linear(frame, ws, (curve, x_transform_kind, x_offset), x_predict, names, result)
    return result


def _fill_linear(frame: pd.DataFrame, ws: list[str], spec: tuple[str, str, float],
                 x_predict: str, names: dict[str, str], result: StatResult) -> None:
    """Последовательная ANOVA, выбор модели, параметры, предсказания и кривые."""
    alpha, has_block, has_group = result.parameters["alpha"], "block" in frame, "group" in frame
    factors = [f for f in ("block", "group", "facx") if f in frame]
    model = Terms(frame, factors, ws)
    labels = {**names, **{w: (names["x"] + ("" if w == "w1" else "^" + w[1:])
                              if spec[0] == "polynomial" else f"{spec[1]}({names['x']})")
                          for w in ws}}
    seq = sequential_rows(model, _linear_terms(ws, has_block, has_group), labels)
    raw_seq = sequential_rows(model, _linear_terms(ws, has_block, has_group),
                              {w: w for w in [*ws, "group", "block", "facx"]})
    terms = _curve_terms(raw_seq, ws, has_block, has_group, alpha)
    params, fit = parameter_rows(model, terms, labels, alpha)
    result.parameters.update({"sequential": seq, "parameter_tests": params,
                              "model_terms": [_label(t, labels) for t in terms]})
    lof = next(r for r in seq if r["term"] == "+ " + names["facx"])
    result.statistics.update({"F_lack_of_fit": lof.get("F"), "p_lack_of_fit": lof.get("p"),
                              "ms_resid": seq[-2]["ms"], "df_resid": seq[-2]["df"]})
    if lof.get("p", 1.0) < alpha:
        result.add_warning("curve_lack_of_fit", "Средние по уровням x значимо отклоняются от "
                           "кривой (гл. 12.8).")
    _linear_curves(model, terms, fit, frame, spec, ws, x_predict, result)
    for row in params:
        result.estimates.append(Estimate(row["term"], row["estimate"], row["ci_lower"],
                                         row["ci_upper"], 1 - alpha, row["se"]))


def _linear_curves(model: Terms, terms: list[tuple[str, ...]], fit: dict, frame: pd.DataFrame,
                   spec: tuple[str, str, float], ws: list[str], x_predict: str,
                   result: StatResult) -> None:
    """Предсказания в точках пользователя и кривые на сетке по группам."""
    alpha = result.parameters["alpha"]
    groups = model.levels["group"] if any("group" in t for t in terms) else [None]
    points = np.asarray(parse_numbers(x_predict), float)
    grid = np.linspace(frame["x"].min(), frame["x"].max(), _GRID)
    preds, curves = [], []
    for g in groups:
        if points.size:
            w = _w_values(points, frame, spec[0], spec[1], spec[2], ws)
            preds += _linear_predictions(model, terms, fit, w, points, g, alpha)
        band = _linear_predictions(model, terms, fit,
                                   _w_values(grid, frame, spec[0], spec[1], spec[2], ws),
                                   grid, g, alpha)
        curves.append({"group": g, "x": grid.tolist(), "fitted": [r["fitted"] for r in band],
                       "lower": [r["mean_lower"] for r in band],
                       "upper": [r["mean_upper"] for r in band]})
    result.parameters.update({"predictions": preds, "curves": curves,
                              "points": {"x": frame["x"].tolist(), "y": frame["y"].tolist()}})


# --- nonlinear_in_blocks -----------------------------------------------------------


class NonlinearDesign:
    """Нелинейная кривая α + Block + β·g(x; θ) с вложенными моделями групп."""

    def __init__(self, frame: pd.DataFrame, curve_name: str) -> None:
        self.frame, self.curve = frame, CURVES[curve_name]
        self.x, self.y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
        self.blocks = _levels(frame, "block")
        self.groups = _levels(frame, "group")
        self.m = len(self.curve.names) - 2  # число нелинейных параметров θ

    def n_theta(self, kind: str) -> int:
        """θ общие для всех групп, кроме модели «all»."""
        return self.m * (len(self.groups) if kind == "all" else 1)

    def design(self, kind: str, theta: np.ndarray, x: np.ndarray | None = None,
               block: list | None = None, group: list | None = None) -> np.ndarray:
        """Линейная часть при фиксированных θ: [1, блоки, группы (α), g или g·группа (β)]."""
        x = self.x if x is None else x
        block = self.frame["block"].tolist() if block is None and self.blocks else block
        group = self.frame["group"].tolist() if group is None and self.groups else group
        cols = [np.ones_like(x)]
        cols += [np.array([b == lev for b in block], float) for lev in self.blocks[1:]]
        if kind != "common":
            cols += [np.array([g == lev for g in group], float) for lev in self.groups[1:]]
        cols += self._shape_columns(kind, theta, x, group)
        return np.column_stack(cols)

    def _shape_columns(self, kind: str, theta: np.ndarray, x: np.ndarray,
                       group: list | None) -> list[np.ndarray]:
        """g(x; θ) одной колонкой или по группам (отдельные β и, для «all», θ)."""
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if kind in ("common", "alpha") or not self.groups:
                return [self.curve.shape(x, theta)]
            out = []
            for k, lev in enumerate(self.groups):
                th = theta[k * self.m:(k + 1) * self.m] if kind == "all" else theta
                out.append(self.curve.shape(x, th) * np.array([g == lev for g in group], float))
            return out

    def residuals(self, kind: str, theta: np.ndarray) -> np.ndarray:
        """y − X(θ)β̂(θ) — вариационная проекция."""
        d = self.design(kind, theta)
        if not np.all(np.isfinite(d)):
            return np.full_like(self.y, 1e10)
        return self.y - d @ np.linalg.lstsq(d, self.y, rcond=None)[0]


def _levels(frame: pd.DataFrame, col: str) -> list[str]:
    """Уровни фактора в естественном порядке (или пусто)."""
    from core.stats.factorial import _natural_order
    return _natural_order(frame[col]) if col in frame else []


def fit_kind(nd: NonlinearDesign, kind: str, start: np.ndarray) -> dict | None:
    """θ̂ по МНК остатков проекции, затем линейные параметры и ковариация по якобиану."""
    theta0 = np.tile(start, nd.n_theta(kind) // nd.m)
    sol = optimize.least_squares(lambda th: nd.residuals(kind, th), theta0, method="lm",
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    if not sol.success:
        return None
    theta = sol.x
    d = nd.design(kind, theta)
    lin = np.linalg.lstsq(d, nd.y, rcond=None)[0]
    jac = _jacobian(nd, kind, theta, lin, d)
    rss = float(((nd.y - d @ lin) ** 2).sum())
    df = len(nd.y) - jac.shape[1]
    return {"kind": kind, "theta": theta, "lin": lin, "rss": rss, "df": df, "s2": rss / df,
            "cov": rss / df * np.linalg.pinv(jac.T @ jac), "n_lin": d.shape[1]}


def _jacobian(nd: NonlinearDesign, kind: str, theta: np.ndarray, lin: np.ndarray,
              d: np.ndarray) -> np.ndarray:
    """∂μ/∂(линейные, θ): линейная часть — сам план, θ — центральные разности."""
    cols = [d]
    for j in range(theta.size):
        h = 1e-6 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        cols.append(((nd.design(kind, up) - nd.design(kind, down)) @ lin / (2 * h))[:, None])
    return np.column_stack(cols)


def _row_and_grad(nd: NonlinearDesign, fit: dict, x: np.ndarray, group: str | None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """μ̄(x) по блокам и градиент по (линейные, θ) для дельта-метода."""
    kind, theta, lin = fit["kind"], fit["theta"], fit["lin"]
    blocks = nd.blocks or [None]
    means, grads = [], []
    for b in blocks:
        kw = {"x": x, "block": [b] * x.size if b is not None else None,
              "group": [group] * x.size if group is not None else None}
        d = nd.design(kind, theta, **kw)
        cols = [d]
        for j in range(theta.size):
            h = 1e-6 * max(1.0, abs(theta[j]))
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            cols.append(((nd.design(kind, up, **kw) - nd.design(kind, down, **kw)) @ lin
                         / (2 * h))[:, None])
        means.append(d @ lin)
        grads.append(np.column_stack(cols))
    return np.mean(means, axis=0), np.mean(grads, axis=0)


def _nl_predictions(nd: NonlinearDesign, fit: dict, x: np.ndarray, group: str | None,
                    alpha: float) -> list[dict]:
    """Предсказания для среднего блока с приближённым ДИ (дельта-метод)."""
    mean, grad = _row_and_grad(nd, fit, x, group)
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", grad, fit["cov"], grad), 0, None))
    crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    return [{"x": float(x[i]), "group": group, "fitted": float(mean[i]), "se_mean": float(se[i]),
             "mean_lower": float(mean[i] - crit * se[i]),
             "mean_upper": float(mean[i] + crit * se[i])} for i in range(x.size)]


def _nl_labels(nd: NonlinearDesign, fit: dict, names: dict[str, str]) -> list[str]:
    """Подписи: [1], блоки, группы (сдвиг α), β (по группам), θ (по группам для «all»)."""
    kind = fit["kind"]
    labels = ["alpha"] + [f"{names['block']} {b}" for b in nd.blocks[1:]]
    if kind != "common":
        labels += [f"{names['group']} {g}" for g in nd.groups[1:]]
    per_group = kind in ("alpha_beta", "all") and nd.groups
    labels += [f"beta {g}" for g in nd.groups] if per_group else ["beta"]
    theta_names = list(nd.curve.names[2:])
    if kind == "all" and nd.groups:
        return labels + [f"{t} {g}" for g in nd.groups for t in theta_names]
    return labels + theta_names


def _nl_parameters(nd: NonlinearDesign, fit: dict, names: dict[str, str],
                   alpha: float) -> list[dict]:
    """Оценки с приближёнными SE, t, P и ДИ."""
    est = np.r_[fit["lin"], fit["theta"]]
    se = np.sqrt(np.clip(np.diag(fit["cov"]), 0, None))
    crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for lab, b, s_ in zip(_nl_labels(nd, fit, names), est, se):
        t = b / s_ if s_ > 0 else float("nan")
        rows.append({"term": lab, "estimate": float(b), "se": float(s_), "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), fit["df"])),
                     "ci_lower": float(b - crit * s_), "ci_upper": float(b + crit * s_)})
    return rows


def _comparisons(fits: dict[str, dict]) -> list[dict]:
    """Вложенные модели групп: F прироста против остатка модели «all» (как Table 15.4)."""
    ref = fits["all"]
    rows, prev = [], None
    for kind in GROUP_MODELS:
        fit = fits[kind]
        row = {"model": kind, "rss": fit["rss"], "df": fit["df"]}
        if prev is not None and prev["df"] > fit["df"]:
            ddf = prev["df"] - fit["df"]
            f = (prev["rss"] - fit["rss"]) / ddf / ref["s2"]
            row.update({"df_change": ddf, "F": float(f), "p": float(stats.f.sf(f, ddf, ref["df"]))})
        rows.append(row)
        prev = fit
    return rows


def _choose(rows: list[dict], alpha: float) -> str:
    """Самая простая модель, после которой усложнение незначимо."""
    chosen = "common"
    for row in rows[1:]:
        if row.get("p", 1.0) >= alpha:
            break
        chosen = row["model"]
    return chosen


def _lack_of_fit(frame: pd.DataFrame, fit: dict) -> dict | None:
    """Неадекватность: модель против Block + FacX(.Group) — остаток чистой ошибки (Table 15.13)."""
    cells = frame["facx"] + ("|" + frame["group"] if "group" in frame else "")
    data = frame.assign(cell=cells)
    factors = ["cell"] + (["block"] if "block" in frame else [])
    model = Terms(data, factors, [])
    fac = model.fit([("block",), ("cell",)] if "block" in frame else [("cell",)])
    ddf = fit["df"] - fac["df"]
    if fac["df"] < 1 or ddf < 1:
        return None
    f = (fit["rss"] - fac["rss"]) / ddf / fac["s2"]
    return {"df": ddf, "ss": fit["rss"] - fac["rss"], "F": float(f),
            "p": float(stats.f.sf(f, ddf, fac["df"])), "df_resid": fac["df"],
            "ms_resid": fac["s2"]}


def nonlinear_in_blocks(
    values: list[float],
    x: list,
    block: list[str] | None = None,
    group: list[str] | None = None,
    model: str = "exponential",
    group_model: str = "auto",
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Нелинейная кривая отклика в RCBD и по группам (Exercises 17.1b, 17.6; стр. 476)."""
    if model not in CURVES or group_model not in ("auto", *GROUP_MODELS):
        raise ValueError(f"model ∈ {tuple(CURVES)}, group_model ∈ (auto, {', '.join(GROUP_MODELS)})")
    settings = _settings(alpha, transform_kind, offset)
    frame, dropped = _collect(values, x, block, group)
    names = _names(column_names)
    result = _start_result("nonlinear_in_blocks", frame, dropped, settings, names, model=model,
                           group_model=group_model)
    if not (_transform_column(frame, "y", transform_kind, offset, result)
            and _guard(frame, len(CURVES[model].names) + 1, result)):
        return result
    if model == "power" and (frame["x"] <= 0).any():
        result.add_warning("x_transform_domain", "Степенная кривая требует x > 0", Severity.ERROR)
        return result
    _fill_nonlinear(NonlinearDesign(frame, model), x_predict, names, result)
    return result


def _start_theta(nd: NonlinearDesign) -> np.ndarray | None:
    """Старт θ: сетка по данным с вычтенными эффектами блоков."""
    y = nd.y.copy()
    if nd.blocks:
        y = y - nd.frame.groupby("block")["y"].transform("mean").to_numpy() + y.mean()
    start = _start(nd.curve, nd.x, y)
    return None if start is None else start[2:]


def _fill_nonlinear(nd: NonlinearDesign, x_predict: str, names: dict[str, str],
                    result: StatResult) -> None:
    """Подгонка моделей, выбор, неадекватность, параметры и кривые."""
    alpha = result.parameters["alpha"]
    start = _start_theta(nd)
    kinds = GROUP_MODELS if nd.groups else ("common",)
    fits = {k: fit_kind(nd, k, start) for k in kinds} if start is not None else {}
    if not fits or any(f is None for f in fits.values()):
        result.add_warning("nonlinear_not_converged", "Итерации не сошлись: выбери другую "
                           "кривую.", Severity.ERROR)
        return
    chosen = "common"
    if nd.groups:
        rows = _comparisons(fits)
        chosen = _choose(rows, alpha) if result.parameters["group_model"] == "auto" else \
            result.parameters["group_model"]
        result.parameters["group_models"] = rows
    _report_nonlinear(nd, fits[chosen], x_predict, names, result)


def _report_nonlinear(nd: NonlinearDesign, fit: dict, x_predict: str, names: dict[str, str],
                      result: StatResult) -> None:
    """Параметры, неадекватность, предсказания и кривые выбранной модели."""
    alpha = result.parameters["alpha"]
    params = _nl_parameters(nd, fit, names, alpha)
    lof = _lack_of_fit(nd.frame, fit)
    groups = nd.groups or [None]
    points = np.asarray(parse_numbers(x_predict), float)
    grid = np.linspace(nd.x.min(), nd.x.max(), _GRID)
    curves = []
    for g in groups:
        band = _nl_predictions(nd, fit, grid, g, alpha)
        curves.append({"group": g, "x": grid.tolist(), "fitted": [r["fitted"] for r in band],
                       "lower": [r["mean_lower"] for r in band],
                       "upper": [r["mean_upper"] for r in band]})
    result.parameters.update({
        "selected_model": fit["kind"], "parameter_tests": params, "lack_of_fit": lof,
        "predictions": [r for g in groups for r in _nl_predictions(nd, fit, points, g, alpha)]
        if points.size else [], "curves": curves, "formula": nd.curve.formula,
        "points": {"x": nd.x.tolist(), "y": nd.y.tolist()}})
    result.statistics.update({"rss": fit["rss"], "df_resid": fit["df"], "ms_resid": fit["s2"]})
    if lof:
        result.statistics.update({"F_lack_of_fit": lof["F"], "p_lack_of_fit": lof["p"]})
        if lof["p"] < alpha:
            result.add_warning("curve_lack_of_fit", "Средние по уровням x значимо отклоняются от "
                               "кривой (гл. 12.8).")
    result.add_warning("nonlinear_approximate", "SE, t и F нелинейной модели приближённые.",
                       Severity.INFO)
    for row in params:
        result.estimates.append(Estimate(row["term"], row["estimate"], row["ci_lower"],
                                         row["ci_upper"], 1 - alpha, row["se"]))
