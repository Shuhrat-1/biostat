"""Регрессия в планируемых опытах и ковариационный анализ (гл. 15.3–15.5 книги).

- ``blocked_regression`` — линейный отклик на количественную обработку в
  RCBD: Block (структура, не тестируется) + x + фактор x (неадекватность),
  параметры параллельных линий по блокам и прямая, усреднённая по блокам
  (Tables 15.13, 15.14).
- ``ancova`` — ковариата и группы обработок (один фактор или A × B,
  необязательные блоки): проверка связи ковариаты с обработками, модель
  отдельных линий, обратное исключение по маргинальности с переподгонкой
  (Table 15.18), параметры итоговой модели (Table 15.16), скорректированные
  средние при среднем ковариаты с SE, SED и LSD и средние без ковариаты
  (Table 15.17).
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _natural_order, _new_result, _settings
from core.stats.regression_checks import _transform_column, influence_measures

_COVARIATE = "cov"
_FEW_PER_GROUP = 5  # ≤ 5 наблюдений на группу: отдельные линии ненадёжны (стр. 410)


# --- Члены модели -------------------------------------------------------------


class Terms:
    """Построитель матрицы плана: факторы (first-level-zero) и количественные переменные."""

    def __init__(self, frame: pd.DataFrame, factors: list[str], variates: list[str]) -> None:
        self.frame = frame
        self.levels = {f: _natural_order(frame[f]) for f in factors}
        self.variates = set(variates)

    def columns(self, term: tuple[str, ...]) -> np.ndarray:
        """Столбцы члена: произведение индикаторов факторов и значений переменных."""
        cols = np.ones((len(self.frame), 1))
        for name in term:
            if name in self.variates:
                part = self.frame[name].to_numpy(float)[:, None]
            else:
                part = np.column_stack([(self.frame[name] == lev).to_numpy(float)
                                        for lev in self.levels[name][1:]])
            cols = np.einsum("ij,ik->ijk", cols, part).reshape(len(self.frame), -1)
        return cols

    def design(self, terms: list[tuple[str, ...]]) -> np.ndarray:
        """[1] + члены."""
        return np.column_stack([np.ones((len(self.frame), 1))] + [self.columns(t) for t in terms])

    def fit(self, terms: list[tuple[str, ...]]) -> dict:
        """МНК: параметры, RSS, ранг, df, s²."""
        design = self.design(terms)
        y = self.frame["y"].to_numpy(float)
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(((y - design @ beta) ** 2).sum())
        df = len(y) - int(rank)
        return {"design": design, "beta": beta, "rss": rss, "rank": int(rank), "df": df,
                "s2": rss / df if df > 0 else float("nan")}


def _label(term: tuple[str, ...], names: dict[str, str]) -> str:
    """Член с именами колонок пользователя: ('cov', 'A') → 'Damage.Trt'."""
    return ".".join(names.get(t, t) for t in term)


def _marginal(term: tuple[str, ...], terms: list[tuple[str, ...]]) -> bool:
    """Член входит в другой член модели (принцип маргинальности, стр. 399)."""
    return any(set(term) < set(other) for other in terms)


# --- Последовательная таблица и отбор -------------------------------------------


def sequential_rows(model: Terms, terms: list[tuple[str, ...]], names: dict[str, str]) -> list[dict]:
    """Инкрементные SS и df; F против ResMS полной модели (Tables 15.4, 15.13)."""
    full = model.fit(terms)
    prev, rows = model.fit([]), []
    for k, term in enumerate(terms):
        cur = model.fit(terms[:k + 1])
        df, ss = cur["rank"] - prev["rank"], prev["rss"] - cur["rss"]
        row = {"term": "+ " + _label(term, names), "df": df, "ss": ss}
        if df > 0:
            f = ss / df / full["s2"]
            row.update({"ms": ss / df, "F": f, "p": float(stats.f.sf(f, df, full["df"]))})
        rows.append(row)
        prev = cur
    total = float(((model.frame["y"] - model.frame["y"].mean()) ** 2).sum())
    return rows + [{"term": "Residual", "df": full["df"], "ss": full["rss"], "ms": full["s2"]},
                   {"term": "Total", "df": len(model.frame) - 1, "ss": total}]


def _drop_test(model: Terms, current: list[tuple[str, ...]], term: tuple[str, ...]) -> dict:
    """Маргинальный F-тест исключения члена по текущей (переподогнанной) модели."""
    big = model.fit(current)
    small = model.fit([t for t in current if t != term])
    df = big["rank"] - small["rank"]
    f = (small["rss"] - big["rss"]) / df / big["s2"]
    return {"term": term, "df": df, "df_resid": big["df"], "F": float(f),
            "p": float(stats.f.sf(f, df, big["df"]))}


def backward_selection(model: Terms, terms: list[tuple[str, ...]], protected: set[tuple[str, ...]],
                       alpha: float) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Исключать наименее значимый допустимый член при P > α, переподгоняя модель (стр. 414)."""
    current, models = list(terms), []
    while True:
        eligible = [t for t in current if t not in protected and not _marginal(t, current)]
        tests = [_drop_test(model, current, t) for t in eligible]
        models.append({"terms": list(current), "tests": tests})
        if not tests:
            return current, models
        worst = max(tests, key=lambda r: r["p"])
        if worst["p"] <= alpha:
            return current, models
        current.remove(worst["term"])


# --- Параметры, средние, линии --------------------------------------------------


def parameter_rows(model: Terms, terms: list[tuple[str, ...]], names: dict[str, str],
                   alpha: float) -> tuple[list[dict], dict]:
    """Оценки first-level-zero с SE, t, P и ДИ (Tables 15.14, 15.16)."""
    fit = model.fit(terms)
    labels = ["[1]"]
    for t in terms:
        labels += _column_labels(model, t, names)
    cov = fit["s2"] * np.linalg.pinv(fit["design"].T @ fit["design"])
    se = np.sqrt(np.diag(cov))
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = [{"term": lab, "estimate": float(b), "se": float(s), "t": float(b / s),
             "p": float(2 * stats.t.sf(abs(b / s), fit["df"])), "ci_lower": float(b - t_crit * s),
             "ci_upper": float(b + t_crit * s)} for lab, b, s in zip(labels, fit["beta"], se)]
    return rows, {**fit, "cov": cov, "t_crit": t_crit}


def _column_labels(model: Terms, term: tuple[str, ...], names: dict[str, str]) -> list[str]:
    """Подписи столбцов члена: Trt +N, Damage.Variety N …"""
    parts = [[names.get(n, n)] if n in model.variates else
             [f"{names.get(n, n)} {lev}" for lev in model.levels[n][1:]] for n in term]
    return [".".join(combo) for combo in product(*parts)]


def _row_for(model: Terms, terms: list[tuple[str, ...]], values: dict[str, object]) -> np.ndarray:
    """Строка плана для заданных уровней факторов и значений переменных."""
    one = pd.DataFrame({k: [v] for k, v in values.items()})
    probe = Terms.__new__(Terms)
    probe.frame, probe.levels, probe.variates = one, model.levels, model.variates
    return probe.design(terms)[0]


def adjusted_means(model: Terms, terms: list[tuple[str, ...]], fit: dict, treatments: list[str],
                   at: dict[str, float], average_over: list[str]) -> list[dict]:
    """Средние по группам при заданных x, усреднённые по структурным факторам (стр. 409, 412)."""
    rows = []
    for combo in product(*(model.levels[t] for t in treatments)):
        others = list(product(*(model.levels[a] for a in average_over))) or [()]
        design = np.mean([_row_for(model, terms, {**dict(zip(treatments, combo)),
                                                  **dict(zip(average_over, rest)), **at})
                          for rest in others], axis=0)
        rows.append({"group": " × ".join(combo), "row": design,
                     "mean": float(design @ fit["beta"]),
                     "se": float(np.sqrt(design @ fit["cov"] @ design))})
    return rows


def _sed_summary(means: list[dict], fit: dict) -> dict:
    """Наименьший и наибольший SED и LSD между скорректированными средними."""
    seds = [float(np.sqrt((a["row"] - b["row"]) @ fit["cov"] @ (a["row"] - b["row"])))
            for a, b in combinations(means, 2)]
    return {"sed_min": min(seds), "sed_max": max(seds),
            "lsd_min": fit["t_crit"] * min(seds), "lsd_max": fit["t_crit"] * max(seds)}


def group_lines(model: Terms, terms: list[tuple[str, ...]], fit: dict, treatments: list[str],
                average_over: list[str]) -> list[dict]:
    """Сдвиг и наклон по ковариате для каждой группы с SE (Example 15.4B, стр. 416)."""
    rows = []
    for g in adjusted_means(model, terms, fit, treatments, {_COVARIATE: 0.0}, average_over):
        at_one = adjusted_means(model, terms, fit, treatments, {_COVARIATE: 1.0}, average_over)
        slope_row = next(r for r in at_one if r["group"] == g["group"])["row"] - g["row"]
        rows.append({"group": g["group"], "intercept": g["mean"], "intercept_se": g["se"],
                     "slope": float(slope_row @ fit["beta"]),
                     "slope_se": float(np.sqrt(slope_row @ fit["cov"] @ slope_row))})
    return rows


# --- Подготовка данных ----------------------------------------------------------


def _to_float(values: list) -> np.ndarray:
    """Строки и числа в float; нечисло → NaN (десятичная запятая допустима)."""
    out = []
    for v in values:
        try:
            out.append(float(str(v).replace(",", ".")) if v not in ("", None) else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.asarray(out, float)


def _frame(values: list[float], numeric: dict[str, list], factors: dict[str, list | None]
           ) -> tuple[pd.DataFrame, int]:
    """Полные строки: отклик, количественные и факторы (пустая метка — пропуск)."""
    data = {"y": np.asarray(values, float)}
    data.update({k: _to_float(v) for k, v in numeric.items()})
    data.update({k: pd.Series(v, dtype=object).replace("", np.nan)
                 for k, v in factors.items() if v is not None})
    frame = pd.DataFrame(data)
    clean = frame.dropna().reset_index(drop=True)
    for k, v in factors.items():
        if v is not None:
            clean[k] = clean[k].astype(str)
    return clean, len(frame) - len(clean)


def _start(method: str, frame: pd.DataFrame, dropped: int, settings: dict,
           names: dict[str, str]) -> StatResult:
    """Результат с настройками и предупреждением о пропусках."""
    result = _new_result(method, frame, settings, column_names=names)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    return result


def _residual_plots(fit: dict, frame: pd.DataFrame) -> dict:
    """Стандартизованные остатки итоговой модели."""
    m = influence_measures(fit["design"][:, _independent(fit["design"])],
                           frame["y"].to_numpy(float))
    r = m["std_residual"]
    blom = stats.norm.ppf((np.arange(1, r.size + 1) - 0.375) / (r.size + 0.25))
    return {"residual_vs_fitted": {"fitted": m["fitted"].tolist(), "residuals": r.tolist()},
            "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
            "std_residuals": r.tolist()}


def _independent(design: np.ndarray) -> list[int]:
    """Индексы линейно независимых столбцов (жадно слева направо)."""
    keep: list[int] = []
    for j in range(design.shape[1]):
        if np.linalg.matrix_rank(design[:, keep + [j]]) > len(keep):
            keep.append(j)
    return keep


# --- blocked_regression ---------------------------------------------------------


def blocked_regression(
    values: list[float],
    x: list,
    block: list[str],
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Регрессия на количественную обработку внутри блоков с проверкой неадекватности (15.3)."""
    settings = _settings(alpha, transform_kind, offset)
    names = {"x": "x", "block": "Block", **(column_names or {})}
    frame, dropped = _frame(values, {"x": x}, {"block": block})
    frame["facx"] = frame["x"].map(lambda v: f"{v:g}")
    result = _start("blocked_regression", frame, dropped, settings, names)
    if not _transform_column(frame, "y", transform_kind, offset, result) or not _blocked_guard(
            frame, result):
        return result
    model = Terms(frame, ["block", "facx"], ["x"])
    labels = {**names, "facx": "Fac" + names["x"]}
    seq = sequential_rows(model, [("block",), ("x",), ("facx",)], labels)
    result.parameters["sequential"] = seq
    _blocked_fit(model, labels, seq, result)
    return result


def _blocked_guard(frame: pd.DataFrame, result: StatResult) -> bool:
    """Нужны ≥ 2 блока и ≥ 3 значения x (иначе неадекватность не проверить)."""
    if frame["block"].nunique() < 2 or frame["x"].nunique() < 3:
        result.add_warning("blocked_regression_invalid", "Нужно ≥ 2 блоков и ≥ 3 уровней x",
                           Severity.ERROR)
        return False
    return True


def _blocked_fit(model: Terms, names: dict[str, str], seq: list[dict], result: StatResult) -> None:
    """Параллельные линии по блокам, прямая для средних условий опыта (стр. 408–409)."""
    alpha = result.parameters["alpha"]
    rows, fit = parameter_rows(model, [("block",), ("x",)], names, alpha)
    lof = seq[2]
    result.parameters["parameter_tests"] = rows
    blocks = model.levels["block"]
    w = np.zeros(len(rows))
    w[0] = 1.0
    w[1:len(blocks)] = 1.0 / len(blocks)
    icept, se = float(w @ fit["beta"]), float(np.sqrt(w @ fit["cov"] @ w))
    slope = rows[-1]
    result.parameters["line"] = {"intercept": icept, "intercept_se": se,
                                 "slope": slope["estimate"], "slope_se": slope["se"]}
    result.statistics.update({"F_x": seq[1]["F"], "p_x": seq[1]["p"], "F_lack_of_fit": lof["F"],
                              "p_lack_of_fit": lof["p"], "ms_resid": fit["s2"],
                              "df_resid": fit["df"], "intercept_averaged": icept})
    result.estimates += [Estimate("intercept (среднее по блокам)", icept, std_error=se),
                         Estimate("slope", slope["estimate"], slope["ci_lower"],
                                  slope["ci_upper"], 1 - alpha, slope["se"])]
    result.p_value = seq[1]["p"]
    result.parameters["plots"] = _residual_plots(fit, model.frame)
    if lof["p"] < alpha:
        result.add_warning("lack_of_fit", "Отклонения средних по уровням x от прямой "
                           "значимы: прямая не описывает зависимость (гл. 12.8).")


# --- ancova -----------------------------------------------------------------------


def _treatment_terms(has_b: bool) -> list[tuple[str, ...]]:
    """A или A + B + A.B (Example 15.4B)."""
    return [("A",), ("B",), ("A", "B")] if has_b else [("A",)]


def _full_terms(has_b: bool, has_block: bool) -> list[tuple[str, ...]]:
    """[Block] + cov + обработки + cov.обработки (ковариата первой, стр. 411)."""
    trt = _treatment_terms(has_b)
    return ([("block",)] if has_block else []) + [(_COVARIATE,)] + trt + [
        (_COVARIATE, *t) for t in trt]


def _covariate_check(frame: pd.DataFrame, has_b: bool, has_block: bool,
                     names: dict[str, str]) -> list[dict]:
    """Связана ли ковариата с обработками: ANOVA ковариаты (шаг 1, стр. 410)."""
    sub = frame.assign(y=frame[_COVARIATE])
    model = Terms(sub, [f for f in ("A", "B", "block") if f in sub], [])
    terms = ([("block",)] if has_block else []) + _treatment_terms(has_b)
    rows = sequential_rows(model, terms, names)
    return [r for r in rows if r["term"] not in ("+ " + names.get("block", "block"),)]


def _ancova_guard(frame: pd.DataFrame, has_b: bool, result: StatResult) -> bool:
    """≥ 2 уровня у факторов обработок и остаточные df у модели отдельных линий."""
    factors = ["A", "B"] if has_b else ["A"]
    if any(frame[f].nunique() < 2 for f in factors):
        result.add_warning("factor_one_level", "У фактора обработки один уровень", Severity.ERROR)
        return False
    if frame[_COVARIATE].nunique() < 2:
        result.add_warning("no_x_variation", "Ковариата не варьирует", Severity.ERROR)
        return False
    return True


def ancova(
    values: list[float],
    group: list[str],
    covariate: list,
    factor_b: list[str] | None = None,
    block: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Ковариационный анализ как регрессия с группами (гл. 15.4–15.5)."""
    settings = _settings(alpha, transform_kind, offset)
    names = {"A": "A", "B": "B", "block": "Block", _COVARIATE: "x"}
    cn = column_names or {}
    names.update({k: cn[s] for k, s in (("A", "group"), ("B", "factor_b"), ("block", "block"),
                                        (_COVARIATE, "covariate")) if s in cn})
    frame, dropped = _frame(values, {_COVARIATE: covariate},
                            {"A": group, "B": factor_b, "block": block})
    result = _start("ancova", frame, dropped, settings, names)
    has_b, has_block = factor_b is not None, block is not None
    if not _transform_column(frame, "y", transform_kind, offset, result) or not _ancova_guard(
            frame, has_b, result):
        return result
    _ancova_fill(frame, has_b, has_block, names, result)
    return result


def _ancova_fill(frame: pd.DataFrame, has_b: bool, has_block: bool, names: dict[str, str],
                 result: StatResult) -> None:
    """Шаги ANCOVA: связь ковариаты, отдельные линии, отбор, итоговая модель."""
    alpha = result.parameters["alpha"]
    model = Terms(frame, [f for f in ("A", "B", "block") if f in frame], [_COVARIATE])
    full = _full_terms(has_b, has_block)
    result.parameters["covariate_check"] = _covariate_check(frame, has_b, has_block, names)
    result.parameters["sequential"] = sequential_rows(model, full, names)
    protected = {("block",), (_COVARIATE,)}
    final, models = backward_selection(model, full, protected, alpha)
    result.parameters["selection"] = [
        {"terms": [_label(t, names) for t in m["terms"]],
         "tests": [{**r, "term": _label(r["term"], names)} for r in m["tests"]]} for m in models]
    result.parameters["selected_terms"] = [_label(t, names) for t in final]
    _ancova_final(model, final, has_b, has_block, names, result)
    _ancova_warnings(frame, has_b, result)


def _ancova_final(model: Terms, final: list[tuple[str, ...]], has_b: bool, has_block: bool,
                  names: dict[str, str], result: StatResult) -> None:
    """Параметры, средние при среднем ковариаты (если линии параллельны) или линии групп."""
    alpha = result.parameters["alpha"]
    rows, fit = parameter_rows(model, final, names, alpha)
    result.parameters["parameter_tests"] = rows
    treatments = [f for f in ("A", "B") if any(f in t for t in final)]
    avg = ["block"] if has_block else []
    parallel = not any(_COVARIATE in t and len(t) > 1 for t in final)
    result.parameters["model_type"] = "parallel_lines" if parallel else "separate_lines"
    mean_x = float(model.frame[_COVARIATE].mean())
    if treatments and parallel:
        means = adjusted_means(model, final, fit, treatments, {_COVARIATE: mean_x}, avg)
        result.parameters["adjusted_means"] = [{k: v for k, v in m.items() if k != "row"}
                                               for m in means]
        result.statistics.update(_sed_summary(means, fit))
        result.parameters["unadjusted_means"] = _unadjusted(model, treatments, has_block)
    elif treatments:
        result.parameters["group_lines"] = group_lines(model, final, fit, treatments, avg)
    result.statistics.update({"covariate_mean": mean_x, "ms_resid": fit["s2"],
                              "df_resid": fit["df"]})
    result.parameters["plots"] = _residual_plots(fit, model.frame)
    for r in rows:
        result.estimates.append(Estimate(r["term"], r["estimate"], r["ci_lower"], r["ci_upper"],
                                         1 - alpha, r["se"]))


def _unadjusted(model: Terms, treatments: list[str], has_block: bool) -> list[dict]:
    """Средние модели отдельных групп без ковариаты (Table 15.17b)."""
    terms = ([("block",)] if has_block else []) + (
        [("A",), ("B",), ("A", "B")] if len(treatments) == 2 else [(treatments[0],)])
    _, fit = parameter_rows(model, terms, {}, 0.05)
    means = adjusted_means(model, terms, fit, treatments, {_COVARIATE: 0.0},
                           ["block"] if has_block else [])
    return [{k: v for k, v in m.items() if k != "row"} for m in means]


def _ancova_warnings(frame: pd.DataFrame, has_b: bool, result: StatResult) -> None:
    """Ковариата зависит от обработок; мало наблюдений на группу (стр. 410)."""
    alpha = result.parameters["alpha"]
    check = [r for r in result.parameters["covariate_check"] if "p" in r]
    related = [r["term"].lstrip("+ ") for r in check if r["p"] < alpha]
    if related:
        result.add_warning("covariate_related", "Ковариата различается между обработками: "
                           f"{', '.join(related)} — интерпретируй коррекцию осторожно.",
                           params={"terms": ", ".join(related)})
    cells = frame.groupby(["A", "B"] if has_b else ["A"]).size()
    if cells.min() <= _FEW_PER_GROUP:
        result.add_warning("ancova_few_per_group", f"В группах ≤ {_FEW_PER_GROUP} наблюдений: "
                           "проверка отдельных линий ненадёжна.", Severity.INFO,
                           params={"n": int(cells.min())})
