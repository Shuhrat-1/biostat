"""Расчётные распознаватели карты (decision-tree-spec v2, разделы 5 и 8).

В отличие от структурных, им нужна подгонка модели: остатки, девианс,
VIF. Модель строится по ролям колонок формы:

- отклик + фактор (+ второй фактор с взаимодействием, + блок) — модель
  вариантов, по ней правила допущений B4 и диагностики DIAG;
- y + x — простая регрессия, по ней правила формы связи A2 и DIAG.4.

Подгонка одна на вызов (``Context.memo``). Карта вызывает эти правила
вторым запросом, лениво — по текущему выбору колонок. Пороги с пометкой
[BioStat] — эвристики, в книге их нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
from scipy import stats

from core.stats.outliers import outlier_candidate
from core.stats.welch_rule import welch_advice
from core.validation import column_names as names
from core.validation.recognize import COMPUTED, HINT, STRONG, Context, Rule, Signal, rule
from core.validation.recognizers import (
    _count_like, _treatment_labels, _trials_column, _trials_guess, binary_response,
    in_glm_branch, many_zeros, right_skew, transform_branch,
)

_ALPHA = 0.05
_SHAPIRO_MAX = 5000  # предел scipy.stats.shapiro


@dataclass(frozen=True)
class Fit:
    """Подгонка МНК: отклик, матрица модели, остатки, рычаги, номера строк."""

    y: np.ndarray
    X: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    leverage: np.ndarray
    rows: np.ndarray  # номера строк файла (с 1) для подогнанных наблюдений
    df: int

    @property
    def s2(self) -> float:
        return float(self.resid @ self.resid / self.df)

    def standardized(self) -> np.ndarray:
        """Стандартизованные остатки r / (s·√(1 − h)) [кн. стр. 95]."""
        return self.resid / np.sqrt(self.s2 * np.clip(1 - self.leverage, 1e-12, None))


def _ols(y: np.ndarray, X: np.ndarray, rows: np.ndarray) -> Fit | None:
    """МНК через QR; рычаги — диагональ проекционной матрицы."""
    q, r = np.linalg.qr(X)
    keep = np.abs(np.diag(r)) > 1e-10 * max(1.0, np.abs(r).max())
    q = q[:, keep]
    rank = int(keep.sum())
    df = y.size - rank
    if df < 1:
        return None
    fitted = q @ (q.T @ y)
    return Fit(y, X, y - fitted, fitted, np.sum(q * q, axis=1), rows, df)


def _dummies(labels: list[str]) -> np.ndarray:
    """Кодирование уровней «treatment»: первый уровень — базовый."""
    levels = sorted(set(labels))
    return np.array([[lab == lv for lv in levels[1:]] for lab in labels], dtype=float)


def _factor_terms(ctx: Context) -> list[str]:
    """Колонки-факторы модели вариантов: основной, второй, блок."""
    names = [ctx.factor, ctx.role("factor_b"), ctx.role("block")]
    return [n for i, n in enumerate(names) if n and n not in names[:i]]


def _covariate(ctx: Context) -> np.ndarray | None:
    """Значения ковариаты по строкам, если она назначена и числовая."""
    col = ctx.role("covariate")
    return ctx.numbers(col) if col and col in ctx.columns_of("numeric") else None


def _factor_fit(ctx: Context) -> Fit | None:
    """Модель вариантов: y ~ A (+ B + A×B) (+ блок) (+ ковариата).

    Ковариата входит в модель: иначе разброс, который она объясняет,
    выглядит как неоднородность или ненормальность остатков [кн. стр. 409].
    """
    resp, terms = ctx.response, _factor_terms(ctx)
    if resp is None or not terms or ctx.factor is None:
        return None
    y, z = ctx.numbers(resp), _covariate(ctx)
    labels = [ctx.labels(t) for t in terms]
    ok = ~np.isnan(y) & np.all([[bool(v) for v in col] for col in labels], axis=0)
    if z is not None:
        ok &= ~np.isnan(z)
    if ok.sum() < 4:
        return None
    cols = [[v for v, k in zip(col, ok) if k] for col in labels]
    parts = [np.ones((int(ok.sum()), 1))] + [_dummies(c) for c in cols]
    if ctx.role("factor_b") and len(terms) >= 2:
        parts.append(_dummies([f"{a}\x1f{b}" for a, b in zip(cols[0], cols[1])]))
    if z is not None:
        parts.append(z[ok].reshape(-1, 1))
    return _ols(y[ok], np.hstack(parts), np.flatnonzero(ok) + 1)


def _xy(ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Пары (x, y) без пропусков и номера их строк."""
    x, resp = ctx.role("x"), ctx.response
    if not (x and resp):
        return None
    xv, yv = ctx.numbers(x), ctx.numbers(resp)
    ok = ~np.isnan(xv) & ~np.isnan(yv)
    if ok.sum() < 4:
        return None
    return xv[ok], yv[ok], np.flatnonzero(ok) + 1


def _line_fit(ctx: Context) -> Fit | None:
    """Простая регрессия y ~ x."""
    data = _xy(ctx)
    if data is None:
        return None
    x, y, rows = data
    return _ols(y, np.column_stack([np.ones_like(x), x]), rows)


def factor_fit(ctx: Context) -> Fit | None:
    return ctx.memo("factor_fit", lambda: _factor_fit(ctx))


def line_fit(ctx: Context) -> Fit | None:
    return ctx.memo("line_fit", lambda: _line_fit(ctx))


def model_fit(ctx: Context) -> Fit | None:
    """Модель для диагностики: регрессия, если задан x, иначе модель вариантов."""
    return line_fit(ctx) if ctx.role("x") else factor_fit(ctx)


def _groups(ctx: Context) -> list[np.ndarray]:
    """Значения отклика по вариантам — комбинациям всех факторов (без пропусков).

    В факториале группа — ячейка A × B, а не уровень A: иначе эффект второго
    фактора попадает в разброс «внутри группы».
    """
    return ctx.memo("groups", lambda: _split_groups(ctx))


def _split_groups(ctx: Context, drop: int | None = None) -> list[np.ndarray]:
    """Группы по вариантам; ``drop`` — номер строки файла, которую не учитывать."""
    resp = ctx.response
    if not (resp and ctx.factor):
        return []
    y, labels = ctx.numbers(resp), np.array(_treatment_labels(ctx), dtype=object)
    ok = ~np.isnan(y) & (labels != "") & _kept(y.size, drop)
    return [y[ok & (labels == level)] for level in sorted(set(labels[ok]))]


def _kept(n: int, drop: int | None) -> np.ndarray:
    """Маска строк без строки ``drop`` (номера строк файла — с единицы)."""
    keep = np.ones(n, dtype=bool)
    if drop is not None:
        keep[drop - 1] = False
    return keep


@dataclass(frozen=True)
class Spread:
    """Разброс внутри варианта: среднее, оценка дисперсии и её df."""

    mean: float
    var: float
    df: float


def _spreads(ctx: Context, drop: int | None = None) -> list[Spread]:
    """Дисперсии вариантов для проверки однородности (группы с n ≥ 2).

    ``drop`` — номер строки файла, без которой считать (кандидат в выбросы).

    Без блоков — выборочные дисперсии групп, как в книге для CRD
    [кн. стр. 105–107]. С блоком книга формального теста не даёт (для RCBD —
    только графики остатков, стр. 137), поэтому [предл.]: дисперсии считаются
    по остаткам модели «вариант + блок», см. :func:`_adjusted`; так же при ковариате.
    """
    if not (ctx.response and ctx.factor):
        return []
    if ctx.role("block") or _covariate(ctx) is not None:
        return ctx.memo(f"spreads:{drop}", lambda: _adjusted(ctx, drop))
    groups = _groups(ctx) if drop is None else _split_groups(ctx, drop)
    return [Spread(float(g.mean()), float(g.var(ddof=1)), g.size - 1.0)
            for g in groups if g.size >= 2]


def _adjusted(ctx: Context, drop: int | None = None) -> list[Spread]:
    """Дисперсии вариантов по остаткам модели y ~ вариант (+ блок) (+ ковариата) [предл.].

    Сырой разброс варианта в блочном опыте включает различия блоков, а при
    ковариате — её влияние, поэтому сравниваются выборочные дисперсии
    остатков по вариантам (df = n − 1). Имитация при равных дисперсиях (RCBD
    от 3 × 8 до 10 × 3, 4000 повторов): Бартлетт по таким дисперсиям
    ошибается в 2–4 % случаев при номинальных 5 %.
    """
    y, z = ctx.numbers(ctx.response), _covariate(ctx)
    cells = np.array(_treatment_labels(ctx), dtype=object)
    ok = ~np.isnan(y) & (cells != "") & _kept(y.size, drop)
    parts = []
    if ctx.role("block"):
        blocks = np.array(ctx.labels(ctx.role("block")), dtype=object)
        ok &= blocks != ""
        parts.append(lambda: _dummies(list(blocks[ok])))
    if z is not None:
        ok &= ~np.isnan(z)
        parts.append(lambda: z[ok].reshape(-1, 1))
    if ok.sum() < 4:
        return []
    y, cells = y[ok], cells[ok]
    X = np.hstack([np.ones((y.size, 1)), _dummies(list(cells))] + [make() for make in parts])
    fit = _ols(y, X, np.flatnonzero(ok) + 1)
    if fit is None:
        return []
    picks = [cells == level for level in sorted(set(cells))]
    return [Spread(float(y[p].mean()), float(fit.resid[p].var(ddof=1)), p.sum() - 1.0)
            for p in picks if p.sum() >= 2]


# --- B4. Допущения ---------------------------------------------------------------

def _shapiro(ctx: Context) -> float | None:
    fit = factor_fit(ctx)
    if fit is None or fit.resid.size < 3 or fit.df < 2:
        return None
    return float(stats.shapiro(fit.resid[:_SHAPIRO_MAX]).pvalue)


def _bartlett_statistic(spreads: list[Spread]) -> float:
    """X² Бартлетта по дисперсиям групп и их df [кн. стр. 105–106]."""
    df = np.array([s.df for s in spreads])
    var = np.array([s.var for s in spreads])
    total = df.sum()
    pooled = float(df @ var / total)
    correction = 1 + (np.sum(1 / df) - 1 / total) / (3 * (df.size - 1))
    return float((total * np.log(pooled) - df @ np.log(var)) / correction)


def _bartlett(ctx: Context) -> tuple[float, float, int] | None:
    """(X², критическое χ²₀.₉₅(t − 1), t) по вариантам с n ≥ 2 [кн. стр. 105–107]."""
    spreads = _spreads(ctx)
    if len(spreads) < 2 or any(s.var <= 0 for s in spreads):
        return None
    crit = float(stats.chi2.ppf(1 - _ALPHA, len(spreads) - 1))
    return _bartlett_statistic(spreads), crit, len(spreads)


def _departure_matters(ctx: Context) -> bool:
    """При больших группах Шапиро отвергает и пустяки; по ЦПТ важна лишь заметная
    асимметрия остатков (|g| > 0.5) [BioStat]."""
    sizes = [g.size for g in _groups(ctx)]
    if not sizes or min(sizes) <= 30:
        return True
    fit = factor_fit(ctx)
    return fit is not None and abs(float(stats.skew(fit.resid))) > 0.5


@rule("B4.1", "B4", "nonnormal", STRONG, "кн. стр. 101; порог BioStat", cls=COMPUTED)
def nonnormal_residuals(ctx: Context) -> Signal | None:
    p = ctx.memo("shapiro", lambda: _shapiro(ctx)) if _anova_applies(ctx) else None
    if p is None or p >= _ALPHA or not _departure_matters(ctx):
        return None
    params = {"p": round(p, 4)}
    if len(_groups(ctx)) == 2:
        params["method"] = "mann_whitney"
    return Signal("Остатки модели отклоняются от нормальности (Шапиро–Уилк p = {p}).", params)


def _outlier_candidates(fit: Fit) -> np.ndarray:
    """Позиции наблюдений — кандидатов в выбросы, от самого крайнего.

    Два признака: (1) правило DIAG.1 по стандартизованным остаткам; (2) тест
    выброса для самого крайнего наблюдения с поправкой Бонферрони
    (:mod:`core.stats.outliers`) — он работает и на малых опытах.
    """
    r = fit.standardized()
    hits = set(np.flatnonzero(np.abs(r) > 2)) if _too_many_large(np.abs(r)) else set()
    worst = outlier_candidate(r, fit.df)
    if worst is not None:
        hits.add(worst)
    return np.array(sorted(hits, key=lambda i: -abs(r[i])), dtype=int)


def _skewed_without(ctx: Context, row: int) -> bool:
    """Данные скошены и без строки-кандидата: асимметрия или SD растёт со средним.

    Признаки считаются без кандидата, потому что один выброс сам создаёт и
    асимметрию, и «связь» SD со средним (на трёх группах — в 70 % случаев).
    """
    v = ctx.numbers(ctx.response).copy()
    v[row - 1] = np.nan
    v = v[~np.isnan(v)]
    right = v.size >= 8 and v.min() > 0 and float(stats.skew(v)) > 1
    return right or _tracks_mean(_spreads(ctx, drop=row)) is not None


def _log_scale_clears(ctx: Context, fit: Fit, hits: np.ndarray) -> bool:
    """Крайнее значение объясняется шкалой: на log-шкале кандидатов в выбросы нет.

    Если данные скошены и без кандидата, большие значения ожидаемы; выбросом
    считается то, что выделяется и после преобразования, которое книга для
    таких данных и предлагает [кн. стр. 113–116].
    """
    if fit.y.min() <= 0 or not _skewed_without(ctx, int(fit.rows[hits[0]])):
        return False
    logged = _ols(np.log(fit.y), fit.X, fit.rows)
    return logged is not None and _outlier_candidates(logged).size == 0


def _outlier_row(ctx: Context) -> int | None:
    """Номер строки самого резкого кандидата в выбросы в модели вариантов."""
    def find() -> int | None:
        fit = factor_fit(ctx)
        if fit is None or fit.df < 3:
            return None
        hits = _outlier_candidates(fit)
        if hits.size == 0 or _log_scale_clears(ctx, fit, hits):
            return None
        return int(fit.rows[hits[0]])

    return ctx.memo("outlier_row", find)


_OUTLIER_NOTE = " Возможно, это из-за выброса в строке {outlier_row} — сначала проверьте запись."


def outlier_aware(fn: Rule) -> Rule:
    """При кандидате в выбросы сигнал правила — подсказка с оговоркой, а не указание.

    Выброс раздувает дисперсию своей группы и ломает нормальность остатков;
    книга велит сначала проверить запись, а не менять метод [кн. стр. 109–110].
    """
    @wraps(fn)
    def wrapped(ctx: Context) -> Signal | None:
        signal = fn(ctx)
        row = _outlier_row(ctx) if signal is not None else None
        if row is None:
            return signal
        return Signal(signal.note + _OUTLIER_NOTE, {**signal.params, "outlier_row": row}, HINT)

    return wrapped


def _anova_applies(ctx: Context) -> bool:
    """Отклик годится для линейной модели: не 0/1 и не счёты с избытком нулей.

    Для бинарных данных и при множестве нулей ни трансформация, ни проверка
    остатков ANOVA не помогают [кн. стр. 126, 501, 512] — нужна ветка GLM.
    """
    return binary_response(ctx) is None and many_zeros(ctx) is None


def _welch_params(ctx: Context) -> tuple[str, dict] | None:
    """Две группы: случай правила welch_rule (отношение SD и баланс n) и параметры."""
    groups = _groups(ctx)
    advice = welch_advice(groups[0], groups[1])
    if advice is None:
        return None
    if not advice.reliable:
        return "small_n", {"sd_ratio": round(advice.sd_ratio, 2), "n_min": advice.n_min}
    return advice.case, {"sd_ratio": round(advice.sd_ratio, 2),
                         "n_ratio": round(advice.n_ratio, 2), "method": "two_sample_t"}


@rule("B4.2", "B4", "heterogeneous", STRONG,
      "кн. стр. 105–107; две группы — порог BioStat [предл.: Ruxton 2006]", cls=COMPUTED)
@outlier_aware
def heterogeneous_variances(ctx: Context) -> Signal | None:
    if not _anova_applies(ctx):
        return None
    if len(_groups(ctx)) == 2:  # не Бартлетт: правило, общее с two_sample_t
        res = ctx.memo("welch_rule", lambda: _welch_params(ctx))
        if res is None or _skew_explains_spread(ctx):
            return None
        case, p = res
        if case == "small_n":  # при n < 8 отношение SD ненадёжно: справка, не совет
            return Signal("SD двух групп различаются в {sd_ratio} {sd_ratio|раз|раза|раз},"
                          " но в меньшей группе всего {n_min}"
                          " {n_min|наблюдение|наблюдения|наблюдений}: по таким выборкам"
                          " о дисперсиях судить нельзя.", {**p, "variant": "small_n"}, strength=HINT)
        if case == "welch_liberal":
            return Signal("SD различаются в {sd_ratio} {sd_ratio|раз|раза|раз} при неравных n"
                          " (в {n_ratio} {n_ratio|раз|раза|раз}),"
                          " большая SD — в меньшей группе: P Стьюдента занижен, t-тест в"
                          " варианте Уэлча.", {**p, "variant": "welch_liberal"})
        if case == "welch_unbalanced":
            return Signal("SD различаются в {sd_ratio} {sd_ratio|раз|раза|раз} при неравных n"
                          " (в {n_ratio} {n_ratio|раз|раза|раз}),"
                          " большая SD — в большей группе: t-тест в варианте Уэлча.",
                          {**p, "variant": "welch_unbalanced"})
        return Signal("SD двух групп различаются в {sd_ratio} {sd_ratio|раз|раза|раз}:"
                      " t-тест в варианте Уэлча.",
                      {**p, "variant": "welch"})
    res = ctx.memo("bartlett", lambda: _bartlett(ctx))
    if res is None or res[0] <= res[1]:
        return None
    return Signal("Дисперсии групп неоднородны (Бартлетт X² = {x2} > {crit}).",
                  {"x2": round(res[0], 2), "crit": round(res[1], 2)})


def _skew_explains_spread(ctx: Context) -> bool:
    """Правая асимметрия и ненормальные остатки: SD различаются из-за самой асимметрии.

    Тогда отклик преобразуют (или берут ранговый тест), а совет Уэлча только
    спорил бы с ними [кн. стр. 115].
    """
    return right_skew(ctx) is not None and nonnormal_residuals(ctx) is not None


def _sd_tracks_mean(ctx: Context) -> float | None:
    """r между log SD и log среднего групп; None — связи нет [BioStat]."""
    return _tracks_mean(_spreads(ctx))


def _tracks_mean(spreads: list[Spread]) -> float | None:
    """Связь SD и среднего по готовым дисперсиям групп (≥ 3 групп, SD ×2, r ≥ 0.8)."""
    means = np.array([s.mean for s in spreads])
    if len(spreads) < 3 or means.min() <= 0:
        return None
    sds = np.sqrt([s.var for s in spreads])
    if sds.min() <= 0 or sds.max() / sds.min() < 2:
        return None
    r = float(np.corrcoef(np.log(means), np.log(sds))[0, 1])
    return r if r >= 0.8 else None


@rule("B4.3", "B4", "transform_groups", STRONG, "кн. стр. 96–97, 115; критерий BioStat",
      cls=COMPUTED)
@outlier_aware
def sd_grows_with_mean(ctx: Context) -> Signal | None:
    r = ctx.memo("sd_mean", lambda: _sd_tracks_mean(ctx)) if _anova_applies(ctx) else None
    if r is None:
        return None
    if len(_spreads(ctx)) < _MIN_GROUPS_FOR_TREND:
        # По трём точкам r ≥ 0.8 часто выходит случайно (13 % при 3 × 3, аудит 01-07).
        return Signal("SD трёх групп растут вместе со средними (r = {r} по логарифмам), но"
                      " по трём группам такая связь часто случайна: трансформация отклика"
                      " (log) — под вопросом.", {"r": round(r, 2), "variant": "three"},
                      strength=HINT)
    return Signal("SD групп растут вместе со средними (r = {r} по логарифмам):"
                  " трансформация отклика (log).", {"r": round(r, 2)})


_MIN_GROUPS_FOR_TREND = 4  # групп, с которых связь SD и среднего — уверенный сигнал [BioStat]


@rule("B4.4", "B4", "pre_transform", HINT, "кн. стр. 102, 115; порог BioStat", cls=COMPUTED)
def skewed_residuals(ctx: Context) -> Signal | None:
    fit = factor_fit(ctx) if _anova_applies(ctx) else None
    if fit is None or fit.resid.size < 8:
        return None
    g = float(stats.skew(fit.resid))
    if abs(g) > 1:
        return Signal("Остатки асимметричны (коэффициент асимметрии {g}): попробуйте"
                      " трансформацию отклика.",
                      {"g": round(g, 2), "branch": transform_branch(ctx)})
    return None


@rule("B4.5", "B4", "welch_anova", STRONG, "кн. стр. 107; предл.: Welch 1951", cls=COMPUTED)
@outlier_aware
def heterogeneous_unrelated(ctx: Context) -> Signal | None:
    if len(_groups(ctx)) < 3 or heterogeneous_variances(ctx) is None:
        return None
    if sd_grows_with_mean(ctx) is not None:
        return None
    return Signal("Дисперсии неоднородны и не растут со средним: ANOVA Уэлча"
                  " (книга: взвешенный ANOVA или Уэлч).")


@rule("B4.6", "B4", "anova_permutation_test", STRONG, "кн. стр. 102", cls=COMPUTED)
@outlier_aware
def nonnormal_homogeneous(ctx: Context) -> Signal | None:
    if len(_groups(ctx)) < 3 or nonnormal_residuals(ctx) is None:
        return None
    if heterogeneous_variances(ctx) is not None:
        return None
    return Signal("Остатки ненормальны при однородных дисперсиях: перестановочный тест"
                  " сохранит F и средние.")


# --- A1–A2. Регрессия ---------------------------------------------------------------

def _max_vif(ctx: Context) -> tuple[str, float] | None:
    """Наибольший VIF среди числовых x (все числовые, кроме отклика)."""
    resp = ctx.response
    xs = [c for c in ctx.columns_of("numeric") if c != resp]
    if resp is None or len(xs) < 2:
        return None
    data = np.column_stack([ctx.numbers(c) for c in xs])
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < len(xs) + 2 or np.any(data.std(axis=0) == 0):
        return None
    vif = np.diag(np.linalg.pinv(np.corrcoef(data, rowvar=False)))
    i = int(np.argmax(vif))
    return xs[i], float(vif[i])


@rule("A1.4", "A1", "collinearity", STRONG, "кн. стр. 362", cls=COMPUTED)
def collinear_x(ctx: Context) -> Signal | None:
    res = _max_vif(ctx)
    if res is None or res[1] <= 10:
        return None
    return Signal("Сильная коллинеарность x: VIF «{c}» = {v} > 10 — отбор переменных"
                  " ненадёжен.", {"c": res[0], "v": round(res[1], 1)})


def _lack_of_fit(ctx: Context) -> tuple[float, float] | None:
    """F и p неадекватности прямой по повторам x [кн. стр. 308–312]."""
    data, fit = _xy(ctx), line_fit(ctx)
    if data is None or fit is None:
        return None
    x, y, _ = data
    levels = np.unique(x)
    ss_pe = sum(float(((y[x == lv] - y[x == lv].mean()) ** 2).sum()) for lv in levels)
    df_pe, df_lof = x.size - levels.size, levels.size - 2
    if levels.size < 3 or df_pe < 1 or ss_pe <= 0:
        return None
    ss_lof = float(fit.resid @ fit.resid) - ss_pe
    f = (ss_lof / df_lof) / (ss_pe / df_pe)
    return f, float(stats.f.sf(f, df_lof, df_pe))


@rule("A2.2", "A2", "curves", STRONG, "кн. стр. 312", cls=COMPUTED)
def lack_of_fit(ctx: Context) -> Signal | None:
    res = _lack_of_fit(ctx)
    if res is None or res[1] >= _ALPHA:
        return None
    return Signal("Неадекватность прямой значима (F = {f}, p = {p}): связь не прямая.",
                  {"f": round(res[0], 2), "p": round(res[1], 4)})


def _quadratic_p(ctx: Context) -> float | None:
    """p квадратичного члена в y ~ x + (x − x̄)² [BioStat: критерий дуги]."""
    data = _xy(ctx)
    if data is None or np.unique(data[0]).size < 4:
        return None
    x, y, rows = data
    xc = x - x.mean()
    fit = _ols(y, np.column_stack([np.ones_like(x), x, xc ** 2]), rows)
    if fit is None or fit.X.shape[1] < 3:
        return None
    cov = np.linalg.pinv(fit.X.T @ fit.X) * fit.s2
    beta = np.linalg.lstsq(fit.X, y, rcond=None)[0]
    t = beta[2] / np.sqrt(cov[2, 2])
    return float(2 * stats.t.sf(abs(t), fit.df))


@rule("A2.3", "A2", "curves", STRONG, "кн. стр. 326; критерий BioStat", cls=COMPUTED)
def curved_residuals(ctx: Context) -> Signal | None:
    p = _quadratic_p(ctx)
    if p is None or p >= _ALPHA:
        return None
    return Signal("Остатки прямой изогнуты дугой (квадратичный член p = {p}): кривая.",
                  {"p": round(p, 4)})


@rule("A2.4", "A2", "pre_transform", HINT, "кн. стр. 330; порог BioStat", cls=COMPUTED)
def spread_grows(ctx: Context) -> Signal | None:
    fit = line_fit(ctx)
    if fit is None or fit.y.size < 10:
        return None
    rho, p = stats.spearmanr(np.abs(fit.resid), fit.fitted)
    if rho > 0.5 and p < _ALPHA:
        return Signal("Разброс y растёт вместе с подогнанным значением (ρ = {r}):"
                      " трансформация отклика.", {"r": round(float(rho), 2)})
    return None


def _runs_z(resid: np.ndarray) -> tuple[int, float, float] | None:
    """Тест серий знаков остатков: (серии, ожидаемо, z) [кн. стр. 98]."""
    signs = resid[resid != 0] > 0
    n1, n2 = int(signs.sum()), int((~signs).sum())
    if n1 < 2 or n2 < 2:
        return None
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n = n1 + n2
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    return runs, mu, (runs - mu) / np.sqrt(var) if var > 0 else 0.0


@rule("A2.5", "A2", "trigonometric_regression", HINT, "предл.", cls=COMPUTED)
def periodic_residuals(ctx: Context) -> Signal | None:
    x, fit = ctx.role("x"), line_fit(ctx)
    if not (x and fit) or not names.is_time(x) or fit.y.size < 12:
        return None
    order = np.argsort(fit.X[:, 1], kind="stable")
    res = _runs_z(fit.resid[order])
    if res is None or res[2] >= -2:
        return None
    return Signal("x похож на время, а остатки идут волнами (серий {runs} при ожидаемых"
                  " {exp}): возможна периодичность.", {"runs": res[0], "exp": round(res[1], 1)})


# --- G. GLM ---------------------------------------------------------------------------

def _glm_deviance(ctx: Context) -> tuple[float, int] | None:
    """Остаточный девианс и df GLM по модели вариантов или регрессии."""
    # Лёгкий импорт: statsmodels.api тянет графику и tsa (~0.5 с).
    from statsmodels.genmod import families
    from statsmodels.genmod.generalized_linear_model import GLM

    fit = model_fit(ctx)
    if fit is None or binary_response(ctx) is not None:
        return None
    trials = _trials_column(ctx)
    if trials is None and _trials_guess(ctx) is not None:
        return None  # возможно, доли из m: Пуассон по числу успехов ничего не скажет
    if trials is not None:
        m = ctx.numbers(trials)[fit.rows - 1]
        endog, family = np.column_stack([fit.y, m - fit.y]), families.Binomial()
    elif _count_like(fit.y):
        endog, family = fit.y, families.Poisson()
    else:
        return None
    res = GLM(endog, fit.X, family=family).fit()
    return float(res.deviance), int(res.df_resid)


@rule("G.1", "G", "overdispersion", STRONG, "кн. стр. 488–493; порог BioStat", cls=COMPUTED)
def overdispersion(ctx: Context) -> Signal | None:
    res = _glm_deviance(ctx) if in_glm_branch(ctx) else None
    if res is None or res[1] < 1:
        return None
    dev, df = res
    crit = float(stats.chi2.ppf(1 - _ALPHA, df))
    # На тысячах строк χ² ловит и φ ≈ 1.03; нужна заметная сверхдисперсия [BioStat].
    if dev <= crit or dev / df <= 1.2:
        return None
    return Signal("Остаточный девианс {d} > χ²₀.₉₅({df}) = {crit}: разброс больше, чем"
                  " допускает модель (сверхдисперсия), — оценивайте параметр дисперсии φ"
                  " по данным и сравнивайте по F.",
                  {"d": round(dev, 1), "df": df, "crit": round(crit, 1)})


# --- DIAG -----------------------------------------------------------------------------

def _row_list(rows: np.ndarray, limit: int = 5) -> str:
    shown = ", ".join(str(int(r)) for r in rows[:limit])
    return shown + (" …" if rows.size > limit else "")


def _too_many_large(r: np.ndarray) -> bool:
    """|r| > 2 бывает у ~5 % нормальных остатков: сигнал — если таких заметно
    больше (биномиальный запас 2σ) или есть одиночный экстремум сверх поправки
    Бонферрони (не меньше 3) [BioStat]."""
    n = r.size
    extreme = max(3.0, float(stats.norm.ppf(1 - 0.025 / n)))
    excess = 0.05 * n + 2 * np.sqrt(0.0475 * n)
    return bool(r.max() > extreme or np.sum(r > 2) > excess)


@rule("DIAG.1", "DIAG", "outliers", STRONG, "кн. стр. 95, 109; порог BioStat", cls=COMPUTED)
def outliers(ctx: Context) -> Signal | None:
    fit = model_fit(ctx)
    if fit is None or fit.df < 3:
        return None
    hits = _outlier_candidates(fit)
    if hits.size == 0 or (not ctx.role("x") and _log_scale_clears(ctx, fit, hits)):
        return None  # нет кандидатов, или большие значения — свойство шкалы
    r = np.abs(fit.standardized())
    big = fit.rows[r > 2]
    if _too_many_large(r):
        return Signal("|Стандартизованный остаток| > 2 у {k} {k|наблюдения|наблюдений|наблюдений}"
                      " (строки {rows}):"
                      " проверьте записи.", {"k": int(big.size), "rows": _row_list(big)})
    return Signal("Наблюдение в строке {rows} резко выделяется на фоне остальных (тест"
                  " выброса с поправкой Бонферрони): проверьте запись.",
                  {"rows": _row_list(fit.rows[hits]), "variant": "single"})


# На 10–12 наблюдениях Дарбин–Уотсон выходит за (1, 3) случайно в 8–13 % прогонов
# (аудит 01-07); с 15 — в пределах 5 % [BioStat].
_MIN_N_SERIAL = 15


@rule("DIAG.2", "DIAG", "dead_end_serial", HINT, "кн. стр. 98–100; пороги BioStat",
      cls=COMPUTED)
def serial_correlation(ctx: Context) -> Signal | None:
    fit = model_fit(ctx)
    if fit is None or fit.resid.size < _MIN_N_SERIAL:
        return None
    e = fit.resid  # в порядке строк файла
    dw = float(np.sum(np.diff(e) ** 2) / (e @ e))
    if 1 < dw < 3:
        return None
    return Signal("Остатки по порядку строк коррелированы (Дарбин–Уотсон {dw}):"
                  " серийная корреляция.", {"dw": round(dw, 2)})


@rule("DIAG.4", "DIAG", "regression_diagnostics", STRONG, "кн. стр. 333–336; порог BioStat",
      cls=COMPUTED)
def influential(ctx: Context) -> Signal | None:
    fit = line_fit(ctx)
    if fit is None or fit.df < 2:
        return None
    p, n = fit.X.shape[1], fit.y.size
    r = fit.standardized()
    cook = r ** 2 * fit.leverage / (p * np.clip(1 - fit.leverage, 1e-12, None))
    # 2p/N из книги на тысячах строк превышают и обычные крайние x, поэтому
    # рычаг ещё и > 0.2 (Хубер); влияние — Кук > F₀.₅(p, N − p) (Кук, 1977) [BioStat].
    high = fit.leverage > max(2 * p / n, 0.2)
    flagged = fit.rows[high | (cook > stats.f.ppf(0.5, p, fit.df))]
    if flagged.size == 0:
        return None
    return Signal("Точки с большим рычагом или влиянием (расстояние Кука): строки {rows}"
                  " — проверьте их вклад в наклон.",
                  {"rows": _row_list(flagged)})
