"""Правила-распознаватели карты (decision-tree-spec v2, раздел 5).

Каждая функция — одна строка реестра: смотрит на данные и роли колонок,
возвращает :class:`Signal` (сработало) или None. Структурные правила
реализованы здесь; расчётные (нормальность, Бартлетт, неадекватность,
сверхдисперсия) — в ``recognizers_computed``.

Пороги с пометкой [BioStat] — эвристики, в книге их нет.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from core.validation.recognize import (
    HINT, STRONG, STRUCTURAL, Context, Signal, declare, rule,
)

# --- Помощники ----------------------------------------------------------------

_BLOCK_NAME = re.compile(r"^(block|blk|bloco|rep|replicate|блок|повтор(ность)?)\b", re.I)
_CONTROL = re.compile(r"^(control|ctrl|check|standard|untreated|testemunha|"
                      r"контроль|контр|стандарт)$", re.I)
_BEFORE = re.compile(r"(before|pre|до|t0|day ?1|день ?1)$", re.I)
_AFTER = re.compile(r"(after|post|после|t1|day ?2|день ?2)$", re.I)
_TIME = re.compile(r"(time|day|date|week|month|year|время|день|дата|неделя|месяц|год|сутки|mês|mes)", re.I)
_PERCENT = re.compile(r"(%|pct|percent|процент|доля)", re.I)


def _valid(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


def _is_int(arr: np.ndarray) -> bool:
    v = _valid(arr)
    return v.size > 0 and bool(np.all(v == np.round(v)))


def _count_like(v: np.ndarray) -> bool:
    """Похоже на счёты: целые ≥ 0, не только 0/1, есть малые значения.

    Условие «минимум ≤ 10» [BioStat] отсекает целочисленные измерения
    вроде урожая в граммах (333–708): у счётов почти всегда есть малые.
    """
    return (_is_int(v) and v.min() >= 0 and v.min() <= 10
            and not set(np.unique(v)) <= {0.0, 1.0})


def _score_like(v: np.ndarray) -> bool:
    """Немного целых значений в узком диапазоне — похоже на баллы [BioStat]."""
    return _is_int(v) and 2 < np.unique(v).size <= 10 and np.ptp(v) <= 10


def _pairs(xs: list[str], ys: list[str]) -> Counter:
    """Число строк в каждой паре меток (x, y); пустые метки пропускаются."""
    return Counter((x, y) for x, y in zip(xs, ys) if x and y)


def _cells(ctx: Context, a: str, b: str) -> Counter:
    """Число строк в каждой паре уровней колонок (a, b)."""
    return _pairs(ctx.labels(a), ctx.labels(b))


def _treatment_columns(ctx: Context) -> list[str]:
    """Колонки факторов вариантов: основной, второй, третий (без повторов)."""
    names = [ctx.factor, ctx.role("factor_b"), ctx.role("factor_c")]
    return [n for i, n in enumerate(names) if n and n not in names[:i]]


def _treatment_labels(ctx: Context) -> list[str]:
    """Метка варианта по строкам: комбинация уровней всех факторов («a1|b2»).

    Вариант в факторном опыте — комбинация уровней, а не уровень одного
    фактора [кн. гл. 8]; пустая метка любого фактора — пропуск ("").
    """
    def build() -> list[str]:
        cols = [ctx.labels(c) for c in _treatment_columns(ctx)]
        return ["|".join(row) if all(row) else "" for row in zip(*cols)]

    return ctx.memo("treatment_labels", build)


def _levels(ctx: Context, column: str) -> list[str]:
    return sorted({v for v in ctx.labels(column) if v})


def _response_values(ctx: Context) -> np.ndarray | None:
    col = ctx.response
    if col is None or col not in ctx.columns_of("numeric"):
        return None
    v = _valid(ctx.numbers(col))
    return v if v.size >= 2 else None


# --- R2. Тип отклика ----------------------------------------------------------

@rule("R2.1", "R2", "G1", STRONG, "кн. стр. 501")
def binary_response(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is not None and set(np.unique(v)) <= {0.0, 1.0}:
        return Signal("Отклик принимает только 0 и 1: бинарные данные (доля из m = 1).")
    return None


@rule("R2.2", "R2", "G2", HINT, "кн. стр. 102, 502; порог BioStat")
def count_response(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is None or not _count_like(v):
        return None
    return Signal("Отклик — неотрицательные целые: похоже на счёты (GLM, Пуассон).")


def _successes_and_trials(ctx: Context, col: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Пары (успехи, m) по полным строкам, если колонка годится как число испытаний.

    Отклик и m — целые, отклик ≥ 0, m ≥ 1 и m ≥ отклика в каждой строке.
    """
    resp = ctx.response
    if resp is None or col == resp:
        return None
    v, m = ctx.numbers(resp), ctx.numbers(col)
    ok = ~np.isnan(v) & ~np.isnan(m)
    v, m = v[ok], m[ok]
    if v.size < 2 or not (_is_int(v) and _is_int(m)) or v.min() < 0:
        return None
    return (v, m) if np.all(m >= v) and np.all(m >= 1) else None


def _trials_column(ctx: Context) -> str | None:
    """Колонка числа испытаний m — только та, что назначена в слот ``trials``.

    Перебора колонок здесь нет: посторонняя целая колонка (год, номер делянки)
    тоже ≥ отклика в каждой строке. Догадка без слота — :func:`_trials_guess`.
    """
    col = ctx.role("trials")
    return col if col and _successes_and_trials(ctx, col) is not None else None


def _variable_columns(ctx: Context, keep_response: bool = False) -> list[str]:
    """Числовые колонки, годные как переменные: не константа и не номера подряд.

    Год сбора, один на весь файл, и номер делянки — не переменные анализа.
    """
    out = []
    for col in ctx.columns_of("numeric"):
        v = _valid(ctx.numbers(col))
        if col == ctx.response and not keep_response:
            continue
        if v.size >= 2 and np.ptp(v) > 0 and not _index_like(v):
            out.append(col)
    return out


# Методы ветки G: только в их форме считаются сверхдисперсия и модель счётов.
_GLM_METHODS = frozenset({"glm_analysis", "glmm_analysis", "count_regression"})


def in_glm_branch(ctx: Context) -> bool:
    """Пользователь анализирует долю или счёт: форма метода GLM или его слоты.

    Целые измерения (время, баллы) похожи на счёты; пока человек сравнивает
    средние, сверхдисперсия пуассоновской модели ему ни о чём не говорит.
    """
    return ctx.method in _GLM_METHODS or ctx.role("count", "trials") is not None


def _index_like(m: np.ndarray) -> bool:
    """Все значения разные и идут подряд: номера делянок или строк, а не m."""
    return m.size > 2 and np.unique(m).size == m.size and np.ptp(m) == m.size - 1


def _trials_guess(ctx: Context) -> str | None:
    """Колонка, похожая на число испытаний, когда слот ``trials`` не задан [предл.].

    Кандидат — числовая колонка без роли, где m ≥ отклика везде и m > отклика
    хотя бы в одной строке; не год (все значения в 1900–2100), не номера
    подряд и не единицы (бинарные данные). Постоянное m допустимо: одинаковое
    число семян или особей на единицу — обычный случай.
    """
    if ctx.role("trials"):
        return None
    taken = set(ctx.roles.values())
    for col in ctx.columns_of("numeric"):
        pair = None if col in taken else _successes_and_trials(ctx, col)
        if pair is None:
            continue
        v, m = pair
        year = m.min() >= 1900 and m.max() <= 2100
        if np.any(m > v) and not year and not _index_like(m) and not np.all(m == 1):
            return col
    return None


@rule("R2.3", "R2", "G1", STRONG, "предл.")
def proportion_with_trials(ctx: Context) -> Signal | None:
    col = _trials_column(ctx)
    if col and not np.all(_valid(ctx.numbers(col)) == 1):
        return Signal("Колонка «{m}» ≥ отклика в каждой строке: отклик — число успехов из m.",
                      {"m": col})
    return None


@rule("R2.12", "R2", "G1", HINT, "предл.")
def possible_trials(ctx: Context) -> Signal | None:
    col = _trials_guess(ctx)
    if col is None:
        return None
    return Signal("Колонка «{m}» может быть числом испытаний m: укажите её в слоте"
                  " «Число испытаний m», если отклик — успехи из m.", {"m": col})


@rule("R2.4", "R2", "G1", HINT, "кн. стр. 120")
def proportion_scale(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is None or _is_int(v):
        return None
    in_unit = v.min() >= 0 and v.max() <= 1
    named = bool(_PERCENT.search(ctx.response or "")) and v.min() >= 0 and v.max() <= 100
    if in_unit or named:
        return Signal("Значения похожи на доли или проценты: биномиальная GLM или logit.")
    return None


@rule("R2.5", "R2", "ordinal_or_contingency", STRONG, "кн. стр. 513")
def label_response(ctx: Context) -> Signal | None:
    resp = ctx.response
    if resp and resp in ctx.columns_of("categorical"):
        return Signal("Отклик — метки, а не числа: порядковая регрессия или таблица частот.",
                      {"variant": "response"})
    if resp is None and not ctx.columns_of("numeric") and len(ctx.columns_of("categorical")) >= 2:
        return Signal("В файле только колонки меток: таблица сопряжённости или порядковые баллы.",
                      {"variant": "file"})
    return None


@rule("R2.6", "R2", "count_regression", HINT, "кн. стр. 126, 512; порог BioStat")
def many_zeros(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is None or not _count_like(v):
        return None
    share = float(np.mean(v == 0))
    if share < _ZERO_SHARE or len(np.unique(v)) <= 2:
        return None
    if share >= _ZERO_SHARE_STRONG:
        return Signal("Нулей {share}%: счёты с избытком нулей — модель для избыточных"
                      " нулей, а не ANOVA и не трансформация.",
                      {"share": round(100 * share), "variant": "half"}, strength=STRONG)
    return Signal("Нулей {share}%: возможно избыточное число нулей.",
                  {"share": round(100 * share)})


_ZERO_SHARE = 0.3  # доля нулей, с которой R2.6 подсказывает [BioStat]
_ZERO_SHARE_STRONG = 0.5  # с которой указывает уверенно [BioStat; кн. стр. 126, 512]


@rule("R2.7", "R2", "pre_log", HINT, "кн. стр. 115; порог BioStat")
def right_skew(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is None or v.size < 8 or v.min() <= 0:
        return None
    skew = float(stats.skew(v))
    if skew > 1:
        return Signal("Все значения > 0, асимметрия {skew}: кандидат на log-преобразование.",
                      {"skew": round(skew, 2)})
    return None


@rule("R2.8", "R2", "dead_end_censored", HINT, "кн. стр. 274; предл.")
def detection_limit(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if v is None or v.size < 10:
        return None
    low = v.min()
    share = float(np.mean(v == low))
    # Целые измерения совпадают и без предела обнаружения; два совпадения — случайность.
    if share >= 0.1 and np.sum(v == low) >= 3 and not _is_int(v):
        return Signal("{share}% значений равны минимуму {low}: возможен предел обнаружения.",
                      {"share": round(100 * share), "low": float(low)})
    return None


@rule("R2.9", "R2", "G1", STRONG, "кн. стр. 126")
def small_trials(ctx: Context) -> Signal | None:
    col = _trials_column(ctx)
    if col is None:
        return None
    m = _valid(ctx.numbers(col))
    if 1 < np.median(m) < 10:
        return Signal("Доли из малых выборок (медиана m = {m}): GLM, а не logit.",
                      {"m": float(np.median(m))})
    return None


@rule("R2.10", "R2", "pre_logit", HINT, "кн. стр. 502")
def large_trials(ctx: Context) -> Signal | None:
    col = _trials_column(ctx)
    if col is None:
        return None
    v, m = _successes_and_trials(ctx, col)
    p = v / m
    if m.min() > 20 and np.mean((p > 0.1) & (p < 0.9)) >= 0.9:
        return Signal("m > 20, доли не у краёв: допустим и logit с линейной моделью.")
    return None


@rule("R2.11", "R2", "numeric_codes", HINT, "Witte стр. 406; порог BioStat")
def numeric_codes(ctx: Context) -> Signal | None:
    """Отклик — подряд идущие целые 0/1…k: возможно, коды классов (Witte, 21.4).

    Witte: смотреть на одно наблюдение — количество это или класс. Порог BioStat:
    3–10 кодов без пропусков в ряду и в среднем ≥ 5 строк на код (f_e ≥ 5, стр. 380).
    """
    v = _response_values(ctx)
    if v is None or not _is_int(v):
        return None
    codes = np.unique(v)
    lo, k = int(codes[0]), codes.size
    if not (3 <= k <= 10 and lo in (0, 1) and np.array_equal(codes, np.arange(lo, lo + k))):
        return None
    if v.size < 5 * k:
        return None
    return Signal("«{c}» — целые {lo}…{hi}: если это коды классов, а не количества или баллы,"
                  " — частоты и χ².", {"c": ctx.response, "lo": lo, "hi": lo + k - 1})


# --- B0. Связанность измерений -------------------------------------------------

@rule("B0.1", "B0", "paired", HINT, "предл.")
def before_after_names(ctx: Context) -> Signal | None:
    nums = ctx.columns_of("numeric")
    before = [c for c in nums if _BEFORE.search(c.strip())]
    after = [c for c in nums if _AFTER.search(c.strip())]
    if before and after:
        return Signal("Колонки «{a}» и «{b}» похожи на измерения до/после.",
                      {"a": before[0], "b": after[0]})
    return None


@rule("B0.2", "B0", "paired", HINT, "предл.; порог BioStat")
def correlated_twins(ctx: Context) -> Signal | None:
    nums = ctx.columns_of("numeric")
    if len(nums) < 2 or len(nums) > 12:
        return None
    for a, b in combinations(nums, 2):
        x, y = ctx.numbers(a), ctx.numbers(b)
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            continue
        ratio = np.mean(y[ok]) / np.mean(x[ok]) if np.mean(x[ok]) else 0
        if 0.5 < ratio < 2 and np.corrcoef(x[ok], y[ok])[0, 1] > 0.7:
            return Signal("«{a}» и «{b}» в одной шкале и сильно связаны: возможно, это"
                          " повторные измерения одних объектов.", {"a": a, "b": b})
    return None


@rule("B0.3", "B0", "mixed_model", HINT, "кн. стр. 100")
def repeated_over_time(ctx: Context) -> Signal | None:
    time_cols = [c for c in ctx.table.header if _TIME.search(c)]
    ids = [p.name for p in ctx.table.profiles
           if p.detected_type.value in ("categorical", "id") and p.n_unique < p.n_total
           and p.name not in time_cols]
    for t in time_cols:
        hit = next((o for o in ids if _measured_repeatedly(ctx, o, t)), None)
        if hit:
            return Signal("Колонка времени «{t}» и повторяющиеся объекты «{o}»: повторные"
                          " измерения, наблюдения коррелированы.", {"t": t, "o": hit})
    return None


def _measured_repeatedly(ctx: Context, obj: str, time: str) -> bool:
    """Объект измерен в несколько моментов, в каждый — не более раза [BioStat].

    Фактор вариантов с повторностями встречается в один момент много раз, а
    при одном значении времени (год сбора) повторных измерений нет вовсе.
    """
    cells = _cells(ctx, obj, time)
    if not cells or set(cells.values()) != {1}:
        return False
    times = Counter(o for o, _ in cells)
    return min(times.values()) >= 2


# --- B1. Структура единиц ------------------------------------------------------

@rule("B0.4", "B0", "wilcoxon", HINT, "Witte стр. 405; порог BioStat")
def paired_scores(ctx: Context) -> Signal | None:
    """Пары «до/после» в баллах → T Уилкоксона (Witte, Fig. 21.1)."""
    a, b = ctx.role("before"), ctx.role("after")
    if not (a and b):
        return None
    x, y = _valid(ctx.numbers(a)), _valid(ctx.numbers(b))
    if x.size < 2 or y.size < 2 or not (_score_like(x) and _score_like(y)):
        return None
    return Signal("«{a}» и «{b}» — немного целых значений (баллы?): для пар — знаково-"
                  "ранговый тест Уилкоксона.", {"a": a, "b": b})


@rule("B1.1", "B1", "blocks", HINT, "предл.")
def block_name(ctx: Context) -> Signal | None:
    if ctx.role("block"):
        return None
    named = [c for c in ctx.table.header if _BLOCK_NAME.search(c.strip())]
    if named:
        return Signal("Колонка «{c}» похожа на блоки / повторности.", {"c": named[0]})
    return None


@dataclass(frozen=True)
class Layout:
    """Раскладка вариантов по блокам: строки в ячейках, списки уровней."""

    cells: Counter  # (вариант, блок) -> число строк
    treatments: list[str]
    blocks: list[str]
    n_factors: int

    @property
    def complete(self) -> bool:
        return len(self.cells) == len(self.treatments) * len(self.blocks)


def _layout(ctx: Context) -> Layout | None:
    """Ячейки «вариант × блок», где вариант — комбинация всех факторов."""
    blk = ctx.role("block")
    if not (ctx.factor and blk):
        return None
    cells = _pairs(_treatment_labels(ctx), ctx.labels(blk))
    return Layout(cells, sorted({t for t, _ in cells}), sorted({b for _, b in cells}),
                  len(_treatment_columns(ctx)))


@rule("B1.2", "B1", "rcbd_anova", STRONG, "кн. стр. 58")
def rcbd_layout(ctx: Context) -> Signal | None:
    lay = _layout(ctx)
    if lay is None or not lay.complete or set(lay.cells.values()) != {1}:
        return None
    t, b = len(lay.treatments), len(lay.blocks)
    if lay.n_factors == 1:
        return Signal("Каждый вариант ровно раз в каждом блоке ({t} × {b}): RCBD.",
                      {"t": t, "b": b})
    if split_plot_layout(ctx) is not None:  # блоки те же, но рандомизация в два этапа
        return None
    return Signal("Каждая комбинация факторов ровно раз в каждом блоке ({t} × {b}):"
                  " факторный опыт в полных блоках.",
                  {"t": t, "b": b, "method": "factorial_anova", "variant": "factorial"})


@rule("B1.3", "B1", "rcbd_subsampling_anova", STRONG, "кн. стр. 46, 145")
def subsamples(ctx: Context) -> Signal | None:
    lay = _layout(ctx)
    if lay is None:
        return None
    counts = set(lay.cells.values())
    if len(counts) == 1 and (k := counts.pop()) > 1:
        return Signal("В каждой ячейке блок × вариант {k} строк: субвыборки"
                      " (анализ на уровне делянок, не строк).", {"k": k})
    return None


@rule("B1.4", "B1", "bibd_anova", STRONG, "кн. стр. 232–234")
def bibd_layout(ctx: Context) -> Signal | None:
    lay = _layout(ctx)
    if lay is None:
        return None
    cells, trts = lay.cells, lay.treatments
    by_block: dict[str, set] = {}
    for (t, b) in cells:
        by_block.setdefault(b, set()).add(t)
    sizes = {len(s) for s in by_block.values()}
    if set(cells.values()) != {1} or len(sizes) != 1 or sizes.pop() >= len(trts):
        return None
    lam = {sum(1 for s in by_block.values() if {a, c} <= s) for a, c in combinations(trts, 2)}
    if len(lam) == 1 and (lam_val := lam.pop()) > 0:
        return Signal("Блок меньше числа вариантов, каждая пара вместе {lam} раз: BIBD.",
                      {"lam": lam_val})
    return None


@rule("B1.5", "B1", "latin_square_anova", STRONG, "кн. стр. 209")
def latin_layout(ctx: Context) -> Signal | None:
    trt, row, col = ctx.factor, ctx.role("row"), ctx.role("column")
    if not (trt and row and col):
        return None
    t = len(_levels(ctx, trt))
    ok = all(len(c) == t * t and set(c.values()) == {1}
             for c in (_cells(ctx, trt, row), _cells(ctx, trt, col)))
    if ok and len(_levels(ctx, row)) == len(_levels(ctx, col)) == t:
        return Signal("Каждый вариант ровно раз в каждом ряду и колонке: латинский квадрат.")
    return None


@rule("B1.6", "B1", "split_plot_anova", STRONG, "кн. стр. 220")
def split_plot_layout(ctx: Context) -> Signal | None:
    a, b = ctx.role("factor_a"), ctx.role("factor_b")
    unit = ctx.role("whole_plot", "unit")
    if not (a and b and unit):
        return None
    a_in_unit = {u: set() for u in _levels(ctx, unit)}
    b_in_unit = {u: set() for u in _levels(ctx, unit)}
    for u, x, y in zip(ctx.labels(unit), ctx.labels(a), ctx.labels(b)):
        if u:
            a_in_unit[u].add(x)
            b_in_unit[u].add(y)
    if all(len(s) == 1 for s in a_in_unit.values()) and all(
            len(s) > 1 for s in b_in_unit.values()):
        return Signal("«{a}» постоянен внутри «{u}», «{b}» меняется: split-plot.",
                      {"a": a, "b": b, "u": unit})
    return None


@rule("B1.7", "B1", "unbalanced", STRONG, "кн. стр. 258, 427")
def unequal_replication(ctx: Context) -> Signal | None:
    lay = _layout(ctx)
    if lay is None:
        return None
    if len(set(lay.cells.values())) > 1 or (not lay.complete and not bibd_layout(ctx)):
        return Signal("Повторность в ячейках блок × вариант неравная или есть пустые"
                      " ячейки: несбалансированный анализ или смешанная модель.")
    return None


# --- B2. Структура вариантов ---------------------------------------------------

def _crossed(ctx: Context) -> tuple[Counter, int] | None:
    a, b = ctx.role("factor_a"), ctx.role("factor_b")
    if not (a and b):
        return None
    full = len(_levels(ctx, a)) * len(_levels(ctx, b))
    return _cells(ctx, a, b), full


@rule("B2.1", "B2", "factorial_anova", STRONG, "кн. стр. 172")
def full_factorial(ctx: Context) -> Signal | None:
    crossed = _crossed(ctx) if _response_values(ctx) is not None else None
    if crossed and len(crossed[0]) == crossed[1] and len(set(crossed[0].values())) == 1:
        return Signal("Все комбинации факторов с равной повторностью: факторный анализ.")
    return None


@rule("B2.2", "B2", "unbalanced_anova", STRONG, "кн. стр. 258")
def unbalanced_factorial(ctx: Context) -> Signal | None:
    # Без числового отклика это таблица частот: неравные ячейки — сами данные.
    crossed = _crossed(ctx) if _response_values(ctx) is not None else None
    if crossed and (len(crossed[0]) < crossed[1] or len(set(crossed[0].values())) > 1):
        missing = crossed[1] - len(crossed[0])
        return Signal("Комбинации факторов неполные (пустых: {missing}) или повторность"
                      " неравная: несбалансированный факторный анализ.", {"missing": missing})
    return None


@rule("B2.3", "B2", "nested_anova", STRONG, "кн. стр. 173")
def nested_members(ctx: Context) -> Signal | None:
    group = ctx.role("group", "factor_a")
    member = ctx.role("member", "factor_b")
    if not (group and member) or group == member:
        return None
    parents: dict[str, set] = {}
    for g, m in _cells(ctx, group, member):
        parents.setdefault(m, set()).add(g)
    if len(parents) > 1 and all(len(p) == 1 for p in parents.values()):
        return Signal("Каждый уровень «{m}» встречается только в одной группе «{g}»:"
                      " вложенная структура.", {"m": member, "g": group})
    return None


@rule("B2.4", "B2", "control_present", HINT, "кн. стр. 179; предл.")
def control_level(ctx: Context) -> Signal | None:
    for col in filter(None, (ctx.role(s) for s in ("group", "factor_a", "factor_b"))):
        hit = next((lv for lv in _levels(ctx, col) if _CONTROL.match(lv)), None)
        if hit:
            return Signal("В «{c}» есть уровень «{lv}»: контроль.", {"c": col, "lv": hit})
    return None


@rule("B2.5", "B2", "ancova", HINT, "кн. стр. 409")
def covariate_candidate(ctx: Context) -> Signal | None:
    resp, trt = ctx.response, ctx.factor
    if not (resp and trt) or ctx.role("covariate"):
        return None
    others = [c for c in _variable_columns(ctx)
              if c not in ctx.roles.values() and not _BLOCK_NAME.search(c)]
    if others:
        return Signal("Есть ещё числовая колонка «{c}»: если это сопутствующая"
                      " переменная — ковариационный анализ.", {"c": others[0]})
    return None


@rule("B2.6", "B2", "factor_codes", HINT, "предл.: Witte стр. 406")
def numeric_factor_codes(ctx: Context) -> Signal | None:
    """Числовая колонка с немногими повторяющимися целыми — возможно, номера вариантов.

    Пока тип колонки «числовая», методы для групп файлу не предлагаются.
    Срабатывает, только если фактор ещё не назначен.
    """
    if ctx.factor:
        return None
    for col in ctx.columns_of("numeric"):
        v = _valid(ctx.numbers(col))
        if col in ctx.roles.values() or v.size < 6 or not _is_int(v):
            continue
        levels, counts = np.unique(v, return_counts=True)
        if 2 <= levels.size <= _MAX_CODE_LEVELS and counts.min() >= 2:
            return Signal("В числовой колонке «{c}» всего {k} разных значений, каждое"
                          " повторяется: если это номера вариантов, смените её тип на"
                          " «категория» — станут доступны методы сравнения групп.",
                          {"c": col, "k": int(levels.size)})
    return None


_MAX_CODE_LEVELS = 10  # больше — скорее измерение, чем номера вариантов [BioStat]


# --- B3–B5. Уровни, допущения (структурная часть), сравнения ---------------------

@rule("B3.1", "B3", "levels", STRONG, "—")
def level_count(ctx: Context) -> Signal | None:
    if ctx.factor is None:
        return None
    t = len(_levels(ctx, ctx.factor))
    if t >= 2:
        return Signal("Уровней фактора: {t}.", {"t": t, "branch": "B4a" if t == 2 else "B4b"})
    return None


def _group_sizes(ctx: Context) -> list[int]:
    resp, trt = ctx.response, ctx.factor
    if not (resp and trt):
        return []
    v = ctx.numbers(resp)
    labels = ctx.labels(trt)
    counts = Counter(lab for lab, x in zip(labels, v) if lab and not np.isnan(x))
    return list(counts.values())


def _treatment_sizes(ctx: Context) -> list[int]:
    """Число наблюдений отклика в каждой комбинации факторов."""
    resp = ctx.response
    if not (resp and ctx.factor):
        return []
    v = ctx.numbers(resp)
    counts = Counter(t for t, x in zip(_treatment_labels(ctx), v) if t and not np.isnan(x))
    return list(counts.values())


@rule("B4.7", "B4", "clt_note", HINT, "BioStat")
def large_groups(ctx: Context) -> Signal | None:
    sizes = _group_sizes(ctx)
    if sizes and min(sizes) > 30:
        return Signal("Во всех группах n > 30: при отклонении от нормальности параметрика"
                      " всё ещё вероятно допустима (эвристика ЦПТ, не правило книги).")
    return None


@rule("B4.8", "B4", "kruskal_wallis", HINT, "кн. стр. 127; Witte стр. 405; порог BioStat")
def score_response(ctx: Context) -> Signal | None:
    """Баллы → ранговый тест по числу групп (Witte, Fig. 21.1): 2 — U, 3+ — H."""
    v = _response_values(ctx)
    if v is None:
        return None
    if _score_like(v):
        method = "mann_whitney" if len(_group_sizes(ctx)) == 2 else "kruskal_wallis"
        return Signal("Отклик — немного целых значений (баллы?): ранговые методы или"
                      " порядковая регрессия.", {"method": method})
    return None


@rule("B5.1", "B5", "polynomial_contrasts", HINT, "кн. стр. 87, 190")
def numeric_levels(ctx: Context) -> Signal | None:
    trt = ctx.factor
    if trt is None:
        return None
    levels = _levels(ctx, trt)
    numeric = [x for x in map(_to_float, levels) if x is not None]
    if len(levels) >= 3 and len(numeric) == len(levels) and not _code_like(numeric):
        return Signal("Уровни «{c}» — числа (дозы?): тренд по дозе вместо попарных"
                      " сравнений.", {"c": trt})
    return None


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _code_like(values: list[float]) -> bool:
    """Номера 1 … k или 0 … k − 1 подряд: коды вариантов, а не дозы."""
    ordered = sorted(values)
    return ordered[0] in (0.0, 1.0) and all(b - a == 1 for a, b in zip(ordered, ordered[1:]))


@rule("B5.2", "B5", "dunnett", HINT, "кн. стр. 201")
def control_for_dunnett(ctx: Context) -> Signal | None:
    hit = control_level(ctx)
    if hit and len(_levels(ctx, hit.params["c"])) >= 3:  # при двух уровнях сравнение одно
        return Signal("Есть контроль «{lv}»: сравнения каждого варианта с контролем"
                      " (Даннет).", {"lv": hit.params["lv"]})
    return None


@rule("B5.3", "B5", "model_tables", HINT, "кн. стр. 205")
def structured_treatments(ctx: Context) -> Signal | None:
    if ctx.role("factor_b", "member"):
        return Signal("Варианты структурированы: сравнивать по таблицам средних модели,"
                      " а не все пары подряд.")
    return None


# --- A1–A2. Регрессия -----------------------------------------------------------

@rule("A1.1", "A1", "multiple_regression", HINT, "кн. гл. 14")
def several_numeric(ctx: Context) -> Signal | None:
    n = len(_variable_columns(ctx, keep_response=True))
    if n >= 3:
        return Signal("Числовых колонок {n}: возможна множественная регрессия.", {"n": n})
    return None


@rule("A1.2", "A1", "regression_with_groups", HINT, "кн. стр. 382")
def x_with_groups(ctx: Context) -> Signal | None:
    """Группа, назначенная в форме, — уверенно; найденная в файле — подсказка."""
    if not ctx.role("x"):
        return None
    if ctx.factor:
        return Signal("Есть колонка групп «{g}»: линии по группам.", {"g": ctx.factor},
                      strength=STRONG)
    free = [c for c in ctx.columns_of("categorical") if c not in ctx.roles.values()]
    if free:
        return Signal("Есть колонка групп «{g}»: линии по группам.", {"g": free[0]})
    return None


@rule("A1.3", "A1", "blocked_regression", HINT, "кн. стр. 406")
def x_with_blocks(ctx: Context) -> Signal | None:
    if not ctx.role("x"):
        return None
    target = {"branch": "A1.blocks"}  # прямая или кривая в блоках — решает форма связи
    if ctx.role("block"):  # блок назначен в форме: структура опыта известна
        return Signal("Есть блоки: регрессия внутри блоков.", target, strength=STRONG)
    if any(_BLOCK_NAME.search(c.strip()) for c in ctx.table.header):
        return Signal("Есть блоки: регрессия внутри блоков.", target)
    return None


@rule("A1.5", "A1", "selection", STRONG, "кн. стр. 366")
def selection_size(ctx: Context) -> Signal | None:
    k = len(_variable_columns(ctx, keep_response=True)) - 1
    if k < 3:
        return None
    if k <= 12:
        return Signal("Кандидатов x: {k} — можно перебрать все подмножества.",
                      {"k": k, "method": "regression_subsets", "variant": "subsets"})
    return Signal("Кандидатов x: {k} — пошаговый отбор.",
                  {"k": k, "method": "stepwise_selection", "variant": "stepwise"})


@rule("A2.1", "A2", "lack_of_fit", STRONG, "кн. стр. 308")
def repeated_x(ctx: Context) -> Signal | None:
    x = ctx.role("x")
    if x is None:
        return None
    v = _valid(ctx.numbers(x))
    levels, counts = np.unique(v, return_counts=True)
    if levels.size >= 3 and counts.max() > 1:
        return Signal("Значения x повторяются ({k} уровней): можно проверить неадекватность"
                      " прямой.", {"k": int(levels.size)})
    return None


# --- G. GLM ---------------------------------------------------------------------

@rule("G.2", "G", "count_regression", HINT, "кн. стр. 512; порог BioStat")
def overdispersed_counts(ctx: Context) -> Signal | None:
    v = _response_values(ctx)
    if not in_glm_branch(ctx) or v is None or v.size < 10 or not _count_like(v) or v.mean() == 0:
        return None
    ratio = float(v.var(ddof=1) / v.mean())
    if ratio > 2:
        return Signal("Дисперсия счётов в {r} раз больше среднего: возможна"
                      " отрицательная биномиальная модель.", {"r": round(ratio, 1)})
    return None


@rule("G.3", "G", "binary_note", STRONG, "кн. стр. 501")
def binary_trials(ctx: Context) -> Signal | None:
    if binary_response(ctx) is not None:
        return Signal("Бинарные данные: φ не оценивается; по возможности суммируйте по"
                      " единицам опыта.")
    return None


@rule("G.4", "G", "glmm_or_block", HINT, "предл.")
def random_grouping(ctx: Context) -> Signal | None:
    stratum = ctx.role("stratum_1")
    if stratum is None:
        return None
    k = len(_levels(ctx, stratum))
    if k >= 5:
        return Signal("У «{c}» {k} уровней: подходит как случайный фактор (GLMM).",
                      {"c": stratum, "k": k, "variant": "glmm"})
    return Signal("У «{c}» всего {k} уровней: лучше как фиксированный блок в GLM.",
                  {"c": stratum, "k": k, "variant": "block"})


# --- DIAG -----------------------------------------------------------------------

@rule("DIAG.3", "DIAG", "power_note", STRONG, "кн. стр. 521")
def residual_df(ctx: Context) -> Signal | None:
    sizes = _treatment_sizes(ctx)
    if not sizes:
        return None
    n, t = sum(sizes), len(sizes)
    df = n - t
    if (lay := _layout(ctx)) and rcbd_layout(ctx):
        df = (len(lay.treatments) - 1) * (len(lay.blocks) - 1)
    if 0 < df < 10:
        return Signal("Остаточных df {df} < 10: мощность сравнений мала.", {"df": df})
    return None


# --- Проверяемые самим методом ----------------------------------------------------

declare("A2.6", "A2", "extrapolation_note", STRONG, "кн. стр. 304, 459", cls=STRUCTURAL)

# Расчётные правила (этап 2 карты) — отдельным модулем; импорт в конце, потому
# что они пользуются помощниками этого модуля.
from core.validation import recognizers_computed  # noqa: E402, F401
