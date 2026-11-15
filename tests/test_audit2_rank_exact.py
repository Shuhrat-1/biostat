"""Аудит 02-02: точное перестановочное P ранговых тестов с рангами при совпадениях.

Эталон — полный перебор (`audit-02/f_ranks.py`): все расстановки знаков для
знаково-рангового теста, все разбиения на группы для Манна–Уитни.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from core.stats import rank_exact
from core.stats.rank_exact import rank_sum_p, signed_rank_p
from tests.bridge_runner import codes, run

PAIRS = {"before": "before", "after": "after"}
GROUPS = {"value": "y", "group": "g"}
WEB = Path(__file__).resolve().parents[1] / "web/src"


def _brute_signed_rank(diff: list[float]) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    observed, total = min(r[d > 0].sum(), r[d < 0].sum()), r.sum()
    hits = sum(1 for s in itertools.product((False, True), repeat=d.size)
               if min(w := r[np.array(s)].sum(), total - w) <= observed + 1e-9)
    return hits / 2 ** d.size


def _brute_rank_sum(a: list[float], b: list[float], alternative: str = "two-sided") -> float:
    r = stats.rankdata(np.concatenate([a, b]))
    n, mean = len(a), len(a) * (len(a) + len(b) + 1) / 2
    sums = np.array([r[list(c)].sum() for c in itertools.combinations(range(r.size), n)])
    if alternative == "greater":
        return float((sums >= r[:n].sum() - 1e-9).mean())
    if alternative == "less":
        return float((sums <= r[:n].sum() + 1e-9).mean())
    return float((np.abs(sums - mean) >= abs(r[:n].sum() - mean) - 1e-9).mean())


def _two_groups(a: list[float], b: list[float]) -> dict[str, list]:
    return {"g": ["a"] * len(a) + ["b"] * len(b), "y": list(a) + list(b)}


# --- Таблица находки 02-02: было (нормальное приближение) → точное ---------------------------

@pytest.mark.parametrize(("before", "after", "exact", "was"), [
    # 6 пар, одна нулевая разность, остальные 5 одного знака: меньше 2/2⁵ недостижимо
    ([10, 12, 9, 14, 11, 13], [10, 13.5, 11, 17, 15.5, 18], 0.0625, 0.043),
    # 7 пар, баллы 1–5
    ([2, 3, 1, 4, 2, 3, 2], [3, 4, 3, 5, 3, 5, 2], 0.03125, 0.023),
    # контроль: 8 пар без нулей и совпадений
    ([10, 12, 9, 14, 11, 13, 8, 15], [11.1, 14.2, 12.3, 18.4, 16.5, 19.6, 15.7, 14.2],
     0.015625, 0.0156),
])
def test_wilcoxon_p_is_exact_with_ties_and_zeros(before, after, exact, was) -> None:
    res = run("wilcoxon", {"before": before, "after": after}, PAIRS, {})
    assert res["p_value"] == pytest.approx(exact, abs=1e-12)
    assert res["parameters"]["p_method"] == "exact"
    assert exact >= was - 1e-4  # приближение занижало P
    assert exact == pytest.approx(_brute_signed_rank(np.subtract(before, after)), abs=1e-12)


@pytest.mark.parametrize(("a", "b", "exact"), [
    ([1, 2, 3], [4, 5, 5], 0.100),               # 3 и 3 с совпадением: было 0.077
    ([1, 2, 2, 3], [3, 4, 4, 5], 4 / 70),        # 4 и 4, баллы: было 0.040 → 0.057
    ([1, 2, 2, 3, 3], [3, 4, 4, 5, 5], 6 / 252),  # 5 и 5, баллы: было 0.019 → 0.024
    ([1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8], 2 / 70),  # контроль без совпадений
])
def test_mann_whitney_p_is_exact_with_ties(a, b, exact) -> None:
    res = run("mann_whitney", _two_groups(a, b), GROUPS, {})
    assert res["p_value"] == pytest.approx(exact, abs=1e-12)
    assert res["parameters"]["p_method"] == "exact"
    assert exact == pytest.approx(_brute_rank_sum(a, b), abs=1e-12)


def test_conclusion_at_alpha_005_no_longer_flips() -> None:
    """Первая и третья строки таблицы: при α = 0.05 вывод был «значимо», точное P — нет."""
    w = run("wilcoxon", {"before": [10, 12, 9, 14, 11, 13],
                         "after": [10, 13.5, 11, 17, 15.5, 18]}, PAIRS, {})
    m = run("mann_whitney", _two_groups([1, 2, 2, 3], [3, 4, 4, 5]), GROUPS, {})
    assert w["p_value"] > 0.05 and m["p_value"] > 0.05


def test_one_sample_wilcoxon_is_exact_too() -> None:
    values = [3, 4, 3, 5, 3, 5, 2, 4]  # баллы против μ₀ = 2: одна нулевая разность
    res = run("one_sample_wilcoxon", {"y": values}, {"value": "y"}, {"popmean": 2})
    assert res["p_value"] == pytest.approx(_brute_signed_rank(np.subtract(values, 2)), abs=1e-12)
    assert res["parameters"]["p_method"] == "exact"


# --- Свойства расчёта ------------------------------------------------------------------------

def test_dp_equals_brute_force_on_random_tied_data() -> None:
    rng = np.random.default_rng(3)
    for _ in range(150):
        n1, n2 = rng.integers(1, 7, 2)
        a, b = rng.integers(1, 6, n1).astype(float), rng.integers(1, 6, n2).astype(float)
        for alt in ("two-sided", "greater", "less"):
            assert rank_sum_p(a, b, alt) == pytest.approx(_brute_rank_sum(a, b, alt), abs=1e-12)
        d = rng.integers(-3, 5, rng.integers(1, 11)).astype(float)
        if np.any(d):
            assert signed_rank_p(d) == pytest.approx(_brute_signed_rank(d), abs=1e-12)


def test_without_ties_equals_scipy_exact_for_every_alternative() -> None:
    rng = np.random.default_rng(11)
    a, b, d = rng.normal(size=7), rng.normal(size=9), rng.normal(size=15)
    for alt in ("two-sided", "greater", "less"):
        assert rank_sum_p(a, b, alt) == pytest.approx(
            stats.mannwhitneyu(a, b, alternative=alt, method="exact").pvalue, abs=1e-12)
        assert signed_rank_p(d, alt) == pytest.approx(
            stats.wilcoxon(d, alternative=alt, mode="exact").pvalue, abs=1e-12)


def test_alternative_direction_matches_the_method() -> None:
    """`greater` — первая группа (или разности a − b) больше; сумма односторонних P ≥ 1."""
    cols = _two_groups([3, 4, 4, 5, 5], [1, 2, 2, 3, 3])
    up = run("mann_whitney", cols, GROUPS, {"alternative": "greater"})["p_value"]
    down = run("mann_whitney", cols, GROUPS, {"alternative": "less"})["p_value"]
    assert up == pytest.approx(3 / 252, abs=1e-12) and down > 0.98 and up + down >= 1


def test_large_samples_use_normal_approximation_and_say_so() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.integers(1, 8, 60).astype(float), rng.integers(2, 9, 61).astype(float)
    res = run("mann_whitney", _two_groups(a.tolist(), b.tolist()), GROUPS, {})
    assert res["parameters"]["p_method"] == "normal"
    assert res["p_value"] == pytest.approx(stats.mannwhitneyu(a, b).pvalue, rel=1e-12)
    d = rng.integers(-5, 8, 260).astype(float)
    d[d == 0] = 1
    res = run("one_sample_wilcoxon", {"y": d.tolist()}, {"value": "y"}, {})
    assert res["parameters"]["p_method"] == "normal"


def test_exact_and_normal_agree_near_the_limit() -> None:
    """У предела (50 + 50) точное и нормальное P близки — переход не даёт скачка."""
    rng = np.random.default_rng(8)
    a, b = rng.integers(1, 8, 50).astype(float), rng.integers(2, 9, 50).astype(float)
    assert a.size + b.size == rank_exact.MAX_RANK_SUM_N
    assert rank_sum_p(a, b) == pytest.approx(stats.mannwhitneyu(a, b).pvalue, rel=0.05)


def test_p_method_is_labelled_in_three_languages() -> None:
    i18n = (WEB / "i18n.tsx").read_text("utf-8")
    assert i18n.count('"result.pMethod.exact": ') == 3
    assert i18n.count('"result.pMethod.normal": ') == 3
    assert "result.pMethod.${pMethod}" in (WEB / "components.tsx").read_text("utf-8")


# --- Краскел–Уоллис: точного P нет, пометка при малых группах ---------------------------------

def test_kruskal_wallis_notes_small_groups() -> None:
    """3 группы по 3: χ²-приближение даёт 0.027 при точном 0.0036 (6 разбиений из 1680)."""
    y = [1.1, 2.3, 3.2, 4.4, 5.1, 6.6, 7.2, 8.9, 9.5]
    res = run("kruskal_wallis", {"g": [f"g{i // 3}" for i in range(9)], "y": y}, GROUPS, {})
    assert res["p_value"] == pytest.approx(0.0273, abs=1e-4)  # P не меняется
    (note,) = [w for w in res["warnings"] if w["code"] == "kruskal_small_groups"]
    assert note["params"] == {"n_min": 3, "need": 5} and "χ²-приближению" in note["message"]


def test_kruskal_wallis_is_silent_when_every_group_has_five() -> None:
    rng = np.random.default_rng(2)
    cols = {"g": [f"g{i // 5}" for i in range(15)], "y": rng.normal(size=15).round(2).tolist()}
    assert "kruskal_small_groups" not in codes(run("kruskal_wallis", cols, GROUPS, {}))
    text = (WEB / "coreMessages.tsx").read_text("utf-8")
    block = text.split("kruskal_small_groups: {")[1].split("\n  },")[0]
    assert all(f"{lang}: `" in block for lang in ("ru", "en", "pt"))
