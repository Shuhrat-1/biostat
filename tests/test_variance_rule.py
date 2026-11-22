"""Одно правило однородности дисперсий для метода и карты (variance_rule).

Было: one_way_anova и anova_permutation_test предупреждали по тесту Левене,
а карта (B4.2, B4.3, B4.5, B4.6) решала по Бартлетту и связи SD со средним —
на одних данных метод и карта могли разойтись. Теперь оба вызывают
``core.stats.variance_rule`` (три и более групп) и ``welch_rule`` (две).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.anova import one_way_anova
from core.stats.anova_checks import anova_permutation_test
from core.stats.variance_rule import bartlett, spreads_of, variance_advice
from core.validation.parser import parse_file
from core.validation.recognize import recognize
from tests.map_scenarios import grouped, q

_VAR_CODES = {"variance_heterogeneity", "variance_grows_with_mean",
              "variance_grows_with_mean_three", "variance_two_groups"}


def _signals(groups: dict[str, np.ndarray]) -> set[str]:
    cols = grouped(groups)
    text = "group,y\n" + "\n".join(f"{g},{y}" for g, y in zip(cols["group"], cols["y"])) + "\n"
    out = recognize(parse_file(text.encode()), {"value": "y", "group": "group"})
    return {f["id"] for f in out["fired"]}


def _codes(res) -> set[str]:
    return {w.code for w in res.warnings}


def _rounded(groups: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Те же числа, что видит карта (grouped округляет до 4 знаков)."""
    return [np.round(v, 4) for v in groups.values()]


CASES = {
    # дисперсии различаются, средние равны → Уэлч
    "welch": {k: 10 + s * q(12, i) for i, (k, s) in enumerate({"a": 1, "b": 4, "c": 1}.items())},
    # SD растёт со средним, 4 группы → log
    "transform": {k: m + 0.3 * m * q(10, i) for i, (k, m) in enumerate({"a": 2, "b": 5, "c": 12, "d": 30}.items())},
    # то же по трём группам → справка
    "transform_three": {k: m + 0.3 * m * q(10, i) for i, (k, m) in enumerate({"a": 2, "b": 6, "c": 20}.items())},
    # однородно
    "ok": {k: m + q(10, i) for i, (k, m) in enumerate({"a": 10, "b": 11, "c": 12}.items())},
}
EXPECT = {"welch": ("variance_heterogeneity", "B4.5"),
          "transform": ("variance_grows_with_mean", "B4.3"),
          "transform_three": ("variance_grows_with_mean_three", "B4.3"),
          "ok": (None, None)}


@pytest.mark.parametrize("case", list(CASES))
def test_method_and_map_agree(case: str) -> None:
    code, signal = EXPECT[case]
    groups = CASES[case]
    method = _codes(one_way_anova(*_rounded(groups))) & _VAR_CODES
    fired = _signals(groups) & {"B4.2", "B4.3", "B4.5"}
    assert method == ({code} if code else set())
    if signal:
        assert signal in fired
    else:
        assert not fired


def test_welch_case_also_fires_bartlett_signal() -> None:
    assert {"B4.2", "B4.5"} <= _signals(CASES["welch"])


def test_bartlett_matches_scipy() -> None:
    groups = _rounded(CASES["welch"])
    x2, crit = bartlett(spreads_of(groups))
    assert x2 == pytest.approx(stats.bartlett(*groups).statistic)
    assert crit == pytest.approx(stats.chi2.ppf(0.95, 2))


def test_advice_none_for_two_groups() -> None:
    assert variance_advice(spreads_of(_rounded(CASES["welch"])[:2])) is None


def test_two_groups_follow_welch_rule() -> None:
    a, b = 10 + q(12), 10 + 4 * q(12, 1)
    assert "variance_two_groups" in _codes(one_way_anova(a, b))
    small = _codes(one_way_anova(10 + q(4), 10 + 4 * q(4, 1)))
    assert "unequal_variance_small_n" in small and "variance_two_groups" not in small


def test_levene_is_reference_only() -> None:
    """Левене остаётся в statistics, но предупреждений не даёт."""
    res = one_way_anova(*_rounded(CASES["ok"]))
    assert "levene_p" in res.statistics
    assert not _codes(res) & (_VAR_CODES | {"levene_undefined"})


def test_permutation_warns_exactly_when_map_rejects_it() -> None:
    """B4.6 ведёт к перестановочному тесту только при однородных дисперсиях."""
    het = anova_permutation_test(*_rounded(CASES["welch"]), n_permutations=99, seed=1)
    ok = anova_permutation_test(*_rounded(CASES["ok"]), n_permutations=99, seed=1)
    assert "permutation_unequal_variances" in _codes(het)
    assert "permutation_unequal_variances" not in _codes(ok)
