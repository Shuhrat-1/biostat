"""Правило «Стьюдент или Уэлч» по отношению SD и балансу n (core.stats.welch_rule)."""

from __future__ import annotations

import numpy as np
import pytest

from core.stats.hypothesis import two_sample_t
from core.stats.welch_rule import welch_advice

BASE = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.25, 0.25, 0.0])  # SD ≈ 0.9


def _group(sd_mult: float, n: int) -> np.ndarray:
    """n значений с SD, пропорциональным sd_mult (повтор BASE по кругу)."""
    return 10 + sd_mult * np.resize(BASE, n)


@pytest.mark.parametrize("sd_b, n_b, case", [
    (1.8, 10, None),                 # n равны, SD < 2 раз — Стьюдент
    (2.5, 10, "welch"),              # n равны, SD > 2 раз
    (1.3, 20, None),                 # n неравны, SD < 1.5 раза
    (1.8, 20, "welch_unbalanced"),   # большая SD в большей группе
    (1 / 1.8, 20, "welch_liberal"),  # большая SD в меньшей группе
])
def test_cases(sd_b: float, n_b: int, case: str | None):
    advice = welch_advice(_group(1.0, 10), _group(sd_b, n_b))
    assert (advice.case if advice else None) == case


def test_ratios_are_larger_over_smaller():
    advice = welch_advice(_group(3.0, 10), _group(1.0, 30))
    assert advice.sd_ratio == pytest.approx(3.0, rel=0.05)
    assert advice.n_ratio == pytest.approx(3.0)
    assert advice.case == "welch_liberal"


def test_degenerate_groups_are_silent():
    assert welch_advice(np.array([1.0]), np.array([1.0, 2.0])) is None
    assert welch_advice(np.array([2.0, 2.0, 2.0]), np.array([1.0, 3.0, 5.0])) is None


def test_two_sample_t_warning_code_follows_case():
    r = two_sample_t(_group(1.0, 10), _group(1 / 1.8, 20))
    assert r.parameters["variant"] == "student"
    w = next(w for w in r.warnings if w.code.startswith("unequal_variance"))
    assert w.code == "unequal_variance_liberal"
    assert w.params["n_ratio"] == 2.0


def test_two_sample_t_silent_when_rule_allows_student():
    r = two_sample_t(_group(1.0, 10), _group(1.8, 10))
    assert not any(w.code.startswith("unequal_variance") for w in r.warnings)
    assert "levene_p" in r.statistics  # Левене — справка


def test_auto_follows_rule_not_levene():
    assert two_sample_t(_group(1.0, 10), _group(2.5, 10), variant="auto").parameters[
        "variant"] == "welch"
    assert two_sample_t(_group(1.0, 10), _group(1.8, 10), variant="auto").parameters[
        "variant"] == "student"
