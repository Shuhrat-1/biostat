"""q_critical — критическое q стьюдентизированного размаха вместо scipy ppf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.qrange import q_critical


@pytest.mark.parametrize(("alpha", "k", "df"), [
    (0.05, 3, 10), (0.05, 6, 94), (0.01, 10, 5), (0.10, 20, 200), (0.05, 4, 1.5)])
def test_q_critical_equals_scipy_ppf(alpha, k, df):
    """brentq в скобке Бонферрони = scipy.studentized_range.ppf до 1e-9."""
    ref = stats.studentized_range.ppf(1 - alpha, k, df)
    assert q_critical(alpha, k, df) == pytest.approx(ref, rel=1e-9)


def test_q_critical_two_means_is_t():
    """k = 2: q = √2·t(1 − α/2, df) точно."""
    assert q_critical(0.05, 2, 7) == pytest.approx(np.sqrt(2) * stats.t.ppf(0.975, 7), rel=1e-15)
    with pytest.raises(ValueError):
        q_critical(0.05, 1, 7)
