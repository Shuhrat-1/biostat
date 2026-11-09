"""Векторный _design_matrix (designs.py): то же пространство столбцов, что у строковой версии."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.stats import designs


def _string_design(frame: pd.DataFrame, terms: list[tuple[str, ...]]) -> np.ndarray:
    """Прежняя реализация: склейка уровней строкой и get_dummies."""
    blocks = [np.ones((len(frame), 1))]
    for term in terms:
        labels = frame[list(term)].astype(str).agg("|".join, axis=1)
        blocks.append(pd.get_dummies(labels).to_numpy(float))
    return np.hstack(blocks)


@pytest.mark.parametrize("terms", [
    [("a",)], [("a",), ("b",), ("a", "b")], [("c",), ("a", "c"), ("a", "b", "c")]])
def test_design_matrix_same_space_as_string_version(terms):
    """Те же столбцы с точностью до порядка: ранг, RSS и подогнанные значения совпадают."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.choice(list("abc"), 300), "b": rng.choice(["1", "2", "10"], 300),
                          "c": rng.integers(0, 40, 300).astype(str), "y": rng.normal(size=300)})
    new, old = designs._design_matrix(frame, terms), _string_design(frame, terms)
    assert new.shape == old.shape
    assert sorted(map(tuple, new.T)) == sorted(map(tuple, old.T))
    y = frame["y"].to_numpy()
    resid = y - old @ np.linalg.lstsq(old, y, rcond=None)[0]
    rss, rank = designs._rss(frame, terms)
    assert rank == np.linalg.matrix_rank(old)
    assert rss == pytest.approx(float(resid @ resid))
