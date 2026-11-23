"""Советы проверок допущений (B4) в опыте с блоками.

Было: при роли «блок» остатки считались с учётом блоков, но сигналы вели к
методам без блоков — трансформация (transformed_means), Уэлч, перестановочный
тест, Краскел–Уоллис (находка помощника на шаге M2). Теперь при полном RCBD
(каждый вариант ровно раз в каждом блоке, кн. стр. 58):

- B4.3 / B4.4 → лист ANOVA с блоками (трансформация — её параметр);
- B4.5 → подсказка на том же листе: аналога Уэлча с блоками в BioStat нет;
- B4.6 / B4.8 → критерий Фридмана (блок — «объект»), подсказкой;
- блоки иной структуры (пропуски, повторы) → однофакторные советы B4 молчат,
  ведут структурные сигналы B1.

Шаг «После ANOVA» (B5.4): при роли «блок» — подсказка о сравнениях по модели с
блоками; листья методов без блока карта помечает «без блоков» (фронт).

Синтетика — детерминированные квантили (как в test_recognize_computed):
вариант i в блоке j = эффект варианта + эффект блока + отклонение.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.validation.recognize import HINT, STRONG, recognize
from tests.test_recognize import _table

NORMAL8 = stats.norm.ppf(np.linspace(0.06, 0.94, 8))
SKEWED8 = stats.expon.ppf(np.linspace(0.06, 0.94, 8))
MIX = [3, 6, 0, 5, 1, 7, 2, 4]  # порядок квантилей по блокам; у каждого варианта — свой сдвиг
BLOCK_EFFECT = np.linspace(-1.5, 1.5, 8)


def _rcbd(devs: dict[str, tuple[float, np.ndarray]]) -> dict[str, list]:
    """Полный RCBD: вариант → (среднее, отклонения по 8 блокам)."""
    rows = [(k, f"b{j + 1}", m + BLOCK_EFFECT[j] + d[MIX[(j + 3 * i) % 8]])
            for i, (k, (m, d)) in enumerate(devs.items()) for j in range(8)]
    return {"t": [r[0] for r in rows], "blk": [r[1] for r in rows],
            "y": [round(float(r[2]), 4) for r in rows]}


def _fired(cols: dict[str, list], block: bool = True) -> dict[str, dict]:
    roles = {"value": "y", "group": "t", **({"block": "blk"} if block else {})}
    return {f["id"]: f for f in recognize(_table(cols), roles)["fired"]}


def _p(f: dict, rid: str) -> dict:
    return f[rid]["message"]["params"]


SD_TRACKS = _rcbd({k: (m, 0.25 * m * NORMAL8)
                   for k, m in {"a": 4, "b": 10, "c": 25, "d": 60}.items()})
# SD одного варианта (не крайнего по средней) вчетверо больше: связи SD со средней нет.
HETERO = _rcbd({"a": (10, NORMAL8), "b": (20, NORMAL8), "c": (30, 4 * NORMAL8), "d": (40, NORMAL8)})
NONNORMAL = _rcbd({k: (m, 2 * SKEWED8) for k, m in {"a": 0, "b": 1, "c": 2, "d": 3}.items()})
SCORES = {"t": [t for t in "abc" for _ in range(8)],
          "blk": [f"b{j + 1}" for _ in "abc" for j in range(8)],
          "y": [1, 2, 2, 3, 1, 2, 3, 2, 2, 3, 3, 4, 2, 3, 4, 3, 3, 4, 4, 5, 3, 4, 5, 4]}


def test_rcbd_layout_is_recognised():
    for cols in (SD_TRACKS, HETERO, NONNORMAL, SCORES):
        assert "B1.2" in _fired(cols)


def test_sd_tracks_mean_points_to_rcbd_leaf():
    f = _fired(SD_TRACKS)
    assert _p(f, "B4.3")["branch"] == "B4b.rcbd"
    assert _p(f, "B4.3")["variant"] == "blocks"
    assert f["B4.3"]["strength"] == STRONG
    plain = _fired(SD_TRACKS, block=False)  # без роли «блок» — прежний лист
    assert "branch" not in _p(plain, "B4.3") and plain["B4.3"]["suggests"] == "transform_groups"


def test_heterogeneous_with_blocks_is_note_on_rcbd_leaf():
    f = _fired(HETERO)
    assert _p(f, "B4.5") == {"branch": "B4b.rcbd", "variant": "blocks"}
    assert f["B4.5"]["strength"] == HINT


def test_nonnormal_with_blocks_points_to_friedman():
    f = _fired(NONNORMAL)
    assert {"B4.1", "B4.6"} <= set(f)
    assert _p(f, "B4.6")["method"] == "friedman" and f["B4.6"]["strength"] == HINT
    if "B4.4" in f:
        assert _p(f, "B4.4")["branch"] == "B4b.rcbd"


def test_scores_in_blocks_point_to_friedman():
    f = _fired(SCORES)
    assert _p(f, "B4.8") == {"method": "friedman", "variant": "blocks"}
    assert _p(_fired(SCORES, block=False), "B4.8")["method"] == "kruskal_wallis"


def test_other_block_structures_silence_one_way_advice():
    """Пропуск в ячейке: не RCBD — советы B4 для однофакторных методов молчат."""
    cols = {k: v[:-1] for k, v in HETERO.items()}
    f = _fired(cols)
    assert "B1.2" not in f
    assert not {"B4.3", "B4.4", "B4.5", "B4.6", "B4.8"} & set(f)
    scores = {k: v[:-1] for k, v in SCORES.items()}
    assert "B4.8" not in _fired(scores)


def test_b5_with_blocks_points_to_comparisons_with_blocks():
    """B5.4: шаг «После ANOVA» при роли «блок» — сравнения по модели с блоками."""
    f = _fired(SD_TRACKS)
    assert f["B5.4"]["strength"] == HINT
    assert _p(f, "B5.4") == {"branch": "B5.mc_tukey"}
    assert "B5.4" not in _fired(SD_TRACKS, block=False)
