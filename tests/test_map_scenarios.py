"""Регрессионные тесты карты на сценариях аудита 1 (``tests/map_scenarios.py``).

Каждый тест назван по находке аудита (``claude/audit-01-report.md``) и
проверяет сигналы ядра на сценарии, где находка воспроизводилась. Контур
(зелёный / янтарный) считает фронт по этим сигналам и таблице ``targets``
дерева, поэтому цели сигналов сверяются с ``web/src/decisionTree.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.report.schema import Severity
from core.stats.hypothesis import two_sample_t
from core.validation.parser import parse_file
from core.validation.recognize import HINT, STRONG, recognize
from tests.map_scenarios import SCENARIOS, grouped, q

BY_ID = {s["id"]: s for s in SCENARIOS}
TARGETS = json.loads((Path(__file__).parent.parent / "web" / "src" / "decisionTree.json")
                     .read_text(encoding="utf-8"))["targets"]


def _table(columns: dict[str, list]):
    names = list(columns)
    rows = zip(*(columns[n] for n in names))
    text = ",".join(names) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n"
    return parse_file(text.encode("utf-8"))


def _run(columns: dict[str, list], roles: dict[str, str],
         method: str | None = None) -> dict[str, dict]:
    out = recognize(_table(columns), roles, method=method)
    assert out["errors"] == []
    return {f["id"]: f for f in out["fired"]}


def _stage(sid: str, stage: str, method: str | None = None) -> dict[str, dict]:
    sc = BY_ID[sid]
    return _run(sc["columns"], sc["stages"][stage], method)


@pytest.mark.parametrize("sid, stage", [(s["id"], st) for s in SCENARIOS for st in s["stages"]])
def test_no_rule_fails_on_any_scenario(sid: str, stage: str):
    """Ни одно правило не падает ни на одном плане (ошибки видны в ``errors``)."""
    _stage(sid, stage)


def test_scenarios_cover_the_audit_list():
    assert len(SCENARIOS) == 46 and len(BY_ID) == 46
    assert all(s["expect"] and s["source"] for s in SCENARIOS)


# --- 01-02. Варианты, закодированные числами --------------------------------------------

def test_01_02_numeric_treatment_codes_get_a_hint():
    """S41: варианты 1–4 — числовая колонка; до смены типа фактор назначить нельзя."""
    sc = BY_ID["S41"]
    fired = _run(sc["columns"], {"value": "y"})
    assert fired["B2.6"]["message"]["params"] == {"c": "variant", "k": 4}
    assert fired["B2.6"]["strength"] == HINT and TARGETS["factor_codes"] == []
    # Данные книги Table 6.1 записаны так же: группы 1–4.
    assert _run(BY_ID["S07"]["columns"], {"value": "y"})["B2.6"]["message"]["params"]["c"] == "group"


def test_01_02_factor_column_is_not_offered_as_covariate_or_dose():
    """S41: колонка, назначенная фактором, — не ковариата (B2.5) и не доза (B5.1)."""
    fired = _stage("S41", "one-way")
    assert not {"B2.5", "B5.1", "B2.6"} & set(fired)


def test_01_02_real_doses_still_suggest_a_trend():
    doses = grouped({str(d): 10 + 0.05 * d + q(5, i) for i, d in enumerate((0, 50, 100, 150))})
    assert "B5.1" in _run(doses, {"value": "y", "group": "group"})


# --- 01-04. Неадекватность, группы и блоки в регрессии ----------------------------------

def test_01_04_repeated_x_is_a_note_not_a_branch():
    """S27 (Welham Table 12.7): значимая неадекватность ведёт к кривой без спора с A2.1."""
    fired = _stage("S27", "xy")
    assert fired["A2.1"]["suggests"] == "lack_of_fit" and TARGETS["lack_of_fit"] == []
    assert fired["A2.2"]["strength"] == STRONG and TARGETS["curves"] == ["A2.curve"]


def test_01_04_assigned_block_and_group_are_strong():
    """S31 (Table 15.11) и S30 (Table 15.1): назначенные блок и группа — структура опыта."""
    assert _stage("S31", "full")["A1.3"]["strength"] == STRONG
    assert _stage("S31", "xy")["A1.3"]["strength"] == HINT
    with_group = _stage("S30", "full")["A1.2"]
    assert with_group["strength"] == STRONG and with_group["message"]["params"] == {"g": "stand"}
    assert _stage("S30", "xy")["A1.2"]["strength"] == HINT


def test_01_04_block_column_is_not_guessed_as_groups():
    assert "A1.2" not in _stage("S31", "full")


# --- 01-06. Совет Уэлча на малых выборках ------------------------------------------------

def _two_groups(n: int) -> dict[str, list]:
    """Две группы по n, SD различаются в 4 раза (синтетика)."""
    return grouped({"a": 20 + 4 * q(n), "b": 22 + q(n, 2)})


def test_01_06_small_samples_give_a_note_instead_of_welch_advice():
    fired = _run(_two_groups(5), {"value": "y", "group": "group"})
    params = fired["B4.2"]["message"]["params"]
    assert fired["B4.2"]["strength"] == HINT
    assert params["variant"] == "small_n" and params["n_min"] == 5 and "method" not in params
    large = _run(_two_groups(10), {"value": "y", "group": "group"})["B4.2"]
    assert large["strength"] == STRONG and large["message"]["params"]["method"] == "two_sample_t"


def test_01_06_two_sample_t_informs_instead_of_warning():
    small = _two_groups(5)
    a, b = np.array(small["y"][:5]), np.array(small["y"][5:])
    warning = next(w for w in two_sample_t(a, b).warnings if w.code.startswith("unequal_variance"))
    assert warning.code == "unequal_variance_small_n" and warning.severity == Severity.INFO
    big = _two_groups(10)
    a, b = np.array(big["y"][:10]), np.array(big["y"][10:])
    warning = next(w for w in two_sample_t(a, b).warnings if w.code.startswith("unequal_variance"))
    assert warning.code == "unequal_variance" and warning.severity == Severity.WARNING


# --- 01-07. «SD растёт со средним» по трём группам ---------------------------------------

def test_01_07_three_groups_give_a_hint_four_a_strong_signal():
    three = grouped({k: m * (1 + 0.2 * q(6, i)) for i, (k, m) in
                     enumerate((("a", 10), ("b", 30), ("c", 90)))})
    fired = _run(three, {"value": "y", "group": "group"})
    assert fired["B4.3"]["strength"] == HINT
    assert fired["B4.3"]["message"]["params"]["variant"] == "three"
    assert _stage("S07", "one-way")["B4.3"]["strength"] == STRONG  # Table 6.1: 4 группы


# --- 01-08. Целые измерения — не счёты ---------------------------------------------------

def test_01_08_integer_measurements_get_no_count_noise():
    """S01 (Witte Table 14.1, время в минутах): t-тест без подсказок про счёты."""
    fired = _stage("S01", "one-way", "two_sample_t")
    assert not {"G.1", "G.2", "R2.8", "B5.2"} & set(fired)


def test_01_08_overdispersion_is_judged_inside_the_glm_branch():
    """S36 (Welham Table 18.10): в форме GLM сверхдисперсия считается."""
    assert "G.1" in _stage("S36", "full", "glm_analysis")
    assert "G.1" not in _stage("S36", "one-way", "one_way_anova")


def test_01_08_year_and_plot_number_are_not_variables():
    fired = _stage("S42", "one-way")
    assert not {"B2.5", "A1.1", "A1.5"} & set(fired)


# --- 01-10. Допущения ANOVA на бинарном и счётном отклике --------------------------------

def test_01_10_zero_heavy_counts_point_to_the_count_model_only():
    """S37: 69 % нулей — модель для избыточных нулей, без log-ANOVA."""
    fired = _stage("S37", "one-way")
    assert fired["R2.6"]["strength"] == STRONG
    assert fired["R2.6"]["message"]["params"]["variant"] == "half"
    assert not {"B4.1", "B4.2", "B4.3", "B4.4", "B4.5", "B4.6"} & set(fired)


def test_01_10_binary_response_gets_no_anova_assumption_signals():
    """S35: отклик 0/1 — биномиальная GLM, без перестановочного ANOVA."""
    fired = _stage("S35", "one-way")
    assert "R2.1" in fired
    assert not {"B4.1", "B4.2", "B4.3", "B4.4", "B4.5", "B4.6"} & set(fired)


def test_01_10_counts_without_excess_zeros_keep_the_transformation_route():
    """S36: для счётов без избытка нулей книга допускает и log (Table 6.1)."""
    assert "B4.3" in _stage("S36", "one-way")


# --- 01-13. Таблица частот — не факториал -------------------------------------------------

def test_01_13_frequency_table_is_not_an_unbalanced_factorial():
    """S39 (Witte Table 19.3): без числового отклика B2.1 / B2.2 молчат."""
    fired = _stage("S39", "full")
    assert not {"B2.1", "B2.2"} & set(fired)
    assert fired["R2.5"]["message"]["params"]["variant"] == "file"
    assert TARGETS["ordinal_or_contingency#file"] == ["R2.contingency"]


def test_01_13_label_response_points_to_ordinal_regression():
    fired = _stage("S38", "full")
    assert fired["R2.5"]["message"]["params"]["variant"] == "response"
    assert TARGETS["ordinal_or_contingency#response"] == ["R2.ordinal"]


def test_01_13_factorial_with_numeric_response_still_recognized():
    assert "B2.1" in _stage("S15", "full")
