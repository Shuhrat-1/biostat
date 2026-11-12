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
from core.validation.recognize import COMPUTED, HINT, STRONG, plural_form, recognize
from tests.map_scenarios import SCENARIOS, grouped, q

BY_ID = {s["id"]: s for s in SCENARIOS}
_TREE = json.loads((Path(__file__).parent.parent / "web" / "src" / "decisionTree.json")
                   .read_text(encoding="utf-8"))
TARGETS = _TREE["targets"]
NODES = {n["id"]: n for n in _TREE["nodes"]}


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


# --- 01-01. Выброс не меняет метод -------------------------------------------------------

_B4_SOFT = ("B4.2", "B4.5")


def test_01_01_outlier_softens_assumption_signals():
    """S10 (кальций, Table 4.5, одно значение ×3): сначала проверить запись [кн. стр. 109–110]."""
    fired = _stage("S10", "one-way")
    assert fired["DIAG.1"]["strength"] == STRONG
    for rid in _B4_SOFT:
        assert fired[rid]["strength"] == HINT
        assert fired[rid]["message"]["params"]["outlier_row"] == 11
        assert "выброса в строке 11" in fired[rid]["message"]["message"]


def test_01_01_outlier_found_in_a_small_trial():
    """3 группы × 3: |r| не может превысить √6 — выброс ловит удалённый остаток [предл.]."""
    data = grouped({"a": [20.1, 20.9, 19.6], "b": [24.8, 25.5, 24.2], "c": [29.9, 30.4, 47.0]})
    fired = _run(data, {"value": "y", "group": "group"})
    assert fired["DIAG.1"]["message"]["params"] == {"rows": "9", "variant": "single"}
    assert all(f["strength"] == HINT for rid, f in fired.items() if rid in ("B4.2", "B4.3", "B4.5"))


def test_01_01_skewed_data_are_not_called_outliers():
    """S07 (Table 6.1): большие значения — свойство шкалы; совет log остаётся уверенным."""
    fired = _stage("S07", "one-way")
    assert fired["B4.3"]["strength"] == STRONG
    assert "outlier_row" not in fired["B4.3"]["message"]["params"] and "DIAG.1" not in fired


def test_01_01_outlier_does_not_pass_for_skew_it_created_itself():
    """Признаки асимметрии проверяются без кандидата (решение 11.11.2026).

    Синтетика tests/test_outliers.py: 4 группы по 4, SD ≈ 2; одно значение
    завышено на 14. Оно само создаёт «SD растёт со средним» (r ≥ 0.8), и на
    log-шкале уже не выделяется — раньше карта давала уверенный совет log.
    Без этой строки связи нет, значит это выброс.
    """
    from tests.test_outliers import GROUPS

    fired = _run(grouped({f"g{i}": v for i, v in enumerate(GROUPS)}),
                 {"value": "y", "group": "group"})
    assert fired["DIAG.1"]["message"]["params"]["rows"] == "13"
    assert fired["B4.3"]["strength"] == HINT
    assert fired["B4.3"]["message"]["params"]["outlier_row"] == 13


def test_01_01_clean_data_are_untouched():
    assert not {"DIAG.1", "B4.2", "B4.5"} & set(_stage("S05", "one-way"))


# --- 01-03. Допущения на листьях планов; трансформация — параметр метода -----------------

def test_01_03_assumption_signals_fire_for_a_factorial():
    """S14 (Welham Example 8.1): на факториале жуков дисперсия растёт со средним."""
    fired = _stage("S14", "full")
    assert fired["B4.3"]["strength"] == STRONG and fired["B4.3"]["node"] == "B4"
    assert TARGETS["transform_groups"] == ["B4b.transform.means"]


def test_01_03_designs_have_their_own_transformation_parameter():
    """Подсказка о трансформации ведёт к параметру метода — он должен быть у каждого плана."""
    from core.validation.catalog import CATALOG

    specs = {sp.method: sp for sp in CATALOG}
    for method in ("rcbd_anova", "rcbd_subsampling_anova", "factorial_anova", "nested_anova",
                   "latin_square_anova", "split_plot_anova", "bibd_anova", "unbalanced_anova",
                   "ancova", "mixed_model"):
        assert any(p.name == "transform_kind" for p in specs[method].params), method


# --- 01-09. Две группы с асимметрией ------------------------------------------------------

def test_01_09_skewed_two_groups_get_no_welch_advice():
    """S03: при правой асимметрии SD различаются из-за неё — ранговый тест или log."""
    fired = _stage("S03", "one-way")
    assert "B4.2" not in fired
    assert fired["B4.1"]["message"]["params"]["method"] == "mann_whitney"
    assert fired["R2.7"]["message"]["params"]["branch"] == "B4a.transform"


def test_01_09_two_group_branch_has_a_transformation_leaf():
    leaf = NODES["B4a.transform.means"]
    assert leaf["method"] == "transformed_means" and NODES[leaf["parent"]]["parent"] == "B4a"
    # При трёх и более группах подсказка ведёт в свою ветку.
    assert _stage("S10", "one-way")["R2.7"]["message"]["params"]["branch"] == "B4b.transform"


def test_01_09_unequal_spread_without_skew_still_advises_welch():
    """S02 при n = 10 в группе: симметричные данные, SD ×4 — совет Уэлча остаётся."""
    fired = _run(_two_groups(10), {"value": "y", "group": "group"})
    assert fired["B4.2"]["strength"] == STRONG


# --- 01-11. ANCOVA -------------------------------------------------------------------------

def test_01_11_assigned_covariate_points_to_ancova_and_enters_the_model():
    """S32 (Welham Table 15.15): с ковариатой в модели ложной неоднородности нет."""
    start = _stage("S32", "one-way")
    assert start["B2.5"]["strength"] == HINT and {"B4.2", "B4.3"} <= set(start)
    fired = _stage("S32", "full")
    assert fired["B2.5"]["strength"] == STRONG
    assert fired["B2.5"]["message"]["params"] == {"c": "damage", "variant": "assigned"}
    assert not {"B4.2", "B4.3", "B4.5"} & set(fired)


# --- 01-05. Колонки без роли: блок, второй фактор, вложенность -----------------------------

def _method_leaves(method: str) -> list[str]:
    return [i for i, n in NODES.items() if n.get("method") == method]


def test_01_05_unassigned_block_column_is_noticed():
    """Welham Table 7.1, стр. 137: картофель в RCBD, в форме указан только вариант."""
    hit = _stage("S13", "one-way")["B1.8"]
    assert hit["strength"] == HINT and hit["suggests"] == "rcbd_anova"
    assert hit["message"]["params"] == {"c": "block", "t": 5, "b": 4}
    assert _method_leaves("rcbd_anova") == ["B4b.rcbd"]
    assert "B1.8" not in _stage("S13", "full")


def test_01_05_unassigned_subjects_are_noticed():
    """Witte Progress Check 17.2, стр. 331: шесть субъектов в трёх условиях."""
    params = _stage("S24", "one-way")["B1.8"]["message"]["params"]
    assert params == {"c": "subject", "t": 3, "b": 6}


def test_01_05_unassigned_rows_and_columns_point_to_the_latin_square():
    """Welham Table 9.1, стр. 209: люпин, латинский квадрат 4 × 4."""
    params = _stage("S20", "one-way")["B1.8"]["message"]["params"]
    assert params == {"a": "row", "b": "col", "method": "latin_square_anova",
                      "variant": "latin"}
    assert _method_leaves("latin_square_anova") == ["B1.latin"]


def test_01_05_unassigned_incomplete_blocks_point_to_the_block_branch():
    """Welham Table 9.20, стр. 232: BIBD, 7 сортов в 7 блоках по 4."""
    sc = BY_ID["S21"]
    params = _run(sc["columns"], {"value": "y", "group": "variety"})["B1.8"]["message"]["params"]
    assert params == {"c": "block", "branch": "B1", "variant": "incomplete"}
    assert NODES["B1"]["kind"] != "leaf"


def test_01_05_unassigned_second_factor_is_noticed():
    """Welham Example 8.1, стр. 151: жуки, 2 × 2 по 10 самок, в форме один фактор."""
    hit = _stage("S14", "one-way")["B2.7"]
    assert hit["strength"] == HINT and hit["suggests"] == "factorial_anova"
    assert hit["message"]["params"] == {"c": "mate", "f": "species", "k": 10}
    assert "B2.7" not in _stage("S14", "full")


def test_01_05_factor_nested_in_a_free_column_is_noticed():
    """Welham Table 8.10, стр. 173: изоляты внутри видов, в форме только изолят."""
    hit = _stage("S22", "one-way")["B2.8"]
    assert hit["strength"] == HINT and hit["suggests"] == "nested_anova"
    assert hit["message"]["params"] == {"c": "species", "f": "isolate", "variant": "outer"}
    assert "B2.8" not in _stage("S22", "full")


def test_01_05_rows_within_a_plant_are_not_independent_replicates():
    """Welham стр. 46, 100: 8 растений по 4 даты — строки растения не повторности."""
    params = _stage("S25", "one-way")["B2.8"]["message"]["params"]
    assert params == {"c": "plant", "f": "treatment"}


def test_01_05_plain_one_way_data_get_no_structure_hints():
    """Welham Table 4.5: однофакторный опыт без лишних колонок — правила молчат."""
    fired = _stage("S05", "one-way")
    assert not {"B1.8", "B2.7", "B2.8"} & set(fired)
    labels = dict(BY_ID["S05"]["columns"], note=["ok"] * len(BY_ID["S05"]["columns"]["y"]))
    assert not {"B1.8", "B2.7", "B2.8"} & set(_run(labels, BY_ID["S05"]["stages"]["one-way"]))


def test_01_05_numbered_blocks_are_noticed_too():
    """Блоки, записанные номерами 1…4 (числовая колонка), — тот же картофель."""
    cols = dict(BY_ID["S13"]["columns"])
    codes = {b: i + 1 for i, b in enumerate(dict.fromkeys(cols["block"]))}
    cols["block"] = [codes[b] for b in cols["block"]]
    hit = _run(cols, BY_ID["S13"]["stages"]["one-way"])["B1.8"]
    assert hit["message"]["params"]["c"] == "block"


# --- 01-16. Тексты: согласование числа, один уровень фактора ----------------------------

@pytest.mark.parametrize("n, word", [(1, "уровень"), (2, "уровня"), (4, "уровня"),
                                     (5, "уровней"), (11, "уровней"), (21, "уровень"),
                                     (22, "уровня"), (112, "уровней")])
def test_01_16_russian_plural_forms(n: int, word: str):
    assert plural_form(n, ["уровень", "уровня", "уровней"]) == word


def test_01_16_plural_of_fractions_and_two_form_languages():
    assert plural_form(2.5, ["раз", "раза", "раз"]) == "раза"
    assert plural_form(5.0, ["раз", "раза", "раз"]) == "раза"  # печатается как «5.0»
    assert [plural_form(n, ["row", "rows"]) for n in (1, 2, 5)] == ["row", "rows", "rows"]


def test_01_16_signal_texts_agree_in_number():
    """«4 уровня», а не «4 уровней»; «2 раза», а не «2 раз» (S13 в блоках по два — BIBD)."""
    codes = _run(BY_ID["S41"]["columns"], {"value": "y"})["B2.6"]["message"]["message"]
    assert "всего 4 разных значения" in codes
    two = {"dose": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
           "y": [2.1, 2.4, 1.9, 3.0, 3.3, 2.8, 4.2, 3.9, 4.4, 5.1, 4.8, 5.3]}
    assert "(4 уровня)" in _run(two, {"x": "dose", "y": "y"})["A2.1"]["message"]["message"]


def test_01_16_single_level_factor_gets_a_note():
    hit = _stage("S45", next(iter(BY_ID["S45"]["stages"])))["B3.1"]
    assert hit["strength"] == HINT
    assert hit["message"]["params"]["branch"] == "D" and NODES["D"]["kind"] != "leaf"
    assert "один уровень" in hit["message"]["message"]


# --- 01-12. Связанные измерения и случайная группировка -----------------------------------

def _computed(columns: dict[str, list], roles: dict[str, str]) -> dict[str, dict]:
    out = recognize(_table(columns), roles, cls=COMPUTED)
    assert out["errors"] == []
    return {f["id"]: f for f in out["fired"]}


def test_01_12_random_grouping_with_a_continuous_response_is_a_mixed_model():
    """Welham Table 16.4, стр. 427–440: сорняки на 59 полях 37 ферм, отклик — log."""
    fired = _stage("S40", "full")
    hit = fired["B1.9"]
    assert hit["strength"] == STRONG and hit["node"] == "B1" and hit["suggests"] == "mixed_model"
    assert hit["message"]["params"] == {"c": "farm", "k": 37}
    assert NODES["B1.mixed"]["method"] == "mixed_model"
    assert "G.4" not in fired  # GLMM — для долей и счётов, не для измерений


def test_01_12_random_grouping_with_counts_still_points_to_glmm():
    """Счёты жуков в ловушках на шести участках [BioStat]: группировка — правило G.4."""
    counts = {"site": [s for s in "abcdef" for _ in range(4)],
              "trap": ["t1", "t2", "t3", "t4"] * 6,
              "beetles": [3, 0, 5, 2, 7, 4, 9, 6, 1, 0, 2, 1,
                          12, 8, 15, 10, 4, 3, 6, 2, 0, 1, 0, 2]}
    fired = _run(counts, {"value": "beetles", "factor_a": "trap", "stratum_1": "site"})
    assert fired["G.4"]["message"]["params"]["variant"] == "glmm" and "B1.9" not in fired


def test_01_12_few_random_levels_only_hint():
    cols = dict(BY_ID["S25"]["columns"])
    cols["bench"] = ["b1" if p in ("p1", "p2", "p5", "p6") else "b2" for p in cols["plant"]]
    hit = _run(cols, {"value": "height", "factor_a": "treatment", "stratum_1": "bench"})["B1.9"]
    assert hit["strength"] == HINT and hit["message"]["params"]["variant"] == "few"


def test_01_12_each_subject_once_in_each_condition_is_rm_anova():
    """Witte Progress Check 17.2, стр. 331: 6 субъектов × 3 уровня шума."""
    hit = _stage("S24", "full")["B0.5"]
    assert hit["strength"] == STRONG and hit["suggests"] == "rm_anova"
    assert hit["message"]["params"] == {"s": "subject", "c": "condition", "n": 6, "k": 3}
    assert NODES["B0.rm"]["method"] == "rm_anova"
    assert "Каждый из 6 объектов" in hit["message"]["message"]


def test_01_12_incomplete_repeated_measures_go_to_the_mixed_model():
    """Тот же опыт без одного измерения: полной таблицы субъект × условие нет."""
    cols = {k: v[:-1] for k, v in BY_ID["S24"]["columns"].items()}
    hit = _run(cols, BY_ID["S24"]["stages"]["full"])["B0.5"]
    assert hit["strength"] == HINT
    assert hit["message"]["params"]["method"] == "mixed_model"


def test_01_12_repeated_scores_point_to_friedman():
    """Witte Fig. 21.1, стр. 405: баллы в связанных выборках → ранговый тест."""
    scores = {"judge": [j for j in ("j1", "j2", "j3", "j4", "j5", "j6") for _ in range(3)],
              "wine": ["a", "b", "c"] * 6,
              "score": [3, 5, 4, 2, 4, 4, 3, 5, 5, 1, 3, 2, 2, 5, 3, 3, 4, 5]}
    hit = _run(scores, {"value": "score", "condition": "wine", "subject": "judge"})["B0.5"]
    assert hit["strength"] == HINT and hit["message"]["params"]["method"] == "friedman"


def test_01_12_assigned_pair_selects_the_paired_branch():
    """Witte Review 15.7, стр. 280: GPA до и после — пара назначена в форме."""
    hit = _stage("S23", "full")["B0.1"]
    assert hit["strength"] == STRONG and TARGETS["paired"] == ["B0.paired"]
    assert _computed(BY_ID["S23"]["columns"], BY_ID["S23"]["stages"]["full"]) == {}


def test_01_12_skewed_differences_point_to_wilcoxon():
    """Разности с длинным правым хвостом (логнормальные квантили): парный t ненадёжен."""
    before = [round(20 + 2 * z, 2) for z in q(16)]
    gain = np.exp(1.2 * q(16)[::-1])
    cols = {"before": before, "after": [round(b + g, 2) for b, g in zip(before, gain)]}
    hit = _computed(cols, {"before": "before", "after": "after"})["B0.6"]
    assert hit["strength"] == STRONG and hit["suggests"] == "wilcoxon"
    assert NODES["B0.paired.w"]["method"] == "wilcoxon" and hit["message"]["params"]["p"] < 0.05


def test_01_12_one_wild_pair_is_a_record_to_check_not_a_reason_for_ranks():
    """Одна пара с опечаткой (разность на порядок больше): подсказка, а не указание."""
    before = [round(20 + 2 * z, 2) for z in q(12)]
    after = [round(b + 1 + 0.5 * z, 2) for b, z in zip(before, q(12)[::-1])]
    after[4] = before[4] + 15
    hit = _computed({"before": before, "after": after},
                    {"before": "before", "after": "after"})["B0.6"]
    assert hit["strength"] == HINT
    assert hit["message"]["params"]["variant"] == "outlier" and hit["message"]["params"]["row"] == 5
