"""Тесты распознавателей карты: реестр и структурные правила на данных книги.

Дизайн каждого book-датасета известен из книги (RCBD картофеля Table 7.1,
латинский квадрат Table 9.1, split-plot Table 9.10, BIBD Table 9.20,
факториал жуков Example 8.1, вложенные изоляты Table 8.10), поэтому
правило должно сработать на «своём» дизайне и молчать на чужом.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.validation import recognize as rz
from core.validation.parser import parse_file
from core.validation.recognize import recognize
from tests.book_ch18_data import APHIDS, DEMETHYLATION
from tests.test_blocking import LAYOUT, PLOT_YIELD, ROWS
from tests.test_designs import BIBD_A, BIBD_B, COMPETITION, LUPIN_LS
from tests.test_factorial import _BRASSICA, BEETLES
from tests.test_treatment_means import CALCIUM

SPEC_IDS = (
    [f"R2.{i}" for i in range(1, 11)] + [f"B0.{i}" for i in range(1, 4)]
    + [f"B1.{i}" for i in range(1, 8)] + [f"B2.{i}" for i in range(1, 6)] + ["B3.1"]
    + [f"B4.{i}" for i in range(1, 9)] + [f"B5.{i}" for i in range(1, 4)]
    + [f"A1.{i}" for i in range(1, 6)] + [f"A2.{i}" for i in range(1, 7)]
    + [f"G.{i}" for i in range(1, 5)] + [f"DIAG.{i}" for i in range(1, 5)]
)
# Сверх спецификации v2 — правила по Witte, Fig. 21.1 (стр. 405–406).
WITTE_IDS = ["B0.4", "R2.11"]
# Добавлены при исправлениях (ноябрь 2026): догадка о числе испытаний m.
EXTRA_IDS = ["R2.12"]


def _table(columns: dict[str, list]):
    """CSV из колонок -> ParsedTable тем же путём, что файл на сайте."""
    names = list(columns)
    rows = zip(*(columns[n] for n in names))
    text = ",".join(names) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n"
    return parse_file(text.encode("utf-8"))


def _fired(columns: dict[str, list], **roles: str) -> dict[str, dict]:
    out = recognize(_table(columns), roles)
    return {f["id"]: f for f in out["fired"]}


def _potato() -> dict[str, list]:
    data: dict[str, list] = {"block": [], "treatment": [], "yield": []}
    for b, cells in LAYOUT.items():
        for p, t in enumerate(cells):
            data["block"].append(f"B{b}")
            data["treatment"].append(t)
            data["yield"].append(PLOT_YIELD[b][p])
    return data


# --- Реестр ---------------------------------------------------------------------

def test_registry_matches_spec():
    """56 правил спецификации + правила Witte + R2.12, без дублей; реализованы 40
    структурных (+ Witte, + R2.12) и 15 расчётных; без реализации только A2.6
    (проверяет сам метод)."""
    from core.validation import recognizers  # noqa: F401

    ids = [r.id for r in rz.REGISTRY]
    added = WITTE_IDS + EXTRA_IDS
    assert sorted(ids) == sorted(SPEC_IDS + added)
    assert len(ids) == len(set(ids)) == 56 + len(added)
    done = [r for r in rz.REGISTRY if r.fn is not None]
    assert sum(r.cls == rz.STRUCTURAL for r in done) == 40 + len(added)
    assert sum(r.cls == rz.COMPUTED for r in done) == 15
    assert [r.id for r in rz.REGISTRY if r.fn is None] == ["A2.6"]
    for r in rz.REGISTRY:
        assert r.strength in (rz.STRONG, rz.HINT) and r.source


def test_output_is_json_and_message_codes():
    out = recognize(_table(_potato()), {"value": "yield", "group": "treatment",
                                        "block": "block"})
    json.dumps(out, allow_nan=False)
    assert len(out["registry"]) == 56 + len(WITTE_IDS + EXTRA_IDS)
    assert all(f["message"]["code"] == "rec_" + f["id"].replace(".", "_")
               for f in out["fired"])


def test_failing_rule_does_not_break_others(monkeypatch):
    def boom(ctx):
        raise RuntimeError("сбой правила")

    bad = rz.Recognizer("X.1", "X", "x", rz.HINT, rz.STRUCTURAL, "test", fn=boom)
    monkeypatch.setattr(rz, "REGISTRY", rz.REGISTRY + [bad])
    out = recognize(_table(_potato()), {"value": "yield", "group": "treatment",
                                        "block": "block"})
    ids = {f["id"] for f in out["fired"]}
    assert "X.1" not in ids and "B1.2" in ids


def test_unknown_role_columns_are_ignored():
    fired = _fired(_potato(), value="нет_такой", group="тоже_нет")
    assert "B3.1" not in fired and "B1.2" not in fired


# --- B1. Структура единиц на дизайнах книги ---------------------------------------

def test_rcbd_potato_table_7_1():
    fired = _fired(_potato(), value="yield", group="treatment", block="block")
    assert "B1.2" in fired and fired["B1.2"]["message"]["params"] == {"t": 5, "b": 4}
    assert not {"B1.3", "B1.4", "B1.7"} & set(fired)


def test_subsampling_potato_rows_table_7_5():
    data: dict[str, list] = {"block": [], "treatment": [], "yield": []}
    for b, cells in LAYOUT.items():
        for p, t in enumerate(cells):
            for v in ROWS[b][p]:
                data["block"].append(f"B{b}")
                data["treatment"].append(t)
                data["yield"].append(v)
    fired = _fired(data, value="yield", group="treatment", block="block")
    assert fired["B1.3"]["message"]["params"] == {"k": 4}
    assert "B1.2" not in fired


@pytest.mark.parametrize("design,lam", [(BIBD_A, 1), (BIBD_B, 2)])
def test_bibd_table_9_20(design, lam):
    data = {"block": [f"B{i}" for i, b in enumerate(design, 1) for _ in b],
            "trt": [f"T{t}" for b in design for t in b]}
    data["y"] = list(np.round(np.linspace(10, 20, len(data["trt"])), 2))
    fired = _fired(data, value="y", group="trt", block="block")
    assert fired["B1.4"]["message"]["params"] == {"lam": lam}
    assert not {"B1.2", "B1.7"} & set(fired)


def test_latin_square_lupin_table_9_1():
    data: dict[str, list] = {"row": [], "col": [], "trt": [], "height": []}
    for i, cells in enumerate(LUPIN_LS, 1):
        for j, cell in enumerate(cells, 1):
            trt, value = cell.split()
            for k, v in zip(data, (f"R{i}", f"C{j}", trt, value)):
                data[k].append(v)
    roles = dict(value="height", group="trt", row="row", column="col")
    assert "B1.5" in _fired(data, **roles)
    # Испорченная ячейка: вариант CH дважды в первом ряду — уже не квадрат.
    # (Перестановка ролей ряд/вариант квадрат не ломает: он симметричен.)
    data["trt"][1] = "CH"
    assert "B1.5" not in _fired(data, **roles)


def test_split_plot_competition_table_9_10():
    data: dict[str, list] = {"plot": [], "irr": [], "species": [], "y": []}
    for block, plots in COMPETITION.items():
        for wp, (irr, cells) in enumerate(plots, 1):
            for species, value in cells:
                sp = species if species != "-" else "none_sp"
                for k, v in zip(data, (f"{block}-{wp}", irr, sp, value)):
                    data[k].append(v)
    roles = dict(value="y", whole_plot="plot")
    assert "B1.6" in _fired(data, factor_a="irr", factor_b="species", **roles)
    assert "B1.6" not in _fired(data, factor_a="species", factor_b="irr", **roles)


def test_unequal_replication_flags_b1_7():
    data = _potato()
    for key in data:
        data[key] = data[key][:-1]  # одна делянка потеряна
    fired = _fired(data, value="yield", group="treatment", block="block")
    assert "B1.7" in fired and "B1.2" not in fired


def _noise(n: int) -> np.ndarray:
    """N(0, 1) без генератора: нормальные квантили в перемешанном порядке."""
    from scipy import stats

    q = stats.norm.ppf((np.arange(n) + 0.5) / n)
    return q[(np.arange(n) * 7 + 3) % n] if n % 7 else q[(np.arange(n) * 5 + 3) % n]


def _factorial_blocks(samples: int = 1, b_effect: float = 5.0) -> dict[str, list]:
    """Факториал A (2) × B (3) в 4 полных блоках; ``samples`` образцов на делянку.

    Синтетика: отклик = 50 + эффект A + сильный эффект B + блок + шум N(0, 1).
    """
    data: dict[str, list] = {"blk": [], "A": [], "B": [], "unit": [], "y": []}
    noise = iter(_noise(24 * samples))
    for b in range(4):
        for i, a in enumerate(("a1", "a2")):
            for j, lv in enumerate(("b1", "b2", "b3")):
                for _ in range(samples):
                    y = 50 + 2 * i + b_effect * (j - 1) + b + next(noise)
                    row = (f"k{b + 1}", a, lv, f"k{b + 1}-{a}-{lv}", round(float(y), 3))
                    for key, value in zip(data, row):
                        data[key].append(value)
    return data


_FACTORIAL_ROLES = dict(value="y", factor_a="A", factor_b="B", block="blk")


def test_factorial_in_blocks_is_rcbd_not_subsamples():
    """Вариант — комбинация факторов: в ячейке блок × (A, B) одна строка.

    Раньше ячейкой считалась пара блок × A (по 3 строки) и B1.3 давало strong
    «субвыборки». Теперь B1.2 ведёт к factorial_anova (блок — его слот).
    """
    fired = _fired(_factorial_blocks(), **_FACTORIAL_ROLES)
    assert "B1.3" not in fired and "B1.7" not in fired
    params = fired["B1.2"]["message"]["params"]
    assert params == {"t": 6, "b": 4, "method": "factorial_anova", "variant": "factorial"}
    assert "B2.1" in fired


def test_factorial_in_blocks_with_two_samples_per_plot_is_subsampling():
    fired = _fired(_factorial_blocks(samples=2), unit="unit", **_FACTORIAL_ROLES)
    assert fired["B1.3"]["message"]["params"] == {"k": 2}
    assert "B1.2" not in fired


def test_factorial_in_blocks_lost_plot_is_unbalanced():
    data = {k: v[:-1] for k, v in _factorial_blocks().items()}
    fired = _fired(data, **_FACTORIAL_ROLES)
    assert "B1.7" in fired and "B1.2" not in fired and "B1.3" not in fired


def test_one_factor_rcbd_signal_unchanged():
    """Однофакторный RCBD картофеля (Table 7.1): прежний текст, без method."""
    fired = _fired(_potato(), value="yield", group="treatment", block="block")
    assert fired["B1.2"]["message"]["params"] == {"t": 5, "b": 4}


def test_split_plot_with_blocks_is_not_reported_as_factorial_rcbd():
    """Split-plot (Table 9.10): комбинации тоже раз в блоке, но рандомизация в два этапа."""
    data: dict[str, list] = {"block": [], "plot": [], "irr": [], "species": [], "y": []}
    for block, plots in COMPETITION.items():
        for wp, (irr, cells) in enumerate(plots, 1):
            for species, value in cells:
                sp = species if species != "-" else "none_sp"
                for k, v in zip(data, (f"B{block}", f"{block}-{wp}", irr, sp, value)):
                    data[k].append(v)
    roles = dict(value="y", factor_a="irr", factor_b="species", block="block")
    assert "B1.2" in _fired(data, **roles)  # без роли делянки отличить нельзя
    fired = _fired(data, whole_plot="plot", **roles)
    assert "B1.6" in fired and "B1.2" not in fired


def test_residual_df_counts_factor_combinations():
    """DIAG.3: в факториале 2 × 3 в 4 блоках остаточных df (6 − 1)(4 − 1) = 15 ≥ 10."""
    assert "DIAG.3" not in _fired(_factorial_blocks(), **_FACTORIAL_ROLES)
    two_blocks = {k: v[:12] for k, v in _factorial_blocks().items()}
    assert _fired(two_blocks, **_FACTORIAL_ROLES)["DIAG.3"]["message"]["params"] == {"df": 5}


def test_block_name_hint_only_without_block_role():
    assert "B1.1" in _fired(_potato(), value="yield", group="treatment")
    assert "B1.1" not in _fired(_potato(), value="yield", group="treatment", block="block")


# --- B2. Структура вариантов -------------------------------------------------------

def _beetles() -> dict[str, list]:
    data: dict[str, list] = {"species": [], "mate": [], "eggs": []}
    for (s, m), vals in BEETLES.items():
        for v in vals:
            for k, x in zip(data, (s, m, v)):
                data[k].append(x)
    return data


def test_full_factorial_beetles_example_8_1():
    fired = _fired(_beetles(), value="eggs", factor_a="species", factor_b="mate")
    assert "B2.1" in fired and "B2.2" not in fired and "B5.3" in fired


def test_missing_cell_is_unbalanced():
    data = _beetles()
    keep = [i for i, (s, m) in enumerate(zip(data["species"], data["mate"]))
            if not (s == "vulg" and m == "intra")]
    data = {k: [v[i] for i in keep] for k, v in data.items()}
    fired = _fired(data, value="eggs", factor_a="species", factor_b="mate")
    assert fired["B2.2"]["message"]["params"] == {"missing": 1}
    assert "B2.1" not in fired


def test_nested_isolates_table_8_10():
    data: dict[str, list] = {"rep": [], "species": [], "isolate": [], "res": []}
    for rep, cells in _BRASSICA.items():
        for cell in cells.split(","):
            iso, res = cell.split()
            for k, v in zip(data, (rep, iso[0], iso, res)):
                data[k].append(v)
    assert "B2.3" in _fired(data, value="res", group="species", member="isolate")
    assert "B2.3" not in _fired(data, value="res", group="species", member="rep")


def test_control_level_and_dunnett_hint():
    fired = _fired(_potato(), value="yield", group="treatment", block="block")
    assert fired["B2.4"]["message"]["params"] == {"c": "treatment", "lv": "Control"}
    assert "B5.2" in fired


# --- B3–B5, DIAG -------------------------------------------------------------------

def test_levels_and_low_residual_df_calcium():
    data = {"ca": [g for g, v in CALCIUM.items() for _ in v],
            "root": [x for v in CALCIUM.values() for x in v]}
    fired = _fired(data, value="root", group="ca")
    assert fired["B3.1"]["message"]["params"]["branch"] == "B4b"
    assert "DIAG.3" not in fired  # df = 20 − 4 = 16
    two = {k: v[:10] for k, v in data.items()}  # группы A и B: df = 8
    fired = _fired(two, value="root", group="ca")
    assert fired["B3.1"]["message"]["params"]["branch"] == "B4a"
    assert fired["DIAG.3"]["message"]["params"] == {"df": 8}


def test_rcbd_residual_df_uses_blocks():
    """RCBD картофеля: df = (5 − 1)(4 − 1) = 12 ≥ 10, предупреждения нет."""
    assert "DIAG.3" not in _fired(_potato(), value="yield", group="treatment", block="block")


def test_numeric_levels_suggest_trend():
    data = {"dose": [d for d in (0, 10, 20, 40) for _ in range(3)],
            "y": list(range(12))}
    data["dose"] = [f"{d}" for d in data["dose"]]
    t = _table(data)
    # Уровни-числа определяются как числовая колонка — назначаем её фактором явно.
    out = recognize(t, {"value": "y", "group": "dose"})
    assert "B5.1" in {f["id"] for f in out["fired"]}


# --- R2. Тип отклика (данные гл. 18) ---------------------------------------------------

def test_trials_column_demethylation():
    data = {"dose": [r[0] for r in DEMETHYLATION], "y": [r[1] for r in DEMETHYLATION],
            "m": [r[2] for r in DEMETHYLATION]}
    fired = _fired(data, value="y", trials="m")
    assert fired["R2.3"]["message"]["params"] == {"m": "m"}
    # m = 60–100, но доли у краёв (59 из 60): logit книга не разрешает.
    assert "R2.10" not in fired and "R2.9" not in fired and "R2.1" not in fired
    # Без слота «число испытаний» — только подсказка, не strong.
    fired = _fired(data, value="y")
    assert "R2.3" not in fired
    assert fired["R2.12"]["strength"] == "hint"
    assert fired["R2.12"]["message"]["params"] == {"m": "m"}


def test_small_trials_suggest_glm():
    data = {"y": [1, 3, 0, 4, 2, 5], "m": [6, 6, 6, 6, 6, 6]}
    fired = _fired(data, value="y", trials="m")
    assert "R2.3" in fired and fired["R2.9"]["message"]["params"] == {"m": 6.0}
    assert "R2.9" not in _fired(data, value="y")  # m не назначено — правило молчит


def _aphids_by_year() -> dict[str, list]:
    """Синтетика: 4 варианта × 6 повторов, счёты тли (средние 5…20), год сбора 2024."""
    counts = {"t1": [3, 5, 4, 7, 6, 5], "t2": [9, 12, 8, 11, 10, 13],
              "t3": [14, 17, 13, 16, 18, 15], "t4": [19, 23, 18, 21, 22, 20]}
    return {"trt": [t for t, v in counts.items() for _ in v],
            "aphids": [x for v in counts.values() for x in v], "year": [2024] * 24}


def test_year_column_is_not_trials():
    """Год ≥ счёта в каждой строке, но это не число испытаний.

    Раньше R2.3 давало strong «доля из m, m = year» и уводило в биномиальную GLM.
    """
    fired = _fired(_aphids_by_year(), value="aphids", group="trt")
    assert "R2.3" not in fired and "R2.12" not in fired
    assert "R2.2" in fired  # счёты → Пуассон, как и должно быть


def test_plot_numbers_are_not_trials():
    data = _aphids_by_year()
    data["plot"] = list(range(101, 125))
    assert "R2.12" not in _fired(data, value="aphids", group="trt")


def _successes_of_20() -> dict[str, list]:
    """Синтетика: успехи из m = 20, доли ~0.2 / 0.4 / 0.6 / 0.8 по вариантам."""
    k = {"a": [3, 4, 5, 4, 2, 6], "b": [8, 7, 9, 8, 10, 6],
         "c": [12, 13, 11, 12, 14, 10], "d": [16, 17, 15, 18, 16, 14]}
    return {"trt": [t for t, v in k.items() for _ in v],
            "k": [x for v in k.values() for x in v], "m": [20] * 24}


def test_constant_trials_column_hint_then_strong_with_slot():
    """Постоянное m (20 семян на чашку) — обычный случай: без слота hint, со слотом strong."""
    fired = _fired(_successes_of_20(), value="k", group="trt")
    assert "R2.3" not in fired and fired["R2.12"]["message"]["params"] == {"m": "m"}
    assert fired["R2.12"]["strength"] == "hint" and fired["R2.12"]["suggests"] == "G1"
    fired = _fired(_successes_of_20(), value="k", group="trt", trials="m")
    assert fired["R2.3"]["strength"] == "strong" and "R2.12" not in fired


def test_column_with_another_role_is_not_guessed_as_trials():
    assert "R2.12" not in _fired(_successes_of_20(), value="k", group="trt", covariate="m")


# --- B0.3. Повторные измерения ---------------------------------------------------------

def test_treatment_factor_is_not_a_repeated_object():
    """Год один на весь файл, а «объект» — фактор вариантов с повторностями.

    Раньше B0.3 сообщало «повторные измерения, объекты = trt».
    """
    assert "B0.3" not in _fired(_aphids_by_year(), value="aphids", group="trt")
    two_years = _aphids_by_year()
    two_years["year"] = [2023, 2024] * 12  # вариант × год: по 3 строки в ячейке
    assert "B0.3" not in _fired(two_years, value="aphids", group="trt")


def test_objects_measured_at_several_times_are_repeated_measures():
    data = {"plant": [p for p in ("p1", "p2", "p3", "p4") for _ in range(3)],
            "day": [1, 7, 14] * 4,
            "height": [5.1, 9.8, 15.2, 4.7, 9.1, 14.0, 5.5, 10.4, 16.1, 4.9, 9.5, 14.8]}
    fired = _fired(data, value="height")
    assert fired["B0.3"]["message"]["params"] == {"t": "day", "o": "plant"}


def test_counts_aphids():
    data = {"field": [str(r[0]) for r in APHIDS], "y": [r[2] for r in APHIDS]}
    fired = _fired(data, value="y", group="field")
    assert "R2.2" in fired and "R2.1" not in fired


def test_integer_yield_is_not_counts():
    """Урожай картофеля записан целыми (333–708), но это не счёты."""
    fired = _fired(_potato(), value="yield", group="treatment", block="block")
    assert not {"R2.2", "R2.6", "G.2"} & set(fired)


def test_binary_response():
    data = {"g": ["a", "b"] * 10, "dead": [0, 1, 1, 0, 1] * 4}
    fired = _fired(data, value="dead", group="g")
    assert {"R2.1", "G.3"} <= set(fired) and "R2.2" not in fired


def test_label_only_file_suggests_contingency():
    data = {"variety": ["A", "B"] * 15, "disease": ["x", "y", "z"] * 10}
    assert "R2.5" in _fired(data)


def test_right_skew_and_detection_limit():
    rng = np.random.default_rng(1)
    skewed = list(np.round(rng.lognormal(0, 1, 40), 3))
    assert "R2.7" in _fired({"y": skewed}, value="y")
    censored = [0.05] * 6 + list(np.round(rng.uniform(0.1, 2, 34), 3))
    assert "R2.8" in _fired({"y": censored}, value="y")


# --- B0, A ----------------------------------------------------------------------------

def test_before_after_names_and_correlated_twins():
    rng = np.random.default_rng(2)
    base = rng.normal(50, 10, 20)
    data = {"weight_before": list(np.round(base, 1)),
            "weight_after": list(np.round(base + rng.normal(2, 1, 20), 1))}
    fired = _fired(data)
    assert "B0.1" in fired and "B0.2" in fired


def test_regression_hints():
    rng = np.random.default_rng(3)
    x = [d for d in (1, 2, 3, 4) for _ in range(3)]
    data = {"x": x, "y": list(np.round(np.array(x) * 2 + rng.normal(0, 1, 12), 2))}
    fired = _fired(data, x="x", y="y")
    assert fired["A2.1"]["message"]["params"] == {"k": 4}
    assert "A1.1" not in fired and "A1.5" not in fired  # две числовые колонки
    for name in ("z1", "z2", "z3"):
        data[name] = list(np.round(rng.normal(size=12), 2))
    fired = _fired(data, x="x", y="y")
    assert "A1.1" in fired
    assert fired["A1.5"]["message"]["params"]["k"] == 4
    assert fired["A1.5"]["message"]["params"]["method"] == "regression_subsets"
    assert fired["A1.5"]["message"]["params"]["variant"] == "subsets"


def test_scores_point_to_rank_test_by_group_count():
    """Witte, Fig. 21.1 (стр. 405): ранги/баллы — U при двух группах, H при трёх и более."""
    scores = [1, 3, 2, 5, 4, 6, 2, 7, 3, 5, 4, 6]
    two = _fired({"g": ["a"] * 6 + ["b"] * 6, "y": scores}, value="y", group="g")
    assert two["B4.8"]["message"]["params"]["method"] == "mann_whitney"
    three = _fired({"g": ["a", "b", "c"] * 4, "y": scores}, value="y", group="g")
    assert three["B4.8"]["message"]["params"]["method"] == "kruskal_wallis"


def test_paired_scores_point_to_wilcoxon():
    """Witte, Fig. 21.1 (стр. 405): пары в рангах/баллах — T Уилкоксона; измерения — молчит."""
    data = {"before": [3, 4, 2, 5, 3, 4, 2, 3], "after": [5, 6, 3, 6, 5, 5, 4, 4]}
    assert _fired(data, before="before", after="after")["B0.4"]["suggests"] == "wilcoxon"
    data = {"before": [61.2, 70.4, 58.3, 66.0], "after": [63.1, 71.9, 60.2, 66.8]}
    assert "B0.4" not in _fired(data, before="before", after="after")


def test_consecutive_integer_codes_hint_at_classes():
    """Witte, 21.4 (стр. 406): 1/2/3 могут быть классами, а не количествами → χ²."""
    data = {"g": ["a", "b"] * 15, "y": [1, 2, 3] * 10}
    params = _fired(data, value="y", group="g")["R2.11"]["message"]["params"]
    assert params == {"c": "y", "lo": 1, "hi": 3}
    gaps = {"g": ["a", "b"] * 15, "y": [1, 3, 5] * 10}  # не подряд — не коды
    assert "R2.11" not in _fired(gaps, value="y", group="g")
    few = {"g": ["a", "b"] * 3, "y": [1, 2, 3] * 2}  # < 5 строк на код
    assert "R2.11" not in _fired(few, value="y", group="g")
