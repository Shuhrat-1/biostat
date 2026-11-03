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
    """56 правил ровно из спецификации, без дублей; реализованы 40 структурных и
    15 расчётных; без реализации только A2.6 (проверяет сам метод)."""
    from core.validation import recognizers  # noqa: F401

    ids = [r.id for r in rz.REGISTRY]
    assert sorted(ids) == sorted(SPEC_IDS)
    assert len(ids) == len(set(ids)) == 56
    done = [r for r in rz.REGISTRY if r.fn is not None]
    assert sum(r.cls == rz.STRUCTURAL for r in done) == 40
    assert sum(r.cls == rz.COMPUTED for r in done) == 15
    assert [r.id for r in rz.REGISTRY if r.fn is None] == ["A2.6"]
    for r in rz.REGISTRY:
        assert r.strength in (rz.STRONG, rz.HINT) and r.source


def test_output_is_json_and_message_codes():
    out = recognize(_table(_potato()), {"value": "yield", "group": "treatment",
                                        "block": "block"})
    json.dumps(out, allow_nan=False)
    assert len(out["registry"]) == 56
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
    fired = _fired(data, value="y")
    assert fired["R2.3"]["message"]["params"] == {"m": "m"}
    # m = 60–100, но доли у краёв (59 из 60): logit книга не разрешает.
    assert "R2.10" not in fired and "R2.9" not in fired and "R2.1" not in fired


def test_small_trials_suggest_glm():
    data = {"y": [1, 3, 0, 4, 2, 5], "m": [6, 6, 6, 6, 6, 6]}
    fired = _fired(data, value="y")
    assert "R2.3" in fired and fired["R2.9"]["message"]["params"] == {"m": 6.0}


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
