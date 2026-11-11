"""Сценарии карты выбора метода: данные → роли → ожидаемый лист → источник.

Собраны в аудите 1 (ноябрь 2026, ``claude/audit-01-report.md``): 46 типичных
планов опыта и ловушек. Каждый сценарий — словарь: id, title, columns
(колонки файла), stages (этапы: какие роли выставил пользователь), expect
(допустимые листья дерева), source (откуда известен верный метод). Данные
книг берутся из тестов, где они сверены с таблицами книг; остальное —
синтетика с заданным свойством, без генератора случайных чисел.

Регрессионные тесты карты — ``tests/test_map_scenarios.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from tests import witte_data as w
from tests.book_ch12_data import TRANSECT, TRITICUM
from tests.book_ch14_data import EXAMINE, EXAMINE_COLUMNS
from tests.book_ch15_data import FORAGE, FOREST, TGW
from tests.book_ch16_data import SOSR
from tests.book_ch17_data import OLSEN_P, TEMPERATURE, YIELD
from tests.book_ch18_data import APHIDS, DEMETHYLATION
from tests.test_blocking import LAYOUT, PLOT_YIELD
from tests.test_designs import BIBD_B, COMPETITION, LUPIN_LS
from tests.test_factorial import _BRASSICA, BEETLES as BEETLES_FACTORIAL
from tests.test_transforms import BEETLES as BEETLES_EGGS
from tests.test_treatment_means import CALCIUM

Columns = dict[str, list]


def q(n: int, shift: int = 0) -> np.ndarray:
    """N(0, 1) без генератора: нормальные квантили, циклически сдвинутые."""
    base = stats.norm.ppf((np.arange(n) + 0.5) / n)
    step = next(s for s in (7, 5, 3, 11) if n % s)
    return np.roll(base[(np.arange(n) * step) % n], shift)


def grouped(groups: dict[str, list | np.ndarray], g: str = "group", y: str = "y") -> Columns:
    return {g: [k for k, v in groups.items() for _ in v],
            y: [round(float(x), 4) for v in groups.values() for x in v]}


def cells(values: dict[tuple, list | np.ndarray], names: tuple[str, ...], y: str = "y") -> Columns:
    out: Columns = {n: [] for n in names}
    out[y] = []
    for key, vals in values.items():
        for v in vals:
            for n, k in zip(names, key):
                out[n].append(k)
            out[y].append(round(float(v), 4))
    return out


def scenario(sid: str, title: str, columns: Columns, stages: dict[str, dict[str, str]],
             expect: list[str], source: str, note: str = "") -> dict:
    return dict(id=sid, title=title, columns=columns, stages=stages, expect=expect,
                source=source, note=note)


# --- Данные книг в виде колонок -------------------------------------------------------

def potato() -> Columns:
    data: Columns = {"block": [], "treatment": [], "yield": []}
    for b, plots in LAYOUT.items():
        for p, t in enumerate(plots):
            for k, v in zip(data, (f"B{b}", t, PLOT_YIELD[b][p])):
                data[k].append(v)
    return data


def split_plot() -> Columns:
    data: Columns = {"block": [], "plot": [], "irrigation": [], "species": [], "y": []}
    for block, plots in COMPETITION.items():
        for wp, (irr, sub) in enumerate(plots, 1):
            for species, value in sub:
                sp = species if species != "-" else "none_sp"
                for k, v in zip(data, (f"B{block}", f"{block}-{wp}", irr, sp, value)):
                    data[k].append(v)
    return data


def latin() -> Columns:
    data: Columns = {"row": [], "col": [], "treatment": [], "y": []}
    for r, line in enumerate(LUPIN_LS, 1):
        for c, cell in enumerate(line, 1):
            trt, value = cell.split()
            for k, v in zip(data, (f"R{r}", f"C{c}", trt, float(value))):
                data[k].append(v)
    return data


def bibd() -> Columns:
    """Раскладка BIBD Table 9.20, план (b): 7 сортов, 7 блоков по 4 (отклик синтетический)."""
    data: Columns = {"block": [], "variety": [], "y": []}
    noise = iter(q(sum(len(b) for b in BIBD_B)))
    for b, trts in enumerate(BIBD_B, 1):
        for t in trts:
            for k, v in zip(data, (f"B{b}", f"V{t}", round(20 + t + 0.5 * b + next(noise), 3))):
                data[k].append(v)
    return data


def nested() -> Columns:
    data: Columns = {"rep": [], "species": [], "isolate": [], "resistance": []}
    for rep, line in _BRASSICA.items():
        for cell in line.split(","):
            iso, res = cell.split()
            for k, v in zip(data, (f"r{rep}", iso[0], iso, float(res))):
                data[k].append(v)
    return data


def letters() -> Columns:
    """Witte Table 19.3: потерянные письма, по строке на письмо."""
    table = {("no", "campus"): 19, ("no", "downtown"): 21, ("no", "suburbia"): 40,
             ("yes", "campus"): 51, ("yes", "downtown"): 39, ("yes", "suburbia"): 30}
    return {"returned": [r for (r, _), n in table.items() for _ in range(n)],
            "place": [p for (_, p), n in table.items() for _ in range(n)]}


def rm_long(matrix: list[list[float]], labels: list[str]) -> Columns:
    data: Columns = {"subject": [], "condition": [], "score": []}
    for s, row in enumerate(matrix, 1):
        for lab, v in zip(labels, row):
            for k, x in zip(data, (f"s{s}", lab, v)):
                data[k].append(x)
    return data


CA = {f"Ca{k}": v for k, v in CALCIUM.items()}


def build() -> list[dict]:
    s: list[dict] = []
    one_way = lambda g="group", y="y": {"one-way": {"value": y, "group": g}}  # noqa: E731

    # --- CRD, один фактор, две группы ---------------------------------------------------
    s.append(scenario(
        "S01", "2 группы, нормально, равные n (Witte, EPO)",
        grouped({"epo": w.EPO_X1, "control": w.EPO_X2}), one_way(), ["B4a.t"],
        "Witte Table 14.1, стр. 245: t для двух независимых выборок"))
    s.append(scenario(
        "S02", "2 группы, SD ×4, n 6 и 15, большая SD в меньшей группе",
        grouped({"a": 20 + 4 * q(6), "b": 22 + q(15)}), one_way(), ["B4a.t"],
        "Welham стр. 33 (Уэлч при неравных дисперсиях); decision-tree-spec B4.2",
        "ожидается сигнал B4.2#welch_liberal"))
    s.append(scenario(
        "S03", "2 группы, сильная правая асимметрия, n = 10",
        grouped({"a": np.exp(1 + q(10)), "b": np.exp(1.8 + q(10, 3))}), one_way(), ["B4a.mw"],
        "decision-tree-spec B4a; Welham стр. 115 (log) и стр. 127 (ранговые)"))
    s.append(scenario(
        "S04", "2 группы, баллы (Witte, ТВ)",
        grouped({"favorable": w.__dict__.get("TV_FAVORABLE", [4, 5, 5, 5, 10, 12, 20, 49]),
                 "unfavorable": [0, 0, 1, 2, 14, 42, 43]}), one_way(), ["B4a.mw"],
        "Witte Table 20.2, стр. 389: U Манна–Уитни"))

    # --- CRD, один фактор, 3+ групп -----------------------------------------------------
    s.append(scenario(
        "S05", "4 группы, нормально (Welham, кальций)", grouped(CA), one_way(), ["B4b.anova"],
        "Welham Table 4.5, стр. 81: однофакторный ANOVA"))
    s.append(scenario(
        "S06", "3 группы, неравные n 5 / 3 / 4 (Witte)",
        grouped(dict(zip(["h0", "h24", "h48"], w.SLEEP_UNEQUAL))), one_way(), ["B4b.anova"],
        "Witte Review 16.10, стр. 445: ANOVA при неравных n"))
    s.append(scenario(
        "S07", "4 группы, дисперсия растёт со средним (Welham, яйца жуков)",
        grouped({k: v for k, v in BEETLES_EGGS.items()}), one_way(), ["B4b.transform.means"],
        "Welham Example 6.1, Table 6.1, стр. 113–118: log-трансформация"))
    s.append(scenario(
        "S08", "3 группы, одна с SD ×5, средние не связаны с SD",
        grouped({"a": 10 + q(10), "b": 20 + 5 * q(10, 2), "c": 30 + q(10, 4)}), one_way(),
        ["B4b.welch"], "Welham стр. 107: взвешенный ANOVA или Уэлч"))
    s.append(scenario(
        "S09", "3 группы, остатки асимметричны, дисперсии равны",
        grouped({k: m + stats.expon.ppf((np.arange(10) + 0.5) / 10) for k, m in
                 (("a", 5), ("b", 6), ("c", 8))}), one_way(), ["B4b.permutation"],
        "Welham стр. 102–103: перестановочный тест"))
    outlier = {k: list(v) for k, v in CA.items()}
    outlier["CaC"][0] = outlier["CaC"][0] * 3
    s.append(scenario(
        "S10", "4 группы, одно значение ×3 (выброс; Welham, кальций)", grouped(outlier),
        one_way(), ["B4b.anova"],
        "Welham стр. 109–110: выброс — проверить запись, метод не менять",
        "ожидается предупреждение DIAG.1, а не смена метода"))
    s.append(scenario(
        "S11", "3 группы по n = 3 (Witte, сон)",
        grouped(dict(zip(["h0", "h24", "h48"], w.SLEEP_B))), one_way(), ["B4b.anova"],
        "Witte Tables 16.3, 16.6, стр. 300: однофакторный ANOVA",
        "малые выборки: срабатывают ли пороги вообще"))
    s.append(scenario(
        "S12", "3 группы, баллы 1–5",
        grouped({"a": [1, 2, 2, 3, 1, 2, 3, 2], "b": [3, 3, 4, 2, 4, 3, 5, 3],
                 "c": [4, 5, 5, 3, 4, 5, 4, 5]}), one_way(), ["B4b.kruskal"],
        "Welham стр. 127; Witte Fig. 21.1, стр. 405: H Краскела–Уоллиса"))

    # --- Блоки и структура вариантов ------------------------------------------------------
    s.append(scenario(
        "S13", "RCBD 5 × 4 (Welham, картофель)", potato(),
        {"one-way": {"value": "yield", "group": "treatment"},
         "full": {"value": "yield", "group": "treatment", "block": "block"}}, ["B4b.rcbd"],
        "Welham Table 7.4, стр. 137: ANOVA для RCBD"))
    beet = cells(BEETLES_FACTORIAL, ("species", "mate"), "eggs")
    s.append(scenario(
        "S14", "Факториал 2 × 2, CRD (Welham, жуки)", beet,
        {"one-way": {"value": "eggs", "group": "species"},
         "full": {"value": "eggs", "factor_a": "species", "factor_b": "mate"}}, ["B2.factorial"],
        "Welham Example 8.1, стр. 151–160: факторный ANOVA"))
    smoke = dict(zip(("minutes", "crowd", "danger"), w.SMOKE))
    s.append(scenario(
        "S15", "Факториал 3 × 2, n = 2 в ячейке (Witte, дым)",
        {"crowd": [f"c{x}" for x in smoke["crowd"]], "danger": smoke["danger"],
         "minutes": smoke["minutes"]},
        {"full": {"value": "minutes", "factor_a": "crowd", "factor_b": "danger"}},
        ["B2.factorial"], "Witte Table 18.3, стр. 350: двухфакторный ANOVA"))
    noise36 = q(36).reshape(9, 4)
    three = {(a, b): 20 + 3 * i + 2 * j + noise36[3 * i + j]
             for i, a in enumerate(("low", "mid", "high")) for j, b in enumerate(("v1", "v2", "v3"))}
    s.append(scenario(
        "S16", "Факториал 3 × 3, CRD, n = 4", cells(three, ("fert", "variety")),
        {"full": {"value": "y", "factor_a": "fert", "factor_b": "variety"}}, ["B2.factorial"],
        "Welham стр. 172: все комбинации, равная повторность"))
    fb: Columns = {"blk": [], "nitrogen": [], "variety": [], "y": []}
    noise = iter(q(24))
    for b in range(4):
        for i, a in enumerate(("n0", "n1")):
            for j, v in enumerate(("v1", "v2", "v3")):
                row = (f"k{b + 1}", a, v, round(50 + 2 * i + 5 * (j - 1) + b + next(noise), 3))
                for k, x in zip(fb, row):
                    fb[k].append(x)
    s.append(scenario(
        "S17", "Факториал 2 × 3 в 4 полных блоках", fb,
        {"no-block": {"value": "y", "factor_a": "nitrogen", "factor_b": "variety"},
         "full": {"value": "y", "factor_a": "nitrogen", "factor_b": "variety", "block": "blk"}},
        ["B2.factorial"], "Welham стр. 172, 166 (факториал с блоками)"))
    sub: Columns = {"block": [], "treatment": [], "plot": [], "y": []}
    noise = iter(q(40))
    for b in range(4):
        for t in range(5):
            for _ in range(2):
                row = (f"B{b + 1}", f"T{t + 1}", f"B{b + 1}-P{t + 1}",
                       round(30 + 2 * t + 3 * b + next(noise), 3))
                for k, x in zip(sub, row):
                    sub[k].append(x)
    s.append(scenario(
        "S18", "RCBD 5 × 4, по 2 измерения на делянку", sub,
        {"rcbd": {"value": "y", "group": "treatment", "block": "block"},
         "full": {"value": "y", "group": "treatment", "block": "block", "unit": "plot"}},
        ["B1.subsample"], "Welham стр. 145–147: субвыборки, анализ на уровне делянок"))
    sp = split_plot()
    s.append(scenario(
        "S19", "Split-plot (Welham, орошение × вид)", sp,
        {"factorial": {"value": "y", "factor_a": "irrigation", "factor_b": "species",
                       "block": "block"},
         "full": {"value": "y", "factor_a": "irrigation", "factor_b": "species",
                  "block": "block", "whole_plot": "plot"}}, ["B1.split"],
        "Welham Table 9.10–9.12, стр. 220–226: split-plot; как RCBD — неверно"))
    s.append(scenario(
        "S20", "Латинский квадрат 4 × 4 (Welham, люпин)", latin(),
        {"one-way": {"value": "y", "group": "treatment"},
         "full": {"value": "y", "factor_a": "treatment", "row": "row", "column": "col"}},
        ["B1.latin"], "Welham Table 9.1–9.2, стр. 209–212: латинский квадрат"))
    s.append(scenario(
        "S21", "BIBD: 7 сортов, 7 блоков по 4 (раскладка Welham)", bibd(),
        {"full": {"value": "y", "group": "variety", "block": "block"}}, ["B1.bibd"],
        "Welham Table 9.20, стр. 232–234: BIBD"))
    s.append(scenario(
        "S22", "Вложенная схема: изоляты внутри видов (Welham)", nested(),
        {"one-way": {"value": "resistance", "group": "isolate"},
         "full": {"value": "resistance", "group": "species", "member": "isolate"}},
        ["B2.nested"], "Welham Table 8.10, стр. 173–178: вложенный ANOVA"))

    # --- Связанные измерения ----------------------------------------------------------------
    s.append(scenario(
        "S23", "Парные до/после, широкий формат (Witte, GPA)",
        {"gpa_before": w.GPA_CONTROL, "gpa_after": w.GPA_EXERCISE},
        {"full": {"before": "gpa_before", "after": "gpa_after"}}, ["B0.paired.t"],
        "Witte Review 15.7, стр. 280: t для связанных выборок"))
    noise_rm = rm_long(w.RM_NOISE, w.RM_NOISE_LABELS)
    s.append(scenario(
        "S24", "Повторные измерения, 6 субъектов × 3 условия (Witte, шум)", noise_rm,
        {"one-way": {"value": "score", "group": "condition"},
         "full": {"value": "score", "condition": "condition", "subject": "subject"}},
        ["B0.rm", "B0.repeated"],
        "Witte Progress Check 17.2, стр. 331: ANOVA повторных измерений; Welham стр. 100"))
    plants: Columns = {"plant": [], "treatment": [], "day": [], "height": []}
    for p in range(8):
        for d_i, day in enumerate((7, 14, 21, 28)):
            row = (f"p{p + 1}", "ctrl" if p < 4 else "fert", day,
                   round(5 + (0.5 if p < 4 else 0.7) * day + q(32)[4 * p + d_i], 2))
            for k, x in zip(plants, row):
                plants[k].append(x)
    s.append(scenario(
        "S25", "Рост 8 растений, 4 даты, 2 варианта (время)", plants,
        {"one-way": {"value": "height", "group": "treatment"},
         "full": {"value": "height", "factor_a": "treatment", "stratum_1": "plant"}},
        ["B0.repeated"], "Welham стр. 100: повторные измерения коррелированы → смешанная модель"))

    # --- Регрессия ------------------------------------------------------------------------------
    xy = lambda x, y: {"xy": {"x": x, "y": y}}  # noqa: E731
    s.append(scenario(
        "S26", "Прямая: длина и масса семян, n = 190 (Welham)",
        {"weight": [r[1] for r in TRITICUM], "length": [r[2] for r in TRITICUM]},
        xy("weight", "length"), ["A3.slr", "A3.details"],
        "Welham Example 12.1, стр. 289–300: простая линейная регрессия"))
    s.append(scenario(
        "S27", "Неадекватность прямой: жуки по трансекте (Welham)",
        {"distance": [r[0] for r in TRANSECT], "count": [r[1] for r in TRANSECT]},
        xy("distance", "count"), ["A2.curve.xt", "A2.curve.poly", "A2.curve.nonlin"],
        "Welham Table 12.7, стр. 310–312: F неадекватности 4.32, p = 0.013"))
    s.append(scenario(
        "S28", "Кривая насыщения: урожай и Olsen P (Welham)",
        {"olsen_p": OLSEN_P, "yield": YIELD}, xy("olsen_p", "yield"),
        ["A2.curve.xt", "A2.curve.poly", "A2.curve.nonlin"],
        "Welham Table 17.1, Example 17.1, стр. 452–458: преобразование x / полином"))
    s.append(scenario(
        "S29", "Периодичность: температура по месяцам (Welham)",
        {"month": list(range(1, 13)), "temperature": TEMPERATURE}, xy("month", "temperature"),
        ["A2.periodic"], "Welham Table 17.5, стр. 460–462: тригонометрическая регрессия"))
    s.append(scenario(
        "S30", "Регрессия с группами: 3 типа древостоя (Welham)",
        {"stand": [r[0] for r in FOREST], "density": [r[1] for r in FOREST],
         "diameter": [r[2] for r in FOREST]},
        {"xy": {"x": "density", "y": "diameter"},
         "full": {"x": "density", "y": "diameter", "group": "stand"}}, ["A1.groups"],
        "Welham Table 15.1, Example 15.1, стр. 382–395: линии по группам"))
    s.append(scenario(
        "S31", "Доза N в блочном опыте 3 × 4 (Welham, фураж)",
        {"block": [f"B{r[0]}" for r in FORAGE], "nitrogen": [r[2] for r in FORAGE],
         "yield": [r[3] for r in FORAGE]},
        {"xy": {"x": "nitrogen", "y": "yield"},
         "full": {"value": "yield", "x": "nitrogen", "block": "block"}},
        ["A1.blocks.line", "A1.blocks.curve"],
        "Welham Table 15.11, Example 15.3, стр. 406–408: регрессия в блоках"))
    tgw = {"trt": [r[3].replace("+", "plus").replace("-", "minus") for r in TGW],
           "damage": [r[4] for r in TGW], "tgw": [r[5] for r in TGW]}
    s.append(scenario(
        "S32", "ANCOVA: масса 1000 зёрен, ковариата — повреждение % (Welham)", tgw,
        {"one-way": {"value": "tgw", "group": "trt"},
         "full": {"value": "tgw", "group": "trt", "covariate": "damage"}},
        ["B2.ancova", "A1.ancova"],
        "Welham Table 15.15, Example 15.4, стр. 409–416: ковариационный анализ"))
    ex = {c: [r[i] for r in EXAMINE] for i, c in enumerate(EXAMINE_COLUMNS)}
    s.append(scenario(
        "S33", f"Множественная регрессия: {len(EXAMINE_COLUMNS)} колонок (Welham, EXAMINE)", ex,
        {"y": {"y": EXAMINE_COLUMNS[-1]}},
        ["A1.multi.mlr", "A1.multi.subsets", "A1.multi.stepwise"],
        "Welham Example 14.1–14.2, стр. 349–372: множественная регрессия и отбор"))

    # --- Не нормальный по природе отклик --------------------------------------------------------
    dem = {"dose": [r[0] for r in DEMETHYLATION], "normal": [r[1] for r in DEMETHYLATION],
           "total": [r[2] for r in DEMETHYLATION]}
    s.append(scenario(
        "S34", "Доли k из m по дозе (Welham, деметилирование)", dem,
        {"value-only": {"value": "normal"},
         "full": {"value": "normal", "trials": "total", "covariate": "dose"}},
        ["G1.dose", "G1.glm"], "Welham Table 18.1, Example 18.1, стр. 481–487: биномиальная GLM"))
    germ = {"treatment": [t for t in ("ctrl", "scar", "soak") for _ in range(12)],
            "germinated": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                           1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]}
    s.append(scenario(
        "S35", "Бинарный отклик 0/1 (всхожесть), 3 варианта × 12 семян", germ,
        {"one-way": {"value": "germinated", "group": "treatment"},
         "full": {"value": "germinated", "factor_a": "treatment"}}, ["G1.glm"],
        "Welham стр. 501: бинарные данные — биномиальная GLM, m = 1"))
    aph = {"field": [f"F{r[0]}" for r in APHIDS], "aphids": [r[2] for r in APHIDS]}
    s.append(scenario(
        "S36", "Счёты со сверхдисперсией (Welham, тля, 3 поля × 15 проб)", aph,
        {"one-way": {"value": "aphids", "group": "field"},
         "full": {"value": "aphids", "factor_a": "field"}}, ["G2.poisson", "G2.negbin"],
        "Welham Table 18.10, Example 18.3, стр. 504–507: пуассоновская GLM с φ"))
    zeros = {"trap": [t for t in ("edge", "mid", "core") for _ in range(12)],
             "beetles": [0, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 12, 0, 2, 0, 0, 0, 9,
                         0, 4, 0, 0, 15, 0, 0, 0, 6, 0, 0, 21]}
    s.append(scenario(
        "S37", "Счёты, 2/3 нулей", zeros, {"one-way": {"value": "beetles", "group": "trap"}},
        ["G2.negbin"], "Welham стр. 126, 512: избыток нулей — не Пуассон"))
    scores = {"variety": [v for v in ("A", "B", "C") for _ in range(10)],
              "disease": (["healthy"] * 6 + ["mild"] * 3 + ["severe"] * 1
                          + ["healthy"] * 3 + ["mild"] * 4 + ["severe"] * 3
                          + ["healthy"] * 1 + ["mild"] * 3 + ["severe"] * 6)}
    s.append(scenario(
        "S38", "Порядковый балл метками: healthy / mild / severe", scores,
        {"full": {"response": "disease", "factor_a": "variety"}}, ["R2.ordinal"],
        "Welham стр. 513: порядковая регрессия"))
    s.append(scenario(
        "S39", "Две классификации, 200 строк-наблюдений (Witte, письма)", letters(),
        {"full": {"factor_a": "returned", "factor_b": "place"}}, ["R2.contingency"],
        "Witte Tables 19.3–19.5, стр. 372–375: χ² независимости"))

    # --- Наблюдение ---------------------------------------------------------------------------
    sosr = {"farm": [f"farm{r[0]}" for r in SOSR], "field": [f"f{r[0]}_{r[1]}" for r in SOSR],
            "year": [f"y{r[2]}" for r in SOSR],
            "log_weeds": [round(float(np.log10(r[3] + 1)), 4) for r in SOSR]}
    s.append(scenario(
        "S40", "Наблюдение: 59 полей на 37 фермах, 3 года (Welham, FSE)", sosr,
        {"one-way": {"value": "log_weeds", "group": "year"},
         "full": {"value": "log_weeds", "factor_a": "year", "stratum_1": "farm"}},
        ["B1.mixed", "B0.repeated"],
        "Welham Table 16.4, Example 16.1, стр. 427–440: смешанная модель"))

    # --- Ловушки ------------------------------------------------------------------------------
    codes = grouped({str(k): v for k, v in zip((1, 2, 3, 4), CALCIUM.values())}, g="variant")
    s.append(scenario(
        "S41", "Коды вариантов 1, 2, 3, 4 (те же данные, что S05)", codes,
        {"one-way": {"value": "y", "group": "variant"}}, ["B4b.anova"],
        "Welham Table 4.5; Witte стр. 406: число может быть кодом класса",
        "колонка вариантов определяется как числовая"))
    yearplot = grouped({"t1": [3, 5, 4, 7, 6, 5], "t2": [9, 12, 8, 11, 10, 13],
                        "t3": [14, 17, 13, 16, 18, 15], "t4": [19, 23, 18, 21, 22, 20]},
                       g="trt", y="aphids")
    yearplot["year"] = [2024] * 24
    yearplot["plot"] = list(range(101, 125))
    s.append(scenario(
        "S42", "Счёты + год 2024 + номер делянки 101–124", yearplot,
        {"one-way": {"value": "aphids", "group": "trt"}}, ["G2.poisson", "G2.negbin"],
        "Welham стр. 502: счёты — GLM; год и номер делянки не переменные анализа"))
    ident = grouped(CA)
    ident["sample_id"] = [f"S{i:03d}" for i in range(1, 21)]
    s.append(scenario(
        "S43", "Колонка ID образца (уникальная) + те же данные, что S05", ident, one_way(),
        ["B4b.anova"], "Welham Table 4.5"))
    gaps = grouped(CA)
    gaps["y"][3] = ""
    gaps["y"][11] = "NA"
    gaps["group"][17] = ""
    s.append(scenario(
        "S44", "Пропуски: пустая ячейка, NA, пустая метка группы", gaps, one_way(),
        ["B4b.anova"], "Welham стр. 85: CRD допускает неравную повторность"))
    s.append(scenario(
        "S45", "Один уровень фактора",
        grouped({"only": 10 + q(8)}), one_way(), ["D.onesample", "D.ci", "D.describe"],
        "сравнивать нечего: описание или одновыборочный тест"))
    s.append(scenario(
        "S46", "3 группы по 2 наблюдения",
        grouped({"a": [10.1, 11.4], "b": [13.0, 12.2], "c": [15.9, 17.1]}), one_way(),
        ["B4b.anova"], "Welham гл. 10, стр. 521: ≥ 10 остаточных df для надёжных выводов",
        "ожидается предупреждение DIAG.3 (df = 3)"))
    return s


SCENARIOS = build()
