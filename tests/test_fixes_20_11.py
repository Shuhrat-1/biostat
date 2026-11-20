"""Исправления 20.11.2026 (сообщение владельца, пять пунктов).

1. На 375 px страница не шире экрана: таблица оценок — в контейнере со своей
   прокруткой, списки и источник сигнала на карте не раздвигают страницу.
2. Строки таблицы ANOVA (Between, Within, Total, Residual…) — на ru / en / pt.
3. Оценки «пшеница — среднее» с ДИ и SE (те же SE, что у графика «средние ± SE»).
4. Смена языка не сбрасывает выбор метода; в en / pt не осталось русских подписей.
5. web/tsconfig.tsbuildinfo не лежит в git.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from core.stats.anova import one_way_anova, treatment_means
from core.stats.oneway import welch_anova
from core.stats.plotdata import bar_means_data
from core.validation.catalog import CATALOG

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")
LABELS = (WEB / "extraLabels.ts").read_text(encoding="utf-8")
GROUPS = {"пшеница": [5.1, 5.3, 5.5, 5.2], "ячмень": [4.6, 4.8, 4.9], "овёс": [3.9, 4.0]}


# --- 1 -------------------------------------------------------------------------------------

def test_1_estimates_table_scrolls_inside_its_own_box() -> None:
    i = COMPONENTS.index("{result.estimates.length > 0 && (")
    assert '<div style={{ overflowX: "auto", marginBottom: "0.9rem" }}>' in COMPONENTS[i:i + 200]
    assert 'maxWidth: "100%"' in (WEB / "styles.ts").read_text(encoding="utf-8")
    assert '<div style={{ maxWidth: "100%" }}>' in COMPONENTS  # список параметра
    assert 'whiteSpace: "nowrap"' not in (WEB / "SensoryMap.tsx").read_text(encoding="utf-8")


# --- 2 -------------------------------------------------------------------------------------

@pytest.mark.parametrize(("lang", "rows"), [
    ("ru", ("Между группами", "Внутри групп", "Всего")),
    ("en", ("Between groups", "Within groups", "Total")),
    ("pt", ("Entre grupos", "Dentro dos grupos", "Total")),
])
def test_2_anova_rows_are_translated(lang: str, rows: tuple[str, str, str]) -> None:
    block = LABELS[LABELS.index(f"  {lang}: {{", LABELS.index("const TERMS")):]
    block = block[:block.index("},")]
    for term, label in zip(("Between", "Within", "Total"), rows):
        assert f'{term}: "{label}"' in block, (lang, term)
    assert "termLabel(r.term, lang)" in COMPONENTS
    assert "termLabel(String(r.term), lang)" in (WEB / "resultExport.ts").read_text(encoding="utf-8")


# --- 3 -------------------------------------------------------------------------------------

def test_3_one_way_means_use_the_anova_residual_like_treatment_means() -> None:
    """Решение 20.11: SE = √(MS_within / n_i), ДИ по t с df остатка (Welham, стр. 85)."""
    r = one_way_anova(*GROUPS.values(), labels=list(GROUPS))
    ref = treatment_means(*GROUPS.values(), labels=list(GROUPS))
    ms, df = r.statistics["ms_within"], r.statistics["df_within"]
    for est, ref_est, values in zip(r.estimates, ref.estimates, GROUPS.values()):
        assert est.std_error == pytest.approx(np.sqrt(ms / len(values)))
        assert (est.std_error, est.ci_lower, est.ci_upper) == pytest.approx(
            (ref_est.std_error, ref_est.ci_lower, ref_est.ci_upper))
        half = stats.t.ppf(0.975, df) * est.std_error
        assert est.ci_upper - est.value == pytest.approx(half)
    assert r.parameters["se_basis"] == "pooled"


def test_3_means_plot_uses_the_same_se_as_the_table() -> None:
    r = one_way_anova(*GROUPS.values(), labels=list(GROUPS))
    plot = bar_means_data(GROUPS, pooled=True)
    assert plot["pooled"] is True
    assert plot["errors"] == pytest.approx([e.std_error for e in r.estimates])
    bridge = (WEB / "pyodide" / "bridge.ts").read_text(encoding="utf-8")
    assert '_groups, pooled=_method in ("one_way_anova", "treatment_means"))' in bridge
    assert 'add(bundle.bar_means.pooled ? "bar_means_pooled" : "bar_means"' in (
        WEB / "charts.tsx").read_text(encoding="utf-8")
    assert 'result.parameters?.se_basis === "pooled"' in COMPONENTS


def test_3_welch_keeps_the_group_se() -> None:
    """У Уэлча дисперсии групп не считаются равными: SE — по самой группе."""
    r = welch_anova(*GROUPS.values(), labels=list(GROUPS))
    assert [e.std_error for e in r.estimates] == pytest.approx(bar_means_data(GROUPS)["errors"])


def test_3_estimate_names_read_as_group_and_kind() -> None:
    assert "estimateLabel(e.name, result.parameters?.labels, lang)" in COMPONENTS
    assert "if (!m || !Array.isArray(groups) || !groups.includes(m[2])) return name;" in LABELS
    assert 'ru: { mean: "среднее", median: "медиана" }' in LABELS


# --- 4 -------------------------------------------------------------------------------------

def test_4_language_switch_keeps_the_method() -> None:
    panel = COMPONENTS[COMPONENTS.index("export function MethodPanel("):]
    assert "const sameTable = lastTable.current === tableKey;" in panel
    assert "if (sameTable && current) {" in panel and "setActiveGroup(current.group);" in panel
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    assert app.count("nextTable();") == 4 and "tableKey={tableKey}" in app
    lang_effect = app[app.index("// При смене языка подписи методов"):]
    lang_effect = lang_effect[:lang_effect.index("}, [lang]);")]
    assert "nextTable" not in lang_effect and "runParse" not in lang_effect  # без повторного разбора
    assert "if (table) void relabel();" in lang_effect
    assert "setTable((cur) => (cur ? { ...cur, suggestions } : cur));" in app
    assert "export async function tableSuggestions(lang: string)" in (
        WEB / "pyodide" / "bridge.ts").read_text(encoding="utf-8")


def _all_sources() -> set[str]:
    import json

    from core.validation import recognize
    import core.validation.recognizers  # noqa: F401
    import core.validation.recognizers_computed  # noqa: F401

    found = {r.source for r in recognize.REGISTRY}

    def walk(v: object) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "source" and isinstance(x, str):
                    found.add(x)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(json.loads((WEB / "decisionTree.json").read_text(encoding="utf-8")))
    return found


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_4_source_hints_are_in_the_interface_language(lang: str) -> None:
    """«предл.: Witte гл. 3–4» и другие пометки источника — без русских слов."""
    sensory = (WEB / "SensoryMap.tsx").read_text(encoding="utf-8")
    block = sensory[sensory.index(f"  {lang}: [["):]
    block = block[:block.index("]],") + 2]
    pairs = re.findall(r'\["([^"]+)", "([^"]+)"\]', block)
    for src in _all_sources():
        out = src
        for ru, tr in pairs:
            out = out.replace(ru, tr)
        assert not re.search("[а-яё]", out, re.I), (lang, src, out)


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_4_no_russian_left_in_the_method_catalog(lang: str) -> None:
    cyr = re.compile("[а-яё]", re.I)
    for spec in CATALOG:
        d = spec.to_dict([], lang)
        texts = [d["label"], d["description"], d["group"], *d["slot_labels"].values(),
                 *(p["label"] for p in d["params"])]
        assert not [x for x in texts if cyr.search(x)], (spec.method, lang)


# --- 5 -------------------------------------------------------------------------------------

def test_5_tsbuildinfo_is_ignored_not_tracked() -> None:
    assert "*.tsbuildinfo" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    tracked = subprocess.run(["git", "ls-files", "*.tsbuildinfo"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8")
    if tracked.returncode != 0:
        pytest.skip("не git-репозиторий")
    assert tracked.stdout.strip() == ""
