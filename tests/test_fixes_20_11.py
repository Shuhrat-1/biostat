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

from core.stats.anova import one_way_anova
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

@pytest.mark.parametrize("method", [one_way_anova, welch_anova])
def test_3_group_means_carry_se_and_ci_like_the_plot(method) -> None:
    r = method(*GROUPS.values(), labels=list(GROUPS))
    plot = bar_means_data(GROUPS)["errors"]
    for est, values, se in zip(r.estimates, GROUPS.values(), plot):
        arr = np.asarray(values)
        assert est.std_error == pytest.approx(se) and est.ci_level == 0.95
        half = stats.t.ppf(0.975, arr.size - 1) * se  # ДИ по t с n − 1 df
        assert (est.ci_lower, est.ci_upper) == pytest.approx((arr.mean() - half, arr.mean() + half))


def test_3_single_observation_has_no_se() -> None:
    r = one_way_anova([5.0, 5.2], [4.0], labels=["a", "b"])
    one = next((e for e in r.estimates if e.name == "mean_b"), None)
    assert one is None or (one.std_error is None and one.ci_lower is None)


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
    lang_effect = app[app.index("// При смене языка переразбираем"):]
    assert "nextTable" not in lang_effect[:lang_effect.index("}, [lang]);")]


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
