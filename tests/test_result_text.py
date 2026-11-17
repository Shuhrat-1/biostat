"""Аудит 04-02: результат говорит вывод словами и не показывает ключи ядра.

(1) строка вывода под P по семейству гипотезы, α метода, P < 0.001;
(2) подписи статистик (`web/src/extraLabels.ts`) на трёх языках;
(3) таблица ANOVA у однофакторного ANOVA, RCBD и средних на шкале трансформации;
(4) название метода вместо id, без повтора η² и ω² в плашках.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
I18N = (WEB / "i18n.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _results() -> dict[str, dict]:
    import warnings

    from tests.bridge_runner import CASES, run, table

    warnings.filterwarnings("ignore")
    return {m: run(m, table(m)) for m in CASES}


def _node(script: str) -> object:
    exe = shutil.which("node")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout if exe else ""
    if not exe or tuple(int(x) for x in version.strip().lstrip("v").split(".")[:2]) < (22, 6):
        pytest.skip("нужен Node 22.6+ для запуска web/src/*.ts")
    out = subprocess.run([exe, "--experimental-strip-types", "--no-warnings", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _clean(value: object) -> object:
    """JSON без NaN: node не читает NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


@lru_cache(maxsize=1)
def _front(tmp: str = "/tmp") -> dict:
    """Вывод и подписи, посчитанные кодом фронта для результатов всех методов."""
    data = Path(tmp) / "biostat_results_127.json"
    data.write_text(json.dumps(_clean(_results()), ensure_ascii=False), encoding="utf-8")
    script = f"""
const fs = require("fs");
Promise.all([import({json.dumps((WEB / 'conclusion.ts').as_uri())}),
             import({json.dumps((WEB / 'extraLabels.ts').as_uri())})]).then(([c, l]) => {{
  const res = JSON.parse(fs.readFileSync({json.dumps(str(data))}, "utf8"));
  const out = {{ lines: {{}}, labels: {{}}, methods: c.CONCLUSION_METHODS }};
  for (const [m, r] of Object.entries(res)) {{
    out.lines[m] = c.conclusionLine(r, (k) => k);
    for (const k of Object.keys(r.statistics))
      for (const lang of ["ru", "en", "pt"]) out.labels[lang + "|" + k] = l.statLabel(k, lang);
  }}
  out.p = [0.0004, 0.0123, 0.05].map(c.pText);
  out.alpha = c.conclusionKey({{ method: "two_sample_t", p_value: 0.03, statistics: {{}},
    parameters: {{ alpha: 0.01 }} }});
  process.stdout.write(JSON.stringify(out));
}});"""
    return _node(script)


# --- (1) вывод словами -------------------------------------------------------------------

def test_every_method_with_p_gets_a_conclusion() -> None:
    front = _front()
    with_p = {m for m, r in _results().items() if r["p_value"] is not None}
    assert with_p == set(front["methods"]) and len(with_p) == 36
    for m in with_p:
        key = front["lines"][m].split(":")[0]
        assert re.fullmatch(r"conclusion\.[a-z_]+\.(yes|no)", key), m
        assert I18N.count(f'"{key}": ') == 3, key
    assert all(front["lines"][m] is None for m in _results() if m not in with_p)


def test_conclusion_carries_the_statistic_and_p() -> None:
    lines = _front()["lines"]
    assert lines["one_way_anova"] == "conclusion.means.yes: F(3, 116) = 54.7, P < 0.001"
    assert lines["two_sample_t"].startswith("conclusion.means.yes: t(")
    assert lines["rcbd_anova"].startswith("conclusion.means.yes: F(3, 87) = ")
    assert lines["polynomial_regression"].startswith("conclusion.model.yes: F(2, 117) = ")
    assert re.fullmatch(r"conclusion\.props\.no: P = 1\.000", lines["mcnemar_test"])
    # P поправлено (Хюйн–Фельдт): df без поправки не пишем
    assert re.fullmatch(r"conclusion\.means\.no: F = [\d.]+, P = 0\.\d{3}", lines["rm_anova"])


def test_p_format_and_alpha_of_the_method() -> None:
    front = _front()
    assert front["p"] == ["< 0.001", "= 0.012", "= 0.050"]
    assert front["alpha"] == "conclusion.means.no"  # P = 0.03 при α = 0.01


def test_wording_never_claims_no_difference() -> None:
    texts = re.findall(r'"conclusion\.[a-z_]+\.no": "([^"]+)"', I18N)
    assert len(texts) == 36
    for text in texts:
        assert re.search(r"обнаружен|detected|detetad", text), text
    assert not re.search(r"различий нет|нет различий|не различаются", I18N)


def test_result_panel_shows_conclusion_under_p() -> None:
    assert "<strong>P {pText(result.p_value)}</strong>" in COMPONENTS
    assert "{conclusion}" in COMPONENTS and "p = {formatNumber(result.p_value)}" not in COMPONENTS


# --- (2) подписи статистик ---------------------------------------------------------------

def test_no_core_key_leaks_into_plaques() -> None:
    labels = _front()["labels"]
    assert len(labels) > 900
    leaked = sorted(k for k, v in labels.items() if "_" in v)
    assert leaked == []


@pytest.mark.parametrize(("key", "lang", "label"), [
    ("df_between", "ru", "df (между группами)"), ("levene_p", "en", "Levene test, P"),
    ("shapiro_p_residuals", "pt", "Shapiro–Wilk, P (resíduos)"),
    ("hl_ci_lower", "ru", "Сдвиг Ходжеса–Лемана: нижняя граница ДИ"),
    ("F_A.B", "en", "F (A.B)"), ("ms_within", "pt", "QM (dentro dos grupos)"),
])
def test_labels_examples(key: str, lang: str, label: str) -> None:
    assert _front()["labels"][f"{lang}|{key}"] == label


# --- (3) таблица ANOVA -------------------------------------------------------------------

@pytest.mark.parametrize("method", ["one_way_anova", "transformed_means", "rcbd_anova",
                                    "rcbd_subsampling_anova"])
def test_anova_table_adds_up(method: str) -> None:
    r = _results()[method]
    rows = r["parameters"]["anova_table"]
    total = rows[-1]
    assert total["term"] == "Total"
    assert sum(x["ss"] for x in rows[:-1]) == pytest.approx(total["ss"], rel=1e-9)
    assert sum(x["df"] for x in rows[:-1]) == total["df"] == r["n"] - 1
    tested = next(x for x in rows if x.get("p") == r["p_value"])
    assert tested["F"] == pytest.approx(r["statistics"]["F"])


def test_subsampling_table_tests_plots_against_samples() -> None:
    rows = _results()["rcbd_subsampling_anova"]["parameters"]["anova_table"]
    strata = [x.get("stratum") for x in rows]
    assert strata == ["block", "unit", "unit", "sample", None]
    assert "anova.stratum.sample" in I18N and I18N.count('"anova.stratum.sample": ') == 3
    assert "r.stratum && r.stratum !== prevStratum" in COMPONENTS  # у «Total» нет страты


# --- (4) заголовок и повторы -------------------------------------------------------------

def test_title_uses_method_name_and_effect_sizes_are_not_repeated() -> None:
    assert "{methodLabel ?? result.method}" in COMPONENTS
    assert "!effectKeys.has(k)" in COMPONENTS
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    assert ".find((sg) => sg.method === result.method)?.label" in app
