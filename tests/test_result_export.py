"""Аудит 05-07 и 04-12: выгрузка результата и графиков.

У результата — CSV и Excel одним набором: «Итог» и по разделу на таблицу (Excel —
лист на раздел), с защитой от формул из 03-04. У каждого графика — PNG (×2, белый
фон) и SVG.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
CHARTS = (WEB / "charts.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")
I18N = (WEB / "i18n.tsx").read_text(encoding="utf-8")


def _clean(v: object) -> object:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


@lru_cache(maxsize=1)
def _exports() -> dict:
    """Разделы, CSV и имена листов, посчитанные кодом фронта для всех методов."""
    import warnings

    from tests.bridge_runner import CASES, run, table

    warnings.filterwarnings("ignore")
    results = {m: run(m, table(m)) for m in CASES}
    hostile = dict(results["one_way_anova"])
    hostile["estimates"] = [{"name": "=HYPERLINK(\"http://evil.example\")", "value": 1.0,
                             "ci_lower": None, "ci_upper": None, "ci_level": None, "std_error": None}]
    results["hostile"] = hostile
    data = Path(tempfile.gettempdir()) / "biostat_results_130.json"
    data.write_text(json.dumps(_clean(results), ensure_ascii=False), encoding="utf-8")
    exe = shutil.which("node")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             encoding="utf-8").stdout if exe else ""
    if not exe or tuple(int(x) for x in version.strip().lstrip("v").split(".")[:2]) < (22, 6):
        pytest.skip("нужен Node 22.6+ для запуска web/src/resultExport.ts")
    # resultExport.ts импортирует соседние модули без расширения (так требует сборщик):
    # node их сам не найдёт, поэтому сначала собираем модуль esbuild'ом из web/node_modules.
    esbuild = ROOT / "web" / "node_modules" / "esbuild"
    if not esbuild.exists():
        pytest.skip("нет web/node_modules (npm install)")
    bundle = Path(tempfile.gettempdir()) / "biostat_result_export_130.mjs"
    script = f"""
const fs = require("fs");
require({json.dumps(str(esbuild))}).buildSync({{ entryPoints: [{json.dumps(str(WEB / 'resultExport.ts'))}],
  bundle: true, format: "esm", platform: "node", outfile: {json.dumps(str(bundle))}, logLevel: "error" }});
import({json.dumps(bundle.as_uri())}).then((m) => {{
  const res = JSON.parse(fs.readFileSync({json.dumps(str(data))}, "utf8"));
  const out = {{}};
  for (const [k, r] of Object.entries(res)) {{
    const secs = m.resultSections(r, (key) => key, "ru", "Метод " + k, null);
    out[k] = {{ sections: secs, csv: m.sectionsToCsv(secs), sheets: m.sheetNames(secs) }};
  }}
  process.stdout.write(JSON.stringify(out));
}});"""
    out = subprocess.run([exe, "-e", script],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def test_every_result_exports_a_summary_and_its_tables() -> None:
    ex = _exports()
    assert len(ex) == 72
    for method, e in ex.items():
        titles = [s["title"] for s in e["sections"]]
        assert titles[0] == "export.summary", method
        summary = dict((r[0], r[1]) for r in e["sections"][0]["rows"][1:] if len(r) == 2)
        assert summary["export.method"] == f"Метод {method}" and "n" in summary


def test_tables_follow_the_screen() -> None:
    ex = _exports()
    anova = [s["title"] for s in ex["one_way_anova"]["sections"]]
    assert anova[:3] == ["export.summary", "export.estimates", "export.anova"]
    rows = next(s["rows"] for s in ex["one_way_anova"]["sections"] if s["title"] == "export.anova")
    assert rows[0][:2] == ["Член", "df"] and [r[0] for r in rows[1:]] == ["Between", "Within", "Total"]
    assert len(ex["tukey_hsd"]["sections"]) >= 2  # парные сравнения — отдельный раздел
    summary = ex["one_way_anova"]["sections"][0]["rows"]
    assert ["export.conclusion", "conclusion.means.yes: F(3, 116) = 54.7, P < 0.001"] in summary
    assert ["df (между группами)", 3] in summary  # подписи статистик, числа как числа


def test_no_object_text_and_valid_sheet_names() -> None:
    for method, e in _exports().items():
        cells = [str(c) for s in e["sections"] for r in s["rows"] for c in r]
        assert not any("[object Object]" in c or c == "undefined" for c in cells), method
        names = e["sheets"]
        assert len(names) == len(e["sections"]) and len({n.lower() for n in names}) == len(names)
        assert all(0 < len(n) <= 31 and not set(n) & set("[]:*?/\\") for n in names), method


def test_csv_sections_are_separated_and_formulas_are_neutralised() -> None:
    e = _exports()["hostile"]
    csv = e["csv"]
    assert csv.startswith("﻿export.summary\n")
    assert "\n\nexport.estimates\n" in csv
    assert "'=HYPERLINK" in csv and "\n=HYPERLINK" not in csv and ";=HYPERLINK" not in csv


def test_result_panel_has_csv_and_excel_next_to_json() -> None:
    buttons = COMPONENTS[COMPONENTS.index("function ExportButtons("):COMPONENTS.index("/** Формат числа")]
    for needle in ("sectionsToCsv(sections())", "sheetNames(all).forEach", "XLSX.utils.aoa_to_sheet(all[i].rows)",
                   ">CSV</button>", ">Excel</button>", 't("result.download")'):
        assert needle in buttons, needle


def test_every_chart_saves_png_and_svg() -> None:
    assert "const PNG_SCALE = 2;" in CHARTS and "chart.renderToSVGString()" in CHARTS
    assert "ctx.fillStyle = c.panel;" in CHARTS  # белый фон PNG
    assert 'fileName={`biostat_${key}`}' in CHARTS and '(["png", "svg"] as const)' in CHARTS


def test_new_texts_in_three_languages() -> None:
    keys = ["plots.save", "export.group", "export.summary", "export.item", "export.value",
            "export.method", "export.conclusion", "export.caveat", "export.estimates",
            "export.estimate", "export.ciLower", "export.ciUpper", "export.anova"]
    for key in keys:
        assert I18N.count(f'"{key}": ') == 3, key
