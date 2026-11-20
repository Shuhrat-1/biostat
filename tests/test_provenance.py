"""Задание P4: по выгрузке видно, чем, когда и на каком файле посчитан результат.

JSON — блок provenance, Excel — лист «Сведения», под результатом — строка
«BioStat <версия> · <дата расчёта> · файл SHA-256 <12 знаков>». CSV не меняется.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
APP = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "src" / "components.tsx").read_text(encoding="utf-8")
SAMPLE = "сорт;урожай\nпшеница;5,1\nячмень;4,7\n".encode("utf-8")


def _need_node() -> str:
    exe = shutil.which("node")
    if not exe or not (WEB / "node_modules" / "esbuild").exists():
        pytest.skip("нужны node и web/node_modules (npm install)")
    return exe


@lru_cache(maxsize=1)
def _front() -> dict:
    """Функции provenance.ts на настоящих байтах, собранные esbuild."""
    exe = _need_node()
    tmp = Path(tempfile.gettempdir())
    data, bundle = tmp / "biostat_p4_sample.csv", tmp / "biostat_p4_provenance.mjs"
    data.write_bytes(SAMPLE)
    script = f"""
require({json.dumps(str(WEB / 'node_modules' / 'esbuild'))}).buildSync({{
  entryPoints: [{json.dumps(str(WEB / 'src' / 'provenance.ts'))}], bundle: true, format: "esm",
  platform: "node", outfile: {json.dumps(str(bundle))}, logLevel: "error" }});
import({json.dumps(bundle.as_uri())}).then(async (m) => {{
  const bytes = new Uint8Array(require("fs").readFileSync({json.dumps(str(data))}));
  const sha = [await m.sha256Hex(bytes), await m.sha256Hex(bytes)];
  const p = m.buildProvenance({{
    computed_at: m.isoWithOffset(new Date(Date.UTC(2026, 10, 20, 21, 10, 5))),
    input_file: {{ name: "f.csv", size_bytes: bytes.length, sha256: sha[0], encoding: "utf-8",
                  delimiter: ";", decimal: ",", has_header: true }},
    runtime: {{ pyodide: "0.26.4", numpy: "1.26.4", scipy: "1.12.0", pandas: "2.2.0", statsmodels: "0.14.2" }},
    method: {{ id: "one_way_anova", name: "Однофакторный ANOVA" }},
    parameters: {{ alpha: 0.05 }}, columns: {{ value: "урожай", group: "сорт" }},
    language: "ru", study_context: "experiment" }});
  process.stdout.write(JSON.stringify({{ sha, p, rows: m.provenanceRows(p, (k) => k),
    line: m.provenanceLine(p, (k) => k) }}));
}});"""
    out = subprocess.run([exe, "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def test_file_hash_is_stable_and_matches_hashlib() -> None:
    sha = _front()["sha"]
    assert sha[0] == sha[1] == hashlib.sha256(SAMPLE).hexdigest()


def test_every_field_is_filled() -> None:
    front = _front()
    p = front["p"]
    for key in ("app_version", "computed_at", "input_file", "runtime", "method", "parameters",
                "columns", "language", "study_context"):
        assert p[key] not in ("", None, {}), key
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", p["computed_at"])
    assert set(p["runtime"]) == {"pyodide", "numpy", "scipy", "pandas", "statsmodels"}
    values = dict((r[0], r[1]) for r in front["rows"][1:])
    assert values["input_file.sha256"] == hashlib.sha256(SAMPLE).hexdigest()
    assert values["method.id"] == "one_way_anova" and values["columns.group"] == "сорт"
    assert all(v != "" for k, v in values.items() if k != "build_date"), values


def test_line_under_the_result() -> None:
    line = _front()["line"]
    sha12 = hashlib.sha256(SAMPLE).hexdigest()[:12]
    assert re.fullmatch(rf"BioStat \S+ · \d{{4}}-\d\d-\d\d \d\d:\d\d · provenance\.file SHA-256 {sha12}", line)


def test_wiring_json_excel_and_unchanged_csv() -> None:
    assert "...(provenance ? { provenance } : {})" in COMPONENTS
    assert 'all.push({ title: t("provenance.sheet"), rows: provenanceRows(provenance, t) })' in COMPONENTS
    assert "sectionsToCsv(sections())" in COMPONENTS  # CSV без служебных строк
    assert "sha256: await sha256Hex(bytes)" in APP and APP.count("await sha256Hex(bytes)") == 2
    assert "parameters: effectiveParams(method, kwargs)," in APP  # и значения по умолчанию
    assert "runtime: await runtimeVersions()" in APP and "{provenanceLine(provenance, t)}" in COMPONENTS
    bridge = (WEB / "src" / "pyodide" / "bridge.ts").read_text(encoding="utf-8")
    assert "import json, pyodide, numpy, scipy, pandas, statsmodels" in bridge


def test_built_site_carries_the_commit_hash() -> None:
    _need_node()
    head = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8")
    if head.returncode != 0:
        pytest.skip("нет git")
    out = Path(tempfile.mkdtemp(prefix="biostat_p4_dist_"))
    npx = shutil.which("npx") or "npx"
    subprocess.run([npx, "vite", "build", "--outDir", str(out), "--emptyOutDir", "--logLevel", "error"],
                   cwd=WEB, check=True, capture_output=True, shell=False)
    bundle = "".join(p.read_text(encoding="utf-8") for p in (out / "assets").glob("*.js"))
    assert f'"{head.stdout.strip()}"' in bundle
    shutil.rmtree(out, ignore_errors=True)
