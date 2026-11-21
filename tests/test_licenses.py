"""Шаг 4 (P2): лицензия кода и лицензии сторонних компонентов, которые раздаёт сайт.

LICENSE — дословный Apache License 2.0 (sha256 официального LICENSE-2.0.txt; тот же
текст — первые 202 строки echarts/LICENSE). Сайт раздаёт /LICENSE.txt и
/third-party-licenses.txt, которые собирает web/build-licenses.mjs; рядом с рантаймом
Pyodide — его LICENSE.txt (MPL-2.0) и SOURCE.txt.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MODULES = WEB / "node_modules"
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
PYODIDE_LICENSE_SHA256 = "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5"  # тег 0.26.4
CPYTHON_LICENSE_SHA256 = "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf"  # тег v3.12.1
COPYRIGHT = "Copyright 2026 Shuhrat Maksumov"

needs_modules = pytest.mark.skipif(not MODULES.exists(), reason="нет web/node_modules (npm install)")


def _lf(path: Path) -> bytes:
    """Байты файла с LF: на Windows git (core.autocrlf) может извлечь текст с CRLF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(_lf(path)).hexdigest()


def _node(script: str) -> str:
    """Выполнить ES-модуль в каталоге web/ и вернуть stdout."""
    done = subprocess.run(["node", "--input-type=module", "-e", script], cwd=WEB, capture_output=True,
                          encoding="utf-8", check=True)
    return done.stdout


def _licenses_module() -> dict:
    url = (WEB / "build-licenses.mjs").as_uri()
    return json.loads(_node(f'import * as m from "{url}"; console.log(JSON.stringify('
                            "{components: m.components(), text: m.thirdPartyText(), bundled: m.BUNDLED}));"))


def test_license_is_the_verbatim_apache_2_0_text() -> None:
    assert _sha(ROOT / "LICENSE") == APACHE_SHA256
    if MODULES.exists():
        echarts = _lf(MODULES / "echarts" / "LICENSE").splitlines(keepends=True)
        assert b"".join(echarts[:202]) == _lf(ROOT / "LICENSE")


def test_notice_readme_and_package_name_the_owner_and_license() -> None:
    assert (ROOT / "NOTICE").read_text(encoding="utf-8") == f"BioStat\n{COPYRIGHT}\n"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Лицензия" in readme and COPYRIGHT in readme and "Apache License 2.0" in readme
    assert json.loads((WEB / "package.json").read_text(encoding="utf-8"))["license"] == "Apache-2.0"
    lock = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"][""]["license"] == "Apache-2.0"


def test_stored_runtime_license_texts_are_the_tagged_originals() -> None:
    pyodide = WEB / "licenses" / "pyodide-0.26.4.LICENSE"
    cpython = WEB / "licenses" / "cpython-3.12.1.LICENSE"
    assert _sha(pyodide) == PYODIDE_LICENSE_SHA256
    assert pyodide.read_text(encoding="utf-8").startswith("Mozilla Public License Version 2.0\n")
    assert _sha(cpython) == CPYTHON_LICENSE_SHA256
    assert "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" in cpython.read_text(encoding="utf-8")


def test_generated_files_are_built_not_committed() -> None:
    scripts = json.loads((WEB / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert "npm run build:licenses" in scripts["prebuild"] and "npm run build:licenses" in scripts["dev"]
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "web/public/LICENSE.txt" in ignore and "web/public/third-party-licenses.txt" in ignore


@needs_modules
def test_build_writes_license_txt_and_full_third_party_texts() -> None:
    subprocess.run(["node", "build-licenses.mjs"], cwd=WEB, check=True, capture_output=True, encoding="utf-8")
    assert (WEB / "public" / "LICENSE.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()
    raw = (WEB / "public" / "third-party-licenses.txt").read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    mod = _licenses_module()
    assert text == mod["text"]
    for name, entry in mod["bundled"].items():
        for file in entry["files"]:
            assert (MODULES / name / file).read_text(encoding="utf-8").strip() in text, f"{name}/{file}"
    for stored in (WEB / "licenses").iterdir():
        assert stored.read_text(encoding="utf-8").strip() in text, stored.name
    assert "https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.26/LICENSE" in text  # в архиве лицензии нет


def _notice_rows() -> set[tuple[str, str, str]]:
    md = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([\w.-]+) \| ([\w.]+) \| ([\w. -]+?) \|$", md, re.M)
    return {row for row in rows if row[0] != "Компонент"}


@needs_modules
def test_notices_file_lists_exactly_what_the_site_loads() -> None:
    components = _licenses_module()["components"]
    assert {(c["name"], c["version"], c["license"]) for c in components} == _notice_rows()
    assert sum(c["served"] == "jsdelivr" for c in components) == 11  # как на странице «Конфиденциальность»


@needs_modules
def test_bundled_table_matches_the_real_bundle() -> None:
    """Пакеты node_modules в sourcemap сборки = ключи BUNDLED (лишних и пропущенных нет)."""
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as out:
        subprocess.run(["node", str(MODULES / "vite" / "bin" / "vite.js"), "build", "--sourcemap",
                        "--outDir", out, "--emptyOutDir", "--logLevel", "error"],
                       cwd=WEB, check=True, capture_output=True, encoding="utf-8")
        sources = [s for m in Path(out, "assets").glob("*.map")
                   for s in json.loads(m.read_text(encoding="utf-8"))["sources"]]
    found = {re.search(r"node_modules/((?:@[^/]+/)?[^/]+)/", s).group(1) for s in sources if "node_modules/" in s}
    assert found == set(_licenses_module()["bundled"])


@needs_modules
def test_pyodide_runtime_folder_gets_license_and_source_notice() -> None:
    subprocess.run(["node", "build-pyodide.mjs"], cwd=WEB, check=True, capture_output=True, encoding="utf-8")
    target = WEB / "public" / "pyodide" / "v0.26.4"
    assert _sha(target / "LICENSE.txt") == PYODIDE_LICENSE_SHA256
    source = (target / "SOURCE.txt").read_text(encoding="utf-8")
    assert "Mozilla Public License 2.0" in source
    assert "https://github.com/pyodide/pyodide/tree/0.26.4" in source
