"""Аудит 03-03: рантайм Pyodide со своего домена.

Пять файлов рантайма (pyodide.mjs, pyodide.asm.js/.wasm, python_stdlib.zip,
pyodide-lock.json) копирует web/build-pyodide.mjs из npm-пакета при сборке; пакеты
остаются на jsDelivr, но их адреса и sha256 берутся из своего lock-файла. Проверено
в Chrome: рантайм приходит со своего адреса, подменённый на CDN numpy отклонён
(«Не удалось загрузить пакеты Python: numpy»), подмена рантайма на CDN ни на что
не влияет — сайт его больше не запрашивает.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
WORKER = (WEB / "src" / "pyodide" / "worker.ts").read_text(encoding="utf-8")
BUILD = (WEB / "build-pyodide.mjs").read_text(encoding="utf-8")


def _version() -> str:
    return re.search(r'PYODIDE_VERSION = "([\d.]+)"', WORKER).group(1)


def test_worker_loads_the_runtime_from_the_site() -> None:
    assert "cdn.jsdelivr" not in WORKER
    assert "const PYODIDE_BASE = `/pyodide/v${PYODIDE_VERSION}/`;" in WORKER
    assert "loadPyodide({ indexURL: base })" in WORKER


def test_versions_agree_everywhere() -> None:
    version = _version()
    assert f'export const PYODIDE_VERSION = "{version}";' in BUILD
    lock = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    assert lock["node_modules/pyodide"]["version"] == version
    assert "if (installed !== PYODIDE_VERSION)" in BUILD  # иная версия в node_modules — сборка падает


def test_runtime_is_copied_on_every_build_and_not_committed() -> None:
    scripts = json.loads((WEB / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert "npm run build:pyodide" in scripts["prebuild"] and "npm run build:pyodide" in scripts["dev"]
    assert scripts["build:pyodide"] == "node build-pyodide.mjs"
    assert "web/public/pyodide/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_own_lock_points_packages_to_the_cdn_with_the_same_hashes() -> None:
    exe = shutil.which("node")
    source = WEB / "node_modules" / "pyodide" / "pyodide-lock.json"
    if not exe or not source.exists():
        pytest.skip("нужны node и web/node_modules (npm install)")
    script = (f"import({json.dumps((WEB / 'build-pyodide.mjs').as_uri())}).then((m) => {{"
              f"const lock = JSON.parse(require('fs').readFileSync({json.dumps(str(source))}, 'utf8'));"
              f"process.stdout.write(JSON.stringify(m.rewriteLock(lock)));}})")
    out = subprocess.run([exe, "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    mine, original = json.loads(out.stdout), json.loads(source.read_text(encoding="utf-8"))
    cdn = f"https://cdn.jsdelivr.net/pyodide/v{_version()}/full/"
    assert mine["packages"].keys() == original["packages"].keys()
    for name, pkg in mine["packages"].items():
        assert pkg["file_name"] == cdn + original["packages"][name]["file_name"], name
        assert pkg["sha256"] == original["packages"][name]["sha256"], name
    assert mine["info"] == original["info"]
