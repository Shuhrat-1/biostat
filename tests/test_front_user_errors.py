"""Аудит 03-06, 03-07, 03-08: понятные ошибки загрузки и гигиена репозитория."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"


def _error_key(message: str) -> str | None:
    exe = shutil.which("node")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, encoding="utf-8").stdout if exe else ""
    if not exe or tuple(int(x) for x in version.strip().lstrip("v").split(".")[:2]) < (22, 6):
        pytest.skip("нужен Node 22.6+ для запуска web/src/userErrors.ts")
    script = (f"import({json.dumps((WEB / 'userErrors.ts').as_uri())}).then((m) => "
              f"process.stdout.write(JSON.stringify(m.errorKey({json.dumps(message)}))))")
    out = subprocess.run([exe, "--experimental-strip-types", "--no-warnings", "-e", script],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize(("message", "key"), [
    ("packages_failed:six, scipy", "error.packages"),
    ("_csv.Error: field larger than field limit (131072)", "error.cellTooBig"),
    ("Traceback (most recent call last):\n  ...\nMemoryError", "error.fileTooBig"),
    ("Нет колонки: yield", None),
])
def test_03_06_raw_errors_map_to_translated_keys(message: str, key: str | None) -> None:
    assert _error_key(message) == key


def test_03_06_messages_exist_in_three_languages() -> None:
    i18n = (WEB / "i18n.tsx").read_text(encoding="utf-8")
    for key in ("error.packages", "error.cellTooBig", "error.fileTooBig"):
        assert i18n.count(f'"{key}": ') == 3, key
    assert i18n.count("{names}") == 3


def test_03_06_parser_errors_are_the_ones_the_front_recognises() -> None:
    """Одна ячейка длиннее предела модуля csv: ядро бросает ровно то, что узнаёт фронт."""
    from core.validation.parser import parse_file

    with pytest.raises(Exception, match="field larger than field limit"):
        parse_file(("a,b\n1," + "x" * 200_000 + "\n2,3\n").encode())


def test_03_06_worker_stops_when_a_package_does_not_import() -> None:
    worker = (WEB / "pyodide" / "worker.ts").read_text(encoding="utf-8")
    assert "await loadPackages(py);" in worker and "errorCallback" in worker
    assert "throw new Error(`${PACKAGES_FAILED}" in worker
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    assert app.count("setError(explainError(e, t))") == 2


def test_03_07_08_no_stray_files_in_the_repository() -> None:
    assert not (ROOT / "package-lock.json").exists()          # пустой lock-файл в корне
    assert not (ROOT / "web" / "web").exists()                # второй лишний lock-файл
    assert (ROOT / "web" / "package-lock.json").exists()      # рабочий остаётся
    assert not (ROOT / "biostat-changes.tar.gz").exists()
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "core.zip" not in ignored  # core.zip хранится в репозитории: патчи его несут
