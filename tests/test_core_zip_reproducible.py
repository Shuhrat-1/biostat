"""core.zip воспроизводим и не «пачкает» git после сборки.

Было: build-core.mjs (npm run build / dev) и build_core.py записывали в архив
время файлов, поэтому core.zip после каждой сборки выглядел изменённым, и
`git am` патча с новым core.zip отказывал: «core.zip: does not match index».
Стало: файлы по алфавиту, время 1980-01-01, концы строк LF, и архив не
перезаписывается, если содержимое то же.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import build_core

ROOT = Path(__file__).resolve().parents[1]


def _tree(tmp: Path, text: bytes) -> Path:
    (tmp / "core" / "stats").mkdir(parents=True)
    (tmp / "core" / "__init__.py").write_bytes(b"")
    (tmp / "core" / "stats" / "b.py").write_bytes(text)
    (tmp / "core" / "stats" / "__pycache__").mkdir()
    (tmp / "core" / "stats" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"x")
    return tmp


def test_same_files_give_byte_identical_archives(tmp_path: Path) -> None:
    files = build_core.core_files(_tree(tmp_path, b"x = 1\n"))
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    build_core.write_archive(a, files)
    build_core.write_archive(b, files)
    assert a.read_bytes() == b.read_bytes()
    infos = zipfile.ZipFile(a).infolist()
    assert [i.filename for i in infos] == ["core/__init__.py", "core/stats/b.py"]  # без __pycache__
    assert {i.date_time for i in infos} == {build_core.FIXED_TIME}


def test_windows_line_endings_do_not_change_the_archive(tmp_path: Path) -> None:
    crlf = build_core.core_files(_tree(tmp_path / "w", b"x = 1\r\ny = 2\r\n"))
    lf = build_core.core_files(_tree(tmp_path / "l", b"x = 1\ny = 2\n"))
    assert crlf == lf


def test_unchanged_core_is_not_rewritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "core.zip"
    monkeypatch.setattr(build_core, "OUTPUT", out)
    monkeypatch.setattr(build_core, "core_files", lambda: {"core/a.py": b"x = 1\n"})
    assert build_core.build() is True
    before = out.stat().st_mtime_ns, out.read_bytes()
    assert build_core.build() is False
    assert (out.stat().st_mtime_ns, out.read_bytes()) == before


def test_committed_archive_matches_the_core() -> None:
    assert build_core.archive_contents(build_core.OUTPUT) == build_core.core_files()


def test_node_builder_packs_the_same_files() -> None:
    exe, admzip = shutil.which("node"), ROOT / "web" / "node_modules" / "adm-zip"
    if not exe or not admzip.exists():
        pytest.skip("нужны node и web/node_modules (npm install)")
    script = (f"import({json.dumps((ROOT / 'web' / 'build-core.mjs').as_uri())}).then((m) => {{"
              "const crypto = require('crypto'); const out = {};"
              "for (const [k, v] of m.coreFiles()) out[k] = crypto.createHash('sha256').update(v).digest('hex');"
              "process.stdout.write(JSON.stringify(Object.keys(out).map((k) => [k, out[k]])));})")
    run = subprocess.run([exe, "-e", script], capture_output=True, text=True, encoding="utf-8",
                         check=True, cwd=ROOT / "web")
    node = json.loads(run.stdout)
    python = [[k, hashlib.sha256(v).hexdigest()] for k, v in build_core.core_files().items()]
    assert node == python  # тот же порядок, те же байты с LF
    builder = (ROOT / "web" / "build-core.mjs").read_text(encoding="utf-8")
    assert "if (sameAsExisting(files))" in builder and "header.time = FIXED_TIME" in builder
