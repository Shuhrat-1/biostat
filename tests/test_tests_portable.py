"""Тесты, которые запускают node, должны проходить и на Windows.

На Windows `subprocess.run(..., text=True)` читает вывод в кодировке системы
(cp1252), а node пишет UTF-8: «урожай» превращалось в «ÑƒÑ€Ð¾Ð¶Ð°Ð¹», а байт 0x81
обрывал чтение. Путь «/tmp» на Windows не существует.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in TESTS.glob("test_*.py")
            if p.name != Path(__file__).name}


def test_text_subprocess_output_is_read_as_utf8() -> None:
    for name, src in _sources().items():
        for call in re.findall(r"subprocess\.run\((?:[^()]|\([^()]*\))*\)", src):
            if "text=True" in call:
                assert 'encoding="utf-8"' in call, (name, call[:80])


def test_no_hard_coded_unix_temp_dir() -> None:
    for name, src in _sources().items():
        assert '"/tmp' not in src, name
