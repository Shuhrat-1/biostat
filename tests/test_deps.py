"""Зависимости сборки: vite — последний выпуск своей основной ветки 5.x (аудит 3).

Новую мажорную версию не берём (решение по аудиту 3): уязвимость esbuild
GHSA-67mh-4wv8-2f99 касается только dev-сервера и закрыта лишь в vite 6.4.4+.
"""

from __future__ import annotations

import json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.lstrip("^~").split("."))


def test_vite_range_starts_at_latest_5x() -> None:
    declared = json.loads((WEB / "package.json").read_text(encoding="utf-8"))["devDependencies"]["vite"]
    assert declared.startswith("^") and _version(declared) >= (5, 4, 21)


def test_lock_resolves_vite_5_and_matches_the_range() -> None:
    lock = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    resolved = _version(lock["node_modules/vite"]["version"])
    assert resolved[0] == 5 and resolved >= (5, 4, 21)
    assert lock[""]["devDependencies"]["vite"] == "^5.4.21"
