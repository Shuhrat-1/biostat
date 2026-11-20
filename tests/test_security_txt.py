"""Задание P6: web/public/.well-known/security.txt по RFC 9116.

Тест падает, когда срок Expires истекает или уходит дальше 12 месяцев от даты
последнего коммита файла, — это напоминание продлить срок.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "web" / "public" / ".well-known" / "security.txt"


def _fields() -> dict[str, str]:
    lines = [line for line in FILE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return dict(line.split(": ", 1) for line in lines)


def _committed_at() -> datetime:
    """Дата последнего коммита файла; файл ещё не в git — сейчас."""
    out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", str(FILE)], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8")
    stamp = out.stdout.strip() if out.returncode == 0 else ""
    return datetime.fromisoformat(stamp) if stamp else datetime.now(timezone.utc)


def test_required_fields_rfc_9116() -> None:
    fields = _fields()
    assert fields["Contact"] == "mailto:security@biostat.app"
    assert fields["Preferred-Languages"] == "ru, en, pt"
    assert fields["Canonical"] == "https://biostat.app/.well-known/security.txt"
    assert set(fields) == {"Contact", "Expires", "Preferred-Languages", "Canonical"}


def test_expires_is_in_the_future_and_within_a_year_of_the_commit() -> None:
    expires = datetime.fromisoformat(_fields()["Expires"].replace("Z", "+00:00"))
    assert expires > datetime.now(timezone.utc), "security.txt просрочен — продлите Expires"
    assert expires <= _committed_at() + timedelta(days=366), "Expires дальше 12 месяцев от коммита"


def test_served_as_a_real_file_not_the_spa() -> None:
    redirects = (ROOT / "web" / "public" / "_redirects").read_text(encoding="utf-8")
    assert "/.well-known/security.txt" in redirects  # отмечено среди настоящих файлов
    assert FILE.read_bytes().endswith(b"\n") and b"\r" not in FILE.read_bytes()
