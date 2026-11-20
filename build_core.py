"""Упаковка расчётного ядра в zip для загрузки в Pyodide.

Запускать после любого изменения в ``core/``:
    python build_core.py

Результат кладётся в ``web/public/core/core.zip`` — оттуда его забирает
браузер и распаковывает в виртуальную ФС Pyodide.

Архив воспроизводим: файлы по алфавиту, одно и то же время у всех записей,
концы строк LF (на Windows git может выдать CRLF). Если содержимое не
изменилось, файл не перезаписывается — сборка не оставляет ``core.zip``
изменённым в git. Те же правила у ``web/build-core.mjs``.
"""

from __future__ import annotations

import pathlib
import zipfile

ROOT = pathlib.Path(__file__).parent
CORE_DIR = ROOT / "core"
OUTPUT = ROOT / "web" / "public" / "core" / "core.zip"
# Минимальная дата формата zip: время файлов не должно попадать в архив.
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def core_files(root: pathlib.Path = ROOT) -> dict[str, bytes]:
    """Путь в архиве → содержимое с LF, по алфавиту, без кеша и байт-кода."""
    files = sorted(
        f for f in (root / "core").rglob("*.py") if "__pycache__" not in f.parts
    )
    return {
        f.relative_to(root).as_posix(): f.read_bytes().replace(b"\r\n", b"\n")
        for f in files
    }


def archive_contents(path: pathlib.Path) -> dict[str, bytes] | None:
    """Содержимое существующего архива; None — архива нет или он битый."""
    try:
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile):
        return None


def write_archive(path: pathlib.Path, files: dict[str, bytes]) -> None:
    """Записать архив с фиксированным временем и правами у каждой записи."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compresslevel=9)


def build() -> bool:
    """Собрать core/ в zip; вернуть True, если архив перезаписан."""
    files = core_files()
    if archive_contents(OUTPUT) == files:
        print(f"core.zip не изменился ({len(files)} файлов) — не перезаписываю")
        return False
    write_archive(OUTPUT, files)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Упаковано {len(files)} файлов -> {OUTPUT} ({size_kb:.1f} КБ)")
    return True


if __name__ == "__main__":
    build()
