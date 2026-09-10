"""Упаковка расчётного ядра в zip для загрузки в Pyodide.

Запускать после любого изменения в ``core/``:
    python build_core.py

Результат кладётся в ``web/public/core/core.zip`` — оттуда его забирает
браузер и распаковывает в виртуальную ФС Pyodide.
"""

from __future__ import annotations

import pathlib
import zipfile

ROOT = pathlib.Path(__file__).parent
CORE_DIR = ROOT / "core"
OUTPUT = ROOT / "web" / "public" / "core" / "core.zip"


def build() -> None:
    """Собрать core/ в zip, пропуская кеш и байт-код."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in CORE_DIR.rglob("*.py")
        if "__pycache__" not in f.parts
    ]
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(ROOT).as_posix())

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Упаковано {len(files)} файлов -> {OUTPUT} ({size_kb:.1f} КБ)")


if __name__ == "__main__":
    build()
