"""Патч 129: мелкие пункты аудитов 4 и 5.

«Сравнение с эталоном» начинается с t-теста; 04-09 пример в полях планирования,
«да / нет», поля не налезают; 04-10 строка «чего не хватает», колонка одного
списка не предлагается в другом; 04-12 пример данных на трёх языках; 05-06 язык
запоминается; 05-08 «Рассчитать» → «Остановить» через 1 с; 05-10 колонка блоков
подставляется и при выборе карточкой, если кандидат один.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.validation.catalog import CATALOG, GROUPS

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
APP = (WEB / "App.tsx").read_text(encoding="utf-8")
COMPONENTS = (WEB / "components.tsx").read_text(encoding="utf-8")
I18N = (WEB / "i18n.tsx").read_text(encoding="utf-8")


def _first_by_group() -> dict[str, str]:
    first: dict[str, str] = {}
    for spec in CATALOG:
        first.setdefault(spec.group, spec.method)
    return first


def test_most_general_method_comes_first_in_every_tab() -> None:
    assert _first_by_group() == {
        "Описательная статистика": "describe_all", "Проверка допущений": "normality_test",
        "Сравнение с эталоном": "one_sample_t", "Сравнение двух групп": "two_sample_t",
        "Сравнение нескольких групп": "one_way_anova", "Планирование опыта": "replication_power",
        "Парные измерения": "paired_t", "Связь переменных": "simple_linear_regression",
    }
    assert set(_first_by_group()) == set(GROUPS)


# --- 04-09 ---------------------------------------------------------------------------------

def test_04_09_planning_text_fields_have_a_book_example() -> None:
    from core.stats.sampling import sampling_scheme_power

    spec = next(s for s in CATALOG if s.method == "sampling_scheme_power")
    examples = {p.name: p.to_dict()["example"] for p in spec.params}
    assert examples["components"] and examples["schemes"]
    res = sampling_scheme_power(examples["components"], examples["schemes"], delta=0.16)
    assert len(res.parameters["schemes"]) == 2  # пример проходит проверку формата ядра


def test_04_09_example_button_yes_no_and_spacing() -> None:
    field = COMPONENTS[COMPONENTS.index("function ParamField("):]
    assert 'boxSizing: "border-box"' in field and "onClick={() => onChange(param.example)}" in field
    assert "placeholder={param.example ?" in field
    for key in ("param.example", "param.useExample", "opt.position_terms.yes", "opt.position_terms.no"):
        assert I18N.count(f'"{key}": ') == 3, key
    assert '"opt.position_terms.yes": "да"' in I18N and '"opt.position_terms.no": "não"' in I18N


# --- 04-10 ---------------------------------------------------------------------------------

def test_04_10_disabled_button_says_what_is_missing() -> None:
    assert ".filter((slot) => !optional.has(slot) && !columns[slot as keyof ColumnSelection])" in COMPONENTS
    assert 't("method.missing").replace("{list}", missing.join(", "))' in COMPONENTS
    assert "aria-describedby={missing.length > 0 ? `${uid}-missing` : undefined}" in COMPONENTS
    assert I18N.count('"method.missing": ') == 3 and I18N.count("{list}") == 3


def test_04_10_column_from_one_list_is_not_offered_in_another() -> None:
    assert "!Object.entries(columns).some(([other, v]) => other !== slot && v === name)" in COMPONENTS


# --- 04-12: пример данных ------------------------------------------------------------------

def _example(lang: str) -> bytes:
    exe = shutil.which("node")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             encoding="utf-8").stdout if exe else ""
    if not exe or tuple(int(x) for x in version.strip().lstrip("v").split(".")[:2]) < (22, 6):
        pytest.skip("нужен Node 22.6+ для запуска web/src/exampleData.ts")
    script = (f"import({json.dumps((WEB / 'exampleData.ts').as_uri())}).then((m) => "
              f"process.stdout.write(JSON.stringify([...m.exampleFileBytes({json.dumps(lang)})])))")
    out = subprocess.run([exe, "--experimental-strip-types", "--no-warnings", "-e", script],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return bytes(json.loads(out.stdout))


@pytest.mark.parametrize(("lang", "header", "crop"), [
    ("ru", ["сорт", "урожай", "высота", "площадь"], "пшеница"),
    ("en", ["variety", "yield", "height", "area"], "wheat"),
    ("pt", ["variedade", "produção", "altura", "área"], "trigo"),
])
def test_04_12_example_in_the_interface_language(lang: str, header: list[str], crop: str) -> None:
    from core.validation.parser import parse_file

    table = parse_file(_example(lang))
    assert table.header == header and table.n_rows == 18
    assert crop in table.column_values(header[0])
    types = {p.name: p.detected_type.value for p in table.profiles}
    assert types[header[1]] == "numeric"
    assert "exampleFileBytes(lang)" in APP


# --- 05-06, 05-08, 05-10 -------------------------------------------------------------------

def test_05_06_language_is_remembered_safely() -> None:
    stored = APP[APP.index("function storedLang()"):APP.index("function AppInner(")]
    assert stored.count("try {") == 2 and stored.count("} catch {") == 2
    assert "LANGUAGES.some((l) => l.code === v)" in stored
    assert 'useState<Lang>(() => storedLang() ?? "ru")' in APP and "saveLang(lang);" in APP


def test_05_08_run_turns_into_stop_after_one_second() -> None:
    assert "window.setTimeout(() => setCanStop(true), 1000)" in COMPONENTS
    assert "{canStop && onCancel ? (" in COMPONENTS and 't("run.cancel")' in COMPONENTS
    assert APP.count("onCancel={handleCancel}") == 3  # плашка остаётся + две формы


def test_05_10_block_hint_counts_candidates() -> None:
    from core.validation.parser import parse_file
    from core.validation.recognize import recognize

    # по две строки в ячейке «сорт × блок»: раскладка блоков не видна, остаётся имя
    rows = [("A", "b1", "r1"), ("A", "b1", "r2"), ("A", "b2", "r1"), ("A", "b2", "r2"),
            ("B", "b1", "r1"), ("B", "b1", "r2"), ("B", "b2", "r1"), ("B", "b2", "r2")]
    body = "".join(f"{v};{b};{r};{i}\n" for i, (v, b, r) in enumerate(rows))
    one = parse_file(("variety;block;plot;yield\n" + body).encode())
    two = parse_file(("variety;block;rep;yield\n" + body).encode())
    fired = {f["id"]: f for f in recognize(one, {"value": "yield", "group": "variety"})["fired"]}
    assert fired["B1.1"]["message"]["params"] == {"c": "block", "n": 1}
    fired = {f["id"]: f for f in recognize(two, {"value": "yield", "group": "variety"})["fired"]}
    assert fired["B1.1"]["message"]["params"]["n"] == 2


def test_05_10_card_choice_fills_the_single_block_column() -> None:
    fill = (WEB / "columnFill.ts").read_text(encoding="utf-8")
    assert "(n === undefined || n === 1)" in fill and "export function fillBlock(" in fill
    assert "const block = fillBlock(carryColumns(columns, spec), spec, blockHint);" in COMPONENTS
    assert "blockHint={map.blockHint}" in APP
    assert "blockHint: blockCandidate(recognition?.fired ?? [])," in (
        WEB / "useMapSync.ts").read_text(encoding="utf-8")
