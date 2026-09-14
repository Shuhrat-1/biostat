"""Тесты сквозного разбора файла: детекция + парсинг + профили."""

from __future__ import annotations

import pytest

from core.validation.parser import parse_file, suggest_methods

EU_FILE = (
    "сорт;урожай;площадь\n"
    "пшеница;5,2;120\n"
    "пшеница;5,5;118\n"
    "ячмень;4,8;95\n"
    "ячмень;4,6;99\n"
).encode("utf-8")

DIRTY_FILE = (
    "дата;сорт;урожай\n"
    "2020-05-01;пшеница;5,2\n"
    "02.05.2020;ячмень;н/д\n"
    "2020-05-03;овёс;4,1\n"
).encode("cp1251")


def test_parse_european_file():
    t = parse_file(EU_FILE)
    assert t.dialect.delimiter == ";"
    assert t.dialect.decimal == ","
    assert t.header == ["сорт", "урожай", "площадь"]
    assert t.n_rows == 4


def test_parse_cp1251_preserves_cyrillic():
    t = parse_file(DIRTY_FILE)
    assert t.dialect.encoding == "cp1251"
    assert "сорт" in t.header


def test_numeric_column_extraction():
    t = parse_file(EU_FILE)
    values = t.numeric_column("урожай")
    assert values == pytest.approx([5.2, 5.5, 4.8, 4.6])


def test_numeric_column_skips_missing():
    t = parse_file(DIRTY_FILE)
    values = t.numeric_column("урожай")
    assert len(values) == 2  # 'н/д' пропущено
    assert values == pytest.approx([5.2, 4.1])


def test_groups_by():
    """Длинный формат: значения разбиваются по метке группы."""
    t = parse_file(EU_FILE)
    groups = t.groups_by("урожай", "сорт")
    assert set(groups) == {"пшеница", "ячмень"}
    assert groups["пшеница"] == pytest.approx([5.2, 5.5])
    assert groups["ячмень"] == pytest.approx([4.8, 4.6])


def test_unknown_column_raises():
    t = parse_file(EU_FILE)
    with pytest.raises(ValueError):
        t.numeric_column("нет такой")


def test_manual_override():
    """Ручное переопределение параметров имеет приоритет."""
    t = parse_file(EU_FILE, decimal=".")
    assert t.dialect.decimal == "."
    # С точкой как десятичной '5,2' уже не число.
    assert t.numeric_column("урожай") == []


def test_no_header_mode():
    t = parse_file(EU_FILE, has_header=False)
    assert t.header[0] == "col1"
    assert t.n_rows == 5  # строка заголовка стала данными


def test_quality_summary_clean():
    clean = b"a,b\n1,2\n3,4\n5,6\n"
    q = parse_file(clean).quality_summary()
    assert q["verdict"] == "ok"
    assert q["n_missing"] == 0
    assert q["n_columns"] == 2


def test_quality_summary_dirty():
    q = parse_file(DIRTY_FILE).quality_summary()
    assert q["verdict"] in {"warnings", "problems"}
    assert q["problem_columns"]
    assert q["top_issues"]


def test_empty_file():
    t = parse_file(b"")
    assert t.n_rows == 0
    assert any(w["code"] == "file_empty" for w in t.warnings)


def test_preview_limited():
    many = b"x\n" + b"\n".join(str(i).encode() for i in range(100))
    t = parse_file(many)
    assert t.n_rows == 100
    assert len(t.preview_rows) == 20


def test_ragged_rows_survive():
    ragged = b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n"
    t = parse_file(ragged)
    assert len(t.profiles) == 3


def test_serializable():
    t = parse_file(EU_FILE)
    d = t.to_dict()
    assert set(d) >= {"dialect", "header", "preview_rows", "profiles", "quality"}
    assert d["quality"]["n_columns"] == 3


def test_suggest_methods_numeric_and_categorical():
    t = parse_file(EU_FILE)
    methods = {s["method"] for s in suggest_methods(t.profiles)}
    assert "describe_all" in methods
    assert "one_way_anova" in methods
    assert "simple_linear_regression" in methods  # две числовые колонки


def test_type_override_enables_grouping():
    """После смены числовой колонки на категорию она годна для групп."""
    data = (
        "pclass,age\n"
        + "\n".join(f"{(i % 3) + 1},{20 + i}" for i in range(60))
        + "\n"
    ).encode("utf-8")
    t = parse_file(data, type_overrides={"pclass": "categorical"})
    pclass = next(p for p in t.profiles if p.name == "pclass")
    assert pclass.detected_type == "categorical"
    groups = t.groups_by("age", "pclass")
    assert len(groups) == 3


def test_type_override_invalid_ignored():
    """Опечатка в типе игнорируется, не ломает разбор."""
    t = parse_file(EU_FILE, type_overrides={"урожай": "чепуха"})
    assert t.n_rows == 4


def test_rows_slice_pagination():
    """Порционная выдача строк для прогрессивного предпросмотра."""
    data = ("n\n" + "\n".join(str(i) for i in range(100)) + "\n").encode()
    t = parse_file(data)
    first = t.rows_slice(0, 15)
    assert len(first) == 15
    assert first[0][0] == "0"
    second = t.rows_slice(15, 15)
    assert second[0][0] == "15"


def test_rows_slice_beyond_end():
    """Срез за концом данных обрезается, не падает."""
    data = ("n\n" + "\n".join(str(i) for i in range(20)) + "\n").encode()
    t = parse_file(data)
    tail = t.rows_slice(15, 50)
    assert len(tail) == 5


def test_rows_slice_normalizes_ragged():
    """Неровные строки дополняются до ширины заголовка."""
    t = parse_file(b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    sliced = t.rows_slice(0, 3)
    assert all(len(row) == 3 for row in sliced)
    assert sliced[1] == ["4", "5", ""]  # дополнено пустым


def test_paired_columns_alignment():
    """Парные колонки выравниваются по строкам, пары с пропуском выпадают."""
    data = "b,a\n120,115\n118,116\n125,н/д\n130,128\n".encode("cp1251")
    t = parse_file(data)
    before, after = t.paired_columns("b", "a")
    # Третья строка (125, н/д) выпадает целиком.
    assert len(before) == len(after) == 3
    assert before == pytest.approx([120.0, 118.0, 130.0])
    assert after == pytest.approx([115.0, 116.0, 128.0])


def test_paired_columns_both_missing():
    """Строка выпадает, если пропуск в любой из двух колонок."""
    data = b"b,a\n1,2\n,5\n3,\n7,8\n"
    t = parse_file(data)
    before, after = t.paired_columns("b", "a")
    assert len(before) == 2  # только строки 1 и 4 полные


def test_usability_good_file():
    t = parse_file(b"x,y\n1,2\n3,4\n")
    u = t.usability()
    assert u["ok"] is True
    assert u["reason"] is None


def test_usability_empty():
    assert parse_file(b"").usability()["reason"] == "file_empty"


def test_usability_header_only():
    assert parse_file(b"a,b,c\n").usability()["reason"] == "no_data_rows"


def test_usability_no_numeric():
    data = b"name,city\nAnna,Lisbon\nBob,Porto\n"
    assert parse_file(data).usability()["reason"] == "no_numeric"


def test_summary_values():
    """Ключевые метрики по числовым колонкам для подстановки в параметры."""
    data = b"x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n"
    t = parse_file(data)
    sv = t.summary_values()
    assert set(sv) == {"x", "y"}
    assert sv["x"]["mean"] == pytest.approx(3.0)
    assert sv["x"]["median"] == pytest.approx(3.0)
    assert sv["y"]["mean"] == pytest.approx(30.0)
    assert "std" in sv["x"]


def test_summary_values_excludes_categorical():
    """Категориальные колонки в сводку не попадают."""
    data = (
        "sort,val\n" + "\n".join(f"{'ab'[i % 2]},{i}" for i in range(10)) + "\n"
    ).encode()
    t = parse_file(data)
    assert "sort" not in t.summary_values()
    assert "val" in t.summary_values()


def test_suggest_methods_numeric_only():
    t = parse_file(b"x,y\n1,2\n3,4\n5,6\n")
    methods = {s["method"] for s in suggest_methods(t.profiles)}
    assert "describe_all" in methods
    assert "one_way_anova" not in methods  # нет категориальной колонки
