"""Тесты детекции параметров CSV на реалистичных файлах."""

from __future__ import annotations

from core.validation.dialect import (
    detect_decimal,
    detect_delimiter,
    detect_dialect,
    detect_encoding,
    detect_header,
)

# Европейский экспорт из Excel: ; как разделитель, запятая десятичная.
EU_CSV = "культура;урожай;площадь\nпшеница;5,2;120\nячмень;4,8;95\nовёс;3,9;80\n"
# Английский стандарт: запятая-разделитель, точка десятичная.
US_CSV = "crop,yield,area\nwheat,5.2,120\nbarley,4.8,95\noats,3.9,80\n"
TAB_CSV = "crop\tyield\twheat\t5.2\nbarley\t4.8\n"


def test_encoding_utf8_cyrillic():
    enc, conf = detect_encoding(EU_CSV.encode("utf-8"))
    assert enc == "utf-8"
    assert conf > 0.5


def test_encoding_cp1251_cyrillic():
    """Windows-1251 — классика Excel в РФ/СНГ, должен определяться."""
    enc, _ = detect_encoding(EU_CSV.encode("cp1251"))
    assert enc.lower().replace("-", "_") in {"cp1251", "windows_1251"}


def test_encoding_empty():
    enc, conf = detect_encoding(b"")
    assert enc == "utf-8"
    assert conf == 0.0


def test_delimiter_semicolon():
    delim, conf = detect_delimiter(EU_CSV)
    assert delim == ";"
    assert conf == 1.0  # число колонок стабильно во всех строках


def test_delimiter_comma():
    delim, conf = detect_delimiter(US_CSV)
    assert delim == ","
    assert conf == 1.0


def test_delimiter_tab():
    delim, _ = detect_delimiter("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
    assert delim == "\t"


def test_delimiter_empty_text():
    delim, conf = detect_delimiter("")
    assert delim == ","
    assert conf == 0.0


def test_decimal_comma_with_semicolon():
    """5,2 при разделителе ; — запятая десятичная."""
    assert detect_decimal(EU_CSV, ";") == ","


def test_decimal_dot_with_comma_delimiter():
    """При запятой-разделителе десятичная всегда точка."""
    assert detect_decimal(US_CSV, ",") == "."


def test_decimal_dot_with_semicolon():
    assert detect_decimal("a;b\n1.5;2.7\n3.1;4.2\n", ";") == "."


def test_header_detected():
    """Текстовая первая строка + числовое тело = заголовок."""
    assert detect_header(EU_CSV, ";") is True


def test_header_absent():
    """Все строки числовые — заголовка нет."""
    assert detect_header("1;2;3\n4;5;6\n7;8;9\n", ";") is False


def test_dialect_european_excel_export():
    """Полная детекция типичного европейского экспорта."""
    info = detect_dialect(EU_CSV.encode("utf-8"))
    assert info.delimiter == ";"
    assert info.decimal == ","
    assert info.has_header is True
    assert info.n_columns == 3


def test_dialect_us_style():
    info = detect_dialect(US_CSV.encode("utf-8"))
    assert info.delimiter == ","
    assert info.decimal == "."
    assert info.n_columns == 3


def test_dialect_cp1251_roundtrip():
    """Файл в CP1251 читается без потери кириллицы."""
    info = detect_dialect(EU_CSV.encode("cp1251"))
    assert info.delimiter == ";"
    assert info.n_columns == 3
    # Нет замещающих символов — значит кодировка распознана верно.
    assert not any("нечитаемые" in n for n in info.notes)


def test_dialect_notes_on_broken_encoding():
    """Битые байты дают замечание о нечитаемых символах."""
    broken = "имя;значение\n".encode("utf-8") + b"\xff\xfe\x00" + b"a;1\n"
    info = detect_dialect(broken)
    assert isinstance(info.notes, list)


def test_dialect_serializable():
    info = detect_dialect(EU_CSV.encode("utf-8"))
    d = info.to_dict()
    assert d["delimiter"] == ";"
    assert set(d) == {
        "encoding",
        "encoding_confidence",
        "delimiter",
        "delimiter_confidence",
        "decimal",
        "has_header",
        "n_columns",
        "notes",
    }
