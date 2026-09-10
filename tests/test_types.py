"""Тесты типизации колонок на классических дефектах Excel-экспорта."""

from __future__ import annotations

from core.validation.types import (
    ColumnType,
    is_missing,
    match_date_format,
    parse_number,
    profile_column,
    profile_table,
)


def test_missing_tokens():
    for token in ["", "NA", "n/a", "NULL", "-", "?", "NaN", "#Н/Д"]:
        assert is_missing(token), token
    assert not is_missing("0")
    assert not is_missing("пшеница")


def test_missing_tokens_russian():
    """Русские обозначения — аудитория пишет именно так."""
    for token in ["н/д", "Н/Д", "нет данных", "б/д", "отсутствует"]:
        assert is_missing(token), token


def test_parse_number_dot():
    assert parse_number("5.2") == 5.2
    assert parse_number("-3") == -3.0
    assert parse_number("abc") is None


def test_parse_number_comma_decimal():
    """Европейский формат: 5,2 -> 5.2"""
    assert parse_number("5,2", decimal=",") == 5.2


def test_parse_number_thousands_separator():
    """Excel часто отдаёт '1 200,5' с пробелом-разделителем тысяч."""
    assert parse_number("1 200,5", decimal=",") == 1200.5
    assert parse_number("1\xa0200,5", decimal=",") == 1200.5


def test_match_date_formats():
    assert match_date_format("2020-03-04") == "ISO (ГГГГ-ММ-ДД)"
    assert match_date_format("04.03.2020") == "точки (ДД.ММ.ГГГГ)"
    assert match_date_format("03/04/2020") is not None
    assert match_date_format("пшеница") is None


def test_clean_numeric_column():
    p = profile_column("урожай", ["5.2", "4.8", "3.9", "6.1"])
    assert p.detected_type == ColumnType.NUMERIC
    assert p.confidence == 1.0
    assert p.n_type_mismatch == 0
    assert not p.issues


def test_numeric_column_with_string_intruders():
    """Классика: числовая колонка с текстовыми ячейками (не пропусками)."""
    values = ["5.2", "4.8", "ошибка ввода", "6.1"] + ["5.0"] * 40
    p = profile_column("урожай", values)
    assert p.detected_type == ColumnType.NUMERIC
    assert p.n_type_mismatch == 1
    assert "ошибка ввода" in p.mismatch_examples
    assert any(i["code"] == "numeric_intruders" for i in p.issues)


def test_european_decimal_column():
    p = profile_column("урожай", ["5,2", "4,8", "3,9"], decimal=",")
    assert p.detected_type == ColumnType.NUMERIC
    assert p.confidence == 1.0


def test_mixed_date_formats_flagged():
    """Смешанные форматы дат в одной колонке — дефект Excel."""
    values = ["2020-03-04", "05.03.2020", "2020-03-06", "07.03.2020"]
    p = profile_column("дата", values)
    assert p.detected_type == ColumnType.DATE
    assert len(p.date_formats) == 2
    assert any(i["code"] == "mixed_date_formats" for i in p.issues)


def test_ambiguous_slash_date_warned():
    values = ["03/04/2020", "05/06/2020", "07/08/2020"]
    p = profile_column("дата", values)
    assert p.detected_type == ColumnType.DATE
    assert any(i["code"] == "ambiguous_slash_date" for i in p.issues)


def test_categorical_column():
    p = profile_column("сорт", ["пшеница", "ячмень", "овёс", "пшеница"])
    assert p.detected_type == ColumnType.CATEGORICAL
    assert p.n_unique == 3


def test_mixed_type_column_flagged():
    """Половина числа, половина текст — подозрительно."""
    values = ["1", "2", "3", "4", "текст", "ещё текст", "нет", "пусто"]
    p = profile_column("смесь", values)
    assert p.detected_type == ColumnType.CATEGORICAL
    assert any(i["code"] == "mixed_types" for i in p.issues)


def test_missing_values_counted():
    p = profile_column("x", ["1", "NA", "3", "", "5"])
    assert p.n_missing == 2
    assert p.n_total == 5
    assert p.missing_ratio == 0.4


def test_high_missing_ratio_flagged():
    p = profile_column("x", ["1"] + ["NA"] * 9)
    assert p.missing_ratio == 0.9
    assert any(i["code"] == "missing_high" for i in p.issues)


def test_empty_column():
    p = profile_column("x", ["", "NA", "-"])
    assert p.detected_type == ColumnType.EMPTY


def test_profile_table_ragged_rows():
    """Строки разной длины не должны ломать профилирование."""
    header = ["a", "b", "c"]
    rows = [["1", "2", "3"], ["4", "5"], ["6", "7", "8", "9"]]
    profiles = profile_table(header, rows)
    assert len(profiles) == 3
    assert all(p.n_total == 3 for p in profiles)


def test_profile_table_realistic():
    """Реалистичная таблица полевого опыта."""
    header = ["дата", "сорт", "урожай", "площадь"]
    rows = [
        ["2020-05-01", "пшеница", "5,2", "120"],
        ["2020-05-02", "ячмень", "4,8", "95"],
        ["2020-05-03", "овёс", "н/д", "80"],
    ]
    profiles = profile_table(header, rows, decimal=",")
    by_name = {p.name: p for p in profiles}
    assert by_name["дата"].detected_type == ColumnType.DATE
    assert by_name["сорт"].detected_type == ColumnType.CATEGORICAL
    assert by_name["урожай"].detected_type == ColumnType.NUMERIC
    assert by_name["урожай"].n_missing == 1


def test_profile_serializable():
    p = profile_column("x", ["1", "2", "3"])
    d = p.to_dict()
    assert d["detected_type"] == "numeric"
    assert "missing_ratio" in d
    assert "likely_categorical" in d


def test_likely_categorical_coded_column():
    """Числовая колонка с малым числом уникальных при большом n — подсказка."""
    # 3 уникальных (1,2,3) на 90 строк — как pclass.
    values = ["1", "2", "3"] * 30
    p = profile_column("pclass", values)
    assert p.detected_type == ColumnType.NUMERIC  # авто оставляет числом
    assert p.likely_categorical is True
    assert any(i["code"] == "likely_categorical" for i in p.issues)


def test_continuous_numeric_not_flagged():
    """Непрерывные измерения не помечаются как категория даже при малом n."""
    p = profile_column("урожай", ["5.2", "4.8", "3.9", "6.1"])
    assert p.likely_categorical is False


def test_force_type_numeric_to_categorical():
    """Ручное переопределение: числовую колонку трактуем как категорию."""
    values = ["1", "2", "3"] * 30
    p = profile_column("pclass", values, force_type=ColumnType.CATEGORICAL)
    assert p.detected_type == ColumnType.CATEGORICAL


def test_force_type_in_table():
    """Переопределение через profile_table по имени колонки."""
    header = ["pclass", "age"]
    rows = [["1", "22"], ["3", "35"], ["2", "58"], ["1", "40"]]
    profiles = profile_table(
        header, rows, type_overrides={"pclass": ColumnType.CATEGORICAL}
    )
    by_name = {p.name: p for p in profiles}
    assert by_name["pclass"].detected_type == ColumnType.CATEGORICAL
    assert by_name["age"].detected_type == ColumnType.NUMERIC


def test_likely_id_integer_column():
    """Целочисленная колонка с уникальными значениями — подсказка ID."""
    values = [str(i) for i in range(1, 101)]
    p = profile_column("id", values)
    assert p.likely_id is True
    assert any(i["code"] == "likely_id" for i in p.issues)


def test_likely_id_string_column():
    """Строковые коды тоже распознаются как возможный ID."""
    values = [f"P{i:04d}" for i in range(100)]
    p = profile_column("code", values)
    assert p.likely_id is True


def test_continuous_not_flagged_as_id():
    """Дробные измерения (цены) не помечаются как ID даже при уникальности."""
    import random

    random.seed(3)
    values = [f"{random.gauss(30, 15):.2f}" for _ in range(100)]
    p = profile_column("fare", values)
    assert p.likely_id is False


def test_force_id_type():
    """Ручное назначение типа ID."""
    values = [str(i) for i in range(50)]
    p = profile_column("id", values, force_type=ColumnType.ID)
    assert p.detected_type == ColumnType.ID


def test_id_duplicate_detection():
    """Повторяющиеся ID (грязные данные) выносятся в предупреждение."""
    values = ["1", "2", "3", "3", "4", "5", "5", "5", "6", "7", "8"]
    p = profile_column("id", values, force_type=ColumnType.ID)
    assert p.detected_type == ColumnType.ID
    assert p.n_duplicates == 3
    assert any(i["code"] == "id_not_unique" for i in p.issues)


def test_id_unique_no_warning():
    """Уникальный ID не даёт предупреждения о дубликатах."""
    values = [str(i) for i in range(20)]
    p = profile_column("id", values, force_type=ColumnType.ID)
    assert not any(i["code"] == "id_not_unique" for i in p.issues)
