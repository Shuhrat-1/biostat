"""Словари имён колонок (аудит 01-14): блок, время, до / после, контроль, проценты.

Имена — как их пишут в полевых журналах на ru / en / pt. Правила по именам —
подсказки [предл.]; тесты проверяют и сами словари, и правила карты на них.
"""

from __future__ import annotations

import pytest

from core.validation import column_names as names
from tests.test_recognize import _fired

LANGS = {"ru", "en", "pt"}


def test_01_14_every_kind_has_words_in_three_languages() -> None:
    for kind, langs in names.WORDS.items():
        assert set(langs) == LANGS, kind
        assert all(langs[lang] for lang in LANGS), kind


def test_01_14_dictionary_words_are_already_normalized() -> None:
    """Слово в словаре записано так, как его даёт normalize: иначе оно не найдётся."""
    for langs in names.WORDS.values():
        for word in (w for ws in langs.values() for w in ws):
            assert names.normalize(word) == word


@pytest.mark.parametrize("name", [
    "block", "Blocks", "Block_ID", "block1", "BlockNo", "rep_no", "Rep.", "Replicate_1",
    "replication", "Повторность", "повторение", "Блоки", "blocos", "repetição", "bloque",
])
def test_01_14_block_names(name: str) -> None:
    assert names.is_block(name)


@pytest.mark.parametrize("name", ["report", "blockage", "reproduction", "R", "yield"])
def test_01_14_not_block_names(name: str) -> None:
    assert not names.is_block(name)


@pytest.mark.parametrize("name", [
    "Day", "sampling_day", "treatment_date", "DAS", "hour", "срок", "час_учёта",
    "дней после посева", "mês", "dia", "semana", "ano",
])
def test_01_14_time_names(name: str) -> None:
    assert names.is_time(name)


@pytest.mark.parametrize("name", ["daylength", "yearly_yield", "timing", "monthly", "t"])
def test_01_14_not_time_names(name: str) -> None:
    assert not names.is_time(name)


@pytest.mark.parametrize(("before", "after"), [
    ("weight_before", "weight_after"), ("До обработки", "После обработки"),
    ("antes", "depois"), ("initial", "final"), ("t0", "t1"), ("день 1", "день 2"),
    ("Day1", "Day2"), ("baseline", "followup"),
])
def test_01_14_before_after_names(before: str, after: str) -> None:
    assert names.is_before(before) and not names.is_after(before)
    assert names.is_after(after) and not names.is_before(after)


@pytest.mark.parametrize("name", ["доза", "prediction", "postharvest", "d14"])
def test_01_14_not_before_after_names(name: str) -> None:
    assert not names.is_before(name) and not names.is_after(name)


@pytest.mark.parametrize("level", [
    "Control", "контроль", "CK", "untreated", "без обработки", "Control (water)",
    "контроль (вода)", "controle", "testemunha", "Sem tratamento",
])
def test_01_14_control_levels(level: str) -> None:
    assert names.is_control(level)


@pytest.mark.parametrize("level", ["biocontrol", "K", "0", "water", "N60"])
def test_01_14_not_control_levels(level: str) -> None:
    assert not names.is_control(level)


@pytest.mark.parametrize("name", ["germination_%", "pct_infected", "Доля всхожих",
                                  "porcentagem", "Percent cover"])
def test_01_14_percent_names(name: str) -> None:
    assert names.is_percent(name)


# --- Правила карты на словарях --------------------------------------------------

def _plots(block: str) -> dict[str, list]:
    """Четыре сорта в трёх повторностях, по два растения с делянки.

    Две строки в ячейке: раскладка сама блоков не выдаёт (иначе сработало бы
    B1.8), остаётся подсказка по имени колонки.
    """
    y = [4.1, 4.8, 5.3, 4.4, 4.5, 5.0, 5.9, 4.6, 3.9, 4.7, 5.5, 4.2]
    return {block: [b for b in (1, 2, 3) for _ in range(8)],
            "variety": [v for _ in range(3) for v in "ABCD" for _ in range(2)],
            "yield": [round(x + d, 1) for x in y for d in (0.0, 0.3)]}


@pytest.mark.parametrize("block", ["Block_ID", "blocks", "repetição", "Повторение"])
def test_01_14_block_hint_on_real_headers(block: str) -> None:
    fired = _fired(_plots(block), response="yield", group="variety")
    assert fired["B1.1"]["message"]["params"] == {"c": block}


def test_01_14_before_after_hint_on_phrases() -> None:
    data = {"До обработки": [12.1, 10.4, 11.8, 9.9, 13.0, 10.7],
            "После обработки": [10.2, 9.1, 10.9, 9.0, 11.1, 9.8]}
    params = _fired(data)["B0.1"]["message"]["params"]
    assert params == {"a": "До обработки", "b": "После обработки"}


@pytest.mark.parametrize("level", ["CK", "без обработки", "Control (water)"])
def test_01_14_control_hint_on_real_levels(level: str) -> None:
    data = {"treatment": [t for t in (level, "N60", "N120") for _ in range(4)],
            "yield": [3.1, 3.4, 2.9, 3.3, 4.0, 4.2, 3.8, 4.1, 4.6, 4.9, 4.4, 4.7]}
    fired = _fired(data, response="yield", group="treatment")
    assert fired["B2.4"]["message"]["params"]["lv"] == level


def test_01_14_daylength_is_not_a_time_column() -> None:
    """Каждое растение — при одном значении длины дня в каждой камере: не время."""
    data = {"plant": [p for p in ("p1", "p2", "p3", "p4") for _ in range(3)],
            "daylength": [8, 12, 16] * 4,
            "height": [21, 25, 31, 19, 24, 30, 22, 27, 33, 20, 26, 29]}
    assert "B0.3" not in _fired(data, response="height")


def test_01_14_das_is_a_time_column() -> None:
    data = {"plant": [p for p in ("p1", "p2", "p3", "p4") for _ in range(3)],
            "DAS": [14, 28, 42] * 4,
            "height": [21, 25, 31, 19, 24, 30, 22, 27, 33, 20, 26, 29]}
    assert _fired(data, response="height")["B0.3"]["message"]["params"] == {"t": "DAS", "o": "plant"}
