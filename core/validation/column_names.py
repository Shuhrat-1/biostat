"""Словари имён колонок и уровней: блок, время, до / после, контроль, проценты (ru / en / pt).

Данные, а не регулярные выражения: чтобы добавить слово, его дописывают в
список нужного языка. Имя режется на слова по любым не-буквам (пробел, «_»,
точка, скобки), по границе строчная → прописная (``BlockNo``) и по цифрам в
конце слова (``block1``); у латиницы снимается диакритика (``repetição`` →
``repeticao``). Слово имени сравнивается со словарём целиком, поэтому
``daylength`` — не «день», а ``yearly_yield`` — не «год».

Все правила по именам — подсказки [предл.]: имя колонки ничего не доказывает.
"""

from __future__ import annotations

import re
import unicodedata

WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "block": {
        "en": ("block", "blocks", "blk", "rep", "reps", "replicate", "replicates",
               "replication", "replications"),
        "pt": ("bloco", "blocos", "repeticao", "repeticoes", "bloque", "bloques"),
        "ru": ("блок", "блоки", "блока", "повтор", "повт", "повторность", "повторности",
               "повторение", "повторения"),
    },
    "time": {
        "en": ("time", "day", "days", "date", "week", "weeks", "month", "months", "year",
               "years", "hour", "hours", "das", "dap", "dat", "doy"),
        "pt": ("tempo", "dia", "dias", "data", "semana", "semanas", "mes", "meses", "ano",
               "anos", "hora", "horas"),
        "ru": ("время", "день", "дни", "дня", "дней", "дата", "неделя", "недели", "недель",
               "месяц", "месяцы", "месяца", "год", "годы", "года", "час", "часы", "сутки",
               "суток", "срок", "сроки"),
    },
    "before": {
        "en": ("before", "pre", "baseline", "initial", "start", "t0", "d0"),
        "pt": ("antes", "inicial", "inicio"),
        "ru": ("до", "исходный", "исходная", "исходное", "начало", "начальный",
               "начальная", "начальное"),
    },
    "after": {
        "en": ("after", "post", "final", "end", "followup", "t1"),
        "pt": ("depois", "apos", "fim"),
        "ru": ("после", "конечный", "конечная", "конечное", "конец", "итог", "итоговый",
               "итоговая", "итоговое"),
    },
    "percent": {
        "en": ("pct", "percent", "percentage", "perc", "proportion", "share", "fraction"),
        "pt": ("porcentagem", "percentagem", "percentual", "porcento", "proporcao",
               "porcentaje"),
        "ru": ("процент", "проценты", "процентов", "проц", "доля", "доли"),
    },
    "control": {
        "en": ("control", "ctrl", "ck", "check", "untreated", "standard"),
        "pt": ("controle", "controlo", "testemunha", "padrao", "testigo"),
        "ru": ("контроль", "контр", "стандарт", "эталон"),
    },
}

# Контроль, названный фразой: уровень начинается с неё (после нормализации).
CONTROL_PHRASES = ("без обработки", "no treatment", "not treated", "sem tratamento",
                   "nao tratado", "sin tratamiento")
# «День 1 / день 2» как до / после: слово времени и номер измерения.
_FIRST, _SECOND = "1", "2"

_SETS = {kind: frozenset(w for words in langs.values() for w in words)
         for kind, langs in WORDS.items()}
_CAMEL = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")
_WORD = re.compile(r"[a-zа-яё]+\d*|\d+")


def normalize(name: str) -> str:
    """Нижний регистр, слова через пробел, латиница без диакритики."""
    spaced = _CAMEL.sub(" ", name.strip())
    plain = "".join(_ascii_base(ch) for ch in spaced.lower())
    return " ".join(_WORD.findall(plain))


def _ascii_base(ch: str) -> str:
    """Буква латиницы без диакритики; кириллица (й, ё) остаётся как есть."""
    base = unicodedata.normalize("NFKD", ch)[0]
    return base if base.isascii() else ch


def words(name: str) -> list[str]:
    """Слова имени; слово с цифрой на конце даёт и вариант без неё (block1 → block)."""
    out: list[str] = []
    for token in normalize(name).split():
        out.append(token)
        stem = token.rstrip("0123456789")
        if stem and stem != token:
            out.append(stem)
    return out


def _has(name: str, kind: str) -> bool:
    return any(w in _SETS[kind] for w in words(name))


def is_block(name: str) -> bool:
    """Имя колонки похоже на блок или повторность."""
    return _has(name, "block")


def is_time(name: str) -> bool:
    """Имя колонки похоже на время, дату или срок."""
    return _has(name, "time")


def is_percent(name: str) -> bool:
    """Имя колонки похоже на проценты или долю (слово или знак %)."""
    return "%" in name or _has(name, "percent")


def _numbered_time(name: str, number: str) -> bool:
    """«День 1», «day2»: слово времени и сразу за ним номер измерения."""
    tokens = normalize(name).split()
    joined = [t for t in tokens if t.rstrip("0123456789") in _SETS["time"] and t.endswith(number)]
    spaced = any(a in _SETS["time"] and b == number for a, b in zip(tokens, tokens[1:]))
    return bool(joined) or spaced


def is_before(name: str) -> bool:
    """Имя колонки похоже на измерение «до»."""
    return _has(name, "before") or _numbered_time(name, _FIRST)


def is_after(name: str) -> bool:
    """Имя колонки похоже на измерение «после»."""
    return _has(name, "after") or _numbered_time(name, _SECOND)


def is_control(level: str) -> bool:
    """Уровень фактора похож на контроль (слово или фраза «без обработки»)."""
    return _has(level, "control") or normalize(level).startswith(CONTROL_PHRASES)
