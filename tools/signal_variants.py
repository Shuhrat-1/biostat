"""Вызовы Signal(...) в правилах карты: текст, вариант и явная сила.

Общий разбор исходника recognizers.py и recognizers_computed.py (AST) для теста карты
(тексты treeLabels.json) и генератора страниц методов (пометка «(подсказка)» у варианта
сигнала). Ключ — id правила или id#variant, как в treeLabels.json.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

STRENGTHS = {"HINT": "hint", "STRONG": "strong"}  # имена констант core.validation.recognize


@dataclass(frozen=True)
class SignalCall:
    """Один вызов Signal(...) в правиле: strength — явная сила вызова или None."""

    rule: str
    variant: str | None
    text: str
    strength: str | None

    @property
    def key(self) -> str:
        return f"{self.rule}#{self.variant}" if self.variant else self.rule


def _rule_id(fn: ast.FunctionDef) -> str | None:
    return next((d.args[0].value for d in fn.decorator_list
                 if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "rule"), None)


def _variant(call: ast.Call) -> str | None:
    if len(call.args) > 1 and isinstance(call.args[1], ast.Dict):
        for k, v in zip(call.args[1].keys, call.args[1].values):
            if getattr(k, "value", None) == "variant":
                return v.value
    return None


def _strength(call: ast.Call) -> str | None:
    kw = next((k.value for k in call.keywords if k.arg == "strength"), None)
    return STRENGTHS.get(getattr(kw, "id", "")) if kw is not None else None


def module_calls(src: str) -> list[SignalCall]:
    """Вызовы Signal(...) внутри функций с декоратором @rule(...)."""
    out = []
    for fn in (n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)):
        rid = _rule_id(fn)
        if rid is None:
            continue
        out += [SignalCall(rid, _variant(c), ast.literal_eval(c.args[0]), _strength(c))
                for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "Signal"]
    return out


def rule_calls() -> dict[str, SignalCall]:
    """Все вызовы правил карты по ключу id / id#variant."""
    from core.validation import recognizers, recognizers_computed

    out: dict[str, SignalCall] = {}
    for module in (recognizers, recognizers_computed):
        for call in module_calls(Path(module.__file__).read_text(encoding="utf-8")):
            out[call.key] = call
    return out
