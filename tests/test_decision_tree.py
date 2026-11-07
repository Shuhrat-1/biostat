"""Целостность дерева карты (web/src/decisionTree.json) и его подписей.

Дерево — данные фронта, но сверяется здесь: рядом каталог методов ядра и
реестр распознавателей. Тест падает, если метод добавлен в ядро, а на
карте его нет, если пресет ссылается на несуществующий параметр, или если
подпись распознавателя разошлась с текстом правила.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from core.validation import recognizers  # noqa: F401  регистрирует правила
from core.validation.catalog import CATALOG
from core.validation.recognize import REGISTRY

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
DATA = json.loads((WEB / "decisionTree.json").read_text(encoding="utf-8"))
TREE = DATA["nodes"]
TARGETS = DATA["targets"]
LABELS = json.loads((WEB / "treeLabels.json").read_text(encoding="utf-8"))
BY_ID = {n["id"]: n for n in TREE}
SPECS = {s.method: s for s in CATALOG}
KINDS = {"choice", "test", "pre", "leaf", "post"}
DEAD_ENDS = {"P.splitpower", "R2.censored", "R2.multivariate", "DIAG.serial"}


def _children(node_id: str) -> list[dict]:
    return [n for n in TREE if n["parent"] == node_id]


# --- Структура ------------------------------------------------------------------

def test_ids_unique_single_root_parents_exist():
    assert len(BY_ID) == len(TREE)
    roots = [n["id"] for n in TREE if n["parent"] is None]
    assert roots == ["R0"]
    assert all(n["parent"] in BY_ID for n in TREE if n["parent"] is not None)
    assert all(n["kind"] in KINDS for n in TREE)


def test_every_node_reachable_from_root_without_cycles():
    for node in TREE:
        seen, cur = set(), node
        while cur["parent"] is not None:
            assert cur["id"] not in seen, f"цикл через {cur['id']}"
            seen.add(cur["id"])
            cur = BY_ID[cur["parent"]]
        assert cur["id"] == "R0"


def test_leaves_have_no_children_inner_nodes_have_some():
    for node in TREE:
        kids = _children(node["id"])
        assert bool(kids) == (node["kind"] != "leaf"), node["id"]


def _path(node_id: str) -> list[str]:
    out = [node_id]
    while BY_ID[out[-1]]["parent"] is not None:
        out.append(BY_ID[out[-1]]["parent"])
    return out[::-1]


def test_fig_21_1_one_sample_is_a_difference_question():
    """Witte, Fig. 21.1 (стр. 405): t одной выборки — проверка различия с μ₀, лист под
    «Различия» (B0) рядом с описательным D.onesample."""
    leaves = {n["id"] for n in TREE if n.get("method") == "one_sample_t"}
    assert leaves == {"D.onesample", "B0.onesample"}
    assert _path("B0.onesample")[:4] == ["R0", "R2", "Q", "B0"]


# --- Связь с каталогом ядра --------------------------------------------------------

def test_every_catalog_method_is_on_the_map():
    on_map = {n["method"] for n in TREE if n.get("method")}
    assert set(SPECS) - on_map == set()


def test_leaf_methods_exist_or_are_dead_ends():
    for n in (n for n in TREE if n["kind"] == "leaf"):
        if n.get("status") == "missing":
            assert "method" not in n, n["id"]
        else:
            assert n.get("method") in SPECS, n["id"]


def test_dead_ends_match_spec():
    assert {n["id"] for n in TREE if n.get("status") == "missing"} == DEAD_ENDS


def test_glmm_leaves_marked_experimental():
    for n in TREE:
        if n.get("method") == "glmm_analysis":
            assert n.get("status") == "experimental", n["id"]


def test_presets_use_real_params_and_options():
    for n in (n for n in TREE if n.get("preset")):
        params = {p.name: p for p in SPECS[n["method"]].params}
        for key, value in n["preset"].items():
            assert key in params, f"{n['id']}: нет параметра {key}"
            assert not params[key].options or value in params[key].options, n["id"]


def test_sources_are_tagged():
    for n in (n for n in TREE if n.get("source")):
        assert re.match(r"^(кн\. |BioStat|предл\.)", n["source"]), n["id"]


# --- Связь с реестром распознавателей ------------------------------------------------

def test_recognizer_nodes_attached_to_tree():
    registry_nodes = {r.node for r in REGISTRY}
    tree_rnodes = {n["rnode"] for n in TREE if n.get("rnode")}
    assert registry_nodes == tree_rnodes


def test_every_suggestion_resolves_to_tree():
    """suggests распознавателя — узел, метод на карте или ключ targets."""
    on_map = {n["method"] for n in TREE if n.get("method")}
    for r in REGISTRY:
        s = r.suggests
        keyed = s in TARGETS or any(k.startswith(s + "#") for k in TARGETS)
        assert s in BY_ID or s in on_map or keyed, f"{r.id}: «{s}» не ведёт на карту"


def test_targets_point_to_existing_nodes_and_are_used():
    used = {r.suggests for r in REGISTRY}
    for key, nodes in TARGETS.items():
        assert key.split("#")[0] in used, f"лишний ключ targets: {key}"
        assert all(n in BY_ID for n in nodes), key


# --- Подписи --------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["ru", "en", "pt"])
def test_every_node_labelled(lang):
    names = LABELS[lang]["nodes"]
    assert set(names) == set(BY_ID)
    assert all(text.strip() for text in names.values())


CONTEXTS = {"experiment", "observation", "unknown"}


def test_context_notes_consistent_across_languages():
    ru = LABELS["ru"]["ctx"]
    assert set(ru) <= set(BY_ID)
    for node, notes in ru.items():
        assert set(notes) <= CONTEXTS, node
        for lang in ("en", "pt"):
            assert set(LABELS[lang]["ctx"][node]) == set(notes), (node, lang)


def _python_notes() -> dict[str, str]:
    """Русские тексты Signal(...) из правил: ключ id или id#variant."""
    from core.validation import recognizers_computed

    out = {}
    for module in (recognizers, recognizers_computed):
        out.update(_module_notes(Path(module.__file__).read_text(encoding="utf-8")))
    return out


def _module_notes(src: str) -> dict[str, str]:
    out = {}
    for fn in (n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)):
        rid = next((d.args[0].value for d in fn.decorator_list
                    if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "rule"), None)
        if rid is None:
            continue
        for call in (c for c in ast.walk(fn)
                     if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "Signal"):
            variant = None
            if len(call.args) > 1 and isinstance(call.args[1], ast.Dict):
                for k, v in zip(call.args[1].keys, call.args[1].values):
                    if getattr(k, "value", None) == "variant":
                        variant = v.value
            out[f"{rid}#{variant}" if variant else rid] = ast.literal_eval(call.args[0])
    return out


def test_recognizer_templates_match_rules_in_all_languages():
    notes = _python_notes()
    assert set(LABELS["ru"]["rec"]) == set(notes)
    for key, ru in notes.items():
        assert LABELS["ru"]["rec"][key] == ru, f"{key}: ru разошёлся с правилом"
        holes = set(re.findall(r"\{(\w+)\}", ru))
        for lang in ("en", "pt"):
            assert set(re.findall(r"\{(\w+)\}", LABELS[lang]["rec"][key])) == holes, (key, lang)


def test_every_implemented_recognizer_has_a_template():
    implemented = {r.id for r in REGISTRY if r.fn is not None}
    covered = {k.split("#")[0] for k in LABELS["ru"]["rec"]}
    assert implemented == covered
