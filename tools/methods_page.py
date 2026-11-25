"""Страница «Методы и проверка» (шаг P5): данные собираются, а не пишутся вручную.

Источники данных:
- каталог ядра (core/validation/catalog.py) — методы, группы, названия на трёх языках;
- дерево карты (web/src/decisionTree.json) — источник метода (книга, страница);
- тесты (tests/test_*.py) — какие таблицы и примеры книг тест сверяет, вызывая метод;
- tools/methods_reference.json — независимая сверка аудита 2 (разовая, с датой и коммитом);
- BIOSTAT_RULES ниже — пороги с пометкой [BioStat] на трёх языках; тест сверяет их с кодом.

Пишет web/public/methods.json и страницы /methods/, /ru/methods/, /pt/methods/ — через
python tools/static_pages.py; tests/test_methods_page.py сверяет файлы с генератором.
"""

from __future__ import annotations

import ast
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validation.catalog import CATALOG  # noqa: E402
from core.validation.catalog_i18n import GROUPS, LABELS, tr  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import static_pages as sp  # noqa: E402
from static_pages import _inline  # noqa: E402

LANGS = ("en", "ru", "pt")
TREE = ROOT / "web" / "src" / "decisionTree.json"
REFERENCE = ROOT / "tools" / "methods_reference.json"
JSON_OUT = ROOT / "web" / "public" / "methods.json"
METHODS = [m.method for m in CATALOG]

BOOK_REF = re.compile(r"\b(Tables?|Example|Exercises?|Fig\.|Figure|Progress Checks?|Review(?: Question)?)\s+"
                      r"(\d+\.\d+[a-z]?(?:[A-Z](?![a-z]))?(?:\s?[–-]\s?\d*\.?\d+[a-zA-Z]?)?)")
PAGE_REF = re.compile(r"стр\.\s(\d+(?:[–-]\d+)?)")
WITTE = re.compile(r"Witte|Progress Check|Review")
WELHAM = re.compile(r"Welham|Книга|кн\.|книг")
EXTERNAL = re.compile(r"\b(lme4|GenStat|glmmML|glmmADMB|SAS|SPSS|Minitab)\b")
BRIDGE_CALLS = {"run", "plots", "summary_plots"}  # tests/bridge_runner.py: запуск метода через мост
REF_NAMES = {"Tables": "Table", "Exercises": "Exercise", "Progress Checks": "Progress Check",
             "Review Question": "Review", "Figure": "Fig."}


# --- тесты: какие методы вызывает тест и на какие страницы книг ссылается ---------------

def _called(node: ast.AST) -> set[str]:
    """Методы, которые узел вызывает: функция ядра m(...) или мост run("m", ...)."""
    found: set[str] = set()
    for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name in METHODS:
            found.add(name)
        first = call.args[0] if call.args else None
        if name in BRIDGE_CALLS and isinstance(first, ast.Constant) and first.value in METHODS:
            found.add(first.value)
    return found


def _book(context: str, module_doc: str) -> str | None:
    """Чья нумерация в ссылках теста: Witte, Welham или неизвестно."""
    if WITTE.search(context):
        return "witte"
    if WELHAM.search(context):
        return "welham"
    if WITTE.search(module_doc) and not WELHAM.search(module_doc):
        return "witte"
    return "welham" if WELHAM.search(module_doc) else None


def _refs(context: str) -> set[str]:
    refs = {f"{REF_NAMES.get(kind, kind)} {num}" for kind, num in BOOK_REF.findall(context)}
    return refs | {f"стр. {page}" for page in PAGE_REF.findall(context)}


def _test_functions(path: Path):
    """(функция-тест, её текст и комментарии, вызванные методы, докстринг модуля)."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    helpers = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    module_doc = ast.get_docstring(tree) or ""
    for fn in (f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name.startswith("test")):
        source = ast.get_source_segment(text, fn) or ""
        context = (ast.get_docstring(fn) or "") + "\n" + "\n".join(re.findall(r"#.*", source))
        called = _called(fn)
        for name in {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} & set(helpers):
            called |= _called(helpers[name])
        yield fn, context, called, module_doc


def tests_by_method() -> dict[str, dict]:
    """Метод → число тестов, файлы, ссылки на Welham и Witte, внешние эталоны."""
    rows = {m: {"tests": 0, "files": set(), "welham": set(), "witte": set(), "external": set()}
            for m in METHODS}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for _fn, context, called, module_doc in _test_functions(path):
            book, refs = _book(context, module_doc), _refs(context)
            external = set(EXTERNAL.findall(context))
            for m in called:
                row = rows[m]
                row["tests"] += 1
                row["files"].add(path.name)
                row["external"] |= external
                if book and refs:
                    row[book] |= refs
    return rows


# --- дерево карты и сверка аудита 2 --------------------------------------------------------

def tree_sources() -> dict[str, list[str]]:
    """Метод → источники его листьев в дереве карты (без повторов, по порядку)."""
    out: dict[str, list[str]] = {m: [] for m in METHODS}
    for node in json.loads(TREE.read_text(encoding="utf-8"))["nodes"]:
        method, source = node.get("method"), node.get("source")
        if method in out and source and source not in out[method]:
            out[method].append(source)
    return out


def _natural(ref: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", ref)]


def methods_data() -> dict:
    """Всё содержимое methods.json."""
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    coverage, sources = tests_by_method(), tree_sources()
    methods = []
    for spec in CATALOG:
        cov = coverage[spec.method]
        methods.append({
            "id": spec.method,
            "group": {lang: tr(GROUPS, spec.group, lang) for lang in LANGS},
            "name": {lang: tr(LABELS, spec.label, lang) for lang in LANGS},
            "sources": sources[spec.method],
            "proposed": any("предл." in s for s in sources[spec.method]),
            "tests": cov["tests"],
            "test_files": sorted(cov["files"]),
            "book_refs": {book: sorted(cov[book], key=_natural) for book in ("welham", "witte")},
            "external_reference_in_tests": sorted(cov["external"]),
            "audit_reference": reference["methods"].get(spec.method),
        })
    audit = {k: reference[k] for k in ("audit", "date", "commit")}
    return {"methods_count": len(methods), "audit": audit, "methods": methods,
            "biostat_thresholds": [{"anchors": r["anchors"], **r["text"]} for r in BIOSTAT_RULES]}


# --- пороги [BioStat] -----------------------------------------------------------------------

def _rule(anchors: list[str], ru: str, en: str, pt: str) -> dict:
    return {"anchors": anchors, "text": {"ru": ru, "en": en, "pt": pt}}


BIOSTAT_RULES = [
    _rule(["(|g₁| > 0.5) [BioStat]", "LARGE_GROUP = 30", "MIN_SKEW = 0.5", "асимметрия остатков [BioStat]",
           '@rule("B4.1"', '@rule("B4.7"'],
          "Нормальность остатков (B4.1, B4.7): если в каждой группе больше 30 наблюдений, нарушением "
          "считается только заметная асимметрия, |g₁| > 0.5; при меньших группах решает тест Шапиро–Уилка.",
          "Normality of residuals (B4.1, B4.7): with more than 30 observations in every group, only marked "
          "skewness, |g₁| > 0.5, counts as a violation; with smaller groups the Shapiro–Wilk test decides.",
          "Normalidade dos resíduos (B4.1, B4.7): com mais de 30 observações em cada grupo, só uma "
          "assimetria marcada, |g₁| > 0,5, conta como violação; com grupos menores decide o teste de "
          "Shapiro–Wilk."),
    _rule(["_MIN_PAIRS = 6", '@rule("B0.6"'],
          "Нормальность разностей пар (B0.6) проверяется с 6 пар: на меньшем числе тест почти ничего не "
          "замечает.",
          "Normality of paired differences (B0.6) is checked from 6 pairs on: with fewer the test detects "
          "almost nothing.",
          "A normalidade das diferenças emparelhadas (B0.6) é verificada a partir de 6 pares: com menos o "
          "teste quase nada deteta."),
    _rule(["Пороги — BioStat [предл.", "Порог 8 — BioStat", "две группы — порог BioStat",
           "решение BioStat 8.11.2026"],
          "Две группы (B4.2, two_sample_t): по умолчанию t Стьюдента; совет Уэлча — при отношении SD > 2, "
          "если n уравновешены, и > 1.5, если отношение n больше 1.5; при n < 8 в группе — только справка.",
          "Two groups (B4.2, two_sample_t): Student's t by default; Welch is advised when the SD ratio is "
          "> 2 with balanced n, or > 1.5 when the n ratio exceeds 1.5; with n < 8 in a group it is a note "
          "only.",
          "Dois grupos (B4.2, two_sample_t): t de Student por omissão; Welch é aconselhado quando a razão "
          "dos DP é > 2 com n equilibrados, ou > 1,5 quando a razão dos n excede 1,5; com n < 8 num grupo "
          "é só uma nota."),
    _rule(["Связь SD со средним — критерий BioStat", "SD_RATIO_FOR_TREND = 2.0", "R_FOR_TREND = 0.8",
           "MIN_GROUPS_FOR_TREND = 4", '@rule("B4.3"'],
          "SD растёт со средним (B4.3, сигнал к log-трансформации): r между log SD и log среднего групп "
          "≥ 0.8 при SD, различающихся не меньше чем вдвое; уверенный сигнал — с 4 групп.",
          "SD grows with the mean (B4.3, a cue for a log transformation): r between log SD and log mean of "
          "the groups ≥ 0.8 with SDs differing at least twofold; a confident signal from 4 groups on.",
          "O DP cresce com a média (B4.3, indício para transformação log): r entre log DP e log média dos "
          "grupos ≥ 0,8 com DP que diferem pelo menos para o dobro; sinal seguro a partir de 4 grupos."),
    _rule(['@rule("B4.4"'],
          "Асимметрия остатков (B4.4): |g₁| > 1 (от 8 остатков) — подсказка трансформировать отклик.",
          "Skewed residuals (B4.4): |g₁| > 1 (from 8 residuals) — a hint to transform the response.",
          "Resíduos assimétricos (B4.4): |g₁| > 1 (a partir de 8 resíduos) — sugestão de transformar a "
          "resposta."),
    _rule(['@rule("R2.7"'],
          "Асимметрия отклика (R2.7): g₁ > 1 при всех значениях > 0 (от 8 наблюдений) — кандидат на "
          "log-преобразование.",
          "Skewed response (R2.7): g₁ > 1 with all values > 0 (from 8 observations) — a candidate for a "
          "log transformation.",
          "Resposta assimétrica (R2.7): g₁ > 1 com todos os valores > 0 (a partir de 8 observações) — "
          "candidata a transformação log."),
    _rule(["Условие «минимум ≤ 10» [BioStat]", '@rule("R2.2"'],
          "Похоже на счёты (R2.2): неотрицательные целые, не только 0 и 1, минимум ≤ 10 (иначе это скорее "
          "целочисленные измерения, например граммы).",
          "Looks like counts (R2.2): non-negative integers, not only 0 and 1, minimum ≤ 10 (otherwise more "
          "likely integer measurements such as grams).",
          "Parece de contagens (R2.2): inteiros não negativos, não só 0 e 1, mínimo ≤ 10 (caso contrário, "
          "mais provavelmente medições inteiras, como gramas)."),
    _rule(["_ZERO_SHARE = 0.3", "_ZERO_SHARE_STRONG = 0.5", '@rule("R2.6"'],
          "Много нулей в отклике (R2.6): подсказка с доли нулей 30 %, уверенное указание — с 50 %.",
          "Many zeros in the response (R2.6): a hint from a 30 % share of zeros, a firm pointer from 50 %.",
          "Muitos zeros na resposta (R2.6): sugestão a partir de 30 % de zeros, indicação firme a partir "
          "de 50 %."),
    _rule(['@rule("G.2"'],
          "Разброс счётов (G.2): дисперсия больше среднего более чем вдвое (от 10 наблюдений) — возможна "
          "отрицательная биномиальная модель.",
          "Spread of counts (G.2): variance more than twice the mean (from 10 observations) — a negative "
          "binomial model may fit.",
          "Dispersão de contagens (G.2): variância mais do dobro da média (a partir de 10 observações) — "
          "pode servir um modelo binomial negativo."),
    _rule(["нужна заметная сверхдисперсия [BioStat]", '@rule("G.1"'],
          "Сверхдисперсия в GLM (G.1): девианс значимо больше числа степеней свободы и девианс/df > 1.2.",
          "Overdispersion in a GLM (G.1): the deviance is significantly larger than its degrees of freedom "
          "and deviance/df > 1.2.",
          "Sobredispersão num GLM (G.1): a deviance é significativamente maior do que os graus de "
          "liberdade e deviance/gl > 1,2."),
    _rule(["похоже на баллы [BioStat]", '@rule("B0.4"', '@rule("B4.8"'],
          "Похоже на баллы (B0.4, B4.8): целые, от 3 до 10 разных значений, размах не больше 10 — карта "
          "советует ранговые методы.",
          "Looks like scores (B0.4, B4.8): integers, 3 to 10 distinct values, range at most 10 — the map "
          "suggests rank methods.",
          "Parece de pontuações (B0.4, B4.8): inteiros, 3 a 10 valores distintos, amplitude até 10 — o "
          "mapa sugere métodos de postos."),
    _rule(["_MAX_CODE_LEVELS = 10", '@rule("R2.11"', "количество это или класс. Порог BioStat"],
          "Числовые коды классов (R2.11): 3–10 подряд идущих целых от 0 или 1, в среднем не меньше 5 строк "
          "на код; больше 10 разных значений — скорее измерение, чем номера вариантов.",
          "Numeric class codes (R2.11): 3–10 consecutive integers from 0 or 1, on average at least 5 rows "
          "per code; more than 10 distinct values are more likely a measurement than treatment numbers.",
          "Códigos numéricos de classes (R2.11): 3–10 inteiros consecutivos a partir de 0 ou 1, em média "
          "pelo menos 5 linhas por código; mais de 10 valores distintos são mais provavelmente uma medição."),
    _rule(['@rule("B0.2"'],
          "Возможные повторные измерения (B0.2): две числовые колонки в одной шкале (отношение средних от "
          "0.5 до 2) и r > 0.7 по 5 и более парам.",
          "Possible repeated measures (B0.2): two numeric columns on one scale (ratio of means 0.5 to 2) "
          "and r > 0.7 over 5 or more pairs.",
          "Possíveis medições repetidas (B0.2): duas colunas numéricas na mesma escala (razão das médias "
          "de 0,5 a 2) e r > 0,7 em 5 ou mais pares."),
    _rule(["не более раза [BioStat]"],
          "Повторные измерения по ролям: каждый объект измерен не меньше чем в двух моментах, в каждый — "
          "не больше одного раза.",
          "Repeated measures from roles: every subject is measured at two or more times, at most once at "
          "each time.",
          "Medições repetidas pelos papéis: cada objeto é medido em dois ou mais momentos, no máximo uma "
          "vez em cada."),
    _rule(["_MIN_RANDOM_LEVELS = 5", '@rule("B1.9"'],
          "Случайный фактор (B1.9): при меньше чем 5 уровнях компонента дисперсии оценивается ненадёжно — "
          "карта советует фиксированный блок.",
          "Random factor (B1.9): with fewer than 5 levels the variance component is estimated poorly — "
          "the map suggests a fixed block.",
          "Fator aleatório (B1.9): com menos de 5 níveis a componente de variância é mal estimada — o mapa "
          "sugere um bloco fixo."),
    _rule(["[BioStat: критерий дуги]", '@rule("A2.3"'],
          "Изогнутость прямой (A2.3): значим квадратичный член (p < 0.05) при 4 и более разных x.",
          "Curvature of a straight line (A2.3): the quadratic term is significant (p < 0.05) with 4 or more "
          "distinct x.",
          "Curvatura da reta (A2.3): o termo quadrático é significativo (p < 0,05) com 4 ou mais x "
          "distintos."),
    _rule(['@rule("A2.4"'],
          "Разброс растёт с подогнанным значением (A2.4): ρ Спирмена > 0.5 и p < 0.05, от 10 точек.",
          "Spread grows with the fitted value (A2.4): Spearman ρ > 0.5 and p < 0.05, from 10 points.",
          "A dispersão cresce com o valor ajustado (A2.4): ρ de Spearman > 0,5 e p < 0,05, a partir de 10 "
          "pontos."),
    _rule(["Бонферрони (не меньше 3) [BioStat]", '@rule("DIAG.1"'],
          "Выбросы в остатках (DIAG.1): доля |r| > 2 заметно больше ожидаемых 5 % или есть остаток больше "
          "порога Бонферрони (не меньше 3).",
          "Outlying residuals (DIAG.1): the share of |r| > 2 is clearly above the expected 5 %, or a residual "
          "exceeds the Bonferroni limit (at least 3).",
          "Resíduos atípicos (DIAG.1): a proporção de |r| > 2 está claramente acima dos 5 % esperados, ou "
          "um resíduo excede o limite de Bonferroni (pelo menos 3)."),
    _rule(["с 15 — в пределах 5 % [BioStat]", '@rule("DIAG.2"'],
          "Серийная корреляция остатков (DIAG.2): Дарбин–Уотсон вне (1, 3), проверка с 15 наблюдений — на "
          "10–12 статистика выходит за эти пределы случайно в 8–13 % случаев.",
          "Serial correlation of residuals (DIAG.2): Durbin–Watson outside (1, 3), checked from 15 "
          "observations on — with 10–12 the statistic leaves these limits by chance in 8–13 % of cases.",
          "Correlação serial dos resíduos (DIAG.2): Durbin–Watson fora de (1, 3), verificada a partir de 15 "
          "observações — com 10–12 a estatística sai destes limites por acaso em 8–13 % dos casos."),
    _rule(["(Кук, 1977) [BioStat]", '@rule("DIAG.4"'],
          "Влиятельные точки регрессии (DIAG.4): рычаг больше 2p/N и больше 0.2; влияние — расстояние Кука "
          "больше F₀.₅(p, N − p).",
          "Influential points in regression (DIAG.4): leverage above 2p/N and above 0.2; influence — Cook's "
          "distance above F₀.₅(p, N − p).",
          "Pontos influentes na regressão (DIAG.4): alavanca acima de 2p/N e acima de 0,2; influência — "
          "distância de Cook acima de F₀.₅(p, N − p)."),
    _rule(["[BioStat] Источник: перестановочное", "[BioStat] Предел точного расчёта"],
          "Точный P ранговых тестов при совпадениях — по перестановочному распределению; считается до 200 "
          "ненулевых разностей (знаково-ранговый тест) и до 100 наблюдений (Манн–Уитни).",
          "Exact P of rank tests with ties comes from the permutation distribution; it is computed up to "
          "200 non-zero differences (signed-rank test) and up to 100 observations (Mann–Whitney).",
          "O P exato dos testes de postos com empates vem da distribuição de permutação; é calculado até "
          "200 diferenças não nulas (teste de postos sinalizados) e até 100 observações (Mann–Whitney)."),
    _rule(["[BioStat] Признаки разделения категорий"],
          "Разделение категорий в порядковой регрессии: полное — log L > −10⁻⁶·n; частичное — |β| > 10 "
          "при SE > 100.",
          "Separation of categories in ordinal regression: complete — log L > −10⁻⁶·n; partial — |β| > 10 "
          "with SE > 100.",
          "Separação de categorias na regressão ordinal: completa — log L > −10⁻⁶·n; parcial — |β| > 10 "
          "com EP > 100."),
    _rule(["(эвристика BioStat"],
          "Факторный ANOVA: исключённое из модели взаимодействие с P < 0.10 отмечается как пограничное.",
          "Factorial ANOVA: an interaction left out of the model with P < 0.10 is flagged as borderline.",
          "ANOVA fatorial: uma interação excluída do modelo com P < 0,10 é assinalada como limítrofe."),
    _rule(["[порог BioStat] больше уровней"],
          "ANCOVA: неадекватность не проверяется, если разных x больше 200 (x непрерывный).",
          "ANCOVA: lack of fit is not tested when there are more than 200 distinct x (x is continuous).",
          "ANCOVA: a falta de ajuste não é testada quando há mais de 200 x distintos (x é contínuo)."),
    _rule(["эвристика BioStat: доля перекрытия"],
          "Регрессия по группам: предупреждение, если диапазоны x двух групп перекрываются меньше чем "
          "наполовину.",
          "Regression with groups: a warning when the x ranges of two groups overlap by less than half.",
          "Regressão com grupos: aviso quando as amplitudes de x de dois grupos se sobrepõem menos de "
          "metade."),
]


def biostat_marked_lines() -> list[tuple[str, str]]:
    """Строки ядра с пометкой BioStat ([BioStat], «порог BioStat» в правилах карты и т. п.),
    кроме общих заголовков модулей."""
    lines = []
    for path in sorted((ROOT / "core").rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "BioStat" in line and "эвристики, в книге их нет" not in line:
                lines.append((path.relative_to(ROOT).as_posix(), line.strip()))
    return lines


# --- страница -------------------------------------------------------------------------------

TEXTS: dict[str, dict] = {
    "ru": {
        "title": "Методы и проверка — BioStat",
        "description": "Все методы BioStat: источник в книге, какие таблицы и примеры книг сверяют тесты, "
                       "независимая сверка и что эталоном не проверено.",
        "h1": "Методы и проверка",
        "updated": "Собрано генератором из каталога, карты и тестов; последнее изменение: 21.11.2026.",
        "back": "← Вернуться в BioStat",
        "intro": [
            "В BioStat {n} методов. Для каждого ниже: источник, тесты, которые его вызывают, таблицы и "
            "примеры книг, числа которых сверяют эти тесты, и независимая сверка. Те же данные в машинном "
            "виде — [methods.json](/methods.json).",
            "«кн.» — Welham S. J., Gezan S. A., Clark S. J., Mead A. Statistical Methods in Biology (CRC "
            "Press, 2015), главный источник. «предл.» — метода или его части в Welham нет; указан его "
            "первоисточник или Witte R. S., Witte J. S. Statistics (11-е изд., Wiley, 2017).",
            "Независимая сверка — аудит {audit} ({date}, коммит {commit}): числа пересчитаны другим путём "
            "(scipy, statsmodels, формулы, точный перебор). Это разовая проверка, не автоматический тест. "
            "Все {n} методов в том же аудите прогнаны на граничных случаях (пустые ячейки, константы, смена "
            "единиц, перестановка строк).",
        ],
        "unverified": "Не проверено независимым эталоном",
        "unverified_note": "{k} методов из {n}: в аудите {audit} только граничные случаи. Ссылки на книги "
                           "в тестах у части из них есть — см. карточку метода.",
        "proposed": "Пометка «предл.»",
        "proposed_note": "{k} методов: метод или его часть — не из Welham.",
        "thresholds": "Пороги [BioStat]",
        "thresholds_note": "Пороги подсказок карты и правил методов, которых нет в книгах, — эвристики "
                           "BioStat:",
        "source": "Источник", "tests": "Тестов, вызывающих метод", "welham": "Welham — сверяют тесты",
        "witte": "Witte — сверяют тесты", "external": "Числа других программ в тестах",
        "audit": "Независимая сверка (аудит {audit})", "edge_only": "нет — только граничные случаи",
        "no_refs": "Ссылок на книги в тестах нет", "in_tests": "в тестах — числа {ext}",
        "tag": "предл.",
    },
    "en": {
        "title": "Methods and validation — BioStat",
        "description": "All BioStat methods: book source, which book tables and examples the tests check, "
                       "independent cross-check and what has no reference check.",
        "h1": "Methods and validation",
        "updated": "Built by a generator from the catalogue, the map and the tests; last updated: 2026-11-21.",
        "back": "← Back to BioStat",
        "intro": [
            "BioStat has {n} methods. For each one below: its source, the tests that call it, the book tables "
            "and examples whose numbers those tests check, and the independent cross-check. The same data in "
            "machine-readable form: [methods.json](/methods.json).",
            "“book” — Welham S. J., Gezan S. A., Clark S. J., Mead A. Statistical Methods in Biology (CRC "
            "Press, 2015), the main source. “proposed” — the method or part of it is not in Welham; its "
            "original source or Witte R. S., Witte J. S. Statistics (11th ed., Wiley, 2017) is given.",
            "Independent cross-check — audit {audit} ({date}, commit {commit}): the numbers were recomputed "
            "another way (scipy, statsmodels, formulas, exact enumeration). It was a one-off check, not an "
            "automated test. In the same audit all {n} methods were run on edge cases (empty cells, constants, "
            "change of units, shuffled rows).",
        ],
        "unverified": "Not checked against an independent reference",
        "unverified_note": "{k} of {n} methods: edge cases only in audit {audit}. Some of them do have book "
                           "references in their tests — see the method entry.",
        "proposed": "The “proposed” mark",
        "proposed_note": "{k} methods: the method or part of it does not come from Welham.",
        "thresholds": "[BioStat] thresholds",
        "thresholds_note": "Thresholds of map hints and method rules that are not in the books are BioStat "
                           "heuristics:",
        "source": "Source", "tests": "Tests that call the method", "welham": "Welham — checked by tests",
        "witte": "Witte — checked by tests", "external": "Numbers from other software in the tests",
        "audit": "Independent cross-check (audit {audit})", "edge_only": "none — edge cases only",
        "no_refs": "No book references in the tests", "in_tests": "the tests use numbers from {ext}",
        "tag": "proposed",
    },
    "pt": {
        "title": "Métodos e verificação — BioStat",
        "description": "Todos os métodos do BioStat: fonte no livro, que tabelas e exemplos os testes "
                       "verificam, verificação independente e o que não tem referência.",
        "h1": "Métodos e verificação",
        "updated": "Gerado a partir do catálogo, do mapa e dos testes; última alteração: 21.11.2026.",
        "back": "← Voltar ao BioStat",
        "intro": [
            "O BioStat tem {n} métodos. Para cada um, abaixo: a fonte, os testes que o chamam, as tabelas e "
            "exemplos dos livros cujos números esses testes verificam e a verificação independente. Os mesmos "
            "dados em formato legível por máquina: [methods.json](/methods.json).",
            "«livro» — Welham S. J., Gezan S. A., Clark S. J., Mead A. Statistical Methods in Biology (CRC "
            "Press, 2015), a fonte principal. «proposta» — o método ou parte dele não está no Welham; é "
            "indicada a fonte original ou Witte R. S., Witte J. S. Statistics (11.ª ed., Wiley, 2017).",
            "Verificação independente — auditoria {audit} ({date}, commit {commit}): os números foram "
            "recalculados por outra via (scipy, statsmodels, fórmulas, enumeração exata). Foi uma verificação "
            "pontual, não um teste automático. Na mesma auditoria, todos os {n} métodos foram corridos em "
            "casos-limite (células vazias, constantes, mudança de unidades, linhas baralhadas).",
        ],
        "unverified": "Sem verificação por referência independente",
        "unverified_note": "{k} de {n} métodos: só casos-limite na auditoria {audit}. Alguns têm referências "
                           "aos livros nos testes — ver a ficha do método.",
        "proposed": "A marca «proposta»",
        "proposed_note": "{k} métodos: o método ou parte dele não vem do Welham.",
        "thresholds": "Limiares [BioStat]",
        "thresholds_note": "Os limiares das sugestões do mapa e das regras dos métodos que não estão nos "
                           "livros são heurísticas do BioStat:",
        "source": "Fonte", "tests": "Testes que chamam o método", "welham": "Welham — verificado nos testes",
        "witte": "Witte — verificado nos testes", "external": "Números de outros programas nos testes",
        "audit": "Verificação independente (auditoria {audit})", "edge_only": "nenhuma — só casos-limite",
        "no_refs": "Sem referências aos livros nos testes", "in_tests": "os testes usam números de {ext}",
        "tag": "proposta",
    },
}


def source_words() -> dict[str, list[tuple[str, str]]]:
    """Перевод источников дерева — тот же словарь, что у карты (SensoryMap.tsx)."""
    text = (ROOT / "web" / "src" / "SensoryMap.tsx").read_text(encoding="utf-8")
    block = re.search(r"const SOURCE_WORDS[^=]*= \{(.*?)\n\};", text, re.S).group(1)
    words = {}
    for lang in ("en", "pt"):
        body = re.search(rf"{lang}: \[(.*?)\]\],", block, re.S).group(1)
        words[lang] = re.findall(r'\["([^"]+)", "([^"]+)"\]', body)
    return words


def format_source(source: str, lang: str, words: dict[str, list[tuple[str, str]]]) -> str:
    if lang == "ru":
        return source
    for ru, tr_ in words[lang]:
        source = tr_.join(source.split(ru))
    return source


def _shown(refs: list[str]) -> list[str]:
    """Таблицы, примеры, упражнения; голые страницы — только если других ссылок нет."""
    named = [r for r in refs if not r.startswith("стр. ")]
    return named or refs


def _page_ref(ref: str, lang: str) -> str:
    return ref if lang == "ru" else ref.replace("стр. ", "p. ")


def _date(iso: str, lang: str) -> str:
    """2026-11-12 → 12.11.2026 для ru и pt, как на остальных страницах."""
    return iso if lang == "en" else ".".join(reversed(iso.split("-")))


def _li(items: list[str]) -> str:
    return "      <ul>\n" + "".join(f"        <li>{i}</li>\n" for i in items) + "      </ul>\n"


def method_href(mid: str, lang: str, detailed: set[str]) -> str:
    """Куда ведёт название метода в списке: страница-объяснение, если она есть, иначе карточка."""
    return sp.page_path("methods/" + mid, lang) if mid in detailed else f"#{mid}"


def _overview(data: dict, t: dict, lang: str, detailed: set[str]) -> str:
    """Вводные абзацы и три сводных списка: без эталона, «предл.», пороги [BioStat]."""
    esc, audit, n = html.escape, data["audit"], data["methods_count"]
    fmt = {"n": n, "audit": audit["audit"], "date": _date(audit["date"], lang), "commit": audit["commit"]}
    out = "".join(f"      <p>{_inline(p.format(**fmt))}</p>\n" for p in t["intro"])
    link = lambda m: f'<a href="{method_href(m["id"], lang, detailed)}">{esc(m["name"][lang])}</a>'  # noqa: E731
    bare = [m for m in data["methods"] if not m["audit_reference"]]
    items = [link(m) + (f' — {esc(t["in_tests"].format(ext=", ".join(m["external_reference_in_tests"])))}'
                        if m["external_reference_in_tests"] else "") for m in bare]
    out += f'      <h2>{esc(t["unverified"])}</h2>\n'
    out += f'      <p>{esc(t["unverified_note"].format(k=len(bare), **fmt))}</p>\n{_li(items)}'
    proposed = [m for m in data["methods"] if m["proposed"]]
    out += f'      <h2>{esc(t["proposed"])}</h2>\n'
    out += f'      <p>{esc(t["proposed_note"].format(k=len(proposed)))}</p>\n{_li([link(m) for m in proposed])}'
    out += f'      <h2>{esc(t["thresholds"])}</h2>\n      <p>{esc(t["thresholds_note"])}</p>\n'
    return out + _li([esc(r[lang]) for r in data["biostat_thresholds"]])


def check_facts(m: dict, lang: str) -> list[str]:
    """Факты о проверке метода (HTML-строки): тесты, ссылки на книги, эталоны, аудит 2.

    Тот же список — в карточке списка и в блоке «Источники» страницы метода.
    """
    esc, t = html.escape, TEXTS[lang]
    facts = [f'{esc(t["tests"])}: {m["tests"]}']
    for book in ("welham", "witte"):
        if m["book_refs"][book]:
            refs = ", ".join(_page_ref(r, lang) for r in _shown(m["book_refs"][book]))
            facts.append(f'{esc(t[book])}: {esc(refs)}')
    if not (m["book_refs"]["welham"] or m["book_refs"]["witte"]):
        facts.append(esc(t["no_refs"]))
    if m["external_reference_in_tests"]:
        facts.append(f'{esc(t["external"])}: {esc(", ".join(m["external_reference_in_tests"]))}')
    audit = m["audit_reference"][lang] if m["audit_reference"] else t["edge_only"]
    facts.append(f'{esc(t["audit"].format(audit=AUDIT_ID))}: {esc(audit)}')
    return facts


def _method(m: dict, t: dict, lang: str, words: dict, detailed: set[str]) -> str:
    """Карточка метода: заголовок с якорем (название — ссылка на страницу-объяснение, если она есть)
    и список фактов."""
    esc = html.escape
    tag = f' <span class="tag">{esc(t["tag"])}</span>' if m["proposed"] else ""
    source = f'{esc(t["source"])}: {esc("; ".join(format_source(s, lang, words) for s in m["sources"]))}'
    name = esc(m["name"][lang])
    if m["id"] in detailed:
        name = f'<a href="{method_href(m["id"], lang, detailed)}">{name}</a>'
    return (f'      <h3 id="{m["id"]}">{name}{tag}</h3>\n'
            f'      <p class="muted"><code>{m["id"]}</code></p>\n{_li([source] + check_facts(m, lang))}')


def group_id(group_ru: str) -> str:
    """Якорь группы в списке (/methods/#group-…): по русскому ключу группы каталога."""
    return "group-" + GROUP_IDS[group_ru]


def body(data: dict, lang: str) -> str:
    """Тело страницы на языке lang."""
    import method_pages  # здесь: method_pages сам импортирует этот модуль

    t, words, detailed = TEXTS[lang], source_words(), set(method_pages.available(lang))
    out, group = _overview(data, t, lang, detailed), None
    for m in data["methods"]:
        if m["group"][lang] != group:
            group = m["group"][lang]
            out += f'      <h2 id="{group_id(m["group"]["ru"])}">{html.escape(group)}</h2>\n'
        out += _method(m, t, lang, words, detailed)
    return out


def json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def pages(data: dict | None = None) -> dict[Path, str]:
    """HTML трёх языковых версий и methods.json."""
    data = data or methods_data()
    out = {sp.PUBLIC / sp.page_path("methods", lang).strip("/") / "index.html":
           sp.shell("methods", TEXTS[lang], lang, body(data, lang)) for lang in LANGS}
    out[JSON_OUT] = json_text(data)
    return out


GROUP_IDS = {  # якоря групп; ключи — группы CATALOG (тест сверяет, что все на месте)
    "Описательная статистика": "descriptive", "Сравнение с эталоном": "reference",
    "Сравнение двух групп": "two-groups", "Парные измерения": "paired",
    "Сравнение нескольких групп": "several-groups", "Связь переменных": "relationships",
    "Проверка допущений": "assumptions", "Планирование опыта": "planning",
}
AUDIT_ID = json.loads(REFERENCE.read_text(encoding="utf-8"))["audit"]
