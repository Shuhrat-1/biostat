"""Переводы строк каталога методов (EN, PT).

Русские строки в catalog.py служат ключами по умолчанию. Здесь — их
переводы. Функция translate подставляет нужный язык при сериализации;
отсутствующий перевод откатывается к русскому ключу.
"""

from __future__ import annotations

# Группы методов.
GROUPS: dict[str, dict[str, str]] = {
    "en": {
        "Описательная статистика": "Descriptive statistics",
        "Сравнение с эталоном": "Comparison to a reference",
        "Сравнение двух групп": "Two-group comparison",
        "Сравнение нескольких групп": "Multi-group comparison",
        "Парные измерения": "Paired measurements",
        "Связь переменных": "Relationship between variables",
        "Проверка допущений": "Assumption checks",
    },
    "pt": {
        "Описательная статистика": "Estatística descritiva",
        "Сравнение с эталоном": "Comparação com referência",
        "Сравнение двух групп": "Comparação de dois grupos",
        "Сравнение нескольких групп": "Comparação de vários grupos",
        "Парные измерения": "Medições emparelhadas",
        "Связь переменных": "Relação entre variáveis",
        "Проверка допущений": "Verificação de pressupostos",
    },
}

# Названия методов.
LABELS: dict[str, dict[str, str]] = {
    "en": {
        "Описательная статистика": "Descriptive statistics",
        "Доверительный интервал среднего": "Confidence interval of the mean",
        "Проверка нормальности (Шапиро-Уилк)": "Normality test (Shapiro-Wilk)",
        "Одновыборочный t-тест": "One-sample t-test",
        "Двухвыборочный t-тест": "Two-sample t-test",
        "Тест Манна-Уитни (непараметрический)": "Mann-Whitney test (nonparametric)",
        "Однофакторный ANOVA": "One-way ANOVA",
        "Парный t-тест": "Paired t-test",
        "Тест Уилкоксона (непараметрический)": "Wilcoxon test (nonparametric)",
        "Пост-хок Tukey HSD": "Post-hoc Tukey HSD",
        "Тест Краскела-Уоллиса (непараметрический)": "Kruskal-Wallis test (nonparametric)",
        "Простая линейная регрессия": "Simple linear regression",
    },
    "pt": {
        "Описательная статистика": "Estatística descritiva",
        "Доверительный интервал среднего": "Intervalo de confiança da média",
        "Проверка нормальности (Шапиро-Уилк)": "Teste de normalidade (Shapiro-Wilk)",
        "Одновыборочный t-тест": "Teste t para uma amostra",
        "Двухвыборочный t-тест": "Teste t para duas amostras",
        "Тест Манна-Уитни (непараметрический)": "Teste de Mann-Whitney (não paramétrico)",
        "Однофакторный ANOVA": "ANOVA de um fator",
        "Парный t-тест": "Teste t emparelhado",
        "Тест Уилкоксона (непараметрический)": "Teste de Wilcoxon (não paramétrico)",
        "Пост-хок Tukey HSD": "Pós-hoc Tukey HSD",
        "Тест Краскела-Уоллиса (непараметрический)": "Teste de Kruskal-Wallis (não paramétrico)",
        "Простая линейная регрессия": "Regressão linear simples",
    },
}

# Описания методов.
DESCRIPTIONS: dict[str, dict[str, str]] = {
    "en": {
        "Центр, разброс, форма, квартили.": "Center, spread, shape, quartiles.",
        "Интервальная оценка среднего по t-распределению.": (
            "Interval estimate of the mean via the t-distribution."
        ),
        "Проверяет, согласуются ли данные с нормальным распределением.": (
            "Tests whether data are consistent with a normal distribution."
        ),
        "Сравнивает среднее выборки с заданным значением.": (
            "Compares the sample mean to a given value."
        ),
        "Сравнивает средние двух групп (авто-выбор Стьюдент/Уэлч).": (
            "Compares the means of two groups (auto Student/Welch)."
        ),
        "Сравнение двух групп без предположения о нормальности.": (
            "Compares two groups without assuming normality."
        ),
        "Сравнивает средние нескольких групп.": "Compares the means of several groups.",
        "Сравнивает связанные измерения (до/после) на одних объектах.": "Compares paired measurements (before/after) on the same subjects.",
        "Парное сравнение без предположения о нормальности разностей.": "Paired comparison without assuming normality of differences.",
        "Все попарные сравнения групп после значимого ANOVA.": (
            "All pairwise group comparisons after a significant ANOVA."
        ),
        "Сравнение нескольких групп без предположения о нормальности.": (
            "Compares several groups without assuming normality."
        ),
        "Модель y = b0 + b1·x с оценками и доверительными интервалами.": (
            "Model y = b0 + b1·x with estimates and confidence intervals."
        ),
    },
    "pt": {
        "Центр, разброс, форма, квартили.": "Centro, dispersão, forma, quartis.",
        "Интервальная оценка среднего по t-распределению.": (
            "Estimativa por intervalo da média via distribuição t."
        ),
        "Проверяет, согласуются ли данные с нормальным распределением.": (
            "Verifica se os dados são consistentes com uma distribuição normal."
        ),
        "Сравнивает среднее выборки с заданным значением.": (
            "Compara a média da amostra com um valor dado."
        ),
        "Сравнивает средние двух групп (авто-выбор Стьюдент/Уэлч).": (
            "Compara as médias de dois grupos (Student/Welch automático)."
        ),
        "Сравнение двух групп без предположения о нормальности.": (
            "Compara dois grupos sem assumir normalidade."
        ),
        "Сравнивает средние нескольких групп.": "Compara as médias de vários grupos.",
        "Сравнивает связанные измерения (до/после) на одних объектах.": "Compara medições emparelhadas (antes/depois) nos mesmos sujeitos.",
        "Парное сравнение без предположения о нормальности разностей.": "Comparação emparelhada sem assumir normalidade das diferenças.",
        "Все попарные сравнения групп после значимого ANOVA.": (
            "Todas as comparações par a par após uma ANOVA significativa."
        ),
        "Сравнение нескольких групп без предположения о нормальности.": (
            "Compara vários grupos sem assumir normalidade."
        ),
        "Модель y = b0 + b1·x с оценками и доверительными интервалами.": (
            "Modelo y = b0 + b1·x com estimativas e intervalos de confiança."
        ),
    },
}

# Метки слотов и параметров.
LABELS_MISC: dict[str, dict[str, str]] = {
    "en": {
        "Числовая колонка": "Numeric column",
        "Колонка групп": "Grouping column",
        "Колонка «до»": "Before column",
        "Колонка «после»": "After column",
        "X (независимая)": "X (independent)",
        "Y (зависимая)": "Y (dependent)",
        "Гипотеза": "Alternative",
        "Уровень доверия": "Confidence level",
        "Эталонное среднее μ": "Reference mean μ",
        "Уровень значимости α": "Significance level α",
    },
    "pt": {
        "Числовая колонка": "Coluna numérica",
        "Колонка групп": "Coluna de grupos",
        "Колонка «до»": "Coluna «antes»",
        "Колонка «после»": "Coluna «depois»",
        "X (независимая)": "X (independente)",
        "Y (зависимая)": "Y (dependente)",
        "Гипотеза": "Hipótese",
        "Уровень доверия": "Nível de confiança",
        "Эталонное среднее μ": "Média de referência μ",
        "Уровень значимости α": "Nível de significância α",
    },
}


def tr(table: dict[str, dict[str, str]], text: str, lang: str) -> str:
    """Перевести строку через таблицу; откат к исходному русскому ключу."""
    if lang == "ru":
        return text
    return table.get(lang, {}).get(text, text)
