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
        "Средние групп с SED и LSD": "Group means with SED and LSD",
        "Перестановочный F-тест (ANOVA)": "Permutation F-test (ANOVA)",
        "Тест Бартлетта (равенство дисперсий)": "Bartlett's test (equal variances)",
        "Остатки ANOVA и выбросы": "ANOVA residuals and outliers",
        "ANOVA с трансформацией отклика": "ANOVA on a transformed response",
        "Блочный ANOVA (RCBD)": "Randomized block ANOVA (RCBD)",
        "Блочный ANOVA с повторными измерениями делянки": "Block ANOVA with repeated measurements per plot",
        "Факторный ANOVA (A×B, A×B×C)": "Factorial ANOVA (A×B, A×B×C)",
        "Вложенный ANOVA (группа / член, контроль)": "Nested ANOVA (group / member, controls)",
        "Контрасты вариантов": "Treatment contrasts",
        "Полиномиальный тренд по уровням": "Polynomial trend across levels",
        "Сравнения средних с поправкой (Тьюки, Даннет, FDR)": "Adjusted mean comparisons (Tukey, Dunnett, FDR)",
        "Латинский квадрат": "Latin square",
        "Расщеплённые делянки (split-plot)": "Split-plot design",
        "Неполные блоки (BIBD)": "Incomplete blocks (BIBD)",
        "Повторность и мощность (по пилоту)": "Replication and power (from a pilot)",
        "Тест эквивалентности (TOST)": "Equivalence test (TOST)",
        "Неортогональный ANOVA (неравная повторность)": "Non-orthogonal ANOVA (unequal replication)",
        "Простая линейная регрессия": "Simple linear regression",
        "Регрессия: ANOVA, тесты, предсказания": "Regression: ANOVA, tests, predictions",
        "Калибровка (обратное предсказание)": "Calibration (inverse prediction)",
        "Корреляция Пирсона": "Pearson correlation",
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
        "Средние групп с SED и LSD": "Médias dos grupos com SED e LSD",
        "Перестановочный F-тест (ANOVA)": "Teste F de permutação (ANOVA)",
        "Тест Бартлетта (равенство дисперсий)": "Teste de Bartlett (igualdade de variâncias)",
        "Остатки ANOVA и выбросы": "Resíduos da ANOVA e valores atípicos",
        "ANOVA с трансформацией отклика": "ANOVA com resposta transformada",
        "Блочный ANOVA (RCBD)": "ANOVA em blocos casualizados (RCBD)",
        "Блочный ANOVA с повторными измерениями делянки": "ANOVA em blocos com medições repetidas por parcela",
        "Факторный ANOVA (A×B, A×B×C)": "ANOVA fatorial (A×B, A×B×C)",
        "Вложенный ANOVA (группа / член, контроль)": "ANOVA hierárquica (grupo / membro, controlos)",
        "Контрасты вариантов": "Contrastes de tratamentos",
        "Полиномиальный тренд по уровням": "Tendência polinomial entre níveis",
        "Сравнения средних с поправкой (Тьюки, Даннет, FDR)": "Comparações de médias com correção (Tukey, Dunnett, FDR)",
        "Латинский квадрат": "Quadrado latino",
        "Расщеплённые делянки (split-plot)": "Parcelas subdivididas (split-plot)",
        "Неполные блоки (BIBD)": "Blocos incompletos (BIBD)",
        "Повторность и мощность (по пилоту)": "Repetição e potência (a partir de um piloto)",
        "Тест эквивалентности (TOST)": "Teste de equivalência (TOST)",
        "Неортогональный ANOVA (неравная повторность)": "ANOVA não ortogonal (repetição desigual)",
        "Простая линейная регрессия": "Regressão linear simples",
        "Регрессия: ANOVA, тесты, предсказания": "Regressão: ANOVA, testes, previsões",
        "Калибровка (обратное предсказание)": "Calibração (previsão inversa)",
        "Корреляция Пирсона": "Correlação de Pearson",
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
        "Средние с ДИ по общей дисперсии ANOVA, SED, LSD и ДИ разностей.": (
            "Means with CIs from the pooled ANOVA variance, SED, LSD and CIs of differences."
        ),
        "F-тест без предположения о нормальности: p по случайным перестановкам.": (
            "F-test without assuming normality: p from random permutations."
        ),
        "Проверяет, равны ли дисперсии групп перед ANOVA.": (
            "Tests whether group variances are equal before ANOVA."
        ),
        "Стандартизованные остатки, графики допущений и кандидаты в выбросы.": (
            "Standardized residuals, assumption plots and outlier candidates."
        ),
        "ANOVA на шкале log, √ или logit; средние и ДИ переведены обратно.": (
            "ANOVA on the log, √ or logit scale; means and CIs back-transformed."
        ),
        "Каждый вариант по одному разу в каждом блоке; блоки отделяются от фона.": (
            "Each treatment once in every block; block differences are separated from background."
        ),
        "Несколько измерений на делянку (ряды, растения): варианты против разброса делянок.": (
            "Several measurements per plot (rows, plants): treatments tested against plot-to-plot variation."
        ),
        "Главные эффекты и взаимодействия; средние по членам, отобранным по маргинальности.": (
            "Main effects and interactions; means for the terms selected by marginality."
        ),
        "Различия между группами и между членами внутри групп (контроль против остальных).": (
            "Differences between groups and between members within groups (control vs the rest)."
        ),
        "Заранее заданные сравнения групп вариантов (A + C vs B) с разбиением SS вариантов.": "Pre-planned comparisons of treatment sets (A + C vs B) partitioning the treatment SS.",
        "Линейный и квадратичный тренд по количественным уровням и проверка остатка.": "Linear and quadratic trend across quantitative levels, with a lack-of-fit test.",
        "Все пары или каждый вариант против контроля с контролем ошибки I рода.": "All pairs or each treatment against a control, with Type I error control.",
        "Каждый вариант раз в каждом ряду и колонке: два перекрёстных блока.": "Each treatment once in every row and column: two crossed blocking factors.",
        "A на крупных делянках, B на их частях: у A и B разные ошибки.": "A on whole plots, B on subplots: A and B are tested against different errors.",
        "Блок меньше числа вариантов: внутриблочные средние, эффективность EF.": "Blocks smaller than the number of treatments: intra-block means and efficiency factor.",
        "Сколько повторностей нужно, чтобы обнаружить разность δ с нужной мощностью.": "How many replicates are needed to detect a difference δ with the target power.",
        "Доказать, что варианты не отличаются от стандарта больше чем на c.": "Show that treatments differ from the standard by no more than c.",
        "Факторы с неравной повторностью или пустыми ячейками: порядок подгонки, отбор модели.": "Factors with unequal replication or empty cells: fitting order and model selection.",
        "Таблица ANOVA, t-тесты параметров, ДИ предсказаний, R², неадекватность.": "ANOVA table, parameter t-tests, prediction intervals, R², lack of fit.",
        "Оценить x по новому измерению y с пределами Филлера.": "Estimate x from a new measurement of y with Fieller limits.",
        "Модель y = b0 + b1·x с оценками и доверительными интервалами.": (
            "Model y = b0 + b1·x with estimates and confidence intervals."
        ),
        "Сила и направление линейной связи двух переменных (r) и тест ρ = 0.": (
            "Strength and direction of the linear relationship between two variables (r) and a test of ρ = 0."
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
        "Средние с ДИ по общей дисперсии ANOVA, SED, LSD и ДИ разностей.": (
            "Médias com IC da variância conjunta da ANOVA, SED, LSD e IC das diferenças."
        ),
        "F-тест без предположения о нормальности: p по случайным перестановкам.": (
            "Teste F sem assumir normalidade: p a partir de permutações aleatórias."
        ),
        "Проверяет, равны ли дисперсии групп перед ANOVA.": (
            "Verifica se as variâncias dos grupos são iguais antes da ANOVA."
        ),
        "Стандартизованные остатки, графики допущений и кандидаты в выбросы.": (
            "Resíduos padronizados, gráficos de pressupostos e candidatos a valores atípicos."
        ),
        "ANOVA на шкале log, √ или logit; средние и ДИ переведены обратно.": (
            "ANOVA na escala log, √ ou logit; médias e IC retransformados."
        ),
        "Каждый вариант по одному разу в каждом блоке; блоки отделяются от фона.": (
            "Cada tratamento uma vez em cada bloco; as diferenças entre blocos são separadas do ruído."
        ),
        "Несколько измерений на делянку (ряды, растения): варианты против разброса делянок.": (
            "Várias medições por parcela (linhas, plantas): tratamentos testados contra a variação entre parcelas."
        ),
        "Главные эффекты и взаимодействия; средние по членам, отобранным по маргинальности.": (
            "Efeitos principais e interações; médias dos termos selecionados pela marginalidade."
        ),
        "Различия между группами и между членами внутри групп (контроль против остальных).": (
            "Diferenças entre grupos e entre membros dentro dos grupos (controlo contra os restantes)."
        ),
        "Заранее заданные сравнения групп вариантов (A + C vs B) с разбиением SS вариантов.": "Comparações planeadas de conjuntos de tratamentos (A + C vs B), com partição da SS de tratamentos.",
        "Линейный и квадратичный тренд по количественным уровням и проверка остатка.": "Tendência linear e quadrática entre níveis quantitativos, com teste da falta de ajuste.",
        "Все пары или каждый вариант против контроля с контролем ошибки I рода.": "Todos os pares ou cada tratamento contra um controlo, com controlo do erro de tipo I.",
        "Каждый вариант раз в каждом ряду и колонке: два перекрёстных блока.": "Cada tratamento uma vez em cada linha e coluna: dois fatores de bloqueio cruzados.",
        "A на крупных делянках, B на их частях: у A и B разные ошибки.": "A nas parcelas principais, B nas subparcelas: A e B são testados contra erros diferentes.",
        "Блок меньше числа вариантов: внутриблочные средние, эффективность EF.": "Blocos menores que o número de tratamentos: médias intrabloco e fator de eficiência.",
        "Сколько повторностей нужно, чтобы обнаружить разность δ с нужной мощностью.": "Quantas repetições são necessárias para detetar uma diferença δ com a potência pretendida.",
        "Доказать, что варианты не отличаются от стандарта больше чем на c.": "Mostrar que os tratamentos não diferem do padrão mais do que c.",
        "Факторы с неравной повторностью или пустыми ячейками: порядок подгонки, отбор модели.": "Fatores com repetição desigual ou células vazias: ordem de ajustamento e seleção do modelo.",
        "Таблица ANOVA, t-тесты параметров, ДИ предсказаний, R², неадекватность.": "Tabela ANOVA, testes t dos parâmetros, intervalos de previsão, R², falta de ajustamento.",
        "Оценить x по новому измерению y с пределами Филлера.": "Estimar x a partir de uma nova medição de y com limites de Fieller.",
        "Модель y = b0 + b1·x с оценками и доверительными интервалами.": (
            "Modelo y = b0 + b1·x com estimativas e intervalos de confiança."
        ),
        "Сила и направление линейной связи двух переменных (r) и тест ρ = 0.": (
            "Força e direção da relação linear entre duas variáveis (r) e teste de ρ = 0."
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
        "Колонка вариантов": "Treatment column",
        "Колонка блоков": "Block column",
        "Колонка делянок": "Plot column",
        "Фактор A": "Factor A",
        "Фактор B": "Factor B",
        "Фактор C (необязательно)": "Factor C (optional)",
        "Колонка блоков (необязательно)": "Block column (optional)",
        "Колонка членов группы": "Group member column",
        "Контрасты (A + C vs B; A vs C)": "Contrasts (A + C vs B; A vs C)",
        "Поправка на множественность": "Multiplicity adjustment",
        "Степень полинома": "Polynomial degree",
        "Процедура": "Procedure",
        "Контрольный уровень (для сравнений с контролем)": "Control level (for comparisons with a control)",
        "Колонка уровней (дозы, числа)": "Level column (doses, numbers)",
        "Колонка вариантов (фактор A)": "Treatment column (factor A)",
        "Колонка рядов": "Row column",
        "Колонка колонок (столбцов)": "Column (position) column",
        "Фактор B (необязательно)": "Factor B (optional)",
        "Фактор целых делянок (A)": "Whole-plot factor (A)",
        "Фактор субделянок (B)": "Subplot factor (B)",
        "Колонка целых делянок (необязательно)": "Whole-plot column (optional)",
        "Разность, которую нужно обнаружить (δ)": "Difference to detect (δ)",
        "Целевая мощность": "Target power",
        "Максимум повторностей": "Maximum replication",
        "Фоновая дисперсия s² (0 — из данных)": "Background variance s² (0 = from data)",
        "Стандарт (уровень или комбинация)": "Standard (level or combination)",
        "Граница эквивалентности c": "Equivalence margin c",
        "Граница в единицах": "Margin units",
        "Посторонний фактор (необязательно)": "Extraneous factor (optional)",
        "Модель": "Model",
        "Сравнить с Хили–Уэстмакоттом": "Compare with Healy–Westmacott",
        "Переменная X": "Variable X",
        "Переменная Y": "Variable Y",
        "Гипотеза": "Alternative",
        "Уровень доверия": "Confidence level",
        "Эталонное среднее μ": "Reference mean μ",
        "Уровень значимости α": "Significance level α",
        "Число перестановок": "Number of permutations",
        "Трансформация": "Transformation",
        "Сдвиг c": "Offset c",
        "X (объясняющая)": "X (explanatory)",
        "Y (отклик)": "Y (response)",
        "Через начало координат (α = 0)": "Through the origin (α = 0)",
        "Проверяемое значение наклона": "Slope value under H0",
        "Проверяемое значение α": "Intercept value under H0",
        "Точки предсказания x (через ;)": "Prediction points x (separated by ;)",
        "Параметризация": "Parameterization",
        "Проверить неадекватность (повторы x)": "Test lack of fit (replicated x)",
        "X (целевая величина)": "X (target variable)",
        "Y (замена, измерение)": "Y (substitute measurement)",
        "Новые значения y (через ;)": "New y values (separated by ;)",
        "Повторов на новый образец r": "Replicates per new sample r",
    },
    "pt": {
        "Числовая колонка": "Coluna numérica",
        "Колонка групп": "Coluna de grupos",
        "Колонка «до»": "Coluna «antes»",
        "Колонка «после»": "Coluna «depois»",
        "X (независимая)": "X (independente)",
        "Y (зависимая)": "Y (dependente)",
        "Колонка вариантов": "Coluna de tratamentos",
        "Колонка блоков": "Coluna de blocos",
        "Колонка делянок": "Coluna de parcelas",
        "Фактор A": "Fator A",
        "Фактор B": "Fator B",
        "Фактор C (необязательно)": "Fator C (opcional)",
        "Колонка блоков (необязательно)": "Coluna de blocos (opcional)",
        "Колонка членов группы": "Coluna de membros do grupo",
        "Контрасты (A + C vs B; A vs C)": "Contrastes (A + C vs B; A vs C)",
        "Поправка на множественность": "Correção para multiplicidade",
        "Степень полинома": "Grau do polinómio",
        "Процедура": "Procedimento",
        "Контрольный уровень (для сравнений с контролем)": "Nível de controlo (para comparações com controlo)",
        "Колонка уровней (дозы, числа)": "Coluna de níveis (doses, números)",
        "Колонка вариантов (фактор A)": "Coluna de tratamentos (fator A)",
        "Колонка рядов": "Coluna de linhas",
        "Колонка колонок (столбцов)": "Coluna de colunas",
        "Фактор B (необязательно)": "Fator B (opcional)",
        "Фактор целых делянок (A)": "Fator das parcelas principais (A)",
        "Фактор субделянок (B)": "Fator das subparcelas (B)",
        "Колонка целых делянок (необязательно)": "Coluna de parcelas principais (opcional)",
        "Разность, которую нужно обнаружить (δ)": "Diferença a detetar (δ)",
        "Целевая мощность": "Potência pretendida",
        "Максимум повторностей": "Repetição máxima",
        "Фоновая дисперсия s² (0 — из данных)": "Variância residual s² (0 = dos dados)",
        "Стандарт (уровень или комбинация)": "Padrão (nível ou combinação)",
        "Граница эквивалентности c": "Margem de equivalência c",
        "Граница в единицах": "Unidades da margem",
        "Посторонний фактор (необязательно)": "Fator estranho (opcional)",
        "Модель": "Modelo",
        "Сравнить с Хили–Уэстмакоттом": "Comparar com Healy–Westmacott",
        "Переменная X": "Variável X",
        "Переменная Y": "Variável Y",
        "Гипотеза": "Hipótese",
        "Уровень доверия": "Nível de confiança",
        "Эталонное среднее μ": "Média de referência μ",
        "Уровень значимости α": "Nível de significância α",
        "Число перестановок": "Número de permutações",
        "Трансформация": "Transformação",
        "Сдвиг c": "Desvio c",
        "X (объясняющая)": "X (explicativa)",
        "Y (отклик)": "Y (resposta)",
        "Через начало координат (α = 0)": "Pela origem (α = 0)",
        "Проверяемое значение наклона": "Valor do declive sob H0",
        "Проверяемое значение α": "Valor da ordenada sob H0",
        "Точки предсказания x (через ;)": "Pontos de previsão x (separados por ;)",
        "Параметризация": "Parametrização",
        "Проверить неадекватность (повторы x)": "Testar falta de ajustamento (x repetidos)",
        "X (целевая величина)": "X (variável alvo)",
        "Y (замена, измерение)": "Y (medição substituta)",
        "Новые значения y (через ;)": "Novos valores de y (separados por ;)",
        "Повторов на новый образец r": "Repetições por nova amostra r",
    },
}


def tr(table: dict[str, dict[str, str]], text: str, lang: str) -> str:
    """Перевести строку через таблицу; откат к исходному русскому ключу."""
    if lang == "ru":
        return text
    return table.get(lang, {}).get(text, text)
