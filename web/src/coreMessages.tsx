/**
 * Перевод структурированных сообщений ядра (качество данных,
 * предупреждения методов). Ядро отдаёт {code, params, message},
 * здесь code+params превращаются в переведённую строку.
 *
 * Шаблоны с плейсхолдерами {name}. Если код неизвестен — показываем
 * message (русский фолбэк из ядра).
 */

import { useLang, type Lang } from "./i18n";

export interface CoreMessage {
  code: string;
  message: string;
  params?: Record<string, unknown>;
}

const TEMPLATES: Record<string, Record<Lang, string>> = {
  likely_categorical: {
    ru: `Уникальных значений всего {n_unique} на {n_present} строк — возможно, это закодированная категория. Смени тип, если нужны группы.`,
    en: `Only {n_unique} unique values across {n_present} rows — this may be an encoded category. Change the type if you need groups.`,
    pt: `Apenas {n_unique} valores únicos em {n_present} linhas — pode ser uma categoria codificada. Altere o tipo se precisar de grupos.`,
  },
  likely_id: {
    ru: `Почти все значения уникальны ({n_unique} из {n_present}) — возможно, это идентификатор. Смени тип на ID, чтобы исключить из расчётов.`,
    en: `Almost all values are unique ({n_unique} of {n_present}) — this may be an identifier. Change the type to ID to exclude it from analyses.`,
    pt: `Quase todos os valores são únicos ({n_unique} de {n_present}) — pode ser um identificador. Altere o tipo para ID para o excluir das análises.`,
  },
  id_not_unique: {
    ru: `ID не уникален: {n_duplicates} повторяющихся значений (примеры: {examples}). Проверь целостность данных.`,
    en: `ID is not unique: {n_duplicates} repeated values (examples: {examples}). Check data integrity.`,
    pt: `O ID não é único: {n_duplicates} valores repetidos (exemplos: {examples}). Verifique a integridade dos dados.`,
  },
  numeric_intruders: {
    ru: `В числовой колонке {n_bad} нечисловых значений (примеры: {examples})`,
    en: `The numeric column has {n_bad} non-numeric values (examples: {examples})`,
    pt: `A coluna numérica tem {n_bad} valores não numéricos (exemplos: {examples})`,
  },
  mixed_date_formats: {
    ru: `Смешаны форматы дат в одной колонке: {formats}. Типичный дефект экспорта из Excel.`,
    en: `Mixed date formats in one column: {formats}. A typical Excel export defect.`,
    pt: `Formatos de data misturados numa coluna: {formats}. Um defeito típico da exportação do Excel.`,
  },
  ambiguous_slash_date: {
    ru: `Формат со слэшем неоднозначен: 03/04/2020 — это 3 апреля или 4 марта? Проверь порядок дня и месяца.`,
    en: `Slash date format is ambiguous: is 03/04/2020 April 3 or March 4? Check the day/month order.`,
    pt: `O formato de data com barras é ambíguo: 03/04/2020 é 3 de abril ou 4 de março? Verifique a ordem dia/mês.`,
  },
  mixed_types: {
    ru: `Смешанные типы: {numeric_pct}% значений числовые. Возможно, колонка должна быть числовой.`,
    en: `Mixed types: {numeric_pct}% of values are numeric. The column might be meant as numeric.`,
    pt: `Tipos misturados: {numeric_pct}% dos valores são numéricos. A coluna pode ser suposta numérica.`,
  },
  missing_high: {
    ru: `Более половины значений пропущено ({missing_pct}%)`,
    en: `More than half the values are missing ({missing_pct}%)`,
    pt: `Mais de metade dos valores estão em falta ({missing_pct}%)`,
  },
  missing_notable: {
    ru: `Заметная доля пропусков: {missing_pct}%`,
    en: `Notable share of missing values: {missing_pct}%`,
    pt: `Proporção notável de valores em falta: {missing_pct}%`,
  },
  empty_column: {
    ru: `Колонка пуста`,
    en: `Column is empty`,
    pt: `A coluna está vazia`,
  },
  normality_violated: {
    ru: `Данные не прошли тест на нормальность. Рассмотри непараметрический аналог или трансформацию.`,
    en: `Data failed the normality test. Consider a nonparametric method or a transformation.`,
    pt: `Os dados não passaram no teste de normalidade. Considere um método não paramétrico ou uma transformação.`,
  },
  unequal_variance: {
    ru: `Дисперсии групп неравны, но задан классический t-тест. Рассмотри поправку Уэлча.`,
    en: `Group variances are unequal, but the classic t-test was requested. Consider Welch's correction.`,
    pt: `As variâncias dos grupos são desiguais, mas foi pedido o teste t clássico. Considere a correção de Welch.`,
  },
  variance_heterogeneity: {
    ru: `Дисперсии групп неравны (тест Левене). Рассмотри Уэлча ANOVA или Краскела-Уоллиса.`,
    en: `Group variances are unequal (Levene's test). Consider Welch's ANOVA or Kruskal-Wallis.`,
    pt: `As variâncias dos grupos são desiguais (teste de Levene). Considere a ANOVA de Welch ou Kruskal-Wallis.`,
  },
  autocorrelation: {
    ru: `Возможна автокорреляция остатков (Дарбин-Уотсон далёк от 2). Проверь порядок наблюдений.`,
    en: `Residual autocorrelation is possible (Durbin-Watson far from 2). Check the observation order.`,
    pt: `É possível autocorrelação dos resíduos (Durbin-Watson longe de 2). Verifique a ordem das observações.`,
  },
  missing_dropped: {
    ru: `Часть значений с пропусками удалена перед расчётом.`,
    en: `Some values with missing data were dropped before the calculation.`,
    pt: `Alguns valores com dados em falta foram removidos antes do cálculo.`,
  },
  n_too_small: {
    ru: `Слишком мало наблюдений для этого расчёта.`,
    en: `Too few observations for this calculation.`,
    pt: `Observações insuficientes para este cálculo.`,
  },
  too_few_groups: {
    ru: `Нужно минимум 2 группы.`,
    en: `At least 2 groups are required.`,
    pt: `São necessários pelo menos 2 grupos.`,
  },
  empty: {
    ru: `Нет данных после очистки.`,
    en: `No data after cleaning.`,
    pt: `Sem dados após a limpeza.`,
  },
  empty_group: {
    ru: `Одна из групп пуста.`,
    en: `One of the groups is empty.`,
    pt: `Um dos grupos está vazio.`,
  },
  empty_columns: {
    ru: `Есть колонки без данных.`,
    en: `Some columns have no data.`,
    pt: `Algumas colunas não têm dados.`,
  },
  n_out_of_range: {
    ru: `Число наблюдений вне диапазона применимости теста.`,
    en: `The number of observations is outside the test's valid range.`,
    pt: `O número de observações está fora do intervalo válido do teste.`,
  },
  no_x_variation: {
    ru: `Переменная X не варьирует — наклон регрессии не определён.`,
    en: `The X variable does not vary — the regression slope is undefined.`,
    pt: `A variável X não varia — o declive da regressão é indefinido.`,
  },
  rcbd_duplicate_cells: {
    ru: `Несколько наблюдений на один вариант в блоке. Если это измерения одной делянки (ряды, растения) — выбери анализ с повторными измерениями делянки и укажи колонку делянок.`,
    en: `Several observations per treatment within a block. If these are measurements of one plot (rows, plants), choose the analysis with repeated measurements per plot and select the plot column.`,
    pt: `Várias observações por tratamento dentro de um bloco. Se forem medições da mesma parcela (linhas, plantas), escolha a análise com medições repetidas por parcela e indique a coluna de parcelas.`,
  },
  rcbd_unit_mixed: {
    ru: `В одной делянке встречаются разные варианты — проверь колонки делянок и вариантов.`,
    en: `One plot contains different treatments — check the plot and treatment columns.`,
    pt: `Uma parcela contém tratamentos diferentes — verifique as colunas de parcelas e de tratamentos.`,
  },
  factorial_unbalanced: {
    ru: `Не все комбинации уровней присутствуют с одинаковой повторностью (в каждом блоке): структура неортогональна, такой анализ здесь не поддерживается.`,
    en: `Not all level combinations are present with equal replication (within each block): the structure is non-orthogonal and is not supported here.`,
    pt: `Nem todas as combinações de níveis estão presentes com igual repetição (em cada bloco): a estrutura não é ortogonal e não é suportada aqui.`,
  },
  factor_one_level: {
    ru: `У одного из факторов меньше 2 уровней.`,
    en: `One of the factors has fewer than 2 levels.`,
    pt: `Um dos fatores tem menos de 2 níveis.`,
  },
  nested_no_members: {
    ru: `Ни в одной группе нет двух членов — вложенный член пуст. Для простого сравнения групп выбери однофакторный ANOVA.`,
    en: `No group has two members, so the nested term is empty. To compare groups only, use one-way ANOVA.`,
    pt: `Nenhum grupo tem dois membros, pelo que o termo hierárquico está vazio. Para comparar apenas grupos, use a ANOVA de um fator.`,
  },
  interaction_borderline: {
    ru: `Взаимодействие {term} исключено из предсказания, но P = {p} близко к порогу: сравни его вклад (доля SS) с главными эффектами и реши, можно ли им пренебречь.`,
    en: `The interaction {term} was dropped from prediction, but P = {p} is close to the threshold: compare its contribution (SS share) with the main effects and decide whether it can be ignored.`,
    pt: `A interação {term} foi excluída da previsão, mas P = {p} está perto do limiar: compare a sua contribuição (fração da SS) com os efeitos principais e decida se pode ser ignorada.`,
  },
  no_significant_terms: {
    ru: `Ни один член модели не значим: предсказание — общее среднее.`,
    en: `No model term is significant: the prediction is the overall mean.`,
    pt: `Nenhum termo do modelo é significativo: a previsão é a média geral.`,
  },
  contrast_invalid: {
    ru: `Контраст задан неверно. Формат: «A + C vs B; A vs C» или «F1=0.5, F2=-0.5»; коэффициенты в сумме 0, уровни — как в данных. В факторном анализе контрасты должны быть ортогональны.`,
    en: `Invalid contrast. Use "A + C vs B; A vs C" or "F1=0.5, F2=-0.5"; coefficients must sum to 0 and levels must match the data. In a factorial analysis the contrasts must be orthogonal.`,
    pt: `Contraste inválido. Use «A + C vs B; A vs C» ou «F1=0.5, F2=-0.5»; os coeficientes devem somar 0 e os níveis devem coincidir com os dados. Na análise fatorial os contrastes devem ser ortogonais.`,
  },
  contrasts_not_orthogonal: {
    ru: `Контрасты неортогональны: их суммы квадратов не складываются, вклад каждого зависит от остальных. Остаток посчитан по совместной подгонке.`,
    en: `The contrasts are not orthogonal: their sums of squares do not add up and each one's contribution depends on the others. The remainder uses the joint fit.`,
    pt: `Os contrastes não são ortogonais: as somas de quadrados não se somam e o contributo de cada um depende dos outros. O resto usa o ajustamento conjunto.`,
  },
  levels_not_numeric: {
    ru: `Для полиномиального тренда уровни фактора должны быть числами (дозы, концентрации).`,
    en: `For a polynomial trend the factor levels must be numbers (doses, concentrations).`,
    pt: `Para uma tendência polinomial os níveis do fator devem ser números (doses, concentrações).`,
  },
  comparison_invalid: {
    ru: `Настройки сравнений несовместимы: Даннету нужен контрольный уровень; Тьюки и aLSD — только двусторонние; односторонняя гипотеза — только против контроля; контроль должен быть уровнем из данных.`,
    en: `Incompatible comparison settings: Dunnett needs a control level; Tukey and aLSD are two-sided only; a one-sided alternative needs a control; the control must be a level in the data.`,
    pt: `Definições de comparação incompatíveis: Dunnett requer um nível de controlo; Tukey e aLSD são apenas bilaterais; a hipótese unilateral requer um controlo; o controlo tem de ser um nível dos dados.`,
  },
  latin_square_invalid: {
    ru: `Не латинский квадрат: нужно одинаковое число рядов, колонок и вариантов, и каждый вариант ровно один раз в каждом ряду и каждой колонке.`,
    en: `Not a Latin square: the numbers of rows, columns and treatments must be equal, with each treatment exactly once in every row and every column.`,
    pt: `Não é um quadrado latino: o número de linhas, colunas e tratamentos deve ser igual, com cada tratamento exatamente uma vez em cada linha e coluna.`,
  },
  split_plot_no_units: {
    ru: `Для расщеплённых делянок укажи колонку блоков или колонку целых делянок.`,
    en: `For a split-plot analysis select the block column or the whole-plot column.`,
    pt: `Para parcelas subdivididas indique a coluna de blocos ou a coluna de parcelas principais.`,
  },
  split_plot_invalid: {
    ru: `Не стандартные расщеплённые делянки: в целой делянке должен быть один уровень A и каждый уровень B ровно один раз; у всех уровней A — одинаковое число целых делянок.`,
    en: `Not a standard split-plot: each whole plot must have one level of A and every level of B exactly once; all levels of A need the same number of whole plots.`,
    pt: `Não é um split-plot padrão: cada parcela principal deve ter um nível de A e cada nível de B exatamente uma vez; todos os níveis de A precisam do mesmo número de parcelas principais.`,
  },
  bibd_invalid: {
    ru: `Не BIBD: вариант не должен повторяться в блоке, блоки одного размера, повторность равная и каждая пара вариантов встречается в блоках одинаковое число раз (λ). Для других неполных блоков нужен общий анализ (гл. 11, 16).`,
    en: `Not a BIBD: no treatment may repeat within a block, blocks must be the same size, replication equal and every pair of treatments must share blocks equally often (λ). Other incomplete-block designs need a general analysis (ch. 11, 16).`,
    pt: `Não é um BIBD: nenhum tratamento pode repetir-se num bloco, os blocos devem ter o mesmo tamanho, a repetição deve ser igual e cada par de tratamentos deve partilhar blocos o mesmo número de vezes (λ). Outros delineamentos em blocos incompletos requerem uma análise geral (cap. 11, 16).`,
  },
  bibd_complete_blocks: {
    ru: `Блоки содержат все варианты — это RCBD: используй блочный ANOVA.`,
    en: `Blocks contain all treatments: this is an RCBD, use the randomized block ANOVA.`,
    pt: `Os blocos contêm todos os tratamentos: é um RCBD, use a ANOVA em blocos.`,
  },
  bibd_no_block_residual: {
    ru: `Число блоков равно числу вариантов: в страте блоков нет остаточных df, межблочный F не строится. Объединение внутри- и межблочной информации — смешанная модель (гл. 16).`,
    en: `The number of blocks equals the number of treatments: the block stratum has no residual df, so no between-block F. Combining intra- and inter-block information needs a mixed model (ch. 16).`,
    pt: `O número de blocos é igual ao de tratamentos: o estrato de blocos não tem df residuais, pelo que não há F entre blocos. Combinar a informação intra e interblocos requer um modelo misto (cap. 16).`,
  },
  power_invalid: {
    ru: `Задай разность δ > 0, которую опыт должен обнаружить, и целевую мощность между 0 и 1.`,
    en: `Set a difference δ > 0 that the experiment should detect and a target power between 0 and 1.`,
    pt: `Defina uma diferença δ > 0 que a experiência deve detetar e uma potência pretendida entre 0 e 1.`,
  },
  power_not_reached: {
    ru: `Целевая мощность не достигнута до {n_max} повторностей: уменьши число вариантов, снизь фоновую вариацию или повтори опыт.`,
    en: `The target power is not reached within {n_max} replicates: reduce the number of treatments, lower the background variation or repeat the experiment.`,
    pt: `A potência pretendida não é atingida até {n_max} repetições: reduza o número de tratamentos, diminua a variação de fundo ou repita a experiência.`,
  },
  few_residual_df: {
    ru: `При малом числе повторностей меньше 10 остаточных df — оценка фоновой вариации будет ненадёжной.`,
    en: `With few replicates there are fewer than 10 residual df, so the background variance estimate will be unreliable.`,
    pt: `Com poucas repetições há menos de 10 df residuais, pelo que a estimativa da variação de fundo será pouco fiável.`,
  },
  equivalence_invalid: {
    ru: `Укажи стандарт (уровень или комбинацию уровней, как в данных) и границу эквивалентности c > 0.`,
    en: `Specify the standard (a level or combination of levels as in the data) and an equivalence margin c > 0.`,
    pt: `Indique o padrão (um nível ou combinação de níveis como nos dados) e uma margem de equivalência c > 0.`,
  },
  equivalence_inconclusive: {
    ru: `Ни эквивалентность, ни неэквивалентность не доказаны: опыту не хватает точности для выбранной границы c.`,
    en: `Neither equivalence nor inequivalence is shown: the experiment is not precise enough for the chosen margin c.`,
    pt: `Não se demonstra equivalência nem não equivalência: a experiência não é suficientemente precisa para a margem c escolhida.`,
  },
  lack_of_fit: {
    ru: `Отклонения средних по уровням x от прямой значимы: прямая не описывает зависимость.`,
    en: `Deviations of the group means at each x from the fitted line are significant: a straight line does not describe the relationship.`,
    pt: `Os desvios das médias em cada x em relação à reta são significativos: a reta não descreve a relação.`,
  },
  lack_of_fit_unavailable: {
    ru: `Проверка неадекватности требует модели со свободным членом, не менее 3 уровней x и повторённых значений x.`,
    en: `The lack-of-fit test needs a model with an intercept, at least 3 levels of x and replicated x values.`,
    pt: `O teste de falta de ajustamento requer um modelo com ordenada na origem, pelo menos 3 níveis de x e valores de x repetidos.`,
  },
  extrapolation: {
    ru: `Вне наблюдённого диапазона x [{low}; {high}]: {values} — форма связи там не проверена (экстраполяция).`,
    en: `Outside the observed range of x [{low}; {high}]: {values} — the form of the relationship there is unchecked (extrapolation).`,
    pt: `Fora do intervalo observado de x [{low}; {high}]: {values} — a forma da relação aí não foi verificada (extrapolação).`,
  },
  calibration_no_values: {
    ru: `Укажи новые значения y через «;».`,
    en: `Enter the new y values separated by “;”.`,
    pt: `Indique os novos valores de y separados por “;”.`,
  },
  fieller_undefined: {
    ru: `Наклон не отличается значимо от нуля (g = {g} ≥ 1): пределы Филлера не существуют.`,
    en: `The slope is not significantly different from zero (g = {g} ≥ 1): Fieller limits do not exist.`,
    pt: `O declive não difere significativamente de zero (g = {g} ≥ 1): os limites de Fieller não existem.`,
  },
  deletion_outliers: {
    ru: `Наблюдений с |удалённым остатком| > 2: {count} (при нормальности ожидается около {expected}), строки {rows}. Проверь их по первичным записям; удалять без найденной причины нельзя.`,
    en: `Observations with |deletion residual| > 2: {count} (about {expected} expected under normality), rows {rows}. Check them against the original records; do not remove them without a found cause.`,
    pt: `Observações com |resíduo de eliminação| > 2: {count} (cerca de {expected} esperadas sob normalidade), linhas {rows}. Verifique-as nos registos originais; não as remova sem causa identificada.`,
  },
  influential_points: {
    ru: `Влиятельных наблюдений (модифицированная статистика Кука выше порога): {count}, строки {rows}. Сравни подгонку с ними и без них; удалять без найденной ошибки нельзя.`,
    en: `Influential observations (modified Cook's statistic above the threshold): {count}, rows {rows}. Compare the fit with and without them; do not remove them without a found error.`,
    pt: `Observações influentes (estatística de Cook modificada acima do limiar): {count}, linhas {rows}. Compare o ajustamento com e sem elas; não as remova sem um erro identificado.`,
  },
  nonorthogonal_design: {
    ru: `Факторы неортогональны ({pairs}): суммы квадратов зависят от порядка подгонки — сравни последовательные таблицы и SS типа II.`,
    en: `The factors are non-orthogonal ({pairs}): sums of squares depend on the fitting order — compare the sequential tables and Type II SS.`,
    pt: `Os fatores não são ortogonais ({pairs}): as somas de quadrados dependem da ordem de ajustamento — compare as tabelas sequenciais e as SS de tipo II.`,
  },
  prediction_not_estimable: {
    ru: `Часть комбинаций не оценивается итоговой моделью (пустые ячейки при взаимодействии): их предсказания пропущены.`,
    en: `Some combinations cannot be estimated by the final model (empty cells with an interaction): their predictions are omitted.`,
    pt: `Algumas combinações não são estimáveis pelo modelo final (células vazias com interação): as previsões foram omitidas.`,
  },
  healy_westmacott_approximate: {
    ru: `Таблица Хили–Уэстмакотта приближённая: SS вариантов завышены, SE занижены; при многих пропусках она преувеличивает значимость. Точная таблица — основная.`,
    en: `The Healy–Westmacott table is approximate: treatment SS are inflated and SEs too small; with many missing values it overstates significance. The exact table is the main one.`,
    pt: `A tabela de Healy–Westmacott é aproximada: as SS de tratamentos estão inflacionadas e os EP subestimados; com muitos valores em falta exagera a significância. A tabela exata é a principal.`,
  },
  healy_westmacott_unavailable: {
    ru: `Хили–Уэстмакотт доступен только для полной факторной модели без блока и без пустых ячеек.`,
    en: `Healy–Westmacott is available only for a full factorial model without blocks or empty cells.`,
    pt: `Healy–Westmacott só está disponível para um modelo fatorial completo sem blocos nem células vazias.`,
  },
  rcbd_unequal_subsamples: {
    ru: `Число измерений на делянку разное — для такого (несбалансированного) случая нужны смешанные модели.`,
    en: `The number of measurements per plot differs — this unbalanced case needs mixed models.`,
    pt: `O número de medições por parcela difere — este caso não equilibrado requer modelos mistos.`,
  },
  rcbd_incomplete: {
    ru: `Не каждый вариант встречается ровно один раз в каждом блоке: дизайн неполный или несбалансированный.`,
    en: `Not every treatment appears exactly once in every block: the design is incomplete or unbalanced.`,
    pt: `Nem todos os tratamentos aparecem exatamente uma vez em cada bloco: o delineamento está incompleto ou desequilibrado.`,
  },
  rcbd_too_small: {
    ru: `Нужно не меньше 2 блоков и 2 вариантов.`,
    en: `At least 2 blocks and 2 treatments are required.`,
    pt: `São necessários pelo menos 2 blocos e 2 tratamentos.`,
  },
  levene_undefined: {
    ru: `Тест Левене не определён: в группах слишком мало наблюдений. Равенство дисперсий не проверено.`,
    en: `Levene's test is undefined: the groups have too few observations. Equality of variances was not checked.`,
    pt: `O teste de Levene não está definido: os grupos têm poucas observações. A igualdade de variâncias não foi verificada.`,
  },
  transform_domain_log: {
    ru: `Логарифм определён только для положительных значений. Добавь сдвиг c (предложение по данным: c = {suggested_offset}).`,
    en: `The logarithm is defined only for positive values. Add an offset c (suggested from the data: c = {suggested_offset}).`,
    pt: `O logaritmo só está definido para valores positivos. Adicione um desvio c (sugestão a partir dos dados: c = {suggested_offset}).`,
  },
  transform_domain: {
    ru: `Значения вне области трансформации (√ — неотрицательные; logit — строго между 0 и 1 или 0 и 100). Проверь данные или задай сдвиг c.`,
    en: `Values are outside the transformation's domain (√ needs non-negative values; logit needs values strictly between 0 and 1 or 0 and 100). Check the data or set an offset c.`,
    pt: `Valores fora do domínio da transformação (√ exige valores não negativos; logit exige valores estritamente entre 0 e 1 ou 0 e 100). Verifique os dados ou defina um desvio c.`,
  },
  back_transform_out_of_range: {
    ru: `После обратного преобразования границы интервала вышли за допустимую шкалу (например, отрицательные счёты). Для таких данных лучше обобщённая линейная модель (GLM).`,
    en: `After back-transformation the interval limits fall outside the valid scale (e.g. negative counts). A generalized linear model (GLM) is better suited to such data.`,
    pt: `Após a retransformação, os limites do intervalo saem da escala válida (por exemplo, contagens negativas). Um modelo linear generalizado (GLM) é mais adequado a estes dados.`,
  },
  zero_group_variance: {
    ru: `В одной из групп все значения одинаковы — тест Бартлетта неприменим.`,
    en: `All values in one of the groups are identical — Bartlett's test cannot be applied.`,
    pt: `Todos os valores de um dos grupos são iguais — o teste de Bartlett não se aplica.`,
  },
  bartlett_unequal_variances: {
    ru: `Дисперсии групп различаются (тест Бартлетта). Если дисперсия растёт вместе со средним — попробуй трансформацию отклика; тест чувствителен и к ненормальности.`,
    en: `Group variances differ (Bartlett's test). If the variance grows with the mean, try transforming the response; the test is also sensitive to non-normality.`,
    pt: `As variâncias dos grupos diferem (teste de Bartlett). Se a variância cresce com a média, experimente transformar a resposta; o teste também é sensível à não normalidade.`,
  },
  potential_outliers: {
    ru: `Наблюдений с |стандартизованным остатком| > 2: {count} (при нормальности ожидается около {expected}). Проверь их по первичным записям; удалять без найденной причины нельзя.`,
    en: `Observations with |standardized residual| > 2: {count} (about {expected} expected under normality). Check them against the original records; do not remove them without a found cause.`,
    pt: `Observações com |resíduo padronizado| > 2: {count} (cerca de {expected} esperadas sob normalidade). Verifique-as nos registos originais; não as remova sem causa identificada.`,
  },
  permutation_unequal_variances: {
    ru: `Дисперсии групп неоднородны (тест Левене); перестановочный тест предполагает однородные дисперсии.`,
    en: `Group variances are heterogeneous (Levene's test); the permutation test assumes homogeneous variances.`,
    pt: `As variâncias dos grupos são heterogéneas (teste de Levene); o teste de permutação pressupõe variâncias homogéneas.`,
  },
  no_residual_variation: {
    ru: `Внутри групп нет вариации — стандартные ошибки равны нулю, доверительные интервалы не определены.`,
    en: `There is no variation within groups — standard errors are zero and confidence intervals are undefined.`,
    pt: `Não há variação dentro dos grupos — os erros padrão são nulos e os intervalos de confiança indefinidos.`,
  },
  unadjusted_comparisons: {
    ru: `Попарные p и интервалы разностей без поправки на множественность. Для всех попарных сравнений используй Tukey HSD.`,
    en: `Pairwise p-values and difference intervals are not adjusted for multiplicity. For all pairwise comparisons use Tukey HSD.`,
    pt: `Os valores p e intervalos das diferenças não estão ajustados para multiplicidade. Para todas as comparações par a par use Tukey HSD.`,
  },
  no_variation_corr: {
    ru: `Одна из переменных не варьирует — корреляция не определена.`,
    en: `One of the variables does not vary — the correlation is undefined.`,
    pt: `Uma das variáveis não varia — a correlação é indefinida.`,
  },
  perfect_correlation: {
    ru: `Точки лежат точно на одной прямой (|r| = 1): t-статистика не определена.`,
    en: `The points lie exactly on a straight line (|r| = 1): the t-statistic is undefined.`,
    pt: `Os pontos estão exatamente numa reta (|r| = 1): a estatística t é indefinida.`,
  },
  correlation_linear_only: {
    ru: `r измеряет только линейную связь: r ≈ 0 не исключает нелинейной зависимости, а корреляция не означает причинности. Смотри диаграмму рассеяния.`,
    en: `r measures only linear association: r ≈ 0 does not rule out a nonlinear relationship, and correlation does not imply causation. Check the scatter plot.`,
    pt: `r mede apenas associação linear: r ≈ 0 não exclui uma relação não linear, e correlação não implica causalidade. Veja o diagrama de dispersão.`,
  },
  file_empty: {
    ru: `Файл пустой — в нём нет данных для анализа.`,
    en: `The file is empty — there is no data to analyze.`,
    pt: `O ficheiro está vazio — não há dados para analisar.`,
  },
  no_data_rows: {
    ru: `В файле есть заголовок, но нет строк с данными. Проверьте, что файл содержит записи под названиями колонок.`,
    en: `The file has a header but no data rows. Check that it contains records under the column names.`,
    pt: `O ficheiro tem cabeçalho mas nenhuma linha de dados. Verifique se contém registos sob os nomes das colunas.`,
  },
  no_columns: {
    ru: `Не удалось распознать колонки. Возможно, это не таблица CSV — проверьте формат файла.`,
    en: `Could not detect columns. This may not be a CSV table — check the file format.`,
    pt: `Não foi possível detetar colunas. Pode não ser uma tabela CSV — verifique o formato do ficheiro.`,
  },
  no_numeric: {
    ru: `В таблице нет числовых колонок. Для статистики нужна хотя бы одна колонка с числами. Если числа распознались как текст — измените тип колонки в разделе «Детали по колонкам».`,
    en: `The table has no numeric columns. Statistics need at least one column of numbers. If numbers were read as text, change the column type under "Column details".`,
    pt: `A tabela não tem colunas numéricas. A estatística precisa de pelo menos uma coluna de números. Se os números foram lidos como texto, altere o tipo da coluna em "Detalhes das colunas".`,
  },
  low_encoding_confidence: {
    ru: `Кодировка определена с низкой уверенностью — проверьте предпросмотр.`,
    en: `Encoding detected with low confidence — check the preview.`,
    pt: `Codificação detetada com baixa confiança — verifique a pré-visualização.`,
  },
  unknown_encoding_utf8: {
    ru: `Неизвестная кодировка, использована UTF-8.`,
    en: `Unknown encoding, UTF-8 was used.`,
    pt: `Codificação desconhecida, foi usado UTF-8.`,
  },
  unreadable_chars: {
    ru: `В тексте есть нечитаемые символы — вероятно, кодировка неверна.`,
    en: `The text has unreadable characters — the encoding is probably wrong.`,
    pt: `O texto tem caracteres ilegíveis — a codificação está provavelmente errada.`,
  },
  unstable_delimiter: {
    ru: `Разделитель нестабилен по строкам — проверьте предпросмотр.`,
    en: `The delimiter is inconsistent across rows — check the preview.`,
    pt: `O separador é inconsistente entre linhas — verifique a pré-visualização.`,
  },
  too_many_rows: {
    ru: `Строк {n_rows} — для браузера это много. Профилирование сделано по первым {limit}.`,
    en: `{n_rows} rows — that's a lot for the browser. Profiling used the first {limit}.`,
    pt: `{n_rows} linhas — é muito para o navegador. A análise usou as primeiras {limit}.`,
  },
};

/** Подставить {name} из params в шаблон. */
function fill(template: string, params: Record<string, unknown>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) =>
    key in params ? String(params[key]) : `{${key}}`,
  );
}

/** Перевести одно сообщение ядра на указанный язык. */
export function translateMessage(msg: CoreMessage, lang: Lang): string {
  const template = TEMPLATES[msg.code]?.[lang];
  if (!template) return msg.message; // неизвестный код — русский фолбэк
  return fill(template, msg.params ?? {});
}

/** Хук: функция перевода сообщений ядра на текущем языке. */
export function useCoreMessage(): (msg: CoreMessage) => string {
  const lang = useLang();
  return (msg: CoreMessage) => translateMessage(msg, lang);
}
