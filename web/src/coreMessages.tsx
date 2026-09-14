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
