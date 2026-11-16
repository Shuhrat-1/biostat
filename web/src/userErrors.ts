/** Сырые ошибки загрузки и разбора → понятный текст на языке интерфейса (аудит 03-06). */

/** Префикс ошибки Worker'а: пакет Python не загрузился или не прошёл проверку sha256. */
export const PACKAGES_FAILED = "packages_failed:";

const KNOWN: Array<[RegExp, string]> = [
  [/^packages_failed:/, "error.packages"],
  [/field larger than field limit/, "error.cellTooBig"],
  [/MemoryError|out of memory|Array buffer allocation failed/i, "error.fileTooBig"],
];

/** Ключ перевода для известной ошибки; null — ошибку показываем как есть. */
export function errorKey(message: string): string | null {
  const hit = KNOWN.find(([pattern]) => pattern.test(message));
  return hit ? hit[1] : null;
}

/** Текст ошибки для пользователя: перевод известной, иначе исходное сообщение. */
export function explainError(error: unknown, t: (key: string) => string): string {
  const message = error instanceof Error ? error.message : String(error);
  const key = errorKey(message);
  if (!key) return message;
  const names = message.startsWith(PACKAGES_FAILED) ? message.slice(PACKAGES_FAILED.length) : "";
  return t(key).replace("{names}", names);
}
