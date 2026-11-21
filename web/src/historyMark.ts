/**
 * Пометка записи истории «здесь приложение» для «← Вернуться в BioStat» на статичных
 * страницах (web/public/back.js). Адрес записи оттуда не виден — сайт отдаёт
 * Referrer-Policy: no-referrer, и Navigation API прячет url чужих записей, — а
 * состояние записи видно. В пометке нет данных пользователя; без Navigation API
 * (Firefox, Safari) ничего не делается.
 */

export const APP_HISTORY_STATE = { biostat: "app" } as const;

type NavigationLike = { updateCurrentEntry?: (options: { state: unknown }) => void };

export function markAppHistoryEntry(win: { navigation?: NavigationLike } = window as never): boolean {
  try {
    if (typeof win.navigation?.updateCurrentEntry !== "function") return false;
    win.navigation.updateCurrentEntry({ state: APP_HISTORY_STATE });
    return true;
  } catch {
    return false;
  }
}
