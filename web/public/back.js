// «← Вернуться в BioStat» на статичных страницах: если в этой вкладке уже открыто
// приложение, вернуться к нему по истории — браузер восстановит файл и расчёт из
// кеша «назад/вперёд», — а не открыть его заново. Запись приложения узнаётся по
// состоянию { biostat: "app" } (src/historyMark.ts): url чужих записей Navigation API
// прячет из-за Referrer-Policy: no-referrer. Navigation API есть не во всех
// браузерах; без него, при Ctrl/Cmd/Shift-клике и когда приложения в истории нет,
// ссылка работает как обычная. Ничего не хранит и никуда не обращается.
(() => {
  const link = document.querySelector("a[data-back]");
  const nav = window.navigation;
  if (!link || !nav || typeof nav.entries !== "function" || !nav.currentEntry) return;
  link.addEventListener("click", (event) => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const entries = nav.entries();
    for (let i = nav.currentEntry.index - 1; i >= 0; i -= 1) {
      const entry = entries[i];
      const state = entry && typeof entry.getState === "function" ? entry.getState() : null;
      if (state && state.biostat === "app") {
        event.preventDefault();
        nav.traverseTo(entries[i].key);
        return;
      }
    }
  });
})();
