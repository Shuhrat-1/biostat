/**
 * Формулы страниц методов → MathML при сборке (KaTeX 0.16.11, только devDependency).
 *
 * Список TeX пишет Python-генератор (python tools/static_pages.py → content/math_tex.json):
 * в TeX подставлены числа примера, а их даёт только расчёт ядра. Скрипт рендерит
 * каждую формулу katex.renderToString(tex, { output: "mathml", displayMode }) и пишет
 * content/math_cache.json; формулы, которых больше нет в списке, удаляются. Шрифты,
 * CSS и JS KaTeX на страницы не попадают — только MathML.
 *
 * Запуск: node web/build-math.mjs, затем снова python tools/static_pages.py.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import katex from "katex";

const content = join(dirname(fileURLToPath(import.meta.url)), "..", "content");
const TEX_LIST = join(content, "math_tex.json");
const CACHE = join(content, "math_cache.json");

/** MathML одной формулы; ошибка TeX роняет сборку с самой формулой в тексте. */
export function render(tex, displayMode) {
  return katex.renderToString(tex, { output: "mathml", displayMode, throwOnError: true, strict: "ignore" });
}

/** Кеш для списка формул: { inline: {tex: mathml}, display: {...} }, ключи по порядку. */
export function buildCache(list) {
  const cache = {};
  for (const kind of ["inline", "display"]) {
    cache[kind] = {};
    for (const tex of [...list[kind]].sort()) cache[kind][tex] = render(tex, kind === "display");
  }
  return cache;
}

function main() {
  const list = JSON.parse(readFileSync(TEX_LIST, "utf8"));
  const cache = buildCache(list);
  writeFileSync(CACHE, JSON.stringify(cache, null, 1) + "\n");
  console.log(`MathML: ${Object.keys(cache.inline).length} встроенных, ${Object.keys(cache.display).length} выключных`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
