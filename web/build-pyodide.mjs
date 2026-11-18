/**
 * Рантайм Pyodide со своего домена (аудит 03-03).
 *
 * Пять файлов рантайма берутся из npm-пакета pyodide (его целостность проверяет
 * npm по package-lock.json) и кладутся в web/public/pyodide/v<версия>/:
 * pyodide.mjs, pyodide.asm.js, pyodide.asm.wasm, python_stdlib.zip и
 * pyodide-lock.json. Пакеты (numpy, scipy…) остаются на jsDelivr — scipy весит
 * 44.9 МБ, больше предела Cloudflare Pages (25 МБ на файл). Поэтому в своей копии
 * lock-файла имена пакетов заменены полными адресами jsDelivr, а sha256 оставлены:
 * Pyodide скачивает пакет с CDN и сверяет его со *своим* lock-файлом (fetch с
 * integrity). Подменённый пакет браузер отклонит.
 */

import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** Должна совпадать с PYODIDE_VERSION в src/pyodide/worker.ts (проверяет тест). */
export const PYODIDE_VERSION = "0.26.4";
export const PACKAGES_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const RUNTIME = ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip"];

const webDir = dirname(fileURLToPath(import.meta.url));
const source = join(webDir, "node_modules", "pyodide");
const target = join(webDir, "public", "pyodide", `v${PYODIDE_VERSION}`);

/** Lock-файл с полными адресами пакетов на CDN; хеши не меняются. */
export function rewriteLock(lock, cdn = PACKAGES_CDN) {
  const packages = Object.fromEntries(Object.entries(lock.packages).map(([name, pkg]) =>
    [name, { ...pkg, file_name: new URL(pkg.file_name, cdn).toString() }]));
  return { ...lock, packages };
}

function main() {
  const installed = JSON.parse(readFileSync(join(source, "package.json"), "utf8")).version;
  if (installed !== PYODIDE_VERSION) {
    throw new Error(`npm pyodide ${installed}, а нужен ${PYODIDE_VERSION}: выполните npm ci`);
  }
  mkdirSync(target, { recursive: true });
  for (const file of RUNTIME) copyFileSync(join(source, file), join(target, file));
  const lock = JSON.parse(readFileSync(join(source, "pyodide-lock.json"), "utf8"));
  writeFileSync(join(target, "pyodide-lock.json"), JSON.stringify(rewriteLock(lock)));
  console.log(`pyodide ${PYODIDE_VERSION}: рантайм → public/pyodide/v${PYODIDE_VERSION}/`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
