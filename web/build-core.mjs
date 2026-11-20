/**
 * Упаковка расчётного ядра (Python core/) в zip для Pyodide.
 *
 * Node-версия build_core.py — нужна для деплоя: хостинг (Cloudflare
 * Pages и т.п.) запускает при сборке Node, а не Python, поэтому zip
 * собирается здесь и встроен в `npm run build`.
 *
 * Результат: web/public/core/core.zip — браузер распаковывает его в
 * виртуальную ФС Pyodide.
 *
 * Архив воспроизводим: файлы по алфавиту, одно время у всех записей, концы
 * строк LF (на Windows git может выдать CRLF). Если содержимое не изменилось,
 * файл не перезаписывается — `npm run build` и `npm run dev` больше не
 * оставляют core.zip изменённым в git. Те же правила у build_core.py.
 */

import { existsSync, readdirSync, readFileSync, statSync, mkdirSync, writeFileSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const scriptDir = dirname(fileURLToPath(import.meta.url));
// Скрипт лежит в web/, корень проекта — на уровень выше.
const root = join(scriptDir, "..");
const coreDir = join(root, "core");
const outPath = join(root, "web", "public", "core", "core.zip");
// Минимальная дата формата zip (как в build_core.py): время файлов в архив не попадает.
const FIXED_TIME = new Date(1980, 0, 1, 0, 0, 0);

/** Рекурсивно собрать все .py, пропуская __pycache__. */
function collectPyFiles(dir) {
  const result = [];
  for (const entry of readdirSync(dir)) {
    if (entry === "__pycache__") continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      result.push(...collectPyFiles(full));
    } else if (entry.endsWith(".py")) {
      result.push(full);
    }
  }
  return result;
}

/** Путь в архиве → содержимое с LF, по алфавиту (как sorted() в Python). */
export function coreFiles() {
  const entries = collectPyFiles(coreDir).map((file) => [
    // Путь внутри архива — относительно корня, с прямыми слэшами.
    relative(root, file).split("\\").join("/"),
    Buffer.from(readFileSync(file).toString("latin1").replace(/\r\n/g, "\n"), "latin1"),
  ]);
  entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return new Map(entries);
}

/** Совпадает ли содержимое существующего архива с файлами ядра. */
function sameAsExisting(files) {
  if (!existsSync(outPath)) return false;
  try {
    const entries = new AdmZip(outPath).getEntries().filter((e) => !e.isDirectory);
    return entries.length === files.size
      && entries.every((e) => files.get(e.entryName)?.equals(e.getData()));
  } catch {
    return false;
  }
}

function build() {
  const files = coreFiles();
  if (sameAsExisting(files)) {
    console.log(`core.zip не изменился (${files.size} файлов) — не перезаписываю`);
    return;
  }
  const zip = new AdmZip();
  for (const [name, data] of files) {
    zip.addFile(name, data);
    zip.getEntry(name).header.time = FIXED_TIME;
  }
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, zip.toBuffer());
  const sizeKb = (statSync(outPath).size / 1024).toFixed(1);
  console.log(`Упаковано ${files.size} файлов -> core.zip (${sizeKb} КБ)`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) build();
