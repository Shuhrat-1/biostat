/**
 * Упаковка расчётного ядра (Python core/) в zip для Pyodide.
 *
 * Node-версия build_core.py — нужна для деплоя: хостинг (Cloudflare
 * Pages и т.п.) запускает при сборке Node, а не Python, поэтому zip
 * собирается здесь и встроен в `npm run build`.
 *
 * Результат: web/public/core/core.zip — браузер распаковывает его в
 * виртуальную ФС Pyodide.
 */

import { readdirSync, statSync, mkdirSync, writeFileSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const scriptDir = dirname(fileURLToPath(import.meta.url));
// Скрипт лежит в web/, корень проекта — на уровень выше.
const root = join(scriptDir, "..");
const coreDir = join(root, "core");
const outPath = join(root, "web", "public", "core", "core.zip");

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

function build() {
  const files = collectPyFiles(coreDir);
  const zip = new AdmZip();
  for (const file of files) {
    // Путь внутри архива — относительно корня, с прямыми слэшами.
    const archivePath = relative(root, file).split("\\").join("/");
    zip.addLocalFile(file, dirname(archivePath));
  }
  mkdirSync(dirname(outPath), { recursive: true });
  zip.writeZip(outPath);

  const sizeKb = (statSync(outPath).size / 1024).toFixed(1);
  console.log(`Упаковано ${files.length} файлов -> core.zip (${sizeKb} КБ)`);
}

build();
