/**
 * Лицензии для сайта (шаг P2): при сборке пишет в web/public/
 *   LICENSE.txt              — лицензия кода BioStat (копия LICENSE из корня);
 *   third-party-licenses.txt — полные тексты лицензий того, что сайт раздаёт.
 *
 * Минификация вырезает из бандла тексты лицензий ECharts, zrender и d3, а BSD-3 и
 * Apache-2.0 (п. 4) требуют передавать их вместе с копией, — поэтому отдельный файл.
 * Состав проверяется: пакет из зависимостей package.json без записи в BUNDLED (или
 * пакет Python из замыкания REQUIRED_PACKAGES без записи в PYTHON) роняет сборку.
 * Перечень для людей — THIRD_PARTY_NOTICES.md в корне (тест сверяет его с этим файлом).
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const webDir = dirname(fileURLToPath(import.meta.url));
const modules = join(webDir, "node_modules");

/**
 * Пакеты npm в бандле (проверено по sourcemap сборки): имя → SPDX и файлы лицензии.
 * standalone — в бандл идёт готовая сборка пакета (xlsx.mjs), его npm-зависимости
 * (adler-32, cfb, codepage… — тоже SheetJS, Apache-2.0) в бандл не входят.
 */
export const BUNDLED = {
  react: { license: "MIT", files: ["LICENSE"] },
  "react-dom": { license: "MIT", files: ["LICENSE"] },
  scheduler: { license: "MIT", files: ["LICENSE"] },
  echarts: { license: "Apache-2.0", files: ["LICENSE", "NOTICE", "licenses/LICENSE-d3"] },
  zrender: { license: "BSD-3-Clause", files: ["LICENSE"] },
  tslib: { license: "0BSD", files: ["LICENSE.txt"] },
  xlsx: { license: "Apache-2.0", files: ["LICENSE"], standalone: true },
};

/** Зависимости, которые в бандл не попадают: трансформация сборки React (browserify). */
export const NOT_BUNDLED = ["loose-envify", "js-tokens"];

/** Пакеты Python с jsDelivr: SPDX по METADATA и файлам лицензий колёс Pyodide 0.26.4. */
export const PYTHON = {
  numpy: "BSD-3-Clause",
  scipy: "BSD-3-Clause",
  pandas: "BSD-3-Clause",
  statsmodels: "BSD-3-Clause",
  patsy: "BSD-2-Clause",
  openblas: "BSD-3-Clause",
  "python-dateutil": "Apache-2.0 AND BSD-3-Clause",
  packaging: "Apache-2.0 OR BSD-2-Clause",
  pytz: "MIT",
  six: "MIT",
  "charset-normalizer": "MIT",
};

/** Пакеты Python без файла лицензии внутри (openblas — только libopenblas.so): где текст. */
export const LICENSE_ELSEWHERE = {
  openblas: (v) => `https://github.com/OpenMathLib/OpenBLAS/blob/v${v}/LICENSE`,
};

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
/** Текст лицензии с окончаниями строк LF (у tslib — CRLF), без хвостовых пустых строк. */
const licenseText = (path) => readFileSync(path, "utf8").replace(/\r\n/g, "\n").trimEnd();
const version = (name) => readJson(join(modules, name, "package.json")).version;

/** Замыкание зависимостей package.json; падает на пакете без записи в BUNDLED. */
export function bundledPackages() {
  const todo = Object.keys(readJson(join(webDir, "package.json")).dependencies);
  const seen = new Set();
  while (todo.length) {
    const name = todo.pop();
    if (seen.has(name) || NOT_BUNDLED.includes(name)) continue;
    if (!BUNDLED[name]) throw new Error(`${name}: нет лицензии в BUNDLED (build-licenses.mjs)`);
    seen.add(name);
    if (BUNDLED[name].standalone) continue;
    todo.push(...Object.keys(readJson(join(modules, name, "package.json")).dependencies ?? {}));
  }
  return [...seen].sort();
}

/** Пакеты Python, которые загружает Worker (замыкание по lock-файлу Pyodide). */
export function pythonPackages() {
  const lock = readJson(join(modules, "pyodide", "pyodide-lock.json")).packages;
  const worker = readFileSync(join(webDir, "src", "pyodide", "worker.ts"), "utf8");
  const todo = [...worker.match(/REQUIRED_PACKAGES = \[([^\]]+)\]/)[1].matchAll(/"([\w-]+)"/g)]
    .map((m) => m[1]);
  const seen = new Map();
  while (todo.length) {
    const name = todo.pop();
    if (seen.has(name)) continue;
    if (!PYTHON[name]) throw new Error(`${name}: нет лицензии в PYTHON (build-licenses.mjs)`);
    seen.set(name, lock[name].version);
    todo.push(...lock[name].depends);
  }
  return [...seen].sort().map(([name, ver]) => ({ name, version: ver, license: PYTHON[name] }));
}

/** Рантайм Pyodide со своего домена и CPython внутри него; тексты — web/licenses/. */
export function runtimeComponents() {
  const lock = readJson(join(modules, "pyodide", "pyodide-lock.json"));
  const pyodide = lock.info.version;
  const python = lock.info.python;
  return [
    { name: "pyodide", version: pyodide, license: "MPL-2.0",
      source: `https://github.com/pyodide/pyodide/tree/${pyodide}`,
      text: join(webDir, "licenses", `pyodide-${pyodide}.LICENSE`) },
    { name: "CPython", version: python, license: "PSF-2.0",
      source: `https://github.com/python/cpython/tree/v${python}`,
      text: join(webDir, "licenses", `cpython-${python}.LICENSE`) },
  ];
}

/** Все компоненты одним списком: имя, версия, SPDX, где раздаётся. */
export function components() {
  const npm = bundledPackages().map((name) =>
    ({ name, version: version(name), license: BUNDLED[name].license, served: "bundle" }));
  const runtime = runtimeComponents().map(({ name, version: v, license }) =>
    ({ name, version: v, license, served: "runtime" }));
  const python = pythonPackages().map((p) => ({ ...p, served: "jsdelivr" }));
  return [...npm, ...runtime, ...python];
}

const RULE = "=".repeat(78);

function section(title, texts) {
  return `${RULE}\n${title}\n${RULE}\n\n${texts.join("\n\n")}\n\n`;
}

/** Текст /third-party-licenses.txt. */
export function thirdPartyText() {
  const head = "Third-party software served by https://biostat.app\n\n" +
    "BioStat itself is licensed under the Apache License 2.0: https://biostat.app/LICENSE.txt\n\n";
  const npm = bundledPackages().map((name) => section(`${name} ${version(name)} (${BUNDLED[name].license})` +
    " - JavaScript bundle", BUNDLED[name].files.map((f) => licenseText(join(modules, name, f)))));
  const runtime = runtimeComponents().map((c) => section(`${c.name} ${c.version} (${c.license})` +
    ` - /pyodide/ runtime\nSource code: ${c.source}`, [licenseText(c.text)]));
  const py = pythonPackages().map((p) => `  ${p.name} ${p.version} (${p.license})` +
    (LICENSE_ELSEWHERE[p.name] ? `\n    license text: ${LICENSE_ELSEWHERE[p.name](p.version)}` : ""))
    .join("\n");
  const tail = section("Python packages loaded from https://cdn.jsdelivr.net/pyodide/ (not served by" +
    " BioStat;\nthe license texts are inside each package file unless a link is given)", [py]);
  return head + npm.join("") + runtime.join("") + tail;
}

function main() {
  const publicDir = join(webDir, "public");
  mkdirSync(publicDir, { recursive: true });
  writeFileSync(join(publicDir, "LICENSE.txt"), readFileSync(join(webDir, "..", "LICENSE")));
  writeFileSync(join(publicDir, "third-party-licenses.txt"), thirdPartyText());
  console.log("лицензии → public/LICENSE.txt, public/third-party-licenses.txt");
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
