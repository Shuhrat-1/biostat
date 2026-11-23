/**
 * Проверка страниц методов в браузере (шаг M1, п. 7 задания): собранный сайт под CSP.
 *
 *   cd web && npm run build && cd ..
 *   node tools/check_method_pages.mjs
 *
 * Нужен Playwright: npm i -g playwright && npx playwright install chromium. Если он не виден
 * из проекта — PLAYWRIGHT=<путь к playwright/index.mjs>; Chrome вместо Chromium — PW_CHANNEL=chrome.
 *
 * Скрипт сам раздаёт web/dist с заголовками блока «/*» из web/dist/_headers и для каждой
 * страницы /…/methods/<id>/ и словаря проверяет: на 375 px scrollWidth == clientWidth;
 * в консоли нет ошибок (в т. ч. CSP); на 375 px видна кнопка «↑ К списку» 44 × 44, на 1280 —
 * оглавление; у виджета «между / внутри» ползунок меняет F, у виджета Уэлча ползунок средних меняет
 * P Уэлча, ползунок n — подпись. Код выхода 1 — есть ошибки.
 */

import { createServer } from "node:http";
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { extname, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(ROOT, "web", "dist");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8", ".zip": "application/zip" };

/** Заголовки блока «/*» из _headers (как на Cloudflare Pages для всех путей). */
function siteHeaders() {
  const lines = readFileSync(join(DIST, "_headers"), "utf8").split("\n");
  const start = lines.findIndex((l) => l.trim() === "/*");
  const out = {};
  for (const line of lines.slice(start + 1)) {
    if (!/^\s+\S/.test(line)) break;
    const i = line.indexOf(":");
    out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}

function serve() {
  const headers = siteHeaders();
  const server = createServer((req, res) => {
    let path = join(DIST, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (existsSync(path) && statSync(path).isDirectory()) path = join(path, "index.html");
    if (!existsSync(path)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { ...headers, "Content-Type": TYPES[extname(path)] ?? "application/octet-stream" });
    res.end(readFileSync(path));
  });
  return new Promise((resolve) => server.listen(0, () => resolve(server)));
}

/** Пути страниц методов в dist: /ru/methods/one_way_anova/ и т. п. */
function methodPages(dir = DIST, base = "") {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (!statSync(full).isDirectory() || name === "assets" || name === "pyodide") continue;
    const url = `${base}/${name}`;
    if (/\/methods\/[^/]+$/.test(url) && existsSync(join(full, "index.html"))) out.push(`${url}/`);
    out.push(...methodPages(full, url));
  }
  return out;
}

async function checkPage(page, url, errors) {
  const out = [];
  for (const width of [375, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(url);
    const m = await page.evaluate(() => {
      const btn = document.querySelector(".to-list"), toc = document.querySelector("aside.toc");
      const r = btn && btn.getBoundingClientRect();
      return { scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth,
        button: r && getComputedStyle(btn).display !== "none" ? [Math.round(r.width), Math.round(r.height)] : null,
        toc: Boolean(toc && getComputedStyle(toc).display !== "none") };
    });
    if (m.scroll !== m.client) out.push(`${width}px: ширина ${m.scroll} > ${m.client}`);
    const isMethod = !url.endsWith("/symbols/");
    if (isMethod && width === 375 && JSON.stringify(m.button) !== "[44,44]") out.push(`375px: кнопка «↑» ${m.button}`);
    if (isMethod && width === 1280 && (m.button || !m.toc)) out.push("1280px: нет оглавления или видна кнопка");
  }
  if (await page.locator(".gw-kb").count()) {
    const before = await page.locator(".gw-w-p").innerText();
    await page.locator(".gw-kb").fill("0.1");
    if ((await page.locator(".gw-w-p").innerText()) === before) out.push("ползунок средних не меняет P Уэлча");
    await page.locator(".gw-nb").fill("3");
    if ((await page.locator(".gw-nb-value").innerText()) !== "3") out.push("ползунок n не обновляет подпись");
  }
  if (await page.locator(".gs-kb").count()) {
    const before = await page.locator(".gs-f").innerText();
    await page.locator(".gs-kb").fill("0");
    if ((await page.locator(".gs-f").innerText()) === before) out.push("ползунок не меняет F");
  }
  return out.concat(errors.splice(0));
}

async function main() {
  let chromium;
  try { ({ chromium } = await import(process.env.PLAYWRIGHT ?? "playwright")); } catch {
    console.error("Нужен Playwright: npm i -g playwright && npx playwright install chromium (или PLAYWRIGHT=путь)");
    process.exit(2);
  }
  const server = await serve();
  const origin = `http://localhost:${server.address().port}`;
  const browser = await chromium.launch(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {});
  const page = await browser.newPage();
  const errors = [];
  page.on("console", (msg) => { if (msg.type() === "error") errors.push(`консоль: ${msg.text()}`); });
  page.on("pageerror", (e) => errors.push(`ошибка: ${e.message}`));
  let failed = 0;
  for (const path of methodPages()) {
    const problems = await checkPage(page, origin + path, errors);
    console.log(`${problems.length ? "✗" : "✓"} ${path}${problems.length ? "\n    " + problems.join("\n    ") : ""}`);
    failed += problems.length ? 1 : 0;
  }
  await browser.close();
  server.close();
  process.exit(failed ? 1 : 0);
}

main();
