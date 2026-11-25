/**
 * Раздача собранного сайта (web/dist) для браузерных проверок: заголовки блока «/*» из
 * web/dist/_headers (как на Cloudflare Pages для всех путей), каталог — его index.html.
 * Общий модуль tools/check_method_pages.mjs и tools/check_method_link.mjs.
 */

import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { extname, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const DIST = join(ROOT, "web", "dist");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8", ".zip": "application/zip" };

/** Заголовки блока «/*» из _headers. */
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

/** HTTP-сервер на свободном порту; адрес — server.address().port. */
export function serve() {
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

/** Playwright: из проекта, глобально или по PLAYWRIGHT=<путь к playwright/index.mjs>. */
export async function playwright() {
  try {
    return await import(process.env.PLAYWRIGHT ?? "playwright");
  } catch {
    console.error("Нужен Playwright: npm i -g playwright && npx playwright install chromium (или PLAYWRIGHT=путь)");
    process.exit(2);
  }
}
