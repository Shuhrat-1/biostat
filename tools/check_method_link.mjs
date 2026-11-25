/**
 * Калькулятор с методом из адреса (задание L1): /?method=<id>&lang=<ru|en|pt> — ссылка
 * «Проверить метод в калькуляторе →» со страницы метода. Собранный сайт под CSP.
 *
 *   cd web && npm run build && cd ..
 *   node tools/check_method_link.mjs
 *
 * Playwright — как для tools/check_method_pages.mjs (PLAYWRIGHT=путь, PW_CHANNEL=chrome). Пакеты
 * Pyodide приложение берёт с cdn.jsdelivr.net; без сети — PYODIDE_DIR=<каталог распакованного
 * pyodide-0.26.4> (файлы отдаются оттуда). Первый запуск Pyodide — до минуты на сценарий.
 *
 * Сценарии: (1) pt + ANOVA — интерфейс pt, строка «Método selecionado…», параметры убраны из адреса,
 * в режиме планирования строки нет, после перезагрузки метод не навязывается, пример → форма ANOVA;
 * (2) paired_t + пример → форма парного t-теста: пример подходит ко всем методам каталога, поэтому
 * «не подходит» — файлом из одной числовой колонки (ANOVA нужен фактор): строка, форма как без ссылки,
 * крестик; следующий файл — строка уходит, метод не навязывается; (3) неизвестный id — ничего;
 * (4) 375 px без горизонтальной прокрутки. Везде — консоль без ошибок и нарушений CSP.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { playwright, serve } from "./static_server.mjs";

const PYO = process.env.PYODIDE_DIR ?? "";
const { chromium } = await playwright();
const server = await serve();
const BASE = `http://localhost:${server.address().port}`;
const browser = await chromium.launch(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {});
const EXAMPLE = { ru: "Попробовать на примере", en: "Try an example", pt: "Experimentar com um exemplo" };
const PLAN = { ru: "Планирование опыта (без данных)", en: "Plan an experiment (no data)", pt: "Planear um ensaio (sem dados)" };
let fail = 0;
const ok = (cond, msg, extra = "") => { if (!cond) fail++; console.log(`${cond ? "✓" : "✗"} ${msg}${cond ? "" : " " + extra}`); };

async function start(url, width = 1280) {
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  if (PYO) await page.route("https://cdn.jsdelivr.net/pyodide/**", (route) => {
    const file = join(PYO, new URL(route.request().url()).pathname.split("/full/")[1]);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "nf" });
    route.fulfill({ status: 200, body: readFileSync(file),
      headers: { "content-type": "application/octet-stream", "access-control-allow-origin": "*" } });
  });
  const log = [];
  page.on("console", (m) => { if (m.type() === "error" || /Content Security Policy/i.test(m.text())) log.push(m.text()); });
  page.on("pageerror", (e) => log.push("pageerror: " + e.message));
  await page.goto(BASE + url);
  return { page, log };
}
const ready = (page, lang) => page.getByRole("button", { name: EXAMPLE[lang] }).waitFor({ timeout: 180000 });
const note = (page) => page.locator("[data-linked-method]");
const pressed = (page) => page.locator('#method-list button[aria-pressed="true"]').first().innerText();
const noScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth === document.documentElement.clientWidth);

// 1. pt + ANOVA → строка, пример → форма ANOVA, адрес без параметров
{
  const { page, log } = await start("/?method=one_way_anova&lang=pt");
  await ready(page, "pt");
  ok(await page.evaluate(() => document.documentElement.lang) === "pt", "1: интерфейс на pt");
  ok(new URL(page.url()).search === "", "1: method и lang убраны из адреса сразу", page.url());
  const text = await note(page).innerText().catch(() => "");
  ok(text.startsWith("Método selecionado: ANOVA a um fator"), "1: строка «Método selecionado: ANOVA a um fator…»", text);
  await page.getByRole("button", { name: PLAN.pt }).click();
  await page.getByText("Planeamento do ensaio:").waitFor({ timeout: 60000 });
  ok(await note(page).count() === 0, "1: в режиме планирования строки нет");
  await page.reload(); await ready(page, "pt");
  ok(await note(page).count() === 0, "1: после перезагрузки метод не навязывается");
  ok(log.length === 0, "1: консоль без ошибок и нарушений CSP", JSON.stringify(log));
  await page.close();
}
{
  const { page, log } = await start("/?method=one_way_anova&lang=pt");
  await ready(page, "pt");
  await page.getByRole("button", { name: EXAMPLE.pt }).click();
  await page.locator("#method-list").waitFor({ timeout: 120000 }); await page.waitForTimeout(1500);
  const sel = await pressed(page);
  ok(/ANOVA a um fator/.test(sel), "1: после примера открыта форма ANOVA", sel);
  ok(await note(page).count() === 0, "1: строка исчезла после открытия формы");
  ok(new URL(page.url()).search === "", "1: параметра в адресе нет");
  ok(log.length === 0, "1: консоль без ошибок", JSON.stringify(log));
  await page.close();
}
// 2. Пример подходит ко всем 71 методу каталога (предложения фильтруются только по типам колонок),
// поэтому «не подходит» проверяется файлом с одной числовой колонкой: ANOVA нужен фактор.
const ONE_COLUMN = "y\n" + Array.from({ length: 12 }, (_, i) => (10 + (i * 7) % 5).toFixed(1)).join("\n") + "\n";
const upload = (page) => page.setInputFiles('input[type="file"]', { name: "one_column.csv", mimeType: "text/csv", buffer: Buffer.from(ONE_COLUMN) });
let defaultCard = "";
{
  const { page } = await start("/?lang=en");
  await ready(page, "en");
  await upload(page);
  await page.locator("#method-list").waitFor({ timeout: 120000 }); await page.waitForTimeout(1500);
  defaultCard = await pressed(page);
  await page.close();
}
{
  const { page, log } = await start("/?method=paired_t&lang=en");
  await ready(page, "en");
  await page.getByRole("button", { name: EXAMPLE.en }).click();
  await page.locator("#method-list").waitFor({ timeout: 120000 }); await page.waitForTimeout(1500);
  ok(/Paired t-test/.test(await pressed(page)), "2: paired_t + пример → форма парного t-теста (пример подходит)");
  ok(log.length === 0, "2: консоль без ошибок", JSON.stringify(log));
  await page.close();
}
{
  const { page, log } = await start("/?method=one_way_anova&lang=en");
  await ready(page, "en");
  await upload(page);
  await page.locator("#method-list").waitFor({ timeout: 120000 }); await page.waitForTimeout(1500);
  const text = await note(page).innerText().catch(() => "");
  ok(text.startsWith("The method “One-way ANOVA” does not fit these data"), "2: строка «не подходит»", text);
  ok(await pressed(page) === defaultCard, "2: форма на первом методе, как без ссылки", defaultCard);
  await page.getByRole("button", { name: EXAMPLE.en }).waitFor({ state: "detached" }).catch(() => {});
  await page.getByRole("button", { name: "Cancel the selection" }).click();
  ok(await note(page).count() === 0, "2: крестик убирает строку");
  ok(log.length === 0, "2: консоль без ошибок", JSON.stringify(log));
  await page.close();
}
{
  const { page, log } = await start("/?method=one_way_anova&lang=en");
  await ready(page, "en");
  await upload(page);
  await page.locator("[data-linked-method=unfit]").waitFor({ timeout: 120000 });
  await page.getByRole("button", { name: "Load another file" }).first().click();
  await page.setInputFiles('input[type="file"]', { name: "example2.csv", mimeType: "text/csv",
    buffer: Buffer.from("group;y\na;1.0\na;1.2\na;0.9\nb;2.1\nb;2.0\nb;2.3\nc;3.1\nc;2.9\nc;3.0\n") });
  await page.waitForTimeout(3000);
  ok(await note(page).count() === 0, "2: строка «не подходит» уходит со следующим файлом");
  ok(!/One-way ANOVA/.test(await pressed(page)), "2: следующему файлу метод не навязывается", await pressed(page));
  ok(log.length === 0, "2: консоль без ошибок", JSON.stringify(log));
  await page.close();
}
// 3. неизвестный метод → ничего
{
  const { page, log } = await start("/?method=nonsense");
  await ready(page, "ru"); await page.waitForTimeout(1500);
  ok(await page.locator('[role="status"]').filter({ hasText: "Выбран метод" }).count() === 0, "3: строки нет");
  ok(new URL(page.url()).search === "", "3: адрес очищен");
  ok(log.length === 0, "3: консоль без ошибок", JSON.stringify(log));
  await page.close();
}
// 4. 375 px: строка без горизонтальной прокрутки, в т. ч. «не подходит»
for (const [url, lang] of [["/?method=one_way_anova&lang=pt", "pt"], ["/?method=one_way_anova&lang=ru", "ru"]]) {
  const { page, log } = await start(url, 375);
  await ready(page, lang);
  ok(await noScroll(page), `4: 375 px без прокрутки (${lang}, строка выбора)`);
  if (lang === "ru") {
    await upload(page);
    await page.locator("#method-list").waitFor({ timeout: 120000 }); await page.waitForTimeout(1500);
    ok(await noScroll(page), "4: 375 px без прокрутки (ru, «не подходит»)");
  }
  ok(log.length === 0, `4: консоль без ошибок (${lang})`, JSON.stringify(log));
  await page.close();
}
await browser.close();
server.close();
process.exit(fail ? 1 : 0);
