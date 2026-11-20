import { execSync } from "node:child_process";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Версия сборки для блока provenance (задание P4): на Cloudflare Pages —
 * CF_PAGES_COMMIT_SHA, локально — короткий хэш HEAD, без git — "dev".
 */
function appVersion(): string {
  const sha = process.env.CF_PAGES_COMMIT_SHA;
  if (sha) return sha.slice(0, 7);
  try {
    return execSync("git rev-parse --short=7 HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString().trim() || "dev";
  } catch {
    return "dev";
  }
}

export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(appVersion()),
    __BUILD_DATE__: JSON.stringify(new Date().toISOString()),
  },
  // Pyodide грузится в Worker'е как внешний модуль (/pyodide/) — не бандлим.
  optimizeDeps: { exclude: ["pyodide"] },
  // Worker с Pyodide — ES-модуль: внутри динамический import pyodide.mjs.
  worker: { format: "es" },
  // fs.allow: страница замера (bench.html) берёт ../tools/bench_data.py.
  server: { port: 5173, open: true, fs: { allow: [".."] } },
});
