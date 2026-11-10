import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Pyodide грузится с CDN как внешний модуль — не пытаемся его бандлить.
  optimizeDeps: { exclude: ["pyodide"] },
  // Worker с Pyodide — ES-модуль: внутри динамический import pyodide.mjs с CDN.
  worker: { format: "es" },
  // fs.allow: страница замера (bench.html) берёт ../tools/bench_data.py.
  server: { port: 5173, open: true, fs: { allow: [".."] } },
});
