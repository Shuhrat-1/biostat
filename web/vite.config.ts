import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Pyodide грузится с CDN как внешний модуль — не пытаемся его бандлить.
  optimizeDeps: { exclude: ["pyodide"] },
  server: { port: 5173, open: true },
});
