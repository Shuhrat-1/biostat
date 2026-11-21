import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { markAppHistoryEntry } from "./historyMark";

const container = document.getElementById("root");
if (!container) throw new Error("Не найден элемент #root");
markAppHistoryEntry();

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
