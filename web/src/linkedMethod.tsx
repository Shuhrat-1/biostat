/**
 * Метод из адреса: /?method=<id>&lang=<ru|en|pt> — ссылка «Проверить метод в калькуляторе»
 * со страницы-объяснения (задание L1). Параметры читаются один раз при загрузке и сразу
 * убираются из адреса: перезагрузка страницы выбор не навязывает, язык дальше помнит
 * приложение. Выбор применяется только к первой таблице после перехода.
 */

import { useEffect, useState } from "react";
import { LANGUAGES, useLang, useT, type Lang } from "./i18n";
import { c } from "./styles";

interface LaunchParams {
  method: string | null;
  lang: Lang | null;
}

let launch: LaunchParams | null = null;

/** method и lang из адреса — один раз за загрузку; оба параметра тут же убираются из адреса. */
export function launchParams(): LaunchParams {
  if (launch) return launch;
  launch = { method: null, lang: null };
  try {
    const url = new URL(window.location.href);
    const lang = url.searchParams.get("lang");
    launch = {
      method: url.searchParams.get("method") || null,
      lang: LANGUAGES.some((l) => l.code === lang) ? (lang as Lang) : null,
    };
    if (url.searchParams.has("method") || url.searchParams.has("lang")) {
      url.searchParams.delete("method");
      url.searchParams.delete("lang");
      window.history.replaceState(window.history.state, "", url.pathname + url.search + url.hash);
    }
  } catch {
    // адрес не разобрался — работаем как без параметров
  }
  return launch;
}

type Names = Record<Lang, string>;
/** waiting — ждём первую таблицу; unfit — метод к ней не подходит; done — выбор исчерпан. */
type Phase = "waiting" | "unfit" | "done";

/** Название метода из /methods.json; неизвестный id или сбой запроса — null, молча. */
async function methodNames(id: string): Promise<Names | null> {
  try {
    const response = await fetch("/methods.json");
    if (!response.ok) return null;
    const data = (await response.json()) as { methods: { id: string; name: Names }[] };
    return data.methods.find((m) => m.id === id)?.name ?? null;
  } catch {
    return null;
  }
}

/**
 * Состояние выбора из ссылки. fileKey меняется только с новым файлом или примером:
 * строка «не подходит» держится до следующего файла.
 */
export function useLinkedMethod(
  suggestions: { method: string }[] | null,
  fileKey: string | null,
  requestMethod: (method: string) => void,
) {
  const id = launchParams().method;
  const [names, setNames] = useState<Names | null>(null);
  const [phase, setPhase] = useState<Phase>("waiting");
  const [unfitFile, setUnfitFile] = useState<string | null>(null);

  useEffect(() => {
    if (id) void methodNames(id).then(setNames);
  }, [id]);

  useEffect(() => {
    if (!id || !names || !suggestions) return;
    if (phase === "waiting") {
      if (suggestions.some((sg) => sg.method === id)) {
        requestMethod(id);
        setPhase("done");
      } else {
        setPhase("unfit");
        setUnfitFile(fileKey);
      }
    } else if (phase === "unfit" && fileKey !== unfitFile) {
      setPhase("done");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [names, suggestions, fileKey]);

  return { names: id ? names : null, phase, cancel: () => setPhase("done") };
}

/** Строка над загрузкой («Выбран метод…») или над таблицей («не подходит») с крестиком. */
export function LinkedMethodNote({ names, text, onCancel }: {
  names: Names;
  text: "selected" | "unfit";
  onCancel: () => void;
}) {
  const t = useT();
  const lang = useLang();
  const message = t(`link.${text}`).replace("{name}", names[lang] ?? names.en);
  return (
    <div
      role="status"
      data-linked-method={text}
      style={{
        display: "flex", alignItems: "flex-start", gap: "0.6rem", marginBottom: "1rem",
        padding: "0.6rem 0.8rem", borderRadius: 8, border: `1px solid ${c.accent}`,
        background: c.selectedBg, color: c.text, fontSize: "0.9rem", lineHeight: 1.45,
      }}
    >
      <span style={{ flex: 1, minWidth: 0 }}>{message}</span>
      <button
        type="button"
        onClick={onCancel}
        aria-label={t("link.cancel")}
        title={t("link.cancel")}
        style={{
          border: "none", background: "transparent", color: c.muted, cursor: "pointer",
          fontSize: "1.1rem", lineHeight: 1, minWidth: 44, minHeight: 44, margin: "-0.6rem -0.8rem -0.6rem 0",
        }}
      >
        ×
      </button>
    </div>
  );
}
