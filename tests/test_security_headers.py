"""Аудит 03-02: защитные заголовки Cloudflare Pages (`web/public/_headers`)."""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _headers() -> dict[str, str]:
    """Заголовки блока «/*» — они действуют на все ответы сайта."""
    out: dict[str, str] = {}
    active = False
    for line in (WEB / "public" / "_headers").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(" "):
            active = line.strip() == "/*"
        elif active:
            name, _, value = line.strip().partition(":")
            out[name] = value.strip()
    return out


def _csp() -> dict[str, str]:
    parts = (p.strip().split(" ", 1) for p in _headers()["Content-Security-Policy"].split(";"))
    return {p[0]: (p[1] if len(p) > 1 else "") for p in parts if p[0]}


def test_03_02_csp_blocks_inline_script_and_foreign_requests() -> None:
    csp = _csp()
    assert csp["default-src"] == "'none'"
    assert "'unsafe-inline'" not in csp["script-src"] and "'unsafe-eval'" not in csp["script-src"]
    assert "'wasm-unsafe-eval'" in csp["script-src"]  # без него Pyodide не запустится
    for directive in ("script-src", "connect-src"):
        hosts = [s for s in csp[directive].split() if s.startswith("http")]
        assert len(hosts) == 1 and hosts[0].startswith("https://cdn.jsdelivr.net/pyodide/v")
    assert csp["frame-ancestors"] == "'none'" and csp["object-src"] == "'none'"
    assert csp["base-uri"] == "'none'" and csp["form-action"] == "'none'"


def test_03_02_csp_path_matches_the_pyodide_version_in_the_worker() -> None:
    """Сменили версию Pyodide в worker.ts и забыли CSP — сайт перестанет загружаться."""
    worker = (WEB / "src" / "pyodide" / "worker.ts").read_text(encoding="utf-8")
    version = re.search(r'PYODIDE_VERSION = "([\d.]+)"', worker).group(1)
    expected = f"https://cdn.jsdelivr.net/pyodide/v{version}/full/"
    assert expected in worker.replace("${PYODIDE_VERSION}", version)
    csp = _csp()
    assert expected in csp["script-src"].split() and expected in csp["connect-src"].split()


def test_03_02_other_protective_headers_are_set() -> None:
    h = _headers()
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Referrer-Policy"] == "no-referrer"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Cross-Origin-Opener-Policy"] == "same-origin"
    assert "camera=()" in h["Permissions-Policy"] and "geolocation=()" in h["Permissions-Policy"]
    assert "Cross-Origin-Embedder-Policy" not in h  # сломал бы загрузку Pyodide с CDN
