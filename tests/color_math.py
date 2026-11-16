"""Цветовые расчёты для тестов палитры: контраст WCAG 2.2, симуляция дальтонизма
(Machado, Oliveira, Fernandes, 2009, полная выраженность), различие ΔE2000 (CIE)."""

from __future__ import annotations

import math

import numpy as np

CVD = {
    "протан": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "дейтан": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "тритан": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}
VIEWS = (None, "протан", "дейтан", "тритан")
_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _linear(hexcode: str) -> np.ndarray:
    v = np.array([int(hexcode.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def contrast(a: str, b: str) -> float:
    """Отношение контрастов WCAG."""
    la, lb = sorted((float(_linear(x) @ [0.2126, 0.7152, 0.0722]) for x in (a, b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def lab(hexcode: str, view: str | None = None) -> np.ndarray:
    """CIELAB цвета, каким его видит человек с дальтонизмом вида ``view``."""
    lin = _linear(hexcode) if view is None else CVD[view] @ _linear(hexcode)
    xyz = (_XYZ @ np.clip(lin, 0, 1)) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: str, b: str, view: str | None = None) -> float:
    """ΔE2000 между двумя цветами в данном виде зрения."""
    (l1, a1, b1), (l2, a2, b2) = lab(a, view), lab(b, view)
    cbar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1, c2 = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1, h2 = math.degrees(math.atan2(b1, a1p)) % 360, math.degrees(math.atan2(b2, a2p)) % 360
    dh = 0.0 if c1 * c2 == 0 else (h2 - h1 + 180) % 360 - 180
    d_hh = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(dh / 2))
    lm, cm = (l1 + l2) / 2, (c1 + c2) / 2
    hm = (h1 + h2) / 2 if abs(h1 - h2) <= 180 else (h1 + h2 + 360) / 2 % 360
    if c1 * c2 == 0:
        hm = h1 + h2
    t = (1 - 0.17 * math.cos(math.radians(hm - 30)) + 0.24 * math.cos(math.radians(2 * hm))
         + 0.32 * math.cos(math.radians(3 * hm + 6)) - 0.20 * math.cos(math.radians(4 * hm - 63)))
    sl = 1 + 0.015 * (lm - 50) ** 2 / math.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cm, 1 + 0.015 * cm * t
    rt = (-2 * math.sqrt(cm ** 7 / (cm ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-((hm - 275) / 25) ** 2))))
    return math.sqrt(((l2 - l1) / sl) ** 2 + ((c2 - c1) / sc) ** 2 + (d_hh / sh) ** 2
                     + rt * ((c2 - c1) / sc) * (d_hh / sh))
