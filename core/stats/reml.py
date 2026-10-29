"""REML для моделей компонент дисперсии и тесты фиксированных членов (гл. 16 книги).

V = Σ σ²_k Z_k Z_kᵀ + σ² I. Компоненты оцениваются методом Фишера (scoring) по
REML (Patterson, Thompson, 1971); отрицательные оценки допускаются, если V
остаётся положительно определённой (стр. 433–434). Фиксированные эффекты —
обобщённый МНК (eBLUE), случайные — eBLUP (стр. 436).

Тесты членов (стр. 431–432): статистика Вальда W = b̂ᵀ(LᵀΦL)⁻¹b̂ с χ²(q), и
приближённый F = W/q со знаменательными df по Кенварду–Роджеру (1997) —
как в GenStat, на котором построена книга: df считаются по матрице средней
информации (average information). В сбалансированных дизайнах F и df
совпадают с многострато́вой ANOVA (Example 16.1E).
"""

from __future__ import annotations

import numpy as np

_MAX_ITER = 100
_TOL = 1e-10


def _projection(v_inv: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Φ = (XᵀV⁻¹X)⁻ и REML-проектор P = V⁻¹ − V⁻¹XΦXᵀV⁻¹."""
    phi = np.linalg.pinv(x.T @ v_inv @ x)
    return phi, v_inv - v_inv @ x @ phi @ x.T @ v_inv


def _inverse_pd(v: np.ndarray) -> np.ndarray | None:
    """V⁻¹ через Холецкого; None, если V не положительно определена."""
    try:
        chol = np.linalg.cholesky(v)
    except np.linalg.LinAlgError:
        return None
    inv_l = np.linalg.inv(chol)
    return inv_l.T @ inv_l


def _score_info(y: np.ndarray, p: np.ndarray, vs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Градиент REML и ожидаемая информация I_ij = ½ tr(P V_i P V_j)."""
    py = p @ y
    pv = [p @ v for v in vs]
    score = np.array([-0.5 * (np.trace(a) - py @ v @ py) for a, v in zip(pv, vs)])
    info = np.array([[0.5 * np.sum(a * b.T) for b in pv] for a in pv])
    return score, info


def reml_fit(y: np.ndarray, x: np.ndarray, vs: list[np.ndarray]) -> dict:
    """Fisher scoring с половинным шагом, пока V не станет положительно определённой."""
    theta = np.full(len(vs), float(np.var(y)) / len(vs))
    converged = False
    for it in range(_MAX_ITER):
        v_inv = _inverse_pd(sum(t * v for t, v in zip(theta, vs)))
        _, p = _projection(v_inv, x)
        score, info = _score_info(y, p, vs)
        step = np.linalg.lstsq(info, score, rcond=None)[0]
        new = _safe_step(theta, step, vs)
        if new is None:
            break
        if np.max(np.abs(new - theta)) < _TOL * max(1.0, np.max(np.abs(theta))):
            theta, converged = new, True
            break
        theta = new
    return _finish(y, x, vs, theta, converged and it < _MAX_ITER)


def _safe_step(theta: np.ndarray, step: np.ndarray, vs: list[np.ndarray]) -> np.ndarray | None:
    """Шаг (с делением пополам), сохраняющий V положительно определённой."""
    for _ in range(30):
        new = theta + step
        if new[-1] > 0 and _inverse_pd(sum(t * v for t, v in zip(new, vs))) is not None:
            return new
        step = step / 2
    return None


def _finish(y: np.ndarray, x: np.ndarray, vs: list[np.ndarray], theta: np.ndarray,
            converged: bool) -> dict:
    """GLS-оценки, ковариации и матрицы информации в точке оптимума."""
    v_inv = _inverse_pd(sum(t * v for t, v in zip(theta, vs)))
    phi, p = _projection(v_inv, x)
    _, info = _score_info(y, p, vs)
    py = p @ y
    ai = np.array([[0.5 * (py @ vi @ p @ vj @ py) for vj in vs] for vi in vs])
    return {"theta": theta, "beta": phi @ x.T @ v_inv @ y, "phi": phi, "v_inv": v_inv,
            "x": x, "vs": vs, "info": info, "ai": ai, "converged": converged,
            "loglik": _reml_loglik(y, x, v_inv, p)}


def _reml_loglik(y: np.ndarray, x: np.ndarray, v_inv: np.ndarray, p: np.ndarray) -> float:
    """Остаточное лог-правдоподобие с точностью до константы."""
    _, logdet_vinv = np.linalg.slogdet(v_inv)
    _, logdet_xvx = np.linalg.slogdet(x.T @ v_inv @ x)
    return float(-0.5 * (-logdet_vinv + logdet_xvx + y @ p @ y))


def refit_fixed(fit: dict, y: np.ndarray, x: np.ndarray) -> dict:
    """GLS для другой матрицы фиксированных эффектов при тех же компонентах."""
    phi, _ = _projection(fit["v_inv"], x)
    return {**fit, "x": x, "phi": phi, "beta": phi @ x.T @ fit["v_inv"] @ y}


def kenward_roger_df(fit: dict, contrast: np.ndarray) -> float:
    """Знаменательные df Кенварда–Роджера для гипотезы Lᵀβ = 0 (L: p × q)."""
    x, v_inv, phi, vs = fit["x"], fit["v_inv"], fit["phi"], fit["vs"]
    w = np.linalg.pinv(fit["ai"])
    p_mats = [-(x.T @ v_inv @ v @ v_inv @ x) for v in vs]
    q = contrast.shape[1]
    theta_m = contrast @ np.linalg.pinv(contrast.T @ phi @ contrast) @ contrast.T
    m = [theta_m @ phi @ pm @ phi for pm in p_mats]
    r = range(len(vs))
    a1 = sum(w[i, j] * np.trace(m[i]) * np.trace(m[j]) for i in r for j in r)
    a2 = sum(w[i, j] * np.trace(m[i] @ m[j]) for i in r for j in r)
    return _kr_m(a1, a2, q)


def _kr_m(a1: float, a2: float, q: int) -> float:
    """Формулы Kenward & Roger (1997) для оценки df знаменателя."""
    if a2 <= 0:
        return float("inf")
    b = (a1 + 6 * a2) / (2 * q)
    g = ((q + 1) * a1 - (q + 4) * a2) / ((q + 2) * a2)
    d = 3 * q + 2 * (1 - g)
    c1, c2, c3 = g / d, (q - g) / d, (q + 2 - g) / d
    e_star = 1 / (1 - a2 / q)
    v_star = (2 / q) * (1 + c1 * b) / ((1 - c2 * b) ** 2 * (1 - c3 * b))
    rho = v_star / (2 * e_star ** 2)
    denom = q * rho - 1
    return float(4 + (q + 2) / denom) if abs(denom) > 1e-12 else float("inf")


def wald(fit: dict, contrast: np.ndarray) -> float:
    """W = (Lᵀβ̂)ᵀ (LᵀΦL)⁻¹ (Lᵀβ̂)."""
    b = contrast.T @ fit["beta"]
    return float(b @ np.linalg.pinv(contrast.T @ fit["phi"] @ contrast) @ b)


def blups(fit: dict, y: np.ndarray, zs: list[np.ndarray]) -> list[np.ndarray]:
    """eBLUP случайных эффектов: ũ_k = σ²_k Z_kᵀ V⁻¹ (y − Xβ̂) (стр. 436)."""
    resid = fit["v_inv"] @ (y - fit["x"] @ fit["beta"])
    return [fit["theta"][k] * z.T @ resid for k, z in enumerate(zs)]
