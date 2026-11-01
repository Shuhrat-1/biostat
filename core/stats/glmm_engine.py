"""Движок обобщённых линейных смешанных моделей (GLMM) — экспериментальный.

Модель: g(μ) = Xβ + ZΛu, u ~ N(0, I), Λ = diag(θ_k) для уровней k-го случайного
фактора (случайные интерсепты, скрещённые или вложенные). Распределения —
биномиальное (logit, probit, cloglog) и пуассоновское (log).

- Условная мода u* при данных (θ, β) — PIRLS (Ньютон по штрафному
  правдоподобию h(u) = Σ log f(y | η) − ½‖u‖²).
- Аппроксимация Лапласа, как devfun lme4: −2ℓ = −2Σ log f(y | u*) + ‖u*‖² +
  log det(I + ΛZᵀWZΛ) (контрольное значение теста lme4 elston.R совпадает).
  Z и I + ΛZᵀWZΛ — разреженные (scipy.sparse, LU), log det — по диагонали U.
- Адаптивная квадратура Гаусса–Эрмита (один случайный фактор): интеграл по
  каждой группе отдельно, узлы в моде с масштабом по кривизне; один узел ≡
  Лаплас.
- (θ, β) — максимум приближённого правдоподобия: стадия 1 по θ с совместной
  модой (β, u) (как nAGQ = 0), стадия 2 по (θ, β) (L-BFGS-B, θ ≥ 0). X внутри
  стандартизуется. Ковариация (θ, β) = 2·H⁻¹ численного гессиана −2ℓ (как
  use.hessian в lme4); условная (RX) — через дополнение Шура.

Ни statsmodels, ни scipy не дают частотного GLMM, поэтому это собственный код
со сверкой: 1 узел = Лаплас, θ → 0 = обычная GLM, эталоны lme4 (cbpp, grouseticks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse, special, stats
from scipy.sparse.linalg import splu

_LINKS = ("logit", "probit", "cloglog", "log")
_PIRLS_ITER = 60
_PIRLS_TOL = 1e-10


@dataclass
class GLMMData:
    """Отклик, число испытаний (биномиальное), планы X и Z, принадлежность столбцов Z."""

    y: np.ndarray
    m: np.ndarray
    x: np.ndarray
    z: sparse.csr_matrix
    groups: list[np.ndarray]  # индексы столбцов Z каждого случайного фактора
    family: str
    link: str
    offset: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        self.z = sparse.csr_matrix(self.z)
        if self.offset is None:
            self.offset = np.zeros_like(self.y)
        if self.link not in _LINKS:
            raise ValueError(f"link ∈ {_LINKS}")


# --- Семейство и связь ------------------------------------------------------------


def inverse_link(eta: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    """μ-шкала (доля p или среднее) и dμ/dη."""
    if link == "logit":
        p = special.expit(eta)
        return p, p * (1 - p)
    if link == "probit":
        return stats.norm.cdf(eta), stats.norm.pdf(eta)
    if link == "cloglog":
        e = np.exp(np.clip(eta, -700, 50))
        return -np.expm1(-e), e * np.exp(-e)
    mu = np.exp(np.clip(eta, -700, 300))
    return mu, mu


def loglik_terms(d: GLMMData, eta: np.ndarray) -> np.ndarray:
    """log f(y | η) по наблюдениям, с полной константой (как family$aic в R)."""
    return _loglik(d, d.y, d.m, eta)


def _loglik(d: GLMMData, y: np.ndarray, m: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """log f(y | η) для произвольной формы η (y и m транслируются)."""
    mu, _ = inverse_link(eta, d.link)
    if d.family == "poisson":
        return stats.poisson.logpmf(y, mu)
    return stats.binom.logpmf(y, m, np.clip(mu, 1e-300, 1 - 1e-16))


def deviance_terms(d: GLMMData, eta: np.ndarray) -> np.ndarray:
    """Отклонения d_i от насыщенной модели (Σ d_i — девианс GLM, остатки отклонения)."""
    mu, _ = inverse_link(eta, d.link)
    if d.family == "poisson":
        return 2 * (special.xlogy(d.y, d.y / np.maximum(mu, 1e-300)) - (d.y - mu))
    fit = np.clip(mu, 1e-300, 1 - 1e-16) * d.m
    return 2 * (special.xlogy(d.y, d.y / fit) + special.xlogy(d.m - d.y, (d.m - d.y) /
                                                              np.maximum(d.m - fit, 1e-300)))


def _score_weight(d: GLMMData, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """∂ log f/∂η и ожидаемый вес W = (dμ/dη)²/Var (Фишер-скоринг)."""
    mu, dmu = inverse_link(eta, d.link)
    if d.family == "poisson" and d.link == "log":  # каноническая: без переполнения μ²/μ
        return d.y - mu, mu
    if d.link == "logit":
        return d.y - d.m * mu, d.m * dmu
    if d.family == "poisson":
        var = np.maximum(mu, 1e-300)
        return (d.y - mu) * dmu / var, dmu ** 2 / var
    var = np.maximum(mu * (1 - mu), 1e-300)
    return (d.y - d.m * mu) * dmu / var, d.m * dmu ** 2 / var


# --- Условная мода и Лаплас ----------------------------------------------------------


def lambda_diag(d: GLMMData, theta: np.ndarray) -> np.ndarray:
    """Диагональ Λ: θ_k для столбцов k-го фактора."""
    lam = np.zeros(d.z.shape[1])
    for k, idx in enumerate(d.groups):
        lam[idx] = theta[k]
    return lam


def conditional_mode(d: GLMMData, theta: np.ndarray, beta: np.ndarray,
                     u0: np.ndarray | None = None) -> dict:
    """PIRLS: u* = argmax Σ log f(y | Xβ + ZΛu) − ½‖u‖², с половинным шагом."""
    lam = lambda_diag(d, theta)
    zl = (d.z @ sparse.diags(lam)).tocsr()
    eye = sparse.identity(lam.size, format="csc")
    fixed = d.x @ beta + d.offset
    u = np.zeros(d.z.shape[1]) if u0 is None else u0.copy()

    def objective(v: np.ndarray) -> float:
        return float(loglik_terms(d, fixed + zl @ v).sum() - 0.5 * v @ v)

    current = objective(u)
    for _ in range(_PIRLS_ITER):
        score, w = _score_weight(d, fixed + zl @ u)
        hess = (zl.T @ sparse.diags(w) @ zl + eye).tocsc()
        step = splu(hess).solve(zl.T @ score - u)
        for _ in range(30):
            trial = objective(u + step)
            if trial >= current - 1e-12:
                break
            step /= 2
        u, previous, current = u + step, current, trial
        if abs(current - previous) < _PIRLS_TOL * (1 + abs(current)):
            break
    eta = fixed + zl @ u
    _, w = _score_weight(d, eta)
    hess = (zl.T @ sparse.diags(w) @ zl + eye).tocsc()
    return {"u": u, "eta": eta, "w": w, "hess": hess, "lu": splu(hess), "zl": zl, "lam": lam}


def laplace(d: GLMMData, theta: np.ndarray, beta: np.ndarray,
            u0: np.ndarray | None = None) -> dict:
    """−2ℓ по Лапласу: −2Σ log f(y | u*) + ‖u*‖² + log det(I + ΛZᵀWZΛ) (как devfun lme4)."""
    mode = conditional_mode(d, theta, beta, u0)
    logdet = float(np.log(np.abs(mode["lu"].U.diagonal())).sum())
    uu = float(mode["u"] @ mode["u"])
    m2ll = float(-2 * loglik_terms(d, mode["eta"]).sum()) + uu + logdet
    return {**mode, "m2ll": m2ll}


def agq(d: GLMMData, theta: np.ndarray, beta: np.ndarray, n_nodes: int,
        u0: np.ndarray | None = None) -> dict:
    """Адаптивная квадратура Гаусса–Эрмита для одного случайного фактора (−2ℓ)."""
    if len(d.groups) != 1:
        raise ValueError("Квадратура — только для одного случайного фактора")
    mode = conditional_mode(d, theta, beta, u0)
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    member = np.asarray(d.z.argmax(axis=1)).ravel()  # уровень фактора наблюдения
    scale = np.sqrt(2) / np.sqrt(mode["hess"].diagonal())
    u_nodes = mode["u"][:, None] + scale[:, None] * nodes[None, :]  # уровни × узлы
    eta = (d.x @ beta + d.offset)[:, None] + theta[0] * u_nodes[member]
    h = np.zeros_like(u_nodes)
    np.add.at(h, member, _loglik(d, d.y[:, None], d.m[:, None], eta))
    vals = np.log(weights) + h - 0.5 * u_nodes ** 2 + nodes ** 2
    total = special.logsumexp(vals, axis=1) + np.log(scale) - 0.5 * np.log(2 * np.pi)
    return {**mode, "m2ll": float(-2 * total.sum())}


# --- Оценивание (θ, β) ---------------------------------------------------------------


def joint_mode(d: GLMMData, theta: np.ndarray, beta0: np.ndarray, u0: np.ndarray | None) -> dict:
    """Совместная мода (β, u) при данном θ — как nAGQ = 0 в lme4 (первая стадия)."""
    beta, u = beta0.copy(), (np.zeros(d.z.shape[1]) if u0 is None else u0.copy())
    for _ in range(_PIRLS_ITER):
        mode = conditional_mode(d, theta, beta, u)
        u = mode["u"]
        score, _ = _score_weight(d, mode["eta"])
        step = np.linalg.lstsq(schur(d, mode), d.x.T @ score, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"beta": beta, "u": u}


def schur(d: GLMMData, mode: dict) -> np.ndarray:
    """Информация по β после исключения u: XᵀWX − XᵀWZΛ(ΛZᵀWZΛ + I)⁻¹ΛZᵀWX."""
    xw = d.x * mode["w"][:, None]
    cross = np.asarray(mode["zl"].T @ xw)
    return d.x.T @ xw - cross.T @ mode["lu"].solve(cross)


def _m2ll(d: GLMMData, theta: np.ndarray, beta: np.ndarray, n_agq: int, state: dict) -> float:
    """−2ℓ(θ, β) (Лаплас или квадратура) с тёплым стартом; неудача → +∞."""
    try:
        fit = laplace(d, theta, beta, state.get("u")) if n_agq <= 1 else \
            agq(d, theta, beta, n_agq, state.get("u"))
    except np.linalg.LinAlgError:
        return 1e300
    if not np.isfinite(fit["m2ll"]):
        return 1e300
    state["u"] = fit["u"]
    return fit["m2ll"]


def _stage_one(d: GLMMData, beta0: np.ndarray, theta0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Первая стадия: только θ, β — из совместной моды (дёшево, хороший старт)."""
    state: dict = {"beta": beta0}

    def f(theta: np.ndarray) -> float:
        jm = joint_mode(d, theta, state["beta"], state.get("u"))
        state.update(jm)
        return _m2ll(d, theta, jm["beta"], 1, {"u": jm["u"]})

    res = optimize.minimize(f, theta0, method="L-BFGS-B", bounds=[(0.0, None)] * theta0.size,
                            options={"maxiter": 500, "ftol": 1e-12})
    return res.x, joint_mode(d, res.x, state["beta"], state.get("u"))["beta"]


def fit_glmm(d: GLMMData, beta0: np.ndarray, n_agq: int = 1,
             theta0: np.ndarray | None = None) -> dict:
    """Максимум приближённого правдоподобия по (θ ≥ 0, β): две стадии, как в lme4."""
    k = len(d.groups)
    theta_s, beta_s = _stage_one(d, beta0, np.ones(k) if theta0 is None else theta0)
    state: dict = {}
    bounds = [(0.0, None)] * k + [(None, None)] * beta0.size
    res = optimize.minimize(lambda p: _m2ll(d, p[:k], p[k:], n_agq, state),
                            np.r_[theta_s, beta_s], method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-9,
                                     "maxls": 50})
    theta, beta = res.x[:k], res.x[k:]
    final = laplace(d, theta, beta, state.get("u"))
    m2ll = _m2ll(d, theta, beta, n_agq, {"u": final["u"]})
    return {"theta": theta, "beta": beta, "m2ll": float(m2ll), "u": final["u"],
            "eta": final["eta"], "mode": final, "n_agq": n_agq,
            "converged": bool(res.success) and np.isfinite(m2ll),
            "b": final["lam"] * final["u"]}


def conditional_covariance(d: GLMMData, fit: dict) -> np.ndarray:
    """Var(β̂) из обращённой информации (β, u) в моде при фиксированных θ (RX в lme4)."""
    return np.linalg.pinv(schur(d, fit["mode"]))


def hessian_covariance(d: GLMMData, fit: dict) -> np.ndarray:
    """Ковариация (θ, β) = 2·H⁻¹, H — численный гессиан −2ℓ (как use.hessian в lme4)."""
    params = np.r_[fit["theta"], fit["beta"]]
    k = len(d.groups)
    state = {"u": fit["u"]}

    def f(p: np.ndarray) -> float:
        return _m2ll(d, np.abs(p[:k]), p[k:], fit["n_agq"], state)

    h = 1e-4 * np.maximum(1.0, np.abs(params))
    n = params.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h[i], np.eye(n)[j] * h[j]
            hess[i, j] = hess[j, i] = (f(params + ei + ej) - f(params + ei - ej)
                                       - f(params - ei + ej) + f(params - ei - ej)) / (4 * h[i] * h[j])
    return 2 * np.linalg.pinv(hess)


def scaling_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X_s со стандартизованными непостоянными столбцами и M: β = M·β_s (столбец 0 — [1])."""
    centre = np.where(np.ptp(x, axis=0) > 0, x.mean(axis=0), 0.0)
    scale = np.where(np.ptp(x, axis=0) > 0, x.std(axis=0), 1.0)
    centre[0], scale[0] = 0.0, 1.0
    xs = (x - centre) / scale
    m = np.diag(1 / scale)
    m[0, 1:] = -centre[1:] / scale[1:]
    return xs, m


def fit_scaled(d: GLMMData, beta0: np.ndarray, n_agq: int = 1) -> dict:
    """Подгонка на стандартизованном X (устойчивость), оценки и ковариации — в исходной шкале."""
    xs, m = scaling_matrix(d.x)
    ds = GLMMData(d.y, d.m, xs, d.z, d.groups, d.family, d.link, d.offset)
    fit = fit_glmm(ds, np.linalg.solve(m, beta0), n_agq)
    k = len(d.groups)
    cov = hessian_covariance(ds, fit)
    t = np.zeros((k + m.shape[0],) * 2)
    t[:k, :k], t[k:, k:] = np.eye(k), m
    return {**fit, "beta": m @ fit["beta"], "cov": t @ cov @ t.T,
            "cov_conditional": m @ conditional_covariance(ds, fit) @ m.T, "data": ds}
