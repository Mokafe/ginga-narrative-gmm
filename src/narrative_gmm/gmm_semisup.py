from __future__ import annotations
import numpy as np

def logsumexp(a: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    return amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=keepdims))

def log_mvn_pdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """
    Log pdf of multivariate normal N(mu, Sigma) for each row in X.
    Uses Cholesky with jitter escalation for numerical stability.
    """
    d = X.shape[1]
    S = Sigma.copy()
    for _ in range(8):
        try:
            L = np.linalg.cholesky(S)
            break
        except np.linalg.LinAlgError:
            S.flat[::d+1] += jitter
            jitter *= 10
    diff = (X - mu)
    y = np.linalg.solve(L, diff.T)
    maha = np.sum(y * y, axis=0)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)

def e_step(X: np.ndarray, phi: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
    n = X.shape[0]
    K = len(phi)
    logp = np.zeros((n, K), dtype=float)
    for k in range(K):
        logp[:, k] = np.log(phi[k] + 1e-12) + log_mvn_pdf(X, mu[k], Sigma[k])
    logZ = logsumexp(logp, axis=1, keepdims=True)
    w = np.exp(logp - logZ)
    ll = float(np.sum(logZ))
    return w, ll

def m_step_weighted(X: np.ndarray, w: np.ndarray, reg: float = 1e-3, diag: bool = True):
    n, d = X.shape
    K = w.shape[1]
    Nk = np.sum(w, axis=0) + 1e-12
    phi = Nk / np.sum(Nk)
    mu = (w.T @ X) / Nk[:, None]
    Sigma = np.zeros((K, d, d), dtype=float)
    for k in range(K):
        diff = X - mu[k]
        S = (w[:, k, None] * diff).T @ diff / Nk[k]
        if diag:
            S = np.diag(np.diag(S))
        S.flat[::d+1] += reg
        Sigma[k] = S
    return phi, mu, Sigma

def fit_gmm_unsup(
    X: np.ndarray,
    K: int = 2,
    reg: float = 1e-3,
    diag: bool = True,
    seed: int | None = 0,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-4,
):
    """
    Unsupervised EM with multiple random restarts; returns best by log-likelihood.
    If seed=None, randomness is non-deterministic.
    """
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_init):
        mu = X[rng.choice(len(X), size=K, replace=False)]
        S0 = np.cov(X.T) + reg * np.eye(X.shape[1])
        if diag:
            S0 = np.diag(np.diag(S0))
        Sigma = np.stack([S0.copy() for _ in range(K)], axis=0)
        phi = np.ones(K) / K

        prev = None
        for it in range(max_iter):
            w, ll = e_step(X, phi, mu, Sigma)
            phi, mu, Sigma = m_step_weighted(X, w, reg=reg, diag=diag)
            if prev is not None and abs(ll - prev) < tol:
                break
            prev = ll

        if (best is None) or (ll > best["ll"]):
            best = {"phi": phi, "mu": mu, "Sigma": Sigma, "ll": ll, "it": it + 1}

    return best

def fit_gmm_semisup(
    X_u: np.ndarray,
    X_l: np.ndarray,
    y_l: np.ndarray,
    K: int = 2,
    alpha: float = 20.0,
    reg: float = 1e-3,
    diag: bool = True,
    seed: int | None = 0,
    n_init: int = 20,
    max_iter: int = 400,
    tol: float = 1e-4,
):
    """
    Semi-supervised EM:
      - unlabeled responsibilities from E-step
      - labeled anchors injected as alpha * one-hot pseudo-counts in M-step
    """
    X_init = X_u if len(X_u) > 0 else X_l
    init = fit_gmm_unsup(X_init, K=K, reg=reg, diag=diag, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol)
    phi, mu, Sigma = init["phi"], init["mu"], init["Sigma"]

    Y = np.zeros((len(X_l), K), dtype=float)
    for i, lab in enumerate(y_l):
        if 0 <= int(lab) < K:
            Y[i, int(lab)] = 1.0

    prev = None
    for it in range(max_iter):
        w_u, ll_u = e_step(X_u, phi, mu, Sigma) if len(X_u) > 0 else (np.zeros((0, K)), 0.0)
        X_all = np.vstack([X_u, X_l])
        w_all = np.vstack([w_u, alpha * Y])
        phi, mu, Sigma = m_step_weighted(X_all, w_all, reg=reg, diag=diag)

        if prev is not None and abs(ll_u - prev) < tol:
            break
        prev = ll_u

    return {"phi": phi, "mu": mu, "Sigma": Sigma, "it": it + 1}
