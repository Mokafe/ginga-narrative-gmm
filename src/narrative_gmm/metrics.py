from __future__ import annotations
import numpy as np
import pandas as pd
from .gmm_semisup import e_step

def add_posteriors(df: pd.DataFrame, X: np.ndarray, phi, mu, Sigma) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Add posterior-derived fields:
      cluster, p_max, margin, entropy, p_k...
    """
    w_all, ll = e_step(X, phi, mu, Sigma)
    cluster = w_all.argmax(axis=1)
    p_sorted = np.sort(w_all, axis=1)
    p_max = p_sorted[:, -1]
    margin = (p_sorted[:, -1] - p_sorted[:, -2]) if w_all.shape[1] >= 2 else p_sorted[:, -1]
    entropy = -np.sum(w_all * np.log(w_all + 1e-12), axis=1)

    out = df.copy()
    out["cluster"] = cluster
    out["p_max"] = p_max
    out["margin"] = margin
    out["entropy"] = entropy
    for k in range(w_all.shape[1]):
        out[f"p_{k}"] = w_all[:, k]
    return out, w_all

def boundary_topN(preds: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Boundary points = low margin (ambiguous) and high entropy (uncertain).
    """
    return preds.sort_values(["margin", "entropy"], ascending=[True, False]).head(top_n).copy()

def boundary_context(preds: pd.DataFrame, boundary: pd.DataFrame, context_w: int = 2) -> pd.DataFrame:
    """
    Context window ±context_w steps around each boundary center step.
    """
    ctx_rows = []
    for _, r in boundary.iterrows():
        gs = int(r["global_step"])
        ctx = preds[(preds["global_step"] >= gs - context_w) & (preds["global_step"] <= gs + context_w)].copy()
        ctx["boundary_center_step"] = gs
        ctx_rows.append(ctx)
    return pd.concat(ctx_rows, ignore_index=True) if ctx_rows else preds.iloc[0:0].copy()
