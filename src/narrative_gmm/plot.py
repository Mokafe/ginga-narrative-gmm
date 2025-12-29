from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter_entropy(preds: pd.DataFrame, boundary: pd.DataFrame | None = None, title: str = ""):
    plt.figure(figsize=(8,6))
    plt.scatter(preds["m"], preds["iso"], s=30 + 250*preds["entropy"], c=preds["cluster"], alpha=0.8)
    if boundary is not None and len(boundary) > 0:
        plt.scatter(boundary["m"], boundary["iso"], s=120, facecolors="none", edgecolors="black", linewidths=1.6)
    plt.xlabel("Morality (m)")
    plt.ylabel("Isolation (iso)")
    plt.title(title or "Clusters in (m, iso) space (size=entropy)")
    plt.show()

def _draw_cov_ellipse(mean, cov, n_std=2.0, num=200):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    axes = n_std * np.sqrt(np.maximum(vals, 1e-12))
    ellipse = (vecs @ (axes[:,None] * circle)) + mean[:,None]
    return ellipse[0], ellipse[1]

def centers_ellipses(preds: pd.DataFrame, mu: np.ndarray, Sigma: np.ndarray, title: str = ""):
    plt.figure(figsize=(8,6))
    plt.scatter(preds["m"], preds["iso"], s=18, c=preds["cluster"], alpha=0.75)
    K = mu.shape[0]
    for k in range(K):
        plt.scatter(mu[k,0], mu[k,1], s=160, marker="X")
        x_e, y_e = _draw_cov_ellipse(mu[k], Sigma[k], n_std=2.0)
        plt.plot(x_e, y_e, linewidth=2)
    plt.xlabel("Morality (m)")
    plt.ylabel("Isolation (iso)")
    plt.title(title or "Cluster centers (X) + 2σ covariance ellipses")
    plt.show()

def time_series_entropy_and_clusters(preds: pd.DataFrame, use_cluster_name: bool = True):
    preds_time = preds.sort_values("global_step").copy()
    chapter_starts = preds_time.groupby("scene_id")["global_step"].min().sort_values().to_list()

    plt.figure(figsize=(12,3))
    plt.plot(preds_time["global_step"], preds_time["entropy"])
    for s in chapter_starts:
        plt.axvline(s, linewidth=1, alpha=0.25)
    plt.xlabel("global_step")
    plt.ylabel("entropy")
    plt.title("Boundary-ness over time (vertical lines = chapter starts)")
    plt.show()

    if use_cluster_name and "cluster_name" in preds_time.columns:
        order = list(dict.fromkeys(preds_time["cluster_name"].tolist()))
        name_to_y = {nm:i for i,nm in enumerate(order)}
        preds_time["cluster_y"] = preds_time["cluster_name"].map(name_to_y)
        plt.figure(figsize=(12,2))
        plt.scatter(preds_time["global_step"], preds_time["cluster_y"], s=18)
        for s in chapter_starts:
            plt.axvline(s, linewidth=1, alpha=0.25)
        plt.yticks(range(len(order)), order)
        plt.xlabel("global_step")
        plt.title("Cluster meaning over time")
        plt.show()
    else:
        plt.figure(figsize=(12,2))
        plt.scatter(preds_time["global_step"], preds_time["cluster"], s=18)
        for s in chapter_starts:
            plt.axvline(s, linewidth=1, alpha=0.25)
        plt.xlabel("global_step")
        plt.ylabel("cluster")
        plt.title("Cluster sequence over time")
        plt.show()

def chapter_bars(scene_density: pd.DataFrame, scene_transitions: pd.DataFrame):
    dens = scene_density.sort_values("entropy_sum", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12,4))
    plt.bar(range(len(dens)), dens["entropy_sum"])
    plt.xticks(range(len(dens)), dens["scene_id"], rotation=45, ha="right")
    plt.ylabel("entropy_sum (turning density)")
    plt.title("Turning density by chapter")
    plt.tight_layout()
    plt.show()

    tran = scene_transitions.set_index("scene_id").loc[dens["scene_id"]].reset_index()
    plt.figure(figsize=(12,4))
    plt.bar(range(len(tran)), tran["n_transitions"])
    plt.xticks(range(len(tran)), tran["scene_id"], rotation=45, ha="right")
    plt.ylabel("n_transitions")
    plt.title("Cluster transitions by chapter")
    plt.tight_layout()
    plt.show()
