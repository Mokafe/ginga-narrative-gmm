from __future__ import annotations
import numpy as np
import pandas as pd

LABEL_NAME = {0: "Communal_Happiness", 1: "Self_Protection"}

def auto_name_clusters(preds: pd.DataFrame, K: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Auto-name clusters using labeled anchors.
    Uses posterior-weighted vote for robustness.
    Returns:
      preds2 with cluster_name column
      mapping table DataFrame (cluster, cluster_name, name_confidence)
    """
    preds2 = preds.copy()
    lab = preds2[preds2["label"] >= 0].copy()

    cluster_name = {k: f"Cluster_{k}" for k in range(K)}
    cluster_name_conf = {k: 0.0 for k in range(K)}

    if len(lab) > 0:
        soft_rows = []
        for k in range(K):
            if f"p_{k}" in lab.columns:
                w = lab[f"p_{k}"].to_numpy()
            else:
                w = np.ones(len(lab), dtype=float)
            soft_rows.append({
                "cluster": k,
                0: float(np.sum(w * (lab["label"].to_numpy() == 0))),
                1: float(np.sum(w * (lab["label"].to_numpy() == 1))),
            })
        soft = pd.DataFrame(soft_rows).set_index("cluster")[[0, 1]]

        for k in range(K):
            M = soft.loc[k]
            total = float(M.sum())
            if total <= 0:
                continue
            winner = int(M.idxmax())
            conf = float(M.max() / total)  # 0.5=ambiguous, 1.0=perfect
            cluster_name[k] = LABEL_NAME.get(winner, f"Label_{winner}")
            cluster_name_conf[k] = conf

        # collision resolution (K=2)
        if K == 2 and len(set(cluster_name.values())) == 1:
            ks = [0, 1]
            k_keep = max(ks, key=lambda x: cluster_name_conf[x])
            k_flip = 1 - k_keep
            keep_name = cluster_name[k_keep]
            flip_name = "Self_Protection" if keep_name == "Communal_Happiness" else "Communal_Happiness"
            cluster_name[k_flip] = flip_name

    preds2["cluster_name"] = preds2["cluster"].map(cluster_name)
    map_df = pd.DataFrame([{
        "cluster": k,
        "cluster_name": cluster_name[k],
        "name_confidence": cluster_name_conf[k],
    } for k in range(K)])

    return preds2, map_df

def compute_scene_turning_density(preds: pd.DataFrame, boundary_steps: set[int]) -> pd.DataFrame:
    preds2 = preds.copy()
    preds2["uncertainty"] = 1.0 - preds2["p_max"]
    preds2["is_boundary_topN"] = preds2["global_step"].astype(int).isin(boundary_steps)

    scene_stats = preds2.groupby(["scene_id", "chapter_title"]).agg(
        n_events=("global_step", "count"),
        n_boundary_topN=("is_boundary_topN", "sum"),
        boundary_topN_ratio=("is_boundary_topN", "mean"),
        entropy_sum=("entropy", "sum"),
        entropy_mean=("entropy", "mean"),
        entropy_max=("entropy", "max"),
        uncertainty_mean=("uncertainty", "mean"),
    ).reset_index()

    idx_peak = preds2.groupby("scene_id")["entropy"].idxmax()
    peaks = preds2.loc[idx_peak, [
        "scene_id","chapter_title","global_step","local_step","event","evidence_1",
        "m","iso","cluster","p_max","margin","entropy"
    ]].rename(columns={
        "global_step":"peak_global_step",
        "local_step":"peak_local_step",
        "event":"peak_event",
        "evidence_1":"peak_evidence_1",
    })

    scene_density = scene_stats.merge(peaks, on=["scene_id","chapter_title"], how="left") \
                             .sort_values("entropy_sum", ascending=False)
    return scene_density

def compute_scene_transitions(preds: pd.DataFrame) -> pd.DataFrame:
    seq = preds.sort_values(["scene_id","global_step"]).copy()
    seq["prev_cluster_in_scene"] = seq.groupby("scene_id")["cluster"].shift(1)
    seq["is_transition_in_scene"] = (seq["cluster"] != seq["prev_cluster_in_scene"]) & seq["prev_cluster_in_scene"].notna()

    out = seq.groupby(["scene_id","chapter_title"]).agg(
        n_events=("global_step","count"),
        n_transitions=("is_transition_in_scene","sum"),
    ).reset_index()
    out["transition_rate"] = out["n_transitions"] / (out["n_events"] - 1).clip(lower=1)
    return out

def compute_global_transition_points(preds: pd.DataFrame) -> pd.DataFrame:
    g = preds.sort_values("global_step").copy()
    g["prev_cluster"] = g["cluster"].shift(1)
    g["is_transition"] = (g["cluster"] != g["prev_cluster"]) & g["prev_cluster"].notna()
    tp = g[g["is_transition"]].copy()
    cols = ["global_step","scene_id","chapter_title","local_step","prev_cluster","cluster","p_max","margin","entropy","event","evidence_1"]
    return tp[cols]
