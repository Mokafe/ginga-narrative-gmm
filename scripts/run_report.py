#!/usr/bin/env python3
"""
Narrative GMM Turning Points - batch runner (no notebook required).

Example:
  python scripts/run_report.py --json data/sample/sample_g_1_reconstructed_keep_last9.json --out outputs --plots
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from narrative_gmm.io_json import load_events_df
from narrative_gmm.gmm_semisup import fit_gmm_semisup
from narrative_gmm.metrics import add_posteriors, boundary_topN, boundary_context
from narrative_gmm.report import (
    auto_name_clusters, compute_scene_turning_density, compute_scene_transitions, compute_global_transition_points
)
from narrative_gmm import plot as plot_mod

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to narrative JSON")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--diag-cov", action="store_true", default=True)
    ap.add_argument("--full-cov", dest="diag_cov", action="store_false")
    ap.add_argument("--seed", type=int, default=229)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--context-w", type=int, default=2)
    ap.add_argument("--plots", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load JSON -> flat table
    df = load_events_df(args.json)
    df.to_csv(out_dir / "events_all.csv", index=False, encoding="utf-8-sig")

    # 2) Prepare data
    FEATURES = ["m", "iso"]
    X_all = df[FEATURES].to_numpy()
    mask_l = df["label"].to_numpy() >= 0
    X_l = X_all[mask_l]
    y_l = df.loc[mask_l, "label"].to_numpy().astype(int)
    X_u = X_all[~mask_l]

    # 3) Fit semi-supervised GMM (anchors = labeled points)
    ss = fit_gmm_semisup(
        X_u, X_l, y_l,
        K=args.K, alpha=args.alpha, reg=args.reg, diag=args.diag_cov,
        seed=args.seed, n_init=args.n_init, max_iter=args.max_iter, tol=args.tol
    )
    np.savez(out_dir / "gmm_params.npz", phi=ss["phi"], mu=ss["mu"], Sigma=ss["Sigma"])

    # 4) Predict posteriors + uncertainty metrics
    preds, w_all = add_posteriors(df, X_all, ss["phi"], ss["mu"], ss["Sigma"])

    # 5) Auto-name clusters using anchor labels
    preds, name_map = auto_name_clusters(preds, K=args.K)
    name_map.to_csv(out_dir / "cluster_name_map.csv", index=False, encoding="utf-8-sig")

    preds.to_csv(out_dir / "preds_all.csv", index=False, encoding="utf-8-sig")

    # 6) Boundary TopN (+context)
    bnd = boundary_topN(preds, top_n=args.topn)
    bnd.to_csv(out_dir / "boundary_top20.csv", index=False, encoding="utf-8-sig")
    ctx = boundary_context(preds, bnd, context_w=args.context_w)
    ctx.to_csv(out_dir / "boundary_context.csv", index=False, encoding="utf-8-sig")

    # 7) Chapter-level summaries
    boundary_steps = set(bnd["global_step"].astype(int).tolist())
    scene_density = compute_scene_turning_density(preds, boundary_steps)
    scene_density.to_csv(out_dir / "scene_turning_density.csv", index=False, encoding="utf-8-sig")

    scene_transitions = compute_scene_transitions(preds)
    scene_transitions.to_csv(out_dir / "scene_cluster_transitions.csv", index=False, encoding="utf-8-sig")

    transition_points = compute_global_transition_points(preds)
    transition_points.to_csv(out_dir / "cluster_transition_points.csv", index=False, encoding="utf-8-sig")

    if args.plots:
        plot_mod.scatter_entropy(preds, bnd, title="Clusters (size=entropy, circles=boundary TopN)")
        plot_mod.centers_ellipses(preds, ss["mu"], ss["Sigma"])
        plot_mod.time_series_entropy_and_clusters(preds, use_cluster_name=True)
        plot_mod.chapter_bars(scene_density, scene_transitions)

    print("Saved outputs:")
    for p in sorted(out_dir.glob("*")):
        print(" -", p.as_posix())

if __name__ == "__main__":
    main()
