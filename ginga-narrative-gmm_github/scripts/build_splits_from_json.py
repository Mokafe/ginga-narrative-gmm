#!/usr/bin/env python3
"""
Build train/valid/test CSV splits from narrative JSON.

Default split_mode='scene' avoids leakage across a chapter (scene_id).
Example:
  python scripts/build_splits_from_json.py --json data/sample/sample_g_1_reconstructed_keep_last9.json --out data/derived
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from narrative_gmm.io_json import load_events_df

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/derived")
    ap.add_argument("--split-mode", choices=["scene","time","random"], default="scene")
    ap.add_argument("--train", type=float, default=0.6)
    ap.add_argument("--valid", type=float, default=0.2)
    ap.add_argument("--test",  type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=229)
    ap.add_argument("--numeric-only", action="store_true", help="Keep only numeric columns for ML")
    return ap.parse_args()

def _ratios_ok(a,b,c):
    return abs((a+b+c)-1.0) < 1e-6

def main():
    args = parse_args()
    if not _ratios_ok(args.train, args.valid, args.test):
        raise ValueError("train+valid+test must sum to 1.0")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_events_df(args.json)

    if args.numeric_only:
        keep = ["global_step","scene_id","m","iso","label"]
        df = df[keep].copy()

    rng = np.random.default_rng(args.seed)

    if args.split_mode == "scene":
        scenes = sorted(df["scene_id"].unique().tolist())
        # deterministic shuffle if seed fixed
        scenes = list(np.array(scenes)[rng.permutation(len(scenes))])
        n = len(scenes)
        n_train = max(1, int(round(n * args.train)))
        n_valid = max(1, int(round(n * args.valid)))
        n_test = max(1, n - n_train - n_valid)

        # adjust if rounding breaks
        while n_train + n_valid + n_test > n:
            n_train = max(1, n_train - 1)
        while n_train + n_valid + n_test < n:
            n_test += 1

        tr = set(scenes[:n_train])
        va = set(scenes[n_train:n_train+n_valid])
        te = set(scenes[n_train+n_valid:n_train+n_valid+n_test])

        train_df = df[df["scene_id"].isin(tr)].copy()
        valid_df = df[df["scene_id"].isin(va)].copy()
        test_df  = df[df["scene_id"].isin(te)].copy()

    elif args.split_mode == "time":
        df2 = df.sort_values("global_step").reset_index(drop=True)
        n = len(df2)
        n_train = int(round(n * args.train))
        n_valid = int(round(n * args.valid))
        train_df = df2.iloc[:n_train].copy()
        valid_df = df2.iloc[n_train:n_train+n_valid].copy()
        test_df  = df2.iloc[n_train+n_valid:].copy()

    else:  # random (point-wise)
        idx = rng.permutation(len(df))
        n = len(df)
        n_train = int(round(n * args.train))
        n_valid = int(round(n * args.valid))
        tr_idx = idx[:n_train]
        va_idx = idx[n_train:n_train+n_valid]
        te_idx = idx[n_train+n_valid:]
        train_df = df.iloc[tr_idx].copy()
        valid_df = df.iloc[va_idx].copy()
        test_df  = df.iloc[te_idx].copy()

    train_df.to_csv(out_dir/"train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(out_dir/"valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir/"test.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print(" -", (out_dir/"train.csv").as_posix(), len(train_df))
    print(" -", (out_dir/"valid.csv").as_posix(), len(valid_df))
    print(" -", (out_dir/"test.csv").as_posix(), len(test_df))

if __name__ == "__main__":
    main()
