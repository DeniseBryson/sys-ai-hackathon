#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a public VLP CSV schema")
    p.add_argument("path", type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.path)
    if "x" not in df.columns or "y" not in df.columns:
        raise SystemExit("Dataset must contain x and y columns")

    led_columns = [c for c in df.columns if c.startswith("led_")]
    if len(led_columns) not in (9, 36):
        raise SystemExit(f"Expected 9 or 36 LED columns, found {len(led_columns)}")

    rss = df[led_columns].to_numpy(np.float32)
    coords = df[["x", "y"]].to_numpy(np.float32)
    print(f"rows={len(df)} columns={len(df.columns)} led_channels={len(led_columns)}")
    print(f"rss_shape={rss.shape} rss_min={np.nanmin(rss):.6g} rss_max={np.nanmax(rss):.6g}")
    print(f"x_mm=[{coords[:, 0].min():.0f}, {coords[:, 0].max():.0f}]")
    print(f"y_mm=[{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}]")
    print(f"NaNs={np.isnan(rss).sum()} infs={np.isinf(rss).sum()}")
    if np.isnan(rss).any() or np.isinf(rss).any():
        raise SystemExit("RSS data contain NaN or inf values")


if __name__ == "__main__":
    main()
