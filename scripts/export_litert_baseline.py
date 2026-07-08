#!/usr/bin/env python3
"""Export the trained PyTorch starter MLP to TFLite and Pico assets."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import litert_torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.baseline_model import build_mlp
from vlp_hackathon.export import tflite_to_c_array


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def main() -> None:
    state_path = ROOT / "models" / "ourmlp_task1.pt"
    scaling_path = ROOT / "models" / "ourmlp_task1_scaling.npz"
    out_path = ROOT / "models" / "ourmlp_task1.tflite"
    firmware_dir = ROOT / "firmware" / "vlp_serial"
    firmware_path = firmware_dir / "vlp_model.tflite"

    scaling = np.load(scaling_path)
    rss_scale = float(scaling["rss_scale"])
    target_min_cm = np.asarray(scaling["target_min_cm"], dtype=np.float32)
    target_range_cm = np.asarray(scaling["target_range_cm"], dtype=np.float32)
    hidden_sizes = [int(h) for h in scaling["hidden_sizes"]]

    model = build_mlp(9, hidden_sizes)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    example = torch.zeros(1, 9, dtype=torch.float32)
    edge_model = litert_torch.convert(
        model,
        sample_args=(example,),
        strict_export=True,
    )
    tflite_bytes = edge_model.model_content()
    out_path.write_bytes(tflite_bytes)
    firmware_path.write_bytes(tflite_bytes)

    tflite_to_c_array(
        firmware_path,
        firmware_dir / "model_data.cc",
        firmware_dir / "model_data.h",
    )

    (firmware_dir / "preprocess_data.h").write_text(
        "#pragma once\n\n"
        f"constexpr float kRssScale = {c_float(rss_scale)};\n",
        encoding="utf-8",
    )
    (firmware_dir / "target_scale_data.h").write_text(
        "#pragma once\n\n"
        f"constexpr float kTargetXMinCm = {c_float(target_min_cm[0])};\n"
        f"constexpr float kTargetYMinCm = {c_float(target_min_cm[1])};\n"
        f"constexpr float kTargetXRangeCm = {c_float(target_range_cm[0])};\n"
        f"constexpr float kTargetYRangeCm = {c_float(target_range_cm[1])};\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_path} ({len(tflite_bytes)} bytes)")
    print(f"Wrote {firmware_path}")
    print("Updated model_data.cc/.h and preprocessing/target scaling headers")


if __name__ == "__main__":
    main()
