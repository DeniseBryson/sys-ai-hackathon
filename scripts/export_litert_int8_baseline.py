#!/usr/bin/env python3
"""Export the starter MLP to a fully int8 TFLite model and Pico assets."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e import quantize_pt2e

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.baseline_model import BaselineMLP
from vlp_hackathon.dataset import CONF2_3X3_LED_INDICES
from vlp_hackathon.export import tflite_to_c_array


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def load_calibration_inputs(rss_scale: float) -> np.ndarray:
    train_csv = ROOT / "data" / "train_clean_3x3_1cm.csv"
    df = pd.read_csv(train_csv)
    feature_columns = [f"led_{i}" for i in CONF2_3X3_LED_INDICES.tolist()]
    x_raw = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    return (x_raw / rss_scale).astype(np.float32)


def main() -> None:
    state_path = ROOT / "models" / "baseline_task1_int8.pt"
    scaling_path = ROOT / "models" / "baseline_task1_int8_scaling.npz"
    out_path = ROOT / "models" / "baseline_task1_int8.tflite"
    firmware_dir = ROOT / "firmware" / "vlp_serial"
    firmware_path = firmware_dir / "vlp_model.tflite"

    scaling = np.load(scaling_path)
    rss_scale = float(scaling["rss_scale"])

    model = BaselineMLP(9)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    x_calib = load_calibration_inputs(rss_scale)
    example = (torch.from_numpy(x_calib[:1]),)

    exported = torch.export.export(model, example, strict=True).module()
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=False)
    )
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)

    with torch.no_grad():
        for row in x_calib:
            prepared(torch.from_numpy(row[None, :]))

    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    quantized = move_exported_model_to_eval(quantized)
    edge_model = litert_torch.convert(
        quantized,
        sample_args=example,
        strict_export=True,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    tflite_bytes = edge_model.model_content()
    out_path.write_bytes(tflite_bytes)
    firmware_path.write_bytes(tflite_bytes)

    tflite_to_c_array(
        firmware_path,
        firmware_dir / "model_data.cc",
        firmware_dir / "model_data.h",
    )

    target_min_cm = np.asarray(scaling["target_min_cm"], dtype=np.float32)
    target_range_cm = np.asarray(scaling["target_range_cm"], dtype=np.float32)
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
