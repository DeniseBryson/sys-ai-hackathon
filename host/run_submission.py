#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.aging import AgingConfig, age_rss_episodes
from vlp_hackathon.dataset import DatasetConfig, load_task_split
from vlp_hackathon.metrics import euclidean_errors_cm
from vlp_hackathon.protocol import get_device_info, send_predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a Pico submission on the selected task, split, and clean/raw source. "
            "The hidden test CSV is AES-GCM encrypted and is decrypted in memory only when "
            "a valid test password is available."
        )
    )
    p.add_argument("--task", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--source", choices=["clean", "raw"], default="clean")
    p.add_argument("--clean-3x3-1cm-train", type=Path, default=ROOT / "data" / "train_clean_3x3_1cm.csv")
    p.add_argument("--clean-3x3-1cm-validation", type=Path, default=ROOT / "data" / "validation_clean_3x3_1cm.csv")
    p.add_argument("--raw-3x3-1cm-train", type=Path, default=ROOT / "data" / "train_raw_3x3_1cm.csv")
    p.add_argument("--raw-3x3-1cm-validation", type=Path, default=ROOT / "data" / "validation_raw_3x3_1cm.csv")
    p.add_argument("--clean-6x6-8cm-train", type=Path, default=ROOT / "data" / "train_clean_6x6_8cm.csv")
    p.add_argument("--clean-6x6-8cm-validation", type=Path, default=ROOT / "data" / "validation_clean_6x6_8cm.csv")
    p.add_argument(
        "--hidden-clean-3x3-1cm-test",
        type=Path,
        default=ROOT / "data" / "hidden" / "test_clean_3x3_1cm.csv.aesgcm",
    )
    p.add_argument(
        "--hidden-raw-3x3-1cm-test",
        type=Path,
        default=ROOT / "data" / "hidden" / "test_raw_3x3_1cm.csv.aesgcm",
    )
    p.add_argument(
        "--hidden-clean-6x6-8cm-test",
        type=Path,
        default=ROOT / "data" / "hidden" / "test_clean_6x6_8cm.csv.aesgcm",
    )
    p.add_argument(
        "--test-password",
        default=None,
        help=(
            "Password for the hidden test set. Prefer the VLP_TEST_PASSWORD environment "
            "variable or interactive prompt to avoid shell history."
        ),
    )
    p.add_argument("--port", required=True, help="USB CDC port, e.g. /dev/ttyACM0 or COM5")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42, help="Sampling/shuffle seed; dataset splits are fixed")
    p.add_argument("--shuffle", action="store_true", help="Shuffle selected samples before streaming")
    p.add_argument("--uf2", type=Path, default=None, help="Firmware UF2 for size reporting and flashing")
    p.add_argument(
        "--flash",
        dest="flash",
        action="store_true",
        default=True,
        help="Flash --uf2 with picotool before evaluation (default: enabled)",
    )
    p.add_argument(
        "--no-flash",
        dest="flash",
        action="store_false",
        help="Skip flashing the firmware before evaluation",
    )
    p.add_argument("--picotool", default="picotool")
    p.add_argument("--settle-seconds", type=float, default=3.0)
    p.add_argument("--aging-max-hours", type=float, default=50_000.0)
    p.add_argument("--aging-episodes", type=int, default=10)
    p.add_argument("--aging-seed", type=int, default=123)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def resolve_test_password(args: argparse.Namespace) -> str | None:
    if args.split != "test":
        return None
    password = args.test_password or os.environ.get("VLP_TEST_PASSWORD")
    if password:
        return password
    if sys.stdin.isatty():
        return getpass.getpass("Hidden test password: ")
    raise SystemExit(
        "The test split is encrypted. Set VLP_TEST_PASSWORD, pass --test-password, "
        "or run interactively and enter the test password."
    )


def flash_firmware(args: argparse.Namespace) -> None:
    if not args.flash:
        return
    if args.uf2 is None:
        raise SystemExit("--flash requires --uf2")
    if not args.uf2.exists():
        raise SystemExit(f"UF2 not found: {args.uf2}")
    cmd = [args.picotool, "load", "-f", "-x", str(args.uf2)]
    print("Flashing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    time.sleep(args.settle_seconds)


def summarize_positioning_errors(errors: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}mean_positioning_error_cm": float(errors.mean()),
        f"{prefix}median_positioning_error_cm": float(np.median(errors)),
        f"{prefix}p95_positioning_error_cm": float(np.percentile(errors, 95)),
    }


def summarize_episode_errors(
    errors: np.ndarray,
    episode_ids: np.ndarray,
    episode_hours: np.ndarray,
) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    for episode_id, episode_age_hours in enumerate(episode_hours):
        episode_errors = errors[episode_ids == episode_id]
        summary: dict[str, float | int] = {
            "episode": int(episode_id),
            "samples": int(len(episode_errors)),
            "aging_hours": float(episode_age_hours),
        }
        summary.update(summarize_positioning_errors(episode_errors))
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    password = resolve_test_password(args)

    if args.task == 2 and args.source != "raw":
        print("Warning: Task 2 is normally evaluated with --source raw", file=sys.stderr)
    if args.task in (1, 3, 4) and args.source == "raw":
        print("Warning: this task is normally based on clean data", file=sys.stderr)

    flash_firmware(args)

    cfg = DatasetConfig(
        clean_3x3_1cm_train_path=args.clean_3x3_1cm_train,
        clean_3x3_1cm_validation_path=args.clean_3x3_1cm_validation,
        raw_3x3_1cm_train_path=args.raw_3x3_1cm_train,
        raw_3x3_1cm_validation_path=args.raw_3x3_1cm_validation,
        clean_6x6_8cm_train_path=args.clean_6x6_8cm_train,
        clean_6x6_8cm_validation_path=args.clean_6x6_8cm_validation,
        hidden_clean_3x3_1cm_test_path=args.hidden_clean_3x3_1cm_test,
        hidden_raw_3x3_1cm_test_path=args.hidden_raw_3x3_1cm_test,
        hidden_clean_6x6_8cm_test_path=args.hidden_clean_6x6_8cm_test,
        test_password=password,
    )
    try:
        ds = load_task_split(
            cfg,
            task=args.task,
            split=args.split,
            source=args.source,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    x = ds.x
    y = ds.y_cm
    order = np.arange(len(x))
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(order)
    if args.max_samples > 0:
        order = order[: min(args.max_samples, len(order))]
    x = x[order]
    y = y[order]

    if len(x) == 0:
        raise SystemExit("No samples selected")

    aging_metadata = None
    if args.task == 4:
        aging_metadata = age_rss_episodes(
            x,
            config=AgingConfig(max_hours=args.aging_max_hours, seed=args.aging_seed),
            episode_count=args.aging_episodes,
        )
        x = aging_metadata.x
        print(
            "Task 4 aging: "
            f"{len(aging_metadata.episode_hours)} contiguous episode(s), "
            f"episode ages in [{aging_metadata.episode_hours.min():.1f}, "
            f"{aging_metadata.episode_hours.max():.1f}] h",
            flush=True,
        )

    print(
        f"Evaluating task={args.task} split={args.split} source={args.source} "
        f"samples={len(x)} features={x.shape[1]}",
        flush=True,
    )

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: pip install -r requirements.txt") from exc

    predictions = np.empty((len(x), 2), dtype=np.float32)
    invoke_ms = np.empty(len(x), dtype=np.float64)
    round_trip_ms = np.empty(len(x), dtype=np.float64)

    with serial.Serial(args.port, args.baud, timeout=args.timeout, write_timeout=args.timeout) as ser:
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        info = get_device_info(ser)
        if info.input_features != x.shape[1]:
            raise SystemExit(
                f"Firmware model expects {info.input_features} features, but this task sends "
                f"{x.shape[1]}. Re-export and rebuild the matching model."
            )

        for i, features in enumerate(x):
            t0 = time.perf_counter_ns()
            response = send_predict(ser, features)
            t1 = time.perf_counter_ns()
            predictions[i] = [response.x_cm, response.y_cm]
            invoke_ms[i] = response.invoke_us / 1000.0
            round_trip_ms[i] = (t1 - t0) / 1e6
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(x)} samples", flush=True)

    errors = euclidean_errors_cm(predictions, y)
    result = {
        "task": args.task,
        "split": args.split,
        "source": args.source,
        "samples": int(len(x)),
        "input_features": int(x.shape[1]),
        "mean_device_invoke_ms": float(invoke_ms.mean()),
        "mean_serial_round_trip_ms": float(round_trip_ms.mean()),
        "p95_serial_round_trip_ms": float(np.percentile(round_trip_ms, 95)),
        "tflite_model_bytes": int(info.tflite_model_bytes),
        "tensor_arena_bytes": int(info.tensor_arena_bytes),
        "uf2_firmware_bytes": int(args.uf2.stat().st_size) if args.uf2 and args.uf2.exists() else None,
    }
    result.update(summarize_positioning_errors(errors))
    if aging_metadata is not None:
        episode_metrics = summarize_episode_errors(
            errors,
            aging_metadata.episode_ids,
            aging_metadata.episode_hours,
        )
        episode_mean_errors = np.array(
            [episode["mean_positioning_error_cm"] for episode in episode_metrics],
            dtype=np.float32,
        )
        result.update(
            {
                "aging_episodes": int(len(aging_metadata.episode_hours)),
                "aging_min_episode_hours": float(aging_metadata.episode_hours.min()),
                "aging_max_episode_hours": float(aging_metadata.episode_hours.max()),
                "aging_mean_episode_mean_positioning_error_cm": float(episode_mean_errors.mean()),
                "aging_worst_episode_mean_positioning_error_cm": float(episode_mean_errors.max()),
                "aging_episode_metrics": episode_metrics,
            }
        )

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
