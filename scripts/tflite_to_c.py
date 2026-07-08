#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.export import tflite_to_c_array


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a .tflite flatbuffer into Pico C++ source/header files")
    p.add_argument("tflite", type=Path)
    p.add_argument("--cc", type=Path, default=ROOT / "firmware" / "vlp_serial" / "model_data.cc")
    p.add_argument("--header", type=Path, default=ROOT / "firmware" / "vlp_serial" / "model_data.h")
    args = p.parse_args()
    tflite_to_c_array(args.tflite, args.cc, args.header)
    print(f"Wrote {args.cc}")
    print(f"Wrote {args.header}")


if __name__ == "__main__":
    main()
