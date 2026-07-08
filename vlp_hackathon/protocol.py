from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

REQUEST_MAGIC = b"VLP1"
RESPONSE_MAGIC = b"VLR1"
INFO_MAGIC = b"VLI1"

CMD_PREDICT = 1
CMD_INFO = 2
STATUS_OK = 0

# request: magic[4], command u8, n_features u16, reserved u8
REQUEST_HEADER = struct.Struct("<4sBHB")
# response: magic[4], status u8, reserved[3], x f32, y f32, invoke_us u32
PREDICT_RESPONSE = struct.Struct("<4sB3xffI")
# info response: magic[4], status u8, input_features u16, reserved u8,
#                tflite_model_bytes u32, tensor_arena_bytes u32
INFO_RESPONSE = struct.Struct("<4sBHxII")


@dataclass(frozen=True)
class PredictionResponse:
    x_cm: float
    y_cm: float
    invoke_us: int


@dataclass(frozen=True)
class DeviceInfo:
    input_features: int
    tflite_model_bytes: int
    tensor_arena_bytes: int


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise TimeoutError(f"Serial read ended with {remaining} bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_predict(stream: BinaryIO, features: np.ndarray) -> PredictionResponse:
    features = np.asarray(features, dtype="<f4").reshape(-1)
    if len(features) > 65535:
        raise ValueError("Too many input features")
    stream.write(REQUEST_HEADER.pack(REQUEST_MAGIC, CMD_PREDICT, len(features), 0))
    stream.write(features.tobytes(order="C"))
    if hasattr(stream, "flush"):
        stream.flush()
    raw = _read_exact(stream, PREDICT_RESPONSE.size)
    magic, status, x_cm, y_cm, invoke_us = PREDICT_RESPONSE.unpack(raw)
    if magic != RESPONSE_MAGIC:
        raise RuntimeError(f"Bad response magic {magic!r}")
    if status != STATUS_OK:
        raise RuntimeError(f"Pico returned status {status}")
    return PredictionResponse(float(x_cm), float(y_cm), int(invoke_us))


def get_device_info(stream: BinaryIO) -> DeviceInfo:
    stream.write(REQUEST_HEADER.pack(REQUEST_MAGIC, CMD_INFO, 0, 0))
    if hasattr(stream, "flush"):
        stream.flush()
    raw = _read_exact(stream, INFO_RESPONSE.size)
    magic, status, input_features, model_bytes, arena_bytes = INFO_RESPONSE.unpack(raw)
    if magic != INFO_MAGIC:
        raise RuntimeError(f"Bad info magic {magic!r}")
    if status != STATUS_OK:
        raise RuntimeError(f"Pico returned status {status}")
    return DeviceInfo(int(input_features), int(model_bytes), int(arena_bytes))
