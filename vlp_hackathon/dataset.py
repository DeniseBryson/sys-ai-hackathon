from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .security import decrypt_file_to_bytes

SplitName = Literal["train", "validation", "test"]
SourceName = Literal["clean", "raw"]

CONF2_3X3_LED_INDICES = np.array([0, 2, 4, 12, 14, 16, 24, 26, 28], dtype=np.int64)
ALL_6X6_LED_INDICES = np.arange(36, dtype=np.int64)


@dataclass(frozen=True)
class DatasetConfig:
    clean_3x3_1cm_train_path: Path
    clean_3x3_1cm_validation_path: Path
    raw_3x3_1cm_train_path: Path
    raw_3x3_1cm_validation_path: Path
    clean_6x6_8cm_train_path: Path
    clean_6x6_8cm_validation_path: Path
    hidden_clean_3x3_1cm_test_path: Path | None = None
    hidden_raw_3x3_1cm_test_path: Path | None = None
    hidden_clean_6x6_8cm_test_path: Path | None = None
    test_password: str | None = None


@dataclass
class LoadedSplit:
    x: np.ndarray
    y_cm: np.ndarray
    row_ids: np.ndarray
    led_indices: np.ndarray
    task: int
    split: SplitName
    source: SourceName


def _validate_frame(df: pd.DataFrame, led_indices: np.ndarray) -> None:
    required = ["x", "y", *[f"led_{i}" for i in led_indices.tolist()]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Dataset is empty")


def _read_csv(path: Path, led_indices: np.ndarray) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    _validate_frame(df, led_indices)
    return df


def _read_encrypted_csv(path: Path, password: str | None, led_indices: np.ndarray) -> pd.DataFrame:
    if password is None or password == "":
        raise ValueError("A test password is required for the hidden test split")
    plaintext = decrypt_file_to_bytes(path, password)
    # Keep plaintext in memory only. No decrypted CSV is written to disk.
    df = pd.read_csv(BytesIO(plaintext))
    _validate_frame(df, led_indices)
    return df


def _variant_for_task(task: int, source: SourceName) -> tuple[str, np.ndarray]:
    """Return the fixed dataset variant and model input channels."""
    if task in (1, 2):
        if source == "clean":
            return "clean_3x3_1cm", CONF2_3X3_LED_INDICES
        return "raw_3x3_1cm", CONF2_3X3_LED_INDICES

    if task == 3:
        if source != "clean":
            raise ValueError("Task 3 is provided only as the clean 8 cm 6x6 dataset")
        return "clean_6x6_8cm", ALL_6X6_LED_INDICES

    if task == 4:
        if source != "clean":
            raise ValueError("Task 4 starts from the clean 3x3 dataset before aging is applied")
        return "clean_3x3_1cm", CONF2_3X3_LED_INDICES

    raise ValueError("task must be one of 1, 2, 3, 4")


def _public_path(config: DatasetConfig, variant: str, split: SplitName) -> Path:
    if split == "test":
        raise ValueError("_public_path cannot resolve the hidden test split")
    mapping = {
        ("clean_3x3_1cm", "train"): config.clean_3x3_1cm_train_path,
        ("clean_3x3_1cm", "validation"): config.clean_3x3_1cm_validation_path,
        ("raw_3x3_1cm", "train"): config.raw_3x3_1cm_train_path,
        ("raw_3x3_1cm", "validation"): config.raw_3x3_1cm_validation_path,
        ("clean_6x6_8cm", "train"): config.clean_6x6_8cm_train_path,
        ("clean_6x6_8cm", "validation"): config.clean_6x6_8cm_validation_path,
    }
    return mapping[(variant, split)]


def _hidden_path(config: DatasetConfig, variant: str) -> Path:
    mapping = {
        "clean_3x3_1cm": config.hidden_clean_3x3_1cm_test_path,
        "raw_3x3_1cm": config.hidden_raw_3x3_1cm_test_path,
        "clean_6x6_8cm": config.hidden_clean_6x6_8cm_test_path,
    }
    path = mapping[variant]
    if path is None:
        raise ValueError(f"No encrypted hidden test path configured for {variant}")
    return path


def load_task_split(
    config: DatasetConfig,
    *,
    task: int,
    split: SplitName,
    source: SourceName,
) -> LoadedSplit:
    """Load one fixed benchmark split.

    The package already contains separate train and validation CSVs. The loader does
    not create or reshuffle splits at runtime. Hidden test CSVs are decrypted only in
    memory when the runner receives the test password.

    Coordinates in the CSV files are millimeters and are converted to cm.
    """
    variant, led_indices = _variant_for_task(task, source)

    if split == "test":
        df = _read_encrypted_csv(
            _hidden_path(config, variant),
            config.test_password,
            led_indices,
        )
    else:
        df = _read_csv(_public_path(config, variant, split), led_indices)

    feature_columns = [f"led_{i}" for i in led_indices.tolist()]
    x = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y_cm = df[["x", "y"]].to_numpy(dtype=np.float32, copy=True) / 10.0

    return LoadedSplit(
        x=x,
        y_cm=y_cm,
        row_ids=np.arange(len(df), dtype=np.int64),
        led_indices=led_indices.copy(),
        task=task,
        split=split,
        source=source,
    )
