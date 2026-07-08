from __future__ import annotations
from pathlib import Path


def tflite_to_c_array(
    tflite_path: Path,
    cc_path: Path,
    h_path: Path,
    *,
    symbol: str = "g_vlp_model_data",
) -> None:
    data = tflite_path.read_bytes()
    h_path.write_text(
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n\n"
        f"extern const unsigned char {symbol}[];\n"
        f"extern const unsigned int {symbol}_len;\n",
        encoding="utf-8",
    )
    lines = [
        '#include "model_data.h"',
        "",
        f"alignas(16) const unsigned char {symbol}[] = {{",
    ]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int {symbol}_len = {len(data)}u;",
            "",
        ]
    )
    cc_path.write_text("\n".join(lines), encoding="utf-8")
