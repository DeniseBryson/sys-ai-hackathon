#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${DEPS_DIR:-$ROOT/.deps}"
BUILD_DIR="${BUILD_DIR:-$ROOT/.build/pico-tflmicro}"
PICO_SDK_TAG="${PICO_SDK_TAG:-2.3.0}"
PICO_BOARD="${PICO_BOARD:-pico}"
PICOTOOL_VERSION="${PICOTOOL_VERSION:-2.3.0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$DEPS_DIR" "$(dirname "$BUILD_DIR")" "$ROOT/firmware/build"

if [[ ! -d "$DEPS_DIR/pico-sdk/.git" ]]; then
  git clone --depth 1 --branch "$PICO_SDK_TAG" \
    https://github.com/raspberrypi/pico-sdk.git "$DEPS_DIR/pico-sdk"
  git -C "$DEPS_DIR/pico-sdk" submodule update --init --depth 1
fi

if [[ ! -d "$DEPS_DIR/pico-tflmicro/.git" ]]; then
  git clone --depth 1 https://github.com/raspberrypi/pico-tflmicro.git \
    "$DEPS_DIR/pico-tflmicro"
fi

FLATBUFFERS_BASE="$DEPS_DIR/pico-tflmicro/src/third_party/flatbuffers/include/flatbuffers/base.h"
if [[ -f "$FLATBUFFERS_BASE" ]] && grep -Fq "#ifndef ARDUINO" "$FLATBUFFERS_BASE"; then
  perl -0pi -e 's/#ifndef ARDUINO\n#include <cstdint>\n#endif/#include <cstdint>/' \
    "$FLATBUFFERS_BASE"
fi

export PICO_SDK_PATH="$DEPS_DIR/pico-sdk"
PICOTOOL_INSTALL_DIR="${PICOTOOL_INSTALL_DIR:-$DEPS_DIR/picotool}"
PICOTOOL_BIN="$PICOTOOL_INSTALL_DIR/picotool/picotool"

PICOTOOL_CMAKE_ARGS=(
  "-DPICOTOOL_FETCH_FROM_GIT_PATH=$PICOTOOL_INSTALL_DIR"
)
if [[ -x "$PICOTOOL_BIN" ]] && "$PICOTOOL_BIN" version "$PICOTOOL_VERSION" >/dev/null 2>&1; then
  PICOTOOL_CMAKE_ARGS+=(
    "-DPICOTOOL_FORCE_FETCH_FROM_GIT=0"
    "-Dpicotool_DIR=$PICOTOOL_INSTALL_DIR/picotool"
  )
else
  PICOTOOL_CMAKE_ARGS+=(
    "-DPICOTOOL_FORCE_FETCH_FROM_GIT=1"
  )
fi

DEST="$DEPS_DIR/pico-tflmicro/examples/vlp_serial"
rm -rf "$DEST"
cp -R "$ROOT/firmware/vlp_serial" "$DEST"

ROOT_CMAKE="$DEPS_DIR/pico-tflmicro/CMakeLists.txt"
LINE='add_subdirectory("examples/vlp_serial")'
if ! grep -Fq "$LINE" "$ROOT_CMAKE"; then
  printf '\n%s\n' "$LINE" >> "$ROOT_CMAKE"
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  CACHED_PICO_BOARD="$(sed -n 's/^PICO_BOARD:STRING=//p' "$BUILD_DIR/CMakeCache.txt" | tail -n 1)"
  if [[ -n "$CACHED_PICO_BOARD" && "$CACHED_PICO_BOARD" != "$PICO_BOARD" ]]; then
    echo "Build directory was configured for PICO_BOARD=$CACHED_PICO_BOARD; reconfiguring for PICO_BOARD=$PICO_BOARD"
    rm -rf "$BUILD_DIR"
  fi
fi

cmake -S "$DEPS_DIR/pico-tflmicro" -B "$BUILD_DIR" \
  -DPICO_SDK_PATH="$PICO_SDK_PATH" \
  -DPICO_BOARD="$PICO_BOARD" \
  "${PICOTOOL_CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target vlp_pico -j "$JOBS"

UF2="$BUILD_DIR/examples/vlp_serial/vlp_pico.uf2"
if [[ ! -f "$UF2" ]]; then
  echo "Build completed but expected UF2 was not found at: $UF2" >&2
  exit 1
fi
cp "$UF2" "$ROOT/firmware/build/vlp_pico.uf2"
echo "Built: $ROOT/firmware/build/vlp_pico.uf2"
