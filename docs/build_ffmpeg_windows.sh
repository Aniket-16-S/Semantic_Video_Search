#!/usr/bin/env bash
# =============================================================================
# docs/build_ffmpeg_windows.sh
# =============================================================================
# Build a stripped-down static FFmpeg for Windows x64 using MinGW-w64.
# Run this script inside MSYS2 MinGW 64-bit shell (NOT a standard CMD/PS window).
#
# Requirements
# ------------
#   MSYS2 with MinGW-w64 toolchain:
#     pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make
#   ~2 GB free disk space
#
# Usage (inside MSYS2 MinGW64 shell)
# ------------------------------------
#   chmod +x docs/build_ffmpeg_windows.sh
#   ./docs/build_ffmpeg_windows.sh
#
# Output
# ------
#   native/windows/x64/
#     include/   ← FFmpeg headers
#     lib/       ← libavformat.a, libavcodec.a, libswresample.a, libavutil.a
#
# After building, set in .env:
#   FFMPEG_WINDOWS_X64_DIR=<project_root>/native/windows/x64
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
FFMPEG_VERSION="6.1.1"
FFMPEG_SRC="ffmpeg-${FFMPEG_VERSION}"
FFMPEG_ARCHIVE="${FFMPEG_SRC}.tar.bz2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/native/windows/x64"

echo "============================================================"
echo "  FFmpeg ${FFMPEG_VERSION} — Windows x64 Static Build (MinGW)"
echo "  Output : ${OUTPUT_DIR}"
echo "============================================================"

# ── Download ──────────────────────────────────────────────────────────────────
if [ ! -d "${FFMPEG_SRC}" ]; then
    echo "[1/4] Downloading FFmpeg ${FFMPEG_VERSION}…"
    curl -L -o "${FFMPEG_ARCHIVE}" \
        "https://ffmpeg.org/releases/${FFMPEG_ARCHIVE}"
    tar -xf "${FFMPEG_ARCHIVE}"
fi

cd "${FFMPEG_SRC}"

# ── Configure ─────────────────────────────────────────────────────────────────
echo "[2/4] Configuring for Windows x64…"
./configure \
    --cross-prefix=x86_64-w64-mingw32- \
    --arch=x86_64 \
    --target-os=mingw32 \
    --enable-static \
    --disable-shared \
    --disable-programs \
    --disable-doc \
    --disable-everything \
    --enable-protocol=file \
    --enable-demuxer=mov,mp4,mp3,wav,aac,matroska,avi \
    --enable-decoder=aac,mp3,flac,pcm_s16le,pcm_s16be,vorbis,opus,h264,hevc \
    --enable-parser=aac,mp3,flac,h264,hevc \
    --enable-filter=aresample \
    --enable-swresample \
    --disable-avfilter \
    --disable-avdevice \
    --disable-postproc \
    --extra-ldflags="-static -lws2_32 -lbcrypt" \
    --prefix="${OUTPUT_DIR}"

# ── Build ──────────────────────────────────────────────────────────────────────
echo "[3/4] Building (this takes 5–10 minutes)…"
make -j"$(nproc)" 2>&1 | tail -20

# ── Install ───────────────────────────────────────────────────────────────────
echo "[4/4] Installing to ${OUTPUT_DIR}…"
make install

echo ""
echo "============================================================"
echo "  Build complete."
echo "  Static libs : ${OUTPUT_DIR}/lib/"
echo "  Headers     : ${OUTPUT_DIR}/include/"
echo ""
echo "  Add to .env (or set as environment variable before flutter build):"
echo "  FFMPEG_WINDOWS_X64_DIR=${OUTPUT_DIR}"
echo "============================================================"
