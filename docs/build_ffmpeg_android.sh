#!/usr/bin/env bash
# =============================================================================
# docs/build_ffmpeg_android.sh
# =============================================================================
# Cross-compile a stripped-down static FFmpeg for Android ARM64 using the
# Android NDK. The output is consumed by media_core_ffi/build.dart when
# building the Flutter app for Android.
#
# Requirements
# ------------
#   - Run on Linux or macOS (WSL2 on Windows is supported)
#   - Android NDK r26b installed (set NDK variable below)
#   - ~2 GB free disk space for the build
#
# Usage
# -----
#   chmod +x docs/build_ffmpeg_android.sh
#   ./docs/build_ffmpeg_android.sh
#
# Output
# ------
#   native/android/arm64/
#     include/   ← FFmpeg headers
#     lib/       ← libavformat.a, libavcodec.a, libswresample.a, libavutil.a
#
# Set FFMPEG_ANDROID_ARM64_DIR=<project_root>/native/android/arm64 in .env
# before running `flutter build apk`.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
NDK="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/26.3.11579264}"
FFMPEG_VERSION="6.1.1"
FFMPEG_SRC="ffmpeg-${FFMPEG_VERSION}"
FFMPEG_ARCHIVE="${FFMPEG_SRC}.tar.bz2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/native/android/arm64"

TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/linux-x86_64"
SYSROOT="${TOOLCHAIN}/sysroot"
TARGET="aarch64-linux-android"
API_LEVEL=26

echo "============================================================"
echo "  FFmpeg ${FFMPEG_VERSION} — Android ARM64 Static Build"
echo "  NDK        : ${NDK}"
echo "  Toolchain  : ${TOOLCHAIN}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

# ── Download FFmpeg source ─────────────────────────────────────────────────────
if [ ! -d "${FFMPEG_SRC}" ]; then
    echo "[1/4] Downloading FFmpeg ${FFMPEG_VERSION}…"
    wget -q "https://ffmpeg.org/releases/${FFMPEG_ARCHIVE}"
    tar -xf "${FFMPEG_ARCHIVE}"
fi

cd "${FFMPEG_SRC}"

# ── Configure ─────────────────────────────────────────────────────────────────
echo "[2/4] Configuring for Android ARM64…"
./configure \
    --cross-prefix="${TOOLCHAIN}/bin/${TARGET}${API_LEVEL}-" \
    --cc="${TOOLCHAIN}/bin/${TARGET}${API_LEVEL}-clang" \
    --cxx="${TOOLCHAIN}/bin/${TARGET}${API_LEVEL}-clang++" \
    --target-os=android \
    --arch=aarch64 \
    --cpu=armv8-a \
    --sysroot="${SYSROOT}" \
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
    --enable-cross-compile \
    --extra-cflags="-fPIC -DANDROID -D__ANDROID_API__=${API_LEVEL}" \
    --extra-ldflags="-lz -ldl" \
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
echo "  Add to .env:"
echo "  FFMPEG_ANDROID_ARM64_DIR=${OUTPUT_DIR}"
echo "============================================================"
