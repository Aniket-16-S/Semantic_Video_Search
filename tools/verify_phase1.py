"""
tools/verify_phase1.py
======================
Phase 1 milestone verification runner.

Executes all three gating checks required to close out Phase 1:

  Check 1 — Graph Shape Verification
    Load siglip_vision_int8.onnx via ORT.
    Assert output shape is [1, 512] or [1, 768].
    Assert ONNX Runtime actually runs the model without error.

  Check 2 — Size Budget Assertions
    siglip_vision_int8.onnx  ≤ 300 MB
    whisper_tiny_int8/ total  ≤  50 MB  (sum of all .onnx files)
    ppocr_det_int8.onnx       ≤  10 MB
    ppocr_rec_int8.onnx       ≤  15 MB

  Check 3 — C Library Smoke Test  (optional — requires compiled .dll/.so)
    Load media_core.dll (Windows) or libmedia_core.so (Android) via ctypes.
    Call extract_audio_pcm() on a real video file.
    Assert sample_count > 0.
    Call free_audio_buffer() and assert no crash (memory safety).

Usage
-----
    # Run all checks (C lib test skipped if --native-lib not provided)
    python tools/verify_phase1.py

    # Full run including C lib
    python tools/verify_phase1.py \\
        --siglip       assets/models/siglip_vision_int8.onnx \\
        --whisper-dir  assets/models/whisper_tiny_int8/ \\
        --det          assets/models/ppocr_det_int8.onnx \\
        --rec          assets/models/ppocr_rec_int8.onnx \\
        --native-lib   assets/native/windows/x64/media_core.dll \\
        --test-video   path/to/sample.mp4

Exit codes
----------
  0 — All requested checks passed
  1 — One or more checks failed
"""

import argparse
import ctypes
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verify_phase1")

# ── Project paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ASSETS_DIR   = _PROJECT_ROOT / "assets" / "models"

# ── Default paths ──────────────────────────────────────────────────────────────
_DEFAULT_SIGLIP      = str(_ASSETS_DIR / "siglip_vision_int8.onnx")
_DEFAULT_WHISPER_DIR = str(_ASSETS_DIR / "whisper_tiny_int8")
_DEFAULT_DET         = str(_ASSETS_DIR / "ppocr_det_int8.onnx")
_DEFAULT_REC         = str(_ASSETS_DIR / "ppocr_rec_int8.onnx")

# ── Size budgets (MB) ──────────────────────────────────────────────────────────
_SIGLIP_BUDGET  = 300.0
_WHISPER_BUDGET =  50.0
_DET_BUDGET     =  10.0
_REC_BUDGET     =  15.0


# ══════════════════════════════════════════════════════════════════════════════
# Result tracker
# ══════════════════════════════════════════════════════════════════════════════

class Results:
    def __init__(self):
        self._results: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self._results.append((name, passed, detail))
        icon = "✓ PASS" if passed else "✗ FAIL"
        msg  = f"[{icon}] {name}"
        if detail:
            msg += f" — {detail}"
        if passed:
            log.info(msg)
        else:
            log.error(msg)

    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        log.info("")
        log.info("=" * 60)
        log.info("  Phase 1 Verification Summary")
        log.info("=" * 60)
        all_pass = True
        for name, passed, detail in self._results:
            icon = "✓" if passed else "✗"
            log.info("  %s  %s", icon, name)
            if not passed and detail:
                log.info("      → %s", detail)
            if not passed:
                all_pass = False
        log.info("=" * 60)
        if all_pass:
            log.info("  All checks PASSED ✓")
        else:
            n_fail = sum(1 for _, p, _ in self._results if not p)
            log.error("  %d check(s) FAILED ✗", n_fail)
        return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# Check 1 — SigLIP graph shape
# ══════════════════════════════════════════════════════════════════════════════

def check_siglip_shape(siglip_path: str, results: Results) -> None:
    """Load SigLIP vision INT8 and assert output is [1, 512|768]."""
    name = "SigLIP Vision — Graph Shape"

    if not os.path.exists(siglip_path):
        results.record(name, False, f"File not found: {siglip_path}")
        return

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(siglip_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        out   = sess.run(["image_embeds"], {"pixel_values": dummy})
        shape = out[0].shape

        if len(shape) == 2 and shape[0] == 1 and shape[1] in (512, 768):
            results.record(name, True, f"output shape = {shape}")
        else:
            results.record(name, False, f"unexpected shape {shape} — expected (1,512) or (1,768)")

    except Exception as exc:
        results.record(name, False, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Check 2 — Size budgets
# ══════════════════════════════════════════════════════════════════════════════

def _file_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else -1.0


def _dir_onnx_mb(directory: str) -> float:
    if not os.path.isdir(directory):
        return -1.0
    total = sum(
        f.stat().st_size for f in Path(directory).glob("*.onnx")
    )
    return total / 1e6


def check_sizes(
    siglip_path: str,
    whisper_dir: str,
    det_path: str,
    rec_path: str,
    results: Results,
) -> None:
    checks = [
        ("SigLIP vision size", siglip_path,  _file_mb,     _SIGLIP_BUDGET),
        ("Whisper Tiny size",  whisper_dir,  _dir_onnx_mb, _WHISPER_BUDGET),
        ("PP-OCR DET size",    det_path,     _file_mb,     _DET_BUDGET),
        ("PP-OCR REC size",    rec_path,     _file_mb,     _REC_BUDGET),
    ]

    for name, path, measure_fn, budget in checks:
        mb = measure_fn(path)
        if mb < 0:
            results.record(name, False, f"not found: {path}")
        elif mb <= budget:
            results.record(name, True, f"{mb:.1f} MB ≤ {budget:.0f} MB")
        else:
            results.record(name, False, f"{mb:.1f} MB EXCEEDS {budget:.0f} MB limit")


# ══════════════════════════════════════════════════════════════════════════════
# Check 3 — C library smoke test
# ══════════════════════════════════════════════════════════════════════════════

class _AudioBufferResult(ctypes.Structure):
    """Mirrors the AudioBufferResult C struct for ctypes."""
    _fields_ = [
        ("data",         ctypes.POINTER(ctypes.c_int16)),
        ("sample_count", ctypes.c_int),
    ]


def check_native_lib(lib_path: str, test_video: str, results: Results) -> None:
    """
    Load media_core shared library, call extract_audio_pcm, verify output.

    Memory safety check: calls free_audio_buffer after consuming samples.
    If the program does not crash after free_audio_buffer, the basic memory
    contract is confirmed (valgrind / ASAN should be used for deeper analysis).
    """
    load_name   = "C library load"
    call_name   = "C library — extract_audio_pcm"
    memory_name = "C library — free_audio_buffer (no crash)"

    if not os.path.exists(lib_path):
        for n in (load_name, call_name, memory_name):
            results.record(n, False, f"Library not found: {lib_path}")
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        lib = ctypes.CDLL(lib_path)
        results.record(load_name, True, os.path.basename(lib_path))
    except OSError as exc:
        results.record(load_name, False, str(exc))
        for n in (call_name, memory_name):
            results.record(n, False, "skipped — library failed to load")
        return

    # ── Configure function signatures ─────────────────────────────────────────
    lib.extract_audio_pcm.restype  = _AudioBufferResult
    lib.extract_audio_pcm.argtypes = [ctypes.c_char_p]

    lib.free_audio_buffer.restype  = None
    lib.free_audio_buffer.argtypes = [ctypes.POINTER(ctypes.c_int16)]

    # ── Call ──────────────────────────────────────────────────────────────────
    if not test_video or not os.path.exists(test_video):
        results.record(call_name, False,
                       f"Test video not found: {test_video} (pass --test-video)")
        results.record(memory_name, False, "skipped — no test video")
        return

    try:
        video_bytes = test_video.encode("utf-8")
        buf = lib.extract_audio_pcm(video_bytes)

        if buf.sample_count > 0 and buf.data:
            duration_s = buf.sample_count / 16000.0
            results.record(
                call_name, True,
                f"{buf.sample_count:,} samples ({duration_s:.1f}s of speech at 16kHz)",
            )
        else:
            results.record(
                call_name, False,
                "sample_count=0 — file may be silent, video-only, or codec not supported",
            )

        # ── Memory release ─────────────────────────────────────────────────────
        lib.free_audio_buffer(buf.data)
        results.record(memory_name, True, "no crash after free_audio_buffer()")

    except Exception as exc:
        results.record(call_name,  False, str(exc))
        results.record(memory_name, False, "skipped — call raised exception")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 milestone verification runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--siglip",      default=_DEFAULT_SIGLIP,
                        help=f"SigLIP INT8 vision encoder path (default: {_DEFAULT_SIGLIP})")
    parser.add_argument("--whisper-dir", default=_DEFAULT_WHISPER_DIR,
                        help=f"Whisper Tiny ONNX directory (default: {_DEFAULT_WHISPER_DIR})")
    parser.add_argument("--det",         default=_DEFAULT_DET,
                        help=f"PP-OCR detection INT8 path (default: {_DEFAULT_DET})")
    parser.add_argument("--rec",         default=_DEFAULT_REC,
                        help=f"PP-OCR recognition INT8 path (default: {_DEFAULT_REC})")
    parser.add_argument("--native-lib",  default=None,
                        help="Path to compiled media_core .dll/.so (optional)")
    parser.add_argument("--test-video",  default=None,
                        help="Path to a sample video file for C lib smoke test")
    args = parser.parse_args()

    results = Results()

    log.info("=" * 60)
    log.info("  Phase 1 Verification Runner")
    log.info("=" * 60)

    # Check 1: SigLIP graph shape
    log.info("\n[Check 1] SigLIP graph shape verification…")
    check_siglip_shape(args.siglip, results)

    # Check 2: Size budgets
    log.info("\n[Check 2] Size budget assertions…")
    check_sizes(args.siglip, args.whisper_dir, args.det, args.rec, results)

    # Check 3: C library (optional)
    if args.native_lib:
        log.info("\n[Check 3] C library smoke test…")
        check_native_lib(args.native_lib, args.test_video, results)
    else:
        log.info("\n[Check 3] C library smoke test — SKIPPED (pass --native-lib to enable)")

    # Summary
    all_pass = results.summary()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
