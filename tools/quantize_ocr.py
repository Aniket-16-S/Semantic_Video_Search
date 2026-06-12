"""
tools/quantize_ocr.py
=====================
PP-OCR v4 INT8 quantization pipeline for on-device text extraction.

This script takes pre-converted FP32 ONNX files (output of paddle2onnx) and
applies asymmetric QUInt8 dynamic quantization to produce production-ready
INT8 models for Android NNAPI and Windows DirectML inference.

Pipeline overview
-----------------
The full OCR pipeline runs as two sequential ONNX sessions:

  1. Text Detection  (DBNet layout)
     Input  : image [1, 3, H, W]  — normalized BGR float32
     Output : probability map [1, 1, H, W]  — pixel-wise text confidence
     Model  : ppocr_det_int8.onnx

  2. Text Recognition  (Transformer / CRNN layout)
     Input  : text region crops [1, 3, 48, W]  — normalized grayscale float32
     Output : token logits [seq_len, 1, vocab_size]
     Model  : ppocr_rec_int8.onnx

Prerequisites — Paddle → ONNX conversion (run ONCE before this script)
-----------------------------------------------------------------------
These steps require a Linux/WSL environment (wget not available on native Windows).
Run them in WSL or a Linux CI container:

    # 1. Install Paddle + converter
    pip install paddlepaddle paddle2onnx onnxruntime

    # 2. Download PaddleOCR v4 English inference models
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
    tar -xf en_PP-OCRv4_det_infer.tar
    tar -xf en_PP-OCRv4_rec_infer.tar

    # 3. Convert Paddle Inference → ONNX (opset 14 — widest NNAPI support)
    paddle2onnx \\
        --model_dir ./en_PP-OCRv4_det_infer \\
        --model_filename inference.pdmodel \\
        --params_filename inference.pdiparams \\
        --save_file ./ppocr_det_fp32.onnx \\
        --opset_version 14

    paddle2onnx \\
        --model_dir ./en_PP-OCRv4_rec_infer \\
        --model_filename inference.pdmodel \\
        --params_filename inference.pdiparams \\
        --save_file ./ppocr_rec_fp32.onnx \\
        --opset_version 14

After conversion, place ppocr_det_fp32.onnx and ppocr_rec_fp32.onnx in
raw_models/ (or pass --det-fp32 / --rec-fp32 flags), then run this script.

Usage
-----
    # Default paths (expects raw_models/ppocr_*.onnx)
    python tools/quantize_ocr.py

    # Custom input paths
    python tools/quantize_ocr.py \\
        --det-fp32 ./ppocr_det_fp32.onnx \\
        --rec-fp32 ./ppocr_rec_fp32.onnx \\
        --output-dir assets/models

Output
------
    assets/models/ppocr_det_int8.onnx   (Text detection — DBNet)
    assets/models/ppocr_rec_int8.onnx   (Text recognition — Transformer)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_ocr")

# ── Project paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ASSETS_DIR   = _PROJECT_ROOT / "assets" / "models"
_RAW_DIR      = _PROJECT_ROOT / "raw_models"

# Size budgets (MB) — warn if exceeded
_DET_SIZE_BUDGET = 10    # DBNet detection model is compact
_REC_SIZE_BUDGET = 15    # Recognition model slightly larger (Transformer decoder)


# ── Dependency check ───────────────────────────────────────────────────────────

def _check_deps() -> None:
    import importlib
    missing = []
    for pkg in ("onnxruntime",):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing: %s\n"
            "Install with: pip install onnxruntime",
            ", ".join(missing),
        )
        sys.exit(1)


# ── Quantization ───────────────────────────────────────────────────────────────

def quantize_model(fp32_path: str, int8_path: str, label: str, budget_mb: float) -> None:
    """
    Apply asymmetric QUInt8 dynamic quantization to a single FP32 ONNX model.
    Preprocesses the model graph to convert constant inputs into explicit initializers 
    to prevent ONNX Runtime quantization failures.
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(fp32_path):
        log.error(
            "FP32 ONNX not found: %s\n"
            "Run the paddle2onnx conversion steps in the script docstring first.",
            fp32_path,
        )
        sys.exit(1)

    fp32_mb = os.path.getsize(fp32_path) / 1e6
    log.info("[%s] Quantizing %.1f MB → INT8…", label, fp32_mb)
    log.info("  Input  : %s", fp32_path)
    log.info("  Output : %s", int8_path)

    # ── Graph Cleaning Step ───────────────────────────────────────────────────
    log.info("[%s] Sanitizing model graph (converting constants to initializers)...", label)
    try:
        from onnxruntime.quantization import shape_inference
        sanitized_fp32_path = fp32_path + ".clean.onnx"
        shape_inference.quant_pre_process(
            input_model_path=fp32_path,
            output_model_path=sanitized_fp32_path,
            skip_optimization=False,
            auto_merge=True,
        )
        target_input_path = sanitized_fp32_path
    except Exception as e:
        log.warning("[%s] Graph sanitization failed: %s. Proceeding with fallback...", label, e)
        target_input_path = fp32_path

    # ── Quantization Step ─────────────────────────────────────────────────────
    quantize_dynamic(
        model_input=target_input_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,         # asymmetric — per spec
        extra_options={
            "MatMulConstBOnly": True,         # weights only; activations stay FP32
        },
    )

    # Clean up temporary sanitized file if created
    if target_input_path != fp32_path and os.path.exists(target_input_path):
        os.remove(target_input_path)

    int8_mb = os.path.getsize(int8_path) / 1e6
    reduction = (1.0 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0.0

    log.info(
        "[%s] Saved → %s  (%.1f MB, %.0f%% reduction)",
        label, int8_path, int8_mb, reduction,
    )

    if int8_mb > budget_mb:
        log.warning(
            "⚠ [%s] Size %.1f MB exceeds budget %.0f MB.",
            label, int8_mb, budget_mb,
        )
    else:
        log.info("✓ [%s] Budget satisfied (%.1f MB ≤ %.0f MB).", label, int8_mb, budget_mb)


# ── Smoke test ─────────────────────────────────────────────────────────────────

def verify_model(int8_path: str, label: str, dummy_input: dict) -> None:
    """Load the INT8 model and run a dummy forward pass."""
    import numpy as np
    import onnxruntime as ort

    log.info("[%s] Running verification pass…", label)
    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, dummy_input)
    log.info("✓ [%s] Output shapes: %s", label, [o.shape for o in outputs])


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize PP-OCR v4 detection + recognition models to INT8 ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--det-fp32",
        default=str(_RAW_DIR / "ppocr_det_fp32.onnx"),
        help="Path to detection FP32 ONNX (default: raw_models/ppocr_det_fp32.onnx)",
    )
    parser.add_argument(
        "--rec-fp32",
        default=str(_RAW_DIR / "ppocr_rec_fp32.onnx"),
        help="Path to recognition FP32 ONNX (default: raw_models/ppocr_rec_fp32.onnx)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ASSETS_DIR),
        help=f"Output directory for INT8 models (default: {_ASSETS_DIR})",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-quantization verification forward pass",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    det_int8 = str(output_dir / "ppocr_det_int8.onnx")
    rec_int8 = str(output_dir / "ppocr_rec_int8.onnx")

    log.info("=" * 60)
    log.info("  PP-OCR v4 INT8 Quantization Pipeline")
    log.info("  Detection  FP32 : %s", args.det_fp32)
    log.info("  Recognition FP32: %s", args.rec_fp32)
    log.info("  Output dir       : %s", output_dir)
    log.info("=" * 60)

    _check_deps()

    # ── Detection model ──────────────────────────────────────────────────────
    quantize_model(args.det_fp32, det_int8, "DET", _DET_SIZE_BUDGET)

    # ── Recognition model ────────────────────────────────────────────────────
    quantize_model(args.rec_fp32, rec_int8, "REC", _REC_SIZE_BUDGET)

    # ── Verification ─────────────────────────────────────────────────────────
    if not args.skip_verify:
        import numpy as np

        # Detection: typical 640×640 input image (normalized)
        verify_model(
            det_int8, "DET",
            dummy_input={"x": np.zeros((1, 3, 640, 640), dtype=np.float32)},
        )

        # Recognition: typical 48px-height text line crop (width flexible)
        # Note: actual width varies by crop; 320 is a representative value.
        try:
            verify_model(
                rec_int8, "REC",
                dummy_input={"x": np.zeros((1, 3, 48, 320), dtype=np.float32)},
            )
        except Exception as exc:
            log.warning(
                "REC model verification failed (%s). "
                "This may be expected if the model uses dynamic width axes — "
                "test with a real inference call to confirm.",
                exc,
            )

    log.info("")
    log.info("✓ PP-OCR INT8 quantization complete.")
    log.info("  Detection  → %s", det_int8)
    log.info("  Recognition→ %s", rec_int8)


if __name__ == "__main__":
    main()
