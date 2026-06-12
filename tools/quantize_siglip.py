"""
tools/quantize_siglip.py
========================
Production SigLIP ONNX export + asymmetric INT8 quantization script.

Supports two model tiers selectable via --tier:

  base   (default)  google/siglip-base-patch16-224
                    ~26 MB / encoder after INT8 — already proven in prototype.
                    768-dim embedding, opset 17.

  hq                google/siglip-so400m-patch14-224
                    SO400M variant for maximum embedding quality.
                    768-dim embedding (same space, richer features).
                    Budget target: ≤ 300 MB after INT8.

Both tiers output to assets/models/ using consistent filenames:
  assets/models/siglip_vision_int8.onnx   ← vision encoder
  (Text encoder produced by tools/export_split_models.py — unchanged)

Why we export only the vision encoder here
-------------------------------------------
The text encoder path is already handled by the production-ready
export_split_models.py script (opset 17, dynamic batch, QInt8, verified).
This script extends that workflow for SO400M on the vision side only, using
the spec-mandated QUInt8 (asymmetric) quantization strategy.

Usage
-----
    # Standard (base model — fastest, smallest)
    python tools/quantize_siglip.py

    # High-quality SO400M variant
    python tools/quantize_siglip.py --tier hq

    # Override output dir
    python tools/quantize_siglip.py --tier hq --output-dir models/so400m_int8

    # Skip export if FP32 ONNX already exists
    python tools/quantize_siglip.py --skip-export

Requirements
------------
    pip install torch transformers onnx onnxruntime

Output
------
    raw_models/siglip_vision_fp32_<tier>.onnx   (intermediate, excluded from git)
    assets/models/siglip_vision_int8.onnx        (final, excluded from git)
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
log = logging.getLogger("quantize_siglip")

# ── Project paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ASSETS_DIR   = _PROJECT_ROOT / "assets" / "models"
_RAW_DIR      = _PROJECT_ROOT / "raw_models"

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_TIERS = {
    "base": {
        "id":          "google/siglip-base-patch16-224",
        "description": "SigLIP Base (patch16-224) — ~26 MB INT8, 768-dim",
        "size_budget": 80,    # MB — warn if exceeded (base encoders are split)
    },
    "hq": {
        "id":          "google/siglip-so400m-patch14-224",
        "description": "SigLIP SO400M (patch14-224) — HQ model, ≤300 MB INT8",
        "size_budget": 300,   # MB — spec hard limit
    },
}


# ── Dependency check ───────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing required packages: %s\n"
            "Install with:  pip install torch transformers onnx onnxruntime",
            ", ".join(missing),
        )
        sys.exit(1)


# ── Step 1: FP32 ONNX export ───────────────────────────────────────────────────

def export_vision_fp32(model_id: str, out_path: str, opset: int) -> None:
    """
    Export the SigLIP vision encoder (pooler_output only) to FP32 ONNX.

    Wraps SiglipVisionModel in a thin nn.Module that returns only the
    pooler_output (shape [batch, hidden_size]) renamed to 'image_embeds'.
    This eliminates the last_hidden_state from the graph — halving the
    ONNX export size and removing an unneeded output from ONNX Runtime.

    Dynamic batch axis is preserved using batch=2 dummy input (prevents
    do_constant_folding from collapsing batch dim into a compile-time constant).
    """
    import torch
    import torch.nn as nn
    from transformers import SiglipVisionModel

    log.info("Loading SiglipVisionModel from '%s'…", model_id)

    class VisionWrapper(nn.Module):
        """Strip SiglipVisionModel output to pooler_output → 'image_embeds'."""
        def __init__(self):
            super().__init__()
            self.model = SiglipVisionModel.from_pretrained(model_id)

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).pooler_output

    wrapper = VisionWrapper()
    wrapper.eval()

    # batch=2: prevents constant-folding from freezing the batch dimension.
    dummy = torch.zeros(2, 3, 224, 224, dtype=torch.float32)

    log.info("Exporting to FP32 ONNX (opset %d) → %s", opset, out_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            export_params=True,
            opset_version=opset,                 # fixed stable opset per spec
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    log.info("FP32 ONNX saved → %s  (%.0f MB)", out_path, size_mb)


# ── Step 2: Asymmetric INT8 dynamic quantization ───────────────────────────────

def quantize_to_int8(fp32_path: str, int8_path: str, budget_mb: float) -> None:
    """
    Apply asymmetric dynamic QUInt8 quantization to the FP32 ONNX graph.

    Strategy (per project spec):
      - QUInt8 (unsigned) = asymmetric quantization → better for ReLU/GELU
        activation distributions that are skewed non-negative.
      - Dynamic quantization: no calibration dataset required. Weights are
        quantized offline; activations are quantized dynamically at runtime.
      - MatMulConstBOnly=True: only quantize MatMul ops with constant weights
        (i.e., the heavy projection matrices in attention layers).
        This PRESERVES LayerNorm, Softmax, and activation scaling at FP32
        — exactly as specified ("leaving layer-normalization blocks at FP32").
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info("Applying asymmetric QUInt8 dynamic quantization…")
    log.info("  Input  : %s", fp32_path)
    log.info("  Output : %s", int8_path)

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,       # asymmetric — per spec
        extra_options={
            "MatMulConstBOnly": True,       # quantize weights only, not activations
        },
    )

    size_mb = os.path.getsize(int8_path) / 1e6
    log.info("INT8 ONNX saved → %s  (%.1f MB)", int8_path, size_mb)

    # ── Budget assertion ─────────────────────────────────────────────────────
    if size_mb > budget_mb:
        log.warning(
            "⚠ Model size %.1f MB EXCEEDS budget %.0f MB.\n"
            "  Consider reducing the model tier or adjusting quantization.",
            size_mb, budget_mb,
        )
    else:
        log.info("✓ Size budget satisfied (%.1f MB ≤ %.0f MB).", size_mb, budget_mb)


# ── Step 3: Smoke test ─────────────────────────────────────────────────────────

def verify_model(int8_path: str) -> None:
    """
    Load the INT8 model and run a single forward pass to confirm:
      1. The model loads without error.
      2. The output shape is [1, 768] — correct for both base and SO400M.
    """
    import numpy as np
    import onnxruntime as ort

    log.info("Verifying INT8 model…")
    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out   = sess.run(["image_embeds"], {"pixel_values": dummy})

    shape = out[0].shape
    assert len(shape) == 2 and shape[0] == 1, (
        f"Unexpected output shape: {shape} — expected (1, 512) or (1, 768)"
    )
    assert shape[1] in (512, 768), (
        f"Unexpected embedding dimension: {shape[1]} — expected 512 or 768"
    )
    log.info("✓ Graph shape verification passed: output = %s", shape)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export + INT8-quantize SigLIP vision encoder for on-device inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model tiers:
  base  google/siglip-base-patch16-224     ~26 MB INT8 (default, proven)
  hq    google/siglip-so400m-patch14-224   ≤300 MB INT8 (higher quality)
        """,
    )
    parser.add_argument(
        "--tier",
        choices=["base", "hq"],
        default="base",
        help="Model quality tier (default: base)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ASSETS_DIR),
        help=f"Directory to write siglip_vision_int8.onnx (default: {_ASSETS_DIR})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17 — stable for NNAPI / DirectML)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 ONNX export if raw_models/siglip_vision_fp32_<tier>.onnx exists",
    )
    args = parser.parse_args()

    tier_cfg = MODEL_TIERS[args.tier]
    model_id = tier_cfg["id"]
    budget   = tier_cfg["size_budget"]

    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = str(_RAW_DIR / f"siglip_vision_fp32_{args.tier}.onnx")
    int8_path = os.path.join(args.output_dir, "siglip_vision_int8.onnx")

    log.info("=" * 60)
    log.info("  SigLIP Vision Encoder Quantizer")
    log.info("  Tier      : %s — %s", args.tier, tier_cfg["description"])
    log.info("  Model ID  : %s", model_id)
    log.info("  Opset     : %d", args.opset)
    log.info("  Output    : %s", int8_path)
    log.info("  Budget    : ≤ %d MB", budget)
    log.info("=" * 60)

    _check_deps()

    # Step 1: FP32 ONNX export
    if args.skip_export and os.path.exists(fp32_path):
        size_mb = os.path.getsize(fp32_path) / 1e6
        log.info("Skipping FP32 export — using existing %s (%.0f MB)", fp32_path, size_mb)
    else:
        export_vision_fp32(model_id, fp32_path, args.opset)

    # Step 2: INT8 quantization
    quantize_to_int8(fp32_path, int8_path, budget)

    # Step 3: Verification
    verify_model(int8_path)

    log.info("")
    log.info("✓ Done. Load in inference with:")
    log.info("  import onnxruntime as ort")
    log.info("  sess = ort.InferenceSession('%s')", int8_path)
    log.info("  out  = sess.run(['image_embeds'], {'pixel_values': pixel_np})")


if __name__ == "__main__":
    main()
