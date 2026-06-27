"""
tools/export_split_models.py
============================
One-time script: exports the SigLIP vision and text encoders as SEPARATE
ONNX files and INT8-quantizes each.

Why separate encoders?
----------------------
The current model_quantized.onnx (210 MB) contains BOTH the 12-layer ViT
vision encoder AND the 12-layer text transformer in a single graph.

  During frame upload  -> only the vision encoder is needed.
  During text search   -> only the text encoder is needed.

Running both for every call wastes ~50% of compute.  Split models fix this:
  vision_encoder_int8.onnx  ~26 MB   → upload pipeline only
  text_encoder_int8.onnx    ~26 MB   → search pipeline only

How the embeddings still match
-------------------------------
SigLIP is trained with contrastive learning on (image, text) pairs, so both
encoders share the SAME 768-dimensional embedding space.  A text embedding
for "person riding bicycle" will have high cosine similarity to any frame
embedding showing that scene, even though the two vectors were produced by
completely separate models.

Run once:
    python tools/export_split_models.py

Outputs (written to models/siglip_int8/):
    vision_encoder.onnx         FP32 vision-only graph  (~105 MB)
    vision_encoder_int8.onnx    INT8 quantized           (~26  MB)
    text_encoder.onnx           FP32 text-only graph     (~105 MB)
    text_encoder_int8.onnx      INT8 quantized           (~26  MB)

Requirements: torch, transformers, onnx, onnxruntime
(all already in requirements.txt)
"""

import os
import sys
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_OUTPUT_DIR  = _PROJECT_ROOT / "models" / "siglip_int8"
_MODEL_NAME  = "google/siglip-base-patch16-224"

_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISION_ONNX     = str(_OUTPUT_DIR / "vision_encoder.onnx")
VISION_INT8     = str(_OUTPUT_DIR / "vision_encoder_int8.onnx")
TEXT_ONNX       = str(_OUTPUT_DIR / "text_encoder.onnx")
TEXT_INT8       = str(_OUTPUT_DIR / "text_encoder_int8.onnx")


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("  Install with: pip install torch transformers onnx onnxruntime")
        sys.exit(1)


# ── Wrapper modules (torch.onnx.export needs Module subclasses) ───────────

def _make_vision_wrapper():
    import torch
    import torch.nn as nn
    from transformers import SiglipVisionModel

    class VisionWrapper(nn.Module):
        """
        Wraps SiglipVisionModel and exposes only the pooler_output,
        renamed to 'image_embeds' in the ONNX graph.

        SiglipVisionModel returns BaseModelOutputWithPooling:
          .last_hidden_state  [batch, num_patches, hidden]   (not exported)
          .pooler_output      [batch, hidden_size=768]       ← this is image_embeds
        """
        def __init__(self):
            super().__init__()
            print(f"  Loading SiglipVisionModel from '{_MODEL_NAME}'…")
            self.model = SiglipVisionModel.from_pretrained(_MODEL_NAME)

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).pooler_output

    return VisionWrapper()


def _make_text_wrapper():
    import torch
    import torch.nn as nn
    from transformers import SiglipTextModel

    class TextWrapper(nn.Module):
        """
        Wraps SiglipTextModel and exposes only the pooler_output,
        renamed to 'text_embeds' in the ONNX graph.

        SiglipTextModel returns BaseModelOutputWithPooling:
          .last_hidden_state  [batch, seq_len, hidden]   (not exported)
          .pooler_output      [batch, hidden_size=768]   ← this is text_embeds
        """
        def __init__(self):
            super().__init__()
            print(f"  Loading SiglipTextModel from '{_MODEL_NAME}'…")
            self.model = SiglipTextModel.from_pretrained(_MODEL_NAME)

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

    return TextWrapper()


# ── Export functions ──────────────────────────────────────────────────────

def export_vision_encoder() -> None:
    import torch

    print("\n[1/4] Exporting vision encoder to ONNX (FP32)…")
    wrapper = _make_vision_wrapper()
    wrapper.eval()

    # Use batch=2 (not 1!) so that do_constant_folding cannot collapse the
    # batch dimension into a constant.  The dynamic_axes dict marks axis-0 as
    # 'batch_size' (symbolic), so any batch size works at runtime — batch=2
    # here is only to give the tracer a non-trivial shape to work with.
    dummy_pixels = torch.zeros(2, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_pixels,),
            VISION_ONNX,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True
        )

    size_mb = os.path.getsize(VISION_ONNX) / 1e6
    print(f"  Saved → {VISION_ONNX}  ({size_mb:.0f} MB)")


def export_text_encoder() -> None:
    import torch

    print("\n[2/4] Exporting text encoder to ONNX (FP32)…")
    wrapper = _make_text_wrapper()
    wrapper.eval()

    # Same reasoning as the vision encoder: batch=2 prevents constant-folding
    # from freezing the batch dimension; dynamic_axes keeps it symbolic.
    dummy_ids  = torch.zeros(2, 64, dtype=torch.long)
    dummy_mask = torch.ones(2, 64, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            TEXT_ONNX,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids":      {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "text_embeds":    {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True
        )

    size_mb = os.path.getsize(TEXT_ONNX) / 1e6
    print(f"  Saved → {TEXT_ONNX}  ({size_mb:.0f} MB)")


def quantize_model(fp32_path: str, int8_path: str, label: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\n[{label}] INT8 quantizing {os.path.basename(fp32_path)}…")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        # MatMulConstBOnly=True: quantize only MatMul ops with constant weights
        # (safer for transformer models — avoids quantizing dynamic activations)
        op_types_to_quantize=['MatMul', 'Attention', 'Gather', 'EmbedLayerNormalization'],
        extra_options={"MatMulConstBOnly": True},
    )
    size_mb = os.path.getsize(int8_path) / 1e6
    print(f"  Saved → {int8_path}  ({size_mb:.0f} MB)")


def verify_models() -> None:
    """Smoke-test: verify both exported models accept dynamic batch sizes.

    Tests batch sizes 1, 2, and 8 to confirm the dynamic axis is truly live
    end-to-end (FP32 export → INT8 quantization → ORT inference).
    A static/frozen batch would raise a shape mismatch on anything != 1.
    """
    import numpy as np
    import onnxruntime as ort

    print("\n[Verify] Running dynamic-batch smoke tests…")
    _BATCH_SIZES = [1, 2, 8]  # must all succeed; failure = frozen batch dim

    # ── Vision encoder ────────────────────────────────────────────────
    sess_v = ort.InferenceSession(VISION_INT8, providers=["CPUExecutionProvider"])
    print("  Vision encoder (vision_encoder_int8.onnx):")
    for bs in _BATCH_SIZES:
        dummy_pix = np.zeros((bs, 3, 224, 224), dtype=np.float32)
        out_v = sess_v.run(["image_embeds"], {"pixel_values": dummy_pix})
        assert out_v[0].shape == (bs, 768), (
            f"    FAIL batch={bs}: expected ({bs}, 768), got {out_v[0].shape}\n"
            f"    → Dynamic batch axis was frozen during export or quantization.\n"
            f"    → Re-run this script to regenerate the models."
        )
        print(f"    ✓ batch={bs}: input ({bs},3,224,224) → output {out_v[0].shape}")

    # ── Text encoder ──────────────────────────────────────────────────
    sess_t = ort.InferenceSession(TEXT_INT8, providers=["CPUExecutionProvider"])
    print("  Text encoder (text_encoder_int8.onnx):")
    for bs in _BATCH_SIZES:
        dummy_ids  = np.zeros((bs, 64), dtype=np.int64)
        dummy_mask = np.ones((bs, 64),  dtype=np.int64)
        out_t = sess_t.run(["text_embeds"], {"input_ids": dummy_ids, "attention_mask": dummy_mask})
        assert out_t[0].shape == (bs, 768), (
            f"    FAIL batch={bs}: expected ({bs}, 768), got {out_t[0].shape}\n"
            f"    → Dynamic batch axis was frozen during export or quantization.\n"
            f"    → Re-run this script to regenerate the models."
        )
        print(f"    ✓ batch={bs}: input ({bs},64) → output {out_t[0].shape}")

    print("\n  All dynamic-batch checks passed.")
    print("  Restart the API server to activate split-model mode.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(" SigLIP Split Model Export — tools/export_split_models.py")
    print("=" * 60)
    print(f" Model  : {_MODEL_NAME}")
    print(f" Output : {_OUTPUT_DIR}")

    _check_deps()

    # Step 1 & 2: FP32 ONNX exports
    export_vision_encoder()
    export_text_encoder()

    # Step 3 & 4: INT8 quantization
    quantize_model(VISION_ONNX, VISION_INT8, "3/4")
    quantize_model(TEXT_ONNX,   TEXT_INT8,   "4/4")

    # Step 5: Smoke test
    verify_models()

    print("\n" + "=" * 60)
    print(" Done!  Restart the API server to activate split-model mode.")
    print("=" * 60)


if __name__ == "__main__":
    main()
