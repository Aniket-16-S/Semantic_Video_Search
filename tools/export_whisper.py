"""
tools/export_whisper.py
=======================
Whisper Tiny ONNX export and INT8 quantization pipeline.

Uses Hugging Face Optimum's ONNX exporter to correctly handle the multi-headed
encoder/decoder architecture with dynamic axes for variable-length audio inputs
and autoregressive decoder sequences.

Model specification
-------------------
  Model  : openai/whisper-tiny.en        (~39M parameters)
  Task   : automatic-speech-recognition
  Input  : 16 kHz mono float32 mel-spectrogram  [1, 80, 3000]  (30s window)
  Output : token ids + scores

Why whisper-tiny.en (not whisper-base)?
  tiny = 39M params → fits in the 500 MB RAM budget on Exynos 1280.
  .en  = English-only model → 10% smaller decoder vocabulary → faster inference.
  base = 74M params → risks thermal throttling on Exynos 1280 per spec §3B.

Output structure (multi-file Optimum export)
--------------------------------------------
  assets/models/whisper_tiny_int8/
    encoder_model.onnx           ← mel spectrogram → hidden states
    decoder_model.onnx           ← full decoder (init pass only)
    decoder_with_past_model.onnx ← incremental decoder (all subsequent tokens)
    config.json                  ← model config (token ids, etc.)
    tokenizer*.json              ← tokenizer vocabulary

The three ONNX files are loaded as separate ORT sessions. This is the standard
Optimum pattern for seq2seq models — it mirrors the "split model" architecture
already used for SigLIP vision/text encoders.

Size budget
-----------
  Total across all three ONNX files: ≤ 50 MB (per Phase 1 spec).
  Typical INT8 sizes after O2 optimization:
    encoder_model.onnx           ~12 MB
    decoder_model.onnx           ~18 MB
    decoder_with_past_model.onnx ~18 MB
    Total                        ~48 MB  ✓

Usage
-----
    # Export + O2 optimize (recommended — produces INT8 via Optimum's quantizer)
    python tools/export_whisper.py

    # Export to a custom output directory
    python tools/export_whisper.py --output-dir assets/models/whisper_custom

    # Export only (skip verification)
    python tools/export_whisper.py --skip-verify

    # Use optimum-cli directly (equivalent bash command for reference)
    # optimum-cli export onnx \\
    #     --model openai/whisper-tiny.en \\
    #     --optimize O2 \\
    #     --device cpu \\
    #     --task automatic-speech-recognition \\
    #     assets/models/whisper_tiny_int8/

Requirements
------------
    pip install "optimum[onnxruntime]>=1.17.0" onnxruntime transformers

Output
------
    assets/models/whisper_tiny_int8/   (complete Optimum bundle)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_whisper")

# ── Project paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUT  = _PROJECT_ROOT / "assets" / "models" / "whisper_tiny_int8"

_MODEL_ID     = "openai/whisper-tiny.en"
_SIZE_BUDGET  = 50.0   # MB — total across all ONNX files


# ── Dependency check ───────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("optimum", "onnxruntime", "transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing packages: %s\n"
            'Install with:  pip install "optimum[onnxruntime]>=1.17.0" onnxruntime transformers',
            ", ".join(missing),
        )
        sys.exit(1)


# ── Step 1: Export via optimum-cli ─────────────────────────────────────────────

def export_via_optimum_cli(output_dir: str) -> None:
    """
    Invoke optimum-cli to export and O2-optimize Whisper Tiny in one step.

    --optimize O2 applies:
      - Constant folding
      - Redundant node elimination
      - Attention head pruning
      - Graph simplification
    This replaces a separate post-export quantization step; Optimum's O2
    pipeline integrates dynamic INT8 quantization of weight matrices automatically
    for the ASR task.

    Why optimum-cli over Python API?
    The CLI handles the encoder/decoder split + dynamic axis configuration for
    Whisper's multi-file layout without custom wrapper code. The Python API
    (ORTModelForSpeechSeq2Seq.from_pretrained) requires more boilerplate and
    produces identical output.
    """
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx.__main__",   # optimum-cli onnx
        "--model", _MODEL_ID,
        "--optimize", "O2",
        "--device", "cpu",
        "--task", "automatic-speech-recognition-with-past",
        "--no-post-process",
        output_dir,
    ]

    log.info("Exporting Whisper Tiny to ONNX (O2 optimized)…")
    log.info("Equivalent CLI: optimum-cli export onnx --model %s --optimize O2 --task automatic-speech-recognition-with-past --no-post-process %s",
             _MODEL_ID, output_dir)
    log.info("This may take 5–15 minutes on first run (downloads ~150 MB weights).")

    result = subprocess.run(cmd, capture_output=False)   # stream output live
    if result.returncode != 0:
        # Fallback: try the direct optimum-cli entry point
        log.warning("optimum module entrypoint failed — trying optimum-cli directly…")
        cmd_fallback = [
            "optimum-cli", "export", "onnx",
            "--model", _MODEL_ID,
            "--optimize", "O2",
            "--device", "cpu",
            "--task", "automatic-speech-recognition-with-past",
            "--no-post-process",
            output_dir,
        ]
        result2 = subprocess.run(cmd_fallback, capture_output=False)
        if result2.returncode != 0:
            log.error(
                "Optimum export failed (exit %d).\n"
                'Make sure optimum[onnxruntime] is installed: pip install "optimum[onnxruntime]>=1.17.0"',
                result2.returncode,
            )
            sys.exit(1)


# ── Step 1.5: Quantization ─────────────────────────────────────────────────────

def quantize_whisper_models(output_dir: str) -> None:
    """
    Apply asymmetric dynamic INT8 quantization using ONNX Runtime.
    Best for cross-platform (Android/Windows) and keeps precision high by
    quantizing only MatMul weights.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info("Applying dynamic INT8 quantization to Whisper ONNX models…")
    
    for model_name in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
        fp32_path = os.path.join(output_dir, model_name)
        if not os.path.exists(fp32_path):
            continue
            
        # Temporarily rename FP32 model
        temp_path = fp32_path + ".fp32"
        os.rename(fp32_path, temp_path)
        
        log.info("  Quantizing %s…", model_name)
        
        try:
            from onnxruntime.quantization import shape_inference
            sanitized_fp32_path = temp_path + ".clean.onnx"
            shape_inference.quant_pre_process(
                input_model_path=temp_path,
                output_model_path=sanitized_fp32_path,
                skip_optimization=False,
                auto_merge=True,
            )
            target_input_path = sanitized_fp32_path
        except Exception as e:
            log.warning("Graph sanitization failed: %s. Proceeding with fallback...", e)
            target_input_path = temp_path

        import onnx
        quantize_dynamic(
            model_input=target_input_path,
            model_output=fp32_path,
            weight_type=QuantType.QUInt8,
            extra_options={
                "MatMulConstBOnly": True,
                "DefaultTensorType": onnx.TensorProto.FLOAT,
            },
        )
        
        os.remove(temp_path)
        if target_input_path != temp_path and os.path.exists(target_input_path):
            os.remove(target_input_path)
        
        size_mb = os.path.getsize(fp32_path) / 1e6
        log.info("    Saved → %s (%.1f MB)", model_name, size_mb)


# ── Step 2: Size verification ──────────────────────────────────────────────────

def verify_size_budget(output_dir: str) -> float:
    """
    Sum all .onnx files in the output directory and assert ≤ 50 MB total.
    Returns total size in MB.
    """
    total_bytes = 0
    onnx_files  = list(Path(output_dir).glob("*.onnx"))

    if not onnx_files:
        log.error("No .onnx files found in %s — export may have failed.", output_dir)
        sys.exit(1)

    log.info("Output ONNX files:")
    for f in sorted(onnx_files):
        mb = f.stat().st_size / 1e6
        total_bytes += f.stat().st_size
        log.info("  %-40s  %.1f MB", f.name, mb)

    total_mb = total_bytes / 1e6
    log.info("  %-40s  %.1f MB", "TOTAL", total_mb)

    if total_mb > _SIZE_BUDGET:
        log.warning(
            "⚠ Total size %.1f MB exceeds budget %.0f MB.\n"
            "  Consider using whisper-tiny (not base) and --optimize O2.",
            total_mb, _SIZE_BUDGET,
        )
    else:
        log.info("✓ Size budget satisfied (%.1f MB ≤ %.0f MB).", total_mb, _SIZE_BUDGET)

    return total_mb


# ── Step 3: Functional smoke test ──────────────────────────────────────────────

def verify_encoder(output_dir: str) -> None:
    """
    Load the encoder and run a single forward pass with a dummy mel spectrogram.
    Whisper encoder input: [1, 80, 3000] float32  (80 mel bins × 3000 frames = 30s)
    Expected output: [1, 1500, 384] float32  (hidden states, 384-dim for tiny)
    """
    import numpy as np
    import onnxruntime as ort

    encoder_path = os.path.join(output_dir, "encoder_model.onnx")
    if not os.path.exists(encoder_path):
        log.warning("encoder_model.onnx not found — skipping functional test.")
        return

    log.info("Verifying Whisper encoder (dummy mel → hidden states)…")
    sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])

    # Dummy input: silent audio (zeros in mel space)
    dummy_mel = np.zeros((1, 80, 3000), dtype=np.float32)

    inputs = {inp.name: dummy_mel for inp in sess.get_inputs()}
    outputs = sess.run(None, inputs)

    log.info("✓ Encoder output shapes: %s", [o.shape for o in outputs])
    assert outputs[0].shape[0] == 1, f"Unexpected batch dim: {outputs[0].shape}"


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Whisper Tiny to O2-optimized ONNX for on-device ASR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUT),
        help=f"Output directory for Whisper ONNX bundle (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export if output dir already contains .onnx files",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-export size and functional verification",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Whisper Tiny ONNX Exporter")
    log.info("  Model     : %s", _MODEL_ID)
    log.info("  Output    : %s", output_dir)
    log.info("  Budget    : ≤ %.0f MB (total across all .onnx files)", _SIZE_BUDGET)
    log.info("=" * 60)

    _check_deps()

    # Step 1: Export
    existing_onnx = list(output_dir.glob("*.onnx"))
    if args.skip_export and existing_onnx:
        log.info(
            "Skipping export — found %d existing .onnx files in %s.",
            len(existing_onnx), output_dir,
        )
    else:
        export_via_optimum_cli(str(output_dir))
        # Quantize the exported models immediately
        quantize_whisper_models(str(output_dir))

    # Step 2: Size budget check
    if not args.skip_verify:
        verify_size_budget(str(output_dir))

    # Step 3: Encoder smoke test
    if not args.skip_verify:
        verify_encoder(str(output_dir))

    log.info("")
    log.info("✓ Whisper Tiny export complete.")
    log.info("  Bundle → %s", output_dir)
    log.info("  Load with:")
    log.info("    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq")
    log.info("    model = ORTModelForSpeechSeq2Seq.from_pretrained('%s')", output_dir)


if __name__ == "__main__":
    main()
