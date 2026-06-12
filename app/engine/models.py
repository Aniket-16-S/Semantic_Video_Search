"""
app/engine/models.py
====================
Thread-safe singleton SigLIP inference engine with automatic split-model support
and dual-tier quality selection.

Model Tier Selection
--------------------
Set SIGLIP_TIER in .env to switch between two quality modes:

  SIGLIP_TIER=base  (default)
    Model  : google/siglip-base-patch16-224
    Dir    : models/siglip_int8/
    Size   : ~26 MB per encoder (split mode)
    Speed  : Fast — ideal for real-time frame indexing

  SIGLIP_TIER=hq
    Model  : google/siglip-so400m-patch14-224
    Dir    : assets/models/
    Size   : ≤ 300 MB (vision encoder, INT8)
    Speed  : Slower — use for nightly/background indexing jobs
    Generate: python tools/quantize_siglip.py --tier hq

Architecture — two loading modes
---------------------------------
SPLIT MODE  (preferred — run tools/export_split_models.py once to enable)
    vision_encoder_int8.onnx  ~26 MB  ← used only during frame upload
    text_encoder_int8.onnx    ~26 MB  ← used only during text search
    • Each encoder runs in its own ORT session with zero dummy inputs.
    • ~50% faster per call vs the combined model.

COMBINED MODE  (automatic fallback if split files are missing)
    model_quantized.onnx  ~210 MB  ← both encoders in one graph
    • Dummy pixel_values fed to the text path; dummy input_ids to the vision path.
    • Slower but always available — nothing needs to be regenerated.

The engine detects which mode to use at startup and switches transparently.
No calling code needs to change; get_image_features / get_text_features work
identically in both modes.

How the two encoders align in embedding space
----------------------------------------------
SigLIP is trained with contrastive learning on (image, text) pairs.
Both encoders output L2-normalised 768-dim vectors in the SAME shared space.
A text embedding for "dog running in park" will have high cosine similarity
to image embeddings of frames showing exactly that — regardless of which
encoder produced each vector.  That alignment is the foundation of semantic search.
"""

import os
from threading import Lock

import numpy as np
import onnxruntime as ort
from transformers import SiglipProcessor

from app.core.config import settings
from app.utils.hardware import get_execution_providers


class SigLipEngine:
    """
    Singleton ONNX-based SigLIP inference engine.

    Public API
    ----------
    get_components()                           → (session, processor, device)
    get_image_features(pixel_values_np)        → np.ndarray [N, 768] float32 L2-norm
    get_text_features(input_ids, attn_mask)    → np.ndarray [N, 768] float32 L2-norm
    is_split_mode                              → bool property
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal: session factory
    # ------------------------------------------------------------------
    def _make_session(self, model_path: str, label: str = "") -> ort.InferenceSession:
        """Build a hardware-optimised ORT session from a .onnx file."""
        sess_opts = ort.SessionOptions()

        if settings.ORT_INTRA_OP_THREADS > 0:
            sess_opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
        if settings.ORT_INTER_OP_THREADS > 0:
            sess_opts.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
        if settings.ORT_ENABLE_MEMORY_ARENA:
            sess_opts.enable_cpu_mem_arena = True

        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers, provider_options = get_execution_providers()
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options if provider_options else None,
        )

        display = label or os.path.basename(model_path)
        active = session.get_providers()[0]
        print(f"  [SigLipEngine] Loaded '{display}' via [{active}]")
        return session

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        # ── Resolve paths from active tier (base | hq) ──────────────────────
        # settings.ACTIVE_ONNX_DIR and settings.ACTIVE_VISION_FILE are @property
        # accessors that resolve to the correct directory and filename based on
        # the SIGLIP_TIER env var.  No other code needs to change when switching.
        active_dir    = settings.ACTIVE_ONNX_DIR
        active_vision = settings.ACTIVE_VISION_FILE

        vision_path   = os.path.join(active_dir, active_vision)
        text_path     = os.path.join(active_dir, settings.ONNX_TEXT_MODEL_FILE)
        combined_path = os.path.join(active_dir, settings.ONNX_MODEL_FILE)

        tier = settings.SIGLIP_TIER.upper()
        flag = settings.USE_SPLIT_MODELS   # True | False | None

        print(f"[SigLipEngine] Active tier: {tier} | dir: {active_dir}")

        if flag is True:
            # ── FORCED SPLIT ─────────────────────────────────────────────
            # Caller explicitly requires split models — raise early if missing.
            for path, label in [(vision_path, "vision"), (text_path, "text")]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"USE_SPLIT_MODELS=true but {label} encoder not found:\n"
                        f"  {path}\n"
                        f"  For tier=base: run 'python tools/export_split_models.py'\n"
                        f"  For tier=hq:   run 'python tools/quantize_siglip.py --tier hq'"
                    )
            print("[SigLipEngine] USE_SPLIT_MODELS=true → loading split encoders.")
            self._load_split(vision_path, text_path)

        elif flag is False:
            # ── FORCED COMBINED ──────────────────────────────────────────
            print("[SigLipEngine] USE_SPLIT_MODELS=false → loading combined model.")
            self._load_combined(combined_path)

        else:
            # ── AUTO-DETECT (default) ────────────────────────────────────
            if os.path.exists(vision_path) and os.path.exists(text_path):
                print("[SigLipEngine] Split models detected (auto) → loading vision + text separately.")
                self._load_split(vision_path, text_path)
            else:
                print(
                    f"[SigLipEngine] Split models not found (auto) → combined model.\n"
                    f"  Missing: {vision_path if not os.path.exists(vision_path) else text_path}\n"
                    f"  For tier=base: run 'python tools/export_split_models.py'\n"
                    f"  For tier=hq:   run 'python tools/quantize_siglip.py --tier hq'\n"
                    f"  Or set USE_SPLIT_MODELS=false in .env to silence this message."
                )
                self._load_combined(combined_path)

        # Processor is shared — same tokeniser / image processor for both modes.
        # MODEL_NAME is a @property that returns the correct HuggingFace ID for
        # the active tier (base or hq).
        self.processor = SiglipProcessor.from_pretrained(settings.MODEL_NAME)
        self.device = settings.DEVICE

    def _load_split(self, vision_path: str, text_path: str) -> None:
        """Load separate vision + text encoder sessions."""
        self._use_split = True
        self.vision_session = self._make_session(vision_path, "vision_encoder_int8")
        self.text_session   = self._make_session(text_path,   "text_encoder_int8")
        self.session = None   # not used in split mode

        self._vision_inputs  = {i.name for i in self.vision_session.get_inputs()}
        self._vision_outputs = [o.name for o in self.vision_session.get_outputs()]
        self._text_inputs    = {i.name for i in self.text_session.get_inputs()}
        self._text_outputs   = [o.name for o in self.text_session.get_outputs()]

        # Legacy compat aliases
        self._input_names  = self._vision_inputs | self._text_inputs
        self._output_names = self._vision_outputs + self._text_outputs

        print(
            f"  Vision inputs : {sorted(self._vision_inputs)}  outputs: {self._vision_outputs}\n"
            f"  Text   inputs : {sorted(self._text_inputs)}    outputs: {self._text_outputs}"
        )

    def _load_combined(self, model_path: str) -> None:
        """Load the combined single-ONNX model (both encoders in one graph)."""
        self._use_split = False
        self.session = self._make_session(model_path, "combined_model_int8")
        # In combined mode both split pointers alias the same session
        self.vision_session = self.session
        self.text_session   = self.session

        self._input_names  = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._vision_inputs  = self._input_names
        self._vision_outputs = self._output_names
        self._text_inputs    = self._input_names
        self._text_outputs   = self._output_names

        print(f"  Inputs : {sorted(self._input_names)}  Outputs: {self._output_names}")

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------
    @property
    def is_split_mode(self) -> bool:
        """True if running with separate vision + text encoder sessions."""
        return self._use_split

    # ------------------------------------------------------------------
    # Compatibility accessor
    # ------------------------------------------------------------------
    def get_components(self):
        """Return (session, processor, device_str) for callers that need them."""
        sess = self.vision_session if self._use_split else self.session
        return sess, self.processor, self.device

    # ------------------------------------------------------------------
    # Image embedding  (used during video upload / frame processing)
    # ------------------------------------------------------------------
    def get_image_features(self, pixel_values) -> np.ndarray:
        """
        Run vision-encoder inference.

        Parameters
        ----------
        pixel_values : np.ndarray  [N, 3, 224, 224]  float32
            Pre-processed by SiglipProcessor.

        Returns
        -------
        np.ndarray  [N, 768]  float32, L2-normalised
        """
        if hasattr(pixel_values, "cpu"):
            pixel_values = pixel_values.cpu().numpy()
        pixel_values = np.asarray(pixel_values, dtype=np.float32)

        if self._use_split:
            # ── SPLIT MODE: clean call — no dummy inputs ────────────────
            # vision_encoder expects only pixel_values → outputs image_embeds
            outputs = self.vision_session.run(["image_embeds"], {"pixel_values": pixel_values})
            return self._l2_normalise(outputs[0])

        else:
            # ── COMBINED MODE: feed dummy input_ids alongside pixel_values
            feed = {"pixel_values": pixel_values}
            if "input_ids" in self._input_names:
                feed["input_ids"] = np.zeros((pixel_values.shape[0], 64), dtype=np.int64)

            if "image_embeds" in self._output_names:
                outputs = self.session.run(["image_embeds"], feed)
                embeddings = outputs[0]
            else:
                outputs = self.session.run(None, feed)
                embeddings = outputs[1] if len(outputs) >= 2 else outputs[0].mean(axis=1)

            return self._l2_normalise(embeddings)

    # ------------------------------------------------------------------
    # Text embedding  (used during search query)
    # ------------------------------------------------------------------
    def get_text_features(self, input_ids, attention_mask=None) -> np.ndarray:
        """
        Run text-encoder inference.

        Parameters
        ----------
        input_ids      : np.ndarray  [N, seq_len]  int64
        attention_mask : np.ndarray  [N, seq_len]  int64  (optional)

        Returns
        -------
        np.ndarray  [N, 768]  float32, L2-normalised
        """
        if hasattr(input_ids, "cpu"):
            input_ids = input_ids.cpu().numpy()
        input_ids = np.asarray(input_ids, dtype=np.int64)

        if self._use_split:
            # ── SPLIT MODE: clean call — no dummy pixel_values ──────────
            # text_encoder expects input_ids + attention_mask → text_embeds
            feed: dict = {"input_ids": input_ids}
            if "attention_mask" in self._text_inputs:
                if attention_mask is not None:
                    if hasattr(attention_mask, "cpu"):
                        attention_mask = attention_mask.cpu().numpy()
                    feed["attention_mask"] = np.asarray(attention_mask, dtype=np.int64)
                else:
                    feed["attention_mask"] = np.ones(input_ids.shape, dtype=np.int64)

            outputs = self.text_session.run(["text_embeds"], feed)
            return self._l2_normalise(outputs[0])

        else:
            # ── COMBINED MODE: feed dummy pixel_values alongside text inputs
            feed = {"input_ids": input_ids}
            if "attention_mask" in self._input_names:
                if attention_mask is not None:
                    if hasattr(attention_mask, "cpu"):
                        attention_mask = attention_mask.cpu().numpy()
                    feed["attention_mask"] = np.asarray(attention_mask, dtype=np.int64)
                else:
                    feed["attention_mask"] = np.ones(input_ids.shape, dtype=np.int64)

            if "pixel_values" in self._input_names:
                feed["pixel_values"] = np.zeros(
                    (input_ids.shape[0], 3, 224, 224), dtype=np.float32
                )

            if "text_embeds" in self._output_names:
                outputs = self.session.run(["text_embeds"], feed)
                embeddings = outputs[0]
            else:
                outputs = self.session.run(None, feed)
                embeddings = outputs[1] if len(outputs) >= 2 else outputs[0].mean(axis=1)

            return self._l2_normalise(embeddings)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _l2_normalise(embeddings: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return (embeddings / norm).astype(np.float32)
