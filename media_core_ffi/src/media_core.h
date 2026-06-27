/**
 * media_core.h
 * ============
 * Public FFI interface for the media_core_ffi Flutter Native Asset.
 *
 * All symbols declared here are exported from the compiled shared library
 * (media_core.dll on Windows, libmedia_core.so on Android) and are callable
 * directly from Dart via dart:ffi.
 *
 * Dart binding generation
 * -----------------------
 * Run ffigen against this header to auto-generate the Dart bindings:
 *
 *   dart run ffigen --config ffigen.yaml
 *
 * The generated file is committed to the repo at:
 *   lib/src/media_core_bindings.dart
 *
 * Threading model
 * ---------------
 * extract_audio_pcm() blocks the calling thread for the full duration of the
 * demux/decode/resample pipeline.  Always call it from a Dart Isolate
 * (compute() or Isolate.spawn()) — never from the UI thread.
 *
 * Memory contract
 * ---------------
 * The caller (Dart) MUST call free_audio_buffer(result.data) exactly once
 * after consuming the PCM samples.  Failure to do so leaks native heap memory
 * that is invisible to the Dart GC.
 */

#ifndef MEDIA_CORE_H
#define MEDIA_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── Export macros ────────────────────────────────────────────────────────── */

#if defined(_WIN32) || defined(_WIN64)
  #ifdef MEDIA_CORE_EXPORTS
    #define MEDIA_CORE_API __declspec(dllexport)
  #else
    #define MEDIA_CORE_API __declspec(dllimport)
  #endif
#else
  /* GCC / Clang: mark symbols as visible in the shared object */
  #define MEDIA_CORE_API __attribute__((visibility("default")))
#endif


/* ── Result types ─────────────────────────────────────────────────────────── */

/**
 * AudioBufferResult
 * -----------------
 * Returned by extract_audio_pcm().  Passed back across the FFI boundary
 * by value (struct, not pointer) so Dart can inspect both fields without
 * dereferencing anything.
 *
 * Fields
 * ------
 * data         : Pointer to a malloc()'d array of int16_t PCM samples.
 *                Layout: 16 kHz, mono, signed 16-bit little-endian (S16LE).
 *                NULL on failure.
 * sample_count : Number of valid int16_t elements in data[].
 *                0 on failure or if the file contains no speech (all VAD-gated).
 */
typedef struct {
    int16_t* data;
    int      sample_count;
} AudioBufferResult;

/**
 * VideoFrame
 * ----------
 * Represents a single normalized CHW Float32 frame (384x384).
 */
typedef struct {
    float*   data;         // [3 * 384 * 384] floats
    double   timestamp_s;  // pts_time
} VideoFrame;

/**
 * VideoExtractionResult
 * ---------------------
 * A batch of extracted keyframes ready for SigLIP / OCR ONNX sessions.
 */
typedef struct {
    VideoFrame* frames;
    int         frame_count;
} VideoExtractionResult;

/**
 * WhisperMelResult
 * ----------------
 * Holds the computed 80-bin Mel-spectrogram for Whisper ONNX encoder.
 */
typedef struct {
    float* data;
    int    size;
} WhisperMelResult;

/**
 * OcrBoundingBox
 * --------------
 * Represents the 4 corners of a detected text line:
 * [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
 */
typedef struct {
    float pts[8];
} OcrBoundingBox;

typedef struct {
    OcrBoundingBox* boxes;
    int box_count;
} OcrBoxResult;

/**
 * OcrCropResult
 * -------------
 * Represents a dynamically-sized, perspective-warped text line.
 * CHW Float32 tensor mapped to [-1.0, 1.0].
 */
typedef struct {
    float* data;
    int    width;
    int    height; // Usually 48 for SVTR models
} OcrCropResult;

/* ── Core API ─────────────────────────────────────────────────────────────── */

/**
 * ocr_extract_bboxes
 * ------------------
 * Extracts bounding boxes natively from the DBNet Float32 heatmap output.
 */
MEDIA_CORE_API OcrBoxResult ocr_extract_bboxes(const float* heatmap, int width, int height, float threshold);

MEDIA_CORE_API void free_ocr_boxes(OcrBoxResult result);

/**
 * ocr_crop_and_warp
 * -----------------
 * Native C++ bilinear interpolation engine to warp the 4-point bounding box
 * out of the original RGB image into a flat, fixed-height tensor.
 */
MEDIA_CORE_API OcrCropResult ocr_crop_and_warp(const uint8_t* image_rgb, int img_w, int img_h, OcrBoundingBox box, int target_height);

MEDIA_CORE_API void free_ocr_crop(OcrCropResult result);

/**
 * whisper_compute_mel
 * -------------------
 * Computes the 80-bin Mel-spectrogram from 16kHz PCM audio.
 * Uses pocketfft for STFT.
 */
MEDIA_CORE_API WhisperMelResult whisper_compute_mel(const int16_t* pcm_data, int sample_count);

MEDIA_CORE_API void free_whisper_mel(WhisperMelResult result);

/**
 * whisper_decode_tokens
 * ---------------------
 * Maps an array of BPE token IDs to a UTF-8 string using tokenizer.json.
 */
MEDIA_CORE_API char* whisper_decode_tokens(const int* tokens, int token_count, const char* tokenizer_json_path);

MEDIA_CORE_API void free_string(char* str);

/**
 * extract_video_frames
 * --------------------
 * Extracts scene-changed keyframes natively using FFmpeg and SAD pixel-diffing.
 * Normalizes RGB24 to Float32 CHW inside C++ to skip Dart processing.
 */
MEDIA_CORE_API VideoExtractionResult extract_video_frames(const char* video_path, double scene_threshold);

MEDIA_CORE_API void free_video_frames(VideoExtractionResult result);

/**
 * extract_audio_pcm
 * -----------------
 * Demux, decode, and resample the audio track of any supported video/audio
 * container file into a normalised 16 kHz mono S16LE PCM stream, then apply
 * a simple energy-based Voice Activity Detector (VAD) to suppress silence.
 *
 * Supported input containers : MP4, MOV, MKV, AVI, MP3, WAV, AAC, FLAC
 * Supported audio codecs     : AAC, MP3, FLAC, PCM S16, Opus, Vorbis
 * Output format              : 16000 Hz · Mono · int16_t (S16LE)
 *
 * Parameters
 * ----------
 * video_path   : Null-terminated UTF-8 path to the media file.
 *
 * Returns
 * -------
 * AudioBufferResult with:
 *   data != NULL, sample_count > 0   → success; caller must free data
 *   data == NULL, sample_count == 0  → failure (file not found, no audio track,
 *                                      codec error, or zero speech detected)
 *
 * Thread safety
 * -------------
 * Re-entrant: multiple calls with different paths may run concurrently.
 * Each call allocates its own independent output buffer.
 */
MEDIA_CORE_API AudioBufferResult extract_audio_pcm(const char* video_path);


/**
 * free_audio_buffer
 * -----------------
 * Release the native heap memory returned by extract_audio_pcm().
 *
 * MUST be called exactly once per successful extract_audio_pcm() call.
 * Passing NULL is safe (no-op).
 *
 * Parameters
 * ----------
 * buffer : The data pointer from AudioBufferResult.data.
 */
MEDIA_CORE_API void free_audio_buffer(int16_t* buffer);


/* ── VAD helpers (also exported for testing/standalone use) ───────────────── */

/**
 * vad_rms
 * -------
 * Compute the Root Mean Square energy of a PCM segment.
 *
 * Parameters
 * ----------
 * buf       : Pointer to int16_t PCM samples.
 * n_samples : Number of samples to evaluate.
 *
 * Returns
 * -------
 * RMS value in the range [0, 32767].  Returns 0 when n_samples == 0.
 */
MEDIA_CORE_API int vad_rms(const int16_t* buf, int n_samples);


/**
 * vad_is_speech
 * -------------
 * Classify a PCM segment as speech (1) or silence (0) using an energy gate.
 *
 * Parameters
 * ----------
 * buf       : Pointer to int16_t PCM samples.
 * n_samples : Number of samples to evaluate.
 * threshold : RMS threshold above which the segment is classified as speech.
 *             Use 300 (VAD_RMS_THRESHOLD) for the default production gate.
 *
 * Returns
 * -------
 * 1 if RMS > threshold (speech), 0 otherwise (silence).
 */
MEDIA_CORE_API int vad_is_speech(const int16_t* buf, int n_samples, int threshold);


#ifdef __cplusplus
}
#endif

#endif /* MEDIA_CORE_H */
