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


/* ── Core API ─────────────────────────────────────────────────────────────── */

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
