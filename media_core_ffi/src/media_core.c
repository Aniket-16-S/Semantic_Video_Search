/**
 * media_core.c
 * ============
 * Unified low-level audio extraction, resampling, and Voice Activity Detection
 * engine for the media_core_ffi Flutter Native Asset plugin.
 *
 * Compiled by Flutter's Native Assets build system:
 *   - Windows  : MSVC (cl.exe) or MinGW-w64 (gcc), links against static FFmpeg libs
 *   - Android  : Android NDK Clang (aarch64-linux-android26-clang), static FFmpeg libs
 *
 * The build.dart orchestration script (in the package root) drives compilation
 * automatically whenever `flutter run` or `flutter build` is invoked.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design decisions
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * 1. Single-file compilation unit
 *    All VAD logic lives here alongside the demux pipeline to minimise the
 *    build system complexity inside Native Assets (one source → one .o → one .so).
 *    The public VAD symbols are still exported (see media_core.h) for unit tests.
 *
 * 2. Incremental buffer growth (1-minute slabs)
 *    The output PCM buffer starts at 1 minute of capacity and doubles whenever
 *    it fills.  This strategy avoids a full-file pre-scan while staying O(1)
 *    amortised per sample — essential for long videos with large audio tracks.
 *
 * 3. VAD applied per-decoded-chunk, not per-sample
 *    The RMS gate is evaluated on each converted chunk (typically 1024–4096
 *    samples after SwrContext resampling).  Only chunks that exceed the energy
 *    threshold are appended to the output buffer, eliminating silent segments
 *    before they ever reach the Dart layer or Whisper Tiny.
 *
 * 4. goto-based cleanup (idiomatic C for multi-resource teardown)
 *    libav* requires strict teardown ordering; goto avoids nested if-else
 *    pyramids while keeping each label's responsibility clear.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Memory contract (see also media_core.h)
 * ─────────────────────────────────────────────────────────────────────────────
 * extract_audio_pcm() malloc()s the output buffer.
 * The caller (Dart FFI) MUST call free_audio_buffer(result.data) when done.
 * Passing NULL to free_audio_buffer is a safe no-op.
 */

#include "media_core.h"

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>   /* sqrt() for RMS computation */

/* ── Constants ────────────────────────────────────────────────────────────── */

/** Output sample rate fed to Whisper Tiny (must be exactly 16000 Hz). */
#define TARGET_SAMPLE_RATE  16000

/**
 * RMS energy threshold for Voice Activity Detection.
 *
 * int16_t PCM samples range from −32768 to +32767.
 * An RMS of 300 corresponds roughly to −40 dBFS — well above quantisation
 * noise but comfortably below natural speech levels (typically 1000–8000).
 *
 * Tune this value if you observe:
 *   - Too many silent chunks kept   → raise the threshold
 *   - Soft speech being dropped     → lower the threshold
 */
#define VAD_RMS_THRESHOLD   300

/**
 * Initial PCM buffer capacity in samples.
 * 1 minute @ 16 kHz = 960 000 int16_t samples ≈ 1.8 MB — a reasonable slab
 * for most short-to-medium clips before any realloc() is needed.
 */
#define INITIAL_CAPACITY_SAMPLES  (TARGET_SAMPLE_RATE * 60)


/* ══════════════════════════════════════════════════════════════════════════
 * Section 1 — VAD helpers (also exported for standalone testing)
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * vad_rms — Root Mean Square energy of a PCM segment.
 *
 * Uses 64-bit accumulator to avoid overflow when squaring int16_t values
 * (max squared value: 32767² = 1,073,676,289 — fits safely in int64_t for
 * any practical chunk size up to ~4 billion samples before saturation).
 */
MEDIA_CORE_API int vad_rms(const int16_t* buf, int n_samples)
{
    if (buf == NULL || n_samples <= 0) return 0;

    int64_t accumulator = 0;
    for (int i = 0; i < n_samples; ++i) {
        int64_t s = (int64_t)buf[i];
        accumulator += s * s;
    }
    return (int)sqrt((double)accumulator / (double)n_samples);
}


/**
 * vad_is_speech — Classify a PCM segment as speech or silence.
 *
 * Returns 1 (speech) when RMS > threshold, 0 (silence) otherwise.
 * A segment classified as silence will be discarded by extract_audio_pcm().
 */
MEDIA_CORE_API int vad_is_speech(const int16_t* buf, int n_samples, int threshold)
{
    return vad_rms(buf, n_samples) > threshold ? 1 : 0;
}


/* ══════════════════════════════════════════════════════════════════════════
 * Section 2 — Memory helpers
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * free_audio_buffer — Release the buffer returned by extract_audio_pcm().
 *
 * Dart FFI callers MUST invoke this exactly once per successful extraction.
 * Passing NULL is safe (no-op matching free(NULL) semantics).
 */
MEDIA_CORE_API void free_audio_buffer(int16_t* buffer)
{
    if (buffer != NULL) {
        free(buffer);
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 * Section 3 — Core audio demux / decode / resample pipeline
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * extract_audio_pcm
 * -----------------
 * Full pipeline: open container → find audio stream → decode → resample to
 * 16 kHz Mono S16LE → VAD gate → return PCM buffer to Dart FFI.
 *
 * See media_core.h for the complete API contract.
 *
 * Implementation notes
 * --------------------
 * a) avformat_open_input / avformat_find_stream_info
 *    Standard libavformat stream probing.  av_log is suppressed at the
 *    library level; errors are signalled exclusively through the return value.
 *
 * b) avcodec_find_decoder / avcodec_alloc_context3 / avcodec_open2
 *    Standard codec setup — parameters copied from the demuxed stream so
 *    the codec context inherits channel layout, sample rate, etc.
 *
 * c) SwrContext (libswresample)
 *    Converts whatever input format the codec emits (e.g. FLTP, S16P, S32)
 *    into AV_SAMPLE_FMT_S16 at TARGET_SAMPLE_RATE, mono.
 *    swr_get_delay() is queried before each swr_convert() call to correctly
 *    estimate the maximum number of output samples (accounts for internal
 *    resampler buffering and fractional sample delays).
 *
 * d) Double-free safety
 *    output_buffer is set to NULL in the failure-before-allocation path so
 *    the codec_cleanup / cleanup goto labels safely skip free().
 *
 * e) av_packet_unref / av_frame_free
 *    Called unconditionally in all paths to prevent libavcodec memory leaks.
 */
MEDIA_CORE_API AudioBufferResult extract_audio_pcm(const char* video_path)
{
    /* ── Initialise result to failure state ─────────────────────────────── */
    AudioBufferResult result = { NULL, 0 };

    if (video_path == NULL) return result;

    /* ── libavformat context ─────────────────────────────────────────────── */
    AVFormatContext* format_ctx = NULL;

    if (avformat_open_input(&format_ctx, video_path, NULL, NULL) < 0) {
        /* File not found, permission denied, or unsupported container */
        return result;
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        /* Corrupt / truncated file — cannot determine stream parameters */
        goto cleanup_format;
    }

    /* ── Find the first audio stream ─────────────────────────────────────── */
    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            break;
        }
    }
    if (audio_stream_idx == -1) {
        /* No audio track — visual-only file (e.g. silent screen recording) */
        goto cleanup_format;
    }

    /* ── Codec setup ─────────────────────────────────────────────────────── */
    AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
    const AVCodec*     codec        = avcodec_find_decoder(codec_params->codec_id);
    if (codec == NULL) {
        /* Codec not compiled into this static FFmpeg build */
        goto cleanup_format;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (codec_ctx == NULL) goto cleanup_format;

    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0)
        goto cleanup_codec;

    if (avcodec_open2(codec_ctx, codec, NULL) < 0)
        goto cleanup_codec;

    /* ── SwrContext: normalise to 16 kHz · Mono · S16LE ─────────────────── */
    /*
     * ch_layout handling:
     *   Modern FFmpeg (≥5.1) uses AVChannelLayout.  Older libav still exposes
     *   channel_layout (uint64_t).  We use the legacy integer API here for
     *   broadest NDK / MinGW compatibility.  If you link against FFmpeg ≥6.0
     *   on a platform that deprecates the integer API, switch to
     *   swr_alloc_set_opts2() with AVChannelLayout structs.
     */
    SwrContext* swr_ctx = swr_alloc_set_opts(
        NULL,
        AV_CH_LAYOUT_MONO,          /* output: mono */
        AV_SAMPLE_FMT_S16,          /* output: signed 16-bit */
        TARGET_SAMPLE_RATE,         /* output: 16 000 Hz */
        codec_ctx->channel_layout,  /* input: from codec */
        codec_ctx->sample_fmt,      /* input: from codec (e.g. FLTP) */
        codec_ctx->sample_rate,     /* input: from codec */
        0, NULL
    );

    if (swr_ctx == NULL) goto cleanup_codec;

    if (swr_init(swr_ctx) < 0) {
        swr_free(&swr_ctx);
        goto cleanup_codec;
    }

    /* ── Output buffer ───────────────────────────────────────────────────── */
    int      capacity      = INITIAL_CAPACITY_SAMPLES;
    int16_t* output_buffer = (int16_t*)malloc((size_t)capacity * sizeof(int16_t));
    if (output_buffer == NULL) {
        swr_free(&swr_ctx);
        goto cleanup_codec;
    }
    int total_samples = 0;

    /* ── Decode loop ─────────────────────────────────────────────────────── */
    AVPacket packet;
    av_init_packet(&packet);   /* zero-initialise — required before av_read_frame */
    packet.data = NULL;
    packet.size = 0;

    AVFrame* frame = av_frame_alloc();
    if (frame == NULL) {
        free(output_buffer);
        output_buffer = NULL;
        swr_free(&swr_ctx);
        goto cleanup_codec;
    }

    while (av_read_frame(format_ctx, &packet) >= 0) {

        if (packet.stream_index != audio_stream_idx) {
            av_packet_unref(&packet);
            continue;   /* Skip video / subtitle packets */
        }

        /* Send compressed packet to decoder */
        if (avcodec_send_packet(codec_ctx, &packet) == 0) {

            /* Receive all decoded frames for this packet */
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {

                /*
                 * Estimate maximum output samples after resampling.
                 * swr_get_delay() returns the number of samples buffered inside
                 * the resampler that haven't been emitted yet (due to fractional
                 * sample-rate ratios).  Adding these to frame->nb_samples and
                 * rounding UP guarantees the output buffer is always large enough.
                 */
                int max_out = (int)av_rescale_rnd(
                    swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                    TARGET_SAMPLE_RATE,
                    codec_ctx->sample_rate,
                    AV_ROUND_UP
                );

                /* Grow output buffer if necessary (amortised doubling) */
                if (total_samples + max_out > capacity) {
                    while (capacity < total_samples + max_out)
                        capacity *= 2;
                    int16_t* new_buf = (int16_t*)realloc(
                        output_buffer,
                        (size_t)capacity * sizeof(int16_t)
                    );
                    if (new_buf == NULL) {
                        /* OOM — return what we have so far */
                        av_frame_unref(frame);
                        av_packet_unref(&packet);
                        goto flush_and_return;
                    }
                    output_buffer = new_buf;
                }

                /* Resample decoded frame into the tail of output_buffer */
                uint8_t* out_ptr = (uint8_t*)(output_buffer + total_samples);
                int converted = swr_convert(
                    swr_ctx,
                    &out_ptr,
                    max_out,
                    (const uint8_t**)frame->data,
                    frame->nb_samples
                );

                if (converted > 0) {
                    /*
                     * VAD gate: compute RMS on the freshly resampled chunk.
                     * Only speech chunks are retained in the output buffer.
                     * Silence chunks are discarded (total_samples NOT advanced).
                     */
                    if (vad_is_speech(output_buffer + total_samples, converted, VAD_RMS_THRESHOLD)) {
                        total_samples += converted;
                    }
                    /* else: chunk is silence — overwrite it on next iteration */
                }

                av_frame_unref(frame);
            } /* while avcodec_receive_frame */
        }

        av_packet_unref(&packet);
    } /* while av_read_frame */

flush_and_return:
    /*
     * Flush resampler: drain any remaining samples buffered in swr_ctx.
     * This handles files where the last decoded frame was silence-gated but
     * the resampler still holds partial output samples internally.
     */
    {
        int flushed;
        do {
            if (total_samples + TARGET_SAMPLE_RATE > capacity) {
                capacity *= 2;
                int16_t* tmp = (int16_t*)realloc(output_buffer, (size_t)capacity * sizeof(int16_t));
                if (tmp == NULL) break;
                output_buffer = tmp;
            }
            uint8_t* flush_ptr = (uint8_t*)(output_buffer + total_samples);
            flushed = swr_convert(swr_ctx, &flush_ptr, TARGET_SAMPLE_RATE, NULL, 0);
            if (flushed > 0 && vad_is_speech(output_buffer + total_samples, flushed, VAD_RMS_THRESHOLD)) {
                total_samples += flushed;
            }
        } while (flushed > 0);
    }

    /* ── Package result ──────────────────────────────────────────────────── */
    result.data         = output_buffer;
    result.sample_count = total_samples;

    /* ── Teardown ─────────────────────────────────────────────────────────── */
    av_frame_free(&frame);
    swr_free(&swr_ctx);

cleanup_codec:
    avcodec_free_context(&codec_ctx);

cleanup_format:
    avformat_close_input(&format_ctx);

    return result;
}
