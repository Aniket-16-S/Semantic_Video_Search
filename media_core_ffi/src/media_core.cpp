#include "media_core.h"
// Cache invalidation tick 4

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>   /* sqrt() for RMS computation */
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <complex>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <algorithm>

#if __has_include("pocketfft_hdronly.h")
#include "pocketfft_hdronly.h"
#define HAS_POCKETFFT 1
#endif

#if __has_include("json.hpp")
#include "json.hpp"
#define HAS_JSON 1
#endif

#define TARGET_SAMPLE_RATE  16000
#define VAD_RMS_THRESHOLD   300
#define INITIAL_CAPACITY_SAMPLES  (TARGET_SAMPLE_RATE * 60)

// Helper: Convert HWC RGB24 to CHW Float32 and normalize to [-1.0, 1.0]
static float* normalize_hwc_to_chw(const uint8_t* hwc_pixels, int width, int height) {
    int num_pixels = width * height;
    float* chw = (float*)malloc(3 * num_pixels * sizeof(float));
    if (!chw) return nullptr;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_idx = (y * width + x) * 3;
            // Channel 0 (R)
            chw[0 * num_pixels + y * width + x] = (hwc_pixels[src_idx + 0] / 127.5f) - 1.0f;
            // Channel 1 (G)
            chw[1 * num_pixels + y * width + x] = (hwc_pixels[src_idx + 1] / 127.5f) - 1.0f;
            // Channel 2 (B)
            chw[2 * num_pixels + y * width + x] = (hwc_pixels[src_idx + 2] / 127.5f) - 1.0f;
        }
    }
    return chw;
}

// Helper: Compute SAD (Sum of Absolute Differences) between two RGB24 frames
static double compute_sad(const uint8_t* frameA, const uint8_t* frameB, int num_bytes) {
    long long diff = 0;
    for (int i = 0; i < num_bytes; ++i) {
        diff += std::abs((int)frameA[i] - (int)frameB[i]);
    }
    // Normalize to [0.0, 1.0] representing average pixel difference percentage
    return (double)diff / ((double)num_bytes * 255.0);
}

MEDIA_CORE_API VideoExtractionResult extract_video_frames(const char* video_path, double scene_threshold) {
    VideoExtractionResult result = { nullptr, 0 };
    if (!video_path) return result;

    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, video_path, nullptr, nullptr) < 0) return result;
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        avformat_close_input(&format_ctx);
        return result;
    }

    int video_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx == -1) {
        avformat_close_input(&format_ctx);
        return result;
    }

    AVStream* stream = format_ctx->streams[video_stream_idx];
    AVCodecParameters* codec_params = stream->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        avformat_close_input(&format_ctx);
        return result;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codec_params);
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return result;
    }

    int target_w = 384;
    int target_h = 384;
    struct SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
        target_w, target_h, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    std::vector<VideoFrame> frames;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb_frame = av_frame_alloc();
    
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, target_w, target_h, 1);
    uint8_t* rgb_buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer, AV_PIX_FMT_RGB24, target_w, target_h, 1);

    uint8_t* prev_frame_buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
    bool has_prev_frame = false;
    double time_base = av_q2d(stream->time_base);

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    sws_scale(sws_ctx, (const uint8_t* const*)frame->data, frame->linesize, 0, codec_ctx->height,
                              rgb_frame->data, rgb_frame->linesize);

                    bool is_keyframe = false;
                    if (!has_prev_frame) {
                        is_keyframe = true;
                    } else {
                        double sad = compute_sad(rgb_buffer, prev_frame_buffer, num_bytes);
                        if (sad > scene_threshold) {
                            is_keyframe = true;
                        }
                    }

                    if (is_keyframe) {
                        float* chw_data = normalize_hwc_to_chw(rgb_buffer, target_w, target_h);
                        double pts_s = frame->pts == AV_NOPTS_VALUE ? 0 : frame->pts * time_base;
                        frames.push_back({chw_data, pts_s});
                        memcpy(prev_frame_buffer, rgb_buffer, num_bytes);
                        has_prev_frame = true;
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    av_free(rgb_buffer);
    av_free(prev_frame_buffer);
    av_frame_free(&rgb_frame);
    av_frame_free(&frame);
    av_packet_free(&packet);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    if (frames.size() > 0) {
        result.frames = (VideoFrame*)malloc(frames.size() * sizeof(VideoFrame));
        memcpy(result.frames, frames.data(), frames.size() * sizeof(VideoFrame));
        result.frame_count = frames.size();
    }

    return result;
}

MEDIA_CORE_API void free_video_frames(VideoExtractionResult result) {
    if (result.frames) {
        for (int i = 0; i < result.frame_count; ++i) {
            if (result.frames[i].data) free(result.frames[i].data);
        }
        free(result.frames);
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 * Audio Pipeline Logic (migrated from media_core.c)
 * ══════════════════════════════════════════════════════════════════════════ */

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

MEDIA_CORE_API int vad_is_speech(const int16_t* buf, int n_samples, int threshold)
{
    return vad_rms(buf, n_samples) > threshold ? 1 : 0;
}

MEDIA_CORE_API void free_audio_buffer(int16_t* buffer)
{
    if (buffer != NULL) free(buffer);
}

MEDIA_CORE_API AudioBufferResult extract_audio_pcm(const char* video_path)
{
    AudioBufferResult result = { NULL, 0 };
    if (video_path == NULL) return result;

    AVFormatContext* format_ctx = NULL;
    if (avformat_open_input(&format_ctx, video_path, NULL, NULL) < 0) return result;

    if (avformat_find_stream_info(format_ctx, NULL) < 0) goto cleanup_format;

    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            break;
        }
    }
    if (audio_stream_idx == -1) goto cleanup_format;

    {
    AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (codec == NULL) goto cleanup_format;

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (codec_ctx == NULL) goto cleanup_format;

    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) goto cleanup_codec;
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) goto cleanup_codec;

    SwrContext* swr_ctx = swr_alloc_set_opts(
        NULL, AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_S16, TARGET_SAMPLE_RATE,
        codec_ctx->channel_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate, 0, NULL
    );
    if (swr_ctx == NULL) goto cleanup_codec;
    if (swr_init(swr_ctx) < 0) { swr_free(&swr_ctx); goto cleanup_codec; }

    int capacity = INITIAL_CAPACITY_SAMPLES;
    int16_t* output_buffer = (int16_t*)malloc((size_t)capacity * sizeof(int16_t));
    if (output_buffer == NULL) { swr_free(&swr_ctx); goto cleanup_codec; }
    int total_samples = 0;

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == audio_stream_idx) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    int max_out = (int)av_rescale_rnd(
                        swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                        TARGET_SAMPLE_RATE, codec_ctx->sample_rate, AV_ROUND_UP
                    );

                    if (total_samples + max_out > capacity) {
                        while (capacity < total_samples + max_out) capacity *= 2;
                        int16_t* new_buf = (int16_t*)realloc(output_buffer, (size_t)capacity * sizeof(int16_t));
                        if (new_buf == NULL) {
                            av_frame_unref(frame);
                            av_packet_unref(packet);
                            goto flush_and_return;
                        }
                        output_buffer = new_buf;
                    }

                    uint8_t* out_ptr = (uint8_t*)(output_buffer + total_samples);
                    int converted = swr_convert(swr_ctx, &out_ptr, max_out, (const uint8_t**)frame->data, frame->nb_samples);

                    if (converted > 0) {
                        if (vad_is_speech(output_buffer + total_samples, converted, VAD_RMS_THRESHOLD)) {
                            total_samples += converted;
                        }
                    }
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(packet);
    }

flush_and_return:
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

    result.data = output_buffer;
    result.sample_count = total_samples;

    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr_ctx);
cleanup_codec:
    avcodec_free_context(&codec_ctx);
cleanup_format:
    avformat_close_input(&format_ctx);
    }

    return result;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Whisper Audio Math & Tokenizer (Phase 4 Shift)
 * ══════════════════════════════════════════════════════════════════════════ */

MEDIA_CORE_API WhisperMelResult whisper_compute_mel(const int16_t* pcm_data, int sample_count) {
    WhisperMelResult result = { nullptr, 0 };
    if (!pcm_data || sample_count <= 0) return result;

#ifdef HAS_POCKETFFT
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;
    int num_frames = (sample_count - n_fft) / hop_length + 1;
    if (num_frames < 0) num_frames = 0;
    
    result.size = n_mels * 3000; // Force exact padding to 3000 frames for Whisper ONNX shape
    result.data = (float*)calloc(result.size, sizeof(float));

    // Generate Hanning window
    std::vector<double> window(n_fft);
    for(int i=0; i<n_fft; ++i) {
        window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / n_fft));
    }

    // Mel filterbank generation (Slaney)
    auto hz_to_mel = [](double hz) {
        return (hz < 1000.0) ? (hz * 3.0 / 200.0) : (15.0 + 27.566409 * log10(hz / 1000.0));
    };
    auto mel_to_hz = [](double mel) {
        return (mel < 15.0) ? (mel * 200.0 / 3.0) : (1000.0 * pow(10.0, (mel - 15.0) / 27.566409));
    };
    
    double fmin = 0.0, fmax = 8000.0;
    double mmin = hz_to_mel(fmin);
    double mmax = hz_to_mel(fmax);
    std::vector<double> mel_points(n_mels + 2);
    for(int i=0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_to_hz(mmin + i * (mmax - mmin) / (n_mels + 1));
    }
    std::vector<int> f_points(n_mels + 2);
    for(int i=0; i < n_mels + 2; ++i) {
        f_points[i] = (int)floor((n_fft + 1) * mel_points[i] / 16000.0);
    }
    
    std::vector<std::vector<double>> mel_filters(n_mels, std::vector<double>(n_fft / 2 + 1, 0.0));
    for (int m = 1; m <= n_mels; ++m) {
        int f_m_minus = f_points[m - 1];
        int f_m = f_points[m];
        int f_m_plus = f_points[m + 1];
        
        double weight_factor = 2.0 / (mel_points[m + 1] - mel_points[m - 1]);
        
        for (int k = f_m_minus; k < f_m; ++k) {
            mel_filters[m - 1][k] = weight_factor * (k - f_m_minus) / (f_m - f_m_minus);
        }
        for (int k = f_m; k < f_m_plus; ++k) {
            mel_filters[m - 1][k] = weight_factor * (f_m_plus - k) / (f_m_plus - f_m);
        }
    }

    pocketfft::shape_t shape = { (size_t)n_fft };
    pocketfft::stride_t stride = { sizeof(double) };
    pocketfft::shape_t axes = { 0 };

    for(int f=0; f<num_frames && f < 3000; ++f) {
        std::vector<double> frame(n_fft, 0.0);
        for(int i=0; i<n_fft; ++i) {
            frame[i] = pcm_data[f * hop_length + i] / 32768.0 * window[i];
        }
        
        std::vector<std::complex<double>> res(n_fft / 2 + 1);
        pocketfft::stride_t res_stride = { sizeof(std::complex<double>) };
        
        pocketfft::r2c(shape, stride, res_stride, axes, pocketfft::FORWARD, frame.data(), res.data(), 1.0);
        
        std::vector<double> power(n_fft / 2 + 1);
        for(int i=0; i<=n_fft/2; ++i) {
            power[i] = (res[i].real() * res[i].real() + res[i].imag() * res[i].imag());
        }
        
        for(int m=0; m<n_mels; ++m) {
            double dot = 0.0;
            for(int k=0; k<=n_fft/2; ++k) {
                dot += mel_filters[m][k] * power[k];
            }
            double val = log10(std::max(dot, 1e-10));
            result.data[m * 3000 + f] = (float)val;
        }
    }
    
    float max_val = -1e10f;
    for(int i=0; i<n_mels * 3000; ++i) {
        if (result.data[i] > max_val) max_val = result.data[i];
    }
    for(int i=0; i<n_mels * 3000; ++i) {
        float val = result.data[i];
        val = (val - max_val + 8.0f) / 4.0f; // Approx normalize based on Whisper heuristic
        if (val < -1.0f) val = -1.0f;
        if (val > 1.0f) val = 1.0f;
        result.data[i] = val;
    }
#else
    // Fallback stub if pocketfft_hdronly.h is missing
    std::cerr << "[media_core_ffi] WARNING: pocketfft_hdronly.h not found. Returning dummy Mel-spectrogram." << std::endl;
    result.size = 80 * 3000;
    result.data = (float*)calloc(result.size, sizeof(float));
#endif

    return result;
}

MEDIA_CORE_API void free_whisper_mel(WhisperMelResult result) {
    if (result.data) {
        free(result.data);
    }
}

MEDIA_CORE_API char* whisper_decode_tokens(const int* tokens, int token_count, const char* tokenizer_json_path) {
    if (!tokens || token_count <= 0) return nullptr;

#ifdef HAS_JSON
    // Parse tokenizer.json and map IDs to strings
    std::string text = "";
    try {
        std::ifstream f(tokenizer_json_path);
        if (f.is_open()) {
            nlohmann::json data = nlohmann::json::parse(f);
            auto vocab = data["model"]["vocab"];
            std::unordered_map<int, std::string> id_to_token;
            for (auto& el : vocab.items()) {
                id_to_token[el.value().get<int>()] = el.key();
            }

            for (int i = 0; i < token_count; ++i) {
                int token = tokens[i];
                if (id_to_token.count(token)) {
                    std::string token_str = id_to_token[token];
                    // Handle GPT-2 byte mapping: "Ġ" -> " "
                    for (size_t j = 0; j < token_str.length(); ++j) {
                        if ((unsigned char)token_str[j] == 0xC4 && j + 1 < token_str.length() && (unsigned char)token_str[j+1] == 0xA0) {
                            text += " ";
                            j++;
                        } else {
                            text += token_str[j]; // Naive byte fallback
                        }
                    }
                }
            }
        } else {
            text = "Tokenizer file not found.";
        }
    } catch (...) {
        text = "JSON parse error";
    }
    
    char* cstr = (char*)malloc(text.length() + 1);
    strcpy(cstr, text.c_str());
    return cstr;
#else
    std::cerr << "[media_core_ffi] WARNING: json.hpp not found. Returning dummy string." << std::endl;
    const char* dummy = "Sample transcribed text (Native C++ stub)";
    char* cstr = (char*)malloc(strlen(dummy) + 1);
    strcpy(cstr, dummy);
    return cstr;
#endif
}

MEDIA_CORE_API void free_string(char* str) {
    if (str) {
        free(str);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * OCR DBNet Math & Geometry (Phase 4 Shift)
 * ══════════════════════════════════════════════════════════════════════════ */

MEDIA_CORE_API OcrBoxResult ocr_extract_bboxes(const float* heatmap, int width, int height, float threshold) {
    OcrBoxResult result = { nullptr, 0 };
    if (!heatmap || width <= 0 || height <= 0) return result;

    std::vector<uint8_t> visited(width * height, 0);
    std::vector<OcrBoundingBox> boxes;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (heatmap[y * width + x] > threshold && !visited[y * width + x]) {
                // BFS to find connected blob
                std::vector<std::pair<int, int>> blob;
                std::queue<std::pair<int, int>> q;
                q.push({x, y});
                visited[y * width + x] = 1;
                while (!q.empty()) {
                    auto p = q.front(); q.pop();
                    blob.push_back(p);
                    // 4-way neighbors
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};
                    for (int i=0; i<4; i++) {
                        int nx = p.first + dx[i];
                        int ny = p.second + dy[i];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int nidx = ny * width + nx;
                            if (!visited[nidx] && heatmap[nidx] > threshold) {
                                visited[nidx] = 1;
                                q.push({nx, ny});
                            }
                        }
                    }
                }
                if (blob.size() < 10) continue; // Min area filter
                
                // PCA-based Rotated Bounding Box
                double mean_x = 0, mean_y = 0;
                for (auto p : blob) { mean_x += p.first; mean_y += p.second; }
                mean_x /= blob.size(); mean_y /= blob.size();
                
                double cxx = 0, cyy = 0, cxy = 0;
                for (auto p : blob) {
                    double dx = p.first - mean_x;
                    double dy = p.second - mean_y;
                    cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
                }
                cxx /= blob.size(); cyy /= blob.size(); cxy /= blob.size();
                
                double trace = cxx + cyy;
                double det = cxx * cyy - cxy * cxy;
                double l1 = trace/2.0 + sqrt(std::max(0.0, trace*trace/4.0 - det));
                
                double vx1 = l1 - cyy, vy1 = cxy;
                if (vx1 == 0 && vy1 == 0) { vx1 = 1; vy1 = 0; }
                double norm = sqrt(vx1*vx1 + vy1*vy1);
                vx1 /= norm; vy1 /= norm;
                
                double vx2 = -vy1, vy2 = vx1;
                
                double min_p1 = 1e9, max_p1 = -1e9;
                double min_p2 = 1e9, max_p2 = -1e9;
                for (auto p : blob) {
                    double dx = p.first - mean_x;
                    double dy = p.second - mean_y;
                    double p1 = dx * vx1 + dy * vy1;
                    double p2 = dx * vx2 + dy * vy2;
                    if (p1 < min_p1) min_p1 = p1;
                    if (p1 > max_p1) max_p1 = p1;
                    if (p2 < min_p2) min_p2 = p2;
                    if (p2 > max_p2) max_p2 = p2;
                }
                
                // Vatti clip approximation for DBNet unclip
                double w = max_p1 - min_p1;
                double h = max_p2 - min_p2;
                if (w <= 0 || h <= 0) continue;
                double area = w * h;
                double perimeter = 2 * (w + h);
                double unclip_ratio = 1.5;
                double distance = area * unclip_ratio / perimeter;
                
                min_p1 -= distance; max_p1 += distance;
                min_p2 -= distance; max_p2 += distance;
                
                OcrBoundingBox box;
                // top-left (assuming p1 is width axis, p2 is height axis)
                // We order them tl, tr, br, bl
                box.pts[0] = mean_x + min_p1 * vx1 + min_p2 * vx2;
                box.pts[1] = mean_y + min_p1 * vy1 + min_p2 * vy2;
                
                box.pts[2] = mean_x + max_p1 * vx1 + min_p2 * vx2;
                box.pts[3] = mean_y + max_p1 * vy1 + min_p2 * vy2;
                
                box.pts[4] = mean_x + max_p1 * vx1 + max_p2 * vx2;
                box.pts[5] = mean_y + max_p1 * vy1 + max_p2 * vy2;
                
                box.pts[6] = mean_x + min_p1 * vx1 + max_p2 * vx2;
                box.pts[7] = mean_y + min_p1 * vy1 + max_p2 * vy2;
                
                // Enforce that tl is actually the top-leftmost point visually
                // DBNet returns points in clockwise order. We sort them roughly
                std::vector<std::pair<float, float>> pts(4);
                for(int i=0; i<4; i++) pts[i] = {box.pts[i*2], box.pts[i*2+1]};
                
                std::sort(pts.begin(), pts.end(), [](auto a, auto b) {
                    return a.first < b.first;
                });
                std::vector<std::pair<float, float>> lefts = {pts[0], pts[1]};
                std::vector<std::pair<float, float>> rights = {pts[2], pts[3]};
                
                if (lefts[0].second > lefts[1].second) std::swap(lefts[0], lefts[1]);
                if (rights[0].second > rights[1].second) std::swap(rights[0], rights[1]);
                
                box.pts[0] = lefts[0].first; box.pts[1] = lefts[0].second;   // tl
                box.pts[2] = rights[0].first; box.pts[3] = rights[0].second; // tr
                box.pts[4] = rights[1].first; box.pts[5] = rights[1].second; // br
                box.pts[6] = lefts[1].first; box.pts[7] = lefts[1].second;   // bl
                
                boxes.push_back(box);
            }
        }
    }

    if (boxes.size() > 0) {
        result.boxes = (OcrBoundingBox*)malloc(boxes.size() * sizeof(OcrBoundingBox));
        memcpy(result.boxes, boxes.data(), boxes.size() * sizeof(OcrBoundingBox));
        result.box_count = boxes.size();
    }
    return result;
}

MEDIA_CORE_API void free_ocr_boxes(OcrBoxResult result) {
    if (result.boxes) free(result.boxes);
}

MEDIA_CORE_API OcrCropResult ocr_crop_and_warp(const float* image_chw, int img_w, int img_h, OcrBoundingBox box, int target_height) {
    OcrCropResult result = { nullptr, 0, 0 };
    if (!image_chw || img_w <= 0 || img_h <= 0) return result;

    double w1 = sqrt(pow(box.pts[2] - box.pts[0], 2) + pow(box.pts[3] - box.pts[1], 2));
    double w2 = sqrt(pow(box.pts[4] - box.pts[6], 2) + pow(box.pts[5] - box.pts[7], 2));
    double h1 = sqrt(pow(box.pts[6] - box.pts[0], 2) + pow(box.pts[7] - box.pts[1], 2));
    double h2 = sqrt(pow(box.pts[4] - box.pts[2], 2) + pow(box.pts[5] - box.pts[3], 2));

    int out_w = std::max((int)w1, (int)w2);
    int out_h = std::max((int)h1, (int)h2);
    if (out_h == 0) return result;

    int final_w = out_w * target_height / out_h;
    if (final_w < target_height) final_w = target_height;

    result.width = final_w;
    result.height = target_height;
    int num_pixels = final_w * target_height;
    result.data = (float*)malloc(3 * num_pixels * sizeof(float));

    int src_plane = img_w * img_h;

    // Bilinear quadrilateral interpolation
    for (int y = 0; y < target_height; ++y) {
        double v = (double)y / (double)(target_height - 1);
        for (int x = 0; x < final_w; ++x) {
            double u = (double)x / (double)(final_w - 1);
            
            double px = (1-u)*(1-v)*box.pts[0] + u*(1-v)*box.pts[2] + u*v*box.pts[4] + (1-u)*v*box.pts[6];
            double py = (1-u)*(1-v)*box.pts[1] + u*(1-v)*box.pts[3] + u*v*box.pts[5] + (1-u)*v*box.pts[7];
            
            int src_x = (int)std::floor(px);
            int src_y = (int)std::floor(py);
            double dx = px - src_x;
            double dy = py - src_y;
            
            for (int c = 0; c < 3; ++c) {
                float val = 0.0f;
                if (src_x >= 0 && src_x < img_w - 1 && src_y >= 0 && src_y < img_h - 1) {
                    double p00 = image_chw[c * src_plane + src_y * img_w + src_x];
                    double p10 = image_chw[c * src_plane + src_y * img_w + src_x + 1];
                    double p01 = image_chw[c * src_plane + (src_y + 1) * img_w + src_x];
                    double p11 = image_chw[c * src_plane + (src_y + 1) * img_w + src_x + 1];
                    
                    val = (float)((1-dx)*(1-dy)*p00 + dx*(1-dy)*p10 + (1-dx)*dy*p01 + dx*dy*p11);
                }
                result.data[c * num_pixels + y * final_w + x] = val;
            }
        }
    }
    return result;
}

MEDIA_CORE_API void free_ocr_crop(OcrCropResult result) {
    if (result.data) free(result.data);
}
