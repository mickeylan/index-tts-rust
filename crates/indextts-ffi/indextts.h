#ifndef INDEXTTS_H
#define INDEXTTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef INDEXTTS_BUILD
#    define INDEXTTS_API __declspec(dllexport)
#  else
#    define INDEXTTS_API __declspec(dllimport)
#  endif
#else
#  define INDEXTTS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct IndexTtsModelHandle;
struct IndexTtsVoiceHandle;
struct IndexTtsEmotionHandle;
struct IndexTtsRequestHandle;
typedef struct IndexTtsModelHandle *indextts_model_t;
typedef struct IndexTtsVoiceHandle *indextts_voice_t;
typedef struct IndexTtsEmotionHandle *indextts_emotion_t;
typedef struct IndexTtsRequestHandle *indextts_request_t;

typedef struct {
    const char *model_dir;
    int32_t device_index; /* -1 = CPU; 0 or greater = CUDA device (CUDA build only) */
    int32_t precision;    /* 0 = float32 */
    uint64_t reserved[8];
} indextts_model_options_t;

typedef struct {
    const char *text;
    const char *language;
    uint64_t seed;
    float duration_factor;
    int32_t do_sample;
    int32_t num_beams;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t reserved[4];
} indextts_generate_options_t;

typedef enum {
    INDEXTTS_EMOTION_NONE = 0,
    INDEXTTS_EMOTION_TEXT = 1,
    INDEXTTS_EMOTION_REFERENCE = 2,
    INDEXTTS_EMOTION_VECTOR = 3
} indextts_emotion_mode_t;

typedef struct {
    int32_t mode;
    const char *text;
    indextts_emotion_t reference;
    const float *vector;
    size_t vector_length;
    float strength;
    uint64_t reserved[4];
} indextts_emotion_options_t;

typedef struct {
    indextts_generate_options_t base;
    indextts_emotion_options_t emotion;
    uint64_t reserved[4];
} indextts_generate_options_v2_t;

typedef struct {
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t sample_rate;
    uint32_t max_reference_seconds;
    uint32_t max_semantic_tokens;
    uint32_t max_concurrent_requests_per_model;
    int32_t supports_cuda;
    int32_t supports_cpu;
    int32_t supports_cancellation;
    int32_t supports_request_cancellation;
    int32_t supports_voice_cache;
    int32_t supports_emotion_text;
    int32_t supports_emotion_reference;
    int32_t supports_emotion_vector;
    int32_t supports_target_duration;
    int32_t supports_sampling;
    int32_t supports_beam_search;
    uint64_t reserved[8];
} indextts_capabilities_t;

typedef struct {
    char runtime_version[64];
    char model_version[64];
    char model_manifest_sha256[65];
    char backend[32];
    char device[32];
    uint64_t reserved[8];
} indextts_model_info_t;

typedef struct {
    char reference_sha256[65];
    float duration_seconds;
    uint32_t source_sample_rate;
    uint32_t source_channels;
    uint64_t cache_bytes;
    uint64_t reserved[8];
} indextts_voice_info_t;

typedef struct {
    float *samples;
    size_t sample_count;
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t reserved[4];
} indextts_audio_out_t;

#define INDEXTTS_OK 0
#define INDEXTTS_ERROR -1
#define INDEXTTS_PANIC -2
#define INDEXTTS_CANCELLED -3

INDEXTTS_API uint32_t indextts_abi_version(void);
INDEXTTS_API int32_t indextts_get_capabilities(indextts_capabilities_t *capabilities);
INDEXTTS_API int32_t indextts_model_get_info(indextts_model_t model, indextts_model_info_t *info);
INDEXTTS_API int32_t indextts_voice_get_info(indextts_voice_t voice, indextts_voice_info_t *info);
INDEXTTS_API void indextts_model_options_init(indextts_model_options_t *options);
INDEXTTS_API void indextts_generate_options_init(indextts_generate_options_t *options);
INDEXTTS_API void indextts_generate_options_v2_init(indextts_generate_options_v2_t *options);
INDEXTTS_API int32_t indextts_model_load(const indextts_model_options_t *options, indextts_model_t *out_model);
INDEXTTS_API int32_t indextts_voice_prepare(indextts_model_t model, const char *reference_audio_path, indextts_voice_t *out_voice);
INDEXTTS_API int32_t indextts_generate(indextts_model_t model, indextts_voice_t voice, const indextts_generate_options_t *options, indextts_audio_out_t *out_audio);
INDEXTTS_API int32_t indextts_emotion_prepare_reference(indextts_model_t model, const char *reference_audio_path, indextts_emotion_t *out_emotion);
INDEXTTS_API int32_t indextts_generate_v2(indextts_model_t model, indextts_voice_t voice, const indextts_generate_options_v2_t *options, indextts_audio_out_t *out_audio);
INDEXTTS_API void indextts_emotion_free(indextts_emotion_t emotion);
INDEXTTS_API int32_t indextts_request_create(indextts_model_t model, indextts_request_t *out_request);
INDEXTTS_API int32_t indextts_generate_request_v2(indextts_model_t model, indextts_request_t request, indextts_voice_t voice, const indextts_generate_options_v2_t *options, indextts_audio_out_t *out_audio);
INDEXTTS_API int32_t indextts_request_cancel(indextts_request_t request);
INDEXTTS_API void indextts_request_free(indextts_request_t request);
/* Thread-safe cooperative cancellation; returns immediately. */
INDEXTTS_API int32_t indextts_model_cancel(indextts_model_t model);
INDEXTTS_API void indextts_audio_free(indextts_audio_out_t *audio);
INDEXTTS_API void indextts_voice_free(indextts_voice_t voice);
INDEXTTS_API void indextts_model_free(indextts_model_t model);

/* Pointer remains valid until a later API call changes the process-wide error. */
INDEXTTS_API const char *indextts_last_error(void);
/* Static pointer valid for the lifetime of the process. */
INDEXTTS_API const char *indextts_version(void);

#ifdef __cplusplus
}
#endif
#endif
