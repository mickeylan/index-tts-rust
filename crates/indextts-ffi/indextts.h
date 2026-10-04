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
typedef struct IndexTtsModelHandle *indextts_model_t;
typedef struct IndexTtsVoiceHandle *indextts_voice_t;

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

INDEXTTS_API void indextts_model_options_init(indextts_model_options_t *options);
INDEXTTS_API void indextts_generate_options_init(indextts_generate_options_t *options);
INDEXTTS_API int32_t indextts_model_load(const indextts_model_options_t *options, indextts_model_t *out_model);
INDEXTTS_API int32_t indextts_voice_prepare(indextts_model_t model, const char *reference_audio_path, indextts_voice_t *out_voice);
INDEXTTS_API int32_t indextts_generate(indextts_model_t model, indextts_voice_t voice, const indextts_generate_options_t *options, indextts_audio_out_t *out_audio);
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
