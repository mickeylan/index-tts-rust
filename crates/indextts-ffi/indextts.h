/* IndexTTS-2.5 C API Header
 * Auto-generated from Rust FFI bindings
 */

#ifndef INDEXTTS_H
#define INDEXTTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Handle types */
typedef void* indextts_model_t;
typedef void* indextts_voice_t;

/* Model loading options */
typedef struct {
    const char* model_dir;        /* Model directory path (UTF-8) */
    int32_t device_index;         /* CUDA device index */
    int32_t precision;            /* 0=float32, 1=bfloat16 */
    uint64_t reserved[8];         /* Reserved for future use */
} indextts_model_options_t;

/* Generation options */
typedef struct {
    const char* text;             /* Text to synthesize (UTF-8) */
    const char* language;         /* Language: ZH, EN, JA, ES, AR */
    uint64_t seed;                /* Random seed (0 = random) */
    float duration_factor;        /* Duration factor (1.0 = default) */
    int32_t do_sample;            /* 0 = greedy, 1 = sample */
    int32_t num_beams;            /* Number of beams */
    float temperature;            /* Sampling temperature */
    int32_t top_k;                /* Top-k sampling */
    float top_p;                  /* Top-p (nucleus) sampling */
    float repetition_penalty;     /* Repetition penalty */
    uint64_t reserved[4];         /* Reserved */
} indextts_generate_options_t;

/* Audio output */
typedef struct {
    float* samples;               /* Audio samples, normalized to [-1, 1] */
    size_t sample_count;          /* Number of samples */
    uint32_t sample_rate;         /* Sample rate in Hz (22050) */
    uint32_t channels;            /* Number of channels (1 = mono) */
    uint64_t reserved[4];         /* Reserved */
} indextts_audio_out_t;

/* Error codes */
#define INDEXTTS_OK              0
#define INDEXTTS_ERROR          -1

/* API Functions */

/* Load an IndexTTS model
 * Returns: INDEXTTS_OK on success, INDEXTTS_ERROR on failure
 * Use indextts_last_error() to get error message
 */
int indextts_model_load(const indextts_model_options_t* options, 
                        indextts_model_t* out_model);

/* Prepare a voice from reference audio
 * Returns: INDEXTTS_OK on success, INDEXTTS_ERROR on failure
 */
int indextts_voice_prepare(indextts_model_t model,
                           const char* reference_audio_path,
                           indextts_voice_t* out_voice);

/* Generate speech
 * Returns: INDEXTTS_OK on success, INDEXTTS_ERROR on failure
 */
int indextts_generate(indextts_model_t model,
                      indextts_voice_t voice,
                      const indextts_generate_options_t* options,
                      indextts_audio_out_t* out_audio);

/* Free audio data
 * Must be called to release audio memory
 */
void indextts_audio_free(indextts_audio_out_t* audio);

/* Free voice handle */
void indextts_voice_free(indextts_voice_t voice);

/* Free model handle */
void indextts_model_free(indextts_model_t model);

/* Get last error message
 * Returns: Error message string, or NULL if no error
 * Note: Valid until next API call
 */
const char* indextts_last_error(void);

/* Get library version
 * Returns: Version string (e.g., "0.1.0")
 */
const char* indextts_version(void);

#ifdef __cplusplus
}
#endif

#endif /* INDEXTTS_H */
