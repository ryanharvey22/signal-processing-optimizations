#ifndef GCFCR_OGAE_H
#define GCFCR_OGAE_H

/* Allocation-free feature-autoencoder inference for IEEE-754 float targets.
 * The model and all pointed-to arrays are immutable and may live in flash.
 * Input, workspace, scores, and label storage must all be disjoint. One
 * workspace is needed per caller.
 * A model can be shared between concurrent callers with separate workspaces.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OGAE_OK = 0,
    OGAE_INVALID_ARGUMENT = -1,
    OGAE_WORKSPACE_TOO_SMALL = -2,
    OGAE_NUMERIC_ERROR = -3
} ogae_status;

typedef enum { OGAE_SPECTRAL = 0, OGAE_TEMPORAL_CONV = 1 } ogae_model_kind;
typedef enum { OGAE_LOCAL_PRODUCTS = 0, OGAE_NORMALIZED_IQ = 1 } ogae_conv_frontend;

typedef struct {
    uint16_t samples;       /* Power of two, currently exporter fixes 512. */
    uint16_t features;      /* Spectral bins, or 3*samples for temporal features. */
    uint16_t hidden;        /* Zero selects a single affine encoder. */
    uint16_t latent;
    uint16_t references;
    uint16_t classes;
    const float *weight0;   /* Output-major: (hidden ? hidden : latent) x features. */
    const float *bias0;
    const float *weight1;   /* latent x hidden; NULL when hidden == 0. */
    const float *bias1;
    const float *codes;     /* references x latent, already unit-normalized. */
    const uint16_t *reference_classes; /* Compact class index per reference. */
    const int64_t *class_labels;       /* Sorted original labels. */
    const float *twiddle_real;         /* samples / 2 FFT twiddles. */
    const float *twiddle_imag;
    ogae_model_kind kind;   /* Zero preserves existing spectral initializers. */
    ogae_conv_frontend conv_frontend;
    uint16_t conv_layers;   /* Temporal kind: three to seven k5/stride2 layers. */
    const uint16_t *conv_channels;
    const float * const *conv_weights; /* Output-channel, input-channel, tap. */
    const float * const *conv_biases;
    /* Temporal kind uses weight0/bias0 for the pooled projection, and no FFT. */
} ogae_model;

/* Zero means invalid dimensions. Bytes = returned count * sizeof(float).
 * Spectral: 2*samples + features + hidden + latent floats.
 * Temporal: features + largest odd-stage output + largest even-stage output
 *           + 2*last_conv_channels + latent. Retains frontend for debug parity.
 */
size_t ogae_workspace_floats(const ogae_model *model);

/* Optional startup validation of generated/custom model numeric arrays.
 * Array lengths follow the descriptor; caller must supply those lengths.
 * The caller/exporter supplies unit-normalized codes and correct FFT twiddles;
 * validation checks dimensions, finiteness and class mapping, not those semantics.
 */
ogae_status ogae_validate_model(const ogae_model *model);

/* Full raw IQ -> selected frontend/encoder -> latent matching.
 * Spectral uses FFT/log-power; temporal uses local IQ features/circular conv.
 * iq_interleaved: 2*samples floats in Re,Im,Re,Im order.
 * class_scores: classes floats; label receives an original class label.
 * No allocation, mutable global state, runtime sin/cos, or decoder is used.
 * On error output buffers may be partially written and must be discarded.
 */
ogae_status ogae_predict(const ogae_model *model, const float *iq_interleaved,
                         float *workspace, size_t workspace_floats,
                         float *class_scores, int64_t *label);

/* Matched-filter control with train-only unit-energy complex reference frames.
 * Samples are interleaved Re/Im; score is |Hermitian dot| / query norm, with
 * per-class maximum and stable first-label ties. This is aligned correlation,
 * not a delay/Doppler search. Unit-energy references are a caller contract.
 */
typedef struct {
    uint16_t samples;
    uint16_t references;
    uint16_t classes;
    const float *templates;
    const uint16_t *reference_classes;
    const int64_t *class_labels;
} ogae_aligned_model;

ogae_status ogae_aligned_predict(const ogae_aligned_model *model,
                                 const float *iq_interleaved,
                                 float *class_scores, int64_t *label);

/* After successful predict, these views support parity/debug checks.
 * They remain valid until the caller reuses the same workspace.
 */
const float *ogae_features(const ogae_model *model, const float *workspace);
const float *ogae_latent(const ogae_model *model, const float *workspace);

#ifdef __cplusplus
}
#endif
#endif
