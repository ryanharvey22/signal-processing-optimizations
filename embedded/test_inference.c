/* Shared native / Cortex-M functional harness. Generated parity fixtures and
 * expected predictions are in flash; no clock or throughput is asserted here.
 */
#include "ogae_model.h"
#include "ogae_golden.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

static float workspace[OGAE_EXPORTED_WORKSPACE_FLOATS];
static float scores[OGAE_EXPORTED_CLASSES];
static float invalid_iq[2u * OGAE_EXPORTED_SAMPLES];
static float invalid_code[OGAE_EXPORTED_LATENT];

static int compare(const float *actual, const float *expected, size_t count,
                   float *largest) {
    size_t index;
    for (index = 0; index < count; ++index) {
        float error = fabsf(actual[index] - expected[index]);
        if (error > *largest) *largest = error;
        if (!isfinite(actual[index]) || error > 0.0003f) return 0;
    }
    return 1;
}

static int test_code_contract(const ogae_model *source) {
    ogae_model model = *source;
    float bank[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const uint16_t mapping[2] = {0, 1};
    const uint16_t invalid_mapping[2] = {0, 2};
    const int64_t labels[2] = {2, 8};
    float query[2] = {0.6f, 0.8f};
    const float before[2] = {0.6f, 0.8f};
    float output[2];
    int64_t label;
    model.latent = 2;
    model.references = 2;
    model.classes = 2;
    model.codes = bank;
    model.reference_classes = mapping;
    model.class_labels = labels;
    if (ogae_match_codes(&model, query, output, &label) != OGAE_OK || label != 8 ||
        output[0] != query[0] || output[1] != query[1] || memcmp(query, before, sizeof(query)))
        return 0;
    query[0] = query[1] = 0.0f;
    if (ogae_match_codes(&model, query, output, &label) != OGAE_OK || label != 2 ||
        output[0] != 0.0f || output[1] != 0.0f) return 0;
    model.reference_classes = invalid_mapping;
    if (ogae_match_codes(&model, query, output, &label) != OGAE_INVALID_ARGUMENT) return 0;
    model.reference_classes = mapping;
    bank[0] = bank[1] = 0.707106781f;
    query[0] = query[1] = FLT_MAX;
    if (ogae_match_codes(&model, query, output, &label) != OGAE_NUMERIC_ERROR) return 0;
    return 1;
}

int main(void) {
    unsigned test;
    float largest = 0.0f;
    int64_t label;
    const ogae_model *model = &ogae_exported_model;
    if (ogae_validate_model(model) != OGAE_OK ||
        ogae_workspace_floats(model) != OGAE_EXPORTED_WORKSPACE_FLOATS) return 1;
    for (test = 0; test < OGAE_EXPORTED_GOLDEN_CASES; ++test) {
        const float *iq = ogae_exported_golden_iq + (size_t)test * 2u * model->samples;
        if (ogae_predict(model, iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                         scores, &label) != OGAE_OK ||
            label != ogae_exported_golden_labels[test] ||
            !compare(ogae_features(model, workspace), ogae_exported_golden_features + (size_t)test * model->features, model->features, &largest) ||
            !compare(ogae_latent(model, workspace), ogae_exported_golden_latent + (size_t)test * model->latent, model->latent, &largest) ||
            !compare(scores, ogae_exported_golden_scores + (size_t)test * model->classes, model->classes, &largest)) {
            printf("embedded parity failed at case %u\n", test);
            return 2;
        }
        /* The exported code is immutable flash data. Its existing normalization
         * must be reused exactly by the public code-only matching interface. */
        if (ogae_match_codes(model,
                             ogae_exported_golden_latent + (size_t)test * model->latent,
                             scores, &label) != OGAE_OK ||
            label != ogae_exported_golden_labels[test] ||
            !compare(scores, ogae_exported_golden_scores + (size_t)test * model->classes,
                     model->classes, &largest)) {
            printf("embedded latent-only parity failed at case %u\n", test);
            return 8;
        }
    }
#ifdef OGAE_EXPORTED_HAS_ALIGNED
    for (test = 0; test < OGAE_EXPORTED_GOLDEN_CASES; ++test) {
        const float *iq = ogae_exported_golden_iq + (size_t)test * 2u * model->samples;
        if (ogae_aligned_predict(&ogae_exported_aligned_model, iq, scores, &label) != OGAE_OK ||
            label != ogae_exported_golden_aligned_labels[test] ||
            !compare(scores, ogae_exported_golden_aligned_scores + (size_t)test * model->classes, model->classes, &largest)) {
            printf("embedded aligned matched-filter parity failed at case %u\n", test);
            return 7;
        }
    }
#endif
    if (ogae_predict(model, ogae_exported_golden_iq, workspace,
                     OGAE_EXPORTED_WORKSPACE_FLOATS - 1, scores, &label) != OGAE_WORKSPACE_TOO_SMALL)
        return 3;
    invalid_iq[0] = NAN;
    if (ogae_predict(model, invalid_iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                     scores, &label) != OGAE_NUMERIC_ERROR) return 4;
    invalid_iq[0] = INFINITY;
    if (ogae_predict(model, invalid_iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                     scores, &label) != OGAE_NUMERIC_ERROR) return 5;
    if (ogae_predict(NULL, invalid_iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                     scores, &label) != OGAE_INVALID_ARGUMENT) return 6;
    invalid_code[0] = NAN;
    if (ogae_match_codes(model, invalid_code, scores, &label) != OGAE_NUMERIC_ERROR) return 9;
    invalid_code[0] = INFINITY;
    if (ogae_match_codes(model, invalid_code, scores, &label) != OGAE_NUMERIC_ERROR) return 10;
    if (ogae_match_codes(NULL, invalid_code, scores, &label) != OGAE_INVALID_ARGUMENT ||
        ogae_match_codes(model, NULL, scores, &label) != OGAE_INVALID_ARGUMENT ||
        ogae_match_codes(model, invalid_code, NULL, &label) != OGAE_INVALID_ARGUMENT ||
        ogae_match_codes(model, invalid_code, scores, NULL) != OGAE_INVALID_ARGUMENT)
        return 11;
    if (!test_code_contract(model)) return 12;
    /* Integer error scale keeps the semihosting harness independent of printf's
     * optional floating-point formatter and its substantial flash footprint. */
    printf("embedded parity passed: %u cases, max error <= %lu parts per billion, workspace %lu bytes\n",
           (unsigned)OGAE_EXPORTED_GOLDEN_CASES, (unsigned long)(largest * 1e9f + 1),
           (unsigned long)sizeof(workspace));
    return 0;
}
