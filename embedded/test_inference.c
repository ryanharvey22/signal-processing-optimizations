/* Shared native / Cortex-M functional harness. Generated parity fixtures and
 * expected predictions are in flash; no clock or throughput is asserted here.
 */
#include "ogae_model.h"
#include "ogae_golden.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static float workspace[OGAE_EXPORTED_WORKSPACE_FLOATS];
static float scores[OGAE_EXPORTED_CLASSES];
static float invalid_iq[2u * OGAE_EXPORTED_SAMPLES];

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
    /* Integer error scale keeps the semihosting harness independent of printf's
     * optional floating-point formatter and its substantial flash footprint. */
    printf("embedded parity passed: %u cases, max error <= %lu parts per billion, workspace %lu bytes\n",
           (unsigned)OGAE_EXPORTED_GOLDEN_CASES, (unsigned long)(largest * 1e9f + 1),
           (unsigned long)sizeof(workspace));
    return 0;
}
