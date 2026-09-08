/* POSIX native-host timing for the exact exported C inference pipeline.
 * This is a warm-cache benchmark over a small rotating golden-frame set, not
 * physical MCU timing, a full-dataset accuracy measurement or a baseline race.
 * Workspace and outputs are fixed static arrays; the harness allocates no heap.
 */
#define _POSIX_C_SOURCE 200809L
#include "ogae_model.h"
#include "ogae_golden.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef OGAE_BENCH_MODEL_NAME
#define OGAE_BENCH_MODEL_NAME "unspecified-export"
#endif
#ifndef OGAE_BENCH_FRONTEND
#define OGAE_BENCH_FRONTEND "unspecified-see-export-manifest"
#endif
#ifndef OGAE_BENCH_BUILD_ID
#define OGAE_BENCH_BUILD_ID "unspecified"
#endif
#if OGAE_EXPORTED_GOLDEN_CASES < 1
#error "A nonempty golden frame set is required"
#endif

#if defined(__aarch64__)
#define OGAE_BENCH_ARCH "aarch64"
#elif defined(__x86_64__)
#define OGAE_BENCH_ARCH "x86_64"
#else
#define OGAE_BENCH_ARCH "other-native-host"
#endif
#if defined(__clang__)
#define OGAE_BENCH_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#define OGAE_BENCH_COMPILER "gcc " __VERSION__
#else
#define OGAE_BENCH_COMPILER "other"
#endif

static float workspace[OGAE_EXPORTED_WORKSPACE_FLOATS];
static float scores[OGAE_EXPORTED_CLASSES];
static volatile uint64_t label_sink;
static volatile float score_sink;

static void json_string(const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    putchar('"');
    while (*cursor) {
        unsigned char character = *cursor++;
        if (character == '"' || character == '\\') {
            putchar('\\');
            putchar((int)character);
        } else if (character < 0x20u) {
            printf("\\u%04x", (unsigned)character);
        } else {
            putchar((int)character);
        }
    }
    putchar('"');
}

static int compare_vector(const float *actual, const float *expected, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!isfinite(actual[index]) || !isfinite(expected[index]) ||
            fabsf(actual[index] - expected[index]) > 0.0003f)
            return 0;
    }
    return 1;
}

static int verify_cases(void) {
    size_t frame, column;
    int64_t label;
    const ogae_model *model = &ogae_exported_model;
#ifdef OGAE_EXPORTED_HAS_ALIGNED
    /* Optional matched-filter fixtures belong to the separate parity harness;
     * referencing them avoids unused-constant warnings without timing them. */
    (void)&ogae_exported_aligned_model;
    (void)ogae_exported_golden_aligned_scores;
    (void)ogae_exported_golden_aligned_labels;
#endif
    if (ogae_validate_model(model) != OGAE_OK ||
        ogae_workspace_floats(model) != OGAE_EXPORTED_WORKSPACE_FLOATS) {
        fputs("invalid exported model/workspace\n", stderr);
        return 0;
    }
    for (frame = 0; frame < OGAE_EXPORTED_GOLDEN_CASES; ++frame) {
        const float *iq = ogae_exported_golden_iq + frame * 2u * model->samples;
        if (ogae_predict(model, iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                         scores, &label) != OGAE_OK ||
            label != ogae_exported_golden_labels[frame]) {
            fprintf(stderr, "golden prediction mismatch at frame %lu\n", (unsigned long)frame);
            return 0;
        }
        if (!compare_vector(ogae_features(model, workspace),
                            ogae_exported_golden_features + frame * model->features, model->features) ||
            !compare_vector(ogae_latent(model, workspace),
                            ogae_exported_golden_latent + frame * model->latent, model->latent)) {
            fprintf(stderr, "golden intermediate mismatch at frame %lu\n", (unsigned long)frame);
            return 0;
        }
        for (column = 0; column < model->classes; ++column) {
            float expected = ogae_exported_golden_scores[frame * model->classes + column];
            if (!isfinite(scores[column]) || !isfinite(expected) ||
                fabsf(scores[column] - expected) > 0.0003f) {
                fprintf(stderr, "golden score mismatch at frame %lu\n", (unsigned long)frame);
                return 0;
            }
        }
    }
    return 1;
}

static int run_frames(unsigned long iterations) {
    unsigned long iteration;
    size_t frame = 0;
    int64_t label;
    const ogae_model *model = &ogae_exported_model;
    for (iteration = 0; iteration < iterations; ++iteration) {
        const float *iq = ogae_exported_golden_iq + frame * 2u * model->samples;
        if (ogae_predict(model, iq, workspace, OGAE_EXPORTED_WORKSPACE_FLOATS,
                         scores, &label) != OGAE_OK) {
            fputs("inference failed during timing\n", stderr);
            return 0;
        }
        /* Observable outputs prevent dead-call elimination, including LTO.
         * Status checks, rotation and sink updates are included in elapsed time.
         */
        label_sink ^= (uint64_t)label + (uint64_t)iteration;
        score_sink += scores[frame % model->classes];
        if (++frame == OGAE_EXPORTED_GOLDEN_CASES) frame = 0;
    }
    return 1;
}

static int monotonic_now(struct timespec *result) {
    if (clock_gettime(CLOCK_MONOTONIC, result) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC)");
        return 0;
    }
    return 1;
}

static double elapsed_seconds(const struct timespec *begin, const struct timespec *end) {
    return (double)(end->tv_sec - begin->tv_sec) +
           (double)(end->tv_nsec - begin->tv_nsec) * 1e-9;
}

int main(int argc, char **argv) {
    unsigned long iterations = 1000;
    unsigned long warmup;
    unsigned trial, i, j;
    double seconds[3], sorted_us[3];
    struct timespec begin, end;
    const ogae_model *model = &ogae_exported_model;
    if (argc > 2) {
        fputs("usage: benchmark_native [positive-iterations-per-trial]\n", stderr);
        return 1;
    }
    if (argc == 2) {
        char *tail;
        if (argv[1][0] < '0' || argv[1][0] > '9') {
            fputs("iterations must be a positive integer <= 100000000\n", stderr);
            return 1;
        }
        errno = 0;
        iterations = strtoul(argv[1], &tail, 10);
        if (errno || *tail || iterations == 0 || iterations > 100000000ul) {
            fputs("iterations must be a positive integer <= 100000000\n", stderr);
            return 1;
        }
    }
    if (!verify_cases()) return 2;
    warmup = OGAE_EXPORTED_GOLDEN_CASES * 2ul;
    if (warmup < 100ul) warmup = 100ul;
    if (!run_frames(warmup)) return 3;
    for (trial = 0; trial < 3; ++trial) {
        if (!monotonic_now(&begin) || !run_frames(iterations) || !monotonic_now(&end)) return 3;
        seconds[trial] = elapsed_seconds(&begin, &end);
        if (!(seconds[trial] > 0.0) || !isfinite(seconds[trial])) {
            fputs("nonpositive or nonfinite measurement duration\n", stderr);
            return 3;
        }
        sorted_us[trial] = seconds[trial] * 1e6 / (double)iterations;
    }
    /* Median of three trial means, not an individual-frame latency percentile. */
    for (i = 1; i < 3; ++i) {
        double value = sorted_us[i];
        j = i;
        while (j && sorted_us[j - 1] > value) {
            sorted_us[j] = sorted_us[j - 1];
            --j;
        }
        sorted_us[j] = value;
    }
    fputs("{\"benchmark\":\"native_c_full_inference\",\"model\":", stdout);
    json_string(OGAE_BENCH_MODEL_NAME);
    fputs(",\"frontend\":", stdout);
    json_string(OGAE_BENCH_FRONTEND);
    fputs(",\"build_id\":", stdout);
    json_string(OGAE_BENCH_BUILD_ID);
    fputs(",\"architecture\":", stdout);
    json_string(OGAE_BENCH_ARCH);
    fputs(",\"compiler\":", stdout);
    json_string(OGAE_BENCH_COMPILER);
    printf(",\"physical_mcu\":false,\"warm_cache\":true,\"limited_golden_cases\":true,"
           "\"golden_cases\":%u,\"golden_parity_passed\":true,\"samples\":%u,"
           "\"features\":%u,\"latent\":%u,\"references\":%u,\"classes\":%u,"
           "\"workspace_bytes\":%lu,\"score_bytes\":%lu,"
           "\"warmup_iterations\":%lu,\"iterations_per_trial\":%lu,\"trials\":3,"
           "\"trial_seconds\":[%.9f,%.9f,%.9f],"
           "\"median_trial_mean_us\":%.6f,\"minimum_trial_mean_us\":%.6f,"
           "\"maximum_trial_mean_us\":%.6f,\"fps_at_median_trial\":%.6f,"
           "\"label_sink\":%" PRIu64 ",\"score_sink\":%.9g}\n",
           (unsigned)OGAE_EXPORTED_GOLDEN_CASES, (unsigned)model->samples,
           (unsigned)model->features, (unsigned)model->latent,
           (unsigned)model->references, (unsigned)model->classes,
           (unsigned long)sizeof(workspace), (unsigned long)sizeof(scores),
           warmup, iterations, seconds[0], seconds[1], seconds[2],
           sorted_us[1], sorted_us[0], sorted_us[2], 1e6 / sorted_us[1],
           label_sink, (double)score_sink);
    return 0;
}
