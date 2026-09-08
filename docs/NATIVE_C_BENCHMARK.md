# Native C inference benchmark

[embedded/benchmark_native.c](../embedded/benchmark_native.c) measures complete
calls to the exported `ogae_predict` pipeline on a POSIX native host. Use it on
both x86-64 and AArch64 with the **same frozen model and golden frame headers**.
It uses the shared API for spectral and temporal-convolution exports. Supply an
explicit model name and frontend description from the export manifest; an
unspecified label is reported honestly rather than inferred from dimensions.

This is a warm-cache microbenchmark over a limited rotating input set. It is not
a physical Cortex-M measurement, full-dataset accuracy evaluation or speed
comparison against the Python FFT matched-filter bank. Keep those results in
their own clearly identified measurements.

## Build and run

First generate `ogae_model.h` and `ogae_golden.h` with the embedded exporter, as
shown in [the embedded guide](../embedded/README.md). Use the same generated
headers on each architecture, not a fresh training run. For the existing CI
fixture paths:

```sh
cc -std=c99 -O3 -Wall -Wextra -Werror -pedantic \
  -Iembedded -Iartifacts/generated \
  embedded/ogae.c embedded/test_inference.c -lm -o parity
./parity
cc -std=c99 -O3 -Wall -Wextra -Werror -pedantic \
  '-DOGAE_BENCH_MODEL_NAME="frozen-latent"' \
  '-DOGAE_BENCH_FRONTEND="spectral-v1"' \
  -Iembedded -Iartifacts/generated \
  embedded/ogae.c embedded/benchmark_native.c -lm -o benchmark-native
./benchmark-native 1000 > native-c-results.json
python -m json.tool native-c-results.json
```

Replace the illustrative frontend name with the actual export's frontend; a
convolutional model must not be labeled spectral-v1. The optional compile-time
`OGAE_BENCH_BUILD_ID` string can identify the commit or CI run. Record the model
artifact SHA-256, generated-header hashes, compiler command/options, CPU model,
OS, CPU affinity and frequency policy alongside the JSON. The harness records
its detected instruction-set architecture and compiler version. Do not use
`-ffast-math`: the inference API's nonfinite-input checks depend on normal
floating-point semantics.

The optional integer argument is the number of inferences per trial, default
1,000, with three trials. Invalid arguments fail. Increase it if a trial is too
short for useful timing; keep the same value and input set across comparisons.
The harness first validates the model and checks every golden label, score and
intermediate feature/latent vector. It then warms the pipeline for at least
100 calls and at least two complete passes over the golden cases. Initialization
and parity checks are outside the measured interval.

## What the JSON measures

Timing uses POSIX `CLOCK_MONOTONIC` around each batch of repeated calls. Every
call processes a raw interleaved IQ frame through its actual front end, encoder
and latent reference matching. The frame index rotates through the golden set.
Status checks, frame rotation and volatile output-consumption overhead are
included. Workspace and scores use fixed static arrays; this harness does not
allocate a heap buffer or recreate the model inside the loop.

| Field | Meaning |
|---|---|
| `trial_seconds` | Three measured batch durations |
| `median_trial_mean_us` | Median of the three batch-mean times per frame |
| `minimum_trial_mean_us`, `maximum_trial_mean_us` | Range of batch means |
| `fps_at_median_trial` | Reciprocal throughput corresponding to the median batch mean |
| `workspace_bytes`, `score_bytes` | These fixed arrays only, not whole-process RSS or firmware size |
| `golden_parity_passed` | Model outputs and intermediates matched the exported fixtures before timing |
| `physical_mcu: false` | A native host measurement; it does not measure Cortex-M silicon |
| `warm_cache`, `limited_golden_cases` | Explicit limitations of this repeated-fixture measurement |

The maximum trial mean is **not** a p95 individual-frame latency. The three
trials do not provide a statistically strong tail-latency distribution. Capture
independent full-dataset runs and a suitable latency sampling protocol if those
metrics are required. Keep background training or other CPU-intensive work out
of final benchmark runs.

## CI integration

Add the compile and execute commands to the existing native x86-64/AArch64
matrix after its C parity step. Upload `native-c-results.json` alongside the
NumPy result, environment record and artifact hashes. Run this as a separate
step so a C inference failure cannot be hidden by a successful Python run.
A nonzero harness exit status must fail the job. JSON parsing should also fail
on malformed or nonfinite output. As a small harness check, invoking
`./benchmark-native 0` must fail without emitting a success JSON record.

No baseline timing is emitted by this harness. A C-to-C speed comparison would
require an equally optimized matched-filter implementation, the same input
corpus, the same timing boundaries and a stated reference-bank configuration.
The small aligned control in the parity fixture cannot substitute for the large
FFT bank that establishes the validation accuracy frontier.
