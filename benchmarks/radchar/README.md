# Frozen RadChar benchmark

**The accuracy-and-cost goal was not met.** The selected autoencoder achieved
82.66% accuracy on 5,000 held-out frames, compared with 84.80% for the strongest
validation-selected matched-filter control. The paired difference is **-2.14
percentage points**, with a 95% interval of **[-3.06, -1.22] percentage points**.
Its complete inference path requires **53.1 times fewer estimated ordinary
arithmetic operations**. That estimate does not establish measured FLOP/s,
latency, throughput, energy, or superiority over the latest published methods.

The frozen evidence is [spec.json](spec.json), [test_report.json](test_report.json),
[validation_candidates.json](validation_candidates.json), and
[additional_validation.json](additional_validation.json). The primary comparator,
parameters, artifact hashes and selection rule were fixed before test evaluation.
No model, reference bank, hyperparameter or epoch was tuned on these test results.

## Held-out results

| Frozen model | Validation accuracy | Test accuracy | Estimated ordinary operations/frame |
|---|---:|---:|---:|
| Selected coherent autoencoder, reconstruction weight 0.01 | 82.34% | 82.66% (4,133/5,000) | 2,666,832 |
| Same architecture, reconstruction weight 0 | 82.00% | 82.30% (4,115/5,000) | 2,666,832 |
| Circular FFT matched filter, 1,024 references/class | 85.32% | 84.80% (4,240/5,000) | 141,580,800 |
| Aligned matched filter, 1,024 references/class | 74.80% | 75.42% (3,771/5,000) | 20,976,640 |

The FFT control searches all circular translations against 5,120 actual training
waveforms. Training SNR ranks its references, with seeded tie breaking; query SNR
is never an inference input. This control measures five-class frame discrimination
with unknown circular delay. It does not implement a Doppler grid, linear-delay
search, clutter-adaptive detector or detection probability at a fixed false-alarm
rate. The largest bank in this grid is not proof that a stronger comparator cannot
exist. The declared accuracy-superiority gate is false and arithmetic-cost gate is
true; native latency has a separate gate.

The selected reconstruction objective improves the point estimate by 0.36
percentage points over its fixed-architecture zero-reconstruction control, with
a paired 95% interval of **[-0.38, +1.08] percentage points**. This interval includes
zero, so this run does not demonstrate an accuracy benefit from reconstruction.
Both intervals use 2,000 paired row-bootstrap resamples over the same 5,000 test
rows. They describe one training seed and do not account for unrecorded dependence
between rows or variability across repeated training runs.

## Dataset and selection scope

RadChar-Tiny is the **published synthetic radar dataset**, not measured radar
captures. The legacy `dataset.synthetic: false` field means that the external
HDF5 release was used instead of this repository's small generated engineering
fixture. The release file SHA-256 is
`e37643d3c691089109ba34893e5a5112c6be5bc1bd2922c81f14127456aa8119`.

The complete 50,000-row population was assigned to 40,000 training, 5,000
validation and 5,000 test rows with seed 42 before applying any caps. Frames have
512 complex samples and five class labels. Split membership, IQ and labels are
hashed in the reports. Rows are disjoint; this release supplies no independent
realization or acquisition-session group identifier. Generalization to measured
captures, new receivers, waveform families, image signals or IMU motion data has
not been established. Other nested RadChar releases were not used as independent
holdouts.

Training fits the encoder and prototype banks using training rows only. Validation
chooses epochs, architectures and bank configurations; the highest validation
accuracy wins, with lower estimated operations breaking ties. The frozen bundle
retains alternative models and the matched reconstruction ablation for reporting,
without choosing a replacement from their test scores. Some exploratory training
runs began before start snapshots were recorded, and source files changed during
some runs. The manifest explicitly records this provenance limit: end snapshots
are not exact training-source attestations. Frozen inference artifacts and golden
outputs have independent hashes.

Recent radar representation learning and adaptive matched filtering use different
datasets and decision protocols. Those published results were not reproduced as a
controlled head-to-head experiment here. See
[comparison scope](../../docs/COMPARISON_SCOPE.md) for the implemented controls and
relevant research protocols. **No state-of-the-art claim is supported.**

## Selected model and arithmetic scope

[latent.npz](latent.npz) contains the selected deployment model. RMS-normalized
I/Q enters 16 learned complex filters with 65-sample circular kernels and stride
two. Response magnitude squared removes global phase after coherent accumulation;
explicit signed power gain/bias and ReLU follow. Five real convolution stages use
channels 12/16/16/16/16, kernel five, stride two, circular padding and folded
normalization. Global mean/max pooling produces 32 features, projected into a
normalized **16-dimensional code**. Cosine matching uses **five training-derived
prototypes**, one per class. The receptive field is 313 samples; analytic circular
shift invariance holds for multiples of the total stride, 64 samples, subject to
floating-point rounding.

Training used 60 epochs, batch size 128 and seed 42. A decoder with reconstruction
weight 0.01 learns 128 normalized spectral features. Reconstruction is lossy and
does not recover original IQ phase. The decoder is omitted from inference, which
uses no FFT or logarithm. Removing phase after the first 65 samples can discard
coherent relationships across more distant pulses; the measured result does not
establish that this architecture preserves full-frame matched-filter information.

The operation estimate includes normalization, the complex first stage, real
convolutions, pooling, projection, code normalization and bank matching. Each real
multiply/add/divide counts as one ordinary operation; a complex MAC counts as
eight. The selected model additionally performs **two square roots per frame**,
reported separately. The FFT control includes the query FFT, template products,
inverse FFTs and magnitude reductions, using the estimate 5*N*log2(N) per complex
FFT. Comparisons, finiteness checks, memory traffic, indexing, compiler instructions
and special-function cost are not interchangeable with ordinary FLOPs. The
53.1 ratio is 141,580,800 / 2,666,832; it is not a hardware speedup measurement.

## Memory and embedded implementation

The selected allocation-free float32 C99 export has the following fixed payloads:

| Resource | Bytes |
|---|---:|
| Constant model numeric arrays | 34,284 |
| Exact caller-owned inference workspace | 26,816 |
| Input IQ frame | 4,096 |
| Five output scores | 20 |
| Output label | 8 |

The mutable buffers total 30,940 bytes per caller before stack and application
state. Model descriptors, ten convolution pointer-table entries, alignment, code,
startup, libm and stack are additional. Inspect the linked image and stack report
for a complete target budget; numeric arrays alone do not establish that firmware
fits. The provisional target is 64 KiB SRAM and 256 KiB flash. C direct convolutions
reuse bounded buffers and need no heap or im2col allocation. Each concurrent caller
needs its own workspace.

Python's 50,984-byte model-array count includes an expanded complex-kernel cache;
it is neither process RSS nor the C flash-array total. NumPy inference also uses
chunked temporary arrays and BLAS workspace. The evaluated FFT control retains
62,996,480 bytes of NumPy arrays, so it is an unconstrained accuracy reference and
does not fit the provisional MCU budget. A memory-constrained performance claim
would require the strongest feasible control under that same complete budget.

No intrinsics are used. The C kernels are portable; any assembly in the supporting
Cortex-M startup and cycle-counter code has explanatory comment blocks. No speedup
from a handwritten SIMD kernel is claimed. See the
[embedded guide](../../embedded/README.md) for error handling, export, numerical
parity, memory ownership and physical-board cycle measurement.

## Negative validation experiments

The retained validation results include pooled-spectrum encoding (74.10%), a
full-512-bin spectral encoder (70.50%), local-product convolution (72.90%),
four-stage normalized-IQ convolution (79.86%), six-stage normalized-IQ convolution
(81.70%), and the smaller 8-channel/33-sample coherent model (81.98%). These
alternatives did not surpass the selected wide coherent model or the 85.32%
matched-filter validation control.

Bounded post-training probes also used training-only statistics: covariance
shrinkage 0.01/0.1/0.5/1, 1/4/16 prototypes per class, and high-SNR training-reference
subsets. None improved either coherent model, so their original weights and banks
were retained. The four-stage IQ model's larger bank reached 80.24%, below the
selected model. All these choices and negative results preceded test evaluation;
the extra validation reports explicitly record that test IQ was not loaded.

## Architecture measurements

The same frozen model was measured on Windows x86-64, Linux x86-64, and Linux
AArch64. All native timings use warm caches.

| Windows x86-64, Ryzen 7 6800H | Full IQ-to-label latency | Throughput |
|---|---:|---:|
| Selected model, C99 / Clang 21.1.0 | 526.33 us, median of three trial means | 1,900 frames/s |
| Selected model, NumPy / one BLAS thread | 530.45 us, batch-1 p50 | 5,933 frames/s at batch 128 |
| FFT matched-filter control, NumPy / one BLAS thread | 27,905.65 us, batch-1 p50 | 56.36 frames/s at batch 128 |

The C timing uses 10,000 calls per trial over eight frozen cases, after 100 warmup
calls. It is not a p50 over individual calls. Python uses 100 timed single-frame
calls per trial, three alternating method orders, and separate batch-128 trials.
The Python paths give a 52.6x batch-1 median latency ratio on this host. Comparing
the C model to Python/NumPy matching would also compare implementations and
runtime overhead; it is not an isolated algorithm or ISA speedup. The C report
includes exact compiled source hashes; the local Python report discloses an
end-of-run source snapshot limitation. Raw evidence:
[Windows C](architecture_results/windows-c.json) and
[Windows Python](architecture_results/windows-python.json).

| Linux runner | Selected C, median trial mean | Selected NumPy, batch-1 p50 | FFT control NumPy, batch-1 p50 | Selected / FFT batch-128 frames/s |
|---|---:|---:|---:|---:|
| x86-64, AMD EPYC 9V74 | 726.02 us | 559.38 us | 21,556.04 us | 8,735 / 75.66 |
| AArch64, CPU part 0xd49 | 469.91 us | 509.97 us | 21,242.85 us | 7,478 / 64.01 |

Both Linux Python runs reproduced all 5,000 frozen predictions exactly for the
selected model and both waveform matched filters. The selected model's batch-1
median latency was 38.5x lower than the FFT control on x86-64 and 41.7x lower on
AArch64. These are comparisons of the tested NumPy implementations on each host,
not hardware-counter FLOP/s. Python includes FFT/BLAS and interpreter overhead.
Portable C is slower than NumPy for this model on the tested Linux x86-64 runner;
it is supplied for allocation-free deployment and does not universally win latency.
The hosts and compilers differ, so the cross-host ratio does not isolate ISA effects.

C parity checks eight frozen cases, including intermediate features, codes,
scores and labels. C timings use three trials of 1,000 full calls on Linux, after
100 warmups, with GCC 13. NumPy timings cover three method-order trials with 100
single-frame calls per method and separate batch-128 timing. The one auxiliary
near-tie described below also occurs on both Linux CPUs; all other models'
predictions are exact. The checked-in architecture reports retain trial values,
hardware/runtime details and hashes.

| Embedded target | Frozen-model check | Linked flash, text + data | Static SRAM, data + bss |
|---|---|---:|---:|
| Cortex-M4F, fpv4-sp-d16 | Cross-build and QEMU MPS2-AN386 functional pass | 140,180 B | 33,280 B |
| Cortex-M7, fpv5-sp-d16 | Cross-build and QEMU MPS2-AN500 functional pass | 140,180 B | 33,280 B |
| Cortex-M33, fpv5-sp-d16 | Cross-build only | 140,172 B | 33,280 B |

These linked sizes include the eight-case golden test harness and newlib support.
The linker enforces 256 KiB flash and 64 KiB SRAM and reserves at least 8 KiB
beyond static state for test stack/semihosting heap. Static SRAM is not a measured
stack high-water mark; production firmware needs its own complete memory budget.
The M33 build assumes the optional FPU and a supported hard-float ABI.
No physical Cortex-M timing, energy, DMA or interrupt behavior was measured.

[CI run 34178151748](https://github.com/ryanharvey22/signal-processing-optimizations/actions/runs/34178151748)
passed preparation, both native CPU jobs and the embedded job at commit
`5b6e05aa4d150e5ed6c201d6b9c3ecd1699e88b9`. The preparation job also passed
50 regression tests and nine compiled-C subtests. Architecture artifacts and
their provenance are retained in [architecture_results](architecture_results/).

Prior synthetic-fixture and model-parity checks establish implementation coverage,
not physical target accuracy or latency. Native warm-cache microbenchmarks over a
small golden set and full-holdout classification timings have different scopes;
report them separately. QEMU execution cannot establish MCU cycles, energy or a
matched-filter speedup. See the [native benchmark guide](../../docs/NATIVE_C_BENCHMARK.md).


## Numerical replay and reproduction

The original frozen labels, first-64 score blocks, latent codes, learned models,
and test report are unchanged. Reordering one float32 auxiliary spectral baseline
query creates a top-score tie: source row 37180 changes from class 3 to class 1.
On Windows this auxiliary control therefore measures 60.28% native accuracy
versus 60.30% in the frozen evaluation. Its exact-prediction flag is false.
The selected autoencoder and both waveform matched filters retain strict exact
prediction checks across all 5,000 rows.

[auxiliary_replay.json](auxiliary_replay.json) and
[auxiliary_replay.npz](auxiliary_replay.npz) are explicitly **post-freeze diagnostic
supplements**. Their original-order replay reproduced every original auxiliary
label before publication. Hashes bind the frozen spec, models and goldens;
row IDs and class-column labels are checked. Every auxiliary score must agree
with this full-score supplement within the pre-existing rtol=3e-4, atol=3e-5,
including rows with unchanged predictions. Disagreements are separately listed
with both margins and actual accuracy. The observed maximum score residual on
the affected Windows baseline is 6.56e-7. Approximate auxiliary numerical agreement
does not count as exact parity or accuracy-superiority evidence.

Rebuild the published frozen inference bundle without training:

```bash
python -m pip install -r requirements-optimized.txt
python scripts/fetch_radchar_tiny.py --output data/radchar/RadChar-Tiny.h5
python scripts/optimized_filter.py rebuild --spec benchmarks/radchar --h5 data/radchar/RadChar-Tiny.h5 --out experiments/reproduced
python scripts/optimized_filter.py native --experiment experiments/reproduced --output experiments/reproduced/native.json --repeats 100
python scripts/export_embedded.py --model experiments/reproduced/latent.npz --output build/reproduced/ogae_model.h --golden-output build/reproduced/ogae_golden.h --golden-iq experiments/reproduced/queries.npz --cases 8
```

The upstream RadChar frame is 512 samples at 3.2 MHz, or 160 us. The measured
Windows C time is longer than that interval. Full-rate continuous acquisition
with this model is not demonstrated on this host or any physical MCU.
Scheduling, DMA, interrupts, receiver preprocessing, cold flash/cache effects and
application buffers are outside these inference timings.
