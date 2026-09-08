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
prototypes**, one per class. The receptive field is 313 samples; exact circular
shift invariance holds for multiples of the total stride, 64 samples.

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

Frozen-artifact measurements are being collected. This table intentionally has no
latency or throughput values until the associated run artifacts are available.

| Target | Required evidence | Measurement status |
|---|---|---|
| Native x86-64 | Same frozen weights/queries, parity, compiler/CPU settings, complete-call timing | Pending |
| Native AArch64 | Same frozen weights/queries, parity, compiler/CPU settings, complete-call timing | Pending |
| Cortex-M4F/M7 | Cross-build and supported QEMU functional execution | Pending frozen-model run report; emulator timing is not physical timing |
| Cortex-M33 | Cross-build with explicit FPU/startup assumptions | Pending frozen-model build report |
| Physical Cortex-M board | Exact chip, placement/cache settings, cycles and energy where measured | Not measured |

Prior synthetic-fixture and model-parity checks establish implementation coverage,
not physical target accuracy or latency. Native warm-cache microbenchmarks over a
small golden set and full-holdout classification timings have different scopes;
report them separately. QEMU execution cannot establish MCU cycles, energy or a
matched-filter speedup. See the [native benchmark guide](../../docs/NATIVE_C_BENCHMARK.md).
