# Comparison scope and embedded targets

Sources checked on 2026-09-07. This document defines what the experiments can
support; it is not an exhaustive claim to have reproduced every recent radar
method. The executable selection and reporting rules live in the review protocol
and experiment scripts. Dataset provenance is described in [DATASETS.md](../DATASETS.md).

## Recent work and comparable tasks

| Method or source | What it evaluates | How it relates to this repository |
|---|---|---|
| Aligned noncoherent waveform bank | Maximum Hermitian-correlation magnitude against known training examples | Direct implemented control; handles global phase, but does not search delay or Doppler. |
| Circular FFT waveform bank | Maximum correlation over every circular frame translation | Direct implemented control; includes query FFT, template multiplication, inverse FFTs and class reduction. It is not a zero-padded linear-delay or Doppler-grid detector. |
| Spectral prototypes and PCA prototypes | Classification using the same normalized log-power front end | Direct implemented controls for the contribution of learned encoding. Training-only standardization is a separate candidate. |
| [RadCharSSL, arXiv:2501.03461v3](https://arxiv.org/html/2501.03461v3) | Masked signal modeling, RF-domain pretraining and few-shot radar recognition | A relevant modern representation-learning comparator. Its RadChar-SSL pretraining, n-shot fine-tuning and RadChar-Eval data are a different protocol. Its published accuracy is not a directly comparable score for Tiny's 40,000/5,000/5,000 split. |
| [Complex-valued VAE versus ANMF-Tyler, arXiv:2511.19805v1](https://arxiv.org/html/2511.19805v1) | Radar out-of-distribution detection under clutter, using reconstruction and latent scores | Relevant to learned detection and adaptive matched filtering, but a different decision problem. Five-class waveform accuracy does not measure detection probability at a controlled false-alarm rate. |

RadCharSSL evaluates ResNet1D, MS-TCN and WaveNet encoders after masked
pretraining. Its one-shot definition includes one labeled example per class and
SNR level, and it evaluates on a separate RadChar-Eval population. Reproducing
that study requires its training budgets, pretraining data, split files and
metrics. Neither a larger Tiny training set nor a smaller model gives a valid
basis to claim superiority over that study without a matched experiment.
[Primary paper](https://arxiv.org/html/2501.03461v3).

The VAE/ANMF paper compares several scores on synthetic and experimental radar
clutter. That motivates a separate detector experiment with noise/clutter-only
training or calibration data, a defined target model, and ROC measurements.
This repository's current waveform-discrimination experiment does not reproduce
that detector protocol. It makes no ANMF, CFAR, clutter-detection or universal
matched-filter superiority claim. [Primary paper](https://arxiv.org/html/2511.19805v1).

## A strong matched-filter accuracy frontier

The following are **validation-only** results from RadChar Tiny: 40,000 training
rows, 5,000 validation rows, seed 42 and the fixed 80/10/10 protocol. Templates
are actual training examples ranked by training SNR, with deterministic random
tie breaking. Every class receives the stated number of templates; query SNR
is never an inference input. All circular translations are searched.

| Templates per class | Total templates | Validation accuracy | Estimated real FLOPs/query | NumPy resident arrays |
|---:|---:|---:|---:|---:|
| 64 | 320 | 72.92% | 8,870,400 | 3,937,280 bytes |
| 256 | 1,280 | 81.16% | 35,412,480 | 15,749,120 bytes |
| 1,024 | 5,120 | 85.32% | 141,580,800 | 62,996,480 bytes |

Evidence: [64-template preview](../experiments/baseline_val_stronger_preview.json)
and [larger-bank preview](../experiments/baseline_val_large_preview.json).
These previews establish the candidate accuracy curve, not final test scores
or isolated hardware timing. A learned model cannot claim an unconstrained
accuracy win merely by beating the 64-template bank. A resource-constrained win
must name the memory/latency budget and show the best feasible baseline under
that budget as well as the unconstrained accuracy frontier. The largest tested
bank is not a proof that no stronger matched-filter configuration exists.

FLOPs are arithmetic estimates, not processor instructions or measured cycles.
The FFT convention is approximately 5*N*log2(N) real operations per radix-2
complex transform. Report complete inference latency, throughput, memory and
accuracy separately. A high FLOP/s rate alone does not mean fewer operations
per decision or lower energy.

## Dataset and model limits

The [RadChar release](https://github.com/abcxyzi/RadChar) is synthetic radar data;
its release variants are nested and must not be treated as independent holdouts.
The current loader guarantees disjoint rows and honors explicit groups when
provided. Tiny supplies no source-realization or acquisition-session group ID,
so the result does not establish independence of unrecorded source groups.
Field-data transfer, unseen waveform families, receiver changes and clutter
robustness require separate evaluation.

The v1 spectral autoencoder reconstructs spectral features. Its front end
discards global phase and amplitude and is invariant to circular translation.
The temporal convolution candidate is different: its local-product front end
retains relative phase, while its coherent-IQ option retains normalized I/Q.
Those temporal front ends are shift-equivariant, and the subsampled pooled code
is exactly shift-invariant only for multiples of the total stride, not arbitrary
single-sample shifts. The default temporal decoder learns spectral features
during training and is omitted during inference. Neither design claims arbitrary
original-IQ reconstruction. Class discrimination does not establish pulse-
compression resolution, range/Doppler estimation, arbitrary image recognition
or IMU activity recognition. Identify the actual model and front end by the
frozen artifact, rather than applying v1 properties to every candidate.

## Candidate Cortex-M deployment hosts

These are concrete **candidate hosts for physical validation**, not measured
performance endorsements. The selected device and its memory layout matter;
Cortex-M33 DSP/FPU support is implementation-dependent. Arm describes those
options in its [Cortex-M33 support material](https://support.arm.com/compute-ip/cortex-m33).
The separate [embedded implementation guide](../embedded/README.md) documents
the allocation-free C interface, export, compilation and functional checks.

| Candidate | Documented resources | Practical implication |
|---|---|---|
| [STM32L476RG, Cortex-M4 with single-precision FPU](https://www.st.com/en/microcontrollers-microprocessors/stm32l476rg.html) | Up to 80 MHz, 1 MiB flash, 128 KiB SRAM; DSP instructions | A constrained floating-point host for testing a compact encoder, bounded frame buffers and FFT workspace. |
| [STM32H743/753 family, Cortex-M7](https://www.st.com/en/microcontrollers-microprocessors/stm32h743-753.html) | Up to 480 MHz, 1–2 MiB flash and 1 MiB SRAM, including TCM and caches | More timing and memory headroom; placement, cache state and DMA coherency still require board-specific validation. |
| [STM32H563RI, Cortex-M33](https://www.st.com/en/microcontrollers-microprocessors/stm32h563ri.html) | Up to 250 MHz, 2 MiB flash, 640 KiB SRAM, single-precision FPU and DSP | A modern host candidate; startup, security configuration and application memory reservations must be included. |

An AArch64 Linux benchmark does not validate these microcontrollers. Arm's
[Neon description](https://www.arm.com/technologies/neon) concerns Cortex-A/R
SIMD, while the Cortex-M DSP/FPU programming model is different. Likewise,
compilation or emulator execution establishes functional portability, not
physical latency, energy, interrupt tolerance or flash wait-state behavior.
**No physical Cortex-M board has been measured for this work.**

The provisional 64 KiB SRAM / 256 KiB flash application target is a design budget,
not a description of every chip in the table. The current NumPy FFT control
keeps raw templates, their conjugate transpose and precomputed spectra. With
complex64 FFTs it occupies 12,304 bytes per template, including labels and row
IDs: 21 total templates fit in 256 KiB of arrays alone. A tailored deployment
storing only complex64 spectra plus int64 labels would need 4,104 bytes per
template, allowing 63 total before code and workspace. This latter figure is an
engineering lower-storage design estimate, not the implementation tested here.
Flash storage and SRAM workspace are separate budgets; neither array count
establishes that complete firmware fits or meets a deadline.

## IMUs, sensor hubs and programmable processors

An IMU is not automatically a general-purpose processor that accepts this C
model. ST's [LSM6DSOX](https://www.st.com/en/mems-and-sensors/lsm6dsox.html)
provides dedicated embedded processing, including a finite-state machine and
machine-learning core. Its documented decision-tree workflow is not an arbitrary
FFT-plus-autoencoder firmware interface. See ST's
[machine-learning-core application note](https://www.st.com/resource/en/application_note/dm00563460-lsm6dsox-machine-learning-core-stmicroelectronics.pdf).
A separate programmable host MCU is the appropriate default execution target.

Some sensor hubs are programmable. Bosch's
[BHI260AP datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bhi260ap-ds000.pdf)
describes a six-axis IMU integrated with a programmable Fuser2 **ARC EM4** core,
20/50 MHz operating modes, an FPU and 256 KiB SRAM. It is not an Arm Cortex-M;
its vendor SDK, firmware framework and actual application-memory allowance need
a separate port. No BHI260AP build or physical measurement is provided here.

Radar IQ and accelerometer/gyroscope time series are different inputs. Using the
architecture for motion signals would require a defined sensor preprocessing
path, a motion dataset, retraining, and subject/session-separated evaluation.
This work does not claim deployment on arbitrary IMUs or reuse of radar-trained
weights for motion recognition.
