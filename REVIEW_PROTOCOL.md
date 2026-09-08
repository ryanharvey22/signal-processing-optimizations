# Independent review protocol

This document defines the evidence required to claim that the latent autoencoder
improves signal discrimination. It is an acceptance protocol, not a declaration
that the implementation is perfect or that the performance objective has been
achieved. The review is current to **September 7, 2026**.

## Task and claim boundaries

The primary repository task is closed-set classification of a finite set of
signal families from fixed-length complex IQ frames. Its learned matcher encodes
a query and compares the representation with references constructed from
training data.

This is different from detecting a known waveform in additive Gaussian noise,
estimating a delay or Doppler shift, discriminating an unknown class, or detecting
a target in sea clutter. A classifier accuracy result does not establish a
detection probability at a fixed false-alarm rate. A deterministic compression
cannot create information absent from the received frame.

The initial compact candidate is a **feature autoencoder**: a normalized,
pooled power spectrum is encoded, and the training decoder reconstructs those
features. It does not reconstruct the original IQ waveform. Removing spectral
phase provides global phase and circular-shift invariance but can make
physically different waveforms indistinguishable. Pooling introduces further
information loss. These are modeling choices to evaluate, not free accuracy
improvements. Tests and documentation must preserve this distinction.

If only locally generated synthetic data are available, label every resulting
accuracy table as a synthetic experiment and describe the generator. Synthetic
smoke-test success is not a RadChar result or evidence of generalization to
measured radar signals.

## Current primary literature

| Source and date | Why it matters | Comparison boundary |
| --- | --- | --- |
| [Multi-task Learning for Radar Signal Characterisation](https://arxiv.org/abs/2306.13105), submitted June 19, 2023; revised April 30, 2024 | Introduces RadChar and the IQ Signal Transformer for classification and parameter regression. [Official RadChar repository](https://github.com/abcxyzi/RadChar) documents five signal families and 512-sample complex frames. | Reproduce the dataset, split, training budget, and classification task before comparing numerical results. |
| [Few-Shot Radar Signal Recognition through Self-Supervised Learning and Radio Frequency Domain Adaptation](https://arxiv.org/abs/2501.03461v3), July 15, 2025 | Studies masked ResNet1D, MS-TCN, and WaveNet autoencoders followed by classification. [Official RadCharSSL release](https://github.com/abcxyzi/RadCharSSL) provides the associated datasets. | Its pretraining corpus, few-shot label budgets, and separate RadChar-Eval set differ from a random RadChar-Tiny split. Published accuracy is not a directly comparable baseline here. |
| [Latent-space metrics for Complex-Valued VAE out-of-distribution detection under radar clutter](https://arxiv.org/abs/2511.19805), November 25, 2025 | Compares reconstruction and latent-space scores with ANMF-Tyler on synthetic and experimental radar data. It illustrates that relative performance depends on the data and scoring rule. | Binary clutter/OOD detection and its fixed false-alarm operating points differ from closed-set waveform classification. |
| [Learning to deform the matched filter](https://arxiv.org/abs/2608.31149), August 31, 2026 | Studies bounded learned modifications to an explicit matched filter in an optical wireless hardware experiment. | Receiver adaptation and error-vector magnitude are different objectives and data from RadChar classification. This is recent related work, not a reproduced baseline. |
| [Explainable deformable matched filtering reveals measurable departures from classical receiver theory in optical wireless communications](https://arxiv.org/abs/2608.30826), August 31, 2026 | Uses low-dimensional learned filter deformations to characterize practical receiver mismatch. | Its optical channel conditions and receiver metrics do not establish a classification comparison in this repository. |

The search identifies relevant recent work, not an exhaustive proof of the
latest state of the art. A claim of outperforming a named published technique
requires a runnable reproduction or an equivalent, explicitly justified
evaluation. Beating the baselines implemented here alone does not establish
state-of-the-art performance.

## Data isolation and reproducibility

1. Create stable, disjoint train, validation, and test IDs from the complete
   sample population before applying any sample cap. A cap must select within
   its assigned split; it must not change split membership.
2. Store the dataset identity and hash, split seed and fractions, selected IDs,
   class map, model configuration, source commit, and dependency versions.
   Distinguish the data-split seed from the model-training seed.
3. Keep all views or augmentations of one original waveform in the same split.
   If a dataset exposes a shared generation, capture, or transmitter group,
   split and resample by that group. A random row split alone cannot establish
   generalization to unseen captures or generators.
4. Fit feature statistics, PCA, reference selection, centroids, prototypes, and
   encoder parameters on training data only. Validation selects hyperparameters,
   template count, stopping epoch, and operating thresholds.
5. Freeze candidate and baseline configurations before the final test
   evaluation. Do not repeatedly tune against a reported test set. If development
   resumes after inspecting test results, disclose the reuse and obtain a fresh
   holdout before making a confirmatory claim.
6. Recompute reference codes using the selected final encoder. Bind the bank
   to the encoder, preprocessing configuration, dimensions, class map, and data
   manifest. Reject incompatible artifacts instead of silently continuing.
7. Use the same query IDs and the same permitted training information for
   every method. Record absent classes or empty subsets as evaluation errors,
   not zero-cost or zero-denominator successes.

## Required comparison matrix

| Method | Required purpose |
| --- | --- |
| Class-mean waveform matcher | Retain the original inexpensive reference, but identify cancellation caused by unaligned phase or delay. It is not sufficient as the only competitor. |
| Multi-template aligned waveform matcher | Score actual training exemplars or valid train-derived medoids with magnitude of complex correlation. Select reference count using validation. |
| Multi-template FFT correlation matcher | Search delays when the task includes them. Specify circular correlation versus zero-padded linear correlation and the supported lag range. |
| Feature-only prototype or nearest-reference matcher | Tests whether the invariant preprocessing, rather than the autoencoder, explains the result. |
| PCA/linear compression with matching | Tests whether a nonlinear encoder is needed for the observed compression advantage. |
| Direct classifier control | A direct classifier using comparable input features or a compact IQ network tests the cost and value of the matching stage. Describe its actual capacity and training budget. |
| Feature autoencoder with latent references | Candidate under evaluation. Reconstruction and supervised losses, prototype construction, and epoch-selection criterion must be recorded. |

Select each baseline's configuration by the same declared validation rule.
Report all rows so that a faster but less accurate baseline is not concealed.
Template banks of 1, 4, and 16 references per class are an initial search grid,
not proof that the classical accuracy frontier has been exhausted.

An optional learned-template waveform matcher must decode actual waveforms
before performing waveform correlation. Decoding pooled power features does
not supply a phase-bearing IQ template and must not be described as doing so.

For noise-present/absent detection claims, add independent noise-only or
clutter-only examples, validation-calibrated thresholds, and detection
probability at fixed false-alarm rates. Closed-set accuracy cannot substitute.

## Accuracy and statistical evaluation

Report sample count, accuracy, macro F1, confusion matrix, and accuracy by class
and SNR. Include the number of examples behind every subgroup metric.

Compare methods on identical held-out queries using paired accuracy differences.
Report a 95% confidence interval, preferably with a paired bootstrap over
independent waveform groups when available. Use fixed bootstrap seeds. State
whether the interval describes a fixed trained model or also includes variation
across training seeds; these are different sources of uncertainty.

Use multiple prespecified training seeds for claims about a training procedure.
One frozen model evaluated on two architectures is still one model replicate.
If only one seed is practical, label that limitation and avoid claims of
training stability. Apply a prespecified correction or clearly identify
exploratory comparisons if many models or subgroups are tested.

Retain difficult examples and low-SNR strata. Do not remove failures or average
away a class collapse solely to improve an aggregate result. Identical
predictions produce a zero paired difference, not evidence of superiority.

## Complete operation and memory accounting

The deployed path starts with an IQ frame and ends with class scores or a
decision. Count normalization, FFT, power formation, pooling, nonlinear feature
transforms, encoder layers, latent normalization, reference matching, and
reduction. Precomputed template FFTs and train-only bank construction are
offline costs; identify them separately.

Report MACs and the convention **one MAC = two FLOPs**, plus estimates or
separate counts for divisions, square roots, logarithms, comparisons, and
integer operations where relevant. FFT estimates must state their convention;
they are not measured processor instruction counts. Do not silently mix
different operation definitions across methods.

Exclude the decoder from online counts only when deployed inference actually
excludes it. Report training time and offline bank construction separately.
Folding a train-only affine standardizer into an encoder layer is permissible
after numerical equivalence is checked; feature extraction still has a cost.

As a scale check, the original 1024-to-512-to-256-to-64 dense encoder uses
671,744 MACs, approximately 1,343,488 FLOPs before bias and matching operations.
Five aligned length-512 complex template correlations require approximately
20,480 real arithmetic operations before normalization and reduction.
Compression of references alone therefore does not prove lower total cost.
Show the effect of reference-bank size and identify the measured break-even
region.

Report encoder weights, online references, scratch memory, and peak process
memory separately. Compressed file size does not represent resident memory.
Block or stream large comparison banks instead of allocating an unbounded
query-by-reference distance matrix.

## Native architecture evaluation

Use the same frozen artifact and golden query data on native x86-64 and ARM
workers. Downloading a shared artifact is preferable to independent retraining.
Record its hash on each worker.

Before timing, compare preprocessing outputs, latent vectors, scores, and
predictions against the reference implementation with documented absolute and
relative tolerances. Report prediction disagreements and score margins;
rounding can change an almost tied decision. A different architecture passing
unrelated random tests is not cross-architecture equivalence.

Record CPU model, architecture and enabled instruction set, OS, compiler and
flags, numerical library/runtime versions, thread count, batch size, warmup,
and number of trials. If dispatch has scalar and assembly paths, exercise and
compare both. Native runners establish CPU evidence, not GPU or embedded-board
performance.

Measure batch-one p50 and p95 latency separately from batched throughput.
Benchmark the complete inference path for every method, with equivalent
thread settings and prepared in-memory inputs. Report data loading and
end-to-end file processing separately. Synchronize asynchronous devices if
they are measured. Alternate method order across repeated trials, and state
whether startup or model loading is included.

Fewer FLOPs per query, lower latency, more queries per second, and higher
achieved FLOPs per second are distinct metrics. None is a substitute for the
others.

## Required critical tests and retest prompts

| Failure to challenge | Required check or retest prompt |
| --- | --- |
| Split leakage | “Show that all three ID intersections are empty even when train and evaluation caps differ.” |
| Prototype leakage or stale weights | “Rebuild the bank from training IDs using the frozen final checkpoint; show matching hashes and repeat evaluation.” |
| Phase dependence | “Rotate IQ by a global complex phase, including a sign inversion; compare scores and labels for paths claiming phase invariance.” |
| Delay ambiguity | “Test supported shifts and document whether invariance uses circular shifts, linear shifts, or an explicit lag search.” |
| Information loss | “Inspect confused classes and examples with similar pooled spectra; show feature-only and PCA controls before crediting the autoencoder.” |
| Invalid input | “Exercise zero energy, nonfinite values, wrong dimensions, empty banks, missing classes, and incompatible artifacts; fail clearly or return a documented finite result.” |
| Hidden information leakage | “Run a random-label sentinel on independent held-out IDs; investigate performance materially above chance.” |
| Incomplete operation count | “Start from raw IQ and account for every preprocessing and inference stage, including FFT and normalization.” |
| Timing bias | “Repeat alternating trials at batch one and at a declared throughput batch size; include all method-specific preprocessing.” |
| Architecture divergence | “Run the same artifact and golden queries through reference, scalar, and available optimized paths; inspect numerical and prediction differences.” |
| Unsupported claim | “Identify the exact named competitor, task, data, and measured result behind the claim; otherwise narrow the statement.” |

## Acceptance gates

A claim that the user's joint accuracy-and-efficiency objective has been met
requires all applicable gates below:

- [ ] Reproducible train/validation/test isolation passes, with immutable
      manifests and no test-driven tuning.
- [ ] Required baselines use comparable data and validation selection.
- [ ] The final held-out paired accuracy improvement has a strictly positive
      lower confidence bound against the declared strongest comparable
      accuracy baseline.
- [ ] Total online operation count is lower under the same counting convention.
- [ ] Batch-one latency is lower in repeated native measurements on both
      x86-64 and ARM; p95 and throughput are reported without hidden tradeoffs.
- [ ] Memory measurements and numerical equivalence pass.
- [ ] Per-class and per-SNR results do not conceal material failures.
- [ ] Literature claims are confined to reproduced comparisons.
- [ ] Dataset and training-seed limitations are explicitly disclosed.

The gate status must be derived from stored results, not assumed from the
design. If any gate fails, report the measured Pareto tradeoff, the failing
gate, and the next falsifiable experiment. A useful implementation can be
delivered while a research objective remains unproven; it must not be labeled
a universal matched-filter replacement or a state-of-the-art result.
