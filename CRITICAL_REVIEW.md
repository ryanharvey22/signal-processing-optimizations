# Critical implementation review

Reviewed on September 7, 2026 against the working implementation on
codex/latent-autoencoder-matched-filter. This is an independent evidence log.
The research acceptance gates in [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md)
remain separate from code correctness.

## Reproduced findings

| ID | Severity | Evidence | Requested correction | Status |
| --- | --- | --- | --- | --- |
| CR-01 | Medium | Constructing FeatureClassifier with labels [0.9, 1.9] silently produced [0, 1]. | Validate label dtype, sign, uniqueness, and int64 range before conversion. | Fixed; independent direct reproduction now rejects the invalid input. |
| CR-02 | Medium | Finite classifier weights and inputs of order 1e38 produced infinite scores instead of an error. A decision from those scores is arbitrary. | Reject nonfinite affine outputs, consistently with LatentAutoencoder. | Fixed; independent direct reproduction now rejects the invalid input. |
| CR-03 | Medium | FeatureClassifier.operation_counts(96) raised a TypeError because an unavailable FFT estimate was added to integers. | Preserve an explicit unavailable estimate or raise a documented unsupported-estimate error. | Fixed; independent retest returns an explicit unavailable estimate. |
| CR-04 | High for statistical claims | A paired bootstrap with all rows in one group returned [1, 1] as its confidence interval and reported one independent unit. Such a degenerate empirical bootstrap does not establish population superiority. | Require at least two independent units and disclose the limitations of small group counts. | Fixed; independent direct reproduction now rejects the invalid input. |
| CR-05 | Medium | Raw-bank normalization could overflow a finite complex64 training waveform and silently produce zero references. | Use stable scaling or reject amplitudes outside a documented finite arithmetic domain, in both fitting and prediction. | Fixed by an explicit safe-domain check; reviewer independently confirmed both aligned and FFT matchers reject 1e20 inputs and accept empty query batches. |
| CR-06 | Medium | The initial native benchmark measured all methods in a single fixed sequence. | Alternate or randomize method order across repeated trials; report latency and throughput separately. | Fixed; independently exercised with disposable synthetic review data. |
| CR-07 | High for reproducibility claims | The initial freeze/rebuild flow recreated golden predictions using the current preprocessing code. A shared semantic change could pass architecture parity while differing from the original test evaluation. | Bind source/frontend identity and retain frozen prediction or numerical fingerprints for comparison with rebuilt artifacts. | Fixed; independently exercised with disposable synthetic review data. |

These cases were found using small direct reproductions rather than inferred
from hypothetical inputs. Numerical rejection is an acceptable documented
contract; a silent zero bank, infinite score, or changed label is not.

## Scientific interpretation still requiring evidence

- The feature autoencoder reconstructs pooled log-power features. Global phase
  and circular-shift invariance are intentional, but spectral-phase information
  is lost. It cannot support a promise of original-IQ reconstruction or arbitrary
  phase-sensitive discrimination.
- The initial PCA control centers raw features, whereas the autoencoder and
  direct classifier use train-fitted per-feature scaling. A difference between
  them does not isolate nonlinearity alone. Add a standardized control or
  explicitly limit that interpretation.
- The primary matched-filter comparator is selected from the implemented
  aligned and circular-delay banks. A gate against that comparator is a
  matched-filter-bank result, not proof of superiority to every control or to
  current published techniques. Show direct-classifier and PCA results too.
- Native x86-64/ARM functional parity and Cortex-M cross-compilation establish
  specific compatibility evidence. They do not establish physical Cortex-M4F,
  Cortex-M7, or Cortex-M33 cycle counts, latency, power, or flash wait-state
  behavior. Those require measurements on named boards or must remain unverified.
- A Cortex-M speed claim needs a comparable baseline on that target and the
  complete IQ-to-decision path, including FFT, logarithms, normalization, and
  memory access. Encoder-only cycle counts are insufficient.
- The reported arithmetic estimate is an algorithmic convention, not a count
  of processor instructions or a measurement of achieved FLOPs per second.

## Verification performed

The first full local pytest invocation printed all completion dots but did not
exit. A subsequent run exposed an existing Windows temporary-directory
permission problem in pytest-of-dalto. The reviewer terminated only that test
session and reran using a new isolated temporary directory, with pytest's cache
provider disabled.

The resulting run completed with exit status zero:
**24 tests passed and one was skipped in 5.01 seconds** after the fixes. The skipped test requires a local C compiler; native compiled parity remains a CI responsibility. The command used the repository's
.venv/Scripts/python.exe, pytest tests, a fresh --basetemp, and
-p no:cacheprovider.

Additional direct scripts reproduced CR-01 through CR-04 and independently retested their fixes, including fractional, negative, and overflowing labels. The reviewer also independently checked the CR-05 correction. Focused regression tests now cover these corrected paths.

The C implementation has been read for FFT ordering, feature formulas,
normalization, workspace layout, class reduction, and error handling. Its
radix-2 butterfly ordering and inference sequence are consistent with the
Python candidate by inspection. This statement is not a compiled numerical
parity result or a microcontroller timing result.

## Review disposition

All seven reproduced findings above are corrected and independently retested. This closes those specific findings, not the research acceptance gates. The revised phase-local convolutional candidate still requires implementation review, frozen test evaluation, and architecture evidence.

## Follow-up verification and convolutional design review

The reviewer independently exercised three alternating native benchmark trials
on disposable synthetic data. The recorded second method order reversed the
first and each method produced three trials. A freeze/rebuild round trip
preserved predictions. Deliberately changing the candidate prediction function
then caused rebuild to fail before publishing its inference bundle. No reserved
real test queries were evaluated in this review exercise.

The next candidate uses power and adjacent complex products to retain local
phase information. The review required an explicit circular boundary rule,
global-phase invariance without claiming carrier-offset invariance, and a full
operation count of roughly 250,880 convolution MACs for three kernel-5,
stride-2 stages at length 512. MACs are not FLOPs.

A concrete export trap was raised: folding input mean subtraction into a
zero-padded convolution changes boundary behavior. The design instead uses no
input train-mean subtraction and folds a following BatchNorm into the preceding
convolution using evaluation running statistics, which is boundary-safe. Numeric
Torch-evaluation versus exported-runtime parity must still test boundaries.

Three such stages have a 29-sample receptive field; RadChar pulses can be longer.
A fourth same-width stage would reach 61 samples at an additional 40,960 MACs.
This is a validation candidate if confusion patterns justify it, not a claim that
more layers necessarily improve accuracy. Circular striding also does not
provide exact invariance to every input shift.
## Second convolutional audit

The revised convolutional implementation was reviewed before the reserved real
test set was evaluated. The following new reproductions and modeling constraints
were sent to the implementation team:

| ID | Finding | Required correction | Status |
| --- | --- | --- | --- |
| CR-08 | A negative infinite intermediate affine/convolution result was clamped to zero by ReLU, allowing a later finite latent code. The C implementation rejects before activation, so Python and C disagreed. | Check each preactivation for finite values before ReLU, in all encoder/classifier paths. | Fixed; negative-overflow reproductions now reject in convolutional, dense, and direct-classifier paths. |
| CR-09 | Torch normalization maps [1e-14, 0] to approximately [0.01, 0], while deployed NumPy normalization maps it to zero. Renormalizing the Torch reference then creates a unit prototype absent from deployed training codes. | Apply the same hard-zero normalization rule when constructing training-reference codes and during deployment. | Fixed: unnormalized Torch projections now pass through the same NumPy normalizer as deployed codes. |
| CR-10 | After the supported public mutation chunk_size = -1, encode loops execute zero iterations and return uninitialized np.empty latent outputs. | Validate chunk_size at use or through a validated setter; reject invalid mutations. | Fixed; independent zero/negative mutation checks now reject before producing output. |

The initial phase-local reconstruction loss also asked a shift-invariant pooled
code to reconstruct position-dependent sequences. Identically encoded shifted
inputs cannot both be reconstructed at their original absolute positions.
Reconstructing a shift-invariant pooled spectrum instead resolves that specific
symmetry conflict, while remaining a lossy cross-feature reconstruction task.
An otherwise identical zero-reconstruction ablation is needed before crediting
the reconstruction objective with a classification benefit. A decoder that does
not train must not supply the epoch-selection tie breaker for that control.

The same-backbone direct-classifier control is scientifically relevant after
changing from a spectral MLP to a temporal convolutional encoder. A comparison
with only the old spectral classifier would confound the frontend, backbone,
and training objective.

If raw IQ plus power becomes a separate frontend candidate, its semantic kind
and version must be stored in the artifact and passed through C export. Both
raw and phase-local variants have three channels; their shapes cannot identify
their meaning. Raw IQ does not inherit analytic global-phase invariance.

The updated suite completed with **31 passed, one skipped in 5.34 seconds**.
The skip still reflects the absence of a local C compiler. A separate independent
comparison constructed four Torch convolution/BatchNorm/ReLU stages with
nontrivial running statistics and a pooled projection. Folded NumPy inference
matched on zero frames, a last-sample impulse, and random frames for both saved
frontends: maximum latent absolute errors were 1.34e-7 for temporal features and
1.19e-7 for raw IQ plus power, below the 4e-6 check tolerance. This is complete
Python-network parity evidence; it is not compiled C or MCU timing evidence.

The invariant spectral reconstruction target, zero-weight control, consistent
first-best-validation-epoch selection, and explicit saved frontend selector are
now implemented. Their classification benefit remains an empirical question for
the frozen evaluation and matched controls. The original phase-local candidate
and the raw-IQ variant must retain separate invariance claims.

## Integrated CLI, API, and native benchmark audit

A disposable synthetic end-to-end exercise passed base fitting, convolutional
fitting with reconstruction and zero-weight control, evaluation, freezing, and
rebuilding. It also verified explicit convolutional model dispatch, preservation
of the raw-IQ frontend selector, ownership of replacement reference banks, and
rejection of deliberately changed convolutional inference semantics before
publishing a rebuilt bundle. These were small functional fixtures, not radar
accuracy evidence; the reviewer did not open the reserved real test queries.

The native C timing harness was inspected. It uses CLOCK_MONOTONIC, validates
golden outputs before timing, rotates inputs, consumes outputs through volatile
sinks, and reports the median of three trial means. The documentation correctly
distinguishes this statistic from individual-frame p95 latency and makes no
physical-MCU or C-versus-Python matched-filter speed claim. Compiled execution
remains a CI requirement.

Three remaining protocol/API clarifications were requested:

- Training code provenance must capture a start snapshot as well as an end
  snapshot. An end-of-training file hash may describe source edited while the
  process was running, rather than the already imported code that trained the
  model. Existing runs without a start snapshot must disclose that limitation.
  Frozen numerical references protect later inference semantics but do not
  establish historical training-source identity.
- Reapplying fit-conv to a convolutional base must not rename that previous
  model spectral_ae or overwrite the actual spectral control. Require a
  spectral base or retain prior candidates with accurate, distinct names.
- The manifest must not claim a zero-reconstruction control when that weight
  was omitted, or inherit a stale control from a different frontend/backbone.
  Require the ablation or describe only the controls actually run.

## Comparison assembly and coherent-filter review

The three earlier integration clarifications are now implemented: training
records start/end snapshots and source changes, convolution fitting requires a
spectral base, and the CLI requires both zero and positive reconstruction
weights. Historical exploratory runs without start snapshots still cannot claim
an exact training-source attestation.

The first comparison assembler could drop the spectral autoencoder when given
only derived convolution experiments, and could overwrite a source's explicit
test-used flag with false. Both were corrected during this audit. Independent
disposable-artifact checks verified that a derived-only assembly retains the
spectral control and rejects a source marked test-used. Assembly now records its
own source hash and the source manifest/file for each selected model.

Two failure-recovery/provenance requirements were sent before any real test
selection was allowed: mark test evaluation durably before the first prediction,
so a later evaluation/export exception cannot reopen tuning, and propagate/check
inherited experiment provenance. Source artifacts must be hash-checked before
inheritance, with the base manifest hash captured rather than only its path.
These are safeguards against accidental workflow mistakes, not an assertion that
local files are tamper-proof.

The coherent first layer was reviewed as a tied complex-linear filter followed
by magnitude squared. The tied block [A,-B; B,A] and absence of an additive complex
bias preserve global-phase equivariance before power and invariance afterward.
Post-power BatchNorm folds to a separate real gain and bias; a negative learned
gain must not be folded into the complex kernel amplitude. This implementation
keeps that signed affine stage after the square, as required.

Four focused coherent-model tests passed. An additional independent fixture
compared a complete Torch evaluation network with the folded NumPy runtime,
including signed power gains, nontrivial tail BatchNorm running statistics,
zero input, first/last-sample impulses, and random frames. Maximum latent
absolute error was 1.788e-7, below the 5e-6 check tolerance. This establishes
numerical Python export parity for that fixture; it does not establish coherent
C export, physical MCU timing, or a radar accuracy advantage.

For the default eight complex length-33 filters, the first stage alone requires
67,584 complex MACs, conventionally 270,336 real MACs, before the real tail and
matching. With five stride-two real tail stages, the receptive field spans 281
samples. Circular shift invariance is exact only for multiples of 64 samples;
global-phase invariance does not imply carrier-frequency invariance.

A zero-reconstruction control supports a reconstruction-benefit claim only when
its backbone, latent width, bank size, seed, minibatches, and selection budget
match the positive-weight candidate. Selecting the best zero-weight control
across a wider latent/bank grid is a useful performance baseline but is not by
itself that matched ablation. The current single-width/single-bank validation
runs avoid this ambiguity. No reviewer test used the reserved real test split.

## Guard and cache follow-up verification

The test-start marker and inherited-source safeguards are implemented and were
independently exercised with newly generated synthetic data. Forcing bundle
creation to fail after predictions left a durable marker without a test report.
Direct and inherited tuning guards, baseline strengthening, derived-model
fitting, and comparison assembly all rejected further selection. An edited
experiment manifest could not resume evaluation; restoring the identical
manifest allowed recovery. No real test queries were used in this exercise.

Composite provenance now records resolved source paths and checks existing
source lineages. A separate fixture assembled unrelated sources A and B with B
selected and no shared-base link. Starting test evaluation only in non-winning
A correctly locked the composite. The checks prevent accidental local workflow
reuse; they do not promise protection against deliberate deletion of provenance
or sources that are no longer available on the filesystem.

A controlled two-by-two latent-width/reference-bank selection fixture selected
the positive candidate at width 4 and bank count 2, retained that exact zero-loss
ablation, and separately retained the globally best zero-loss control at width 8
and bank count 1. Inherited manifest hashes and snapshots matched their sources,
all copied model artifacts reloaded, and a deliberately corrupted inherited
artifact was rejected before training. These injected selection scores test the
workflow only and are not classification-performance evidence.

| ID | Finding | Required correction | Status |
| --- | --- | --- | --- |
| CR-11 | Coherent inference used an initialization-only expanded kernel cache, while public complex coefficient arrays remained mutable. Zeroing those arrays left current inference unchanged but changed save/reload latent outputs by up to 0.0833407. | Prevent unsupported coefficient mutation or synchronize the cache without adding avoidable per-query work. | Fixed: coefficient arrays reject in-place writes and attribute replacement; an explicit replacement model rebuilds the cache. |

Independent cache retests rejected scalar writes, fill operations, and attribute
replacement. Constructing a replacement model and replacing reference banks
still worked, preserved the original instance, and saved/reloaded consistently.
The demonstrated coherent cache divergence is closed.

## Frozen test result and auxiliary numerical disagreement

The test split has now been evaluated and frozen; no further model tuning was
performed in this review. The selected coherent encoder scored 82.66% against
84.80% for the validation-selected FFT matched filter. Their paired accuracy
difference is -2.14 percentage points, with the reported row-bootstrap 95%
interval [-3.06, -1.22] percentage points. Estimated ordinary arithmetic is
2,666,832 versus 141,580,800 operations per query. The accuracy-superiority gate
is false: the original objective of beating both accuracy and cost is not met.
The bootstrap does not model unrecorded dependence in this release.

A read-only audit of spectral_standardized found one native prediction
mismatch among 5,000 frozen queries. Query 2081 is source row 37180, true class 3,
SNR -18 dB. Its frozen prediction is 3; reordered batch-128 inference predicts 1.
The current scores of classes 1 and 3 are exactly tied at 0.6612148881. Replaying
batch-128 inference in the original source-row order reproduces all 5,000 frozen
labels, with class 3 leading class 1 on this row by 1.1920929e-7. Float64 matching
from the same float32 frontend also favors class 3, by approximately 1.93368e-7.

The first 64 frozen score vectors reproduce exactly. Across all rows, the
maximum score change between original and reordered batch inference is
6.55651e-7. Batch sizes 1, 16, and 32 reproduce all labels; 64, 128, and 256 yield
the single tie-related flip. Model and query hashes were checked before this
audit. This evidence identifies batch-dependent rounding at an almost tied
auxiliary decision; no frozen parameters or test reports were changed.

The review recommendation is to preserve exact prediction requirements for the
selected encoder and both waveform matched-filter controls. Auxiliary mismatch
diagnostics should retain exact_predictions_equal=false, list the affected
rows and score margins, and report measured native accuracy separately from
frozen accuracy. A near-tie diagnostic is not proof of exact architecture parity.
A criterion using current scores alone can hide semantic drift that collapses
scores into ties; the original artifact contains only 64 full score vectors.
Any additional score replay reference must be separately identified and must
not rewrite the original predictions, results, or their provenance.
