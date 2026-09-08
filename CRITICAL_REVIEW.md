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
| CR-06 | Medium | The initial native benchmark measured all methods in a single fixed sequence. | Alternate or randomize method order across repeated trials; report latency and throughput separately. | Fix in progress; independent retest pending. |
| CR-07 | High for reproducibility claims | The initial freeze/rebuild flow recreated golden predictions using the current preprocessing code. A shared semantic change could pass architecture parity while differing from the original test evaluation. | Bind source/frontend identity and retain frozen prediction or numerical fingerprints for comparison with rebuilt artifacts. | Fix in progress; independent retest pending. |

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

**Changes requested.** Close each concrete finding with a focused regression
check. Then review the frozen test and architecture reports without promoting
engineering correctness into a state-of-the-art or universal matched-filter
claim. Update this log with retest evidence as findings are resolved.
