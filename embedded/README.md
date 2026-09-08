# Embedded inference

`ogae.c` runs the selected exported autoencoder on one 512-sample complex IQ frame at a time. The same allocation-free C99 API supports spectral MLP and convolution models, using float32 with no intrinsics, heap, decoder or runtime sine/cosine. The safe NPZ metadata selects the model kind and frontend when generating the header.

| Model | Frontend and encoder | Example workspace | Numeric flash arrays |
|---|---|---:|---:|
| Spectral MLP | Pooled FFT/log-power, 128 → 48 → 16 | 4,864 B | 30,322 B |
| Four-layer CNN | Local complex products or normalized I/Q/power; 3 → 12 → 16 → 16 → 16, mean/max pool → 16 | 26,816 B | 17,530 B |
| Six-layer CNN | Same frontend; two additional 16-channel stages | 26,816 B | 27,902 B |

The examples use five latent references and five classes. The provisional resource target is 64 KiB SRAM / 256 KiB flash. All examples additionally need 4,096 bytes for input IQ, 20 bytes for five output scores and 8 bytes for the label. The convolution runtime directly evaluates kernel-five, stride-two circular convolutions, with no im2col buffer. It supports three through seven layers. Exact workspace equals retained frontend features plus the maximum odd-stage output plus maximum even-stage output plus pooled features and latent code. Keeping frontend features enables full intermediate parity checks. This formula remains correct for nonmonotone channel widths.

The header JSON reports the actual model's numeric arrays, exact workspace and pointer-table entries. Descriptor and pointer-table size depend on the target ABI. Stack, startup, libm, application data and optional raw reference/golden arrays are additional; inspect the linked ELF and compiler stack-usage report for the complete budget. Numeric array size is not firmware size. Sixteen convolution golden frames consume substantially more test flash than spectral fixtures because each feature frame contains 1,536 floats; use eight cases for a compact Cortex-M fixture if needed.

The spectral representation discards waveform phase, amplitude and circular alignment. The temporal-product frontend preserves local phase differences and is analytically invariant to global phase; the normalized-I/Q frontend preserves coherent phase and relies on training augmentation for phase robustness. Neither is analytically invariant to carrier offset. Strided convolution with global pooling is exactly shift-invariant only for multiples of 2^layers; other shifts are learned robustness. No raw-IQ reconstruction or IMU-specific accuracy is claimed.

The C API is in `ogae.h`. Keep input, workspace, scores, and label buffers disjoint. The model is immutable and can be shared across callers; each concurrent caller needs its own workspace. Run `ogae_validate_model` at initialization, then call `ogae_predict`. A caller-supplied custom descriptor must provide valid FFT twiddles and normalized reference codes; startup validation checks array finiteness, dimensions and class mapping, not those numerical semantics. Nonfinite inputs, insufficient workspace and arithmetic overflow return explicit errors.

## Match already-encoded queries

`ogae_match_codes(model, normalized_code, class_scores, &label)` performs only
the reference-bank comparison for one code from the same frozen encoder. The
query has `model->latent` float elements and must already be normalized; the
encoder's all-zero output is also accepted. The function treats the query as
read-only, does not renormalize it, allocates nothing and requires no workspace.
It returns the same per-class maximum scores and label as the matching stage of
`ogae_predict`, with stable ties choosing the first sorted class label.

Validate the exported model once at startup. Keep the code, score output, label
and model storage disjoint. Nonfinite code elements and dot-product overflow
return `OGAE_NUMERIC_ERROR`; discard outputs on failure. A code from a different
encoder or preprocessing version is incompatible even when its length matches.
The shared C parity harness checks every exported golden code against its scores
and label, plus zero/tie behavior, read-only input, invalid pointers and numerical
errors. This interface is useful when encoding happened upstream; report its
latency separately from complete raw-IQ inference.
## Export and native functional checks

Run from the repository root, substituting the trained artifact path:

```sh
python scripts/export_embedded.py --model model.npz \
  --output build/embedded/ogae_model.h \
  --golden-output build/embedded/ogae_golden.h --cases 16
cc -std=c99 -O3 -Wall -Wextra -Werror -pedantic \
  -Iembedded -Ibuild/embedded embedded/ogae.c embedded/test_inference.c \
  -lm -o build/embedded/parity
build/embedded/parity
```

The default golden frames include zero, impulse, tone, chirp and random inputs. Supply `--golden-iq queries.npz` with a complex `iq` array for actual held-out frames; this is inference parity, not retraining. The harness checks intermediate features, latent vectors, class scores (absolute tolerance 3e-4), exact labels, and error paths. `tests/test_embedded.py` also tests linear/hidden spectral encoders, both convolution frontends, deep stacks and nonmonotone channel widths; native compilation is skipped only on hosts without a C compiler.

For an aligned complex matched-filter control, add `--aligned-bank train_refs.npz`, containing complex `iq` and integer `labels` arrays built exclusively from training references. The exporter normalizes the bank, emits a C control descriptor and independent NumPy golden scores, and the harness checks both pipelines. The control computes the maximum normalized Hermitian-correlation magnitude per class with no delay/Doppler search. It uses no workspace beyond output scores. Each stored 512-sample raw reference adds 4,096 flash bytes; limit fixture banks deliberately. A small control bank establishes implementation parity, not superiority over a large bank or modern radar detector.

## Cortex-M compilation and QEMU functional check

Required Ubuntu packages are `gcc-arm-none-eabi`, `libnewlib-arm-none-eabi`, and `qemu-system-arm`. Compile the M4F fixture:

```sh
arm-none-eabi-gcc -std=c99 -O3 -mcpu=cortex-m4 -mthumb \
  -mfpu=fpv4-sp-d16 -mfloat-abi=hard -fdata-sections -ffunction-sections \
  -fstack-usage -Iembedded -Ibuild/embedded \
  embedded/ogae.c embedded/test_inference.c embedded/qemu/startup.c \
  -nostartfiles --specs=rdimon.specs -Tembedded/qemu/linker.ld \
  -Wl,--gc-sections,-Map=build/embedded/cortex-m4.map -lm \
  -o build/embedded/cortex-m4.elf
arm-none-eabi-size build/embedded/cortex-m4.elf
timeout 30 qemu-system-arm -M mps2-an386 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel build/embedded/cortex-m4.elf
```

Compile-check Cortex-M7 with `-mcpu=cortex-m7 -mfpu=fpv5-sp-d16` and Cortex-M33 with `-mcpu=cortex-m33 -mfpu=fpv5-sp-d16`, retaining Thumb and hard-float flags. The supplied startup/linker script is a semihosted MPS2 functional fixture and enforces the provisional memory budget; it is not a production firmware image or a Cortex-M33 TrustZone startup. Each physical board requires its vendor's startup, linker map, FPU and memory configuration.

QEMU documents `mps2-an386` as a Cortex-M4 board model in its [MPS2 board documentation](https://www.qemu.org/docs/master/system/arm/mps2.html). Emulator success establishes functional execution only. It does not establish physical Cortex-M latency, cycle count, energy use, flash wait-state behavior or a matched-filter speedup.

## Measure a physical device

`cortex_m_cycles.h` provides optional DWT cycle-counter access for boards that document accessible cycle counting. Locked or security-restricted DWT registers can fault; check the actual chip documentation before use. The empty assembly barriers are documented and emit no SIMD/intrinsics. Compare complete calls on identical frames:

```c
uint32_t before = ogae_dwt_cycles();
ogae_status status = ogae_predict(&ogae_exported_model, iq, workspace,
                                 OGAE_EXPORTED_WORKSPACE_FLOATS, scores, &label);
uint32_t elapsed = ogae_dwt_cycles() - before;
/* Consume status, scores, and label; repeat for ogae_aligned_predict. */
```

Unsigned subtraction handles one counter wrap; keep each interval shorter than 2^32 cycles. Report CPU clock, exact chip, compiler/options, flash/RAM placement, cache/wait-state settings, interrupt policy, bank size, whole-call median/p95 cycles and accuracy for the same queries. Keep calls observable and do not infer cycles from FLOP estimates. The JSON separates ordinary arithmetic estimates from logarithms/square roots (convolution inference uses no FFT or logarithm); actual instruction count depends on compiler and libm. Cortex-M0/M0+ and implementations without an FPU need separate software-floating-point cost evaluation. Do not enable `-ffast-math`, which invalidates nonfinite-input checks.

## Coherent first-stage filters

The same API also exports CoherentAutoencoder. A learned complex filter sums
normalized I/Q before computing response power. Its signed power normalization
gain and bias remain explicit; the real convolution stages use folded weights.
Both real and imaginary filter coefficients are immutable in Python so cached
inference, serialization, and C export agree.

For the default eight complex channels, 33-sample kernel and five real stages,
the exact workspace is 18,624 bytes and numeric flash tables total 26,092 bytes
with five 16-dimensional prototypes. The wider 16-channel, 65-sample variant
has a larger operation count; use the generated JSON for its exact memory.
Neither path uses heap allocation, intrinsics, or a deployment decoder.

Local native C parity covers nine configurations, including signed power gains
and different kernel lengths. The validation workflow additionally compiles
Cortex-M4F/M7/M33 and runs supported M4F/M7 QEMU fixtures. Physical MCU latency
and energy must be measured separately using the board cycle-counter harness.
