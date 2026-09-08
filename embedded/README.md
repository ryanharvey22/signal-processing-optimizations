# Embedded inference

`ogae.c` runs the same exported feature autoencoder on one complex IQ frame at a time: 512-point radix-2 FFT, pooled normalized log-power spectrum, folded affine encoder, latent normalization, and maximum cosine reference score per class. It uses C99 and float32, no intrinsics, no heap, no decoder, and no runtime sine/cosine. The feature representation intentionally discards waveform phase, amplitude, and circular time alignment; this is not raw-IQ reconstruction or an IMU-specific model.

The provisional resource target is 64 KiB SRAM / 256 KiB flash. A 128 → 48 → 16 encoder needs exactly 4,864 bytes of caller workspace, plus 4,096 bytes for the input frame, 20 bytes for five output scores, and 8 bytes for the label. Immutable weights, five reference codes, mappings, labels, and precomputed FFT twiddles occupy 30,322 bytes before descriptor alignment. The header JSON reports the actual selected model. Stack, startup, libm, application data, and optional raw reference/golden arrays are additional; use the linked ELF and compiler stack-usage report for the complete budget. Do not interpret model-array size as firmware size.

The C API is in `ogae.h`. Keep input, workspace, scores, and label buffers disjoint. The model is immutable and can be shared across callers; each concurrent caller needs its own workspace. Run `ogae_validate_model` at initialization, then call `ogae_predict`. A caller-supplied custom descriptor must provide valid FFT twiddles and normalized reference codes; startup validation checks array finiteness, dimensions and class mapping, not those numerical semantics. Nonfinite inputs, insufficient workspace and arithmetic overflow return explicit errors.

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

The default golden frames include zero, impulse, tone, chirp and random inputs. Supply `--golden-iq queries.npz` with a complex `iq` array for actual held-out frames; this is inference parity, not retraining. The harness checks intermediate features, latent vectors, class scores (absolute tolerance 3e-4), exact labels, and error paths. `tests/test_embedded.py` also tests both linear and hidden encoders; native compilation is skipped only on hosts without a C compiler.

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

Unsigned subtraction handles one counter wrap; keep each interval shorter than 2^32 cycles. Report CPU clock, exact chip, compiler/options, flash/RAM placement, cache/wait-state settings, interrupt policy, bank size, whole-call median/p95 cycles and accuracy for the same queries. Keep calls observable and do not infer cycles from FLOP estimates. The JSON separates ordinary arithmetic estimates from logarithms/square roots; actual instruction count depends on compiler and libm. Cortex-M0/M0+ and implementations without an FPU need separate software-floating-point cost evaluation. Do not enable `-ffast-math`, which invalidates nonfinite-input checks.
