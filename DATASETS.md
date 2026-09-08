# Datasets and evaluation splits

RadChar is the primary radar benchmark. MNIST remains an image pipeline sanity
check. The optimized experiment also supplies a small, explicitly synthetic radar
fixture for tests and architecture checks when release data is unavailable.

## RadChar release

Sources: [upstream RadChar repository](https://github.com/abcxyzi/RadChar),
[dataset download](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023), and
[the original paper](https://arxiv.org/abs/2306.13105).

The release contains five pulsed radar families at SNRs from -20 through 20 dB,
with 512 complex IQ samples per frame at 3.2 MHz. Tiny contains 50,000 frames;
Small, Baseline and Large contain larger nested populations. **Tiny is a subset
of Small, which is a subset of Baseline, which is a subset of Large.** Different
variants are not independent training and test datasets. Use one release file
and create disjoint splits within it. The primary-source schema has an `iq`
dataset and a structured `labels` dataset.

| Field | Meaning |
|---|---|
| `index` | Identifier supplied by the release |
| `signal_type` | Integer class label |
| `number_of_pulses` | Pulse count |
| `pulse_width` | Pulse width in seconds |
| `time_delay` | Pulse delay in seconds |
| `pulse_repetition_interval` | PRI in seconds |
| `signal_to_noise_ratio` | SNR in dB |

Class IDs are 0 coherent pulse train, 1 Barker, 2 polyphase Barker, 3 Frank, and
4 linear frequency modulation. Class labels, SNR and pulse parameters are
metadata for evaluation; they are never supplied as inference features.

Place a downloaded file at `data/radchar/RadChar-Tiny.h5`, set `RADCHAR_H5` for
legacy dataset factory calls, or pass an explicit HDF5 path to the optimized
experiment. Data loading does not download anything. Release files and generated
waveform arrays are excluded from Git. Retain the upstream citation and consult
the source's current usage terms before redistributing dataset files.

## Fixed train, validation and test assignments

The default split is now **80% training, 10% validation, 10% test**, generated
with `numpy.random.default_rng(seed)` and default seed 42. Validation and test
have distinct memberships. This changes the old protocol, which used the same
holdout for both names; old checkpoints and reported scores are not directly
comparable with a newly split experiment.

Assignment happens on the **full population before any sample cap**. Selected
indices are sorted for reading. A cap takes a seeded random subset from its
already assigned split, across the file rather than from a class-ordered prefix.
Caps at different sizes are nested for the same seed. Changing a training or
validation cap therefore cannot move a test row into the training set.

Known integer group IDs in a `group_id`/`group` label field or a top-level
`group_ids` dataset are assigned as indivisible units. In this case, 80/10/10
refers to groups, and row counts may differ when groups have different sizes.
A cap can omit some examples from a selected group but cannot put them in a
different split. The standard Tiny release has no explicit independent-waveform
or acquisition-session group ID, so its manifest claims **row separation only**.
It does not establish independence of latent source realizations that are not
identified by the release. Do not manufacture group IDs from class or SNR.

Training alone fits encoders, spectral/PCA transforms, template banks and
prototypes. Validation selects hyperparameters and checkpoints. Evaluate the
held-out test set after those choices are frozen. Any augmented replicas of a
clean waveform must inherit the original waveform's split; split the originals
before augmentation, not the resulting noisy copies.

## Loading and provenance

`gcfcr.optimized.data.load_experiment_data` returns `ExperimentData` containing
`train`, `val`, `test` and a manifest. Each `SignalSplit` exposes:

| Attribute | Shape and meaning |
|---|---|
| `iq` | `(N, 512)` complex64 input samples |
| `y` | `(N,)` int64 class labels |
| `snr_db` | `(N,)` float32 SNR metadata |
| `row_ids` | `(N,)` physical row offsets in this file |
| `group_ids` | Known original groups, or `None` when unavailable |

The manifest records the source file's SHA-256, source population size, split
protocol version, seed, fractions, selected counts, class counts, row hashes,
waveform hashes and label hashes. File hashing reads bounded chunks. Its data
arrays contain only the requested split members; the loader never reads the
whole waveform dataset merely to discard most of it. HDF5 selections are sorted
and issued in bounded batches. Metadata and chosen waveform arrays reside in
RAM; for very large releases, set explicit caps appropriate to the machine.

```python
from gcfcr.optimized.data import load_experiment_data

data = load_experiment_data(
    "data/radchar/RadChar-Tiny.h5",
    seed=42, train_cap=40000, val_cap=5000, test_cap=5000,
)
assert not set(data.train.row_ids) & set(data.test.row_ids)
```

The legacy PyTorch `RadCharDataset` and `build_dataset("radchar", ...)` share the
same split helper. `radchar_train_fraction` defaults to 0.8;
`radchar_val_fraction` defaults to 0.1. Fractions must be positive and sum to less
than one so a test split remains. `radchar_max_samples` now caps the selected
split after assignment. `split="all"` remains available for descriptive work;
it is not an independent evaluation holdout.

## Synthetic engineering fixture

Calling `load_experiment_data` without an HDF5 path generates
`synthetic-radar-fixture-v1`. This is original test data, **not RadChar**, a
reimplementation of its generator, or evidence of state-of-the-art performance.
The five families are unmodulated pulse trains, binary Barker-coded pulses,
a generic quadriphase code, Frank-coded pulses, and chirps. Each example is a
fresh random realization. Phase, circular translation, carrier offset,
amplitude, pulse count and timing parameters are drawn independently of class;
additive complex Gaussian noise uses SNRs -12, -6, 0, 6, 12 and 18 dB.

The fixture assigns one independent realization to each row/group. It contains
no replicated clean waveform with separately generated noise across the splits.
The manifest explicitly identifies the source as synthetic and records the seed
and waveform hash. Its simplified signal family definitions and circular-delay
assumption deliberately limit the interpretation of accuracy results. Use it for
mathematical checks, reproducible smoke training and deployment parity checks;
use the actual release for the radar comparison.

## MNIST image sanity check

MNIST is loaded through torchvision on demand. Merely importing the radar
pipeline does not import torchvision. The official training population has
60,000 images and the official test population has 10,000 images. Their digit
counts are **not exactly equal**. This repository uses a fixed seed-42 random
90/10 split of the official training population for training and validation;
test uses the official held-out population. Per-digit counts should be measured
from the actual data, not assumed to be 6,000 or 1,000.

Each image is float32, shape `(1, 28, 28)`, with values in `[0, 1]`. The legacy
MLP autoencoder flattens this to 784 elements. MNIST results do not validate the
radar front end, nuisance handling, or matched-filter discrimination claims.

## Optimized manifest terminology

[The upstream RadChar release](https://github.com/abcxyzi/RadChar) consists of
synthetic radar signals. RadChar results are not measurements on captured radar
signals or evidence of deployed sensor performance. In the initial optimized
manifest schema, the legacy synthetic Boolean distinguishes the local
engineering fixture from externally loaded HDF5 data; false for RadChar does
not mean physical measurements. The report states the dataset interpretation
explicitly. The release uses 512 samples at 3.2 MHz (160 microseconds per frame);
continuous operation at that rate requires separate end-to-end timing evidence.
