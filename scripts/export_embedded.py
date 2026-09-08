#!/usr/bin/env python3
"""Export a safe NPZ latent autoencoder to allocation-free C99 flash tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.optimized.autoencoder import LatentAutoencoder
from gcfcr.optimized.conv_autoencoder import ConvLatentAutoencoder, temporal_features
from gcfcr.optimized.features import spectral_features


def _symbol(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError("symbol must be a C identifier beginning with a letter")
    return value


def _float(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if not any(character in text for character in ".eE"):
        text += ".0"
    return text + "f"


def _array(name: str, array: np.ndarray, ctype: str = "float") -> str:
    values = np.asarray(array).reshape(-1)
    if ctype == "float":
        render = _float
    elif ctype == "int64_t":
        render = lambda value: f"INT64_C({int(value)})"
    else:
        render = lambda value: str(int(value))
    rows = ["    " + ", ".join(render(value) for value in values[start:start + 8]) for start in range(0, len(values), 8)]
    return f"static const {ctype} {name}[{len(values)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _aligned_bank(model: LatentAutoencoder, iq: np.ndarray | None, labels: np.ndarray | None):
    if iq is None and labels is None:
        return None
    if iq is None or labels is None:
        raise ValueError("aligned IQ and labels must be supplied together")
    iq = np.asarray(iq)
    labels = np.asarray(labels)
    if iq.ndim != 2 or iq.shape[1] != 512 or not len(iq) or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
        raise ValueError("aligned references must be finite complex IQ frames")
    if labels.shape != (len(iq),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("aligned labels must be a matching integer vector")
    if len(iq) > 65535 or not np.array_equal(np.unique(labels), model.classes):
        raise ValueError("aligned references must fit uint16 and cover exactly the model classes")
    energy = np.sqrt(np.sum(np.abs(iq.astype(np.complex128)) ** 2, axis=1, keepdims=True))
    if not np.isfinite(energy).all():
        raise ValueError("aligned reference energy overflowed")
    normalized = np.divide(iq, energy, out=np.zeros(iq.shape, np.complex128), where=energy > 1e-12).astype(np.complex64)
    compact = np.searchsorted(model.classes, labels).astype(np.uint16)
    return normalized, compact

def export_model(model: LatentAutoencoder | ConvLatentAutoencoder, output: Path, *, symbol: str = "ogae_exported", samples: int = 512, aligned_iq: np.ndarray | None = None, aligned_labels: np.ndarray | None = None) -> dict:
    symbol = _symbol(symbol)
    if samples != 512:
        raise ValueError("embedded exporter currently supports 512-sample IQ frames")
    convolution = isinstance(model, ConvLatentAutoencoder)
    hidden = 0 if convolution else (model.weights[0].shape[0] if len(model.weights) == 2 else 0)
    features = 3 * samples if convolution else model.input_dim
    dimensions = (samples, features, hidden, model.latent_dim, len(model.prototypes), len(model.classes))
    if any(value < 0 or value > 65535 for value in dimensions) or (not convolution and samples % features):
        raise ValueError("model dimensions do not fit the embedded descriptor")
    prefix = symbol.upper()
    reference_classes = np.searchsorted(model.classes, model.prototype_labels).astype(np.uint16)
    pointer_tables = []
    extra_fields = []
    if convolution:
        channels = [len(weight) for weight in model.conv_weights]
        if any(channel > 256 for channel in channels):
            raise ValueError("embedded convolution supports at most 256 channels per stage")
        stage_sizes = [(samples >> (index + 1)) * channel for index, channel in enumerate(channels)]
        odd, even = max(stage_sizes[::2]), max(stage_sizes[1::2])
        workspace_floats = features + odd + even + 2 * channels[-1] + model.latent_dim
        arrays = [(f"{symbol}_weight0", model.projection_weight, "float"), (f"{symbol}_bias0", model.projection_bias, "float"), (f"{symbol}_conv_channels", np.asarray(channels, np.uint16), "uint16_t")]
        for index, (weight, bias) in enumerate(zip(model.conv_weights, model.conv_biases)):
            arrays += [(f"{symbol}_conv_weight{index}", weight, "float"), (f"{symbol}_conv_bias{index}", bias, "float")]
        for name in ("weight", "bias"):
            pointers = ", ".join(f"{symbol}_conv_{name}{index}" for index in range(len(channels)))
            pointer_tables.append(f"static const float * const {symbol}_conv_{name}s[{len(channels)}] = {{ {pointers} }};")
        frontend = getattr(model, "frontend", "temporal")
        if frontend not in ("temporal", "iq"):
            raise ValueError("unsupported convolution frontend")
        enum = "OGAE_NORMALIZED_IQ" if frontend == "iq" else "OGAE_LOCAL_PRODUCTS"
        extra_fields = ["    .kind = OGAE_TEMPORAL_CONV,", f"    .conv_frontend = {enum}, .conv_layers = {len(channels)},", f"    .conv_channels = {symbol}_conv_channels,", f"    .conv_weights = {symbol}_conv_weights, .conv_biases = {symbol}_conv_biases,"]
        architecture = f"conv_{frontend}"
    else:
        workspace_floats = 2 * samples + features + hidden + model.latent_dim
        angles = -2 * np.pi * np.arange(samples // 2) / samples
        arrays = [(f"{symbol}_weight0", model.weights[0], "float"), (f"{symbol}_bias0", model.biases[0], "float")]
        if hidden:
            arrays += [(f"{symbol}_weight1", model.weights[1], "float"), (f"{symbol}_bias1", model.biases[1], "float")]
        arrays += [(f"{symbol}_twiddle_real", np.cos(angles).astype(np.float32), "float"), (f"{symbol}_twiddle_imag", np.sin(angles).astype(np.float32), "float")]
        extra_fields = ["    .kind = OGAE_SPECTRAL,", f"    .twiddle_real = {symbol}_twiddle_real, .twiddle_imag = {symbol}_twiddle_imag,"]
        architecture = "spectral_mlp"
    arrays += [(f"{symbol}_codes", model.prototypes, "float"), (f"{symbol}_reference_classes", reference_classes, "uint16_t"), (f"{symbol}_class_labels", model.classes, "int64_t")]
    aligned = _aligned_bank(model, aligned_iq, aligned_labels)
    if aligned is not None:
        references, compact = aligned
        arrays += [(f"{symbol}_aligned_iq", np.stack((references.real, references.imag), axis=-1), "float"), (f"{symbol}_aligned_classes", compact, "uint16_t")]
    lines = [f"#ifndef {prefix}_MODEL_H\n#define {prefix}_MODEL_H", '#include "ogae.h"', "/* Generated immutable folded weights and selected frontend descriptor. */", f"#define {prefix}_WORKSPACE_FLOATS {workspace_floats}u", f"#define {prefix}_SAMPLES {samples}u", f"#define {prefix}_FEATURES {features}u", f"#define {prefix}_LATENT {model.latent_dim}u", f"#define {prefix}_CLASSES {len(model.classes)}u"]
    lines.extend(_array(name, array, ctype) for name, array, ctype in arrays)
    lines.extend(pointer_tables)
    fields = [f"static const ogae_model {symbol}_model = {{", f"    .samples = {samples}, .features = {features}, .hidden = {hidden},", f"    .latent = {model.latent_dim}, .references = {len(model.prototypes)}, .classes = {len(model.classes)},", f"    .weight0 = {symbol}_weight0, .bias0 = {symbol}_bias0,", f"    .weight1 = {symbol}_weight1, .bias1 = {symbol}_bias1," if hidden else "    .weight1 = NULL, .bias1 = NULL,", f"    .codes = {symbol}_codes, .reference_classes = {symbol}_reference_classes,", f"    .class_labels = {symbol}_class_labels,", *extra_fields, "};"]
    lines.append("\n".join(fields))
    if aligned is not None:
        lines.append(f"#define {prefix}_HAS_ALIGNED 1\nstatic const ogae_aligned_model {symbol}_aligned_model = {{\n" + f"    .samples = 512, .references = {len(aligned[0])}, .classes = {len(model.classes)},\n" + f"    .templates = {symbol}_aligned_iq, .reference_classes = {symbol}_aligned_classes,\n" + f"    .class_labels = {symbol}_class_labels\n" + "};")
    lines.append("#endif\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    rom_arrays = sum(np.asarray(array).size * (8 if ctype == "int64_t" else 2 if ctype == "uint16_t" else 4) for _, array, ctype in arrays)
    return {"architecture": architecture, "samples": samples, "features": features, "hidden": hidden, "latent": model.latent_dim, "references": len(model.prototypes), "classes": len(model.classes), "workspace_floats": workspace_floats, "workspace_bytes": 4 * workspace_floats, "caller_input_bytes": 2 * samples * 4, "caller_score_bytes": len(model.classes) * 4, "model_numeric_flash_bytes": rom_arrays, "descriptor_bytes": "ABI-dependent", "extra_pointer_table_entries": 2 * len(model.conv_weights) if convolution else 0, "stack_heap": "kernel uses no heap; compiler stack usage must be measured separately", "arithmetic": model.operation_counts(samples)}

def export_golden(model: LatentAutoencoder | ConvLatentAutoencoder, iq: np.ndarray, output: Path, *, symbol: str = "ogae_exported", aligned_iq: np.ndarray | None = None, aligned_labels: np.ndarray | None = None) -> None:
    symbol = _symbol(symbol)
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.ndim != 2 or iq.shape[1] != 512 or not len(iq):
        raise ValueError("golden IQ must be a nonempty complex matrix with 512 samples")
    if isinstance(model, ConvLatentAutoencoder):
        features = model.extract_features(iq) if hasattr(model, "extract_features") else temporal_features(iq)
    else:
        features = spectral_features(iq, model.input_dim)
    latent = model.encode_features(features)
    scores = model.match_codes(latent)
    labels = model.classes[np.argmax(scores, axis=1)]
    interleaved = np.stack((iq.real, iq.imag), axis=-1).reshape(len(iq), -1)
    prefix = symbol.upper()
    lines = [f"#ifndef {prefix}_GOLDEN_H\n#define {prefix}_GOLDEN_H", "#include <stdint.h>", f"#define {prefix}_GOLDEN_CASES {len(iq)}u", "/* Functional float32 parity vectors. These do not measure target accuracy. */"]
    lines += [_array(f"{symbol}_golden_{name}", array, ctype) for name, array, ctype in (("iq", interleaved, "float"), ("features", features, "float"), ("latent", latent, "float"), ("scores", scores, "float"), ("labels", labels, "int64_t"))]
    aligned = _aligned_bank(model, aligned_iq, aligned_labels)
    if aligned is not None:
        references, compact = aligned
        energy = np.sqrt(np.sum(np.abs(iq) ** 2, axis=1, keepdims=True))
        unit = np.divide(iq, energy, out=np.zeros_like(iq), where=energy > 1e-12)
        reference_scores = np.abs(unit @ references.conj().T)
        aligned_scores = np.column_stack([reference_scores[:, compact == c].max(axis=1) for c in range(len(model.classes))])
        aligned_predictions = model.classes[aligned_scores.argmax(axis=1)]
        lines += [_array(f"{symbol}_golden_aligned_scores", aligned_scores), _array(f"{symbol}_golden_aligned_labels", aligned_predictions, "int64_t")]
    lines.append("#endif\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def diagnostic_iq(cases: int = 16) -> np.ndarray:
    if cases < 1:
        raise ValueError("cases must be positive")
    rng = np.random.default_rng(7321)
    iq = (rng.normal(size=(cases, 512)) + 1j * rng.normal(size=(cases, 512))).astype(np.complex64)
    iq[0] = 0
    if cases > 1:
        iq[1] = 0
        iq[1, 37] = 1 + 2j
    if cases > 2:
        iq[2] = np.exp(2j * np.pi * 37 * np.arange(512) / 512)
    if cases > 3:
        iq[3] = np.exp(2j * np.pi * 0.0007 * np.arange(512) ** 2)
    return iq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="safe NPZ model from LatentAutoencoder.save")
    parser.add_argument("--output", type=Path, required=True, help="generated model header, normally ogae_model.h")
    parser.add_argument("--golden-output", type=Path, help="optional generated parity header, normally ogae_golden.h")
    parser.add_argument("--golden-iq", type=Path, help="NPZ containing complex64 iq; otherwise generate diagnostic frames")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--symbol", default="ogae_exported")
    parser.add_argument("--aligned-bank", type=Path, help="optional train-only NPZ complex iq and integer labels for a C matched-filter control")
    args = parser.parse_args()
    if args.cases < 1 or args.cases > 4096:
        parser.error("--cases must be between 1 and 4096")
    with np.load(args.model, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    model_class = ConvLatentAutoencoder if metadata.get("format") == "gcfcr-conv-latent-ae" else LatentAutoencoder
    model = model_class.load(args.model)
    aligned_iq, aligned_labels = None, None
    if args.aligned_bank:
        with np.load(args.aligned_bank, allow_pickle=False) as data:
            aligned_iq, aligned_labels = data["iq"], data["labels"]
    report = export_model(model, args.output, symbol=args.symbol, aligned_iq=aligned_iq, aligned_labels=aligned_labels)
    if aligned_iq is not None:
        references = len(aligned_iq)
        report["aligned_control"] = {"references": references, "real_ops_estimate": 4 * 512 + references * (8 * 512 + 4), "sqrt_evaluations": references + 1, "workspace_bytes": 0, "scope": "aligned phase-invariant correlation, train-only references required"}
    if args.golden_output:
        if args.golden_iq:
            with np.load(args.golden_iq, allow_pickle=False) as data:
                iq = data["iq"][:args.cases]
        else:
            iq = diagnostic_iq(args.cases)
        export_golden(model, iq, args.golden_output, symbol=args.symbol, aligned_iq=aligned_iq, aligned_labels=aligned_labels)
        report["golden_cases"] = len(iq)
    elif args.golden_iq:
        parser.error("--golden-iq requires --golden-output")
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
