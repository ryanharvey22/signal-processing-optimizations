"""
Experiment matrix: **Classical matched filter** (class-mean templates, waveform domain).

RadChar: per-class mean complex IQ from train, unit energy; query unit-normalized;
score |⟨μ_c, x⟩| or FFT circular correlation peak. MNIST: class-mean + cosine.

See ``matched_filter.md`` and ``scripts/eval_matched_filter_baseline.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gcfcr.data.pipeline import build_dataset
from gcfcr.methods._collate import collate_mnist_flat, collate_radchar_iq

ScoringMode = Literal["time", "fft"]


@torch.no_grad()
def fit_radchar_class_mean_templates(
    loader: DataLoader,
    num_classes: int,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    sums = torch.zeros(num_classes, n_samples, dtype=torch.complex64, device=device)
    counts = torch.zeros(num_classes, device=device)
    for iq, y in loader:
        iq = iq.to(device)
        y = y.to(device)
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] += iq[m].sum(dim=0).to(torch.complex64)
                counts[c] += m.sum().float()
    mu = sums / counts.clamp(min=1).unsqueeze(1).to(torch.complex64)
    denom = mu.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    return mu / denom


@torch.no_grad()
def fit_mnist_class_mean_templates(
    loader: DataLoader,
    num_classes: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    sums = torch.zeros(num_classes, dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] += x[m].sum(dim=0)
                counts[c] += m.sum().float()
    mu = sums / counts.clamp(min=1).unsqueeze(1)
    return F.normalize(mu, dim=1)


def normalize_radchar_iq_energy(iq: torch.Tensor) -> torch.Tensor:
    return iq / iq.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)


@torch.no_grad()
def score_radchar_time_domain(
    iq_unit: torch.Tensor, templates: torch.Tensor
) -> torch.Tensor:
    inner = (iq_unit.unsqueeze(1) * templates.conj().unsqueeze(0)).sum(dim=-1)
    return inner.abs()


@torch.no_grad()
def score_radchar_fft_peak(
    iq_unit: torch.Tensor, templates_fft_conj: torch.Tensor
) -> torch.Tensor:
    x_fft = torch.fft.fft(iq_unit, dim=-1)
    corr = torch.fft.ifft(
        x_fft.unsqueeze(1) * templates_fft_conj.unsqueeze(0),
        dim=-1,
    )
    return corr.abs().amax(dim=-1)


@torch.no_grad()
def predict_radchar(
    iq: torch.Tensor,
    templates: torch.Tensor,
    mode: ScoringMode,
    templates_fft_conj: torch.Tensor | None,
) -> torch.Tensor:
    iq_u = normalize_radchar_iq_energy(iq)
    if mode == "time":
        scores = score_radchar_time_domain(iq_u, templates)
    else:
        assert templates_fft_conj is not None
        scores = score_radchar_fft_peak(iq_u, templates_fft_conj)
    return scores.argmax(dim=1)


@torch.no_grad()
def score_mnist_cosine(x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    xn = F.normalize(x, dim=1)
    return xn @ templates.T


@torch.no_grad()
def evaluate_accuracy(
    *,
    dataset: Literal["mnist", "radchar"],
    split: Literal["val", "test"],
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, int, int]:
    """Returns (accuracy, correct, total)."""
    if dataset == "mnist":
        num_classes, dim = 10, 784
        collate = collate_mnist_flat
        train_ld = DataLoader(
            build_dataset("mnist", data_dir=data_dir, split="train"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        templates = fit_mnist_class_mean_templates(train_ld, num_classes, dim, device)
        eval_ld = DataLoader(
            build_dataset("mnist", data_dir=data_dir, split=split),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        correct, total = 0, 0
        for x, y in eval_ld:
            x = x.to(device)
            y = y.to(device)
            scores = score_mnist_cosine(x, templates)
            pred = scores.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    else:
        num_classes, n_samples = 5, 512
        collate = collate_radchar_iq
        train_ld = DataLoader(
            build_dataset("radchar", data_dir=data_dir, split="train"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        templates = fit_radchar_class_mean_templates(
            train_ld, num_classes, n_samples, device
        )
        eval_ld = DataLoader(
            build_dataset("radchar", data_dir=data_dir, split=split),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        correct, total = 0, 0
        for iq, y in eval_ld:
            iq = iq.to(device)
            y = y.to(device)
            iq_u = normalize_radchar_iq_energy(iq)
            scores = score_radchar_time_domain(iq_u, templates)
            pred = scores.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)

    acc = correct / max(total, 1)
    return acc, correct, total


@torch.no_grad()
def bench_radchar_single_frame_ms(
    frame_iq: torch.Tensor,
    templates: torch.Tensor,
    mode: ScoringMode,
    templates_fft_conj: torch.Tensor | None,
    repeats: int,
) -> float:
    """Average milliseconds per single-frame predict (warm outside)."""
    start = time.perf_counter()
    for _ in range(repeats):
        _ = predict_radchar(frame_iq, templates, mode, templates_fft_conj)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s * 1000.0) / max(repeats, 1)


@torch.no_grad()
def run_radchar_cpu_runtime_benchmark(
    *,
    data_dir: Path,
    split: Literal["val", "test", "all"],
    mode: ScoringMode,
    batch_size: int,
    num_workers: int,
    max_eval_samples: int | None,
    train_max_samples: int | None,
    frame_repeats: int,
    warmup_batches: int,
    cpu_threads: int,
) -> Dict[str, Any]:
    """
    Build class-mean templates on train, time discrimination on eval split (CPU).

    Returns numeric fields for printing or logging.
    """
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")

    train_ds = build_dataset(
        "radchar",
        data_dir=data_dir,
        split="train",
        radchar_max_samples=train_max_samples,
    )
    eval_ds = build_dataset(
        "radchar",
        data_dir=data_dir,
        split=split,
        radchar_max_samples=max_eval_samples,
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_radchar_iq,
        num_workers=num_workers,
    )
    eval_ld = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_radchar_iq,
        num_workers=num_workers,
    )

    num_classes, n_samples = 5, 512
    t0 = time.perf_counter()
    templates = fit_radchar_class_mean_templates(
        train_ld, num_classes, n_samples, device
    )
    template_build_s = time.perf_counter() - t0
    templates_fft_conj = (
        torch.fft.fft(templates, dim=-1).conj() if mode == "fft" else None
    )

    first_batch_iq, _ = next(iter(eval_ld))
    frame_iq = first_batch_iq[:1].to(device)
    _ = predict_radchar(frame_iq, templates, mode, templates_fft_conj)

    single_frame_ms = bench_radchar_single_frame_ms(
        frame_iq, templates, mode, templates_fft_conj, frame_repeats
    )

    warmup_done = 0
    for iq, _ in eval_ld:
        iq = iq.to(device)
        _ = predict_radchar(iq, templates, mode, templates_fft_conj)
        warmup_done += 1
        if warmup_done >= warmup_batches:
            break

    correct = 0
    total = 0
    t1 = time.perf_counter()
    for iq, y in eval_ld:
        iq = iq.to(device)
        y = y.to(device)
        pred = predict_radchar(iq, templates, mode, templates_fft_conj)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    discrim_s = time.perf_counter() - t1

    acc = correct / max(total, 1)
    frames_per_s = total / discrim_s if discrim_s > 0 else float("inf")
    mean_frame_ms = (discrim_s * 1000.0) / max(total, 1)

    return {
        "mode": mode,
        "split": split,
        "samples": total,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "template_build_s": template_build_s,
        "single_frame_ms": single_frame_ms,
        "frame_repeats": frame_repeats,
        "dataset_discrimination_s": discrim_s,
        "mean_frame_ms": mean_frame_ms,
        "fps": frames_per_s,
        "accuracy": acc,
        "correct": correct,
        "total": total,
    }
