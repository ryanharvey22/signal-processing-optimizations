#!/usr/bin/env python3
"""
CLI for timing **classical matched filter** on RadChar (same method module as eval).

Implementation: ``gcfcr.methods.classical_matched_filter.run_radchar_cpu_runtime_benchmark``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.methods.classical_matched_filter import run_radchar_cpu_runtime_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--split", choices=("val", "test", "all"), default="val")
    p.add_argument("--mode", choices=("time", "fft"), default="fft")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--frame-repeats", type=int, default=2000)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.device != "cpu":
        raise ValueError("This benchmark is intended for CPU; use --device cpu.")

    m = run_radchar_cpu_runtime_benchmark(
        data_dir=args.data_dir,
        split=args.split,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=args.max_eval_samples,
        train_max_samples=args.train_max_samples,
        frame_repeats=args.frame_repeats,
        warmup_batches=args.warmup_batches,
        cpu_threads=args.cpu_threads,
    )

    print("matched_filter_runtime_benchmark")
    print(f"  mode={m['mode']}  split={m['split']}  device=cpu")
    print(
        f"  samples={m['samples']}  batch_size={m['batch_size']}  "
        f"cpu_threads={m['cpu_threads']}"
    )
    print(f"  template_build_s={m['template_build_s']:.6f}")
    print(
        f"  single_frame_ms={m['single_frame_ms']:.6f}  "
        f"(repeats={m['frame_repeats']})"
    )
    print(
        f"  dataset_discrimination_s={m['dataset_discrimination_s']:.6f}  "
        f"mean_frame_ms={m['mean_frame_ms']:.6f}  fps={m['fps']:.2f}"
    )
    print(
        f"  accuracy={m['accuracy']:.4f}  ({m['correct']}/{m['total']})"
    )


if __name__ == "__main__":
    main()
