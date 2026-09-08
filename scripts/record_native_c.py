"""Bind native C timing output to the exact model, generated headers and binary."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.results.read_text(encoding="utf-8-sig"))
    bundle = json.loads((args.experiment / "bundle.json").read_text())
    model = bundle["models"]["latent"]
    if sha256(args.experiment / model["file"]) != model["sha256"]:
        raise ValueError("inference model checksum mismatch")
    row.update({
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_sha256": model["sha256"],
        "bundle_sha256": sha256(args.experiment / "bundle.json"),
        "generated_model_sha256": sha256(args.generated / "ogae_model.h"),
        "generated_goldens_sha256": sha256(args.generated / "ogae_golden.h"),
        "binary_sha256": sha256(args.binary),
        "inference_operation_estimate": model["operations"]["real_flops_estimate"],
        "estimated_arithmetic_ops_per_second": model["operations"]["real_flops_estimate"] * row["fps_at_median_trial"],
        "operation_rate_scope": "arithmetic estimate times measured throughput; not hardware counter FLOPS"
    })
    args.results.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
