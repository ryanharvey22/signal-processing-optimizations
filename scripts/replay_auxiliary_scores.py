#!/usr/bin/env python3
"""Create an immutable, explicitly post-freeze full-score diagnostic supplement."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from optimized_filter import load_models, sha256, write_json, environment, code_snapshot
from numerical_verification import ATOL, RTOL, classes_for, scores_chunked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if any((args.spec / name).exists() for name in ("auxiliary_replay.json", "auxiliary_replay.npz")):
        raise ValueError("post-freeze supplements must not be overwritten")
    spec = json.loads((args.spec / "spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.experiment / "bundle.json").read_text(encoding="utf-8"))
    if sha256(args.experiment / "queries.npz") != bundle["queries_sha256"]:
        raise ValueError("query checksum mismatch")
    if sha256(args.experiment / "experiment.json") != bundle["experiment_sha256"]:
        raise ValueError("experiment checksum mismatch")
    if sha256(args.spec / "frozen_goldens.npz") != spec["frozen_goldens_sha256"]:
        raise ValueError("frozen golden checksum mismatch")
    models = load_models(args.experiment, manifest)
    arrays, hashes = {}, {}
    with np.load(args.experiment / "queries.npz", allow_pickle=False) as query, \
         np.load(args.spec / "frozen_goldens.npz", allow_pickle=False) as golden:
        order = np.random.default_rng(spec["seed"] + 991).permutation(len(query["iq"]))
        original_iq = query["iq"][np.argsort(order)]
        arrays["row_ids"] = query["row_ids"]
        for name, model in models.items():
            if manifest["models"][name]["sha256"] != spec["models"][name]["sha256"]:
                raise ValueError("frozen model identity changed")
            if name == "latent" or manifest["models"][name]["kind"] == "waveform":
                continue
            scores = scores_chunked(model, original_iq, 128)[order]
            classes = classes_for(model)
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite replay scores")
            np.testing.assert_array_equal(classes[scores.argmax(axis=1)], golden[name + "_prediction"])
            np.testing.assert_array_equal(query[name + "_prediction"], golden[name + "_prediction"])
            np.testing.assert_allclose(scores[:64], golden[name + "_scores"], rtol=RTOL, atol=ATOL)
            arrays[name + "_scores"], arrays[name + "_classes"] = scores, classes
            hashes[name] = spec["models"][name]["sha256"]
    np.savez_compressed(args.spec / "auxiliary_replay.npz", **arrays)
    write_json(args.spec / "auxiliary_replay.json", {
        "format": "post-freeze-auxiliary-score-replay-v1",
        "purpose": "Numerical diagnostic captured after final test; not original evaluation scores or new model selection.",
        "frozen_spec_sha256": sha256(args.spec / "spec.json"),
        "frozen_goldens_sha256": sha256(args.spec / "frozen_goldens.npz"),
        "original_bundle_sha256": sha256(args.experiment / "bundle.json"),
        "scores_sha256": sha256(args.spec / "auxiliary_replay.npz"),
        "model_sha256": hashes, "all_original_auxiliary_labels_reproduced": True,
        "evaluation_order": "Original split order, then reordered by seed+991 permutation for bundle alignment",
        "outer_chunk_size": 128, "dtype": "float32", "blas_threads": 1,
        "rtol": RTOL, "atol": ATOL, "environment": environment(),
        "code_snapshot": code_snapshot(), "generator_sha256": sha256(Path(__file__)),
        "verification_sha256": sha256(Path(__file__).with_name("numerical_verification.py"))})


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
