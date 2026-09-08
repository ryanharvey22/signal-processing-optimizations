"""Assemble validation-selected models before any constituent opens its test set."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from optimized_filter import code_snapshot, load_models, sha256, write_json, ensure_tunable

def assemble(experiments, out):
    if out.exists():
        raise ValueError("choose a new output directory")
    sources = []
    for directory in experiments:
        ensure_tunable(directory)
        manifest = json.loads((directory / "experiment.json").read_text())
        if manifest.get("test_used_for_selection") is not False:
            raise ValueError("source must explicitly attest no test selection")
        load_models(directory, manifest)  # Validate every source artifact hash.
        if sources and manifest["dataset"] != sources[0][1]["dataset"]:
            raise ValueError("all experiments must share identical data and splits")
        sources.append((directory, manifest))
    if not sources:
        raise ValueError("at least one experiment is required")
    def rank(row):
        return row["validation_accuracy"], -row["operations"]["real_flops_estimate"]
    winner_index = max(range(len(sources)), key=lambda i: rank(sources[i][1]["models"]["latent"]))
    target = copy.deepcopy(sources[winner_index][1])
    target["models"] = {}
    target["validation_source_manifests"] = []
    selection = {}
    candidates = []
    for index, (directory, manifest) in enumerate(sources):
        target["validation_source_manifests"].append({
            "directory_name": directory.name, "directory_path": str(directory.resolve()), "manifest_sha256": sha256(directory / "experiment.json"),
            "training_start_snapshot": manifest.get("training_start_snapshot"),
            "training_sources_changed_during_run": manifest.get("training_sources_changed_during_run", "not recorded"),
            "training_versions": manifest.get("training_versions"),
        })
        for name, row in manifest["models"].items():
            if name == "latent":
                key = "latent" if index == winner_index else f"encoder_candidate_{index}"
            elif name == "conv_no_reconstruction":
                key = "encoder_no_reconstruction" if index == winner_index else f"no_reconstruction_{index}"
            else:
                key = name
            if key not in selection or rank(row) > rank(selection[key][1]):
                selection[key] = (directory, copy.deepcopy(row))
        history_path = directory / "validation_candidates.json"
        if history_path.exists():
            candidates.append({"experiment": directory.name, "candidates": json.loads(history_path.read_text())})
    # Spectral wins are permitted; they have their own direct classifier controls.
    target["selected_encoder_source"] = sources[winner_index][0].name
    target["selection"] = "highest validation accuracy across the declared encoder experiments; ties favor lower estimated operations"
    target["primary_comparator"] = max(("mf_aligned", "mf_fft"), key=lambda name: rank(selection[name][1]))
    target["code_snapshot"] = code_snapshot()
    target["assembly_source_sha256"] = sha256(Path(__file__))
    target["test_used_for_selection"] = False
    target["training_source_provenance_limit"] = (
        "Some exploratory runs began before start snapshots were implemented and while source files changed. "
        "Their end snapshots are not exact training-source attestations. Frozen inference parameters and golden outputs are independently hashed."
    )
    out.mkdir(parents=True)
    for name, (directory, row) in selection.items():
        source = directory / row["file"]
        row["selection_origin"] = {
            "directory_name": directory.name, "model_file": row["file"],
            "manifest_sha256": sha256(directory / "experiment.json"),
        }
        row["file"] = name + ".npz"
        shutil.copyfile(source, out / row["file"])
        target["models"][name] = row
    write_json(out / "experiment.json", target)
    write_json(out / "validation_candidates.json", candidates)
    write_json(out / "data_manifest.json", target["dataset"])
    print(json.dumps({"selected": target["selected_encoder_source"],
                      "validation": {name: row["validation_accuracy"] for name, row in target["models"].items()}}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.experiments, args.out)

if __name__ == "__main__":
    main()
