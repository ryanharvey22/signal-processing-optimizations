#!/usr/bin/env python3
"""Fit on train/validation, freeze a test evaluation, and benchmark exact artifacts."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcfcr.optimized.autoencoder import LatentAutoencoder, fit
from gcfcr.optimized.features import spectral_features
from gcfcr.optimized.baselines import WaveformMatchedFilter, FeaturePrototypeMatcher
from gcfcr.optimized.classifier import FeatureClassifier, fit_classifier
from gcfcr.optimized.data import load_experiment_data
from gcfcr.optimized.metrics import classification_metrics, paired_accuracy_interval

LOADERS = {"latent": LatentAutoencoder, "waveform": WaveformMatchedFilter,
           "feature": FeaturePrototypeMatcher, "classifier": FeatureClassifier}

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def data_from_config(config, h5=None):
    options = dict(config)
    options["h5_path"] = h5 if h5 is not None else options.pop("h5_path", None)
    return load_experiment_data(**options)

def cost(model, samples=512):
    counts = model.operation_counts(samples)
    value = counts.get("real_flops_estimate")
    if not isinstance(value, (int, float)):
        raise ValueError("benchmark requires a defined conventional operation estimate")
    return value

def predict_chunked(model, iq, chunk=128):
    return np.concatenate([model.predict(iq[i:i + chunk]) for i in range(0, len(iq), chunk)])

def progress(name, epochs):
    def callback(row):
        if row["epoch"] == 1 or row["epoch"] % 10 == 0 or row["epoch"] == epochs:
            print(f"{name} epoch={row['epoch']} val_accuracy={row['validation_accuracy']:.5f}", flush=True)
    return callback

def save_model(out, name, kind, model, params, accuracy):
    filename = name + ".npz"
    model.save(out / filename)
    return {"kind": kind, "file": filename, "sha256": sha256(out / filename),
            "parameters": params, "validation_accuracy": accuracy,
            "storage_bytes": model.storage_bytes, "operations": model.operation_counts()}

def load_models(directory, manifest):
    result = {}
    for name, row in manifest["models"].items():
        if Path(row["file"]).name != row["file"]:
            raise ValueError("artifact model paths must be filenames")
        path = directory / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"model hash mismatch: {name}")
        result[name] = LOADERS[row["kind"]].load(path)
    return result

def fit_experiment(args):
    import torch
    torch.set_num_threads(args.threads)
    out = args.out
    if (out / "experiment.json").exists():
        raise ValueError("experiment already exists; choose a new output to preserve validation history")
    out.mkdir(parents=True, exist_ok=True)
    data_config = {"h5_path": str(args.h5.resolve()) if args.h5 else None, "seed": args.seed,
                   "train_cap": args.train_cap, "val_cap": args.val_cap,
                   "test_cap": args.test_cap, "synthetic_total": args.synthetic_total}
    data = data_from_config(data_config)
    write_json(out / "data_manifest.json", data.manifest)
    print(f"data={data.manifest['source']} train={len(data.train)} val={len(data.val)} "
          f"test_reserved={len(data.test)}", flush=True)
    train = spectral_features(data.train.iq, args.bins)
    val = spectral_features(data.val.iq, args.bins)
    models, candidates = {}, []
    best_key, best = None, None
    for hidden in args.hidden_dims:
        for latent in args.latent_dims:
            for k in args.prototype_counts:
                params = dict(hidden_dim=hidden, latent_dim=latent, prototypes_per_class=k,
                              epochs=args.epochs, seed=args.seed, batch_size=args.batch_size)
                name = f"ae-h{hidden}-l{latent}-p{k}"
                model, history = fit(train, data.train.y, val, data.val.y, **params,
                                     progress=progress(name, args.epochs))
                accuracy = float(np.mean(model.predict_features(val) == data.val.y))
                row = {"method": name, "parameters": params, "validation_accuracy": accuracy,
                       "operations": model.operation_counts(), "history": history}
                candidates.append(row)
                model.save(out / (name + ".npz"))
                write_json(out / "validation_candidates.json", candidates)
                key = (accuracy, -cost(model))
                if best_key is None or key > best_key:
                    best_key, best = key, (model, params, accuracy)
    models["latent"] = save_model(out, "latent", "latent", *best)
    # The same feature frontend and supervised data budget control for whether
    # reconstruction/latent matching actually adds value over direct prediction.
    for hidden in sorted(set([0, *args.hidden_dims])):
        name = "linear" if hidden == 0 else "classifier"
        model, history = fit_classifier(train, data.train.y, val, data.val.y, hidden_dim=hidden,
                epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
                progress=progress(name, args.epochs))
        accuracy = float(np.mean(model.classes[model.scores_features(val).argmax(1)] == data.val.y))
        params = {"hidden_dim": hidden, "epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size}
        if name not in models or accuracy > models[name]["validation_accuracy"]:
            models[name] = save_model(out, name, "classifier", model, params, accuracy)
        candidates.append({"method": name, "parameters": params, "validation_accuracy": accuracy, "history": history})
    for mode in ("aligned", "fft"):
        best_key, best = None, None
        for k in args.mf_bank_counts:
            params = {"mode": mode, "templates_per_class": k, "seed": args.seed}
            model = WaveformMatchedFilter(**params).fit(data.train.iq, data.train.y)
            accuracy = float(np.mean(predict_chunked(model, data.val.iq) == data.val.y))
            candidates.append({"method": "mf_" + mode, "parameters": params, "validation_accuracy": accuracy,
                               "operations": model.operation_counts()})
            print(f"mf_{mode} templates/class={k} val_accuracy={accuracy:.5f}", flush=True)
            key = (accuracy, -cost(model))
            if best_key is None or key > best_key:
                best_key, best = key, (model, params, accuracy)
        models["mf_" + mode] = save_model(out, "mf_" + mode, "waveform", *best)
    for pca in (None, args.pca_dim):
        name = "spectral" if pca is None else "pca"
        best_key, best = None, None
        for k in args.prototype_counts:
            params = {"pca_dim": pca, "prototypes_per_class": k, "bins": args.bins}
            model = FeaturePrototypeMatcher(**params).fit(data.train.iq, data.train.y)
            accuracy = float(np.mean(predict_chunked(model, data.val.iq) == data.val.y))
            candidates.append({"method": name, "parameters": params, "validation_accuracy": accuracy,
                               "operations": model.operation_counts()})
            print(f"{name} prototypes/class={k} val_accuracy={accuracy:.5f}", flush=True)
            key = (accuracy, -cost(model))
            if best_key is None or key > best_key:
                best_key, best = key, (model, params, accuracy)
        models[name] = save_model(out, name, "feature", *best)
    # Primary comparator is declared from validation accuracy only, before test.
    primary = max(("mf_aligned", "mf_fft"),
                  key=lambda name: (models[name]["validation_accuracy"], -models[name]["operations"]["real_flops_estimate"]))
    manifest = {"format": "optimized-filter-experiment-v1", "data_config": data_config,
                "dataset": data.manifest, "models": models, "primary_comparator": primary,
                "selection": "highest validation accuracy; ties favor lower estimated operations",
                "test_used_for_selection": False, "seed": args.seed,
                "training_versions": {"torch": torch.__version__, "numpy": np.__version__},
                "scope": "five-class frame discrimination, not universal known-signal detection superiority"}
    write_json(out / "validation_candidates.json", candidates)
    write_json(out / "experiment.json", manifest)
    print(json.dumps({name: row["validation_accuracy"] for name, row in models.items()}), flush=True)

def create_bundle(directory, manifest, data, models):
    # Query payload is private experiment output, never committed as a dataset.
    order = np.random.default_rng(manifest["seed"] + 991).permutation(len(data.test))
    iq = data.test.iq[order]
    arrays = {"iq": iq, "y": data.test.y[order], "snr_db": data.test.snr_db[order],
              "row_ids": data.test.row_ids[order], "latent_codes": models["latent"].encode(iq[:64])}
    for name, model in models.items():
        arrays[name + "_prediction"] = predict_chunked(model, iq)
        arrays[name + "_scores"] = model.scores(iq[:64])
    query_path = directory / "queries.npz"
    np.savez_compressed(query_path, **arrays)
    bundle = {"format": "optimized-filter-native-bundle-v1", "models": manifest["models"],
              "queries_sha256": sha256(query_path), "seed": manifest["seed"],
              "primary_comparator": manifest["primary_comparator"],
              "source_sha256": data.manifest["source_sha256"],
              "experiment_sha256": sha256(directory / "experiment.json")}
    write_json(directory / "bundle.json", bundle)

def evaluate(args):
    directory = args.experiment
    if (directory / "test_report.json").exists():
        raise ValueError("test report already exists; frozen results must not be overwritten")
    manifest = json.loads((directory / "experiment.json").read_text())
    models = load_models(directory, manifest)
    data = data_from_config(manifest["data_config"], args.h5)
    if data.manifest != manifest["dataset"]:
        raise ValueError("data identity/split manifest changed")
    predictions, results = {}, {}
    for name, model in models.items():
        prediction = predict_chunked(model, data.test.iq)
        predictions[name] = prediction
        results[name] = {**classification_metrics(data.test.y, prediction, snr_db=data.test.snr_db),
                         "operations": model.operation_counts(), "storage_bytes": model.storage_bytes}
    primary = manifest["primary_comparator"]
    interval = paired_accuracy_interval(data.test.y, predictions["latent"], predictions[primary],
                                       groups=data.test.group_ids, seed=manifest["seed"])
    report = {"dataset": manifest["dataset"], "results": results, "primary_comparator": primary,
              "paired_accuracy_difference": interval,
              "accuracy_superiority_gate": interval["ci95"][0] > 0,
              "arithmetic_cost_gate": cost(models["latent"]) < cost(models[primary]),
              "native_latency_gate": "pending separate native measurements",
              "state_of_the_art_claim": False,
              "limits": ["Rows are disjoint; unrecorded realization dependence is unknown.",
                         "Spectral features discard phase and cannot reconstruct original IQ.",
                         "Arithmetic estimates are not measured hardware instructions; special functions are separate.",
                         "Published results using different datasets/tasks are not a head-to-head comparison."]}
    write_json(directory / "test_report.json", report)
    create_bundle(directory, manifest, data, models)
    print(json.dumps({name: value["accuracy"] for name, value in results.items()}), flush=True)
    print(json.dumps({"primary": primary, "paired_interval": interval,
                      "accuracy_gate": report["accuracy_superiority_gate"],
                      "operations_gate": report["arithmetic_cost_gate"]}), flush=True)

def environment():
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np.show_runtime()
    cpu = platform.processor()
    if sys.platform.startswith("linux"):
        text = Path("/proc/cpuinfo").read_text()
        cpu = next((line.split(":", 1)[1].strip() for line in text.splitlines()
                    if line.startswith("model name") or line.startswith("CPU part")), cpu)
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": cpu,
            "python": sys.version, "numpy": np.__version__, "threadpools": threadpool_info(),
            "numpy_runtime": capture.getvalue(), "pid": os.getpid()}

def native(args):
    directory = args.experiment
    bundle = json.loads((directory / "bundle.json").read_text())
    if sha256(directory / "queries.npz") != bundle["queries_sha256"]:
        raise ValueError("query artifact checksum mismatch")
    if sha256(directory / "experiment.json") != bundle["experiment_sha256"]:
        raise ValueError("experiment manifest checksum mismatch")
    models = load_models(directory, bundle)
    with np.load(directory / "queries.npz", allow_pickle=False) as d:
        iq = d["iq"]
        rows = {}
        for name, model in models.items():
            scores = model.scores(iq[:64])
            np.testing.assert_allclose(scores, d[name + "_scores"], rtol=3e-4, atol=3e-5)
            predictions = predict_chunked(model, iq)
            if not np.array_equal(predictions, d[name + "_prediction"]):
                raise ValueError(f"{name}: native predictions differ from frozen golden results")
            for _ in range(args.warmup):
                model.predict(iq[:1])
            samples = []
            for i in range(args.repeats):
                frame = iq[i % len(iq):i % len(iq) + 1]
                start = time.perf_counter_ns()
                model.predict(frame)
                samples.append((time.perf_counter_ns() - start) / 1000)
            batch = iq[:min(args.batch_size, len(iq))]
            model.predict(batch)
            batch_times = []
            for _ in range(12):
                start = time.perf_counter_ns()
                model.predict(batch)
                batch_times.append((time.perf_counter_ns() - start) / 1e9)
            rows[name] = {"accuracy": float(np.mean(predictions == d["y"])),
                "batch1_p50_us": float(np.median(samples)), "batch1_p95_us": float(np.quantile(samples, 0.95)),
                "batch_size": len(batch), "batch_frames_per_second": float(len(batch) / np.median(batch_times)),
                "storage_bytes": model.storage_bytes, "operations": model.operation_counts(),
                "golden_predictions_equal": True}
        np.testing.assert_allclose(models["latent"].encode(iq[:64]), d["latent_codes"], rtol=3e-4, atol=3e-5)
    primary = bundle["primary_comparator"]
    result = {"environment": environment(), "bundle_sha256": sha256(directory / "bundle.json"),
              "queries_sha256": bundle["queries_sha256"], "results": rows,
              "primary_comparator": primary, "latency_gate":
              rows["latent"]["batch1_p50_us"] < rows[primary]["batch1_p50_us"],
              "measurement": "wall-clock input-IQ through prediction; warm caches; single process; includes Python/FFT/BLAS",
              "hardware_microcontroller_measurement": False}
    write_json(args.output, result)
    print(json.dumps({name: {"p50_us": row["batch1_p50_us"], "fps": row["batch_frames_per_second"]}
                      for name, row in rows.items()}), flush=True)

def freeze(args):
    """Publish learned parameters/specification; raw reference/query IQ stays outside Git."""
    source, target = args.experiment, args.output
    manifest = json.loads((source / "experiment.json").read_text())
    if not (source / "test_report.json").exists():
        raise ValueError("evaluate selected models before freezing")
    target.mkdir(parents=True, exist_ok=True)
    manifest["data_config"]["h5_path"] = None
    for name, row in manifest["models"].items():
        if row["kind"] != "waveform":
            shutil.copyfile(source / row["file"], target / row["file"])
    write_json(target / "spec.json", manifest)
    shutil.copyfile(source / "test_report.json", target / "test_report.json")

def rebuild(args):
    """Recreate only deterministic waveform references; never retrain or retune."""
    spec = json.loads((args.spec / "spec.json").read_text())
    target = args.out
    target.mkdir(parents=True, exist_ok=True)
    data = data_from_config(spec["data_config"], args.h5)
    if data.manifest != spec["dataset"]:
        raise ValueError("frozen dataset/source/split does not match")
    for name, row in spec["models"].items():
        if row["kind"] == "waveform":
            model = WaveformMatchedFilter(**row["parameters"]).fit(data.train.iq, data.train.y)
            model.save(target / row["file"])
            # Archive container timestamps vary; golden hashes bind this rebuilt
            # artifact, while deterministic source IDs/parameters bind its origin.
            row["sha256"] = sha256(target / row["file"])
        else:
            source = args.spec / row["file"]
            if sha256(source) != row["sha256"]:
                raise ValueError("frozen learned parameter checksum mismatch")
            shutil.copyfile(source, target / row["file"])
    write_json(target / "experiment.json", spec)
    models = load_models(target, spec)
    create_bundle(target, spec, data, models)
    print(f"rebuilt frozen inference bundle at {target}", flush=True)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fit")
    f.add_argument("--h5", type=Path)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--hidden-dims", type=int, nargs="+", default=[0, 48])
    f.add_argument("--latent-dims", type=int, nargs="+", default=[8, 16])
    f.add_argument("--prototype-counts", type=int, nargs="+", default=[1, 4])
    f.add_argument("--mf-bank-counts", type=int, nargs="+", default=[1, 4, 16])
    f.add_argument("--bins", type=int, default=128)
    f.add_argument("--pca-dim", type=int, default=16)
    f.add_argument("--train-cap", type=int, default=40000)
    f.add_argument("--val-cap", type=int, default=5000)
    f.add_argument("--test-cap", type=int, default=5000)
    f.add_argument("--synthetic-total", type=int, default=6000)
    f.add_argument("--batch-size", type=int, default=256)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--threads", type=int, default=1)
    e = commands.add_parser("evaluate")
    e.add_argument("--experiment", type=Path, required=True)
    e.add_argument("--h5", type=Path)
    n = commands.add_parser("native")
    n.add_argument("--experiment", type=Path, required=True)
    n.add_argument("--output", type=Path, required=True)
    n.add_argument("--repeats", type=int, default=300)
    n.add_argument("--warmup", type=int, default=30)
    n.add_argument("--batch-size", type=int, default=128)
    z = commands.add_parser("freeze")
    z.add_argument("--experiment", type=Path, required=True)
    z.add_argument("--output", type=Path, required=True)
    r = commands.add_parser("rebuild")
    r.add_argument("--spec", type=Path, required=True)
    r.add_argument("--h5", type=Path)
    r.add_argument("--out", type=Path, required=True)
    return p

def main():
    args = parser().parse_args()
    if args.command == "native" and (args.repeats < 10 or args.warmup < 0 or args.batch_size < 1):
        raise ValueError("native timing requires repeats>=10, warmup>=0, batch_size>=1")
    with threadpool_limits(limits=getattr(args, "threads", 1)):
        {"fit": fit_experiment, "evaluate": evaluate, "native": native, "freeze": freeze, "rebuild": rebuild}[args.command](args)

if __name__ == "__main__":
    main()
