"""Portable, validated transfer of already computed classical probabilities.

Only trusted artifacts from this repository are loaded (pickle payloads).
Hardware and local package builds are provenance, not cache identity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, TEST_CSV, TRAIN_CSV


def portable_file_hash(path):
    """Ignore checkout newline conversion for Python sources and raw CSV files."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def classical_contract():
    pipeline = ast.parse((PROJECT_ROOT / "src/dl_hybrid_pipeline.py").read_text(encoding="utf-8"))
    names = {"fold_features", "classical_predictions", "splits", "browser_mapping"}
    contract = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name in names}
    for name in ("features.py", "parser.py", "postprocess.py", "config.py"):
        tree = ast.parse((PROJECT_ROOT / "src" / name).read_text(encoding="utf-8"))
        contract[name] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return contract


def subset_indices(y, split_function):
    subsets = {}
    for fold, (train, val) in enumerate(split_function(y), 1):
        subsets[f"outer{fold}"] = (train, val, True)
        for inner, (relative_train, relative_val) in enumerate(split_function(y[train]), 1):
            subsets[f"outer{fold}_inner{inner}"] = (train[relative_train], train[relative_val], False)
    return subsets


def import_classical_bundle(bundle, cache, y, classes, model_names, split_function):
    """Validate data, extraction code, fold membership, columns and file hashes."""
    bundle, cache = Path(bundle), Path(cache)
    metadata = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    expected_data = {p.name: portable_file_hash(p) for p in (TRAIN_CSV, TEST_CSV)}
    if metadata.get("format_version") != 1 or metadata["data_hashes"] != expected_data:
        raise ValueError("Transfer bundle does not match the raw train/test data")
    if metadata["classical_contract"] != classical_contract():
        raise ValueError("Transfer bundle uses different classical feature/model definitions")
    if metadata["classes"] != np.asarray(classes).tolist() or metadata["model_names"] != list(model_names):
        raise ValueError("Transfer bundle class columns or model list differ")
    subsets = subset_indices(y, split_function)
    with TEST_CSV.open(encoding="utf-8") as stream:
        n_test = sum(1 for _ in stream)
    cache.mkdir(parents=True, exist_ok=True)
    imported = []
    for item in metadata["items"]:
        tag, filename = item["tag"], item["filename"]
        if Path(filename).name != filename or tag not in subsets:
            raise ValueError("Invalid transfer file or fold tag")
        source = bundle / filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Transfer checksum differs: {filename}")
        train, val, outer = subsets[tag]
        if item["train_indices"] != train.tolist() or item["eval_indices"] != val.tolist():
            raise ValueError(f"Transfer fold indices differ: {tag}")
        with source.open("rb") as stream:
            probabilities = pickle.load(stream)
        expected_shape = (len(val) + (n_test if outer else 0), len(classes))
        if set(probabilities) != set(model_names):
            raise ValueError(f"Transfer model keys differ: {tag}")
        for p in probabilities.values():
            if (p.shape != expected_shape or not np.isfinite(p).all() or np.any(p < 0)
                    or not np.allclose(p.sum(1), 1, atol=1e-5)):
                raise ValueError(f"Invalid transfer probabilities: {tag}")
        target = cache / filename
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
            (cache / f"{tag}_classical_origin.json").write_text(
                json.dumps({"bundle_manifest": metadata, "item": item}, indent=2), encoding="utf-8")
            imported.append(tag)
    return imported


def export_classical_bundle(run, destination, last_logged_epoch=None):
    """Export a stopped legacy run after checking the pre-edit source snapshot."""
    from src.dl_hybrid_pipeline import MODEL_NAMES, splits
    run, destination = Path(run), Path(destination)
    origin = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((run / "transfer_source_contract.json").read_text(encoding="utf-8"))
    if snapshot["contract"] != classical_contract():
        raise ValueError("Classical definitions changed since the original run")
    for name, sha in snapshot["source_hashes"].items():
        if origin["files"][name] != sha:
            raise ValueError(f"Source snapshot differs from original manifest: {name}")
    data_hashes = {p.name: portable_file_hash(p) for p in (TRAIN_CSV, TEST_CSV)}
    if data_hashes != snapshot["data_hashes"]:
        raise ValueError("Raw data changed since original run")
    with TRAIN_CSV.open(encoding="utf-8") as stream:
        classes, y = np.unique([line.split(",", 1)[0] for line in stream], return_inverse=True)
    subsets = subset_indices(y, splits)
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for source in sorted(run.glob("*_classical.pkl")):
        tag = source.name.removesuffix("_classical.pkl")
        train, val, _ = subsets[tag]
        shutil.copyfile(source, destination / source.name)
        items.append({"tag": tag, "filename": source.name,
                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "train_indices": train.tolist(), "eval_indices": val.tolist()})
    metadata = {"format_version": 1, "data_hashes": data_hashes,
                "classical_contract": classical_contract(), "classes": classes.tolist(),
                "model_names": list(MODEL_NAMES), "origin_manifest": origin,
                "items": items, "network_checkpoint_available": False,
                "last_network_epoch_logged_before_stop": last_logged_epoch}
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {len(items)} classical cache(s) to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-logged-epoch", type=int)
    args = parser.parse_args()
    export_classical_bundle(args.legacy_run, args.output, args.last_logged_epoch)
