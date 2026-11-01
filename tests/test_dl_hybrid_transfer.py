"""Reject transfer corruption and preserve the exact class/fold alignment."""
import hashlib
import json
import pickle

import numpy as np
import pytest

from src import dl_hybrid_transfer as transfer
from src.dl_hybrid_pipeline import splits


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    y = np.repeat(np.arange(3), 8)
    classes = np.array(["a", "b", "c"])
    train.write_text("\n".join(f"{classes[c]},Firefox,Action" for c in y) + "\n", encoding="utf-8")
    test.write_text("Firefox,Action\n" * 4, encoding="utf-8")
    monkeypatch.setattr(transfer, "TRAIN_CSV", train)
    monkeypatch.setattr(transfer, "TEST_CSV", test)
    location = tmp_path / "bundle"
    location.mkdir()
    tr, va, _ = transfer.subset_indices(y, splits)["outer1_inner1"]
    probabilities = {"model": np.full((len(va), 3), 1 / 3, dtype=np.float32)}
    path = location / "outer1_inner1_classical.pkl"
    path.write_bytes(pickle.dumps(probabilities))
    metadata = {"format_version": 1,
                "data_hashes": {p.name: transfer.portable_file_hash(p) for p in (train, test)},
                "classical_contract": transfer.classical_contract(), "classes": classes.tolist(),
                "model_names": ["model"], "items": [{"tag": "outer1_inner1", "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "train_indices": tr.tolist(), "eval_indices": va.tolist()}]}
    (location / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return location, tmp_path / "cache", y, classes


def test_transfer_validates_and_imports_once(bundle):
    location, cache, y, classes = bundle
    assert transfer.import_classical_bundle(location, cache, y, classes, ["model"], splits) == ["outer1_inner1"]
    assert transfer.import_classical_bundle(location, cache, y, classes, ["model"], splits) == []
    assert (cache / "outer1_inner1_classical_origin.json").exists()


def test_transfer_rejects_corrupt_payload_before_deserialization(bundle):
    location, cache, y, classes = bundle
    with (location / "outer1_inner1_classical.pkl").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        transfer.import_classical_bundle(location, cache, y, classes, ["model"], splits)


def test_transfer_rejects_class_order_and_fold_changes(bundle):
    location, cache, y, classes = bundle
    with pytest.raises(ValueError, match="class columns"):
        transfer.import_classical_bundle(location, cache, y, classes[::-1], ["model"], splits)
    path = location / "manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["items"][0]["train_indices"].reverse()
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fold indices"):
        transfer.import_classical_bundle(location, cache, y, classes, ["model"], splits)


def test_portable_text_hash_ignores_checkout_line_endings(tmp_path):
    first, second = tmp_path / "lf.py", tmp_path / "crlf.py"
    first.write_bytes(b"x = 1\ny = 2\n")
    second.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert transfer.portable_file_hash(first) == transfer.portable_file_hash(second)
