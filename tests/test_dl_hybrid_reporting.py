"""Final artifact contract, exercised only in pytest's temporary directory."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import dl_hybrid_pipeline as pipeline


def test_submission_reference_alignment_and_arbitration_export(tmp_path, monkeypatch):
    tables, submissions, raw = (tmp_path / name for name in ("tables", "submissions", "raw"))
    submissions.mkdir()
    raw.mkdir()
    ids = np.arange(1, 325)
    pd.DataFrame({"RowId": ids, "prediction": "placeholder"}).to_csv(raw / "sample_submission.csv", index=False)
    monkeypatch.setattr(pipeline, "TABLES_DIR", tables)
    monkeypatch.setattr(pipeline, "SUBMISSIONS_DIR", submissions)
    monkeypatch.setattr(pipeline, "TEST_CSV", raw / "test.csv")
    monkeypatch.setattr(pipeline, "REFERENCE_FILES", {"reference_0.974": "reference0974.csv"})
    monkeypatch.setattr(pipeline, "bootstrap_delta", lambda *args: [-.01, .02])
    classes = np.array(["a", "b", "c"])
    y = np.tile(np.arange(3), 3)
    oof_probability = np.eye(3)[y] * .8 + .2 / 3
    before_test = np.tile([.8, .1, .1], (324, 1))
    after_test = before_test.copy()
    after_test[3] = [.1, .8, .1]
    final = classes[after_test.argmax(1)]
    # Reversed physical order must be aligned by RowId rather than CSV position.
    reference = pd.DataFrame({"RowId": ids, "prediction": final}).iloc[::-1]
    for filename in ("reference0974.csv", "submission_hada_jehd.csv"):
        reference.to_csv(submissions / filename, index=False)
    oof = {name: oof_probability.copy() for name in (*pipeline.MODEL_NAMES, "dl")}
    test = {name: after_test.copy() for name in oof}
    train_sessions = [{"user": user, "browser": "Firefox"} for user in classes]
    test_sessions = [{"browser": "Firefox"} for _ in ids]
    report = pipeline.write_report(
        y, classes, train_sessions, test_sessions, oof, test, oof_probability, oof_probability,
        before_test, after_test, oof_probability, [], "test-only", SimpleNamespace(
            epochs=28, batch_size=64, reference_0980="submission_hada_jehd.csv"))
    submission = pd.read_csv(submissions / "submission_dl_hybrid_final.csv")
    assert submission.columns.tolist() == ["RowId", "prediction"]
    assert len(submission) == 324 and not submission.isna().any().any()
    np.testing.assert_array_equal(submission.RowId, ids)
    np.testing.assert_array_equal(submission.prediction, final)
    concordance = pd.read_csv(tables / "dl_hybrid_concordance.csv", index_col=0)
    assert concordance.loc["reference_0.980", "ensemble_with_dl"] == 324
    arbitrations = pd.read_csv(tables / "dl_hybrid_arbitrations.csv")
    assert arbitrations.RowId.tolist() == [4]
    assert arbitrations.before_top1.tolist() == ["a"]
    assert arbitrations.after_top1.tolist() == ["b"]
    assert arbitrations.dl_agrees_with_final.tolist() == [True]
    assert report["browser_violations"] == 0
    assert report["unique_test_classes"] == 2


def test_paired_bootstrap_of_identical_predictions_is_zero():
    y = np.repeat(np.arange(3), 4)
    np.testing.assert_array_equal(pipeline.bootstrap_delta(y, y, y, 3, iterations=30), [0., 0.])
