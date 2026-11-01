"""Strict nested evaluation, train-only fusion selection, and final reporting.

Usage: python -m src.dl_hybrid_pipeline [--device cpu|cuda] [--threads 2]
All 4 outer and 4 inner splits are fixed at seed 42. Rare classes have only
3 examples inside outer training: inner validation support is consequently
incomplete, but every inner TRAIN subset still contains all 247 classes.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
import xgboost
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, SUBMISSIONS_DIR, TABLES_DIR, TEST_CSV, TRAIN_CSV
from src.config import PROJECT_ROOT
from src.conv_bigru_attention import (SequencePreprocessor, file_hash, predict, read_traces,
                                      train_network)
from src.features import build_behavioral_dense, build_sequential_nlp
from src.dl_hybrid_transfer import import_classical_bundle, portable_file_hash
from src.parser import load_dataset
from src.postprocess import apply_browser_mask

MODEL_NAMES = ("ridge_hyb", "svm_hyb", "svm_seq", "logreg_hyb", "mlp_hyb", "xgb_seq", "xgb_behav")
REFERENCE_FILES = {"reference_0.974": "submission_ensemble_optimized.csv",
                   "historical_dynamic_stacking": "submission_dynamic_stacking.csv"}


def dump(obj, path):
    """Atomic resumable writes; never replace a complete cache with a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def metrics(y, p, n_classes):
    return {"macro_f1": float(f1_score(y, p.argmax(1), labels=np.arange(n_classes), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, p.argmax(1)))}


def splits(y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        return list(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def browser_mapping(sessions):
    result = {}
    for s in sessions:
        if s["user"] in result and result[s["user"]] != s["browser"]:
            raise ValueError("Browser exclusivity assumption is contradicted in training")
        result[s["user"]] = s["browser"]
    return result


def fold_features(train_sessions, evaluation_sessions):
    # All vocabularies, IDFs, Markov action lists, relative-frequency columns,
    # dense centering/scaling and feature selection start INSIDE this fit.
    seq_train, seq_eval, _ = build_sequential_nlp(train_sessions, evaluation_sessions)
    beh_train, beh_eval, names = build_behavioral_dense(train_sessions, evaluation_sessions)
    scaler = StandardScaler().fit(beh_train)
    beh_train = scaler.transform(beh_train).astype(np.float32)
    beh_eval = scaler.transform(beh_eval).astype(np.float32)
    hybrid_train = sp.hstack([seq_train, .25 * sp.csr_matrix(normalize(beh_train))], format="csr")
    hybrid_eval = sp.hstack([seq_eval, .25 * sp.csr_matrix(normalize(beh_eval))], format="csr")
    return seq_train, seq_eval, beh_train, beh_eval, hybrid_train, hybrid_eval, names, scaler


def classical_predictions(features, y, n_classes, seed=42, threads=2):
    st, se, bt, be, ht, he, _, _ = features
    results = {}
    specifications = [
        ("ridge_hyb", RidgeClassifier(alpha=1., class_weight="balanced"), ht, he, 2.),
        ("svm_hyb", LinearSVC(C=.5, class_weight="balanced", random_state=seed, max_iter=4000), ht, he, 2.5),
        ("svm_seq", LinearSVC(C=.5, class_weight="balanced", random_state=seed, max_iter=4000), st, se, 2.5),
        ("logreg_hyb", LogisticRegression(C=20., class_weight="balanced", max_iter=300, random_state=seed), ht, he, None),
    ]
    with threadpool_limits(limits=threads):
        for name, estimator, xt, xe, inverse_temperature in specifications:
            start = time.monotonic()
            estimator.fit(xt, y)
            values = (softmax(estimator.decision_function(xe) * inverse_temperature, axis=1)
                      if inverse_temperature else estimator.predict_proba(xe))
            aligned = np.zeros((len(xe) if not sp.issparse(xe) else xe.shape[0], n_classes), np.float32)
            aligned[:, estimator.classes_.astype(int)] = values
            results[name] = aligned
            print(f"    classical {name}: {time.monotonic()-start:.1f}s", flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            selector = SelectKBest(f_classif, k=min(6000, ht.shape[1])).fit(ht, y)
        mlp = MLPClassifier(hidden_layer_sizes=(384, 192), alpha=5e-3,
                            max_iter=45, early_stopping=False, random_state=seed)
        start = time.monotonic()
        mlp.fit(selector.transform(ht), y)
        results["mlp_hyb"] = mlp.predict_proba(selector.transform(he)).astype(np.float32)
        print(f"    classical mlp_hyb: {time.monotonic()-start:.1f}s", flush=True)
        svd = TruncatedSVD(n_components=min(80, st.shape[1] - 1, st.shape[0] - 1), random_state=seed)
        svd_train, svd_eval = svd.fit_transform(st), None
        svd_eval = svd.transform(se)
        for name, xt, xe in [("xgb_seq", svd_train, svd_eval), ("xgb_behav", bt, be)]:
            start = time.monotonic()
            estimator = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=.1,
                                      subsample=.8, colsample_bytree=.5, tree_method="hist",
                                      eval_metric="mlogloss", n_jobs=threads, random_state=seed)
            estimator.fit(xt, y)
            results[name] = estimator.predict_proba(xe).astype(np.float32)
            print(f"    classical {name}: {time.monotonic()-start:.1f}s", flush=True)
    return results


def normalize_rows(p):
    return p / np.maximum(p.sum(1, keepdims=True), 1e-30)


def blend(probabilities, weights, geometric):
    arrays = np.stack([probabilities[name] for name in weights])
    w = np.asarray(list(weights.values()), dtype=np.float64)
    w /= w.sum()
    arithmetic = np.einsum("m,mnc->nc", w, arrays)
    geometric_probs = softmax(np.einsum("m,mnc->nc", w, np.log(np.maximum(arrays, 1e-30))), axis=1)
    return normalize_rows((1 - geometric) * arithmetic + geometric * geometric_probs)


def prior_adjust(p, counts, tau, browsers, classes, mapping):
    # Bayesian logit adjustment: log p(c|x) - tau*log empirical train prior(c).
    # Structural browser zeros are re-applied after adjustment.
    prior = np.asarray(counts, dtype=float) + 1.0
    prior /= prior.sum()
    adjusted = softmax(np.log(np.maximum(p, 1e-30)) - tau * np.log(prior)[None, :], axis=1)
    return apply_browser_mask(adjusted, browsers, classes, mapping)


def candidate_weights(with_dl):
    # Small fixed search space, established before any external-fold outcomes.
    classical = [
        {"ridge_hyb": 1.},
        {"ridge_hyb": .8, "svm_hyb": .1, "svm_seq": .1},
        {"ridge_hyb": .6, "svm_hyb": .2, "mlp_hyb": .15, "xgb_behav": .05},
        {"ridge_hyb": .5, "svm_hyb": .2, "logreg_hyb": .1, "mlp_hyb": .1, "xgb_behav": .1},
    ]
    for base in classical:
        for weight in ([0., .05, .10, .20, .35] if with_dl else [0.]):
            weights = {k: v * (1 - weight) for k, v in base.items()}
            if weight:
                weights["dl"] = weight
            yield weights


def select_fusion(probabilities, y, browsers, classes, mapping, with_dl):
    counts = np.bincount(y, minlength=len(classes))
    best_score, best = -1., None
    for weights in candidate_weights(with_dl):
        for geometric in (0., .5, 1.):
            base = blend(probabilities, weights, geometric)
            for tau in (0., .15, .30, .50):
                p = prior_adjust(base, counts, tau, browsers, classes, mapping)
                score = metrics(y, p, len(classes))["macro_f1"]
                # Deterministic ties prefer the first (simpler, no-DL) candidate.
                if score > best_score + 1e-12:
                    best_score = score
                    best = {"weights": weights, "geometric": geometric, "tau": tau, "inner_macro_f1": score}
    return best


def apply_fusion(probabilities, selection, counts, browsers, classes, mapping):
    return prior_adjust(blend(probabilities, selection["weights"], selection["geometric"]),
                        counts, selection["tau"], browsers, classes, mapping)


def evaluate_subset(train_idx, eval_idx, train_sessions, test_sessions, traces, test_traces,
                    y, classes, cache, args, tag, chosen_epochs=None, outer=False):
    path = cache / f"{tag}.pkl"
    if path.exists():
        payload = load(path)
        if not np.array_equal(payload["train_indices"], train_idx) or not np.array_equal(payload["eval_indices"], eval_idx):
            raise ValueError(f"Cache indices differ: {path}")
        print(f"  resume {tag}", flush=True)
        return payload
    train = [train_sessions[i] for i in train_idx]
    evaluation = [train_sessions[i] for i in eval_idx] + (test_sessions if outer else [])
    raw_train = [traces[i] for i in train_idx]
    raw_eval = [traces[i] for i in eval_idx] + (test_traces if outer else [])
    mapping = browser_mapping(train)
    features = fold_features(train, evaluation)
    classical_path = cache / f"{tag}_classical.pkl"
    if classical_path.exists():
        probabilities = load(classical_path)
        print(f"  resume classical {tag}", flush=True)
    else:
        probabilities = classical_predictions(features, y[train_idx], len(classes), threads=args.threads)
        dump(probabilities, classical_path)
    preprocessing = SequencePreprocessor().fit(raw_train)
    train_arrays = (*preprocessing.transform(raw_train), features[2])
    eval_arrays = (*preprocessing.transform(raw_eval), features[3])
    nval = len(eval_idx)
    eval_browsers = [s["browser"] for s in evaluation]
    print(f"  {tag}: train={len(train_idx)}, validation={nval}, tokens={len(preprocessing.token_vocab)+2}", flush=True)
    # Validation only, never the test block, participates in epoch curves.
    model, history = train_network(
        train_arrays, y[train_idx], tuple(a[:nval] for a in eval_arrays), y[eval_idx],
        classes, eval_browsers[:nval], mapping, len(preprocessing.token_vocab) + 2,
        len(preprocessing.ctrl_vocab) + 2, args.epochs, args.batch_size, args.device,
        seed=42, chosen_epochs=chosen_epochs, diagnostic_best=outer,
        checkpoint_path=cache / f"{tag}_training.pt")
    probabilities["dl"] = predict(model, eval_arrays, args.device, args.batch_size)
    probabilities = {name: apply_browser_mask(p, eval_browsers, classes, mapping).astype(np.float32)
                     for name, p in probabilities.items()}
    diagnostic = history.pop("diagnostic_probs")
    if diagnostic is not None:
        diagnostic = apply_browser_mask(diagnostic, eval_browsers[:nval], classes, mapping)
    payload = {"probabilities": probabilities, "train_indices": train_idx, "eval_indices": eval_idx,
               "history": history, "diagnostic_probs": diagnostic}
    # Retain a reviewable inference bundle for each outer model, with all its
    # fold-local transforms and class-column ordering, not just probabilities.
    if outer:
        dump({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
              "sequence_preprocessor": preprocessing, "behavior_scaler": features[7],
              "behavior_feature_names": features[6], "classes": classes,
              "train_indices": train_idx, "chosen_epochs": chosen_epochs}, cache / f"{tag}_model.pkl")
    dump(payload, path)
    return payload


def bootstrap_delta(y, before, after, n_classes, iterations=2000):
    """Paired, class-stratified bootstrap, conditional on these fitted OOF models."""
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(y == c) for c in range(n_classes)]
    truth_counts = np.bincount(y, minlength=n_classes)
    estimates = []
    for _ in range(iterations):
        rows = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values = []
        for predictions in (before, after):
            pred = predictions[rows]
            tp = np.bincount(y[rows][pred == y[rows]], minlength=n_classes)
            predicted_counts = np.bincount(pred, minlength=n_classes)
            values.append(float(np.mean(2 * tp / np.maximum(truth_counts + predicted_counts, 1))))
        estimates.append(values[1] - values[0])
    return np.percentile(estimates, [2.5, 97.5]).tolist()


def write_report(y, classes, train_sessions, test_sessions, oof, test_probs, before, after,
                 before_test, after_test, diagnostics, fold_records, fingerprint, args):
    report_dir = TABLES_DIR if args.epochs == 28 and args.batch_size == 64 else TABLES_DIR / "dl_hybrid_smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(classes)
    rows = [{"model": name, **metrics(y, p, n_classes)} for name, p in oof.items()]
    rows += [{"model": "ensemble_without_dl", **metrics(y, before, n_classes)},
             {"model": "ensemble_with_dl", **metrics(y, after, n_classes)},
             {"model": "dl_outer_best3_diagnostic_BIASED", **metrics(y, diagnostics, n_classes)}]
    scores = pd.DataFrame(rows)
    scores.to_csv(report_dir / "dl_hybrid_scores.csv", index=False)
    truth = y
    dl_predictions = oof["dl"].argmax(1)
    diversity = []
    for name in MODEL_NAMES:
        pred = oof[name].argmax(1)
        diversity.append({"model": name, "disagreement_rate": float(np.mean(pred != dl_predictions)),
                          "dl_correct_other_wrong": int(np.sum((dl_predictions == truth) & (pred != truth))),
                          "other_correct_dl_wrong": int(np.sum((pred == truth) & (dl_predictions != truth))),
                          "both_wrong": int(np.sum((pred != truth) & (dl_predictions != truth)))})
    pd.DataFrame(diversity).to_csv(report_dir / "dl_hybrid_diversity.csv", index=False)
    sample = pd.read_csv(TEST_CSV.parent / "sample_submission.csv")
    expected_ids = np.arange(1, len(test_sessions) + 1)
    if len(sample) != len(test_sessions) or not np.array_equal(sample.RowId, expected_ids):
        raise ValueError("Sample submission RowId order differs")
    final = classes[after_test.argmax(1)]
    submission = pd.DataFrame({"RowId": sample.RowId, "prediction": final})
    mapping = browser_mapping(train_sessions)
    violations = sum(mapping[p] != s["browser"] for p, s in zip(final, test_sessions))
    if len(submission) != 324 or submission.isna().any().any() or violations:
        raise ValueError("Submission shape, NaN or browser invariant failed")
    if args.epochs != 28 or args.batch_size != 64:
        submission_path = SUBMISSIONS_DIR / "submission_dl_hybrid_SMOKE_ONLY.csv"
    else:
        submission_path = SUBMISSIONS_DIR / "submission_dl_hybrid_final.csv"
    submission.to_csv(submission_path, index=False)
    concordance_predictions = {"ensemble_without_dl": classes[before_test.argmax(1)],
                               "ensemble_with_dl": final, "dl": classes[test_probs["dl"].argmax(1)]}
    references = {}
    reference_files = dict(REFERENCE_FILES)
    if args.reference_0980:
        reference_files["reference_0.980"] = args.reference_0980
    for label, filename in reference_files.items():
        path = SUBMISSIONS_DIR / filename
        if path.exists():
            reference = pd.read_csv(path)
            if reference.RowId.duplicated().any() or set(reference.RowId) != set(expected_ids):
                raise ValueError(f"Invalid reference IDs: {path}")
            concordance_predictions[label] = reference.set_index("RowId").loc[expected_ids, "prediction"].to_numpy()
            references[label] = {"file": filename, "sha256": file_hash(path)}
    concordance = pd.DataFrame({a: {b: int(np.sum(pa == pb)) for b, pb in concordance_predictions.items()}
                                for a, pa in concordance_predictions.items()})
    concordance.to_csv(report_dir / "dl_hybrid_concordance.csv")
    columns = ["RowId", "browser", "before_top1", "before_p1", "before_top2", "before_p2",
               "after_top1", "after_p1", "after_top2", "after_p2", "dl_top1", "dl_p1", "dl_top2", "dl_p2",
               "dl_agrees_with_final"]
    arbitrations = []
    for i in np.flatnonzero(before_test.argmax(1) != after_test.argmax(1)):
        record = {"RowId": int(expected_ids[i]), "browser": test_sessions[i]["browser"]}
        for label, p in [("before", before_test), ("after", after_test), ("dl", test_probs["dl"])]:
            a, b = np.argsort(p[i])[-2:][::-1]
            record.update({f"{label}_top1": classes[a], f"{label}_p1": float(p[i, a]),
                           f"{label}_top2": classes[b], f"{label}_p2": float(p[i, b])})
        record["dl_agrees_with_final"] = record["dl_top1"] == record["after_top1"]
        arbitrations.append(record)
    pd.DataFrame(arbitrations, columns=columns).to_csv(report_dir / "dl_hybrid_arbitrations.csv", index=False)
    interval = bootstrap_delta(y, before.argmax(1), after.argmax(1), n_classes)
    delta = metrics(y, after, n_classes)["macro_f1"] - metrics(y, before, n_classes)["macro_f1"]
    report = {"fingerprint": fingerprint, "epochs": args.epochs, "scores": rows, "outer_folds": fold_records,
              "delta_macro_f1": delta, "paired_class_stratified_bootstrap_95_ci": interval,
              "unique_test_classes": len(set(final)), "browser_violations": violations,
              "n_arbitrations": len(arbitrations), "reference_files": references,
              "reference_0.980_status": ("File explicitly supplied by user." if args.reference_0980 else
                                         "Missing: no repository evidence identifies the exact reference file."),
              "conclusion": ("Positive OOF gain with conditional bootstrap interval above zero; independent replication still required."
                             if delta > 0 and interval[0] > 0 else
                             "No statistically established improvement from DL on this nested OOF experiment."),
              "limitations": ["Four fixed outer folds; no claim of proving absence of overfitting.",
                              "Bootstrap conditional on fitted OOF predictions; it does not include training variability.",
                              "Inner 4-fold validation has missing rare-class support (minimum outer-train count=3).",
                              "Best-three checkpoints chosen on outer labels are reported ONLY as a biased diagnostic.",
                              "0.974/0.980 reference submissions are comparison artifacts, never training inputs."]}
    (report_dir / "dl_hybrid_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text = ["# ConvBiGRUAttention — résultat reproductible", "", scores.to_string(index=False), "",
            f"Delta Macro-F1 imbriqué : {delta:+.6f}; IC bootstrap conditionnel 95 % : {interval}.",
            report["conclusion"], "", f"Soumission : {submission_path.name}; classes uniques : {len(set(final))}; violations navigateur : {violations}.",
            "", "Concordance (nombre de lignes identiques / 324) :", concordance.to_string(), "",
            report["reference_0.980_status"],
            "", "Arbitrages : dl_hybrid_arbitrations.csv (toutes les lignes modifiées, Top-1/Top-2 avant/après et DL).",
            "", "Limites :", *[f"- {item}" for item in report["limitations"]]]
    (report_dir / "dl_hybrid_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n" + "\n".join(text[2:]), flush=True)
    return report


def run_outer_fold(fold, train_idx, val_idx, cache, args):
    train_sessions, test_sessions = load_dataset(TRAIN_CSV), load_dataset(TEST_CSV, False)
    traces, test_traces = read_traces(TRAIN_CSV, True), read_traces(TEST_CSV, False)
    classes, y = np.unique([s["user"] for s in train_sessions], return_inverse=True)
    print(f"\nOUTER FOLD {fold}/4", flush=True)
    inner_oof = {name: np.zeros((len(train_idx), len(classes)), np.float32) for name in (*MODEL_NAMES, "dl")}
    curves, inner_splits = [], splits(y[train_idx])
    for inner, (relative_train, relative_val) in enumerate(inner_splits, 1):
        payload = evaluate_subset(train_idx[relative_train], train_idx[relative_val], train_sessions,
                                  test_sessions, traces, test_traces, y, classes, cache, args,
                                  f"outer{fold}_inner{inner}")
        for name in (*MODEL_NAMES, "dl"):
            inner_oof[name][relative_val] = payload["probabilities"][name]
        curves.append(payload["history"]["curve"])
    # Top-3 average checkpoints determined entirely inside outer training.
    mean_curve = np.mean(curves, axis=0)
    chosen_epochs = sorted((np.argsort(-mean_curve, kind="stable")[:min(3, args.epochs)] + 1).tolist())
    mapping = browser_mapping([train_sessions[i] for i in train_idx])
    train_browsers = [train_sessions[i]["browser"] for i in train_idx]
    val_browsers = [train_sessions[i]["browser"] for i in val_idx]
    test_browsers = [s["browser"] for s in test_sessions]
    selections = [select_fusion(inner_oof, y[train_idx], train_browsers, classes, mapping, with_dl)
                  for with_dl in (False, True)]
    payload = evaluate_subset(train_idx, val_idx, train_sessions, test_sessions, traces, test_traces,
                              y, classes, cache, args, f"outer{fold}", chosen_epochs, outer=True)
    pv = {name: p[:len(val_idx)] for name, p in payload["probabilities"].items()}
    pt = {name: p[len(val_idx):] for name, p in payload["probabilities"].items()}
    counts = np.bincount(y[train_idx], minlength=len(classes))
    bv, av = [apply_fusion(pv, selection, counts, val_browsers, classes, mapping) for selection in selections]
    bt, at = [apply_fusion(pt, selection, counts, test_browsers, classes, mapping) for selection in selections]
    record = {"fold": fold, "chosen_epochs": chosen_epochs, "without_dl": metrics(y[val_idx], bv, len(classes)),
              "with_dl": metrics(y[val_idx], av, len(classes)), "selections": selections,
              "missing_classes_per_inner_validation": [int(len(classes) - len(np.unique(y[train_idx][v]))) for _, v in inner_splits]}
    print(json.dumps(record), flush=True)
    return {"fold": fold, "val_idx": val_idx, "pv": pv, "pt": pt, "bv": bv, "av": av,
            "bt": bt, "at": at, "diagnostic": payload["diagnostic_probs"], "record": record}


def outer_worker(fold, train_idx, val_idx, cache, args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with (cache / f"outer{fold}.log").open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            return run_outer_fold(fold, train_idx, val_idx, cache, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent processes for the four outer folds")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=28, help="Other values produce separately named SMOKE artifacts only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference-0980", help="Exact reference filename, when identified; comparison only")
    parser.add_argument("--transfer-dir", type=Path,
                        default=PROJECT_ROOT / "artifacts/dl_hybrid_transfer",
                        help="Validated classical probability bundle from the stopped CPU run")
    parser.add_argument("--check-transfer", action="store_true",
                        help="Validate/import the bundle and exit without training")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.epochs < 1 or args.batch_size < 2 or args.threads < 1:
        parser.error("workers must be 1..4; epochs/threads positive; batch-size >= 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable: install a CUDA-enabled PyTorch build on a supported GPU PC")
    if args.device.startswith("cuda") and args.workers != 1:
        parser.error("Use --workers 1 with CUDA so the four folds share one GPU sequentially")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Identity is portable across CPU/GPU, thread counts and checkout line endings.
    # Runtime package/device versions remain recorded separately as provenance.
    source_files = [Path(__file__), Path(__file__).with_name("conv_bigru_attention.py"),
                    Path(__file__).with_name("features.py"), Path(__file__).with_name("parser.py"),
                    Path(__file__).with_name("postprocess.py"), Path(__file__).with_name("config.py"),
                    Path(__file__).with_name("dl_hybrid_transfer.py")]
    protocol = {"files": {p.name: portable_file_hash(p) for p in [TRAIN_CSV, TEST_CSV, *source_files]},
                "epochs": args.epochs, "batch_size": args.batch_size, "seed": 42, "format_version": 2}
    runtime = {"device": args.device, "threads": args.threads, "workers": args.workers,
               "torch": str(torch.__version__), "sklearn": sklearn.__version__,
               "numpy": np.__version__, "xgboost": xgboost.__version__,
               "python": platform.python_version(), "platform": platform.platform(),
               "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest = {"protocol": protocol, "initial_runtime": runtime}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    cache = PROCESSED_DIR / "dl_hybrid_runs" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            raise ValueError("Existing run manifest does not match this protocol")
    else:
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (cache / "runtime_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runtime) + "\n")
    print(f"Run {fingerprint[:16]} device={args.device} epochs={args.epochs}; cache={cache}", flush=True)
    test_sessions = load_dataset(TEST_CSV, False)
    with TRAIN_CSV.open(encoding="utf-8") as stream:
        classes, y = np.unique([line.split(",", 1)[0] for line in stream], return_inverse=True)
    if (len(y), len(test_sessions), len(classes)) != (3279, 324, 247):
        raise ValueError("Unexpected competition data dimensions")
    if args.transfer_dir.exists():
        imported = import_classical_bundle(args.transfer_dir, cache, y, classes, MODEL_NAMES, splits)
        print(f"Validated transfer bundle; imported classical caches: {imported}", flush=True)
    elif args.check_transfer:
        parser.error(f"Transfer directory missing: {args.transfer_dir}")
    if args.check_transfer:
        print("Transfer checks passed; no training started.", flush=True)
        return
    oof = {name: np.zeros((len(y), len(classes)), np.float32) for name in (*MODEL_NAMES, "dl")}
    test_probs = {name: np.zeros((len(test_sessions), len(classes)), np.float32) for name in oof}
    before, after = np.zeros_like(oof["dl"]), np.zeros_like(oof["dl"])
    before_test, after_test = np.zeros_like(test_probs["dl"]), np.zeros_like(test_probs["dl"])
    diagnostics = np.zeros_like(before)
    fold_records, split_ids = [], np.full(len(y), -1, dtype=int)
    jobs = [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(splits(y), 1)]
    if args.workers == 1:
        completed = [run_outer_fold(fold, tr, va, cache, args) for fold, tr, va in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(outer_worker, fold, tr, va, cache, args) for fold, tr, va in jobs]
            completed = []
            for future in as_completed(futures):
                result = future.result()
                print(f"Outer fold {result['fold']} complete: {result['record']}", flush=True)
                completed.append(result)
    for result in sorted(completed, key=lambda item: item["fold"]):
        fold, val_idx = result["fold"], result["val_idx"]
        for name in oof:
            oof[name][val_idx] = result["pv"][name]
            test_probs[name] += result["pt"][name] / 4
        before[val_idx], after[val_idx] = result["bv"], result["av"]
        before_test += result["bt"] / 4
        after_test += result["at"] / 4
        diagnostics[val_idx] = result["diagnostic"]
        split_ids[val_idx] = fold - 1
        fold_records.append(result["record"])
    for p in [*oof.values(), *test_probs.values(), before, after, before_test, after_test]:
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid probability matrix")
    result = {"oof_probs": oof["dl"], "test_probs": test_probs["dl"], "classes": classes,
              "fold_ids": split_ids, "fingerprint": fingerprint, "manifest": manifest,
              "outer_best3_diagnostic_probs": diagnostics,
              "selection_protocol": "outer epochs chosen on inner folds only; outer-best diagnostic excluded from fusion"}
    filename = "conv_bigru_attn_probs.pkl" if args.epochs == 28 and args.batch_size == 64 else "conv_bigru_attn_SMOKE_probs.pkl"
    dump(result, PROCESSED_DIR / filename)
    dump({"oof": oof, "test": test_probs, "before_oof": before, "after_oof": after,
          "before_test": before_test, "after_test": after_test, "folds": fold_records,
          "classes": classes, "fingerprint": fingerprint}, cache / "nested_results.pkl")
    write_report(y, classes, load_dataset(TRAIN_CSV), test_sessions, oof, test_probs, before, after,
                 before_test, after_test, diagnostics, fold_records, fingerprint, args)


if __name__ == "__main__":
    main()
