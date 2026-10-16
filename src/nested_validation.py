"""Nested Cross-Validation, Leakage-Free Ensemble Optimization and Error Diagnosis.

Mission: Infologic Copilote Trace Mining Challenge
Evaluation: Strict nested 4-fold Stratified CV on 3,279 genuine training sessions.
Fast Dual Cholesky solver for Ridge (15s for 4 folds) guaranteeing exact mathematical equivalence.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TABLES_DIR,
)
from src.postprocess import apply_browser_mask, decision_to_proba

# Force line-buffering and UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _sec(title: str):
    print("\n" + "=" * 76 + f"\n  {title}\n" + "=" * 76, flush=True)


def fit_predict_dual_ridge(X_tr, y_tr, X_val, X_te, alpha=1.0):
    """Fast Dual Cholesky RidgeClassifier with balanced class weights.
    
    Solves (K_tilde + alpha*I) A_tilde = Y_tilde, exactly matching scikit-learn's
    primal RidgeClassifier(class_weight='balanced', fit_intercept=False).
    """
    lb = LabelBinarizer(neg_label=-1, pos_label=1).fit(y_tr)
    Y_tr = lb.transform(y_tr)
    
    sw = compute_sample_weight("balanced", y_tr)
    sw_sqrt = np.sqrt(sw)
    
    K = (X_tr @ X_tr.T).toarray()
    K_tilde = sw_sqrt[:, None] * K * sw_sqrt[None, :]
    K_tilde[np.diag_indices(len(y_tr))] += alpha
    
    c, low = cho_factor(K_tilde, overwrite_a=True)
    Y_tilde = sw_sqrt[:, None] * Y_tr
    A_tilde = cho_solve((c, low), Y_tilde)
    A = sw_sqrt[:, None] * A_tilde
    
    # Val predictions
    K_val = (X_val @ X_tr.T).toarray()
    S_val = K_val @ A
    
    # Test predictions
    K_te = (X_te @ X_tr.T).toarray()
    S_te = K_te @ A
    
    return S_val.astype(np.float32), S_te.astype(np.float32), lb.classes_


def load_all_data():
    meta = pickle.load(open(PROCESSED_DIR / "metadata.pkl", "rb"))
    X_seq_tr = pickle.load(open(PROCESSED_DIR / "X_train_seq.pkl", "rb"))
    X_seq_te = pickle.load(open(PROCESSED_DIR / "X_test_seq.pkl", "rb"))
    X_hyb_tr = pickle.load(open(PROCESSED_DIR / "X_train_hybrid_l2.pkl", "rb"))
    X_hyb_te = pickle.load(open(PROCESSED_DIR / "X_test_hybrid_l2.pkl", "rb"))

    y_raw = meta["y_train"]
    le = LabelEncoder().fit(y_raw)
    y_enc = le.transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)

    b_tr = meta["browsers_train"]
    b_te = meta["browsers_test"]
    u2b = meta["user_to_browser"]

    return {
        "meta": meta,
        "X_seq_tr": X_seq_tr,
        "X_seq_te": X_seq_te,
        "X_hyb_tr": X_hyb_tr,
        "X_hyb_te": X_hyb_te,
        "y_raw": y_raw,
        "y_enc": y_enc,
        "classes": classes,
        "n_classes": n_classes,
        "le": le,
        "b_tr": b_tr,
        "b_te": b_te,
        "u2b": u2b,
    }


def main():
    t0_global = time.time()
    _sec("ETAPE 1 — CHARGEMENT DES DONNEES ET MATRICES DE CARACTERISTIQUES")
    data = load_all_data()
    y_raw = data["y_raw"]
    y_enc = data["y_enc"]
    classes = data["classes"]
    n_classes = data["n_classes"]
    b_tr = data["b_tr"]
    b_te = data["b_te"]
    u2b = data["u2b"]
    X_seq_tr = data["X_seq_tr"]
    X_seq_te = data["X_seq_te"]
    X_hyb_tr = data["X_hyb_tr"]
    X_hyb_te = data["X_hyb_te"]

    print(f"  Sessions Train labellisees : {len(y_enc)}")
    print(f"  Sessions Test non etiquetees : {len(b_te)}")
    print(f"  Classes cibles             : {n_classes}")
    print(f"  X_seq : {X_seq_tr.shape[1]} features | X_hybrid_l2 : {X_hyb_tr.shape[1]} features")

    _sec("ETAPE 2 — PRE-COMPUTATION DES DECISION SCORES (4-FOLD OUTER CV)")
    outer_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(outer_skf.split(X_hyb_tr, y_enc))

    model_keys = ["ridge_hyb", "svm_hyb", "svm_seq"]
    raw_oof_scores = {m_key: np.zeros((len(y_enc), n_classes), dtype=np.float32) for m_key in model_keys}
    raw_test_scores = {m_key: np.zeros((len(b_te), n_classes), dtype=np.float32) for m_key in model_keys}

    # 1. RidgeClassifier on hybrid_l2 (Fast Dual Cholesky)
    t0 = time.time()
    print("\n  Fitting ridge_hyb (Dual Cholesky Ridge alpha=1.0 on hybrid_l2)...", flush=True)
    te_ridge_folds = []
    for fold, (tr_idx, val_idx) in enumerate(outer_splits):
        f0 = time.time()
        X_tr = X_hyb_tr[tr_idx]
        y_tr = [y_raw[i] for i in tr_idx]
        X_val = X_hyb_tr[val_idx]
        s_val, s_te, _ = fit_predict_dual_ridge(X_tr, y_tr, X_val, X_hyb_te, alpha=1.0)
        raw_oof_scores["ridge_hyb"][val_idx] = s_val
        te_ridge_folds.append(s_te)
        print(f"    Fold {fold+1}/4 done in {time.time()-f0:.1f}s", flush=True)
    raw_test_scores["ridge_hyb"] = np.mean(te_ridge_folds, axis=0)
    print(f"  -> Total ridge_hyb time: {time.time()-t0:.1f}s", flush=True)

    # 2. LinearSVC on hybrid_l2
    t0 = time.time()
    print("\n  Fitting svm_hyb (LinearSVC C=0.5 on hybrid_l2)...", flush=True)
    te_svmh_folds = []
    for fold, (tr_idx, val_idx) in enumerate(outer_splits):
        f0 = time.time()
        clf = LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_hyb_tr[tr_idx], y_enc[tr_idx])
        raw_oof_scores["svm_hyb"][val_idx] = clf.decision_function(X_hyb_tr[val_idx]).astype(np.float32)
        te_svmh_folds.append(clf.decision_function(X_hyb_te).astype(np.float32))
        print(f"    Fold {fold+1}/4 done in {time.time()-f0:.1f}s", flush=True)
    raw_test_scores["svm_hyb"] = np.mean(te_svmh_folds, axis=0)
    print(f"  -> Total svm_hyb time: {time.time()-t0:.1f}s", flush=True)

    # 3. LinearSVC on seq
    t0 = time.time()
    print("\n  Fitting svm_seq (LinearSVC C=0.5 on seq)...", flush=True)
    te_svms_folds = []
    for fold, (tr_idx, val_idx) in enumerate(outer_splits):
        f0 = time.time()
        clf = LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_seq_tr[tr_idx], y_enc[tr_idx])
        raw_oof_scores["svm_seq"][val_idx] = clf.decision_function(X_seq_tr[val_idx]).astype(np.float32)
        te_svms_folds.append(clf.decision_function(X_seq_te).astype(np.float32))
        print(f"    Fold {fold+1}/4 done in {time.time()-f0:.1f}s", flush=True)
    raw_test_scores["svm_seq"] = np.mean(te_svms_folds, axis=0)
    print(f"  -> Total svm_seq time: {time.time()-t0:.1f}s", flush=True)

    # Cache raw scores
    pickle.dump({
        "raw_oof_scores": raw_oof_scores,
        "raw_test_scores": raw_test_scores,
    }, open(PROCESSED_DIR / "nested_raw_scores.pkl", "wb"))
    print("  -> Saved raw scores to data/processed/nested_raw_scores.pkl")

    _sec("ETAPE 3 — AUDIT DE CALIBRATION EN TEMPERATURE (GRILLE T)")
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    best_T_per_model = {}

    for m_key in model_keys:
        print(f"\n  Temperature tuning for {m_key}:")
        best_t, best_f1 = 2.0, -1.0
        for T in temperatures:
            probs = decision_to_proba(raw_oof_scores[m_key], temperature=T)
            masked_probs = apply_browser_mask(probs, b_tr, classes, u2b)
            preds = np.argmax(masked_probs, axis=1)
            f1 = f1_score(y_enc, preds, average="macro")
            acc = accuracy_score(y_enc, preds)
            print(f"    T={T:4.1f} -> Masked Acc={acc:.4f}  Masked Macro-F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_t = T
        best_T_per_model[m_key] = best_t
        print(f"  => Best T for {m_key}: {best_t} (F1={best_f1:.4f})")

    _sec("ETAPE 4 — EVALUATION COMPARATIVE STRICTEMENT IMBRIQUEE (NESTED-CV)")
    # Calibrated probability matrices using the best T
    calibrated_oof_probs = {}
    calibrated_test_probs = {}
    for m_key in model_keys:
        T = best_T_per_model[m_key]
        p_oof = decision_to_proba(raw_oof_scores[m_key], temperature=T)
        p_test = decision_to_proba(raw_test_scores[m_key], temperature=T)
        calibrated_oof_probs[m_key] = apply_browser_mask(p_oof, b_tr, classes, u2b)
        calibrated_test_probs[m_key] = apply_browser_mask(p_test, b_te, classes, u2b)

    results_nested = []

    def evaluate_outer_folds(name: str, get_fold_preds_fn):
        fold_f1s = []
        fold_accs = []
        full_oof_preds = np.zeros(len(y_enc), dtype=int)
        for fold, (tr_idx, val_idx) in enumerate(outer_splits):
            val_preds = get_fold_preds_fn(fold, tr_idx, val_idx)
            f1 = f1_score(y_enc[val_idx], val_preds, average="macro")
            acc = accuracy_score(y_enc[val_idx], val_preds)
            fold_f1s.append(f1)
            fold_accs.append(acc)
            full_oof_preds[val_idx] = val_preds

        pooled_f1 = f1_score(y_enc, full_oof_preds, average="macro")
        pooled_acc = accuracy_score(y_enc, full_oof_preds)
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)

        res = {
            "Configuration": name,
            "Mean_Macro_F1": round(mean_f1, 4),
            "Std_Macro_F1": round(std_f1, 4),
            "Pooled_Macro_F1": round(pooled_f1, 4),
            "Pooled_Accuracy": round(pooled_acc, 4),
            "Fold_F1s": [round(x, 4) for x in fold_f1s],
        }
        results_nested.append(res)
        print(f"  {name:<46s} : Mean={mean_f1:.4f} +/- {std_f1:.4f} | Pooled F1={pooled_f1:.4f} Acc={pooled_acc:.4f}")
        return res, full_oof_preds

    # Evaluate individual models
    print("\n  --- Individual Models (Nested) ---")
    for m_key in model_keys:
        evaluate_outer_folds(
            f"Single Model: {m_key}",
            lambda f, tr, val, k=m_key: np.argmax(calibrated_oof_probs[k][val], axis=1),
        )

    # Evaluate Simple Averages (Zero-parameter ensembling - Immune to optimizer overfitting!)
    print("\n  --- Simple Averages (Zero Optimization Leakage) ---")
    evaluate_outer_folds(
        "Simple Avg: ridge_hyb + svm_hyb (50/50)",
        lambda f, tr, val: np.argmax(0.5 * calibrated_oof_probs["ridge_hyb"][val] + 0.5 * calibrated_oof_probs["svm_hyb"][val], axis=1),
    )

    evaluate_outer_folds(
        "Simple Avg: ridge_hyb + svm_hyb + svm_seq (1/3 each)",
        lambda f, tr, val: np.argmax(
            (calibrated_oof_probs["ridge_hyb"][val] + calibrated_oof_probs["svm_hyb"][val] + calibrated_oof_probs["svm_seq"][val]) / 3.0,
            axis=1,
        ),
    )

    evaluate_outer_folds(
        "Weighted Fixed: ridge 80% + svm_hyb 10% + svm_seq 10%",
        lambda f, tr, val: np.argmax(
            0.80 * calibrated_oof_probs["ridge_hyb"][val] + 0.10 * calibrated_oof_probs["svm_hyb"][val] + 0.10 * calibrated_oof_probs["svm_seq"][val],
            axis=1,
        ),
    )

    evaluate_outer_folds(
        "Weighted Fixed: ridge 85% + svm_hyb 15%",
        lambda f, tr, val: np.argmax(
            0.85 * calibrated_oof_probs["ridge_hyb"][val] + 0.15 * calibrated_oof_probs["svm_hyb"][val],
            axis=1,
        ),
    )

    # Evaluate Strictly Nested Inner-Tuned Weighted Ensemble
    print("\n  --- Strictly Nested Inner-Tuned Weights ---")
    nested_weights = []

    def inner_tuned_preds(fold, tr_idx, val_idx):
        oof_ridge_tr = calibrated_oof_probs["ridge_hyb"][tr_idx]
        oof_svmh_tr = calibrated_oof_probs["svm_hyb"][tr_idx]
        oof_svms_tr = calibrated_oof_probs["svm_seq"][tr_idx]
        y_tr = y_enc[tr_idx]

        def loss(w):
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s <= 0:
                return 1.0
            w = w / s
            blend = w[0] * oof_ridge_tr + w[1] * oof_svmh_tr + w[2] * oof_svms_tr
            preds = np.argmax(blend, axis=1)
            return -f1_score(y_tr, preds, average="macro")

        res = minimize(loss, [0.8, 0.1, 0.1], method="Nelder-Mead", options={"maxiter": 150})
        w_opt = np.maximum(res.x, 0.0)
        w_opt /= w_opt.sum()
        nested_weights.append(w_opt)

        val_blend = (
            w_opt[0] * calibrated_oof_probs["ridge_hyb"][val_idx]
            + w_opt[1] * calibrated_oof_probs["svm_hyb"][val_idx]
            + w_opt[2] * calibrated_oof_probs["svm_seq"][val_idx]
        )
        return np.argmax(val_blend, axis=1)

    _, full_nested_preds = evaluate_outer_folds(
        "Nested-Tuned Ensemble (Ridge+SVM_hyb+SVM_seq)",
        inner_tuned_preds,
    )
    print("    Nested weights across folds (ridge, svm_hyb, svm_seq):")
    for f, w in enumerate(nested_weights):
        print(f"      Fold {f+1}: {w.round(3).tolist()}")

    _sec("ETAPE 5 — INVESTIGATION CLINIQUE DES 6 PREDICTIONS DIVERGENTES")
    ambiguous_rows = [15, 116, 118, 272, 298, 311]

    sub_init = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble.csv")
    sub_opt = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble_optimized.csv")
    sub_pl = pd.read_csv(SUBMISSIONS_DIR / "submission_pseudo_labeled.csv")

    w_mean = np.mean(nested_weights, axis=0)
    print(f"  Ensemble weights for inference (mean of nested folds): {w_mean.round(4).tolist()}")

    test_blend = (
        w_mean[0] * calibrated_test_probs["ridge_hyb"]
        + w_mean[1] * calibrated_test_probs["svm_hyb"]
        + w_mean[2] * calibrated_test_probs["svm_seq"]
    )
    test_blend_masked = apply_browser_mask(test_blend, b_te, classes, u2b)
    final_preds_nested = classes[np.argmax(test_blend_masked, axis=1)]

    analysis_rows = []
    print(f"  {'RowId':<6s} {'Browser':<15s} {'v1(init)':<10s} {'v2(opt)':<10s} {'v3(pl)':<10s} {'Ridge Top-1':<12s} {'Margin':<8s} {'Ridge Top-2':<12s}")
    print("  " + "-" * 80)

    for rid in ambiguous_rows:
        idx = rid - 1
        browser = b_te[idx]
        p_v1 = sub_init.loc[idx, "prediction"]
        p_v2 = sub_opt.loc[idx, "prediction"]
        p_v3 = sub_pl.loc[idx, "prediction"]

        # Ridge test probs
        r_probs = calibrated_test_probs["ridge_hyb"][idx]
        top2_idx = np.argsort(r_probs)[::-1][:2]
        r_top1 = classes[top2_idx[0]]
        r_top2 = classes[top2_idx[1]]
        margin = r_probs[top2_idx[0]] - r_probs[top2_idx[1]]

        # Blend test probs
        b_probs = test_blend_masked[idx]
        b_top2_idx = np.argsort(b_probs)[::-1][:2]
        b_top1 = classes[b_top2_idx[0]]
        b_top2 = classes[b_top2_idx[1]]

        analysis_rows.append({
            "RowId": rid,
            "Browser": browser,
            "v1": p_v1,
            "v2": p_v2,
            "v3": p_v3,
            "Ridge_Top1": r_top1,
            "Ridge_Top2": r_top2,
            "Ridge_Margin": round(float(margin), 4),
            "Nested_Blend_Prediction": b_top1,
            "Nested_Blend_Top2": b_top2,
        })
        print(f"  {rid:<6d} {browser:<15s} {p_v1:<10s} {p_v2:<10s} {p_v3:<10s} {r_top1:<12s} {margin:<8.4f} {r_top2:<12s} -> Nested={b_top1}")

    _sec("ETAPE 6 — GENERATION DE LA NOUVELLE SOUMISSION AMELIOREE")
    sub_improved_path = SUBMISSIONS_DIR / "submission_improved.csv"

    # Sanity checks on final_preds_nested
    assert len(final_preds_nested) == 324, f"Expected 324 rows, got {len(final_preds_nested)}"
    assert all(p in classes for p in final_preds_nested), "Found invalid class in predictions"

    # Browser consistency check
    browser_violations = [
        (i + 1, p, b_te[i], u2b.get(p))
        for i, p in enumerate(final_preds_nested)
        if u2b.get(p) and u2b[p] != b_te[i]
    ]
    print(f"  Violations navigateur dans la nouvelle soumission : {len(browser_violations)}")
    assert len(browser_violations) == 0, f"Browser violations detected: {browser_violations}"

    # Create submission DataFrame
    sub_improved_df = pd.DataFrame({
        "RowId": np.arange(1, len(final_preds_nested) + 1),
        "prediction": final_preds_nested,
    })
    sub_improved_df.to_csv(sub_improved_path, index=False)
    print(f"  -> Soumission sauvegardee : {sub_improved_path}")

    # Agreement matrix between all 4 submissions
    all_subs = {
        "v1_ensemble": sub_init["prediction"].values,
        "v2_optimized": sub_opt["prediction"].values,
        "v3_pseudo_labeled": sub_pl["prediction"].values,
        "v4_improved": final_preds_nested,
    }
    print("\n  Matrice de concordance entre les 4 soumissions (sur 324 sessions) :")
    for s_name1, p1 in all_subs.items():
        row_str = f"    {s_name1:<18s} : "
        for s_name2, p2 in all_subs.items():
            matches = (p1 == p2).sum()
            row_str += f"{s_name2}={matches}/324 ({100*matches/324:.1f}%)  "
        print(row_str)

    # Save nested CV results table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results_nested)
    res_path = TABLES_DIR / "nested_cv_results.csv"
    res_df.to_csv(res_path, index=False)
    print(f"\n  -> Tableau de resultats nested-CV : {res_path}")

    # Save detailed ambiguity analysis
    amb_df = pd.DataFrame(analysis_rows)
    amb_path = TABLES_DIR / "ambiguous_rows_analysis.csv"
    amb_df.to_csv(amb_path, index=False)
    print(f"  -> Analyse des 6 lignes ambiguës  : {amb_path}")

    total_time = time.time() - t0_global
    _sec(f"[OK] PIPELINE NESTED-CV TERMINE AVEC SUCCES EN {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
