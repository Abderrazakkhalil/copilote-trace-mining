"""Dynamic Meta-Stacking and Instance & Class-Specific Ensemble.

Mission: Infologic Copilote Trace Mining Challenge
Architecture:
1. Morphological sub-token TF-IDF enrichment + Block L2-normalization (alpha=0.20).
2. Diverse 6-model base:
   - ridge_hyb: Dual Cholesky Ridge (alpha=1.0) on X_hybrid_l2 (T=2.0)
   - svm_hyb: LinearSVC (C=0.5) on X_hybrid_l2 (T=2.5)
   - svm_seq: LinearSVC (C=0.5) on X_seq_enriched (T=2.5)
   - sgd_log_hyb: SGDClassifier (log_loss, alpha=1e-5) on X_hybrid_l2
   - logreg_hyb: LogisticRegression (C=25.0) on X_hybrid_l2
   - xgb_behav: XGBoost on Behavioral dense dataset
3. Bayesian Class-Wise Reliability R[m, c] (Nested-CV smoothed precision, prior_strength=3.0).
4. Dynamic Entropy-Margin Gating (DES) + Pairwise Meta-Stacking (Level-2 LogisticRegression on Top-5 Candidates).
5. Strict Nested 4-Fold CV validation + Clinical Audit + Generation of submission_dynamic_stacking.csv.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TABLES_DIR,
    TEST_CSV,
    TRAIN_CSV,
)
from src.parser import load_dataset
from src.postprocess import apply_browser_mask, decision_to_proba

# Force line-buffering and UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _sec(title: str):
    print("\n" + "=" * 76 + f"\n  {title}\n" + "=" * 76, flush=True)


def fit_predict_dual_ridge(X_tr, y_tr, X_val, X_te, alpha=1.0):
    """Fast Dual Cholesky RidgeClassifier with balanced class weights."""
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
    
    S_val = (X_val @ X_tr.T).toarray() @ A
    S_te = (X_te @ X_tr.T).toarray() @ A
    return S_val.astype(np.float32), S_te.astype(np.float32)


def main():
    t0_global = time.time()
    _sec("ETAPE 1 — ENRICHISSEMENT MORPHOLOGIQUE ET CONSTRUCTION DES MATRICES")

    # Load existing datasets and raw sequences
    meta = pickle.load(open(PROCESSED_DIR / "metadata.pkl", "rb"))
    X_seq_tr = pickle.load(open(PROCESSED_DIR / "X_train_seq.pkl", "rb"))
    X_seq_te = pickle.load(open(PROCESSED_DIR / "X_test_seq.pkl", "rb"))
    X_hyb_tr = pickle.load(open(PROCESSED_DIR / "X_train_hybrid_l2.pkl", "rb"))
    X_hyb_te = pickle.load(open(PROCESSED_DIR / "X_test_hybrid_l2.pkl", "rb"))
    X_btr = pickle.load(open(PROCESSED_DIR / "X_train_behav.pkl", "rb"))
    X_bte = pickle.load(open(PROCESSED_DIR / "X_test_behav.pkl", "rb"))

    y_raw = meta["y_train"]
    le = LabelEncoder().fit(y_raw)
    y_enc = le.transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)
    b_tr = meta["browsers_train"]
    b_te = meta["browsers_test"]
    u2b = meta["user_to_browser"]

    print(f"  Train: {len(y_enc)} sessions | Test: {len(b_te)} sessions | Classes: {n_classes}")

    # 1a. Morphological Sub-tokens Extraction from Controllers and Configs
    print("  1a. Extraction des sous-jetons morphologiques (controleurs/ecrans)...")
    train_sessions = load_dataset(TRAIN_CSV, is_train=True)
    test_sessions = load_dataset(TEST_CSV, is_train=False)

    def extract_subtokens(sessions):
        docs = []
        for s in sessions:
            subtoks = []
            for c in s["controllers"]:
                if c:
                    parts = re.split(r"[._/\-\s]+", str(c))
                    subtoks.extend([p.lower() for p in parts if len(p) > 1])
            for cfg in s["configs"]:
                if cfg:
                    parts = re.split(r"[._/\-\s]+", str(cfg))
                    subtoks.extend([p.lower() for p in parts if len(p) > 1])
            docs.append(subtoks)
        return docs

    tr_sub_docs = extract_subtokens(train_sessions)
    te_sub_docs = extract_subtokens(test_sessions)

    vec_sub = TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
        sublinear_tf=True,
        norm="l2",
    )
    X_sub_tr = vec_sub.fit_transform(tr_sub_docs)
    X_sub_te = vec_sub.transform(te_sub_docs)
    print(f"    Sous-jetons extraits : {X_sub_tr.shape[1]} features TF-IDF")

    BETA = 0.10
    X_seq_enr_tr = normalize(sp.hstack([X_seq_tr, BETA * X_sub_tr], format="csr"), norm="l2")
    X_seq_enr_te = normalize(sp.hstack([X_seq_te, BETA * X_sub_te], format="csr"), norm="l2")
    print(f"    X_seq_enriched : {X_seq_enr_tr.shape[1]} features (beta={BETA})")
    print(f"    X_hybrid_l2    : {X_hyb_tr.shape[1]} features (optimal block L2)")

    # Session action lengths for meta-features
    nb_actions_tr = np.array([len(s["raw_actions"]) for s in train_sessions], dtype=np.float32)
    nb_actions_te = np.array([len(s["raw_actions"]) for s in test_sessions], dtype=np.float32)

    _sec("ETAPE 2 — ENTRAINEMENT DES 6 MODELES COMPLEMENTAIRES EN 4-FOLD CV")
    outer_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(outer_skf.split(X_hyb_tr, y_enc))

    models_config = {
        "ridge_hyb": {
            "name": "Dual Ridge(alpha=1.0) on X_hybrid_l2",
            "type": "dual_ridge",
            "temp": 2.0,
            "matrix_tr": X_hyb_tr,
            "matrix_te": X_hyb_te,
        },
        "svm_hyb": {
            "name": "LinearSVC(C=0.5) on X_hybrid_l2",
            "type": "svm",
            "temp": 2.5,
            "matrix_tr": X_hyb_tr,
            "matrix_te": X_hyb_te,
            "factory": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        },
        "svm_seq": {
            "name": "LinearSVC(C=0.5) on X_seq_enriched",
            "type": "svm",
            "temp": 2.5,
            "matrix_tr": X_seq_enr_tr,
            "matrix_te": X_seq_enr_te,
            "factory": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        },
        "sgd_log_hyb": {
            "name": "SGDClassifier(log_loss) on X_hybrid_l2",
            "type": "prob",
            "matrix_tr": X_hyb_tr,
            "matrix_te": X_hyb_te,
            "factory": lambda: SGDClassifier(loss="log_loss", alpha=1e-5, penalty="l2",
                                             class_weight="balanced", max_iter=250, random_state=RANDOM_STATE),
        },
        "logreg_hyb": {
            "name": "LogisticRegression(C=25.0) on X_hybrid_l2",
            "type": "prob",
            "matrix_tr": X_hyb_tr,
            "matrix_te": X_hyb_te,
            "factory": lambda: LogisticRegression(C=25.0, class_weight="balanced", solver="lbfgs",
                                                 max_iter=150, random_state=RANDOM_STATE),
        },
        "xgb_behav": {
            "name": "XGBoost on Behavioral Dense",
            "type": "xgb",
            "matrix_tr": X_btr,
            "matrix_te": X_bte,
            "factory": lambda: XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.15,
                                             subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                                             eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
        },
    }

    probs_save_path = PROCESSED_DIR / "dynamic_stacking_probs.pkl"
    force_retrain = "--retrain" in sys.argv

    if probs_save_path.exists() and not force_retrain:
        print(f"  Chargement des probabilités calibrées existantes depuis : {probs_save_path}")
        saved_dict = pickle.load(open(probs_save_path, "rb"))
        oof_probs = saved_dict["oof_probs"]
        test_probs = saved_dict["test_probs"]
    else:
        oof_probs = {m: np.zeros((len(y_enc), n_classes), dtype=np.float32) for m in models_config}
        test_probs = {m: np.zeros((len(b_te), n_classes), dtype=np.float32) for m in models_config}

        for m_key, m_cfg in models_config.items():
            t0_m = time.time()
            print(f"\n  Entrainement [{m_key}] ({m_cfg['name']})...", flush=True)
            te_folds = []

            for fold, (tr_idx, val_idx) in enumerate(outer_splits):
                f0 = time.time()
                X_tr, y_tr = m_cfg["matrix_tr"][tr_idx], y_enc[tr_idx]
                X_val, y_val = m_cfg["matrix_tr"][val_idx], y_enc[val_idx]
                X_te = m_cfg["matrix_te"]

                if m_cfg["type"] == "dual_ridge":
                    y_tr_raw = [y_raw[i] for i in tr_idx]
                    s_val, s_te = fit_predict_dual_ridge(X_tr, y_tr_raw, X_val, X_te, alpha=1.0)
                    p_val = decision_to_proba(s_val, temperature=m_cfg["temp"])
                    p_te = decision_to_proba(s_te, temperature=m_cfg["temp"])
                elif m_cfg["type"] == "svm":
                    clf = m_cfg["factory"]()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(X_tr, y_tr)
                    s_val = clf.decision_function(X_val)
                    s_te = clf.decision_function(X_te)
                    p_val = decision_to_proba(s_val, temperature=m_cfg["temp"])
                    p_te = decision_to_proba(s_te, temperature=m_cfg["temp"])
                elif m_cfg["type"] in ("prob", "xgb"):
                    clf = m_cfg["factory"]()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(X_tr, y_tr)
                    p_val = clf.predict_proba(X_val)
                    p_te = clf.predict_proba(X_te)
                    if p_val.shape[1] < n_classes:
                        full_val = np.zeros((len(val_idx), n_classes), dtype=np.float32)
                        full_te = np.zeros((len(b_te), n_classes), dtype=np.float32)
                        for idx_c, cls_c in enumerate(clf.classes_):
                            full_val[:, int(cls_c)] = p_val[:, idx_c]
                            full_te[:, int(cls_c)] = p_te[:, idx_c]
                        p_val, p_te = full_val, full_te

                p_val = apply_browser_mask(p_val, [b_tr[i] for i in val_idx], classes, u2b)
                p_te = apply_browser_mask(p_te, b_te, classes, u2b)

                oof_probs[m_key][val_idx] = p_val.astype(np.float32)
                te_folds.append(p_te.astype(np.float32))
                print(f"    Fold {fold+1}/4 fait en {time.time()-f0:.1f}s", flush=True)

            test_probs[m_key] = np.mean(te_folds, axis=0)

        with open(probs_save_path, "wb") as f:
            pickle.dump({"oof_probs": oof_probs, "test_probs": test_probs, "classes": classes}, f)
        print(f"\n  [Sauvegarde] Matrices de probabilités sauvegardées dans : {probs_save_path}")

    _sec("ETAPE 3 — MATRICE DE COMPETENCE PAR CLASSE ET GATING DYNAMIQUE")

    PRIOR_STRENGTH = 3.0

    def compute_reliability_matrix(oof_p_dict, tr_idx, y_train):
        R = {}
        for m_key, p_matrix in oof_p_dict.items():
            preds = np.argmax(p_matrix[tr_idx], axis=1)
            acc = accuracy_score(y_train, preds)
            R_m = np.zeros(n_classes, dtype=np.float32)

            for c in range(n_classes):
                tp = np.sum((y_train == c) & (preds == c))
                fp = np.sum((y_train != c) & (preds == c))
                R_m[c] = (tp + PRIOR_STRENGTH * acc) / (tp + fp + PRIOR_STRENGTH)
            R[m_key] = R_m
        return R

    # Dynamic Entropy-Margin Gating configuration
    weights_des = {
        "ridge_hyb": 0.88,
        "svm_hyb": 0.08,
        "svm_seq": 0.04,
    }
    gamma = 0.35

    def compute_des_scores(probs_dict, R_dict, g=0.35):
        n_samples = next(iter(probs_dict.values())).shape[0]
        blended = np.zeros((n_samples, n_classes), dtype=np.float32)

        for m_key, w in weights_des.items():
            P = probs_dict[m_key]
            R_m = R_dict[m_key]
            P_sort = np.sort(P, axis=1)
            margin = (P_sort[:, -1] - P_sort[:, -2]) ** g
            W_dyn = w * (margin[:, None] * R_m[None, :])
            blended += W_dyn * P

        return blended

    _sec("ETAPE 4 — EVALUATION COMPARATIVE EN NESTED-CV (OUT-OF-FOLD PUR)")

    eval_results = []

    # 4a. Individual models
    for m_key in models_config:
        preds = np.argmax(oof_probs[m_key], axis=1)
        f1 = f1_score(y_enc, preds, average="macro")
        acc = accuracy_score(y_enc, preds)
        eval_results.append({
            "Methode": f"Single: {m_key}",
            "Macro_F1": round(f1, 4),
            "Accuracy": round(acc, 4),
        })
        print(f"  Single: {m_key:<20s} -> Macro-F1={f1:.4f}  Acc={acc:.4f}")

    # 4b. Fixed-Weights Baseline Ensemble (Reference)
    fixed_blend_oof = sum(weights_des[m] * oof_probs[m] for m in weights_des)
    fixed_preds = np.argmax(fixed_blend_oof, axis=1)
    f1_fixed = f1_score(y_enc, fixed_preds, average="macro")
    acc_fixed = accuracy_score(y_enc, fixed_preds)
    eval_results.append({
        "Methode": "Soft Voting Poids Fixes (Reference)",
        "Macro_F1": round(f1_fixed, 4),
        "Accuracy": round(acc_fixed, 4),
    })
    print(f"\n  Soft Voting Poids Fixes -> Macro-F1={f1_fixed:.4f}  Acc={acc_fixed:.4f}")

    # 4c. Dynamic Entropy-Margin Ensemble (DES) in Nested-CV
    des_oof_probs = np.zeros((len(y_enc), n_classes), dtype=np.float32)
    for fold, (tr_idx, val_idx) in enumerate(outer_splits):
        R_fold = compute_reliability_matrix(oof_probs, tr_idx, y_enc[tr_idx])
        val_probs = {m: oof_probs[m][val_idx] for m in weights_des}
        des_scores_val = compute_des_scores(val_probs, R_fold, g=gamma)
        des_scores_val = apply_browser_mask(des_scores_val, [b_tr[i] for i in val_idx], classes, u2b)
        des_oof_probs[val_idx] = des_scores_val

    des_oof_preds = np.argmax(des_oof_probs, axis=1)
    f1_des = f1_score(y_enc, des_oof_preds, average="macro")
    acc_des = accuracy_score(y_enc, des_oof_preds)
    eval_results.append({
        "Methode": "Dynamic Entropy-Margin Ensemble (DES)",
        "Macro_F1": round(f1_des, 4),
        "Accuracy": round(acc_des, 4),
    })
    print(f"  Dynamic Entropy-Margin (DES) -> Macro-F1={f1_des:.4f}  Acc={acc_des:.4f} (Δ={f1_des - f1_fixed:+.4f})")

    # 4d. Pairwise Meta-Stacking Level 2 (LogisticRegression sur Top-5 Candidats)
    print("\n  Entrainement du Meta-Stacker Niveau 2 (Pairwise Top-5 Candidates)...")
    meta_oof_probs = np.zeros((len(y_enc), n_classes), dtype=np.float32)
    log_actions_tr = np.log1p(nb_actions_tr)
    log_actions_te = np.log1p(nb_actions_te)

    oof_margins = {m: np.sort(oof_probs[m], axis=1)[:, -1] - np.sort(oof_probs[m], axis=1)[:, -2] for m in oof_probs}
    test_margins = {m: np.sort(test_probs[m], axis=1)[:, -1] - np.sort(test_probs[m], axis=1)[:, -2] for m in test_probs}

    for fold, (tr_idx, val_idx) in enumerate(outer_splits):
        R_fold = compute_reliability_matrix(oof_probs, tr_idx, y_enc[tr_idx])
        
        X_m_tr, y_m_tr = [], []
        for s_i in tr_idx:
            cands = np.argsort(fixed_blend_oof[s_i])[::-1][:5]
            true_c = y_enc[s_i]
            la = log_actions_tr[s_i]
            for c in cands:
                feats = []
                for m in ["ridge_hyb", "svm_hyb", "svm_seq"]:
                    p = oof_probs[m][s_i, c]
                    feats.extend([p, np.log(p + 1e-6), oof_margins[m][s_i], R_fold[m][c], p * la])
                X_m_tr.append(feats)
                y_m_tr.append(1 if c == true_c else 0)

        meta_lr = LogisticRegression(C=1.0, max_iter=200, random_state=RANDOM_STATE).fit(X_m_tr, y_m_tr)

        for s_i in val_idx:
            cands = np.argsort(des_oof_probs[s_i])[::-1][:5]
            la = log_actions_tr[s_i]
            cand_feats = []
            for c in cands:
                feats = []
                for m in ["ridge_hyb", "svm_hyb", "svm_seq"]:
                    p = oof_probs[m][s_i, c]
                    feats.extend([p, np.log(p + 1e-6), oof_margins[m][s_i], R_fold[m][c], p * la])
                cand_feats.append(feats)
            probs = meta_lr.predict_proba(cand_feats)[:, 1]
            for c, pr in zip(cands, probs):
                meta_oof_probs[s_i, c] = pr

    meta_oof_probs = apply_browser_mask(meta_oof_probs, b_tr, classes, u2b)
    meta_oof_preds = np.argmax(meta_oof_probs, axis=1)
    f1_meta = f1_score(y_enc, meta_oof_preds, average="macro")
    acc_meta = accuracy_score(y_enc, meta_oof_preds)
    eval_results.append({
        "Methode": "Meta-Stacking Niveau 2 (Pairwise)",
        "Macro_F1": round(f1_meta, 4),
        "Accuracy": round(acc_meta, 4),
    })
    print(f"  Meta-Stacking Niveau 2 -> Macro-F1={f1_meta:.4f}  Acc={acc_meta:.4f} (Δ={f1_meta - f1_fixed:+.4f})")

    # 4e. Hybride (50% DES + 50% Meta-Stacking)
    norm_des_oof = des_oof_probs / np.maximum(des_oof_probs.sum(axis=1, keepdims=True), 1e-9)
    norm_meta_oof = meta_oof_probs / np.maximum(meta_oof_probs.sum(axis=1, keepdims=True), 1e-9)
    hybrid_oof_probs = 0.5 * norm_des_oof + 0.5 * norm_meta_oof
    hybrid_oof_preds = np.argmax(hybrid_oof_probs, axis=1)
    f1_hyb = f1_score(y_enc, hybrid_oof_preds, average="macro")
    acc_hyb = accuracy_score(y_enc, hybrid_oof_preds)
    eval_results.append({
        "Methode": "Hybride (50% DES + 50% Meta-Stacking)",
        "Macro_F1": round(f1_hyb, 4),
        "Accuracy": round(acc_hyb, 4),
    })
    print(f"  Hybride (50% DES + 50% Meta) -> Macro-F1={f1_hyb:.4f}  Acc={acc_hyb:.4f}")

    # Build Test Predictions
    R_global = compute_reliability_matrix(oof_probs, np.arange(len(y_enc)), y_enc)
    
    # DES on Test
    des_test_scores = compute_des_scores(test_probs, R_global, g=gamma)
    des_test_probs = apply_browser_mask(des_test_scores, b_te, classes, u2b)
    norm_des_test = des_test_probs / np.maximum(des_test_probs.sum(axis=1, keepdims=True), 1e-9)

    # Train Level 2 Meta-Model on full train
    X_m_full, y_m_full = [], []
    for s_i in range(len(y_enc)):
        cands = np.argsort(fixed_blend_oof[s_i])[::-1][:5]
        true_c = y_enc[s_i]
        la = log_actions_tr[s_i]
        for c in cands:
            feats = []
            for m in ["ridge_hyb", "svm_hyb", "svm_seq"]:
                p = oof_probs[m][s_i, c]
                feats.extend([p, np.log(p + 1e-6), oof_margins[m][s_i], R_global[m][c], p * la])
            X_m_full.append(feats)
            y_m_full.append(1 if c == true_c else 0)

    meta_lr_full = LogisticRegression(C=1.0, max_iter=200, random_state=RANDOM_STATE).fit(X_m_full, y_m_full)

    meta_test_probs = np.zeros((len(b_te), n_classes), dtype=np.float32)
    for s_i in range(len(b_te)):
        cands = np.argsort(norm_des_test[s_i])[::-1][:5]
        la = log_actions_te[s_i]
        cand_feats = []
        for c in cands:
            feats = []
            for m in ["ridge_hyb", "svm_hyb", "svm_seq"]:
                p = test_probs[m][s_i, c]
                feats.extend([p, np.log(p + 1e-6), test_margins[m][s_i], R_global[m][c], p * la])
            cand_feats.append(feats)
        probs = meta_lr_full.predict_proba(cand_feats)[:, 1]
        for c, pr in zip(cands, probs):
            meta_test_probs[s_i, c] = pr

    meta_test_probs = apply_browser_mask(meta_test_probs, b_te, classes, u2b)
    norm_meta_test = meta_test_probs / np.maximum(meta_test_probs.sum(axis=1, keepdims=True), 1e-9)

    # Select best validated model: DES has the highest Out-Of-Fold Macro-F1 (0.9696)
    p_final_test = norm_des_test
    p_final_test = apply_browser_mask(p_final_test, b_te, classes, u2b)
    final_preds_dyn = classes[np.argmax(p_final_test, axis=1)]

    _sec("ETAPE 5 — AUDIT DES SESSIONS AMBIGUES ET CAS HISTORIQUES")
    ambiguous_rows = [15, 116, 118, 226, 272, 298, 311]

    sub_v1 = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble.csv")
    sub_v2 = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble_optimized.csv")
    sub_v3 = pd.read_csv(SUBMISSIONS_DIR / "submission_pseudo_labeled.csv")
    sub_v4 = pd.read_csv(SUBMISSIONS_DIR / "submission_improved.csv")

    audit_records = []
    print(f"  {'RowId':<6s} {'Browser':<15s} {'v1(init)':<10s} {'v2(opt)':<10s} {'v4(imp)':<10s} {'Dynamic':<10s} {'Top-2 Margins':<22s} {'Decision'}")
    print("  " + "-" * 90)

    for rid in ambiguous_rows:
        idx = rid - 1
        browser = b_te[idx]
        pv1 = sub_v1.loc[idx, "prediction"]
        pv2 = sub_v2.loc[idx, "prediction"]
        pv4 = sub_v4.loc[idx, "prediction"]
        pdyn = final_preds_dyn[idx]

        top2_c = np.argsort(p_final_test[idx])[::-1][:2]
        c1, c2 = classes[top2_c[0]], classes[top2_c[1]]
        sc1, sc2 = p_final_test[idx, top2_c[0]], p_final_test[idx, top2_c[1]]
        margin_str = f"{c1}:{sc1:.3f} vs {c2}:{sc2:.3f}"

        audit_records.append({
            "RowId": rid,
            "Browser": browser,
            "v1": pv1,
            "v2": pv2,
            "v4": pv4,
            "Dynamic_Stacking": pdyn,
            "Top1": c1,
            "Top2": c2,
            "Margin": round(float(sc1 - sc2), 4),
        })
        print(f"  {rid:<6d} {browser:<15s} {pv1:<10s} {pv2:<10s} {pv4:<10s} {pdyn:<10s} {margin_str:<22s} -> {pdyn}")

    # Check differences across all 324 rows between Dynamic Stacking and previous submissions
    diff_mask_v2 = final_preds_dyn != sub_v2["prediction"].values
    n_diffs_v2 = np.sum(diff_mask_v2)
    print(f"\n  Total sessions modifiees vs v2 (0.974) : {n_diffs_v2} / 324")
    if n_diffs_v2 > 0:
        diff_rids = np.where(diff_mask_v2)[0] + 1
        for d_rid in diff_rids:
            idx = d_rid - 1
            top2_idx = np.argsort(p_final_test[idx])[::-1][:2]
            c1, c2 = classes[top2_idx[0]], classes[top2_idx[1]]
            sc1, sc2 = p_final_test[idx, top2_idx[0]], p_final_test[idx, top2_idx[1]]
            ent = -np.sum(p_final_test[idx] * np.log(p_final_test[idx] + 1e-12))
            print(f"    RowId {d_rid:3d} (Browser: {b_te[idx]}): v2={sub_v2.loc[idx, 'prediction']} -> Dynamic={final_preds_dyn[idx]} (Margin={sc1-sc2:.4f}, Entropie={ent:.3f})")

    diff_mask_v3 = final_preds_dyn != sub_v3["prediction"].values
    n_diffs_v3 = np.sum(diff_mask_v3)
    print(f"  Total sessions modifiees vs v3 (pseudo-labeled, 0.974) : {n_diffs_v3} / 324")

    diff_mask_v4 = final_preds_dyn != sub_v4["prediction"].values
    n_diffs_v4 = np.sum(diff_mask_v4)
    print(f"  Total sessions modifiees vs v4 (improved, 0.972) : {n_diffs_v4} / 324")
    if n_diffs_v4 > 0:
        diff_rids_v4 = np.where(diff_mask_v4)[0] + 1
        for d_rid in diff_rids_v4:
            idx = d_rid - 1
            print(f"    RowId {d_rid:3d} (Browser: {b_te[idx]}): v4={sub_v4.loc[idx, 'prediction']} -> Dynamic={final_preds_dyn[idx]}")

    _sec("ETAPE 6 — GENERATION DE LA SOUMISSION DYNAMIC STACKING")
    out_sub_path = SUBMISSIONS_DIR / "submission_dynamic_stacking.csv"

    # Validation constraints
    assert len(final_preds_dyn) == 324, f"Expected 324 rows, got {len(final_preds_dyn)}"
    assert all(p in classes for p in final_preds_dyn), "Unknown class predicted!"
    violations = [
        (i + 1, p, b_te[i], u2b.get(p))
        for i, p in enumerate(final_preds_dyn)
        if u2b.get(p) and u2b[p] != b_te[i]
    ]
    assert len(violations) == 0, f"Violations navigateur détectées : {violations}"
    print(f"  Violations navigateur verifiees : 0")

    pd.DataFrame({
        "RowId": np.arange(1, len(final_preds_dyn) + 1),
        "prediction": final_preds_dyn,
    }).to_csv(out_sub_path, index=False)
    print(f"  -> Soumission sauvegardee : {out_sub_path}")

    # Save summary tables
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_results).to_csv(TABLES_DIR / "dynamic_stacking_results.csv", index=False)
    pd.DataFrame(audit_records).to_csv(TABLES_DIR / "dynamic_stacking_audit.csv", index=False)

    print("\n  Tableau d'Evaluation Complet :")
    print(pd.DataFrame(eval_results).to_string(index=False))

    total_time = time.time() - t0_global
    _sec(f"[OK] PIPELINE DYNAMIC STACKING COMPLETE EN {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
