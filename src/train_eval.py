"""Entraînement, validation croisée, ensembling et visualisation.

Exécute 5 familles de modèles × 3 datasets en StratifiedKFold(4),
applique le post-processing navigateur, optimise un Soft Voting pondéré,
et génère les figures de comparaison des modèles.

Benchmark exploratoire : les matrices de features sont préapprises sur tout
le train. Pour une évaluation où les transformations sont réapprises dans
chaque pli, utiliser src.dl_hybrid_pipeline.

Usage
-----
    python -m src.train_eval
"""

from __future__ import annotations

import sys
import io

# ── Force UTF-8 on Windows (PowerShell/cmd default is CP1252) ──────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    PROCESSED_DIR, MODEL_FIGURES_DIR as FIGURES_DIR, TABLES_DIR, SUBMISSIONS_DIR,
    RANDOM_STATE, N_SPLITS,
)
from src.postprocess import apply_browser_mask, decision_to_proba


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constantes & Noms d'affichage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET_NAMES = ["seq", "behav", "hybrid"]
MODEL_NAMES   = ["logreg", "svm", "mlp", "dtree", "xgboost"]

DATASET_DISPLAY = {
    "seq":    "Sequential_NLP",
    "behav":  "Behavioral_Dense",
    "hybrid": "Hybrid_Combined",
}
MODEL_DISPLAY = {
    "logreg":  "LogReg",
    "svm":     "LinearSVM",
    "mlp":     "MLP",
    "dtree":   "DecisionTree",
    "xgboost": "XGBoost",
}

# Palette de couleurs pour les figures
COLORS_DS = {"seq": "#4C72B0", "behav": "#DD8452", "hybrid": "#55A868"}
COLOR_ENSEMBLE = "#C44E52"
COLOR_RAW    = "#AAAAAA"
COLOR_MASKED = "#4C72B0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fabrique de modèles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_model(name: str):
    """Instancie un modèle avec les hyper-paramètres fixés."""
    if name == "logreg":
        return LogisticRegression(
            C=5.0, class_weight="balanced", solver="lbfgs",
            max_iter=500, random_state=RANDOM_STATE,
        )
    if name == "svm":
        return LinearSVC(
            C=1.0, class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-2,
            max_iter=60, early_stopping=True,
            random_state=RANDOM_STATE,
        )
    if name == "dtree":
        return DecisionTreeClassifier(
            max_depth=25, min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.5,
            tree_method="hist", eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Modèle inconnu : {name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Préparation des données par pli (anti-leakage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _prepare_fold(
    model_name: str,
    dataset_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_seq, X_behav, X_hybrid,
    X_test_seq, X_test_behav, X_test_hybrid,
):
    """Prépare (X_tr, X_val, X_te) adaptés au couple (modèle, dataset).

    Applique TruncatedSVD et/ou StandardScaler **dans le pli** pour
    les modèles qui en ont besoin, en fittant exclusivement sur X_tr.
    """
    needs_svd = model_name in ("mlp", "dtree", "xgboost")
    needs_scale = model_name == "mlp"
    n_svd = 128 if model_name == "mlp" else 80

    # ── behav ───────────────────────────────────────────────────────────
    if dataset_name == "behav":
        X_tr  = X_behav[train_idx]
        X_val = X_behav[val_idx]
        X_te  = X_test_behav.copy()
        if needs_scale:
            sc = StandardScaler()
            X_tr  = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
            X_te  = sc.transform(X_te)
        return X_tr, X_val, X_te

    # ── seq ─────────────────────────────────────────────────────────────
    if dataset_name == "seq":
        if needs_svd:
            svd = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)
            X_tr  = svd.fit_transform(X_seq[train_idx])
            X_val = svd.transform(X_seq[val_idx])
            X_te  = svd.transform(X_test_seq)
            if needs_scale:
                sc = StandardScaler()
                X_tr  = sc.fit_transform(X_tr)
                X_val = sc.transform(X_val)
                X_te  = sc.transform(X_te)
            return X_tr, X_val, X_te
        # logreg / svm : sparse directement
        return X_seq[train_idx], X_seq[val_idx], X_test_seq

    # ── hybrid ──────────────────────────────────────────────────────────
    if dataset_name == "hybrid":
        if needs_svd:
            svd = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)
            s_tr  = svd.fit_transform(X_seq[train_idx])
            s_val = svd.transform(X_seq[val_idx])
            s_te  = svd.transform(X_test_seq)
            # Concaténer avec les features denses brutes
            X_tr  = np.hstack([s_tr,  X_behav[train_idx]])
            X_val = np.hstack([s_val, X_behav[val_idx]])
            X_te  = np.hstack([s_te,  X_test_behav])
            if needs_scale:
                sc = StandardScaler()
                X_tr  = sc.fit_transform(X_tr)
                X_val = sc.transform(X_val)
                X_te  = sc.transform(X_te)
            return X_tr, X_val, X_te
        # logreg / svm : hybrid sparse pré-construit
        return X_hybrid[train_idx], X_hybrid[val_idx], X_test_hybrid

    raise ValueError(f"Dataset inconnu : {dataset_name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Extraction de probabilités
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_proba(model, model_name: str, X, n_classes: int) -> np.ndarray:
    """Extrait les probabilités de classe depuis un modèle fitté."""
    if model_name == "svm":
        scores = model.decision_function(X)
        probs = decision_to_proba(scores, temperature=2.0)
    else:
        probs = model.predict_proba(X)

    # Sécurité : si le modèle n'a pas vu toutes les classes, compléter
    if probs.shape[1] < n_classes:
        n_samples = probs.shape[0]
        full = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, c in enumerate(model.classes_):
            full[:, int(c)] = probs[:, i]
        return full
    return probs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boucle d'évaluation principale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_all(
    X_seq, X_behav, X_hybrid,
    X_test_seq, X_test_behav, X_test_hybrid,
    y_encoded: np.ndarray,
    le: LabelEncoder,
    metadata: dict,
) -> tuple[list[dict], dict, dict]:
    """Évalue les 15 combinaisons (5 modèles × 3 datasets) en 4-Fold CV.

    Returns
    -------
    results : list[dict]        – tableau de métriques.
    all_oof_masked : dict       – ``{key: ndarray(n_train, n_classes)}``.
    all_test_masked : dict      – ``{key: ndarray(n_test, n_classes)}``.
    """
    n_train   = len(y_encoded)
    n_classes = len(le.classes_)
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE,
    )

    browsers_train  = metadata["browsers_train"]
    browsers_test   = metadata["browsers_test"]
    user_to_browser = metadata["user_to_browser"]
    classes = le.classes_                       # labels originaux triés

    results: list[dict] = []
    all_oof_masked:  dict[str, np.ndarray] = {}
    all_test_masked: dict[str, np.ndarray] = {}

    combo_idx = 0
    total = len(DATASET_NAMES) * len(MODEL_NAMES)

    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            combo_idx += 1
            key = f"{dataset_name}_{model_name}"
            print(f"  [{combo_idx:02d}/{total}] {key:<25s}", end="", flush=True)

            t0 = time.time()
            oof_probs = np.zeros((n_train, n_classes), dtype=np.float64)
            test_folds: list[np.ndarray] = []

            for fold_idx, (tr_idx, va_idx) in enumerate(
                skf.split(X_seq, y_encoded)
            ):
                print(".", end="", flush=True)

                X_tr, X_va, X_te = _prepare_fold(
                    model_name, dataset_name, tr_idx, va_idx,
                    X_seq, X_behav, X_hybrid,
                    X_test_seq, X_test_behav, X_test_hybrid,
                )

                model = _make_model(model_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_encoded[tr_idx])

                oof_probs[va_idx] = _get_proba(
                    model, model_name, X_va, n_classes,
                )
                test_folds.append(
                    _get_proba(model, model_name, X_te, n_classes)
                )

            elapsed = time.time() - t0
            test_probs = np.mean(test_folds, axis=0)

            # ── Métriques brutes (avant masquage) ───────────────────────
            raw_preds = np.argmax(oof_probs, axis=1)
            raw_acc = accuracy_score(y_encoded, raw_preds)
            raw_f1  = f1_score(y_encoded, raw_preds, average="macro")

            # ── Métriques masquées ──────────────────────────────────────
            m_oof = apply_browser_mask(
                oof_probs, browsers_train, classes, user_to_browser,
            )
            m_test = apply_browser_mask(
                test_probs, browsers_test, classes, user_to_browser,
            )

            m_preds = np.argmax(m_oof, axis=1)
            m_acc = accuracy_score(y_encoded, m_preds)
            m_f1  = f1_score(y_encoded, m_preds, average="macro")

            delta = m_f1 - raw_f1
            print(
                f"  F1={m_f1:.4f} (Δ={delta:+.4f})  [{elapsed:.1f}s]"
            )

            all_oof_masked[key]  = m_oof
            all_test_masked[key] = m_test

            results.append({
                "Dataset":              dataset_name,
                "Model":                model_name,
                "Raw_Accuracy":         round(raw_acc, 4),
                "Raw_Macro_F1":         round(raw_f1, 4),
                "Masked_Accuracy":      round(m_acc, 4),
                "Masked_Macro_F1":      round(m_f1, 4),
                "Gain_Macro_F1_Masking": round(delta, 4),
                "Train_Time_Sec":       round(elapsed, 1),
            })

    return results, all_oof_masked, all_test_masked


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Optimisation de l'Ensemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_optimal_weights(
    oof_list: list[np.ndarray],
    y_true: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Trouve les poids optimaux pour le Soft Voting via differential evolution.

    Les poids sont contraints positifs et normalisés pour sommer à 1.
    L'objectif minimisé est ``-Macro_F1(argmax(Σ w_i p_i), y_true)``.
    """
    k = len(oof_list)

    def neg_f1(raw_w):
        w = np.asarray(raw_w)
        w = w / w.sum()
        blended = sum(wi * pi for wi, pi in zip(w, oof_list))
        preds = np.argmax(blended, axis=1)
        return -f1_score(y_true, preds, average="macro")

    bounds = [(0.001, 1.0)] * k
    res = differential_evolution(
        neg_f1, bounds,
        seed=RANDOM_STATE, maxiter=300, tol=1e-7, popsize=25,
    )
    opt_w = res.x / res.x.sum()
    return opt_w, -res.fun


def _hard_voting_score(
    oof_list: list[np.ndarray],
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Vote majoritaire : accuracy et macro-F1."""
    votes = np.array([np.argmax(p, axis=1) for p in oof_list])  # (k, n)
    n = votes.shape[1]
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        c = Counter(votes[:, i].tolist())
        preds[i] = c.most_common(1)[0][0]
    return (
        float(accuracy_score(y_true, preds)),
        float(f1_score(y_true, preds, average="macro")),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _plot_benchmark(results: list[dict], ensemble_f1: float):
    """Fig 6 — Barres groupées : Masked Macro-F1 par Modèle × Dataset + Ensemble."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    n_m = len(MODEL_NAMES)
    bar_w = 0.22
    x = np.arange(n_m)

    # Lookup rapide
    lut: dict[tuple, float] = {}
    for r in results:
        lut[(r["Dataset"], r["Model"])] = r["Masked_Macro_F1"]

    # Barres par dataset
    for i, ds in enumerate(DATASET_NAMES):
        offset = (i - 1) * bar_w
        vals = [lut.get((ds, m), 0) for m in MODEL_NAMES]
        bars = ax.bar(
            x + offset, vals, bar_w,
            label=DATASET_DISPLAY[ds],
            color=COLORS_DS[ds], edgecolor="white", lw=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    # Barre Ensemble
    ens_x = n_m + 0.6
    ax.bar(
        ens_x, ensemble_f1, bar_w * 2.5,
        label="Ensemble (Soft Voting)",
        color=COLOR_ENSEMBLE, edgecolor="white", lw=0.5,
    )
    ax.text(
        ens_x, ensemble_f1 + 0.004,
        f"{ensemble_f1:.3f}", ha="center", va="bottom",
        fontsize=9, fontweight="bold",
    )

    # Mise en forme
    xticks = list(x) + [ens_x]
    xlabels = [MODEL_DISPLAY[m] for m in MODEL_NAMES] + ["Ensemble"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Masked Macro-F1 (OOF)")
    ax.set_title(
        "Fig. 6 — Benchmark : 5 Modèles × 3 Datasets "
        "+ Ensemble (Soft Voting Pondéré)"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = FIGURES_DIR / "fig6_benchmark_models_datasets.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    -> {path}")


def _plot_masking_impact(results: list[dict]):
    """Fig 7 — Raw vs Masked Macro-F1 pour les 5 modèles sur Hybrid."""
    fig, ax = plt.subplots(figsize=(10, 5))

    hybrid = {r["Model"]: r for r in results if r["Dataset"] == "hybrid"}

    x = np.arange(len(MODEL_NAMES))
    w = 0.32

    raw_vals    = [hybrid[m]["Raw_Macro_F1"]    for m in MODEL_NAMES]
    masked_vals = [hybrid[m]["Masked_Macro_F1"] for m in MODEL_NAMES]

    bars_r = ax.bar(
        x - w / 2, raw_vals, w,
        label="Raw (sans masque)", color=COLOR_RAW, edgecolor="white",
    )
    bars_m = ax.bar(
        x + w / 2, masked_vals, w,
        label="Masked (navigateur)", color=COLOR_MASKED, edgecolor="white",
    )

    for bar, v in zip(bars_r, raw_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#666",
        )
    for bar, v in zip(bars_m, masked_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )

    # Annotations Δ
    for i, m in enumerate(MODEL_NAMES):
        delta = masked_vals[i] - raw_vals[i]
        y_pos = max(raw_vals[i], masked_vals[i]) + 0.025
        ax.annotate(
            f"Δ={delta:+.3f}", xy=(i, y_pos), ha="center",
            fontsize=7.5, color=COLOR_ENSEMBLE, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_NAMES])
    ax.set_ylabel("Macro-F1 (OOF)")
    ax.set_title(
        "Fig. 7 — Impact du Post-Processing "
        "(Masquage Navigateur) sur Hybrid"
    )
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = FIGURES_DIR / "fig7_postprocessing_impact.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    -> {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Point d'entrée principal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_global = time.time()

    # ── 1. Chargement ───────────────────────────────────────────────────
    print("=" * 72)
    print("  CHARGEMENT DES DONNEES")
    print("=" * 72)

    X_seq         = _load_pkl(PROCESSED_DIR / "X_train_seq.pkl")
    X_behav       = _load_pkl(PROCESSED_DIR / "X_train_behav.pkl")
    X_hybrid      = _load_pkl(PROCESSED_DIR / "X_train_hybrid.pkl")
    X_test_seq    = _load_pkl(PROCESSED_DIR / "X_test_seq.pkl")
    X_test_behav  = _load_pkl(PROCESSED_DIR / "X_test_behav.pkl")
    X_test_hybrid = _load_pkl(PROCESSED_DIR / "X_test_hybrid.pkl")
    metadata      = _load_pkl(PROCESSED_DIR / "metadata.pkl")

    le = LabelEncoder()
    y_encoded = le.fit_transform(metadata["y_train"])
    n_classes = len(le.classes_)
    print(f"  {X_seq.shape[0]} train, {X_test_seq.shape[0]} test, {n_classes} classes")

    # ── 2. Évaluation de toutes les combinaisons ───────────────────────
    print("\n" + "=" * 72)
    print("  BENCHMARK : 5 Modeles x 3 Datasets (Stratified 4-Fold CV)")
    print("=" * 72)

    results, all_oof_masked, all_test_masked = evaluate_all(
        X_seq, X_behav, X_hybrid,
        X_test_seq, X_test_behav, X_test_hybrid,
        y_encoded, le, metadata,
    )

    # Sauvegarder le benchmark CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = TABLES_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n    -> {csv_path}")

    # ── 3. Tableau récapitulatif ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TABLEAU RECAPITULATIF")
    print("=" * 72)
    header = (
        f"  {'Dataset':<10s} {'Model':<10s} "
        f"{'Raw_Acc':>8s} {'Raw_F1':>8s} "
        f"{'Mask_Acc':>9s} {'Mask_F1':>8s} "
        f"{'dF1':>8s} {'Time':>7s}"
    )
    print(header)
    print("  " + "-" * 74)
    for r in results:
        print(
            f"  {r['Dataset']:<10s} {r['Model']:<10s} "
            f"{r['Raw_Accuracy']:>8.4f} {r['Raw_Macro_F1']:>8.4f} "
            f"{r['Masked_Accuracy']:>9.4f} {r['Masked_Macro_F1']:>8.4f} "
            f"{r['Gain_Macro_F1_Masking']:>+8.4f} "
            f"{r['Train_Time_Sec']:>6.1f}s"
        )

    # ── 4. Ensemble (Soft Voting Pondéré Optimal) ──────────────────────
    print("\n" + "=" * 72)
    print("  ENSEMBLE (Soft Voting Pondere Optimal)")
    print("=" * 72)

    # Sélectionner le meilleur dataset par famille de modèle
    best_per_model: dict[str, str] = {}
    for model_name in MODEL_NAMES:
        best_key = None
        best_f1  = -1.0
        for ds in DATASET_NAMES:
            key = f"{ds}_{model_name}"
            f1 = [
                r["Masked_Macro_F1"]
                for r in results
                if r["Dataset"] == ds and r["Model"] == model_name
            ][0]
            if f1 > best_f1:
                best_f1 = f1
                best_key = key
        best_per_model[model_name] = best_key
        print(
            f"    {MODEL_DISPLAY[model_name]:<15s} -> "
            f"{best_key} (F1={best_f1:.4f})"
        )

    ensemble_oof  = [all_oof_masked[best_per_model[m]]  for m in MODEL_NAMES]
    ensemble_test = [all_test_masked[best_per_model[m]] for m in MODEL_NAMES]

    # Optimiser les poids
    print("\n  Optimisation des poids (differential_evolution)...")
    opt_weights, opt_f1 = _find_optimal_weights(ensemble_oof, y_encoded)

    print("  Poids optimaux :")
    for m, w in zip(MODEL_NAMES, opt_weights):
        print(f"    {MODEL_DISPLAY[m]:<15s} : {w:.4f}")
    print(f"\n  Soft Voting Macro-F1 (OOF) : {opt_f1:.4f}")

    # Comparaison Hard Voting
    hard_acc, hard_f1 = _hard_voting_score(ensemble_oof, y_encoded)
    print(f"  Hard Voting Macro-F1 (OOF) : {hard_f1:.4f}")

    # Meilleur individuel
    best_ind_f1 = max(r["Masked_Macro_F1"] for r in results)
    best_ind = [r for r in results if r["Masked_Macro_F1"] == best_ind_f1][0]
    print(
        f"  Meilleur individuel        : "
        f"{best_ind['Dataset']}_{best_ind['Model']} "
        f"(F1={best_ind_f1:.4f})"
    )
    print(f"\n  Gain Ensemble vs Best Ind. : {opt_f1 - best_ind_f1:+.4f}")

    # ── 5. Prédictions finales ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PREDICTIONS FINALES")
    print("=" * 72)

    blended_oof  = sum(w * p for w, p in zip(opt_weights, ensemble_oof))
    blended_test = sum(w * p for w, p in zip(opt_weights, ensemble_test))

    # Vérification OOF
    oof_preds_enc = np.argmax(blended_oof, axis=1)
    oof_acc = accuracy_score(y_encoded, oof_preds_enc)
    oof_f1  = f1_score(y_encoded, oof_preds_enc, average="macro")
    print(f"  Ensemble OOF : Accuracy={oof_acc:.4f}, Macro-F1={oof_f1:.4f}")

    # Prédictions test
    final_preds_enc = np.argmax(blended_test, axis=1)
    final_preds = le.inverse_transform(final_preds_enc)

    # Fichier de soumission
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({
        "RowId": np.arange(1, len(final_preds) + 1),
        "prediction": final_preds,
    })
    sub_path = SUBMISSIONS_DIR / "submission_ensemble.csv"
    submission.to_csv(sub_path, index=False)
    print(f"    -> {sub_path}")

    # Probabilités ensemble (pour analyse d'erreurs)
    _save_pkl(
        {
            "oof_probs":  blended_oof,
            "test_probs": blended_test,
            "oof_preds":  le.inverse_transform(oof_preds_enc),
            "test_preds": final_preds,
            "weights":    dict(zip(MODEL_NAMES, opt_weights.tolist())),
            "classes":    le.classes_.tolist(),
        },
        PROCESSED_DIR / "ensemble_probs.pkl",
    )
    print(f"    -> {PROCESSED_DIR / 'ensemble_probs.pkl'}")

    # ── 6. Figures ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FIGURES")
    print("=" * 72)
    _plot_benchmark(results, opt_f1)
    _plot_masking_impact(results)

    # ── Terminé ─────────────────────────────────────────────────────────
    total_time = time.time() - t_global
    print("\n" + "=" * 72)
    print(
        f"  [OK] PIPELINE COMPLET en {total_time:.0f}s "
        f"({total_time / 60:.1f} min)"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
