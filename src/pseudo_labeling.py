"""Pseudo-Labeling pour amelioration Kaggle.

Pipeline :
1. Charge metadata, soumissions precedentes et probabilites OOF/test.
2. Filtre les sessions test par INTERSECTION des deux soumissions.
3. Re-parse train.csv + test.csv filtrees pour construire train_extended.
4. Rebuild features (TF-IDF + Behavioral) sur train_extended.
5. Re-entraine le Super-Ensemble optimise (LinearSVC + RidgeClassifier).
6. Applique le masquage navigateur et genere la soumission finale.

Usage: python -m src.pseudo_labeling
"""
from __future__ import annotations
import os, io, sys, time, pickle, warnings
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.svm import LinearSVC
from scipy.optimize import differential_evolution

from src.config import (
    PROCESSED_DIR, SUBMISSIONS_DIR,
    TRAIN_CSV, TEST_CSV, RANDOM_STATE, N_SPLITS,
)
from src.parser import load_dataset
from src.features import build_sequential_nlp, build_behavioral_dense
from src.postprocess import apply_browser_mask, decision_to_proba

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _sec(title):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72, flush=True)

def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _oof_eval(factory, X_train, X_test, y_enc, classes, b_tr, b_te, u2b,
              n_cls, label, is_svm=False, temp=2.0):
    """Cross-validation 4-fold OOF + test-proba avec masquage navigateur."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y_enc), n_cls))
    folds = []
    for tri, vai in skf.split(X_train, y_enc):
        Xtr = X_train[tri]; Xva = X_train[vai]
        m = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, y_enc[tri])
        if is_svm:
            oof[vai] = decision_to_proba(m.decision_function(Xva), temperature=temp)
            folds.append(decision_to_proba(m.decision_function(X_test), temperature=temp))
        else:
            vp = m.predict_proba(Xva)
            tp = m.predict_proba(X_test)
            if vp.shape[1] < n_cls:
                fv = np.zeros((len(vai), n_cls)); ft = np.zeros((X_test.shape[0], n_cls))
                for i, c in enumerate(m.classes_):
                    fv[:, int(c)] = vp[:, i]; ft[:, int(c)] = tp[:, i]
                vp, tp = fv, ft
            oof[vai] = vp; folds.append(tp)
    t_prob = np.mean(folds, axis=0)
    oom = apply_browser_mask(oof,    b_tr, classes, u2b)
    tem = apply_browser_mask(t_prob, b_te, classes, u2b)
    raw_f1 = f1_score(y_enc, np.argmax(oof, axis=1), average="macro")
    msk_f1 = f1_score(y_enc, np.argmax(oom, axis=1), average="macro")
    print(f"    {label:<42s}  Raw={raw_f1:.4f}  Masked={msk_f1:.4f}", flush=True)
    return oom, tem, raw_f1, msk_f1


def _opt_weights(oof_list, y):
    """Optimisation differentielle des poids de l'ensemble."""
    k = len(oof_list)
    def neg(w):
        w = np.asarray(w); w = w / w.sum()
        return -f1_score(y, np.argmax(sum(wi*pi for wi, pi in zip(w, oof_list)), axis=1), average="macro")
    r = differential_evolution(neg, [(0.001, 1.)] * k, seed=RANDOM_STATE, maxiter=200, tol=1e-6, popsize=20)
    w = r.x / r.x.sum()
    return w, -r.fun


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    t0g = time.time()

    # ── 1. Chargement ────────────────────────────────────────────────────────
    _sec("ETAPE 1 — Chargement des donnees et probabilites")

    meta      = _load(PROCESSED_DIR / "metadata.pkl")
    y_raw     = meta["y_train"]
    btr_orig  = meta["browsers_train"]
    bte       = meta["browsers_test"]
    u2b       = meta["user_to_browser"]

    sub_init = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble.csv")
    sub_opt  = pd.read_csv(SUBMISSIONS_DIR / "submission_ensemble_optimized.csv")
    assert len(sub_init) == len(sub_opt) == len(bte)

    ep              = _load(PROCESSED_DIR / "ensemble_probs.pkl")
    test_probs_init = ep["test_probs"]   # (324, 247)

    le      = LabelEncoder().fit(y_raw)
    classes = le.classes_
    n_cls   = len(classes)
    y_enc   = le.transform(y_raw)

    print(f"  Train original : {len(y_raw)} sessions, {n_cls} utilisateurs", flush=True)
    print(f"  Test           : {len(bte)} sessions", flush=True)

    # ── 2. Filtrage par Intersection + Seuil de Confiance ───────────────────
    _sec("ETAPE 2 — Filtrage Intersection + Seuil de Confiance")

    THRESHOLD  = 0.97
    preds_init = sub_init["prediction"].values
    preds_opt  = sub_opt["prediction"].values
    max_p_test = test_probs_init.max(axis=1)

    mask_agree = (preds_init == preds_opt)
    mask_conf  = (max_p_test > THRESHOLD)
    # Critere principal : intersection parfaite des deux ensembles
    mask_pl    = mask_agree

    print(f"  Intersection (accord des 2 ensembles)       : {mask_agree.sum()} / {len(bte)} ({100*mask_agree.mean():.1f}%)", flush=True)
    print(f"  Seuil confiance > {THRESHOLD}               : {mask_conf.sum()} / {len(bte)} ({100*mask_conf.mean():.1f}%)", flush=True)
    print(f"  => Pseudo-labels retenus (intersection)     : {mask_pl.sum()} / {len(bte)} ({100*mask_pl.mean():.1f}%)", flush=True)

    pseudo_labels   = preds_opt[mask_pl]
    pseudo_browsers = np.array(bte)[mask_pl]

    print(f"\n  Distribution pseudo-labels (top 10) :", flush=True)
    for user, freq in Counter(pseudo_labels).most_common(10):
        print(f"    {user}: {freq}", flush=True)

    # Verification : les 6 desaccords ont une confiance tres faible
    print(f"\n  Confiance moy. (sessions concordantes)   : {max_p_test[mask_agree].mean():.4f}", flush=True)
    print(f"  Confiance moy. (sessions discordantes)   : {max_p_test[~mask_agree].mean():.4f}", flush=True)

    # ── 3. Construction du Train Elargi ──────────────────────────────────────
    _sec("ETAPE 3 — Construction du Train Elargi (Pseudo-Labeling)")

    print("  Parsing train.csv...", flush=True)
    train_sessions_orig = load_dataset(TRAIN_CSV, is_train=True)
    print(f"    {len(train_sessions_orig)} sessions", flush=True)

    print("  Parsing test.csv...", flush=True)
    test_sessions_all = load_dataset(TEST_CSV, is_train=False)
    print(f"    {len(test_sessions_all)} sessions", flush=True)

    # Assigner les pseudo-labels aux sessions test selectionnees
    test_idx_pl = np.where(mask_pl)[0]
    pl_sessions = []
    for i, idx in enumerate(test_idx_pl):
        s = dict(test_sessions_all[idx])
        s["user"] = pseudo_labels[i]
        pl_sessions.append(s)

    train_ext = train_sessions_orig + pl_sessions
    browsers_ext = list(btr_orig) + list(pseudo_browsers)
    y_ext_raw = np.array([s["user"] for s in train_ext])
    y_ext_enc = le.transform(y_ext_raw)

    print(f"\n  Train original : {len(train_sessions_orig)} sessions", flush=True)
    print(f"  Pseudo-labels  : {len(pl_sessions)} sessions (+{100*len(pl_sessions)/len(train_sessions_orig):.1f}%)", flush=True)
    print(f"  Train elargi   : {len(train_ext)} sessions total", flush=True)

    # ── 4. Reconstruction des Features ──────────────────────────────────────
    _sec("ETAPE 4 — Reconstruction des Features (TF-IDF + Behavioral)")

    print("  4a. Sequential TF-IDF...", flush=True)
    t_a = time.time()
    X_ext_seq, X_test_seq, _ = build_sequential_nlp(train_ext, test_sessions_all)
    print(f"    X_ext_seq : {X_ext_seq.shape}  [{time.time()-t_a:.1f}s]", flush=True)

    print("  4b. Behavioral Dense...", flush=True)
    t_b = time.time()
    X_ext_behav, X_test_behav, _ = build_behavioral_dense(train_ext, test_sessions_all)
    print(f"    X_ext_behav: {X_ext_behav.shape}  [{time.time()-t_b:.1f}s]", flush=True)

    print("  4c. Block L2-Normalisation (alpha=0.25)...", flush=True)
    sc = StandardScaler()
    Xbts = sc.fit_transform(X_ext_behav)
    Xbes = sc.transform(X_test_behav)
    Xbtl2 = normalize(Xbts, norm="l2")
    Xbel2 = normalize(Xbes, norm="l2")
    alpha = 0.25
    X_ext_hyb  = sp.hstack([X_ext_seq, alpha * sp.csr_matrix(Xbtl2)]).tocsr()
    X_test_hyb = sp.hstack([X_test_seq, alpha * sp.csr_matrix(Xbel2)]).tocsr()
    print(f"    X_ext_hyb  : {X_ext_hyb.shape}", flush=True)
    print(f"    X_test_hyb : {X_test_hyb.shape}", flush=True)

    # ── 5. Re-entrainement Super-Ensemble ────────────────────────────────────
    _sec("ETAPE 5 — Re-entrainement Super-Ensemble sur Train Elargi")

    pool = {}

    def run(lbl, Xtr, Xte, factory, is_svm=False, temp=2.0):
        t0 = time.time()
        print(f"  [{lbl}]", end=" ", flush=True)
        oom, tem, rf, mf = _oof_eval(
            factory, Xtr, Xte, y_ext_enc, classes,
            browsers_ext, bte, u2b, n_cls,
            label=lbl, is_svm=is_svm, temp=temp)
        print(f"       [{time.time()-t0:.1f}s]", flush=True)
        pool[lbl] = (oom, tem, mf)

    run("svm_seq_C0.5_T2.5", X_ext_seq, X_test_seq,
        lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        is_svm=True, temp=2.5)

    run("svm_hyb_C0.5_T2.5", X_ext_hyb, X_test_hyb,
        lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        is_svm=True, temp=2.5)

    run("ridge_hyb_T2.0", X_ext_hyb, X_test_hyb,
        lambda: RidgeClassifier(alpha=1.0, class_weight="balanced"),
        is_svm=True, temp=2.0)

    run("logreg_hyb_C20", X_ext_hyb, X_test_hyb,
        lambda: LogisticRegression(C=20.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=300,
                                   random_state=RANDOM_STATE))

    # ── 6. Optimisation des Poids ────────────────────────────────────────────
    _sec("ETAPE 6 — Optimisation des Poids (Differential Evolution)")

    cands     = ["svm_seq_C0.5_T2.5", "svm_hyb_C0.5_T2.5", "ridge_hyb_T2.0", "logreg_hyb_C20"]
    oof_list  = [pool[l][0] for l in cands]
    test_list = [pool[l][1] for l in cands]
    f1_list   = [pool[l][2] for l in cands]

    print("  Candidats :", flush=True)
    for l, f in zip(cands, f1_list):
        print(f"    {l:<42s} F1={f:.4f}", flush=True)

    print("\n  Optimisation...", flush=True)
    opt_w, opt_f1 = _opt_weights(oof_list, y_ext_enc)

    print("\n  Poids optimaux :", flush=True)
    for l, w in zip(cands, opt_w):
        print(f"    {l:<42s} w={w:.4f}", flush=True)

    F1_REF = 0.9794
    print(f"\n  OOF Macro-F1 (train elargi) : {opt_f1:.4f}", flush=True)
    print(f"  Reference (sans PL)         : {F1_REF:.4f}", flush=True)
    print(f"  Gain OOF estime             : {opt_f1 - F1_REF:+.4f}", flush=True)

    blended_oof  = sum(w * p for w, p in zip(opt_w, oof_list))
    blended_test = sum(w * p for w, p in zip(opt_w, test_list))
    print(f"\n  OOF final : Acc={accuracy_score(y_ext_enc, np.argmax(blended_oof,axis=1)):.4f}"
          f"  F1={f1_score(y_ext_enc, np.argmax(blended_oof,axis=1), average='macro'):.4f}", flush=True)

    # ── 7. Post-Processing & Soumission Finale ───────────────────────────────
    _sec("ETAPE 7 — Post-Processing et Generation de la Soumission Finale")

    blended_test_masked = apply_browser_mask(blended_test, bte, classes, u2b)
    final_preds = le.inverse_transform(np.argmax(blended_test_masked, axis=1))

    # Verification contrainte navigateur
    violations = sum(
        1 for pred, browser in zip(final_preds, bte)
        if u2b.get(pred) and u2b[pred] != browser
    )
    print(f"  Violations navigateur apres masquage : {violations}", flush=True)
    assert violations == 0

    # Comparaison avec les soumissions precedentes
    print(f"  Accord vs submission_ensemble.csv           : "
          f"{(final_preds == preds_init).sum()} / {len(final_preds)}", flush=True)
    print(f"  Accord vs submission_ensemble_optimized.csv : "
          f"{(final_preds == preds_opt).sum()} / {len(final_preds)}", flush=True)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "submission_pseudo_labeled.csv"
    pd.DataFrame({
        "RowId": np.arange(1, len(final_preds) + 1),
        "prediction": final_preds,
    }).to_csv(sub_path, index=False)
    print(f"\n  -> {sub_path}", flush=True)

    # ── 8. Resume ────────────────────────────────────────────────────────────
    _sec("RESUME")
    print(f"  Pseudo-labels ajoutes           : {len(pl_sessions)} / {len(bte)}"
          f"  ({100*len(pl_sessions)/len(bte):.1f}% du test set)", flush=True)
    print(f"  Taille train elargi             : {len(train_ext)} sessions", flush=True)
    print(f"  OOF Macro-F1 (train elargi)     : {opt_f1:.4f}", flush=True)
    print(f"  Reference super_ensemble (orig) : {F1_REF:.4f}", flush=True)
    print(f"  Gain OOF estime (PL)            : {opt_f1 - F1_REF:+.4f}", flush=True)

    dist = Counter(final_preds)
    print(f"\n  Distribution predictions test (top 10) :", flush=True)
    for user, freq in dist.most_common(10):
        print(f"    {user}: {freq}", flush=True)
    print(f"  Classes predites distinctes : {len(dist)} / {n_cls}", flush=True)

    # Sauvegarde probs
    _save({
        "oof_probs_ext":  blended_oof,
        "test_probs_ext": blended_test_masked,
        "test_preds":     final_preds,
        "weights":        dict(zip(cands, opt_w.tolist())),
        "classes":        classes.tolist(),
        "n_pseudo":       int(len(pl_sessions)),
        "oof_f1":         float(opt_f1),
    }, PROCESSED_DIR / "pseudo_ensemble_probs.pkl")
    print(f"\n  -> data/processed/pseudo_ensemble_probs.pkl", flush=True)

    total = time.time() - t0g
    _sec(f"[OK] COMPLET en {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
