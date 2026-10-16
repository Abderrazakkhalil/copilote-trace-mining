"""Optimisation post-benchmark : L2-Normalization par bloc, Tuning HP, Super-Ensemble.

Usage: python -m src.optimize_models
"""
from __future__ import annotations
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys, io, pickle, time, warnings
# Force line-buffering so background task logs are flushed immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.svm import LinearSVC
from src.config import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SUBMISSIONS_DIR, RANDOM_STATE, N_SPLITS
from src.postprocess import apply_browser_mask, decision_to_proba

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def _load(p):
    with open(p, "rb") as f: return pickle.load(f)

def _save(o, p):
    with open(p, "wb") as f: pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)

def _sec(t): print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72, flush=True)


def _oof_eval(factory, X_train, X_test, y_encoded, classes, b_tr, b_te, u2b,
              n_cls, label, is_svm=False, temp=2.0, needs_scale=False, needs_kbest=0):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y_encoded), n_cls))
    folds = []
    for tri, vai in skf.split(X_train, y_encoded):
        Xtr = X_train[tri]; Xva = X_train[vai]; Xte = X_test
        if needs_scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr); Xva = sc.transform(Xva); Xte = sc.transform(Xte)
        if needs_kbest > 0:
            sel = SelectKBest(f_classif, k=min(needs_kbest, Xtr.shape[1]))
            Xtr = sel.fit_transform(Xtr, y_encoded[tri])
            Xva = sel.transform(Xva); Xte = sel.transform(Xte)
        m = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, y_encoded[tri])
        if is_svm:
            oof[vai] = decision_to_proba(m.decision_function(Xva), temperature=temp)
            folds.append(decision_to_proba(m.decision_function(Xte), temperature=temp))
        else:
            vp = m.predict_proba(Xva); tp = m.predict_proba(Xte)
            if vp.shape[1] < n_cls:
                fv = np.zeros((len(vai), n_cls)); ft = np.zeros((Xte.shape[0], n_cls))
                for i, c in enumerate(m.classes_): fv[:, int(c)] = vp[:, i]; ft[:, int(c)] = tp[:, i]
                vp, tp = fv, ft
            oof[vai] = vp; folds.append(tp)
    tprob = np.mean(folds, axis=0)
    rf = f1_score(y_encoded, np.argmax(oof, axis=1), average="macro")
    oom = apply_browser_mask(oof, b_tr, classes, u2b)
    tem = apply_browser_mask(tprob, b_te, classes, u2b)
    mf = f1_score(y_encoded, np.argmax(oom, axis=1), average="macro")
    print(f"    {label:<42s}  Raw={rf:.4f}  Masked={mf:.4f}", flush=True)
    return oom, tem, rf, mf


def _opt_w(oof_list, y):
    k = len(oof_list)
    def neg(w):
        w = np.asarray(w); w = w / w.sum()
        return -f1_score(y, np.argmax(sum(wi * pi for wi, pi in zip(w, oof_list)), axis=1), average="macro")
    r = differential_evolution(neg, [(0.001, 1.)] * k, seed=RANDOM_STATE, maxiter=200, tol=1e-6, popsize=20)
    w = r.x / r.x.sum()
    return w, -r.fun


def main():
    t0g = time.time(); rows = []
    _sec("CHARGEMENT")
    meta = _load(PROCESSED_DIR / "metadata.pkl")
    Xst = _load(PROCESSED_DIR / "X_train_seq.pkl")
    Xse = _load(PROCESSED_DIR / "X_test_seq.pkl")
    Xbtr = _load(PROCESSED_DIR / "X_train_behav.pkl")
    Xbte = _load(PROCESSED_DIR / "X_test_behav.pkl")
    y_raw = meta["y_train"]; btr = meta["browsers_train"]
    bte = meta["browsers_test"]; u2b = meta["user_to_browser"]
    le = LabelEncoder(); ye = le.fit_transform(y_raw)
    cls = le.classes_; nc = len(cls)
    print(f"  {len(ye)} train, n_classes={nc}")
    print(f"  behav L2-norm brute (mean): {np.linalg.norm(Xbtr, axis=1).mean():.2f}")
    BF = 0.9620; BE = 0.9660

    _sec("ETAPE 1 - L2-Normalization Hybrid")
    sc_beh = StandardScaler()
    Xbts = sc_beh.fit_transform(Xbtr); Xbes = sc_beh.transform(Xbte)
    Xbtl2 = normalize(Xbts, norm="l2"); Xbel2 = normalize(Xbes, norm="l2")
    print(f"  behav L2 apres norm (mean): {np.linalg.norm(Xbtl2, axis=1).mean():.4f}")
    print(f"  seq   L2 mean            : {sp.linalg.norm(Xst, axis=1).mean():.4f}")
    print("  Recherche alpha (1-fold rapide)...")
    skf1 = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    tri0, vai0 = next(iter(skf1.split(Xst, ye)))
    ba, bf, ar = 0.25, -1.0, []
    for alpha in [0.15, 0.25, 0.35]:
        Htr = sp.hstack([Xst[tri0], alpha * sp.csr_matrix(Xbtl2[tri0])]).tocsr()
        Hva = sp.hstack([Xst[vai0], alpha * sp.csr_matrix(Xbtl2[vai0])]).tocsr()
        svm = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); svm.fit(Htr, ye[tri0])
        f1 = f1_score(ye[vai0], svm.predict(Hva), average="macro")
        ar.append((alpha, f1))
        print(f"    alpha={alpha:.2f}  val_F1={f1:.4f}")
        if f1 > bf: bf = f1; ba = alpha
    print(f"  => best alpha={ba}  F1={bf:.4f}")
    Xhtl2 = sp.hstack([Xst, ba * sp.csr_matrix(Xbtl2)]).tocsr()
    Xhel2 = sp.hstack([Xse, ba * sp.csr_matrix(Xbel2)]).tocsr()
    _save(Xhtl2, PROCESSED_DIR / "X_train_hybrid_l2.pkl")
    _save(Xhel2, PROCESSED_DIR / "X_test_hybrid_l2.pkl")
    print("  hybrid_l2 matrices saved.")
    rows += [
        {"Etape": "Baseline", "Model": "seq_svm", "Masked_F1": BF, "Note": "benchmark"},
        {"Etape": "Baseline", "Model": "ensemble_init", "Masked_F1": BE, "Note": "benchmark"},
        {"Etape": "Etape1", "Model": f"svm_hybrid_l2_a={ba}", "Masked_F1": bf, "Note": "1-fold"},
    ]

    _sec("ETAPE 2 - Tuning HP")
    pool = {}; ci = [0]

    def run(lbl, Xtr, Xte, fac, is_svm=False, temp=2.0, nk=0):
        ci[0] += 1; t0 = time.time()
        print(f"  [{ci[0]:02d}]", end=" ", flush=True)
        om, tm, rf, mf = _oof_eval(
            fac, Xtr, Xte, ye, cls, btr, bte, u2b, nc,
            label=lbl, is_svm=is_svm, temp=temp, needs_kbest=nk)
        print(f"       [{time.time()-t0:.1f}s]", flush=True)
        pool[lbl] = (om, tm, mf)
        rows.append({"Etape": "Tuning", "Model": lbl, "Masked_F1": round(mf, 4), "Note": ""})
        return om, tm, mf

    # Grid reduite : 3C x 2T = 6 combos par dataset (budget temps ~3min total)
    C_GRID = [0.5, 1.0, 2.0]
    T_GRID = [2.5, 6.0]   # T=2.5 (calibration douce) vs T=6.0 (piquee)

    print("\n  2a. LinearSVC seq (C x Temp):", flush=True)
    sseq = {}
    for C in C_GRID:
        for T in T_GRID:
            _, _, mf = run(f"svm_seq_C{C}_T{T}", Xst, Xse,
                lambda C=C: LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE),
                is_svm=True, temp=T)
            sseq[(C, T)] = mf
    bsk = max(sseq, key=sseq.get)
    print(f"\n  => Best SVM seq: C={bsk[0]},T={bsk[1]},F1={sseq[bsk]:.4f}", flush=True)

    print("\n  2b. LinearSVC hybrid_l2 (C x Temp):", flush=True)
    shyb = {}
    for C in C_GRID:
        for T in T_GRID:
            _, _, mf = run(f"svm_hyb_C{C}_T{T}", Xhtl2, Xhel2,
                lambda C=C: LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE),
                is_svm=True, temp=T)
            shyb[(C, T)] = mf
    bhk = max(shyb, key=shyb.get)
    print(f"\n  => Best SVM hyb: C={bhk[0]},T={bhk[1]},F1={shyb[bhk]:.4f}", flush=True)

    print("\n  2c. RidgeClassifier hybrid_l2:")
    for T in [2.0, 4.0]:
        run(f"ridge_hyb_T{T}", Xhtl2, Xhel2,
            lambda: RidgeClassifier(alpha=1.0, class_weight="balanced"),
            is_svm=True, temp=T)

    print("\n  2d. LogisticRegression (C eleve):")
    for C in [20.0, 40.0]:
        run(f"logreg_seq_C{C}", Xst, Xse,
            lambda C=C: LogisticRegression(
                C=C, class_weight="balanced", solver="lbfgs",
                max_iter=300, random_state=RANDOM_STATE))
        run(f"logreg_hyb_C{C}", Xhtl2, Xhel2,
            lambda C=C: LogisticRegression(
                C=C, class_weight="balanced", solver="lbfgs",
                max_iter=300, random_state=RANDOM_STATE))

    print("\n  2e. MLP + SelectKBest(6000) hybrid_l2:")
    run("mlp_hyb_kbest6k", Xhtl2, Xhel2,
        lambda: MLPClassifier(
            hidden_layer_sizes=(384, 192), alpha=5e-3,
            max_iter=45, early_stopping=True, random_state=RANDOM_STATE),
        nk=6000)

    _sec("ETAPE 3 - Super-Ensemble")
    bsl = f"svm_seq_C{bsk[0]}_T{bsk[1]}"
    bhl = f"svm_hyb_C{bhk[0]}_T{bhk[1]}"
    bri = max([k for k in pool if k.startswith("ridge_")], key=lambda k: pool[k][2])
    blr = max([k for k in pool if k.startswith("logreg_")], key=lambda k: pool[k][2])
    mk = "mlp_hyb_kbest6k"
    cands = [bsl, bhl, bri, blr, mk]
    ofl = [pool[l][0] for l in cands]
    tfl = [pool[l][1] for l in cands]
    f1l = [pool[l][2] for l in cands]
    print("\n  Candidats :")
    for l, f in zip(cands, f1l): print(f"    {l:<45s} F1={f:.4f}")
    print("\n  Optimisation poids...")
    ow, of1 = _opt_w(ofl, ye)
    print("\n  Poids :")
    for l, w in zip(cands, ow): print(f"    {l:<45s} w={w:.4f}")
    print(f"\n  Soft-Voting Macro-F1 (OOF): {of1:.4f}  (gain vs baseline: {of1 - BE:+.4f})")
    bo = sum(w * p for w, p in zip(ow, ofl))
    bt = sum(w * p for w, p in zip(ow, tfl))
    oa = accuracy_score(ye, np.argmax(bo, axis=1))
    of = f1_score(ye, np.argmax(bo, axis=1), average="macro")
    print(f"  OOF final: Acc={oa:.4f} F1={of:.4f}")
    fp = le.inverse_transform(np.argmax(bt, axis=1))
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sp2 = SUBMISSIONS_DIR / "submission_ensemble_optimized.csv"
    pd.DataFrame({
        "RowId": np.arange(1, len(fp) + 1),
        "prediction": fp
    }).to_csv(sp2, index=False)
    print(f"  -> {sp2}")
    rows.append({"Etape": "Super-Ensemble", "Model": "super_ensemble",
                 "Masked_F1": round(of1, 4),
                 "Note": str(dict(zip(cands, ow.round(3).tolist())))})

    adf = pd.DataFrame(rows)
    ap = TABLES_DIR / "optimization_ablation.csv"
    adf.to_csv(ap, index=False)
    _sec("ABLATION"); print(adf.to_string(index=False)); print(f"\n  -> {ap}")

    # Figure 8
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,
        "xtick.labelsize": 9, "ytick.labelsize": 10, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimization: L2-Block Norm & Tuning", fontsize=15, fontweight="bold")
    ax = axes[0]; av = [r[0] for r in ar]; fv = [r[1] for r in ar]
    bars = ax.bar([str(a) for a in av], fv, color=["#4C72B0", "#55A868", "#DD8452"])
    ax.axhline(0.8754, color="red", linestyle="--", lw=1.5, label="hybrid_svm base (0.8754)")
    ax.axhline(BF, color="purple", linestyle=":", lw=1.5, label=f"seq_svm ({BF})")
    [ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f"{f:.4f}",
             ha="center", va="bottom", fontsize=9) for b, f in zip(bars, fv)]
    ax.set_xlabel("Alpha"); ax.set_ylabel("F1 (1-fold)")
    ax.set_title("Alpha search - Hybrid L2"); ax.legend(fontsize=8)
    ax.set_ylim(max(0, min(fv) - 0.05), max(fv) + 0.05)
    ax = axes[1]
    steps = ["seq_svm\nbase", "SVM seq\ntuned", "SVM hyb\ntuned",
             "Best\nLogReg", "MLP\nkbest6k", "Super\nEnsemble"]
    scores = [BF, sseq[bsk], shyb[bhk], pool[blr][2], pool[mk][2], of1]
    colors = ["#AAAAAA", "#4C72B0", "#55A868", "#DD8452", "#8172B2", "#C44E52"]
    bars2 = ax.bar(steps, scores, color=colors, edgecolor="white", linewidth=0.5)
    [ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001, f"{s:.4f}",
             ha="center", va="bottom", fontsize=8.5) for b, s in zip(bars2, scores)]
    ax.axhline(BE, color="black", linestyle="--", lw=1.5, label=f"Ensemble init ({BE})")
    ax.set_ylabel("Masked Macro-F1 (OOF)")
    ax.set_title("Progression - Tuning & Super-Ensemble"); ax.legend(fontsize=8)
    ax.set_ylim(min(scores) - 0.05, max(scores) + 0.025)
    plt.tight_layout()
    fp2 = FIGURES_DIR / "fig8_hybrid_l2_and_tuning_gain.png"
    fig.savefig(fp2); plt.close()
    print(f"\n  -> {fp2}")
    total = time.time() - t0g
    _sec(f"[OK] COMPLET en {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
