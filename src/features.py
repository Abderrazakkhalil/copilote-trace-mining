"""Feature Engineering — Construction des 3 datasets de classification.

Datasets produits dans ``data/processed/`` :

1. **Sequential_NLP** (``X_train_seq.pkl``, ``X_test_seq.pkl``)
   Matrice sparse TF-IDF multi-granulaire + ancrage début/fin.

2. **Behavioral_Dense** (``X_train_behav.pkl``, ``X_test_behav.pkl``)
   Matrice dense de ~240 descripteurs comportementaux (rythme, Markov,
   RLE, fréquences relatives, diversité, ergonomie).

3. **Hybrid_Combined** (``X_train_hybrid.pkl``, ``X_test_hybrid.pkl``)
   Concaténation horizontale du Dense standardisé + Sparse.

Les constructeurs ajustent leurs transformations sur les sessions reçues en
premier argument. Le point d'entrée historique les ajuste sur tout le train
pour le benchmark exploratoire ; src.dl_hybrid_pipeline les réapprend dans
chaque pli pour l'évaluation imbriquée. Markov et RLE sont dans le bloc dense.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    TRAIN_CSV, TEST_CSV, PROCESSED_DIR, RANDOM_STATE,
)
from src.parser import load_dataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_div(a: float, b: float) -> float:
    """Division protégée contre les cas b=0."""
    return a / b if b > 0 else 0.0


def _shannon_entropy(counts: Counter) -> float:
    """Entropie de Shannon sur un compteur de fréquences."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _identity(x):
    """Identité utilisée comme preprocessor / tokenizer pour TfidfVectorizer."""
    return x


def _build_tfidf(
    train_docs: list[list[str]],
    test_docs: list[list[str]],
    *,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_features: Optional[int] = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """Construit un TF-IDF sur des listes de tokens pré-découpés.

    Utilise un ``analyzer`` passthrough pour préserver les tokens multi-mots
    contenant des espaces et caractères spéciaux.
    """
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    X_train = vec.fit_transform(train_docs)
    X_test  = vec.transform(test_docs)
    return X_train, X_test, vec


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dataset 1 : Sequential_NLP (sparse)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_sequential_nlp(
    train_sessions: list[dict],
    test_sessions: list[dict],
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict]:
    """Construit le dataset Sequential_NLP (sparse TF-IDF multi-granulaire).

    Returns
    -------
    X_train, X_test : csr_matrix
    info : dict  – nombre de features par bloc, noms des vectorizers.
    """
    blocks_train: list[sp.csr_matrix] = []
    blocks_test:  list[sp.csr_matrix] = []
    info: dict = {}

    # ── TF-IDF Tokens complets (raw_actions) ────────────────────────────
    tr, te, vec = _build_tfidf(
        [s["raw_actions"] for s in train_sessions],
        [s["raw_actions"] for s in test_sessions],
        ngram_range=(1, 2), min_df=3, max_features=8000,
    )
    blocks_train.append(tr); blocks_test.append(te)
    info["tfidf_raw"] = tr.shape[1]

    # ── TF-IDF Actions de base ──────────────────────────────────────────
    tr, te, vec = _build_tfidf(
        [s["base_actions"] for s in train_sessions],
        [s["base_actions"] for s in test_sessions],
        ngram_range=(1, 3), min_df=3, max_features=2500,
    )
    blocks_train.append(tr); blocks_test.append(te)
    info["tfidf_base"] = tr.shape[1]

    # ── TF-IDF Contrôleurs ──────────────────────────────────────────────
    tr, te, vec = _build_tfidf(
        [[c for c in s["controllers"] if c is not None] for s in train_sessions],
        [[c for c in s["controllers"] if c is not None] for s in test_sessions],
        ngram_range=(1, 2), min_df=2, max_features=3000,
    )
    blocks_train.append(tr); blocks_test.append(te)
    info["tfidf_ctrl"] = tr.shape[1]

    # ── TF-IDF Configurations ───────────────────────────────────────────
    tr, te, vec = _build_tfidf(
        [[c for c in s["configs"] if c is not None] for s in train_sessions],
        [[c for c in s["configs"] if c is not None] for s in test_sessions],
        ngram_range=(1, 2), min_df=2, max_features=1000,
    )
    blocks_train.append(tr); blocks_test.append(te)
    info["tfidf_cfg"] = tr.shape[1]

    # ── TF-IDF Domaines ─────────────────────────────────────────────────
    tr, te, vec = _build_tfidf(
        [[d for d in s["domains"] if d is not None] for s in train_sessions],
        [[d for d in s["domains"] if d is not None] for s in test_sessions],
        ngram_range=(1, 2), min_df=2, max_features=500,
    )
    blocks_train.append(tr); blocks_test.append(te)
    info["tfidf_dom"] = tr.shape[1]

    # ── Ancrage Début / Fin (One-Hot) ───────────────────────────────────
    def _anchor_features(sessions: list[dict]) -> np.ndarray:
        """Extrait les 5 ancres textuelles (str) par session."""
        rows = []
        for s in sessions:
            ba = s["base_actions"]
            ct = s["controllers"]
            first_ba  = ba[0]  if len(ba) >= 1 else "__EMPTY__"
            second_ba = ba[1]  if len(ba) >= 2 else "__EMPTY__"
            last_ba   = ba[-1] if len(ba) >= 1 else "__EMPTY__"
            first_ct  = ct[0]  if (len(ct) >= 1 and ct[0] is not None) else "__NONE__"
            last_ct   = ct[-1] if (len(ct) >= 1 and ct[-1] is not None) else "__NONE__"
            rows.append([first_ba, second_ba, last_ba, first_ct, last_ct])
        return np.array(rows, dtype=object)

    anchors_train = _anchor_features(train_sessions)
    anchors_test  = _anchor_features(test_sessions)

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=3,
        sparse_output=True,
    )
    oh_train = ohe.fit_transform(anchors_train)
    oh_test  = ohe.transform(anchors_test)
    blocks_train.append(oh_train); blocks_test.append(oh_test)
    info["onehot_anchors"] = oh_train.shape[1]

    # ── Assemblage final ────────────────────────────────────────────────
    X_train = sp.hstack(blocks_train, format="csr")
    X_test  = sp.hstack(blocks_test,  format="csr")
    info["total"] = X_train.shape[1]

    return X_train, X_test, info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dataset 2 : Behavioral_Dense (dense)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Les 4 navigateurs connus (colonnes One-Hot fixes)
BROWSERS = ["Firefox", "Google Chrome", "Microsoft Edge", "Opera"]

# Les 5 actions majeures pour le RLE détaillé
RLE_MAJOR_ACTIONS = [
    "Exécution d'un bouton",
    "Saisie dans un champ",
    "Sélection d'un écran",
    "Sélection d'un onglet",
    "Raccourci",
]


def _compute_rle(actions: list[str]) -> list[tuple[str, int]]:
    """Run-Length Encoding d'une séquence d'actions."""
    if not actions:
        return []
    runs: list[tuple[str, int]] = []
    current = actions[0]
    count = 1
    for a in actions[1:]:
        if a == current:
            count += 1
        else:
            runs.append((current, count))
            current = a
            count = 1
    runs.append((current, count))
    return runs


def _extract_dense_features(
    session: dict,
    top10_actions: list[str],
    all_base_actions: list[str],
    all_domains: list[str],
) -> np.ndarray:
    """Extrait le vecteur dense d'une session.

    Parameters
    ----------
    session : dict – sortie de ``parse_session``.
    top10_actions : list[str] – 10 actions les plus fréquentes (fit sur train).
    all_base_actions : list[str] – 35 actions de base ordonnées (fit sur train).
    all_domains : list[str] – 76 domaines ordonnés (fit sur train).
    """
    ba = session["base_actions"]
    ct = session["controllers"]
    cf = session["configs"]
    dm = session["domains"]
    wc = session["window_counts"]
    tm = session["time_markers"]
    n_act = max(len(ba), 1)  # protection division par zéro

    features: list[float] = []

    # ── A. Contexte & Volume (6 col.) ───────────────────────────────────
    browser = session["browser"]
    for b in BROWSERS:
        features.append(1.0 if browser == b else 0.0)
    features.append(float(np.log1p(len(ba))))                       # log_nb_actions
    max_time = float(tm[-1]) if tm else 0.0
    features.append(float(np.log1p(max_time)))                      # log_duree_totale

    # ── B. Dynamique temporelle par fenêtre de 5s (8 col.) ──────────────
    if wc:
        wc_arr = np.array(wc, dtype=np.float64)
        features.append(float(np.mean(wc_arr)))                     # moyenne
        features.append(float(np.median(wc_arr)))                   # médiane
        features.append(float(np.max(wc_arr)))                      # max burst
        features.append(float(np.std(wc_arr)))                      # écart-type
        features.append(float(np.percentile(wc_arr, 90)))           # Q90
        features.append(_safe_div(                                  # taux_ralentissement
            float(np.sum(wc_arr <= 1)), float(len(wc_arr))
        ))
    else:
        features.extend([0.0] * 6)

    # Taux multitache tX : proportion de marqueurs temporels consécutifs
    # identiques (deux tX de suite sans action entre eux)
    if len(tm) >= 2:
        consecutive_t = sum(
            1 for i in range(1, len(tm))
            if session["time_markers"][i] == session["time_markers"][i - 1]
        )
        features.append(_safe_div(float(consecutive_t), float(len(tm) - 1)))
    else:
        features.append(0.0)

    # Ratio activité au démarrage : actions dans les 6 premières fenêtres / total
    if wc:
        start_actions = sum(wc[:6])
        total_wc = sum(wc)
        features.append(_safe_div(float(start_actions), float(max(total_wc, 1))))
    else:
        features.append(0.0)

    # ── C. Diversité & Ergonomie (8 col.) ────────────────────────────────
    ctrl_non_none = [c for c in ct if c is not None]
    dom_non_none  = [d for d in dm if d is not None]
    cfg_non_none  = [c for c in cf if c is not None]

    features.append(float(len(set(ctrl_non_none))))                 # nb_ecrans_uniques
    features.append(float(len(set(dom_non_none))))                  # nb_domaines_uniques
    features.append(float(len(set(cfg_non_none))))                  # nb_configs_uniques
    features.append(_shannon_entropy(Counter(ba)))                  # entropie_actions
    features.append(_shannon_entropy(Counter(ctrl_non_none)))       # entropie_ecrans

    # ratio_clavier : (Raccourci + Chainage) / n_actions
    ba_counts = Counter(ba)
    n_clavier = ba_counts.get("Raccourci", 0) + ba_counts.get("Chainage", 0)
    features.append(_safe_div(float(n_clavier), float(n_act)))

    # ratio_saisie_ecran : Saisie / (1 + Sélection écran + Sélection onglet)
    n_saisie = ba_counts.get("Saisie dans un champ", 0)
    n_sel_ecran = ba_counts.get("Sélection d'un écran", 0)
    n_sel_onglet = ba_counts.get("Sélection d'un onglet", 0)  # note: l'apostrophe
    # Tenter aussi la variante typographique de l'apostrophe
    n_sel_onglet += ba_counts.get("Sélection d\u2019un onglet", 0)
    features.append(_safe_div(float(n_saisie), float(1 + n_sel_ecran + n_sel_onglet)))

    # ratio_suffixe_1
    features.append(_safe_div(float(session["suffix_ones"]), float(n_act)))

    # ── D. Run-Length Encoding / Dynamique de répétition (13 col.) ───────
    # Taux auto-répétition globale
    if len(ba) >= 2:
        n_repeat = sum(1 for i in range(1, len(ba)) if ba[i] == ba[i - 1])
        features.append(_safe_div(float(n_repeat), float(len(ba) - 1)))
    else:
        features.append(0.0)

    # RLE global : longueur moyenne et max des séries identiques consécutives
    rle = _compute_rle(ba)
    if rle:
        run_lengths = [r[1] for r in rle]
        features.append(float(np.mean(run_lengths)))                # mean run length
        features.append(float(np.max(run_lengths)))                 # max run length
    else:
        features.extend([0.0, 0.0])

    # RLE par action majeure : mean + max pour chacune des 5 actions
    for major_action in RLE_MAJOR_ACTIONS:
        major_runs = [r[1] for r in rle if r[0] == major_action]
        if major_runs:
            features.append(float(np.mean(major_runs)))
            features.append(float(np.max(major_runs)))
        else:
            features.extend([0.0, 0.0])

    # ── E. Chaînes de Markov d'ordre 1 (100 col.) ───────────────────────
    # Matrice de transition 10x10 sur les top-10 actions
    top10_idx = {a: i for i, a in enumerate(top10_actions)}
    trans_counts = np.zeros((10, 10), dtype=np.float64)
    row_totals   = np.zeros(10, dtype=np.float64)

    for i in range(len(ba) - 1):
        a_from = ba[i]
        a_to   = ba[i + 1]
        if a_from in top10_idx and a_to in top10_idx:
            trans_counts[top10_idx[a_from], top10_idx[a_to]] += 1
        if a_from in top10_idx:
            row_totals[top10_idx[a_from]] += 1

    # P(B | A) = count(A->B) / max(count(A), 1)
    for r in range(10):
        denom = max(row_totals[r], 1.0)
        trans_counts[r, :] /= denom

    features.extend(trans_counts.ravel().tolist())                  # 100 col

    # ── F. Profil de fréquences relatives (111 col.) ─────────────────────
    # 35 base_actions
    for action in all_base_actions:
        features.append(_safe_div(float(ba_counts.get(action, 0)), float(n_act)))

    # 76 domains
    dom_counts = Counter(dom_non_none)
    n_dom = max(len(dom_non_none), 1)
    for domain in all_domains:
        features.append(_safe_div(float(dom_counts.get(domain, 0)), float(n_dom)))

    return np.array(features, dtype=np.float32)


def build_behavioral_dense(
    train_sessions: list[dict],
    test_sessions: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construit le dataset Behavioral_Dense (dense, ~240 colonnes).

    Returns
    -------
    X_train, X_test : np.ndarray (float32)
    feature_names : list[str]
    """
    # ── Déterminer le top-10 actions et les vocabulaires sur TRAIN ───────
    all_ba_counter: Counter = Counter()
    all_dom_counter: Counter = Counter()
    for s in train_sessions:
        all_ba_counter.update(s["base_actions"])
        all_dom_counter.update(d for d in s["domains"] if d is not None)

    top10_actions = [a for a, _ in all_ba_counter.most_common(10)]
    all_base_actions = sorted(all_ba_counter.keys())     # 35 actions ordonnées
    all_domains      = sorted(all_dom_counter.keys())    # 76 domaines ordonnés

    # ── Construire les noms de colonnes ─────────────────────────────────
    names: list[str] = []

    # A. Contexte & Volume (6)
    names.extend([f"browser_{b}" for b in BROWSERS])
    names.extend(["log_nb_actions", "log_duree_totale"])

    # B. Dynamique temporelle (8)
    names.extend([
        "wc_mean", "wc_median", "wc_max_burst", "wc_std", "wc_q90",
        "taux_ralentissement", "taux_multitache_tX", "ratio_activite_demarrage",
    ])

    # C. Diversité & Ergonomie (8)
    names.extend([
        "nb_ecrans_uniques", "nb_domaines_uniques", "nb_configs_uniques",
        "entropie_actions", "entropie_ecrans",
        "ratio_clavier", "ratio_saisie_ecran", "ratio_suffixe_1",
    ])

    # D. RLE (13)
    names.extend(["taux_auto_repetition", "rle_mean_global", "rle_max_global"])
    for ma in RLE_MAJOR_ACTIONS:
        short = ma.split("'")[-1].strip().replace(" ", "_")[:20]
        names.extend([f"rle_mean_{short}", f"rle_max_{short}"])

    # E. Markov (100)
    for a_from in top10_actions:
        for a_to in top10_actions:
            sf = a_from[:15].replace(" ", "_")
            st = a_to[:15].replace(" ", "_")
            names.append(f"markov_{sf}_to_{st}")

    # F. Fréquences relatives (35 + 76 = 111)
    for a in all_base_actions:
        names.append(f"freq_ba_{a[:30]}")
    for d in all_domains:
        names.append(f"freq_dom_{d[:30]}")

    # ── Extraction ──────────────────────────────────────────────────────
    X_train = np.vstack([
        _extract_dense_features(s, top10_actions, all_base_actions, all_domains)
        for s in train_sessions
    ])
    X_test = np.vstack([
        _extract_dense_features(s, top10_actions, all_base_actions, all_domains)
        for s in test_sessions
    ])

    # ── Nettoyage NaN / Inf ─────────────────────────────────────────────
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    assert X_train.shape[1] == len(names), (
        f"Mismatch: {X_train.shape[1]} colonnes vs {len(names)} noms"
    )

    return X_train, X_test, names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dataset 3 : Hybrid_Combined (dense standardisé + sparse)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_hybrid(
    X_train_behav: np.ndarray,
    X_test_behav: np.ndarray,
    X_train_seq: sp.csr_matrix,
    X_test_seq: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, StandardScaler]:
    """Standardise le dense puis concatène avec le sparse.

    Returns
    -------
    X_train_hybrid, X_test_hybrid : csr_matrix
    scaler : StandardScaler (fit sur train uniquement)
    """
    scaler = StandardScaler()
    behav_train_scaled = scaler.fit_transform(X_train_behav)
    behav_test_scaled  = scaler.transform(X_test_behav)

    # Nettoyage post-scaling (au cas où des std=0 produisent des NaN)
    behav_train_scaled = np.nan_to_num(behav_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    behav_test_scaled  = np.nan_to_num(behav_test_scaled,  nan=0.0, posinf=0.0, neginf=0.0)

    X_train_hybrid = sp.hstack(
        [sp.csr_matrix(behav_train_scaled), X_train_seq], format="csr"
    )
    X_test_hybrid = sp.hstack(
        [sp.csr_matrix(behav_test_scaled), X_test_seq], format="csr"
    )
    return X_train_hybrid, X_test_hybrid, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sauvegarde
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_pickle(obj, path: Path):
    """Sauvegarde un objet au format pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _check_matrix(name: str, X):
    """Vérifie l'absence de NaN / Inf et affiche les dimensions."""
    if sp.issparse(X):
        data = X.data
    else:
        data = X.ravel()
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    shape_str = f"{X.shape[0]:>5d} x {X.shape[1]:>6d}"
    status = "OK" if (n_nan == 0 and n_inf == 0) else f"NaN={n_nan} Inf={n_inf}"
    print(f"  {name:<30s}  {shape_str}   [{status}]")
    if n_nan > 0 or n_inf > 0:
        print(f"    ⚠ ATTENTION : valeurs invalides détectées !")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t0 = time.time()

    print("=" * 72)
    print("  CHARGEMENT DES SESSIONS")
    print("=" * 72)
    train_sessions = load_dataset(TRAIN_CSV, is_train=True)
    test_sessions  = load_dataset(TEST_CSV,  is_train=False)
    print(f"  train : {len(train_sessions)} sessions")
    print(f"  test  : {len(test_sessions)} sessions")

    # ── Dataset 1 : Sequential_NLP ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DATASET 1 : Sequential_NLP (sparse TF-IDF)")
    print("=" * 72)
    X_train_seq, X_test_seq, seq_info = build_sequential_nlp(
        train_sessions, test_sessions,
    )
    for block_name, n_feat in seq_info.items():
        print(f"    {block_name:<20s} : {n_feat:>6d} features")

    # ── Dataset 2 : Behavioral_Dense ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DATASET 2 : Behavioral_Dense (dense)")
    print("=" * 72)
    X_train_behav, X_test_behav, feature_names = build_behavioral_dense(
        train_sessions, test_sessions,
    )
    print(f"    {len(feature_names)} colonnes denses")

    # ── Dataset 3 : Hybrid_Combined ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DATASET 3 : Hybrid_Combined (dense scaled + sparse)")
    print("=" * 72)
    X_train_hybrid, X_test_hybrid, scaler = build_hybrid(
        X_train_behav, X_test_behav, X_train_seq, X_test_seq,
    )

    # ── Métadonnées ─────────────────────────────────────────────────────
    y_train = np.array([s["user"] for s in train_sessions])
    browsers_train = [s["browser"] for s in train_sessions]
    browsers_test  = [s["browser"] for s in test_sessions]

    # Dictionnaire user -> navigateur exclusif (247 clés)
    user_to_browser: dict[str, str] = {}
    for s in train_sessions:
        u = s["user"]
        if u not in user_to_browser:
            user_to_browser[u] = s["browser"]

    metadata = {
        "y_train":          y_train,
        "browsers_train":   browsers_train,
        "browsers_test":    browsers_test,
        "user_to_browser":  user_to_browser,
        "feature_names_behav": feature_names,
    }

    # ── Sauvegarde ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SAUVEGARDE")
    print("=" * 72)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    _save_pickle(X_train_seq,    PROCESSED_DIR / "X_train_seq.pkl")
    _save_pickle(X_test_seq,     PROCESSED_DIR / "X_test_seq.pkl")
    _save_pickle(X_train_behav,  PROCESSED_DIR / "X_train_behav.pkl")
    _save_pickle(X_test_behav,   PROCESSED_DIR / "X_test_behav.pkl")
    _save_pickle(X_train_hybrid, PROCESSED_DIR / "X_train_hybrid.pkl")
    _save_pickle(X_test_hybrid,  PROCESSED_DIR / "X_test_hybrid.pkl")
    _save_pickle(metadata,       PROCESSED_DIR / "metadata.pkl")

    # Feature names aussi en JSON pour inspection humaine
    with open(PROCESSED_DIR / "feature_names_behav.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    for name, path in [
        ("X_train_seq",    "X_train_seq.pkl"),
        ("X_test_seq",     "X_test_seq.pkl"),
        ("X_train_behav",  "X_train_behav.pkl"),
        ("X_test_behav",   "X_test_behav.pkl"),
        ("X_train_hybrid", "X_train_hybrid.pkl"),
        ("X_test_hybrid",  "X_test_hybrid.pkl"),
        ("metadata",       "metadata.pkl"),
    ]:
        fpath = PROCESSED_DIR / path
        size_kb = fpath.stat().st_size / 1024
        print(f"    {name:<20s} -> {fpath.name}  ({size_kb:.0f} KB)")

    # ── Vérification finale ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERIFICATION FINALE (NaN / Inf)")
    print("=" * 72)
    _check_matrix("X_train_seq (sparse)",    X_train_seq)
    _check_matrix("X_test_seq (sparse)",     X_test_seq)
    _check_matrix("X_train_behav (dense)",   X_train_behav)
    _check_matrix("X_test_behav (dense)",    X_test_behav)
    _check_matrix("X_train_hybrid (sparse)", X_train_hybrid)
    _check_matrix("X_test_hybrid (sparse)",  X_test_hybrid)

    elapsed = time.time() - t0
    print(f"\n  [OK] FEATURE ENGINEERING TERMINE en {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
