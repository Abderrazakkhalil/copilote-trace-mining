"""Analyse exploratoire methodologique en 5 etapes.

Demarche : [Question] -> [Mesure] -> [Constat] -> [Consequence pipeline ML]

Etape 1 – Audit macroscopique & heterogeneite des sequences
Etape 2 – Distribution de la variable cible Y (utilisateurs)
Etape 3 – Variable contextuelle Navigateur & alignement Train/Test
Etape 4 – Sparsite lexicale, OOV & gain de la decomposition grammaticale
Etape 5 – Dynamique sequentielle et temporelle
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.config import (
    TRAIN_CSV, TEST_CSV, FIGURES_DIR, TABLES_DIR,
)
from src.parser import load_dataset

# ── Style global des figures ────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

COLORS = {
    "train": "#2563EB",   # bleu
    "test":  "#F59E0B",   # ambre
    "accent": "#10B981",  # emeraude
    "warn":  "#EF4444",   # rouge
    "gray":  "#6B7280",
    "purple": "#8B5CF6",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _descriptive(arr: np.ndarray) -> dict:
    """Statistiques descriptives completes."""
    return {
        "min":    int(np.min(arr)),
        "Q1":     int(np.percentile(arr, 25)),
        "median": int(np.median(arr)),
        "mean":   round(float(np.mean(arr)), 1),
        "std":    round(float(np.std(arr)), 1),
        "P95":    int(np.percentile(arr, 95)),
        "max":    int(np.max(arr)),
    }


def _save_fig(fig: plt.Figure, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.relative_to(FIGURES_DIR.parent.parent)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etape 1 : Audit Macroscopique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step1_macro_structure(train, test):
    """Dimension du dataset et variabilite de longueur des traces."""
    print("\n" + "=" * 72)
    print("  ETAPE 1 : Audit Macroscopique & Heterogeneite des Sequences")
    print("=" * 72)

    metrics = {}

    # Longueur totale (actions + marqueurs temporels)
    train_total_len = np.array([len(s["raw_actions"]) + len(s["time_markers"]) for s in train])
    test_total_len  = np.array([len(s["raw_actions"]) + len(s["time_markers"]) for s in test])

    # Nombre d'actions IHM pures
    train_n_act = np.array([len(s["raw_actions"]) for s in train])
    test_n_act  = np.array([len(s["raw_actions"]) for s in test])

    metrics["n_sessions_train"] = len(train)
    metrics["n_sessions_test"]  = len(test)
    metrics["total_length_train"] = _descriptive(train_total_len)
    metrics["total_length_test"]  = _descriptive(test_total_len)
    metrics["n_actions_train"] = _descriptive(train_n_act)
    metrics["n_actions_test"]  = _descriptive(test_n_act)

    # Sessions pathologiques
    short_train = int(np.sum(train_n_act < 5))
    short_test  = int(np.sum(test_n_act < 5))
    zero_act_train = int(np.sum(train_n_act == 0))
    zero_act_test  = int(np.sum(test_n_act == 0))

    metrics["pathological"] = {
        "short_lt5_train": short_train,
        "short_lt5_test":  short_test,
        "zero_actions_train": zero_act_train,
        "zero_actions_test":  zero_act_test,
    }

    print(f"  Sessions : train={len(train)}, test={len(test)}")
    print(f"  Longueur totale (train) : {metrics['total_length_train']}")
    print(f"  Actions IHM  (train)    : {metrics['n_actions_train']}")
    print(f"  Actions IHM  (test)     : {metrics['n_actions_test']}")
    print(f"  Pathologiques (<5 act)  : train={short_train}, test={short_test}")
    print(f"  Zero actions            : train={zero_act_train}, test={zero_act_test}")

    # ── Figure 1 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, label, color in [
        (axes[0], train_n_act, "Train", COLORS["train"]),
        (axes[1], test_n_act,  "Test",  COLORS["test"]),
    ]:
        log_data = np.log10(data[data > 0])
        ax.hist(log_data, bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        med = np.median(data)
        moy = np.mean(data)
        ax.axvline(np.log10(med), color=COLORS["warn"], ls="--", lw=1.5,
                   label=f"Mediane = {int(med)}")
        ax.axvline(np.log10(moy), color=COLORS["accent"], ls="-.", lw=1.5,
                   label=f"Moyenne = {moy:.0f}")
        ax.set_xlabel("Nombre d'actions IHM (echelle log10)")
        ax.set_ylabel("Nombre de sessions")
        ax.set_title(f"{label} (n={len(data)})")
        ax.legend(fontsize=8)

    fig.suptitle("Fig. 1 - Distribution du nombre d'actions par session", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "fig1_sequence_lengths.png")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etape 2 : Variable Cible Y
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step2_target_distribution(train):
    """Equilibre des 247 classes (utilisateurs)."""
    print("\n" + "=" * 72)
    print("  ETAPE 2 : Analyse de la Variable Cible Y")
    print("=" * 72)

    metrics = {}
    user_counts = Counter(s["user"] for s in train)
    counts = np.array(sorted(user_counts.values(), reverse=True))

    metrics["n_classes"] = len(user_counts)
    metrics["sessions_per_user"] = _descriptive(counts)
    metrics["imbalance_ratio"] = round(float(counts.max()) / float(counts.min()), 1)

    rare = int(np.sum(counts <= 5))
    frequent = int(np.sum(counts >= 30))
    metrics["users_rare_le5"]      = rare
    metrics["users_rare_le5_pct"]  = round(100.0 * rare / len(counts), 1)
    metrics["users_frequent_ge30"]     = frequent
    metrics["users_frequent_ge30_pct"] = round(100.0 * frequent / len(counts), 1)

    print(f"  Classes uniques         : {metrics['n_classes']}")
    print(f"  Sessions/utilisateur    : {metrics['sessions_per_user']}")
    print(f"  Ratio desequilibre      : {metrics['imbalance_ratio']}x")
    print(f"  Rares (<= 5 sessions)   : {rare} ({metrics['users_rare_le5_pct']}%)")
    print(f"  Frequents (>= 30 sess.) : {frequent} ({metrics['users_frequent_ge30_pct']}%)")

    # ── Figure 2 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gauche : histogramme + KDE du nombre de sessions par utilisateur
    ax = axes[0]
    ax.hist(counts, bins=30, color=COLORS["train"], alpha=0.7, edgecolor="white", linewidth=0.5)
    med_val = np.median(counts)
    mean_val = np.mean(counts)
    min_val = counts.min()
    ax.axvline(min_val, color=COLORS["warn"], ls=":", lw=2,
               label=f"Min = {min_val} (seuil critique)")
    ax.axvline(med_val, color=COLORS["accent"], ls="--", lw=1.5,
               label=f"Mediane = {int(med_val)}")
    ax.axvline(mean_val, color=COLORS["purple"], ls="-.", lw=1.5,
               label=f"Moyenne = {mean_val:.1f}")
    ax.set_xlabel("Nombre de sessions par utilisateur")
    ax.set_ylabel("Nombre d'utilisateurs")
    ax.set_title("Distribution des sessions / utilisateur")
    ax.legend(fontsize=8)

    # Droite : courbe de Lorenz
    ax = axes[1]
    sorted_asc = np.sort(counts)
    cumsum = np.cumsum(sorted_asc)
    lorenz = cumsum / cumsum[-1]
    x_axis = np.arange(1, len(lorenz) + 1) / len(lorenz)

    ax.plot(x_axis, lorenz, color=COLORS["train"], lw=2, label="Courbe de Lorenz")
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=1, label="Egalite parfaite")
    ax.fill_between(x_axis, lorenz, x_axis, alpha=0.15, color=COLORS["train"])

    # Gini
    _trapz = getattr(np, "trapezoid", np.trapz)
    gini = 1 - 2 * _trapz(lorenz, x_axis)
    ax.text(0.15, 0.75, f"Gini = {gini:.3f}", fontsize=11, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("Proportion cumulee d'utilisateurs (rang)")
    ax.set_ylabel("Proportion cumulee de sessions")
    ax.set_title("Concentration des sessions (Lorenz)")
    ax.legend(fontsize=8, loc="upper left")

    metrics["gini"] = round(float(gini), 4)

    fig.suptitle("Fig. 2 - Desequilibre de la variable cible (247 utilisateurs)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "fig2_target_imbalance.png")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etape 3 : Navigateur & Alignement Train/Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step3_browser_context(train, test):
    """Distribution des navigateurs et fidelite utilisateur."""
    print("\n" + "=" * 72)
    print("  ETAPE 3 : Navigateur & Alignement Train/Test")
    print("=" * 72)

    metrics = {}

    browsers_train = Counter(s["browser"] for s in train)
    browsers_test  = Counter(s["browser"] for s in test)
    all_browsers = sorted(set(browsers_train) | set(browsers_test))

    n_train = len(train)
    n_test  = len(test)

    train_pct = {b: round(100.0 * browsers_train[b] / n_train, 1) for b in all_browsers}
    test_pct  = {b: round(100.0 * browsers_test.get(b, 0) / n_test, 1) for b in all_browsers}

    metrics["browser_pct_train"] = train_pct
    metrics["browser_pct_test"]  = test_pct

    print(f"  {'Navigateur':<20} {'Train %':>8} {'Test %':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for b in all_browsers:
        print(f"  {b:<20} {train_pct[b]:>7.1f}% {test_pct[b]:>7.1f}%")

    # Fidelite : combien de navigateurs par utilisateur ?
    user_browsers = {}
    for s in train:
        user_browsers.setdefault(s["user"], set()).add(s["browser"])

    n_browsers_per_user = [len(v) for v in user_browsers.values()]
    mono_browser = sum(1 for n in n_browsers_per_user if n == 1)
    multi_browser = sum(1 for n in n_browsers_per_user if n > 1)

    metrics["mono_browser_users"] = mono_browser
    metrics["multi_browser_users"] = multi_browser
    metrics["fidelity_pct"] = round(100.0 * mono_browser / len(user_browsers), 1)

    print(f"\n  Fidelite navigateur : {mono_browser}/{len(user_browsers)} "
          f"utilisateurs mono-navigateur ({metrics['fidelity_pct']}%)")

    # Utilisateurs exclusifs par navigateur
    exclusive = {}
    for b in all_browsers:
        users_of_b = {u for u, bs in user_browsers.items() if b in bs}
        excl = {u for u, bs in user_browsers.items() if bs == {b}}
        exclusive[b] = {
            "total_users": len(users_of_b),
            "exclusive_users": len(excl),
            "sessions": browsers_train[b],
        }
    metrics["browser_exclusivity"] = exclusive

    for b, info in exclusive.items():
        print(f"    {b:<20} {info['sessions']:>5} sessions, "
              f"{info['total_users']:>3} users ({info['exclusive_users']} exclusifs)")

    # ── Figure 3 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gauche : comparaison % Train vs Test
    ax = axes[0]
    x = np.arange(len(all_browsers))
    w = 0.35
    bars1 = ax.bar(x - w/2, [train_pct[b] for b in all_browsers], w,
                   color=COLORS["train"], alpha=0.8, label="Train", edgecolor="white")
    bars2 = ax.bar(x + w/2, [test_pct[b] for b in all_browsers], w,
                   color=COLORS["test"], alpha=0.8, label="Test", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(all_browsers, fontsize=8)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Proportions Train vs Test")
    ax.legend(fontsize=8)
    # Annotation des barres
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    # Droite : sessions + utilisateurs exclusifs par navigateur
    ax = axes[1]
    sessions_vals = [exclusive[b]["sessions"] for b in all_browsers]
    exclusive_vals = [exclusive[b]["exclusive_users"] for b in all_browsers]
    total_user_vals = [exclusive[b]["total_users"] for b in all_browsers]

    ax.bar(x - w/2, sessions_vals, w, color=COLORS["train"], alpha=0.8,
           label="Sessions (train)", edgecolor="white")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, total_user_vals, w, color=COLORS["accent"], alpha=0.6,
            label="Utilisateurs", edgecolor="white")
    ax2.bar(x + w/2, exclusive_vals, w, color=COLORS["purple"], alpha=0.9,
            label="Exclusifs", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_browsers, fontsize=8)
    ax.set_ylabel("Nombre de sessions")
    ax2.set_ylabel("Nombre d'utilisateurs")
    ax.set_title("Sessions & utilisateurs par navigateur")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle("Fig. 3 - Analyse du navigateur (contexte & fidelite)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "fig3_browser_analysis.png")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etape 4 : Sparsite Lexicale & OOV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step4_vocabulary_and_grammar(train, test):
    """Sparsite du vocabulaire brut vs decomposition grammaticale."""
    print("\n" + "=" * 72)
    print("  ETAPE 4 : Sparsite Lexicale, OOV & Decomposition Grammaticale")
    print("=" * 72)

    metrics = {}

    # ── Vocabulaire brut ────────────────────────────────────────────────
    train_raw_counts = Counter()
    test_raw_counts  = Counter()
    for s in train:
        train_raw_counts.update(s["raw_actions"])
    for s in test:
        test_raw_counts.update(s["raw_actions"])

    vocab_train = set(train_raw_counts.keys())
    vocab_test  = set(test_raw_counts.keys())
    oov_raw = vocab_test - vocab_train

    hapax = sum(1 for c in train_raw_counts.values() if c == 1)
    rare  = sum(1 for c in train_raw_counts.values() if c <= 5)

    metrics["raw_vocab_train"] = len(vocab_train)
    metrics["raw_vocab_test"]  = len(vocab_test)
    metrics["raw_hapax"]       = hapax
    metrics["raw_hapax_pct"]   = round(100.0 * hapax / len(vocab_train), 1)
    metrics["raw_rare_le5"]    = rare
    metrics["raw_rare_le5_pct"] = round(100.0 * rare / len(vocab_train), 1)
    metrics["raw_oov_count"]   = len(oov_raw)
    metrics["raw_oov_pct"]     = round(100.0 * len(oov_raw) / max(len(vocab_test), 1), 1)

    print(f"  Vocabulaire brut (train)   : {len(vocab_train)} tokens uniques")
    print(f"  Hapax (1 seule fois)       : {hapax} ({metrics['raw_hapax_pct']}%)")
    print(f"  Rares (<= 5 fois)          : {rare} ({metrics['raw_rare_le5_pct']}%)")
    print(f"  Vocabulaire brut (test)    : {len(vocab_test)} tokens uniques")
    print(f"  OOV brut (test\\train)      : {len(oov_raw)} ({metrics['raw_oov_pct']}%)")

    # ── Vocabulaire par sous-composant ──────────────────────────────────
    components = ["base_actions", "controllers", "configs", "domains"]
    comp_labels = ["Action", "(Controleur)", "<Config>", "$Domaine$"]
    comp_metrics = {}

    for comp, label in zip(components, comp_labels):
        train_c = Counter()
        test_c  = Counter()
        for s in train:
            train_c.update(v for v in s[comp] if v is not None)
        for s in test:
            test_c.update(v for v in s[comp] if v is not None)

        vt = set(train_c.keys())
        ve = set(test_c.keys())
        oov = ve - vt

        comp_metrics[label] = {
            "vocab_train": len(vt),
            "vocab_test":  len(ve),
            "oov_count":   len(oov),
            "oov_pct":     round(100.0 * len(oov) / max(len(ve), 1), 1),
        }

        print(f"  {label:<16} vocab={len(vt):>5}, OOV={len(oov):>3} ({comp_metrics[label]['oov_pct']}%)")

    metrics["components"] = comp_metrics

    # Distribution de frequence des tokens bruts
    freq_bins = {"1 fois": 0, "2-5 fois": 0, "6-50 fois": 0, ">50 fois": 0}
    for c in train_raw_counts.values():
        if c == 1:
            freq_bins["1 fois"] += 1
        elif c <= 5:
            freq_bins["2-5 fois"] += 1
        elif c <= 50:
            freq_bins["6-50 fois"] += 1
        else:
            freq_bins[">50 fois"] += 1
    metrics["freq_distribution"] = freq_bins

    # ── Figure 4 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Gauche : barplot horizontal log - taille vocabulaire + OOV
    ax = axes[0]
    labels_all = ["Brut"] + comp_labels
    vocab_sizes = [len(vocab_train)] + [comp_metrics[l]["vocab_train"] for l in comp_labels]
    oov_pcts    = [metrics["raw_oov_pct"]] + [comp_metrics[l]["oov_pct"] for l in comp_labels]

    y_pos = np.arange(len(labels_all))
    bars = ax.barh(y_pos, vocab_sizes, color=[COLORS["warn"]] + [COLORS["train"]]*4,
                   alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_all, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Taille du vocabulaire (echelle log)")
    ax.set_title("Vocabulaire brut vs decompose")

    for i, (bar, oov) in enumerate(zip(bars, oov_pcts)):
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                f"OOV={oov}%", va="center", fontsize=8, fontweight="bold",
                color=COLORS["warn"] if oov > 5 else COLORS["accent"])

    # Droite : repartition par tranche de frequence
    ax = axes[1]
    fbin_labels = list(freq_bins.keys())
    fbin_vals   = list(freq_bins.values())
    colors_freq = [COLORS["warn"], "#F97316", COLORS["test"], COLORS["accent"]]
    wedges, texts, autotexts = ax.pie(
        fbin_vals, labels=fbin_labels, autopct="%1.1f%%",
        colors=colors_freq, startangle=90,
        textprops={"fontsize": 9},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Repartition des tokens bruts\npar tranche de frequence")

    fig.suptitle("Fig. 4 - Sparsite lexicale & gain de la decomposition",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "fig4_sparsity_and_parsing.png")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etape 5 : Dynamique Sequentielle et Temporelle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step5_temporal_and_sequential_dynamics(train):
    """Rythme temporel, repetitivite et actions frequentes."""
    print("\n" + "=" * 72)
    print("  ETAPE 5 : Dynamique Sequentielle et Temporelle")
    print("=" * 72)

    metrics = {}

    # ── Duree des sessions ──────────────────────────────────────────────
    durations = []
    for s in train:
        if s["time_markers"]:
            durations.append(s["time_markers"][-1])
        else:
            durations.append(0)
    durations = np.array(durations)
    metrics["duration_seconds"] = _descriptive(durations)
    print(f"  Duree des sessions (sec)  : {metrics['duration_seconds']}")

    # ── Actions par fenetre de 5s ───────────────────────────────────────
    all_window_counts = []
    for s in train:
        all_window_counts.extend(s["window_counts"])
    all_window_counts = np.array(all_window_counts) if all_window_counts else np.array([0])

    wc_nonzero = all_window_counts[all_window_counts > 0]
    metrics["window_5s"] = {
        "mean":   round(float(np.mean(wc_nonzero)), 2) if len(wc_nonzero) else 0,
        "median": int(np.median(wc_nonzero)) if len(wc_nonzero) else 0,
        "P90":    int(np.percentile(wc_nonzero, 90)) if len(wc_nonzero) else 0,
        "max_burst": int(np.max(wc_nonzero)) if len(wc_nonzero) else 0,
    }
    print(f"  Actions / fenetre 5s      : {metrics['window_5s']}")

    # ── Repetition immediate ────────────────────────────────────────────
    total_transitions = 0
    repeat_count = 0
    for s in train:
        ba = s["base_actions"]
        for i in range(len(ba) - 1):
            total_transitions += 1
            if ba[i] == ba[i + 1]:
                repeat_count += 1

    metrics["immediate_repeat_rate"] = round(100.0 * repeat_count / max(total_transitions, 1), 2)
    print(f"  Taux repetition immediate : {metrics['immediate_repeat_rate']}% "
          f"({repeat_count}/{total_transitions})")

    # ── Top 10 actions de base ──────────────────────────────────────────
    base_counter = Counter()
    for s in train:
        base_counter.update(s["base_actions"])

    total_base = sum(base_counter.values())
    top10_actions = base_counter.most_common(10)
    metrics["top10_actions"] = [
        {"action": a, "count": c, "pct": round(100.0 * c / total_base, 2)}
        for a, c in top10_actions
    ]

    print(f"\n  Top 10 actions de base ({total_base} total) :")
    for info in metrics["top10_actions"]:
        print(f"    {info['pct']:>5.1f}%  {info['action']}")

    # ── Top 10 bi-grammes ───────────────────────────────────────────────
    bigram_counter = Counter()
    for s in train:
        ba = s["base_actions"]
        for i in range(len(ba) - 1):
            bigram_counter[(ba[i], ba[i + 1])] += 1

    top10_bigrams = bigram_counter.most_common(10)
    total_bigrams = sum(bigram_counter.values())
    metrics["top10_bigrams"] = [
        {"bigram": f"{a} -> {b}", "count": c, "pct": round(100.0 * c / total_bigrams, 2)}
        for (a, b), c in top10_bigrams
    ]

    print(f"\n  Top 10 bi-grammes ({total_bigrams} total) :")
    for info in metrics["top10_bigrams"]:
        print(f"    {info['pct']:>5.2f}%  {info['bigram']}")

    # ── Figure 5 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gauche : deux distributions superposees
    ax = axes[0]
    # Distribution des durees (haut)
    dur_pos = durations[durations > 0]
    ax.hist(np.log10(dur_pos), bins=50, color=COLORS["train"], alpha=0.7,
            edgecolor="white", linewidth=0.5, label="Duree session (log10 sec)")
    ax.set_xlabel("log10(secondes)")
    ax.set_ylabel("Nombre de sessions")
    ax.set_title("Distribution des durees de sessions")
    ax.axvline(np.log10(np.median(dur_pos)), color=COLORS["warn"], ls="--", lw=1.5,
               label=f"Mediane = {int(np.median(dur_pos))}s")
    ax.legend(fontsize=8)

    # Droite : Top 10 actions
    ax = axes[1]
    top10_labels = [info["action"][:35] for info in metrics["top10_actions"]]
    top10_pcts   = [info["pct"] for info in metrics["top10_actions"]]

    y_pos = np.arange(len(top10_labels))
    bars = ax.barh(y_pos, top10_pcts, color=COLORS["train"], alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Frequence (%)")
    ax.set_title("Top 10 des actions IHM de base")

    for bar, pct in zip(bars, top10_pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8, fontweight="bold")

    fig.suptitle("Fig. 5 - Dynamique temporelle & actions frequentes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "fig5_temporal_and_actions.png")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rapport Markdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_report(all_metrics: dict, report_path: Path):
    """Genere un rapport synthetique en Markdown."""

    m1 = all_metrics["step1"]
    m2 = all_metrics["step2"]
    m3 = all_metrics["step3"]
    m4 = all_metrics["step4"]
    m5 = all_metrics["step5"]

    lines = []
    L = lines.append

    L("# Rapport d'Analyse Exploratoire - Copilote Trace Mining")
    L("")
    L("## Etape 1 : Audit Macroscopique & Heterogeneite des Sequences")
    L("")
    L("**Question** : Quelle est la dimension du dataset et la variabilite "
      "de longueur des traces ?")
    L("")
    L(f"| Metrique | Train | Test |")
    L(f"|---|---|---|")
    L(f"| Sessions | {m1['n_sessions_train']} | {m1['n_sessions_test']} |")
    L(f"| Mediane actions | {m1['n_actions_train']['median']} | {m1['n_actions_test']['median']} |")
    L(f"| Moyenne actions | {m1['n_actions_train']['mean']} | {m1['n_actions_test']['mean']} |")
    L(f"| Ecart-type | {m1['n_actions_train']['std']} | {m1['n_actions_test']['std']} |")
    L(f"| Min | {m1['n_actions_train']['min']} | {m1['n_actions_test']['min']} |")
    L(f"| P95 | {m1['n_actions_train']['P95']} | {m1['n_actions_test']['P95']} |")
    L(f"| Max | {m1['n_actions_train']['max']} | {m1['n_actions_test']['max']} |")
    L("")
    p = m1["pathological"]
    L(f"**Sessions pathologiques** : {p['short_lt5_train']} sessions train "
      f"et {p['short_lt5_test']} sessions test ont < 5 actions IHM. "
      f"{p['zero_actions_train']} sessions train et {p['zero_actions_test']} sessions test "
      f"ont 0 actions.")
    L("")
    L("**Consequence pipeline ML** : La forte heterogeneite (ecart-type >> mediane) "
      "impose un Feature Engineering invariant a la longueur des sequences (ex: ratios, "
      "distributions normalisees, statistiques d'ordre). Les sessions tres courtes "
      "devront etre traitees avec attention car elles portent peu d'information discriminante.")
    L("")

    L("---")
    L("## Etape 2 : Variable Cible Y (247 utilisateurs)")
    L("")
    L("**Question** : Les classes sont-elles equilibrees ?")
    L("")
    L(f"| Metrique | Valeur |")
    L(f"|---|---|")
    L(f"| Classes uniques | {m2['n_classes']} |")
    L(f"| Sessions/user : min | {m2['sessions_per_user']['min']} |")
    L(f"| Sessions/user : mediane | {m2['sessions_per_user']['median']} |")
    L(f"| Sessions/user : moyenne | {m2['sessions_per_user']['mean']} |")
    L(f"| Sessions/user : max | {m2['sessions_per_user']['max']} |")
    L(f"| Ratio desequilibre (max/min) | {m2['imbalance_ratio']}x |")
    L(f"| Gini | {m2['gini']} |")
    L(f"| Utilisateurs rares (<= 5 sessions) | {m2['users_rare_le5']} ({m2['users_rare_le5_pct']}%) |")
    L(f"| Utilisateurs frequents (>= 30 sessions) | {m2['users_frequent_ge30']} ({m2['users_frequent_ge30_pct']}%) |")
    L("")
    L(f"**Consequence pipeline ML** : Avec un ratio de desequilibre de {m2['imbalance_ratio']}x "
      f"et {m2['users_rare_le5']} utilisateurs rares, la validation croisee DOIT etre stratifiee "
      f"(StratifiedKFold, n_splits=4 car min={m2['sessions_per_user']['min']} sessions). "
      f"Des metriques comme le macro-F1 ou le balanced accuracy sont indispensables.")
    L("")

    L("---")
    L("## Etape 3 : Navigateur & Alignement Train/Test")
    L("")
    L("**Question** : Distribution stable entre Train et Test ? Dependance user-navigateur ?")
    L("")
    L(f"| Navigateur | Train (%) | Test (%) |")
    L(f"|---|---|---|")
    for b in sorted(m3["browser_pct_train"].keys()):
        L(f"| {b} | {m3['browser_pct_train'][b]} | {m3['browser_pct_test'][b]} |")
    L("")
    L(f"**Fidelite** : {m3['fidelity_pct']}% des utilisateurs n'utilisent qu'un seul navigateur "
      f"({m3['mono_browser_users']}/{m3['mono_browser_users']+m3['multi_browser_users']}).")
    L("")
    L("**Consequence pipeline ML** : Le navigateur est un feature contextuel potentiellement "
      "discriminant (fidelite forte). La distribution quasi-identique Train/Test ecarte "
      "tout risque de covariate shift sur cette variable.")
    L("")

    L("---")
    L("## Etape 4 : Sparsite Lexicale & Decomposition Grammaticale")
    L("")
    L("**Question** : Peut-on encoder navement les tokens bruts ?")
    L("")
    L(f"| Niveau | Vocabulaire Train | OOV Test |")
    L(f"|---|---|---|")
    L(f"| Brut | {m4['raw_vocab_train']} | {m4['raw_oov_count']} ({m4['raw_oov_pct']}%) |")
    for label, info in m4["components"].items():
        L(f"| {label} | {info['vocab_train']} | {info['oov_count']} ({info['oov_pct']}%) |")
    L("")
    L(f"**Hapax** : {m4['raw_hapax']} tokens ({m4['raw_hapax_pct']}%) ne sont vus qu'une seule "
      f"fois dans train. {m4['raw_rare_le5']} tokens ({m4['raw_rare_le5_pct']}%) sont vus <= 5 fois.")
    L("")
    L("**Consequence pipeline ML** : L'encodage brut (one-hot/TF-IDF sur les tokens complets) "
      "souffrirait d'une sparsite extreme et d'un taux d'OOV eleve. La decomposition "
      "grammaticale (Action, Controleur, Config, Domaine) reduit drastiquement le vocabulaire "
      "et le taux d'OOV. Le Feature Engineering doit s'appuyer sur les sous-composants.")
    L("")

    L("---")
    L("## Etape 5 : Dynamique Sequentielle et Temporelle")
    L("")
    L("**Question** : Rythme temporel, repetitivite et gestes dominants ?")
    L("")
    L(f"| Metrique | Valeur |")
    L(f"|---|---|")
    L(f"| Duree mediane (sec) | {m5['duration_seconds']['median']} |")
    L(f"| Duree max (sec) | {m5['duration_seconds']['max']} |")
    L(f"| Actions/fenetre 5s (moyenne) | {m5['window_5s']['mean']} |")
    L(f"| Actions/fenetre 5s (max burst) | {m5['window_5s']['max_burst']} |")
    L(f"| Taux repetition immediate | {m5['immediate_repeat_rate']}% |")
    L("")
    L("**Top 5 actions de base** :")
    L("")
    for info in m5["top10_actions"][:5]:
        L(f"- {info['action']} ({info['pct']}%)")
    L("")
    L("**Consequence pipeline ML** : Le taux de repetition immediate eleve "
      "suggere que la compression (run-length encoding) pourrait etre utile. "
      "Les features temporels (duree, rythme, bursts) et les n-grammes d'actions "
      "captureront des patterns comportementaux discriminants entre utilisateurs.")
    L("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n  Rapport -> {report_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 72)
    print("  CHARGEMENT DES DONNEES")
    print("=" * 72)

    train = load_dataset(TRAIN_CSV, is_train=True)
    test  = load_dataset(TEST_CSV,  is_train=False)
    print(f"  train : {len(train)} sessions")
    print(f"  test  : {len(test)} sessions")

    all_metrics = {}

    all_metrics["step1"] = step1_macro_structure(train, test)
    all_metrics["step2"] = step2_target_distribution(train)
    all_metrics["step3"] = step3_browser_context(train, test)
    all_metrics["step4"] = step4_vocabulary_and_grammar(train, test)
    all_metrics["step5"] = step5_temporal_and_sequential_dynamics(train)

    # Sauvegarder les metriques
    metrics_path = TABLES_DIR / "eda_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, ensure_ascii=False, indent=2)
    print(f"\n  Metriques -> {metrics_path}")

    # Generer le rapport
    report_path = TABLES_DIR / "eda_report.md"
    _generate_report(all_metrics, report_path)

    print("\n" + "=" * 72)
    print("  [OK] EDA COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
