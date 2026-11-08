"""Outils de l'exploration positionnelle et du rééquilibrage global.

Le rééquilibrage reprend l'ajustement de colonnes inspiré de Sinkhorn du
script historique. Il ne réalise pas un transport optimal itératif et ne
force aucun nombre d'identités ni aucune prédiction individuelle. Ses
paramètres doivent être choisis sur des validations internes uniquement.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.postprocess import apply_browser_mask


def positional_documents(sessions: list[dict], width: int = 5) -> tuple[list, list]:
    """Tokens des premières/dernières actions, sans apprendre de vocabulaire."""
    if width < 1:
        raise ValueError("width must be positive")
    opening, closing = [], []
    for session in sessions:
        actions = session["raw_actions"]
        opening.append(actions[:width])
        closing.append(actions[-width:])
    return opening, closing


def cosine_profile_similarities(X_train, y_train, X_evaluation, classes):
    """Similarité aux profils moyens appris exclusivement sur le train reçu.

    Le vocabulaire et la pondération de X doivent eux aussi être appris dans
    le train du pli. Aucune étiquette d'évaluation n'est utilisée ici.
    """
    labels = np.asarray(y_train)
    if X_train.shape[0] != len(labels) or X_train.shape[1] != X_evaluation.shape[1]:
        raise ValueError("Feature/label dimensions differ")
    centers = []
    for label in classes:
        selected = labels == label
        if not selected.any():
            raise ValueError(f"No training example for class {label}")
        centers.append(sp.csr_matrix(np.asarray(X_train[selected].mean(axis=0)).reshape(1, -1)))
    centers = normalize(sp.vstack(centers, format="csr"))
    similarities = normalize(sp.csr_matrix(X_evaluation)) @ centers.T
    return similarities.toarray().astype(np.float32)


def rebalance_probabilities(probabilities, train_counts, browsers, classes,
                            user_to_browser, *, alpha=0., beta=0., tau=0., smoothing=.15):
    """Ajustement global doux des colonnes, suivi du masque navigateur.

    P'[i,c] ∝ P[i,c] / masse[c]**beta × cible[c]**alpha × fréquence[c]**(-tau).
    La cible combine les fréquences train et un prior uniforme. alpha=beta=
    tau=0 laisse les scores inchangés, hormis le masque et la normalisation.
    Les comptes train proviennent du train interne pendant la sélection.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    counts = np.asarray(train_counts, dtype=np.float64)
    classes = np.asarray(classes)
    if (p.ndim != 2 or not p.shape[0] or p.shape[1] != len(classes)
            or counts.shape != (len(classes),) or len(browsers) != p.shape[0]):
        raise ValueError("Probability, count, browser and class dimensions differ")
    if (not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(counts).all()
            or (counts <= 0).any() or not np.isfinite([alpha, beta, tau, smoothing]).all()
            or min(alpha, beta, tau) < 0 or not 0 <= smoothing <= 1):
        raise ValueError("Expected finite nonnegative probabilities and positive train counts")
    p = apply_browser_mask(p, browsers, classes, user_to_browser)
    if (p.sum(axis=1) <= 0).any():
        raise ValueError("A session has no positive browser-compatible probability")
    frequencies = counts / counts.sum()
    target = p.shape[0] * ((1 - smoothing) * frequencies + smoothing / len(classes))
    weights = ((target + 1e-6) ** alpha * frequencies ** (-tau)
               / (p.sum(axis=0) + 1e-6) ** beta)
    return apply_browser_mask(p * weights, browsers, classes, user_to_browser)
