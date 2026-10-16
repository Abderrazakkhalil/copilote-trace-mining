"""Post-processing utilities for browser masking and probability calibration.

Functions
---------
apply_browser_mask
    Zeroes out probabilities for browser-incompatible users, renormalizes.
decision_to_proba
    Converts LinearSVC decision scores to probabilities via softmax.
"""

from __future__ import annotations

import numpy as np


def apply_browser_mask(
    probs: np.ndarray,
    session_browsers: list[str],
    classes: np.ndarray,
    user_to_browser: dict[str, str],
) -> np.ndarray:
    """Masquage déterministe par navigateur.

    Pour chaque échantillon *i* et classe *j*, si le navigateur connu de
    l'utilisateur ``classes[j]`` (via ``user_to_browser``) ne correspond
    pas au navigateur de la session ``session_browsers[i]``, la probabilité
    ``probs[i, j]`` est mise à 0.  Chaque ligne est ensuite renormalisée
    pour que la somme vaille 1.

    Parameters
    ----------
    probs : ndarray, shape (n_samples, n_classes)
        Matrice de probabilités brutes.
    session_browsers : list[str], longueur n_samples
        Navigateur de chaque session (train ou test).
    classes : array-like of str, longueur n_classes
        Labels de classe ordonnés (colonnes de *probs*).
    user_to_browser : dict
        ``{user_label: navigateur_exclusif}``.

    Returns
    -------
    ndarray, shape (n_samples, n_classes)
        Probabilités masquées et renormalisées.
    """
    masked = probs.copy()
    browsers_arr = np.asarray(session_browsers)

    for j, user in enumerate(classes):
        expected = user_to_browser.get(user)
        if expected is not None:
            mismatch = browsers_arr != expected
            masked[mismatch, j] = 0.0

    # Renormaliser chaque ligne
    row_sums = masked.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    masked /= row_sums

    return masked


def decision_to_proba(
    scores: np.ndarray,
    temperature: float = 2.0,
) -> np.ndarray:
    """Convertit la sortie de ``LinearSVC.decision_function`` en probabilités.

    Applique un softmax avec température :
    ``P_j = exp((s_j − max) × T) / Σ_k exp((s_k − max) × T)``

    Parameters
    ----------
    scores : ndarray, shape (n_samples, n_classes)
        Scores bruts de ``decision_function``.
    temperature : float
        Facteur de mise à l'échelle (plus élevé → distribution plus piquée).

    Returns
    -------
    ndarray, shape (n_samples, n_classes)
        Probabilités calibrées.
    """
    scaled = scores * temperature
    scaled -= scaled.max(axis=1, keepdims=True)   # stabilité numérique
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
