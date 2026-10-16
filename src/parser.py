"""Audit, normalisation et parsing des traces Copilote.

Workflow
--------
1. ``audit_raw_vocabulary``  – détecte toutes les anomalies syntaxiques.
2. ``parse_session``         – décompose **une** ligne CSV en dict structuré.
3. ``load_dataset``          – charge un fichier entier en liste de dicts.

Les regex sont conçues à partir de la grammaire observée dans les logs :
  ActionDeBase(Controleur)<Configuration>$Domaine$1
où chaque suffixe est optionnel.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import TRAIN_CSV, TEST_CSV, TABLES_DIR

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Regex Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Marqueur temporel : t5, t10, t1445 …
RE_TIME = re.compile(r"^t(\d+)$")

# Token temporel collé à la fin d'un token d'action (virgule manquante).
# Ex : "Création d'un écran(AccueilController)t5" -> deux tokens séparés.
# On exige que le tX soit précédé d'un délimiteur de la grammaire (parenthèse,
# chevron, dollar) ou d'un chiffre, pour ne pas confondre avec des mots
# comme "onglet1" où le 't' fait partie du mot.
RE_GLUED_TIME = re.compile(r"^(.+[)>$\d])(t\d+)$")

# Grammaire complète d'un token d'action IHM :
#   ActionDeBase  ( Controleur )  < Configuration >  $ Domaine $  1
#
# Chaque bloc entre marques est optionnel.
RE_ACTION = re.compile(
    r"^"
    r"(?P<base>[^(<$]+?)"                    # action de base (tout sauf ( < $)
    r"(?:\((?P<ctrl>[^)]*)\))?"              # (Controleur)           optionnel
    r"(?:<(?P<cfg>[^>]*)>)?"                 # <Configuration>        optionnel
    r"(?:\$(?P<dom>[^$]*)\$)?"               # $Domaine$              optionnel
    r"(?P<sfx>1)?"                           # suffixe "1"            optionnel
    r"$"
)

# Détection de parenthèses imbriquées (traces d'erreurs Java, etc.)
RE_NESTED_PARENS = re.compile(r"\([^)]*\([^)]*\)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pré-nettoyage : séparer les marqueurs temporels collés
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _split_glued_tokens(tokens: list[str]) -> list[str]:
    """Sépare les marqueurs temporels collés par erreur à la fin d'un token.

    On vérifie que la partie gauche n'est pas elle-même un pur marqueur
    temporel (pour ne pas casser 'tt5' s'il existait – cas hypothétique).
    Seuls les cas où la partie gauche contient au moins un caractère
    non-digit (typiquement un token d'action) sont séparés.
    """
    out: list[str] = []
    for tok in tokens:
        if RE_TIME.match(tok):
            out.append(tok)
            continue
        m = RE_GLUED_TIME.match(tok)
        if m and not RE_TIME.match(m.group(1)):
            out.append(m.group(1))
            out.append(m.group(2))
        else:
            out.append(tok)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Audit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _iter_raw_tokens(*csv_paths: Path):
    """Itère sur tous les tokens bruts (après split sur virgule)."""
    for csv_path in csv_paths:
        with open(csv_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",")
                # Skip user (train) / browser (both) – on veut les événements.
                if csv_path.name == "train.csv":
                    yield from parts[2:]  # skip user + browser
                else:
                    yield from parts[1:]  # skip browser


def audit_raw_vocabulary(
    *csv_paths: Path,
    save_path: Optional[Path] = None,
) -> dict:
    """Scanne tous les tokens bruts et détecte les anomalies syntaxiques.

    Catégories d'anomalies :
    - ``glued_time``      : tX collé après un token d'action (virgule manquante).
    - ``nested_parens``   : parenthèses imbriquées (ex : traces Java).
    - ``no_regex_match``  : tokens ni temporels ni conformes à RE_ACTION après
                            nettoyage des collages temporels.

    Returns
    -------
    dict avec les clés ``glued_time``, ``nested_parens``, ``no_regex_match``,
    chacune contenant un dict ``{token: count}``.
    """
    glued_time: Counter   = Counter()
    nested_parens: Counter = Counter()
    no_match: Counter     = Counter()

    for raw_tok in _iter_raw_tokens(*csv_paths):
        tok = raw_tok.strip()
        if not tok:
            continue

        # 1) Marqueur temporel pur → OK
        if RE_TIME.match(tok):
            continue

        # 2) Marqueur temporel collé ?
        m = RE_GLUED_TIME.match(tok)
        if m and not RE_TIME.match(m.group(1)):
            glued_time[tok] += 1
            # Après séparation, vérifier la partie action
            action_part = m.group(1)
        else:
            action_part = tok

        # 3) Parenthèses imbriquées ?
        if RE_NESTED_PARENS.search(action_part):
            nested_parens[action_part] += 1
            continue  # on considère ces tokens comme identifiés

        # 4) La partie action matche-t-elle la grammaire standard ?
        if not RE_ACTION.match(action_part):
            no_match[action_part] += 1

    report = {
        "glued_time": dict(glued_time.most_common()),
        "nested_parens": dict(nested_parens.most_common()),
        "no_regex_match": dict(no_match.most_common()),
    }

    if save_path is None:
        save_path = TABLES_DIR / "parsing_audit.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Parsing d'un token d'action
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_action_token(tok: str) -> dict:
    """Décompose un token d'action selon la grammaire IHM.

    Pour les cas spéciaux (parenthèses imbriquées / Java), on extrait
    le texte avant la première '(' comme action de base, et l'intérieur
    complet comme contrôleur.

    Returns
    -------
    dict avec ``base_action``, ``controller``, ``config``,
    ``domain``, ``suffix_one`` (bool).
    """
    m = RE_ACTION.match(tok)
    if m:
        return {
            "base_action": m.group("base").strip(),
            "controller":  m.group("ctrl"),
            "config":      m.group("cfg"),
            "domain":      m.group("dom"),
            "suffix_one":  m.group("sfx") == "1",
        }

    # Fallback : parenthèses imbriquées ou format non standard.
    # On prend tout avant la première '(' comme base, le reste comme ctrl.
    idx = tok.find("(")
    if idx != -1:
        base = tok[:idx].strip()
        # Extraire l'intérieur entre la première '(' et la dernière ')'
        last_paren = tok.rfind(")")
        if last_paren > idx:
            ctrl = tok[idx + 1 : last_paren]
        else:
            ctrl = tok[idx + 1 :]
        # Chercher config/domaine/suffixe après la dernière ')'
        remainder = tok[last_paren + 1 :] if last_paren > idx else ""
    else:
        base = tok.strip()
        ctrl = None
        remainder = ""

    # Parser le remainder pour <config>, $domain$, suffixe 1
    cfg_m = re.search(r"<([^>]*)>", remainder)
    dom_m = re.search(r"\$([^$]*)\$", remainder)
    sfx   = remainder.rstrip().endswith("1")

    return {
        "base_action": base,
        "controller":  ctrl,
        "config":      cfg_m.group(1) if cfg_m else None,
        "domain":      dom_m.group(1) if dom_m else None,
        "suffix_one":  sfx,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Parsing d'une session
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_window_counts(time_markers: list[int], n_actions_between: list[int], window: int = 5) -> list[int]:
    """Nombre d'actions par tranche de ``window`` secondes.

    ``n_actions_between[i]`` est le nombre d'actions entre le marqueur
    ``time_markers[i-1]`` (ou le début) et ``time_markers[i]``.

    On répartit les actions uniformément sur la durée entre deux marqueurs,
    puis on agrège par fenêtre de ``window`` secondes.
    """
    if not time_markers:
        return []

    total_time = time_markers[-1]
    n_windows = (total_time // window) + 1
    counts = [0] * n_windows

    # Bornes : [prev_t, cur_t)
    prev_t = 0
    for i, cur_t in enumerate(time_markers):
        n_act = n_actions_between[i]
        duration = cur_t - prev_t
        if duration > 0 and n_act > 0:
            # Distribuer uniformément les actions sur [prev_t, cur_t)
            for j in range(n_act):
                t = prev_t + (j + 0.5) * duration / n_act
                w = int(t // window)
                if w < n_windows:
                    counts[w] += 1
        elif n_act > 0:
            # Durée nulle : tout va dans la fenêtre courante
            w = int(cur_t // window)
            if w < n_windows:
                counts[w] += n_act
        prev_t = cur_t

    return counts


def parse_session(tokens: list[str], user: Optional[str] = None) -> dict:
    """Parse une séquence de tokens (événements) en un dict structuré.

    Parameters
    ----------
    tokens : list[str]
        Séquence d'événements **après** extraction de user/browser.
        Le premier token est ``browser``, le reste les événements.
    user : str or None
        Label de l'utilisateur (None pour test.csv).

    Returns
    -------
    dict avec les clés documentées dans le docstring du module.
    """
    browser = tokens[0]
    raw_events = tokens[1:]

    # Pré-nettoyage : séparer les marqueurs temporels collés
    events = _split_glued_tokens(raw_events)

    # Séparer actions et marqueurs temporels
    raw_actions: list[str] = []
    time_markers: list[int] = []
    n_actions_between: list[int] = []   # actions entre tX[i-1] et tX[i]
    action_count_since_last_t = 0

    for ev in events:
        ev = ev.strip()
        if not ev:
            continue
        m_time = RE_TIME.match(ev)
        if m_time:
            time_markers.append(int(m_time.group(1)))
            n_actions_between.append(action_count_since_last_t)
            action_count_since_last_t = 0
        else:
            raw_actions.append(ev)
            action_count_since_last_t += 1

    # Décomposer chaque action
    base_actions: list[str] = []
    controllers:  list[Optional[str]] = []
    configs:      list[Optional[str]] = []
    domains:      list[Optional[str]] = []
    suffix_ones   = 0

    for act in raw_actions:
        parsed = _parse_action_token(act)
        base_actions.append(parsed["base_action"])
        controllers.append(parsed["controller"])
        configs.append(parsed["config"])
        domains.append(parsed["domain"])
        if parsed["suffix_one"]:
            suffix_ones += 1

    # Fenêtres temporelles
    window_counts = _compute_window_counts(time_markers, n_actions_between)

    return {
        "user":          user,
        "browser":       browser,
        "raw_actions":   raw_actions,
        "base_actions":  base_actions,
        "controllers":   controllers,
        "configs":       configs,
        "domains":       domains,
        "suffix_ones":   suffix_ones,
        "time_markers":  time_markers,
        "window_counts": window_counts,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chargement complet d'un fichier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset(csv_path: Path, is_train: bool = True) -> list[dict]:
    """Charge et parse toutes les sessions d'un fichier CSV.

    Parameters
    ----------
    csv_path : Path
        Chemin vers train.csv ou test.csv.
    is_train : bool
        Si True, le premier token est ``user`` (cible).

    Returns
    -------
    list[dict] – une entrée par session (ligne).
    """
    sessions: list[dict] = []
    with open(csv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if not parts or (len(parts) < 2):
                continue
            if is_train:
                user = parts[0]
                tokens = parts[1:]   # browser + events
            else:
                user = None
                tokens = parts       # browser + events
            sessions.append(parse_session(tokens, user=user))
    return sessions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Point d'entrée test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 72)
    print("  AUDIT SYNTAXIQUE DES TOKENS BRUTS")
    print("=" * 72)

    report = audit_raw_vocabulary(TRAIN_CSV, TEST_CSV)

    for category, items in report.items():
        n = len(items)
        total = sum(items.values())
        print(f"\n  [{category}]  {n} tokens distincts, {total} occurrences")
        # Afficher les 10 plus fréquents
        for tok, cnt in list(items.items())[:10]:
            print(f"    {cnt:>6d}x  {tok}")
        if n > 10:
            print(f"    ... et {n - 10} autres tokens")

    save_path = TABLES_DIR / "parsing_audit.json"
    print(f"\n  Rapport sauvegarde -> {save_path}")

    # ── Vérification du chargement ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CHARGEMENT DES SESSIONS")
    print("=" * 72)

    train = load_dataset(TRAIN_CSV, is_train=True)
    test  = load_dataset(TEST_CSV,  is_train=False)

    print(f"\n  train.csv : {len(train):>6d} sessions")
    print(f"  test.csv  : {len(test):>6d} sessions")

    # Aperçu d'une session
    s = train[0]
    print(f"\n  Exemple (train[0]):")
    print(f"    user         = {s['user']}")
    print(f"    browser      = {s['browser']}")
    print(f"    #actions     = {len(s['raw_actions'])}")
    print(f"    #base_actions= {len(s['base_actions'])}")
    print(f"    #time_marks  = {len(s['time_markers'])}")
    if s["time_markers"]:
        print(f"    time range   = t{s['time_markers'][0]} -> t{s['time_markers'][-1]}")
    print(f"    suffix_ones  = {s['suffix_ones']}")
    print(f"    window_counts (first 10) = {s['window_counts'][:10]}")

    # Quelques stats globales
    users = set(s["user"] for s in train)
    browsers = Counter(s["browser"] for s in train)
    print(f"\n  Utilisateurs uniques (train) : {len(users)}")
    print(f"  Navigateurs (train) :")
    for b, c in browsers.most_common():
        print(f"    {c:>6d}x  {b}")

    print("\n  [OK] Parsing termine avec succes.")
