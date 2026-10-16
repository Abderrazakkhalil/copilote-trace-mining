# Rapport d'Analyse Exploratoire - Copilote Trace Mining

## Etape 1 : Audit Macroscopique & Heterogeneite des Sequences

**Question** : Quelle est la dimension du dataset et la variabilite de longueur des traces ?

| Metrique | Train | Test |
|---|---|---|
| Sessions | 3279 | 324 |
| Mediane actions | 340 | 365 |
| Moyenne actions | 692.4 | 662.3 |
| Ecart-type | 1005.7 | 810.2 |
| Min | 0 | 9 |
| P95 | 2386 | 2287 |
| Max | 11977 | 6203 |

*(Normalement longeur totale des sessions c'est 14468 mais ici on a compté que les actions et non pas les horloges qui sont 2491)*

**Sessions pathologiques** : 3 sessions train et 0 sessions test ont < 5 actions IHM. 1 sessions train et 0 sessions test ont 0 actions.

**Consequence pipeline ML** : La forte heterogeneite (ecart-type >> mediane) impose un Feature Engineering invariant a la longueur des sequences (ex: ratios, distributions normalisees, statistiques d'ordre). Les sessions tres courtes devront etre traitees avec attention car elles portent peu d'information discriminante.

---
## Etape 2 : Variable Cible Y (247 utilisateurs)

**Question** : Les classes sont-elles equilibrees ?

| Metrique | Valeur |
|---|---|
| Classes uniques | 247 |
| Sessions/user : min | 4 |
| Sessions/user : mediane | 12 |
| Sessions/user : moyenne | 13.3 |
| Sessions/user : max | 75 |
| Ratio desequilibre (max/min) | 18.8x |
| Gini | 0.3095 |
| Utilisateurs rares (<= 5 sessions) | 33 (13.4%) |
| Utilisateurs frequents (>= 30 sessions) | 10 (4.0%) |

**Consequence pipeline ML** : Avec un ratio de desequilibre de 18.8x et 33 utilisateurs rares, la validation croisee DOIT etre stratifiee (StratifiedKFold, n_splits=4 car min=4 sessions). Des metriques comme le macro-F1 ou le balanced accuracy sont indispensables.

---
## Etape 3 : Navigateur & Alignement Train/Test

**Question** : Distribution stable entre Train et Test ? Dependance user-navigateur ?

| Navigateur | Train (%) | Test (%) |
|---|---|---|
| Firefox | 44.7 | 45.4 |
| Google Chrome | 40.8 | 41.0 |
| Microsoft Edge | 13.8 | 13.0 |
| Opera | 0.7 | 0.6 |

**Fidelite** : 100.0% des utilisateurs n'utilisent qu'un seul navigateur (247/247).

**Consequence pipeline ML** : Le navigateur est un feature contextuel potentiellement discriminant (fidelite forte). La distribution quasi-identique Train/Test ecarte tout risque de covariate shift sur cette variable.

---
## Etape 4 : Sparsite Lexicale & Decomposition Grammaticale

**Question** : Peut-on encoder navement les tokens bruts ?

| Niveau | Vocabulaire Train | OOV Test |
|---|---|---|
| Brut | 7086 | 206 (7.3%) |
| Action | 35 | 0 (0.0%) |
| (Controleur) | 415 | 8 (3.7%) |
| <Config> | 219 | 2 (1.4%) |
| $Domaine$ | 76 | 0 (0.0%) |

**Hapax** : 1981 tokens (28.0%) ne sont vus qu'une seule fois dans train. 4004 tokens (56.5%) sont vus <= 5 fois.

**Consequence pipeline ML** : L'encodage brut (one-hot/TF-IDF sur les tokens complets) souffrirait d'une sparsite extreme et d'un taux d'OOV eleve. La decomposition grammaticale (Action, Controleur, Config, Domaine) reduit drastiquement le vocabulaire et le taux d'OOV. Le Feature Engineering doit s'appuyer sur les sous-composants.

---
## Etape 5 : Dynamique Sequentielle et Temporelle

**Question** : Rythme temporel, repetitivite et gestes dominants ?

| Metrique | Valeur |
|---|---|
| Duree mediane (sec) | 395 |
| Duree max (sec) | 11790 |
| Actions/fenetre 5s (moyenne) | 4.58 |
| Actions/fenetre 5s (max burst) | 1859 |
| Taux repetition immediate | 18.08% |

**Top 5 actions de base** :

- Exécution d'un bouton (22.49%)
- Saisie dans un champ (11.7%)
- Sélection d'un écran (9.11%)
- Fermeture d'une dialogue (7.53%)
- Affichage d'une dialogue (7.44%)

**Consequence pipeline ML** : Le taux de repetition immediate eleve suggere que la compression (run-length encoding) pourrait etre utile. Les features temporels (duree, rythme, bursts) et les n-grammes d'actions captureront des patterns comportementaux discriminants entre utilisateurs.
