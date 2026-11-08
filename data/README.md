# Données des sessions Copilote

Placer les fichiers originaux obtenus auprès des enseignants ou de l'espace
Kaggle du BE dans `data/raw/`, sans modifier leur ordre ni leur encodage UTF-8.
Le dépôt ne redistribue pas `train.csv` et `test.csv`. Les copies déjà présentes
sur la machine de travail sont conservées et ignorées par Git.

| Fichier | Format | Effectif attendu |
|---|---|---:|
| `raw/train.csv` | `utilisateur,navigateur,événement,...` | 3 279 sessions, 247 utilisateurs |
| `raw/test.csv` | `navigateur,événement,...` | 324 sessions |
| `raw/sample_submission.csv` | En-tête `RowId,prediction` | 324 lignes, identifiants de 1 à 324 |

Les deux fichiers de traces **n'ont pas d'en-tête** ; leurs lignes ont des
longueurs variables. Utiliser `src.parser.load_dataset`, et non une lecture
tabulaire à nombre de colonnes fixe. Les événements sont des actions telles que
`Action(Controleur)<Config>$Domaine$1` ou des marqueurs temporels `tX`.

Les empreintes SHA-256 dans [manifest.json](manifest.json) portent sur le texte
UTF-8 après normalisation des fins de ligne, pour accepter les clones Windows
et Linux. Elles identifient les données utilisées pour l'audit du dépôt.

Depuis la racine :

```bash
python -m src.validate_data
```

`data/processed/` accueille uniquement des matrices, probabilités et checkpoints
recalculables ; son contenu reste local. Les données brutes existaient dans
l'historique Git avant ce nettoyage : les ignorer et les désindexer ne les
efface pas des anciens commits.
