# Identification des utilisateurs de l’ERP Copilote

**[Consulter / télécharger le rapport PDF complet](report/BE_ML_Rapport.pdf)**

Projet de BE — **École Centrale de Lyon, MOD 7.2 : Introduction aux sciences des données**.
À partir des traces de sessions IHM de l’ERP **Copilote (Infologic)**, nous cherchons à reconnaître **247 utilisateurs** : **3 279 sessions d’apprentissage** et **324 sessions de test**.

**Auteurs :** Salma Ihssane ,Anas Meniari et Abderrazak Khalil.
**Enseignant :** Antoine Becquet.

## Comprendre les traces pour choisir la méthode

| Observation de l’EDA | Conséquence méthodologique |
|---|---|
| De **4 à 75 sessions par utilisateur** | **Macro-F1** pour donner le même poids aux classes ; validation stratifiée à **4 plis**, puis sélection et évaluation séparées en **nested CV** |
| Un seul navigateur observé pour chacun des 247 utilisateurs | **Masque navigateur** en sortie : le SVM lexical passe de 94,17 à 96,20 % de Macro-F1, soit **+2,03 points** |
| **7 086 tokens** distincts et de nombreux événements rares | Décomposition de `Action(Controleur)<Config>$Domaine$1` ; traitement des marqueurs `t5` collés et des parenthèses Java imbriquées |
| Marqueurs temporels `tX`, parfois répétés (`t10`) | Hypothèse de multi-fenêtrage ; rythme agrégé par fenêtres de **5 secondes**, avec protection contre les intervalles nuls |
| **18,08 %** de répétitions immédiates | Conservation des motifs courts, des transitions et des séries répétées |

Les doublons temporels suggèrent plusieurs fenêtres simultanées ; ils ne donnent pas l’instant exact de chaque clic. Le masque suppose que la fidélité au navigateur observée dans le train se maintient au test.

## Trois représentations complémentaires

1. **Séquentielle** : TF-IDF sous-linéaire, normalisation L2 des blocs, unigrammes, bigrammes/trigrammes et ancrages de début/fin. Les transitions de **Markov** et le **Run-Length Encoding (RLE)** complètent cette lecture de la séquence ; leur implémentation se trouve dans le bloc dense.
2. **Comportementale** : **246 descripteurs** dans la représentation historique complète, résumant volume, durée, diversité, entropie, vitesse, répétitions et rythme. La dimension dépend du vocabulaire appris dans chaque pli.
3. **Hybride rééquilibrée** : standardisation du bloc comportemental sur le train, normalisation **L2 par session**, puis pondération **0,25** avant concaténation au TF-IDF. Après normalisation et réglage, le SVM hybride passe de **87,54 % à 96,38 %** de Macro-F1.

## Benchmark des cinq familles

Macro-F1 **local (%) après masque navigateur**, issu du [benchmark exploratoire](outputs/tables/benchmark_results.csv). L’hybride de ce tableau est la concaténation initiale, avant rééquilibrage.

| Modèle | Séquence TF-IDF | Comportement | Hybride initial |
|---|---:|---:|---:|
| Régression logistique | 92,36 | 46,93 | 86,76 |
| **SVM linéaire** | **96,20** | 65,28 | 87,54 |
| MLP | 87,05 | 77,67 | 85,54 |
| Arbre de décision | 19,28 | 65,61 | 62,15 |
| **XGBoost** | 71,70 | **83,01** | 85,62 |

Les matrices de ce benchmark historique sont préparées sur tout le train. Le pipeline imbriqué conservé réapprend les transformations dans chaque pli ; ces deux évaluations sont distinguées dans [la documentation de reproductibilité](docs/REPRODUCIBILITY.md).

## Progression sur le leaderboard public Kaggle

| Palier décrit dans le rapport | Approche | Score public |
|---|---|---:|
| Socle classique | Feature engineering, vote souple pondéré des cinq familles et masque navigateur | **97,4 %** |
| Itérations intermédiaires | Ridge, profils moyens et similarités cosinus, cinq premières/dernières actions, rééquilibrage global inspiré de Sinkhorn | **97,2 → 97,9 → 98,0 %** |
| Palier final | **ConvBiGRUAttention** : convolution, GRU bidirectionnelle, attention, position et rythme | **98,4 %** |

Le rééquilibrage exploratoire décrit dans le rapport étend les prédictions de **224 à 229 identités distinctes**. Les routines génériques sont dans `src/sinkhorn.py` ; le script historique comportant également des ancres par ligne est archivé localement.

Le rapport annonce une fenêtre neuronale de **126 événements sous contrainte CPU**. Le code disponible et les résultats locaux conservés correspondent à une fenêtre de **256 événements**. L’option `--max-length 126` permet désormais d’exécuter la contrainte du rapport, avec des sorties distinctes. Le score public de 98,4 % n’a pas été recalculé lors de ce nettoyage.

L’[export local du pipeline imbriqué](outputs/tables/dl_hybrid_report.md) mesure **98,17 % de Macro-F1 externe pour la fusion avec le réseau**, contre 97,79 % sans lui. Ce score local est distinct du score public ; la correspondance exacte du fichier local avec celui envoyé sur Kaggle reste à confirmer dans les archives de soumission.

## Repères dans le dépôt

```text
copilote-trace-mining/
├── report/                 # PDF final, source LaTeX disponible et illustration
├── figures/                # Les cinq figures d’EDA du rapport (fig1 à fig5)
├── data/                   # Format attendu et empreintes des CSV
│   ├── raw/                # train/test locaux ; modèle de soumission versionné
│   └── processed/          # Matrices et checkpoints générés, ignorés par Git
├── src/
│   ├── parser.py           # Grammaire des traces, anomalies et marqueurs tX
│   ├── eda.py              # Analyse exploratoire et cinq figures
│   ├── features.py         # Représentations lexicales et comportementales
│   ├── train_eval.py       # Benchmark exploratoire et vote souple
│   ├── optimize_models.py  # Hybride L2, réglages, Ridge et ensemble
│   ├── postprocess.py      # Masque navigateur et conversion des scores
│   ├── sinkhorn.py         # Positions, profils cosinus et rééquilibrage global
│   ├── conv_bigru_attention.py  # Architecture séquentielle et entraînement
│   ├── dl_hybrid_pipeline.py    # Validation imbriquée, fusion et soumission
│   ├── dl_hybrid_transfer.py    # Transferts contrôlés des probabilités
│   ├── validate_data.py    # Vérification des données avant calcul
│   └── config.py           # Chemins ancrés sur la racine et graine 42
├── outputs/
│   ├── tables/             # EDA, benchmark, ablations, résultats et provenance
│   └── submissions/        # Trois soumissions jalons documentées
├── tests/                  # Tests du parsing, réseau, fusion, reprise et sorties
├── docs/                   # Reproductibilité, bilan du tri et sujet d’origine
├── requirements.txt        # Dépendances directes
└── requirements-dev.txt    # Dépendances et tests
```

Le dossier `_archive_local/`, exclu de Git, conserve les essais, anciennes soumissions, caches et journaux. Le [bilan du tri](docs/CLEANUP.md) décrit les déplacements ; l’inventaire détaillé reste sur la machine.

## Installation et exécution

Utiliser **Python 3.10 ou ultérieur**. Depuis la racine :

```bash
python -m venv .venv
```

Activer l’environnement avec `.\.venv\Scripts\Activate.ps1` sous PowerShell, ou `source .venv/bin/activate` sous Linux/macOS, puis :

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Les tests utilisent des données synthétiques et ne nécessitent pas les CSV du BE. `requirements.txt` suffit pour exécuter les scripts sans pytest ; `requirements-repro.txt` enregistre les versions de la machine de vérification.

Obtenir les données auprès de la source autorisée et placer `train.csv` et `test.csv` dans `data/raw/` selon [data/README.md](data/README.md), puis :

```bash
python -m src.validate_data
python -m src.parser
python -m src.eda
```

Pour reconstruire les représentations et les expériences classiques historiques, exécuter dans cet ordre :

```bash
python -m src.features
python -m src.train_eval
python -m src.optimize_models
```

Pour lancer le protocole imbriqué directement à partir des CSV, avec la fenêtre de 126 événements indiquée dans le rapport :

```bash
python -u -m src.dl_hybrid_pipeline --device cpu --workers 1 --threads 4 --max-length 126 --reference-0980 submission_hada_jehd.csv
```

Ce calcul réentraîne les transformations et les modèles ; il demande **plusieurs heures sur CPU**. La soumission est écrite dans `outputs/submissions/submission_dl_hybrid_len126.csv` et les tables dans `outputs/tables/dl_hybrid_len126/`. La référence 98,0 % sert uniquement à la concordance. Pour le protocole de longueur historique, utiliser `--max-length 256` ; une exécution complète écrit `submission_dl_hybrid_final.csv`.

Les détails de reprise, les limites de validation et la provenance des résultats sont dans [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Le [PDF final](report/BE_ML_Rapport.pdf) reste le document remis ; la source LaTeX disponible est décrite dans [report/README.md](report/README.md).
