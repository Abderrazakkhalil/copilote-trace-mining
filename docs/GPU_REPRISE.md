# Reprendre sur un autre PC avec GPU

Le processus CPU a été arrêté après 23 époques complètes du premier pli interne.
L'ancien programme ne sauvegardait pas les poids à chaque époque : ces poids
étaient uniquement en RAM. Le réseau recommencera ce premier entraînement à
l'époque 1. Les probabilités des sept modèles classiques du même pli sont
conservées dans Git et seront réutilisées automatiquement.

## 1. Récupérer le dépôt

```shell
git clone https://github.com/Abderrazakkhalil/copilote-trace-mining.git
cd copilote-trace-mining
```

Si le dépôt est déjà cloné sur l'autre PC, utiliser `git pull --ff-only` dans
ce dépôt. Les CSV bruts nécessaires sont déjà suivis par Git, ainsi que la
référence `outputs/submissions/submission_hada_jehd.csv` identifiée comme 0,980.
L'archive locale `data/processed.zip` et les scripts exploratoires locaux ne
sont pas nécessaires à ce pipeline.

## 2. Préparer Python et PyTorch CUDA

Utiliser un environnement Python 3.10 ou 3.11 compatible avec les dépendances
figées du projet. Par exemple sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
```

Sous Linux, l'activation équivalente est `source .venv/bin/activate`.

Installer ensuite **PyTorch avec CUDA** en utilisant la commande adaptée au PC
donnée par le [sélecteur officiel PyTorch](https://docs.pytorch.org/get-started/locally/).
`requirements-repro.txt` exclut volontairement PyTorch pour conserver ce choix.
Le PC CPU initial utilisait PyTorch 2.14.1+cpu ; les autres versions exactes
figurent dans le fichier de dépendances et dans le manifeste du transfert.

Vérifier que PyTorch voit effectivement le GPU :

```shell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU CUDA indisponible')"
```

`CUDA: True` est nécessaire avant la commande d'entraînement suivante. Le chemin
`--device cuda` suppose un GPU NVIDIA pris en charge et un pilote compatible.

## 3. Vérifier le transfert sans entraîner

```shell
python -m pytest tests/ -q
python -m src.dl_hybrid_pipeline --check-transfer --workers 1 --threads 4 --reference-0980 submission_hada_jehd.csv
```

La commande vérifie le paquet puis copie les probabilités classiques dans le
cache du protocole courant. Elle doit se terminer par :

```text
Validated transfer bundle; imported classical caches: ['outer1_inner1']
Transfer checks passed; no training started.
```

Si l'import avait déjà été effectué, la liste des caches nouvellement importés
sera vide. Les données, les fonctions de construction et de modélisation,
l'ordre des classes, les indices train/validation et le SHA-256 du fichier
doivent correspondre. Un paquet incompatible est refusé.

## 4. Lancer sur GPU

```shell
python -u -m src.dl_hybrid_pipeline --device cuda --workers 1 --threads 4 --reference-0980 submission_hada_jehd.csv
```

Le journal doit afficher `resume classical outer1_inner1` pour le premier pli.
Le réseau démarre à l'époque 1, faute de sauvegarde de l'ancien réseau CPU.
Les quatre plis externes passent séquentiellement sur le GPU. Les sept modèles
classiques utilisent toujours le CPU ; `--threads 4` contrôle leur parallélisme.

La même commande peut être lancée avec `--device cpu` pour une reprise locale.
Ce guide ne constitue pas un test GPU : les tests de reprise ont été exécutés
sur CPU, où une interruption/reprise reproduit exactement les poids d'une
exécution continue.

## 5. Arrêter et reprendre les prochains entraînements

Le nouveau programme écrit à chaque fin d'époque :

```text
data/processed/dl_hybrid_runs/<empreinte>/outer1_inner1_training.pt
```

Le message de fin d'époque indique `saved=True`. Le checkpoint comprend poids,
états AdamW et scheduler, états aléatoires, courbes et checkpoints nécessaires à
la moyenne finale. L'écriture atomique conserve le précédent fichier valide si
une interruption survient pendant la sauvegarde.

Faire `Ctrl+C` pour arrêter, puis relancer la même commande : le programme
reprend à l'époque suivant le dernier checkpoint complet. Le travail de
l'époque inachevée sera refait. Un pli entièrement terminé est directement
réutilisé depuis son fichier de probabilités.

Pour changer à nouveau de PC, arrêter le processus puis copier **le dossier
entier** `data/processed/dl_hybrid_runs/<empreinte>/` vers le même emplacement
relatif dans l'autre clone. Ces futurs checkpoints sont ignorés par Git : un
`git push` seul ne les transfère pas. Le paquet classique présent dans
`artifacts/dl_hybrid_transfer/`, lui, est bien inclus dans ce commit.

Le matériel, le nombre de threads et les fins de lignes Windows/Linux ne
changent pas l'identifiant du protocole. Les versions de chaque session sont
enregistrées séparément. Les tenseurs recalculés du pli sont aussi vérifiés
avant chargement du checkpoint. Un changement de code, de données ou de recette
d'entraînement déclenche un autre cache ou un refus de reprise. Un passage
CPU/GPU peut modifier la trajectoire numérique ; l'identité bit à bit entre
matériels différents n'est pas annoncée.

Le mécanisme suit les [recommandations PyTorch sur les checkpoints complets et
le chargement entre appareils](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

## 6. Résultats attendus à la fin

- `data/processed/conv_bigru_attn_probs.pkl` : OOF et test du réseau.
- `outputs/submissions/submission_dl_hybrid_final.csv` : soumission vérifiée.
- `outputs/tables/dl_hybrid_report.md` et `.json` : comparaison OOF sans/avec DL.
- `outputs/tables/dl_hybrid_concordance.csv` : références 0,974 et 0,980.
- `outputs/tables/dl_hybrid_arbitrations.csv` : tous les changements avec Top-1/Top-2.

Aucun score OOF global du nouveau réseau ni aucune soumission DL finale n'était
disponible lors de l'arrêt CPU. La grille peut retenir un poids DL nul si les
validations internes ne soutiennent pas son ajout.
