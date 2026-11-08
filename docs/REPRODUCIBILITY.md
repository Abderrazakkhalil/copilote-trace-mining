# Traçabilité et reproduction

## Ce que mesurent les résultats

Le rapport final du groupe décrit la progression
publique Kaggle : **97,4 %**, puis **97,2 à 98,0 %**, enfin **98,4 %**. Ces
valeurs proviennent du rapport et des indications du groupe ; les étiquettes
test ne sont pas disponibles pour les recalculer localement.

Le [benchmark CSV](../outputs/tables/benchmark_results.csv) contient les scores
exploratoires à quatre plis des cinq familles sur les trois représentations.
`src.features` y préapprend les vocabulaires et le scaler sur tout le train :
ce benchmark historique ne constitue donc pas une évaluation entièrement
indépendante des transformations. `src.optimize_models` sélectionne aussi les
hyperparamètres et les poids à partir des scores OOF disponibles ; son score
de vote est un score de sélection.

Le pipeline `src.dl_hybrid_pipeline` réapprend les vocabulaires, IDF,
descripteurs dépendant des fréquences, scaler et sélections de variables dans
chaque train de pli. Les quatre plis internes choisissent la fusion et les
époques ; les quatre plis externes évaluent ce choix. Avec les classes rares,
certains trains externes n'ont que trois exemples : une validation interne à
quatre plis ne contient donc pas toutes les classes dans chacun de ses blocs
de validation, même si chaque train interne conserve les 247 classes.

Les [résultats neuronaux conservés](../outputs/tables/dl_hybrid_report.md)
ont été importés de l'export local `results_20261101_223919` :

| Variante évaluée | Macro-F1 externe |
|---|---:|
| Ensemble sans réseau | 97,7851 % |
| Ensemble avec réseau | 98,1682 % |
| Réseau seul | 93,7989 % |

Le gain de fusion vaut 0,383 point. Le bootstrap apparié est conditionnel aux
prédictions déjà apprises ; il ne mesure pas la variabilité de réentraînement.
Le diagnostic « meilleurs checkpoints externes » est biaisé et exclu de la
soumission. Aucun entraînement complet n'a été relancé pendant le nettoyage.

## Écart de longueur de séquence

Le PDF final indique une fenêtre de **126 événements**. Le code et ses tests
disponibles avant nettoyage imposaient **256 événements**, avec les 128
premiers et les 128 derniers pour les longues sessions. Les résultats importés
sont associés à cette implémentation disponible, pas à une exécution démontrée
à 126 événements.

Le prétraitement accepte désormais `--max-length 126` (63 événements au début,
63 à la fin), avec les positions et intervalles temporels d'origine. La valeur
historique de 256 est conservée par défaut. Changer la longueur change
l'empreinte du protocole et isole les sorties :

| Commande / protocole | Soumission | Tables |
|---|---|---|
| 28 époques, batch 64, longueur 256 | `submission_dl_hybrid_final.csv` | `outputs/tables/` |
| 28 époques, batch 64, longueur 126 | `submission_dl_hybrid_len126.csv` | `outputs/tables/dl_hybrid_len126/` |
| Autres époques ou batch | nom contenant `SMOKE_ONLY` | sous-dossier contenant `SMOKE_ONLY` |

Cette option permet d'exécuter la contrainte annoncée ; elle ne démontre pas
que cette nouvelle exécution reproduira le score public de 98,4 %.

## Représentations et rééquilibrage

Les 246 colonnes comportementales sont la dimension historique sur le train
complet. Les vocabulaires d'actions/domaines sont appris par pli ; le nombre
de colonnes peut donc varier. Markov (10 × 10) et RLE sont implémentés dans le
bloc dense, tandis que les n-grammes et ancrages se trouvent dans le bloc
TF-IDF. Le pipeline imbriqué construit l'hybride sous la forme
`[X_seq, 0.25 * L2(StandardScaler(X_behav))]`.

`src.sinkhorn` fournit les documents des cinq premières/dernières actions,
les similarités cosinus aux profils moyens appris sur le train et
l'ajustement global des colonnes inspiré de Sinkhorn. Ce dernier reprend une
étape douce de correction des masses, suivie du masque navigateur ; il ne
s'agit pas d'un solveur itératif de transport optimal. Ses paramètres doivent
être sélectionnés sur des validations internes avant toute mesure externe.

Le passage de **224 à 229 identités** est un constat de l'exploration décrit
dans le rapport. Le script historique comprenait aussi une réaffectation
gloutonne et des ancres imposées à certaines lignes test ; il est archivé.
Le module conservé n'impose aucune identité et ne prétend pas reconstruire à
l'identique cette soumission historique. Il n'est pas appliqué implicitement
au pipeline neuronal final.

## Exécuter et reprendre

Toutes les commandes du README partent de la racine du dépôt. Les chemins de
données et de sorties sont ancrés sur `src/config.py`, indépendamment du
répertoire courant. Les tests sont exécutables sans les données du BE.

Le calcul imbriqué complet comporte 16 entraînements neuronaux internes et
4 externes, ainsi que les modèles classiques correspondants. Prévoir plusieurs
heures sur CPU. Utiliser `--workers 1` et adapter `--threads` à la machine.
Les caches et checkpoints restent dans `data/processed/dl_hybrid_runs/` ; les
écritures atomiques permettent une reprise après arrêt du processus précédent.
Ne pas lancer deux processus écrivant le même cache simultanément.

Les anciens caches et transferts sont dans `_archive_local/` : les changements
de chemins et de sources invalident leurs empreintes. Le pipeline propre
recalcule ses artefacts. `src.dl_hybrid_transfer` reste disponible pour des
transferts futurs dont le manifeste correspond exactement au code courant.

`requirements.txt` contient les dépendances directes ; `requirements-dev.txt`
ajoute pytest. `requirements-repro.txt` enregistre les versions installées sur
la machine de vérification, y compris son build local PyTorch, sans garantir
leur disponibilité sur tous les index ni toutes les plateformes. Le détail
de cet environnement est dans
[cleanup_environment.json](../outputs/tables/cleanup_environment.json).

Les empreintes du PDF, des soumissions conservées et l'identifiant de
l'exécution importée figurent dans
[provenance.json](../outputs/tables/provenance.json). Les sorties importées
restent des témoins de l'exécution passée ; un nouveau calcul écrit ses propres
métadonnées et peut différer selon les versions et le matériel.
