# Audit préalable — ConvBiGRUAttention

Les sources et caches ont été inspectés avant l'implémentation du nouveau module.

- Données : 3 279 sessions train, 324 test, 247 utilisateurs ; minimum 4 sessions
  par classe. Ordre des lignes conservé, RowId tirés du sample_submission.
- `parser.py` : découpage du CSV sans en-tête, réparation des `tX` collés,
  décomposition action/contrôleur/configuration/domaine/suffixe ; comptage des
  actions entre marqueurs temporels, puis statistiques par fenêtre.
- `features.py`, Dataset 1 : TF-IDF sur tokens complets (1–2 grammes, max 8 000),
  actions (1–3, max 2 500), contrôleurs (1–2, max 3 000), configurations
  (1–2, max 1 000), domaines (1–2, max 500), plus ancres début/fin OneHot.
  Cache train : `(3279, 15139)` ; TF sous-linéaire, normalisation L2 par bloc.
- Dataset 2 : `(3279, 246)` ; navigateur/volume, rythme par fenêtre, diversité,
  entropies, ergonomie, répétitions RLE, 100 transitions de Markov, fréquences
  relatives d'actions et domaines. Les listes de colonnes sont apprises du train.
- Hybride initial : dense standardisé concaténé au sparse. Hybride L2 optimisé :
  `[TF-IDF, 0.25 * normalize(StandardScaler(X_behav))]`, 15 385 colonnes.
- `postprocess.apply_browser_mask` : annule les classes dont le navigateur
  exclusif observé dans le train diffère de celui de la session ; renormalise.
  Le nouveau pipeline vérifie l'exclusivité et reconstruit la table par pli.
- Découpage externe conservé : `StratifiedKFold(4, shuffle=True, random_state=42)`.

## Défauts à ne pas propager

Les TF-IDF, listes comportementales et certains scalers des caches historiques
ont été ajustés sur tout le train avant les découpages. Cela expose les
transformations aux observations de validation, même sans utiliser leurs labels.
L'alpha hybride 0,25 a aussi été choisi historiquement sur un pli. Il est ici
traité comme une constante héritée, sans nouveau réglage sur les plis externes.
Les scores historiques sont donc des repères exploratoires, pas une preuve
imbriquée de généralisation.

`chrono_data.py` utilise les actions de base sans configuration, apprend son
vocabulaire sur tout le train et groupe ses statistiques de temps par action,
malgré le nom `screen_stats`. `clean_scientific_pipeline.py` standardise aussi
le comportemental avant les plis du réseau existant. Aucun de ces caches n'est
réutilisé pour l'évaluation stricte du nouveau modèle.

Certains scripts (`break_0985_pipeline.py`, `sinkhorn_positional_ensemble.py`)
contiennent des ancres ou décisions manuelles par RowId. Ils ne sont ni importés
ni exécutés. Les soumissions historiques ne sont lues qu'après la prédiction
finale, pour produire une concordance descriptive.

## Conventions temporelles et protocole retenu

Les marqueurs `tX` ne fournissent pas de latence observée par action. Comme
l'extracteur chrono existant, on partage uniformément chaque intervalle entre
les actions qui le précèdent ; la queue sans marqueur final reçoit 0,5 seconde.
Les deltas sont calculés avant la coupure Head+Tail. Les tokens sont les triplets
`(action, contrôleur/écran, configuration)` et la branche contrôleur conserve
séparément l'information hiérarchique. Positions originales indexées à partir
de zéro : `t/L`, `log1p(t)/8`, `log1p(L-t)/8`, saut uniquement à l'indice 128.
Le second temps est un z-score en espace `log1p(dt)`, centré sur la médiane du
contrôleur et divisé par son écart-type ; écrêtage fixé à [-5, 5].

Tout vocabulaire, IDF, liste de colonnes, scaler, SVD et sélection supervisée
est réappris sur le sous-ensemble d'entraînement correspondant. Le padding est
exclu des statistiques BatchNorm temporelles et du BiGRU via packed sequences.
Une session vide est représentée par une étape UNK synthétique.

Une classe de 4 exemples n'en conserve que 3 dans le train externe. Les quatre
plis internes sont conservés : certaines validations internes n'ont donc pas
toutes les classes ; chaque train interne les conserve toutes. Le Macro-F1
utilise explicitement les 247 labels et `zero_division=0`.

Les prédictions internes utilisent une moyenne prédéfinie des trois derniers
checkpoints. Les trois époques de l'entraînement externe sont sélectionnées
à partir des courbes de validation INTERNES uniquement. Après moyenne des poids,
BatchNorm est recalculée uniquement sur le train correspondant. La moyenne des
trois meilleurs checkpoints selon la validation externe demandée est également
calculée, mais uniquement comme diagnostic signalé biaisé : elle n'alimente
ni la fusion imbriquée ni la soumission.

La recherche de fusion porte sur une grille fixe de poids, de mélange
arithmétique/géométrique et de logit adjustment bayésien. Aucune contrainte sur
le nombre de classes test n'est imposée. Sinkhorn n'est pas nécessaire puisque
l'alternative logit adjustment demandée est implémentée.

Références méthodologiques :
- [scikit-learn : validation imbriquée](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [PyTorch : packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
- [PyTorch : moyenne des poids et BatchNorm](https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema)

La référence 0,974 est documentée comme `submission_ensemble_optimized.csv`.
L'utilisateur a identifié `submission_hada_jehd.csv` comme référence 0,980 durant
l'exécution. Elle est fournie avec `--reference-0980 submission_hada_jehd.csv`,
uniquement pour le rapport final. `submission_dynamic_stacking.csv` est aussi
comparé sous son nom historique, sans lui attribuer de score Kaggle.

## Arrêt et migration sur GPU

À la demande de l'utilisateur, le processus CPU a été arrêté après 23 époques
complètes du premier entraînement interne. Aucune de ces époques n'avait été
persistée : le réseau repartira de l'époque 1. Le cache des sept modèles
classiques du même pli est conservé dans `artifacts/dl_hybrid_transfer/`, avec
validation du contrat de construction, des données, des classes et des indices.

La version préparée pour GitHub ajoute un checkpoint atomique à chaque époque
(poids, optimiseur, scheduler, états aléatoires, checkpoints sélectionnés et
courbes). Le nom du cache ne dépend plus du matériel ni des fins de lignes ;
chaque environnement est enregistré dans l'historique d'exécution. Un test CPU
vérifie l'égalité exacte des poids entre entraînement continu et reprise après
interruption. L'exécution CUDA devra être vérifiée sur le PC GPU.
