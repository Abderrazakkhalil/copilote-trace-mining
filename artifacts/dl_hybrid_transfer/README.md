# Cache classique transféré depuis le PC CPU

Ce paquet contient `outer1_inner1_classical.pkl` (environ 4,1 Mio) et son manifeste.
Il préserve les probabilités de validation de sept modèles déjà entraînés :
Ridge hybride, SVM hybride, SVM séquentiel, LogisticRegression hybride, MLP
hybride, XGBoost séquentiel et XGBoost comportemental.

Le manifeste documente les données, le code de construction et de modélisation,
les classes, les indices de plis, les versions initiales et le SHA-256 du cache.
Le pipeline importe automatiquement ce paquet après vérification. Les CSV de
référence Kaggle ne servent pas à construire ce cache.

Le réseau CPU avait terminé 23 époques lorsque le processus a été arrêté.
**Aucun poids de ces époques n'avait été sauvegardé**. Le réseau doit donc
recommencer à l'époque 1 ; ce paquet ne prétend pas être un checkpoint neuronal.

Le nouveau code sauvegarde chaque époque future dans `data/processed/`.
Instructions de transfert et de reprise : [GPU_REPRISE.md](../../docs/GPU_REPRISE.md).
