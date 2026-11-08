# Soumissions jalons conservées

| Fichier | Rôle |
|---|---|
| `submission_ensemble_optimized.csv` | Référence classique à 97,4 % publics, identifiée dans les archives du projet |
| `submission_hada_jehd.csv` | Référence intermédiaire à 98,0 %, identifiée par le groupe |
| `submission_dl_hybrid_final.csv` | Sortie finale importée de l'exécution locale du pipeline imbriqué |

Chaque fichier contient 324 prédictions au format `RowId,prediction`.
Les références ne servent qu'aux comparaisons de concordance du pipeline ;
elles ne sont pas des étiquettes d'apprentissage.

Le rapport annonce 98,4 % sur le leaderboard public pour le palier neuronal.
La correspondance exacte entre la soumission envoyée sur Kaggle et le fichier
neuronal local ne peut pas être certifiée par les seuls artefacts disponibles.
Voir [la traçabilité](../../docs/REPRODUCIBILITY.md).
