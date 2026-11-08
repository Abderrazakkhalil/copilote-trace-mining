# Plan et bilan du tri

Le tri annoncé avant déplacement a été appliqué le 8 novembre 2026.
Les fichiers ont été déplacés ou sauvegardés, sans suppression définitive.
L'archive `_archive_local/` est ignorée par Git ; son inventaire détaillé se
trouve dans `_archive_local/manifest.csv` sur la machine de travail.

| Conservé dans le dépôt | Destination |
|---|---|
| Parseur, EDA, features, benchmark et optimisation classiques | `src/` |
| Réseau ConvBiGRUAttention, pipeline imbriqué, utilitaires de transfert | `src/` |
| Outils génériques positionnels, profils cosinus et rééquilibrage | `src/sinkhorn.py` |
| Tests du parseur, chemins, réseau, fusion, reprise et transferts | `tests/` |
| Cinq figures d'analyse du rapport | `figures/` |
| Sujet et indications d'origine, utilitaire d'extraction | `docs/assignment/` |
| Benchmark, ablations, EDA, résultats neuronaux et provenance | `outputs/tables/` |
| Trois soumissions jalons | `outputs/submissions/` |
| Format attendu, manifeste des données et modèle de soumission | `data/` |

| Archivé sur la machine | Chemin dans `_archive_local/` |
|---|---|
| Scripts neural boost, chrono, essais alternatifs, stacking et pseudo-labeling | `src/` |
| Ancien script Sinkhorn avec ancres par ligne, ancien nested sur matrices globales | `src/` |
| `test_weights`, diagnostics de lignes ambiguës, notes temporaires | `src/`, `scratch_doc_notes.py` |
| Tests spécifiques aux variantes retirées | `tests/` |
| Notebooks Colab, notebook de travail du sujet, lanceurs CPU | `notebooks/`, `enonce/`, `scripts/` |
| Anciens transferts et leurs guides | `artifacts/`, `docs/` |
| Exports de résultats et ZIP, dont les sorties finales recopiées dans `outputs/` | dossiers datés, fichiers ZIP |
| Matrices, probabilités, checkpoints et caches | `data/processed/` |
| Soumissions intermédiaires, dont corrections chirurgicales | `outputs/submissions/` |
| Journaux, audits de diagnostic et figures supplémentaires | `outputs/` |
| Anciennes versions des fichiers réécrits | `before_cleanup/` |
| Dossier du rapport, PDF, source LaTeX, README et illustration retirés à la demande du groupe | `report_removed/` |

Le déplacement initial couvre **213 fichiers, 578 543 882 octets** (environ
552 Mio), auxquels s'ajoutent les sauvegardes des fichiers réécrits.
Les CSV train/test restent en place dans `data/raw/`, exclus du suivi Git.
Leur présence dans les anciens commits n'est pas modifiée.

Pour retrouver un fichier, consulter l'inventaire et copier sa destination
archivée vers son ancien chemin. Réintroduire un script exploratoire peut
nécessiter de restaurer ses dépendances et ses données intermédiaires : cette
archive conserve l'état de travail, elle n'est pas un second pipeline maintenu.

Les limites scientifiques ou les écarts avec le PDF sont documentés dans
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Vérifications effectuées : 32 tests réussis, métriques EDA régénérées à
l'identique, cinq figures reconstruites et matrices de 15 139 variables
séquentielles et 246 descripteurs comportementaux sans NaN ni infini.
Les trois soumissions respectent leurs 324 lignes et le masque navigateur ;
les liens locaux et les empreintes des soumissions sont contrôlés.
