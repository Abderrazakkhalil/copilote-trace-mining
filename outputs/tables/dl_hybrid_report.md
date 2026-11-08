# ConvBiGRUAttention — résultat reproductible

                           model  macro_f1  accuracy
                       ridge_hyb  0.969131  0.971333
                         svm_hyb  0.964059  0.966758
                         svm_seq  0.964269  0.967063
                      logreg_hyb  0.927550  0.941751
                         mlp_hyb  0.957000  0.960049
                         xgb_seq  0.722338  0.756938
                       xgb_behav  0.831009  0.842025
                              dl  0.937989  0.946020
             ensemble_without_dl  0.977851  0.978347
                ensemble_with_dl  0.981682  0.981092
dl_outer_best3_diagnostic_BIASED  0.939242  0.945410

Delta Macro-F1 imbriqué : +0.003830; IC bootstrap conditionnel 95 % : [0.0014013799983688985, 0.006443577839845351].
Positive OOF gain with conditional bootstrap interval above zero; independent replication still required.

Soumission : submission_dl_hybrid_final.csv; classes uniques : 226; violations navigateur : 0.

Concordance (nombre de lignes identiques / 324) :
                             ensemble_without_dl  ensemble_with_dl   dl  reference_0.974  historical_dynamic_stacking  reference_0.980
ensemble_without_dl                          324               322  316              322                          322              318
ensemble_with_dl                             322               324  318              320                          320              316
dl                                           316               318  324              317                          318              312
reference_0.974                              322               320  317              324                          323              317
historical_dynamic_stacking                  322               320  318              323                          324              318
reference_0.980                              318               316  312              317                          318              324

File explicitly supplied by user.

Arbitrages : dl_hybrid_arbitrations.csv (toutes les lignes modifiées, Top-1/Top-2 avant/après et DL).

Limites :
- Four fixed outer folds; no claim of proving absence of overfitting.
- Bootstrap conditional on fitted OOF predictions; it does not include training variability.
- Inner 4-fold validation has missing rare-class support (minimum outer-train count=3).
- Best-three checkpoints chosen on outer labels are reported ONLY as a biased diagnostic.
- 0.974/0.980 reference submissions are comparison artifacts, never training inputs.
