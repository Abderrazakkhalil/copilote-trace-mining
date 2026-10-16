import pickle, numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from src.postprocess import apply_browser_mask, decision_to_proba

raw = pickle.load(open('data/processed/nested_raw_scores.pkl', 'rb'))
meta = pickle.load(open('data/processed/metadata.pkl', 'rb'))
y = meta['y_train']
classes = np.array(sorted(list(set(y))))
n_classes = len(classes)
u2b = meta['user_to_browser']
b_tr = meta['browsers_train']
b_te = meta['browsers_test']
y_enc = np.array([np.where(classes == u)[0][0] for u in y])

pr_oof = apply_browser_mask(decision_to_proba(raw['raw_oof_scores']['ridge_hyb'], temperature=2.0), b_tr, classes, u2b)
psh_oof = apply_browser_mask(decision_to_proba(raw['raw_oof_scores']['svm_hyb'], temperature=2.5), b_tr, classes, u2b)
pss_oof = apply_browser_mask(decision_to_proba(raw['raw_oof_scores']['svm_seq'], temperature=2.5), b_tr, classes, u2b)

pr_te = apply_browser_mask(decision_to_proba(raw['raw_test_scores']['ridge_hyb'], temperature=2.0), b_te, classes, u2b)
psh_te = apply_browser_mask(decision_to_proba(raw['raw_test_scores']['svm_hyb'], temperature=2.5), b_te, classes, u2b)
pss_te = apply_browser_mask(decision_to_proba(raw['raw_test_scores']['svm_seq'], temperature=2.5), b_te, classes, u2b)

def get_R(P):
    preds = np.argmax(P, axis=1)
    acc = (y_enc == preds).mean()
    R = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((y_enc == c) & (preds == c))
        fp = np.sum((y_enc != c) & (preds == c))
        R[c] = (tp + 3.0 * acc) / (tp + fp + 3.0)
    return R

Rr = get_R(pr_oof)
Rsh = get_R(psh_oof)
Rss = get_R(pss_oof)

mr = (np.sort(pr_te, axis=1)[:, -1] - np.sort(pr_te, axis=1)[:, -2])**0.5
msh = (np.sort(psh_te, axis=1)[:, -1] - np.sort(psh_te, axis=1)[:, -2])**0.5
mss = (np.sort(pss_te, axis=1)[:, -1] - np.sort(pss_te, axis=1)[:, -2])**0.5

# Test across different base weights
for wr in [0.80, 0.85, 0.90]:
    wsh = (1.0 - wr) * 0.6
    wss = (1.0 - wr) * 0.4
    Wr = wr * mr[:, None] * Rr[None, :]
    Wsh = wsh * msh[:, None] * Rsh[None, :]
    Wss = wss * mss[:, None] * Rss[None, :]
    b_te = (Wr * pr_te + Wsh * psh_te + Wss * pss_te)
    p_te = classes[np.argmax(b_te, axis=1)]
    s2 = pd.read_csv('outputs/submissions/submission_ensemble_optimized.csv')
    res = {r: (s2.loc[r-1, 'prediction'], p_te[r-1]) for r in [15, 116, 118, 226, 272, 298, 311]}
    print(f"wr={wr:.2f} -> {res}")
