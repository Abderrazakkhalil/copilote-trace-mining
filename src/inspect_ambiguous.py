from collections import Counter
from src.parser import load_dataset
from src.config import TRAIN_CSV, TEST_CSV

train_sessions = load_dataset(TRAIN_CSV, is_train=True)
test_sessions = load_dataset(TEST_CSV, is_train=False)

ambiguous_rids = [15, 116, 118, 272, 298, 311]
pairs = {
    15: ('txk', 'eoe'),
    116: ('cpn', 'zht'),
    118: ('khv', 'tmn'),
    272: ('guh', 'hvh'),
    298: ('crn', 'rmf'),
    311: ('skm', 'slq'),
}

for rid in ambiguous_rids:
    idx = rid - 1
    s_te = test_sessions[idx]
    c1, c2 = pairs[rid]
    tr_c1 = [s for s in train_sessions if s['user'] == c1]
    tr_c2 = [s for s in train_sessions if s['user'] == c2]
    br = s_te['browser']
    n_act = len(s_te['raw_actions'])
    top_te_ctrl = Counter([c for c in s_te['controllers'] if c]).most_common(2)
    top_te_dom = Counter([d for d in s_te['domains'] if d]).most_common(2)
    top_c1_ctrl = Counter([c for s in tr_c1 for c in s['controllers'] if c]).most_common(2)
    top_c1_dom = Counter([d for s in tr_c1 for d in s['domains'] if d]).most_common(2)
    top_c2_ctrl = Counter([c for s in tr_c2 for c in s['controllers'] if c]).most_common(2)
    top_c2_dom = Counter([d for s in tr_c2 for d in s['domains'] if d]).most_common(2)
    print(f"RowId {rid} (Browser: {br}, #acts: {n_act}):")
    print(f"  Test : ctrl={top_te_ctrl}, dom={top_te_dom}")
    print(f"  [{c1}] ({len(tr_c1)} train): ctrl={top_c1_ctrl}, dom={top_c1_dom}")
    print(f"  [{c2}] ({len(tr_c2)} train): ctrl={top_c2_ctrl}, dom={top_c2_dom}")
    print("-" * 60)
