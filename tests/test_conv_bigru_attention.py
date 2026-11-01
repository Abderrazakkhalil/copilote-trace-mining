"""Scientific invariants: fold-local transforms, real positions and padding."""
import numpy as np
import torch

from src.conv_bigru_attention import (ConvBiGRUAttentionNet, CosineLinear,
                                      SequencePreprocessor, Trace, read_traces, train_network)
from src.dl_hybrid_pipeline import (MODEL_NAMES, blend, browser_mapping,
                                    classical_predictions, prior_adjust, splits)


def trace(length, ctrl="known", token="open", dt=1.):
    return Trace([(token, ctrl, "screen")] * length, [ctrl] * length,
                 np.full(length, dt, dtype=np.float32))


def test_head_tail_retains_true_positions_and_original_deltas():
    s = trace(300)
    s.deltas = np.arange(300, dtype=np.float32)
    prep = SequencePreprocessor().fit([s])
    tokens, ctrls, tempo, pos = prep.transform([s])
    indices = np.r_[np.arange(128), np.arange(172, 300)]
    np.testing.assert_allclose(pos[0, :, 0], indices / 300, atol=1e-7)
    np.testing.assert_allclose(pos[0, :, 1], np.log1p(indices) / 8, atol=1e-7)
    np.testing.assert_allclose(pos[0, :, 2], np.log1p(300 - indices) / 8, atol=1e-7)
    assert np.flatnonzero(pos[0, :, 3]).tolist() == [128]
    assert np.isclose(tempo[0, 128, 0], (np.log1p(172) - prep.mean) / prep.std)
    assert tokens.shape == ctrls.shape == (1, 256)


def test_unseen_events_and_stats_do_not_enter_training_preprocessor():
    prep = SequencePreprocessor().fit([trace(3), trace(1, token="rare")])
    before = (prep.mean, prep.std, dict(prep.token_vocab), dict(prep.ctrl_stats))
    tokens, ctrls, tempo, pos = prep.transform([trace(4, ctrl="validation_only", dt=1000), trace(1, token="rare")])
    assert (tokens[0, :4] == 1).all()
    assert (ctrls[0, :4] == 1).all()
    assert tokens[1, 0] == 1  # frequency < 2
    assert before == (prep.mean, prep.std, prep.token_vocab, prep.ctrl_stats)
    assert not pos[:, :, 3].any()
    assert not tempo[0, 4:].any()


def test_parser_alignment_glued_markers_and_censored_tail(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("u,Firefox,Click(C)<screen>,Type(C)t6,t10,Exit(D)\n", encoding="utf-8")
    s = read_traces(path, True)[0]
    assert s.tokens == [("Click", "C", "screen"), ("Type", "C", "<none>"), ("Exit", "D", "<none>")]
    np.testing.assert_array_equal(s.deltas, [3, 3, .5])


def test_padding_cannot_change_logits_or_training_statistics():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    prep = SequencePreprocessor().fit([trace(8), trace(5)])
    seq = prep.transform([trace(8), trace(5)])
    inputs = [torch.from_numpy(a[:, :8]) for a in seq] + [torch.randn(2, 7)]
    model = ConvBiGRUAttentionNet(len(prep.token_vocab) + 2, len(prep.ctrl_vocab) + 2, 7, 3)
    model.eval()
    with torch.no_grad():
        short = model(*inputs)
        padded = [torch.cat([x, torch.zeros((2, 4, *x.shape[2:]), dtype=x.dtype)], dim=1) for x in inputs[:4]]
        # Numeric pad features must also be ignored, even when nonzero.
        padded[2][:, 8:] = 123
        padded[3][:, 8:] = -55
        long = model(*padded, inputs[4])
    torch.testing.assert_close(short, long, rtol=1e-5, atol=1e-5)
    # Compare identical training paths with the same dropout/masking RNG.
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    torch.manual_seed(7)
    first = model(*inputs)
    first.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    first_bn = [bn.running_mean.clone() for bn in model.conv_norms]
    model.load_state_dict(state)
    torch.manual_seed(7)
    # Disable token masking for this BN comparison: differing pad extent draws
    # a different number of random values before spatial dropout.
    model.spatial_dropout.p = 0
    # BN's direct valid-step invariant is independent of stochastic layers.
    bn = model.conv_norms[0]
    x = torch.randn(2, 72, 8)
    mask = torch.ones(2, 8, dtype=torch.bool)
    bn.reset_running_stats()
    out_short = bn(x, mask)
    stats = (bn.running_mean.clone(), bn.running_var.clone())
    bn.reset_running_stats()
    out_long = bn(torch.cat([x, torch.randn(2, 72, 4)], dim=2),
                  torch.cat([mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1))
    torch.testing.assert_close(out_short, out_long[:, :, :8])
    torch.testing.assert_close(stats[0], bn.running_mean)
    torch.testing.assert_close(stats[1], bn.running_var)
    assert torch.isfinite(first).all() and first_bn


def test_empty_session_and_cosine_scale_are_finite():
    prep = SequencePreprocessor().fit([trace(2)])
    arrays = prep.transform([trace(0)])
    model = ConvBiGRUAttentionNet(3, 3, 7, 3).eval()
    with torch.no_grad():
        output = model(*(torch.from_numpy(a[:, :2]) for a in arrays), torch.zeros(1, 7))
    assert torch.isfinite(output).all()
    head = CosineLinear(4, 3)
    assert torch.isclose(head.log_scale.exp(), torch.tensor(18.))
    torch.testing.assert_close(head(torch.ones(2, 4)), head(torch.ones(2, 4) * 5))


def test_inner_splits_keep_all_classes_in_training_and_cover_each_row_once():
    y = np.r_[np.repeat(np.arange(246), 4), np.repeat(246, 8)]
    for outer_train, outer_val in splits(y):
        assert len(set(outer_train) & set(outer_val)) == 0
        coverage = np.zeros(len(outer_train), int)
        for train, val in splits(y[outer_train]):
            assert len(np.unique(y[outer_train][train])) == 247
            coverage[val] += 1
        assert (coverage == 1).all()


def test_fusion_never_revives_browser_incompatible_classes():
    p = {"a": np.array([[.9, .1, 0.], [0., .2, .8]]),
         "b": np.array([[.7, .3, 0.], [0., .4, .6]])}
    classes = np.array(["u", "v", "w"])
    mapping = {"u": "Firefox", "v": "Firefox", "w": "Opera"}
    for geo in (0., .5, 1.):
        fused = blend(p, {"a": .8, "b": .2}, geo)
        masked = prior_adjust(fused, [10, 3, 4], .3, ["Firefox", "Opera"], classes, mapping)
        assert masked[0, 2] == masked[1, 0] == masked[1, 1] == 0
        np.testing.assert_allclose(masked.sum(1), 1)


def test_browser_exclusivity_must_be_observed_in_training():
    import pytest
    with pytest.raises(ValueError, match="exclusivity"):
        browser_mapping([{"user": "u", "browser": "Firefox"}, {"user": "u", "browser": "Opera"}])


def test_outer_labels_cannot_change_fixed_checkpoint_predictions():
    torch.set_num_threads(1)
    prep = SequencePreprocessor().fit([trace(5), trace(3)])
    arrays = tuple(a[:, :5] for a in prep.transform([trace(5), trace(3)] * 3))
    arrays += (np.random.default_rng(42).normal(size=(6, 7)).astype(np.float32),)
    y = np.array([0, 1, 2, 0, 1, 2])
    classes = np.array(["a", "b", "c"])
    mapping = dict.fromkeys(classes, "Firefox")
    states = []
    for external_y in (y, (y + 1) % 3):
        model, history = train_network(arrays, y, arrays, external_y, classes,
                                       ["Firefox"] * 6, mapping, 3, 3, epochs=3,
                                       batch_size=4, chosen_epochs=[1, 2, 3], diagnostic_best=True)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        assert history["chosen_epochs"] == [1, 2, 3]
        assert len(history["curve"]) == 3
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def test_all_classical_branches_produce_aligned_probabilities():
    import scipy.sparse as sp
    rng = np.random.default_rng(42)
    train = rng.normal(size=(36, 12)).astype(np.float32)
    evaluation = rng.normal(size=(5, 12)).astype(np.float32)
    # An obvious class signal keeps this an inexpensive API/integration check.
    y = np.tile(np.arange(3), 12)
    train[:, :3] += np.eye(3, dtype=np.float32)[y] * 4
    st, se = sp.csr_matrix(train), sp.csr_matrix(evaluation)
    features = st, se, train, evaluation, st, se, [], None
    results = classical_predictions(features, y, n_classes=3, threads=1)
    assert set(results) == set(MODEL_NAMES)
    for probabilities in results.values():
        assert probabilities.shape == (5, 3)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
