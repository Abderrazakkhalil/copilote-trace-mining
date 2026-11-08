import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.sinkhorn import cosine_profile_similarities, rebalance_probabilities


def test_global_adjustment_reduces_dominance_without_reviving_wrong_browser():
    p = np.array([[.8, .2, .9], [.7, .3, .1], [.1, .2, .8]])
    classes = np.array(['a', 'b', 'c'])
    mapping = {'a': 'Firefox', 'b': 'Firefox', 'c': 'Opera'}
    browsers = ['Firefox', 'Firefox', 'Opera']
    neutral = rebalance_probabilities(p, [10, 10, 10], browsers, classes, mapping)
    adjusted = rebalance_probabilities(p, [10, 10, 10], browsers, classes, mapping, beta=.5)
    assert adjusted[:2, 0].sum() < neutral[:2, 0].sum()
    assert not adjusted[:2, 2].any() and not adjusted[2, :2].any()
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    np.testing.assert_array_equal(p, [[.8, .2, .9], [.7, .3, .1], [.1, .2, .8]])


def test_profile_centers_use_only_training_examples():
    train = csr_matrix([[1., 0.], [3., 0.], [0., 2.]])
    evaluation = csr_matrix([[1., 1.], [0., 1.]])
    result = cosine_profile_similarities(train, ['a', 'a', 'b'], evaluation, ['a', 'b'])
    np.testing.assert_allclose(result, [[2 ** -.5, 2 ** -.5], [0., 1.]], atol=1e-7)
    with pytest.raises(ValueError, match='No training example'):
        cosine_profile_similarities(train, ['a', 'a', 'b'], evaluation, ['c'])


def test_incompatible_zero_rows_are_rejected():
    with pytest.raises(ValueError, match='browser-compatible'):
        rebalance_probabilities([[1., 0.]], [1, 1], ['Opera'], ['a', 'b'],
                                {'a': 'Firefox', 'b': 'Opera'})
