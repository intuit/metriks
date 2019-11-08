import numpy as np

import metriks


def test_label_mrr_with_errors():
    y_true = np.array([[0, 1, 1, 0],
                       [1, 1, 1, 0],
                       [0, 1, 1, 1]])
    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.9, 0.2, 0.8, 0.4],
                       [0.2, 0.8, 0.9, 0.4]])

    result = metriks.label_mean_reciprocal_rank(y_true, y_prob)

    expected = np.ma.array([1.0, (7/4) / 3, 2/3, 1/3])

    np.testing.assert_allclose(result, expected)
    print(result)


def test_label_mrr_perfect():
    """Test MRR where the probabilities are perfect"""
    y_true = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    y_prob = np.array([
        [0.3, 0.7, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.5, 0.3],
        [0.6, 0.2, 0.4],
        [0.1, 0.2, 0.3],
    ])

    # Check results
    expected = np.ma.array([1.0, 0.0, 0.0], mask=[False, True, True])
    result = metriks.label_mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_label_mrr_zeros():
    """Test MRR where no relevant labels"""
    y_true = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    y_prob = np.array([
        [0.3, 0.7, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.5, 0.3]
    ])

    # Check results
    result = metriks.label_mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_equal(result.mask.all(), True)

    print(result)


def test_label_mrr_some_zeros():
    """Test MRR where some relevant labels"""
    y_true = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])

    y_prob = np.array([
        [0.3, 0.7, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.5, 0.3]
    ])

    # Check results
    expected = np.ma.array([0.0, 1.0, 0.0], mask=[True, False, True])
    result = metriks.label_mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_label_mrr_ones():
    """Test MRR where all labels are relevant and predictions are perfect"""
    y_true = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    y_prob = np.array([
        [0.3, 0.7, 0.0],
        [0.1, 0.9, 0.0],
        [0.2, 0.5, 0.1]
    ])

    # Check results
    expected = np.ma.array([0.5, 1.0, 1/3])
    result = metriks.label_mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


if __name__ == '__main__':
    test_label_mrr_with_errors()
    test_label_mrr_perfect()
    test_label_mrr_zeros()
    test_label_mrr_some_zeros()
    test_label_mrr_ones()


