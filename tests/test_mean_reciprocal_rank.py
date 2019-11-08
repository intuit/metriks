import numpy as np

import metriks


def test_mrr_wikipedia():
    """
    Tests the wikipedia example.

    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    values = [
        ['catten', 'cati', 'cats'],
        ['torii', 'tori', 'toruses'],
        ['viruses', 'virii', 'viri'],
    ]

    # Flag indicating which is actually relevant
    y_true = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])

    # The predicted probability
    y_prob = np.array([
        [0.4, 0.6, 0.3],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.3],
    ])

    # Check results
    expected = np.ma.array([11.0/18.0, 0.0, 0.0], mask=[False, True, True])
    result = metriks.mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_mrr_with_errors():
    """Test MRR where there are errors in the probabilities"""
    y_true = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])

    y_prob = np.array([
        [0.7, 0.6, 0.3],
        [0.9, 0.2, 0.1],  # 3rd probability is an error
        [0.7, 0.8, 0.9],  # 3rd probability is an error
        [0.9, 0.8, 0.3],  # 2nd and 3rd probability are swapped
        [0.4, 0.6, 0.3],
        [0.1, 0.6, 0.9],
        [0.1, 0.6, 0.9],
        [0.1, 0.6, 0.5],
        [0.9, 0.8, 0.7],
    ])

    # Check results
    expected = np.ma.array([11.0 / 18.0, 31.0 / 48.0, 1.0])
    result = metriks.mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_mrr_perfect():
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
    result = metriks.mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_mrr_zeros():
    """Test MRR where there is a sample of all zeros"""
    y_true = np.array([
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 0]
    ])

    y_prob = np.array([
        [0.1, 0.4, 0.35, 0.8],
        [0.3, 0.2, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4]
    ])

    # Check results
    expected = np.ma.array([0.75, 7.0/24.0, 1.0/3.0, 0.0], mask=[False, False, False, True])
    result = metriks.mean_reciprocal_rank(y_true, y_prob)
    np.testing.assert_allclose(result, expected)

    print(result)


if __name__ == '__main__':
    test_mrr_wikipedia()
    test_mrr_with_errors()
    test_mrr_perfect()
    test_mrr_zeros()