import numpy as np

import metriks


def test_precision_at_k_wikipedia():
    """
    Tests the wikipedia example.

    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """

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
    expected = 2.0/6.0
    result = metriks.precision_at_k(y_true, y_prob, 2)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_precision_at_k_with_errors():
    """Test precision_at_k where there are errors in the probabilities"""
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
    expected = 18.0/27.0
    result = metriks.precision_at_k(y_true, y_prob, 3)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_precision_at_k_perfect():
    """Test precision_at_k where the probabilities are perfect"""
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
    expected = 1.0
    result = metriks.precision_at_k(y_true, y_prob, 1)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_precision_at_k_perfect_multiple_true():
    """Test precision_at_k where the probabilities are perfect and there
    are multiple true labels for some samples"""
    y_true = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
    ])

    y_prob = np.array([
        [0.3, 0.7, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.5, 0.3],
        [0.6, 0.2, 0.4],
        [0.1, 0.2, 0.3],
    ])

    # Check results for k=1
    expected = 1.0
    result = metriks.precision_at_k(y_true, y_prob, 1)
    np.testing.assert_allclose(result, expected)

    print(result)

    # Check results for k=2
    expected = 9.0/10.0
    result = metriks.precision_at_k(y_true, y_prob, 2)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_precision_at_k_few_zeros():
    """Test precision_at_k where there are very few zeros"""
    y_true = np.array([
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]
    ])

    y_prob = np.array([
        [0.1, 0.4, 0.35, 0.8, 0.9],
        [0.3, 0.2, 0.7, 0.8, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5]
    ])

    # Check results
    expected = 5.0/6.0
    result = metriks.precision_at_k(y_true, y_prob, 2)
    np.testing.assert_allclose(result, expected)

    print(result)


def test_precision_at_k_zeros():
    """Test precision_at_k where there is a sample of all zeros"""
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
    expected = 4.0/9.0
    result = metriks.precision_at_k(y_true, y_prob, 3)
    np.testing.assert_allclose(result, expected)

    print(result)


if __name__ == '__main__':
    test_precision_at_k_wikipedia()
    test_precision_at_k_with_errors()
    test_precision_at_k_perfect()
    test_precision_at_k_perfect_multiple_true()
    test_precision_at_k_few_zeros()
    test_precision_at_k_zeros()
