import numpy as np
import metriks


def test_confusion_matrix_at_k_wikipedia():
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

    expected = (
        np.array([1, 0, 1]),
        np.array([1, 2, 1]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 2)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


def test_confusion_matrix_at_k_with_errors():
    """Test confusion_matrix_at_k where there are errors in the probabilities"""
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
    expected = (
        np.array([2, 0, 2]),
        np.array([1, 3, 1]),
        np.array([2, 0, 3]),
        np.array([4, 6, 3])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 2)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


def test_confusion_matrix_at_k_perfect():
    """Test confusion_matrix_at_k where the probabilities are perfect"""
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
    expected = (
        np.array([3, 3, 4]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([2, 2, 1])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 1)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


def test_confusion_matrix_at_k_perfect_multiple_true():
    """Test confusion_matrix_at_k where the probabilities are perfect and there
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
        [0.1, 0.05, 0.0],
        [0.1, 0.5, 0.3],
        [0.6, 0.2, 0.4],
        [0.1, 0.2, 0.3],
    ])

    # Check results for k=0
    expected = (
        np.array([2, 2, 2]),
        np.array([0, 0, 0]),
        np.array([3, 3, 3]),
        np.array([0, 0, 0])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 0)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)

    # Check results for k=2
    expected = (
        np.array([2, 1, 2]),
        np.array([0, 1, 0]),
        np.array([0, 0, 0]),
        np.array([3, 3, 3])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 2)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


def test_confusion_matrix_at_k_few_zeros():
    """Test confusion_matrix_at_k where there are very few zeros"""
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

    # Check results for k=2
    expected = (
        np.array([1, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 1]),
        np.array([2, 3, 2, 0, 1]),
        np.array([0, 0, 1, 3, 1])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 2)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)

    # Check results for k=5
    expected = {
        'moved': {'fp': 1, 'tn': 0, 'fn': 0, 'tp': 2},
        'hadAJob': {'fp': 0, 'tn': 0, 'fn': 0, 'tp': 3},
        'farmIncome': {'fp': 0, 'tn': 0, 'fn': 0, 'tp': 3},
        'married': {'fp': 0, 'tn': 0, 'fn': 0, 'tp': 3},
        'alimony': {'fp': 1, 'tn': 0, 'fn': 0, 'tp': 2}
    }
    expected = (
        np.array([0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 1]),
        np.array([0, 0, 0, 0, 0]),
        np.array([2, 3, 3, 3, 2])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 5)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


def test_confusion_matrix_at_k_zeros():
    """Test confusion_matrix_at_k where there is a sample of all zeros"""
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
    expected = {
        'moved': {'fp': 0, 'tn': 2, 'fn': 1, 'tp': 0},
        'hadAJob': {'fp': 1, 'tn': 1, 'fn': 1, 'tp': 0},
        'farmIncome': {'fp': 1, 'tn': 0, 'fn': 1, 'tp': 1},
        'married': {'fp': 2, 'tn': 0, 'fn': 0, 'tp': 1}
    }
    expected = (
        np.array([2, 1, 0, 0]),
        np.array([0, 1, 1, 2]),
        np.array([1, 1, 1, 0]),
        np.array([0, 0, 1, 1])
    )
    result = metriks.confusion_matrix_at_k(y_true, y_prob, 2)
    for i in range(4):
        assert expected[i].all() == result[i].all(), AssertionError(
            'Expected:\n{expected}. \nGot:\n{actual}'
            .format(expected=expected, actual=result)
        )

    print(result)


if __name__ == '__main__':
    test_confusion_matrix_at_k_wikipedia()
    test_confusion_matrix_at_k_with_errors()
    test_confusion_matrix_at_k_perfect()
    test_confusion_matrix_at_k_perfect_multiple_true()
    test_confusion_matrix_at_k_few_zeros()
    test_confusion_matrix_at_k_zeros()
