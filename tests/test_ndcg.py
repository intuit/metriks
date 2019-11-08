import numpy as np
import metriks


def test_ndcg_perfect():
    """
    All predicted rankings match the expected ranking
    """
    y_true = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 1],
                       [1, 1, 1, 0]])

    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.4, 0.8, 0.7, 0.5],
                       [0.9, 0.6, 0.7, 0.2]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_equal([actual], [expected])


def test_ndcg_errors():
    """
    Some samples predicted the order incorrectly
    """
    y_true = np.array([[1, 1, 1, 0],
                       [1, 0, 1, 1],
                       [0, 1, 1, 0]])

    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.5, 0.8, 0.7, 0.4],
                       [0.9, 0.6, 0.7, 0.2]])

    expected = 0.7979077
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_all_ones():
    """
    Every item in each sample is relevant
    """
    y_true = np.array([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]])

    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.5, 0.8, 0.7, 0.4],
                       [0.9, 0.6, 0.7, 0.2]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_allclose([actual], [expected])


def test_all_zeros():
    """
    There are no relevant items in any sample
    """
    y_true = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])

    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.5, 0.8, 0.7, 0.4],
                       [0.9, 0.6, 0.7, 0.2]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_allclose([actual], [expected])


def test_combination():
    """
    Mixture of all relevant, no relevant, and some relevant samples
    """
    y_true = np.array([[0, 0, 0, 0],
                       [0, 1, 1, 1],
                       [1, 1, 1, 1]])

    y_prob = np.array([[0.2, 0.9, 0.8, 0.4],
                       [0.5, 0.8, 0.7, 0.4],
                       [0.9, 0.6, 0.7, 0.2]])

    expected = 0.989156
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_after_0():
    """
    ndcg with no classes removed
    """
    y_true = np.array([[0, 1, 1, 0, 0, 1],
                       [0, 1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 0],
                       [1, 1, 0, 0, 1, 0]])

    y_prob = np.array([[0.5, 0.9, 0.8, 0.4, 0.2, 0.2],
                       [0.9, 0.8, 0.2, 0.5, 0.2, 0.4],
                       [0.2, 0.9, 0.8, 0.4, 0.6, 0.5],
                       [0.2, 0.7, 0.8, 0.4, 0.6, 0.9],
                       [0.2, 0.9, 0.8, 0.5, 0.3, 0.2]])

    expected = 0.8395053
    actual = metriks.ndcg(y_true, y_prob)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_after_1():
    """
    ndcg with top 1 class removed
    """
    y_true = np.array([[0, 1, 1, 0, 0, 1],
                       [1, 1, 1, 0, 1, 0],
                       [1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0, 1],
                       [1, 1, 0, 0, 1, 1]])

    y_prob = np.array([[0.5, 0.9, 0.8, 0.4, 0.2, 0.2],
                       [0.9, 0.8, 0.2, 0.5, 0.2, 0.4],
                       [0.2, 0.9, 0.8, 0.4, 0.6, 0.5],
                       [0.2, 0.7, 0.8, 0.4, 0.6, 0.9],
                       [0.2, 0.2, 0.8, 0.5, 0.3, 0.9]])

    expected = 0.8554782
    actual = metriks.ndcg(y_true, y_prob, 1)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_after_2_perfect():
    """
    ndcg with top 2 classes removed
    """
    y_true = np.array([[1, 1, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 1],
                       [0, 1, 1, 1, 1, 1],
                       [0, 1, 1, 0, 1, 1],
                       [0, 0, 1, 1, 1, 1]])

    y_prob = np.array([[0.5, 0.9, 0.8, 0.4, 0.2, 0.2],
                       [0.9, 0.8, 0.2, 0.5, 0.2, 0.4],
                       [0.2, 0.9, 0.8, 0.4, 0.6, 0.5],
                       [0.2, 0.7, 0.8, 0.4, 0.6, 0.9],
                       [0.2, 0.2, 0.8, 0.5, 0.3, 0.9]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob, 2)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_after_2_errors():
    """
    ndcg with top 2 classes removed
    """
    y_true = np.array([[0, 1, 1, 0, 0, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1],
                       [0, 0, 1, 1, 1, 1]])

    y_prob = np.array([[0.5, 0.9, 0.8, 0.4, 0.2, 0.2],
                       [0.9, 0.8, 0.2, 0.5, 0.2, 0.4],
                       [0.2, 0.9, 0.8, 0.4, 0.6, 0.5],
                       [0.2, 0.7, 0.8, 0.4, 0.6, 0.9],
                       [0.2, 0.2, 0.8, 0.5, 0.3, 0.9]])

    expected = 0.8139061
    actual = metriks.ndcg(y_true, y_prob, 2)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_1_over_0_error():
    """
    ndcg with division by 0
    """
    y_true = np.array([[0, 1, 1, 1]])

    y_prob = np.array([[0.9, 0.8, 0.7, 0.6]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob, 1)

    np.testing.assert_allclose([actual], [expected])


def test_ndcg_0_over_0_error():
    """
    ndcg with division by 0
    """
    y_true = np.array([[0, 0, 0, 0]])

    y_prob = np.array([[0.9, 0.8, 0.7, 0.6]])

    expected = 1.0
    actual = metriks.ndcg(y_true, y_prob, 1)

    np.testing.assert_allclose([actual], [expected])

if __name__ == '__main__':
    test_ndcg_perfect()
    test_ndcg_errors()
    test_ndcg_all_ones()
    test_all_zeros()
    test_combination()
    test_ndcg_after_0()
    test_ndcg_after_1()
    test_ndcg_after_2_perfect()
    test_ndcg_after_2_errors()
    test_ndcg_1_over_0_error()
    test_ndcg_0_over_0_error()
