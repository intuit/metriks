"""
Ranking
=======
Metrics to use for ranking models.
"""

import numpy as np


def check_arrays(y_true, y_prob):
    # Make sure that inputs this conforms to our expectations
    assert isinstance(y_true, np.ndarray), AssertionError(
        'Expect y_true to be a {expected}. Got {actual}'
        .format(expected=np.ndarray, actual=type(y_true))
    )

    assert isinstance(y_prob, np.ndarray), AssertionError(
        'Expect y_prob to be a {expected}. Got {actual}'
        .format(expected=np.ndarray, actual=type(y_prob))
    )

    assert y_true.shape == y_prob.shape, AssertionError(
        'Shapes must match. Got y_true={true_shape}, y_prob={prob_shape}'
        .format(true_shape=y_true.shape, prob_shape=y_prob.shape)
    )

    assert len(y_true.shape) == 2, AssertionError(
        'Shapes should be of rank 2. Got {rank}'
        .format(rank=len(y_true.shape))
    )

    uniques = np.unique(y_true)
    assert len(uniques) <= 2, AssertionError(
        'Expected labels: [0, 1]. Got: {uniques}'
        .format(uniques=uniques)
    )


def check_k(n_items, k):
    # Make sure that inputs conform to our expectations
    assert isinstance(k, int), AssertionError(
        'Expect k to be a {expected}. Got {actual}'
        .format(expected=int, actual=type(k))
    )

    assert 0 <= k <= n_items, AssertionError(
        'Expect 0 <= k <= {n_items}. Got {k}'
        .format(n_items=n_items, k=k)
    )


def recall_at_k(y_true, y_prob, k):
    """
    Calculates recall at k for binary classification ranking problems. Recall
    at k measures the proportion of total relevant items that are found in the
    top k (in ranked order by y_prob). If k=5, there are 6 total relevant documents,
    and 3 of the top 5 items are relevant, the recall at k will be 0.5.

    Samples where y_true is 0 for all labels are filtered out because there will be
    0 true positives and false negatives.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)
        y_prob (~np.ndarray): The predicted probability that the given flag
            is relevant. size=(n_samples, n_items)
        k (int): Number of items to evaluate for relevancy, in descending
            sorted order by y_prob

    Returns:
        recall (~np.ndarray): The recall at k

    Example:
    >>> y_true = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])
    >>> y_prob = np.array([
        [0.4, 0.6, 0.3],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.3],
    ])
    >>> recall_at_k(y_true, y_prob, 2)
    0.6666666666666666

    In the example above, each of the samples has 1 total relevant document.
    For the first sample, there are 0 relevant documents in the top k for k=2,
    because 0.3 is the 3rd value for y_prob in descending order. For the second
    sample, there is 1 relevant document in the top k, because 0.2 is the 2nd
    value for y_prob in descending order. For the third sample, there is 1
    relevant document in the top k, because 0.9 is the 1st value for y_prob in
    descending order. Averaging the values for all of these samples (0, 1, 1)
    gives a value for recall at k of 2/3.
    """
    check_arrays(y_true, y_prob)
    check_k(y_true.shape[1], k)

    # Filter out rows of all zeros
    mask = y_true.sum(axis=1).astype(bool)
    y_prob = y_prob[mask]
    y_true = y_true[mask]

    # Extract shape components
    n_samples, n_items = y_true.shape

    # List of locations indexing
    y_prob_index_order = np.argsort(-y_prob)
    rows = np.reshape(np.arange(n_samples), (-1, 1))
    ranking = y_true[rows, y_prob_index_order]

    # Calculate number true positives for numerator and number of relevant documents for denominator
    num_tp = np.sum(ranking[:, :k], axis=1)
    num_relevant = np.sum(ranking, axis=1)
    # Calculate recall at k
    recall = np.mean(num_tp / num_relevant)

    return recall


def precision_at_k(y_true, y_prob, k):
    """
    Calculates precision at k for binary classification ranking problems.
    Precision at k measures the proportion of items in the top k (in ranked
    order by y_prob) that are relevant (as defined by y_true). If k=5, and
    3 of the top 5 items are relevant, the precision at k will be 0.6.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)
        y_prob (~np.ndarray): The predicted probability that the given flag
            is relevant. size=(n_samples, n_items)
        k (int): Number of items to evaluate for relevancy, in descending
            sorted order by y_prob

    Returns:
        precision_k (~np.ndarray): The precision at k

    Example:
    >>> y_true = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])
    >>> y_prob = np.array([
        [0.4, 0.6, 0.3],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.3],
    ])
    >>> precision_at_k(y_true, y_prob, 2)
    0.3333333333333333

    For the first sample, there are 0 relevant documents in the top k for k=2,
    because 0.3 is the 3rd value for y_prob in descending order. For the second
    sample, there is 1 relevant document in the top k, because 0.2 is the 2nd
    value for y_prob in descending order. For the third sample, there is 1
    relevant document in the top k, because 0.9 is the 1st value for y_prob in
    descending order. Because k=2, the values for precision of k for each sample
    are 0, 1/2, and 1/2 respectively. Averaging these gives a value for precision
    at k of 1/3.
    """
    check_arrays(y_true, y_prob)
    check_k(y_true.shape[1], k)

    # Extract shape components
    n_samples, n_items = y_true.shape

    # List of locations indexing
    y_prob_index_order = np.argsort(-y_prob)
    rows = np.reshape(np.arange(n_samples), (-1, 1))
    ranking = y_true[rows, y_prob_index_order]

    # Calculate number of true positives for numerator
    num_tp = np.sum(ranking[:, :k], axis=1)
    # Calculate precision at k
    precision = np.mean(num_tp / k)

    return precision


def mean_reciprocal_rank(y_true, y_prob):
    """
    Gets a positional score about how well you did at rank 1, rank 2,
    etc. The resulting vector is of size (n_items,) but element 0 corresponds to
    rank 1 not label 0.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)
        y_prob (~np.ndarray): The predicted probability that the given flag
            is relevant. size=(n_samples, n_items)

    Returns:
        mrr (~np.ma.array): The positional ranking score. This will be masked
            for ranks where there were no relevant values. size=(n_items,)
    """

    check_arrays(y_true, y_prob)

    # Extract shape components
    n_samples, n_items = y_true.shape

    # Determine the ranking order
    rank_true = np.flip(np.argsort(y_true, axis=1), axis=1)
    rank_prob = np.flip(np.argsort(y_prob, axis=1), axis=1)

    # Compute reciprocal ranks
    reciprocal = 1.0 / (np.argsort(rank_prob, axis=1) + 1)

    # Now order the reciprocal ranks by the true order
    rows = np.reshape(np.arange(n_samples), (-1, 1))
    cols = rank_true
    ordered = reciprocal[rows, cols]

    # Create a masked array of true labels only
    ma = np.ma.array(ordered, mask=np.isclose(y_true[rows, cols], 0))
    return ma.mean(axis=0)


def label_mean_reciprocal_rank(y_true, y_prob):
    """
    Determines the average rank each label was placed across samples. Only labels that are
    relevant in the true data set are considered in the calculation.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)that
        y_prob (~np.ndarray): The predicted probability  the given flag
            is relevant. size=(n_samples, n_items)
    Returns:
        mrr (~np.ma.array): The positional ranking score. This will be masked
            for ranks where there were no relevant values. size=(n_items,)
    """

    check_arrays(y_true, y_prob)

    rank_prob = np.flip(np.argsort(y_prob, axis=1), axis=1)
    reciprocal = 1 / (np.argsort(rank_prob, axis=1) + 1)
    ma = np.ma.array(reciprocal, mask=~y_true.astype(bool))

    return ma.mean(axis=0)


def ndcg(y_true, y_prob, k=0):
    """
    A score for measuring the quality of a set of ranked results. The resulting score is between 0 and 1.0 -
    results that are relevant and appear earlier in the result set are given a heavier weight, so the
    higher the score, the more relevant your results are

    The optional k param is recommended for data sets where the first few labels are almost always ranked first,
    and hence skew the overall score.  To compute this "NDCG after k" metric, we remove the top k (by predicted
    probability) labels and compute NDCG as usual for the remaining labels.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)that
        y_prob (~np.ndarray): The predicted probability  the given flag
            is relevant. size=(n_samples, n_items)
        k (int): Optional, the top k classes to exclude
    Returns:
        ndcg (~np.float64): The normalized dcg score across all queries, excluding the top k
    """
    # Get the sorted prob indices in descending order
    rank_prob = np.flip(np.argsort(y_prob, axis=1), axis=1)

    # Get the sorted true indices in descending order
    rank_true = np.flip(np.argsort(y_true, axis=1), axis=1)

    prob_samples, prob_items = y_prob.shape
    true_samples, true_items = y_true.shape

    # Compute DCG

    # Order y_true and y_prob by y_prob order indices
    prob_vals = y_prob[np.arange(prob_samples).reshape(prob_samples, 1), rank_prob]
    true_vals = y_true[np.arange(true_samples).reshape(true_samples, 1), rank_prob]

    # Remove the first k columns
    prob_vals = prob_vals[:, k:]
    true_vals = true_vals[:, k:]

    rank_prob_k = np.flip(np.argsort(prob_vals, axis=1), axis=1)

    n_samples, n_items = true_vals.shape

    values = np.arange(n_samples).reshape(n_samples, 1)

    # Construct the dcg numerator, which are the relevant items for each rank
    dcg_numerator = true_vals[values, rank_prob_k]

    # Construct the denominator, which is the log2 of the current rank + 1
    position = np.arange(1, n_items + 1)
    denominator = np.log2(np.tile(position, (n_samples, 1)) + 1.0)

    dcg = np.sum(dcg_numerator / denominator, axis=1)

    # Compute IDCG
    rank_true_idcg = np.flip(np.argsort(true_vals, axis=1), axis=1)

    idcg_true_samples, idcg_true_items = rank_true_idcg.shape

    # Order y_true indices
    idcg_true_vals = true_vals[np.arange(idcg_true_samples).reshape(idcg_true_samples, 1), rank_true_idcg]

    rank_true_k = np.flip(np.argsort(idcg_true_vals, axis=1), axis=1)

    idcg_numerator = idcg_true_vals[values, rank_true_k]

    idcg = np.sum(idcg_numerator / denominator, axis=1)

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        sample_ndcg = np.divide(dcg, idcg)

    # ndcg may be NaN if idcg is 0; this happens when there are no relevant documents
    # in this case, showing anything in any order should be considered correct
    where_are_nans = np.isnan(sample_ndcg)
    where_are_infs = np.isinf(sample_ndcg)
    sample_ndcg[where_are_nans] = 1.0
    sample_ndcg[where_are_infs] = 0.0

    return np.mean(sample_ndcg, dtype=np.float64)


def generate_y_pred_at_k(y_prob, k):
    """
    Generates a matrix of binary predictions from a matrix of probabilities
    by evaluating the top k items (in ranked order by y_prob) as true.

    In the case where multiple probabilities for a sample are identical, the
    behavior is undefined in terms of how the probabilities are ranked by argsort.

    Args:
        y_prob (~np.ndarray): The predicted probability that the given flag
            is relevant. size=(n_samples, n_items)
        k (int): Number of items to evaluate as true, in descending
            sorted order by y_prob

    Returns:
        y_pred (~np.ndarray): A binary prediction that the given flag is
            relevant. size=(n_samples, n_items)

    Example:
    >>> y_prob = np.array([
        [0.4, 0.6, 0.3],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.3],
    ])
    >>> generate_y_pred_at_k(y_prob, 2)
    array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 1, 0]
    ])

    For the first sample, the top 2 values for y_prob are 0.6 and 0.4, so y_pred
    at those positions is 1. For the second sample, the top 2 values for y_prob
    are 0.9 and 0.2, so y_pred at these positions is 1. For the third sample, the
    top 2 values for y_prob are 0.9 and 0.6, so y_pred at these positions in 1.
    """
    n_items = y_prob.shape[1]
    index_array = np.argsort(y_prob, axis=1)
    col_idx = np.arange(y_prob.shape[0]).reshape(-1, 1)
    y_pred = np.zeros(np.shape(y_prob))
    y_pred[col_idx, index_array[:, n_items-k:n_items]] = 1
    return y_pred


def confusion_matrix_at_k(y_true, y_prob, k):
    """
    Generates binary predictions from probabilities by evaluating the top k items
    (in ranked order by y_prob) as true. Uses these binary predictions along with
    true flags to calculate the confusion matrix per label for binary
    classification problems.

    Args:
        y_true (~np.ndarray): Flags (0, 1) which indicate whether a column is
            relevant or not. size=(n_samples, n_items)
        y_prob (~np.ndarray): The predicted probability that the given flag
            is relevant. size=(n_samples, n_items)
        k (int): Number of items to evaluate as true, in descending
            sorted order by y_prob

    Returns:
        tn, fp, fn, tp (tuple of ~np.ndarrays): A tuple of ndarrays containing
            the number of true negatives (tn), false positives (fp),
            false negatives (fn), and true positives (tp) for each item. The
            length of each ndarray is equal to n_items

    Example:
    >>> y_true = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])
    >>> y_prob = np.array([
        [0.4, 0.6, 0.3],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.3],
    ])
    >>> y_pred = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 1, 0]
    ])
    >>> label_names = ['moved', 'hadAJob', 'farmIncome']
    >>> confusion_matrix_at_k(y_true, y_prob, 2)
    (
        np.array([1, 0, 1]),
        np.array([1, 2, 1]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0])
    )

    In the example above, y_pred is not passed into the function, but is
    generated by calling generate_y_pred_at_k with y_prob and k.

    For the first item (moved), the first sample is a false positive, the
    second is a true negative, and the third is a true positive.
    For the second item (hadAJob), the first and third samples are false
    positives, and the second is a true positive.
    For the third item (farmIncome), the first item is a false negative, the
    second is a false positive, and the third is a true positive.
    """
    check_arrays(y_true, y_prob)
    check_k(y_true.shape[1], k)

    y_pred = generate_y_pred_at_k(y_prob, k)

    tp = np.count_nonzero(y_pred * y_true, axis=0)
    tn = np.count_nonzero((y_pred - 1) * (y_true - 1), axis=0)
    fp = np.count_nonzero(y_pred * (y_true - 1), axis=0)
    fn = np.count_nonzero((y_pred - 1) * y_true, axis=0)

    return tn, fp, fn, tp

