try:
    from metriks.__version import __version__
except ImportError:  # pragma: no cover
    __version__ = "dev"

from metriks.ranking import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    label_mean_reciprocal_rank,
    ndcg,
    confusion_matrix_at_k,
)
