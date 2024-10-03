import sys
from pathlib import Path


package_root = str(Path(__file__).resolve().parents[1])
sys.path.append(package_root)

print(package_root)
import pytest
from src.utils.metrics import RecSysMetrics


pytest.error_threshold = 0.00001
pytest.true_ratings = [0.0, 1.0, 2.0, 3.0, 4.0]
pytest.pred_ratings = [0.1, 1.1, 2.1, 3.1, 4.1]


def test_mae():
    assert (
        pytest.approx(
            RecSysMetrics().mae(pytest.true_ratings, pytest.pred_ratings),
            pytest.error_threshold,
        )
        == 0.1
    )


def test_mse():
    assert (
        pytest.approx(
            RecSysMetrics().mse(pytest.true_ratings, pytest.pred_ratings),
            pytest.error_threshold,
        )
        == 0.01
    )


def test_rmse():
    assert (
        pytest.approx(
            RecSysMetrics().rmse(pytest.true_ratings, pytest.pred_ratings),
            pytest.error_threshold,
        )
        == 0.1
    )


pytest.pred_item = [1, 2, 3, 4, 5]
pytest.true_item = [2, 4, 6, 8]


def test_precision_at_k():
    assert RecSysMetrics().precision_at_k(pytest.true_item, pytest.pred_item, 5) == 0.4


def test_recall_at_k():
    assert RecSysMetrics().recall_at_k(pytest.true_item, pytest.pred_item, 5) == 0.5


def test_f1_at_k():
    assert (
        pytest.approx(
            RecSysMetrics().f1_at_k(pytest.true_item, pytest.pred_item, 5),
            pytest.error_threshold,
        )
        == 0.4444444
    )


pytest.user1_relevance = [1, 0, 0]
pytest.user2_relevance = [0, 1, 0]
pytest.user3_relevance = [1, 0, 1]
pytest.users_relevance = [
    pytest.user1_relevance,
    pytest.user2_relevance,
    pytest.user3_relevance,
]


def test_rr_at_k():
    assert RecSysMetrics().rr_at_k(pytest.user1_relevance, 3) == 1.0 / 1
    assert RecSysMetrics().rr_at_k(pytest.user2_relevance, 3) == 1.0 / 2
    assert RecSysMetrics().rr_at_k(pytest.user3_relevance, 3) == 1.0 / 1


def test_mrr_at_k():
    assert (
        RecSysMetrics().mrr_at_k(pytest.users_relevance, 3)
        == (1.0 / 1 + 1.0 / 2 + 1.0 / 1) / 3
    )


def test_ap_at_k():

    assert RecSysMetrics().ap_at_k(pytest.user1_relevance, 3) == 1.0 / 1
    assert RecSysMetrics().ap_at_k(pytest.user2_relevance, 3) == 1.0 / 2
    assert RecSysMetrics().ap_at_k(pytest.user3_relevance, 3) == (1.0 / 1 + 2.0 / 3) / 2


def test_map_at_k():
    assert (
        RecSysMetrics().map_at_k(pytest.users_relevance, 3)
        == (1.0 / 1 + 1.0 / 2 + (1.0 / 1 + 2.0 / 3) / 2) / 3
    )


pytest.user4_weighted_relevance = [0, 2, 0, 1, 0]


def test_dcg_at_k():
    assert RecSysMetrics().dcg_at_k(pytest.user4_weighted_relevance, 5) == 2.5


def test_ndcg_at_k():
    assert RecSysMetrics().nDCG_at_k(pytest.user4_weighted_relevance, 5) == 2.5 / 3.0
