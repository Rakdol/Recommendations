import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class RecSysMetrics(object):

    # Regression Error Metrics
    def mae(self, true_ratings: List[float], pred_ratings: List[float]) -> float:
        return mean_absolute_error(y_true=true_ratings, y_pred=pred_ratings)

    def mse(self, true_ratings: List[float], pred_ratings: List[float]) -> float:
        return mean_squared_error(y_true=true_ratings, y_pred=pred_ratings)

    def rmse(self, true_ratings: List[float], pred_ratings: List[float]) -> float:
        return np.sqrt(self.mse(true_ratings=true_ratings, pred_ratings=pred_ratings))

    def calc_precision_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(p_at_k)
        return np.mean(scores)

    def calc_recall_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(r_at_k)
        return np.mean(scores)

    # Relevance Metrics
    def _precision_at_k(
        self, true_items: List[float], pred_items: List[float], k: int
    ) -> float:
        if k <= 0:
            return 0.0

        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k

        return p_at_k

    def _recall_at_k(
        self, true_items: List[float], pred_items: List[float], k: int
    ) -> float:
        if k <= 0:
            return 0

        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)

        return r_at_k

    def _f1_at_k(
        self, true_items: List[float], pred_items: List[float], k: int
    ) -> float:

        if k <= 0:
            return 0

        precision = self._precision_at_k(
            true_items=true_items, pred_items=pred_items, k=k
        )
        recall = self._recall_at_k(true_items=true_items, pred_items=pred_items, k=k)

        if precision + recall <= 0.0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    # Ranking Metrics
    def rr_at_k(self, user_relevances: List[int], k: int) -> float:
        nonzero_indices = np.asarray(user_relevances).nonzero()[0]
        if nonzero_indices.size > 0 and nonzero_indices[0] + 1 <= k:
            return 1.0 / (nonzero_indices[0] + 1.0)
        return 0.0

    def mrr_at_k(self, users_relevances: List[List[int]], k: int) -> float:
        return float(
            np.mean(
                [
                    self.rr_at_k(user_relevances, k)
                    for user_relevances in users_relevances
                ]
            )
        )

    def ap_at_k(self, user_relevances: List[int], k: int) -> float:
        if sum(user_relevances[:k]) <= 0:
            return 0.0

        nonzero_indices = np.asarray(user_relevances[:k]).nonzero()[0]

        return sum(
            [sum(user_relevances[: idx + 1]) / (idx + 1) for idx in nonzero_indices]
        ) / sum(user_relevances[:k])

    def map_at_k(self, users_relevances: List[List[int]], k: int) -> float:
        return float(
            np.mean(
                [
                    self.ap_at_k(user_relevances, k)
                    for user_relevances in users_relevances
                ]
            )
        )

    def dcg_at_k(self, user_relevances: List[int], k: int) -> float:
        user_relevances = user_relevances[:k]
        if len(user_relevances) <= 0:
            return 0.0

        return user_relevances[0] + np.sum(
            user_relevances[1:] / np.log2(np.arange(2, len(user_relevances) + 1))
        )

    def nDCG_at_k(self, user_relevances: List[int], k: int) -> float:
        dcg_ideal = self.dcg_at_k(sorted(user_relevances, reverse=True), k)

        if not dcg_ideal:
            return 0.0

        return self.dcg_at_k(user_relevances, k) / dcg_ideal
