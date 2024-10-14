from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from base import BaseModel

class RandomMovieRatingModel(BaseModel):
    def __init__(self, unique_user_ids: List[int], unique_movie_ids: List[int]) -> None:

        self.userid_index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        self.movieid_index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))
        self.random_pred_matrix = np.random.uniform(
            0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids))
        )

    def predict(self, test_rating: pd.DataFrame) -> List[float]:

        pred_results = []

        for i, row in test_rating.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]
            if movie_id not in self.movieid_index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue

            user_index = self.userid_index[user_id]
            movie_index = self.movieid_index[movie_id]

            pred_rating = self.random_pred_matrix[user_index, movie_index]
            pred_results.append(pred_rating)

        return pred_results


class RandomMovieRankingModel(BaseModel):
    def __init__(self, movielens_train: pd.DataFrame, top_k: int = 10):
        self.top_k = top_k
        self.user_evaluated_movies = (
            movielens_train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )

    def predict(
        self, unique_user_ids: List[int], unique_movie_ids: List[int]
    ) -> Dict[int, List[int]]:
        pred_user2items = defaultdict(list)
        for user_id in unique_user_ids:
            watched_movie_set = set(self.user_evaluated_movies[user_id])
            all_movie_set = set(unique_movie_ids)
            non_watched_movie = list(watched_movie_set ^ all_movie_set)

            random_pred_ranking = np.random.choice(non_watched_movie, self.top_k)

            pred_user2items[user_id].extend(random_pred_ranking)
        return pred_user2items
