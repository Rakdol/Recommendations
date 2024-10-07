from typing import List, Dict

import numpy as np
import pandas as pd

from collections import defaultdict, Counter

from base import BaseModel


class AssociationModel(BaseModel):
    def __init__(self, movielens_train: pd.DataFrame, **kwargs) -> None:
        self.min_support = kwargs.get("min_support", 0.1)
        self.min_threshold = kwargs.get("min_threshold", 0.1)
        
        self.user_evaluated_movies = None
        self.movielens_train_high_rating = None
        self.user_movie_matrix = None
        self._initialize(movielens_train)
        
    def _initialize(self, movielens_train:pd.DataFrame):
        self.user_evaluated_movies = movielens_train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        self.movielens_train_high_rating = movielens_train[movielens_train.rating >= 4]
        self.user_movie_matrix = movielens_train.pivot(index="user_id", columns = "movie_id", values="rating")
        self.user_movie_matrix[self.user_movie_matrix < 4] = 0
        self.user_movie_matrix[self.user_movie_matrix.isnull()] = 0
        self.user_movie_matrix[self.user_movie_matrix >= 4] = 1
        
    def _get_association_rules(self) -> pd.DataFrame:
        freq_movies = apriori(self.user_movie_matrix, min_support=self.min_support, use_colnames=True)
        rules = association_rules(freq_movies, metric="lift", min_threshold=self.min_threshold)
        return rules
        
    def predict(self, top_k:int = 10):
        pred_user2items = defaultdict(list)
        rules = self._get_association_rules()
        for user_id, data in self.movielens_train_high_rating.groupby("user_id"):
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1
            
            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])
            
            counter = Counter(consequent_movies)
            
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in self.user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                
                if len(pred_user2items[user_id]) == 10:
                    break
                
        return pred_user2items
