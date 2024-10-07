from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd


from base import BaseModel

class AverageRatingModel(BaseModel):
    def __init__(self, movielens_train: pd.DataFrame) -> None:
        self.movieid_index = dict(zip(movielens_train.movie_id.unique(), range(len(movielens_train.movie_id.unique()))))
        self.movie_average_rating = movielens_train.groupby("movie_id").agg({"rating": np.mean})
        
    
    def predict(self, movielens_test: pd.DataFrame) -> pd.DataFrame:
        pred_results = []
        
        for i, row in movielens_test.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]
            
            if movie_id not in self.movieid_index:
                pred_results.append(0)
                continue
            
            movie_index = self.movieid_index[movie_id]
            
            pred_rating = self.movie_average_rating.loc[movie_id, "rating"]
            pred_results.append(pred_rating)
            
        return pred_results
    

class PopularRankingModel(BaseModel):
    def __init__(self, movielens_train:pd.DataFrame, minimum_num_ratings:int = 30):
        
        self.unique_user_ids = movielens_train.user_id.unique()
        
        self.movie_rank = movielens_train.groupby(["movie_id", "movie_title"]).agg({"rating": [np.size, np.mean]})
        self.user_evaluated_movies = movielens_train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        minimum_num_indices = self.movie_rank["rating"]["size"] >= 30
        self.movie_sorted_by_rating = movie_rank[minimum_num_indices].sort_values(by=("rating", "mean"), ascending=False)
        
    def predict(self, top_k:int = 10) -> Dict[int, List[int]]:
        pred_user2items = defaultdict(list)      
        for user_id in self.unique_user_ids:
            for movie_id in popluar_movies:
                if movie_id not in self.user_evaluated_movies:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) >= top_k:
                    break
        return pred_user2items