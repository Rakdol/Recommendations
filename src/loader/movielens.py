import os
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
from uszipcode import SearchEngine


@dataclass(frozen=True)
class MovieLensDataset(object):
    train: pd.DataFrame
    test: pd.DataFrame

    test_user2item: Dict[int, List[int]]

    user_content: pd.DataFrame
    item_content: pd.DataFrame
    rating_content: pd.DataFrame


class MovieLensLoader(object):

    def __init__(
        self,
        num_users: int = 500,
        num_test_items: int = 5,
        data_path: str = "../data/ml-100k/",
    ):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path
        self.generes = [
            "unknown",
            "action",
            "adventure",
            "animation",
            "childrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film_noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci_fi",
            "thriller",
            "war",
            "western",
        ]

    def load(self) -> MovieLensDataset:

        ratings, moveis, users, rating_user_movies = self._load_dataset()
        movielens_train, movielens_test = self._split_data(rating_user_movies)
        # Test For ranking
        movielens_test_user2items = (
            movielens_test[movielens_test["rating"] >= 4]
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )

        return MovieLensDataset(
            movielens_train,
            movielens_test,
            movielens_test_user2items,
            moveis,
            users,
            ratings,
        )

    def _load_dataset(self):
        dateparse = lambda x: datetime.utcfromtimestamp(int(x)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        ratings = pd.read_csv(
            self.data_path + "u.data",
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            parse_dates=["timestamp"],
            date_parser=dateparse,
        )

        movies = pd.read_csv(
            self.data_path + "u.item",
            sep="|",
            encoding="latin-1",
            names=[
                "movie_id",
                "movie_title",
                "release_date",
                "video_release_date",
                "imdb_url",
                "unknown",
                "action",
                "adventure",
                "animation",
                "childrens",
                "comedy",
                "crime",
                "documentary",
                "drama",
                "fantasy",
                "film_noir",
                "horror",
                "musical",
                "mystery",
                "romance",
                "sci_fi",
                "thriller",
                "war",
                "western",
            ],
        )

        movies["release_date"] = pd.to_datetime(movies["release_date"])
        movies["release_date"].fillna(
            value=movies["release_date"].mode()[0], inplace=True
        )
        movies.drop(columns=["video_release_date", "imdb_url"], inplace=True)

        users = pd.read_csv(
            self.data_path + "u.user",
            sep="|",
            encoding="latin-1",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )

        users = self._convert_zip_code(users=users)
        users = self._add_age_group(users=users)

        # user 수를 num_users로 줄인다
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]
        users = users[users.user_id <= max(valid_user_ids)]

        raitng_users = pd.merge(ratings, users, on="user_id", how="inner")
        rating_users_movies = pd.merge(raitng_users, movies, on="movie_id", how="inner")

        return ratings, movies, users, rating_users_movies

    def _split_data(self, movielens: pd.DataFrame):

        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(
            ascending=False, method="first"
        )  # Rank Order for watched movies, rank 1 means that latest watched movie.

        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]

        return movielens_train, movielens_test

    def _add_age_group(self, users: pd.DataFrame) -> pd.DataFrame:
        users["age_group"] = users.age.apply(
            lambda age: (
                "Gradeschooler"
                if 5 <= age <= 12
                else (
                    "Teenager"
                    if 13 <= age <= 19
                    else (
                        "Young"
                        if 20 <= age <= 35
                        else ("Midlife" if 35 <= age <= 55 else "Old")
                    )
                )
            )
        )
        return users

    def _convert_zip_code(self, users: pd.DataFrame) -> pd.DataFrame:
        search = SearchEngine()
        city = []
        for z in users["zip_code"]:
            zipcode = search.by_zipcode(z)
            if zipcode:
                city.append(zipcode.major_city)
            else:
                city.append(np.nan)

        users["city"] = city
        users.city.fillna(value="unknown", inplace=True)

        return users
