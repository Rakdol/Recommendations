import os
from datetime import datetime

import pandas as pd
import numpy as np
from uszipcode import SearchEngine


class MovieLensLoader(object):

    def __init__(
        self,
        num_users: int = 1000,
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

        return ratings, movies, users

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

        return users
