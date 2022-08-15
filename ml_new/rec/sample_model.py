import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])
# timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

# max_timestamp = timestamps.max()
# min_timestamp = timestamps.min()

# timestamp_buckets = np.linspace(
#     min_timestamp, max_timestamp, num=1000,
# )

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))
print(ratings)