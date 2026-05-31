"""Data loading for MovieLens 100k."""

from graphrec.data.loader import (
    load_movie_features,
    load_movies,
    load_ratings,
    load_user_features,
)

__all__ = [
    "load_movie_features",
    "load_movies",
    "load_ratings",
    "load_user_features",
]
