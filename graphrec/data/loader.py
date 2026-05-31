"""MovieLens 100k loading utilities.

The raw dataset (~5 MB) is downloaded on first use and cached under ``.data/`` at
the repo root, so reviewers can run the project end-to-end with no manual setup.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".data"


def _dataset_dir(cache_dir: Path | None = None) -> Path:
    """Return the extracted ml-100k directory, downloading it if necessary."""
    cache = cache_dir or DEFAULT_CACHE
    extracted = cache / "ml-100k"
    if extracted.exists():
        return extracted

    cache.mkdir(parents=True, exist_ok=True)
    with urlopen(ML_100K_URL) as resp:  # noqa: S310 - trusted, pinned URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(cache)
    return extracted


def load_ratings(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load interactions as columns: user_id, movie_id, rating, timestamp."""
    base = _dataset_dir(cache_dir)
    return pd.read_csv(
        base / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def load_movies(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load movie metadata. For titles we only need: movie_id, title.

    ``u.item`` is pipe-separated; the first two fields are the id and title.
    """
    base = _dataset_dir(cache_dir)
    return pd.read_csv(
        base / "u.item",
        sep="|",
        usecols=[0, 1],
        names=["movie_id", "title"],
        encoding="latin-1",
    )


# u.item genre flags, in column order. "unknown" is dropped as a feature node.
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]  # fmt: skip


def load_movie_features(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load movie side-features: movie_id, genres (list[str]), year (nullable)."""
    base = _dataset_dir(cache_dir)
    columns = [
        "movie_id", "title", "release_date", "video_release", "imdb_url",
        *GENRE_NAMES,
    ]  # fmt: skip
    raw = pd.read_csv(base / "u.item", sep="|", names=columns, encoding="latin-1")

    flags = raw[GENRE_NAMES].to_numpy(dtype=bool)
    genres = [
        [GENRE_NAMES[i] for i in range(len(GENRE_NAMES)) if i != 0 and row[i]]
        for row in flags
    ]
    year = pd.to_datetime(
        raw["release_date"], format="%d-%b-%Y", errors="coerce"
    ).dt.year

    return pd.DataFrame({"movie_id": raw["movie_id"], "genres": genres, "year": year})


def load_user_features(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load user side-features: user_id, age, occupation."""
    base = _dataset_dir(cache_dir)
    users = pd.read_csv(
        base / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    return users[["user_id", "age", "occupation"]]
