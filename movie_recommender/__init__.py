"""Movie Recommender - Collaborative filtering with matrix factorization."""

from .data import load_movie_data
from .model import MovieRecommender
from .evaluator import evaluate

__version__ = "1.0.0"
__all__ = [
    "load_movie_data",
    "MovieRecommender",
    "evaluate",
]
