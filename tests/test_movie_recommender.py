"""Tests for movie recommender."""

import pytest
import numpy as np
import pandas as pd
from movie_recommender.data import load_movie_data
from movie_recommender.model import MovieRecommender
from movie_recommender.evaluator import evaluate, print_evaluation_report


class TestDataLoading:
    def test_load_movie_data_returns_dict(self):
        """Test that load_movie_data returns a dictionary with required keys."""
        data = load_movie_data()
        assert isinstance(data, dict)
        assert 'matrix' in data
        assert 'user_mapping' in data
        assert 'item_mapping' in data
        assert 'train_users' in data
        assert 'test_users' in data

    def test_load_movie_data_matrix_shape(self):
        """Test that matrix has expected shape."""
        data = load_movie_data()
        expected_rows = len(data['user_mapping'])
        expected_cols = len(data['item_mapping'])
        assert data['matrix'].shape == (expected_rows, expected_cols)

    def test_load_movie_data_missing_column_raises(self):
        """Test that missing column raises ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("user_id,movie_id\n1,101\n")
            path = f.name
        with pytest.raises(ValueError, match="rating"):
            load_movie_data(filepath=path)
        import os
        os.unlink(path)


class TestModel:
    def test_recommender_init(self):
        """Test model initialization."""
        recommender = MovieRecommender(n_components=10)
        assert recommender.n_components == 10
        assert recommender.model is None
        assert recommender.user_factors is None

    def test_fit_model(self):
        """Test model fitting."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        assert recommender.user_factors is not None
        assert recommender.item_factors is not None
        assert recommender.user_factors.shape[1] == 5

    def test_recommend_returns_list(self):
        """Test that recommendations are returned as a list."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        recommendations = recommender.recommend(
            user_id=1,
            user_mapping=data['user_mapping'],
            item_mapping=data['item_mapping'],
            n_recommendations=5
        )
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert len(recommendations) <= 5

    def test_recommend_invalid_user_raises(self):
        """Test that invalid user raises ValueError."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        with pytest.raises(ValueError, match="not found"):
            recommender.recommend(
                user_id=999,
                user_mapping=data['user_mapping'],
                item_mapping=data['item_mapping']
            )

    def test_predict_rating(self):
        """Test rating prediction."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        rating = recommender.predict_rating(
            user_id=1,
            movie_id=101,
            user_mapping=data['user_mapping'],
            item_mapping=data['item_mapping']
        )
        assert isinstance(rating, (int, float))

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        import tempfile
        import os
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            recommender.save_model(path)
            loaded = MovieRecommender.load_model(path)
            assert loaded.user_factors is not None
            assert loaded.item_factors is not None
        finally:
            os.unlink(path)


class TestEvaluator:
    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns expected keys."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        metrics = evaluate(
            recommender,
            data['user_mapping'],
            data['item_mapping'],
            data['test_users'],
            data['matrix']
        )
        assert 'rmse' in metrics
        assert 'mae' in metrics

    def test_evaluate_metrics_non_negative(self):
        """Test that metrics are non-negative."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        metrics = evaluate(
            recommender,
            data['user_mapping'],
            data['item_mapping'],
            data['test_users'],
            data['matrix']
        )
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0

    def test_print_evaluation_report(self, capsys):
        """Test report printing."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        metrics = evaluate(
            recommender,
            data['user_mapping'],
            data['item_mapping'],
            data['test_users'],
            data['matrix']
        )
        print_evaluation_report(metrics)
        captured = capsys.readouterr()
        assert "MOVIE RECOMMENDER" in captured.out


class TestIntegration:
    def test_end_to_end_pipeline(self):
        """Full pipeline test."""
        data = load_movie_data()
        recommender = MovieRecommender(n_components=5)
        recommender.fit(data)
        recommendations = recommender.recommend(
            user_id=1,
            user_mapping=data['user_mapping'],
            item_mapping=data['item_mapping'],
            n_recommendations=3
        )
        assert len(recommendations) == 3
        assert recommendations[0][0] in data['item_mapping']
