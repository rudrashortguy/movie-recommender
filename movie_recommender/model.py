"""Model implementation."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class MovieRecommender:
    """Movie recommendation system using matrix factorization."""

    def __init__(self, n_components: int = 50, random_state: int = 42):
        """
        Initialize the recommender.

        Args:
            n_components: Number of latent factors.
            random_state: Random seed.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.user_factors = None
        self.item_factors = None

    def fit(self, data: dict):
        """
        Train the recommendation model.

        Args:
            data: Dictionary with 'matrix' and other keys from load_movie_data.
        """
        # Scale the rating matrix
        self.scaled_matrix = self.scaler.fit_transform(data['matrix'])

        # Apply Truncated SVD for matrix factorization
        self.model = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
            algorithm='arpack'
        )
        self.user_factors = self.model.fit_transform(self.scaled_matrix)
        self.item_factors = self.model.components_

    def recommend(self, user_id: int, user_mapping: dict, item_mapping: dict, n_recommendations: int = 10) -> list:
        """
        Get movie recommendations for a user.

        Args:
            user_id: User ID.
            user_mapping: Mapping from user IDs to indices.
            item_mapping: Mapping from item IDs to indices.
            n_recommendations: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_rating) tuples.
        """
        if user_id not in user_mapping:
            raise ValueError(f"User ID {user_id} not found in user mapping")

        user_idx = user_mapping[user_id]
        user_vec = self.user_factors[user_idx]

        # Predict ratings for all movies
        predicted_ratings = np.dot(user_vec, self.item_factors)

        # Create list of (item_idx, predicted_rating)
        item_indices = np.argsort(predicted_ratings)[::-1]
        recommendations = []

        for item_idx in item_indices[:n_recommendations]:
            # Find the original movie ID from the item index
            for movie_id, idx in item_mapping.items():
                if idx == item_idx:
                    recommendations.append((movie_id, predicted_ratings[item_idx]))
                    break

        return recommendations

    def predict_rating(self, user_id: int, movie_id: int, user_mapping: dict, item_mapping: dict) -> float:
        """
        Predict rating for a specific user-movie pair.

        Args:
            user_id: User ID.
            movie_id: Movie ID.
            user_mapping: User ID to index mapping.
            item_mapping: Item ID to index mapping.

        Returns:
            Predicted rating.
        """
        if user_id not in user_mapping or movie_id not in item_mapping:
            raise ValueError("User or movie not found in mappings")

        user_idx = user_mapping[user_id]
        item_idx = item_mapping[movie_id]

        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[item_idx]

        return np.dot(user_vec, item_vec)

    def save_model(self, filepath: str):
        """Save model to disk."""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'n_components': self.n_components,
            'random_state': self.random_state,
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from disk."""
        import joblib
        data = joblib.load(filepath)
        recommender = cls(n_components=data['n_components'], random_state=data['random_state'])
        recommender.model = data['model']
        recommender.scaler = data['scaler']
        recommender.user_factors = data['user_factors']
        recommender.item_factors = data['item_factors']
        return recommender
