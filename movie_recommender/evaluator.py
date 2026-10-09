"""Evaluation module."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(
    model,
    user_mapping: dict,
    item_mapping: dict,
    test_users: np.ndarray,
    full_matrix: np.ndarray,
    scaler=None,
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained MovieRecommender instance.
        user_mapping: User ID to index mapping.
        item_mapping: Item ID to index mapping.
        test_users: Array of test user indices.
        full_matrix: Original rating matrix.
        scaler: Optional scaler.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Get inverse mapping for item indices to IDs
    item_id_to_idx = {v: k for k, v in item_mapping.items()}

    predictions = []
    true_values = []

    for user_idx in test_users:
        # Get original user ID
        for user_id, idx in user_mapping.items():
            if idx == user_idx:
                user_id_orig = user_id
                break

        # Predict ratings for all items rated by this user
        for movie_id, item_idx in item_mapping.items():
            true_rating = full_matrix[user_idx, item_idx]
            if true_rating > 0:  # Only consider rated items
                pred_rating = model.predict_rating(user_id_orig, movie_id, user_mapping, item_mapping)
                predictions.append(pred_rating)
                true_values.append(true_rating)

    if len(predictions) == 0:
        return {"rmse": float('nan'), "mae": float('nan')}

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    metrics = {
        "rmse": np.sqrt(mean_squared_error(true_values, predictions)),
        "mae": mean_absolute_error(true_values, predictions),
    }

    return metrics


def print_evaluation_report(metrics: dict):
    """Print formatted evaluation report."""
    print("=" * 50)
    print("MOVIE RECOMMENDER EVALUATION REPORT")
    print("=" * 50)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print("=" * 50)
