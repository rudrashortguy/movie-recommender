"""Data loading module."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_movie_data(
    filepath: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load movie rating dataset.

    Args:
        filepath: Path to CSV file with user_id, movie_id, rating columns.
        test_size: Proportion for testing.
        random_state: Random seed.

    Returns:
        Dictionary with 'matrix', 'user_mapping', 'item_mapping' and split data.
    """
    if filepath is None:
        filepath = _get_sample_data_path()

    df = pd.read_csv(filepath)

    if 'user_id' not in df.columns or 'movie_id' not in df.columns or 'rating' not in df.columns:
        raise ValueError("CSV must contain 'user_id', 'movie_id', and 'rating' columns")

    # Create user and item mappings
    users = df['user_id'].unique()
    items = df['movie_id'].unique()
    user_mapping = {user: idx for idx, user in enumerate(users)}
    item_mapping = {item: idx for idx, item in enumerate(items)}

    # Create user-item rating matrix
    n_users = len(users)
    n_items = len(items)
    matrix = np.zeros((n_users, n_items))

    for _, row in df.iterrows():
        user_idx = user_mapping[row['user_id']]
        item_idx = item_mapping[row['movie_id']]
        matrix[user_idx, item_idx] = row['rating']

    # Create a simple train/test split
    user_indices = np.arange(n_users)
    train_users, test_users = train_test_split(
        user_indices, test_size=test_size, random_state=random_state
    )

    return {
        'matrix': matrix,
        'user_mapping': user_mapping,
        'item_mapping': item_mapping,
        'train_users': train_users,
        'test_users': test_users,
        'full_data': df,
    }


def _get_sample_data_path() -> str:
    """Return path to bundled sample data."""
    import tempfile
    sample_data = """user_id,movie_id,rating
1,101,5
1,102,3
1,103,4
2,101,2
2,102,4
2,103,5
2,104,3
3,101,5
3,102,3
3,104,4
3,105,2
4,101,4
4,103,5
4,105,3
5,102,3
5,103,4
5,104,5
5,105,4
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(sample_data)
        return f.name
