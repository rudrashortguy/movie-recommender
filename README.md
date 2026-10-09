# Movie Recommender

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A collaborative filtering recommendation system using matrix factorization with scikit-learn.

## Features

- Movie rating matrix construction from user data
- Matrix factorization with SVD
- User-based and item-based recommendations
- Model evaluation with RMSE and MAE
- Command-line interface for training and prediction
- Comprehensive test suite
- GitHub Actions CI

## Installation

```bash
git clone https://github.com/rudrashortguy/movie-recommender.git
cd movie-recommender
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Training a model

```bash
movie-recommender train --input data/movie_ratings.csv --output models/movie_recommender.pkl
```

### Making recommendations

```bash
movie-recommender recommend --model models/movie_recommender.pkl --user-id 123 --num-recommendations 10
```

### Python API

```python
from movie_recommender.data import load_movie_data
from movie_recommender.model import MovieRecommender
from movie_recommender.evaluator import evaluate

# Load data
data = load_movie_data()

# Initialize recommender
recommender = MovieRecommender(n_components=50)

# Train
recommender.fit(data)

# Get recommendations
recommendations = recommender.recommend(user_id=123, n_recommendations=10)
print(f"Top movie: {recommendations[0]}")
```

## Project Structure

```
movie-recommender/
├── data/                  # Data storage (not tracked)
├── movie_recommender/     # Source code
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing
│   ├── model.py           # Recommendation model
│   └── evaluator.py       # Evaluation metrics
├── tests/                 # Unit tests
├── models/                # Trained models (not tracked)
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
└── .github/workflows/tests.yml
```

## Tech Stack

- **Language**: Python 3.8+
- **Matrix Factorization**: scikit-learn (SVD)
- **Data Processing**: pandas, numpy
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Roadmap

- [x] User-item matrix construction
- [x] Matrix factorization (SVD)
- [x] Recommendation generation
- [ ] Add deep learning models (Neural Collaborative Filtering)
- [ ] Support for implicit feedback
- [ ] Real-time recommendation API
- [ ] Integration with TMDB for movie metadata

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.