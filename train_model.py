"""Train the movie review sentiment model.

Reads the IMDb dataset (``IMDB Dataset.csv`` with ``review`` and ``sentiment``
columns), fits a TF-IDF vectorizer and a logistic regression classifier, and
saves both to disk so the Flask app can load them.

Usage:
    python train_model.py
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "IMDB Dataset.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10_000
CV_FOLDS = 5
PARAM_GRID = {"C": [0.1, 1, 10], "solver": ["saga"]}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Return a DataFrame with ``Text`` and numeric ``sentiment`` (1 = positive)."""
    raw = pd.read_csv(path)
    return pd.DataFrame(
        {
            "Text": raw["review"],
            "sentiment": (raw["sentiment"] == "positive").astype(int),
        }
    )


def main() -> int:
    if not DATASET_PATH.exists():
        logger.error("Could not find %s. Download the IMDb dataset and place it here.", DATASET_PATH.name)
        return 1

    df = load_dataset(DATASET_PATH)
    logger.info("Loaded %d reviews.", len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"], df["sentiment"], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train)

    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    logger.info("Running grid search over %s ...", PARAM_GRID)
    search.fit(X_train_vec, y_train)
    logger.info("Best parameters: %s (CV accuracy %.3f)", search.best_params_, search.best_score_)

    test_accuracy = search.score(vectorizer.transform(X_test), y_test)
    logger.info("Held-out test accuracy: %.3f", test_accuracy)

    joblib.dump(search, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Saved %s and %s.", MODEL_PATH.name, VECTORIZER_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
