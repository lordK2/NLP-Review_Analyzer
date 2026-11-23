"""Movie Review Analyzer: a Flask web app for audience sentiment.

For a movie title entered in the browser, the app:
  1. looks up the movie on TMDb,
  2. fetches its user reviews from TMDb,
  3. classifies each review as positive or negative with the trained model
     from ``train_model.py``, and
  4. returns the counts and per-review labels as JSON.

Configuration lives in a ``.env`` file in the project root (never committed):

    TMDB_API_KEY=your_key_here

See ``.env.example`` for the template.
"""

import logging
import os
from pathlib import Path

import joblib
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_TIMEOUT_SECONDS = 10
MAX_TITLE_LENGTH = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


class TMDbError(Exception):
    """Raised when the TMDb API cannot be reached or returns an unusable response."""


def load_model():
    """Return the trained (classifier, vectorizer) pair, or (None, None) if the files are missing."""
    try:
        classifier = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception:
        logger.exception("Could not load model files from %s. Run train_model.py first.", MODEL_PATH.parent)
        return None, None
    logger.info("Loaded sentiment model and vectorizer.")
    return classifier, vectorizer


model, vectorizer = load_model()


def get_api_key() -> str:
    """Read the TMDb key at request time so it is always taken from the current environment."""
    return os.environ.get("TMDB_API_KEY", "").strip()


def tmdb_get(path: str, **params) -> dict:
    """Call a TMDb v3 endpoint and return the decoded JSON body."""
    params["api_key"] = get_api_key()
    try:
        response = requests.get(f"{TMDB_BASE_URL}{path}", params=params, timeout=TMDB_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise TMDbError(f"TMDb request to {path} failed: {exc}") from exc


def find_movie_id(title: str) -> int | None:
    """Return the TMDb ID of the top search result for ``title``, or None if there are no matches."""
    results = tmdb_get("/search/movie", query=title).get("results") or []
    return results[0]["id"] if results else None


def fetch_reviews(movie_id: int) -> list[str]:
    """Return the text of each review on the first page of results for a movie."""
    results = tmdb_get(f"/movie/{movie_id}/reviews").get("results") or []
    return [review["content"] for review in results if review.get("content")]


def classify(reviews: list[str]) -> list[str]:
    """Label each review "Positive" or "Negative" using the trained model."""
    predictions = model.predict(vectorizer.transform(reviews))
    return ["Positive" if int(p) == 1 else "Negative" for p in predictions]


def error_response(message: str, status: int):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    """Analyze reviews for a movie. Request body: {"movie_title": "..."}."""
    if model is None or vectorizer is None:
        return error_response("Model files are missing. Run train_model.py first.", 500)
    if not get_api_key():
        return error_response("TMDB_API_KEY is not set. Add it to the .env file.", 500)

    payload = request.get_json(silent=True) or {}
    title = str(payload.get("movie_title") or "").strip()
    if not title:
        return error_response("Movie title is required.", 400)
    if len(title) > MAX_TITLE_LENGTH:
        return error_response(f"Movie title must be {MAX_TITLE_LENGTH} characters or fewer.", 400)

    try:
        movie_id = find_movie_id(title)
        if movie_id is None:
            return error_response(f"Could not find a movie titled '{title}'.", 404)
        reviews = fetch_reviews(movie_id)
    except TMDbError:
        logger.exception("TMDb lookup failed for '%s'", title)
        return error_response("Could not reach TMDb. Please try again later.", 502)

    if not reviews:
        return error_response("This movie does not have any reviews on TMDb.", 404)

    labels = classify(reviews)
    positive_count = labels.count("Positive")
    return jsonify(
        {
            "positive_count": positive_count,
            "negative_count": len(labels) - positive_count,
            "total_reviews": len(labels),
            "reviews": [{"text": text, "sentiment": label} for text, label in zip(reviews, labels)],
        }
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
