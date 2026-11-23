"""Movie Review Analyzer web app.

Looks up a movie on TMDb, fetches its user reviews, and classifies each review
as positive or negative with the model produced by ``train_model.py``.

Set the TMDb API key before starting the server:
    export TMDB_API_KEY=your_key_here
"""

import logging
import os
from pathlib import Path

import joblib
import requests
from flask import Flask, jsonify, render_template, request

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_TIMEOUT_SECONDS = 10
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


class TMDbError(Exception):
    """Raised when the TMDb API cannot be reached or returns an error."""


def load_model():
    """Load the trained classifier and vectorizer, or return (None, None) on failure."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception:
        logger.exception("Could not load model files. Run train_model.py first.")
        return None, None
    logger.info("Loaded sentiment model and vectorizer.")
    return model, vectorizer


model, vectorizer = load_model()


def tmdb_get(path: str, **params) -> dict:
    """GET a TMDb endpoint and return the decoded JSON body."""
    params["api_key"] = TMDB_API_KEY
    try:
        response = requests.get(f"{TMDB_BASE_URL}{path}", params=params, timeout=TMDB_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise TMDbError(f"TMDb request to {path} failed: {exc}") from exc


def find_movie_id(title: str) -> int | None:
    """Return the TMDb ID of the best match for ``title``, or None if nothing matches."""
    results = tmdb_get("/search/movie", query=title).get("results") or []
    return results[0]["id"] if results else None


def fetch_reviews(movie_id: int) -> list[str]:
    """Return the text of the first page of reviews for a movie."""
    results = tmdb_get(f"/movie/{movie_id}/reviews").get("results") or []
    return [review["content"] for review in results if review.get("content")]


def error_response(message: str, status: int):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    if model is None or vectorizer is None:
        return error_response("Model files are missing. Run train_model.py first.", 500)
    if not TMDB_API_KEY:
        return error_response("The TMDB_API_KEY environment variable is not set.", 500)

    payload = request.get_json(silent=True) or {}
    title = (payload.get("movie_title") or "").strip()
    if not title:
        return error_response("Movie title is required.", 400)

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

    predictions = model.predict(vectorizer.transform(reviews))
    labels = ["Positive" if int(p) == 1 else "Negative" for p in predictions]
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
