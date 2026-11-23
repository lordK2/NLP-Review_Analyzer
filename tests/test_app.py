"""Tests for the Flask routes. TMDb is mocked, so no network or API key is needed."""

from unittest import mock

import pytest
import requests

import app as app_module


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


SEARCH_PAYLOAD = {"results": [{"id": 603, "title": "The Matrix"}]}
REVIEWS_PAYLOAD = {
    "results": [
        {"content": "An absolutely brilliant, thrilling masterpiece. I loved every minute."},
        {"content": "Boring, predictable and far too long. Terrible acting."},
        {"content": "A wonderful film with great performances."},
        {"content": ""},  # empty reviews must be skipped
    ]
}


def fake_tmdb(url, params=None, timeout=None):
    """Stand-in for requests.get that answers the two TMDb endpoints the app calls."""
    if url.endswith("/search/movie"):
        return FakeResponse(SEARCH_PAYLOAD)
    if url.endswith("/movie/603/reviews"):
        return FakeResponse(REVIEWS_PAYLOAD)
    raise AssertionError(f"Unexpected TMDb URL: {url}")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Movie Review Analyzer" in response.data


def test_analyze_returns_counts_and_labelled_reviews(client):
    with mock.patch.object(app_module.requests, "get", side_effect=fake_tmdb) as get:
        response = client.post("/analyze", json={"movie_title": "  The Matrix  "})

    assert response.status_code == 200
    body = response.get_json()
    assert body["total_reviews"] == 3
    assert body["positive_count"] + body["negative_count"] == 3
    assert len(body["reviews"]) == 3
    assert all(item["sentiment"] in {"Positive", "Negative"} for item in body["reviews"])

    # The title is trimmed, and the key is sent as a query parameter, not in the path.
    search_call = get.call_args_list[0]
    assert search_call.kwargs["params"] == {"query": "The Matrix", "api_key": "test-key"}


def test_analyze_labels_clear_reviews_correctly(client):
    with mock.patch.object(app_module.requests, "get", side_effect=fake_tmdb):
        body = client.post("/analyze", json={"movie_title": "The Matrix"}).get_json()

    labels = {item["text"]: item["sentiment"] for item in body["reviews"]}
    assert labels["An absolutely brilliant, thrilling masterpiece. I loved every minute."] == "Positive"
    assert labels["Boring, predictable and far too long. Terrible acting."] == "Negative"


def test_analyze_requires_a_title(client):
    response = client.post("/analyze", json={"movie_title": "   "})
    assert response.status_code == 400
    assert "required" in response.get_json()["error"]


def test_analyze_rejects_overlong_title(client):
    response = client.post("/analyze", json={"movie_title": "x" * (app_module.MAX_TITLE_LENGTH + 1)})
    assert response.status_code == 400


def test_analyze_handles_unknown_movie(client):
    empty = FakeResponse({"results": []})
    with mock.patch.object(app_module.requests, "get", return_value=empty):
        response = client.post("/analyze", json={"movie_title": "Not A Real Movie"})
    assert response.status_code == 404


def test_analyze_handles_movie_without_reviews(client):
    def no_reviews(url, params=None, timeout=None):
        if url.endswith("/search/movie"):
            return FakeResponse(SEARCH_PAYLOAD)
        return FakeResponse({"results": []})

    with mock.patch.object(app_module.requests, "get", side_effect=no_reviews):
        response = client.post("/analyze", json={"movie_title": "The Matrix"})
    assert response.status_code == 404
    assert "reviews" in response.get_json()["error"]


def test_analyze_reports_tmdb_outage_as_bad_gateway(client):
    with mock.patch.object(app_module.requests, "get", side_effect=requests.ConnectionError("down")):
        response = client.post("/analyze", json={"movie_title": "The Matrix"})
    assert response.status_code == 502


def test_analyze_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    response = client.post("/analyze", json={"movie_title": "The Matrix"})
    assert response.status_code == 500
    assert "TMDB_API_KEY" in response.get_json()["error"]


def test_analyze_reports_missing_model(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    response = client.post("/analyze", json={"movie_title": "The Matrix"})
    assert response.status_code == 500
    assert "train_model.py" in response.get_json()["error"]
