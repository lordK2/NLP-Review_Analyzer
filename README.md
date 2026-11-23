# Movie Review Analyzer

Type a movie title, and the app pulls that movie's reviews from [TMDb](https://www.themoviedb.org/) and labels each one as **positive** or **negative** using a machine-learning model trained on IMDb reviews. You get a pie chart of the overall sentiment and the reviews themselves, each tagged with its label.

It is built as a small, complete example of an NLP pipeline: data preparation, model training, a web API, and a browser front end.

## How it works

```
 Training (run once)                          Serving (every request)

 IMDb reviews (CSV)                           Browser: "Inception"
        │                                            │  POST /analyze
        ▼                                            ▼
 train_model.py                               app.py (Flask)
   • TF-IDF vectorizer ─┐                       1. TMDb: search movie → ID
   • Logistic regression│ saved to             2. TMDb: fetch reviews
     (grid-searched)   ─┘ models/              3. vectorizer.transform(reviews)
                                                  model.predict(...)
                                               4. JSON: counts + labelled reviews
                                                        │
                                                        ▼
                                               Browser: pie chart + review list
```

1. **Vectorize:** each review becomes a TF-IDF vector: a weighted count of its words, with common English stop words removed.
2. **Classify:** a logistic regression model predicts positive (1) or negative (0). Its regularization strength (`C`) is chosen by 5-fold cross-validation.
3. **Serve:** the Flask app loads the saved model once at startup and applies it to live TMDb reviews.

## Quick start

Requires **Python 3.10 or newer**.

```bash
# 1. Get the code and create a virtual environment
git clone https://github.com/lordK2/NLP-Review_Analyzer.git
cd NLP-Review_Analyzer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your TMDb API key (see below)
cp .env.example .env               # Windows: copy .env.example .env

# 4. Run the app
python app.py
```

Open <http://127.0.0.1:5000>, enter a movie title, and click **Analyze Reviews**.

The trained model files are included in `models/`, so you do not need to retrain to run the app.

### Getting a TMDb API key

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/signup).
2. Request an API key from your account's **Settings → API** page.
3. Paste the **API Key** value (not the read-access token) into `.env`:

   ```ini
   TMDB_API_KEY=your_key_here
   ```

The `.env` file is listed in `.gitignore`, so Git will not track it. **Never put your key in a source file or commit it.** If a key is ever exposed, revoke it on the TMDb settings page and create a new one.

## Retraining the model (optional)

1. Download the [IMDb 50K movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle. It has 50,000 labelled reviews.
2. Save the file as `data/IMDB_Dataset.csv`. The CSV must have `review` and `sentiment` columns.
3. Run:

   ```bash
   python train_model.py
   ```

The script prints the best parameters, the cross-validated accuracy, and the accuracy on a held-out 20% test split. It then overwrites the files in `models/`. Restart `app.py` afterward to load the new model.

## API

The front end talks to one JSON endpoint, which you can also call directly.

### `POST /analyze`

Request:

```json
{ "movie_title": "The Matrix" }
```

Success (`200`):

```json
{
  "positive_count": 2,
  "negative_count": 1,
  "total_reviews": 3,
  "reviews": [
    { "text": "A wonderful film with great performances.", "sentiment": "Positive" },
    { "text": "Boring, predictable and far too long.", "sentiment": "Negative" }
  ]
}
```

Errors return `{ "error": "message" }` with one of these status codes:

| Status | Meaning |
|--------|---------|
| `400` | Title is missing, blank, or longer than 200 characters |
| `404` | No movie matches the title, or the movie has no reviews on TMDb |
| `500` | The model files or the `TMDB_API_KEY` setting are missing |
| `502` | TMDb could not be reached or returned an error |

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The same tests run automatically on GitHub Actions for every push and pull request.

The tests cover the HTTP routes and their error handling, with TMDb mocked so they run offline without an API key. They also check the label-mapping step of the training script.

## Project structure

```
NLP-Review_Analyzer/
├── app.py               # Flask web server and the /analyze endpoint
├── train_model.py       # Trains the TF-IDF vectorizer and classifier
├── templates/
│   └── index.html       # Single-page front end (Tailwind CSS + Chart.js)
├── models/
│   ├── sentiment_model.pkl   # Trained classifier (GridSearchCV wrapper)
│   └── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── data/                # Put IMDB_Dataset.csv here for retraining (git-ignored)
├── tests/
│   ├── test_app.py      # Route and error-handling tests
│   └── test_train.py    # Data-loading tests
├── .github/workflows/   # CI: runs the test suite on every push
├── .env.example        # Template for the local .env file
├── LICENSE              # MIT License
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Test dependencies
└── pytest.ini
```

## Limitations

- Only the first page of TMDb reviews (about 20) is analyzed.
- The model is a bag-of-words classifier. It does not understand context, so it can miss sarcasm, negation, and mixed opinions. Read the counts as a rough signal.
- The model was trained on IMDb reviews of films. It may be less accurate for other kinds of text.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "TMDB_API_KEY is not set" | Create `.env` from `.env.example` and add your key, then restart the server. |
| "Model files are missing" | Make sure `models/` contains both `.pkl` files, or run `train_model.py`. |
| "Could not reach TMDb" | Check your internet connection and that your key is valid. |
| Model fails to load after an upgrade | The `.pkl` files depend on the scikit-learn version. Install the versions in `requirements.txt`, or retrain. |

## License

This project is released under the [MIT License](LICENSE).
