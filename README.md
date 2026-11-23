# Movie Review Analyzer

A small Flask web app that looks up a movie on [TMDb](https://www.themoviedb.org/), fetches its user reviews, and classifies each one as positive or negative with a logistic regression model trained on IMDb reviews.

## Features

- Search any movie by title
- Classify the first page of TMDb reviews as positive or negative
- Pie chart and per-review breakdown of the results

## Tech stack

- **Backend:** Flask, requests
- **Model:** scikit-learn (TF-IDF features + logistic regression, tuned with GridSearchCV)
- **Frontend:** HTML, Tailwind CSS, Chart.js

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/lordK2/NLP-Review_Analyzer.git
cd NLP-Review_Analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### TMDb API key

Get a free key from [TMDb](https://www.themoviedb.org/settings/api) and expose it as an environment variable. The key is never stored in the code.

```bash
export TMDB_API_KEY=your_key_here     # Windows (PowerShell): $env:TMDB_API_KEY="your_key_here"
```

## Usage

The repository includes trained model files (`sentiment_model.pkl`, `vectorizer.pkl`), so you can run the app straight away:

```bash
python app.py
```

Then open <http://127.0.0.1:5000> and search for a movie.

### Retraining the model (optional)

1. Download the [IMDb 50K movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.
2. Place `IMDB Dataset.csv` in the project root.
3. Run `python train_model.py`. This overwrites the two `.pkl` files.

## Project structure

```
├── app.py               # Flask app: TMDb lookup and sentiment prediction
├── train_model.py       # Trains and saves the TF-IDF vectorizer and classifier
├── templates/
│   └── index.html       # Single-page frontend
├── sentiment_model.pkl  # Trained classifier (GridSearchCV wrapper)
├── vectorizer.pkl       # Fitted TF-IDF vectorizer
└── requirements.txt
```

## Notes

- Only the first page of reviews (up to 20) returned by TMDb is analyzed.
- The model is a bag-of-words classifier, so it can miss sarcasm and nuance. Treat the counts as a rough signal, not a verdict.
