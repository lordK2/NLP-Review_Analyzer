"""Tests for the data-loading step of the training script."""

import pandas as pd

from train_model import load_dataset


def test_load_dataset_maps_labels_and_renames_columns(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {
            "review": ["loved it", "hated it"],
            "sentiment": ["positive", "negative"],
        }
    ).to_csv(csv_path, index=False)

    df = load_dataset(csv_path)

    assert list(df.columns) == ["Text", "sentiment"]
    assert df["Text"].tolist() == ["loved it", "hated it"]
    assert df["sentiment"].tolist() == [1, 0]
