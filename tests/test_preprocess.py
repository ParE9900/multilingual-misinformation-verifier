"""Tests for data preprocessing and unified dataset verification."""

from pathlib import Path
import pandas as pd
import pytest
from src.nli_classifier import NLIClassifier


def test_master_dataset_file_exists():
    """Verify that data/processed/master_dataset.csv exists."""
    dataset_path = Path("data/processed/master_dataset.csv")
    assert dataset_path.exists(), f"Expected dataset at {dataset_path} does not exist"


def test_master_dataset_schema_and_contents():
    """Verify schema, column names, lack of NaNs, and label values in master_dataset.csv."""
    dataset_path = Path("data/processed/master_dataset.csv")
    assert dataset_path.exists()

    df = pd.read_csv(dataset_path)

    # Assert exact columns
    assert list(df.columns) == ["text", "label", "language"]

    # Assert no NaNs
    assert df.isna().sum().sum() == 0

    # Assert not empty
    assert len(df) > 0

    # Assert label values are valid binary classes {0, 1}
    assert set(df["label"].unique()).issubset({0, 1})

    # Assert texts are non-empty strings
    assert (df["text"].str.strip().str.len() > 0).all()

    # Assert languages are non-empty strings
    assert (df["language"].str.strip().str.len() > 0).all()


def test_nli_classifier_skeleton():
    """Verify NLIClassifier initialization, language detection, and stance prediction skeleton."""
    classifier = NLIClassifier(model_name="xlm-roberta-base", device="cpu", max_length=256)
    assert classifier.model_name == "xlm-roberta-base"
    assert classifier.device == "cpu"
    assert classifier.max_length == 256
    assert classifier.model is None
    assert classifier.tokenizer is None

    # Test language detection
    assert classifier.detect_language("This is a simple English sentence.") == "en"
    assert classifier.detect_language("यह एक हिंदी समाचार लेख है।") == "hi"

    # Test stance prediction skeleton
    res = classifier.predict_stance("The earth is round.", "Scientific evidence shows earth is spherical.")
    assert isinstance(res, dict)
    assert "supports" in res
    assert "refutes" in res
    assert res["supports"] == 0.5
    assert res["refutes"] == 0.5
