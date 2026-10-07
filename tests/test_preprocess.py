"""Tests for dataset preprocessing and NLI skeleton."""

from pathlib import Path
import pandas as pd
from src.nli_classifier import NLIClassifier


def test_master_dataset_file_exists():
    dataset_path = Path("data/processed/master_dataset.csv")
    assert dataset_path.exists()


def test_master_dataset_schema_and_contents():
    dataset_path = Path("data/processed/master_dataset.csv")
    df = pd.read_csv(dataset_path)

    assert list(df.columns) == ["text", "label", "language"]
    assert df.isna().sum().sum() == 0
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})
    assert (df["text"].str.strip().str.len() > 0).all()
    assert (df["language"].str.strip().str.len() > 0).all()


def test_nli_classifier_skeleton():
    classifier = NLIClassifier(model_name="xlm-roberta-base", device="cpu", max_length=256)
    assert classifier.model_name == "xlm-roberta-base"
    assert classifier.device == "cpu"
    assert classifier.max_length == 256
    assert classifier.model is None
    assert classifier.tokenizer is None

    assert classifier.detect_language("This is a simple English sentence.") == "en"
    assert classifier.detect_language("यह एक हिंदी समाचार लेख है।") == "hi"

    res = classifier.predict_stance("The earth is round.", "Scientific evidence shows earth is spherical.")
    assert isinstance(res, dict)
    assert res.get("supports") == 0.5
    assert res.get("refutes") == 0.5
