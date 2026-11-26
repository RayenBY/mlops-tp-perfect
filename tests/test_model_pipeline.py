"""Unit tests for model_pipeline.py (run with: python -m pytest)."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_pipeline import _diagnosis_group, load_model, predict, prepare_data, save_model

DATA_PATH = "data/diabetic_data.csv"


@pytest.mark.parametrize(
    "code, expected",
    [
        ("Unknown", "Unknown"),
        ("V45", "Other"),
        ("E888", "Other"),
        ("428", "Circulatory"),
        ("786", "Respiratory"),
        ("250.83", "Diabetes"),
        ("820", "Injury"),
        ("174", "Neoplasms"),
    ],
)
def test_diagnosis_group(code, expected):
    assert _diagnosis_group(code) == expected


def test_prepare_data_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("does/not/exist.csv")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset not available")
def test_prepare_data_split(tmp_path):
    # Small sample of the real dataset so the test stays fast
    sample = tmp_path / "sample.csv"
    pd.read_csv(DATA_PATH, nrows=3000).to_csv(sample, index=False)

    data = prepare_data(str(sample))

    assert set(data) == {"X_train", "X_test", "y_train", "y_test", "groups_train"}
    assert len(data["X_train"]) == len(data["y_train"]) == len(data["groups_train"])
    assert len(data["X_test"]) == len(data["y_test"]) > 0
    assert set(data["y_train"].unique()) <= {0, 1}
    for leaked in ("readmitted", "readmis", "patient_nbr", "encounter_id"):
        assert leaked not in data["X_train"].columns
    assert "age_num" in data["X_train"].columns


@pytest.fixture
def tiny_model():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "b": [1, 0, 1, 0, 1, 0]})
    y = np.array([0, 0, 0, 1, 1, 1])
    pipeline = Pipeline([("modele", LogisticRegression())]).fit(X, y)
    return pipeline, ["a", "b"], X


def test_save_and_load_model(tmp_path, tiny_model):
    pipeline, columns, X = tiny_model
    path = tmp_path / "model.pkl"

    save_model(pipeline, columns, str(path))
    loaded, loaded_columns = load_model(str(path))

    assert loaded_columns == columns
    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.pkl"))


def test_predict(tiny_model):
    pipeline, columns, X = tiny_model
    result = predict(pipeline, columns, X)

    assert list(result.columns) == ["prediction", "proba_readmis"]
    assert len(result) == len(X)
    assert result["proba_readmis"].between(0, 1).all()


def test_predict_missing_column(tiny_model):
    pipeline, columns, X = tiny_model
    with pytest.raises(ValueError):
        predict(pipeline, columns, X.drop(columns=["b"]))
