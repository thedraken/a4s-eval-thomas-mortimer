import numpy as np
import pytest
import datetime
import pandas as pd
import uuid

from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import Dataset, Feature, FeatureType, DataShape
from a4s_eval.metrics.prediction_metrics.nlp_llm_robustness_metrics import llm_answer_consistency, llm_performance_drop

# ---------------------------
# Fixtures
# ---------------------------

# Minimal DataShape
shape = DataShape(features=[
    Feature(pid=uuid.uuid4(), name="text_original", feature_type=FeatureType.TEXT, min_value=None, max_value=None),
    Feature(pid=uuid.uuid4(), name="text_transformed", feature_type=FeatureType.TEXT, min_value=None, max_value=None),
    Feature(pid=uuid.uuid4(), name="y", feature_type=FeatureType.CATEGORICAL, min_value=None, max_value=None)
])

@pytest.fixture
def dataset_basic():
    df = pd.DataFrame({
        "text_original": ["a", "b"],
        "text_transformed": ["a2", "b2"],
        "y": [0, 1]
    })
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)

@pytest.fixture
def dataset_orthogonal_preds():
    df = pd.DataFrame({
        "text_original": ["a"],
        "text_transformed": ["a2"],
        "y": [0]
    })
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)


# ---------------------------
# llm_answer_consistency tests
# ---------------------------

def test_llm_answer_consistency_basic(dataset_basic):
    n = len(dataset_basic.data)
    # Predictions: original and transformed
    y_pred_proba = np.array([[0.9, 0.1], [0.1, 0.9],  # original
                             [0.85, 0.15], [0.15, 0.85]])  # transformed

    results = llm_answer_consistency(None, None, dataset_basic, y_pred_proba)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, Measure)
        assert isinstance(r.score, float)
        assert isinstance(r.time, datetime.datetime)


def test_llm_answer_consistency_similarity_range(dataset_orthogonal_preds):
    y_pred_proba = np.array([[1, 0], [0, 1]])  # original + transformed
    results = llm_answer_consistency(None, None, dataset_orthogonal_preds, y_pred_proba)
    mean_sim = results[0].score
    assert np.isclose(mean_sim, 0.0, atol=1e-6)


# ---------------------------
# llm_performance_drop tests
# ---------------------------

def test_llm_performance_drop_basic(dataset_basic):
    # Original correct, transformed incorrect
    y_pred_proba = np.array([
        [0.9, 0.1],  # original pred = 0
        [0.1, 0.9],  # original pred = 1
        [0.2, 0.8],  # transformed pred = 1 (incorrect)
        [0.8, 0.2]   # transformed pred = 0 (incorrect)
    ])
    results = llm_performance_drop(None, None, dataset_basic, y_pred_proba)

    assert len(results) == 3
    names = [r.name for r in results]
    assert "accuracy_original" in names
    assert "accuracy_transformed" in names
    assert "performance_drop" in names

    drop = [r for r in results if r.name == "performance_drop"][0].score
    assert np.isclose(drop, 1.0, atol=1e-6)


def test_llm_performance_drop_no_change(dataset_orthogonal_preds):
    y_pred_proba = np.array([[0.1, 0.9], [0.1, 0.9]])
    results = llm_performance_drop(None, None, dataset_orthogonal_preds, y_pred_proba)

    acc_o = [r for r in results if r.name == "accuracy_original"][0].score
    acc_t = [r for r in results if r.name == "accuracy_transformed"][0].score
    drop = [r for r in results if r.name == "performance_drop"][0].score

    assert acc_o == 1.0
    assert acc_t == 1.0
    assert drop == 0.0


# ---------------------------
# Invalid datasets
# ---------------------------

def test_llm_answer_consistency_invalid_dataset():
    # Dataset missing data
    dataset = Dataset(pid=uuid.uuid4(), shape=shape, data=None)
    y_pred_proba = np.zeros((2, 2))
    with pytest.raises(ValueError):
        llm_answer_consistency(None, None, dataset, y_pred_proba)


def test_llm_performance_drop_invalid_dataset():
    # Dataset with no label column
    df = pd.DataFrame({"text_original": ["a"], "text_transformed": ["a2"]})
    dataset = Dataset(pid=uuid.uuid4(), shape=shape, data=df)
    y_pred_proba = np.zeros((2, 2))
    with pytest.raises(ValueError):
        llm_performance_drop(None, None, dataset, y_pred_proba)
