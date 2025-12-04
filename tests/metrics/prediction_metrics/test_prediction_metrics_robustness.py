import numpy as np
import pytest
import datetime

from a4s_eval.data_model.measure import Measure
from a4s_eval.metrics.prediction_metrics.nlp_llm_robustness_metrics import llm_answer_consistency, llm_performance_drop


# ------------------------------------------------------------
# Mock dataset class matching expected fields
# ------------------------------------------------------------

class MockDataset:
    def __init__(self, X_original, X_transformed, y=None):
        self.X_original = X_original
        self.X_transformed = X_transformed
        self.y = y


# ------------------------------------------------------------
# llm_answer_consistency TESTS
# ------------------------------------------------------------
def test_llm_answer_consistency_basic():
    n = 3
    dataset = MockDataset(
        X_original=["a", "b", "c"],
        X_transformed=["a2", "b2", "c2"]
    )

    # create predictions with clear similarity
    original_preds = np.array([[0.9, 0.1]] * n)
    transformed_preds = np.array([[0.85, 0.15]] * n)

    y_pred_proba = np.vstack([original_preds, transformed_preds])

    results = llm_answer_consistency(None, None, dataset, y_pred_proba)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, Measure)
        assert isinstance(r.score, float)
        assert isinstance(r.time, datetime.datetime)


def test_llm_answer_consistency_invalid_dataset():
    dataset = type("Bad", (), {})()  # missing fields
    y_pred_proba = np.zeros((2, 2))

    with pytest.raises(ValueError):
        llm_answer_consistency(None, None, dataset, y_pred_proba)


def test_llm_answer_consistency_invalid_prediction_length():
    dataset = MockDataset(["a"], ["a2"])
    y_pred_proba = np.zeros((1, 2))  # should be 2 samples

    with pytest.raises(ValueError):
        llm_answer_consistency(None, None, dataset, y_pred_proba)


def test_llm_answer_consistency_similarity_range():
    dataset = MockDataset(["a"], ["a2"])

    p1 = np.array([1, 0])
    p2 = np.array([0, 1])

    y_pred_proba = np.vstack([p1, p2])  # 1 original + 1 transformed

    results = llm_answer_consistency(None, None, dataset, y_pred_proba)
    mean_sim = results[0].score

    # dot product of orthogonal vectors = 0
    assert np.isclose(mean_sim, 0.0, atol=1e-6)


# ------------------------------------------------------------
# llm_performance_drop TESTS
# ------------------------------------------------------------
def test_llm_performance_drop_basic():
    X_original = ["a", "b"]
    X_transformed = ["a2", "b2"]
    y_true = np.array([0, 1])

    dataset = MockDataset(X_original, X_transformed, y_true)

    # correct predictions for original, incorrect for transformed
    y_pred_proba = np.array([
        [0.9, 0.1],  # original pred = 0 (correct)
        [0.1, 0.9],  # original pred = 1 (correct)
        [0.2, 0.8],  # transformed pred = 1 (incorrect)
        [0.8, 0.2]   # transformed pred = 0 (incorrect)
    ])

    results = llm_performance_drop(None, None, dataset, y_pred_proba)

    assert len(results) == 3
    names = [r.name for r in results]
    assert "accuracy_original" in names
    assert "accuracy_transformed" in names
    assert "performance_drop" in names

    # original accuracy = 1.0
    # transformed accuracy = 0.0
    # drop = (1 - 0) / 1 = 1
    drop = [r for r in results if r.name == "performance_drop"][0].score
    assert np.isclose(drop, 1.0, atol=1e-6)

def test_llm_performance_drop_invalid_dataset():
    dataset = type("Bad", (), {})()  # missing X_original
    y_pred_proba = np.zeros((2, 2))

    with pytest.raises(ValueError):
        llm_performance_drop(None, None, dataset, y_pred_proba)

def test_llm_performance_drop_invalid_prediction_length():
    dataset = MockDataset(["a"], ["a2"], np.array([0]))

    # only 1 prediction, should be 2
    y_pred_proba = np.zeros((1, 2))

    with pytest.raises(ValueError):
        llm_performance_drop(None, None, dataset, y_pred_proba)


def test_llm_performance_drop_no_change():
    # identical predictions for original + transformed
    X_original = ["a"]
    X_transformed = ["a2"]
    y_true = np.array([1])

    dataset = MockDataset(X_original, X_transformed, y_true)

    y_pred_proba = np.array([
        [0.1, 0.9],  # original correct
        [0.1, 0.9]   # transformed identical
    ])

    results = llm_performance_drop(None, None, dataset, y_pred_proba)

    acc_o = [r for r in results if r.name == "accuracy_original"][0].score
    acc_t = [r for r in results if r.name == "accuracy_transformed"][0].score
    drop = [r for r in results if r.name == "performance_drop"][0].score

    assert acc_o == 1.0
    assert acc_t == 1.0
    assert drop == 0.0
