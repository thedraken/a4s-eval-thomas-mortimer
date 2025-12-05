import uuid

import numpy as np
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import DataShape, Dataset, Feature, \
    FeatureType, Model
from a4s_eval.metrics.prediction_metrics.nlp_llm_robustness_metrics import \
    llm_answer_consistency, llm_performance_drop


@pytest.fixture
def dataset_basic():
    data = pd.DataFrame({
        "text_original": ["a", "b"],
        "text_transformed": ["a2", "b2"],
        "y": [0, 1]
    })
    shape = DataShape(
        features=[
            Feature(pid=uuid.uuid4(), name="text_original", feature_type=FeatureType.TEXT, min_value=None, max_value=None),
            Feature(pid=uuid.uuid4(), name="text_transformed", feature_type=FeatureType.TEXT, min_value=None, max_value=None),
        ],
        target=Feature(pid=uuid.uuid4(), name="y", feature_type=FeatureType.INTEGER, min_value=0, max_value=1),
    )
    return Dataset(pid=uuid.uuid4(), shape=shape, data=data)

@pytest.fixture
def model_basic(dataset_basic):
    return Model(pid=uuid.uuid4(), dataset=dataset_basic, model=None)

def test_llm_answer_consistency_basic(dataset_basic, model_basic):
    #features = dataset_basic.shape.features
    ds_shape = dataset_basic.shape

    y_pred_proba = np.vstack([
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.85, 0.15], [0.15, 0.85]])
    ])

    results = llm_answer_consistency(ds_shape, model_basic, dataset_basic, y_pred_proba)
    assert 1.0 >= results[0].score >= 0.0


def test_llm_performance_drop_basic(dataset_basic, model_basic):
    ds_shape = dataset_basic.shape
    y_pred_proba = np.vstack([
        np.array([[0.9, 0.1], [0.1, 0.9]]),  # original correct
        np.array([[0.2, 0.8], [0.8, 0.2]])  # transformed incorrect
    ])

    results = llm_performance_drop(ds_shape, model_basic, dataset_basic, y_pred_proba)

    original_acc = [m.score for m in results if m.name == "original_accuracy"][0]
    transformed_acc = [m.score for m in results if m.name == "transformed_accuracy"][0]
    drop = [m.score for m in results if m.name == "performance_drop"][0]

    assert original_acc > 0
    assert transformed_acc >= 0
    assert drop >= 0
