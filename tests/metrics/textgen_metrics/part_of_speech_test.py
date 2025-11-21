import uuid

import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
)
from a4s_eval.metrics.textgen_metrics.part_of_speech_constraint_metric import \
    PartOfSpeechConstraintMetric
from a4s_eval.service.functional_model import TextGenerationModel


@pytest.fixture
def imdb_dataset() -> pd.DataFrame:
    data = {
        "text": [
            "This movie was fantastic! The acting was superb.",
            "I did not like this film at all.",
            "The plot was boring and predictable.",
        ],
        "label": [1, 0, 0],  # 1=positive, 0=negative
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_shape() -> DataShape:
    def text_feat(feature_name: str) -> Feature:
        return Feature(
            pid=uuid.uuid4(),
            name=feature_name,
            feature_type=FeatureType.TEXT,
            min_value=0,
            max_value=0,
        )
    features = [text_feat("text")]
    target = text_feat("label")
    data_shape = {"features": features, "target": target}
    return DataShape.model_validate(data_shape)

@pytest.fixture
def test_dataset(imdb_dataset: pd.DataFrame, data_shape: DataShape) -> Dataset:
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=imdb_dataset)

def mock_perturb_text(text: str) -> str:
    replacements = {
        "movie": "film",
        "fantastic": "amazing",
        "boring": "dull",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def test_part_of_speech_constraint_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TextGenerationModel,
):
    # Initialise the metric
    metric = PartOfSpeechConstraintMetric()

    # Mock the perturbed text for testing
    if test_dataset.data is not None:
        for i, example in test_dataset.data.iterrows():
            original_text = example["text"]
            perturbed_text = mock_perturb_text(original_text)
            test_dataset.data.at[i, "text"] = perturbed_text

    # Run the metric
    measures = metric(data_shape, ref_model, test_dataset, functional_model)

    # Assertions
    assert len(measures) > 0
    assert all(0 <= m.score <= 1 for m in measures)
    assert any(m.name == "part_of_speech_constraint_metric_avg" for m in measures)

def test_empty_text():
    metric = PartOfSpeechConstraintMetric()
    original_tags = []
    perturbed_tags = []
    score = metric.evaluate_pos_tags(original_tags, perturbed_tags)
    assert score == 0.0

def test_no_drift():
    metric = PartOfSpeechConstraintMetric()
    original_tags = [("movie", "NOUN"), ("was", "VERB")]
    perturbed_tags = [("film", "NOUN"), ("was", "VERB")]
    score = metric.evaluate_pos_tags(original_tags, perturbed_tags)
    assert score == 1.0

def test_full_drift():
    metric = PartOfSpeechConstraintMetric()
    original_tags = [("movie", "NOUN"), ("was", "VERB")]
    perturbed_tags = [("movie", "ADJ"), ("was", "NOUN")]
    score = metric.evaluate_pos_tags(original_tags, perturbed_tags)
    assert score == 0.0