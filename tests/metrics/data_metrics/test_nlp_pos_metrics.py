import uuid
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import DataShape, Dataset, Feature, FeatureType
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import (
    noun_adj_transformation_accuracy,
)


# Mock a "doc" object returned by nlp()
def fake_stanza_doc(pos_tags):
    """
    Returns a fake object that mimics stanza.Document with token.pos attributes.
    pos_tags: list of strings, e.g. ["NN", "JJ"]
    """
    doc = MagicMock()
    # tokens is a list of mock words, each with a pos attribute
    doc.sentences = []
    sentence = MagicMock()
    sentence.words = [MagicMock(pos=tag) for tag in pos_tags]
    doc.sentences.append(sentence)
    return doc


# --- Fixture for DataShape ---
@pytest.fixture
def reference_dataset_basic():
    data = pd.DataFrame({"text_original": ["The big dog", "A fast cat"]})
    shape = DataShape(
        features=[
            Feature(
                pid=uuid.uuid4(),
                name="text_original",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            )
        ]
    )
    return Dataset(pid=uuid.uuid4(), shape=shape, data=data)


@pytest.fixture
def evaluated_dataset_mixed():
    data = pd.DataFrame({"text_transformed": ["The strong dog", "A quick cat"]})
    shape = DataShape(
        features=[
            Feature(
                pid=uuid.uuid4(),
                name="text_transformed",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            )
        ]
    )
    return Dataset(pid=uuid.uuid4(), shape=shape, data=data)


@pytest.fixture
def empty_datasets():
    shape_ref = DataShape(
        features=[
            Feature(
                pid=uuid4(),
                name="text_original",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            )
        ]
    )
    shape_eval = DataShape(
        features=[
            Feature(
                pid=uuid4(),
                name="text_transformed",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            )
        ]
    )
    return (
        Dataset(pid=uuid4(), shape=shape_ref, data=pd.DataFrame()),
        Dataset(pid=uuid4(), shape=shape_eval, data=pd.DataFrame()),
    )


@pytest.fixture
def text_datashape():
    # Provide a default DataShape that includes both expected features
    return DataShape(
        features=[
            Feature(
                pid=uuid.uuid4(),
                name="text_original",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            ),
            Feature(
                pid=uuid.uuid4(),
                name="text_transformed",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            ),
        ]
    )


@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_mixed_accuracy(
    mock_nlp, text_datashape, reference_dataset_basic, evaluated_dataset_mixed
):
    """
    Tests the accuracy of noun and adjective transformations in a mixed
    dataset.

    Args:
        mock_nlp: Mocked NLP pipeline for generating fake documents with predefined
            parts of speech (POS) tags.
        text_datashape: Datashape of the text data being examined.
        reference_dataset_basic: The reference dataset used for comparing the
            evaluation results.
        evaluated_dataset_mixed: The dataset being evaluated, containing mixed
            transformations.

    Raises:
        AssertionError: If the evaluated noun or adjective transformation
            accuracy does not match the expected values.
    """
    # Patch nlp pipeline to return fake docs
    mock_nlp.side_effect = [
        fake_stanza_doc(["NN", "JJ"]),  # ref 1
        fake_stanza_doc(["NN", "NN"]),  # eval 1
        fake_stanza_doc(["NN", "JJ"]),  # ref 2
        fake_stanza_doc(["NN", "NN"]),  # eval 2
    ]

    results = noun_adj_transformation_accuracy(
        datashape=text_datashape,
        reference=reference_dataset_basic,
        evaluated=evaluated_dataset_mixed,
    )

    noun_acc = [m.score for m in results if m.name == "noun_accuracy"][0]
    adj_acc = [m.score for m in results if m.name == "adjective_accuracy"][0]

    assert noun_acc == 1.0
    assert adj_acc == 0.0
