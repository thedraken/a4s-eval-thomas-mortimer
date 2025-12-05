import pytest
import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import uuid

from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import noun_adj_transformation_accuracy

# ---------------------------
# Fixtures for datasets
# ---------------------------

# Minimal DataShape
shape = DataShape(features=[
    Feature(pid=uuid.uuid4(), name="text_original", feature_type=FeatureType.TEXT, min_value=None, max_value=None),
    Feature(pid=uuid.uuid4(), name="text_transformed", feature_type=FeatureType.TEXT, min_value=None, max_value=None)
])

@pytest.fixture
def reference_dataset_basic():
    df = pd.DataFrame({"text_original": ["The big dog"]})
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)

@pytest.fixture
def evaluated_dataset_basic():
    df = pd.DataFrame({"text_transformed": ["The large dog"]})
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)

@pytest.fixture
def evaluated_dataset_mixed():
    df = pd.DataFrame({"text_transformed": ["The strong dog"]})
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)

@pytest.fixture
def evaluated_dataset_empty():
    df = pd.DataFrame({"text_transformed": [""]})
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)

@pytest.fixture
def evaluated_dataset_length_mismatch():
    df = pd.DataFrame({"text_transformed": ["The large dog", "Extra"]})
    return Dataset(pid=uuid.uuid4(), shape=shape, data=df)


# ---------------------------
# Helper for mocking stanza
# ---------------------------

def make_fake_stanza_doc(pos_tags):
    """Creates a mock stanza Document object"""
    mock_word_objs = []
    for tag in pos_tags:
        w = MagicMock()
        w.xpos = tag
        mock_word_objs.append(w)

    mock_sentence = MagicMock()
    mock_sentence.words = mock_word_objs

    mock_doc = MagicMock()
    mock_doc.sentences = [mock_sentence]
    return mock_doc


# ---------------------------
# Tests using fixtures
# ---------------------------

@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_basic_accuracy(mock_nlp, reference_dataset_basic, evaluated_dataset_basic):
    mock_nlp.side_effect = [
        make_fake_stanza_doc(["NN", "JJ"]),
        make_fake_stanza_doc(["NN", "JJ"]),
    ]

    results = noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_basic)

    assert len(results) == 3
    for m in results:
        assert isinstance(m, Measure)
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)

    assert results[0].score == 1.0  # noun_accuracy
    assert results[1].score == 1.0  # adjective_accuracy


@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_mixed_accuracy(mock_nlp, reference_dataset_basic, evaluated_dataset_mixed):
    mock_nlp.side_effect = [
        make_fake_stanza_doc(["NN", "JJ"]),  # reference
        make_fake_stanza_doc(["NN", "NN"]),  # evaluated
    ]

    results = noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_mixed)

    assert results[0].score == 1.0  # noun_accuracy matches
    assert results[1].score == 0.0  # adjective_accuracy does not match


@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_empty_text(mock_nlp, reference_dataset_basic, evaluated_dataset_empty):
    mock_nlp.side_effect = [
        make_fake_stanza_doc([]),
        make_fake_stanza_doc([]),
    ]

    results = noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_empty)

    assert results[0].score == 0.0
    assert results[1].score == 0.0
    assert results[2].score == 0.0


def test_mismatched_dataset_lengths(reference_dataset_basic, evaluated_dataset_length_mismatch):
    with pytest.raises(ValueError, match="Reference and evaluated datasets must have the same number of samples"):
        noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_length_mismatch)


def test_missing_fields():
    # Completely empty dataset
    bad_ref = Dataset(pid=uuid.uuid4(), shape=shape, data=pd.DataFrame())
    bad_eval = Dataset(pid=uuid.uuid4(), shape=shape, data=pd.DataFrame())
    with pytest.raises(ValueError, match="Both reference and evaluated datasets must contain text data"):
        noun_adj_transformation_accuracy(None, bad_ref, bad_eval)
