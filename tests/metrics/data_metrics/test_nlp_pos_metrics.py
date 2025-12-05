import uuid

import pytest
import pandas as pd
from uuid import uuid4
from unittest.mock import patch
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import noun_adj_transformation_accuracy
from a4s_eval.data_model.evaluation import Feature, FeatureType, DataShape, Dataset
from unittest.mock import MagicMock

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
    shape = DataShape(features=[
        Feature(pid=uuid.uuid4(), name="text_original", feature_type=FeatureType.TEXT, min_value=None, max_value=None)
    ])
    return Dataset(pid=uuid.uuid4(), shape=shape, data=data)

@pytest.fixture
def evaluated_dataset_mixed():
    data = pd.DataFrame({"text_transformed": ["The strong dog", "A quick cat"]})
    shape = DataShape(features=[
        Feature(pid=uuid.uuid4(), name="text_transformed", feature_type=FeatureType.TEXT, min_value=None, max_value=None)
    ])
    return Dataset(pid=uuid.uuid4(), shape=shape, data=data)

@pytest.fixture
def empty_datasets():
    shape_ref = DataShape(features=[Feature(pid=uuid4(), name="text_original", feature_type=FeatureType.TEXT, min_value=None, max_value=None)])
    shape_eval = DataShape(features=[Feature(pid=uuid4(), name="text_transformed", feature_type=FeatureType.TEXT, min_value=None, max_value=None)])
    return (
        Dataset(pid=uuid4(), shape=shape_ref, data=pd.DataFrame()),
        Dataset(pid=uuid4(), shape=shape_eval, data=pd.DataFrame())
    )

# --- Test with patched NLP ---
@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_mixed_accuracy(mock_nlp, text_datashape, reference_dataset_basic, evaluated_dataset_mixed):
    # Patch nlp pipeline to return fake docs
    mock_nlp.side_effect = [
        fake_stanza_doc(["NN", "JJ"]),  # reference
        fake_stanza_doc(["NN", "NN"])   # evaluated
    ]

    results = noun_adj_transformation_accuracy(
        datashape=reference_dataset_basic.shape,
        reference=reference_dataset_basic,
        evaluated=evaluated_dataset_mixed
    )

    noun_acc = [m.score for m in results if m.name == "noun_accuracy"][0]
    adj_acc = [m.score for m in results if m.name == "adjective_accuracy"][0]

    assert noun_acc == 1.0
    assert adj_acc == 0.0
