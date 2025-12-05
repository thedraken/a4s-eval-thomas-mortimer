import pytest
import datetime
from unittest.mock import patch, MagicMock

from a4s_eval.data_model.measure import Measure
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import noun_adj_transformation_accuracy

class MockDataset:
    """
    Mock Dataset class
    """
    def __init__(self, X):
        self.X = X


def make_fake_stanza_doc(pos_tags):
    """
    Creates a mock stanza Document object:
    doc.sentences[0].words[i].xpos = tag
    Args:
        pos_tags: The tags of the sentence

    Returns:A mock of the sentence

    """
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


@pytest.fixture
def reference_dataset_basic():
    return MockDataset(["The big dog"])


@pytest.fixture
def evaluated_dataset_basic():
    return MockDataset(["The large dog"])


@pytest.fixture
def evaluated_dataset_mixed():
    return MockDataset(["The strong dog"])


@pytest.fixture
def evaluated_dataset_length_mismatch():
    return MockDataset(["The large dog"])


@pytest.fixture
def evaluated_dataset_empty():
    return MockDataset([""])


# ---------------------------------------------------------------------
# Tests using fixtures
# ---------------------------------------------------------------------

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
        make_fake_stanza_doc(["NN", "JJ"]),  # original
        make_fake_stanza_doc(["NN", "NN"]),  # evaluated
    ]

    results = noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_mixed)

    assert results[0].score == 1.0
    assert results[1].score == 0.0


@patch("a4s_eval.metrics.data_metrics.nlp_pos_metrics.nlp")
def test_length_mismatch(mock_nlp, reference_dataset_basic, evaluated_dataset_length_mismatch):
    mock_nlp.side_effect = [
        make_fake_stanza_doc(["NN", "JJ", "NN"]),
        make_fake_stanza_doc(["NN", "JJ"]),
    ]

    results = noun_adj_transformation_accuracy(None, reference_dataset_basic, evaluated_dataset_length_mismatch)

    assert results[0].score == 1.0
    assert results[1].score == 1.0


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


def test_missing_fields():
    reference = type("Bad", (), {})()
    evaluated = MockDataset(["x"])
    with pytest.raises(ValueError):
        noun_adj_transformation_accuracy(None, reference, evaluated)


def test_mismatched_dataset_lengths():
    reference = MockDataset(["a"])
    evaluated = MockDataset(["a", "b"])
    with pytest.raises(ValueError):
        noun_adj_transformation_accuracy(None, reference, evaluated)
