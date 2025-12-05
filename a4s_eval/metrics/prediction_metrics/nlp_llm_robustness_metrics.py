from sentence_transformers import SentenceTransformer
import numpy as np
import datetime

from a4s_eval.data_model.evaluation import DataShape, Dataset
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(a: str, b: str) -> float:
    emb = embedder.encode([a, b])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))


@prediction_metric(name="NLP Noun Adjective consistency")
def llm_answer_consistency(datashape, model, dataset, y_pred_proba: np.ndarray):
    """
    Computes similarity between predictions on original vs transformed texts.
    Supports datasets with either:
        - dataset.data['text_original'] / dataset.data['text_transformed']
        - dataset.data['text']
    """

    if dataset.data is None:
        raise ValueError("Dataset must contain data")

    # Determine text columns
    if 'text_original' in dataset.data.columns and 'text_transformed' in dataset.data.columns:
        n = len(dataset.data['text_original'])
    elif 'text' in dataset.data.columns:
        n = len(dataset.data['text']) // 2  # first half original, second half transformed
    else:
        raise ValueError("Dataset must contain 'text_original'/'text_transformed' or 'text' column")

    if len(y_pred_proba) != 2 * n:
        raise ValueError("Expected predictions for original+transformed (2N samples)")

    pred_original = y_pred_proba[:n]
    pred_transformed = y_pred_proba[n:]

    # Cosine similarity per sample
    similarities = []
    for p1, p2 in zip(pred_original, pred_transformed):
        sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        similarities.append(sim)

    return [
        Measure(name="mean_prediction_similarity", score=float(np.mean(similarities)), time=datetime.datetime.now()),
        Measure(name="min_similarity", score=float(np.min(similarities)), time=datetime.datetime.now()),
        Measure(name="max_similarity", score=float(np.max(similarities)), time=datetime.datetime.now()),
    ]

@prediction_metric(name="NLP Noun Adjective performance")
def llm_performance_drop(datashape, model, dataset, y_pred_proba: np.ndarray):
    """
    Computes accuracy drop between original and transformed text predictions.
    Uses dataset.data['y'] or dataset.data['label'] for true labels.
    """

    if dataset.data is None:
        raise ValueError("Dataset must contain data")

    # Flexible label detection
    if 'y' in dataset.data.columns:
        y_true = dataset.data['y'].to_numpy()
    elif 'label' in dataset.data.columns:
        y_true = dataset.data['label'].to_numpy()
    else:
        raise ValueError("Dataset must contain 'y' or 'label' column for true labels")

    # Determine number of samples
    if 'text_original' in dataset.data.columns and 'text_transformed' in dataset.data.columns:
        n = len(dataset.data['text_original'])
    elif 'text' in dataset.data.columns:
        n = len(dataset.data['text']) // 2
    else:
        raise ValueError("Dataset must contain text columns to determine sample size")

    if len(y_pred_proba) != 2 * n:
        raise ValueError("Expected predictions for original+transformed samples in y_pred_proba")

    pred_original = np.argmax(y_pred_proba[:n], axis=1)
    pred_transformed = np.argmax(y_pred_proba[n:], axis=1)

    acc_original = float(np.mean(pred_original == y_true[:n]))
    acc_transformed = float(np.mean(pred_transformed == y_true[:n]))
    drop = (acc_original - acc_transformed) / max(acc_original, 1e-9)

    return [
        Measure(name="accuracy_original", score=acc_original, time=datetime.datetime.now()),
        Measure(name="accuracy_transformed", score=acc_transformed, time=datetime.datetime.now()),
        Measure(name="performance_drop", score=drop, time=datetime.datetime.now())
    ]

