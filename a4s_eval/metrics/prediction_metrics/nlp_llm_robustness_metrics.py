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
def llm_answer_consistency(datashape: DataShape, model, dataset: Dataset, y_pred_proba: np.ndarray):
    """
    Computes similarity between predictions on original vs transformed texts.
    Args:
        datashape: Unused argument for this check, but tells us what shape the dataset is in
        model: The model to be analysed, unused for this test
        dataset: The dataset must contain: dataset.X_original, dataset.X_transformed
        y_pred_proba: y_pred_proba has one prediction per sample in dataset.X

    Returns:

    """
    if not hasattr(dataset, "X_original") or not hasattr(dataset, "X_transformed"):
        raise ValueError("Dataset must contain X_original and X_transformed fields for consistency metric.")

    n = len(dataset.X_original)
    if len(y_pred_proba) != 2 * n:
        raise ValueError("Expected predictions for original+transformed (2N samples).")

    pred_original = y_pred_proba[:n]
    pred_transformed = y_pred_proba[n:]

    similarities = []
    for p1, p2 in zip(pred_original, pred_transformed):
        sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        similarities.append(sim)

    mean_prediction_similarity = Measure(name="mean_prediction_similarity", score=float(np.mean(similarities)),
                                        time=datetime.datetime.now())
    min_similarity = Measure(name="min_similarity", score=float(np.min(similarities)), time=datetime.datetime.now())
    max_similarity = Measure(name="max_similarity", score=float(np.max(similarities)), time=datetime.datetime.now())

    return [mean_prediction_similarity, min_similarity, max_similarity]

@prediction_metric(name="NLP Noun Adjective performance")
def llm_performance_drop(datashape: DataShape, model, dataset: Dataset, y_pred_proba: np.ndarray):
    """
    Computes accuracy drop between original and transformed text predictions
    Args:
        datashape: The shape of the data to be analysed
        model: The model of the data
        dataset: Contains labels:
        dataset.X_original
        dataset.X_transformed
        dataset.y
        y_pred_proba: Contains predictions in the order:
        [original samples..., transformed samples...]
    Returns:

    """

    if not hasattr(dataset, "X_original") or not hasattr(dataset, "X_transformed"):
        raise ValueError("Dataset must contain X_original and X_transformed.")

    y_true = dataset.y
    n = len(dataset.X_original)

    if len(y_pred_proba) != 2 * n:
        raise ValueError("Expected predictions for original+transformed samples in y_pred_proba.")

    pred_original = np.argmax(y_pred_proba[:n], axis=1)
    pred_transformed = np.argmax(y_pred_proba[n:], axis=1)

    acc_original = float(np.mean(pred_original == y_true))
    acc_transformed = float(np.mean(pred_transformed == y_true))

    drop = (acc_original - acc_transformed) / max(acc_original, 1e-9)

    accuracy_original = Measure(name="accuracy_original", score=acc_original, time=datetime.datetime.now())
    accuracy_transformed = Measure(name="accuracy_transformed", score=acc_transformed, time=datetime.datetime.now())
    performance_drop = Measure(name="performance_drop", score=drop, time=datetime.datetime.now())

    return [accuracy_original, accuracy_transformed, performance_drop]
