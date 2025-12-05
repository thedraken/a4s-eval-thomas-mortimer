import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from a4s_eval.data_model.evaluation import DataShape, Dataset, FeatureType, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@prediction_metric(name="NLP Noun Adjective consistency")
def llm_answer_consistency(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    """
    Computes similarity between predictions on original vs transformed texts.

    Assumes:
      - dataset.data has N rows (one per logical sample)
      - y_pred_proba has 2N rows:
          first N  -> predictions for original text
          next  N  -> predictions for transformed text
    """

    if dataset.data is None:
        raise ValueError("Dataset must contain data")

    n = len(dataset.data)
    if y_pred_proba.shape[0] != 2 * n:
        raise ValueError("Expected predictions for original+transformed (2N samples).")

    preds_orig = y_pred_proba[:n]
    preds_trans = y_pred_proba[n:]

    sims: list[float] = []
    for p1, p2 in zip(preds_orig, preds_trans):
        sims.append(_cosine_sim(p1, p2))

    now = datetime.datetime.now()
    sims_arr = np.array(sims, dtype=float)

    mean_sim = float(np.mean(sims_arr)) if sims else 0.0
    min_sim = float(np.min(sims_arr)) if sims else 0.0
    max_sim = float(np.max(sims_arr)) if sims else 0.0

    return [
        Measure(name="mean_prediction_similarity", score=mean_sim, time=now),
        Measure(name="min_similarity", score=min_sim, time=now),
        Measure(name="max_similarity", score=max_sim, time=now),
    ]

@prediction_metric(name="NLP Noun Adjective performance")
def llm_performance_drop(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    """
    Computes accuracy drop between original and transformed text predictions.

    Assumes:
      - dataset.data has a 'y' column with gold labels
      - dataset.data has N rows
      - y_pred_proba has 2N rows:
          first N  -> predictions for original text
          next  N  -> predictions for transformed text
    """

    if dataset.data is None:
        raise ValueError("Dataset must contain data")

    text_original = next((f.name for f in datashape.features
                          if f.name == "text_original"), None)
    text_transformed = next((f.name for f in datashape.features
                             if f.name == "text_transformed"), None)

    if text_original is None or text_transformed is None:
        # Not a text-transformation dataset → return neutral measures
        return [
            Measure(name="original_accuracy", score=0.0,
                    time=datetime.datetime.now()),
            Measure(name="transformed_accuracy", score=0.0,
                    time=datetime.datetime.now()),
            Measure(name="performance_drop", score=0.0,
                    time=datetime.datetime.now()),
        ]

    if datashape.target is None or datashape.target.name not in dataset.data.columns:
        raise ValueError("DataShape must specify a target feature")
    y_true = dataset.data[datashape.target.name].to_numpy()

    n = len(y_true)

    original_pred = y_pred_proba[:n]
    transformed_pred = y_pred_proba[n:]

    if len(original_pred) != len(transformed_pred):
        raise ValueError(
            "Predictions must contain original and transformed pairs")

    # Compute accuracies
    y_pred_o = original_pred.argmax(axis=1)
    y_pred_t = transformed_pred.argmax(axis=1)

    acc_o = (y_pred_o == y_true).mean()
    acc_t = (y_pred_t == y_true).mean()

    drop = (acc_o - acc_t)

    now = datetime.datetime.now()
    return [
        Measure(name="original_accuracy", score=float(acc_o), time=now),
        Measure(name="transformed_accuracy", score=float(acc_t), time=now),
        Measure(name="performance_drop", score=float(drop), time=now),
    ]
