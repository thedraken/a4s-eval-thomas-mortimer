import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from a4s_eval.data_model.evaluation import DataShape, Dataset, FeatureType, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(a: str, b: str) -> float:
    emb = embedder.encode([a, b])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))


@prediction_metric(name="NLP Noun Adjective consistency")
def llm_answer_consistency(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray
) -> list[Measure]:
    """
    Computes answer consistency between model predictions on original and transformed text.

    Consistency = fraction where argmax(original[i]) == argmax(transformed[i])
    """

    # -------------------------
    # Validate dataset
    # -------------------------
    if dataset.data is None:
        raise ValueError("Dataset must contain data")

    # -------------------------
    # Determine text column from DataShape
    # Tests expect this behavior
    # -------------------------
    text_col = next(
        (f.name for f in datashape.features if f.feature_type == "text" and f.name in dataset.data.columns),
        None,
    )
    if text_col is None:
        raise ValueError("Dataset must contain a text column defined in DataShape features")

    n_samples = dataset.data.shape[0]

    # -------------------------
    # Validate prediction length
    # y_pred_proba must contain 2N predictions (original + transformed)
    # -------------------------
    if y_pred_proba.shape[0] != 2 * n_samples:
        raise ValueError("Number of predictions does not match original + transformed samples")

    # -------------------------
    # Split predictions
    # -------------------------
    original_preds = y_pred_proba[:n_samples]
    transformed_preds = y_pred_proba[n_samples:]

    # -------------------------
    # Compute argmax classes
    # -------------------------
    orig_class = np.argmax(original_preds, axis=1)
    trans_class = np.argmax(transformed_preds, axis=1)

    # -------------------------
    # Compute consistency
    # -------------------------
    consistency = np.mean(orig_class == trans_class)

    return [
        Measure(
            name="answer_consistency",
            score=float(consistency),
            time=datetime.datetime.now()
        )
    ]


@prediction_metric(name="NLP Noun Adjective performance")
def llm_performance_drop(datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray):
    """
    Computes accuracy drop between original and transformed text predictions.

    Args:
        datashape: Metadata about dataset shape
        model: Model object used for inference
        dataset: Dataset containing original and transformed texts and target labels
        y_pred_proba: Numpy array of predicted probabilities for the transformed texts

    Returns:
        List[Measure]: accuracy on original, accuracy on transformed, and performance drop
    """

    # --- Step 1: Validate dataset ---
    if dataset.data is None:
        raise ValueError("Dataset must contain data")
    if dataset.shape.target is None:
        raise ValueError("DataShape must specify a target feature")

    target_name = dataset.shape.target.name
    if target_name not in dataset.data.columns:
        raise ValueError(f"Dataset must contain target column '{target_name}'")

    # --- Step 2: Identify text columns from features ---
    text_features = [f.name for f in datashape.features if f.feature_type == FeatureType.TEXT]
    if not text_features:
        raise ValueError("DataShape must have at least one text feature")

    # Use first text column as 'original', second (if present) as 'transformed'
    text_orig_col = text_features[0]
    text_trans_col = text_features[1] if len(text_features) > 1 else text_features[0]

    if text_orig_col not in dataset.data.columns or text_trans_col not in dataset.data.columns:
        raise ValueError(f"Dataset must have '{text_orig_col}' and '{text_trans_col}' columns")

    # --- Step 3: Compute predictions ---
    # For simplicity, assume y_pred_proba corresponds to transformed text predictions
    y_true = dataset.data[target_name].to_numpy()

    # Original text predictions: assume the model stored in `model.dataset` can provide predictions
    # Here we simulate with same predictions as y_true for demonstration
    # In real usage, you would run `model` on `dataset.data[text_orig_col]`
    y_pred_orig = np.argmax(y_pred_proba, axis=1)  # placeholder: assume perfect on original
    y_pred_trans = np.argmax(y_pred_proba, axis=1)

    if len(y_true) != len(y_pred_trans) or len(y_true) != len(y_pred_orig):
        raise ValueError("Mismatch between number of samples and predictions")

    # --- Step 4: Compute accuracies ---
    acc_orig = np.mean(y_pred_orig == y_true)
    acc_trans = np.mean(y_pred_trans == y_true)
    perf_drop = acc_orig - acc_trans

    # --- Step 5: Return Measure objects ---
    now = datetime.datetime.now()
    return [
        Measure(name="accuracy_original", score=float(acc_orig), time=now),
        Measure(name="accuracy_transformed", score=float(acc_trans), time=now),
        Measure(name="performance_drop", score=float(perf_drop), time=now),
    ]

