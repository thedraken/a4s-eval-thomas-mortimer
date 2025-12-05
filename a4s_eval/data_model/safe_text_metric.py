from datetime import datetime
from typing import List
from a4s_eval.data_model.measure import Measure


def safe_text_metric(metric_name: str, orig_texts: list, transformed_texts: list, compute_score_fn) -> List[Measure]:
    """
    Generic safe wrapper for text metrics.

    Args:
        metric_name: name of the metric
        orig_texts: list of original text strings
        transformed_texts: list of transformed text strings
        compute_score_fn: function(orig_text:str, transformed_text:str) -> float

    Returns:
        List[Measure]
    """
    measures = []

    current_time = datetime.now()

    if not orig_texts or not transformed_texts:
        return [
            Measure(
                name=metric_name,
                score=-1.0,
                time=current_time,
                feature_pid=None
            )
        ]

    for orig, transformed in zip(orig_texts, transformed_texts):
        if not isinstance(orig, str) or not isinstance(transformed, str):
            continue  # skip invalid rows
        try:
            score = compute_score_fn(orig, transformed)
        except Exception:
            score = -1.0  # fallback on error
        measures.append(
            Measure(
                name=metric_name,
                score=score,
                time=current_time,
                feature_pid=None
            )
        )

    if not measures:
        measures.append(
            Measure(
                name=metric_name,
                score=-1.0,
                time=current_time,
                feature_pid=None
            )
        )

    return measures
