import uuid

import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Feature, FeatureType
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import (
    noun_adj_transformation_accuracy,
)
from a4s_eval.metrics.prediction_metrics.nlp_llm_robustness_metrics import (
    llm_answer_consistency,
    llm_performance_drop,
)


class NLPTransformationEvaluator:
    """
    Loads a dataset with text_original, text_transformed, label.
    Runs predictions on original+transformed and evaluates all metrics.
    """

    def __init__(self, model):
        self.model = model

    def build_feature(self, name: str, values: list, ftype: FeatureType) -> Feature:
        if ftype == FeatureType.TEXT or ftype == FeatureType.CATEGORICAL:
            min_v, max_v = None, None
        elif ftype in (FeatureType.INTEGER, FeatureType.FLOAT):
            min_v = float(min(values))
            max_v = float(max(values))
        elif ftype == FeatureType.DATE:
            min_v = min(values)
            max_v = max(values)
        else:
            min_v, max_v = None, None
        return Feature(
            pid=uuid.uuid4(),
            name=name,
            feature_type=ftype,
            min_value=min_v,
            max_value=max_v,
        )

    def build_datashape(self, df: pd.DataFrame) -> DataShape:
        feat_original = self.build_feature(
            "text_original", df["text_original"].tolist(), FeatureType.TEXT
        )
        feat_transformed = self.build_feature(
            "text_transformed", df["text_transformed"].tolist(), FeatureType.TEXT
        )
        feat_label = self.build_feature(
            "label", df["label"].tolist(), FeatureType.CATEGORICAL
        )
        return DataShape(
            features=[feat_original, feat_transformed], target=feat_label, date=None
        )

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        datashape = self.build_datashape(df)
        return Dataset(pid=uuid.uuid4(), shape=datashape, data=df)

    def run_predictions(self, dataset: Dataset) -> np.ndarray:
        df = dataset.data
        texts = df["text_original"].tolist() + df["text_transformed"].tolist()
        return self.model.predict_probability(texts)

    def evaluate(self, df: pd.DataFrame):
        print("Preparing dataset...")
        dataset = self.prepare_dataset(df)
        print("Running predictions...")
        y_pred = self.run_predictions(dataset)

        print("Running consistency check...")
        consistency = llm_answer_consistency(
            datashape=dataset.shape,
            model=self.model,
            dataset=dataset,
            y_pred_proba=y_pred,
        )

        print("Running performance check...")
        performance = llm_performance_drop(
            datashape=dataset.shape,
            model=self.model,
            dataset=dataset,
            y_pred_proba=y_pred,
        )

        print("Running accuracy check...")
        pos_accuracy = noun_adj_transformation_accuracy(
            datashape=dataset.shape, reference=dataset, evaluated=dataset
        )

        print("Evaluation complete.")
        return {
            "consistency": consistency,
            "performance_drop": performance,
            "pos_accuracy": pos_accuracy,
        }
