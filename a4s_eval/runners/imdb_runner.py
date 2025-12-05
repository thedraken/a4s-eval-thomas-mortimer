import uuid
import numpy as np
import pandas as pd


from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.metrics.data_metrics.nlp_pos_metrics import noun_adj_transformation_accuracy
from a4s_eval.metrics.prediction_metrics.nlp_llm_robustness_metrics import llm_answer_consistency, llm_performance_drop


class NLPTransformationEvaluator:
    """
    Loads an IMDB-style dataset with text_original, text_transformed, label.
    Runs predictions on original+transformed and evaluates via the three metrics.
    """

    def __init__(self, model):
        """
        model: must implement predict_proba(list_of_texts)
        """
        self.model = model

    def build_feature(self, name: str, values: list, ftype: FeatureType) -> Feature:
        """
        Creates a Feature object with correct type and min/max.
        """
        if ftype == FeatureType.TEXT:
            min_v, max_v = None, None
        elif ftype in (FeatureType.INTEGER, FeatureType.FLOAT):
            min_v = float(min(values))
            max_v = float(max(values))
        elif ftype == FeatureType.CATEGORICAL:
            min_v = None
            max_v = None
        elif ftype == FeatureType.DATE:
            min_v = min(values)
            max_v = max(values)
        else:
            min_v = None
            max_v = None

        return Feature(
            pid=uuid.uuid4(),
            name=name,
            feature_type=ftype,
            min_value=min_v,
            max_value=max_v,
        )

    def build_datashape(self, df: pd.DataFrame) -> DataShape:
        """
        Build a DataShape with correct Feature objects.
        """

        feat_original = self.build_feature(
            name="text_original",
            values=df["text_original"].tolist(),
            ftype=FeatureType.TEXT,
        )

        feat_transformed = self.build_feature(
            name="text_transformed",
            values=df["text_transformed"].tolist(),
            ftype=FeatureType.TEXT,
        )

        feat_label = self.build_feature(
            name="label",
            values=df["label"].tolist(),
            ftype=FeatureType.CATEGORICAL,
        )

        datashape = DataShape(
            features=[feat_original, feat_transformed],
            target=feat_label,
            date=None
        )

        return datashape

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Wrap IMDB DataFrame in your Dataset class.
        """

        datashape = self.build_datashape(df)

        dataset = Dataset(
            pid=uuid.uuid4(),
            shape=datashape,
            data=df
        )

        # Prepare expected fields for internal metrics
        dataset.X_original = df["text_original"].tolist()
        dataset.X_transformed = df["text_transformed"].tolist()
        dataset.y = df["label"].to_numpy()

        return dataset

    def run_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        Calls model.predict_proba() on both original+transformed samples.
        """
        texts = dataset.X_original + dataset.X_transformed
        return self.model.predict_proba(texts)

    def evaluate(self, df: pd.DataFrame):
        """
        Full evaluation pipeline.
        """
        dataset = self.prepare_dataset(df)
        y_pred = self.run_predictions(dataset)

        # Prediction metrics
        consistency = llm_answer_consistency(
            datashape=dataset.shape,
            model=self.model,
            dataset=dataset,
            y_pred_proba=y_pred
        )

        performance = llm_performance_drop(
            datashape=dataset.shape,
            model=self.model,
            dataset=dataset,
            y_pred_proba=y_pred
        )

        # Data metric
        pos_accuracy = noun_adj_transformation_accuracy(
            datashape=dataset.shape,
            reference=dataset,
            evaluated=dataset
        )

        return {
            "consistency": consistency,
            "performance_drop": performance,
            "pos_accuracy": pos_accuracy,
        }