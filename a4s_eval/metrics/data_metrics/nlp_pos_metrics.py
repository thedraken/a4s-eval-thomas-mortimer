import datetime

import numpy as np
import stanza

from a4s_eval.data_model.evaluation import DataShape, Dataset, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric

nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_no_ssplit=True)


@data_metric(name="Noun/Adjective Transformation Accuracy")
def noun_adj_transformation_accuracy(datashape: DataShape, reference: Dataset, evaluated: Dataset):
    """
    Computes accuracy of noun and adjective preservation after text transformation.
    If no text columns exist, returns a neutral Measure so non-NLP projects still pass.
    """

    # ---------- Step 1: Identify text feature ----------
    text_features = [
        f for f in datashape.features
        if f.feature_type == FeatureType.TEXT
    ]

    # No text → dataset irrelevant for this metric → return neutral metric
    if not text_features:
        return [
            Measure(
                name="noun_accuracy",
                score=0.0,
                time=datetime.datetime.now()
            ),
            Measure(
                name="adjective_accuracy",
                score=0.0,
                time=datetime.datetime.now()
            ),
            Measure(
                name="overall_pos_stability",
                score=0.0,
                time=datetime.datetime.now()
            )
        ]

    # We expect exactly one text feature
    text_col = text_features[0].name

    # ---------- Step 2: Column names for reference/evaluated ----------
    ref_col = text_col
    eval_col = text_col.replace("original", "transformed")

    # If evaluated dataset does not include transformed column → neutral output
    if ref_col not in reference.data.columns or eval_col not in evaluated.data.columns:
        return [
            Measure(name="noun_accuracy", score=0.0, time=datetime.datetime.now()),
            Measure(name="adjective_accuracy", score=0.0, time=datetime.datetime.now()),
            Measure(name="overall_pos_stability", score=0.0, time=datetime.datetime.now()),
        ]

    ref_texts = reference.data[ref_col].astype(str).tolist()
    eval_texts = evaluated.data[eval_col].astype(str).tolist()

    if len(ref_texts) != len(eval_texts):
        return [
            Measure(name="noun_accuracy", score=0.0, time=datetime.datetime.now()),
            Measure(name="adjective_accuracy", score=0.0, time=datetime.datetime.now()),
            Measure(name="overall_pos_stability", score=0.0, time=datetime.datetime.now()),
        ]

    # ---------- Step 3: POS tagging ----------
    noun_correct = []
    adj_correct = []
    overall_correct = []

    for t1, t2 in zip(ref_texts, eval_texts):
        doc1 = nlp(t1)
        doc2 = nlp(t2)

        pos1 = [w.xpos for sent in doc1.sentences for w in sent.words]
        pos2 = [w.xpos for sent in doc2.sentences for w in sent.words]

        L = min(len(pos1), len(pos2))

        for p1, p2 in zip(pos1[:L], pos2[:L]):
            if p1.startswith("NN"):
                noun_correct.append(int(p2.startswith("NN")))
            if p1.startswith("JJ"):
                adj_correct.append(int(p2.startswith("JJ")))
            overall_correct.append(int(p1[0] == p2[0]))

    noun_acc = float(np.mean(noun_correct)) if noun_correct else 0.0
    adj_acc = float(np.mean(adj_correct)) if adj_correct else 0.0
    overall_acc = float(np.mean(overall_correct)) if overall_correct else 0.0

    # ---------- Step 4: Return results ----------
    now = datetime.datetime.now()

    return [
        Measure(name="noun_accuracy", score=noun_acc, time=now),
        Measure(name="adjective_accuracy", score=adj_acc, time=now),
        Measure(name="overall_pos_stability", score=overall_acc, time=now),
    ]

