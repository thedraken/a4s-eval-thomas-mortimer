import datetime

import numpy as np
import stanza

from a4s_eval.data_model.evaluation import DataShape, Dataset, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric

nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_no_ssplit=True)


def _compute_pos_measures(ref_texts: list[str], eval_texts: list[str]) -> list[Measure]:
    noun_correct: list[int] = []
    adj_correct: list[int] = []
    overall_correct: list[int] = []

    for orig_text, trans_text in zip(ref_texts, eval_texts):
        # stanza expects strings; skip invalid rows
        if not isinstance(orig_text, str) or not isinstance(trans_text, str):
            continue

        orig_doc = nlp(orig_text)
        trans_doc = nlp(trans_text)

        orig_pos = [w.xpos for sent in orig_doc.sentences for w in sent.words]
        trans_pos = [w.xpos for sent in trans_doc.sentences for w in sent.words]

        L = min(len(orig_pos), len(trans_pos))
        for p1, p2 in zip(orig_pos[:L], trans_pos[:L]):
            if p1.startswith("NN"):  # nouns
                noun_correct.append(int(p2.startswith("NN")))
            if p1.startswith("JJ"):  # adjectives
                adj_correct.append(int(p2.startswith("JJ")))
            # rough POS stability: first letter of tag matches
            overall_correct.append(int(p1[:1] == p2[:1]))

    noun_acc = float(np.mean(noun_correct)) if noun_correct else 0.0
    adj_acc = float(np.mean(adj_correct)) if adj_correct else 0.0
    overall_acc = float(np.mean(overall_correct)) if overall_correct else 0.0

    now = datetime.datetime.now()
    return [
        Measure(name="noun_accuracy", score=noun_acc, time=now),
        Measure(name="adjective_accuracy", score=adj_acc, time=now),
        Measure(name="overall_pos_stability", score=overall_acc, time=now),
    ]


def _neutral_pos_measures() -> list[Measure]:
    now = datetime.datetime.now()
    return [
        Measure(name="noun_accuracy", score=0.0, time=now),
        Measure(name="adjective_accuracy", score=0.0, time=now),
        Measure(name="overall_pos_stability", score=0.0, time=now),
    ]


@data_metric(name="Noun/Adjective Transformation Accuracy")
def noun_adj_transformation_accuracy(
    datashape: DataShape | None, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    POS-based stability between reference and evaluated datasets.

    Behaviour:
      * If datashape is None:
          - Expect reference.data['text_original'] and evaluated.data['text_transformed'].
          - Raise ValueError if they are missing (matches unit tests).
      * If datashape is not None:
          - If no TEXT features exist -> return neutral measures (not applicable).
          - If TEXT features exist -> try to use 'text_original' / 'text_transformed'
            or fall back to any shared TEXT column names.
    """
    # Check dataset
    if reference.data is None or evaluated.data is None:
        raise ValueError("Both reference and evaluated datasets must contain data")

    # Columns from DataShape
    ref_col = next((f.name for f in datashape.features
                    if f.feature_type == FeatureType.TEXT and f.name in reference.data.columns and "original" in f.name),
                   None)
    eval_col = next((f.name for f in datashape.features
                     if f.feature_type == FeatureType.TEXT and f.name in evaluated.data.columns and "transformed" in f.name),
                    None)

    if ref_col is None or eval_col is None:
        # Not a text-transformation dataset → return neutral measures
        return [
            Measure(name="noun_accuracy", score=0.0,
                    time=datetime.datetime.now()),
            Measure(name="adjective_accuracy", score=0.0,
                    time=datetime.datetime.now()),
            Measure(name="overall_pos_stability", score=0.0,
                    time=datetime.datetime.now()),
        ]

    ref_texts = reference.data[ref_col].tolist()
    eval_texts = evaluated.data[eval_col].tolist()

    if len(ref_texts) != len(eval_texts):
        raise ValueError("Reference and evaluated datasets must have equal length")

    noun_correct = 0
    noun_total = 0
    adj_correct = 0
    adj_total = 0

    # --- For each pair of sentences ---
    max_size = len(ref_texts)
    counter = 0
    for ref_t, eval_t in zip(ref_texts, eval_texts):
        counter += 1
        print(f"Processing sentence {counter}/{max_size}")
        ref_doc = nlp(ref_t)
        eval_doc = nlp(eval_t)

        for ref_sent, eval_sent in zip(ref_doc.sentences, eval_doc.sentences):

            ref_pos = [w.pos for w in ref_sent.words]
            eval_pos = [w.pos for w in eval_sent.words]

            # Align only to the shared token length
            L = min(len(ref_pos), len(eval_pos))
            ref_pos = ref_pos[:L]
            eval_pos = eval_pos[:L]

            # Count noun matches
            for rp, ep in zip(ref_pos, eval_pos):
                if rp == "NN":
                    noun_total += 1
                    if ep == "NN":
                        noun_correct += 1

                if rp == "JJ":
                    adj_total += 1
                    if ep == "JJ":
                        adj_correct += 1

    noun_acc = noun_correct / noun_total if noun_total > 0 else 0.0
    adj_acc = adj_correct / adj_total if adj_total > 0 else 0.0

    overall = (noun_acc + adj_acc) / 2 if (noun_total + adj_total) > 0 else 0.0

    current_date = datetime.datetime.now()

    return [
        Measure(name="noun_accuracy", score=noun_acc, time=current_date),
        Measure(name="adjective_accuracy", score=adj_acc, time=current_date),
        Measure(name="overall_pos_stability", score=overall, time=current_date),
    ]
