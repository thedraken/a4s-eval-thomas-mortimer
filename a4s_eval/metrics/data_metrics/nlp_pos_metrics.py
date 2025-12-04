from a4s_eval.metric_registries.data_metric_registry import data_metric
from a4s_eval.data_model.measure import Measure
import stanza
import numpy as np
import datetime

nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_no_ssplit=True)


@data_metric(name="Noun/Adjective Transformation Accuracy")
def noun_adj_transformation_accuracy(datashape, reference, evaluated):
    """
    Args:
        datashape: Metadata about the dataset shape
        reference: Original dataset
        evaluated: Transformed dataset

    Returns: A list of measures, that includes the accuracy of the nouns, adjectives and the overall stability

    """
    ref_texts = getattr(reference, "X", None)
    if ref_texts is None and hasattr(reference, "df"):
        ref_texts = reference.df["text"].tolist()

    eval_texts = getattr(evaluated, "X", None)
    if eval_texts is None and hasattr(evaluated, "df"):
        eval_texts = evaluated.df["text"].tolist()

    if ref_texts is None or eval_texts is None:
        raise ValueError(
            "Both reference and evaluated datasets must contain text data (X or df['text'])")

    if len(ref_texts) != len(eval_texts):
        raise ValueError(
            "Reference and evaluated datasets must have the same number of samples")

    noun_correct = []
    adj_correct = []
    overall_correct = []

    for orig_text, trans_text in zip(ref_texts, eval_texts):

        orig_doc = nlp(orig_text)
        trans_doc = nlp(trans_text)

        orig_pos = [w.xpos for sent in orig_doc.sentences for w in sent.words]
        trans_pos = [w.xpos for sent in trans_doc.sentences for w in sent.words]

        # POS-tag sequences might differ in length.
        L = min(len(orig_pos), len(trans_pos))

        for p1, p2 in zip(orig_pos[:L], trans_pos[:L]):

            # Nouns
            if p1.startswith("NN"):
                noun_correct.append(int(p2.startswith("NN")))

            # Adjectives
            if p1.startswith("JJ"):
                adj_correct.append(int(p2.startswith("JJ")))

            # Overall POS category stability
            overall_correct.append(int((p1[0] == p2[0])))

    noun_acc = float(np.mean(noun_correct)) if noun_correct else 0.0
    adj_acc = float(np.mean(adj_correct)) if adj_correct else 0.0
    overall_acc = float(np.mean(overall_correct)) if overall_correct else 0.0

    return [
        Measure(name="noun_accuracy", score=noun_acc, time=datetime.datetime.now()),
        Measure(name="adjective_accuracy", score=adj_acc, time=datetime.datetime.now()),
        Measure(name="overall_pos_stability", score=overall_acc, time=datetime.datetime.now()),
    ]
