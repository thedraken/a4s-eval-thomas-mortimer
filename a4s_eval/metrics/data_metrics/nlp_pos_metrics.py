from a4s_eval.metric_registries.data_metric_registry import data_metric
from a4s_eval.data_model.measure import Measure
import stanza
import numpy as np
import datetime

nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_no_ssplit=True)


@data_metric(name="Noun/Adjective Transformation Accuracy")
def noun_adj_transformation_accuracy(datashape, reference, evaluated):
    """
    Computes how well a transformed dataset preserved or correctly
    altered nouns and adjectives.

    Parameters
    ----------
    datashape : DataShape
    reference : Dataset      (original dataset)
    evaluated : Dataset      (transformed dataset)

    Returns
    -------
    List[Measure]:
        noun_accuracy
        adj_accuracy
        overall_accuracy
    """

    if not hasattr(reference, "X") or not hasattr(evaluated, "X"):
        raise ValueError("Both reference and evaluated datasets must contain field X")

    if len(reference.X) != len(evaluated.X):
        raise ValueError("Reference and evaluated datasets must have same sample count")

    noun_correct = []
    adj_correct = []
    overall_correct = []

    for orig_text, trans_text in zip(reference.X, evaluated.X):

        orig_doc = nlp(orig_text)
        trans_doc = nlp(trans_text)

        # Flatten tokens
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

    # Safe means when lists empty
    noun_acc = float(np.mean(noun_correct)) if noun_correct else 0.0
    adj_acc = float(np.mean(adj_correct)) if adj_correct else 0.0
    overall_acc = float(np.mean(overall_correct)) if overall_correct else 0.0

    return [
        Measure(name="noun_accuracy", score=noun_acc, time=datetime.datetime.now()),
        Measure(name="adjective_accuracy", score=adj_acc, time=datetime.datetime.now()),
        Measure(name="overall_pos_stability", score=overall_acc, time=datetime.datetime.now()),
    ]
