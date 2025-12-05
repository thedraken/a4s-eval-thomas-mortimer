import argparse
import pandas as pd
import numpy as np

from a4s_eval.data_model.measure import Measure
from a4s_eval.runners.imdb_runner import NLPTransformationEvaluator


class DummyModel:
    """Simple model that returns random probability distributions."""
    def predict_proba(self, texts):
        probs = np.random.rand(len(texts), 2)
        return probs / probs.sum(axis=1, keepdims=True)

def print_measure_list(name, measures):
    print(f"\n==== {name} ====")
    for m in measures:
        if isinstance(m, Measure):
            print(f"{m.name}: {m.score}")
        else:
            print(m)

def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP LLM robustness on IMDB dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to IMDB CSV")
    parser.add_argument("--model", type=str, default="dummy",
                        help="Model type. Options: dummy (default), hf-bert, etc.")

    args = parser.parse_args()

    print("Loading dataset:", args.csv)
    df = pd.read_csv(args.csv)

    # Choose model
    if args.model == "dummy":
        model = DummyModel()
    elif args.model == "hf":
        from hf_model import HFClassifier
        model = HFClassifier()
    else:
        raise NotImplementedError("Only 'dummy' is implemented. Add more models here.")

    evaluator = NLPTransformationEvaluator(model=model)

    print("Running evaluation...")
    results = evaluator.evaluate(df)

    print("\n=========== FINAL RESULTS ===========")
    for key, value in results.items():
        print_measure_list(key, value)


if __name__ == "__main__":
    main()
