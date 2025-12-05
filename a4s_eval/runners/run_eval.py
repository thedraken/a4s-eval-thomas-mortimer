import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hf_model import HFClassifier

from a4s_eval.data_model.measure import Measure
from a4s_eval.runners.imdb_runner import NLPTransformationEvaluator


# ---------------------------------------------------------------------------
# Dummy Model (replace with real HFClassifier if desired)
# ---------------------------------------------------------------------------
class DummyModel:
    """Simple model that returns random probability distributions."""
    def predict_proba(self, texts):
        probs = np.random.rand(len(texts), 2)
        return probs / probs.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_performance_drop(measures):
    acc_original = next(m.score for m in measures if m.name == "original_accuracy")
    acc_transformed = next(m.score for m in measures if m.name == "transformed_accuracy")

    plt.figure(figsize=(6, 4))
    plt.bar(["Original", "Transformed"], [acc_original, acc_transformed], color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Performance Drop")
    plt.show()


def plot_consistency(measures):
    similarities = [m.score for m in measures if m.name.startswith("mean") or m.name.startswith("min") or m.name.startswith("max")]
    labels = [m.name for m in measures]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, similarities, color="lightgreen")
    plt.ylim(0, 1)
    plt.ylabel("Cosine Similarity")
    plt.title("Prediction Consistency")
    plt.show()


def plot_pos_accuracy(measures):
    scores = [m.score for m in measures]
    labels = [m.name for m in measures]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color="orange")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("POS Transformation Accuracy")
    plt.show()


# ---------------------------------------------------------------------------
# Pretty-printing helper
# ---------------------------------------------------------------------------
def print_measure_list(name, measures):
    print(f"\n==== {name} ====")
    for m in measures:
        if isinstance(m, Measure):
            print(f"{m.name}: {m.score:.4f}")
        else:
            print(m)


# ---------------------------------------------------------------------------
# Main CLI runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP LLM robustness on IMDB dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to IMDB CSV")
    parser.add_argument("--model", type=str, default="dummy",
                        help="Model type. Options: dummy (default), hf-bert, etc.")
    parser.add_argument("--plot", action="store_true", help="Generate plots for metrics")

    args = parser.parse_args()

    print("Loading dataset:", args.csv)
    df = pd.read_csv(args.csv)

    # Choose model
    if args.model == "dummy":
        model = DummyModel()
    elif args.model == "hf":
        model = HFClassifier()
    else:
        raise NotImplementedError("Only 'dummy' is implemented. Add more models here.")

    evaluator = NLPTransformationEvaluator(model=model)

    print("Running evaluation...")
    results = evaluator.evaluate(df)

    print("\n=========== FINAL RESULTS ===========")
    for key, value in results.items():
        print_measure_list(key, value)

    # Plot if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_performance_drop(results["performance_drop"])
        plot_consistency(results["consistency"])
        plot_pos_accuracy(results["pos_accuracy"])


if __name__ == "__main__":
    main()
