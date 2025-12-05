import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import List

from a4s_eval.data_model.measure import Measure
from a4s_eval.runners.imdb_runner import NLPTransformationEvaluator
from hf_model import HFClassifier


class DummyModel:
    """
    A mock class for demonstrating a basic model structure.

    This class is intended for generating random probability distributions
    for demonstration purposes. Generally, you should use HuggingFace
    """
    def predict_probability(self, texts):
        probs = np.random.rand(len(texts), 2)
        return probs / probs.sum(axis=1, keepdims=True)

def plot_performance_drop(measures: list[Measure]):
    """
    Plots a bar chart to visualise the drop in performance between original
    and transformed accuracies.

    Arguments:
        measures (list): A list of objects, each containing 'name' and 'score' attributes.

    Raises:
        StopIteration: If 'original_accuracy' or 'transformed_accuracy' is not found in the
        provided measures.
    """
    acc_original = next(m.score for m in measures if m.name == "original_accuracy")
    acc_transformed = next(m.score for m in measures if m.name == "transformed_accuracy")

    plt.figure(figsize=(6, 4))
    plt.bar(["Original", "Transformed"], [acc_original, acc_transformed], color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Performance Drop")
    plt.show()


def plot_consistency(measures: list[Measure]):
    similarities = [m.score for m in measures if m.name.startswith("mean") or m.name.startswith("min") or m.name.startswith("max")]
    labels = [m.name for m in measures]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, similarities, color="lightgreen")
    plt.ylim(0, 1)
    plt.ylabel("Cosine Similarity")
    plt.title("Prediction Consistency")
    plt.show()


def plot_pos_accuracy(measures: list[Measure]):
    scores = [m.score for m in measures]
    labels = [m.name for m in measures]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color="orange")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("POS Transformation Accuracy")
    plt.show()

def print_measure_list(name: str, measures: list[Measure | str]):
    """
    Prints a formatted list of measures with associated scores.

    Parameters:
    name (str): The heading to be printed above the list of measures.
    measures (list): A list of measures to be printed.

    """
    print(f"\n==== {name} ====")
    for m in measures:
        if isinstance(m, Measure):
            print(f"{m.name}: {m.score:.4f}")
        else:
            print(m)

def main():
    """
    Parses command-line arguments, loads the IMDB dataset, initialises a
    selected NLP model, evaluates the model's robustness using the dataset,
    and optionally generates performance plots.

    Raises NotImplementedError if an unsupported model type is specified.

    Arguments:
        --csv (str): Path to the IMDB CSV file. This argument is mandatory.
        Check Readme for Thomas Mortimer's project for generation details
        --model (str): Model type to use for evaluation.
        Options include 'dummy' (default) and 'hf'.
        --plot: If specified, generates plots for performance metrics.

    Returns:
        None: The function does not return any value.

    Raises:
        NotImplementedError: Raised when an unsupported model type is specified.

    """
    parser = argparse.ArgumentParser(description="Evaluate NLP LLM robustness on IMDB dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to IMDB CSV")
    parser.add_argument("--model", type=str, default="dummy",
                        help="Model type. Options: dummy (default), hf.")
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
