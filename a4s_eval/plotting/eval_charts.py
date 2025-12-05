import matplotlib.pyplot as plt
from pathlib import Path


def plot_transformation_accuracy(scores: dict, out_dir: str | Path):
    """
    Generates and saves a bar plot illustrating the accuracy of POS transformations.

    Parameters:
    scores (dict): A dictionary where keys are the labels/identifiers of POS transformations
        and values are their corresponding accuracy scores.
    out_dir (str | Path): The directory where the generated plot should be saved. If it does
        not exist, it will be created.

    Returns:
    Path: The path to the saved plot file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(scores.keys())
    values = [scores[k] for k in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("POS Transformation Accuracy")
    plt.ylim(0, 1)

    plt.xticks(rotation=20)
    plt.tight_layout()

    out_path = out_dir / "pos_transformation_accuracy.png"
    plt.savefig(out_path)
    plt.close()

    return out_path


def plot_robustness_consistency(robustness: float, consistency: float, out_dir: str | Path):
    """
    Generates a bar plot for robustness and consistency scores, saves it as an image
    in the specified directory, and returns the file path.

    Parameters:
    robustness: float
        Robustness score as a decimal version of a percentage, e.g. 0.875 for 87.5%.
    consistency: float
        Consistency score as a decimal version of a percentage.
    out_dir: str | Path
        Directory where the resulting plot image will be saved.

    Returns:
    Path
        Path to the saved plot image file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Robustness", "Consistency"]
    values = [robustness, consistency]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Score")
    plt.title("LLM Robustness & Consistency")
    plt.ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / "robustness_consistency.png"
    plt.savefig(out_path)
    plt.close()

    return out_path
