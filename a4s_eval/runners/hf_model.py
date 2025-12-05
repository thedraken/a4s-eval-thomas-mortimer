import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFClassifier:
    """
    HuggingFace sentiment classifier that exposes a predict_proba(texts) API.
    """

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=16):
        # If possible, use GPU, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size

    def predict_probability(self, texts) -> np.ndarray:
        """
        Predict probabilities for a list of strings.
        """
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]

                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)

                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)
