import argparse
import pandas as pd
import stanza
from datasets import load_dataset


stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos")



def transform_text(text):
    """
    Attempts a noun/adjective swap using stanza.
    If stanza unavailable, returns mock transformation.
    """
    doc = nlp(text)
    out = []

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == "NOUN":
                out.append("object")
            elif word.upos == "ADJ":
                out.append("descriptive")
            else:
                out.append(word.text)

    return " ".join(out)

def generate_csv(limit, output):
    print("Downloading IMDB dataset...")
    imdb = load_dataset("imdb")

    rows = []

    print("Processing IMDB samples...")
    for i, sample in enumerate(imdb["train"]):
        if i >= limit:
            break

        original = sample["text"]
        transformed = transform_text(original)
        label = sample["label"]

        rows.append({
            "text_original": original,
            "text_transformed": transformed,
            "label": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved CSV → {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate transformed IMDB CSV")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of IMDB samples to convert")
    parser.add_argument("--output", type=str, default="imdb_transformed.csv",
                        help="Output CSV filename")

    args = parser.parse_args()
    generate_csv(args.limit, args.output)


if __name__ == "__main__":
    main()
