import argparse
import pandas as pd

try:
    import stanza
    stanza.download("en")
    nlp = stanza.Pipeline("en", processors="tokenize,pos")
    USE_STANZA = True
except Exception:
    print("⚠️  Stanza not available — using mock transformation.")
    USE_STANZA = False

from datasets import load_dataset

def transform_text(text):
    """
    Attempts a noun/adjective swap using stanza.
    If stanza unavailable, returns mock transformation.
    """
    if not USE_STANZA:
        return text.replace("good", "great").replace("bad", "terrible")

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
