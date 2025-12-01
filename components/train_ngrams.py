from pathlib import Path

from .ngram_model import SmartNGramModel, load_corpus, save_model


CORPUS_PATH = Path("data/stories.txt")
MODELS_DIR = Path("models")


def train_one_ngram(n: int, model_path: Path, min_count: int, top_k: int):
    stories = load_corpus(CORPUS_PATH)
    print(f"[n={n}] Loaded {len(stories)} stories from {CORPUS_PATH}")

    model = SmartNGramModel(n=n, min_count=min_count, top_k=top_k)
    model.fit(stories)
    save_model(model, model_path)

    print(f"[n={n}] Saved model to {model_path}")


def main():
    configs = [
        (2, MODELS_DIR / "ngram_2.pkl", 2, 12),
        (3, MODELS_DIR / "ngram_3.pkl", 2, 10),
        (4, MODELS_DIR / "ngram_4.pkl", 3, 8),
        (5, MODELS_DIR / "ngram_5.pkl", 3, 6),
    ]

    for n, path, min_count, top_k in configs:
        train_one_ngram(n=n, model_path=path, min_count=min_count, top_k=top_k)


if __name__ == "__main__":
    main()
