# demo_compare_ngrams.py
from pathlib import Path
import pickle

from .ngram_model import SmartNGramModel


MODELS = {
    "2": Path("models/ngram_2.pkl"),
    "3": Path("models/ngram_3.pkl"),
    "4": Path("models/ngram_4.pkl"),
    "5": Path("models/ngram_5.pkl"),
}


def load_model_file(path: Path) -> SmartNGramModel:
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    models: dict[str, SmartNGramModel] = {}
    for n_str, path in MODELS.items():
        if not path.exists():
            print(f"[WARN] Model for n={n_str} not found at {path}, skipping.")
            continue
        models[n_str] = load_model_file(path)
        print(f"[INFO] Loaded n={n_str} from {path}")

    if not models:
        print("No models loaded. Train them first with train_ngrams.py.")
        return

    while True:
        prefix = input("\nEnter story beginning (or 'quit'): ").strip()
        if not prefix or prefix.lower() in {"quit", "exit"}:
            break

        print("\n=== N-gram comparison ===")
        for n_str, model in models.items():
            draft = model.generate_multi(prefix, num_sentences=3, max_tokens=80)
            print(f"\n[n={n_str} draft]:")
            print(draft)


if __name__ == "__main__":
    main()
