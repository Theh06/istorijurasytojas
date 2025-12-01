from pathlib import Path

from components.ngram_model import SmartNGramModel, load_model


MODEL_FILES = {
    2: Path("models/ngram_2.pkl"),
    3: Path("models/ngram_3.pkl"),
    4: Path("models/ngram_4.pkl"),
    5: Path("models/ngram_5.pkl"),
}


def compute_stats(model: SmartNGramModel) -> dict:
    n = model.n
    vocab_size = len(model.vocab)
    total_tokens = model.total_tokens
    num_contexts = len(model.context_counts)

    if num_contexts > 0:
        total_next_variants = sum(len(counter) for counter in model.context_counts.values())
        avg_next_per_context = total_next_variants / num_contexts
        max_next_per_context = max(len(counter) for counter in model.context_counts.values())
    else:
        avg_next_per_context = 0.0
        max_next_per_context = 0

    return {
        "n": n,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "num_contexts": num_contexts,
        "avg_next_per_context": avg_next_per_context,
        "max_next_per_context": max_next_per_context,
    }


def main():
    rows: list[dict] = []

    for n, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[WARN] model file for n={n} not found at {path}, skipping.")
            continue

        print(f"[INFO] loading model n={n} from {path}...")
        model = load_model(path)
        stats = compute_stats(model)
        rows.append(stats)

    if not rows:
        print("No models loaded. Train n-gram models first.")
        return

    print("\nN-gram model statistics:\n")

    header = (
        "n",
        "vocab_size",
        "total_tokens",
        "num_contexts",
        "avg_next_per_context",
        "max_next_per_context",
    )
    col_widths = [len(h) for h in header]
    for row in rows:
        col_widths[0] = max(col_widths[0], len(str(row["n"])))
        col_widths[1] = max(col_widths[1], len(str(row["vocab_size"])))
        col_widths[2] = max(col_widths[2], len(str(row["total_tokens"])))
        col_widths[3] = max(col_widths[3], len(str(row["num_contexts"])))
        col_widths[4] = max(
            col_widths[4],
            len(f"{row['avg_next_per_context']:.2f}"),
        )
        col_widths[5] = max(col_widths[5], len(str(row["max_next_per_context"])))

    def fmt_row(values):
        return "  ".join(
            str(v).rjust(w) for v, w in zip(values, col_widths)
        )

    print(fmt_row(header))
    print(fmt_row("-" * w for w in col_widths))

    for row in rows:
        values = (
            row["n"],
            row["vocab_size"],
            row["total_tokens"],
            row["num_contexts"],
            f"{row['avg_next_per_context']:.2f}",
            row["max_next_per_context"],
        )
        print(fmt_row(values))


if __name__ == "__main__":
    main()
