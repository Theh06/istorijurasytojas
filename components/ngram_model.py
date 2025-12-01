import pickle
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

from .clean import tokenize, OUT_PATH as CORPUS_PATH


class SmartNGramModel:
    def __init__(self, n: int = 4, min_count: int = 3, top_k: int = 8):
        if n < 2:
            raise ValueError("n must be >= 2 for n-gram model")

        self.n = n
        self.min_count = min_count
        self.top_k = top_k

        self.context_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.unigram_counts: Counter = Counter()
        self.vocab: set[str] = set()
        self.total_tokens: int = 0

    def fit(self, texts: list[str]):
        for text in texts:
            tokens = ["<bos>"] + tokenize(text) + ["<eos>"]
            self.unigram_counts.update(tokens)

        for text in texts:
            tokens = ["<bos>"] + tokenize(text) + ["<eos>"]
            norm_tokens = [
                tok if self.unigram_counts[tok] >= self.min_count else "<unk>"
                for tok in tokens
            ]
            self.total_tokens += len(norm_tokens)

            if len(norm_tokens) < self.n:
                continue

            for i in range(len(norm_tokens) - self.n + 1):
                context = tuple(norm_tokens[i : i + self.n - 1])
                next_tok = norm_tokens[i + self.n - 1]
                self.context_counts[context][next_tok] += 1

        self.vocab = set(self.unigram_counts.keys()) | {"<unk>"}

    def _next_dist(self, context: tuple) -> Counter:
        context = tuple(context)
        for k in range(len(context), 0, -1):
            subctx = context[-k:]
            if subctx in self.context_counts:
                return self.context_counts[subctx]

        return self.unigram_counts

    def _sample_next(self, context: tuple) -> str:
        dist = self._next_dist(context)
        if not dist:
            return random.choice(list(self.vocab))

        items = [(tok, c) for tok, c in dist.items() if tok != "<unk>"]
        if not items:
            items = list(dist.items())

        items.sort(key=lambda x: x[1], reverse=True)
        if self.top_k is not None and self.top_k > 0 and len(items) > self.top_k:
            items = items[: self.top_k]

        tokens, counts = zip(*items)
        total = sum(counts)
        r = random.random() * total
        cm = 0.0
        for tok, c in zip(tokens, counts):
            cm += c
            if r <= cm:
                return tok
        return tokens[-1]

    def generate_multi(self, prefix: str, num_sentences: int = 3, max_tokens: int = 80) -> str:
        prefix_tokens = tokenize(prefix)

        if prefix_tokens and prefix_tokens[-1] in {".", "!", "?"}:
            context_tokens = ["<bos>"]
        else:
            context_tokens = ["<bos>"] + prefix_tokens

        if len(context_tokens) < self.n - 1:
            context_tokens = ["<bos>"] * (self.n - 1 - len(context_tokens)) + context_tokens

        context = tuple(context_tokens[-(self.n - 1):])
        generated: list[str] = []
        sentence_count = 0

        for _ in range(max_tokens):
            next_tok = self._sample_next(context)

            if next_tok == "<eos>":
                break

            generated.append(next_tok)
            context = tuple(list(context[1:]) + [next_tok])

            if next_tok in {".", "!", "?"}:
                sentence_count += 1
                if sentence_count >= num_sentences:
                    break

        clean_tokens = [t for t in generated if t != "<unk>"]
        text = " ".join(clean_tokens)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        return text

    def generate(self, prefix: str, max_tokens: int = 40) -> str:
        return self.generate_multi(prefix, num_sentences=1, max_tokens=max_tokens)


def load_corpus(corpus_path: Path = CORPUS_PATH) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path.resolve()}")

    stories: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stories.append(line)

    return stories


def save_model(model: SmartNGramModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> SmartNGramModel:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    with path.open("rb") as f:
        model: SmartNGramModel = pickle.load(f)
    return model
