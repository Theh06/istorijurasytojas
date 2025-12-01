import re
from pathlib import Path


RAW_PATH = Path("data/reddit_short_stories.txt")
OUT_PATH = Path("data/stories.txt")


def tokenize(text: str):
    text = text.lower()

    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def clean_story_block(block: str) -> str | None:
    if "<eos>" in block:
        block = block.split("<eos>")[0]


    block = block.replace("<sos>", " ")
    block = block.replace("<nl>", " ")


    block = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", block)

    block = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", block)
    block = re.sub(r"_([^_]+)_", r"\1", block)

    block = re.split(r"ALTERNATE ENDINGS", block, maxsplit=1)[0]

    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("^"):
            continue
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) < 20:
        return None

    return text


def build_corpus(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH):
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path.resolve()}")

    raw_text = raw_path.read_text(encoding="utf-8")

    stories: list[str] = []

    for chunk in raw_text.split("<sos>"):
        chunk = chunk.strip()
        if not chunk:
            continue

        story = clean_story_block(chunk)
        if story is None:
            continue

        stories.append(story)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in stories:
            f.write(s + "\n")

    print(f"Saved {len(stories)} cleaned stories to {out_path}")


if __name__ == "__main__":
    build_corpus()
