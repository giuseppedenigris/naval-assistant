"""Chunk one markdown file and write a size report followed by every chunk in full."""

import sys
from pathlib import Path

from chunking import MAX, MIN, TARGET, chunk


def count_words(text: str) -> int:
    return len(text.split())


def main(src: str, dst: str) -> None:
    chunks = chunk(Path(src).read_text(encoding="utf-8"), count_words)

    with open(dst, "w", encoding="utf-8") as out:
        out.write(f"file: {src}\n")
        out.write(f"MIN/TARGET/MAX: {MIN}/{TARGET}/{MAX}\n")
        out.write(f"chunks: {len(chunks)}\n\n")
        for i, c in enumerate(chunks):
            out.write(f"{i:4d}  {c.tokens:5d} words  {' · '.join(c.sections)}\n")

        for i, c in enumerate(chunks):
            out.write(f"\n{'=' * 80}\nchunk {i} | {c.tokens} words\n{'=' * 80}\n{c.text}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
