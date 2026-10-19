"""Sections of a markdown document, each labelled [§n] by the position of its header in the document."""

import re
from itertools import count

HEADER = re.compile(r"^(#{1,6})[ \t]+", re.M)
LABEL = re.compile(r"^#{1,6} \[§(\d+)\] (.*)$", re.M)
ANCHOR = re.compile(r"\[\[(?:IMG|TABLE):\w+\]\]")


def label_sections(md: str) -> str:
    n = count(1)
    return HEADER.sub(lambda m: f"{m.group(1)} [§{next(n)}] ", md)


def split_sections(labelled: str) -> list[tuple[int, str, str]]:
    """(number, title, text) of every labelled section; text before the first header is left out."""
    labels = list(LABEL.finditer(labelled))
    ends = [m.start() for m in labels[1:]] + [len(labelled)]
    return [(int(m.group(1)), m.group(2), labelled[m.start():end]) for m, end in zip(labels, ends)]
