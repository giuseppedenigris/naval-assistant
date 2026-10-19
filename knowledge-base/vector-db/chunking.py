"""Markdown chunking: blocks on blank lines, merged by dynamic programming."""

from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN, TARGET, MAX = 600, 800, 1200  # chunk size in tokens
SMALL_PENALTY = 5.0                # extra cost for chunks below MIN
CUT_COST = 1.0                     # cost of cutting inside a section; cutting before a header is free


@dataclass
class Block:
    text: str
    tokens: int
    section: str        # title of the enclosing section
    is_header: bool


@dataclass
class Chunk:
    text: str
    tokens: int
    sections: list[str]


def to_blocks(md: str, count: Callable[[str], int]) -> list[Block]:
    """One block per blank-line separated part; parts over MAX are split recursively."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX, chunk_overlap=0, length_function=count)
    blocks, section = [], ""
    for part in md.split("\n\n"):
        part = part.strip()
        if not part:
            continue
        is_header = part.startswith("#")
        if is_header:
            section = part.lstrip("# ")
        for piece in splitter.split_text(part):
            blocks.append(Block(piece, count(piece), section, is_header))
            is_header = False
    return blocks


def size_cost(tokens: int) -> float:
    cost = ((TARGET - tokens) / TARGET) ** 2
    return cost + SMALL_PENALTY if tokens < MIN else cost


def cut_cost(prev: Block, block: Block) -> float:
    """Cost of cutting between prev and block: free only before a header that follows content."""
    return 0.0 if block.is_header and not prev.is_header else CUT_COST


def segment(blocks: list[Block]) -> list[list[Block]]:
    """Minimum-cost partition into consecutive chunks of at most MAX tokens."""
    n = len(blocks)
    best = [0.0] + [float("inf")] * n   # best[j]: min cost of blocks[:j]
    start = [0] * (n + 1)               # start[j]: first block of the last chunk in best[j]

    for end in range(1, n + 1):
        tokens = 0
        for begin in range(end - 1, -1, -1):
            tokens += blocks[begin].tokens
            if tokens > MAX and begin < end - 1:
                break
            # at begin == 0 every solution pays the same cut cost, so blocks[-1] is harmless
            cost = best[begin] + size_cost(tokens) + cut_cost(blocks[begin - 1], blocks[begin])
            if cost < best[end]:
                best[end], start[end] = cost, begin

    chunks, end = [], n
    while end:
        chunks.append(blocks[start[end]:end])
        end = start[end]
    return chunks[::-1]


def chunk(md: str, count: Callable[[str], int]) -> list[Chunk]:
    return [
        Chunk(
            text="\n\n".join(b.text for b in group),
            tokens=sum(b.tokens for b in group),
            sections=list(dict.fromkeys(b.section for b in group if b.section)),
        )
        for group in segment(to_blocks(md, count))
    ]
