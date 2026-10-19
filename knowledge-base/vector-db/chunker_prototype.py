from dataclasses import dataclass

@dataclass
class Block:
    text: str
    tokens: int
    path: tuple[str, ...]          # es. ("Head", "2")

MIN, TARGET, MAX = 150, 700, 900


def shared(a, b):
    """Quanti header hanno in comune due path."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def size_cost(tokens):
    """Minimo vicino a TARGET, penalità forte sotto MIN."""
    cost = ((TARGET - tokens) / TARGET) ** 2
    return cost + 5 if tokens < MIN else cost


def cut_cost(left, right):
    """Tagliare dentro una sezione costa molto, tra sezioni poco."""
    if left.path == right.path:
        return 3
    return 0.5 * shared(left.path, right.path)


def segment(blocks):
    n = len(blocks)
    best = [0] + [float("inf")] * n    # best[j] = costo minimo per blocks[:j]
    start = [0] * (n + 1)              # inizio dell'ultimo chunk di quella soluzione

    for end in range(1, n + 1):
        tokens = 0
        for begin in range(end - 1, -1, -1):     # chunk candidato: blocks[begin:end]
            tokens += blocks[begin].tokens
            if begin < end - 1:
                too_big = tokens > MAX
                new_chapter = shared(blocks[begin].path, blocks[begin + 1].path) == 0
                if too_big or new_chapter:
                    break
            cut = cut_cost(blocks[begin - 1], blocks[begin]) if begin else 0
            cost = best[begin] + size_cost(tokens) + cut
            if cost < best[end]:
                best[end], start[end] = cost, begin

    chunks, end = [], n
    while end:
        chunks.append(blocks[start[end]:end])
        end = start[end]
    return chunks[::-1]


def breadcrumb(chunk):
    """Header comuni + sottosezioni contenute: 'Head > 1 · 2'."""
    paths = [b.path for b in chunk]
    k = min(shared(paths[0], p) for p in paths)
    parts = list(paths[0][:k])
    children = list(dict.fromkeys(p[k] for p in paths if len(p) > k))
    if children:
        parts.append(" · ".join(children))
    return " > ".join(parts)
