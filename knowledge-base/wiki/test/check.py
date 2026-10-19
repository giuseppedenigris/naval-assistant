"""Check the wiki against raw/: every statement cited, every citation pointing to an existing section with the same title, every [[slug]] to an existing page, every anchor to the cited section."""

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
CITATION = re.compile(r'\[source:\s*([^\]§"]+?)\s*§(\d+)\s*"([^"]*)"\]')
CITATION_START = re.compile(r"\[source:")
LABEL = re.compile(r"^#{1,6} \[§(\d+)\] (.*)$", re.M)
LINK = re.compile(r"\[\[(?!IMG:|TABLE:)([^\]|#]+)[^\]]*\]\]")
ANCHOR = re.compile(r"\[\[(?:IMG|TABLE):\w+\]\]")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
UNCHECKED = {"index", "log"}
MIN_WORDS = 4


def normalize(title: str) -> str:
    return re.sub(r"\W+", " ", title).strip().lower()


def read_sections(path: Path) -> dict[str, tuple[str, str]]:
    """Section number -> (title, text) of a labelled source."""
    text = path.read_text(encoding="utf-8")
    labels = list(LABEL.finditer(text))
    ends = [m.start() for m in labels[1:]] + [len(text)]
    return {m.group(1): (m.group(2), text[m.start():end]) for m, end in zip(labels, ends)}


def statements(body: str) -> list[str]:
    """Paragraphs and list items: the units that need a citation."""
    units, paragraph = [], []
    for line in body.splitlines():
        if LIST_ITEM.match(line) or not line.strip() or line.lstrip().startswith("#"):
            if paragraph:
                units.append(" ".join(paragraph))
                paragraph = []
            if LIST_ITEM.match(line):
                units.append(line.strip())
        else:
            paragraph.append(line.strip())
    if paragraph:
        units.append(" ".join(paragraph))
    return units


def needs_citation(unit: str) -> bool:
    bare = CITATION.sub("", ANCHOR.sub("", LINK.sub("", LIST_ITEM.sub("", unit))))
    return len(re.findall(r"\w+", bare)) >= MIN_WORDS


def main() -> None:
    sources = {path.name: read_sections(path) for path in (ROOT / "raw").glob("*.md")}
    pages = {path.stem: path for path in (ROOT / "wiki").rglob("*.md")}

    outcome = Counter()
    problems = []

    def report(kind: str, message: str) -> None:
        outcome[kind] += 1
        problems.append(f"[{kind}] {message}")

    for slug, path in sorted(pages.items()):
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if target.strip() not in pages:
                report("broken link", f"{slug}: [[{target}]]")
        if slug in UNCHECKED:
            continue
        if not FRONT_MATTER.match(text):
            report("missing front matter", slug)

        scope = []
        for unit in statements(FRONT_MATTER.sub("", text)):
            citations = CITATION.findall(unit)
            if len(CITATION_START.findall(unit)) > len(citations):
                report("malformed citation", f"{slug}: {unit[:120]}")
            for source, n, title in citations:
                if source not in sources:
                    report("unknown source", f"{slug}: {source}")
                elif n not in sources[source]:
                    report("unknown section", f"{slug}: {source} §{n}")
                elif normalize(sources[source][n][0]) != normalize(title):
                    report("title mismatch", f'{slug}: {source} §{n} is "{sources[source][n][0]}", cited as "{title}"')
                else:
                    outcome["valid citation"] += 1
            if not citations and needs_citation(unit):
                report("uncited statement", f"{slug}: {unit[:120]}")
            scope = citations or scope
            for anchor in ANCHOR.findall(unit):
                if not any(anchor in sources.get(source, {}).get(n, ("", ""))[1] for source, n, _ in scope):
                    report("anchor outside cited sections", f"{slug}: {anchor}")

    print(f"pages: {len(pages)}  ({', '.join(f'{k}: {v}' for k, v in sorted(Counter(p.parent.name for p in pages.values()).items()))})")
    print("summary:", dict(sorted(outcome.items())))
    print("\n".join(problems) if problems else "no problems")


if __name__ == "__main__":
    main()
