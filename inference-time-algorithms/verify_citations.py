#!/usr/bin/env python3
"""Cross-check every arXiv citation in a report against the arXiv API.

Usage: verify_citations.py FILE.md [FILE.md ...]

Parses `- **Title:** ...` / `- **arXiv:** [ID](url)` entry pairs, fetches the
registered title for each ID, and reports mismatches. Exists because the
literature reports in this directory are agent-generated and citation
fabrication is the main failure mode we care about.
"""
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def norm(s: str) -> str:
    s = s.replace("$_2$", "2").replace("₂", "2")
    # Headings often carry a parenthesised acronym: "Representation Engineering
    # (RepE): A Top-Down Approach" vs the registered title without it.
    s = re.sub(r"\(([A-Za-z][A-Za-z0-9\-]{1,12})\)", " ", s)
    s = s.lower()
    for long, short in (
        ("large language models", "llms"),
        ("large language model", "llm"),
    ):
        s = s.replace(long, short)
    return re.sub(r"[^a-z0-9]", "", s)


def fetch_titles(ids: list[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for i in range(0, len(ids), 25):
        chunk = ids[i : i + 25]
        url = (
            "http://export.arxiv.org/api/query?id_list="
            + ",".join(chunk)
            + "&max_results=100"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                break
            except Exception as exc:  # transient arXiv API failures
                if attempt == 2:
                    raise
                print(f"  retry after {exc}", file=sys.stderr)
                time.sleep(5)
        for entry in ET.fromstring(data).findall("a:entry", NS):
            aid = entry.find("a:id", NS).text.split("/abs/")[-1].split("v")[0]
            titles[aid] = " ".join(entry.find("a:title", NS).text.split())
        time.sleep(3)
    return titles


def extract_entries(text: str) -> list[tuple[str, str]]:
    """Return (claimed_title, arxiv_id) pairs.

    Reports come from different agents and use two layouts:
      - `- **Title:** X` ... `- **arXiv:** [ID](url)`
      - `#### 3. X` ... `- **arXiv:** [ID](url)`
    For each arXiv line we take the nearest preceding explicit Title field,
    falling back to the nearest preceding heading.
    """
    lines = text.splitlines()
    entries: list[tuple[str, str]] = []
    title_re = re.compile(r"^-?\s*\*\*Title:\*\*\s*(.+?)\s*$")
    head_re = re.compile(r"^#{2,5}\s+(?:[A-Z]?\d+(?:\.\d+)*[.)]?\s*)?(.+?)\s*$")
    # `- **arXiv:** [ID](url)`, `**arXiv:** [ID](url)`, `| **arXiv** | [ID](url) |`
    arxiv_re = re.compile(r"^\|?\s*-?\s*\*\*arXiv:?\*\*:?\s*\|?\s*\[(\d{4}\.\d{4,5})\]")
    for i, line in enumerate(lines):
        m = arxiv_re.match(line.strip())
        if not m:
            continue
        claimed = None
        for j in range(i - 1, max(-1, i - 15), -1):
            stripped = lines[j].strip()
            t = title_re.match(stripped)
            if t:
                claimed = t.group(1)
                break
            h = head_re.match(stripped)
            if h:
                claimed = h.group(1)
                # Headings like `2.1 Ji et al. — "A Survey of ..."` carry the
                # real title in quotes; prefer that over the whole heading.
                q = re.search(r'["“](.+?)["”]', claimed)
                if q:
                    claimed = q.group(1)
                break
        if claimed:
            entries.append((claimed.strip().rstrip(":"), m.group(1)))
    return entries


def check_link_consistency(path: str) -> int:
    """Flag `[ID_A](https://arxiv.org/abs/ID_B)` where the label and the URL
    disagree — a real typo class that title-matching alone would miss."""
    bad = 0
    for label, target in re.findall(
        r"\[([0-9a-zA-Z./\-]+)\]\(https://arxiv\.org/abs/([0-9a-zA-Z./\-]+)\)",
        open(path).read(),
    ):
        if label.replace("arXiv:", "").strip() != target:
            print(f"  LINK-MISMATCH label={label} url={target}")
            bad += 1
    return bad


def check(path: str) -> int:
    text = open(path).read()
    entries = extract_entries(text)
    bad = check_link_consistency(path)
    if not entries:
        print(f"{path}: no parseable entries ({bad} link issues)")
        return bad
    titles = fetch_titles([aid for _, aid in entries])
    for claimed, aid in entries:
        real = titles.get(aid)
        # Headings often carry a method nickname prefix ("PICARD: ...") or a
        # trailing gloss, so accept a substring match either way.
        if real is None:
            print(f"  MISSING  {aid}  claimed={claimed}")
            bad += 1
        elif norm(claimed) not in norm(real) and norm(real) not in norm(claimed):
            print(f"  MISMATCH {aid}\n     claimed: {claimed}\n     actual : {real}")
            bad += 1
    print(f"{path}: {len(entries)} citations, {bad} suspicious")
    return bad


if __name__ == "__main__":
    total = sum(check(p) for p in sys.argv[1:])
    sys.exit(1 if total else 0)
