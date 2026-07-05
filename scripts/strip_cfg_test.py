#!/usr/bin/env python3
"""Remove `#[cfg(test)]`-guarded items (mod/fn/impl/use/...) from Rust files.

Brace matching ignores braces inside line/block comments, `"..."` strings, and
`'{'`/`'}'` char literals, so mid-file test modules and format strings are
handled. Prints a per-file net brace balance of the RESULT (should be 0).
"""
import sys


def _skip_to_close_brace(text, i):
    """`i` points at (or before) the item; find first '{' then its match.
    Returns index just past the matching '}'. Comment/string aware."""
    n = len(text)
    depth = 0
    seen = False
    while i < n:
        c = text[i]
        c2 = text[i + 1] if i + 1 < n else ''
        if c == '/' and c2 == '/':
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '/' and c2 == '*':
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'" and text[i:i + 3] in ("'{'", "'}'"):
            i += 3
            continue
        if c == '{':
            depth += 1
            seen = True
        elif c == '}':
            depth -= 1
            if seen and depth == 0:
                return i + 1
        i += 1
    return n


def _net_braces(text):
    n = len(text)
    depth = 0
    i = 0
    while i < n:
        c = text[i]
        c2 = text[i + 1] if i + 1 < n else ''
        if c == '/' and c2 == '/':
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '/' and c2 == '*':
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'" and text[i:i + 3] in ("'{'", "'}'"):
            i += 3
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    return depth


def strip_file(path):
    src = open(path).read()
    out = []
    i = 0
    n = len(src)
    removed = 0
    while i < n:
        # detect a `#[cfg(test)]` attribute at a line start
        line_start = (i == 0) or (src[i - 1] == '\n')
        if line_start and src[i:].lstrip(' \t').startswith('#[cfg(test)]'):
            # find the item: skip this attr line and any following attr/blank lines
            j = src.find('\n', i)
            j = n if j < 0 else j + 1
            while j < n:
                seg = src[j:src.find('\n', j) if src.find('\n', j) >= 0 else n]
                s = seg.strip()
                if s == '' or s.startswith('#['):
                    j = src.find('\n', j)
                    j = n if j < 0 else j + 1
                    continue
                break
            # j is at the item line
            item_seg = src[j:src.find('\n', j) if src.find('\n', j) >= 0 else n]
            if '{' in item_seg or 'mod ' in item_seg or 'impl' in item_seg or 'fn ' in item_seg:
                end = _skip_to_close_brace(src, j)
            else:
                # single-line item ending with ';'
                semi = src.find(';', j)
                end = (semi + 1) if semi >= 0 else (src.find('\n', j) + 1)
            # also consume one trailing newline
            if end < n and src[end] == '\n':
                end += 1
            removed += src[i:end].count('\n')
            i = end
            continue
        out.append(src[i])
        i += 1
    result = ''.join(out)
    # collapse 3+ blank lines to 1
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    open(path, 'w').write(result)
    return removed, _net_braces(result)


if __name__ == '__main__':
    total = 0
    for p in sys.argv[1:]:
        rem, bal = strip_file(p)
        total += rem
        flag = '' if bal == 0 else '  <<< BRACE IMBALANCE'
        print(f"{rem:6d} lines removed  balance={bal:+d}  {p}{flag}")
    print(f"--- total removed: {total} ---")
