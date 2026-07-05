"""Modular-cache inferlet in Python.

This mirrors the Rust version:
- Build a dependency-ordered module list (rejecting dup ids, missing deps, cycles).
- Open the longest saved module-prefix context snapshot.
- Append only modules that were not cached.
- Save snapshots after every new module so any stable prefix can be reused.

Behavior:
  * first run with a module chain -> ``cache_miss``
  * identical re-run              -> ``cache_hit_modules=N`` (full reuse)
  * only the final task changed   -> reuse the stable earlier prefix
  * use_cache=false               -> never open a saved snapshot
  * save_cache=false              -> never save new snapshots
"""

from inferlet import Context, Sampler

# Bump when the snapshot layout / key meaning changes.
CACHE_SCHEMA = "modular-cache-v1"
# Snapshot namespace — keeps Python snapshots distinct from the Rust / JS ports.
CACHE_NS = "modular-cache-python"
# Key field separator (ASCII unit separator — won't show up in prompt text).
SEP = "\x1f"


def default_modules(prompt: str):
    return [
        {
            "id": "system/base",
            "role": "system",
            "deps": [],
            "text": "You are a concise technical assistant.",
        },
        {
            "id": "style/simple",
            "role": "user",
            "deps": ["system/base"],
            "text": "Explain simply first, then give implementation details.",
        },
        {
            "id": "context/pie",
            "role": "user",
            "deps": ["style/simple"],
            "text": "Pie inferlets can control forward passes, KV cache snapshots, and generation loops.",
        },
        {
            "id": "task/current",
            "role": "user",
            "deps": ["context/pie"],
            "text": prompt,
        },
    ]


def stable_hash(text: str) -> int:
    """FNV-1a 64-bit. Deterministic across runs and processes.

    Python's built-in ``hash()`` is randomized per-process (PYTHONHASHSEED),
    so it must never be used for persistent cache keys. The Rust and JS ports
    use the identical algorithm.
    """
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def prefix_key(modules) -> str:
    """Deterministic snapshot name for a module prefix.

    Folds in the schema version plus every module's id, role, text, and deps,
    so any change anywhere in the prefix changes the key (invalidation)."""
    parts = [CACHE_SCHEMA]
    for m in modules:
        parts.append(m["id"])
        parts.append(m.get("role", "user"))
        parts.append(m["text"])
        parts.extend(m.get("deps", []))
    return f"{CACHE_NS}/{stable_hash(SEP.join(parts)):016x}"


def topo_sort(modules):
    """Dependency-order modules so deps come first.

    Rejects duplicate ids, missing deps, and cycles with clean errors."""
    by_id = {}
    for m in modules:
        mid = m.get("id", "")
        if not str(mid).strip():
            raise ValueError("module id must not be empty")
        role = m.get("role", "user")
        if role not in ("system", "user"):
            raise ValueError(f"unsupported role '{role}' on module {mid}")
        if mid in by_id:
            raise ValueError(f"duplicate module id: {mid}")
        by_id[mid] = m

    visiting = set()
    visited = set()
    ordered = []

    def visit(mid):
        if mid in visited:
            return
        if mid in visiting:
            raise ValueError(f"dependency cycle at module {mid}")
        if mid not in by_id:
            raise ValueError(f"missing module {mid}")

        visiting.add(mid)
        for dep in by_id[mid].get("deps", []):
            visit(dep)
        visiting.remove(mid)

        visited.add(mid)
        ordered.append(by_id[mid])

    # Sorted start ids => deterministic order independent of input order.
    for mid in sorted(by_id.keys()):
        visit(mid)

    return ordered


def open_longest_prefix(modules):
    """Longest saved prefix as (ctx, prefix_len), or (None, 0).

    open() forks the snapshot (it stays immutable), so the context it returns is
    ours to append to. It *raises* on a missing snapshot rather than returning
    None, so a failed open just means a miss at that length.
    """
    for length in range(len(modules), 0, -1):
        name = prefix_key(modules[:length])
        try:
            ctx = Context.open(name)
        except Exception:
            ctx = None
        if ctx is not None:
            return ctx, length
    return None, 0


async def main(input: dict) -> str:
    prompt = input.get(
        "prompt",
        "Explain modular KV caching for LLM serving in simple terms.",
    )
    max_tokens = int(input.get("max_tokens", 256))
    use_cache = bool(input.get("use_cache", True))
    save_cache = bool(input.get("save_cache", True))

    modules = input.get("modules") or default_modules(prompt)
    modules = topo_sort(modules)

    print("--- modular-cache-python ---")
    print(f"modules={len(modules)}")
    print("order=" + " -> ".join(m["id"] for m in modules))
    print(f"use_cache={use_cache} save_cache={save_cache}")

    resume_index = 0
    if use_cache:
        cached, resume_index = open_longest_prefix(modules)
        if cached is not None:
            print(f"cache_hit_modules={resume_index}")
            ctx = cached
        else:
            print("cache_miss")
            ctx = Context()
            resume_index = 0
    else:
        print("cache_miss (use_cache=false)")
        ctx = Context()

    for i in range(resume_index, len(modules)):
        m = modules[i]

        if m.get("role", "user") == "system":
            ctx.system(m["text"])
        else:
            ctx.user(m["text"])

        await ctx.flush()

        if save_cache:
            name = prefix_key(modules[: i + 1])
            # best-effort: save() raises if an earlier run already saved this
            # exact prefix; a cache miss shouldn't abort generation.
            try:
                ctx.save(name)
                print(f"saved={name}")
            except Exception as e:
                print(f"save_skipped={name} ({e})")

    ctx.cue()
    return await ctx.generate(Sampler.argmax(), max_tokens=max_tokens).collect_text()
