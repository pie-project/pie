"""Modular-cache inferlet in Python.

This mirrors the Rust version:
- Build a dependency-ordered module list.
- Open the longest saved module-prefix context snapshot.
- Append only modules that were not cached.
- Save snapshots after every new module.
"""

from inferlet import Context, Model, Sampler, runtime


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
    """Small deterministic FNV-1a hash.

    Python's built-in hash() is randomized between processes, so do not use it
    for persistent cache keys.
    """
    h = 1469598103934665603
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def prefix_key(modules):
    """Build the persistent snapshot name for a module prefix."""
    parts = ["modular-cache-v1"]
    for m in modules:
        parts.append(m["id"])
        parts.append(m.get("role", "user"))
        parts.append(m["text"])
        parts.extend(m.get("deps", []))
    return f"modular-cache-python/{stable_hash('|'.join(parts)):016x}"


def topo_sort(modules):
    """Dependency-order modules so deps come first."""
    by_id = {m["id"]: m for m in modules}
    visiting = set()
    visited = set()
    ordered = []

    def visit(mid):
        if mid in visited:
            return
        if mid in visiting:
            raise RuntimeError(f"cycle detected at module {mid}")
        if mid not in by_id:
            raise RuntimeError(f"missing module {mid}")

        visiting.add(mid)
        for dep in by_id[mid].get("deps", []):
            visit(dep)
        visiting.remove(mid)

        visited.add(mid)
        ordered.append(by_id[mid])

    for mid in sorted(by_id.keys()):
        visit(mid)

    return ordered


def open_longest_prefix(model, modules):
    """Return (ctx, prefix_len) for the longest saved prefix, or (None, 0)."""
    for length in range(len(modules), 0, -1):
        name = prefix_key(modules[:length])
        ctx = Context.open(model, name)
        if ctx is not None:
            return ctx, length
    return None, 0


async def main(input: dict) -> str:
    model = Model.load(runtime.models()[0])

    prompt = input.get(
        "prompt",
        "Explain modular KV caching for LLM serving in simple terms.",
    )
    max_tokens = int(input.get("max_tokens", 256))
    use_cache = bool(input.get("use_cache", True))
    save_cache = bool(input.get("save_cache", True))

    modules = input.get("modules") or default_modules(prompt)
    modules = topo_sort(modules)

    if use_cache:
        cached, resume_index = open_longest_prefix(model, modules)
        if cached is not None:
            print(f"cache_hit_modules={resume_index}")
            ctx = cached.fork()
        else:
            print("cache_miss")
            ctx = Context(model)
            resume_index = 0
    else:
        ctx = Context(model)
        resume_index = 0

    for i in range(resume_index, len(modules)):
        m = modules[i]

        if m.get("role", "user") == "system":
            ctx.system(m["text"])
        else:
            ctx.user(m["text"])

        await ctx.flush()

        if save_cache:
            name = prefix_key(modules[: i + 1])
            ctx.save(name)
            print(f"saved={name}")

    ctx.cue()
    return await ctx.generate(Sampler.argmax(), max_tokens=max_tokens).collect_text()
