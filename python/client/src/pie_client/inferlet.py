"""A Python function as an inferlet: decorate it, hand it to `PieClient.run`.

    @inferlet
    async def beam_search(prompt: str, beams: int = 2, max_tokens: int = 16):
        import inferlet as pie
        ...
        return {"text": text}

    process = await client.run(beam_search, prompt="hi", beams=4)
    print(await process.result())

The function's *source* travels, not its bytecode: the server's Python
language component parses it itself, so the client's Python version does
not matter and tracebacks quote the function's own lines. What that asks of
the function is that it be self-contained -- everything it uses is imported
inside it or passed in as an argument. A closure over a local, or a use of
a module-level name, is refused here with the name, since the server would
only discover it as a NameError.

The manifest is derived: the name from the function's, the version from a
hash of the source (so an edit is a new program and an unchanged one is
never re-uploaded), the parameter schema from the signature's annotations
and defaults, the description from the docstring.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import symtable
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import blake3

_TYPE_NAMES = {str: "string", int: "int", float: "float", bool: "bool"}


@dataclass(frozen=True)
class Inferlet:
    """A function readied to run on pie: its source and the manifest that
    names it."""

    fn: Callable[..., Any]
    name: str
    version: str
    entry: str
    source: str
    description: str | None
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def program(self) -> str:
        """`name@version`, the id the server launches by."""
        return f"{self.name}@{self.version}"

    def manifest_toml(self) -> str:
        lines = [
            "[package]",
            f'name = "{self.name}"',
            f'version = "{self.version}"',
        ]
        if self.description:
            lines.append(f"description = {_toml_string(self.description)}")
        lines += [
            "",
            "[runtime]",
            'language = "python"',
            f'entry = "{self.entry}"',
            'call = "kwargs"',
        ]
        if self.parameters:
            lines += ["", "[parameters]"]
            for pname, spec in self.parameters.items():
                parts = [f'type = "{spec["type"]}"']
                if spec.get("optional"):
                    parts.append("optional = true")
                if spec.get("description"):
                    parts.append(f"description = {_toml_string(spec['description'])}")
                lines.append(f"{pname} = {{{', '.join(parts)}}}")
        return "\n".join(lines) + "\n"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The function still runs locally when called directly."""
        return self.fn(*args, **kwargs)


def inferlet(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    version: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark a function as an inferlet. Bare (`@inferlet`) or with options
    (`@inferlet(name="beam-search", version="1.2.0")`)."""

    def wrap(fn: Callable[..., Any]) -> Inferlet:
        source = function_source(fn)
        check_self_contained(fn, source)
        program_name = name or fn.__name__.replace("_", "-")
        doc = description if description is not None else _first_line(inspect.getdoc(fn))
        parameters = parameter_schema(fn)
        return Inferlet(
            fn=fn,
            name=program_name,
            version=version or hashed_version(source),
            entry=fn.__name__,
            source=source,
            description=doc,
            parameters=parameters,
        )

    return wrap if fn is None else wrap(fn)


def function_source(fn: Callable[..., Any]) -> str:
    """The function's definition with its decorators removed, dedented so
    it stands as a module of its own."""
    try:
        raw = inspect.getsource(fn)
    except (OSError, TypeError) as error:
        raise ValueError(
            f"cannot read the source of {fn.__name__}: an inferlet is sent as source, "
            "so it must be defined in a file or a notebook cell, not in a REPL or `exec`"
        ) from error
    dedented = textwrap.dedent(raw)
    tree = ast.parse(dedented)
    node = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if node is None:
        raise ValueError(f"{fn.__name__} is not a function definition")
    lines = dedented.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 :])


def check_self_contained(fn: Callable[..., Any], source: str) -> None:
    """Refuse a function the server could not run as it stands: one that
    closes over a local, or reaches for a name its own body never binds."""
    code = fn.__code__
    if code.co_freevars:
        names = ", ".join(code.co_freevars)
        raise ValueError(
            f"{fn.__name__} closes over {names}; an inferlet must be self-contained -- "
            "pass them as arguments instead"
        )
    table = symtable.symtable(source, fn.__name__, "exec")
    unbound: set[str] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        for sym in scope.get_symbols():
            if sym.is_global() and sym.is_referenced() and not hasattr(builtins, sym.get_name()):
                unbound.add(sym.get_name())
        for child in scope.get_children():
            visit(child)

    for scope in table.get_children():
        visit(scope)
    if unbound:
        names = ", ".join(sorted(unbound))
        raise ValueError(
            f"{fn.__name__} uses {names} without binding it; an inferlet must be "
            "self-contained -- import inside the function, or pass the value as an argument"
        )


def parameter_schema(fn: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    """`[parameters]` from the signature: annotated `str`/`int`/`float`/
    `bool` become typed entries, a default makes one optional, and anything
    else is passed through as a string."""
    schema: dict[str, dict[str, Any]] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        entry: dict[str, Any] = {"type": _TYPE_NAMES.get(param.annotation, "string")}
        if param.default is not inspect.Parameter.empty:
            entry["optional"] = True
            entry["description"] = f"default: {param.default!r}"
        schema[pname] = entry
    return schema


def hashed_version(source: str) -> str:
    """A semver-shaped version that is a function of the source, so the
    server's copy is content-addressed: `0.<16 bits>.<16 bits>`."""
    digest = blake3.blake3(source.encode("utf-8")).hexdigest()
    return f"0.{int(digest[:4], 16)}.{int(digest[4:8], 16)}"


def _first_line(doc: str | None) -> str | None:
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip() or None


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'
