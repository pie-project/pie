"""The Python language component: the one wasm every Python inferlet runs in.

A Python inferlet is its source, not a build. The host installs this
component once, and for each Python program it launches an instance of it
with the program's source folded into the launch input:

    {"__pie_script__": {"name": "beam-search@0.1.0", "file": "beam-search.py",
                        "source": "..."},
     "input": "<the caller's input, verbatim>"}

`run` unwraps that envelope, executes the source as a fresh module, and
calls its `main` with the parsed input as the one argument, awaiting the
result when it is awaitable, exactly as a compiled Python inferlet's
generated wrapper would. Nothing about the `inferlet` package or
the WIT world is different: the source sees the same `inferlet` the build
would have bundled, because this component bundles it.

The standard-library sweep at the top is deliberate. componentize-py
carries the modules that were imported when it snapshotted the interpreter,
so a module a program can import is one this file imported first; the sweep
imports every standard module that runs under WASI, and a program that
imports one pays nothing.
"""

import importlib
import inspect
import json
import linecache
import sys
import traceback
import types

_STDLIB_SKIP = {
    # Side effects at import: a browser, a printout, a display.
    "antigravity", "this", "turtle", "tkinter", "idlelib",
    # Dead weight or another platform's: the test suite, installers, docs.
    "test", "ensurepip", "venv", "lib2to3", "distutils", "pydoc_data",
    "curses", "msvcrt", "winreg", "winsound", "nt",
}


def _prewarm_stdlib() -> None:
    import pkgutil

    def wanted(name: str) -> bool:
        parts = name.split(".")
        return not any(part.startswith("_") or part in ("test", "tests") for part in parts) and (
            name not in _STDLIB_SKIP
        )

    for name in sorted(getattr(sys, "stdlib_module_names", ())):
        if not wanted(name):
            continue
        try:
            module = importlib.import_module(name)
        except BaseException:  # noqa: BLE001 -- absent under WASI; not offered
            continue
        # A package's submodules are separate files, and only an imported
        # file is carried: walk them (`http.client`, `xml.etree.ElementTree`).
        path = getattr(module, "__path__", None)
        if not path:
            continue
        for info in pkgutil.walk_packages(path, prefix=name + ".", onerror=lambda _: None):
            if not wanted(info.name):
                continue
            try:
                importlib.import_module(info.name)
            except BaseException:  # noqa: BLE001
                pass


_prewarm_stdlib()

# `componentize_py_types.Err` is the exception class componentize-py uses to
# encode the Err arm of a `result<T, E>`. Anything the program raises is
# re-raised as this so the host receives a clean WIT Err carrying the
# traceback instead of a wasm trap.
from componentize_py_types import Err as _WitErr
from wit_world import exports

import inferlet as _inferlet  # bundled: the source imports this by name

ENVELOPE_KEY = "__pie_script__"


def _parse_input(raw: str):
    """The caller's input as the program's `main` expects it: a dict when
    the input is JSON, else the raw string under `input`."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"input": raw}


def _load(script: dict) -> types.ModuleType:
    """Execute the program's source as a fresh module and return it."""
    source = script["source"]
    file = script.get("file") or "main.py"
    module = types.ModuleType("__inferlet__")
    module.__file__ = file
    # Tracebacks quote the program's own lines: register the source with
    # linecache under the file name `compile` is told.
    linecache.cache[file] = (len(source), None, source.splitlines(keepends=True), file)
    code = compile(source, file, "exec")
    exec(code, module.__dict__)
    return module


def _encode(result, fallback) -> str:
    """The `run` return: strings pass through, everything else is JSON."""
    if result is None:
        return fallback() or ""
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump_json") and callable(result.model_dump_json):
        return result.model_dump_json()
    return json.dumps(result, default=str)


class Run(exports.Run):
    async def run(self, input: str) -> str:
        try:
            outer = json.loads(input) if input else {}
        except json.JSONDecodeError:
            outer = None
        if not isinstance(outer, dict) or ENVELOPE_KEY not in outer:
            raise _WitErr(
                "the Python language component was launched without a program: the launch "
                f"input carries no {ENVELOPE_KEY!r} envelope"
            )
        script = outer[ENVELOPE_KEY]
        input_data = _parse_input(outer.get("input", ""))

        try:
            module = _load(script)
            fn = getattr(module, "main", None)
            if fn is None:
                raise _WitErr(
                    f"{script.get('name', 'the program')} defines no `main`; "
                    "a Python inferlet is a module with an `async def main(input: dict)`"
                )
            result = fn(input_data)
            if inspect.isawaitable(result):
                result = await result
            return _encode(result, _inferlet.get_return_value)
        except _WitErr:
            raise
        except BaseException as e:
            raise _WitErr(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
