"""The `@inferlet` decorator: a function becomes a program whose manifest is
derived from its source, and one the server could not run is refused here."""

import pytest

from pie_client import Inferlet, inferlet
from pie_client.inferlet import hashed_version, parameter_schema

LIMIT = 3


def test_a_function_becomes_a_program_named_and_versioned_by_its_source():
    @inferlet
    async def token_count(prompt: str, repeat: int = 1):
        """How many tokens the model's tokenizer sees."""
        from inferlet import model

        return len(model.encode(prompt * repeat))

    assert isinstance(token_count, Inferlet)
    assert token_count.name == "token-count"
    assert token_count.entry == "token_count"
    assert token_count.version == hashed_version(token_count.source)
    assert token_count.program == f"token-count@{token_count.version}"
    assert token_count.source.startswith("async def token_count(prompt: str, repeat: int = 1):")
    assert token_count.description == "How many tokens the model's tokenizer sees."
    manifest = token_count.manifest_toml()
    assert 'language = "python"' in manifest
    assert 'entry = "token_count"' in manifest
    assert 'call = "kwargs"' in manifest
    assert 'prompt = {type = "string"}' in manifest
    assert 'repeat = {type = "int", optional = true, description = "default: 1"}' in manifest


def test_the_same_source_is_the_same_version_and_an_edit_is_a_new_one():
    def first():
        def f(x: int):
            return x

        return f

    def again():
        def f(x: int):
            return x

        return f

    def edited():
        def f(x: int):
            return x + 1

        return f

    assert inferlet(first()).version == inferlet(again()).version
    assert inferlet(first()).version != inferlet(edited()).version
    assert hashed_version("x") != hashed_version("y")
    assert all(part.isdigit() for part in hashed_version("x").split("."))


def test_options_override_what_is_derived():
    @inferlet(name="echo", version="1.2.0", description="Echoes.")
    def anything(x: str):
        """Not this docstring."""
        return x

    assert (anything.name, anything.version, anything.description) == ("echo", "1.2.0", "Echoes.")
    assert anything("hi") == "hi", "the function still runs locally"


def test_a_closure_over_a_local_is_refused_by_name():
    bound = 2
    with pytest.raises(ValueError, match="closes over bound"):

        @inferlet
        def scaled(x: int):
            return x * bound


def test_a_module_level_name_is_refused_by_name():
    with pytest.raises(ValueError, match="uses LIMIT without binding it"):

        @inferlet
        def capped(x: int):
            return min(x, LIMIT)


def test_a_name_bound_inside_the_function_is_not_an_unbound_one():
    @inferlet
    def uses_imports_and_builtins(items: str):
        import json

        parsed = json.loads(items)
        return sorted(len(str(p)) for p in parsed)

    assert uses_imports_and_builtins.name == "uses-imports-and-builtins"


def test_the_parameter_schema_reads_the_signature():
    def f(a: str, b: int, c: float = 0.5, d: bool = False, e=None, *args, **kwargs):
        pass

    assert parameter_schema(f) == {
        "a": {"type": "string"},
        "b": {"type": "int"},
        "c": {"type": "float", "optional": True, "description": "default: 0.5"},
        "d": {"type": "bool", "optional": True, "description": "default: False"},
        "e": {"type": "string", "optional": True, "description": "default: None"},
    }


def test_a_manifest_quotes_what_toml_needs_quoted():
    @inferlet(description='say "hi"\nthen stop')
    def quoted(x: str):
        return x

    assert 'description = "say \\"hi\\"\\nthen stop"' in quoted.manifest_toml()
