
import pytest

from pie_client import Inferlet, inferlet
import blake3

LIMIT = 3


def test_a_function_becomes_a_program_named_by_it_and_keyed_by_its_source():
    @inferlet
    async def token_count(prompt: str, repeat: int = 1):
        """How many tokens the model's tokenizer sees."""
        from inferlet import model

        return len(model.encode(prompt * repeat))

    assert isinstance(token_count, Inferlet)
    assert token_count.name == "token-count"
    assert token_count.hash == blake3.blake3(token_count.source.encode()).hexdigest()
    assert token_count.program == f"token-count@{token_count.hash}"
    assert token_count.source.startswith("async def token_count(prompt: str, repeat: int = 1):")


def test_the_source_ends_with_a_main_that_spreads_the_input():
    @inferlet
    async def token_count(prompt: str, repeat: int = 1):
        return len(prompt) * repeat

    assert token_count.source.endswith(
        "\n\nasync def main(input):\n    return await token_count(**input)\n"
    )



def test_the_same_source_is_the_same_build_and_an_edit_is_a_new_one():
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

    assert inferlet(first()).program == inferlet(again()).program
    assert inferlet(first()).program != inferlet(edited()).program


def test_a_name_overrides_the_functions():
    @inferlet(name="echo")
    def anything(x: str):
        return x

    assert anything.name == "echo"
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
