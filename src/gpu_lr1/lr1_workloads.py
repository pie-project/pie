from __future__ import annotations

from dataclasses import dataclass

from gpu_lr1.lr1 import Grammar


@dataclass(frozen=True)
class NamedLR1Workload:
    name: str
    family: str
    grammar: Grammar
    prefix: tuple[str, ...]
    next_terminal: str
    stack_capacity: int


def arithmetic_grammar(name: str = "arithmetic") -> Grammar:
    return Grammar.from_rules(
        name,
        "E",
        {
            "E": [("E", "+", "T"), ("T",)],
            "T": [("T", "*", "F"), ("F",)],
            "F": [("(", "E", ")"), ("id",)],
        },
    )


def byte_arithmetic_grammar(name: str = "byte-arithmetic") -> Grammar:
    digits = tuple(str(value) for value in range(10))
    return Grammar.from_rules(
        name,
        "E",
        {
            "E": [("E", "+", "T"), ("T",)],
            "T": [("T", "*", "F"), ("F",)],
            "F": [("(", "E", ")"), ("N",)],
            "N": [("N", "D"), ("D",)],
            "D": [(digit,) for digit in digits],
        },
    )


def bounded_byte_arithmetic_grammar(
    max_parenthesis_depth: int,
    name: str = "bounded-byte-arithmetic",
) -> Grammar:
    if max_parenthesis_depth < 0:
        raise ValueError("maximum parenthesis depth must be non-negative")
    rules: dict[str, list[tuple[str, ...]]] = {
        "N": [("N", "D"), ("D",)],
        "D": [(str(value),) for value in range(10)],
    }
    for depth in range(max_parenthesis_depth + 1):
        expression = f"E{depth}"
        term = f"T{depth}"
        factor = f"F{depth}"
        rules[expression] = [
            (expression, "+", term),
            (term,),
        ]
        rules[term] = [
            (term, "*", factor),
            (factor,),
        ]
        rules[factor] = [("N",)]
        if depth < max_parenthesis_depth:
            rules[factor].append(("(", f"E{depth + 1}", ")"))
    return Grammar.from_rules(name, "E0", rules)


def balanced_grammar(name: str = "balanced") -> Grammar:
    return Grammar.from_rules(
        name,
        "S",
        {
            "S": [("(", "S", ")", "S"), ()],
        },
    )


def bounded_balanced_grammar(
    max_nesting_depth: int,
    name: str = "bounded-balanced",
) -> Grammar:
    if max_nesting_depth < 0:
        raise ValueError("maximum nesting depth must be non-negative")
    rules: dict[str, list[tuple[str, ...]]] = {}
    for depth in range(max_nesting_depth + 1):
        sequence = f"S{depth}"
        rules[sequence] = [(sequence, f"I{depth}"), ()]
        if depth < max_nesting_depth:
            rules[f"I{depth}"] = [("(", f"S{depth + 1}", ")")]
        else:
            rules[f"I{depth}"] = [("(", ")")]
    return Grammar.from_rules(name, "S0", rules)


def bounded_arithmetic_ebnf(max_parenthesis_depth: int) -> str:
    if max_parenthesis_depth < 0:
        raise ValueError("maximum parenthesis depth must be non-negative")
    rules = ["root ::= E0"]
    for depth in range(max_parenthesis_depth + 1):
        rules.append(f'E{depth} ::= T{depth} ("+" T{depth})*')
        rules.append(f'T{depth} ::= F{depth} ("*" F{depth})*')
        factor = "[0-9]+"
        if depth < max_parenthesis_depth:
            factor += f' | "(" E{depth + 1} ")"'
        rules.append(f"F{depth} ::= {factor}")
    return "\n".join(rules)


def bounded_balanced_ebnf(max_nesting_depth: int) -> str:
    if max_nesting_depth < 0:
        raise ValueError("maximum nesting depth must be non-negative")
    rules = ["root ::= S0"]
    for depth in range(max_nesting_depth + 1):
        item = (
            f'"(" S{depth + 1} ")"'
            if depth < max_nesting_depth
            else '"(" ")"'
        )
        rules.append(f"S{depth} ::= ({item})*")
    return "\n".join(rules)


def json_structure_grammar(name: str = "json-structure") -> Grammar:
    return Grammar.from_rules(
        name,
        "JSON",
        {
            "JSON": [("Value",)],
            "Value": [
                ("Object",),
                ("Array",),
                ("STRING",),
                ("NUMBER",),
                ("true",),
                ("false",),
                ("null",),
            ],
            "Object": [("{", "MembersOpt", "}")],
            "MembersOpt": [(), ("Members",)],
            "Members": [("Pair",), ("Members", ",", "Pair")],
            "Pair": [("STRING", ":", "Value")],
            "Array": [("[", "ElementsOpt", "]")],
            "ElementsOpt": [(), ("Elements",)],
            "Elements": [("Value",), ("Elements", ",", "Value")],
        },
    )


def reduction_chain_grammar(length: int) -> Grammar:
    if length <= 0:
        raise ValueError("reduction chain length must be positive")
    rules: dict[str, list[tuple[str, ...]]] = {
        f"N{index}": [(f"N{index + 1}",)]
        for index in range(length)
    }
    rules[f"N{length}"] = [("atom",)]
    return Grammar.from_rules(
        f"reduction-chain-{length}",
        "N0",
        rules,
    )


def wide_choice_grammar(width: int) -> Grammar:
    if width <= 0:
        raise ValueError("choice width must be positive")
    terminals = tuple(f"choice_{width}_{index}" for index in range(width))
    return Grammar.from_rules(
        f"wide-choice-{width}",
        "S",
        {"S": [(terminal,) for terminal in terminals]},
    )


def sequence_grammar(length: int) -> Grammar:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    terminals = tuple(f"seq_{length}_{index}" for index in range(length))
    return Grammar.from_rules(
        f"sequence-{length}",
        "S",
        {"S": [terminals]},
    )


def benchmark_lr1_workloads() -> list[NamedLR1Workload]:
    arithmetic = arithmetic_grammar()
    json_grammar = json_structure_grammar()

    workloads = [
        NamedLR1Workload(
            "arithmetic-reduce",
            "mixed",
            arithmetic,
            ("id",),
            "+",
            32,
        ),
        NamedLR1Workload(
            "json-object-close",
            "mixed",
            json_grammar,
            ("{", "STRING", ":", "NUMBER"),
            "}",
            64,
        ),
    ]

    for depth in (1, 8, 32, 128):
        workloads.append(
            NamedLR1Workload(
                f"balanced-depth-{depth}",
                "depth",
                balanced_grammar(f"balanced-depth-{depth}"),
                ("(",) * depth,
                ")",
                depth + 32,
            )
        )

    for length in (1, 4, 16, 64):
        workloads.append(
            NamedLR1Workload(
                f"reduction-chain-{length}",
                "reduction",
                reduction_chain_grammar(length),
                ("atom",),
                "$",
                length + 16,
            )
        )

    for width in (4, 16, 64, 256):
        grammar = wide_choice_grammar(width)
        workloads.append(
            NamedLR1Workload(
                f"wide-choice-{width}",
                "shift",
                grammar,
                (),
                f"choice_{width}_{width - 1}",
                8,
            )
        )

    for length in (8, 32, 128):
        grammar = sequence_grammar(length)
        prefix_length = max(0, length - 1)
        workloads.append(
            NamedLR1Workload(
                f"sequence-{length}",
                "shift",
                grammar,
                tuple(
                    f"seq_{length}_{index}" for index in range(prefix_length)
                ),
                f"seq_{length}_{length - 1}",
                length + 8,
            )
        )
    return workloads
