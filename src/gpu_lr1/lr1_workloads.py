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


def balanced_grammar(name: str = "balanced") -> Grammar:
    return Grammar.from_rules(
        name,
        "S",
        {
            "S": [("(", "S", ")", "S"), ()],
        },
    )


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
    balanced = balanced_grammar()
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
