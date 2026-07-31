"""Shared machinery for the rigorous comparison against XGrammar.

One place for the things every benchmark in this package needs to agree on:
which tokenizer, which documents, how long something took, and how to say so
without overclaiming. The rules are deliberately inconvenient.

*Warm up before timing.* Triton compiles on first call and CUDA allocates on
first use. A measurement that includes either is a measurement of the compiler.

*Report distributions.* A mean hides the tail, and constrained decoding is a
tail-latency business: a step that occasionally synchronises with the device is
worse than one that is steadily slower.

*Say what could not be measured.* A benchmark that did not run is reported as
unanswered rather than dropped, because a missing row is indistinguishable from
a favourable one.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable, Sequence

RESULTS = Path("results")
CORPUS = RESULTS / "jsonschemabench-instances.json"
PERMUTED = RESULTS / "jsonschemabench-permuted.json"

# Tokenizers to sweep. Vocabulary size is the axis that matters for anything
# holding a per-token bitset, so the set spans an order of magnitude.
TOKENIZERS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3-8b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
}


@dataclass
class Distribution:
    """What a timing actually was, rather than what its mean was."""

    count: int
    p50: float
    p90: float
    p99: float
    maximum: float
    minimum: float
    mean: float

    @classmethod
    def of(cls, samples: Sequence[float]) -> Distribution:
        ordered = sorted(samples)
        if not ordered:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        def at(fraction: float) -> float:
            index = min(len(ordered) - 1, int(fraction * len(ordered)))
            return ordered[index]

        return cls(
            count=len(ordered),
            p50=at(0.50),
            p90=at(0.90),
            p99=at(0.99),
            maximum=ordered[-1],
            minimum=ordered[0],
            mean=statistics.fmean(ordered),
        )


@dataclass
class Answer:
    """One reviewer question, and what the measurement said about it."""

    question_id: str
    headline: str
    detail: dict[str, Any] = field(default_factory=dict)
    unanswered: str | None = None

    def render(self) -> str:
        if self.unanswered:
            return f"  {self.question_id}: UNANSWERED - {self.unanswered}"
        return f"  {self.question_id}: {self.headline}"


def load_corpus(path: Path = CORPUS) -> list[dict[str, str]]:
    return json.loads(path.read_text())["instances"]


def load_vocabulary(model: str) -> list[bytes]:
    """The byte string each token id stands for.

    Both engines have to be told the same thing about the tokenizer or the
    comparison measures the tokenizer rather than the grammar.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001 - a token that has no string form
            vocabulary.append(b"")
    return vocabulary


BYTE_VOCABULARY: list[bytes] = [bytes([value]) for value in range(256)]


def time_calls(
    call: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    synchronise: Callable[[], None] | None = None,
) -> Distribution:
    """Time one operation, having first paid for everything that is not it."""
    for _ in range(warmup):
        call()
    if synchronise:
        synchronise()

    samples: list[float] = []
    for _ in range(repeats):
        if synchronise:
            synchronise()
        started = time.perf_counter()
        call()
        if synchronise:
            synchronise()
        samples.append((time.perf_counter() - started) * 1e6)
    return Distribution.of(samples)


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def write_report(name: str, answers: Iterable[Answer], extra: dict[str, Any]) -> Path:
    answers = list(answers)
    payload = {
        "gpu": gpu_name(),
        "answers": [asdict(answer) for answer in answers],
        **extra,
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"rigor-{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    for answer in answers:
        print(answer.render())
    print(f"\nwritten to {path}")
    return path
