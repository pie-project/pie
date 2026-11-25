"""Training and evaluation data, rendered to token ids.

A dataset is a JSONL file of chat examples (OpenAI's chat format) or of
preference pairs (OpenAI's preference format). Every example is rendered once,
here, with the model's own chat template, and the trainer, the evaluation and
pie's score program all read those token ids. pie never renders text again, so
the template cannot drift between training and serving.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .train import Batch as TrainBatch


@dataclass
class Batch(TrainBatch):
    """The trainer's batch, plus the sha256 of the file its rows came from."""

    data_sha256: str | None = None


def prompt(tokenizer, messages: list[dict]) -> list[int]:
    """The token ids of `messages` followed by the opening of the assistant's
    turn: what a model generates its answer from. Thinking is off."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def ends(tokenizer) -> set[int]:
    """The token ids that close a turn: the tokenizer's end of sequence and
    the end ids of the checkpoint's generation config (Gemma 4 closes a turn
    with `<turn|>`, not with its `<eos>`)."""
    ids = {tokenizer.eos_token_id}
    path = getattr(tokenizer, "name_or_path", "")
    config = Path(path) / "generation_config.json"
    if path and config.is_file():
        eos = json.loads(config.read_text()).get("eos_token_id", [])
        ids.update(eos if isinstance(eos, list) else [eos])
    return ids


def example(tokenizer, messages: list[dict], answer: str) -> tuple[list[int], list[float]]:
    """The token ids of `messages` answered by `answer`, ending with the token
    that closes the answer, and the loss weight of predicting each next token:
    1 for the answer's tokens (the closing token included), 0 elsewhere."""
    head = prompt(tokenizer, messages)
    full = messages + [{"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(full, tokenize=False, enable_thinking=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if tokens[: len(head)] != head:
        raise ValueError("the chat template renders the prompt differently once the answer follows it")
    closing = ends(tokenizer)
    end = next(i for i in range(len(head), len(tokens)) if tokens[i] in closing)
    tokens = tokens[: end + 1]
    weights = [0.0] * (len(head) - 1) + [1.0] * (end + 1 - len(head))
    return tokens, weights


@dataclass
class Dataset:
    """Examples read from one JSONL file, each rendered to one or more token rows.

    A chat example is one row; a preference pair is two adjacent rows, the
    preferred answer first.
    """

    examples: list[dict]
    """The examples as the file holds them."""
    tokens: list[list[int]]
    """One token row per answer, `group` rows per example."""
    weights: list[list[float]]
    """Per row, the loss weight of predicting each next token (one shorter than the row)."""
    prompts: list[list[dict]]
    """Per example, the messages its answers follow."""
    answers: list[str]
    """Per example, the reference answer (the preferred one for a pair)."""
    group: int
    """Rows per example: 1 for chat, 2 for preference pairs."""
    sha256: str
    """The sha256 of the file the examples were read from."""

    def __len__(self) -> int:
        return len(self.examples)

    def messages(self, i: int) -> list[dict]:
        """The messages example `i` answers, without its answer."""
        return self.prompts[i]

    def batches(self, rows: int = 8, epochs: int = 1, seed: int = 0):
        """Batches of `rows` rows, shuffled again every epoch; a pair's two rows
        stay together. The last batch of an epoch may be smaller."""
        if rows % self.group:
            raise ValueError(f"rows={rows} splits a preference pair; use a multiple of {self.group}")
        rng = random.Random(seed)
        per = rows // self.group
        for _ in range(epochs):
            order = list(range(len(self.examples)))
            rng.shuffle(order)
            for start in range(0, len(order), per):
                picked = [i * self.group + j for i in order[start : start + per] for j in range(self.group)]
                yield Batch(
                    tokens=[self.tokens[r] for r in picked],
                    weights=[self.weights[r] for r in picked],
                    data_sha256=self.sha256,
                )

    def sample(self, n: int, seed: int = 0) -> "Dataset":
        """`n` examples drawn without replacement, in file order."""
        keep = sorted(random.Random(seed).sample(range(len(self.examples)), n))
        rows = [i * self.group + j for i in keep for j in range(self.group)]
        return Dataset(
            examples=[self.examples[i] for i in keep],
            tokens=[self.tokens[r] for r in rows],
            weights=[self.weights[r] for r in rows],
            prompts=[self.prompts[i] for i in keep],
            answers=[self.answers[i] for i in keep],
            group=self.group,
            sha256=self.sha256,
        )


def _read(path: str | Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], hashlib.sha256(raw).hexdigest()


def chat(path: str | Path, model) -> Dataset:
    """A chat dataset: each line is `{"messages": [...]}` and its last message
    is the assistant's answer, the only part the loss reads."""
    examples, sha = _read(path)
    tokens, weights, prompts, answers = [], [], [], []
    for n, ex in enumerate(examples, 1):
        *head, last = ex["messages"]
        if last["role"] != "assistant":
            raise ValueError(f"{path}:{n}: the last message is not the assistant's answer")
        row, w = example(model.tokenizer, head, last["content"])
        tokens.append(row)
        weights.append(w)
        prompts.append(head)
        answers.append(last["content"])
    return Dataset(examples, tokens, weights, prompts, answers, group=1, sha256=sha)


def pairs(path: str | Path, model) -> Dataset:
    """A preference dataset: each line is OpenAI's preference format,
    `{"input": {"messages": [...]}, "preferred_output": [{...}], "non_preferred_output": [{...}]}`."""
    examples, sha = _read(path)
    tokens, weights, prompts, answers = [], [], [], []
    for ex in examples:
        head = ex["input"]["messages"]
        for side in ("preferred_output", "non_preferred_output"):
            row, w = example(model.tokenizer, head, ex[side][0]["content"])
            tokens.append(row)
            weights.append(w)
        prompts.append(head)
        answers.append(ex["preferred_output"][0]["content"])
    return Dataset(examples, tokens, weights, prompts, answers, group=2, sha256=sha)
