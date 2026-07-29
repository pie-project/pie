from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[bytes, ...]
    eos_token_id: int = 0
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("vocabulary cannot be empty")
        if self.tokens[self.eos_token_id] != b"":
            raise ValueError("the EOS token must use an empty byte sequence")

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def max_token_bytes(self) -> int:
        return max(map(len, self.tokens))

    def padded_bytes(self) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray([len(token) for token in self.tokens], dtype=np.int32)
        padded = np.zeros((self.size, self.max_token_bytes), dtype=np.uint8)
        for token_id, token in enumerate(self.tokens):
            if token:
                padded[token_id, : len(token)] = np.frombuffer(token, dtype=np.uint8)
        return padded, lengths

    @classmethod
    def synthetic(
        cls,
        size: int,
        schemas: Iterable[Mapping[str, Any]] = (),
        seed: int = 0,
        max_token_bytes: int = 12,
    ) -> Vocabulary:
        if size < 257:
            raise ValueError("synthetic vocabulary needs EOS plus 256 byte tokens")

        ordered: list[bytes] = [b""]
        seen = {b""}

        def add(token: bytes) -> None:
            if token not in seen and 0 < len(token) <= max_token_bytes:
                seen.add(token)
                ordered.append(token)

        for value in range(256):
            add(bytes([value]))

        common = [
            b'{"',
            b'":',
            b'","',
            b'":[',
            b'":{"',
            b"}",
            b"]",
            b"},",
            b"],",
            b"true",
            b"false",
            b"null",
            b"[]",
            b"{}",
            b'"',
            b",",
            b":",
            b"-1",
            b"0",
            b"1",
            b"10",
            b"100",
        ]
        for token in common:
            add(token)

        literals = sorted(_collect_schema_literals(schemas))
        for literal in literals:
            add(literal)
            for width in range(2, min(max_token_bytes, len(literal)) + 1):
                for start in range(0, len(literal) - width + 1):
                    add(literal[start : start + width])
                    if len(ordered) >= size:
                        return cls(tuple(ordered[:size]), name=f"synthetic-{size}")

        rng = random.Random(seed)
        alphabet = (
            b'{}[]":,'
            b"abcdefghijklmnopqrstuvwxyz"
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            b"0123456789"
            b" _-./"
        )
        while len(ordered) < size:
            length = rng.randint(2, max_token_bytes)
            token = bytes(rng.choice(alphabet) for _ in range(length))
            add(token)
        return cls(tuple(ordered), name=f"synthetic-{size}")

    @classmethod
    def tiktoken(
        cls,
        encoding_name: str = "gpt2",
        size: int | None = None,
    ) -> Vocabulary:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "install gpu-lr1[tokenizers] to use a tiktoken vocabulary"
            ) from exc

        encoding = tiktoken.get_encoding(encoding_name)
        target_size = size or encoding.n_vocab + 1
        tokens: list[bytes] = [b""]
        token_id = 0
        while len(tokens) < target_size and token_id < encoding.n_vocab:
            try:
                tokens.append(encoding.decode_single_token_bytes(token_id))
            except KeyError:
                pass
            token_id += 1
        if len(tokens) < target_size:
            raise ValueError(
                f"encoding {encoding_name} only supplied {len(tokens) - 1} tokens"
            )
        return cls(
            tuple(tokens),
            eos_token_id=0,
            name=f"tiktoken-{encoding_name}-{target_size}",
        )

    @classmethod
    def huggingface(
        cls,
        model_name: str,
        *,
        size: int | None = None,
        trust_remote_code: bool = False,
    ) -> Vocabulary:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "install gpu-lr1[tokenizers] to use a Hugging Face tokenizer"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        decoder = getattr(tokenizer, "decoder", None)
        if decoder is None and hasattr(tokenizer, "_tokenizer"):
            decoder = tokenizer._tokenizer.decoder
        if decoder is None or "ByteLevel" not in type(decoder).__name__:
            raise ValueError(
                "Hugging Face vocabulary extraction currently requires a "
                "byte-level tokenizer decoder"
            )
        total_size = len(tokenizer)
        target_size = size or total_size
        if target_size > total_size:
            raise ValueError(
                f"tokenizer {model_name} only supplies {total_size} token IDs"
            )
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None or eos_token_id >= target_size:
            raise ValueError(
                "requested vocabulary must include the tokenizer EOS token"
            )

        byte_decoder = {
            character: byte for byte, character in _bytes_to_unicode().items()
        }
        tokens: list[bytes] = []
        for token_id in range(target_size):
            if token_id == eos_token_id:
                tokens.append(b"")
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                tokens.append(f"<invalid-token-{token_id}>".encode("ascii"))
                continue
            try:
                tokens.append(bytes(byte_decoder[character] for character in token))
            except KeyError:
                tokens.append(token.encode("utf-8"))
        return cls(
            tuple(tokens),
            eos_token_id=eos_token_id,
            name=f"huggingface-{model_name.replace('/', '-')}-{target_size}",
        )


def _collect_schema_literals(
    schemas: Iterable[Mapping[str, Any]],
) -> set[bytes]:
    literals: set[bytes] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                for name in properties:
                    literals.add(json.dumps(name, ensure_ascii=True).encode("ascii"))
            if "const" in value:
                literals.add(
                    json.dumps(
                        value["const"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                )
            enum = value.get("enum")
            if isinstance(enum, list):
                for item in enum:
                    literals.add(
                        json.dumps(
                            item,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("ascii")
                    )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for schema in schemas:
        visit(schema)
    return literals


def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values), strict=True))
