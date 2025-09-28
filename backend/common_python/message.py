"""
Message types for PIE backend communication.

This module defines all the request and response message structures used for
communication between the PIE backend and clients using msgspec.
"""

from typing import Optional

import msgspec


# ==============================================================================
# 1. DATA STRUCTURES (using msgspec.Struct and modern type hints)
# ==============================================================================


class HandshakeRequest(msgspec.Struct, gc=False):
    """Request message for handshake with version information."""

    version: str


class HandshakeResponse(msgspec.Struct, gc=False):
    """Response message containing model and tokenizer information."""

    version: str
    model_name: str
    model_traits: list[str]  # Use built-in list
    model_description: str
    prompt_template: str
    prompt_template_type: str
    prompt_stop_tokens: list[str]
    kv_page_size: int
    max_batch_tokens: int
    resources: dict[int, int]  # Use built-in list and tuple
    tokenizer_num_vocab: int
    tokenizer_merge_table: dict[int, bytes]
    tokenizer_special_tokens: dict[str, int]
    tokenizer_split_regex: str
    tokenizer_escape_non_printable: bool


class QueryRequest(msgspec.Struct, gc=False):
    """Request message for querying the model."""

    query: str


class QueryResponse(msgspec.Struct, gc=False):
    """Response message containing query result."""

    value: str


class HeartbeatRequest(msgspec.Struct, gc=False):
    """Request message for keepalive heartbeat."""


class HeartbeatResponse(msgspec.Struct, gc=False):
    """Response message for keepalive heartbeat."""


class ForwardPassRequest(msgspec.Struct, gc=False):
    """Request message for forward pass inference."""

    input_tokens: list[int]
    input_token_positions: list[int]
    input_embed_ptrs: list[int]
    input_embed_positions: list[int]
    adapter: Optional[int]
    adapter_seed: Optional[int]
    mask: list[list[int]]
    kv_page_ptrs: list[int] = msgspec.field(default_factory=list)
    kv_page_last_len: int = 0
    output_token_indices: list[int] = msgspec.field(default_factory=list)
    output_token_samplers: list[dict] = msgspec.field(default_factory=list)
    output_embed_ptrs: list[int] = msgspec.field(default_factory=list)
    output_embed_indices: list[int] = msgspec.field(default_factory=list)


class ForwardPassResponse(msgspec.Struct, gc=False):
    """Response message containing inference results."""

    tokens: list[int]
    dists: list[tuple[list[int], list[float]]]


class EmbedImageRequest(msgspec.Struct, gc=False):
    """Request message for image embedding."""

    embed_ptrs: list[int]
    image_blob: bytes
    position_offset: int


class InitializeAdapterRequest(msgspec.Struct, gc=False):
    """Request message for adapter initialization."""

    adapter_ptr: int
    rank: int
    alpha: float
    population_size: int
    mu_fraction: float
    initial_sigma: float


class UpdateAdapterRequest(msgspec.Struct, gc=False):
    """Request message for adapter updates."""

    adapter_ptr: int
    scores: list[float]
    seeds: list[int]
    max_sigma: float


class UploadAdapterRequest(msgspec.Struct, gc=False):
    """Request message for adapter upload."""

    adapter_ptr: int
    name: str
    adapter_data: bytes


class DownloadAdapterRequest(msgspec.Struct, gc=False):
    """Request message for adapter download."""

    adapter_ptr: int
    name: str


class DownloadAdapterResponse(msgspec.Struct, gc=False):
    """Response message containing adapter data."""

    adapter_data: bytes
