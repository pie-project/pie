"""Driver → runtime handshake contract.

Pie's configuration has two distinct concepts that this module separates:

  * **Budgets** (top-down, user-set in `ModelConfig` / `SchedulerConfig`):
    "you may use up to this much" — `gpu_mem_utilization`, `swap_budget_bytes`,
    `default_token_limit`, scheduler timing knobs, etc. These flow from the
    user's config file into the worker and are consumed there. They never
    come back.

  * **Capacities** (bottom-up, driver-computed in this dataclass): "given
    that budget plus the kernel's real ceiling, here is what I actually
    have" — number of KV pages, page size the attention impl supports,
    forward limits, model-architecture facts read from HF config. These
    flow from the worker back to the server via the `ready_queue` handshake
    and are then forwarded to the Rust runtime when applicable.

Each inference driver (`pie_driver`, `pie_driver_vllm`, future siblings)
implements `Engine.capabilities() -> DriverCapabilities` and the worker
delivers that struct as the ready signal payload. The server uses it as the
source of truth when constructing `pie_runtime.ModelConfig`.

Some fields are forwarded to Rust through existing `pie_runtime.ModelConfig` /
`DeviceConfig` kwargs; others are Python-only for now (used for validation
and logging at the server layer). See server._bootstrap for the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriverCapabilities:
    """Capacities the inference driver reports to pie's runtime.

    Frozen so the value can't drift after the handshake — once the worker
    publishes it, both Python and Rust treat it as immutable truth for the
    lifetime of that worker.
    """

    # ── KV cache ────────────────────────────────────────────────────────
    # Number of KV pages the driver allocated on this rank's GPU.
    # Forwarded to Rust as DeviceConfig.total_pages.
    total_pages: int

    # Block size in tokens that the chosen attention kernel actually uses.
    # The user's `ModelConfig.kv_page_size` is a *preference*; the driver
    # may resolve to a different value (e.g., FlashInfer requires 16/32/64).
    # Forwarded to Rust as ModelConfig.kv_page_size.
    kv_page_size: int

    # Pinned host-side KV slot count for swap-out. 0 = swap disabled.
    # Forwarded to Rust as DeviceConfig.cpu_pages.
    swap_pool_size: int

    # ── Forward limits ─────────────────────────────────────────────────
    # Clamped to what the driver's kernel actually supports. Dynamic
    # allocation drivers can use u32::MAX for sub-limits that do not apply.
    max_forward_tokens: int
    max_forward_requests: int
    max_page_refs: int
    max_logit_rows: int
    max_prob_rows: int
    max_custom_mask_bytes: int
    max_sampler_rows: int
    max_logprob_labels: int

    # ── Model-architecture facts ───────────────────────────────────────
    # First architecture string from the HF config (e.g. "LlamaForCausalLM",
    # "Qwen3ForCausalLM"). Forwarded to Rust as ModelConfig.arch_name.
    arch_name: str

    # Vocabulary size and max context length. Currently Python-only —
    # surfaced for telemetry and future Rust-side admission control.
    vocab_size: int
    max_model_len: int

    # Resolved activation dtype, never "auto". Currently Python-only.
    activation_dtype: str

    # ── Feature support ────────────────────────────────────────────────
    # Whether the driver implements user-supplied attention masks (i.e.,
    # respects `req.has_user_mask` from pie's runtime). False means the
    # driver computes plain causal attention regardless of any inferlet-
    # supplied mask. Inferlets that depend on non-causal patterns must be
    # routed to a driver that returns True (currently only `native`).
    # Defaults False so a driver that omits the field is treated as
    # mask-unaware. Python-only today; the Rust runtime gains an admission
    # hook in a follow-up.
    supports_user_attention_mask: bool = False

    # Whether the driver implements CMA-ES / LoRA adapter operations
    # (`init_adapter`, `update_adapter`, `load_adapter`, `save_adapter`,
    # plus per-batch noise injection at Q/K/V). False means the driver
    # raises NotImplementedError on adapter calls; inferlets that exercise
    # adapter paths must run on a driver that returns True (currently only
    # `native`). Defaults False so a driver that omits the field is treated
    # as adapter-unaware. Python-only today; the Rust runtime gains an
    # admission hook in a follow-up.
    supports_adapters: bool = False

    # ── Filesystem ─────────────────────────────────────────────────────
    # Local directory containing tokenizer.json + config.json. The Rust
    # runtime reads tokenizer.json from `<snapshot_dir>/tokenizer.json`.
    # Forwarded to Rust as part of ModelConfig.tokenizer_path.
    snapshot_dir: str = ""
