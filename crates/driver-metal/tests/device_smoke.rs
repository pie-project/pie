//! The per-family decode SMOKES, retired with the layers they exercised.
//!
//! # What was here
//!
//! Six tests over `metal::Decoder` and the per-family steppers and binders:
//! one token end to end, the first step's taps against a host reference, a
//! paged prefill against a sequential decode, a two-lane fleet held
//! token-exact, a mixed-length fleet, and a thousand tokens without a wedge
//! or a NaN. Five more had already gone with the gpt-oss and gemma4 layers,
//! and two before that with llama's.
//!
//! # Why they could go
//!
//! Every one was gated on `PIE_METAL_SMOKE_CHECKPOINT` pointing at a
//! **qwen3.5-family** snapshot -- a generation whose config interleaves
//! linear attention this crate does not model. So they skipped in CI, they
//! skipped on any machine without that variable, and the path they drove is
//! not the path a deployment takes: `MetalDriver::launch` runs the generic
//! executor and only the generic executor.
//!
//! The refusal itself moved while this file sat here. It used to be a string
//! table (`model::text::serves("qwen3_5")` was false) and is a ROW's answer
//! now, asked through `model::binding::serves` -- which means the refusal is
//! the catalog's to state and not this driver's. Worth knowing before reading
//! the paragraph above as a promise: the row is what refuses, so if a
//! qwen3.5 row grows a Metal text these gates become reachable without a line
//! changing here.
//!
//! # What runs instead, unconditionally, on every `cargo test`
//!
//! - `device_text_fire` -- the whole metal text on the device, for the plain
//!   shape, a mixture, gpt-oss and gemma4's side network, plus a prefill lane.
//! - `device_real_weights` -- six gates against MLX over a REAL checkpoint,
//!   including `a_generation_agrees_with_mlx_token_for_token`, which runs a
//!   prefill and three decodes with real KV carryover and holds all four
//!   tokens to MLX exactly. That is a stronger claim than any smoke here
//!   made: they checked that fires happened, it checks what they computed.
//! - `device_checkpoint_names` -- every MLX snapshot on the machine, every
//!   name the text states resolved against the load plan.
//! - `text_conformance` -- eight structural checks over five texts.
//!
//! # Written down rather than lost
//!
//! Two claims no current test made:
//!
//!   1. **A thousand tokens.** `a_thousand_tokens_decode_without_a_wedge_or_
//!      a_nan` ran a long generation and watched for a hang or a NaN. The
//!      generation gate runs four tokens. Still open.
//!   2. ~~**Two requests in one fire, isolated from each other.**~~ **Closed**
//!      by `device_real_weights::a_request_prefills_the_same_way_beside_
//!      another_one`, which prefills a prompt alone and again beside a longer
//!      unrelated one and holds both distributions bit-identical. It needs no
//!      reference, and it checks the SECOND request as well as the first —
//!      attention is causal, so the first request is insensitive to a leak by
//!      construction and comparing only it would prove nothing.
//!
//!      Worth recording what this reached that the retired smokes could not.
//!      Every device gate in this crate ran either one request holding every
//!      token or one request PER token, a decode fleet. Several requests each
//!      holding several tokens — the shape a served frame takes whenever two
//!      prompts arrive together — was staged by nothing, so
//!      `stage_prefill_fleet` is a third staging shape and the gap was in the
//!      harness rather than in the driver.
//!
//! The first wants the generic executor, a small checkpoint, and time — not a
//! per-family layer.
