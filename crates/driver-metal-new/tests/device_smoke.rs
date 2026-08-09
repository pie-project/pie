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
//! **qwen3.5-family** snapshot -- an architecture no Metal text serves
//! (`model::text::serves("qwen3_5")` is false, and its config interleaves
//! linear attention this crate does not model). So they skipped in CI, they
//! skipped on any machine without that variable, and the path they drove is
//! not the path a deployment takes: `MetalDriver::launch` runs the generic
//! executor and only the generic executor.
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
//! Two claims no current test makes:
//!
//!   1. **A thousand tokens.** `a_thousand_tokens_decode_without_a_wedge_or_
//!      a_nan` ran a long generation and watched for a hang or a NaN. The
//!      generation gate runs four tokens.
//!   2. **Two requests in one fire, isolated from each other.** Every current
//!      device gate runs one request.
//!
//! Both want the generic executor, a small checkpoint, and time -- not a
//! per-family layer.
