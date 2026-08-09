//! Load-time facts a declaration traces against.
//!
//! These are the `config.json`-derived values that the hand-written
//! `LlamaLikeForwardCfg` + `HfConfig` pair carries into the forward today,
//! reduced to what the *declaration* needs: everything here is resolved at
//! trace time and none of it survives into the traced form except as
//! constants and op choices.
//!
//! ## What is left here is the shared vocabulary
//!
//! [`NormPlacement`] and [`QkNorm`] are words more than one family is written
//! in. Every family's own facts moved to `crates/model`, beside that family's
//! declaration -- adding a fact to a model touches that model, and only it.

use serde::{Deserialize, Serialize};


/// Where each sub-layer's norm sits relative to the residual add.
///
/// `Pre` is the standard Llama shape: norm the residual stream *into* the
/// sub-layer, accumulate the sub-layer's projection straight back onto the
/// stream (the `beta=1` GEMM). `Post` is the OLMo-2/OLMo-3 shape: the
/// sub-layer reads the residual stream raw, the norm applies to the
/// sub-layer's OUTPUT, and only then does a separate residual add land it —
/// a genuinely different op ORDER, which is why it is a fact and not an
/// emitter choice. Mirrors the driver's `NormPlacement`
/// (`crates/driver-cuda/csrc/src/model/llama_like/llama_like.hpp`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
}

/// Which q/k-norm convention the checkpoint ships, if any.
///
/// Two conventions exist in the wild (the driver's `rmsnorm_qk` dispatch,
/// `llama_like.cpp`): *per-head* (qwen3, gemma-3 — weight shape
/// `[head_dim]`, each head's channels normalised independently) and
/// *global* (OLMo-2, OLMo-3 — weight shape `[heads * head_dim]`, ONE
/// RMSNorm over the flattened projection). They are different arithmetic —
/// the global form shares one scale across heads — so the tri-state is a
/// fact, and the traced ops differ: per-head traces `RmsnormPerHead`,
/// global traces a plain row `Rmsnorm` (which is exactly what the kernel
/// launches: `kernels::norm::rmsnorm_bf16` over `[N, heads * head_dim]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QkNorm {
    #[default]
    Off,
    PerHead,
    Global,
}


/// The SLIDING WINDOW layer `l` attends over, `-1` for none.
///
/// One reader for a fact every family's backend carries the same way,
/// because the shape of the list is the same everywhere: empty is "no
/// window", one element is a config's single `sliding_window` broadcast
/// to every layer, and a longer list is the per-layer array an
/// alternating architecture states (gemma-2, gemma-4, OLMo-3).
///
/// It lives here rather than on any one facts struct for the reason
/// `NormPlacement` and `QkNorm` do: more than one family is written in
/// these words. Eleven executor sites across four families derived this
/// from `fwd_cfg.per_layer_window_left` -- an array no statement
/// mentioned -- and the dispatch statements carry it now.
///
/// A list SHORTER than the layer count is not an error: the last entry
/// covers the tail, which is what the drivers' fallback meant.
pub fn window_left_at(list: &[i32], l: u32) -> i32 {
    match list.len() {
        0 => -1,
        n => list[(l as usize).min(n - 1)],
    }
}

/// The ROPE THETA layer `l` rotates at.
///
/// The same shape as [`window_left_at`] and for the same reason: most
/// architectures carry one value in config, and some (gemma-4) alternate
/// it per layer between their local and global attention. A driver that
/// read the single one from config was reading the wrong one for half of
/// gemma-4's layers, which is why this is a fact and not a cfg field.
///
/// An EMPTY list is a text that states no rotation, and the accessor
/// says so by answering zero rather than by guessing a default.
pub fn rope_theta_at(list: &[f32], l: u32) -> f32 {
    match list.len() {
        0 => 0.0,
        n => list[(l as usize).min(n - 1)],
    }
}
