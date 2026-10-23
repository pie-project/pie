use serde::{Deserialize, Serialize};

use crate::Op;
use crate::value::ValueId;

/// Recurrent sequence mixers: each reads and advances a per-request state
/// held in a cache. A `*Chunked` op walks a prompt's rows through the state;
/// the plain one takes each row as a request's one step. They are separate
/// ops, not a flag, because they read their rows differently (a chunked scan
/// reads its lanes' runs) and kernels key that by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "ssm")]
pub enum Ssm {
    /// A depthwise causal conv over the last `conv_width` rows (the state
    /// holds the earlier ones), then SiLU.
    CausalConv1d {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        #[input(CACHE)]
        state: ValueId,
        /// A bias a channel, added before the activation (Mamba-2's).
        #[input]
        bias: Option<ValueId>,
        conv_width: u32,
        dilation: u32,
        #[output]
        y: ValueId,
    },
    CausalConv1dChunked {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        #[input(CACHE)]
        state: ValueId,
        /// A bias a channel, added before the activation (Mamba-2's).
        #[input]
        bias: Option<ValueId>,
        conv_width: u32,
        dilation: u32,
        #[output]
        y: ValueId,
    },
    /// A depthwise causal conv with no activation (LFM2's short conv).
    ShortConv {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        #[input(CACHE)]
        state: ValueId,
        conv_width: u32,
        #[output]
        y: ValueId,
    },
    ShortConvChunked {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        #[input(CACHE)]
        state: ValueId,
        conv_width: u32,
        #[output]
        y: ValueId,
    },
    /// A conv whose taps are computed a row (`coeff`) around a `base`
    /// kernel, over blocks of `side` rows.
    DynConv {
        #[input]
        x: ValueId,
        #[input]
        coeff: ValueId,
        #[input]
        base: ValueId,
        side: u32,
        taps: u32,
        group: u32,
        #[output]
        y: ValueId,
    },
    /// Gated DeltaNet's decay and write gates from the `b`/`a` projection.
    GdnGates {
        #[input]
        ba: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        a_log: ValueId,
        #[output(F32)]
        gates: ValueId,
    },
    GatedDelta {
        #[input]
        qkv: ValueId,
        #[input]
        z: ValueId,
        #[input]
        gates: ValueId,
        #[input(CACHE)]
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        #[output]
        y: ValueId,
    },
    GatedDeltaChunked {
        #[input]
        qkv: ValueId,
        #[input]
        z: ValueId,
        #[input]
        gates: ValueId,
        #[input(CACHE)]
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        #[output]
        y: ValueId,
    },
    /// Kimi delta attention.
    Kda {
        #[input]
        mixed: ValueId,
        #[input]
        f: ValueId,
        #[input]
        b: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        a_log: ValueId,
        #[input(CACHE)]
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        gate_floor: f32,
        #[output]
        y: ValueId,
    },
    KdaChunked {
        #[input]
        mixed: ValueId,
        #[input]
        f: ValueId,
        #[input]
        b: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        a_log: ValueId,
        #[input(CACHE)]
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        gate_floor: f32,
        #[output]
        y: ValueId,
    },
    /// Mamba-2's selective scan over the convolved rows `xbc` (`x`, then
    /// `groups` of `B`, then of `C`, each `state_dim` wide): per head, `dt =
    /// softplus(dt + dt_bias)`, `S = exp(-exp(a_log) dt) S + dt x B^T`, and
    /// `(C S + d x) * silu(z)`, RMS-normed a group of heads and scaled by
    /// `norm`, landed bf16.
    Mamba2 {
        #[input]
        xbc: ValueId,
        #[input]
        dt: ValueId,
        #[input]
        z: ValueId,
        #[input]
        a_log: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        d: ValueId,
        #[input]
        norm: ValueId,
        eps: f32,
        #[input(CACHE)]
        state: ValueId,
        heads: u32,
        groups: u32,
        state_dim: u32,
        #[output]
        y: ValueId,
    },
    Mamba2Chunked {
        #[input]
        xbc: ValueId,
        #[input]
        dt: ValueId,
        #[input]
        z: ValueId,
        #[input]
        a_log: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        d: ValueId,
        #[input]
        norm: ValueId,
        eps: f32,
        #[input(CACHE)]
        state: ValueId,
        heads: u32,
        groups: u32,
        state_dim: u32,
        #[output]
        y: ValueId,
    },
}
