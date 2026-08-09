//! The launch arithmetic: what a kernel's `[[thread_position_in_grid]]`
//! contract says its grid must be.
//!
//! **This is the half of `batch/dispatch.rs` that stays.** That file is on the
//! retirement list because it also builds a decode DAG out of a
//! `DecodeGeometry` — a model definition inside the driver. The shapes below
//! are the opposite kind of fact: they are the KERNEL's knowledge, read off
//! each shader's thread-position contract, and no generic executor can do
//! without them. Moving them here is what lets the DAG builder be deleted
//! without taking them along.
//!
//! Nothing here is a decision. When a caller needs one — which tile, whether
//! to batch — that reads `Tuning` and arrives as an argument. Each helper's
//! doc names the kernel whose contract it states, and several of those
//! sentences are load-bearing findings rather than description: [`qmv`]'s
//! round-up records the difference between computing every output and
//! silently dropping the last few.
//!
//! [`super::geometry`] is what turns a stated [`Rule`] into one of these.
//!
//! [`Rule`]: super::geometry::Rule

use crate::batch::QMM_BMS;

/// A dispatch's thread grid and threadgroup, in THREADS — the encoder calls
/// `dispatchThreads`, so a head count multiplies the threadgroup width
/// rather than standing alone. Writing it the other way launches `n_heads`
/// threads total, which is not an error the hardware reports: the kernel's
/// simd reductions just read lanes that were never dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Total threads per axis.
    pub grid: [u32; 3],
    /// Threads per threadgroup per axis.
    pub tg: [u32; 3],
}

/// `affine_qmv_fast` (every Qmv* kind): four outputs per simdgroup, two
/// simdgroups per threadgroup.
///
/// Rounded UP, and the story is the reason the round-up is load-bearing: a
/// truncating count drops every output past the last whole four — and at
/// `n < 4` it drops the dispatch entirely. The shared expert's gate is
/// `hidden -> ONE logit a token`: its grid was `{32, 0, 1}`, no threads
/// ran, its buffer kept the zeros it was allocated with, and every routed
/// token was combined under `sigmoid(0) = 0.5` instead of its own gate.
#[must_use]
pub fn qmv(n: u32) -> Launch {
    Launch {
        grid: [32, n.div_ceil(4), 1],
        tg: [32, 2, 1],
    }
}

/// `rms_single_row` (Rms/FinalRms/QNorm/KNorm): one threadgroup per row,
/// `row_size / 4` threads (N_READS = 4), rows stacked on grid.x.
///
/// Rounded up — the kernel guards its own tail, but a truncating count
/// silently drops the last partial group of four — and capped at the 1024
/// threads Metal allows a threadgroup to be.
#[must_use]
pub fn rms(row_size: u32, n_rows: u32) -> Launch {
    let t = row_size.div_ceil(4).min(1024);
    Launch {
        grid: [t * n_rows, 1, 1],
        tg: [t, 1, 1],
    }
}

/// `rope_neox_decode`: x = frequency index, y = head. In place, so it is
/// dispatched once for Q and once for K.
#[must_use]
pub fn rope(rotary_dims: u32, n_heads: u32) -> Launch {
    let half = rotary_dims / 2;
    Launch {
        grid: [half, n_heads, 1],
        tg: [half, 1, 1],
    }
}

/// `residual_add` (Residual/LayerOut): elementwise over `hidden`.
#[must_use]
pub fn residual(hidden: u32) -> Launch {
    Launch {
        grid: [hidden, 1, 1],
        tg: [256, 1, 1],
    }
}

/// `embed_gather_4bit`: one thread per output channel.
#[must_use]
pub fn embed(hidden: u32) -> Launch {
    Launch {
        grid: [hidden, 1, 1],
        tg: [256, 1, 1],
    }
}

/// `q_gate_split`: deinterleave the 2×-wide q projection into query and
/// gate; one thread per (channel, query head).
#[must_use]
pub fn q_split(head_dim: u32, n_q_heads: u32) -> Launch {
    Launch {
        grid: [head_dim, n_q_heads, 1],
        tg: [head_dim, 1, 1],
    }
}

/// `kv_append`: elementwise (head_dim, kv head) scatter into the ring.
#[must_use]
pub fn kv_append(head_dim: u32, n_kv_heads: u32) -> Launch {
    Launch {
        grid: [head_dim, n_kv_heads, 1],
        tg: [head_dim, 1, 1],
    }
}

/// `sdpa_vector_decode`: one 1024-thread threadgroup per query head.
///
/// The C++ had THREE names for this shape — qwen3.5's, gemma4's sliding
/// and gpt-oss's sink — two with byte-identical bodies and the third their
/// `rows == 1` case; the kernels header collapsed them and this port keeps
/// the one.
#[must_use]
pub fn sdpa(n_q_heads: u32) -> Launch {
    Launch {
        grid: [n_q_heads * 1024, 1, 1],
        tg: [1024, 1, 1],
    }
}

/// `attn_gate`: `attn *= sigmoid(gate)`, elementwise head-major.
#[must_use]
pub fn attn_gate(n_q_heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [n_q_heads * head_dim, 1, 1],
        tg: [256, 1, 1],
    }
}

/// `gated_rms` (the golden `gdn_core` tap): one threadgroup per value
/// head, `v_dim` lanes reducing cooperatively.
#[must_use]
pub fn gated_rms(v_heads: u32, v_dim: u32) -> Launch {
    Launch {
        grid: [v_dim, v_heads, 1],
        tg: [v_dim, 1, 1],
    }
}

/// `silu_mul`: elementwise over the FFN intermediate.
#[must_use]
pub fn silu_mul(intermediate: u32) -> Launch {
    Launch {
        grid: [intermediate, 1, 1],
        tg: [256, 1, 1],
    }
}

/// The router's launch width: one lane per expert, rounded up to a whole
/// simdgroup — the kernel reduces ACROSS simdgroups and a partial one would
/// leave a reduction slot uninitialised. Clamped to the kernel's 1024-lane
/// cap first, which is the same answer as clamping after.
#[must_use]
pub fn router_lane_width(n_experts: u32) -> u32 {
    n_experts.clamp(1, 1024).div_ceil(32) * 32
}

/// `moe_route` top-k: every expert a lane, one row per grid.y.
#[must_use]
pub fn router_topk(n_experts: u32) -> Launch {
    let w = router_lane_width(n_experts);
    Launch {
        grid: [w, 1, 1],
        tg: [w, 1, 1],
    }
}

/// `moe_route_sort`: one threadgroup, sized to the expert count it scans.
#[must_use]
pub fn route_sort(n_experts: u32) -> Launch {
    let w = router_lane_width(n_experts);
    Launch {
        grid: [w, 1, 1],
        tg: [w, 1, 1],
    }
}

/// `route_rows` (gather/scatter/combine over sorted rows): one thread per
/// (channel, row).
#[must_use]
pub fn route_rows(width: u32, rows: u32) -> Launch {
    let w = width.max(1);
    Launch {
        grid: [w, rows.max(1), 1],
        tg: [w.min(256), 1, 1],
    }
}

/// The routed matvec: the dense [`qmv`] row decomposition — same kernel
/// body, a threadgroup owns EIGHT output rows across two simdgroups — with
/// two axes the dense shape does not have: the token row on x and the
/// expert slot on z. They are NOT interchangeable: the kernel selects its
/// expert with `sel = row * slots_per_row + slot`, so folding rows into
/// the slot axis routes every row through row 0's experts.
#[must_use]
pub fn routed_qmv(n: u32, experts_per_token: u32, rows: u32) -> Launch {
    Launch {
        grid: [
            32 * rows.max(1),
            n.max(1).div_ceil(4),
            experts_per_token.max(1),
        ],
        tg: [32, 2, 1],
    }
}

// ── The batched shapes. ──
//
// Same kind of fact as everything above, and they were the other half of
// the launch vocabulary living beside a DAG builder. `model::geometry`
// proves each of these is its M=1 sibling's generalisation, so the pair
// belongs in one module rather than either side of a retirement.

/// The widest row rung at or under `n` rows.
#[must_use]
pub fn qmm_bm(n: u32) -> u32 {
    let mut best = QMM_BMS[0];
    for &bm in &QMM_BMS {
        if n >= bm {
            best = bm;
        }
    }
    best
}

/// `rms_single_row` over `n_rows × n` stacked rows.
#[must_use]
pub fn rms_mb(row_size: u32, n_rows: u32, n: u32) -> Launch {
    let t = row_size.div_ceil(4).min(1024);
    Launch {
        grid: [t * n_rows * n, 1, 1],
        tg: [t, 1, 1],
    }
}

/// Flat elementwise over `width × n`.
#[must_use]
pub fn elementwise_mb(width: u32, n: u32) -> Launch {
    Launch {
        grid: [width * n, 1, 1],
        tg: [256, 1, 1],
    }
}

/// The matvec with rows on the first grid axis.
#[must_use]
pub fn qmv_mb(out_vec: u32, n: u32) -> Launch {
    Launch {
        grid: [32 * n, out_vec.div_ceil(4), 1],
        tg: [32, 2, 1],
    }
}

/// The GEMM grid for `n` rows at a `(bm, bn)` tile.
#[must_use]
pub fn qmm_t(out_vec: u32, n: u32, bn: u32, bm: u32) -> Launch {
    Launch {
        // Exact division on BOTH axes, and the caller guarantees it:
        // `Rule::Qmm` refuses a row count the tile does not divide
        // (`Ungeometric::PartialTile`) because the shader has no `M` argument
        // and reads the row count from the grid.
        grid: [32 * (out_vec / bn), 2 * (n / bm.max(1)), 2],
        tg: [32, 2, 2],
    }
}
