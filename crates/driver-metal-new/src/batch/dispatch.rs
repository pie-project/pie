//! The decode DAG: one dispatch per kernel, in golden-surface order, with
//! the launch geometry each kernel's own contract dictates.
//!
//! # The launch shapes
//!
//! Every shape here was read off a kernel's `[[thread_position_in_grid]]`
//! contract — it is the KERNEL's knowledge, kept in the C++ under
//! `pie/kernels/*.h` beside the shaders. `kernels-metal`'s Rust side is
//! still a signature table, so the shapes live here until it grows a launch
//! module (the same note `sizing.rs` carries for `sorted_rows`); each
//! helper's doc names the kernel whose contract it states. Nothing here is
//! a decision: when a caller needs one — which tile, whether to batch —
//! that reads [`Tuning`] and arrives as an argument.
//!
//! # The DAG
//!
//! [`build_decode_dag`] emits the qwen3.6 per-token decode in the golden
//! kernel-surface order: full gated attention (2×-wide q_proj → QSplit,
//! AttnGate), the 4-way GDN in-projection, GdnCore → GatedRms, and the FFN
//! in one of two shapes. The routed and dense members of this family differ
//! in the mixture and in nothing else, which is why they are one family and
//! not two.
//!
//! Counts, for the default shape: 1 embed + 18 GDN layers × 15 + 6
//! full-attention layers × 20 + 2 tail = 393 dispatches, +1 with device
//! argmax.
//!
//! # Barriers
//!
//! A barrier follows every dispatch except inside an explicit concurrency
//! group ([`concurrent_run_ends`]): kinds that read only activations
//! produced before the group starts and write distinct scratch values.
//! Nothing in the scratch dataflow models KV pages or GDN recurrent state,
//! which is why the groups are an explicit list rather than a derivation
//! from the scratch schedule alone.

use crate::tuning::Tuning;

use super::abi::Kernel;
use super::geometry::DecodeGeometry;
use super::sizing::{RoutedProjection, moe_sorted_rows};

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

/// One dispatch of the decode DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// PSO lookup and dump tag.
    pub kind: Kernel,
    /// Flat position — the argument-table key (unique, token-stable).
    pub ordinal: u32,
    /// The model layer, or `None` for the pre/post-stack singletons.
    pub layer: Option<u32>,
    /// The launch geometry.
    pub launch: Launch,
    /// QmvO/QmvOut/QmvDown: add the block residual in the GEMV epilogue,
    /// dropping the following Residual/LayerOut dispatch.
    pub fuse_residual: bool,
    /// Output columns per threadgroup when this projection runs as the
    /// steel GEMM; 0 = the GEMV.
    pub qmm_bn: u32,
    /// K partitions; above 1 a reduce dispatch follows.
    pub qmm_split: u32,
    /// Rows per threadgroup for that GEMM: a wider block dequantizes each
    /// weight tile once for twice the rows, which only pays once the batch
    /// has threadgroups to spare.
    pub qmm_bm: u32,
}

/// What shape [`build_decode_dag`] builds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DagOptions {
    /// Append the device-argmax tail.
    pub with_argmax: bool,
    /// Fold each block's residual add into the closing GEMV.
    pub fuse_residual: bool,
    /// Split GdnCore into the prep dispatch and the slimmed recurrent core.
    pub gdn_prep: bool,
}

/// Build the per-token decode DAG for this geometry.
#[must_use]
pub fn build_decode_dag(g: &DecodeGeometry, tuning: &Tuning, options: DagOptions) -> Vec<Dispatch> {
    let q_dim = g.n_q_heads * g.head_dim;
    let qg_dim = 2 * q_dim; // q_proj is 2×-wide [query | gate]
    let kv_dim = g.n_kv_heads * g.head_dim;

    let mut dag: Vec<Dispatch> = Vec::new();
    let emit = |dag: &mut Vec<Dispatch>, kind: Kernel, layer: Option<u32>, launch: Launch| {
        let ordinal = u32::try_from(dag.len()).expect("a DAG is hundreds of dispatches");
        dag.push(Dispatch {
            kind,
            ordinal,
            layer,
            launch,
            fuse_residual: false,
            qmm_bn: 0,
            qmm_split: 1,
            qmm_bm: 16,
        });
    };
    let fuse_last = |dag: &mut Vec<Dispatch>| {
        dag.last_mut().expect("just emitted").fuse_residual = true;
    };

    // EMBED ×1. Tied, one table serves both ends of the model. Untied —
    // which is every routed member of this family — they are two tensors,
    // and a kind is a weight name, so they are two kinds: same kernel, same
    // launch, different tensor asked for.
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::EmbedGather
        } else {
            Kernel::EmbedUntied
        },
        None,
        embed(g.hidden),
    );

    for layer in 0..g.n_layers {
        let at = Some(layer);
        if g.is_full_attn(layer) {
            emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
            // q/k/v adjacent so they form one concurrent run: all three
            // read the block norm's output and write disjoint scratch.
            // QSplit consumes q_proj, so it follows the run rather than
            // splitting it in half.
            emit(&mut dag, Kernel::QmvQ, at, qmv(qg_dim));
            emit(&mut dag, Kernel::QmvK, at, qmv(kv_dim));
            emit(&mut dag, Kernel::QmvV, at, qmv(kv_dim));
            emit(
                &mut dag,
                Kernel::QSplit,
                at,
                q_split(g.head_dim, g.n_q_heads),
            );
            emit(&mut dag, Kernel::QNorm, at, rms(g.head_dim, g.n_q_heads));
            emit(&mut dag, Kernel::KNorm, at, rms(g.head_dim, g.n_kv_heads));
            emit(&mut dag, Kernel::Rope, at, rope(g.rotary_dims, g.n_q_heads));
            emit(
                &mut dag,
                Kernel::RopeK,
                at,
                rope(g.rotary_dims, g.n_kv_heads),
            );
            emit(
                &mut dag,
                Kernel::KvAppend,
                at,
                kv_append(g.head_dim, g.n_kv_heads),
            );
            emit(&mut dag, Kernel::Sdpa, at, sdpa(g.n_q_heads));
            emit(
                &mut dag,
                Kernel::AttnGate,
                at,
                attn_gate(g.n_q_heads, g.head_dim),
            );
            emit(&mut dag, Kernel::QmvO, at, qmv(g.hidden));
            if options.fuse_residual {
                fuse_last(&mut dag);
            } else {
                emit(&mut dag, Kernel::Residual, at, residual(g.hidden));
            }
        } else {
            emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
            emit(&mut dag, Kernel::QmvIn, at, qmv(g.gdn_conv_dim));
            emit(&mut dag, Kernel::QmvInZ, at, qmv(g.gdn_v_total));
            emit(&mut dag, Kernel::GdnInA, at, qmv(g.gdn_v_heads));
            emit(&mut dag, Kernel::GdnInB, at, qmv(g.gdn_v_heads));
            if options.gdn_prep {
                // The dv-independent q/k path hoisted to GdnPrep — once per
                // (request, v-head), one simdgroup covering Dk — then the
                // slimmed recurrent core reads its fp32 scratch.
                emit(
                    &mut dag,
                    Kernel::GdnPrep,
                    at,
                    Launch {
                        grid: [32, 1, g.gdn_v_heads],
                        tg: [32, 1, 1],
                    },
                );
                emit(
                    &mut dag,
                    Kernel::GdnCore,
                    at,
                    Launch {
                        grid: [32, g.gdn_v_dim, g.gdn_v_heads],
                        tg: [32, 4, 1],
                    },
                );
            } else {
                // The fused one-dispatch core: tg spans a tile of dv (32
                // simdgroups) so the q/k prologue is computed once per
                // threadgroup and shared, killing the Vd-fold redundancy at
                // full occupancy.
                emit(
                    &mut dag,
                    Kernel::GdnCore,
                    at,
                    Launch {
                        grid: [32, g.gdn_v_dim, g.gdn_v_heads],
                        tg: [32, 32, 1],
                    },
                );
            }
            emit(
                &mut dag,
                Kernel::GatedRms,
                at,
                gated_rms(g.gdn_v_heads, g.gdn_v_dim),
            );
            emit(&mut dag, Kernel::QmvOut, at, qmv(g.hidden));
            if options.fuse_residual {
                fuse_last(&mut dag);
            } else {
                emit(&mut dag, Kernel::Residual, at, residual(g.hidden));
            }
        }
        // The FFN, in one of two shapes. Everything above this line is the
        // same either way.
        emit(&mut dag, Kernel::FfnRms, at, rms(g.hidden, 1));
        if g.is_moe() {
            // The same nine dispatches and kernels the llama family's
            // mixture runs; what this family was missing was the DAG, not
            // the machinery. The sort is what makes the projections
            // tractable: grouping (token, slot) pairs by expert makes one
            // expert's rows contiguous, and a contiguous run against one
            // weight slice is the matmul the driver already has. At M=1 it
            // is a grouping with no padding and the projections stay
            // matvecs — deliberately, so the batched path is not a second
            // implementation.
            let sorted = u32::try_from(moe_sorted_rows(g, tuning, 1, RoutedProjection::Matmul))
                .expect("an M=1 sort is small");
            emit(&mut dag, Kernel::LlRouter, at, qmv(g.n_experts));
            emit(&mut dag, Kernel::GoRouterTopK, at, router_topk(g.n_experts));
            emit(&mut dag, Kernel::LlMoeSort, at, route_sort(g.n_experts));
            emit(
                &mut dag,
                Kernel::LlMoeGather,
                at,
                route_rows(g.hidden, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertGate,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertUp,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertSiluMul,
                at,
                route_rows(g.moe_intermediate, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertDown,
                at,
                routed_qmv(g.hidden, 1, sorted),
            );
            emit(&mut dag, Kernel::LlMoeCombine, at, route_rows(g.hidden, 1));
            // The shared expert: a dense FFN over the SAME FfnRms output the
            // router read, added to the mixture under a one-scalar gate.
            // Emitted after the mixture only because the combine has to see
            // both; nothing here depends on the routed half, and the
            // concurrency groups say so, so the two halves overlap on the
            // GPU.
            if g.has_shared_expert() {
                emit(
                    &mut dag,
                    Kernel::LlSharedGate,
                    at,
                    qmv(g.shared_intermediate),
                );
                emit(&mut dag, Kernel::LlSharedUp, at, qmv(g.shared_intermediate));
                emit(
                    &mut dag,
                    Kernel::SiluMul,
                    at,
                    silu_mul(g.shared_intermediate),
                );
                emit(&mut dag, Kernel::LlSharedDown, at, qmv(g.hidden));
                // The gate goes LAST of the five, immediately before its
                // only reader. It reads `normed`, live the whole way, so it
                // could have gone first — and going first cost a ninth
                // scratch colour on a pool of eight, because a value
                // written five dispatches before its read is live across
                // all five. It is `hidden -> 1`: the cheapest dispatch in
                // the layer, and the one with the least to gain from
                // overlapping.
                emit(&mut dag, Kernel::LlSharedGateProj, at, qmv(1));
                emit(
                    &mut dag,
                    Kernel::LlSharedCombine,
                    at,
                    route_rows(g.hidden, 1),
                );
            }
        } else {
            emit(&mut dag, Kernel::QmvGate, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::QmvUp, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::SiluMul, at, silu_mul(g.intermediate));
            emit(&mut dag, Kernel::QmvDown, at, qmv(g.hidden));
            if options.fuse_residual {
                fuse_last(&mut dag);
            }
        }
        if !options.fuse_residual || g.is_moe() {
            emit(&mut dag, Kernel::LayerOut, at, residual(g.hidden));
        }
    }

    // TAIL: final norm, lm_head (logits always produced), optional argmax.
    emit(&mut dag, Kernel::FinalRms, None, rms(g.hidden, 1));
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::QmvLmHead
        } else {
            Kernel::LmHeadUntied
        },
        None,
        qmv(g.vocab),
    );
    if options.with_argmax {
        emit(
            &mut dag,
            Kernel::Argmax,
            None,
            Launch {
                grid: [1024, 1, 1],
                tg: [1024, 1, 1],
            },
        );
    }
    dag
}

/// Kinds that may run together inside a layer. Each group reads only
/// activations produced before the group starts and writes a distinct
/// scratch value, so the members are mutually independent — not merely
/// pairwise-adjacent-independent — which is what lets a whole group drop
/// its internal barriers.
///
/// * the GDN in-projections and the attention q/k/v projections all read
///   the block norm's output;
/// * gate/up both read the FFN norm's output;
/// * q_norm/k_norm rewrite their own heads in place, and Rope/RopeK
///   likewise rewrite q and k in place from disjoint scratch — the barrier
///   that used to sit between the ropes was ordering nothing.
const fn concurrency_group(kind: Kernel) -> u8 {
    match kind {
        Kernel::QmvIn | Kernel::QmvInZ | Kernel::GdnInA | Kernel::GdnInB => 1,
        Kernel::QmvQ | Kernel::QmvK | Kernel::QmvV => 2,
        Kernel::QmvGate | Kernel::QmvUp => 3,
        Kernel::QNorm | Kernel::KNorm => 4,
        Kernel::Rope | Kernel::RopeK => 5,
        _ => 0, // runs alone
    }
}

/// `run_ends[i]`: the last ordinal of the concurrency run containing `i` —
/// the shape [`color_live_ranges`](super::color::color_live_ranges) and the
/// encoder's barrier placement both consume.
#[must_use]
pub fn concurrent_run_ends(dag: &[Dispatch]) -> Vec<u32> {
    let mut ends: Vec<u32> = (0..dag.len())
        .map(|i| u32::try_from(i).expect("a DAG is hundreds of dispatches"))
        .collect();
    let mut i = 0;
    while i < dag.len() {
        let group = concurrency_group(dag[i].kind);
        let mut j = i;
        if group != 0 {
            while j + 1 < dag.len()
                && dag[j + 1].layer == dag[i].layer
                && concurrency_group(dag[j + 1].kind) == group
            {
                j += 1;
            }
        }
        let end = u32::try_from(j).expect("in range");
        for slot in &mut ends[i..=j] {
            *slot = end;
        }
        i = j + 1;
    }
    ends
}

/// Whether a barrier follows dispatch `i`: true everywhere except inside a
/// concurrency run.
#[must_use]
pub fn barrier_after(dag: &[Dispatch], i: usize, run_ends: &[u32]) -> bool {
    if i + 1 >= dag.len() {
        return true;
    }
    run_ends[i] == u32::try_from(i).expect("in range")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_shape_is_the_documented_393_dispatches() {
        let dag = build_decode_dag(
            &DecodeGeometry::default(),
            &Tuning::default(),
            DagOptions::default(),
        );
        // 1 embed + 18 GDN x 15 + 6 full-attn x 20 + 2 tail.
        assert_eq!(dag.len(), 393);
        assert_eq!(dag[0].kind, Kernel::EmbedGather);
        assert_eq!(dag[dag.len() - 1].kind, Kernel::QmvLmHead);
        // Ordinals are the flat positions — the argument-table key.
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
    }

    #[test]
    fn a_concurrency_run_spans_its_group_and_stops_at_the_layer() {
        let dag = build_decode_dag(
            &DecodeGeometry::default(),
            &Tuning::default(),
            DagOptions::default(),
        );
        let ends = concurrent_run_ends(&dag);
        // Find the first full-attention layer's q/k/v run.
        let q = dag
            .iter()
            .position(|d| d.kind == Kernel::QmvQ)
            .expect("a full-attn layer exists");
        assert_eq!(dag[q + 1].kind, Kernel::QmvK);
        assert_eq!(dag[q + 2].kind, Kernel::QmvV);
        assert_eq!(ends[q] as usize, q + 2, "q/k/v are one run");
        assert!(!barrier_after(&dag, q, &ends));
        assert!(barrier_after(&dag, q + 2, &ends));
    }

    #[test]
    fn the_shared_gate_is_never_a_zero_thread_dispatch() {
        // The sigmoid(0) = 0.5 story: hidden -> 1 logit must still launch.
        let launch = qmv(1);
        assert_eq!(launch.grid, [32, 1, 1]);
    }
}
