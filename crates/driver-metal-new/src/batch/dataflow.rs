//! The decode DAG's activation dataflow: which value each dispatch reads
//! and writes, threaded through the golden-surface order.
//!
//! This is the walk that feeds the colouring
//! ([`schedule_scratch`](super::schedule_scratch)). Live handles — the
//! residual stream, the block norm's output, the attention temporaries,
//! the GDN chain, the mixture's routing values — advance as the walk moves,
//! and every read/write becomes a [`Use`] at the kernel's own activation
//! bind slot.
//!
//! The mixture's values are tracked as ordinary values with honest live
//! ranges rather than pinned, so the colouring SEES the extent instead of
//! being told about it: the sort's outputs are written once and still read
//! five dispatches later, with three matvecs in between allocating freely.
//! The shared expert's gate/up/silu deliberately reuse the dense handles —
//! by the time it runs, the mixture's projections have consumed theirs,
//! and it IS the same three-stage SwiGLU dataflow; private handles would
//! have said the two are different shapes, which they are not.

use super::abi::Kernel;
use super::color::{ScheduleError, ScratchSchedule, Use, schedule_scratch};
use super::dispatch::{Dispatch, concurrent_run_ends};

/// The activation bind slots, from `decode_abi.hpp`'s `bind::` enums —
/// the slots the SCRATCH plane owns (weights and IO have their own
/// tables).
#[allow(missing_docs)] // names are the C++ bind:: constants, one-to-one
mod bi {
    pub const EMBED_OUT: u8 = 4;
    pub const RMS_X: u8 = 0;
    pub const RMS_OUT: u8 = 2;
    pub const QMV_X: u8 = 3;
    pub const QMV_OUT: u8 = 4;
    pub const QMV_RESIDUAL: u8 = 7;
    pub const QSPLIT_IN: u8 = 0;
    pub const QSPLIT_Q: u8 = 1;
    pub const QSPLIT_GATE: u8 = 2;
    pub const ROPE_X: u8 = 0;
    pub const SDPA_Q: u8 = 0;
    pub const SDPA_OUT: u8 = 3;
    pub const ATTN_GATE_ATTN: u8 = 0;
    pub const ATTN_GATE_GATE: u8 = 1;
    pub const KV_APPEND_K: u8 = 0;
    pub const KV_APPEND_V: u8 = 1;
    pub const GDN_MIXED: u8 = 0;
    pub const GDN_CORE_OUT: u8 = 3;
    pub const GDN_A_GATE: u8 = 8;
    pub const GDN_B_GATE: u8 = 9;
    pub const GDN_PREP_MIXED: u8 = 0;
    pub const GDN_PREP_A_GATE: u8 = 6;
    pub const GDN_PREP_B_GATE: u8 = 7;
    pub const GDN_PREP_PRE_Q: u8 = 8;
    pub const GDN_PREP_PRE_K: u8 = 9;
    pub const GDN_PREP_PRE_GATE: u8 = 10;
    pub const GDN_REC_MIXED: u8 = 0;
    pub const GDN_REC_CORE_OUT: u8 = 3;
    pub const GDN_REC_PRE_Q: u8 = 6;
    pub const GDN_REC_PRE_K: u8 = 7;
    pub const GDN_REC_PRE_GATE: u8 = 8;
    pub const GATED_RMS_X: u8 = 0;
    pub const GATED_RMS_Z: u8 = 1;
    pub const GATED_RMS_OUT: u8 = 3;
    pub const RESID_X: u8 = 0;
    pub const RESID_R: u8 = 1;
    pub const RESID_OUT: u8 = 2;
    pub const SILU_GATE: u8 = 0;
    pub const SILU_UP: u8 = 1;
    pub const SILU_OUT: u8 = 2;
    // The routed matvec shares the dense activation binds — same kernel,
    // weight stack indexed per row — so only the routing slots are new.
    pub const QMV_EXPERT_IDS: u8 = 8;
    pub const QMV_TILE_EXPERT: u8 = 12;
    pub const ROUTER_LOGITS: u8 = 0;
    pub const ROUTER_IDS: u8 = 1;
    pub const ROUTER_WEIGHTS: u8 = 2;
    pub const ROW_GATHER_IN: u8 = 0;
    pub const ROW_GATHER_OUT: u8 = 1;
    // gemma4: the fused norm+residual keeps bind::Rms's prefix and
    // appends the residual; the weightless V norm and the layer scalar
    // are tiny ABIs of their own; the PLE combine is `(proj+token)/√2`.
    pub const RR_X: u8 = 0;
    pub const RR_OUT: u8 = 2;
    pub const RR_RESID: u8 = 4;
    pub const VNORM_X: u8 = 0;
    pub const VNORM_OUT: u8 = 1;
    pub const SCALAR_X: u8 = 0;
    pub const SCALAR_OUT: u8 = 2;
    pub const PLE_PROJ: u8 = 0;
    pub const PLE_TOKEN: u8 = 1;
    pub const PLE_OUT: u8 = 2;
    pub const SORT_IDS: u8 = 0;
    pub const SORT_PERM: u8 = 1;
    pub const SORT_ROW_EXPERT: u8 = 2;
    pub const SORT_TILE_EXPERT: u8 = 3;
    pub const SORT_INV: u8 = 5;
    pub const ROWS_IN: u8 = 0;
    pub const ROWS_OUT: u8 = 1;
    pub const ROWS_PERM: u8 = 2;
    pub const COMBINE_Y: u8 = 0;
    pub const COMBINE_WEIGHTS: u8 = 1;
    pub const COMBINE_OUT: u8 = 2;
    pub const COMBINE_INV: u8 = 4;
    pub const SH_ROUTED: u8 = 0;
    pub const SH_SHARED: u8 = 1;
    pub const SH_GATE: u8 = 2;
    pub const SH_OUT: u8 = 3;
}

/// The live handles the walk threads. `None` means "not produced yet";
/// reading one is a malformed DAG and panics with the kind, because every
/// DAG this walk sees comes from `build_decode_dag` and a gap is a port
/// bug to catch in the first test, not a runtime condition.
#[derive(Default)]
struct Live {
    resid: Option<u32>,
    normed: Option<u32>,
    q: Option<u32>,
    gate: Option<u32>,
    kk: Option<u32>,
    vv: Option<u32>,
    attn: Option<u32>,
    mixed: Option<u32>,
    zg: Option<u32>,
    ag: Option<u32>,
    bg: Option<u32>,
    core: Option<u32>,
    gnorm: Option<u32>,
    out: Option<u32>,
    pq: Option<u32>,
    pk: Option<u32>,
    pg: Option<u32>,
    prep_pending: bool,
    gp: Option<u32>,
    up: Option<u32>,
    hh: Option<u32>,
    dn: Option<u32>,
    router_logits: Option<u32>,
    expert_ids: Option<u32>,
    expert_weights: Option<u32>,
    shared_gate: Option<u32>,
    shared_out: Option<u32>,
    perm: Option<u32>,
    row_expert: Option<u32>,
    tile_expert: Option<u32>,
    inv: Option<u32>,
    sorted_x: Option<u32>,
    sorted_out: Option<u32>,
    // gemma4's second stream and its sandwich temporaries.
    ple_tok: Option<u32>,
    ple_proj: Option<u32>,
    ple: Option<u32>,
    ple_gate: Option<u32>,
    ple_act: Option<u32>,
    ple_back: Option<u32>,
    dense_br: Option<u32>,
    router_normed: Option<u32>,
    moe_normed: Option<u32>,
}

fn have(value: Option<u32>, kind: Kernel, what: &str) -> u32 {
    value.unwrap_or_else(|| panic!("{kind:?} reads {what} before anything produced it"))
}

/// Walk the DAG and produce its uses and value count.
#[must_use]
#[allow(clippy::too_many_lines)] // one walk, one dataflow: splitting it hides the threading
pub fn build_scratch_uses(dag: &[Dispatch]) -> (Vec<Use>, usize) {
    let mut uses: Vec<Use> = Vec::new();
    let mut next_value = 0u32;
    let mut fresh = || {
        let v = next_value;
        next_value += 1;
        v
    };
    let mut live = Live::default();

    for (index, d) in dag.iter().enumerate() {
        // Schedules are indexed by DAG position, not argument-table
        // ordinal: M=1 happens to use both as 0..N-1; paged ordinals
        // deliberately live in a disjoint range.
        let o = u32::try_from(index).expect("a DAG is hundreds of dispatches");
        let k = d.kind;
        let rd = |uses: &mut Vec<Use>, b: u8, v: u32| {
            uses.push(Use {
                ordinal: o,
                bind_index: b,
                value: v,
                is_write: false,
            });
        };
        let wr = |uses: &mut Vec<Use>, b: u8, v: u32| {
            uses.push(Use {
                ordinal: o,
                bind_index: b,
                value: v,
                is_write: true,
            });
        };
        match k {
            Kernel::EmbedUntied | Kernel::EmbedGather => {
                let v = fresh();
                wr(&mut uses, bi::EMBED_OUT, v);
                live.resid = Some(v);
            }
            Kernel::Rms | Kernel::FfnRms | Kernel::FinalRms | Kernel::G4FfnPreNorm => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::RMS_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RMS_OUT, v);
                live.normed = Some(v);
            }

            // ── GDN block ──
            Kernel::QmvIn => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.mixed = Some(v);
            }
            Kernel::QmvInZ => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.zg = Some(v);
            }
            Kernel::GdnInA => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.ag = Some(v);
            }
            Kernel::GdnInB => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.bg = Some(v);
            }
            Kernel::GdnPrep | Kernel::GdnPrepSlotted => {
                let (pq, pk, pg) = (fresh(), fresh(), fresh());
                rd(
                    &mut uses,
                    bi::GDN_PREP_MIXED,
                    have(live.mixed, k, "the in-projection"),
                );
                rd(
                    &mut uses,
                    bi::GDN_PREP_A_GATE,
                    have(live.ag, k, "the a-gate"),
                );
                rd(
                    &mut uses,
                    bi::GDN_PREP_B_GATE,
                    have(live.bg, k, "the b-gate"),
                );
                wr(&mut uses, bi::GDN_PREP_PRE_Q, pq);
                wr(&mut uses, bi::GDN_PREP_PRE_K, pk);
                wr(&mut uses, bi::GDN_PREP_PRE_GATE, pg);
                live.pq = Some(pq);
                live.pk = Some(pk);
                live.pg = Some(pg);
                live.prep_pending = true;
            }
            Kernel::GdnCore | Kernel::GdnCoreSlotted => {
                let v = fresh();
                if live.prep_pending {
                    rd(
                        &mut uses,
                        bi::GDN_REC_MIXED,
                        have(live.mixed, k, "the in-projection"),
                    );
                    rd(&mut uses, bi::GDN_REC_PRE_Q, have(live.pq, k, "prep's q"));
                    rd(&mut uses, bi::GDN_REC_PRE_K, have(live.pk, k, "prep's k"));
                    rd(
                        &mut uses,
                        bi::GDN_REC_PRE_GATE,
                        have(live.pg, k, "prep's gate"),
                    );
                    wr(&mut uses, bi::GDN_REC_CORE_OUT, v);
                    live.prep_pending = false;
                } else {
                    rd(
                        &mut uses,
                        bi::GDN_MIXED,
                        have(live.mixed, k, "the in-projection"),
                    );
                    rd(&mut uses, bi::GDN_A_GATE, have(live.ag, k, "the a-gate"));
                    rd(&mut uses, bi::GDN_B_GATE, have(live.bg, k, "the b-gate"));
                    wr(&mut uses, bi::GDN_CORE_OUT, v);
                }
                live.core = Some(v);
            }
            Kernel::GatedRms => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::GATED_RMS_X,
                    have(live.core, k, "the core output"),
                );
                rd(&mut uses, bi::GATED_RMS_Z, have(live.zg, k, "the z-gate"));
                wr(&mut uses, bi::GATED_RMS_OUT, v);
                live.gnorm = Some(v);
            }
            Kernel::QmvOut => {
                let source = have(live.gnorm, k, "the gated norm");
                if d.fuse_residual {
                    let nr = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    rd(
                        &mut uses,
                        bi::QMV_RESIDUAL,
                        have(live.resid, k, "the residual stream"),
                    );
                    wr(&mut uses, bi::QMV_OUT, nr);
                    live.resid = Some(nr);
                } else {
                    let v = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    wr(&mut uses, bi::QMV_OUT, v);
                    live.out = Some(v);
                }
            }

            // ── Full-attention block ──
            Kernel::QmvQ => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.q = Some(v);
            }
            Kernel::QSplit => {
                let (query, gate) = (fresh(), fresh());
                rd(&mut uses, bi::QSPLIT_IN, have(live.q, k, "the 2x-wide q"));
                wr(&mut uses, bi::QSPLIT_Q, query);
                wr(&mut uses, bi::QSPLIT_GATE, gate);
                // The post-split query replaces the 2x-wide buffer.
                live.q = Some(query);
                live.gate = Some(gate);
            }
            Kernel::QmvK => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.kk = Some(v);
            }
            Kernel::QmvV => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.vv = Some(v);
            }
            Kernel::QNorm => {
                let q = have(live.q, k, "the query");
                rd(&mut uses, bi::RMS_X, q);
                wr(&mut uses, bi::RMS_OUT, q);
            }
            Kernel::KNorm => {
                let kk = have(live.kk, k, "the key");
                rd(&mut uses, bi::RMS_X, kk);
                wr(&mut uses, bi::RMS_OUT, kk);
            }
            Kernel::Rope => {
                let q = have(live.q, k, "the query");
                rd(&mut uses, bi::ROPE_X, q);
                wr(&mut uses, bi::ROPE_X, q);
            }
            Kernel::RopeK => {
                let kk = have(live.kk, k, "the key");
                rd(&mut uses, bi::ROPE_X, kk);
                wr(&mut uses, bi::ROPE_X, kk);
            }
            Kernel::KvAppend | Kernel::KvAppendPaged => {
                rd(&mut uses, bi::KV_APPEND_K, have(live.kk, k, "the key"));
                rd(&mut uses, bi::KV_APPEND_V, have(live.vv, k, "the value"));
            }
            Kernel::Sdpa | Kernel::SdpaPaged | Kernel::G4SdpaSliding => {
                let v = fresh();
                rd(&mut uses, bi::SDPA_Q, have(live.q, k, "the query"));
                wr(&mut uses, bi::SDPA_OUT, v);
                live.attn = Some(v);
            }
            Kernel::AttnGate => {
                let attn = have(live.attn, k, "the attention output");
                rd(&mut uses, bi::ATTN_GATE_ATTN, attn);
                rd(
                    &mut uses,
                    bi::ATTN_GATE_GATE,
                    have(live.gate, k, "the query gate"),
                );
                wr(&mut uses, bi::ATTN_GATE_ATTN, attn);
            }
            Kernel::QmvO => {
                let source = have(live.attn, k, "the attention output");
                if d.fuse_residual {
                    let nr = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    rd(
                        &mut uses,
                        bi::QMV_RESIDUAL,
                        have(live.resid, k, "the residual stream"),
                    );
                    wr(&mut uses, bi::QMV_OUT, nr);
                    live.resid = Some(nr);
                } else {
                    let v = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    wr(&mut uses, bi::QMV_OUT, v);
                    live.out = Some(v);
                }
            }
            Kernel::Residual => {
                let nr = fresh();
                rd(
                    &mut uses,
                    bi::RESID_X,
                    have(live.resid, k, "the residual stream"),
                );
                rd(
                    &mut uses,
                    bi::RESID_R,
                    have(live.out, k, "the block output"),
                );
                wr(&mut uses, bi::RESID_OUT, nr);
                live.resid = Some(nr);
            }

            // ── SwiGLU MLP ──
            Kernel::QmvGate => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.gp = Some(v);
            }
            Kernel::QmvUp => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.up = Some(v);
            }
            Kernel::SiluMul | Kernel::LlExpertSiluMul | Kernel::G4Geglu | Kernel::G4ExpertGeglu => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::SILU_GATE,
                    have(live.gp, k, "the gate projection"),
                );
                rd(
                    &mut uses,
                    bi::SILU_UP,
                    have(live.up, k, "the up projection"),
                );
                wr(&mut uses, bi::SILU_OUT, v);
                live.hh = Some(v);
            }
            Kernel::QmvDown => {
                let source = have(live.hh, k, "the SwiGLU output");
                if d.fuse_residual {
                    let nr = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    rd(
                        &mut uses,
                        bi::QMV_RESIDUAL,
                        have(live.resid, k, "the residual stream"),
                    );
                    wr(&mut uses, bi::QMV_OUT, nr);
                    live.resid = Some(nr);
                } else {
                    let v = fresh();
                    rd(&mut uses, bi::QMV_X, source);
                    wr(&mut uses, bi::QMV_OUT, v);
                    live.dn = Some(v);
                }
            }
            Kernel::LayerOut => {
                let nr = fresh();
                rd(
                    &mut uses,
                    bi::RESID_X,
                    have(live.resid, k, "the residual stream"),
                );
                rd(&mut uses, bi::RESID_R, have(live.dn, k, "the FFN output"));
                wr(&mut uses, bi::RESID_OUT, nr);
                live.resid = Some(nr);
            }

            // ── the routed FFN ──
            Kernel::LlRouter => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.router_logits = Some(v);
            }
            Kernel::GoRouterTopK | Kernel::G4RouterTopK => {
                // Both outputs outlive the three matvecs that follow: the
                // ids name each pair's expert; the weights are what the
                // combine finally sums with.
                let (ids, weights) = (fresh(), fresh());
                rd(
                    &mut uses,
                    bi::ROUTER_LOGITS,
                    have(live.router_logits, k, "the router"),
                );
                wr(&mut uses, bi::ROUTER_IDS, ids);
                wr(&mut uses, bi::ROUTER_WEIGHTS, weights);
                live.expert_ids = Some(ids);
                live.expert_weights = Some(weights);
            }
            Kernel::LlMoeSort | Kernel::G4MoeSort => {
                let (perm, row_expert, tile_expert, inv) = (fresh(), fresh(), fresh(), fresh());
                rd(
                    &mut uses,
                    bi::SORT_IDS,
                    have(live.expert_ids, k, "the expert ids"),
                );
                wr(&mut uses, bi::SORT_PERM, perm);
                wr(&mut uses, bi::SORT_ROW_EXPERT, row_expert);
                wr(&mut uses, bi::SORT_TILE_EXPERT, tile_expert);
                wr(&mut uses, bi::SORT_INV, inv);
                live.perm = Some(perm);
                live.row_expert = Some(row_expert);
                live.tile_expert = Some(tile_expert);
                live.inv = Some(inv);
            }
            Kernel::LlMoeGather => {
                let v = fresh();
                rd(&mut uses, bi::ROWS_IN, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::ROWS_OUT, v);
                rd(
                    &mut uses,
                    bi::ROWS_PERM,
                    have(live.perm, k, "the sort's permutation"),
                );
                live.sorted_x = Some(v);
            }
            Kernel::LlExpertGate | Kernel::G4ExpertGate => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.sorted_x, k, "the gathered stack"),
                );
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.gp = Some(v);
            }
            Kernel::LlExpertUp | Kernel::G4ExpertUp => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.sorted_x, k, "the gathered stack"),
                );
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.up = Some(v);
            }
            Kernel::LlExpertDown | Kernel::G4ExpertDown => {
                // Still sorted: the k results per token come back together
                // only at the combine, through the sort's inverse.
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.hh, k, "the expert SwiGLU"));
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.sorted_out = Some(v);
            }
            Kernel::LlMoeCombine | Kernel::G4ExpertCombine => {
                // The mixture's output takes the place a dense down_proj
                // would have written, so the residual add below needs no
                // case of its own.
                let v = fresh();
                rd(
                    &mut uses,
                    bi::COMBINE_Y,
                    have(live.sorted_out, k, "the sorted outputs"),
                );
                rd(
                    &mut uses,
                    bi::COMBINE_WEIGHTS,
                    have(live.expert_weights, k, "the weights"),
                );
                rd(
                    &mut uses,
                    bi::COMBINE_INV,
                    have(live.inv, k, "the sort's inverse"),
                );
                wr(&mut uses, bi::COMBINE_OUT, v);
                live.dn = Some(v);
            }

            // ── the shared expert: every projection reads `normed`, not
            // `sorted_x` — it sees every token whole, which is the point. ──
            Kernel::LlSharedGateProj => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.shared_gate = Some(v);
            }
            Kernel::LlSharedGate => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.gp = Some(v);
            }
            Kernel::LlSharedUp => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.up = Some(v);
            }
            Kernel::LlSharedDown => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.hh, k, "the shared SwiGLU"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.shared_out = Some(v);
            }
            Kernel::LlSharedCombine => {
                // Takes dn's place, so the residual add still reads one
                // value and does not know a mixture happened.
                let v = fresh();
                rd(
                    &mut uses,
                    bi::SH_ROUTED,
                    have(live.dn, k, "the routed output"),
                );
                rd(
                    &mut uses,
                    bi::SH_SHARED,
                    have(live.shared_out, k, "the shared output"),
                );
                rd(
                    &mut uses,
                    bi::SH_GATE,
                    have(live.shared_gate, k, "the shared gate"),
                );
                wr(&mut uses, bi::SH_OUT, v);
                live.dn = Some(v);
            }

            // ── gpt-oss: the biased projections thread like the plain
            // ones (the bias is a weight, not a value); the sink attention
            // reads q and writes attn like the plain SDPA; the clamped
            // SwiGLU is silu_mul's shape under its own kind; the combine
            // reads the sorted outputs through the sort's inverse exactly
            // as llama's does — it binds the same table. ──
            Kernel::GoQmvQ => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.q = Some(v);
            }
            Kernel::GoQmvK => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.kk = Some(v);
            }
            Kernel::GoQmvV => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the block norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.vv = Some(v);
            }
            Kernel::GoSdpaSink | Kernel::GoSdpaSinkPaged => {
                let v = fresh();
                rd(&mut uses, bi::SDPA_Q, have(live.q, k, "the query"));
                wr(&mut uses, bi::SDPA_OUT, v);
                live.attn = Some(v);
            }
            Kernel::GoQmvO => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.attn, k, "the attention output"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.out = Some(v);
            }
            Kernel::GoRouter => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the FFN norm"));
                wr(&mut uses, bi::QMV_OUT, v);
                live.router_logits = Some(v);
            }
            Kernel::GoExpertGate => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.sorted_x, k, "the gathered stack"),
                );
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.gp = Some(v);
            }
            Kernel::GoExpertUp => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.sorted_x, k, "the gathered stack"),
                );
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.up = Some(v);
            }
            Kernel::GoSwiGlu => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::SILU_GATE,
                    have(live.gp, k, "the gate projection"),
                );
                rd(
                    &mut uses,
                    bi::SILU_UP,
                    have(live.up, k, "the up projection"),
                );
                wr(&mut uses, bi::SILU_OUT, v);
                live.hh = Some(v);
            }
            Kernel::GoExpertDown => {
                let v = fresh();
                rd(&mut uses, bi::QMV_X, have(live.hh, k, "the expert SwiGLU"));
                rd(
                    &mut uses,
                    bi::QMV_EXPERT_IDS,
                    have(live.row_expert, k, "the row experts"),
                );
                rd(
                    &mut uses,
                    bi::QMV_TILE_EXPERT,
                    have(live.tile_expert, k, "the tile experts"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.sorted_out = Some(v);
            }
            Kernel::GoExpertCombine => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::COMBINE_Y,
                    have(live.sorted_out, k, "the sorted outputs"),
                );
                rd(
                    &mut uses,
                    bi::COMBINE_WEIGHTS,
                    have(live.expert_weights, k, "the weights"),
                );
                rd(
                    &mut uses,
                    bi::COMBINE_INV,
                    have(live.inv, k, "the sort's inverse"),
                );
                wr(&mut uses, bi::COMBINE_OUT, v);
                live.dn = Some(v);
            }

            // ── gemma4: the PLE stream, the sandwich, the weightless
            // V norms, and the mixture that sits BESIDE the dense MLP. ──
            Kernel::G4PleTokenGather => {
                let v = fresh();
                wr(&mut uses, bi::EMBED_OUT, v);
                live.ple_tok = Some(v);
            }
            Kernel::G4PleProjGemv => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.ple_proj = Some(v);
            }
            Kernel::G4PleProjNorm => {
                let v = have(live.ple_proj, k, "the PLE projection");
                rd(&mut uses, bi::RMS_X, v);
                wr(&mut uses, bi::RMS_OUT, v);
            }
            Kernel::G4PleCombine => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::PLE_PROJ,
                    have(live.ple_proj, k, "the PLE projection"),
                );
                rd(
                    &mut uses,
                    bi::PLE_TOKEN,
                    have(live.ple_tok, k, "the PLE table"),
                );
                wr(&mut uses, bi::PLE_OUT, v);
                live.ple = Some(v);
            }
            // Weightless, in place: V normalised on its way to the cache.
            Kernel::G4VNorm => {
                let v = have(live.vv, k, "the value");
                rd(&mut uses, bi::VNORM_X, v);
                wr(&mut uses, bi::VNORM_OUT, v);
            }
            // The layer projected no V: it reads the K PROJECTION —
            // before KNorm rewrites it — and writes a V of its own.
            Kernel::G4VNormFromK => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::VNORM_X,
                    have(live.kk, k, "the key projection"),
                );
                wr(&mut uses, bi::VNORM_OUT, v);
                live.vv = Some(v);
            }
            // The sandwich's second half and the add it always precedes:
            // normalise the BLOCK's output, then rejoin the stream.
            Kernel::G4AttnPostResidual => {
                let next = fresh();
                rd(
                    &mut uses,
                    bi::RR_X,
                    have(live.out, k, "the attention block"),
                );
                rd(
                    &mut uses,
                    bi::RR_RESID,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RR_OUT, next);
                live.resid = Some(next);
            }
            Kernel::G4FfnPostResidual => {
                let next = fresh();
                rd(&mut uses, bi::RR_X, have(live.dn, k, "the FFN block"));
                rd(
                    &mut uses,
                    bi::RR_RESID,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RR_OUT, next);
                live.resid = Some(next);
            }
            // The mixture's four norms and the router, all reading the
            // POST-ATTENTION residual — the branches are siblings, not a
            // chain; `dn` carries the dense branch through to the add.
            Kernel::G4DenseBranchNorm => {
                let v = fresh();
                rd(&mut uses, bi::RMS_X, have(live.dn, k, "the dense branch"));
                wr(&mut uses, bi::RMS_OUT, v);
                live.dense_br = Some(v);
            }
            Kernel::G4RouterNorm => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::RMS_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RMS_OUT, v);
                live.router_normed = Some(v);
            }
            Kernel::G4Router => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.router_normed, k, "the router norm"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.router_logits = Some(v);
            }
            Kernel::G4MoeNorm => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::RMS_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RMS_OUT, v);
                live.moe_normed = Some(v);
            }
            // Like the shared gather, but off the mixture's OWN entry
            // norm rather than the dense branch's.
            Kernel::G4MoeGather => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::ROWS_IN,
                    have(live.moe_normed, k, "the mixture norm"),
                );
                wr(&mut uses, bi::ROWS_OUT, v);
                rd(
                    &mut uses,
                    bi::ROWS_PERM,
                    have(live.perm, k, "the sort's permutation"),
                );
                live.sorted_x = Some(v);
            }
            Kernel::G4MoeBranchNorm => {
                let v = fresh();
                rd(&mut uses, bi::RMS_X, have(live.dn, k, "the mixture output"));
                wr(&mut uses, bi::RMS_OUT, v);
                live.dn = Some(v);
            }
            // The two branches meet; `dn` from here is what the
            // sandwich's second half normalises, exactly as on a dense
            // layer — which is why G4FfnPostResidual needs no mixture
            // case.
            Kernel::G4BranchAdd => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::RESID_X,
                    have(live.dense_br, k, "the dense branch"),
                );
                rd(
                    &mut uses,
                    bi::RESID_R,
                    have(live.dn, k, "the mixture branch"),
                );
                wr(&mut uses, bi::RESID_OUT, v);
                live.dn = Some(v);
            }
            // ── the per-layer embedding residual ──
            Kernel::G4PleGateGemv => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.ple_gate = Some(v);
            }
            // Gated by the stream, valued by this layer's slice of the
            // table.
            Kernel::G4PleGeglu => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::SILU_GATE,
                    have(live.ple_gate, k, "the PLE gate"),
                );
                rd(&mut uses, bi::SILU_UP, have(live.ple, k, "the PLE stream"));
                wr(&mut uses, bi::SILU_OUT, v);
                live.ple_act = Some(v);
            }
            Kernel::G4PleProjLayerGemv => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::QMV_X,
                    have(live.ple_act, k, "the PLE activation"),
                );
                wr(&mut uses, bi::QMV_OUT, v);
                live.ple_back = Some(v);
            }
            Kernel::G4PleResidualScaled => {
                let next = fresh();
                rd(&mut uses, bi::RR_X, have(live.ple_back, k, "the PLE block"));
                rd(
                    &mut uses,
                    bi::RR_RESID,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::RR_OUT, next);
                live.resid = Some(next);
            }
            Kernel::G4LayerScalar => {
                let next = fresh();
                rd(
                    &mut uses,
                    bi::SCALAR_X,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::SCALAR_OUT, next);
                live.resid = Some(next);
            }

            // The sampled rows, compacted: everything after this is [S, *].
            // A value of its own rather than in place — the input is
            // [N, hidden] and the output is [S, hidden], and aliasing them
            // would make the pool's slot sizing a lie.
            Kernel::G4RowGather => {
                let v = fresh();
                rd(
                    &mut uses,
                    bi::ROW_GATHER_IN,
                    have(live.resid, k, "the residual stream"),
                );
                wr(&mut uses, bi::ROW_GATHER_OUT, v);
                live.resid = Some(v);
            }

            // The tail lm_head reads the final norm from scratch and writes
            // logits to IO, not scratch; argmax touches IO only.
            Kernel::QmvLmHead | Kernel::LmHeadUntied => {
                rd(&mut uses, bi::QMV_X, have(live.normed, k, "the final norm"));
            }
            Kernel::Argmax | Kernel::G4Softcap => {}
            other => panic!("the dataflow walk does not know {other:?}"),
        }
    }
    (uses, next_value as usize)
}

/// The full schedule for a DAG: walk, colour, fan out.
///
/// # Errors
///
/// [`ScheduleError`]; a hazard is unignorable through here too.
pub fn build_scratch_schedule(
    dag: &[Dispatch],
    no_recycle: bool,
) -> Result<ScratchSchedule, ScheduleError> {
    let (uses, value_count) = build_scratch_uses(dag);
    let run_ends = concurrent_run_ends(dag);
    schedule_scratch(dag.len(), &uses, &run_ends, value_count, no_recycle)
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::{DagOptions, build_decode_dag};
    use super::super::geometry::DecodeGeometry;
    use crate::batch::SCRATCH_POOL;
    use crate::tuning::Tuning;

    use super::*;

    #[test]
    fn the_default_dag_schedules_hazard_free_inside_the_pool() {
        let dag = build_decode_dag(
            &DecodeGeometry::default(),
            &Tuning::default(),
            DagOptions::default(),
        );
        let schedule = build_scratch_schedule(&dag, false).expect("no hazards");
        assert!(
            (schedule.coloring.colors_used as usize) <= SCRATCH_POOL,
            "{} colours over a pool of {SCRATCH_POOL}",
            schedule.coloring.colors_used
        );
        assert_eq!(schedule.per_dispatch.len(), dag.len());
    }

    #[test]
    fn a_routed_dag_with_a_shared_expert_also_fits_the_pool() {
        let geometry = DecodeGeometry {
            n_experts: 512,
            experts_per_token: 10,
            moe_intermediate: 768,
            shared_intermediate: 512,
            tied_embeddings: false,
            ..DecodeGeometry::default()
        };
        let dag = build_decode_dag(&geometry, &Tuning::default(), DagOptions::default());
        let schedule = build_scratch_schedule(&dag, false).expect("no hazards");
        assert!(
            (schedule.coloring.colors_used as usize) <= SCRATCH_POOL,
            "{} colours over a pool of {SCRATCH_POOL}",
            schedule.coloring.colors_used
        );
    }

    #[test]
    fn the_gptoss_dag_schedules_hazard_free_inside_the_pool() {
        let g = crate::batch::GptOssGeometry::default();
        let dag = crate::batch::build_gptoss_dag(&g, true);
        let schedule = build_scratch_schedule(&dag, false).expect("no hazards");
        assert!(
            (schedule.coloring.colors_used as usize) <= crate::batch::SCRATCH_POOL,
            "{} colours over a pool of {}",
            schedule.coloring.colors_used,
            crate::batch::SCRATCH_POOL
        );
        assert_eq!(schedule.per_dispatch.len(), dag.len());
    }

    #[test]
    fn fused_residuals_and_the_prep_split_still_schedule() {
        let dag = build_decode_dag(
            &DecodeGeometry::default(),
            &Tuning::default(),
            DagOptions {
                with_argmax: true,
                fuse_residual: true,
                gdn_prep: true,
            },
        );
        let schedule = build_scratch_schedule(&dag, false).expect("no hazards");
        assert!((schedule.coloring.colors_used as usize) <= SCRATCH_POOL);
    }
}
