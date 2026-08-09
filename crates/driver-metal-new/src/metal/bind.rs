//! Binding the decode DAG: weights, state, KV, IO, scratch and constants,
//! by ordinal, into the argument tables — then encoding it.
//!
//! Four passes over one DAG, in the C++'s order (`bind_decode_dag`,
//! `bind_scratch`, `bind_decode_consts`), each reacting to a dispatch's
//! kind rather than to a fixed sequence, so a family that reorders its DAG
//! rebinds correctly. The argument tables are address-only, so constants
//! travel in tiny resident buffers ([`ConstSlots`]) cached by
//! `(ordinal, index)` — a width-change rebind overwrites the same slot's
//! CONTENTS at an address the table already holds, so no encoded byte
//! moves.
//!
//! [`encode_decode_step`] is the walk itself: PSO by kind (the residual-
//! fused GEMV where the dispatch says so), argument table by ordinal, the
//! dispatch, and a barrier everywhere except inside a concurrency run.

use std::collections::HashMap;

use crate::batch::GatedRmsParams;
use crate::batch::{
    DecodeGeometry, Dispatch, ExpertCombineParams, IoSlot, Kernel, MoeRouteParams, RmsParams,
    RoutedProjection, RouterParams, ScratchSchedule, WeightBind, barrier_after,
    concurrent_run_ends, gdn_core_params, is_qmv, is_routed, moe_sorted_rows, qmv_kn, weight_binds,
};
use crate::region::Region as _;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::context::Context;
use super::encoder::{StepEncoder, Visibility};
use super::fire::pod_bytes;
use super::handle::Handle;
use super::program::Pso;
use super::ring::allocate;
use super::storage::DecodeStorage;
use super::tables::Tables;

/// The extra bind ordinals this pass writes that `batch::binds::slot` does
/// not carry (state, IO and const slots, from `decode_abi.hpp`'s `bind::`
/// namespace).
#[allow(missing_docs)] // names are the C++ bind:: constants, one-to-one
pub mod slot {
    pub const EMBED_TOKEN_ID: u8 = 3;
    pub const EMBED_HIDDEN: u8 = 5;
    /// Reserved for the fire path's activation rebind.
    #[allow(dead_code)]
    pub const QMV_X: u8 = 3;
    pub const QMV_OUT: u8 = 4;
    pub const QMV_K: u8 = 5;
    pub const QMV_N: u8 = 6;
    pub const GO_QMV_X_SLOT_STRIDE: u8 = 9;
    pub const GO_QMV_X_ROW_STRIDE: u8 = 10;
    pub const GO_QMV_SLOTS_PER_ROW: u8 = 11;
    /// Bound by the fire path once the router's ids buffer exists.
    #[allow(dead_code)]
    pub const GO_QMV_EXPERT_IDS: u8 = 8;
    pub const RMS_PARAMS: u8 = 3;
    pub const GATED_RMS_PARAMS: u8 = 4;
    pub const QSPLIT_HEAD_DIM: u8 = 3;
    pub const QSPLIT_QG_ROW_STRIDE: u8 = 4;
    pub const QSPLIT_OUT_ROW_STRIDE: u8 = 5;
    pub const ATTN_GATE_ROW_STRIDE: u8 = 2;
    pub const ROPE_POSITION: u8 = 1;
    pub const ROPE_SCALE: u8 = 2;
    pub const ROPE_BASE: u8 = 3;
    pub const ROPE_HEAD_DIM: u8 = 4;
    pub const KV_APPEND_K_PAGES: u8 = 2;
    pub const KV_APPEND_V_PAGES: u8 = 3;
    pub const KV_APPEND_POSITION: u8 = 4;
    pub const KV_APPEND_HEAD_DIM: u8 = 5;
    pub const KV_APPEND_K_HEAD_STRIDE: u8 = 6;
    pub const KV_APPEND_K_SEQ_STRIDE: u8 = 7;
    pub const SDPA_K: u8 = 1;
    pub const SDPA_V: u8 = 2;
    pub const SDPA_GQA_FACTOR: u8 = 4;
    pub const SDPA_N: u8 = 5;
    pub const SDPA_K_HEAD_STRIDE: u8 = 6;
    pub const KV_APPEND_PAGED_HEAD_DIM: u8 = 5;
    pub const KV_APPEND_PAGED_K_HEAD_STRIDE: u8 = 6;
    pub const KV_APPEND_PAGED_K_SEQ_STRIDE: u8 = 7;
    pub const KV_APPEND_PAGED_PAGE_SIZE: u8 = 10;
    pub const KV_APPEND_PAGED_N_KV_HEADS: u8 = 12;
    pub const KV_APPEND_PAGED_SRC_ROW_STRIDE: u8 = 15;
    pub const SDPA_PAGED_GQA_FACTOR: u8 = 4;
    pub const SDPA_PAGED_PAGE_SIZE: u8 = 9;
    pub const SDPA_PAGED_N_KV_HEADS: u8 = 10;
    pub const SDPA_PAGED_SCALE: u8 = 11;
    pub const SDPA_PAGED_WINDOW: u8 = 15;
    pub const SDPA_PAGED_ROWS: u8 = 17;
    pub const ROW_GATHER_ROWS: u8 = 2;
    pub const ROW_GATHER_PARAMS: u8 = 3;
    pub const SDPA_K_SEQ_STRIDE: u8 = 7;
    pub const SDPA_V_HEAD_STRIDE: u8 = 8;
    pub const SDPA_V_SEQ_STRIDE: u8 = 9;
    pub const SDPA_SCALE: u8 = 10;
    pub const GDN_CORE_CONV_STATE: u8 = 1;
    pub const GDN_CORE_RECURRENT_STATE: u8 = 2;
    pub const GDN_CORE_CONV_B: u8 = 5;
    pub const GDN_CORE_CONV_STATE_OUT: u8 = 10;
    pub const GDN_CORE_PARAMS: u8 = 11;
    pub const GDN_RECURRENT_CONV_STATE: u8 = 1;
    pub const GDN_RECURRENT_STATE: u8 = 2;
    pub const GDN_RECURRENT_CONV_B: u8 = 5;
    pub const GDN_RECURRENT_CONV_STATE_OUT: u8 = 9;
    pub const GDN_RECURRENT_PARAMS: u8 = 10;
    pub const GDN_PREP_CONV_STATE: u8 = 1;
    pub const GDN_PREP_CONV_B: u8 = 3;
    pub const GDN_PREP_CONV_STATE_OUT: u8 = 11;
    pub const GDN_PREP_PARAMS: u8 = 12;
    pub const ARGMAX_LOGITS: u8 = 0;
    pub const ARGMAX_NEXT_TOKEN: u8 = 1;
    pub const ARGMAX_PARAMS: u8 = 2;
    pub const ARGMAX_EOS_FLAG: u8 = 3;
    pub const MOE_SORT_PARAMS: u8 = 4;
    pub const MOE_ROWS_PARAMS: u8 = 3;
    pub const ROUTER_TOPK_PARAMS: u8 = 3;
    pub const EXPERT_COMBINE_PARAMS: u8 = 3;
    pub const SHARED_COMBINE_WIDTH: u8 = 4;
}

/// The tiny resident buffers constants live in, cached by
/// `(ordinal, index)` so a rebind overwrites contents in place.
#[derive(Debug, Default)]
pub struct ConstSlots {
    slots: HashMap<(u32, u8), Handle>,
}

impl ConstSlots {
    /// A fresh, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many const slots exist.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether none exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Write `value` into the slot for `(ordinal, index)` — allocating and
    /// binding it on first use, overwriting in place after.
    pub fn bind<T: Copy>(
        &mut self,
        context: &Context,
        tables: &mut Tables,
        ordinal: u32,
        index: u8,
        value: &T,
    ) -> Result<()> {
        let bytes = pod_bytes(std::slice::from_ref(value));
        match self.slots.get(&(ordinal, index)) {
            Some(handle) => {
                // SAFETY: const slots are host-owned; the GPU reads them
                // only mid-step, and rebinds happen at step boundaries.
                unsafe { handle.write(0, bytes)? };
            }
            None => {
                let handle = allocate(context, bytes.len() as u64, "const slot")?;
                // SAFETY: freshly allocated.
                unsafe { handle.write(0, bytes)? };
                tables.bind_address(context, ordinal, index as usize, handle.gpu_address())?;
                self.slots.insert((ordinal, index), handle);
            }
        }
        Ok(())
    }
}

fn bind_handle(
    context: &Context,
    tables: &mut Tables,
    ordinal: u32,
    index: u8,
    handle: &Handle,
) -> Result<()> {
    tables.bind_address(context, ordinal, index as usize, handle.gpu_address())
}

fn io(storage: &DecodeStorage, slot: IoSlot) -> Result<&Handle> {
    storage.io[slot as usize].as_ref().ok_or(Error::Create {
        what: "io slot",
        message: "an IO slot this DAG binds was not allocated".to_string(),
    })
}

/// Pass (a) and (b): load-once weights, then the kind-specific state, KV
/// and IO slots.
///
/// An unstaged weight is an error naming the tensor — a bind that fell
/// back would run the kernel against whatever the pool held at that
/// ordinal.
pub fn bind_decode_dag(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    dag: &[Dispatch],
    g: &DecodeGeometry,
    gdn_prep: bool,
) -> Result<()> {
    for d in dag {
        let ord = d.ordinal;
        for WeightBind { bind_index, tensor } in weight_binds(d.kind, d.layer, g, gdn_prep) {
            let handle = storage.weights.get(&tensor).ok_or_else(|| Error::Create {
                what: "weight bind",
                message: format!("unstaged weight {tensor}"),
            })?;
            bind_handle(context, tables, ord, bind_index, handle)?;
        }
        let layer = d.layer.map(|l| l as usize);
        let gdn_of = |layer: Option<usize>| -> Result<&super::storage::GdnState> {
            layer
                .and_then(|l| storage.gdn[l].as_ref())
                .ok_or(Error::Create {
                    what: "gdn state",
                    message: "a GDN dispatch outside a GDN layer".to_string(),
                })
        };
        match d.kind {
            Kernel::EmbedGather | Kernel::EmbedUntied => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::EMBED_TOKEN_ID,
                    io(storage, IoSlot::TokenId)?,
                )?;
            }
            Kernel::GdnPrep => {
                let s = gdn_of(layer)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_STATE,
                    &s.conv_state,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_STATE_OUT,
                    &s.conv_state_out,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_B,
                    &s.conv_bias_zero,
                )?;
            }
            Kernel::GdnCore => {
                let s = gdn_of(layer)?;
                if gdn_prep {
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_RECURRENT_CONV_STATE,
                        &s.conv_state,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_RECURRENT_STATE,
                        &s.recurrent_state,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_RECURRENT_CONV_STATE_OUT,
                        &s.conv_state_out,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_RECURRENT_CONV_B,
                        &s.conv_bias_zero,
                    )?;
                } else {
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_CORE_CONV_STATE,
                        &s.conv_state,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_CORE_RECURRENT_STATE,
                        &s.recurrent_state,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_CORE_CONV_STATE_OUT,
                        &s.conv_state_out,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        ord,
                        slot::GDN_CORE_CONV_B,
                        &s.conv_bias_zero,
                    )?;
                }
            }
            Kernel::KvAppend => {
                let kv = layer
                    .and_then(|l| storage.kv[l].as_ref())
                    .ok_or(Error::Create {
                        what: "kv slots",
                        message: "a KV append outside a full-attention layer".to_string(),
                    })?;
                bind_handle(context, tables, ord, slot::KV_APPEND_K_PAGES, &kv.k_pages)?;
                bind_handle(context, tables, ord, slot::KV_APPEND_V_PAGES, &kv.v_pages)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_POSITION,
                    io(storage, IoSlot::Position)?,
                )?;
            }
            Kernel::Sdpa | Kernel::GoSdpaSink => {
                let kv = layer
                    .and_then(|l| storage.kv[l].as_ref())
                    .ok_or(Error::Create {
                        what: "kv slots",
                        message: "an attention read outside a full-attention layer".to_string(),
                    })?;
                bind_handle(context, tables, ord, slot::SDPA_K, &kv.k_pages)?;
                bind_handle(context, tables, ord, slot::SDPA_V, &kv.v_pages)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::SDPA_N,
                    io(storage, IoSlot::SeqLen)?,
                )?;
            }
            Kernel::Rope | Kernel::RopeK => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::ROPE_POSITION,
                    io(storage, IoSlot::Position)?,
                )?;
            }
            // Logits ALWAYS land in the IO region — both kinds, because an
            // untied head that wrote nowhere is exactly as silent as one
            // that wrote here.
            Kernel::QmvLmHead | Kernel::LmHeadUntied => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::QMV_OUT,
                    io(storage, IoSlot::Logits)?,
                )?;
            }
            Kernel::Argmax => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_LOGITS,
                    io(storage, IoSlot::Logits)?,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_NEXT_TOKEN,
                    io(storage, IoSlot::NextToken)?,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_PARAMS,
                    &storage.argmax_params,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_EOS_FLAG,
                    &storage.eos_flag,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Swap every GDN dispatch's conv-state read/write binds to `parity`.
///
/// Even is the staged orientation (read `conv_state`, write
/// `conv_state_out`); Odd is the swap. The recurrent state needs no swap —
/// it is in place — and the parity is per SLOT in the C++'s slotted world;
/// at max_slots = 1 the whole pool shares one, which is what this rebinds.
pub fn bind_gdn_parity(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    dag: &[Dispatch],
    gdn_prep: bool,
    parity: crate::store::Parity,
) -> Result<()> {
    let swapped = parity == crate::store::Parity::Odd;
    for d in dag {
        let layer = d.layer.map(|l| l as usize);
        let Some(s) = layer.and_then(|l| storage.gdn[l].as_ref()) else {
            continue;
        };
        let (read, write) = if swapped {
            (&s.conv_state_out, &s.conv_state)
        } else {
            (&s.conv_state, &s.conv_state_out)
        };
        match d.kind {
            Kernel::GdnPrep => {
                bind_handle(context, tables, d.ordinal, slot::GDN_PREP_CONV_STATE, read)?;
                bind_handle(
                    context,
                    tables,
                    d.ordinal,
                    slot::GDN_PREP_CONV_STATE_OUT,
                    write,
                )?;
            }
            Kernel::GdnCore => {
                if gdn_prep {
                    bind_handle(
                        context,
                        tables,
                        d.ordinal,
                        slot::GDN_RECURRENT_CONV_STATE,
                        read,
                    )?;
                    bind_handle(
                        context,
                        tables,
                        d.ordinal,
                        slot::GDN_RECURRENT_CONV_STATE_OUT,
                        write,
                    )?;
                } else {
                    bind_handle(context, tables, d.ordinal, slot::GDN_CORE_CONV_STATE, read)?;
                    bind_handle(
                        context,
                        tables,
                        d.ordinal,
                        slot::GDN_CORE_CONV_STATE_OUT,
                        write,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Pass (c): the scratch pool, from the colouring's per-dispatch table.
pub fn bind_scratch(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    schedule: &ScratchSchedule,
) -> Result<()> {
    for (ordinal, binds) in schedule.per_dispatch.iter().enumerate() {
        for bind in binds {
            let buffer = storage
                .scratch
                .get(bind.color as usize)
                .ok_or(Error::Create {
                    what: "scratch bind",
                    message: "a colour past the pool".to_string(),
                })?;
            bind_handle(
                context,
                tables,
                u32::try_from(ordinal).expect("a DAG is hundreds of dispatches"),
                bind.bind_index,
                buffer,
            )?;
        }
    }
    Ok(())
}

/// The constants a batch width can change: the mixture's routing. Split
/// out so the fire path can rebind these alone — re-walking ~400
/// dispatches to rewrite the two dozen that could have changed pays the
/// whole table for the mixture's share of it.
#[allow(clippy::too_many_arguments)]
pub fn bind_token_consts(
    context: &Context,
    tables: &mut Tables,
    consts: &mut ConstSlots,
    dag: &[Dispatch],
    g: &DecodeGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    row_pitch: u32,
    routed_batched: bool,
) -> Result<()> {
    let rows = n_tokens.max(1);
    let pairs = rows * g.experts_per_token;
    let run = if routed_batched {
        RoutedProjection::Matmul
    } else {
        RoutedProjection::Matvec
    };
    let sorted = u32::try_from(moe_sorted_rows(g, tuning, rows, run)).unwrap_or(u32::MAX);
    let tile = if routed_batched {
        tuning.moe_tile_rows(pairs, g.n_experts)
    } else {
        1
    };
    for d in dag {
        match d.kind {
            Kernel::LlMoeSort | Kernel::LlMoeGather => {
                // One struct for both, so the sort's padding and the
                // gather's bounds cannot disagree about how many rows
                // exist.
                let params = MoeRouteParams {
                    n: pairs,
                    n_experts: g.n_experts,
                    experts_per_token: g.experts_per_token,
                    tile_rows: tile,
                    padded: sorted,
                    width: g.hidden,
                    x_pitch: row_pitch,
                };
                let index = if d.kind == Kernel::LlMoeSort {
                    slot::MOE_SORT_PARAMS
                } else {
                    slot::MOE_ROWS_PARAMS
                };
                consts.bind(context, tables, d.ordinal, index, &params)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Pass (d): every geometry-derived constant, by ordinal. Token-invariant
/// except the mixture's routing, which [`bind_token_consts`] owns and this
/// calls first — a DAG is never left half-bound.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn bind_decode_consts(
    context: &Context,
    tables: &mut Tables,
    consts: &mut ConstSlots,
    dag: &[Dispatch],
    g: &DecodeGeometry,
    tuning: &Tuning,
    max_ctx: u32,
    gdn_prep: bool,
    n_tokens: u32,
    row_pitch: u32,
) -> Result<()> {
    bind_token_consts(
        context, tables, consts, dag, g, tuning, n_tokens, row_pitch, true,
    )?;

    let rope_scale = 1.0f32;
    let rope_base = g.rope_theta.log2();
    let head_stride = u64::from(max_ctx) * u64::from(g.head_dim);
    let seq_stride = u64::from(g.head_dim);
    let gqa_factor = (g.n_q_heads / g.n_kv_heads.max(1)) as i32;
    let sdpa_scale = 1.0 / (g.head_dim as f32).sqrt();

    for d in dag {
        let ord = d.ordinal;
        let k = d.kind;
        if is_qmv(k, g) {
            let kn = qmv_kn(k, g);
            consts.bind(context, tables, ord, slot::QMV_K, &(kn.k as i32))?;
            consts.bind(context, tables, ord, slot::QMV_N, &(kn.n as i32))?;
            if is_routed(k) {
                // The routed matvec reads the SORTED stack: the sort made
                // the pair axis disappear, so these collapse to the dense
                // answer — no slot stride, a row pitch that is the input
                // width, one expert per row named by `row_expert`.
                consts.bind(context, tables, ord, slot::GO_QMV_X_SLOT_STRIDE, &0i32)?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::GO_QMV_X_ROW_STRIDE,
                    &(kn.k as i32),
                )?;
                consts.bind(context, tables, ord, slot::GO_QMV_SLOTS_PER_ROW, &1i32)?;
            }
            continue;
        }
        match k {
            Kernel::EmbedGather | Kernel::EmbedUntied => {
                consts.bind(context, tables, ord, slot::EMBED_HIDDEN, &(g.hidden as i32))?;
            }
            Kernel::Rms | Kernel::FfnRms | Kernel::FinalRms => {
                let params = RmsParams {
                    eps: g.eps,
                    axis_size: g.hidden,
                    w_stride: 1,
                    plus_one: 0,
                    gain: 1.0,
                };
                consts.bind(context, tables, ord, slot::RMS_PARAMS, &params)?;
            }
            Kernel::QNorm | Kernel::KNorm => {
                let params = RmsParams {
                    eps: g.eps,
                    axis_size: g.head_dim,
                    w_stride: 1,
                    plus_one: 0,
                    gain: 1.0,
                };
                consts.bind(context, tables, ord, slot::RMS_PARAMS, &params)?;
            }
            Kernel::GdnPrep | Kernel::GdnPrepSlotted => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_PARAMS,
                    &gdn_core_params(g),
                )?;
            }
            Kernel::GdnCoreSlotted => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::GDN_RECURRENT_PARAMS,
                    &gdn_core_params(g),
                )?;
            }
            Kernel::GdnCore => {
                let index = if gdn_prep {
                    slot::GDN_RECURRENT_PARAMS
                } else {
                    slot::GDN_CORE_PARAMS
                };
                consts.bind(context, tables, ord, index, &gdn_core_params(g))?;
            }
            Kernel::GatedRms => {
                let params = GatedRmsParams {
                    eps: g.eps,
                    vd: g.gdn_v_dim,
                };
                consts.bind(context, tables, ord, slot::GATED_RMS_PARAMS, &params)?;
            }
            Kernel::QSplit => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::QSPLIT_HEAD_DIM,
                    &(g.head_dim as i32),
                )?;
                // Packed by default; a prefill rebinds both to the arena's
                // pitch on row zero's table.
                consts.bind(context, tables, ord, slot::QSPLIT_QG_ROW_STRIDE, &0i32)?;
                consts.bind(context, tables, ord, slot::QSPLIT_OUT_ROW_STRIDE, &0i32)?;
            }
            Kernel::Rope | Kernel::RopeK => {
                consts.bind(context, tables, ord, slot::ROPE_SCALE, &rope_scale)?;
                consts.bind(context, tables, ord, slot::ROPE_BASE, &rope_base)?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::ROPE_HEAD_DIM,
                    &(g.head_dim as i32),
                )?;
            }
            Kernel::KvAppend => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_HEAD_DIM,
                    &(g.head_dim as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_K_SEQ_STRIDE,
                    &seq_stride,
                )?;
            }
            Kernel::KvAppendPaged => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_HEAD_DIM,
                    &(g.head_dim as i32),
                )?;
                // The two preserved M=1 ABI entries: unused by the paged
                // shader but bound so every declared slot has a value.
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_K_SEQ_STRIDE,
                    &seq_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_PAGE_SIZE,
                    &(g.kv_page_size as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_N_KV_HEADS,
                    &(g.n_kv_heads as i32),
                )?;
                // Packed by default; a per-token prefill rebinds row zero's
                // table to the arena's pitch.
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::KV_APPEND_PAGED_SRC_ROW_STRIDE,
                    &0i32,
                )?;
            }
            Kernel::SdpaPaged => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_PAGED_GQA_FACTOR,
                    &gqa_factor,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_PAGED_PAGE_SIZE,
                    &(g.kv_page_size as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_PAGED_N_KV_HEADS,
                    &(g.n_kv_heads as i32),
                )?;
                consts.bind(context, tables, ord, slot::SDPA_PAGED_SCALE, &sdpa_scale)?;
                // Full attention, but the shared kernel takes a window:
                // binding 0 says so; unbound would read one from
                // uninitialized memory — wrong attention, not a crash.
                consts.bind(context, tables, ord, slot::SDPA_PAGED_WINDOW, &0i32)?;
            }
            Kernel::Sdpa => {
                consts.bind(context, tables, ord, slot::SDPA_GQA_FACTOR, &gqa_factor)?;
                consts.bind(context, tables, ord, slot::SDPA_K_HEAD_STRIDE, &head_stride)?;
                consts.bind(context, tables, ord, slot::SDPA_K_SEQ_STRIDE, &seq_stride)?;
                consts.bind(context, tables, ord, slot::SDPA_V_HEAD_STRIDE, &head_stride)?;
                consts.bind(context, tables, ord, slot::SDPA_V_SEQ_STRIDE, &seq_stride)?;
                consts.bind(context, tables, ord, slot::SDPA_SCALE, &sdpa_scale)?;
            }
            Kernel::AttnGate => {
                consts.bind(context, tables, ord, slot::ATTN_GATE_ROW_STRIDE, &0i32)?;
            }
            // Width-invariant routing ends: the router reads one row and
            // writes k logits whatever the batch; the combine sums k slots
            // into one row.
            Kernel::GoRouterTopK => {
                let params = RouterParams {
                    n_experts: g.n_experts,
                    experts_per_token: g.experts_per_token,
                    softmax_over_all: u32::from(!g.norm_topk_prob),
                    logits_pitch: row_pitch,
                };
                consts.bind(context, tables, ord, slot::ROUTER_TOPK_PARAMS, &params)?;
            }
            Kernel::LlMoeCombine => {
                let params = ExpertCombineParams {
                    width: g.hidden,
                    experts_per_token: g.experts_per_token,
                    out_pitch: row_pitch,
                };
                consts.bind(context, tables, ord, slot::EXPERT_COMBINE_PARAMS, &params)?;
            }
            Kernel::LlSharedCombine => {
                // One value per ROW: the kernel indexes gate[row], so this
                // is the row width, not the element count.
                consts.bind(context, tables, ord, slot::SHARED_COMBINE_WIDTH, &g.hidden)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The M=1 PSO table: one pipeline per kind, plus the residual-fused GEMV.
#[derive(Clone, Debug, Default)]
pub struct StepPsos {
    /// Indexed by `Kernel::index()`.
    pub by_kind: HashMap<Kernel, Pso>,
    /// `affine_qmv_fast_residual`, for `fuse_residual` dispatches.
    pub qmv_residual: Option<Pso>,
}

/// Encode `[begin, end)` of the DAG: PSO, table, dispatch, barrier.
///
/// Barriers follow every dispatch except inside a concurrency run. Only
/// the segment that ends the step is special anywhere else (timing hooks
/// land with the attribution wiring).
pub fn encode_decode_step(
    encoder: &mut StepEncoder<'_>,
    tables: &Tables,
    dag: &[Dispatch],
    psos: &StepPsos,
    force_barriers: bool,
    begin: usize,
    end: usize,
) -> Result<()> {
    let run_ends = concurrent_run_ends(dag);
    let stop = end.min(dag.len());
    for i in begin..stop {
        let d = &dag[i];
        let pso = if d.fuse_residual {
            psos.qmv_residual.as_ref()
        } else {
            psos.by_kind.get(&d.kind)
        }
        .ok_or(Error::Create {
            what: "step pso",
            message: "a kind this DAG dispatches has no compiled pipeline".to_string(),
        })?;
        encoder.set_pipeline(pso);
        encoder.set_argument_table_for(tables, d.ordinal)?;
        encoder.dispatch(
            d.launch.grid.map(|v| v as usize),
            d.launch.tg.map(|v| v as usize),
        )?;
        if force_barriers || barrier_after(dag, i, &run_ends) {
            encoder.barrier(Visibility::Device);
        }
    }
    Ok(())
}
