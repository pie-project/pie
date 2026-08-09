//! The multibatch bind pass: paged KV, slotted GDN, and per-token offsets
//! into the shared IO buffers.
//!
//! The M=1 pass binds each IO buffer at its base; a prefill's per-token
//! DAGs share those buffers and each token's table binds them at ITS row —
//! [`MbBindOffsets`] carries the row and the logits offset, and a bind is
//! `gpu_address + offset`, which the argument tables hold as one number.
//!
//! [`bind_gdn_conv_parity`] carries its own race story: `gdn_prep` runs
//! one threadgroup per VALUE head, and `rep = Hv / Hk` value heads share a
//! key head — that head's conv window is READ by all `rep` of them and
//! WRITTEN by the first. Alias the output over the input and the writer
//! shifts the window while its siblings are still reading it: both
//! outcomes are finite numbers, so it reports as a fleet member quietly
//! disagreeing with its identical neighbours rather than as a fault.
//! `rep` is 1 on Qwen3.6-35B-A3B and 3 on Qwen3.6-27B, which is why only
//! the 27B showed it.

use crate::batch::{DecodeGeometry, Dispatch, IoSlot, Kernel, WeightBind, weight_binds};
use crate::store::Parity;
use crate::{Error, Result};

use super::bind::slot;
use super::context::Context;
use super::handle::Handle;
use super::storage::DecodeStorage;
use super::tables::Tables;

/// The multibatch-only bind ordinals this pass adds to `bind::slot`.
#[allow(missing_docs)] // names are the C++ bind:: constants, one-to-one
pub mod slot_mb {
    pub const GDN_PREP_SLOT_OF_TOKEN: u8 = 13;
    pub const GDN_RECURRENT_SLOT_OF_TOKEN: u8 = 11;
    pub const KV_APPEND_PAGED_K_PAGES: u8 = 2;
    pub const KV_APPEND_PAGED_V_PAGES: u8 = 3;
    pub const KV_APPEND_PAGED_POSITION_IDS: u8 = 4;
    pub const KV_APPEND_PAGED_PAGE_INDICES: u8 = 8;
    pub const KV_APPEND_PAGED_PAGE_INDPTR: u8 = 9;
    pub const KV_APPEND_PAGED_REQ_OF_TOKEN: u8 = 11;
    pub const KV_APPEND_PAGED_W_PAGE: u8 = 13;
    pub const KV_APPEND_PAGED_W_OFF: u8 = 14;
    pub const SDPA_PAGED_K_PAGES: u8 = 1;
    pub const SDPA_PAGED_V_PAGES: u8 = 2;
    pub const SDPA_PAGED_POSITION_IDS: u8 = 5;
    pub const SDPA_PAGED_REQ_OF_TOKEN: u8 = 6;
    pub const SDPA_PAGED_PAGE_INDICES: u8 = 7;
    pub const SDPA_PAGED_PAGE_INDPTR: u8 = 8;
    pub const SDPA_PAGED_ATTN_MASK: u8 = 12;
    pub const SDPA_PAGED_ATTN_MASK_STRIDE: u8 = 13;
    pub const SDPA_PAGED_ATTN_MASK_ENABLED: u8 = 14;
}

/// Where this DAG's per-token binds land in the shared IO buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MbBindOffsets {
    /// The token row the per-row scalars are read at.
    pub token_row: u64,
    /// The byte offset the lm_head writes this row's logits at.
    pub logits_bytes: u64,
}

/// One dense attention-mask row's pitch, in bytes.
#[must_use]
pub fn paged_attention_mask_pitch_bytes(g: &DecodeGeometry) -> u64 {
    u64::from(g.total_pages.max(1)) * u64::from(g.kv_page_size.max(1))
}

fn bind_at(
    context: &Context,
    tables: &mut Tables,
    ordinal: u32,
    index: u8,
    handle: &Handle,
    offset: u64,
) -> Result<()> {
    tables.bind_address(
        context,
        ordinal,
        index as usize,
        handle.gpu_address() + offset,
    )
}

fn io(storage: &DecodeStorage, slot: IoSlot) -> Result<&Handle> {
    storage.io[slot as usize].as_ref().ok_or(Error::Create {
        what: "mb io slot",
        message: "an IO slot this MB DAG binds was not allocated (is paging on?)".to_string(),
    })
}

/// Bind a multibatch DAG's weights, state, paged KV and offset IO.
///
/// Reacts to each dispatch's kind, like the M=1 pass; the scratch and
/// constant passes are shared with M=1 and run separately.
#[allow(clippy::too_many_lines)] // one walk, one table
pub fn bind_decode_dag_mb(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    dag: &[Dispatch],
    g: &DecodeGeometry,
    gdn_prep: bool,
    offsets: MbBindOffsets,
) -> Result<()> {
    let row_u32 = offsets.token_row * 4;
    for d in dag {
        let ord = d.ordinal;
        for WeightBind { bind_index, tensor } in weight_binds(d.kind, d.layer, g, gdn_prep) {
            let handle = storage.weights.get(&tensor).ok_or_else(|| Error::Create {
                what: "mb weight bind",
                message: format!("unstaged weight {tensor}"),
            })?;
            bind_at(context, tables, ord, bind_index, handle, 0)?;
        }
        let layer = d.layer.map(|l| l as usize);
        let gdn = |layer: Option<usize>| -> Result<&super::storage::GdnState> {
            layer
                .and_then(|l| storage.gdn[l].as_ref())
                .ok_or(Error::Create {
                    what: "mb gdn state",
                    message: "a slotted GDN dispatch outside a GDN layer".to_string(),
                })
        };
        let kv = |layer: Option<usize>| -> Result<&super::storage::KvSlots> {
            layer
                .and_then(|l| storage.kv[l].as_ref())
                .ok_or(Error::Create {
                    what: "mb kv slots",
                    message: "a paged attention dispatch outside a full-attention layer"
                        .to_string(),
                })
        };
        match d.kind {
            Kernel::EmbedUntied | Kernel::EmbedGather => {
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::EMBED_TOKEN_ID,
                    io(storage, IoSlot::TokenId)?,
                    row_u32,
                )?;
            }
            Kernel::GdnPrepSlotted => {
                let s = gdn(layer)?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_STATE,
                    &s.conv_state,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_STATE_OUT,
                    &s.conv_state_out,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_PREP_CONV_B,
                    &s.conv_bias_zero,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::GDN_PREP_SLOT_OF_TOKEN,
                    io(storage, IoSlot::SlotOfToken)?,
                    row_u32,
                )?;
            }
            Kernel::GdnCoreSlotted => {
                let s = gdn(layer)?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_RECURRENT_CONV_STATE,
                    &s.conv_state,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_RECURRENT_STATE,
                    &s.recurrent_state,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_RECURRENT_CONV_STATE_OUT,
                    &s.conv_state_out,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::GDN_RECURRENT_CONV_B,
                    &s.conv_bias_zero,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::GDN_RECURRENT_SLOT_OF_TOKEN,
                    io(storage, IoSlot::SlotOfToken)?,
                    row_u32,
                )?;
            }
            Kernel::KvAppendPaged => {
                let kv = kv(layer)?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_K_PAGES,
                    &kv.k_pages,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_V_PAGES,
                    &kv.v_pages,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_POSITION_IDS,
                    io(storage, IoSlot::Position)?,
                    row_u32,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_PAGE_INDICES,
                    io(storage, IoSlot::KvPageIndices)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_PAGE_INDPTR,
                    io(storage, IoSlot::KvPageIndptr)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_REQ_OF_TOKEN,
                    io(storage, IoSlot::ReqOfToken)?,
                    row_u32,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_W_PAGE,
                    io(storage, IoSlot::WPage)?,
                    row_u32,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::KV_APPEND_PAGED_W_OFF,
                    io(storage, IoSlot::WOff)?,
                    row_u32,
                )?;
            }
            // The sink form shares the paged ABI's whole IO set; its one
            // extra tensor (the learned sinks) arrives with the weight
            // walk above, at the paged ABI's OWN index — see
            // `slot::SDPA_PAGED_SINKS` for the collision the C++ remapped.
            Kernel::SdpaPaged | Kernel::GoSdpaSinkPaged => {
                let kv = kv(layer)?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_K_PAGES,
                    &kv.k_pages,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_V_PAGES,
                    &kv.v_pages,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_POSITION_IDS,
                    io(storage, IoSlot::Position)?,
                    row_u32,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_REQ_OF_TOKEN,
                    io(storage, IoSlot::ReqOfToken)?,
                    row_u32,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_PAGE_INDICES,
                    io(storage, IoSlot::KvPageIndices)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_PAGE_INDPTR,
                    io(storage, IoSlot::KvPageIndptr)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_ATTN_MASK,
                    io(storage, IoSlot::AttnMask)?,
                    offsets.token_row * paged_attention_mask_pitch_bytes(g),
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_ATTN_MASK_STRIDE,
                    io(storage, IoSlot::AttnMaskStride)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot_mb::SDPA_PAGED_ATTN_MASK_ENABLED,
                    io(storage, IoSlot::AttnMaskEnabled)?,
                    offsets.token_row,
                )?;
            }
            Kernel::Rope | Kernel::RopeK => {
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ROPE_POSITION,
                    io(storage, IoSlot::Position)?,
                    row_u32,
                )?;
            }
            // The compaction reads which body rows the fire samples; the
            // list is whole-fire, so no per-token offset applies.
            Kernel::G4RowGather => {
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ROW_GATHER_ROWS,
                    io(storage, IoSlot::SampleRows)?,
                    0,
                )?;
            }
            Kernel::QmvLmHead | Kernel::LmHeadUntied => {
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::QMV_OUT,
                    io(storage, IoSlot::Logits)?,
                    offsets.logits_bytes,
                )?;
            }
            Kernel::Argmax => {
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_LOGITS,
                    io(storage, IoSlot::Logits)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_NEXT_TOKEN,
                    io(storage, IoSlot::NextToken)?,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_PARAMS,
                    &storage.argmax_params,
                    0,
                )?;
                bind_at(
                    context,
                    tables,
                    ord,
                    slot::ARGMAX_EOS_FLAG,
                    &storage.eos_flag,
                    0,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Point the slotted GDN pair's conv ping-pong at the half holding the
/// history this fire will READ. See the module docs for the shared-window
/// race that makes the two halves distinct buffers.
pub fn bind_gdn_conv_parity(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    dag: &[Dispatch],
    parity: Parity,
) -> Result<()> {
    let even = parity == Parity::Even;
    for d in dag {
        if d.kind != Kernel::GdnPrepSlotted && d.kind != Kernel::GdnCoreSlotted {
            continue;
        }
        let Some(s) = d.layer.and_then(|l| storage.gdn[l as usize].as_ref()) else {
            continue;
        };
        let (input, output) = if even {
            (&s.conv_state, &s.conv_state_out)
        } else {
            (&s.conv_state_out, &s.conv_state)
        };
        if d.kind == Kernel::GdnPrepSlotted {
            bind_at(
                context,
                tables,
                d.ordinal,
                slot::GDN_PREP_CONV_STATE,
                input,
                0,
            )?;
            bind_at(
                context,
                tables,
                d.ordinal,
                slot::GDN_PREP_CONV_STATE_OUT,
                output,
                0,
            )?;
        } else {
            bind_at(
                context,
                tables,
                d.ordinal,
                slot::GDN_RECURRENT_CONV_STATE,
                input,
                0,
            )?;
            bind_at(
                context,
                tables,
                d.ordinal,
                slot::GDN_RECURRENT_CONV_STATE_OUT,
                output,
                0,
            )?;
        }
    }
    Ok(())
}
