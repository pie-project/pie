//! The decode step's resident storage: weights, KV, GDN state, IO and the
//! scratch pool, allocated and staged.
//!
//! The C++ (`loader/heap_bind.cpp::stage_decode_storage`) allocates each
//! region and stages the load plan's weights with its own transform loops.
//! This port allocates the same regions, but the weights come from
//! `model_loader::executor::host::execute_plan` — the engine `transcode.hpp`
//! was a mirror of (see `PARITY-LOADER.md`) — so every TileMap the plan
//! carries (MXFP4 decode, affine encode, casts) has already run by the time
//! a byte reaches the device buffer.
//!
//! Weights land in ONE shared region and each tensor is a [`Handle`] slice
//! of it, keyed by runtime name: hundreds of per-tensor buffers would be
//! hundreds of residency entries, and the argument tables want stable
//! GPU addresses, not a map that reallocates.
//!
//! What is deliberately NOT here yet, implementation-first (the ledger
//! carries each):
//!
//! * zero-copy mapping and weight streaming (`resolve_mappable`, the
//!   stream pack) — memory optimizations; every checkpoint loads correctly
//!   through the copy path, some just resident-larger;
//! * the expert-slab staging arm (`ExpertSlabRequest`) — the slab type
//!   exists (`crate::loader::ExpertSlab`); wiring it here needs the paging
//!   fire path;
//! * elastic sizing of KV/scratch (`alloc_zeroed`'s initial-commit
//!   parameter) — regions allocate at full size for now.

use std::collections::HashMap;
use std::path::Path;

use model_loader::executor::host::execute_plan;
use model_loader::plan::LoadPlan;

use crate::batch::{ArgmaxParams, DecodeGeometry, IO_SLOT_COUNT, IoSlot, SCRATCH_POOL};
use crate::region::Region as _;
use crate::{Error, Result};

use super::context::Context;
use super::fire::pod_bytes;
use super::handle::Handle;
use super::ring::allocate;

/// One GDN layer's persistent state.
#[derive(Debug)]
pub struct GdnState {
    /// The conv taps read this step (ping).
    pub conv_state: Handle,
    /// The conv taps written this step (pong) — distinct, because an
    /// in-place shift races the tap reads.
    pub conv_state_out: Handle,
    /// The recurrent state, in place: each (v-head, v-dim) row is owned by
    /// one threadgroup.
    pub recurrent_state: Handle,
    /// A zeroed bf16 conv bias: the checkpoint ships none, the kernel
    /// binds one.
    pub conv_bias_zero: Handle,
}

/// One full-attention layer's KV pages.
#[derive(Debug)]
pub struct KvSlots {
    /// The key pages.
    pub k_pages: Handle,
    /// The value pages.
    pub v_pages: Handle,
}

/// Everything the bind pass reads: the staged weights and every allocated
/// region. The C++ `BoundDecode`, minus the raw heap plan it no longer
/// needs to carry.
#[derive(Debug)]
pub struct DecodeStorage {
    /// The one region every staged tensor lives in.
    pub weights_region: Handle,
    /// Load-once weights, keyed by runtime tensor name (a tied lm_head
    /// appears once).
    pub weights: HashMap<String, Handle>,
    /// Per layer; `None` for full-attention layers.
    pub gdn: Vec<Option<GdnState>>,
    /// Per layer; `None` for GDN layers.
    pub kv: Vec<Option<KvSlots>>,
    /// The IO region, indexed by [`IoSlot`]; paged-only slots are `None`
    /// when paging is off.
    pub io: [Option<Handle>; IO_SLOT_COUNT],
    /// The device-argmax parameter block (allocated always; inert unless
    /// the DAG carries the argmax tail).
    pub argmax_params: Handle,
    /// `u32[max_tokens]`: 1 where the argmax winner is an EOS id.
    pub eos_flag: Handle,
    /// The activation ping-pong pool.
    pub scratch: Vec<Handle>,
}

fn alloc_zeroed(context: &Context, len: u64, what: &'static str) -> Result<Handle> {
    let handle = allocate(context, len.max(1), what)?;
    // SAFETY: the buffer is seconds old; no command buffer references it.
    unsafe { handle.zero(0, handle.len())? };
    Ok(handle)
}

/// Stage the plan's weights into one region, each tensor a named slice.
///
/// `execute_plan` runs the whole plan on the CPU — checkpoint reads and
/// every load-time transform — and hands back each finalized tensor's
/// bytes; this packs them 256-aligned into a single shared buffer.
pub fn stage_plan_weights(
    context: &Context,
    plan: &LoadPlan,
    snapshot_dir: &Path,
) -> Result<(Handle, HashMap<String, Handle>)> {
    // Arena mode: a Metal plan carries BulkExtentWrite, which addresses the
    // persistent arena by offset — the streaming path refuses it by design.
    // The arena IS the laid-out weights region, so it is copied whole and
    // each tensor becomes a slice at the offset the plan's own memory plan
    // assigned. Tensors the plan publishes outside the arena are appended
    // after it. The transient cost is one host arena beside the device
    // region; the zero-copy mapping the ledger defers would remove it.
    let storage = execute_plan(plan, snapshot_dir).map_err(|err| Error::Create {
        what: "staged weights",
        message: err.to_string(),
    })?;
    let arena_len = storage.arena.len() as u64;
    let names_by_tensor: HashMap<_, _> = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.id, tensor.name.as_str()))
        .collect();
    // (name, arena offset, bytes) for every tensor the arena holds.
    let mut in_arena: Vec<(&str, u64, u64)> = Vec::new();
    for buffer in &plan.buffers {
        let (Some(offset), Some(tensor)) = (buffer.persistent_offset, buffer.tensor) else {
            continue;
        };
        if let Some(name) = names_by_tensor.get(&tensor) {
            in_arena.push((name, offset, buffer.bytes));
        }
    }
    let arena_names: std::collections::HashSet<&str> =
        in_arena.iter().map(|(name, _, _)| *name).collect();
    let mut published: Vec<(&String, &Vec<u8>)> = storage
        .tensors
        .iter()
        .filter(|(name, _)| !arena_names.contains(name.as_str()))
        .collect();
    published.sort_by_key(|(name, _)| name.as_str());
    let extra: u64 = published
        .iter()
        .map(|(_, bytes)| (bytes.len() as u64).div_ceil(256) * 256)
        .sum();
    let region = allocate(context, (arena_len + extra).max(1), "weights region")?;
    // SAFETY: the region is freshly allocated; the GPU has no reference.
    unsafe { region.write(0, &storage.arena)? };
    let mut weights = HashMap::new();
    for (name, offset, bytes) in in_arena {
        weights.insert(name.to_string(), region.slice(offset, bytes)?);
    }
    let mut at = arena_len;
    for (name, bytes) in published {
        // SAFETY: as above; the offset stays inside the sized region.
        unsafe { region.write(at, bytes)? };
        weights.insert(name.clone(), region.slice(at, bytes.len() as u64)?);
        at += (bytes.len() as u64).div_ceil(256) * 256;
    }
    Ok((region, weights))
}

/// A scratch pool of `slots` buffers of `bytes` each — the dump path's
/// no-recycle pool, where every activation value keeps its own buffer so
/// nothing is overwritten before it is read back.
pub fn scratch_pool(context: &Context, slots: usize, bytes: u64) -> Result<Vec<Handle>> {
    (0..slots)
        .map(|_| alloc_zeroed(context, bytes, "scratch slot"))
        .collect()
}

/// Allocate every region the decode step touches and stage the weights.
///
/// Region sizes are the geometry's own arithmetic, matching
/// [`plan_heap`](crate::loader::plan_heap)'s formulas; the paged-KV IO
/// slots exist only when [`DecodeGeometry::paged_kv_enabled`].
pub fn stage_decode_storage(
    context: &Context,
    plan: &LoadPlan,
    snapshot_dir: &Path,
    g: &DecodeGeometry,
    max_ctx: u32,
    scratch_slot_bytes: u64,
) -> Result<DecodeStorage> {
    let (weights_region, weights) = stage_plan_weights(context, plan, snapshot_dir)?;

    // KV: k and v, each [n_kv_heads, max_ctx, head_dim] bf16, full-attention
    // layers only.
    let kv_one = u64::from(g.n_kv_heads) * u64::from(max_ctx) * u64::from(g.head_dim) * 2;
    let mut kv = Vec::with_capacity(g.n_layers as usize);
    for layer in 0..g.n_layers {
        kv.push(if g.is_full_attn(layer) {
            Some(KvSlots {
                k_pages: alloc_zeroed(context, kv_one, "kv k pages")?,
                v_pages: alloc_zeroed(context, kv_one, "kv v pages")?,
            })
        } else {
            None
        });
    }

    // GDN state: conv ping-pongs, recurrent is in place, and slots pack at
    // the natural per-slot stride — the slotted kernel indexes
    // slot * (Kc * CDIM) and slot * (Hv * Vd * Dk). At max_slots = 1 every
    // allocation is byte-identical to the sealed single-slot layout.
    let slots = u64::from(g.max_slots);
    let conv_state = u64::from(g.gdn_conv_dim) * u64::from(g.gdn_conv_k) * 4 * slots;
    let recur_state =
        u64::from(g.gdn_v_heads) * u64::from(g.gdn_v_dim) * u64::from(g.gdn_k_dim) * 4 * slots;
    let conv_bias = u64::from(g.gdn_conv_dim) * 2; // bf16, all zero
    let mut gdn = Vec::with_capacity(g.n_layers as usize);
    for layer in 0..g.n_layers {
        gdn.push(if g.is_full_attn(layer) {
            None
        } else {
            Some(GdnState {
                conv_state: alloc_zeroed(context, conv_state, "gdn conv state")?,
                conv_state_out: alloc_zeroed(context, conv_state, "gdn conv state out")?,
                recurrent_state: alloc_zeroed(context, recur_state, "gdn recurrent state")?,
                conv_bias_zero: alloc_zeroed(context, conv_bias, "gdn conv bias")?,
            })
        });
    }

    // The scratch pool: beta assigns X/Out per dispatch from the colouring.
    let mut scratch = Vec::with_capacity(SCRATCH_POOL);
    for _ in 0..SCRATCH_POOL {
        scratch.push(alloc_zeroed(context, scratch_slot_bytes, "scratch slot")?);
    }

    // IO: per-token scalars widen to u32[max_tokens]; logits stay the
    // historical f32[vocab] at M=1 and densify to bf16[N, vocab] paged.
    let tok = 4 * u64::from(g.max_tokens);
    let mut io: [Option<Handle>; IO_SLOT_COUNT] = std::array::from_fn(|_| None);
    let mut set = |slot: IoSlot, handle: Handle| {
        io[slot as usize] = Some(handle);
    };
    set(IoSlot::TokenId, alloc_zeroed(context, tok, "io token id")?);
    set(IoSlot::Position, alloc_zeroed(context, tok, "io position")?);
    set(IoSlot::SeqLen, alloc_zeroed(context, tok, "io seq len")?);
    let logits_bytes = if g.paged_kv_enabled {
        u64::from(g.vocab) * u64::from(g.max_tokens.max(1)) * 2
    } else {
        u64::from(g.vocab) * 4
    };
    set(
        IoSlot::Logits,
        alloc_zeroed(context, logits_bytes, "io logits")?,
    );
    set(
        IoSlot::NextToken,
        alloc_zeroed(context, tok, "io next token")?,
    );

    if g.paged_kv_enabled {
        let r = u64::from(g.max_requests.max(1));
        let n = u64::from(g.max_tokens.max(1));
        let refs = r * u64::from(g.total_pages.max(1));
        set(
            IoSlot::QoIndptr,
            alloc_zeroed(context, (r + 1) * 4, "io qo indptr")?,
        );
        set(
            IoSlot::KvPageIndptr,
            alloc_zeroed(context, (r + 1) * 4, "io page indptr")?,
        );
        set(
            IoSlot::KvPageIndices,
            alloc_zeroed(context, refs * 4, "io page indices")?,
        );
        set(
            IoSlot::KvLastPageLens,
            alloc_zeroed(context, r * 4, "io last page lens")?,
        );
        set(
            IoSlot::RsSlotIds,
            alloc_zeroed(context, r * 4, "io rs slots")?,
        );
        set(
            IoSlot::RsSlotFlags,
            alloc_zeroed(context, r, "io rs flags")?,
        );
        set(
            IoSlot::ReqOfToken,
            alloc_zeroed(context, n * 4, "io req of token")?,
        );
        set(
            IoSlot::SlotOfToken,
            alloc_zeroed(context, n * 4, "io slot of token")?,
        );
        set(IoSlot::WPage, alloc_zeroed(context, n * 4, "io w page")?);
        set(IoSlot::WOff, alloc_zeroed(context, n * 4, "io w off")?);
        let mask_stride = u64::from(g.total_pages.max(1)) * u64::from(g.kv_page_size.max(1));
        set(
            IoSlot::AttnMask,
            alloc_zeroed(context, n * mask_stride, "io attn mask")?,
        );
        set(
            IoSlot::AttnMaskStride,
            alloc_zeroed(context, 4, "io attn mask stride")?,
        );
        set(
            IoSlot::AttnMaskEnabled,
            alloc_zeroed(context, n, "io attn mask enabled")?,
        );
        // Which body rows the fire samples — at most all of them. The
        // row-gather compaction is the first reader; unallocated, it was
        // the one MB slot the paged block forgot.
        set(
            IoSlot::SampleRows,
            alloc_zeroed(context, n * 4, "io sample rows")?,
        );
    }

    // The argmax substrate: allocated always, inert unless the DAG carries
    // the tail. The executor rewrites vocab and EOS ids per generation.
    let argmax_params = alloc_zeroed(context, size_of::<ArgmaxParams>() as u64, "argmax params")?;
    let params = ArgmaxParams {
        vocab: g.vocab,
        n_eos: 0,
        eos_ids: [0; 8],
    };
    // SAFETY: freshly allocated; no GPU reference yet.
    unsafe { argmax_params.write(0, pod_bytes(std::slice::from_ref(&params)))? };
    let eos_flag = alloc_zeroed(context, tok, "eos flag")?;

    Ok(DecodeStorage {
        weights_region,
        weights,
        gdn,
        kv,
        io,
        argmax_params,
        eos_flag,
        scratch,
    })
}

/// Write one fire's CSR into the IO region — the pass every engine fire
/// runs between validation and encode. The mask row is disabled per
/// token and the mask stride zeroed: a fire that wants a wire mask
/// writes it after this, and a fire that does not must still write the
/// FLAGS, because the paged attention reads them per row and stale
/// flags consume a stale mask.
///
/// # Errors
///
/// An IO slot the geometry did not allocate.
///
/// # Safety
///
/// The previous fire must have retired: this writes buffers the GPU
/// reads.
pub unsafe fn write_fire_io(storage: &DecodeStorage, csr: &crate::batch::FireCsr) -> Result<()> {
    let io = |slot: IoSlot| -> Result<&Handle> {
        storage.io[slot as usize].as_ref().ok_or(Error::Create {
            what: "fire io slot",
            message: format!("IO slot {slot:?} was not allocated (is paging on?)"),
        })
    };
    let write_u32s = |slot: IoSlot, values: &[u32]| -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: the caller holds this function's contract.
        unsafe { io(slot)?.write(0, &bytes) }
    };
    write_u32s(IoSlot::TokenId, &csr.token_ids)?;
    write_u32s(IoSlot::Position, &csr.position_ids)?;
    // The ring attention's per-token KV extent; the paged kernels read
    // positions instead, but the slot is cheap and a mixed bind set may
    // read either.
    let seq_lens: Vec<u32> = csr.position_ids.iter().map(|p| p + 1).collect();
    write_u32s(IoSlot::SeqLen, &seq_lens)?;
    write_u32s(IoSlot::ReqOfToken, &csr.req_of_token)?;
    write_u32s(IoSlot::QoIndptr, &csr.qo_indptr)?;
    write_u32s(IoSlot::KvPageIndices, &csr.kv_page_indices)?;
    write_u32s(IoSlot::KvPageIndptr, &csr.kv_page_indptr)?;
    write_u32s(IoSlot::KvLastPageLens, &csr.kv_last_page_lens)?;
    write_u32s(IoSlot::WPage, &csr.w_page)?;
    write_u32s(IoSlot::WOff, &csr.w_off)?;
    write_u32s(IoSlot::SampleRows, &csr.sample_rows)?;
    write_u32s(IoSlot::AttnMaskStride, &[0])?;
    // SAFETY: as above; the flags are u8 rows.
    unsafe {
        io(IoSlot::AttnMaskEnabled)?.write(0, &vec![0u8; csr.token_ids.len()])?;
    }
    Ok(())
}
