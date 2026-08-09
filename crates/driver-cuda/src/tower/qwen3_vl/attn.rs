//! The vision tower's block-diagonal attention — `vis_helpers.cpp`'s half
//! that is not a GEMM.
//!
//! The port of `driver-cuda/csrc/vision/vis_helpers.cpp:64-136`
//! (`VisAttnRes`, `ensure_ws`, `qwen3vl_vis_attn`). Non-causal MHA over
//! `num_seqs` independent images: image `i` has `seqlens[i]` patches at row
//! offset `Σ_{j<i} seqlens[j]` in `q`/`k`/`v` (`[Σ, NH, HEAD]`), and one
//! FlashInfer multi-sequence prefill makes each query attend only within its
//! own image. `q`/`k` already carry the 2-D RoPE — `k_split_rope_qkv` applied
//! it — and the softmax scale is applied INSIDE flashinfer, which is why
//! `run`'s layer body never scales the scores.
//!
//! # THIS IS THE ONE CALLEE THAT IS STILL C++, AND IT IS NOT THIS STEP'S
//!
//! `pie_k_attn_dispatch_attention_flashinfer_prefill_bf16` and the
//! `pie_x_*_prefill_plan` trio are `driver-cuda/csrc/attn/`'s — the FA2
//! lattice — reached across the C ABI exactly as `fire/attn_score.rs` reaches
//! its own driver-owned launcher. **North star §5 step 8 retires them**, and
//! until it does, a tower whose walk is Rust and whose attention crosses one
//! FFI edge is the complete result of THIS step: the edge is a call, not a
//! host program, and no `<<<>>>` crosses it.
//!
//! `execution.rs` wrote this down before the port: *"`ffi::pie_k_attn_
//! dispatch_attention_flashinfer_prefill_bf16` already exists, so
//! `vis_helpers.cpp:131` needs no forwarder."* It did not, and this module is
//! the call it was talking about.
//!
//! # What the port UNDOES
//!
//! `vis_helpers.cpp`'s header states the one change it made to the adapter it
//! was copied from: *"the dedicated attention workspace is raw-allocated here
//! (32 MiB float / 16 MiB int / a pinned int mirror — the same sizes the
//! driver passes its `AttentionWorkspace`), because the owning class is a
//! DRIVER object and this archive only knows the view."* That boundary is
//! what is being deleted, so the raw `cudaMalloc` trio goes back to being an
//! [`AttentionWorkspace`] — the driver object it was imitating — at the same
//! three sizes.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::bind::PrefillPlan;
use crate::bind::abi::{AttentionWorkspaceView, ffi};
use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps, StagingOps};
use crate::{Error, Result};

/// This module's name in a refusal — `super::WHO`'s reason, one file down.
const WHO: &str = "qwen3vl_vis_attn";

/// The attention page size — `vis_helpers.cpp:56`'s `kVisPageSize`.
///
/// The KV view is paged even though nothing here is a KV cache: a prefill
/// plan is what FlashInfer offers for block-diagonal attention, and a plan
/// takes pages. `run`'s callers must keep each image's patch count a multiple
/// of this when they batch images into ONE call, which is what
/// `qwen3_vl_tower.cu:421`'s `constexpr int PS=16;  // attention page size;
/// per-image rows must be a multiple` records. The live per-image arm never
/// batches, so the constraint is vacuous there — the last page is simply
/// short, which `klpl` states.
const PAGE_SIZE: i32 = 16;

/// The workspace, as the driver's own object rather than three `cudaMalloc`s.
type Workspace = AttentionWorkspace<<LiveStagingOps as StagingOps>::Event>;

/// The tower's dedicated attention resources — `vis_helpers.cpp:38-50`'s
/// `VisAttnRes`, field for field.
///
/// Dedicated, and that is the whole design: the fire's own workspace and plan
/// belong to the decoder's attention and are planned for the decoder's
/// geometry, so a tower planning over them mid-forward would clobber a plan
/// the LLM layers are about to dispatch against. The C++ allocated a second
/// set for that reason and this keeps it.
struct Res {
    /// 32 MiB float / 16 MiB int / one pinned staging slot — `ensure_ws`.
    ws: Workspace,
    /// FlashInfer's prefill plan cache, `make_prefill_plan()`'s.
    plan: PrefillPlan,
    /// Owns the four index buffers below.
    alloc: Allocator,
    /// `(num_seqs, total_tokens, seqlens[0], NH, HEAD)` — the plan/index
    /// signature. Vision images in one batch are equal-sized, so this
    /// captures the shape, and a `None` is "never planned".
    sig: Option<[i32; 5]>,
    /// Query offsets, `[num_seqs + 1]`.
    qo: Option<DeviceBuffer>,
    /// Page offsets, `[num_seqs + 1]`.
    kvpi: Option<DeviceBuffer>,
    /// The identity page map, `[total_pages]`.
    kvidx: Option<DeviceBuffer>,
    /// Last-page occupancies, `[num_seqs]`.
    klpl: Option<DeviceBuffer>,
}

// SAFETY: every field is either plain data or a device/pinned-host address.
// `Workspace` holds four raw pointers and a `cudaEvent_t`, `PrefillPlan` holds
// one opaque handle, and a `DeviceBuffer` is already `Send` — none of them is
// thread-affine, and CUDA contexts are per-process rather than per-thread in
// the runtime API this driver uses. The justification `bind/quant_gemm.rs:484`
// makes for `DequantWeightCache`, for the same kind of value: a `static` that
// exists because the resource must outlive any one call.
unsafe impl Send for Res {}

/// The one instance, raised on first use — `vis_helpers.cpp:51`'s
/// `VisAttnRes& vis_attn_res() { static VisAttnRes v; return v; }` with the
/// `std::mutex` that lived INSIDE it hoisted out, because in Rust the lock is
/// what hands out the `&mut`.
fn res() -> &'static Mutex<Option<Res>> {
    static RES: OnceLock<Mutex<Option<Res>>> = OnceLock::new();
    RES.get_or_init(|| Mutex::new(None))
}

/// `ensure_ws` — the workspace and the plan, once per process.
///
/// The three sizes are `vis_helpers.cpp:58-59` (`kFloatBytes = 32u << 20`,
/// `kIntBytes = 16u << 20`, the pinned mirror sized to the int budget), and
/// they are the sizes the driver passes its own `AttentionWorkspace` — which
/// is now the type holding them, so the equality is structural instead of a
/// comment.
///
/// ONE staging slot. The pool rotates for run-ahead, and a tower dispatches
/// its plan on the stream that just built it, inside one mutex, with no
/// second step in flight — so the slot count is the C++'s single
/// `page_locked_int` and not the fire's depth.
fn ensure(slot: &mut Option<Res>) -> Result<&mut Res> {
    if slot.is_none() {
        let mut ops = LiveStagingOps;
        let ws = Workspace::allocate(&mut ops, 32 << 20, 16 << 20, 1).map_err(|why| {
            // `vis_helpers.cpp:69`'s
            // `throw std::runtime_error("qwen3vl_vis_attn: workspace allocation failed")`,
            // as a value. `execution.rs`'s `WALKED` entry cited that throw as
            // one of this walk's three refusals: "the tower declines rather
            // than attending over an unallocated scratch".
            Error::invalid(WHO, format!("workspace allocation failed: {why:?}"))
        })?;
        *slot = Some(Res {
            ws,
            plan: PrefillPlan::new(),
            alloc: Allocator::new(),
            sig: None,
            qo: None,
            kvpi: None,
            kvidx: None,
            klpl: None,
        });
    }
    slot.as_mut()
        .ok_or_else(|| Error::invalid(WHO, "the attention resources were not raised"))
}

/// Upload a `u32` run into a held buffer, replacing it when it is too small.
fn upload(
    alloc: &Allocator,
    held: &mut Option<DeviceBuffer>,
    src: &[u32],
    stream: StreamRef<'_>,
) -> Result<()> {
    // SAFETY: `u32` has no padding and no invalid bit patterns, so the run is
    // readable as bytes for its own length. `Scratch::upload_f32s` makes the
    // same reinterpretation for the same reason.
    let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
    if held.as_ref().is_none_or(|b| b.len() < bytes.len()) {
        // Dropped BEFORE the new allocation, not after: `cudaFree` and
        // `cudaMalloc` both synchronise the device, and holding two at once
        // would double the peak for no reason. The C++ freed all four
        // unconditionally on a signature change (`vis_helpers.cpp:110-113`).
        *held = None;
        *held = Some(alloc.alloc(bytes.len().max(1))?);
    }
    let buffer = held
        .as_mut()
        .ok_or_else(|| Error::invalid(WHO, "an index buffer went missing"))?;
    buffer.copy_from_host(bytes, stream)
}

/// Non-causal block-diagonal MHA over `seqlens.len()` images.
///
/// `q` is read, `k`/`v` are the launcher's mutable page planes (it takes them
/// non-const; nothing here writes them), and `o` receives `[Σ, NH, HEAD]`.
///
/// # Errors
///
/// The workspace would not allocate, an index buffer would not upload, or a
/// shape that cannot be planned — each the `throw` the C++ made at the same
/// point, as a value.
pub(super) fn attend(
    q: *const c_void,
    k: *mut c_void,
    v: *mut c_void,
    o: *mut c_void,
    seqlens: &[i32],
    heads: i32,
    head_dim: i32,
    stream: StreamRef<'_>,
) -> Result<()> {
    let num_seqs = i32::try_from(seqlens.len())
        .map_err(|_| Error::invalid(WHO, "more images than an int can count"))?;
    if num_seqs == 0 {
        return Ok(());
    }
    let mut guard = res().lock().unwrap_or_else(|e| e.into_inner());
    let st = ensure(&mut guard)?;

    // `vis_helpers.cpp:89-99` — the three host CSRs and the last-page
    // occupancies, built from the per-image patch counts.
    let mut total: i32 = 0;
    let mut qo = vec![0u32; seqlens.len() + 1];
    let mut kvpi = vec![0u32; seqlens.len() + 1];
    let mut klpl = vec![0u32; seqlens.len()];
    for (i, &len) in seqlens.iter().enumerate() {
        if len <= 0 {
            return Err(Error::invalid(
                WHO,
                format!("image {i} has {len} patches, which is not a sequence"),
            ));
        }
        total = total
            .checked_add(len)
            .ok_or_else(|| Error::invalid(WHO, "the batched patch count overflowed an int"))?;
        let pages = len.div_ceil(PAGE_SIZE);
        qo[i + 1] = qo[i] + len.unsigned_abs();
        kvpi[i + 1] = kvpi[i] + pages.unsigned_abs();
        klpl[i] = (len - (pages - 1) * PAGE_SIZE).unsigned_abs();
    }
    let total_pages = kvpi[seqlens.len()];
    let sig = [num_seqs, total, seqlens[0], heads, head_dim];

    if st.sig != Some(sig) {
        // `vis_helpers.cpp:104-127` — the four uploads and the plan, together,
        // because the plan is what reads the shape they describe.
        let kvidx: Vec<u32> = (0..total_pages).collect();
        upload(&st.alloc, &mut st.qo, &qo, stream)?;
        upload(&st.alloc, &mut st.kvpi, &kvpi, stream)?;
        upload(&st.alloc, &mut st.kvidx, &kvidx, stream)?;
        upload(&st.alloc, &mut st.klpl, &klpl, stream)?;
        let view: AttentionWorkspaceView = st.ws.view();
        // NOT `PrefillPlan::plan_prefill`, and the difference is the whole
        // tower: that helper passes `causal_mask = true`, because its caller
        // is a decoder prefill where a token may not see its future. A ViT is
        // bidirectional — every patch attends to every patch of its own image
        // — so this calls the entry directly with the C++'s six flags:
        // `enable_cuda_graph=false, window_left=-1, full_attention_variant=
        // false, hnd_layout=false, causal_mask=false` (`vis_helpers.cpp:
        // 121-126`), plus the two the declaration carries and the C++ left
        // defaulted, `custom_mask` and `wants_prefill_score`, both false.
        //
        // The workspace's plan-update fence (`begin_plan_update` /
        // `end_plan_update`) is NOT taken, exactly as the C++ did not take
        // it. One staging slot, no rotation, and the upload the planner
        // queues is stream-ordered behind the dispatch that preceded it on
        // the same stream. The residual hazard — the CPU writing the pinned
        // mirror while an in-flight dispatch still reads the device copy — is
        // the C++'s and is inherited, not introduced; it is bounded by the
        // fact that a re-plan happens only when the image SHAPE changes.
        //
        // SAFETY: `plan` is this module's own live handle, the three CSR
        // pointers address host `Vec`s that outlive the call, and the view's
        // pointers are the workspace's own. `stream` is live for the borrow.
        unsafe {
            ffi::pie_x_plan_attention_flashinfer_prefill_bf16(
                st.plan.as_ptr(),
                qo.as_ptr(),
                kvpi.as_ptr(),
                klpl.as_ptr(),
                total,
                num_seqs,
                heads,
                heads,
                head_dim,
                PAGE_SIZE,
                view,
                stream.as_raw().cast(),
                false,
                -1,
                false,
                false,
                false,
                false,
                false,
            );
        }
        st.sig = Some(sig);
    }

    let (Some(qo_d), Some(kvpi_d), Some(kvidx_d), Some(klpl_d)) = (
        st.qo.as_ref(),
        st.kvpi.as_ref(),
        st.kvidx.as_ref(),
        st.klpl.as_ref(),
    ) else {
        return Err(Error::invalid(
            WHO,
            "the plan is current but its index buffers are not held",
        ));
    };
    #[allow(clippy::cast_precision_loss)]
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let view: AttentionWorkspaceView = st.ws.view();
    // `vis_helpers.cpp:131-133`. THE ONE FFI EDGE THIS TOWER STILL CROSSES —
    // see the module header, and north star §5 step 8 for what closes it.
    // The argument order is the row's (`table::attn`'s `flashinfer_prefill`),
    // which is where `kv_page_indices` precedes `kv_page_indptr`: the C++
    // call passes `st.kvidx_d` then `st.kvpi_d` for exactly that reason.
    //
    // SAFETY: `q`/`k`/`v`/`o` are the walk's scratch, live until the caller
    // synchronises; the four index buffers are this module's own and were
    // uploaded on this stream; the view is the workspace's; `lse_out` is null,
    // which the launcher reads as "do not write log-sum-exp".
    unsafe {
        ffi::pie_k_attn_dispatch_attention_flashinfer_prefill_bf16(
            st.plan.as_ptr().cast_const(),
            q,
            k,
            v,
            o,
            qo_d.as_ptr().cast::<u32>().cast_const(),
            kvidx_d.as_ptr().cast::<u32>().cast_const(),
            kvpi_d.as_ptr().cast::<u32>().cast_const(),
            klpl_d.as_ptr().cast::<u32>().cast_const(),
            view,
            stream.as_raw().cast(),
            0.0,
            sm_scale,
            std::ptr::null_mut(),
        );
    }
    Ok(())
}
