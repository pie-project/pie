//! What happens when a trace states one of the six FlashInfer FA2 dispatches.
//!
//! # Why these are a file of their own and not six more arms in [`attn`]
//!
//! [`super::attn`]'s arms read operands, widths and the KV layer, and each is
//! three lines. These six read a vocabulary nothing else in the tree touches:
//! the two PLAN CACHES and the full-attention plan beside them, the two
//! attention workspaces, the score sink and its CSR, the custom mask pair and
//! — for the planless form — the HOST mirrors of the CSR, which the planner
//! walks on the CPU. They also do more than resolve: each uploads a plan
//! descriptor to the device before the launch that indexes it, and three of
//! them widen a quantised KV layer first. The routines they call are
//! [`kernels_cuda::attn::fa2`]'s, not `x::attn`'s, and the file boundary is
//! the same boundary.
//!
//! # What moved and what did not
//!
//! `fire::flashinfer_fa2_dispatch`'s six `attn_dispatch_*` entry points and
//! `bind/mod.rs`'s six hand arms were the two halves of what is now one arm
//! each. What each arm still does here, and could not do anywhere else:
//!
//! 1. **The plan.** [`crate::bind::attn_plan`] picks between a family's two
//!    decode plans on `window_of(spec, ..) == -1`, because two-kind families
//!    keep a second decode plan for their full-attention layers and the two
//!    kinds disagree on head dim. The cache is then DESTRUCTURED --
//!    [`fa2d::decode_plan_of`] -- into a `Copy` value a routine can take.
//! 2. **The H2D.** `attention_flashinfer.cu:193-198`, issued immediately
//!    before the fire that reads it: the grid is `plan_info.padded_batch_size`
//!    and the work list the grid indexes is the bytes being uploaded. It is
//!    here rather than in the routine because a routine's `Ctx` has no copy
//!    engine on it, and because the bytes belong to a cache that never
//!    crosses.
//! 3. **The dequant prelude.** `KvWidth::BF16` is the only width the lattice
//!    instantiates, so a quantised layer is widened into its bf16 mirrors
//!    before FA2 sees a page. It stays on this side because
//!    `dequant_kv_cache_layer_to_bf16_active` takes a `&KvLayer`, which has no
//!    `Arg` impl -- a trace statement cannot supply a layer view -- and it is
//!    the same reason `attn::write_kv_to_pages` has not crossed either.
//!
//! A layer whose dtype `KvDType` does not name skips the prelude and the
//! attention still runs, which is the shape the `Declined` the entry points
//! consumed with `let _ =` already had.

use core::ffi::c_void;

use kernels::Refusal;
use kernels::routine::Env;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::attn::kv_paged;
use kernels_cuda::attn::fa2::{
    attention_flashinfer_prefill, dispatch_attention_flashinfer_decode,
    dispatch_attention_flashinfer_decode_capture, dispatch_attention_flashinfer_prefill_bf16,
    dispatch_attention_flashinfer_prefill_capture_bf16,
    dispatch_attention_flashinfer_prefill_custom,
};

use crate::bind::AttnCtx;
use crate::bind::abi::AttentionWorkspaceView;
use kernels_cuda::attn::fa2::plan as ffa2;
use kernels_cuda::attn::fa2::dispatch as fa2d;

use super::super::cx::Cx;
use super::Bound;

/// The two join facts no `Source` could name, refused rather than ignored.
///
/// `spec.aux` and `spec.per_head_dim` are facts about the STATEMENT that
/// change the arithmetic rather than the operands, and this launcher has
/// neither reading — so binding one anyway would read right and compute wrong.
/// The generated guard made the branch not match; here nothing else serves
/// these symbols, so it is a refusal.
fn no_join_extras(cx: &Cx<'_>) -> Result<(), Refusal> {
    let spec = cx.spec();
    if !spec.aux.is_empty() || spec.per_head_dim.is_some() {
        return Err(Refusal::Unstated {
            what: "an FA2 dispatch without an aux value or a per-head reading",
        });
    }
    Ok(())
}

/// `Source::Or(&Out(0), &Attn("o_out"))` — the stated result if the statement
/// declares one, the guard-owned arena slot if it does not.
fn o_or(cx: &Cx<'_>, a: &AttnCtx) -> Result<*mut bf16, Refusal> {
    if let Ok(p) = cx.arg_out(0) {
        return Ok(p.cast::<bf16>());
    }
    if a.o_out.is_null() {
        return Err(Refusal::Unstated { what: "somewhere for the attention output to land" });
    }
    Ok(a.o_out.cast::<bf16>())
}

/// `Source::Or(&Out(1), &Attn("lse_out_d"))` — the decode pair's LSE.
///
/// gpt-oss' sink rescale reads that LSE, so a null here is a launch whose
/// second output nothing owns.
fn lse_or(cx: &Cx<'_>, a: &AttnCtx) -> Result<*mut f32, Refusal> {
    if let Ok(p) = cx.arg_out(1) {
        return Ok(p.cast::<f32>());
    }
    if a.lse_out_d.is_null() {
        return Err(Refusal::Unstated { what: "a second result or a published `lse_out_d`" });
    }
    Ok(a.lse_out_d)
}

/// The plan the fire raised, or the refusal that says it raised none.
///
/// A pure-prefill fire has no decode plan and a pure-decode fire has no
/// prefill plan, and a statement can ask for the one that is not there.
fn plan_ptr(cx: &Cx<'_>, a: &AttnCtx, family: &'static str) -> Result<*const c_void, Refusal> {
    let layer = u32::try_from(cx.layer()).unwrap_or(0);
    let plan = crate::bind::attn_plan(a, cx.spec(), layer, family);
    if plan.is_null() {
        return Err(Refusal::Unstated { what: "the plan this fire did not raise" });
    }
    Ok(plan.cast_const())
}

/// `Source::AttnNonZero`'s test, which was a guard and is a refusal here.
fn published<T>(p: *const T, what: &'static str) -> Result<*const T, Refusal> {
    if p.is_null() { Err(Refusal::Absent { what }) } else { Ok(p) }
}

/// The plan's descriptor, host to device, on the fire's own stream.
///
/// `int_base_bytes` is added to the DESTINATION and not to the offsets: the
/// upload is carved from zero, so a plan sharing an int workspace with another
/// moves as a block and the descriptor's own offsets stay relative.
///
/// # Safety
///
/// `stream` is the fire's, live across the copy, and `workspace.int_buffer`
/// names at least `int_base_bytes + bytes.len()` writable device bytes --
/// which the planner that filled `bytes` carved against.
unsafe fn upload(
    bytes: &[u8],
    workspace: AttentionWorkspaceView,
    int_base_bytes: usize,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: the caller's contract, forwarded.
    //
    // The stream crosses as a raw `*mut c_void` rather than a `StreamRef`.
    // `StreamRef` is `driver-cuda`'s borrow of a `cudaStream_t` and does not
    // exist one crate down; what it bought here was a lifetime on a value
    // that was constructed from a raw pointer two lines earlier and dropped
    // one line later, which is a borrow of nothing.
    let copied =
        unsafe { ffa2::upload_int_plan(bytes, workspace.int_buffer as u64, int_base_bytes, stream) };
    copied.map_err(|why| {
        tracing::error!(%why, "the FA2 plan descriptor did not reach the device");
        Refusal::Device { why: "the FA2 plan descriptor's H2D faulted; see the log" }
    })
}

/// Widen this layer's active pages into its bf16 mirrors.
///
/// `let _ =`, as the entry points had it: a layer whose dtype `KvDType` does
/// not name skips the prelude and the attention below still runs.
fn dequant_prelude(cx: &Cx<'_>, stream: *mut c_void, pages: i32) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let indices = cx.plan()?.kv_page_indices;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    let _ = kv_paged::dequant_kv_cache_layer_to_bf16_active(&ctx, &layer, indices, pages);
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_decode`
fn fa2_decode_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    let o = o_or(cx, a)?;
    let lse = lse_or(cx, a)?;
    // SAFETY: `bind::DecodePlan::as_ptr` is the only producer of this pointer
    // and it hands out its own boxed cache, non-null by the test above; the
    // borrow is shared and ends before this call returns.
    let cache = unsafe { &*plan_ptr(cx, a, "decode")?.cast::<ffa2::DecodePlanCache>() };

    dequant_prelude(cx, stream, cache.num_pages_in_batch)?;
    // SAFETY: the fire's stream, and the workspace the planner carved against.
    unsafe { upload(&cache.int_upload, a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dispatch_attention_flashinfer_decode(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(lse),
        Env(a.workspace.int_buffer),
        Env(a.workspace.float_buffer),
        Env(fa2d::decode_plan_of(cache, ffa2::fa_device())),
        cx.window_left()?,
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
        // `attention_flashinfer.hpp:136`'s default; the outer dispatch never
        // passed it, and a decode step reads one query row per request.
        Env(false),
    )
}

/// `attn::dispatch_attention_flashinfer_decode_capture`
///
/// The score buffers ride the CONTEXT rather than the statement because they
/// must be arena-STABLE: the predicate is folded, so one exec serves a fire
/// that wants scores and one that does not, and an address recorded now has to
/// still be right when it goes true.
fn fa2_decode_capture_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    let o = o_or(cx, a)?;
    let lse = lse_or(cx, a)?;
    published(a.score_out.cast_const(), "the score sink this launcher writes")?;
    published(a.score_indptr_d, "the score index this launcher writes into")?;
    // SAFETY: as `fa2_decode_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "decode")?.cast::<ffa2::DecodePlanCache>() };

    dequant_prelude(cx, stream, cache.num_pages_in_batch)?;
    // SAFETY: as above.
    unsafe { upload(&cache.int_upload, a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dispatch_attention_flashinfer_decode_capture(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(lse),
        Env(a.workspace.int_buffer),
        Env(a.workspace.float_buffer),
        Env(fa2d::decode_plan_of(cache, ffa2::fa_device())),
        Env(a.score_out),
        Env(a.score_indptr_d),
        cx.window_left()?,
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
        Env(false),
    )
}

/// `attn::dispatch_attention_flashinfer_prefill_bf16`
///
/// The pages LOOSE rather than the view whole, and `prefill_workspace` rather
/// than `workspace`: a FlashInfer plan writes its schedule into the workspace
/// it was raised against, so a prefill reading the decode plan's is one
/// clobbering the other.
///
/// No dequant prelude, and there was none in the C++ either -- this is the one
/// FA2 row whose KV comes in already bf16.
fn fa2_prefill_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    let o = o_or(cx, a)?;
    // SAFETY: `bind::PrefillPlan::as_ptr` is the only producer.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };

    // SAFETY: the fire's stream, and the workspace this plan was raised
    // against.
    unsafe { upload(&cache.int_upload, a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dispatch_attention_flashinfer_prefill_bf16(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.qo_indptr),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(a.lse_out_d),
        Env(a.prefill_workspace.int_buffer),
        Env(a.prefill_workspace.float_buffer),
        Env(fa2d::prefill_plan_of(cache, ffa2::fa_device())),
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
    )
}

/// `attn::dispatch_attention_flashinfer_prefill_capture_bf16`
///
/// `score_window` here is the OBSERVATION window ([`AttnCtx::score_window`]),
/// not the attention one -- deliberately not `Source::AttnWindow`. The routine
/// refuses zero and `window_left` is `-1` on a family that attends the whole
/// context, so the same number reads as "no window" to one and "invalid" to
/// the other.
///
/// `AttnCtx::folded_out` is bound by the row and **not read here**: folding is
/// `attn::attn_score_fold_heads`, a separate symbol fired by
/// `fire/attn_score.rs` after this returns.
fn fa2_prefill_capture_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    let o = o_or(cx, a)?;
    published(a.score_out.cast_const(), "the score sink this launcher writes")?;
    published(a.score_indptr_d, "the score index this launcher writes into")?;
    // SAFETY: as `fa2_prefill_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };

    // SAFETY: as above.
    unsafe { upload(&cache.int_upload, a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dispatch_attention_flashinfer_prefill_capture_bf16(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.qo_indptr),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(a.lse_out_d),
        Env(a.prefill_workspace.int_buffer),
        Env(a.prefill_workspace.float_buffer),
        Env(fa2d::prefill_plan_of(cache, ffa2::fa_device())),
        Env(a.score_out),
        Env(a.score_indptr_d),
        Env(a.score_window),
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
    )
}

/// `attn::dispatch_attention_flashinfer_prefill_custom`
///
/// The mask rides the CONTEXT, not the statement, for the reason the score
/// sink does: the predicate is folded, so one exec serves the fire that stages
/// a mask and the fire that does not, and the address recorded now must still
/// be right when it goes true.
///
/// This launcher takes the layer view whole rather than the pages loose, so it
/// dequantises like the decode -- with `num_pages_in_batch` read off the
/// plan's own widened KV indptr tail rather than off a device pointer, exactly
/// as `attention_flashinfer.cu:1244` did.
fn fa2_prefill_custom_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    let o = o_or(cx, a)?;
    published(a.mask_d, "the custom mask this launcher reads")?;
    published(a.mask_indptr_d, "the custom mask's index")?;
    // SAFETY: as `fa2_prefill_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };

    // `:1244`, whole: the page count comes off the plan's widened KV indptr,
    // because the device copy cannot be read from the host.
    let pages = if cache.num_requests > 0 {
        cache.kv_h_buf.get(cache.num_requests as usize).copied().unwrap_or(0)
    } else {
        0
    };
    dequant_prelude(cx, stream, pages)?;
    // SAFETY: as above.
    unsafe { upload(&cache.int_upload, a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dispatch_attention_flashinfer_prefill_custom(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.qo_indptr),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(a.lse_out_d),
        Env(a.prefill_workspace.int_buffer),
        Env(a.prefill_workspace.float_buffer),
        Env(fa2d::prefill_plan_of(cache, ffa2::fa_device())),
        Env(a.mask_d),
        Env(a.mask_indptr_d),
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
    )
}

/// `attn::attention_flashinfer_prefill` -- the PLANLESS prefill.
///
/// # The planning is the ARM's, and that is what `whole` is about
///
/// No cache crosses the fire for this symbol, so one is built here and thrown
/// away — the C++ did the same with a function-local `PrefillPlanInfo` and two
/// `std::vector<IdType>`. The resource is the pair `qo_indptr_h` /
/// `kv_page_indptr_h`: **HOST mirrors of the CSR**, which
/// [`ffa2::plan_prefill`] walks on the CPU. No `Cx` query answers a host
/// pointer, and reading the device CSR host-side is a synchronise a fire may
/// not make. The plan is also R-shaped, which is why the row states `whole`:
/// a row window would leave the arithmetic pointing at another request.
///
/// `:1000` and `:1063-1067` fix four flags this path never varies:
/// `enable_cuda_graph = false`, `full_attention_variant = false`,
/// `causal_mask = true`, `custom_mask = false`.
///
/// `plan` here is a local, and the H2D that reads it is issued before this
/// function returns: `upload_int_plan` copies from a pageable source, which
/// `cudaMemcpyAsync` stages synchronously, and that is what makes a
/// function-local plan legal.
fn fa2_prefill_planless_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = cx.arg_in(0)?.cast::<bf16>().cast_const();
    // `whole`, so this row carries no `Or`: a launcher that cannot be given a
    // row window cannot be handed the guard's arena slot either.
    let o = cx.arg_out(0)?.cast::<bf16>();
    if plan_of.requests <= 0 {
        return Err(Refusal::Empty { what: "the batch" });
    }
    let n = plan_of.requests as usize + 1;
    published(a.qo_indptr_h, "the host QO indptr the planner walks")?;
    published(a.kv_page_indptr_h, "the host KV page indptr the planner walks")?;
    // SAFETY: the fire's contract -- both are host CSRs of `num_requests + 1`
    // entries, which is what `Prepare::FireWide` publishes.
    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(a.qo_indptr_h, n),
            core::slice::from_raw_parts(a.kv_page_indptr_h, n),
        )
    };

    // `:1098`.
    let pages = i32::try_from(kv_h[plan_of.requests as usize]).unwrap_or(i32::MAX);
    dequant_prelude(cx, stream, pages)?;

    // `Source::Div(&Width(&In(0)), &KvLayerField("head_dim"))`: the head
    // COUNT, which nobody carries -- the query's width over the cache's head
    // dim. `.max(1)` is the generated divisor's, and it is what keeps a layer
    // view that states no head dim from dividing by zero.
    let num_q_heads = cx.in_width(0)? / layer.head_dim.max(1);

    let mut cache = ffa2::PrefillPlanCache::new();
    let device = ffa2::plan_device();
    let planned = ffa2::plan_prefill(
        &mut cache,
        qo_h,
        kv_h,
        cx.rows().count,
        plan_of.requests,
        num_q_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.page_size,
        kernels_cuda::attn::plan::Workspace {
            float_bytes: a.workspace.float_bytes,
            int_bytes: a.workspace.int_bytes,
        },
        &device,
        // `:1000`.
        false,
        cx.window_left()?,
        // `:1066-1067`.
        false,
        layer.hnd,
        true,
        false,
        false,
    );
    if let ffa2::Planned::Declined(why) = planned {
        tracing::error!(%why, "the planless FA2 prefill could not plan its own fire");
        return Err(Refusal::Unstated { what: "a plannable FA2 prefill fire; see the log" });
    }

    // SAFETY: the fire's stream. This path plans against `workspace` and not
    // `prefill_workspace`, as the entry point it replaces did.
    unsafe { upload(&cache.int_upload, a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attention_flashinfer_prefill(
        &ctx,
        q,
        o,
        Env(layer.k_bf16_pages.cast::<bf16>()),
        Env(layer.v_bf16_pages.cast::<bf16>()),
        Env(plan_of.qo_indptr),
        Env(plan_of.kv_page_indices),
        Env(plan_of.kv_page_indptr),
        Env(plan_of.kv_last_page_lens),
        Env(a.lse_out_d),
        Env(a.workspace.int_buffer),
        Env(a.workspace.float_buffer),
        Env(fa2d::prefill_plan_of(&cache, ffa2::fa_device())),
        Env(a.logits_soft_cap),
        Env(a.sm_scale),
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_decode",
        arm: Some(fa2_decode_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_decode_capture",
        arm: Some(fa2_decode_capture_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_prefill_bf16",
        arm: Some(fa2_prefill_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        arm: Some(fa2_prefill_capture_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_prefill_custom",
        arm: Some(fa2_prefill_custom_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::attention_flashinfer_prefill",
        arm: Some(fa2_prefill_planless_arm),
        unbound: None,
    },
];
