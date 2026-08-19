//! What a trace that states a FlashInfer FA2 dispatch binds to.
//!
//! The plans ARE unfolded -- `keys::Fa2Decode*` name the decode plan's sixteen
//! leaves and `keys::Fa2Prefill*` the planned prefill's twenty-two, at 28 to 36
//! parameters against a ceiling of 36. Four things hold the rows here and none
//! is a fact:
//!
//! 1. The upload: a replay runs no arm, and the captured H2D node bakes the
//!    source host address rather than the bytes.
//! 2. `no_join_extras`: an emptiness assertion, where a `Source` binds a slot.
//! 3. `dequant_prelude`: a second launch, where a column is one argument list.
//! 4. `lse_slab`: a load-bearing null refusal, per `arms/attn.rs`' null rule.
//!
//! # Two things that would compile and be wrong
//!
//! `AttnCtx` holds two workspace carves and a plan writes its schedule into the
//! one it was raised against, so a prefill reading the decode carve clobbers it
//! invisibly. `keys::AttnWorkspace*` and `fa2_decode_leaves` mean the decode
//! carve; only `fa2_prefill_leaves` means the prefill one. The planless pair
//! plans against `a.workspace` and so keeps the decode carve.
//!
//! `window_left` is `Env<keys::WindowLeft>` and not `Param<0, i32>`, though
//! every attention row states it in `params[0]`: `LaunchSpec::params` is
//! `Vec<u32>`, so an unbounded window arrives as `0xFFFF_FFFF` and
//! `as_declared`'s `U32 -> I32` refuses above `i32::MAX`.
//!
//! Each arm's `q` region reads `in_width(0).unwrap_or(0)`: no body reads the
//! field -- a dispatch is told its rectangle by its plan -- and `?` would
//! refuse a symbolic trailing dim for a number nobody reads.

use core::ffi::c_void;

use kernels::Refusal;
use kernels::routine::{Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::attn::kv_paged;
use kernels_cuda::attn::fa2::{
    attention_flashinfer_prefill, attention_flashinfer_prefill_lse,
    dispatch_attention_flashinfer_decode, dispatch_attention_flashinfer_decode_capture,
    dispatch_attention_flashinfer_decode_lse, dispatch_attention_flashinfer_prefill_bf16,
    dispatch_attention_flashinfer_prefill_capture_bf16,
    dispatch_attention_flashinfer_prefill_custom,
};

use crate::bind::AttnCtx;
use crate::bind::abi::AttentionWorkspaceView;
use kernels_cuda::attn::fa2::plan as ffa2;
use kernels_cuda::attn::fa2::dispatch as fa2d;

use super::super::cx::Cx;
use super::Bound;

/// The two join facts no `Source` could name: they change the arithmetic, not
/// the operands, so binding one would read right and compute wrong.
fn no_join_extras(cx: &Cx<'_>) -> Result<(), Refusal> {
    let spec = cx.spec();
    if !spec.aux.is_empty() || spec.per_head_dim.is_some() {
        return Err(Refusal::Unstated {
            what: "an FA2 dispatch without an aux value or a per-head reading",
        });
    }
    Ok(())
}

/// The stated result if the statement declares one, else the guard-owned arena
/// slot. Splitting the symbol on `o` loses the region's `Val` entirely.
fn o_or(cx: &Cx<'_>, a: &AttnCtx) -> Result<*mut bf16, Refusal> {
    if let Ok(p) = cx.arg_out(0) {
        return Ok(p.cast::<bf16>());
    }
    if a.o_out.is_null() {
        return Err(Refusal::Unstated { what: "somewhere for the attention output to land" });
    }
    Ok(a.o_out.cast::<bf16>())
}

/// The fire's LSE scratch slab, or the refusal saying nothing published one.
/// Load-bearing on the decode pair, whose sink rescale reads that LSE.
fn lse_slab(a: &AttnCtx) -> Result<*mut f32, Refusal> {
    if a.lse_out_d.is_null() {
        return Err(Refusal::Unstated { what: "a second result or a published `lse_out_d`" });
    }
    Ok(a.lse_out_d)
}

/// Which of a D2 pair's two symbols an arm is serving: only where `o` and
/// `lse` come from moves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// One result: `o` may be the guard's arena slot, `lse` the scratch slab.
    Plain,
    /// Two results, both the statement's; `arity_problem` bands these `[2, 2]`
    /// so a text that stopped declaring its LSE is refused at load.
    Lse,
}

/// The plan the fire raised, or the refusal that says it raised none: a
/// pure-prefill fire has no decode plan, and a statement can ask for one.
fn plan_ptr(cx: &Cx<'_>, a: &AttnCtx, family: &'static str) -> Result<*const c_void, Refusal> {
    let layer = u32::try_from(cx.layer()).unwrap_or(0);
    let plan = crate::bind::attn_plan(a, cx.spec(), layer, family);
    if plan.is_null() {
        return Err(Refusal::Unstated { what: "the plan this fire did not raise" });
    }
    Ok(plan.cast_const())
}

/// The null test `Source::AttnNonZero` would have compiled to. See
/// `crate::bind::Facts::q_out` for why a row was the wrong place for it.
fn published<T>(p: *const T, what: &'static str) -> Result<*const T, Refusal> {
    if p.is_null() { Err(Refusal::Absent { what }) } else { Ok(p) }
}

/// The plan's descriptor, host to device, on the fire's own stream.
/// `int_base_bytes` is added to the destination and not to the offsets.
///
/// # Safety
///
/// `stream` is the fire's, live across the copy, and `workspace.int_buffer`
/// names at least `int_base_bytes + bytes.len()` writable device bytes.
unsafe fn upload(
    bytes: &[u8],
    workspace: AttentionWorkspaceView,
    int_base_bytes: usize,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: the caller's contract, forwarded.
    let copied =
        unsafe { ffa2::upload_int_plan(bytes, workspace.int_buffer as u64, int_base_bytes, stream) };
    copied.map_err(|why| {
        tracing::error!(%why, "the FA2 plan descriptor did not reach the device");
        Refusal::Device { why: "the FA2 plan descriptor's H2D faulted; see the log" }
    })
}

/// Widen this layer's active pages into its bf16 mirrors.
///
/// `let _ =`: a layer whose dtype `KvDType` does not name skips the prelude and
/// the attention below still runs.
fn dequant_prelude(cx: &Cx<'_>, stream: *mut c_void, _pages: i32) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let _plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    // THE SEVEN LAYER NUMBERS ARE ASKED FOR INSIDE THE CALLEE NOW: they came
    // off `cx.kv_layer()` here and reach the body off the same borrow through
    // `keys::Kv_`, so there is one source rather than two that could disagree.
    let _ = kv_paged::dequant_kv_cache_layer_to_bf16_active(&ctx);
    Ok(())
}

/// `attn::dequant_kv_cache_layer_to_bf16_active`, as its own arm.
///
/// A HAND ARM AGAIN, and not a regression: the routine takes NOTHING from the
/// statement now — its seven layer numbers are asked for and its pointers come
/// off `Cx` — so there is no operand column to derive, and `derived_arm`
/// refuses an empty one with "nothing states an operand column". A row bound
/// from a column it does not have is the same mistake as a column bound from
/// a row that does not exist.
///
/// The body is `dequant_prelude`'s, whose page count this launch does not use.
///
/// # Errors
///
/// Whatever the layer query or the launch refuses.
pub fn dequant_kv_cache_layer_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    dequant_prelude(cx, stream, 0)
}

/// `attn::dispatch_attention_flashinfer_decode`
///
/// The column is complete on both spellings. What keeps the arm is the header's
/// four, plus `o_or`'s fallback: d[1] is `Slot(Out, 0)`, not nullable.
fn fa2_decode_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    fa2_decode(cx, stream, Form::Plain)
}

/// `attn::dispatch_attention_flashinfer_decode_lse`
///
/// D2's other half: the only text naming it declares both results, so this arm
/// reads `arg_out(0)` and `arg_out(1)` and refuses rather than falling back.
fn fa2_decode_lse_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    fa2_decode(cx, stream, Form::Lse)
}

/// The decode dispatch, both spellings.
fn fa2_decode(cx: &Cx<'_>, stream: *mut c_void, form: Form) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    // The one line that differs between the two symbols: `Form::Lse` has no
    // fallback, so the declared buffer is the buffer the kernel writes.
    let (o, lse) = match form {
        Form::Plain => (o_or(cx, a)?, lse_slab(a)?),
        Form::Lse => (cx.arg_out(0)?.cast::<bf16>(), cx.arg_out(1)?.cast::<f32>()),
    };
    // SAFETY: `bind::DecodePlan::as_ptr` hands out its own boxed cache, non-null
    // by the test above; the shared borrow ends before this call returns.
    let cache = unsafe { &*plan_ptr(cx, a, "decode")?.cast::<ffa2::DecodePlanCache>() };
    // The same resolution the column would do, through the same function: two
    // copies of the offset arithmetic is how the two paths come to disagree.
    let _l = super::super::table::fa2_decode_leaves(cx)?;

    dequant_prelude(cx, stream, cache.num_pages_in_batch)?;
    // SAFETY: the fire's stream, and the workspace the planner carved against.
    unsafe { upload(cache.int_upload.as_slice(), a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    let _k_pages = layer.k_bf16_pages.cast::<u8>();
    let _v_pages = layer.v_bf16_pages.cast::<u8>();
    let _kv_page_indices = plan_of.kv_page_indices;
    let _kv_page_indptr = plan_of.kv_page_indptr;
    let _kv_last_page_lens = plan_of.kv_last_page_lens;
    // THE THREE NUMBERS ARE ASKED FOR INSIDE THE CALLEE NOW. `ctx` carries
    // `Answering::over_facts(cx)`, so `keys::WindowLeft`, `AttnLogitsSoftCap`
    // and `SmScale` reach the body off the same `Cx` this arm read them from
    // -- one source rather than two that could disagree.
    match form {
        Form::Plain => dispatch_attention_flashinfer_decode(
            &ctx,
            q,
            Out { ptr: o, rows: 0, width: 0 },
    ),
        Form::Lse => dispatch_attention_flashinfer_decode_lse(
            &ctx,
            q,
            Out { ptr: o, rows: 0, width: 0 },
            Out { ptr: lse, rows: 0, width: 0 },
    ),
    }
}

/// `attn::dispatch_attention_flashinfer_decode_capture`
///
/// The score buffers ride the context and must be arena-stable: the predicate
/// is folded, so an address recorded now has to be right when it goes true.
fn fa2_decode_capture_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let _layer = cx.kv_layer()?;
    let _plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    let o = o_or(cx, a)?;
    // `lse_slab` and not a probe: this dispatch has one DSL spelling, so there
    // is no `arg_out(1)` to find.
    let _lse = lse_slab(a)?;
    published(a.score_out.cast_const(), "the score sink this launcher writes")?;
    published(a.score_indptr_d, "the score index this launcher writes into")?;
    // SAFETY: as `fa2_decode_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "decode")?.cast::<ffa2::DecodePlanCache>() };
    let _l = super::super::table::fa2_decode_leaves(cx)?;

    dequant_prelude(cx, stream, cache.num_pages_in_batch)?;
    // SAFETY: as above.
    unsafe { upload(cache.int_upload.as_slice(), a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    dispatch_attention_flashinfer_decode_capture(
        &ctx,
        q,
        Out { ptr: o, rows: 0, width: 0 },
        a.score_out,
        a.score_indptr_d,
        Const { v: cx.window_left()? },
        Const { v: a.logits_soft_cap },
        Const { v: a.sm_scale },
    )
}

/// `attn::dispatch_attention_flashinfer_prefill_bf16`
///
/// No dequant prelude: this is the one FA2 row whose KV comes in already bf16.
fn fa2_prefill_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let _layer = cx.kv_layer()?;
    let _plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    let o = o_or(cx, a)?;
    // SAFETY: `bind::PrefillPlan::as_ptr` is the only producer.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };
    let _l = super::super::table::fa2_prefill_leaves(cx)?;

    // SAFETY: the fire's stream, and the workspace this plan was raised against.
    unsafe { upload(cache.int_upload.as_slice(), a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    // The cap and the scale are asked for inside the callee; see the decode
    // arm above.
    dispatch_attention_flashinfer_prefill_bf16(&ctx, q, Out { ptr: o, rows: 0, width: 0 })
}

/// `attn::dispatch_attention_flashinfer_prefill_capture_bf16`
///
/// `score_window` is the observation window, not the attention one: the routine
/// refuses zero, and `window_left` is `-1` on a family attending the whole
/// context, so one number reads as "no window" to one and "invalid" to the other.
fn fa2_prefill_capture_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let _layer = cx.kv_layer()?;
    let _plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    let o = o_or(cx, a)?;
    published(a.score_out.cast_const(), "the score sink this launcher writes")?;
    published(a.score_indptr_d, "the score index this launcher writes into")?;
    // SAFETY: as `fa2_prefill_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };
    let _l = super::super::table::fa2_prefill_leaves(cx)?;

    // SAFETY: as above.
    unsafe { upload(cache.int_upload.as_slice(), a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    dispatch_attention_flashinfer_prefill_capture_bf16(
        &ctx,
        q,
        Out { ptr: o, rows: 0, width: 0 },
        a.score_out,
        a.score_indptr_d,
        a.score_window,
        Const { v: a.logits_soft_cap },
        Const { v: a.sm_scale },
    )
}

/// `attn::dispatch_attention_flashinfer_prefill_custom`
///
/// The mask rides the context for the reason the score sink does: the predicate
/// is folded, so the address recorded now must be right when it goes true. This
/// launcher takes the layer view whole, so it dequantises like the decode.
fn fa2_prefill_custom_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let _layer = cx.kv_layer()?;
    let _plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    let o = o_or(cx, a)?;
    published(a.mask_d, "the custom mask this launcher reads")?;
    published(a.mask_indptr_d, "the custom mask's index")?;
    // SAFETY: as `fa2_prefill_arm`'s.
    let cache = unsafe { &*plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>() };
    let _l = super::super::table::fa2_prefill_leaves(cx)?;

    // The page count comes off the plan's widened KV indptr, not the device copy.
    let pages = if cache.num_requests > 0 {
        cache.kv_h_buf.get(cache.num_requests as usize).copied().unwrap_or(0)
    } else {
        0
    };
    dequant_prelude(cx, stream, pages)?;
    // SAFETY: as above.
    unsafe { upload(cache.int_upload.as_slice(), a.prefill_workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    dispatch_attention_flashinfer_prefill_custom(
        &ctx,
        q,
        Out { ptr: o, rows: 0, width: 0 },
        a.mask_d,
        a.mask_indptr_d,
        Const { v: a.logits_soft_cap },
        Const { v: a.sm_scale },
    )
}

/// `attn::attention_flashinfer_prefill` -- the planless prefill.
///
/// What keeps the arm is `a.qo_indptr_h`/`a.kv_page_indptr_h`, host mirrors of
/// the CSR the planner walks, which no `Cx` query answers.
///
/// It plans into the fire's `prefill_plan` cache and not a local: under capture
/// the memcpy node holds the source address, so a local is a use-after-free
/// whose only symptom is wrong attention.
fn fa2_prefill_planless_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    fa2_prefill_planless(cx, stream, Form::Plain)
}

/// `attn::attention_flashinfer_prefill_lse`
///
/// D2's other half of the planless prefill: the only text naming it declares
/// both results, so this arm has no fallback available to take.
fn fa2_prefill_planless_lse_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    fa2_prefill_planless(cx, stream, Form::Lse)
}

/// The planless prefill, both spellings: the split is about arity, not behaviour.
fn fa2_prefill_planless(cx: &Cx<'_>, stream: *mut c_void, form: Form) -> Result<(), Refusal> {
    no_join_extras(cx)?;
    let a = cx.attn_ctx()?;
    let layer = cx.kv_layer()?;
    let plan_of = cx.plan()?;
    let q = In {
        ptr: cx.arg_in(0)?.cast::<bf16>().cast_const(),
        rows: cx.rows().count,
        width: cx.in_width(0).unwrap_or(0),
    };
    // `whole`, so neither spelling has a fallback on `o`: a launcher that
    // cannot take a row window cannot take the guard's arena slot either.
    let o = cx.arg_out(0)?.cast::<bf16>();
    // `Form::Plain` passes the slab bare, as `Env<*mut f32>` says it may. The
    // arity bands discriminate -- `[0, 1]` here against `[2, 2]` on the `_lse`
    // symbol -- so a text declaring an LSE against this one is refused at load.
    let lse = match form {
        Form::Plain => a.lse_out_d,
        Form::Lse => cx.arg_out(1)?.cast::<f32>(),
    };
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

    let pages = i32::try_from(kv_h[plan_of.requests as usize]).unwrap_or(i32::MAX);
    dequant_prelude(cx, stream, pages)?;

    // The head count, which nobody carries: the query's width over the cache's
    // head dim. Its consumer `ffa2::plan_prefill` is a host planner with no
    // `Refusal` to carry a guard into, so `.max(1)` fabricates a divisor rather
    // than preventing the division, and a partial head truncates the rectangle.
    if layer.head_dim <= 0 {
        return Err(Refusal::Empty { what: "the layer's head dim" });
    }
    let q_width = cx.in_width(0)?;
    if q_width % layer.head_dim != 0 {
        return Err(Refusal::Narrow { what: "the query width, in heads", at: i64::from(q_width) });
    }
    let num_q_heads = q_width / layer.head_dim;

    // SAFETY: `bind::PrefillPlan::as_ptr` hands out its own boxed cache, non-null
    // by `plan_ptr`. The `&mut` is exclusive for this arm: a fire dispatches one
    // statement at a time and holds only the raw pointer.
    let cache =
        unsafe { &mut *plan_ptr(cx, a, "prefill")?.cast::<ffa2::PrefillPlanCache>().cast_mut() };
    let device = ffa2::plan_device();
    let planned = ffa2::plan_prefill(
        cache,
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
        // `enable_cuda_graph`, and not the C++'s value: `false` sizes the
        // tiling from this batch, `true` derives `cta_tile_q` from a bound the
        // batch cannot exceed, so the layout can be baked once.
        true,
        cx.window_left()?,
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
    unsafe { upload(cache.int_upload.as_slice(), a.workspace, cache.int_base_bytes, stream) }?;

    // SAFETY: `stream` is the fire's own, live across the launch.
    // AND IT LENDS THIS FIRE'S ANSWERS: the bodies ask for the plan
    // leaves and the KV pool rather than taking them as parameters.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    let _k_pages = layer.k_bf16_pages.cast::<u8>();
    let _v_pages = layer.v_bf16_pages.cast::<u8>();
    let _kv_page_indices = plan_of.kv_page_indices;
    let _kv_page_indptr = plan_of.kv_page_indptr;
    let _kv_last_page_lens = plan_of.kv_last_page_lens;
    // THE PLAN IS THE ARM'S OWN AGGREGATE, which is why it is `#[unbound]` at
    // the signature: nothing publishes it before the fire, so no column can
    // answer for it. The workspaces beside it used to be threaded through too
    // and are asked for in the body now.
    let prefill_plan = fa2d::prefill_plan_of(cache, ffa2::fa_device());
    match form {
        Form::Plain => attention_flashinfer_prefill(
            &ctx,
            q,
            Out { ptr: o, rows: 0, width: 0 },
            prefill_plan,
            Const { v: a.logits_soft_cap },
            Const { v: a.sm_scale },
        ),
        Form::Lse => attention_flashinfer_prefill_lse(
            &ctx,
            q,
            Out { ptr: o, rows: 0, width: 0 },
            Out { ptr: lse, rows: 0, width: 0 },
            prefill_plan,
            Const { v: a.logits_soft_cap },
            Const { v: a.sm_scale },
        ),
    }
}

/// Every symbol this family binds.
///
/// Each `_lse` symbol shares its twin's body through [`Form`], so a pair cannot
/// drift apart in anything but the two lines the split is about. None is
/// `Bound::derived` and none becomes one: see the header's four.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_decode",
        arm: Some(fa2_decode_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dispatch_attention_flashinfer_decode_lse",
        arm: Some(fa2_decode_lse_arm),
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
    Bound {
        symbol: "attn::attention_flashinfer_prefill_lse",
        arm: Some(fa2_prefill_planless_lse_arm),
        unbound: None,
    },
];
