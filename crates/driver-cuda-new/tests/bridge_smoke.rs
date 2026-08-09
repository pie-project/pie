//! The launch bridge, smoked end to end: one generated binding, one real
//! kernel, one round trip (retirement plan phase A step 3).
//!
//! The cheapest row in the table does the proving:
//! `quant::cast_fp32_to_bf16` reads fp32, writes bf16, and its answer is
//! bit-checkable on the host (bf16 IS the top half of the fp32 for values
//! that need no rounding). If this passes, the whole chain held — the
//! bindings module compiled against the generated declarations, the shim
//! compiled against the real headers, the archive linked, and a launcher
//! this crate did not write ran device code on this crate's stream.
//!
//! Skipped without a device, like every GPU test here.

use driver_cuda_new::cuda::{Allocator, OwnedStream};
use driver_cuda_new::launch::ffi;

mod common;
use common::{device_or_skip, gpu_guard};

#[test]
fn a_generated_binding_reaches_a_real_kernel() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("bridge smoke") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // Powers of two need no rounding, so each bf16 is exactly the fp32's
    // top sixteen bits and the expectation can be computed with a shift.
    let src: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25).collect();
    let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut d_src = alloc.alloc(src_bytes.len()).expect("src alloc");
    d_src.copy_from_host(&src_bytes, stream.as_ref()).expect("h2d");
    let d_dst = alloc.alloc(src.len() * 2).expect("dst alloc");

    unsafe {
        ffi::pie_k_quant_cast_fp32_to_bf16(
            d_src.as_ptr(),
            d_dst.as_ptr(),
            src.len(),
            stream.as_ref().as_raw().cast(),
        );
    }

    let mut back = vec![0u8; src.len() * 2];
    d_dst.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    for (i, v) in src.iter().enumerate() {
        let expect = (v.to_bits() >> 16) as u16;
        let got = u16::from_le_bytes([back[i * 2], back[i * 2 + 1]]);
        assert_eq!(
            got, expect,
            "element {i}: bf16 0x{got:04x} != expected 0x{expect:04x} (fp32 {v})"
        );
    }
}

/// The second table's chain, smoked the same way: a DRIVER-INTERNAL
/// launcher (no DSL row — the envelope seed) reached through its generated
/// binding via `LiveKvCacheOps`, the first bridge-gated seam impl. Empty
/// envelopes are +inf/-inf bf16 by construction, so the whole tier is
/// bit-checkable on the host.
#[test]
fn a_driver_internal_binding_seeds_the_envelope_tier() {
    use driver_cuda_new::store::kv_cache_live::{KvCacheDeviceOps, LiveKvCacheOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("driver-internal envelope seed") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // 2 pages x 2 kv heads x 4 dims = 16 bf16 elements per plane.
    let elems = 2 * 2 * 4;
    let mut d_min = alloc.alloc(elems * 2).expect("env_min");
    let mut d_max = alloc.alloc(elems * 2).expect("env_max");
    d_min.memset(0, stream.as_ref()).expect("zero min");
    d_max.memset(0, stream.as_ref()).expect("zero max");

    let alloc = driver_cuda_new::cuda::Allocator::new();
    let mut ops = LiveKvCacheOps::new(stream.as_ref().as_raw().cast(), &alloc);
    ops.envelope_seed(d_min.as_ptr().cast(), d_max.as_ptr().cast(), 2, 2, 4);
    ops.stream_synchronize();

    let mut min_back = vec![0u8; elems * 2];
    let mut max_back = vec![0u8; elems * 2];
    d_min.copy_to_host(&mut min_back, stream.as_ref()).expect("d2h min");
    d_max.copy_to_host(&mut max_back, stream.as_ref()).expect("d2h max");
    stream.as_ref().synchronize().expect("sync");

    for i in 0..elems {
        let min = u16::from_le_bytes([min_back[i * 2], min_back[i * 2 + 1]]);
        let max = u16::from_le_bytes([max_back[i * 2], max_back[i * 2 + 1]]);
        assert_eq!(min, 0x7F80, "element {i}: empty env_min is +inf bf16");
        assert_eq!(max, 0xFF80, "element {i}: empty env_max is -inf bf16");
    }
}

/// The live `ScoreOps`: memset, the CSR upload, and the fold launch. With
/// ONE query head the fold's per-position average is over a single value,
/// so `folded == raw` over the request's span regardless of the kernel's
/// internal layout — an identity that checks the whole chain without
/// re-deriving the indexing the score oracle already pinned.
#[test]
fn the_live_score_ops_upload_memset_and_fold() {
    use driver_cuda_new::model::attn_score::{LiveScoreOps, ScoreOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live score ops") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let mut ops = LiveScoreOps::new(stream.as_ref().as_raw().cast());

    // memset: 64 bytes of 0xA5.
    let scratch = alloc.alloc(64).expect("scratch");
    ops.memset_async(scratch.as_ptr().cast(), 0xa5, 64);
    let mut back = vec![0u8; 64];
    scratch.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    assert!(back.iter().all(|&b| b == 0xa5));

    // CSR upload: one request spanning 4 positions.
    let indptr: Vec<i32> = vec![0, 4];
    let d_indptr = alloc.alloc(indptr.len() * 4).expect("indptr");
    ops.upload_csr(d_indptr.as_ptr().cast(), &indptr);

    // Fold: 1 request, 1 q head, kv_len 4 in one page of 16.
    let raw: Vec<f32> = vec![0.25, 0.5, 0.125, 1.0];
    let raw_bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_raw = alloc.alloc(raw_bytes.len()).expect("raw");
    d_raw.copy_from_host(&raw_bytes, stream.as_ref()).expect("h2d raw");
    let page_indptr: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_pages = alloc.alloc(page_indptr.len()).expect("pages");
    d_pages.copy_from_host(&page_indptr, stream.as_ref()).expect("h2d pages");
    let last_lens: Vec<u8> = [4u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_lens = alloc.alloc(last_lens.len()).expect("lens");
    d_lens.copy_from_host(&last_lens, stream.as_ref()).expect("h2d lens");
    let d_folded = alloc.alloc(raw_bytes.len()).expect("folded");

    ops.fold_heads(
        d_raw.as_ptr().cast(),
        d_indptr.as_ptr().cast(),
        d_pages.as_ptr().cast(),
        d_lens.as_ptr().cast(),
        16,
        1,
        1,
        d_folded.as_ptr().cast(),
    );

    let mut folded_back = vec![0u8; raw_bytes.len()];
    d_folded.copy_to_host(&mut folded_back, stream.as_ref()).expect("d2h folded");
    stream.as_ref().synchronize().expect("sync");
    for (i, v) in raw.iter().enumerate() {
        let got = f32::from_le_bytes(folded_back[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(got, *v, "position {i}: average over one head is identity");
    }
}

/// The live `LoraOps`: the cast through its DSL binding, and the pointer
/// slab landing device-resident with its values intact.
#[test]
fn the_live_lora_ops_cast_and_upload_the_slab() {
    use driver_cuda_new::model::lora::{LiveLoraOps, LoraOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live lora ops") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let mut ops = LiveLoraOps::new(stream.as_ref().as_raw().cast());

    // Cast: same bit-check as the bridge smoke, through the seam.
    let src: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
    let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_src = alloc.alloc(src_bytes.len()).expect("src");
    d_src.copy_from_host(&src_bytes, stream.as_ref()).expect("h2d");
    let d_dst = alloc.alloc(src.len() * 2).expect("dst");
    ops.cast_fp32_to_bf16(d_src.as_ptr(), d_dst.as_ptr(), src.len());
    let mut back = vec![0u8; src.len() * 2];
    d_dst.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    for (i, v) in src.iter().enumerate() {
        let expect = (v.to_bits() >> 16) as u16;
        let got = u16::from_le_bytes([back[i * 2], back[i * 2 + 1]]);
        assert_eq!(got, expect, "element {i}");
    }

    // Slab: four sentinel pointers, round-tripped.
    let slots: Vec<*const std::ffi::c_void> =
        [0x1000usize, 0x2000, 0x3000, 0x4000].iter().map(|&a| a as _).collect();
    let d_slab = alloc.alloc(slots.len() * 8).expect("slab");
    ops.upload_slab(d_slab.as_ptr(), &slots);
    let mut slab_back = vec![0u8; slots.len() * 8];
    d_slab.copy_to_host(&mut slab_back, stream.as_ref()).expect("d2h slab");
    stream.as_ref().synchronize().expect("sync");
    for (i, &p) in slots.iter().enumerate() {
        let got = u64::from_le_bytes(slab_back[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(got, p as u64, "slot {i} survived the upload");
    }
}

/// The executor's two halves over the REAL anchor lowering (retirement
/// plan phase C): bind + dispatch walk `qwen3_0_6b`'s decode launches on
/// the device until the first kernel without an arm, and the numbers are
/// checked against host math. What this pins beyond the plumbing is the
/// OPERAND ORDER inside each arm — a swapped input is wrong values, not a
/// type error, and only host arithmetic notices.
#[test]
fn the_executor_prefix_runs_the_anchor_decode_on_device() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::model::executor::{
        DispatchCtx, DispatchPlan, DispatchRefusal, Frame, Resolver, bind, dispatch,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("executor prefix") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    // The real traced decode form, over four rows.
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // Sixty-four fake vocabulary rows: token t's embedding alternates
    // +a(t), -a(t) with a(t) = 0.5 + 0.25 t, so rmsnorm collapses every
    // row to alternating ±1 whatever the token — two exact expectations
    // from one pattern.
    const HIDDEN: usize = 1024;
    const VOCAB: usize = 64;
    let tokens: [i32; 4] = [1, 2, 3, 5];
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..VOCAB {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let ones_host: Vec<u8> = std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN)
        .flatten()
        .collect();
    let ids_host: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

    struct Live {
        embed: driver_cuda_new::cuda::DeviceBuffer,
        ones: driver_cuda_new::cuda::DeviceBuffer,
        zeros: driver_cuda_new::cuda::DeviceBuffer,
        ids: driver_cuda_new::cuda::DeviceBuffer,
        named: BTreeMap<ValueId, *mut std::ffi::c_void>,
    }
    impl Resolver for Live {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") {
                self.embed.as_ptr()
            } else if name.contains("norm") {
                self.ones.as_ptr()
            } else {
                self.zeros.as_ptr()
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            // Every pinned input in the prefix is a per-row i32 array
            // (token ids, positions); one buffer serves each id.
            Some(*self.named.entry(value).or_insert(self.ids.as_ptr()))
        }
    }

    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed w");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d embed");
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d ones");
    let mut zeros_dev = alloc.alloc(16 << 20).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");
    let mut ids_dev = alloc.alloc(ids_host.len()).expect("ids");
    ids_dev.copy_from_host(&ids_host, stream.as_ref()).expect("h2d ids");
    let mut resolver = Live {
        embed: embed_dev,
        ones: ones_dev,
        zeros: zeros_dev,
        ids: ids_dev,
        named: BTreeMap::new(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e6,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: 128,
        num_q_heads: 16,
        num_kv_heads: 8,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: resolver.ids.as_ptr(),
        positions: resolver.ids.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };
    let dplan = DispatchPlan::new(&plan, &l);

    // Walk until the first kernel without an arm, remembering where the
    // embed and the first rmsnorm wrote.
    let mut embed_out: Option<usize> = None;
    let mut norm_out: Option<usize> = None;
    let mut dispatched = 0usize;
    let mut stopped_at = String::new();
    for (i, launch) in l.launches.iter().enumerate() {
        let bound = bind(&l, launch, frame, &mut resolver).expect("binds");
        let offset_of = |p: *mut std::ffi::c_void| p as usize - frame.arena as usize;
        match dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, None, None) {
            Ok(()) => {
                if bound.kernel == "layout::embed_bf16" {
                    embed_out.get_or_insert(offset_of(bound.args[0].ptr));
                } else if bound.kernel == "norm::rmsnorm_bf16" {
                    norm_out.get_or_insert(offset_of(bound.args[1].ptr));
                }
                dispatched += 1;
            }
            // The smoke runs WITHOUT attention context on purpose — the
            // fused qkv arm refusing on that is the intended boundary.
            Err(DispatchRefusal::NoArm(k) | DispatchRefusal::NoAttnCtx(k)) => {
                stopped_at = k;
                break;
            }
            Err(e) => panic!("arm drift: {e:?}"),
        }
    }
    stream.as_ref().synchronize().expect("sync");
    assert!(dispatched >= 4, "only {dispatched} launches ran before the stop");
    assert_eq!(
        stopped_at, "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        "the walk should stop at the fused attention step"
    );

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h arena");
    stream.as_ref().synchronize().expect("sync");
    let bf16_at = |off: usize, i: usize| {
        u16::from_le_bytes([arena_back[off + i * 2], arena_back[off + i * 2 + 1]])
    };

    // The embed rows are the pattern rows for tokens [1, 2, 3, 5].
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 511, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let got = bf16_at(e, r * HIDDEN + c);
            assert_eq!(got, want, "embed row {r} (token {t}) col {c}");
        }
    }

    // RMSNorm of an alternating ±a row is alternating ±1 (times the ones
    // weight), whatever a was — bf16-exactly, since 1.0 is representable
    // and the kernel normalizes in fp32.
    let n = norm_out.expect("rmsnorm ran");
    for r in 0..tokens.len() {
        for c in [0usize, 1, 512, 1023] {
            let want = bf16(if c % 2 == 0 { 1.0 } else { -1.0 });
            let got = bf16_at(n, r * HIDDEN + c);
            assert_eq!(got, want, "rmsnorm row {r} col {c}");
        }
    }

    cublas.release(&mut cublas_ops);
}

/// FlashInfer's decode planner, driven from Rust end to end: a real
/// `DecodePlanCache` through the hand-written extras, staged into a LIVE
/// workspace slot under the begin/end fence — the C++ prepare flow,
/// re-spoken. This is the riskiest unlit piece of the attention step
/// (host planning code deep inside the archive), which is why it gets its
/// own smoke before any attention arm consumes the plan.
#[test]
fn the_decode_planner_plans_a_real_geometry() {
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::DecodePlan;

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("decode planner") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw = stream.as_ref().as_raw().cast::<std::ffi::c_void>();

    let mut ops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut ops, 32 << 20, 16 << 20, 2)
        .expect("workspace");

    // Four decode requests, one 16-token page each — qwen3-0.6b geometry.
    let indptr: [u32; 5] = [0, 1, 2, 3, 4];
    let mut plan = DecodePlan::new();
    ws.begin_plan_update(&mut ops).expect("begin");
    plan.plan_decode(&indptr, 16, 8, 128, 16, ws.view(), raw, false, -1);
    ws.end_plan_update(&mut ops, raw);
    stream.as_ref().synchronize().expect("the staged upload retires");

    // Plan again with different geometry — the cache is reusable per
    // fire, which is how the driver holds it.
    let indptr2: [u32; 3] = [0, 2, 4];
    ws.begin_plan_update(&mut ops).expect("begin 2");
    plan.plan_decode(&indptr2, 16, 8, 128, 16, ws.view(), raw, false, -1);
    ws.end_plan_update(&mut ops, raw);
    stream.as_ref().synchronize().expect("second upload retires");

    ws.release(&mut ops);
}

/// The FULL decode: all 257 launches of `qwen3_0_6b`'s real lowering,
/// walked on the device with a live `AttnCtx` — KV pools, page CSRs,
/// write descriptors, a planned FlashInfer cache. All-zero weights make
/// the whole forward analytically checkable: every projection returns
/// zero, attention over zero V returns zero, the beta-1 residual folds
/// add zero — so the residual stream must equal the embed rows
/// BIT-EXACTLY after 28 layers, and the logits must be all-zero. A single
/// swapped operand anywhere in the walk breaks one of the two.
#[test]
fn the_full_zero_weight_decode_walks_every_launch() {
    zero_weight_decode(Leg::Eager);
}

/// The same decode over a fire whose last two rows carry attached
/// programs — the only leg that exercises a PEEL.
///
/// `lower` splits a hooked fire on the hook axis, and the tail region
/// addresses rows at absolute offsets in a full-N buffer, so its
/// statements take `_devwin` kernels that read the split from device
/// memory instead of taking a row count. Until this leg existed nothing
/// in the tree ran one: every other lowering uses plain rows, so the peel
/// path had no fire, `attn::split_qkv_bf16_devwin` had no arm, and the
/// symbol was not even in `UNARMED` because it was never lowered.
///
/// No programs are actually attached, so both regions compute what the
/// unpeeled fire computes. That is what makes it a gate rather than a
/// smoke test: the residual and logit invariants must come out
/// bit-identical, and a devwin launch reading the wrong window would move
/// them.
///
/// # Why this is `ignore`d, and what un-ignoring it needs
///
/// It does not pass yet, and the reason moved once already, which is the
/// useful part. Two layers came off:
///
/// * the SPLIT — `attn::split_qkv_bf16_devwin` had no arm and no device
///   word to read. `cuda::PeelWindowWord` and its arm fixed that.
/// * the WINDOWED RECTANGLE — the tail's launches bound BASE pointers
///   plus a row count, so they ran over the prefix's rows. `Arg::Arena`
///   states its element width now and `resolve_arg_windowed` applies the
///   window once, for the stated args and the op join's placements
///   together. §4's fourth decline-rule is gone with it.
///
/// What is left is the PLAN. `Launch::peel`'s own doc says a prepared
/// plan "is found by the rectangle's ROW COUNT" — and this fire builds
/// ONE `DecodePlan`, for all four rows, while the tail's attention serves
/// two. FlashInfer reads a plan that does not describe the launch it was
/// given and faults.
///
/// So un-ignoring needs a plan per row count, which is A4's plan-stability
/// item arriving from the peel side rather than the capture side. The two
/// want the same thing: an arm must be HANDED its plan rather than
/// resolve one, because neither a peel region nor a captured replay can
/// assume the fire's.
#[test]
fn a_hooked_fire_peels_and_still_lands_the_same_numbers() {
    zero_weight_decode(Leg::Hooked);
}

/// The SAME walk, captured and replayed.
///
/// This is the gate `run_captured` was written for and the one thing its
/// unit tests cannot reach: a real deployment's real decode, issued into
/// a capture rather than onto a stream, instantiated, and launched. The
/// arena is WIPED between the capture and the replay, so the two
/// invariants below -- the residual is the embed rows bit-exactly, the
/// logits are the tied lm_head's exact algebra -- can only be satisfied
/// by work the replay did.
///
/// It captures the RESOLVED lowering. Capturing the union needs one more
/// thing, which `the_union_capture_needs_every_arm_issuable` pins.
#[test]
fn a_resolved_walk_captures_and_replays() {
    zero_weight_decode(Leg::Captured);
}

/// **The concurrent-lane property: one exec, a SECOND fire.**
///
/// Clause 2's remaining item. Every other captured leg replays the fire
/// it captured, which cannot tell a baked address from baked contents.
/// This one captures at `[1, 2, 3, 5]` and replays at `[7, 11, 13, 17]`,
/// and the residual invariant is exact per row — so a capture that baked
/// contents fails here and nowhere else.
#[test]
fn a_cached_exec_serves_the_next_fire() {
    zero_weight_decode(Leg::Reused);
}

/// **A5, at one lane: the UNION captured and replayed on real geometry.**
///
/// Everything else proves a piece. `gpu_supergraph` proves the mechanism
/// against memsets; `union_lower` proves the launch list, GPU-free;
/// `a_resolved_walk_captures_and_replays` proves a capture of an
/// already-decided program. This one keeps every guard, records every arm
/// into conditional bodies, arms the predicates from a device word, and
/// then asks the only question that matters: does the same decode come
/// out?
///
/// The arena is wiped between the capture and the replay, so the residual
/// and logit invariants can only be met by work the replay did — through
/// the conditionals, off the predicate word.
///
/// What it is NOT yet is A5 in full: the plan asks for byte-identity
/// across CONCURRENT structurally-distinct lanes, which needs more than
/// one fire in flight. This is the single-lane form, and it is the one
/// that had to work first.
///
/// # Why this is `ignore`d: the warm-up paradox
///
/// Everything structural is in place. The score dispatch is armed, the
/// tree is right — its launch sits under `cond.slot == 3`,
/// `SLOT_WANTS_ATTN_SCORE`, exactly where it should — and the buffers can
/// be valid and EMPTY, because a body the conditional never enters writes
/// nothing.
///
/// What blocks it is the interaction of two facts each of which is
/// correct alone:
///
/// * a capture must be taken on a WARM fire, because a launcher that
///   allocates its workspace on first use cannot do so inside a capture
///   (established this morning, the hard way — a C++ throw crossing the C
///   ABI and aborting with no message);
/// * a warm-up must walk a VALID program, because walking a union eagerly
///   runs both sides of every guard over the same rows.
///
/// Together they say: a union capture must warm every arm it will record,
/// and a warm-up that walks one valid program cannot warm the arms that
/// program does not take. The score dispatch's first use therefore lands
/// inside the capture, and it faults there.
///
/// The way out is presumably what the C++ arc calls DUAL-PREPARE — warm
/// each variant once before recording the union — and that is the next
/// thing to port. It is not a missing arm and not a missing buffer; both
/// of those were the answers to the previous two attempts, and both are
/// now done.
#[test]
fn the_union_captures_and_replays_the_same_decode() {
    zero_weight_decode(Leg::CapturedUnion);
}

/// Every arm the union states is now armed, and that claim moved.
///
/// This file used to carry a runtime probe: lower with every guard KEPT,
/// walk eagerly, and report the first arm that refused. It found the two
/// gaps it was built for — `pie_lora_qkv_correction`, which refused when a
/// fire staged no adapters, and `attn::write_kv_explicit_bf16`, which had
/// no arm at all — and then it could not survive its own success. With
/// everything armed, walking a union EAGERLY runs both sides of every
/// guard over the same rows, which is not a meaningful program: the
/// explicit KV write and the CSR-derived one both fire, and the fire
/// faults rather than refusing.
///
/// So the check is static now and lives where the other closed set does:
/// `executor_bind`'s corpus gained a `union_lowered` entry, so an arm that
/// only a guard's losing side states is inside `UNARMED` rather than
/// invisible to it. What a static check cannot see is an arm that EXISTS
/// and refuses at runtime, which is what the lora one did — that is
/// covered by actually capturing a union, which is A5.

/// The SECOND fire's tokens — a different set from the captured fire's
/// `[1, 2, 3, 5]`, and all four inside the patterned range so the
/// residual invariant stays exact for every row.
const SECOND_FIRE: [i32; 4] = [7, 11, 13, 17];

/// Which leg of the zero-weight decode to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Leg {
    /// Resolve the guards, issue onto the stream, assert the invariants.
    Eager,
    /// Resolve the guards, warm up, WIPE, capture, replay, assert the
    /// same invariants against what the replay alone produced.
    Captured,
    /// KEEP the guards and capture. The union records every arm and lets
    /// a conditional decide at replay, so this is the leg that asks
    /// whether the whole design produces the right numbers — not the
    /// mechanism in isolation (`gpu_supergraph`) and not the launch list
    /// in isolation (`union_lower`).
    CapturedUnion,
    /// CAPTURE ONCE, SERVE A SECOND FIRE. The exec is instantiated from
    /// one fire's tokens and then replayed against a DIFFERENT fire's,
    /// which is the property a cached exec has to have and the one no
    /// other leg asks for: every leg above replays the fire it captured.
    ///
    /// A capture bakes ADDRESSES, and the whole `FireArrays` design is
    /// the claim that only addresses are baked — the contents are
    /// refreshed per fire. If a capture baked contents instead, this leg
    /// returns the FIRST fire's answer for the second fire's tokens:
    /// fluent, plausible, and wrong, which is this tree's most-named
    /// failure mode.
    Reused,
    /// A fire whose LAST TWO ROWS carry attached programs, which makes
    /// `lower` split it on the hook axis. The tail then addresses rows at
    /// absolute offsets, takes `_devwin` statements, and needs its own
    /// prepared attention state — so this is the only leg that exercises
    /// a peel.
    Hooked,
}

fn zero_weight_decode(leg: Leg) {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, Resolver, run, run_captured,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Arg, Fire, GuardMode, Row, lower_with};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("full zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let mut alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 28;
    const KV_HEADS: i32 = 8;
    const Q_HEADS: i32 = 16;
    const HEAD_DIM: i32 = 128;
    const PAGE: i32 = 16;
    const ROWS: usize = 4;
    // The real vocabulary: the checkpoint is TIED, so the lm_head resolves
    // to "embed" and reads all [vocab, hidden] of it — a 64-row fake table
    // was the first version's illegal address. Pattern rows for the first
    // 64 tokens, zeros beyond: the logits become analytically checkable.
    const VOCAB: usize = 151_936;
    const PATTERNED: usize = 64;

    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let mut rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    if leg == Leg::Hooked {
        // A contiguous SUFFIX; `lower.rs::split_at` refuses anything else.
        for r in rows.iter_mut().skip(ROWS - 2) {
            r.hooked = true;
        }
    }
    // The captured leg KEEPS every guard: that is what makes one capture
    // able to serve fires that differ in their variant bits.
    let mode = if leg == Leg::CapturedUnion { GuardMode::Union } else { GuardMode::Resolve };
    let l = lower_with(&plan, &rows, Fire { captures_across_splits: false }, mode)
        .expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let mut arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // ── Weights: embed pattern, norm ones, everything else zero. ──
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; ROWS] = [1, 2, 3, 5];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let mut zeros_dev = alloc.alloc(8 * 3072 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    // ── Named pins, preallocated from the lowering's own widths. ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * w as usize * 2).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") || name.contains("lm_head") {
                if name.contains("lm_head") { self.zeros } else { self.embed }
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // ── The fire's KV side: pools, views, CSRs, write descriptors. ──
    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 4,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            // The NATIVE alias the C++ `layer_view` maintains: the dispatch
            // reads the bf16 MIRROR planes, and for a native cache those
            // are the storage pages themselves.
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("csr");
        b.copy_from_host(data, stream.as_ref()).expect("h2d csr");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1, 2, 3]));
    let csr_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let csr_lens = up(&u32s(&[1, 1, 1, 1]));
    let w_page = up(&u32s(&[0, 1, 2, 3]));
    let w_off = up(&u32s(&[0, 0, 0, 0]));
    let row_valid = up(&[1u8, 1, 1, 1]);
    let mut ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions = up(&[0i32, 0, 0, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    // ── Workspace + the planned decode cache. ──
    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(&[0, 1, 2, 3, 4], Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1);
    ws.end_plan_update(&mut sops, raw_stream);

    // The guard-owned attention values: q is the dispatch's Named arg
    // (the observed-query pin), o is what the following o_proj reads.
    let fi = l
        .launches
        .iter()
        .position(|x| l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_decode")
        .expect("a decode fire dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    let o_off = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => *at,
        other => panic!("o_proj reads the attention slot, got {other:?}"),
    };

    // The TAIL region's prepared attention state, for the peeled leg.
    //
    // A peel's tail serves rows [split, N) — a different row count, a
    // different set of requests, and therefore a different plan, different
    // KV page CSRs, and output pins that start at the tail's first row.
    // `AttnRegions` is what hands it to the arm; building it here is what
    // makes the peel gate a real test of the handing-over rather than of
    // the vocabulary alone.
    //
    // The CSR indptr is REBASED: FlashInfer reads a prefix sum that starts
    // at zero, so a sub-batch cannot borrow the fire's.
    let split = rows.iter().position(|r| r.hooked).unwrap_or(0);
    let tail_rows = ROWS - split;
    let tail_indptr = up(&u32s(&(0..=tail_rows as u32).collect::<Vec<_>>()));
    let tail_lens = up(&u32s(&vec![1u32; tail_rows]));
    let tail_indices = up(&u32s(&(split as u32..ROWS as u32).collect::<Vec<_>>()));
    let mut tail_plan = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin tail plan");
    tail_plan.plan_decode(
        &(0..=tail_rows as u32).collect::<Vec<_>>(),
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE,
        ws.view(),
        raw_stream,
        false,
        -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);


    // SCORE buffers: valid, stable, and EMPTY.
    //
    // The union records the score-capturing decode dispatch whether or not
    // this fire wants scores, so the addresses must be real — a null
    // faults the instant `WantsAttnScore` goes true. They may be empty,
    // and that is the trick: the CSR says every request folds zero rows,
    // so a body that did run would write nothing, and the conditional
    // means it does not run at all.
    //
    // A later fire in this bucket that DOES want scores needs a bigger
    // slot, and growing it moves the base — which is precisely what
    // `PlanEpoch` exists to notice. Growth bumps the epoch, the captured
    // exec goes stale, and the bucket recaptures. A cost, not a wrong
    // answer.
    let score_indptr = up(&u32s(&vec![0u32; ROWS + 1]));
    let scores = alloc.alloc(4).expect("scores");

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        q_out: named_bufs[&q_pin_value].as_ptr(),
        score_out: scores.as_ptr().cast(),
        score_indptr_d: score_indptr.as_ptr().cast(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        o_out: unsafe { arena.as_ptr().cast::<u8>().add(o_off) }.cast(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: core::ptr::null(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: ROWS as i32,
        num_pages_in_batch: 4,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    // The tail's state: the fire's, with the region's plan, its rebased
    // CSRs, and pins that start at the tail's first row. Everything else
    // (workspace, layers, geometry) is fire-wide and shared.
    let attn_tail = AttnCtx {
        decode_plan: tail_plan.as_ptr(),
        kv_page_indices_d: tail_indices.as_ptr().cast(),
        kv_page_indptr_d: tail_indptr.as_ptr().cast(),
        kv_last_page_lens_d: tail_lens.as_ptr().cast(),
        num_requests: tail_rows as i32,
        first_token: split as i32,
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        o_out: unsafe {
            arena.as_ptr().cast::<u8>().add(o_off + split * HIDDEN * 2)
        }
        .cast(),
        lse_out_d: unsafe {
            lse.as_ptr().cast::<u8>().add(split * Q_HEADS as usize * 4)
        }
        .cast(),
        ..attn.clone()
    };
    let regions = if leg == Leg::Hooked {
        AttnRegions::split(&attn, &attn_tail)
    } else {
        AttnRegions::whole(Some(&attn))
    };


    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let mut peel_win =
        driver_cuda_new::cuda::PeelWindowWord::new(&alloc).expect("peel window word");
    // The window is the TAIL's, not the fire's. `_devwin` statements only
    // occur in a peel's tail region, and the word is what tells them which
    // absolute rows are theirs — publishing the whole fire makes the tail
    // write the prefix's rows too, which is an out-of-bounds store into
    // buffers the lowering sized for the region.
    let split = rows.iter().position(|r| r.hooked).unwrap_or(0) as u32;
    peel_win.set(split, ROWS as u32 - split);
    peel_win.upload(stream.as_ref()).expect("publish the peel window");
    stream.as_ref().synchronize().expect("the window lands");

    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e6,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        // The peel's tail begins where the marked suffix does. The word is
        // what a `_devwin` launch early-outs on, and publishing the WHOLE
        // fire here is right for both legs: the unpeeled one has no split,
        // and the hooked one's regions between them cover every row.
        peel_window: peel_win.device_ptr(),
        rows_total: ROWS as i32,
    };

    // ── The walk: every launch, no refusals allowed. ──
    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let ran = if matches!(leg, Leg::Captured | Leg::CapturedUnion | Leg::Reused) {
        use driver_cuda_new::cuda::{PredicateWord, SupergraphBuilder};
        use driver_cuda_new::model::supergraph::fire_predicates;

        // The word is filled and uploaded BEFORE the capture opens: the
        // host decides the fire's bits, the graph reads them.
        let mut preds = PredicateWord::new(&alloc).expect("predicate word");
        // Under `Union` the conditionals are real and this word is what
        // decides them; under `Resolve` the tree is empty and it decides
        // nothing. The same call serves both, which is the point of
        // `fire_predicates` reading the tree rather than a fire's flags.
        fire_predicates(&rows, &l.conds, &mut preds).expect("the fire's bits");
        preds.upload(stream.as_ref()).expect("upload");
        stream.as_ref().synchronize().expect("the word lands before the capture");

        // DUAL-PREPARE: warm each VARIANT, not just this fire.
        //
        // A capture must be taken warm, because a launcher that allocates
        // its workspace on first use cannot do so inside a capture -- the
        // refusal becomes a C++ `throw` crossing the C ABI, which aborts
        // the process. A warm-up must walk a VALID program, because
        // walking a union eagerly runs both sides of every guard over the
        // same rows. And a union records arms that no single valid program
        // takes.
        //
        // So warm once per variant, each with its own RESOLVED lowering --
        // every one a real program -- and let the union of their arms
        // cover the union lowering's. The numbers they produce are
        // discarded: the arena is wiped below, and what survives is the
        // allocation each launcher did on its first call. This is what the
        // C++ arc calls dual-prepare.
        for marks in [
            Row { samples: true, ..Row::default() },
            Row { samples: true, wants_scores: true, ..Row::default() },
            Row { samples: true, write_desc: true, ..Row::default() },
        ] {
            let warm_rows: Vec<Row> = vec![marks; ROWS];
            let warm = lower_with(
                &plan,
                &warm_rows,
                Fire { captures_across_splits: false },
                GuardMode::Resolve,
            )
            .expect("the warm-up lowers");
            let warm_dplan = DispatchPlan::new(&plan, &warm);
            run(&warm, &warm_dplan, frame, &mut resolver, &ctx, regions, None)
                .unwrap_or_else(|e| panic!("the warm-up walk refused: {e:?}"));
            stream.as_ref().synchronize().expect("the warm-up retires");
        }
        stream.as_ref().synchronize().expect("the warm-up retires");

        // Wipe what the warm-up computed, so that what the invariants
        // read below can only have come from the REPLAY.
        arena.memset(0, stream.as_ref()).expect("wipe the arena");
        stream.as_ref().synchronize().expect("the wipe lands");

        let (ran, graph) = {
            let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
            let mut b = SupergraphBuilder::new(scope.stream(), &preds);
            let ran = run_captured(
                &l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), None, &mut b,
            )
            .unwrap_or_else(|e| panic!("the captured walk refused: {e:?}"));
            drop(b);
            (ran, scope.end().expect("end capture"))
        };
        let exec = graph.instantiate().expect("instantiate");

        // ── A5 AT WIDTH: one exec, two structurally distinct programs ──
        //
        // This is the claim the whole workstream exists to make. The same
        // instantiated graph is launched twice; the ONLY thing that
        // changes between them is one byte of the device predicate word,
        // and that byte selects a different KV-write program — the
        // explicit descriptor form on one launch and the CSR-derived form
        // on the other. Different launches, different kernels, one exec.
        //
        // They must agree. Both write the same cells to the same pages, so
        // the logits that come back are the test: if the conditional were
        // not selecting, or were selecting the wrong body, or the union
        // had folded two programs that are not equivalent, these two byte
        // arrays would differ.
        //
        // The non-zero check is what keeps the equality from being
        // vacuous — two empty buffers also match.
        let lv = logits_value.expect("the last launch writes a named pin");
        let mut lanes: Vec<Vec<u8>> = Vec::new();
        for has_write_desc in [false, true] {
            arena.memset(0, stream.as_ref()).expect("wipe between lanes");
            preds
                .set(driver_cuda_new::cuda::SLOT_HAS_WRITE_DESC, has_write_desc)
                .expect("slot");
            preds.upload(stream.as_ref()).expect("upload");
            stream.as_ref().synchronize().expect("the word lands");
            exec.launch(stream.as_ref()).expect("replay");
            stream.as_ref().synchronize().expect("the replay retires");
            let mut back = vec![0u8; named_bufs[&lv].len()];
            named_bufs[&lv]
                .copy_to_host(&mut back, stream.as_ref())
                .expect("d2h logits");
            stream.as_ref().synchronize().expect("sync");
            lanes.push(back);
        }
        assert!(
            lanes[0].iter().any(|&b| b != 0),
            "the lanes agree only because both are empty"
        );
        assert_eq!(
            lanes[0], lanes[1],
            "ONE exec produced different logits for two structurally \
             distinct programs — the union folded something it should not \
             have, or the conditional is not selecting"
        );

        // ── A SECOND FIRE, on the exec the first one captured ──────────
        //
        // Every leg above replays the fire it CAPTURED, which cannot tell
        // a baked address from baked contents — the answer is right
        // either way. A cache is only worth having if the exec serves the
        // NEXT fire too, so this writes different tokens into the same
        // descriptor buffer and launches the same exec again.
        //
        // The residual invariant below is the discriminator and it is
        // exact: row `r` must be the embed pattern for `SECOND_FIRE[r]`.
        // Had the capture baked contents, this replay returns the first
        // fire's rows.
        if leg == Leg::Reused {
            let bytes: Vec<u8> =
                SECOND_FIRE.iter().flat_map(|t| t.to_le_bytes()).collect();
            ids.copy_from_host(&bytes, stream.as_ref()).expect("the next fire's tokens");
            arena.memset(0, stream.as_ref()).expect("wipe before the second fire");
            preds
                .set(driver_cuda_new::cuda::SLOT_HAS_WRITE_DESC, false)
                .expect("slot");
            preds.upload(stream.as_ref()).expect("upload");
            stream.as_ref().synchronize().expect("the tokens land");
            exec.launch(stream.as_ref()).expect("the second fire replays");
            stream.as_ref().synchronize().expect("the second fire retires");
        }
        ran
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, regions, None)
            .unwrap_or_else(|e| panic!("the walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole decode retires");

    // ── Invariant 1: the residual equals the embed rows, bit-exactly. ──
    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    // Whichever fire ran LAST is the one the arena holds.
    let fired: &[i32] = if leg == Leg::Reused { &SECOND_FIRE } else { &tokens };
    for (r, t) in fired.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    // ── Invariant 2: the logits are the tied lm_head's exact algebra. ──
    //
    // The residual is the embed row (invariant 1), the final norm turns it
    // into alternating ±1, and the tied lm_head dots that against every
    // pattern row: logit[r][t] = Σ_c (±1)(±amp(t)) = HIDDEN · amp(t) for
    // t < PATTERNED (the signs align by construction), 0 beyond. The same
    // value for EVERY row r — and bf16-representable at every checked t.
    let lv = logits_value.expect("the last launch writes a named pin (the logits)");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..ROWS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(HIDDEN as f32 * amp(t as i32));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The FULL prefill: the decode walk's twin over `qwen3_0_6b`'s real
/// prefill lowering — two requests (3 + 4 tokens), the five prefill arms
/// (split, the staged in-place rope, the KV write, the dequant staging,
/// the planned FlashInfer prefill), same zero-weight algebra: residual ==
/// embed rows bit-exactly, logits == the tied lm_head's dot with the
/// pattern table. Causal attention over zero V is zero, so every layer's
/// residual fold adds nothing.
#[test]
fn the_full_zero_weight_prefill_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DispatchCtx, DispatchPlan, Frame, PrefillPlan, Resolver, run,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("full zero-weight prefill") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 28;
    const KV_HEADS: i32 = 8;
    const Q_HEADS: i32 = 16;
    const HEAD_DIM: i32 = 128;
    const PAGE: i32 = 16;
    const TOKENS: usize = 7;
    const VOCAB: usize = 151_936;
    const PATTERNED: usize = 64;

    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Prefill,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; TOKENS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; TOKENS] = [1, 2, 3, 5, 7, 11, 13];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let mut zeros_dev = alloc.alloc(8 * 3072 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(TOKENS * w as usize * 2).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") {
                self.embed
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // Two requests: tokens [0,3) and [3,7), one page each.
    let plane = (2 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 2,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 3] = [0, 3, 7];
    let page_indptr_h: [u32; 3] = [0, 1, 2];
    let last_lens_h: [u32; 2] = [3, 4];
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 1, 2, 0, 1, 2, 3].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h, &page_indptr_h, &last_lens_h,
        Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_prefill_bf16"
        })
        .expect("a prefill fire dispatches attention");
    let o_off = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => *at,
        other => panic!("o_proj reads the attention slot, got {other:?}"),
    };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        layers,
        q_out: core::ptr::null_mut(),
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        o_out: unsafe { arena.as_ptr().cast::<u8>().add(o_off) }.cast(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: 2,
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e6,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &named_bufs,
    };
    let ran = run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), None)
        .unwrap_or_else(|e| panic!("the walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole prefill retires");

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..TOKENS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(HIDDEN as f32 * amp(t as i32));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The qwen3_5 HYBRID's decode, walked end to end on device (E-gate
/// family #1's GPU smoke): 24 layers — 18 GDN (conv → prep → bf16-state
/// recurrence → gated norm) and 6 full-attention (2×-wide gated q, partial
/// rope, flashinfer) — against driver-owned conv/recurrent state slabs and
/// a per-layer seam-value pool, with synthetic weights chosen so the
/// residual stream is analytically checkable:
///
/// * every in-projection reads only EVEN channels (the embed pattern
///   alternates sign per channel, so an even-only weight sums to a real
///   value instead of cancelling to zero — finite activations, no NaN);
/// * every landing projection (`o_proj`, `down`) is zero, so the residual
///   equals the embed rows BIT-EXACTLY after 24 layers (invariant 1);
/// * the tied lm_head then dots the Gemma-folded final norm (±2) against
///   the pattern rows: `logit[r][t] = 2 · HIDDEN · amp(t)` (invariant 2).
#[test]
#[allow(clippy::too_many_lines)]
fn the_hybrid_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, GdnCtx, Resolver, run,
    };
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("hybrid zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 24;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const PAGE: i32 = 16;
    const ROWS: usize = 4;
    const VOCAB: usize = 248_320;
    const PATTERNED: usize = 64;
    // GDN geometry (the 0.8B facts' own numbers).
    const K_H: i32 = 16;
    const V_H: i32 = 16;
    const K_D: i32 = 128;
    const V_D: i32 = 128;
    const CONV_DIM: i32 = 6144;
    const CONV_K: i32 = 4;
    const SLOTS: usize = 4;

    let hybrid = Qwen35HybridFacts::qwen3_5_0_8b();
    // The LIVE L40S cuda set (`emissions.rs`), not the synthetic fixture.
    let cuda = Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        // Dense BF16, whole context — this fixture's own frame.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        window_left: Vec::new(),
    };
    let plan = qwen3_5_hybrid_cuda(&hybrid, &cuda, FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // ── Weights. Embed: the pattern rows. In-projections: even-only
    // small. Landings: zero. Norms: ones (bf16 or fp32 as consumed). ──
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; ROWS] = [1, 2, 3, 5];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    // Even-channel-only in-projection bank, big enough for the widest
    // ([CONV_DIM, HIDDEN]) and reused by every in-proj and the conv (whose
    // [CONV_DIM, 1, K] flat layout just reads a prefix).
    let inproj_elems = CONV_DIM as usize * HIDDEN;
    let mut inproj_host = vec![0u8; inproj_elems * 2];
    let small = bf16(1.0 / 1024.0).to_le_bytes();
    for j in 0..inproj_elems {
        if j % 2 == 0 {
            inproj_host[j * 2] = small[0];
            inproj_host[j * 2 + 1] = small[1];
        }
    }
    let mut inproj_dev = alloc.alloc(inproj_host.len()).expect("inproj");
    inproj_dev.copy_from_host(&inproj_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let ones_f32: Vec<u8> =
        std::iter::repeat_n(1.0f32.to_le_bytes(), V_D as usize).flatten().collect();
    let mut ones_f32_dev = alloc.alloc(ones_f32.len()).expect("ones f32");
    ones_f32_dev.copy_from_host(&ones_f32, stream.as_ref()).expect("h2d");
    let mut zeros_f32_dev = alloc.alloc(V_H as usize * 4).expect("zeros f32");
    zeros_f32_dev.memset(0, stream.as_ref()).expect("zero");
    let mut zeros_dev = alloc.alloc(2 * 3584 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    // ── The seam-value pool: every Named value the lowering states,
    // allocated at fp32 width (the widest dtype any pin carries). ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * w as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        inproj: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        ones_f32: *const std::ffi::c_void,
        zeros_f32: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            if name.ends_with("conv_bias") {
                return None; // the 0.8B conv has no bias — the null path
            }
            Some(if name == "embed" {
                self.embed
            } else if name.contains("a_log") {
                self.zeros_f32
            } else if name.contains("gate_norm") {
                self.ones_f32
            } else if name.contains("dt_bias") {
                self.zeros
            } else if name.contains("norm") {
                self.ones
            } else if name.contains("in_proj")
                || name.contains("conv")
                || name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
            {
                self.inproj
            } else {
                // o_proj, gate_up, down — the landings stay zero so the
                // residual is the embed rows, exactly.
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // ── KV pools for the SIX full-attention layers; placeholder views
    // (never dereferenced) at the GDN indices. ──
    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if !hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                Some((k, v))
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, kv)| {
            let (k, v) = kv.as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(i).expect("layer"),
                num_pages: 4,
                page_size: PAGE,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scheme: KvCacheScheme::Native,
                storage_dtype: DType::Bf16,
                block_size: 0,
                k_pages: k,
                v_pages: v,
                k_scales: core::ptr::null_mut(),
                v_scales: core::ptr::null_mut(),
                k_bf16_pages: k,
                v_bf16_pages: v,
                k_env_min: core::ptr::null_mut(),
                k_env_max: core::ptr::null_mut(),
                hnd_layout: false,
                native_bf16: true,
            }
        })
        .collect();

    // ── GDN state: conv + recurrent slabs for the EIGHTEEN linear
    // layers, slot-indirected. ──
    let conv_stride = (CONV_K * CONV_DIM) as usize;
    let state_stride = (V_H * K_D * V_D) as usize;
    let gdn_slabs: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut c = alloc.alloc(SLOTS * conv_stride * 2).expect("conv slab");
                let mut s = alloc.alloc(SLOTS * state_stride * 2).expect("state slab");
                c.memset(0, stream.as_ref()).expect("zc");
                s.memset(0, stream.as_ref()).expect("zs");
                Some((c, s))
            })
            .collect();
    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let slot_ids =
        up(&[0i32, 1, 2, 3].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let gdn = GdnCtx {
        k_h: K_H,
        v_h: V_H,
        k_d: K_D,
        v_d: V_D,
        conv_dim: CONV_DIM,
        conv_k: CONV_K,
        n_groups: 0,
        conv_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
            .collect(),
        conv_stride_elems: i64::try_from(conv_stride).expect("stride"),
        recurrent_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
            .collect(),
        state_stride_elems: i64::try_from(state_stride).expect("stride"),
        slot_ids_d: slot_ids.as_ptr().cast(),
        write_state: true,
    };

    // ── CSRs, write descriptors, plan, workspace. ──
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1, 2, 3]));
    let csr_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let csr_lens = up(&u32s(&[1, 1, 1, 1]));
    let w_page = up(&u32s(&[0, 1, 2, 3]));
    let w_off = up(&u32s(&[0, 0, 0, 0]));
    let row_valid = up(&[1u8, 1, 1, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 0, 0, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(
        &[0, 1, 2, 3, 4],
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE,
        ws.view(),
        raw_stream,
        false,
        -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_decode"
        })
        .expect("the hybrid decode dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    // The attention output is guard-owned (the dispatch launch records no
    // SSA outputs); the launch AFTER the dispatch reads it first — the
    // sigmoid output gate's `x`.
    let o_out: *mut std::ffi::c_void =
        match &l.args[l.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            other => panic!("the gate reads the attention slot, got {other:?}"),
        };

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        q_out: named_bufs[&q_pin_value].as_ptr(),
        o_out,
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        // The hybrid writes KV through the EXPLICIT kernel, which walks
        // the qo CSR even on decode — one row per request, trivially.
        qo_indptr_d: csr_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: i32::try_from(ROWS).expect("rows"),
        num_pages_in_batch: 4,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e6,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: i32::try_from(VOCAB).expect("vocab"),
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    // ── The walk. ──
    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        inproj: inproj_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        ones_f32: ones_f32_dev.as_ptr(),
        zeros_f32: zeros_f32_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let per_launch_sync = std::env::var("HYBRID_SMOKE_SYNC").is_ok();
    let ran = if per_launch_sync {
        use driver_cuda_new::model::executor::{bind, dispatch};
        for (i, launch) in l.launches.iter().enumerate() {
            let kernel = l.kernels[launch.kernel as usize].clone();
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: bind {e:?}"));
            dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, Some(&attn), Some(&gdn))
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: dispatch {e:?}"));
            stream
                .as_ref()
                .synchronize()
                .unwrap_or_else(|e| panic!("launch {i} {kernel} left the stream poisoned: {e:?}"));
        }
        l.launches.len()
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), Some(&gdn))
            .unwrap_or_else(|e| panic!("the hybrid walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole hybrid decode retires");

    // ── Invariant 1: the residual equals the embed rows, bit-exactly —
    // 18 GDN and 6 attention landings all through zero projections. ──
    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    // ── Invariant 2: the logits are the tied lm_head's exact algebra —
    // the GEMMA final norm folds (1 + 1), so ±2 against ±amp(t):
    // logit[r][t] = 2 · HIDDEN · amp(t). ──
    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..ROWS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(2.0 * HIDDEN as f32 * amp(i32::try_from(t).expect("t")));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The hybrid's PREFILL, walked the same way: two requests over seven
/// tokens, the conv prefill walk + chunked FLA recurrence advancing the
/// GDN state slabs, flashinfer prefill on the full-attention layers.
/// Same synthetic weights, same two invariants as the decode smoke.
#[test]
#[allow(clippy::too_many_lines)]
fn the_hybrid_zero_weight_prefill_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DispatchCtx, DispatchPlan, Frame, GdnCtx, PrefillPlan, Resolver, run,
    };
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("hybrid zero-weight prefill") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 24;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const PAGE: i32 = 16;
    const TOKENS: usize = 7;
    const REQUESTS: usize = 2;
    const VOCAB: usize = 248_320;
    const PATTERNED: usize = 64;
    const K_H: i32 = 16;
    const V_H: i32 = 16;
    const K_D: i32 = 128;
    const V_D: i32 = 128;
    const CONV_DIM: i32 = 6144;
    const CONV_K: i32 = 4;
    const SLOTS: usize = 4;

    let hybrid = Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        // Dense BF16, whole context — this fixture's own frame.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        window_left: Vec::new(),
    };
    let plan = qwen3_5_hybrid_cuda(&hybrid, &cuda, FireClass::Prefill);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; TOKENS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; TOKENS] = [1, 2, 3, 5, 7, 11, 13];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let inproj_elems = CONV_DIM as usize * HIDDEN;
    let mut inproj_host = vec![0u8; inproj_elems * 2];
    let small = bf16(1.0 / 1024.0).to_le_bytes();
    for j in 0..inproj_elems {
        if j % 2 == 0 {
            inproj_host[j * 2] = small[0];
            inproj_host[j * 2 + 1] = small[1];
        }
    }
    let mut inproj_dev = alloc.alloc(inproj_host.len()).expect("inproj");
    inproj_dev.copy_from_host(&inproj_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let ones_f32: Vec<u8> =
        std::iter::repeat_n(1.0f32.to_le_bytes(), V_D as usize).flatten().collect();
    let mut ones_f32_dev = alloc.alloc(ones_f32.len()).expect("ones f32");
    ones_f32_dev.copy_from_host(&ones_f32, stream.as_ref()).expect("h2d");
    let mut zeros_f32_dev = alloc.alloc(V_H as usize * 4).expect("zeros f32");
    zeros_f32_dev.memset(0, stream.as_ref()).expect("zero");
    let mut zeros_dev = alloc.alloc(2 * 3584 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(TOKENS * w as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        inproj: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        ones_f32: *const std::ffi::c_void,
        zeros_f32: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            if name.ends_with("conv_bias") {
                return None;
            }
            Some(if name == "embed" {
                self.embed
            } else if name.contains("a_log") {
                self.zeros_f32
            } else if name.contains("gate_norm") {
                self.ones_f32
            } else if name.contains("dt_bias") {
                self.zeros
            } else if name.contains("norm") {
                self.ones
            } else if name.contains("in_proj")
                || name.contains("conv")
                || name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
            {
                self.inproj
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    let plane = (2 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if !hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                Some((k, v))
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, kv)| {
            let (k, v) = kv.as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(i).expect("layer"),
                num_pages: 2,
                page_size: PAGE,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scheme: KvCacheScheme::Native,
                storage_dtype: DType::Bf16,
                block_size: 0,
                k_pages: k,
                v_pages: v,
                k_scales: core::ptr::null_mut(),
                v_scales: core::ptr::null_mut(),
                k_bf16_pages: k,
                v_bf16_pages: v,
                k_env_min: core::ptr::null_mut(),
                k_env_max: core::ptr::null_mut(),
                hnd_layout: false,
                native_bf16: true,
            }
        })
        .collect();

    let conv_stride = (CONV_K * CONV_DIM) as usize;
    let state_stride = (V_H * K_D * V_D) as usize;
    let gdn_slabs: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut c = alloc.alloc(SLOTS * conv_stride * 2).expect("conv slab");
                let mut s = alloc.alloc(SLOTS * state_stride * 2).expect("state slab");
                c.memset(0, stream.as_ref()).expect("zc");
                s.memset(0, stream.as_ref()).expect("zs");
                Some((c, s))
            })
            .collect();
    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let slot_ids = up(&[0i32, 1].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let gdn = GdnCtx {
        k_h: K_H,
        v_h: V_H,
        k_d: K_D,
        v_d: V_D,
        conv_dim: CONV_DIM,
        conv_k: CONV_K,
        n_groups: 0,
        conv_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
            .collect(),
        conv_stride_elems: i64::try_from(conv_stride).expect("stride"),
        recurrent_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
            .collect(),
        state_stride_elems: i64::try_from(state_stride).expect("stride"),
        slot_ids_d: slot_ids.as_ptr().cast(),
        write_state: true,
    };

    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 3] = [0, 3, 7];
    let page_indptr_h: [u32; 3] = [0, 1, 2];
    let last_lens_h: [u32; 2] = [3, 4];
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 1, 2, 0, 1, 2, 3].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h, &page_indptr_h, &last_lens_h,
        Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_prefill_bf16"
        })
        .expect("the hybrid prefill dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    let o_out: *mut std::ffi::c_void =
        match &l.args[l.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            other => panic!("the gate reads the attention slot, got {other:?}"),
        };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        layers,
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        q_out: named_bufs[&q_pin_value].as_ptr(),
        o_out,
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: i32::try_from(REQUESTS).expect("requests"),
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e6,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: i32::try_from(VOCAB).expect("vocab"),
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        inproj: inproj_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        ones_f32: ones_f32_dev.as_ptr(),
        zeros_f32: zeros_f32_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let ran = run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), Some(&gdn))
        .unwrap_or_else(|e| panic!("the hybrid prefill walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole hybrid prefill retires");

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..TOKENS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(2.0 * HIDDEN as f32 * amp(i32::try_from(t).expect("t")));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// nemotron_h's DECODE, walked live for the first time: four requests
/// through the synthetic fixture's six layers — three mamba (split →
/// conv update → params prep → dt/dA → selective scan → gated norm),
/// one attention, two bare-MLP — with the sigmoid-routed MoE after
/// every mixer layer. Zero weights everywhere except a patterned embed
/// (first 64 tokens), ones on the norms, and the SAME pattern bank as
/// the lm_head, so the two invariants of the other zero-weight walks
/// hold here verbatim: the residual equals the embed rows bit-exactly
/// (every mixer and MoE landing is zero — the router's sigmoid(0) = 0.5
/// weights combine zero expert outputs), and the logits are the exact
/// dot of ±1 against the pattern rows. The first LIVE run of all eleven
/// nemotron arms, including the `LaunchSpec::aux` cross-statement
/// wiring (the scan consumes the split's raw dt and the prep's fp32
/// tables on finite-but-nonzero paths: a = -exp(0) = -1, dt =
/// softplus(0) ≈ 0.693, dA = 0.5 — all finite, all multiplied into a
/// zero state).
#[test]
fn the_nemotron_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, GdnCtx, Resolver, run,
    };
    use model::nemotron_h::forward::facts::NemotronHFacts;
    use model::nemotron_h::forward::nemotron_h_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("nemotron zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 2048;
    const LAYERS: usize = 6;
    const KV_HEADS: i32 = 4;
    const Q_HEADS: i32 = 16;
    const HEAD_DIM: i32 = 128;
    const PAGE: i32 = 16;
    const ROWS: usize = 4;
    const VOCAB: usize = 131_072;
    const PATTERNED: usize = 64;
    // The mamba geometry, the fixture's own.
    const M_HEADS: i32 = 16;
    const M_HEAD_DIM: i32 = 64;
    const STATE: i32 = 128;
    const GROUPS: i32 = 8;
    const CONV_DIM: i32 = 3072;
    const CONV_K: i32 = 4;

    let facts = NemotronHFacts::nemotron_h_synthetic();
    let plan = nemotron_h_cuda(&facts, FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // ── Weights: embed pattern (also the lm_head), norm ones, zeros. ──
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; ROWS] = [1, 2, 3, 5];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    // Big enough for the widest zero bank: the stacked expert banks
    // ([E=32, i=1024, h=2048]) and the in-projection ([4112, 2048]).
    let mut zeros_dev = alloc.alloc(32 * 1024 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    // ── Named pins (×4: the dt/dA pins are fp32). ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * w.max(1) as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") || name.contains("lm_head") {
                self.embed
            } else if name.ends_with("norm") || name.contains("_norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // ── KV pools (uniform; only the attention layer reads its own). ──
    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 4,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    // ── The mamba slabs: one slot per request on layers 0/2/4. ──
    let conv_stride = (CONV_K * CONV_DIM) as usize;
    let state_stride = (M_HEADS * STATE * M_HEAD_DIM) as usize;
    let slabs: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS as u32)
            .map(|i| {
                use model::nemotron_h::forward::facts::NemotronLayerKind;
                if facts.kind(i) != NemotronLayerKind::Mamba {
                    return None;
                }
                let mut c = alloc.alloc(ROWS * conv_stride * 2).expect("conv slab");
                let mut s = alloc.alloc(ROWS * state_stride * 2).expect("state slab");
                c.memset(0, stream.as_ref()).expect("zc");
                s.memset(0, stream.as_ref()).expect("zs");
                Some((c, s))
            })
            .collect();
    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let slot_ids = up(&[0i32, 1, 2, 3].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let gdn = GdnCtx {
        k_h: GROUPS,
        v_h: M_HEADS,
        k_d: STATE,
        v_d: M_HEAD_DIM,
        conv_dim: CONV_DIM,
        conv_k: CONV_K,
        n_groups: GROUPS,
        conv_state: slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
            .collect(),
        conv_stride_elems: i64::try_from(conv_stride).expect("stride"),
        recurrent_state: slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
            .collect(),
        state_stride_elems: i64::try_from(state_stride).expect("stride"),
        slot_ids_d: slot_ids.as_ptr().cast(),
        write_state: true,
    };

    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1, 2, 3]));
    let csr_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let csr_lens = up(&u32s(&[1, 1, 1, 1]));
    let qo_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let w_page = up(&u32s(&[0, 1, 2, 3]));
    let w_off = up(&u32s(&[0, 0, 0, 0]));
    let row_valid = up(&[1u8, 1, 1, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 0, 0, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(
        &[0, 1, 2, 3, 4], Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    // The attention pins, discovered tolerantly: a 2-arg dispatch states
    // its own [q, o] and needs neither.
    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_decode"
        })
        .expect("the decode dispatches attention");
    let dispatch_args = l.launches[fi].args.end - l.launches[fi].args.start;
    let (q_out, o_out) = if dispatch_args >= 2 {
        (core::ptr::null_mut(), core::ptr::null_mut())
    } else {
        let q_pin = match &l.args[l.launches[fi].args.start as usize] {
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Weight(w) => panic!("the dispatch's q is a weight {w}?"),
        };
        let o = match &l.args[l.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            Arg::Weight(w) => panic!("o_proj reads a weight {w}?"),
        };
        (q_pin, o)
    };

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        q_out,
        o_out,
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: ROWS as i32,
        num_pages_in_batch: 4,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e4,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let per_launch_sync = std::env::var("NEMOTRON_SMOKE_SYNC").is_ok();
    let ran = if per_launch_sync {
        use driver_cuda_new::model::executor::{bind, dispatch};
        for (i, launch) in l.launches.iter().enumerate() {
            let kernel = l.kernels[launch.kernel as usize].clone();
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: bind {e:?}"));
            dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, Some(&attn), Some(&gdn))
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: dispatch {e:?}"));
            stream
                .as_ref()
                .synchronize()
                .unwrap_or_else(|e| panic!("launch {i} {kernel} poisoned the stream: {e:?}"));
        }
        l.launches.len()
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), Some(&gdn))
            .unwrap_or_else(|e| panic!("the nemotron walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole decode retires");

    // ── Invariant 1: the residual equals the embed rows, bit-exactly. ──
    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 2047] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    // ── Invariant 2: logits = the pattern bank's exact dot. ──
    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..ROWS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(HIDDEN as f32 * amp(t as i32));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The qwen3_vl vision TOWER through its bridge row — the first live fire
/// of the tower-granularity VL judgment. A tiny synthetic tower (depth 1,
/// hidden 128 / head_dim 64 — flashinfer's real instantiation — merge 2×2,
/// one 4-patch image → ONE merged token) with ALL-ZERO weights: the patch
/// projection, the block, the merger and the pos-embed table are zeros, so
/// the tower's merged token is exactly zero — and the SCATTER must
/// OVERWRITE the anchor row of a hidden buffer pre-filled with a nonzero
/// pattern. Zeroed anchor row + untouched neighbor row = the whole
/// pipeline (host prep, tower, merger, scatter) ran and landed exactly
/// where the anchors say.
#[test]
fn the_qwen3vl_tower_fires_through_the_bridge() {
    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("qwen3_vl tower fire") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: i32 = 128;
    const HEADS: i32 = 2;
    const INTER: i32 = 128;
    const PATCH: i32 = 16;
    const T_PATCH: i32 = 2;
    const MERGE: i32 = 2;
    const IN_CH: i32 = 3;
    const OUT_HIDDEN: i32 = 96;
    const NUM_POS: i32 = 2304;
    const PATCH_DIM: usize = (IN_CH * T_PATCH * PATCH * PATCH) as usize; // 1536
    const N_PATCH: usize = 4; // grid 1×2×2 → one merged token
    const N_ROWS: usize = 2; // anchor row 0 + a guard row 1

    // One zero bank big enough for every weight the tower reads
    // (largest: fc1 [4*hidden, 4*hidden] = 512×512).
    let mut zeros = alloc.alloc(512 * 512 * 2).expect("zeros");
    zeros.memset(0, stream.as_ref()).expect("z");
    // The pos-embed table [2304, hidden] — zeros too.
    let mut pos_tbl = alloc.alloc(NUM_POS as usize * HIDDEN as usize * 2).expect("pos");
    pos_tbl.memset(0, stream.as_ref()).expect("z");

    let z = zeros.as_ptr().cast_const();
    // depth 1 → 12 block pointers; merger 6; no deepstack.
    let block_w: Vec<*const std::ffi::c_void> = vec![z; 12];
    let merger_w: Vec<*const std::ffi::c_void> = vec![z; 6];
    // THREE deepstack mergers (the real tower's count) — the smoke's
    // depth-1 block makes every deepstack tap fire at layer 0.
    let deepstack_w: Vec<*const std::ffi::c_void> = vec![z; 3 * 6];
    let deepstack_layers: [i32; 3] = [0, 0, 0];

    // The image: 4 patches of f32 pixels (host), grid (t,h,w) = (1,2,2).
    let pixels = vec![0.25f32; N_PATCH * PATCH_DIM];
    let pixel_indptr: [u32; 2] = [0, (N_PATCH * PATCH_DIM * 4) as u32];
    let grids: [u32; 3] = [1, 2, 2];
    let anchors: [u32; 1] = [0];

    // The hidden rows, pre-filled with a nonzero pattern.
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let pattern: Vec<u8> = std::iter::repeat_n(bf16(3.0).to_le_bytes(), N_ROWS * OUT_HIDDEN as usize)
        .flatten()
        .collect();
    let mut hidden_rows = alloc.alloc(pattern.len()).expect("hidden");
    hidden_rows.copy_from_host(&pattern, stream.as_ref()).expect("h2d");
    // The deepstack scratch `[3, n_rows, out_hidden]`, pattern-filled:
    // the scatter memsets it whole and adds the (zero) merger outputs,
    // so EVERY byte must come back zero — the proof the deepstack leg
    // ran, not just the main merger.
    let ds_pattern: Vec<u8> =
        std::iter::repeat_n(bf16(5.0).to_le_bytes(), 3 * N_ROWS * OUT_HIDDEN as usize)
            .flatten()
            .collect();
    let mut ds_scratch = alloc.alloc(ds_pattern.len()).expect("ds");
    ds_scratch.copy_from_host(&ds_pattern, stream.as_ref()).expect("h2d");
    stream.as_ref().synchronize().expect("sync");

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    unsafe {
        ffi::pie_k_vision_qwen3vl_scatter(
            z,
            core::ptr::null(),
            pos_tbl.as_ptr().cast_const(),
            block_w.as_ptr(),
            1,
            merger_w.as_ptr(),
            deepstack_w.as_ptr(),
            deepstack_layers.as_ptr(),
            HIDDEN,
            HEADS,
            INTER,
            PATCH,
            T_PATCH,
            MERGE,
            IN_CH,
            OUT_HIDDEN,
            NUM_POS,
            1e-6,
            1e4,
            pixels.as_ptr(),
            pixel_indptr.as_ptr(),
            grids.as_ptr(),
            anchors.as_ptr(),
            1,
            hidden_rows.as_ptr(),
            N_ROWS as i32,
            ds_scratch.as_ptr(),
            3,
            cublas.handle().expect("created").cast(),
            raw_stream,
        );
    }
    stream.as_ref().synchronize().expect("the tower retires");

    let mut ds_back = vec![0u8; ds_pattern.len()];
    ds_scratch.copy_to_host(&mut ds_back, stream.as_ref()).expect("d2h ds");
    stream.as_ref().synchronize().expect("sync");
    assert!(
        ds_back.iter().all(|&b| b == 0),
        "the deepstack scratch must be fully zeroed (memset + zero mergers)"
    );

    let mut back = vec![0u8; pattern.len()];
    hidden_rows.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let at = |r: usize, c: usize| {
        let off = (r * OUT_HIDDEN as usize + c) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for c in [0usize, 1, 47, 95] {
        assert_eq!(at(0, c), 0, "anchor row col {c}: the zero tower must land zero");
        assert_eq!(at(1, c), bf16(3.0), "guard row col {c} must stay untouched");
    }

    cublas.release(&mut cublas_ops);
}

/// gemma-4's STANDALONE vision tower through its encode row — host
/// pixels in, HOST bf16 embedding rows out, the `pie_cuda_encode` shape.
/// A tiny synthetic tower (depth 1, 2×2 patches pooled 2×2 → ONE soft
/// token) with all-zero weights: the encode must write exactly one ZERO
/// row into a pattern-filled host buffer, fill the anchor CSR, and leave
/// the tail untouched.
#[test]
fn the_gemma4_vision_tower_encodes_through_the_bridge() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("gemma4 vision encode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    // The REAL tower geometry — the C++ walk refuses anything else
    // (`hidden=768, heads=12` is a hard check; patch_dim is 3*16*16).
    const HIDDEN: i32 = 768;
    const HEADS: i32 = 12;
    const INTER: i32 = 3072;
    const POS_TABLE: i32 = 16;
    const TEXT_HIDDEN: i32 = 32;
    const POOL: i32 = 3;
    const PATCH: usize = 16;
    const N_PATCH: usize = 9; // 3×3 grid, pooled 3×3 → one soft token
    const OUT_LEN: usize = 1;

    let mut zeros = alloc.alloc(4 * 3072 * 768 * 2).expect("zeros");
    zeros.memset(0, stream.as_ref()).expect("z");
    let z = zeros.as_ptr().cast_const();
    let layer_w: Vec<*const std::ffi::c_void> = vec![z; 41];

    let pixel_dim = 3 * PATCH * PATCH;
    let pixels = vec![0.5f32; N_PATCH * pixel_dim];
    let pixel_indptr: [u32; 2] = [0, (N_PATCH * pixel_dim * 4) as u32];
    // (x, y) per patch, the 3×3 grid.
    let patch_positions: [u32; 18] = [
        0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2,
    ];
    let anchors: [u32; 1] = [0];

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    // Host output rows: OUT_LEN real + 2 guard rows, pattern-filled.
    let mut out_rows = vec![bf16(7.0); (OUT_LEN + 2) * TEXT_HIDDEN as usize];
    let mut out_indptr = [u32::MAX; 2];

    unsafe {
        ffi::pie_k_vision_gemma4_vision_encode(
            z,
            z,
            z,
            layer_w.as_ptr(),
            1,
            HIDDEN,
            HEADS,
            INTER,
            POS_TABLE,
            TEXT_HIDDEN,
            POOL,
            1e-6,
            100.0,
            pixels.as_ptr(),
            pixel_indptr.as_ptr(),
            patch_positions.as_ptr(),
            anchors.as_ptr(),
            1,
            out_rows.as_mut_ptr(),
            out_rows.len() * 2,
            out_indptr.as_mut_ptr(),
            raw_stream,
        );
    }
    stream.as_ref().synchronize().expect("the encode retires");

    assert_eq!(out_indptr[0], 0, "the CSR starts at zero");
    assert_eq!(out_indptr[1], OUT_LEN as u32, "one image, one soft token");
    for (c, v) in out_rows.iter().enumerate().take(TEXT_HIDDEN as usize) {
        assert_eq!(*v, 0, "soft-token col {c}: the zero tower lands zero");
    }
    for c in 0..TEXT_HIDDEN as usize {
        assert_eq!(
            out_rows[(OUT_LEN + 1) * TEXT_HIDDEN as usize + c],
            bf16(7.0),
            "guard row col {c} must stay untouched"
        );
    }
}

/// LoRA slice B on device: `LoraFireState::apply`'s three passes, with
/// every delta chosen so host arithmetic knows the answer exactly.
///
/// Four lanes over four token rows, one row each (so no row-stride
/// question arises), applied at LAYER 1 of 2 — and layer 0's adapter
/// slice holds DIFFERENT values in every lane. That is deliberate: the
/// C++ arm records that reading the state param instead of the op's
/// layer tag applied layer 0's slice everywhere, "the bug the first
/// live A/B caught". Here that bug is a wrong number, not a crash.
///
///   lane 0 (row 0, site Q, low-rank, B = 0)      → q row 0 unchanged
///   lane 1 (row 1, site Q, low-rank, A = B = 1)  → q row 1 += R*H = 128
///   lane 2 (row 2, site V, scale l = 2)          → v row 2 doubled
///   lane 3 (row 3, site Q, scale l = 1)          → q row 3 unchanged
///
/// Every untouched row and every column past `d_out` must survive
/// bit-exactly — the row windows and the widths are as much the claim
/// as the arithmetic is.
#[test]
fn the_lora_apply_lands_its_deltas_on_device() {
    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::model::lora::{
        LORA_SITE_Q, LORA_SITE_V, LiveLoraOps, LoraFireState, LoraForm, LoraLaneView,
        LoraOps, LoraStageArena, LoraStageRows, LoraTable,
    };
    use driver_cuda_new::model::sideband_arena::{DeviceMemory, LiveDeviceMemory};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("lora apply") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const LAYERS: i32 = 2;
    const AT_LAYER: i32 = 1;
    const H: i32 = 64;
    const HQ: i32 = 32;
    const HK: i32 = 16;
    const R: i32 = 2;
    // The staging holds a lane's `d_out` to its SITE's projection width
    // ("lora adapter d_out N != q projection width M"), so a Q lane is
    // `HQ` wide and a V lane `HK` — the widths are a checked contract,
    // not a free parameter.
    const DQ: i32 = HQ;
    const DV: i32 = HK;
    const ROWS: usize = 4;

    // The seam pair the staging draws on, as one value.
    struct Ops<'a> {
        mem: LiveDeviceMemory<'a>,
        lora: LiveLoraOps,
    }
    impl DeviceMemory for Ops<'_> {
        fn alloc(&mut self, bytes: usize) -> Option<*mut std::ffi::c_void> {
            self.mem.alloc(bytes)
        }
        fn free(&mut self, ptr: *mut std::ffi::c_void) {
            self.mem.free(ptr);
        }
        fn synchronize(&mut self) -> bool {
            self.mem.synchronize()
        }
    }
    impl LoraOps for Ops<'_> {
        fn cast_fp32_to_bf16(
            &mut self,
            src: *const std::ffi::c_void,
            dst: *mut std::ffi::c_void,
            elems: usize,
        ) {
            self.lora.cast_fp32_to_bf16(src, dst, elems);
        }
        fn upload_slab(&mut self, dst: *mut std::ffi::c_void, slots: &[*const std::ffi::c_void]) {
            self.lora.upload_slab(dst, slots);
        }
    }

    let up_f32 = |v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut b = alloc.alloc(bytes.len()).expect("f32 up");
        b.copy_from_host(&bytes, stream.as_ref()).expect("h2d");
        b
    };
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let up_bf16 = |v: f32, n: usize| {
        let bytes: Vec<u8> = std::iter::repeat_n(bf16(v).to_le_bytes(), n).flatten().collect();
        let mut b = alloc.alloc(bytes.len()).expect("bf16 up");
        b.copy_from_host(&bytes, stream.as_ref()).expect("h2d");
        b
    };

    // ── Adapter cells, f32 (staging casts them). Layer 0 differs from
    // layer 1 in EVERY lane, so a layer mix-up is a wrong number. ──
    let a_len = (LAYERS * R * H) as usize;
    let b_len = (LAYERS * DQ * R) as usize;
    let l0 = (R * H) as usize;
    let b0 = (DQ * R) as usize;
    // lane 0: A ones (both layers), B zero (both layers).
    let a_zero_b = up_f32(&vec![1.0f32; a_len]);
    let b_zero = up_f32(&vec![0.0f32; b_len]);
    // lane 1: layer 0 zeros, layer 1 ones — delta only at layer 1.
    let mut a_one_h = vec![0.0f32; a_len];
    a_one_h[l0..].fill(1.0);
    let mut b_one_h = vec![0.0f32; b_len];
    b_one_h[b0..].fill(1.0);
    let a_one = up_f32(&a_one_h);
    let b_one = up_f32(&b_one_h);
    // lane 2 (scale, site V): layer 0 = 4, layer 1 = 2.
    let mut s2 = vec![4.0f32; (LAYERS * DV) as usize];
    s2[DV as usize..].fill(2.0);
    let scale_two = up_f32(&s2);
    // lane 3 (scale, site Q): layer 0 = 8, layer 1 = 1 — identity here.
    let mut s3 = vec![8.0f32; (LAYERS * DQ) as usize];
    s3[DQ as usize..].fill(1.0);
    let scale_one = up_f32(&s3);

    let low_rank = |a: *const std::ffi::c_void,
                    b: *const std::ffi::c_void,
                    row: u32,
                    site: u64,
                    d_out: i32| {
        LoraLaneView {
            a,
            b,
            sites_bits: site,
            token_start: row,
            token_count: 1,
            num_layers: LAYERS as u32,
            rank: R as u32,
            d_in: H as u32,
            d_out: d_out as u32,
            form: LoraForm::LowRank,
        }
    };
    let scale = |a: *const std::ffi::c_void, row: u32, site: u64, d_out: i32| LoraLaneView {
        a,
        b: std::ptr::null(),
        sites_bits: site,
        token_start: row,
        token_count: 1,
        num_layers: LAYERS as u32,
        rank: 0,
        d_in: 0,
        d_out: d_out as u32,
        form: LoraForm::Scale,
    };
    let lanes = [
        low_rank(a_zero_b.as_ptr().cast_const(), b_zero.as_ptr().cast_const(), 0, LORA_SITE_Q, DQ),
        low_rank(a_one.as_ptr().cast_const(), b_one.as_ptr().cast_const(), 1, LORA_SITE_Q, DQ),
        scale(scale_two.as_ptr().cast_const(), 2, LORA_SITE_V, DV),
        scale(scale_one.as_ptr().cast_const(), 3, LORA_SITE_Q, DQ),
    ];
    let table = LoraTable { lanes: &lanes };

    // ── The fire's buffers: x all ones, q all ones, v all threes. ──
    let x = up_bf16(1.0, ROWS * H as usize);
    let q = up_bf16(1.0, ROWS * HQ as usize);
    let v = up_bf16(3.0, ROWS * HK as usize);
    let scratch = up_bf16(0.0, ROWS * 128);

    let mut ops = Ops {
        mem: LiveDeviceMemory::new(stream.as_ref()),
        lora: LiveLoraOps::new(raw_stream),
    };
    let mut arena = LoraStageArena::default();
    let rows_view = LoraStageRows {
        y: x.as_ptr().cast_const(),
        norm_x: x.as_ptr().cast_const(),
        q: q.as_ptr(),
        v: v.as_ptr(),
        gate: scratch.as_ptr(),
    };
    let state = LoraFireState::stage(
        &mut ops,
        &mut arena,
        &table,
        LAYERS,
        ROWS as i32,
        H,
        HQ,
        HK,
        128,
        1,
        &rows_view,
        /*grouped_enabled=*/ false,
    )
    .expect("the lane table stages");
    stream.as_ref().synchronize().expect("staging retires");

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    state.apply(
        cublas.handle().expect("created").cast(),
        AT_LAYER,
        x.as_ptr().cast_const(),
        H,
        HQ,
        HK,
        q.as_ptr(),
        v.as_ptr(),
        scratch.as_ptr(),
        raw_stream,
    );
    stream.as_ref().synchronize().expect("the corrections retire");

    let read = |buf: &driver_cuda_new::cuda::DeviceBuffer| {
        let mut back = vec![0u8; buf.len()];
        buf.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
        stream.as_ref().synchronize().expect("sync");
        back
    };
    let q_back = read(&q);
    let v_back = read(&v);
    let at = |back: &[u8], row: usize, width: i32, col: usize| {
        let off = (row * width as usize + col) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };

    // Row 0: B is zero at every layer — the launches ran and added nothing.
    for c in [0usize, 3, 31] {
        assert_eq!(at(&q_back, 0, HQ, c), bf16(1.0), "row 0 col {c}: zero B must not move q");
    }
    // Row 1: xAᵀ = H per rank slot, (xAᵀ)Bᵀ = R*H = 128, folded β=1 onto 1.0.
    // Layer 0's slice is ZERO — 129 here is also the layer-tag proof.
    for c in [0usize, 3, 31] {
        assert_eq!(
            at(&q_back, 1, HQ, c),
            bf16(129.0),
            "row 1 col {c}: the low-rank delta is R*H folded onto the base"
        );
    }
    // The delta spans the whole projection row, and NOTHING past it:
    // the next row's first column is a different lane's business.
    // Row 2 is a V-site lane: q untouched, v doubled (layer 0 would be ×4).
    assert_eq!(at(&q_back, 2, HQ, 0), bf16(1.0), "row 2 is a v-site lane");
    for c in [0usize, 3, 15] {
        assert_eq!(at(&v_back, 2, HK, c), bf16(6.0), "row 2 col {c}: scale ×2");
    }
    // Row 3: a scale of one at layer 1 (layer 0 would be ×8).
    for c in [0usize, 31] {
        assert_eq!(at(&q_back, 3, HQ, c), bf16(1.0), "row 3 col {c}: identity scale");
    }
    // Every row no lane names stays put.
    for r in [0usize, 1, 3] {
        assert_eq!(at(&v_back, r, HK, 0), bf16(3.0), "v row {r} has no lane");
    }

    cublas.release(&mut cublas_ops);
    arena.release(&mut ops);
}

/// The GROUPED lora path — a different lowering of the same math, and a
/// different code path end to end: same-shape lanes fold into one group,
/// their pointer arrays are staged into the device-resident slab, and
/// `apply` does slot arithmetic plus three grouped GEMMs instead of a
/// pair per lane. Two lanes of one shape on disjoint rows, distinguished
/// by their B contents so a swapped slot is a wrong number: row 0 takes
/// B = 1 (delta R*H = 128), row 1 takes B = 0.5 (delta 64). Layer 0's
/// slice is zero in both, so the layer-tag claim holds here too.
#[test]
fn the_lora_grouped_path_lands_its_deltas_on_device() {
    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::model::lora::{
        LORA_SITE_Q, LiveLoraOps, LoraFireState, LoraForm, LoraLaneView, LoraOps,
        LoraStageArena, LoraStageRows, LoraTable,
    };
    use driver_cuda_new::model::sideband_arena::{DeviceMemory, LiveDeviceMemory};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("lora grouped apply") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const LAYERS: i32 = 2;
    const AT_LAYER: i32 = 1;
    const H: i32 = 64;
    const HQ: i32 = 32;
    const HK: i32 = 16;
    const R: i32 = 2;
    const ROWS: usize = 4;

    struct Ops<'a> {
        mem: LiveDeviceMemory<'a>,
        lora: LiveLoraOps,
    }
    impl DeviceMemory for Ops<'_> {
        fn alloc(&mut self, bytes: usize) -> Option<*mut std::ffi::c_void> {
            self.mem.alloc(bytes)
        }
        fn free(&mut self, ptr: *mut std::ffi::c_void) {
            self.mem.free(ptr);
        }
        fn synchronize(&mut self) -> bool {
            self.mem.synchronize()
        }
    }
    impl LoraOps for Ops<'_> {
        fn cast_fp32_to_bf16(
            &mut self,
            src: *const std::ffi::c_void,
            dst: *mut std::ffi::c_void,
            elems: usize,
        ) {
            self.lora.cast_fp32_to_bf16(src, dst, elems);
        }
        fn upload_slab(&mut self, dst: *mut std::ffi::c_void, slots: &[*const std::ffi::c_void]) {
            self.lora.upload_slab(dst, slots);
        }
    }

    let up_f32 = |v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut b = alloc.alloc(bytes.len()).expect("f32 up");
        b.copy_from_host(&bytes, stream.as_ref()).expect("h2d");
        b
    };
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let up_bf16 = |v: f32, n: usize| {
        let bytes: Vec<u8> = std::iter::repeat_n(bf16(v).to_le_bytes(), n).flatten().collect();
        let mut b = alloc.alloc(bytes.len()).expect("bf16 up");
        b.copy_from_host(&bytes, stream.as_ref()).expect("h2d");
        b
    };

    let a_len = (LAYERS * R * H) as usize;
    let b_len = (LAYERS * HQ * R) as usize;
    let mut a_h = vec![0.0f32; a_len];
    a_h[(R * H) as usize..].fill(1.0);
    let a_dev = up_f32(&a_h);
    let mut b1_h = vec![0.0f32; b_len];
    b1_h[(HQ * R) as usize..].fill(1.0);
    let b_one = up_f32(&b1_h);
    let mut b2_h = vec![0.0f32; b_len];
    b2_h[(HQ * R) as usize..].fill(0.5);
    let b_half = up_f32(&b2_h);

    let lane = |b: *const std::ffi::c_void, row: u32| LoraLaneView {
        a: a_dev.as_ptr().cast_const(),
        b,
        sites_bits: LORA_SITE_Q,
        token_start: row,
        token_count: 1,
        num_layers: LAYERS as u32,
        rank: R as u32,
        d_in: H as u32,
        d_out: HQ as u32,
        form: LoraForm::LowRank,
    };
    let lanes = [lane(b_one.as_ptr().cast_const(), 0), lane(b_half.as_ptr().cast_const(), 1)];
    let table = LoraTable { lanes: &lanes };

    let x = up_bf16(1.0, ROWS * H as usize);
    let q = up_bf16(1.0, ROWS * HQ as usize);
    let v = up_bf16(3.0, ROWS * HK as usize);
    let scratch = up_bf16(0.0, ROWS * 128);

    let mut ops = Ops {
        mem: LiveDeviceMemory::new(stream.as_ref()),
        lora: LiveLoraOps::new(raw_stream),
    };
    let mut arena = LoraStageArena::default();
    let rows_view = LoraStageRows {
        y: x.as_ptr().cast_const(),
        norm_x: x.as_ptr().cast_const(),
        q: q.as_ptr(),
        v: v.as_ptr(),
        gate: scratch.as_ptr(),
    };
    let state = LoraFireState::stage(
        &mut ops, &mut arena, &table, LAYERS, ROWS as i32, H, HQ, HK, 128, 1, &rows_view,
        /*grouped_enabled=*/ true,
    )
    .expect("the lane table stages");
    stream.as_ref().synchronize().expect("staging retires");
    // `grouping_desc` spells a group as `<members>xr<rank>`.
    assert!(
        state.grouping_desc().contains("2xr2"),
        "the two same-shape lanes must have GROUPED, got {}",
        state.grouping_desc()
    );

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    state.apply(
        cublas.handle().expect("created").cast(),
        AT_LAYER,
        x.as_ptr().cast_const(),
        H,
        HQ,
        HK,
        q.as_ptr(),
        v.as_ptr(),
        scratch.as_ptr(),
        raw_stream,
    );
    stream.as_ref().synchronize().expect("the grouped corrections retire");

    let mut back = vec![0u8; q.len()];
    q.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let at = |row: usize, col: usize| {
        let off = (row * HQ as usize + col) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for c in [0usize, 5, 31] {
        assert_eq!(at(0, c), bf16(129.0), "grouped row 0 col {c}: B = 1 gives R*H");
        assert_eq!(at(1, c), bf16(65.0), "grouped row 1 col {c}: B = 0.5 gives R*H/2");
    }
    for r in [2usize, 3] {
        assert_eq!(at(r, 0), bf16(1.0), "row {r} is in no group");
    }

    cublas.release(&mut cublas_ops);
    arena.release(&mut ops);
}

/// gemma3n's DECODE walked live: the AltUp rank-K residual end to end —
/// expand one stream into K, predict the others from the active one, run
/// the layer body on the prediction, correct all K from the result, and
/// close with the per-stream projections, the magnitude rescale and the
/// mean. A small synthetic geometry (4 streams, 4 layers, two of them
/// sparse so `gaussian_topk` fires) with zero weights and ones on the
/// norms.
///
/// The claim is not an algebraic identity — AltUp's coefficient
/// projections are zero here, so the arithmetic is the family's own, not
/// something host math re-derives. It is that all TEN new arms fire on
/// device with the shapes the lowering states, no launch faults (the walk
/// syncs after each one), and nothing leaves a NaN or an infinity behind:
/// the failure mode these arms actually have is a wrong dimension read
/// from the wrong operand, which is an illegal address or a poisoned
/// buffer, not a subtly wrong number.
#[test]
fn the_gemma3n_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, Resolver, bind, dispatch,
    };
    use model::gemma3n::forward::facts::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts};
    use model::gemma3n::forward::gemma3n_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("gemma3n zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 256;
    const LAYERS: usize = 4;
    const VOCAB: usize = 256;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 4;
    const HEAD_DIM: i32 = 64;
    const PAGE: i32 = 16;
    const ROWS: usize = 2;
    const STREAMS: i32 = 4;

    let facts = Gemma3nFacts {
        vocab: VOCAB as u32,
        hidden: HIDDEN as u32,
        per_layer_intermediate: vec![512; LAYERS],
        laurel_rank: 32,
        ple_width: 64,
        sparsity_layers: 2,
        altup: Gemma3nAltUpFacts { num_streams: STREAMS as u32, active: 0 },
        attn: Gemma3nAttnFacts {
            heads: Q_HEADS as u32,
            kv_heads: KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
        },
        // Empty reads as "no window" — what this fixture meant before the
        // field existed, and what a zero-weight walk needs either way.
        window_left: Vec::new(),
    };
    let plan = gemma3n_cuda(&facts, FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let tokens: [i32; ROWS] = [1, 2];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..VOCAB {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { 0.5 } else { -0.5 };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), 4096).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    // Every projection is zero; the widest is [512, 256].
    let mut zeros_dev = alloc.alloc(4 << 20).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * (w.max(1) as usize) * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") || name.contains("lm_head") {
                self.embed
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 4,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("csr");
        b.copy_from_host(data, stream.as_ref()).expect("h2d csr");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&[0, 1, 2]));
    let csr_lens = up(&u32s(&[1, 1]));
    let qo_indptr = up(&u32s(&[0, 1, 2]));
    let w_page = up(&u32s(&[0, 1]));
    let w_off = up(&u32s(&[0, 0]));
    let row_valid = up(&[1u8, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions = up(&[0i32, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(
        &[0, 1, 2], Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        q_out: core::ptr::null_mut(),
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        o_out: core::ptr::null_mut(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: ROWS as i32,
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e4,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 30.0,
        ple_dim: 0,
        // The `scale.<name>` constants the trace rides in weight slots,
        // scanned from the lowering itself so a renamed statement shows
        // up as a refusal rather than as a silent default. The AltUp
        // router's is `1/H` (the C++'s own comment: "tanh(modality_router
        // (router_norm(active_in) / H))"); the rest only have to be
        // finite for this smoke's claim, and the shell will read the
        // deployment's when gemma3n gets one.
        scales: {
            let mut m = std::collections::BTreeMap::new();
            for a in &l.args {
                if let Arg::Weight(w) = a
                    && let Some(name) = w.strip_prefix("scale.")
                {
                    let v = if name.ends_with("laurel_scale") {
                        1.0 / (facts.laurel_rank as f32).sqrt()
                    } else {
                        1.0 / HIDDEN as f32
                    };
                    m.insert(name.to_string(), v);
                }
            }
            assert!(!m.is_empty(), "gemma3n states scale constants");
            m
        },
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: STREAMS,
        altup_active: 0,
        // `gaussian_inverse_cdf(0.95)`, the C++'s host derivation — the
        // value only has to be a real threshold for the launch to mean
        // something; the sparse layers are the first two.
        altup_std_mult_by_layer: vec![1.6449, 1.6449, 0.0, 0.0],
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    // Launch by launch, syncing after each: a wrong dimension read from
    // the wrong operand is an illegal address, and this is what names it.
    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &named_bufs,
    };
    for (i, launch) in l.launches.iter().enumerate() {
        let kernel = l.kernels[launch.kernel as usize].clone();
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|e| panic!("launch {i} {kernel}: bind {e:?}"));
        dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, Some(&attn), None)
            .unwrap_or_else(|e| panic!("launch {i} {kernel}: dispatch {e:?}"));
        stream
            .as_ref()
            .synchronize()
            .unwrap_or_else(|e| panic!("launch {i} {kernel} poisoned the stream: {e:?}"));
    }

    // Nothing left a NaN or an infinity in the arena.
    let mut back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let mut bad = 0usize;
    for c in back.chunks_exact(2) {
        let f = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
        if f.is_nan() || f.is_infinite() {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "the AltUp walk left {bad} non-finite bf16 words in the arena");

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// gpt-oss's DECODE walked live: the MXFP4 expert banks, the clamped
/// GLU, the attention sink rescale and YaRN-over-the-original-context,
/// all on a small synthetic geometry with zero weights.
///
/// The MXFP4 arms are the reason this smoke exists in this shape. Their
/// expert banks are POINTER TABLES, not tensors — a device array of one
/// pointer per expert, which the C++ keeps as four fields on its layer
/// struct and this driver reaches by name suffix. The resolver below
/// hands back a real such table, so the launch reads pointers where it
/// expects pointers; a plain buffer there is an illegal address, which
/// is exactly the failure this walk is here to rule out. Per-launch
/// syncs name the offender, and the closing check is that nothing left a
/// NaN or an infinity behind.
#[test]
fn the_gpt_oss_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, Resolver, bind, dispatch,
    };
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    use model::gpt_oss::forward::gpt_oss_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("gpt_oss zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 256;
    const LAYERS: usize = 2;
    const VOCAB: usize = 256;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 4;
    const HEAD_DIM: i32 = 64;
    const PAGE: i32 = 16;
    const ROWS: usize = 2;
    const EXPERTS: usize = 8;

    // The 20b's shape at a size a smoke can hold; every fact that steers
    // a STATEMENT (sinks, yarn, bias, the GLU limit) keeps its real value.
    let facts = GptOssFacts {
        hidden: HIDDEN as u32,
        layers: LAYERS as u32,
        q_heads: Q_HEADS as u32,
        kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate: 256,
        experts: EXPERTS as u32,
        top_k: 2,
        vocab: VOCAB as u32,
        tied_embeddings: false,
        swiglu_limit: 7.0,
        attention_bias: true,
        rope_yarn_original: true,
        attn_sinks: true,
    };
    let plan = gpt_oss_cuda(&facts, &GptOssCudaFacts::gpt_oss_20b_synthetic(), FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let tokens: [i32; ROWS] = [1, 2];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..VOCAB {
        for c in 0..HIDDEN {
            let b = bf16(if c % 2 == 0 { 0.5 } else { -0.5 }).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), 4096).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let mut zeros_dev = alloc.alloc(8 << 20).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");
    // The per-expert POINTER TABLE: E pointers, every one into the zero
    // bank. This is the shape the MXFP4 launches dereference.
    let table_host: Vec<u8> = std::iter::repeat_n(zeros_dev.as_ptr() as u64, EXPERTS)
        .flat_map(u64::to_le_bytes)
        .collect();
    let mut bank_table = alloc.alloc(table_host.len()).expect("bank table");
    bank_table.copy_from_host(&table_host, stream.as_ref()).expect("h2d table");
    stream.as_ref().synchronize().expect("uploads retire");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * (w.max(1) as usize) * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        table: *const std::ffi::c_void,
        named: &'a BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            // The expert banks and their scales are pointer TABLES; the
            // per-expert biases are genuinely absent on this deployment,
            // and the row says they may be null.
            if name.contains("bank") {
                return if name.ends_with("_bias") { None } else { Some(self.table) };
            }
            Some(if name.contains("embed") || name.contains("lm_head") {
                self.embed
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 4,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("csr");
        b.copy_from_host(data, stream.as_ref()).expect("h2d csr");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&[0, 1, 2]));
    let csr_lens = up(&u32s(&[1, 1]));
    let qo_indptr = up(&u32s(&[0, 1, 2]));
    let w_page = up(&u32s(&[0, 1]));
    let w_off = up(&u32s(&[0, 0]));
    let row_valid = up(&[1u8, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions = up(&[0i32, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(
        &[0, 1, 2], Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        q_out: core::ptr::null_mut(),
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        o_out: core::ptr::null_mut(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: ROWS as i32,
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-5,
        rope_theta: 150_000.0,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: HEAD_DIM,
        num_q_heads: Q_HEADS,
        num_kv_heads: KV_HEADS,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        // The 20b's YaRN set, in the launcher's order.
        yarn: [32.0, 32.0, 1.0, 1.0],
        yarn_original_max: 4096,
        glu_limit: facts.swiglu_limit,
        glu_alpha: 1.702,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
    };

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        table: bank_table.as_ptr(),
        named: &named_bufs,
    };
    for (i, launch) in l.launches.iter().enumerate() {
        let kernel = l.kernels[launch.kernel as usize].clone();
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|e| panic!("launch {i} {kernel}: bind {e:?}"));
        dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, Some(&attn), None)
            .unwrap_or_else(|e| panic!("launch {i} {kernel}: dispatch {e:?}"));
        stream
            .as_ref()
            .synchronize()
            .unwrap_or_else(|e| panic!("launch {i} {kernel} poisoned the stream: {e:?}"));
    }

    let mut back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let bad = back
        .chunks_exact(2)
        .filter(|c| {
            let f = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
            f.is_nan() || f.is_infinite()
        })
        .count();
    assert_eq!(bad, 0, "the gpt-oss walk left {bad} non-finite bf16 words in the arena");

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}
