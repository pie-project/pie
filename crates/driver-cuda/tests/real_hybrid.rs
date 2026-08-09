//! The qwen3_5 HYBRID's real-weight A/B (E-gate family #1's parity
//! gate): `Qwen/Qwen3.5-0.8B-Base` from the HF cache, prefilled through
//! `run()` — 18 GDN layers advancing real conv/recurrent state, 6
//! full-attention layers through flashinfer — against a committed
//! transformers reference (`tests/oracle/real_decode/qwen3_5_0_8b.json`).
//!
//! ONE deployment for the family, per the campaign's constraint: the
//! parity anchor, not a per-model loop.
//!
//! The checkpoint's own facts, verified against `config.json`
//! (`text_config`): 24 layers at `full_attention_interval` 4, GDN
//! 16/16 heads at 128/128, conv window 4 (no bias), full attention
//! 8 q / 2 kv heads at 256 with `rotary_dim` 64 and `rope_theta` 1e7,
//! dense MLP 3584, vocab 248320, tied embeddings. `A_log` and the GDN
//! gate norm ship F32 (the `gdn_fp32_parameters` contract's fp32 side —
//! uploaded verbatim, no widening needed on this checkpoint); everything
//! else is bf16.
//!
//! # Why this family's bar is looser than llama_like's
//!
//! The llama_like A/Bs held exact argmax and ±0.25 on top-5. This
//! family cannot, and the reasons are measured, not assumed
//! (2026-08-09, all on this checkpoint and prompt):
//!
//! * transformers bf16 ≈ transformers fp32 here (top logits within
//!   0.03) — HF computes effectively in fp32 with bf16 weights, so HF
//!   is a clean reference, and the gap below is OURS;
//! * our gap is the C++ KERNEL PIPELINE's: real bf16 arenas between
//!   every launch and 18 GDN layers of L2-norm/exp/softplus
//!   nonlinearity. Depth bisection against HF `hidden_states` (layers
//!   4/8/12/16/20/21/22/23) tracks within ~5% residual norm at every
//!   depth — accumulation, no structural break;
//! * the C++ driver's OWN parity harness
//!   (`parity_qwen3_5_multireq.py`) explicitly refuses argmax equality
//!   for this family: "with bf16 + flashinfer's R-dependent prefill
//!   tiling, the very first decoded token can legitimately flip".
//!
//! So the criteria here: our argmax is one of HF's top-5, every HF
//! top-5 id sits in our top-8, top-5 logits within 1.25, probes within
//! 0.6. A structural bug (a swapped binding, a wrong state slab) blows
//! all four; bf16 accumulation passes them.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use driver_cuda::bind::abi::{KvCacheLayerView, KvCacheScheme};
use driver_cuda::bind::{
    AttnCtx, AttnRegions, DispatchCtx, DispatchPlan, Frame, GdnCtx, PrefillPlan, Resolver, run,
};
use driver_cuda::device::{Allocator, DeviceBuffer, OwnedStream};
use driver_cuda::dtype::DType;
use driver_cuda::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};
use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

mod common;
use common::{device_or_skip, gpu_guard};

/// The checkpoint, header-parsed — `real_prefill`'s reader with the one
/// difference this family needs: F32 tensors are admitted (A_log, the
/// GDN gate norm), because their kernels consume fp32 directly.
struct Checkpoint {
    raws: Vec<Vec<u8>>,
    index: BTreeMap<String, (usize, usize, usize)>,
}

impl Checkpoint {
    fn open(cache_dir: &str) -> Option<Self> {
        let home = std::env::var_os("HOME")?;
        let snaps =
            PathBuf::from(home).join(format!(".cache/huggingface/hub/{cache_dir}/snapshots"));
        let snap = std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(Result::ok)
            .find_map(|e| {
                let d = e.path();
                (d.join("model.safetensors").is_file()
                    || d.join("model.safetensors.index.json").is_file())
                .then_some(d)
            })?;
        let files: Vec<PathBuf> = if snap.join("model.safetensors").is_file() {
            vec![snap.join("model.safetensors")]
        } else {
            let idx: serde_json::Value = serde_json::from_slice(
                &std::fs::read(snap.join("model.safetensors.index.json")).ok()?,
            )
            .ok()?;
            let mut shards: Vec<String> = idx["weight_map"]
                .as_object()?
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            shards.into_iter().map(|f| snap.join(f)).collect()
        };
        let mut raws = Vec::new();
        let mut index = BTreeMap::new();
        for (fi, f) in files.iter().enumerate() {
            let raw = std::fs::read(f).ok()?;
            let header_len = u64::from_le_bytes(raw[..8].try_into().ok()?) as usize;
            let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + header_len]).ok()?;
            let payload = 8 + header_len;
            for (name, meta) in header.as_object()? {
                if name == "__metadata__" {
                    continue;
                }
                assert!(
                    meta["dtype"] == "BF16" || meta["dtype"] == "F32",
                    "{name}: this loader speaks bf16 and fp32 only, got {}",
                    meta["dtype"]
                );
                let offs = meta["data_offsets"].as_array()?;
                index.insert(
                    name.clone(),
                    (
                        fi,
                        payload + offs[0].as_u64()? as usize,
                        payload + offs[1].as_u64()? as usize,
                    ),
                );
            }
            raws.push(raw);
        }
        Some(Self { raws, index })
    }

    fn bytes(&self, name: &str) -> &[u8] {
        let (f, b, e) = *self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("checkpoint lacks {name}"));
        &self.raws[f][b..e]
    }
}

/// Bind every trace name the hybrid states to the checkpoint's tensors
/// — the qwen3_5 binder, `bind_qwen3_5_weight`'s vocabulary as data.
fn bind(ckpt: &Checkpoint, sink: &mut dyn FnMut(String, Vec<u8>)) {
    let p = "model.language_model";
    sink(
        "embed".into(),
        ckpt.bytes(&format!("{p}.embed_tokens.weight")).to_vec(),
    );
    sink(
        "final_norm".into(),
        ckpt.bytes(&format!("{p}.norm.weight")).to_vec(),
    );
    for n in 0..24u32 {
        let lp = format!("{p}.layers.{n}");
        let mut w =
            |trace: &str, hf: String| sink(format!("layer.{n}.{trace}"), ckpt.bytes(&hf).to_vec());
        w("attn_norm", format!("{lp}.input_layernorm.weight"));
        w("mlp_norm", format!("{lp}.post_attention_layernorm.weight"));
        w("down", format!("{lp}.mlp.down_proj.weight"));
        if n % 4 == 3 {
            for f in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                w(f, format!("{lp}.self_attn.{f}.weight"));
            }
        } else {
            for f in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                w(f, format!("{lp}.linear_attn.{f}.weight"));
            }
            w("conv", format!("{lp}.linear_attn.conv1d.weight"));
            w("a_log", format!("{lp}.linear_attn.A_log"));
            w("dt_bias", format!("{lp}.linear_attn.dt_bias"));
            w("gate_norm", format!("{lp}.linear_attn.norm.weight"));
            w("o_proj", format!("{lp}.linear_attn.out_proj.weight"));
        }
        // The fused gate‖up bank (`gate_up_fused: true`, gate first).
        let mut fused = ckpt.bytes(&format!("{lp}.mlp.gate_proj.weight")).to_vec();
        fused.extend_from_slice(ckpt.bytes(&format!("{lp}.mlp.up_proj.weight")));
        sink(format!("layer.{n}.gate_up"), fused);
    }
}

struct Live<'a> {
    weights: &'a BTreeMap<String, DeviceBuffer>,
    named: &'a BTreeMap<ValueId, DeviceBuffer>,
}
impl Resolver for Live<'_> {
    fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
        self.weights.get(name).map(|b| b.as_ptr().cast_const())
    }
    fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
        self.named.get(&value).map(|b| b.as_ptr())
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn the_hybrid_matches_transformers_on_real_weights() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("qwen3.5 hybrid A/B") else {
        return;
    };
    let Some(ckpt) = Checkpoint::open("models--Qwen--Qwen3.5-0.8B-Base") else {
        eprintln!("skipped: Qwen3.5-0.8B-Base not in the HF cache");
        return;
    };
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle/real_decode/qwen3_5_0_8b.json"),
        )
        .expect("reference file"),
    )
    .expect("reference json");

    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const TOKENS: usize = 7;
    const VOCAB: usize = 248_320;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const PAGE: i32 = 16;
    const K_H: i32 = 16;
    const V_H: i32 = 16;
    const K_D: i32 = 128;
    const V_D: i32 = 128;
    const CONV_DIM: i32 = 6144;
    const CONV_K: i32 = 4;

    let mut hybrid = Qwen35HybridFacts::qwen3_5_0_8b();
    // Bisection aid: truncate the depth to compare the residual against
    // transformers' per-layer hidden states.
    if let Ok(n) = std::env::var("HYBRID_AB_LAYERS") {
        hybrid.layers = n.parse().expect("layer count");
    }
    // The LIVE L40S cuda set (`emissions.rs`).
    let cuda = Qwen35CudaFacts {
        // fp32 state for the A/B: transformers runs the recurrence with
        // fp32 state (`mamba_ssm_dtype: float32`), and the bf16-state
        // deployment fact is a precision TRADE whose drift the smoke
        // measured at ~1.0 on top logits. The kernels under test are the
        // same C++ either way; the A/B pins the binding, not the trade.
        state_bf16: false,
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
    // A prefill: one request, TOKENS rows -- every row multi-token, which
    // is what `GuardPred::WindowOne` reads (graph.md §4.1).
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            multi_token: true,
            ..Row::default()
        };
        TOKENS
    ];
    let l = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame {
        arena: arena.as_ptr(),
        arena_bytes: l.arena_bytes,
    };

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };

    // ── The real weights, uploaded under their trace names. ──
    let mut weights: BTreeMap<String, DeviceBuffer> = BTreeMap::new();
    bind(&ckpt, &mut |name, host| {
        weights.insert(name, up(&host));
    });

    // ── The seam-value pool. ──
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
    let named_bufs: BTreeMap<ValueId, DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(TOKENS * w as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    // ── KV pools (one page holds the whole prompt) + GDN slabs. ──
    let plane = (PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<Option<(DeviceBuffer, DeviceBuffer)>> = (0..24u32)
        .map(|i| {
            if !hybrid.is_full_attn(i) {
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
            let (k, v) = kv
                .as_ref()
                .map_or((core::ptr::null_mut(), core::ptr::null_mut()), |(k, v)| {
                    (k.as_ptr(), v.as_ptr())
                });
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(i).expect("layer"),
                num_pages: 1,
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
    let gdn_slabs: Vec<Option<(DeviceBuffer, DeviceBuffer)>> = (0..24u32)
        .map(|i| {
            if hybrid.is_full_attn(i) {
                return None;
            }
            let mut c = alloc.alloc(conv_stride * 2).expect("conv slab");
            let mut s = alloc.alloc(state_stride * 4).expect("state slab");
            c.memset(0, stream.as_ref()).expect("zc");
            s.memset(0, stream.as_ref()).expect("zs");
            Some((c, s))
        })
        .collect();
    let slot_ids = up(&0i32.to_le_bytes());
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

    // ── One request over the whole prompt. ──
    let prompt: Vec<i32> = reference["prompt_ids"]
        .as_array()
        .expect("prompt")
        .iter()
        .map(|v| i32::try_from(v.as_i64().expect("id")).expect("id"))
        .collect();
    assert_eq!(prompt.len(), TOKENS);
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 2] = [0, 7];
    let page_indptr_h: [u32; 2] = [0, 1];
    let last_lens_h: [u32; 1] = [7];
    let csr_indices = up(&u32s(&[0]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&prompt
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .collect::<Vec<u8>>());
    let positions = up(&(0i32..7).flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h,
        &page_indptr_h,
        &last_lens_h,
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
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_prefill_bf16"
        })
        .expect("the hybrid prefill dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    let o_out: *mut std::ffi::c_void = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
        Arg::Named { value, .. } => named_bufs[value].as_ptr(),
        other => panic!("the gate reads the attention slot, got {other:?}"),
    };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        prefill_workspace: ws.view(),
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
        num_requests: 1,
        num_pages_in_batch: 1,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        score_window: 32,
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = driver_cuda::device::cublas::LiveCublas;
    let mut cublas = driver_cuda::device::cublas::CublasHandle::create(&mut cublas_ops, raw_stream)
        .expect("cublas");
    let ctx = DispatchCtx {
        // Every row sampled, so no compaction is stated and the gather
        // has no index list to read.
        sampling_indices: core::ptr::null(),
        sampled_rows: 0,
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: 1e-6,
        rope_theta: 1e7,
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
        // Dense fixtures: no routed statement, which this field spells `0`.
        experts_per_token: 0,
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
            moe_ptrs: std::cell::Cell::new(None),
};

    let mut logits_value: Option<ValueId> = None;
    for i in 0..l.launches.len() {
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }

    let mut resolver = Live {
        weights: &weights,
        named: &named_bufs,
    };
    let ran = run(
        &l,
        &dplan,
        frame,
        &mut resolver,
        &ctx,
        AttnRegions::whole(Some(&attn)),
        Some(&gdn),
    )
    .unwrap_or_else(|e| panic!("the hybrid A/B walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len());
    stream.as_ref().synchronize().expect("the fire retires");

    if std::env::var("HYBRID_AB_DEBUG").is_ok() {
        // The residual lives where the embed wrote it; dump the last
        // row's head for the bisection against HF hidden states.
        let e_at = l
            .launches
            .iter()
            .find_map(|x| {
                (l.kernels[x.kernel as usize] == "layout::embed_bf16").then(|| {
                    match &l.args[x.args.start as usize] {
                        Arg::Arena { at, .. } => *at,
                        other => panic!("embed writes the arena, got {other:?}"),
                    }
                })
            })
            .expect("embed ran");
        let mut back = vec![0u8; l.arena_bytes];
        arena.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
        stream.as_ref().synchronize().expect("sync");
        let h = |r: usize, c: usize| {
            let off = e_at + (r * 1024 + c) * 2;
            let bits = u16::from_le_bytes([back[off], back[off + 1]]);
            f32::from_bits(u32::from(bits) << 16)
        };
        let head: Vec<f32> = (0..8).map(|c| h(TOKENS - 1, c)).collect();
        let norm: f32 = (0..1024)
            .map(|c| h(TOKENS - 1, c).powi(2))
            .sum::<f32>()
            .sqrt();
        eprintln!("ours residual last row head: {head:?} norm={norm:.4}");
    }

    // ── The A/B: the last row's logits against transformers'. ──
    let lv = logits_value.expect("the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits
        .copy_to_host(&mut back, stream.as_ref())
        .expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let last = TOKENS - 1;
    let logit = |t: usize| {
        let off = (last * VOCAB + t) * 2;
        let bits = u16::from_le_bytes([back[off], back[off + 1]]);
        f32::from_bits(u32::from(bits) << 16)
    };

    let mut all: Vec<(usize, f32)> = Vec::with_capacity(VOCAB);
    for t in 0..VOCAB {
        all.push((t, logit(t)));
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    if std::env::var("HYBRID_AB_DEBUG").is_ok() {
        eprintln!("ours top8: {:?}", &all[..8]);
    }

    let ids5: Vec<usize> = reference["top5_ids"]
        .as_array()
        .expect("top5")
        .iter()
        .map(|v| usize::try_from(v.as_u64().expect("id")).expect("id"))
        .collect();
    let vals5: Vec<f32> = reference["top5_logits"]
        .as_array()
        .expect("top5")
        .iter()
        .map(|v| {
            let f = v.as_f64().expect("v");
            #[allow(clippy::cast_possible_truncation)]
            {
                f as f32
            }
        })
        .collect();
    let our_argmax = all[0].0;
    assert!(
        ids5.contains(&our_argmax),
        "our argmax {our_argmax} ({}) is not one of HF's top-5 {ids5:?}",
        all[0].1
    );
    let our_top8: Vec<usize> = all[..8].iter().map(|(t, _)| *t).collect();
    for t in &ids5 {
        assert!(
            our_top8.contains(t),
            "HF top-5 token {t} missing from our top-8 {our_top8:?}"
        );
    }
    for (t, hf) in ids5.iter().zip(&vals5) {
        let ours = logit(*t);
        assert!(
            (ours - hf).abs() < 1.25,
            "top-5 token {t}: ours {ours} vs HF {hf}"
        );
    }
    let probes: Vec<usize> = reference["probe_ids"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|v| usize::try_from(v.as_u64().expect("id")).expect("id"))
        .collect();
    let probe_vals: Vec<f32> = reference["probe_logits"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|v| {
            let f = v.as_f64().expect("v");
            #[allow(clippy::cast_possible_truncation)]
            {
                f as f32
            }
        })
        .collect();
    for (t, hf) in probes.iter().zip(&probe_vals) {
        let ours = logit(*t);
        assert!(
            (ours - hf).abs() < 0.6,
            "probe token {t}: ours {ours} vs HF {hf}"
        );
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}
