//! gemma-4's real-weight A/B (E-gate family #2's parity anchor):
//! `google/gemma-4-E2B-it` from the HF cache, prefilled through `run()`
//! — 28 sliding layers through the planless flashinfer prefill, 7 full
//! layers at head_dim 512 through the naive paged kernel, the PLE
//! double-embedding relay, sandwich-norm fusions crossing layer
//! boundaries, 20 KV-shared layers attending through their source's
//! pages — against a committed transformers reference
//! (`tests/oracle/real_decode/gemma4_e2b.json`).
//!
//! ONE deployment for the family, per the campaign's constraint. The
//! gemma-2 arms ride the same rows; its own A/B stays blocked on a
//! gated checkpoint (recorded in the retirement wiki).
//!
//! The checkpoint's facts, verified against `config.json`
//! (`text_config`): 35 layers, full attention every 5th, 8 q / 1 kv
//! head at 256 (sliding) / 512 (full, rotary 128 of 512, theta 1e6 vs
//! the sliding 1e4 — `rope_parameters` splits BY LAYER KIND, which is
//! what `DispatchCtx::rope_theta_by_layer` exists for), MLP 6144
//! doubled to 12288 on the 20 KV-shared trailing layers
//! (`use_double_wide_mlp`), PLE dim 256, vocab 262144 tied, final
//! logit softcap 30. All tensors BF16. The per-layer `layer_scalar`
//! [1] tensors feed `DispatchCtx::scales` under the fused sandwich
//! norm's `ple_norm` weight names — the C++ reads them to host once at
//! load (`read_bf16_scalar_once`) and so does this binder.
//!
//! The fused banks (`qkv`, `gate_up`) are NOT in the checkpoint: the
//! engine's `dense_fused_projection_joins` builds them at load (E2B's
//! banks total ~2.2 GB, inside the 10 GiB budget, so the LIVE binding
//! fuses both), and this binder reproduces the join by host
//! concatenation — q‖k‖v on the non-shared layers, gate‖up everywhere.
//!
//! The bar is TIGHTER than qwen3_5's, because the measurement says it
//! can be (2026-08-09, this checkpoint and prompt): our argmax equals
//! HF's, the residual entering the final norm matches HF's
//! `hidden_states[35]` to three digits (103.41 vs 103.40, components
//! agreeing to bf16 print precision), and every HF top-5 logit lands
//! within 0.19 of ours. The final softcap compresses logits into ±30,
//! which tightens the comparison rather than loosening it. So: argmax
//! EQUAL, HF's top-5 inside our top-8, top-5 logits within 0.75,
//! probes within 0.6 — the slack over the measured 0.19 covers
//! inter-fire jitter (allocation addresses shifting GEMM reduction
//! orders, the qwen3_5 copy-state experiment's ~0.1). A structural bug
//! (a swapped binding, a wrong source pool, a wrong per-layer theta, a
//! per-head norm fired flat) does not fail these bars by 0.1 — it
//! demolishes them; every one of the four defects this A/B caught
//! saturated the softcap.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use driver_cuda::bind::abi::{KvCacheLayerView, KvCacheScheme};
use driver_cuda::bind::{
    AttnCtx, AttnRegions, DispatchCtx, DispatchPlan, Frame, PrefillPlan, Resolver, run,
};
use driver_cuda::device::{Allocator, DeviceBuffer, OwnedStream};
use driver_cuda::dtype::DType;
use driver_cuda::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};
use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
use model::gemma_4::forward::gemma4_cuda;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_ir::trace::{FireClass, ValueId};

mod common;
use common::{device_or_skip, gpu_guard};

/// The checkpoint reader `real_hybrid` carries, unchanged: every tensor
/// in this file is BF16, and the reader admits BF16|F32.
struct Checkpoint {
    raws: Vec<Vec<u8>>,
    index: BTreeMap<String, (usize, usize, usize)>,
}

impl Checkpoint {
    fn open(cache_dir: &str) -> Option<Self> {
        let home = std::env::var_os("HOME")?;
        let snaps =
            PathBuf::from(home).join(format!(".cache/huggingface/hub/{cache_dir}/snapshots"));
        let snap = std::fs::read_dir(&snaps).ok()?.filter_map(Result::ok).find_map(|e| {
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
        let (f, b, e) = *self.index.get(name).unwrap_or_else(|| panic!("checkpoint lacks {name}"));
        &self.raws[f][b..e]
    }
}

/// Bind every trace name the gemma-4 prefill states to the checkpoint —
/// `gemma4.cpp`'s binder plus the engine's fused joins, as data. The
/// KV-shared layers (15..35) state only the Q leg, so k/v/k_norm and
/// the qkv bank exist only below 15; `gate_up` is fused on every layer
/// (both the 6144 and the double-wide 12288 shapes concatenate the same
/// way).
fn bind(ckpt: &Checkpoint, facts: &Gemma4Facts, sink: &mut dyn FnMut(String, Vec<u8>)) {
    let p = "model.language_model";
    sink("embed".into(), ckpt.bytes(&format!("{p}.embed_tokens.weight")).to_vec());
    // The PLE table is `[vocab, 35 * ple_dim]` row-major; a TRUNCATED
    // depth (the bisection aid) states a narrower row, so the table's
    // rows re-pack to the stated stride — without this every truncated
    // run reads the wrong vocab rows and the bisection lies.
    let ple_full = ckpt.bytes(&format!("{p}.embed_tokens_per_layer.weight"));
    let full_row = 35 * facts.ple_dim as usize * 2;
    let want_row = facts.layers as usize * facts.ple_dim as usize * 2;
    if want_row == full_row {
        sink("embed_per_layer".into(), ple_full.to_vec());
    } else {
        let mut packed = Vec::with_capacity(ple_full.len() / full_row * want_row);
        for row in ple_full.chunks_exact(full_row) {
            packed.extend_from_slice(&row[..want_row]);
        }
        sink("embed_per_layer".into(), packed);
    }
    sink(
        "ple_model_proj".into(),
        ckpt.bytes(&format!("{p}.per_layer_model_projection.weight")).to_vec(),
    );
    sink(
        "ple_model_norm".into(),
        ckpt.bytes(&format!("{p}.per_layer_projection_norm.weight")).to_vec(),
    );
    sink("final_norm".into(), ckpt.bytes(&format!("{p}.norm.weight")).to_vec());
    for n in 0..facts.layers {
        let lp = format!("{p}.layers.{n}");
        let mut w =
            |trace: &str, hf: String| sink(format!("layer.{n}.{trace}"), ckpt.bytes(&hf).to_vec());
        w("attn_norm", format!("{lp}.input_layernorm.weight"));
        w("post_attn_norm", format!("{lp}.post_attention_layernorm.weight"));
        w("pre_ffw_norm", format!("{lp}.pre_feedforward_layernorm.weight"));
        w("post_ffw_norm", format!("{lp}.post_feedforward_layernorm.weight"));
        w("q_norm", format!("{lp}.self_attn.q_norm.weight"));
        w("o_proj", format!("{lp}.self_attn.o_proj.weight"));
        w("down", format!("{lp}.mlp.down_proj.weight"));
        w("ple_gate", format!("{lp}.per_layer_input_gate.weight"));
        w("ple_proj", format!("{lp}.per_layer_projection.weight"));
        w("ple_norm", format!("{lp}.post_per_layer_input_norm.weight"));
        if facts.is_kv_shared(n) {
            w("q_proj", format!("{lp}.self_attn.q_proj.weight"));
        } else {
            w("k_norm", format!("{lp}.self_attn.k_norm.weight"));
            let mut qkv = ckpt.bytes(&format!("{lp}.self_attn.q_proj.weight")).to_vec();
            qkv.extend_from_slice(ckpt.bytes(&format!("{lp}.self_attn.k_proj.weight")));
            qkv.extend_from_slice(ckpt.bytes(&format!("{lp}.self_attn.v_proj.weight")));
            sink(format!("layer.{n}.qkv"), qkv);
        }
        let mut gate_up = ckpt.bytes(&format!("{lp}.mlp.gate_proj.weight")).to_vec();
        gate_up.extend_from_slice(ckpt.bytes(&format!("{lp}.mlp.up_proj.weight")));
        sink(format!("layer.{n}.gate_up"), gate_up);
    }
}

/// The per-layer `layer_scalar` [1] tensors, read to host —
/// `read_bf16_scalar_once`, the C++ load-time read.
fn layer_scalars(ckpt: &Checkpoint, layers: u32) -> Vec<f32> {
    (0..layers)
        .map(|n| {
            let b = ckpt.bytes(&format!("model.language_model.layers.{n}.layer_scalar"));
            assert_eq!(b.len(), 2, "layer_scalar {n} is one bf16");
            f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
        })
        .collect()
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
fn gemma4_matches_transformers_on_real_weights() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("gemma-4 E2B A/B") else {
        return;
    };
    let Some(ckpt) = Checkpoint::open("models--google--gemma-4-E2B-it") else {
        eprintln!("skipped: gemma-4-E2B-it not in the HF cache");
        return;
    };
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle/real_decode/gemma4_e2b.json"),
        )
        .expect("reference file"),
    )
    .expect("reference json");

    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const TOKENS: usize = 7;
    const VOCAB: usize = 262_144;
    const PAGE: i32 = 16;

    let mut facts = Gemma4Facts::gemma_4_e2b();
    // Bisection aid: truncate the depth to compare the residual against
    // transformers' per-layer hidden states. The KV-shared count moves
    // with the truncation so the shared/owning split stays E2B's (first
    // 15 layers own).
    if let Ok(n) = std::env::var("GEMMA4_AB_LAYERS") {
        facts.layers = n.parse().expect("layer count");
        facts.kv_shared_layers = facts.layers.saturating_sub(15);
    }
    // The LIVE cuda set: both banks fused (the join's budget admits
    // E2B), native bf16 pages on L40S.
    let mut cuda = Gemma4CudaFacts::gemma_4_e4b_synthetic();
    // The checkpoint's per-layer `layer_scalar`, which the fixture cannot know
    // and the landing needs: with the identity every logit saturates the
    // softcap. Read the same way `serve/load.rs` reads it at load.
    cuda.layer_scalars = layer_scalars(&ckpt, facts.layers);
    let plan = gemma4_cuda(&facts, &cuda, FireClass::Prefill);
    // A prefill: one request, TOKENS rows -- every row multi-token, which
    // is what `GuardPred::WindowOne` reads (graph.md §4.1).
    let rows: Vec<Row> = vec![Row { samples: true, multi_token: true, ..Row::default() }; TOKENS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    // WITH THE BOOT'S ANSWER, not `Boot::default()`. `attn::write_kv_to_pages`
    // is an `untraced!` declaration and `Boot::route` is what resolves it to
    // `_bf16` or `_quantised`; a plan that states no KV dtype resolves it to
    // neither and the walk refuses at launch 15 with `NoArm`. The cache this
    // test builds below is `KvCacheScheme::Native` at `DType::Bf16`, so the
    // fact is `Some(true)` and stating it here is what makes the two agree.
    let dplan = DispatchPlan::with_boot(
        &plan,
        &l,
        driver_cuda::bind::Boot { kv_native_bf16: Some(true) },
    );

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };

    // ── The real weights, uploaded under their trace names. ──
    let mut weights: BTreeMap<String, DeviceBuffer> = BTreeMap::new();
    bind(&ckpt, &facts, &mut |name, host| {
        weights.insert(name, up(&host));
    });

    // ── The seam-value pool. ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width, .. } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width, .. } = a {
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

    // ── KV pools: the 15 owning layers allocate; the 20 shared layers
    // VIEW their source's pool — the load-time decision, as data. ──
    let pools: Vec<Option<(DeviceBuffer, DeviceBuffer)>> = (0..facts.layers)
        .map(|i| {
            if facts.is_kv_shared(i) {
                return None;
            }
            let plane = (PAGE * i32::from(facts.kv_heads as u16) * facts.head_dim_of(i) as i32)
                as usize
                * 2;
            let mut k = alloc.alloc(plane).expect("k pool");
            let mut v = alloc.alloc(plane).expect("v pool");
            k.memset(0, stream.as_ref()).expect("zk");
            v.memset(0, stream.as_ref()).expect("zv");
            Some((k, v))
        })
        .collect();
    let layers: Vec<KvCacheLayerView> = (0..facts.layers)
        .map(|i| {
            let src = facts.kv_source(i).unwrap_or(i);
            let (k, v) = pools[src as usize]
                .as_ref()
                .map(|(k, v)| (k.as_ptr(), v.as_ptr()))
                .expect("the source layer owns a pool");
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(src).expect("layer"),
                num_pages: 1,
                page_size: PAGE,
                num_kv_heads: i32::from(facts.kv_heads as u16),
                head_dim: facts.head_dim_of(i) as i32,
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

    // ── One request over the whole prompt. ──
    let prompt: Vec<i32> = reference["prompt_ids"]
        .as_array()
        .expect("prompt")
        .iter()
        .map(|v| i32::try_from(v.as_i64().expect("id")).expect("id"))
        .collect();
    assert_eq!(prompt.len(), TOKENS);
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    // The HOST CSRs stay alive past `run()`: the planless prefill plans
    // per fire and reads them from the host.
    let qo_indptr_h: [u32; 2] = [0, 7];
    let page_indptr_h: [u32; 2] = [0, 1];
    let last_lens_h: [u32; 1] = [7];
    let csr_indices = up(&u32s(&[0]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&prompt.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions = up(&(0i32..7).flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * facts.q_heads as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    // A RAISED PLAN, where this used to pass none. The sliding layers reach
    // `attn::attention_flashinfer_prefill`, whose arm reads the plan handle
    // off the fire — "nothing states the plan this fire did not raise" is its
    // refusal — so a null handle is not the planless variant, it is no
    // attention at all. Raised over the same three host CSRs the walk binds,
    // at the sliding layers' own head dim; the full layers take the naive
    // paged path and read none of this.
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h,
        &page_indptr_h,
        &last_lens_h,
        facts.q_heads as i32,
        i32::from(facts.kv_heads as u16),
        facts.head_dim as i32,
        PAGE,
        ws.view(),
        raw_stream,
        false,
        -1,
    );
    ws.end_plan_update(&mut sops, raw_stream).expect("end");

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        prefill_workspace: ws.view(),
        layers,
        // No guard-owned pins in this walk: both attention arms take
        // [q, o] as stated values.
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
        qo_indptr_h: qo_indptr_h.as_ptr(),
        kv_page_indptr_h: page_indptr_h.as_ptr(),
        num_requests: 1,
        num_pages_in_batch: 1,
        max_pages_per_request: 0,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        score_window: 32,
        window_left: -1,
        // Sliding layers window at `sliding_window` 512; full layers run
        // unbounded — `per_layer_window_left`, the C++ parse-time table.
        window_left_by_layer: (0..facts.layers)
            .map(|i| if facts.is_full_attn(i) { -1 } else { 512 })
            .collect(),
        logits_soft_cap: 0.0,
        // 1.0 at every gemma-4 attention site in the C++ — the q/k norms
        // carry the scaling, the dispatch does not.
        sm_scale: 1.0,
    };

    // The four scale constants (`std::sqrt`, fp32 — the C++ arms') and
    // the 35 per-layer sandwich scalars under their `ple_norm` names.
    let mut scales = BTreeMap::new();
    let hidden = facts.hidden as f32;
    scales.insert("sqrt_hidden".into(), hidden.sqrt());
    scales.insert("sqrt_ple_dim".into(), (facts.ple_dim as f32).sqrt());
    scales.insert("rsqrt_hidden".into(), 1.0 / hidden.sqrt());
    scales.insert("rsqrt_2".into(), 1.0 / 2f32.sqrt());
    for (n, s) in layer_scalars(&ckpt, facts.layers).into_iter().enumerate() {
        scales.insert(format!("layer.{n}.ple_norm"), s);
    }

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
        rope_theta: 1e4,
        // `rope_parameters` splits by layer kind: sliding 1e4 default,
        // full 1e6 proportional — `per_layer_rope_theta`, expanded.
        rope_theta_by_layer: (0..facts.layers)
            .map(|i| if facts.is_full_attn(i) { 1e6 } else { 1e4 })
            .collect(),
        // `rotary_of(l)` for the Q-ONLY partial ropes, whose dsl
        // statement carries no width: 128 on the full layers.
        rotary_by_layer: (0..facts.layers)
            .map(|i| if facts.is_full_attn(i) { facts.global_rotary_dim } else { 0 })
            .collect(),
        head_dim: facts.head_dim as i32,
        num_q_heads: facts.q_heads as i32,
        num_kv_heads: facts.kv_heads as i32,
        vocab: i32::try_from(VOCAB).expect("vocab"),
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: facts.logit_softcap,
        ple_dim: facts.ple_dim as i32,
        scales,
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

    let mut resolver = Live { weights: &weights, named: &named_bufs };
    let ran = if std::env::var("GEMMA4_AB_TRACE").is_ok() {
        // Launch-by-launch walk with a sync and a last-row norm of the
        // first output after each — the bisection's microscope.
        use driver_cuda::bind::{bind, dispatch};
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
            if kernel == "layout::transpose_bf16_nld_to_lnd" {
                // Read both relay layouts NOW — the arena reuses these
                // slots later, so a post-run read lies.
                let (src_at, dst_at) = match (
                    &l.args[launch.args.start as usize],
                    &l.args[launch.args.start as usize + 1],
                ) {
                    (Arg::Arena { at: s, .. }, Arg::Arena { at: d, .. }) => (*s, *d),
                    other => panic!("the relay rides the arena, got {other:?}"),
                };
                let mut whole = vec![0u8; l.arena_bytes];
                arena.copy_to_host(&mut whole, stream.as_ref()).expect("d2h");
                stream.as_ref().synchronize().expect("sync");
                let d = facts.ple_dim as usize;
                let lcount = facts.layers as usize;
                let bf = |off: usize| {
                    f32::from_bits(
                        u32::from(u16::from_le_bytes([whole[off], whole[off + 1]])) << 16,
                    )
                };
                for layer in 0..lcount.min(3) {
                    let r = TOKENS - 1;
                    let src_off = src_at + ((r * lcount + layer) * d) * 2;
                    let dst_off = dst_at + ((layer * TOKENS + r) * d) * 2;
                    let sn: f32 = (0..d).map(|c| bf(src_off + c * 2).powi(2)).sum::<f32>().sqrt();
                    let dn: f32 = (0..d).map(|c| bf(dst_off + c * 2).powi(2)).sum::<f32>().sqrt();
                    eprintln!(
                        "  relay L{layer} r{r}: src n={sn:.3} [{:.3},{:.3}] dst n={dn:.3} [{:.3},{:.3}]",
                        bf(src_off),
                        bf(src_off + 2),
                        bf(dst_off),
                        bf(dst_off + 2)
                    );
                }
            }
            let out = dplan.spec(i).outs.first().cloned().or_else(|| {
                (launch.args.end > launch.args.start)
                    .then(|| l.args[launch.args.end as usize - 1].clone())
            });
            if let Some(a) = out {
                let (host, width): (Vec<u8>, usize) = match a {
                    Arg::Arena { at, width, .. } => {
                        let w = width as usize;
                        let rows = (launch.rows.end - launch.rows.start) as usize;
                        let mut whole = vec![0u8; l.arena_bytes];
                        arena.copy_to_host(&mut whole, stream.as_ref()).expect("d2h");
                        stream.as_ref().synchronize().expect("sync");
                        (whole[at..at + rows * w * 2].to_vec(), w)
                    }
                    Arg::Named { value, width, .. } => {
                        // The pins are allocated 4 bytes per element
                        // (fp32-capable); the bf16 rows live in the
                        // FIRST half — truncate so the row math below
                        // reads data, not the zeroed tail.
                        let b = &named_bufs[&value];
                        let mut host = vec![0u8; b.len()];
                        b.copy_to_host(&mut host, stream.as_ref()).expect("d2h");
                        stream.as_ref().synchronize().expect("sync");
                        host.truncate(host.len() / 2);
                        (host, width as usize)
                    }
                    Arg::Weight(_) | Arg::Raised { .. } => continue,
                };
                let rows = host.len() / (width * 2);
                let last = rows.saturating_sub(1);
                let v = |c: usize| {
                    let off = (last * width + c) * 2;
                    f32::from_bits(u32::from(u16::from_le_bytes([host[off], host[off + 1]])) << 16)
                };
                let norm: f32 = (0..width).map(|c| v(c).powi(2)).sum::<f32>().sqrt();
                eprintln!(
                    "  #{i:3} {kernel} w={width} phd={:?} wt={:?} last-row norm={norm:.3} head=[{:.3},{:.3},{:.3},{:.3}]",
                    dplan.spec(i).per_head_dim,
                    dplan.spec(i).weight,
                    v(0),
                    v(1),
                    v(2),
                    v(3)
                );
            }
        }
        l.launches.len()
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), None)
            .unwrap_or_else(|e| panic!("the gemma-4 A/B walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len());
    stream.as_ref().synchronize().expect("the fire retires");

    if std::env::var("GEMMA4_AB_DEBUG").is_ok() {
        // The residual right before the final norm — the value the last
        // `norm::rmsnorm_bf16` reads. Its last row compares against
        // transformers' `hidden_states[layers]`.
        let y_at = l
            .launches
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, x)| {
                (l.kernels[x.kernel as usize] == "norm::rmsnorm_bf16").then(|| {
                    match &l.args[x.args.start as usize] {
                        Arg::Arena { at, .. } => (i, *at),
                        other => panic!("the final norm reads the arena, got {other:?}"),
                    }
                })
            })
            .expect("a final norm ran");
        let mut back = vec![0u8; l.arena_bytes];
        arena.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
        stream.as_ref().synchronize().expect("sync");
        let hidden = facts.hidden as usize;
        let h = |r: usize, c: usize| {
            let off = y_at.1 + (r * hidden + c) * 2;
            let bits = u16::from_le_bytes([back[off], back[off + 1]]);
            f32::from_bits(u32::from(bits) << 16)
        };
        let head: Vec<f32> = (0..8).map(|c| h(TOKENS - 1, c)).collect();
        let norm: f32 = (0..hidden).map(|c| h(TOKENS - 1, c).powi(2)).sum::<f32>().sqrt();
        eprintln!(
            "ours residual before final norm (launch {}): last row head {head:?} norm={norm:.4}",
            y_at.0
        );

        // NO RELAY COMPARISON HERE. It used to read `layout::transpose_bf16_
        // nld_to_lnd`'s two sides out of this post-run copy and report a
        // 25-fold disagreement, which is not a defect and not even a
        // measurement: the arena reuses that span, and `mlp::geglu_tanh_bf16`
        // at launch 24 is the first of THIRTY-NINE later launches that write
        // over it. Read at the moment it is written -- `GEMMA4_AB_TRACE`, which
        // syncs after every launch and samples there -- the two sides agree to
        // the bit, as a transpose must. The duplicate here survived because its
        // numbers were plausible and pointed at gemma-4's one genuinely open
        // question, so it read like a lead. It is deleted rather than fixed:
        // `GEMMA4_AB_TRACE` already asks it honestly.
    }

    if std::env::var("GEMMA4_AB_LAYERS").is_ok() {
        // Truncated depth states a truncated model; the logits bar is
        // meaningless there — the residual dump above is the product.
        ws.release(&mut sops);
        cublas.release(&mut cublas_ops);
        return;
    }

    // ── The A/B: the last row's logits against transformers'. ──
    let lv = logits_value.expect("the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
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
    if std::env::var("GEMMA4_AB_DEBUG").is_ok() {
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
    assert_eq!(
        our_argmax, ids5[0],
        "our argmax ({} at {}) is not HF's ({})",
        our_argmax, all[0].1, ids5[0]
    );
    let our_top8: Vec<usize> = all[..8].iter().map(|(t, _)| *t).collect();
    for t in &ids5 {
        assert!(our_top8.contains(t), "HF top-5 token {t} missing from our top-8 {our_top8:?}");
    }
    for (t, hf) in ids5.iter().zip(&vals5) {
        let ours = logit(*t);
        assert!((ours - hf).abs() < 0.75, "top-5 token {t}: ours {ours} vs HF {hf}");
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
        assert!((ours - hf).abs() < 0.6, "probe token {t}: ours {ours} vs HF {hf}");
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}
