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
use driver_cuda::bind::views::{FireStreams, FireViews};
use driver_cuda::device::{Allocator, DeviceBuffer, OwnedStream};
use driver_cuda::dtype::DType;
use driver_cuda::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};
use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_ir::trace::{FireClass, ValueId};

mod common;
use common::{device_or_skip, gpu_guard};

/// The scalars the family texts used to read off their fact structs.
///
/// Upstream lifted `norm_eps`, the rope bases and gpt-oss's sliding window
/// OUT of the facts and onto the forward functions, because two SKUs of one
/// family can differ in those and in nothing else. These tests never read a
/// number back -- they lower, bind and count -- so any well-formed value
/// states the same text, and these are the shipped checkpoints' own.
#[allow(dead_code)]
const EPS: f32 = 1e-6;
/// The common rope base. gpt-oss's is its own.
#[allow(dead_code)]
const THETA: f32 = 1_000_000.0;
/// gpt-oss: YaRN over a 150k base, alternating 128-token windows.
#[allow(dead_code)]
const WINDOWED_THETA: f32 = 150_000.0;
/// The sliding leg's span. `-1` is "no window" and is NOT what gpt-oss says.
#[allow(dead_code)]
const WINDOW: i32 = 128;

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
    /// The prefill schedule this fixture staged, for the statement that NAMES
    /// it.
    ///
    /// `Resolver::raised` defaults to `None` and that default is right for a
    /// resolver holding no raises -- but this one holds one, and hands the
    /// same pointer it puts in `AttnCtx::prefill_plan`. The driver used to
    /// recover the plan from the ctx by guessing from a family string; the
    /// statement says which object it wants now, and a fixture that answers
    /// only weights and names refuses at `RaisedUnbound { key: "fa2.prefill" }`
    /// the moment the prefill launcher binds its first argument.
    ///
    /// One pointer and not a map, because this fixture stages one schedule.
    /// The live path keys on the VALUE for the reason `fire::launch`'s
    /// `LiveResolver` states: two prefills of different head dim both spell
    /// `fa2.prefill`. That cannot arise here.
    prefill_plan: *const std::ffi::c_void,
    /// The RESIDENT half of the raise channel, built exactly as the live path
    /// builds it (`FireViews::build`).
    ///
    /// # Why a fixture needs it now
    ///
    /// The GDN arms used to take their conv and recurrent slabs off the fire's
    /// [`GdnCtx`], so filling that struct was the whole of staging state. The
    /// no-ask sweep moved every per-layer OBJECT onto the operand channel:
    /// `ssm::causal_conv1d_prefill_batched` and its recurrence siblings now
    /// name `"recurrent_state"` in the text, the lowering leaves the operand
    /// `Arg::Raised`, and a resolver that answers only weights, names and
    /// `fa2.prefill` refuses at `RaisedUnbound { key: "recurrent_state" }` on
    /// the first linear layer. Filling `GdnCtx` is still necessary -- it is
    /// where the slabs come FROM -- but it is no longer sufficient.
    ///
    /// `FireViews::build` is the same call the live path makes, over the same
    /// two contexts, so this fixture publishes the slabs by the driver's own
    /// rule rather than by a second one written here. The streams half is
    /// [`FireStreams::default`]: every stream this hybrid names is a seam the
    /// fixture pinned itself and answers through `named`, and a default
    /// `FireStreams` is all-null, which is a refusal and not a wrong pointer
    /// if that ever stops being true.
    views: &'a FireViews,
    /// `plan.runtime`, inverted: which name and which layer each raised value
    /// stands for. The live path keeps the same map for the same reason --
    /// the key on the wire is the vocabulary word, and the LAYER is what
    /// picks one of eighteen slabs.
    runtime: BTreeMap<ValueId, (String, Option<u32>)>,
}
impl Resolver for Live<'_> {
    fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
        self.weights.get(name).map(|b| b.as_ptr().cast_const())
    }
    fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
        // A runtime STREAM the fixture staged answers first, a seam value it
        // pinned answers after -- `fire::launch`'s `LiveResolver` in the same
        // order, and here it is CORRECTNESS and not cost. The two sets are
        // disjoint only because the pin pool excludes `runtime_values`; before
        // it did, every stream resolved to a zeroed pin, and the loudest of
        // those was `"first_token"`: the swept routines read that answer as
        // `ptr as i32`, so a pin handed `write_kv_to_pages` a DEVICE ADDRESS
        // as its write origin and the append walked off the pool.
        if let Some((name, _)) = self.runtime.get(&value) {
            return self.views.streams.named(name);
        }
        self.named.get(&value).map(|b| b.as_ptr())
    }
    fn raised(&mut self, value: ValueId, key: &str) -> Option<*const std::ffi::c_void> {
        // BY THE KEY for the one prep this fixture publishes itself -- which
        // is the case the trait's doc names as the key's own, and it must be
        // tried first because a default `FireStreams` answers the same word
        // with a null it would refuse on.
        if key == "fa2.prefill" {
            return Some(self.prefill_plan);
        }
        // Everything else is a RESIDENT object out of the closed vocabulary,
        // answered from the view arena by the value's own name and layer. An
        // unrecognised value is `None` and therefore a refusal, not some other
        // layer's slab.
        let (name, layer) = self.runtime.get(&value)?;
        self.views.raised(name, *layer, value)
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
        proj_repr: model_dsl::WeightRepr::Bf16,
        window_left: Vec::new(),
    };
    let plan = qwen3_5_hybrid_cuda::<model::qwen_3_5::forward::ShippedW1, model::qwen_3_5::forward::ShippedW2, model::qwen_3_5::forward::ShippedA, model::qwen_3_5::forward::ShippedKv>(&hybrid, &cuda, FireClass::Prefill, EPS, THETA);
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
    // WITH THE BOOT'S ANSWER, not `Boot::default()`, for the reason
    // `real_gemma4.rs` states it: `attn::write_kv_to_pages` is an `untraced!`
    // declaration standing for a CHOICE, and `Boot::route` is what resolves
    // it to `_bf16` or `_quantised`. A plan that states no KV dtype resolves
    // it to neither, and this walk refused at launch 51 with `NoArm`. The
    // caches built below are all `native_bf16: true`, so the fact is
    // `Some(true)` and stating it here is what makes the two agree.
    let dplan = DispatchPlan::with_boot(
        &plan,
        &l,
        driver_cuda::bind::Boot {
            kv_native_bf16: Some(true),
        },
    );

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
    //
    // RUNTIME VALUES ARE NOT SEAMS. The lowering leaves them `Arg::Named` like
    // any other, but the driver stages them per fire and answers them by name;
    // `runtime_values` is the live path's own exclusion and this fixture takes
    // it verbatim, so a stream never gets a pin to be answered from.
    let staged = driver_cuda::bind::views::runtime_values(&plan.runtime);
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width, .. } = a
            && !staged.contains(value)
        {
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
    ws.end_plan_update(&mut sops, raw_stream).expect("end");

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
        max_pages_per_request: 0,
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

    // The streams this fixture staged, under the names the text mints them
    // by. `sampling_indices` stays null on purpose: every row is sampled, so
    // there is no index list, and a gather that named one must REFUSE rather
    // than read an empty pin.
    let views = FireViews::build(
        Some(&attn),
        Some(&gdn),
        FireStreams {
            positions: positions.as_ptr(),
            token_ids: ids.as_ptr(),
            request_of_token: core::ptr::null_mut(),
            qo_indptr: qo_indptr.as_ptr(),
            row_valid: row_valid.as_ptr(),
            sampling_indices: core::ptr::null_mut(),
            first_token: 0,
            qo_indptr_host: qo_indptr_h.as_ptr(),
            kv_page_indptr_host: page_indptr_h.as_ptr(),
            prefill_plan_cache: core::ptr::null_mut(),
        },
    );
    let mut resolver = Live {
        weights: &weights,
        named: &named_bufs,
        prefill_plan: pplan.as_ptr().cast_const(),
        views: &views,
        runtime: plan
            .runtime
            .iter()
            .map(|b| (b.value, (b.name.clone(), b.layer)))
            .collect(),
    };
    // ── The step census, when asked for. ──
    //
    // `real_prefill`'s bisection, over this walk: one launch at a time, the
    // out's rectangle back to the host, and a mean/max/non-finite line per
    // step. It is how a hybrid disagreement is localised to a LAYER and a
    // KERNEL rather than to "the logits are wrong".
    if std::env::var("HYBRID_AB_STEPS").is_ok() {
        let mut host = vec![0u8; TOKENS * 8192 * 2];
        for i in 0..l.launches.len() {
            let launch = &l.launches[i];
            let bound = driver_cuda::bind::bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|e| panic!("bind {i}: {e:?}"));
            driver_cuda::bind::dispatch(
                &bound,
                dplan.spec(i),
                frame,
                &mut resolver,
                &ctx,
                AttnRegions::whole(Some(&attn)).of(&launch.rows),
                Some(&gdn),
            )
            .unwrap_or_else(|e| panic!("dispatch {i}: {e:?}"));
            stream.as_ref().synchronize().expect("step sync");
            let Some(out) = dplan.spec(i).outs.first() else {
                continue;
            };
            // THE ELEMENT WIDTH IS ON THE ARG, and reading it is not optional
            // here: a hybrid's recurrence carries fp32 intermediates (hence
            // `norm::rmsnorm_gated_fp32_in` reading them), so a census that
            // assumes bf16 reports `qwen_gdn_post_conv_prep` at 1e36 with a
            // third of its rows non-finite and sends the reader after a
            // kernel that is fine.
            let (base, width, elem) = match out {
                Arg::Named {
                    value,
                    width,
                    bytes,
                } => (
                    named_bufs[value].as_ptr().cast_const().cast::<u8>(),
                    *width,
                    *bytes as usize,
                ),
                // SAFETY: `at` is an offset the lowering assigned inside the
                // arena this frame was built from.
                Arg::Arena { at, width, bytes } => (
                    unsafe { arena.as_ptr().cast_const().cast::<u8>().add(*at) },
                    *width,
                    *bytes as usize,
                ),
                Arg::Weight(_) | Arg::Raised { .. } => continue,
            };
            let n = TOKENS * width as usize;
            let bytes = n * elem;
            if bytes > host.len() || !matches!(elem, 2 | 4) {
                continue;
            }
            // SAFETY: `base` addresses `bytes` live device bytes -- the
            // rectangle the launch just wrote -- and `host` is that long.
            let rc = unsafe {
                cudarc::runtime::sys::cudaMemcpy(
                    host.as_mut_ptr().cast(),
                    base.cast(),
                    bytes,
                    cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(
                rc,
                cudarc::runtime::sys::cudaError::cudaSuccess,
                "step d2h {i}"
            );
            let (mut sum, mut max, mut bad) = (0f64, 0f32, 0usize);
            for k in 0..n {
                let x = if elem == 4 {
                    f32::from_le_bytes([
                        host[k * 4],
                        host[k * 4 + 1],
                        host[k * 4 + 2],
                        host[k * 4 + 3],
                    ])
                } else {
                    let bits = u16::from_le_bytes([host[k * 2], host[k * 2 + 1]]);
                    f32::from_bits(u32::from(bits) << 16)
                };
                if x.is_finite() {
                    sum += f64::from(x.abs());
                    max = max.max(x.abs());
                } else {
                    bad += 1;
                }
            }
            // The last row's first few elements, when a step is under
            // suspicion: an aggregate says THAT two implementations differ
            // and an element says HOW.
            if std::env::var("HYBRID_AB_HEAD").as_deref() == Ok(&i.to_string()) {
                let at = (TOKENS - 1) * width as usize;
                let head: Vec<f32> = (at..at + 8)
                    .map(|k| {
                        if elem == 4 {
                            f32::from_le_bytes([
                                host[k * 4],
                                host[k * 4 + 1],
                                host[k * 4 + 2],
                                host[k * 4 + 3],
                            ])
                        } else {
                            let b = u16::from_le_bytes([host[k * 2], host[k * 2 + 1]]);
                            f32::from_bits(u32::from(b) << 16)
                        }
                    })
                    .collect();
                println!("HEAD [{i:4}] {head:?}");
            }
            println!(
                "STEP [{i:4}] {:<46} w={:<26} width={width:5} e{elem} mean|x|={:>10.5} max={max:>9.3} bad={bad}",
                l.kernels[l.launches[i].kernel as usize],
                dplan.spec(i).weight.as_deref().unwrap_or("-"),
                sum / n.max(1) as f64,
            );
        }
    }

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
        // THE OUT, not `args.start` -- a launch's argument run begins with
        // its INPUTS, so the old reading picked up `token_ids` and panicked
        // that "embed writes the arena". Where the residual lives is the
        // allocator's choice and not the text's, so both channels are read.
        let e_out = (0..l.launches.len())
            .find_map(|i| {
                (l.kernels[l.launches[i].kernel as usize] == "layout::embed_bf16")
                    .then(|| dplan.spec(i).outs.first().cloned())
                    .flatten()
            })
            .expect("embed ran");
        let (back, e_at) = match e_out {
            Arg::Arena { at, .. } => {
                let mut back = vec![0u8; l.arena_bytes];
                arena.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
                (back, at)
            }
            Arg::Named { value, .. } => {
                let b = &named_bufs[&value];
                let mut back = vec![0u8; b.len()];
                b.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
                (back, 0)
            }
            other => panic!("embed writes somewhere this dump cannot read: {other:?}"),
        };
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
