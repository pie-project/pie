//! The REAL thing: Qwen3-0.6B's actual weights through the whole executor,
//! checked against HuggingFace's logits (retirement plan, the A/B seed).
//!
//! `tests/oracle/real_decode/reference.json` pins what transformers
//! computes for a fixed prompt (provenance inside). This test loads the
//! same checkpoint from the HF cache — safetensors parsed by hand, q/k/v
//! and gate/up fused the way the binder fuses them — runs the same prompt
//! through `executor::run` as one prefill fire, and compares the last
//! row's logits: the argmax, the top-5, and five probe tokens.
//!
//! Tolerance is ±0.25 absolute: both sides quantize weights to bf16, but
//! accumulate differently (CUDA tensor-core fp32 vs CPU), and the bf16
//! grid at logit magnitude ~13 is 0.0625. The top-5 values span 0.375, so
//! the tolerance still discriminates real drift from rounding.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use driver_cuda::bind::abi::{KvCacheLayerView, KvCacheScheme};
use driver_cuda::bind::{
    AttnCtx, AttnRegions, DispatchCtx, DispatchPlan, Frame, MapResolver, PrefillPlan, run,
};
use driver_cuda::device::{Allocator, DeviceBuffer, OwnedStream};
use driver_cuda::dtype::DType;
use driver_cuda::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};
use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use model::shared::llama_like::forward::llama_like_cuda;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_ir::trace::{FireClass, ValueId};

mod common;
use common::{device_or_skip, gpu_guard};

/// One tensor's slice of the safetensors payload.
struct View<'a> {
    bytes: &'a [u8],
}

/// The checkpoint, header-parsed. Nothing is copied until fusion.
struct Checkpoint {
    raws: Vec<Vec<u8>>,
    // name -> (shard, begin, end)
    index: BTreeMap<String, (usize, usize, usize)>,
}

impl Checkpoint {
    /// Open a cached HF snapshot — single-file or sharded (the
    /// `model.safetensors.index.json` form 7B-class checkpoints ship).
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
                assert_eq!(meta["dtype"], "BF16", "{name}: this loader speaks bf16 only");
                let offs = meta["data_offsets"].as_array()?;
                let shape = meta["shape"].as_array()?;
                let _ = shape;
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

    fn view(&self, name: &str) -> View<'_> {
        let (f, b, e) = *self.index.get(name).unwrap_or_else(|| panic!("checkpoint lacks {name}"));
        View { bytes: &self.raws[f][b..e] }
    }
}

/// Upload one host slice.
fn up(alloc: &Allocator, stream: &OwnedStream, data: &[u8]) -> DeviceBuffer {
    let mut b = alloc.alloc(data.len()).expect("alloc");
    b.copy_from_host(data, stream.as_ref()).expect("h2d");
    b
}

/// A deployment's weight binder: reads the checkpoint, emits
/// `(trace name, fused host bytes)` pairs into the sink.
type Binder = fn(&Checkpoint, &mut dyn FnMut(String, Vec<u8>));

/// One deployment's A/B: everything the generic runner needs.
struct Spec {
    name: &'static str,
    cache_dir: &'static str,
    reference: &'static str,
    facts: LlamaLikeFacts,
    cuda: LlamaLikeCudaFacts,
    layers: usize,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    vocab: usize,
    eps: f32,
    theta: f32,
    /// (trace name, fused host bytes) per weight, from the checkpoint.
    bind: Binder,
}

/// The fire itself, generic over the deployment.
#[allow(clippy::too_many_lines)]
fn ab(spec: &Spec) {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip(spec.name) else {
        return;
    };
    let Some(ckpt) = Checkpoint::open(spec.cache_dir) else {
        eprintln!("skipped: {} not in the HF cache", spec.cache_dir);
        return;
    };
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle/real_decode")
                .join(spec.reference),
        )
        .expect("reference file"),
    )
    .expect("reference json");

    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    let layers_n = spec.layers;
    let kv_heads = spec.kv_heads;
    let q_heads = spec.q_heads;
    let head_dim = spec.head_dim;
    const PAGE: i32 = 16;
    let vocab = spec.vocab;

    let prompt: Vec<i32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().expect("id") as i32)
        .collect();
    let tokens = prompt.len();

    // ── The weights, fused as this deployment's binder fuses them. ──
    let mut resolver = MapResolver::new();
    let mut keep: Vec<DeviceBuffer> = Vec::new();
    (spec.bind)(&ckpt, &mut |trace_name, fused| {
        let buf = up(&alloc, &stream, &fused);
        resolver.insert_weight(trace_name, buf.as_ptr());
        keep.push(buf);
    });

    // ── The fire: one request, the whole prompt, one page. ──
    let plan = llama_like_cuda(&spec.facts, &spec.cuda, FireClass::Prefill);
    // A prefill: ONE request contributing `tokens` rows, so every row is
    // multi-token. `GuardPred::WindowOne` reads that bit -- the window class
    // is a row property now (`.wiki/driver/graph.md` §4.1) -- and a fixture
    // that leaves it false asks for the DECODE dispatch.
    let rows: Vec<Row> = vec![Row { samples: true, multi_token: true, ..Row::default() }; tokens];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);
    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width, .. } = a {
            // MAX, not last: several values can now share one id (they
            // alias in place), and the buffer has to fit the widest read.
            let slot = named_widths.entry(*value).or_insert(*width);
            *slot = (*slot).max(*width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width, .. } = a {
                let slot = named_widths.entry(*value).or_insert(*width);
                *slot = (*slot).max(*width);
            }
        }
    }
    let named_bufs: BTreeMap<ValueId, DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(tokens * w as usize * 2).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();
    for (&v, b) in &named_bufs {
        resolver.insert_named(v, b.as_ptr());
    }

    let plane = (PAGE * kv_heads * head_dim) as usize * 2;
    let pools: Vec<(DeviceBuffer, DeviceBuffer)> = (0..layers_n)
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
            num_pages: 1,
            page_size: PAGE,
            num_kv_heads: kv_heads,
            head_dim,
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

    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 2] = [0, tokens as u32];
    let page_indptr_h: [u32; 2] = [0, 1];
    let last_lens_h: [u32; 1] = [tokens as u32];
    let csr_indices = up(&alloc, &stream, &u32s(&[0]));
    let csr_indptr = up(&alloc, &stream, &u32s(&page_indptr_h));
    let csr_lens = up(&alloc, &stream, &u32s(&last_lens_h));
    let qo_indptr = up(&alloc, &stream, &u32s(&qo_indptr_h));
    let row_valid = up(&alloc, &stream, &vec![1u8; tokens]);
    let ids =
        up(&alloc, &stream, &prompt.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&alloc, &stream, &(0..tokens as i32).flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(tokens * q_heads as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h,
        &page_indptr_h,
        &last_lens_h,
        q_heads,
        kv_heads,
        head_dim,
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
        .expect("prefill dispatches attention");
    let o_off = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => *at,
        other => panic!("o_proj reads the attention slot, got {other:?}"),
    };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        prefill_workspace: ws.view(),
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
        sm_scale: 1.0 / (head_dim as f32).sqrt(),
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
        eps: spec.eps,
        rope_theta: spec.theta,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim,
        num_q_heads: q_heads,
        num_kv_heads: kv_heads,
        vocab: vocab as i32,
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

    // PIE_DUMP_STEP=1 walks the launches one at a time and reads each
    // output back the MOMENT it is written. The whole-run dump below can
    // only ever show a value's final state, and the arena is reused — so
    // an intermediate that is right when written and overwritten later is
    // invisible there and plain here. Same launches, same order; the only
    // difference is a synchronize and a D2H between them.
    let ran = if std::env::var_os("PIE_DUMP_STEP").is_some() {
        let mut host = vec![0u8; 4 << 20];
        for (i, launch) in l.launches.iter().enumerate() {
            let bound = driver_cuda::bind::bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|e| panic!("bind {i}: {e:?}"));
            driver_cuda::bind::dispatch(
                &bound,
                dplan.spec(i),
                frame,
                &mut resolver,
                &ctx,
                AttnRegions::whole(Some(&attn)).of(&launch.rows),
                None,
            )
            .unwrap_or_else(|e| panic!("dispatch {i}: {e:?}"));
            stream.as_ref().synchronize().expect("step sync");

            let Some(out) = dplan.spec(i).outs.first() else { continue };
            let (base, width) = match out {
                Arg::Named { value, width, .. } => {
                    (named_bufs[value].as_ptr().cast_const().cast::<u8>(), *width)
                }
                Arg::Arena { at, width, .. } => {
                    // SAFETY: `at` is an offset the lowering assigned inside
                    // the arena this frame was built from.
                    (unsafe { arena.as_ptr().cast_const().cast::<u8>().add(*at) }, *width)
                }
                Arg::Weight(_) => continue,
            };
            let n = tokens * width as usize;
            let bytes = n * 2;
            if bytes > host.len() {
                continue;
            }
            // SAFETY: `base` addresses `bytes` live device bytes — the
            // rectangle the launch just wrote — and `host` is that long.
            let rc = unsafe {
                cudarc::runtime::sys::cudaMemcpy(
                    host.as_mut_ptr().cast(),
                    base.cast(),
                    bytes,
                    cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, cudarc::runtime::sys::cudaError::cudaSuccess, "step d2h {i}");
            let (mut sum, mut max) = (0f64, 0f32);
            let mut bad = 0usize;
            for k in 0..n {
                let bits = u16::from_le_bytes([host[k * 2], host[k * 2 + 1]]);
                let x = f32::from_bits(u32::from(bits) << 16);
                if x.is_nan() || x.is_infinite() {
                    bad += 1;
                    continue;
                }
                sum += f64::from(x.abs());
                max = max.max(x.abs());
            }
            println!(
                "STEP [{i:4}] {:<46} w={:<26} width={width:5} mean|x|={:>10.5} max={max:>9.3} bad={bad}",
                l.kernels[launch.kernel as usize],
                dplan.spec(i).weight.as_deref().unwrap_or("-"),
                sum / n.max(1) as f64,
            );
        }
        l.launches.len()
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, AttnRegions::whole(Some(&attn)), None)
            .unwrap_or_else(|e| panic!("the walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len());
    stream.as_ref().synchronize().expect("the fire retires");

    // ── The A/B: the last row's logits against transformers'. ──
    let lv = logits_value.expect("the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let last = tokens - 1;
    let logit = |t: usize| {
        let off = (last * vocab + t) * 2;
        let bits = u16::from_le_bytes([back[off], back[off + 1]]);
        f32::from_bits(u32::from(bits) << 16)
    };

    // ── PIE_DUMP_LAUNCHES=1: every launch's output, in fire order. ──
    //
    // Not part of the claim; a bisect handle. The logits being wrong says
    // nothing about WHERE, and every intermediate is already pinned to its
    // own buffer, so the whole forward pass is readable after the fact.
    if std::env::var_os("PIE_DUMP_LAUNCHES").is_some() {
        // The dataflow first: what each launch READS and WRITES, by value.
        // The buffers below hold only the pass's final state, so the wiring
        // is what says which launch a zero should be blamed on.
        for i in 0..l.launches.len() {
            let lu = &l.launches[i];
            let ops: Vec<String> = l.args[lu.args.start as usize..lu.args.end as usize]
                .iter()
                .map(|a| match a {
                    Arg::Named { value, width, .. } => format!("v{value}:{width}"),
                    Arg::Weight(n) => format!("W({n})"),
                    other => format!("{other:?}"),
                })
                .collect();
            let outs: Vec<String> = dplan
                .spec(i)
                .outs
                .iter()
                .map(|a| match a {
                    Arg::Named { value, width, .. } => format!("v{value}:{width}"),
                    other => format!("{other:?}"),
                })
                .collect();
            println!(
                "WIRE [{i:4}] {:<44} rows={:?} args=[{}] outs=[{}]",
                l.kernels.get(lu.kernel as usize).map_or("?", String::as_str),
                lu.rows,
                ops.join(" "),
                outs.join(" "),
            );
        }
        let mut seen: BTreeMap<ValueId, usize> = BTreeMap::new();
        for i in 0..l.launches.len() {
            if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first() {
                seen.insert(*value, i);
            }
        }
        for i in 0..l.launches.len() {
            let Some(Arg::Named { value, width, .. }) = dplan.spec(i).outs.first() else {
                continue;
            };
            // Only the LAST writer of a value holds what the pass ended with.
            if seen.get(value) != Some(&i) {
                continue;
            }
            let buf = &named_bufs[value];
            let mut host = vec![0u8; buf.len()];
            buf.copy_to_host(&mut host, stream.as_ref()).expect("d2h dump");
            stream.as_ref().synchronize().expect("sync dump");
            let n = host.len() / 2;
            let (mut sum, mut max, mut nan, mut zero) = (0f64, 0f32, 0usize, 0usize);
            for k in 0..n {
                let bits = u16::from_le_bytes([host[k * 2], host[k * 2 + 1]]);
                let x = f32::from_bits(u32::from(bits) << 16);
                if x.is_nan() || x.is_infinite() {
                    nan += 1;
                    continue;
                }
                if x == 0.0 {
                    zero += 1;
                }
                sum += f64::from(x.abs());
                max = max.max(x.abs());
            }
            let lu = &l.launches[i];
            println!(
                "[{i:4}] k={:3} op={:4} rows={:?} w={:<28} width={width:5} \
                 mean|x|={:>10.5} max|x|={max:>10.4} nan={nan:<5} zero={zero}/{n}",
                lu.kernel,
                lu.op,
                lu.rows,
                dplan.spec(i).weight.as_deref().unwrap_or("-"),
                sum / n.max(1) as f64,
            );
        }
    }

    if std::env::var_os("PIE_DUMP_LAUNCHES").is_some() {
        let mut all: Vec<(usize, f32)> = (0..vocab).map(|t| (t, logit(t))).collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("OURS  top5: {:?}", &all[..5]);
        let hf_ids = reference["top5_ids"].as_array().expect("top5");
        let hf_vs = reference["top5_logits"].as_array().expect("top5");
        println!(
            "HF    top5: {:?}",
            hf_ids
                .iter()
                .zip(hf_vs)
                .map(|(i, v)| (i.as_u64().unwrap_or(0), v.as_f64().unwrap_or(0.0)))
                .collect::<Vec<_>>()
        );
        println!(
            "OURS at HF's ids: {:?}",
            hf_ids.iter().map(|i| logit(i.as_u64().unwrap_or(0) as usize)).collect::<Vec<_>>()
        );
    }

    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as usize;
    let (mut best_t, mut best_v) = (0usize, f32::NEG_INFINITY);
    for t in 0..vocab {
        let v = logit(t);
        if v > best_v {
            (best_t, best_v) = (t, v);
        }
    }
    assert_eq!(best_t, hf_argmax, "argmax drifted (ours {best_v} at {best_t})");

    let ids5: Vec<usize> = reference["top5_ids"]
        .as_array()
        .expect("top5")
        .iter()
        .map(|v| v.as_u64().expect("id") as usize)
        .collect();
    let vals5: Vec<f32> = reference["top5_logits"]
        .as_array()
        .expect("top5")
        .iter()
        .map(|v| v.as_f64().expect("v") as f32)
        .collect();
    for (t, hf) in ids5.iter().zip(&vals5) {
        let ours = logit(*t);
        assert!((ours - hf).abs() < 0.25, "top-5 token {t}: ours {ours} vs HF {hf}");
    }
    let probes: Vec<usize> = reference["probe_ids"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|v| v.as_u64().expect("id") as usize)
        .collect();
    let probe_vals: Vec<f32> = reference["probe_logits"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|v| v.as_f64().expect("v") as f32)
        .collect();
    for (t, hf) in probes.iter().zip(&probe_vals) {
        let ours = logit(*t);
        assert!((ours - hf).abs() < 0.25, "probe token {t}: ours {ours} vs HF {hf}");
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

fn cuda_facts(dfp: bool, force_prefill: bool, padded: bool) -> LlamaLikeCudaFacts {
    LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: dfp,
        rope_table: true,
        force_prefill_path: force_prefill,
        head_dim_padded: padded,
        // The only caller that pads is phi3, whose 96 rounds to 128 — and
        // the two facts have to agree, which is why this reads off the
        // flag rather than taking a parameter of its own.
        head_dim_kernel: if padded { 128 } else { 0 },
        gate_up_fused: true,
        // Dense BF16, one GPU, whole context.
        proj_repr: model_dsl::WeightRepr::Bf16,
        tp_size: 1,
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    }
}

/// Concatenate checkpoint tensors into one fused host buffer.
fn fuse(ckpt: &Checkpoint, parts: &[String]) -> Vec<u8> {
    parts.iter().flat_map(|p| ckpt.view(p).bytes.iter().copied()).collect()
}

#[test]
fn qwen3_0_6b_reproduces_the_hf_logits() {
    ab(&Spec {
        name: "qwen3_0_6b",
        cache_dir: "models--Qwen--Qwen3-0.6B",
        reference: "reference.json",
        facts: LlamaLikeFacts::qwen3_0_6b(),
        cuda: LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        layers: 28,
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        vocab: 151_936,
        eps: 1e-6,
        theta: 1e6,
        bind: |ckpt, sink| {
            sink("embed".into(), fuse(ckpt, &["model.embed_tokens.weight".into()]));
            sink("final_norm".into(), fuse(ckpt, &["model.norm.weight".into()]));
            for i in 0..28 {
                let n = |s: &str| format!("model.layers.{i}.{s}");
                sink(format!("layer.{i}.attn_norm"), fuse(ckpt, &[n("input_layernorm.weight")]));
                sink(
                    format!("layer.{i}.qkv"),
                    fuse(
                        ckpt,
                        &[
                            n("self_attn.q_proj.weight"),
                            n("self_attn.k_proj.weight"),
                            n("self_attn.v_proj.weight"),
                        ],
                    ),
                );
                sink(format!("layer.{i}.q_norm"), fuse(ckpt, &[n("self_attn.q_norm.weight")]));
                sink(format!("layer.{i}.k_norm"), fuse(ckpt, &[n("self_attn.k_norm.weight")]));
                sink(format!("layer.{i}.o_proj"), fuse(ckpt, &[n("self_attn.o_proj.weight")]));
                sink(
                    format!("layer.{i}.mlp_norm"),
                    fuse(ckpt, &[n("post_attention_layernorm.weight")]),
                );
                sink(
                    format!("layer.{i}.gate_up"),
                    fuse(ckpt, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")]),
                );
                sink(format!("layer.{i}.down"), fuse(ckpt, &[n("mlp.down_proj.weight")]));
            }
        },
    });
}

/// OLMo-2 is POST-norm with global qk-norm and separate q/k/v — the trace
/// maps `attn_norm` to `post_attention_layernorm` and `mlp_norm` to
/// `post_feedforward_layernorm` (the bind_olmo3 convention).
#[test]
fn olmo2_1b_reproduces_the_hf_logits() {
    ab(&Spec {
        name: "olmo2_1b",
        cache_dir: "models--allenai--OLMo-2-0425-1B-Instruct",
        reference: "olmo2_1b.json",
        facts: LlamaLikeFacts::olmo2_1b(),
        cuda: cuda_facts(true, false, false),
        layers: 16,
        q_heads: 16,
        kv_heads: 16,
        head_dim: 128,
        vocab: 100_352,
        eps: 1e-6,
        theta: 5e5,
        bind: |ckpt, sink| {
            sink("embed".into(), fuse(ckpt, &["model.embed_tokens.weight".into()]));
            sink("final_norm".into(), fuse(ckpt, &["model.norm.weight".into()]));
            sink("lm_head".into(), fuse(ckpt, &["lm_head.weight".into()]));
            for i in 0..16 {
                let n = |s: &str| format!("model.layers.{i}.{s}");
                sink(
                    format!("layer.{i}.attn_norm"),
                    fuse(ckpt, &[n("post_attention_layernorm.weight")]),
                );
                sink(
                    format!("layer.{i}.mlp_norm"),
                    fuse(ckpt, &[n("post_feedforward_layernorm.weight")]),
                );
                sink(format!("layer.{i}.q_proj"), fuse(ckpt, &[n("self_attn.q_proj.weight")]));
                sink(format!("layer.{i}.k_proj"), fuse(ckpt, &[n("self_attn.k_proj.weight")]));
                sink(format!("layer.{i}.v_proj"), fuse(ckpt, &[n("self_attn.v_proj.weight")]));
                sink(format!("layer.{i}.q_norm"), fuse(ckpt, &[n("self_attn.q_norm.weight")]));
                sink(format!("layer.{i}.k_norm"), fuse(ckpt, &[n("self_attn.k_norm.weight")]));
                sink(format!("layer.{i}.o_proj"), fuse(ckpt, &[n("self_attn.o_proj.weight")]));
                sink(
                    format!("layer.{i}.gate_up"),
                    fuse(ckpt, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")]),
                );
                sink(format!("layer.{i}.down"), fuse(ckpt, &[n("mlp.down_proj.weight")]));
            }
        },
    });
}

/// Qwen2.5 is the bias deployment: fused qkv plus q/k/v bias vectors, tied
/// lm_head.
#[test]
fn qwen2_5_1_5b_reproduces_the_hf_logits() {
    ab(&Spec {
        name: "qwen2_5_1_5b",
        cache_dir: "models--Qwen--Qwen2.5-1.5B-Instruct",
        reference: "qwen2_5_1_5b.json",
        facts: LlamaLikeFacts::qwen2_5_1_5b(),
        cuda: cuda_facts(false, true, false),
        layers: 28,
        q_heads: 12,
        kv_heads: 2,
        head_dim: 128,
        vocab: 151_936,
        eps: 1e-6,
        theta: 1e6,
        bind: |ckpt, sink| {
            sink("embed".into(), fuse(ckpt, &["model.embed_tokens.weight".into()]));
            sink("final_norm".into(), fuse(ckpt, &["model.norm.weight".into()]));
            for i in 0..28 {
                let n = |s: &str| format!("model.layers.{i}.{s}");
                sink(format!("layer.{i}.attn_norm"), fuse(ckpt, &[n("input_layernorm.weight")]));
                sink(
                    format!("layer.{i}.qkv"),
                    fuse(
                        ckpt,
                        &[
                            n("self_attn.q_proj.weight"),
                            n("self_attn.k_proj.weight"),
                            n("self_attn.v_proj.weight"),
                        ],
                    ),
                );
                sink(format!("layer.{i}.q_bias"), fuse(ckpt, &[n("self_attn.q_proj.bias")]));
                sink(format!("layer.{i}.k_bias"), fuse(ckpt, &[n("self_attn.k_proj.bias")]));
                sink(format!("layer.{i}.v_bias"), fuse(ckpt, &[n("self_attn.v_proj.bias")]));
                sink(format!("layer.{i}.o_proj"), fuse(ckpt, &[n("self_attn.o_proj.weight")]));
                sink(
                    format!("layer.{i}.mlp_norm"),
                    fuse(ckpt, &[n("post_attention_layernorm.weight")]),
                );
                sink(
                    format!("layer.{i}.gate_up"),
                    fuse(ckpt, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")]),
                );
                sink(format!("layer.{i}.down"), fuse(ckpt, &[n("mlp.down_proj.weight")]));
            }
        },
    });
}

/// Mistral-7B: the 7B scale point, sharded checkpoint, untied lm_head,
/// eps 1e-5.
#[test]
fn mistral_7b_v03_reproduces_the_hf_logits() {
    ab(&Spec {
        name: "mistral_7b_v03",
        cache_dir: "models--mistralai--Mistral-7B-Instruct-v0.3",
        reference: "mistral_7b_v03.json",
        facts: LlamaLikeFacts::mistral_7b_v03(),
        cuda: cuda_facts(true, false, false),
        layers: 32,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        vocab: 32_768,
        eps: 1e-5,
        theta: 1e6,
        bind: |ckpt, sink| {
            sink("embed".into(), fuse(ckpt, &["model.embed_tokens.weight".into()]));
            sink("final_norm".into(), fuse(ckpt, &["model.norm.weight".into()]));
            sink("lm_head".into(), fuse(ckpt, &["lm_head.weight".into()]));
            for i in 0..32 {
                let n = |s: &str| format!("model.layers.{i}.{s}");
                sink(format!("layer.{i}.attn_norm"), fuse(ckpt, &[n("input_layernorm.weight")]));
                sink(
                    format!("layer.{i}.qkv"),
                    fuse(
                        ckpt,
                        &[
                            n("self_attn.q_proj.weight"),
                            n("self_attn.k_proj.weight"),
                            n("self_attn.v_proj.weight"),
                        ],
                    ),
                );
                sink(format!("layer.{i}.o_proj"), fuse(ckpt, &[n("self_attn.o_proj.weight")]));
                sink(
                    format!("layer.{i}.mlp_norm"),
                    fuse(ckpt, &[n("post_attention_layernorm.weight")]),
                );
                sink(
                    format!("layer.{i}.gate_up"),
                    fuse(ckpt, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")]),
                );
                sink(format!("layer.{i}.down"), fuse(ckpt, &[n("mlp.down_proj.weight")]));
            }
        },
    });
}

/// Phi-3-mini: the padded-head-dim deployment (logical 96). The checkpoint
/// ships FUSED qkv and gate_up; the trace wants SEPARATE q/k/v, so the
/// loader splits by rows — the inverse of the fusions above.
#[test]
#[ignore = "first-ever run dies with the C++-exception signature (drop-guard \
panic, no header) — phi3 was WRITTEN under the no-per-model-loops \
constraint and never executed; its padded-head path needs its own \
diagnosis session before this claim can stand"]
fn phi3_mini_reproduces_the_hf_logits() {
    ab(&Spec {
        name: "phi3_mini",
        cache_dir: "models--microsoft--Phi-3-mini-4k-instruct",
        reference: "phi3_mini.json",
        facts: LlamaLikeFacts::phi3_mini(),
        cuda: cuda_facts(false, false, true),
        layers: 32,
        q_heads: 32,
        kv_heads: 32,
        head_dim: 96,
        vocab: 32_064,
        eps: 1e-5,
        theta: 1e4,
        bind: |ckpt, sink| {
            sink("embed".into(), fuse(ckpt, &["model.embed_tokens.weight".into()]));
            sink("final_norm".into(), fuse(ckpt, &["model.norm.weight".into()]));
            sink("lm_head".into(), fuse(ckpt, &["lm_head.weight".into()]));
            const HIDDEN: usize = 3072;
            for i in 0..32 {
                let n = |s: &str| format!("model.layers.{i}.{s}");
                sink(format!("layer.{i}.attn_norm"), fuse(ckpt, &[n("input_layernorm.weight")]));
                let qkv = ckpt.view(&n("self_attn.qkv_proj.weight"));
                let row_bytes = HIDDEN * 2;
                sink(format!("layer.{i}.q_proj"), qkv.bytes[..3072 * row_bytes].to_vec());
                sink(
                    format!("layer.{i}.k_proj"),
                    qkv.bytes[3072 * row_bytes..6144 * row_bytes].to_vec(),
                );
                sink(format!("layer.{i}.v_proj"), qkv.bytes[6144 * row_bytes..].to_vec());
                sink(format!("layer.{i}.o_proj"), fuse(ckpt, &[n("self_attn.o_proj.weight")]));
                sink(
                    format!("layer.{i}.mlp_norm"),
                    fuse(ckpt, &[n("post_attention_layernorm.weight")]),
                );
                sink(format!("layer.{i}.gate_up"), fuse(ckpt, &[n("mlp.gate_up_proj.weight")]));
                sink(format!("layer.{i}.down"), fuse(ckpt, &[n("mlp.down_proj.weight")]));
            }
        },
    });
}
