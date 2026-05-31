//! `pie-driver-cuda-native` — the fat Rust control plane for the
//! driver/cuda_new rewrite.
//!
//! It owns every orchestration *decision* (construction, memory planning,
//! per-fire dispatch, sampling policy, spec-decode scheduling, TP
//! broadcast) and calls *down* into `libpie_cuda_device` (the C++/CUDA
//! thin device library) for every kernel *sequence*. See `../../PLAN.md`.
//!
//! C-ABI shape mirrors driver/dummy and driver/cuda: a `*_run_inproc`
//! entry handed a `pie-bridge` vtable, plus a `*_request_stop`. During the
//! migration this crate exports `pie_driver_cuda_native_*` so it coexists
//! with the live `pie_driver_cuda_*` driver behind a Cargo feature; at
//! cutover it takes over the canonical `pie_driver_cuda_*` names.

// Scaffolding: modules are stubbed and wired progressively (phases 1-4 in
// PLAN.md), so most items are not yet reachable from the C-ABI entry.
// Remove this once the executor path is live so real dead code surfaces.
#![allow(dead_code)]

mod arch;
mod builder;
mod device;
mod executor;
mod ffi;
mod loader;
mod mem;
mod sampler;
mod spec;
mod tp;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use pie_bridge::ffi::{FfiRequestSink, FfiResponseSource, InProcVTable};
use pie_bridge::{
    __pie_frame_from_desc, Frame, ForwardResponse, PieForwardResponseDesc, PieResponseFrameDesc,
    PieResponsePayloadDesc, PieStatusResponseDesc, RequestPayload, PIE_RESPONSE_PAYLOAD_FORWARD,
    PIE_RESPONSE_PAYLOAD_STATUS,
};

use crate::builder::{BootConfig, Model};
use crate::device::Device;
use crate::mem::Profile;

pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

/// Cooperative shutdown flag, flipped by `request_stop`. The serve loop also
/// exits when the vtable's `recv` returns a non-zero (shutdown) code.
static STOP: AtomicBool = AtomicBool::new(false);

/// KV page size (tokens/page) — must match the runtime's page allocator.
const KV_PAGE_SIZE: usize = 16;

// ── startup config (the `[model]` / `[batching]` TOML the runtime writes) ──
#[derive(serde::Deserialize)]
struct StartupConfig {
    model: ModelSection,
    #[serde(default)]
    batching: BatchingSection,
}
#[derive(serde::Deserialize)]
struct ModelSection {
    snapshot_dir: String,
    #[serde(default)]
    device: String, // "cuda:N"
}
#[derive(serde::Deserialize, Default)]
struct BatchingSection {
    #[serde(default)]
    gpu_mem_utilization: f64,
    #[serde(default)]
    num_kv_pages: usize,
    /// Host KV page slots for swap-out/in (Copy D2H/H2D/H2H). `0` disables.
    #[serde(default)]
    swap_pool_size: usize,
}

/// In-process driver entry. NO IPC on this path — the runtime hands a function-
/// pointer [`InProcVTable`] (`recv` → next request `PieFrameDesc`, `send_response`
/// → reply `PieResponseFrameDesc`); we serve fires by direct call. Steps:
///   1. parse `--config <path>` from argv → `[model]/[batching]` TOML
///   2. `builder::build` → resident `Model` (weights + caches on the GPU)
///   3. emit the READY capability JSON via `ready_cb`
///   4. serve loop: recv → `Model::serve_forward` → send_response
///
/// # Safety
/// `argv` must point to `argc` valid C strings; `ready_cb` must be valid;
/// `vtable`'s fn pointers + ctx must satisfy the `InProcVTable` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_cuda_native_run_inproc(
    argc: c_int,
    argv: *mut *mut c_char,
    _install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
    vtable: InProcVTable,
) -> c_int {
    STOP.store(false, Ordering::SeqCst);
    match unsafe { run_serve(argc, argv, ready_cb, ready_ctx, &vtable) } {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[pie-driver-cuda-native] fatal: {e:#}");
            -1
        }
    }
}

unsafe fn run_serve(
    argc: c_int, argv: *mut *mut c_char, ready_cb: ReadyCb, ready_ctx: *mut c_void,
    vtable: &InProcVTable,
) -> Result<()> {
    let cfg_path = unsafe { parse_config_path(argc, argv) }.context("missing --config <path>")?;
    let text = std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path}"))?;
    let sc: StartupConfig = toml::from_str(&text).context("parsing startup TOML")?;

    let ordinal = sc.model.device.rsplit(':').next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let dev = Device::new(ordinal).map_err(|e| anyhow::anyhow!("Device::new({ordinal}): {e}"))?;
    let util = if sc.batching.gpu_mem_utilization > 0.0 { sc.batching.gpu_mem_utilization } else { 0.9 };
    let num_kv_pages = if sc.batching.num_kv_pages > 0 { sc.batching.num_kv_pages } else { 1024 };
    let boot = BootConfig {
        device_ordinal: ordinal,
        memory_profile: Profile::Auto,
        gpu_mem_utilization: util,
        page_size: KV_PAGE_SIZE,
        num_kv_pages,
        swap_pool_size: sc.batching.swap_pool_size,
    };
    let model = builder::build(&dev, Path::new(&sc.model.snapshot_dir), &boot)?;

    // READY handshake — capability JSON the runtime parses into DriverCapabilities.
    let (total_pages, vocab, max_model_len) = model.caps();
    let caps = serde_json::json!({
        "total_pages": total_pages,
        "kv_page_size": KV_PAGE_SIZE,
        "swap_pool_size": model.swap_pool_size() as u32,
        "max_forward_tokens": total_pages * KV_PAGE_SIZE,
        "max_forward_requests": 256u32,
        "max_page_refs": total_pages,
        "max_logit_rows": u32::MAX,
        "max_prob_rows": u32::MAX,
        "max_custom_mask_bytes": u32::MAX,
        "max_sampler_rows": u32::MAX,
        "max_logprob_labels": u32::MAX,
        "arch_name": format!("{:?}", model.spec.id),
        "vocab_size": vocab,
        "max_model_len": max_model_len,
        "activation_dtype": "bfloat16",
        "snapshot_dir": sc.model.snapshot_dir,
    })
    .to_string();
    let caps_c = CString::new(caps).context("caps JSON has NUL")?;
    unsafe { ready_cb(caps_c.as_ptr(), ready_ctx) };

    serve_loop(&dev, &model, vtable);
    Ok(())
}

/// The fire loop: pull requests off the vtable, dispatch, reply. Exits on
/// `STOP` or a non-zero `recv` (shutdown).
fn serve_loop(dev: &Device, model: &Model, vtable: &InProcVTable) {
    let sink = FfiRequestSink::new(vtable);
    let src = FfiResponseSource::new(vtable);
    while !STOP.load(Ordering::Relaxed) {
        let Some(req) = (unsafe { sink.recv() }) else { break };
        // No rkyv on the in-process path: the descriptor is converted straight
        // to a native owned `Frame`.
        let frame: Frame = __pie_frame_from_desc(req.request);
        let driver_id = frame.driver_id;
        match frame.payload {
            RequestPayload::Forward(fr) => match model.serve_forward(dev, &fr) {
                Ok(resp) => send_forward(&src, req.req_id, driver_id, &resp),
                Err(e) => {
                    eprintln!("[pie-driver-cuda-native] forward failed: {e:#}");
                    send_status(&src, req.req_id, driver_id, -1);
                }
            },
            RequestPayload::Copy(cp) => match model.copy_pages(dev, &cp) {
                Ok(()) => {
                    if std::env::var_os("PIE_CUDA_NEW_TRACE").is_some() {
                        eprintln!(
                            "[pie-driver-cuda-native] copy ok: {:?} {:?} {} page(s)",
                            cp.resource, cp.dir, cp.srcs.len()
                        );
                    }
                    send_status(&src, req.req_id, driver_id, 0)
                }
                Err(e) => {
                    eprintln!("[pie-driver-cuda-native] copy failed: {e:#}");
                    send_status(&src, req.req_id, driver_id, -1);
                }
            },
            RequestPayload::Health => send_status(&src, req.req_id, driver_id, 0),
            // Adapter (LoRA) ops are accepted as no-ops returning success —
            // exact parity with driver/cuda, which stubs Load/Save/ZoInit/
            // ZoUpdate to status 0 and never consumes adapter_bindings in the
            // forward. (load_adapter awaits this status; -1 would surface as an
            // error to the inferlet.)
            RequestPayload::Adapter(_) => send_status(&src, req.req_id, driver_id, 0),
            #[allow(unreachable_patterns)]
            _ => send_status(&src, req.req_id, driver_id, -1),
        }
    }
}

/// Build + post a Forward response descriptor pointing at `resp`'s buffers.
/// `resp` must outlive this call (the vtable consumes the descriptor + its
/// slice pointers synchronously, per the contract).
fn send_forward(src: &FfiResponseSource, req_id: u32, driver_id: u32, resp: &ForwardResponse) {
    let fwd = PieForwardResponseDesc {
        num_requests: resp.num_requests,
        tokens_indptr_ptr: resp.tokens_indptr.as_ptr(),
        tokens_indptr_len: resp.tokens_indptr.len(),
        tokens_ptr: resp.tokens.as_ptr(),
        tokens_len: resp.tokens.len(),
        ..Default::default()
    };
    let rf = PieResponseFrameDesc {
        driver_id,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_FORWARD,
            forward: fwd,
            status: PieStatusResponseDesc::default(),
        },
    };
    src.send(req_id, &rf);
}

fn send_status(src: &FfiResponseSource, req_id: u32, driver_id: u32, status: i32) {
    let rf = PieResponseFrameDesc {
        driver_id,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_STATUS,
            forward: PieForwardResponseDesc::default(),
            status: PieStatusResponseDesc { status },
        },
    };
    src.send(req_id, &rf);
}

/// Scan argv for `--config <path>` (or `--config=<path>`).
unsafe fn parse_config_path(argc: c_int, argv: *mut *mut c_char) -> Option<String> {
    let mut args: Vec<String> = Vec::new();
    for i in 0..argc as isize {
        let p = unsafe { *argv.offset(i) };
        if !p.is_null() {
            if let Ok(s) = unsafe { CStr::from_ptr(p) }.to_str() {
                args.push(s.to_string());
            }
        }
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--config" {
            return it.next().cloned();
        }
        if let Some(v) = a.strip_prefix("--config=") {
            return Some(v.to_string());
        }
    }
    None
}

/// Signal the serve loop to stop. Idempotent; safe from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_cuda_native_request_stop() {
    STOP.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::f32_to_bf16;
    use pie_bridge::{
        PieForwardRequestDesc, PieFrameDesc, PieRequestPayloadDesc, ResponsePayload,
        __pie_response_frame_from_desc, PIE_REQUEST_PAYLOAD_FORWARD,
    };
    use serde_json::json;
    use std::cell::{Cell, RefCell};

    fn syn(seed: f32, n: usize, s: f32) -> Vec<u16> {
        (0..n).map(|i| f32_to_bf16(((i as f32 + seed) * 0.1).sin() * s)).collect()
    }

    fn write_tiny_llama(dir: &Path, vocab: usize, hidden: usize, nq: usize, nkv: usize, hd: usize, inter: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = json!({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": 1,
            "num_attention_heads": nq, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": vocab, "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let (hq, hkv) = (nq * hd, nkv * hd);
        let ts: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], syn(3.0, vocab * hidden, 0.1)),
            ("model.layers.0.input_layernorm.weight".into(), vec![hidden], syn(4.0, hidden, 1.0)),
            ("model.layers.0.post_attention_layernorm.weight".into(), vec![hidden], syn(5.0, hidden, 1.0)),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![hq, hidden], syn(6.0, hq * hidden, 0.1)),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![hkv, hidden], syn(7.0, hkv * hidden, 0.1)),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![hkv, hidden], syn(8.0, hkv * hidden, 0.1)),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![hidden, hq], syn(9.0, hidden * hq, 0.1)),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![inter, hidden], syn(10.0, inter * hidden, 0.1)),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![inter, hidden], syn(11.0, inter * hidden, 0.1)),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![hidden, inter], syn(12.0, hidden * inter, 0.1)),
        ];
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (n, shape, vals) in &ts {
            let begin = data.len();
            for &v in vals { data.extend_from_slice(&v.to_le_bytes()); }
            header.insert(n.clone(), json!({"dtype":"BF16","shape":shape,"data_offsets":[begin,data.len()]}));
        }
        let hj = serde_json::to_vec(&header).unwrap();
        let mut out = (hj.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hj);
        out.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    /// End-to-end: drive the full `run_inproc` serve loop through a mock
    /// `InProcVTable` (one Forward request → captured response) and check the
    /// reply equals greedy prefill. Proves the no-IPC vtable path works.
    #[test]
    fn run_inproc_serve_loop_forward() {
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip run_inproc_serve_loop_forward (no device): {e:#}"); return; }
        };
        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skip run_inproc_serve_loop_forward (<3 GiB free)");
            return;
        }
        let (vocab, hidden, nq, nkv, hd, inter) = (32usize, 16, 2, 1, 8, 32);
        let dir = std::env::temp_dir().join(format!("cuda_new_inproc_{}", std::process::id()));
        write_tiny_llama(&dir, vocab, hidden, nq, nkv, hd, inter);
        let util = ((((total - free) as f64) + 2.0e9) / total as f64).min(0.99);
        let boot = BootConfig {
            device_ordinal: 0, memory_profile: Profile::Auto, gpu_mem_utilization: util,
            page_size: 16, num_kv_pages: 4, swap_pool_size: 0,
        };
        let model = builder::build(&dev, &dir, &boot).unwrap();

        // Request backing arrays (must outlive serve_loop — raw ptrs alias them).
        let token_ids: Vec<u32> = vec![1, 3, 2];
        let position_ids: Vec<u32> = vec![0, 1, 2];
        let qo_indptr: Vec<u32> = vec![0, 3];
        let kv_page_indices: Vec<u32> = vec![0];
        let kv_page_indptr: Vec<u32> = vec![0, 1];
        let kv_last_page_lens: Vec<u32> = vec![3];
        let sampling_indices: Vec<u32> = vec![2];
        let sampling_indptr: Vec<u32> = vec![0, 1];
        let fwd = PieForwardRequestDesc {
            token_ids_ptr: token_ids.as_ptr(), token_ids_len: token_ids.len(),
            position_ids_ptr: position_ids.as_ptr(), position_ids_len: position_ids.len(),
            qo_indptr_ptr: qo_indptr.as_ptr(), qo_indptr_len: qo_indptr.len(),
            kv_page_indices_ptr: kv_page_indices.as_ptr(), kv_page_indices_len: kv_page_indices.len(),
            kv_page_indptr_ptr: kv_page_indptr.as_ptr(), kv_page_indptr_len: kv_page_indptr.len(),
            kv_last_page_lens_ptr: kv_last_page_lens.as_ptr(), kv_last_page_lens_len: kv_last_page_lens.len(),
            sampling_indices_ptr: sampling_indices.as_ptr(), sampling_indices_len: sampling_indices.len(),
            sampling_indptr_ptr: sampling_indptr.as_ptr(), sampling_indptr_len: sampling_indptr.len(),
            ..Default::default()
        };
        let frame = PieFrameDesc {
            driver_id: 7,
            payload: PieRequestPayloadDesc {
                kind: PIE_REQUEST_PAYLOAD_FORWARD, forward: fwd, ..Default::default()
            },
        };

        struct Ctx {
            req: *const PieFrameDesc,
            delivered: Cell<bool>,
            captured: RefCell<Option<Vec<u32>>>,
        }
        let ctx = Ctx { req: &frame, delivered: Cell::new(false), captured: RefCell::new(None) };

        unsafe extern "C" fn recv(ctx: *mut c_void, out: *mut *const PieFrameDesc, id: *mut u32) -> c_int {
            let c = unsafe { &*(ctx as *const Ctx) };
            if c.delivered.get() {
                return 1; // shutdown after the one request
            }
            unsafe { *out = c.req; *id = 7; }
            c.delivered.set(true);
            0
        }
        unsafe extern "C" fn send(ctx: *mut c_void, _id: u32, resp: *const PieResponseFrameDesc) {
            let c = unsafe { &*(ctx as *const Ctx) };
            let rf = __pie_response_frame_from_desc(unsafe { &*resp });
            if let ResponsePayload::Forward(f) = rf.payload {
                *c.captured.borrow_mut() = Some(f.tokens);
            }
        }
        let vt = InProcVTable { recv, send_response: send, ctx: &ctx as *const Ctx as *mut c_void };

        STOP.store(false, Ordering::SeqCst);
        serve_loop(&dev, &model, &vt);

        let toks = ctx.captured.borrow().clone().expect("serve_loop produced no response");
        assert_eq!(toks.len(), 1, "one sampled token for the single request");
        assert!((toks[0] as usize) < vocab, "token in range");
        let greedy = *model.prefill_greedy(&dev, &[1, 3, 2]).unwrap().last().unwrap();
        assert_eq!(toks[0] as i32, greedy, "serve_loop token != greedy prefill");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
