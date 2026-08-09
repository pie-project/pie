//! The thirteen `pie_cuda_*` exports — the cutover's door (retirement
//! plan phase D).
//!
//! The engine consumes a driver through `pie_driver_abi.h`, whose Rust
//! source of truth is `driver_abi::local`. This module DEFINES the symbols
//! that crate declares; a test resolving the declaration against these
//! definitions makes the linker prove the contract, the same way the
//! launch bridge's shim makes the C++ compiler prove the rows.
//!
//! **One provider per binary.** The C++ shell exports the same thirteen
//! names, so the `abi` feature must never be enabled in a build that also
//! links `driver-cuda` — same symbols, duplicate-definition link error,
//! by design rather than by accident.
//!
//! What is real and what refuses, today: `create`/`destroy` manage the
//! shell's state and return honest capabilities; `close_*` succeed on the
//! nothing they have to close; everything else returns
//! `PIE_STATUS_UNSUPPORTED` with the awaited machinery named in its doc —
//! the stated-refusal pattern, so the remaining distance to cutover is
//! enumerable as exactly the functions still refusing.

// Error paths print to stderr with the C++ shell's own prefix — that IS
// the behaviour being replaced (`abi.cpp` writes `[pie-driver-cuda]` to
// cerr), and an ABI boundary has no tracing subscriber to rely on.
#![allow(clippy::print_stderr)]

// Every export takes raw pointers by C-ABI necessity and null-checks them
// before the deref — the same defensive shape the C++ shell has. The
// caller-side contract is `driver_abi::local`'s `unsafe extern` block;
// marking the DEFINITIONS `unsafe fn` would change their ABI type for a
// fact the boundary already states.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use driver_abi::local::{
    PIE_DRIVER_ABI_VERSION, PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_OK,
    PIE_STATUS_UNSUPPORTED, PieChannelDesc, PieChannelEndpointBinding, PieCompletion,
    PieDriver, PieDriverCaps, PieDriverCreateDesc, PieEncodeDesc, PieFrameDesc,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieModelLoadDesc,
    PiePoolResizeDesc, PieProgramDesc, PieStateCopyDesc,
};

/// The shell's state — what `PieDriver` points at.
struct Shell {
    /// The capabilities JSON `create` hands back; owned here so the
    /// pointer in [`PieDriverCaps`] lives as long as the driver.
    caps: Vec<u8>,
    /// `[model] descriptor` from the boot TOML, for HF snapshots whose
    /// descriptor does not ride inside the checkpoint.
    boot_descriptor: Option<std::path::PathBuf>,
    /// The loaded model, once `load_model` succeeds.
    model: Option<LoadedModel>,
    /// Registered programs by id — the C3 hash is the dedup key, so
    /// re-registering a program answers the id it already has.
    programs: std::collections::BTreeMap<u64, ProgramEntry>,
    /// Bound instances by id.
    instances: std::collections::BTreeMap<u64, InstanceEntry>,
    /// The next never-used id (programs and instances share the counter —
    /// simpler, and nothing in the ABI wants them dense).
    next_id: u64,
    /// The runtime's notify callback + its context, from `create`.
    notify: driver_abi::local::PieRuntimeNotifyFn,
    notify_ctx: *mut std::ffi::c_void,
    /// The hybrid's GDN state slabs, allocated on first hybrid launch.
    gdn: Option<GdnState>,
    /// The driver-owned KV pools, allocated on first launch and grown on
    /// demand — decode continuity across launches lives here.
    kv: Option<KvState>,
    /// Registered channels: the pinned host ring endpoints the engine
    /// maps. Device-side rings and fire delivery ride with the launch
    /// integration.
    channels: std::collections::BTreeMap<u64, ChannelState>,
    /// The host-pinned KV swap pool: page-granular, per layer, both
    /// planes — where `copy_kv`'s host-pinned domain lands. Grown on
    /// demand by highest page id touched.
    swap: Option<SwapPool>,
    /// The fire scratch held PER DRIVER, as the C++ holds it: the
    /// attention workspace and both FlashInfer plan caches. Created on
    /// first launch. This is also what the 711-fire soak enforced: the
    /// per-fire version leaked its 48 MB workspace every fire.
    scratch: Option<FireScratch>,
}

/// Driver-lifetime fire scratch.
struct FireScratch {
    ws: crate::model::attention_workspace::AttentionWorkspace<
        cudarc::runtime::sys::cudaEvent_t,
    >,
    decode_plan: crate::model::executor::DecodePlan,
    /// gemma-4's SECOND decode plan — the FULL layers' 512-wide
    /// geometry; single-kind families never plan it.
    decode_plan_full: crate::model::executor::DecodePlan,
    prefill_plan: crate::model::executor::PrefillPlan,
}

/// The pinned swap pool: `layers × [pages × page_bytes]` per plane.
struct SwapPool {
    /// One pinned block per layer: `[k_pages | v_pages]` back to back.
    blocks: Vec<*mut std::ffi::c_void>,
    num_pages: u32,
    page_bytes: usize,
}

impl SwapPool {
    fn free(&self) {
        use crate::model::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        for &b in &self.blocks {
            ops.free_host(b);
        }
    }
    /// The host address of `(layer, plane, page)` — plane 0 is K.
    fn page(&self, layer: usize, plane: usize, page: u32) -> *mut u8 {
        let off = (plane * self.num_pages as usize + page as usize) * self.page_bytes;
        unsafe { self.blocks[layer].cast::<u8>().add(off) }
    }
}

/// One channel's host endpoint: the pinned mirror and the four control
/// words, exactly the C++ registry's binding contract.
struct ChannelState {
    mirror: *mut std::ffi::c_void,
    words: *mut std::ffi::c_void,
    mirror_bytes: usize,
    cell_bytes: usize,
    /// `capacity + 1` — the ring modulus.
    ring: u32,
    host_role: u8,
}

impl ChannelState {
    /// Publish one wire cell: write it at `tail % ring`, then advance the
    /// tail word with release ordering — the reader (the engine) consumes
    /// from the head. The writer side of the C++ ring, host-resident.
    fn publish(&self, cell: &[u8]) -> bool {
        if cell.len() != self.cell_bytes {
            return false;
        }
        let words = self.words.cast::<u64>();
        let head = unsafe { words.add(0).read_volatile() };
        let tail = unsafe { words.add(1).read_volatile() };
        if tail.wrapping_sub(head) >= u64::from(self.ring) {
            return false; // ring full; the engine has not consumed
        }
        let slot = (tail % u64::from(self.ring)) as usize;
        let dst = unsafe { self.mirror.cast::<u8>().add(slot * self.cell_bytes) };
        debug_assert!((slot + 1) * self.cell_bytes <= self.mirror_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(cell.as_ptr(), dst, cell.len());
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { words.add(1).write_volatile(tail + 1) };
        true
    }
}

impl ChannelState {
    fn free(&self) {
        use crate::model::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        ops.free_host(self.mirror);
        ops.free_host(self.words);
    }
}

/// The shell's KV: one (k, v) pool per layer, plus the capacity in
/// pages. A `None` row is a layer that owns no pages — gemma-4's
/// KV-shared trailing layers, whose views ride their source's pool.
struct KvState {
    pools: Vec<Option<(crate::cuda::DeviceBuffer, crate::cuda::DeviceBuffer)>>,
    num_pages: u32,
}

/// The hybrid's driver-owned GDN state: one (conv, recurrent) slab pair
/// per LINEAR model layer, slot-indirected — `RecurrentStateCache`'s
/// role, shell-resident. Slot ids are the ENGINE's (`rs_slot_ids` on the
/// step, `PieStateCopyRange` on state copies); the shell only stores.
struct GdnState {
    /// Indexed by MODEL layer; `None` at full-attention layers.
    slabs: Vec<Option<(crate::cuda::DeviceBuffer, crate::cuda::DeviceBuffer)>>,
    num_slots: u32,
    conv_stride_elems: i64,
    state_stride_elems: i64,
    /// Bytes per element of the recurrent store (2 = bf16 state).
    state_elem_bytes: usize,
}

impl GdnState {
    /// Grow every slab to cover `need` slots, MIGRATING the surviving
    /// slots — the same contract the KV resize keeps.
    fn ensure_slots(
        &mut self,
        need: u32,
        alloc: &crate::cuda::Allocator,
        stream: &crate::cuda::OwnedStream,
    ) -> Result<(), i32> {
        if self.num_slots >= need {
            return Ok(());
        }
        let conv_bytes_old = self.num_slots as usize * self.conv_stride_elems as usize * 2;
        let state_bytes_old =
            self.num_slots as usize * self.state_stride_elems as usize * self.state_elem_bytes;
        for slab in self.slabs.iter_mut().flatten() {
            let mut c = alloc
                .alloc(need as usize * self.conv_stride_elems as usize * 2)
                .map_err(|_| PIE_STATUS_EXHAUSTED)?;
            let mut r = alloc
                .alloc(need as usize * self.state_stride_elems as usize * self.state_elem_bytes)
                .map_err(|_| PIE_STATUS_EXHAUSTED)?;
            c.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            r.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            let d2d = |dst: &crate::cuda::DeviceBuffer,
                       src: &crate::cuda::DeviceBuffer,
                       bytes: usize|
             -> Result<(), i32> {
                use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
                let code = unsafe {
                    cudaMemcpyAsync(
                        dst.as_ptr(),
                        src.as_ptr().cast_const(),
                        bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream.as_ref().as_raw().cast(),
                    )
                };
                (code == cudaError::cudaSuccess).then_some(()).ok_or(PIE_STATUS_DRIVER_ERROR)
            };
            d2d(&c, &slab.0, conv_bytes_old)?;
            d2d(&r, &slab.1, state_bytes_old)?;
            *slab = (c, r);
        }
        stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        self.num_slots = need;
        Ok(())
    }
}

/// What registration keeps of a program today: the identity the engine
/// dedups on. The launch package itself is deep-copied when the `launch`
/// arm lands — it is the caller's transient memory, and copying an IR
/// nothing can execute yet would be bytes without a reader.
struct ProgramEntry {
    program_hash: u64,
    #[allow(dead_code)] // read when launch's compile cache lands
    emitter_version: u32,
}

/// A bound instance: which program, the geometry the binding echoed, and
/// the channels the instance attached.
struct InstanceEntry {
    #[allow(dead_code)] // read when launch resolves frames to instances
    program_id: u64,
    #[allow(dead_code)]
    geometry_class: u32,
    channel_ids: Vec<u64>,
}

/// What a successful `load_model` leaves behind: the parsed config and
/// every weight resident on the device, keyed by BOTH its checkpoint name
/// and (for the llama-like family) the fused trace name the executor asks
/// by.
struct LoadedModel {
    hf: crate::model::config::HfConfig,
    /// The caps JSON `load_model` answered with; owned like `Shell::caps`.
    load_caps: Vec<u8>,
    weights: std::collections::BTreeMap<String, crate::cuda::DeviceBuffer>,
    /// Trace-name RENAMES onto checkpoint names (`layer.3.attn_norm` →
    /// `model.layers.3.input_layernorm.weight`); concats get buffers of
    /// their own in `weights`, renames get a row here — no second copy of
    /// a tensor that already sits on the device.
    aliases: std::collections::BTreeMap<String, String>,
    /// gemma-4's per-layer `layer_scalar` [1] tensors, read to host once
    /// at load (the C++ `read_bf16_scalar_once`) — the fused sandwich
    /// norm's whole-stream multiplier, carried into `DispatchCtx::scales`
    /// per fire. Empty on every other family.
    gemma_layer_scalars: Vec<f32>,
}

impl LoadedModel {
    /// The device pointer for a name — the live half of the executor's
    /// `Resolver::weight`. Checkpoint names, fused names and aliases all
    /// answer. `launch` is its caller; until that arm lands it is only
    /// the load test's assertion surface.
    #[allow(dead_code)]
    fn weight(&self, name: &str) -> Option<*const std::ffi::c_void> {
        if let Some(b) = self.weights.get(name) {
            return Some(b.as_ptr().cast_const());
        }
        let target = self.aliases.get(name)?;
        self.weights.get(target).map(|b| b.as_ptr().cast_const())
    }
}

/// The capabilities this shell can honestly claim today.
const CAPS_JSON: &str =
    r#"{"driver":"driver-cuda-new","status":"phase-d-shell","abi":24}"#;

fn shell(driver: *mut PieDriver) -> Option<&'static mut Shell> {
    // SAFETY: the only non-null `PieDriver` values in circulation are the
    // boxes `pie_cuda_create` leaked; the engine's contract is to pass
    // them back unmodified.
    unsafe { driver.cast::<Shell>().as_mut() }
}

/// Create the driver. Refuses a null descriptor or a mismatched ABI
/// version by returning null, as the C++ shell does.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_create(
    desc: *const PieDriverCreateDesc,
    caps: *mut PieDriverCaps,
) -> *mut PieDriver {
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return std::ptr::null_mut();
    };
    if desc.abi_version != PIE_DRIVER_ABI_VERSION {
        return std::ptr::null_mut();
    }
    // The boot TOML rides in `config_bytes`; `[model] descriptor` is the
    // one key this shell reads today.
    let boot_descriptor = (!desc.config_bytes.ptr.is_null())
        .then(|| unsafe {
            std::slice::from_raw_parts(desc.config_bytes.ptr, desc.config_bytes.len)
        })
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|v| {
            v.get("model")?
                .get("descriptor")?
                .as_str()
                .map(std::path::PathBuf::from)
        });
    let boxed = Box::new(Shell {
        caps: CAPS_JSON.as_bytes().to_vec(),
        boot_descriptor,
        model: None,
        programs: std::collections::BTreeMap::new(),
        instances: std::collections::BTreeMap::new(),
        next_id: 1,
        notify: desc.runtime.notify,
        notify_ctx: desc.runtime.ctx,
        kv: None,
        gdn: None,
        channels: std::collections::BTreeMap::new(),
        swap: None,
        scratch: None,
    });
    let raw = Box::into_raw(boxed);
    if let Some(out) = unsafe { caps.as_mut() } {
        out.json_bytes = unsafe { (*raw).caps.as_ptr() };
        out.json_len = unsafe { (*raw).caps.len() };
    }
    raw.cast()
}

/// Tear the driver down. Null is a no-op, as everywhere in the ABI.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_destroy(driver: *mut PieDriver) {
    if !driver.is_null() {
        let mut shell = unsafe { Box::from_raw(driver.cast::<Shell>()) };
        for ch in shell.channels.values() {
            ch.free();
        }
        if let Some(swap) = &shell.swap {
            swap.free();
        }
        if let Some(mut scratch) = shell.scratch.take() {
            let mut sops = crate::model::attention_workspace::LiveStagingOps;
            scratch.ws.release(&mut sops);
            drop(scratch.decode_plan);
            drop(scratch.prefill_plan);
        }
        drop(shell);
    }
}

/// Load the model: one parse of the snapshot through the Rust loader,
/// the `pie.model/1` descriptor (embedded meta, else the boot TOML's
/// path), and every bf16 weight resident on the device — with the
/// llama-like fused trace names built beside the checkpoint names, so the
/// executor's resolver asks and receives.
///
/// Still awaited here: quantized encodings (refused, not mis-loaded),
/// the memory plan, and KV materialization — those land with `launch`.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_load_model(
    driver: *mut PieDriver,
    load: *const PieModelLoadDesc,
    caps: *mut PieDriverCaps,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(load) = (unsafe { load.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let snapshot = (!load.snapshot_dir.ptr.is_null())
        .then(|| unsafe {
            std::slice::from_raw_parts(load.snapshot_dir.ptr, load.snapshot_dir.len)
        })
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(std::path::PathBuf::from);
    let Some(snapshot) = snapshot else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    match load_impl(state, &snapshot) {
        Ok(()) => {
            let m = state.model.as_ref().expect("load_impl stored the model");
            if let Some(out) = unsafe { caps.as_mut() } {
                out.json_bytes = m.load_caps.as_ptr();
                out.json_len = m.load_caps.len();
            }
            PIE_STATUS_OK
        }
        Err(code) => code,
    }
}

/// The load itself; `i32` errors are the ABI's status codes.
fn load_impl(state: &mut Shell, snapshot: &std::path::Path) -> Result<(), i32> {
    use model_loader::checkpoint::read::{parse_checkpoint_metadata, read_meta};
    use model_loader::types::Encoding;

    let meta = parse_checkpoint_metadata(snapshot).map_err(|e| {
        eprintln!("[driver-cuda-new] load_model: checkpoint parse: {e:?}");
        PIE_STATUS_INVALID_ARGUMENT
    })?;

    // The descriptor: embedded in an artifact, else the boot TOML's path.
    let descriptor_json = match read_meta(&meta, "model/descriptor") {
        Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|_| PIE_STATUS_DRIVER_ERROR)?,
        Ok(None) => {
            let Some(path) = &state.boot_descriptor else {
                eprintln!(
                    "[driver-cuda-new] load_model: no embedded model/descriptor \
                     and no [model] descriptor in the boot config"
                );
                return Err(PIE_STATUS_UNSUPPORTED);
            };
            std::fs::read_to_string(path).map_err(|_| PIE_STATUS_INVALID_ARGUMENT)?
        }
        Err(e) => {
            eprintln!("[driver-cuda-new] load_model: read_meta: {e:?}");
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
    };
    let hf = crate::model::descriptor::parse_pie_model_descriptor(&descriptor_json)
        .map_err(|e| {
            eprintln!("[driver-cuda-new] load_model: descriptor: {e}");
            PIE_STATUS_INVALID_ARGUMENT
        })?;

    // Every raw bf16/fp32 weight, uploaded through one stream — fp32 is
    // the GDN parameter side of the `gdn_fp32_parameters` contract
    // (`A_log`, the gate norm), consumed as fp32 by its kernels.
    // Quantized encodings refuse rather than mis-load.
    let stream = crate::cuda::OwnedStream::new(0).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let alloc = crate::cuda::Allocator::new();
    let mut weights = std::collections::BTreeMap::new();
    let mut host = Vec::new();
    for t in meta.weights() {
        match &t.encoding {
            Encoding::Raw(d) if matches!(format!("{d:?}").as_str(), "BF16" | "F32") => {}
            other => {
                eprintln!("[driver-cuda-new] load_model: {}: unsupported encoding {other:?}", t.name);
                return Err(PIE_STATUS_UNSUPPORTED);
            }
        }
        let file = meta
            .files
            .iter()
            .find(|f| f.id == t.file_id)
            .ok_or(PIE_STATUS_DRIVER_ERROR)?;
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(snapshot.join(&file.path))
            .or_else(|_| std::fs::File::open(&file.path))
            .map_err(|_| PIE_STATUS_INVALID_ARGUMENT)?;
        f.seek(SeekFrom::Start(t.file_offset)).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        host.resize(usize::try_from(t.span_bytes).map_err(|_| PIE_STATUS_DRIVER_ERROR)?, 0);
        f.read_exact(&mut host).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        let mut buf = alloc.alloc(host.len()).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        buf.copy_from_host(&host, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        weights.insert(t.name.clone(), buf);
    }
    stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;

    let mut model = LoadedModel {
        hf,
        load_caps: Vec::new(),
        weights,
        aliases: std::collections::BTreeMap::new(),
        gemma_layer_scalars: Vec::new(),
    };
    fuse_llama_like(&mut model, &alloc, &stream)?;
    alias_gemma4(&mut model, &alloc, &stream)?;
    alias_qwen3_5(&mut model, &alloc, &stream)?;
    model.load_caps = format!(
        r#"{{"model_type":"{}","hidden":{},"layers":{},"vocab":{},"weights":{}}}"#,
        model.hf.model_type,
        model.hf.hidden_size,
        model.hf.num_hidden_layers,
        model.hf.vocab_size,
        model.weights.len(),
    )
    .into_bytes();
    state.model = Some(model);
    Ok(())
}

/// Build the llama-like fused trace names beside the checkpoint names —
/// the A/B harness's binder, generalized over `HfConfig` and promoted
/// into the shell. Families beyond llama-like keep their raw names; a
/// later launch asking for an unfused trace name gets the resolver's
/// drift refusal, which is the honest state until their binders land.
fn fuse_llama_like(
    model: &mut LoadedModel,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
) -> Result<(), i32> {
    let has = |m: &LoadedModel, n: &str| m.weights.contains_key(n);
    if !has(model, "model.embed_tokens.weight") {
        return Ok(()); // not an HF llama-like naming scheme; leave raw
    }
    let alias = |model: &mut LoadedModel, trace: String, ckpt: String| {
        if model.weights.contains_key(&ckpt) {
            model.aliases.insert(trace, ckpt);
        }
    };
    let fuse = |model: &mut LoadedModel,
                trace: String,
                parts: &[String]|
     -> Result<(), i32> {
        if parts.iter().any(|p| !model.weights.contains_key(p)) {
            return Ok(()); // this deployment lacks the part; skip
        }
        let mut host = Vec::new();
        for p in parts {
            let src = &model.weights[p];
            let mut back = vec![0u8; src.len()];
            src.copy_to_host(&mut back, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            host.extend_from_slice(&back);
        }
        let mut buf = alloc.alloc(host.len()).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        buf.copy_from_host(&host, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        model.weights.insert(trace, buf);
        Ok(())
    };
    alias(model, "embed".into(), "model.embed_tokens.weight".into());
    alias(model, "final_norm".into(), "model.norm.weight".into());
    if model.weights.contains_key("lm_head.weight") {
        alias(model, "lm_head".into(), "lm_head.weight".into());
    } else {
        // Tied embeddings: the trace's lm_head name IS "embed".
        alias(model, "lm_head".into(), "model.embed_tokens.weight".into());
    }
    let layers = usize::try_from(model.hf.num_hidden_layers).unwrap_or(0);
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");
        fuse(model, format!("layer.{i}.qkv"), &[
            n("self_attn.q_proj.weight"),
            n("self_attn.k_proj.weight"),
            n("self_attn.v_proj.weight"),
        ])?;
        fuse(model, format!("layer.{i}.gate_up"), &[
            n("mlp.gate_proj.weight"),
            n("mlp.up_proj.weight"),
        ])?;
        // Some checkpoints ship the fused projections ALREADY (phi3's
        // `qkv_proj` and `gate_up_proj`), in the same concatenation order
        // the fuse above builds. Those want an alias, not a copy -- and
        // `alias` is a no-op when the name is absent, so this costs the
        // deployments that split nothing.
        alias(model, format!("layer.{i}.qkv"), n("self_attn.qkv_proj.weight"));
        alias(model, format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.weight"));
        // The norm placement decides the mapping, and `input_layernorm`'s
        // presence IS the placement: pre-norm has it (attn_norm=input,
        // mlp_norm=post_attention); post-norm (olmo2) lacks it
        // (attn_norm=post_attention, mlp_norm=post_feedforward) — the
        // bind_olmo3 convention the A/B verified.
        if model.weights.contains_key(&n("input_layernorm.weight")) {
            alias(model, format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
            alias(model, format!("layer.{i}.mlp_norm"), n("post_attention_layernorm.weight"));
        } else {
            alias(model, format!("layer.{i}.attn_norm"), n("post_attention_layernorm.weight"));
            alias(model, format!("layer.{i}.mlp_norm"), n("post_feedforward_layernorm.weight"));
        }
        for (trace, ckpt) in [
            ("q_norm", "self_attn.q_norm.weight"),
            ("k_norm", "self_attn.k_norm.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("down", "mlp.down_proj.weight"),
            ("q_proj", "self_attn.q_proj.weight"),
            ("k_proj", "self_attn.k_proj.weight"),
            ("v_proj", "self_attn.v_proj.weight"),
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
        ] {
            alias(model, format!("layer.{i}.{trace}"), n(ckpt));
        }
    }
    Ok(())
}

/// Build the qwen3_5 hybrid's trace names — the `real_hybrid` A/B's
/// binder vocabulary, promoted into the shell. The checkpoint naming is
/// the VL config's (`model.language_model.*`); the vision tower and the
/// MTP block stay untouched under their raw names.
/// Build the gemma-4 trace names beside the checkpoint names —
/// `gemma4.cpp`'s binder plus the engine's `dense_fused_projection_joins`
/// (q‖k‖v on the layers that project their own KV, gate‖up everywhere),
/// as the real-weight A/B proved them. Also reads the per-layer
/// `layer_scalar` [1] tensors to host — the load-time
/// `read_bf16_scalar_once`, stashed for the fire's `scales` map.
#[allow(clippy::too_many_lines)]
fn alias_gemma4(
    model: &mut LoadedModel,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
) -> Result<(), i32> {
    let p = "model.language_model";
    if !model.weights.contains_key(&format!("{p}.embed_tokens_per_layer.weight")) {
        return Ok(()); // the PLE table IS the family's signature
    }
    let alias = |model: &mut LoadedModel, trace: String, ckpt: String| {
        if model.weights.contains_key(&ckpt) {
            model.aliases.insert(trace, ckpt);
        }
    };
    let fuse = |model: &mut LoadedModel, trace: String, parts: &[String]| -> Result<(), i32> {
        if parts.iter().any(|q| !model.weights.contains_key(q)) {
            return Ok(());
        }
        let mut host = Vec::new();
        for q in parts {
            let src = &model.weights[q];
            let mut back = vec![0u8; src.len()];
            src.copy_to_host(&mut back, stream.as_ref())
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            host.extend_from_slice(&back);
        }
        let mut buf = alloc.alloc(host.len()).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        buf.copy_from_host(&host, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        model.weights.insert(trace, buf);
        Ok(())
    };
    alias(model, "embed".into(), format!("{p}.embed_tokens.weight"));
    alias(model, "embed_per_layer".into(), format!("{p}.embed_tokens_per_layer.weight"));
    alias(model, "ple_model_proj".into(), format!("{p}.per_layer_model_projection.weight"));
    alias(model, "ple_model_norm".into(), format!("{p}.per_layer_projection_norm.weight"));
    alias(model, "final_norm".into(), format!("{p}.norm.weight"));
    let layers = usize::try_from(model.hf.num_hidden_layers).unwrap_or(0);
    let first_shared =
        layers.saturating_sub(usize::try_from(model.hf.num_kv_shared_layers).unwrap_or(0));
    let mut scalars = Vec::with_capacity(layers);
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        alias(model, format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        alias(model, format!("layer.{i}.post_attn_norm"), n("post_attention_layernorm.weight"));
        alias(model, format!("layer.{i}.pre_ffw_norm"), n("pre_feedforward_layernorm.weight"));
        alias(model, format!("layer.{i}.post_ffw_norm"), n("post_feedforward_layernorm.weight"));
        alias(model, format!("layer.{i}.q_norm"), n("self_attn.q_norm.weight"));
        alias(model, format!("layer.{i}.o_proj"), n("self_attn.o_proj.weight"));
        alias(model, format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        alias(model, format!("layer.{i}.ple_gate"), n("per_layer_input_gate.weight"));
        alias(model, format!("layer.{i}.ple_proj"), n("per_layer_projection.weight"));
        alias(model, format!("layer.{i}.ple_norm"), n("post_per_layer_input_norm.weight"));
        if i >= first_shared {
            // A KV-shared layer states only the Q leg.
            alias(model, format!("layer.{i}.q_proj"), n("self_attn.q_proj.weight"));
        } else {
            alias(model, format!("layer.{i}.k_norm"), n("self_attn.k_norm.weight"));
            fuse(model, format!("layer.{i}.qkv"), &[
                n("self_attn.q_proj.weight"),
                n("self_attn.k_proj.weight"),
                n("self_attn.v_proj.weight"),
            ])?;
        }
        fuse(model, format!("layer.{i}.gate_up"), &[
            n("mlp.gate_proj.weight"),
            n("mlp.up_proj.weight"),
        ])?;
        // The layer scalar, host-read: one bf16.
        let s = model.weights.get(&n("layer_scalar")).map_or(1.0f32, |b| {
            let mut back = [0u8; 2];
            if b.len() == 2
                && b.copy_to_host(&mut back, stream.as_ref()).is_ok()
                && stream.as_ref().synchronize().is_ok()
            {
                f32::from_bits(u32::from(u16::from_le_bytes(back)) << 16)
            } else {
                1.0
            }
        });
        scalars.push(s);
    }
    model.gemma_layer_scalars = scalars;
    Ok(())
}

fn alias_qwen3_5(
    model: &mut LoadedModel,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
) -> Result<(), i32> {
    let p = "model.language_model";
    if !model.weights.contains_key(&format!("{p}.embed_tokens.weight")) {
        return Ok(()); // not the qwen3_5 naming scheme; leave raw
    }
    if model.weights.contains_key(&format!("{p}.embed_tokens_per_layer.weight")) {
        return Ok(()); // gemma-4 shares the prefix; its aliases are its own
    }
    let alias = |model: &mut LoadedModel, trace: String, ckpt: String| {
        if model.weights.contains_key(&ckpt) {
            model.aliases.insert(trace, ckpt);
        }
    };
    alias(model, "embed".into(), format!("{p}.embed_tokens.weight"));
    alias(model, "final_norm".into(), format!("{p}.norm.weight"));
    let layers = usize::try_from(model.hf.num_hidden_layers).unwrap_or(0);
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        alias(model, format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        alias(model, format!("layer.{i}.mlp_norm"), n("post_attention_layernorm.weight"));
        alias(model, format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        let full = model
            .hf
            .layer_types
            .get(i)
            .is_some_and(|t| t == "full_attention");
        if full {
            for f in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                alias(model, format!("layer.{i}.{f}"), n(&format!("self_attn.{f}.weight")));
            }
        } else {
            for f in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                alias(model, format!("layer.{i}.{f}"), n(&format!("linear_attn.{f}.weight")));
            }
            alias(model, format!("layer.{i}.conv"), n("linear_attn.conv1d.weight"));
            alias(model, format!("layer.{i}.conv_bias"), n("linear_attn.conv1d.bias"));
            alias(model, format!("layer.{i}.a_log"), n("linear_attn.A_log"));
            alias(model, format!("layer.{i}.dt_bias"), n("linear_attn.dt_bias"));
            alias(model, format!("layer.{i}.gate_norm"), n("linear_attn.norm.weight"));
            alias(model, format!("layer.{i}.o_proj"), n("linear_attn.out_proj.weight"));
        }
        // The fused gate‖up bank, gate first — the dense MLP's binding.
        let parts = [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")];
        if parts.iter().all(|q| model.weights.contains_key(q)) {
            let mut host = Vec::new();
            for q in &parts {
                let src = &model.weights[q];
                let mut back = vec![0u8; src.len()];
                src.copy_to_host(&mut back, stream.as_ref())
                    .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
                stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
                host.extend_from_slice(&back);
            }
            let mut buf = alloc.alloc(host.len()).map_err(|_| PIE_STATUS_EXHAUSTED)?;
            buf.copy_from_host(&host, stream.as_ref())
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            model.weights.insert(format!("layer.{i}.gate_up"), buf);
        }
    }
    Ok(())
}

/// Register a program: the id lifecycle, with the C3 hash as the dedup
/// key — re-registering answers the existing id. The launch PACKAGE is
/// not yet copied; it is deep-copied when the `launch` arm lands and has
/// a reader for it.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_register_program(
    driver: *mut PieDriver,
    program: *const PieProgramDesc,
    program_id: *mut u64,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { program.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    if desc.abi_version != PIE_DRIVER_ABI_VERSION {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let existing = state
        .programs
        .iter()
        .find(|(_, p)| p.program_hash == desc.program_hash)
        .map(|(&id, _)| id);
    let id = existing.unwrap_or_else(|| {
        let id = state.next_id;
        state.next_id += 1;
        state.programs.insert(
            id,
            ProgramEntry {
                program_hash: desc.program_hash,
                emitter_version: desc.emitter_version,
            },
        );
        id
    });
    if let Some(out) = unsafe { program_id.as_mut() } {
        *out = id;
    }
    PIE_STATUS_OK
}

/// Register a channel endpoint: the C++ registry's binding contract —
/// a pinned host MIRROR of `(capacity + 1)` wire cells and four pinned
/// control words (head 0, tail 1, poison 2, closed 3), both zeroed, with
/// the wire-cell math reproduced exactly (bool bit-packs, everything
/// else is four bytes per element; `capacity + 1 ≤ 64`). Device-side
/// rings and fire delivery ride with the launch integration.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_register_channel(
    driver: *mut PieDriver,
    channel: *const PieChannelDesc,
    binding: *mut PieChannelEndpointBinding,
) -> i32 {
    use crate::model::attention_workspace::{LiveStagingOps, StagingOps};

    const MAX_RING: u64 = 64;
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { channel.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    if desc.abi_version != PIE_DRIVER_ABI_VERSION
        || state.channels.contains_key(&desc.channel_id)
        || desc.dtype > driver_abi::local::PIE_CHANNEL_DTYPE_ACT
    {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let shape = slice_of(desc.shape.ptr, desc.shape.len);
    let mut numel: u64 = 1;
    for &d in shape {
        let Some(next) = numel.checked_mul(u64::from(d)) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        numel = next;
    }
    let wire_bytes: u64 = if desc.dtype == driver_abi::local::PIE_CHANNEL_DTYPE_BOOL {
        numel.div_ceil(8)
    } else {
        match numel.checked_mul(4) {
            Some(b) => b,
            None => return PIE_STATUS_INVALID_ARGUMENT,
        }
    };
    let ring = u64::from(desc.capacity) + 1;
    if wire_bytes == 0 || ring > MAX_RING {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let Some(mirror_bytes) = wire_bytes.checked_mul(ring) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Ok(mirror_bytes) = usize::try_from(mirror_bytes) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };

    let mut ops = LiveStagingOps;
    let Some(mirror) = ops.malloc_host(mirror_bytes) else {
        return PIE_STATUS_EXHAUSTED;
    };
    let word_bytes = 4 * std::mem::size_of::<u64>();
    let Some(words) = ops.malloc_host(word_bytes) else {
        ops.free_host(mirror);
        return PIE_STATUS_EXHAUSTED;
    };
    unsafe {
        std::ptr::write_bytes(mirror.cast::<u8>(), 0, mirror_bytes);
        std::ptr::write_bytes(words.cast::<u8>(), 0, word_bytes);
    }
    state.channels.insert(
        desc.channel_id,
        ChannelState {
            mirror,
            words,
            mirror_bytes,
            cell_bytes: usize::try_from(wire_bytes).unwrap_or(usize::MAX),
            ring: u32::try_from(ring).expect("ring fits u32"),
            host_role: desc.host_role,
        },
    );
    if let Some(out) = unsafe { binding.as_mut() } {
        *out = PieChannelEndpointBinding {
            channel_id: desc.channel_id,
            mirror_base: mirror as u64,
            word_base: words as u64,
            mirror_bytes: mirror_bytes as u64,
            word_bytes: word_bytes as u64,
            cell_bytes: u32::try_from(wire_bytes).unwrap_or(u32::MAX),
            capacity: desc.capacity,
            head_word_index: 0,
            tail_word_index: 1,
            poison_word_index: 2,
            closed_word_index: 3,
        };
    }
    PIE_STATUS_OK
}

/// Bind an instance to a registered program: the id lifecycle, honoring
/// a nonzero `requested_instance_id` and echoing the geometry class.
/// KV-slot and adapter state ride in with the `launch` arm.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_bind_instance(
    driver: *mut PieDriver,
    instance: *const PieInstanceDesc,
    binding: *mut PieInstanceBinding,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { instance.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    if desc.abi_version != PIE_DRIVER_ABI_VERSION
        || !state.programs.contains_key(&desc.program_id)
    {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let id = if desc.requested_instance_id != 0 {
        desc.requested_instance_id
    } else {
        let id = state.next_id;
        state.next_id += 1;
        id
    };
    if state.instances.contains_key(&id) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    state.instances.insert(
        id,
        InstanceEntry {
            program_id: desc.program_id,
            geometry_class: desc.geometry_class,
            channel_ids: slice_of(desc.channel_ids.ptr, desc.channel_ids.len).to_vec(),
        },
    );
    if let Some(out) = unsafe { binding.as_mut() } {
        out.instance_id = id;
        out.geometry_class = desc.geometry_class;
        out.reserved0 = 0;
    }
    PIE_STATUS_OK
}

/// Launch a frame: the executor's fire assembly, promoted from the
/// smokes into the shell.
///
/// What runs today: SINGLE-step, single-sub-batch frames over the loaded
/// llama-like model — the frame's own CSRs become the fire, the KV pools
/// are driver-owned and persist across launches, write targets derive
/// from the CSR tails, and every batch member's terminal cell is
/// published (release) before the runtime is notified. Multi-step
/// frames, device-geometry sub-batches and channel-delivered outputs
/// refuse with UNSUPPORTED until their machinery lands — logits stay in
/// the shell until channels exist to carry them out.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_launch(
    driver: *mut PieDriver,
    frame: *const PieFrameDesc,
    completion: PieCompletion,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(frame) = (unsafe { frame.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    match launch_impl(state, frame) {
        Ok(()) => {
            // Publish every member's terminal cell, then notify.
            if let Some(step) = slice_of(frame.steps.ptr, frame.steps.len).first() {
                for &cell in slice_of(step.terminal_cells.ptr, step.terminal_cells.len) {
                    if !cell.is_null() {
                        unsafe {
                            std::ptr::addr_of_mut!((*cell).outcome).write_volatile(
                                driver_abi::local::PIE_TERMINAL_OUTCOME_SUCCESS,
                            );
                        }
                    }
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            if let Some(notify) = state.notify {
                unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
            }
            PIE_STATUS_OK
        }
        Err(code) => code,
    }
}

/// A borrowed ABI slice as a Rust slice; empty for null.
fn slice_of<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// The families the shell can plan today.
enum FamilyFacts {
    LlamaLike(
        model::families::llama_like::forward::facts::LlamaLikeFacts,
        model::families::llama_like::forward::facts::LlamaLikeCudaFacts,
    ),
    Qwen35(
        model::qwen_3_5::forward::facts::Qwen35HybridFacts,
        model::qwen_3_5::forward::facts::Qwen35CudaFacts,
    ),
    Gemma4(
        model::gemma_4::forward::facts::Gemma4Facts,
        model::gemma_4::forward::facts::Gemma4CudaFacts,
    ),
}

/// gemma-4's facts off the checkpoint's config — the layer schedule
/// reduced to the interval (irregular arrays refuse, qwen3_5's rule),
/// the FULL layers' rotary width by the driver's derivation, the
/// double-wide-MLP and KV-shared counts as stated. The E2B anchor's
/// legs only: `k_eq_v` (26B-A4B's V-from-K mode) and the MoE block
/// refuse until a deployment anchors them.
fn gemma4_facts_from_hf(model: &LoadedModel) -> Result<FamilyFacts, i32> {
    use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
    let hf = &model.hf;
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    let regular = interval > 0
        && hf.layer_types.iter().enumerate().all(|(l, t)| {
            (t == "full_attention") == (l as u32 % interval == interval - 1)
        });
    if !regular {
        eprintln!("[driver-cuda-new] launch: irregular gemma-4 layer_types refuse");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    if hf.gemma4_attention_k_eq_v || hf.gemma4_enable_moe {
        eprintln!("[driver-cuda-new] launch: gemma-4 k_eq_v/MoE legs await their anchor");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let global_d = to_u32(hf.gemma4_global_head_dim.max(hf.head_dim));
    // The FULL layers' partial factor: the per-layer table when the
    // config ships one, else full rotation — `rotary_of`'s input.
    let full_factor = (0..hf.layer_types.len())
        .find(|&l| hf.layer_types[l] == "full_attention")
        .and_then(|l| hf.gemma_per_layer_partial_rotary_factor.get(l).copied())
        .unwrap_or(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let global_rotary = 2u32.max(2 * (0.5 * full_factor * global_d as f32) as u32);
    let facts = Gemma4Facts {
        hidden: to_u32(hf.hidden_size),
        layers: to_u32(hf.num_hidden_layers),
        full_attn_interval: interval,
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        global_head_dim: global_d,
        global_rotary_dim: global_rotary,
        intermediate: to_u32(hf.intermediate_size),
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        kv_shared_layers: to_u32(hf.num_kv_shared_layers),
        ple_dim: to_u32(hf.gemma_hidden_size_per_layer_input),
        double_wide_shared: hf.gemma4_double_wide_mlp,
        logit_softcap: hf.gemma_final_logit_softcap,
    };
    // The LIVE binding: both banks fused (the load's joins built them),
    // native bf16 pages — the A/B's proven set.
    //
    // `window_left` is NOT empty here, and gemma-4 is the family that
    // makes the difference visible: full-attention layers see the whole
    // context and the rest attend a sliding window, on the family's own
    // interval. The shell already derived exactly this list for its
    // decode plans; now the declaration carries it too, and an empty list
    // would have the trace say "no window" while the plan applied one.
    let cuda = Gemma4CudaFacts {
        fused_qkv: true,
        gate_up_fused: true,
        kv_native_bf16: true,
        window_left: (0..facts.layers)
            .map(|l| if facts.is_full_attn(l) { -1 } else { hf.sliding_window.max(0) })
            .collect(),
    };
    Ok(FamilyFacts::Gemma4(facts, cuda))
}

/// The qwen3_5 hybrid's facts, read off the checkpoint's own config —
/// the layer schedule from `layer_types` (reduced to the interval, the
/// Metal driver's reduction; irregular arrays refuse), the GDN geometry
/// from the `linear_*` fields, the rotary width by the driver's
/// `max(2, 2·int(0.5·factor·head_dim))` derivation. Dense MLP only —
/// a MoE config refuses until a MoE deployment anchors that leg.
fn qwen35_facts_from_hf(model: &LoadedModel) -> Result<FamilyFacts, i32> {
    use model::qwen_3_5::forward::facts::{
        Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind,
    };
    use model_compiler::trace::NormVariant;
    let hf = &model.hf;
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    let regular = interval > 0
        && hf.layer_types.iter().enumerate().all(|(l, t)| {
            (t == "full_attention") == (l as u32 % interval == interval - 1)
        });
    if !regular {
        eprintln!("[driver-cuda-new] launch: irregular qwen3_5 layer_types refuse");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    if hf.num_experts > 0 {
        eprintln!("[driver-cuda-new] launch: the qwen3_5 MoE leg awaits its anchor deployment");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rotary =
        2u32.max(2 * (0.5 * hf.partial_rotary_factor * hf.head_dim as f32) as u32);
    let facts = Qwen35HybridFacts {
        layers: to_u32(hf.num_hidden_layers),
        full_attn_interval: interval,
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        norm_variant: NormVariant::Gemma,
        attn: Qwen35FullAttnFacts {
            hidden: to_u32(hf.hidden_size),
            q_heads: to_u32(hf.num_attention_heads),
            kv_heads: to_u32(hf.num_key_value_heads),
            head_dim: to_u32(hf.head_dim),
            rotary_dim: rotary,
            fused_qkv: false,
            norm_variant: NormVariant::Gemma,
        },
        gdn: Qwen35GdnFacts {
            hidden: to_u32(hf.hidden_size),
            key_heads: to_u32(hf.linear_num_key_heads),
            value_heads: to_u32(hf.linear_num_value_heads),
            key_head_dim: to_u32(hf.linear_key_head_dim),
            value_head_dim: to_u32(hf.linear_value_head_dim),
            conv_kernel: to_u32(hf.linear_conv_kernel_dim),
            fused_in_proj: false,
            norm_variant: NormVariant::Gemma,
        },
        mlp: Qwen35MlpKind::Dense { intermediate: to_u32(hf.intermediate_size) },
    };
    // The LIVE L40S cuda set (`emissions.rs`): warp-tiled and the cached
    // prefill env-gated off, bf16 recurrent state, prefill-decode on.
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
        // As llama_like's, and for the same reason.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        window_left: Vec::new(),
    };
    Ok(FamilyFacts::Qwen35(facts, cuda))
}

/// The loaded model's facts, family-dispatched: the qwen3_5 hybrid by
/// its `linear_*` geometry + layer schedule, else the llama-like
/// mapping. Only the qwen3-family pre-norm shape is claimed on the
/// llama-like side; anything else refuses rather than mis-executes.
/// What a row dispatches to: this family's facts, off the checkpoint.
type FactsFrom = fn(&LoadedModel) -> Result<FamilyFacts, i32>;

/// One row per `model_type` this shell can OPEN.
///
/// A table rather than the chain of weight-name sniffs this replaces, for
/// exactly the reason `model::contract::HF_ROWS` is a table: the supported
/// set becomes a VALUE something can iterate. The gap between what the
/// loader can author and what this shell can open is then a test with a
/// closed list (`tests/facts_registry.rs`) rather than a surprise at boot
/// — which is what §3.3's "eight families dispatch but cannot load" was.
///
/// Dispatch is on the model type because that is what the descriptor
/// SAYS. Sniffing a weight name infers the family from a consequence of
/// it, which is how `gemma3` used to be answered by the llama-like
/// derivation: it has `model.embed_tokens.weight` and a pre-norm, so the
/// sniff accepted it and transcribed the wrong facts. A model type with
/// no row now refuses by name, which is this plan's standing rule —
/// refuse what cannot be derived rather than guess it.
const FACTS_ROWS: &[(&str, FactsFrom)] = &[
    // ── llama lineage: dense/GQA decoders the llama_like text serves.
    ("qwen3", llama_like_facts_from_hf),
    ("qwen2", llama_like_facts_from_hf),
    ("llama", llama_like_facts_from_hf),
    ("llama3", llama_like_facts_from_hf),
    ("mistral", llama_like_facts_from_hf),
    ("mistral3", llama_like_facts_from_hf),
    ("ministral3", llama_like_facts_from_hf),
    ("olmo2", llama_like_facts_from_hf),
    ("olmo3", llama_like_facts_from_hf),
    ("phi3", llama_like_facts_from_hf),
    // Qwen3-VL binds the plain Qwen3 TEXT tower; the vision tower is a
    // service behind `pie_cuda_encode`, not part of this decode plan.
    ("qwen3_vl", llama_like_facts_from_hf),
    ("qwen3_vl_text", llama_like_facts_from_hf),
    // ── Gemma-4: nested decoder, PLE, two layer kinds.
    ("gemma4", gemma4_facts_from_hf),
    ("gemma4_text", gemma4_facts_from_hf),
    // ── Qwen3.5 hybrids: GDN linear attention beside full attention.
    ("qwen3_5", qwen35_facts_from_hf),
    ("qwen3_5_text", qwen35_facts_from_hf),
    ("qwen3_5_moe", qwen35_facts_from_hf),
    ("qwen3_5_moe_text", qwen35_facts_from_hf),
];

/// Every `model_type` this shell can open, in table order.
///
/// Public so that `tests/facts_registry.rs` can hold it against the
/// loader's own registry. The two lists answering "which model type is
/// supported" from opposite sides of the load is exactly the pairing
/// `model::contract`'s header describes, and the same failure it names
/// applies here: a family whose forward is declared but whose facts were
/// never written used to surface as a wrong answer rather than a refusal.
#[must_use]
pub fn openable_model_types() -> Vec<&'static str> {
    FACTS_ROWS.iter().map(|(k, _)| *k).collect()
}

/// The facts for a loaded checkpoint, by the model type it declares.
fn facts_from_hf(model: &LoadedModel) -> Result<FamilyFacts, i32> {
    let model_type = model.hf.model_type.as_str();
    match FACTS_ROWS.iter().find(|(k, _)| *k == model_type) {
        Some((_, derive)) => derive(model),
        None => {
            eprintln!(
                "[driver-cuda-new] launch: no facts derivation for \
                 model_type='{model_type}'; the family declares a forward \
                 but nobody has written its facts"
            );
            Err(PIE_STATUS_UNSUPPORTED)
        }
    }
}

/// The llama lineage's facts, off the checkpoint's own config.
fn llama_like_facts_from_hf(model: &LoadedModel) -> Result<FamilyFacts, i32> {
    use model::families::llama_like::forward::facts::{
        LlamaLikeCudaFacts, LlamaLikeFacts, NormPlacement, QkNorm,
    };
    use model_compiler::trace::{NormVariant, RopeKind};
    let hf = &model.hf;
    if !model.weights.contains_key("model.embed_tokens.weight") {
        eprintln!("[driver-cuda-new] launch: only HF llama-like checkpoints execute today");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    // THE GQA RATIO, refused here rather than discovered at launch.
    //
    // FlashInfer's decode instantiates group sizes {1, 2, 3, 4, 8} and
    // reports anything else by THROWING. A throw crossing the C ABI is
    // undefined behaviour; the generated shim now prints the message
    // before it dies, but printing is all it can do — the launcher
    // signatures have nowhere to put a failure. A load DOES: it returns a
    // status code.
    //
    // So a deployment whose q/kv ratio this build cannot serve is turned
    // away while turning it away is still cheap. Qwen2.5-1.5B is the live
    // example — twelve query heads over two kv heads is a group size of
    // six, and six is not in the list.
    let kv_heads = hf.num_key_value_heads.max(1);
    let group_size = hf.num_attention_heads / kv_heads;
    if hf.num_attention_heads % kv_heads != 0
        || !matches!(group_size, 1 | 2 | 3 | 4 | 8)
    {
        eprintln!(
            "[driver-cuda-new] load: this build's decode does not instantiate \
             GQA group size {group_size} ({} q heads over {kv_heads} kv heads); \
             the supported set is 1, 2, 3, 4, 8",
            hf.num_attention_heads
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }

    // NORM PLACEMENT, off the checkpoint. `input_layernorm`'s presence IS
    // the placement, which is the same fact `fuse_llama_like` already
    // binds on: pre-norm ships it, post-norm (olmo2) ships
    // `post_attention` + `post_feedforward` instead. The binder was
    // already correct for both; only this derivation refused.
    let pre_norm = model
        .aliases
        .get("layer.0.attn_norm")
        .is_some_and(|t| t.ends_with("input_layernorm.weight"));

    // QK NORM, three ways, and the checkpoint distinguishes them by
    // SHAPE rather than by any config key. A deployment that norms q and
    // k ships `q_norm`/`k_norm`; whether it norms PER HEAD (qwen3, one
    // gamma of `head_dim`) or over the whole projection (olmo2, one gamma
    // of `q_heads * head_dim`) is the tensor's own extent. Reading the
    // extent is deriving from the checkpoint; assuming one is guessing,
    // and the two lower to different kernels.
    let elems_of = |trace: &str| -> Option<usize> {
        let ckpt = model.aliases.get(trace)?;
        // bf16 gammas throughout this family.
        Some(model.weights.get(ckpt)?.len() / 2)
    };
    let qk_norm = match elems_of("layer.0.q_norm") {
        None => QkNorm::Off,
        Some(n) if n == usize::try_from(hf.head_dim).unwrap_or(0) => QkNorm::PerHead,
        Some(_) => QkNorm::Global,
    };

    // FUSED QKV is a fact about the LOAD, not about the checkpoint:
    // `fuse_llama_like` concatenates q/k/v when all three are present and
    // leaves them alone when they are not. So the honest source is
    // whether the fused name exists, which is what the trace will state.
    // Either spelling counts: `fuse` writes a concatenated buffer under
    // the trace name, while a checkpoint that already ships the fused
    // projection gets an alias to it instead.
    let fused_qkv = model.weights.contains_key("layer.0.qkv")
        || model.aliases.contains_key("layer.0.qkv");

    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let facts = LlamaLikeFacts {
        hidden: to_u32(hf.hidden_size),
        layers: to_u32(hf.num_hidden_layers),
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        intermediate: to_u32(hf.intermediate_size),
        vocab: to_u32(hf.vocab_size),
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Plain,
        norm_placement: if pre_norm { NormPlacement::Pre } else { NormPlacement::Post },
        qk_norm,
        fused_qkv,
        tied_embeddings: hf.tie_word_embeddings,
        qkv_bias: hf.attention_bias,
    };
    let cuda = LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: hf.head_dim != hf.head_dim_kernel,
        // The padded width itself, from the same place the flag reads.
        head_dim_kernel: to_u32(hf.head_dim_kernel),
        gate_up_fused: true,
        // The shell's own frame: one GPU, no collectives, bf16
        // checkpoints. `window_left` empty reads as "no window", which is
        // what this assembly meant before the declaration carried one —
        // the shell derives its own per-layer windows from
        // `hf.sliding_window` where a family has them, and that path is
        // unchanged.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        tp_size: 1,
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    };
    Ok(FamilyFacts::LlamaLike(facts, cuda))
}

/// The fire itself. Everything here is the proven smoke assembly, run
/// against the shell's own state.
#[allow(clippy::too_many_lines)]
fn launch_impl(state: &mut Shell, frame: &PieFrameDesc) -> Result<(), i32> {

    let steps = slice_of(frame.steps.ptr, frame.steps.len);
    if steps.is_empty() {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // Steps run SEQUENTIALLY, each a fire of its own — the frame's
    // producer→consumer ordering. One shared KV, per-step everything else.
    for step in &steps[..steps.len() - 1] {
        step_impl(state, frame, step)?;
    }
    let step = steps.last().expect("nonempty");
    step_impl(state, frame, step)
}

/// One step's fire — the former single-step body.
#[allow(clippy::too_many_lines)]
fn step_impl(
    state: &mut Shell,
    frame: &PieFrameDesc,
    step: &driver_abi::local::PieStepDesc,
) -> Result<(), i32> {
    use crate::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use crate::model::executor::{
        AttnCtx, DecodePlan, DispatchCtx, DispatchPlan, Frame, GdnCtx, PrefillPlan, Resolver,
        run,
    };
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let sub_batches = slice_of(step.sub_batch_indptr.ptr, step.sub_batch_indptr.len);
    if sub_batches.len() > 2 {
        eprintln!("[driver-cuda-new] launch: one sub-batch per step today");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let Some(model) = state.model.as_ref() else {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };
    let family = facts_from_hf(model)?;

    let token_ids = slice_of(step.token_ids.ptr, step.token_ids.len);
    let position_ids = slice_of(step.position_ids.ptr, step.position_ids.len);
    let kv_indices = slice_of(step.kv_page_indices.ptr, step.kv_page_indices.len);
    let kv_indptr = slice_of(step.kv_page_indptr.ptr, step.kv_page_indptr.len);
    let kv_lens = slice_of(step.kv_last_page_lens.ptr, step.kv_last_page_lens.len);
    let qo_indptr = slice_of(step.qo_indptr.ptr, step.qo_indptr.len);
    if token_ids.is_empty()
        || token_ids.len() != position_ids.len()
        || kv_indptr.len() < 2
        || kv_indptr.len() != kv_lens.len() + 1
        || qo_indptr.len() != kv_indptr.len()
    {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    let rows = token_ids.len();
    let requests = kv_lens.len();
    let class = if rows == requests { FireClass::Decode } else { FireClass::Prefill };

    // ── The lowering. ──
    let plan = match &family {
        FamilyFacts::LlamaLike(facts, cuda) => {
            model::families::llama_like::forward::llama_like_cuda(facts, cuda, class)
        }
        FamilyFacts::Qwen35(facts, cuda) => {
            model::qwen_3_5::forward::qwen3_5_hybrid_cuda(facts, cuda, class)
        }
        FamilyFacts::Gemma4(facts, cuda) => {
            model::gemma_4::forward::gemma4_cuda(facts, cuda, class)
        }
    };
    let fire_rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    let lowered = lower(&plan, &fire_rows, Fire { captures_across_splits: false })
        .map_err(|e| {
            eprintln!("[driver-cuda-new] launch: uncovered: {e:?}");
            PIE_STATUS_UNSUPPORTED
        })?;
    let dplan = DispatchPlan::new(&plan, &lowered);

    // ── Device state: KV pools (persistent), fire arrays (per launch). ──
    let stream = crate::cuda::OwnedStream::new(0).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = crate::cuda::Allocator::new();

    let need_pages = frame.required_kv_pages.max(
        kv_indices.iter().copied().max().map_or(1, |m| m + 1),
    );
    let page_size: i32 = 16;
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    // Per-layer pool geometry, family-decided: gemma-4's two layer kinds
    // disagree on head dim and its trailing layers own NO pool (they
    // attend through their source's pages — the load-time decision). A
    // `None` row is a shared layer; its VIEW mirrors the source below.
    let layer_geom: Vec<Option<(i32, u32)>> = match &family {
        FamilyFacts::Gemma4(facts, _) => (0..facts.layers)
            .map(|l| {
                (!facts.is_kv_shared(l))
                    .then(|| (facts.head_dim_of(l) as i32, l))
            })
            .collect(),
        _ => (0..u32::try_from(model.hf.num_hidden_layers).unwrap_or(0))
            .map(|l| Some((head_dim_i, l)))
            .collect(),
    };
    let grow = !matches!(&state.kv, Some(kv) if kv.num_pages >= need_pages);
    if grow {
        let mut pools = Vec::new();
        for g in &layer_geom {
            let Some((d, _)) = g else {
                pools.push(None);
                continue;
            };
            let bytes = need_pages as usize
                * page_size as usize
                * kv_heads_i as usize
                * *d as usize
                * 2;
            let mut k = alloc.alloc(bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?;
            let mut v = alloc.alloc(bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?;
            k.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            v.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            pools.push(Some((k, v)));
        }
        // NOTE: growth REPLACES the pools without migrating pages — decode
        // continuity holds while the page demand is stable, which is the
        // single-frame smoke's world. Page migration rides with resize_pool.
        state.kv = Some(KvState { pools, num_pages: need_pages });
    }
    let kv = state.kv.as_ref().expect("just ensured");
    let kv_source_of = |i: usize| -> usize {
        match &family {
            FamilyFacts::Gemma4(facts, _) => facts
                .kv_source(u32::try_from(i).unwrap_or(0))
                .map_or(i, |s| s as usize),
            _ => i,
        }
    };
    let layers: Vec<crate::launch::KvCacheLayerView> = (0..kv.pools.len())
        .map(|i| {
            let src = kv_source_of(i);
            let (k, v) = kv.pools[src].as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            let d = match &family {
                FamilyFacts::Gemma4(facts, _) => {
                    facts.head_dim_of(u32::try_from(i).unwrap_or(0)) as i32
                }
                _ => head_dim_i,
            };
            crate::launch::KvCacheLayerView {
                layer: i as i32,
                source_layer: src as i32,
                num_pages: kv.num_pages as i32,
                page_size,
                num_kv_heads: kv_heads_i,
                head_dim: d,
                scheme: crate::launch::KvCacheScheme::Native,
                storage_dtype: crate::dtype::DType::Bf16,
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

    let up_u32 = |vals: &[u32]| -> Result<crate::cuda::DeviceBuffer, i32> {
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut b = alloc.alloc(bytes.len().max(4)).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        b.copy_from_host(&bytes, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        Ok(b)
    };
    let d_ids = up_u32(token_ids)?;
    let d_pos = up_u32(position_ids)?;
    let d_kv_indices = up_u32(kv_indices)?;
    let d_kv_indptr = up_u32(kv_indptr)?;
    let d_kv_lens = up_u32(kv_lens)?;
    let d_qo = up_u32(qo_indptr)?;

    // Write targets: each request appends its NEW tokens at the CSR tail.
    // Decode appends one token at `len - 1`; prefill appends its whole
    // window ending there.
    let mut w_page = Vec::with_capacity(rows);
    let mut w_off = Vec::with_capacity(rows);
    for r in 0..requests {
        let pages = &kv_indices[kv_indptr[r] as usize..kv_indptr[r + 1] as usize];
        let total = (pages.len() as u32 - 1) * page_size as u32 + kv_lens[r];
        let toks = (qo_indptr[r + 1] - qo_indptr[r]) as usize;
        for t in 0..toks {
            let pos = total - toks as u32 + t as u32;
            w_page.push(pages[(pos / page_size as u32) as usize]);
            w_off.push(pos % page_size as u32);
        }
    }
    let d_w_page = up_u32(&w_page)?;
    let d_w_off = up_u32(&w_off)?;
    let mut d_valid = alloc.alloc(rows).map_err(|_| PIE_STATUS_EXHAUSTED)?;
    d_valid
        .copy_from_host(&vec![1u8; rows], stream.as_ref())
        .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;

    // ── Workspace + plan caches: DRIVER-lifetime, first-launch built. ──
    let mut sops = LiveStagingOps;
    if state.scratch.is_none() {
        let ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)
            .map_err(|_| PIE_STATUS_EXHAUSTED)?;
        state.scratch = Some(FireScratch {
            ws,
            decode_plan: DecodePlan::new(),
            decode_plan_full: DecodePlan::new(),
            prefill_plan: PrefillPlan::new(),
        });
    }
    let scratch = state.scratch.as_mut().expect("just ensured");
    let (ws, decode_plan, decode_plan_full, prefill_plan) = (
        &mut scratch.ws,
        &mut scratch.decode_plan,
        &mut scratch.decode_plan_full,
        &mut scratch.prefill_plan,
    );
    // Plan for the dispatch the LOWERED text actually states — not the
    // fire class: the hybrid's `prefill_decode` fact routes a
    // single-request decode through the PREFILL flashinfer path
    // (`TokensLE(1)` resolves at lower time).
    let states_decode_dispatch = lowered
        .kernels
        .iter()
        .any(|k| k == "attn::dispatch_attention_flashinfer_decode");
    ws.begin_plan_update(&mut sops).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let decode_plan_full_ptr = if let FamilyFacts::Gemma4(facts, _) = &family {
        if states_decode_dispatch {
            // TWO decode plans, one per layer kind — the C++'s
            // `decode_plan_sliding` / `decode_plan_full` pair, because
            // the kinds disagree on head dim and the planner bakes it in.
            decode_plan.plan_decode_variant(
                kv_indptr,
                model.hf.num_attention_heads,
                kv_heads_i,
                facts.head_dim as i32,
                page_size,
                ws.view(),
                raw_stream,
                false,
                false,
                -1,
            );
            decode_plan_full.plan_decode_variant(
                kv_indptr,
                model.hf.num_attention_heads,
                kv_heads_i,
                facts.global_head_dim as i32,
                page_size,
                ws.view(),
                raw_stream,
                false,
                true,
                -1,
            );
            decode_plan_full.as_ptr()
        } else {
            // gemma-4's prefill is PLANLESS (it plans internally per
            // fire, off the host CSR mirrors) and its 512-wide layers
            // take the naive kernel — nothing to pre-plan.
            core::ptr::null_mut()
        }
    } else if states_decode_dispatch {
        decode_plan.plan_decode(
            kv_indptr,
            model.hf.num_attention_heads,
            kv_heads_i,
            head_dim_i,
            page_size,
            ws.view(),
            raw_stream,
            false,
            -1,
        );
        core::ptr::null_mut()
    } else {
        prefill_plan.plan_prefill(
            qo_indptr,
            kv_indptr,
            kv_lens,
            model.hf.num_attention_heads,
            kv_heads_i,
            head_dim_i,
            page_size,
            ws.view(),
            raw_stream,
            false,
            -1,
        );
        core::ptr::null_mut()
    };
    ws.end_plan_update(&mut sops, raw_stream);

    let arena = alloc.alloc(lowered.arena_bytes.max(64)).map_err(|_| PIE_STATUS_EXHAUSTED)?;
    let exec_frame = Frame { arena: arena.as_ptr(), arena_bytes: lowered.arena_bytes.max(64) };

    let mut named_widths: std::collections::BTreeMap<ValueId, u32> =
        std::collections::BTreeMap::new();
    for a in &lowered.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..lowered.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: std::collections::BTreeMap<ValueId, crate::cuda::DeviceBuffer> =
        std::collections::BTreeMap::new();
    for (&v, &w) in &named_widths {
        // fp32-wide: the GDN seam pins are f32; llama-like's are bf16 and
        // simply leave half the pin unread.
        let mut b =
            alloc.alloc(rows * w as usize * 4).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        b.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        named_bufs.insert(v, b);
    }

    // ── The hybrid's GDN context: driver-owned slabs, instance slots. ──
    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::cuda::DeviceBuffer> = None;
    if let FamilyFacts::Qwen35(facts, cuda) = &family {
        let conv_stride = (facts.gdn.conv_kernel * facts.gdn.conv_dim()) as usize;
        let state_stride = (facts.gdn.value_heads
            * facts.gdn.key_head_dim
            * facts.gdn.value_head_dim) as usize;
        let state_elem = if cuda.state_bf16 { 2 } else { 4 };
        const GDN_SLOTS: u32 = 8;
        if state.gdn.is_none() {
            let mut slabs = Vec::new();
            for l in 0..facts.layers {
                if facts.is_full_attn(l) {
                    slabs.push(None);
                    continue;
                }
                let mut c = alloc
                    .alloc(GDN_SLOTS as usize * conv_stride * 2)
                    .map_err(|_| PIE_STATUS_EXHAUSTED)?;
                let mut r = alloc
                    .alloc(GDN_SLOTS as usize * state_stride * state_elem)
                    .map_err(|_| PIE_STATUS_EXHAUSTED)?;
                c.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
                r.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
                slabs.push(Some((c, r)));
            }
            state.gdn = Some(GdnState {
                slabs,
                num_slots: GDN_SLOTS,
                conv_stride_elems: i64::try_from(conv_stride).unwrap_or(0),
                state_stride_elems: i64::try_from(state_stride).unwrap_or(0),
                state_elem_bytes: state_elem,
            });
        }
        let gdn_state = state.gdn.as_mut().expect("just ensured");
        // The ENGINE assigns slots: `rs_slot_ids`, one per request. The
        // flags this shell executes are RESET (zero the slot before the
        // fire); the fold/buffer machinery is spec-decode's and refuses.
        let rs_slot_ids = slice_of(step.rs_slot_ids.ptr, step.rs_slot_ids.len);
        let rs_flags = slice_of(step.rs_slot_flags.ptr, step.rs_slot_flags.len);
        if rs_slot_ids.len() != requests {
            eprintln!("[driver-cuda-new] launch: hybrid fire without rs_slot_ids");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        if rs_flags.iter().any(|f| f & !driver_abi::local::PIE_RS_FLAG_RESET != 0) {
            eprintln!("[driver-cuda-new] launch: rs fold/buffer flags await spec-decode");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
        let need_slots = rs_slot_ids.iter().copied().max().map_or(1, |m| m + 1);
        gdn_state.ensure_slots(need_slots, &alloc, &stream)?;
        for (r, &slot) in rs_slot_ids.iter().enumerate() {
            if rs_flags.get(r).copied().unwrap_or(0) & driver_abi::local::PIE_RS_FLAG_RESET
                == 0
            {
                continue;
            }
            // Zero the slot's conv + recurrent regions on every slab.
            for slab in gdn_state.slabs.iter().flatten() {
                use cudarc::runtime::sys::{cudaError, cudaMemsetAsync};
                let conv_bytes = gdn_state.conv_stride_elems as usize * 2;
                let st_bytes =
                    gdn_state.state_stride_elems as usize * gdn_state.state_elem_bytes;
                for (buf, stride) in [(&slab.0, conv_bytes), (&slab.1, st_bytes)] {
                    let code = unsafe {
                        cudaMemsetAsync(
                            buf.as_ptr().cast::<u8>().add(slot as usize * stride).cast(),
                            0,
                            stride,
                            stream.as_ref().as_raw().cast(),
                        )
                    };
                    if code != cudaError::cudaSuccess {
                        return Err(PIE_STATUS_DRIVER_ERROR);
                    }
                }
            }
        }
        let slot_ids_h: Vec<i32> =
            rs_slot_ids.iter().map(|&u| i32::try_from(u).unwrap_or(0)).collect();
        let bytes: Vec<u8> = slot_ids_h.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut sbuf = alloc.alloc(bytes.len().max(4)).map_err(|_| PIE_STATUS_EXHAUSTED)?;
        sbuf.copy_from_host(&bytes, stream.as_ref())
            .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        let to_i32 = |v: u32| i32::try_from(v).unwrap_or(0);
        gdn_ctx = Some(GdnCtx {
            k_h: to_i32(facts.gdn.key_heads),
            v_h: to_i32(facts.gdn.value_heads),
            k_d: to_i32(facts.gdn.key_head_dim),
            v_d: to_i32(facts.gdn.value_head_dim),
            conv_dim: to_i32(facts.gdn.conv_dim()),
            conv_k: to_i32(facts.gdn.conv_kernel),
            n_groups: 0,
            conv_state: gdn_state
                .slabs
                .iter()
                .map(|sl| sl.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
                .collect(),
            conv_stride_elems: gdn_state.conv_stride_elems,
            recurrent_state: gdn_state
                .slabs
                .iter()
                .map(|sl| sl.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
                .collect(),
            state_stride_elems: gdn_state.state_stride_elems,
            slot_ids_d: sbuf.as_ptr().cast(),
            write_state: true,
        });
        _slot_ids_buf = Some(sbuf);
    }
    let lse = alloc
        .alloc(rows * model.hf.num_attention_heads as usize * 4)
        .map_err(|_| PIE_STATUS_EXHAUSTED)?;

    // The guard-owned attention values, discovered from the lowering as
    // the smokes discovered them. gemma-4 has NONE: both its attention
    // forms state [q, o] as SSA args, so the pins stay null.
    let (q_pin, o_off) = if matches!(&family, FamilyFacts::Gemma4(..)) {
        (None, None)
    } else {
        let dispatch_name = if states_decode_dispatch {
            "attn::dispatch_attention_flashinfer_decode"
        } else {
            "attn::dispatch_attention_flashinfer_prefill_bf16"
        };
        let fi = lowered
            .launches
            .iter()
            .position(|x| lowered.kernels[x.kernel as usize] == dispatch_name)
            .ok_or(PIE_STATUS_UNSUPPORTED)?;
        let q_pin = lowered.launches[fi]
            .args
            .clone()
            .find_map(|ai| match &lowered.args[ai as usize] {
                Arg::Named { value, .. } => Some(*value),
                _ => None,
            });
        let o_off = match &lowered.args[lowered.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => *at,
            _ => return Err(PIE_STATUS_UNSUPPORTED),
        };
        (q_pin, Some(o_off))
    };

    struct LiveResolver<'a> {
        model: &'a LoadedModel,
        named: &'a std::collections::BTreeMap<ValueId, crate::cuda::DeviceBuffer>,
    }
    impl Resolver for LiveResolver<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            self.model.weight(name)
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // The family's attention scalars: gemma-4 runs sm_scale 1.0 (the
    // q/k norms carry the scaling), per-layer windows (sliding at
    // `sliding_window`, full unbounded), and needs the HOST CSR mirrors
    // for its planless prefill.
    let (sm_scale, window_by_layer, is_gemma4) = match &family {
        FamilyFacts::Gemma4(facts, _) => (
            1.0,
            (0..facts.layers)
                .map(|l| {
                    if facts.is_full_attn(l) { -1 } else { model.hf.sliding_window.max(0) }
                })
                .collect(),
            true,
        ),
        _ => (1.0 / (model.hf.head_dim as f32).sqrt(), Vec::new(), false),
    };
    let attn = AttnCtx {
        decode_plan: decode_plan.as_ptr(),
        decode_plan_full: decode_plan_full_ptr,
        prefill_plan: prefill_plan.as_ptr(),
        workspace: ws.view(),
        layers,
        q_out: q_pin
            .and_then(|v| named_bufs.get(&v).map(|b| b.as_ptr()))
            .unwrap_or(core::ptr::null_mut()),
        o_out: o_off
            .map_or(core::ptr::null_mut(), |off| unsafe {
                arena.as_ptr().cast::<u8>().add(off)
            }
            .cast()),
        kv_page_indices_d: d_kv_indices.as_ptr().cast(),
        kv_page_indptr_d: d_kv_indptr.as_ptr().cast(),
        kv_last_page_lens_d: d_kv_lens.as_ptr().cast(),
        qo_indptr_d: d_qo.as_ptr().cast(),
        qo_indptr_h: if is_gemma4 { qo_indptr.as_ptr() } else { core::ptr::null() },
        kv_page_indptr_h: if is_gemma4 { kv_indptr.as_ptr() } else { core::ptr::null() },
        num_requests: requests as i32,
        num_pages_in_batch: kv_indices.len() as i32,
        first_token: 0,
        w_page_d: d_w_page.as_ptr().cast(),
        w_off_d: d_w_off.as_ptr().cast(),
        row_valid_d: d_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: window_by_layer,
        logits_soft_cap: 0.0,
        sm_scale,
    };

    let mut cublas_ops = crate::cuda::cublas::LiveCublas;
    let mut cublas = crate::cuda::cublas::CublasHandle::create(&mut cublas_ops, raw_stream)
        .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    // gemma-4's per-layer tables and named constants — the C++ parse-time
    // vectors (`per_layer_rope_theta`, `rotary_of`) and the prologue's
    // four `scale.*` values plus the load-read layer scalars.
    let (theta_by_layer, rotary_by_layer, softcap, ple_dim, scales) = match &family {
        FamilyFacts::Gemma4(facts, _) => {
            let hf = &model.hf;
            let theta: Vec<f32> = (0..facts.layers as usize)
                .map(|l| {
                    hf.gemma_per_layer_rope_theta.get(l).copied().unwrap_or({
                        // The C++ parse fallback: full layers (and configs
                        // without a local base) ride `rope_theta`.
                        if facts.is_full_attn(l as u32)
                            || hf.gemma3n_rope_local_base_freq <= 0.0
                        {
                            hf.rope_theta
                        } else {
                            hf.gemma3n_rope_local_base_freq
                        }
                    })
                })
                .collect();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let rotary: Vec<u32> = (0..facts.layers)
                .map(|l| {
                    let f = hf
                        .gemma_per_layer_partial_rotary_factor
                        .get(l as usize)
                        .copied()
                        .unwrap_or(1.0);
                    let d = facts.head_dim_of(l) as f32;
                    2u32.max(2 * (0.5 * f * d) as u32)
                })
                .collect();
            let mut scales = std::collections::BTreeMap::new();
            let hidden = facts.hidden as f32;
            scales.insert("sqrt_hidden".into(), hidden.sqrt());
            scales.insert("sqrt_ple_dim".into(), (facts.ple_dim as f32).sqrt());
            scales.insert("rsqrt_hidden".into(), 1.0 / hidden.sqrt());
            scales.insert("rsqrt_2".into(), 1.0 / 2f32.sqrt());
            for (n, s) in model.gemma_layer_scalars.iter().enumerate() {
                scales.insert(format!("layer.{n}.ple_norm"), *s);
            }
            (theta, rotary, facts.logit_softcap, facts.ple_dim as i32, scales)
        }
        _ => (Vec::new(), Vec::new(), 0.0, 0, std::collections::BTreeMap::new()),
    };
    // The peel window word, uploaded before the walk so a tail region's
    // `_devwin` launch reads a split rather than whatever was there. The
    // engine does not yet mark rows, so the window is the whole fire —
    // which is what an unpeeled fire means and what the lowering's own
    // prefix/tail split degenerates to.
    let mut peel_win =
        crate::cuda::PeelWindowWord::new(&alloc).map_err(|_| PIE_STATUS_EXHAUSTED)?;
    peel_win.set(0, u32::try_from(rows).unwrap_or(0));
    peel_win.upload(stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;

    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: model.hf.rms_norm_eps,
        rope_theta: model.hf.rope_theta,
        rope_theta_by_layer: theta_by_layer,
        rotary_by_layer,
        head_dim: model.hf.head_dim,
        num_q_heads: model.hf.num_attention_heads,
        num_kv_heads: model.hf.num_key_value_heads,
        vocab: model.hf.vocab_size,
        gate_second: false,
        rope_interleaved: false,
        token_ids: d_ids.as_ptr(),
        positions: d_pos.as_ptr(),
        final_logit_softcap: softcap,
        ple_dim,
        scales,
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
        // The fire's peel window, published so a `_devwin` statement in a
        // tail region can early-out per lane. The prefix is the rows that
        // do NOT carry the axis's mark, so the tail begins where the
        // marked suffix does; with no marked rows there is no split and
        // the word says the whole fire.
        peel_window: peel_win.device_ptr(),
        rows_total: i32::try_from(rows).unwrap_or(0),
    };

    let mut resolver = LiveResolver { model, named: &named_bufs };
    let result =
        run(&lowered, &dplan, exec_frame, &mut resolver, &ctx, Some(&attn), gdn_ctx.as_ref());
    let sync = stream.as_ref().synchronize();
    cublas.release(&mut cublas_ops);
    match (result, sync) {
        (Ok(_), Ok(())) => {}
        (Err(e), _) => {
            eprintln!("[driver-cuda-new] launch: refused: {e:?}");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
        (_, Err(e)) => {
            eprintln!("[driver-cuda-new] launch: stream: {e:?}");
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
    }

    // ── Delivery: the LAST row's logits, out through the instance's
    // reader channel. The convention until the launch package's channel
    // table is parsed: the roster's first instance, its first registered
    // channel with `host_role == READER` whose cell is `[vocab]` f32.
    // Device bf16 widens to the f32 wire on the host.
    let logits_value = (0..lowered.launches.len())
        .rev()
        .find_map(|i| {
            dplan.spec(i).outs.first().and_then(|a| match a {
                Arg::Named { value, .. } => Some(*value),
                Arg::Arena { .. } | Arg::Weight(_) => None,
            })
        });
    let instance_ids = slice_of(frame.instance_ids.ptr, frame.instance_ids.len);
    if let (Some(lv), Some(&iid)) = (logits_value, instance_ids.first())
        && let Some(inst) = state.instances.get(&iid)
    {
        let vocab = usize::try_from(model.hf.vocab_size).unwrap_or(0);
        let target = inst.channel_ids.iter().find_map(|cid| {
            state.channels.get(cid).filter(|ch| {
                ch.host_role == driver_abi::local::PIE_CHANNEL_HOST_ROLE_READER
                    && ch.cell_bytes == vocab * 4
            })
        });
        if let (Some(ch), Some(buf)) = (target, named_bufs.get(&lv)) {
            let mut bf16 = vec![0u8; buf.len()];
            buf.copy_to_host(&mut bf16, stream.as_ref())
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            let last = rows - 1;
            let mut cell = vec![0u8; vocab * 4];
            for t in 0..vocab {
                let off = (last * vocab + t) * 2;
                let bits = u16::from_le_bytes([bf16[off], bf16[off + 1]]);
                cell[t * 4..t * 4 + 4]
                    .copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
            }
            if !ch.publish(&cell) {
                eprintln!("[driver-cuda-new] launch: logits ring full; frame dropped its output");
            }
        }
    }
    Ok(())
}

/// Awaits: the MULTIMODAL encoders — image/audio features to embedding
/// rows (the vision/audio towers, which stayed hand-written C++). The
/// plan once mislabeled this as the Sampling-IR path; the desc's fields
/// (`image_pixels`, `audio_features`, `output_rows`) say what it is. A
/// text-only shell refuses it honestly.
/// The audio half of the encode arm: `bind_gemma4_audio`'s name map as
/// the stride-62 table (`vision/gemma4_towers_c.hpp`'s layout), the
/// media row width from the embed projection's own bytes, and the
/// `PieEncodeDesc` audio slices passed straight through.
fn encode_gemma4_audio_arm(
    model: &LoadedModel,
    desc: &PieEncodeDesc,
    out_ptr: *mut std::ffi::c_void,
    out_bytes: usize,
    indptr_ptr: *mut u32,
) -> i32 {
    let Some(ac) = model.hf.gemma_audio.as_ref() else {
        eprintln!("[driver-cuda-new] encode: this deployment carries no audio tower");
        return PIE_STATUS_UNSUPPORTED;
    };
    let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
        model.weights.get(n).map(|b| b.as_ptr().cast_const()).ok_or_else(|| {
            eprintln!("[driver-cuda-new] encode: missing audio weight {n}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let opt = |n: String| -> *const std::ffi::c_void {
        model.weights.get(&n).map_or(core::ptr::null(), |b| b.as_ptr().cast_const())
    };
    let ap = "model.audio_tower";
    let g = |n: &str| need(&format!("{ap}.{n}"));
    let (sscp0_conv, sscp0_norm, sscp1_conv, sscp1_norm, sscp_proj, out_w, out_b, embed) = match (
        g("subsample_conv_projection.layer0.conv.weight"),
        g("subsample_conv_projection.layer0.norm.weight"),
        g("subsample_conv_projection.layer1.conv.weight"),
        g("subsample_conv_projection.layer1.norm.weight"),
        g("subsample_conv_projection.input_proj_linear.weight"),
        g("output_proj.weight"),
        g("output_proj.bias"),
        need("model.embed_audio.embedding_projection.weight"),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f), Ok(gp), Ok(h)) => (a, b, c, d, e, f, gp, h),
        _ => return PIE_STATUS_UNSUPPORTED,
    };
    let depth = usize::try_from(ac.num_hidden_layers).unwrap_or(0);
    let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(depth * 62);
    for l in 0..depth {
        let lp = format!("{ap}.layers.{l}");
        let clip = |base: String, table: &mut Vec<*const std::ffi::c_void>| -> Result<(), i32> {
            table.push(need(&format!("{base}.linear.weight"))?);
            for m in ["input_min", "input_max", "output_min", "output_max"] {
                table.push(opt(format!("{base}.{m}")));
            }
            Ok(())
        };
        let ffn = |base: String, table: &mut Vec<*const std::ffi::c_void>| -> Result<(), i32> {
            table.push(need(&format!("{base}.pre_layer_norm.weight"))?);
            table.push(need(&format!("{base}.post_layer_norm.weight"))?);
            clip(format!("{base}.ffw_layer_1"), table)?;
            clip(format!("{base}.ffw_layer_2"), table)?;
            Ok(())
        };
        let r: Result<(), i32> = (|| {
            ffn(format!("{lp}.feed_forward1"), &mut table)?;
            ffn(format!("{lp}.feed_forward2"), &mut table)?;
            table.push(need(&format!("{lp}.norm_pre_attn.weight"))?);
            table.push(need(&format!("{lp}.norm_post_attn.weight"))?);
            clip(format!("{lp}.self_attn.q_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.k_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.v_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.post"), &mut table)?;
            table.push(need(&format!("{lp}.self_attn.relative_k_proj.weight"))?);
            table.push(need(&format!("{lp}.self_attn.per_dim_scale"))?);
            table.push(need(&format!("{lp}.lconv1d.pre_layer_norm.weight"))?);
            table.push(need(&format!("{lp}.lconv1d.conv_norm.weight"))?);
            clip(format!("{lp}.lconv1d.linear_start"), &mut table)?;
            clip(format!("{lp}.lconv1d.linear_end"), &mut table)?;
            table.push(need(&format!("{lp}.lconv1d.depthwise_conv1d.weight"))?);
            table.push(need(&format!("{lp}.norm_out.weight"))?);
            Ok(())
        })();
        if let Err(e) = r {
            return e;
        }
    }
    let text_hidden = model
        .weights
        .get("model.embed_audio.embedding_projection.weight")
        .map_or(0, |b| b.len() / (usize::try_from(ac.output_proj_dims.max(1)).unwrap_or(1) * 2));
    let Ok(stream) = crate::cuda::OwnedStream::new(0) else {
        return PIE_STATUS_DRIVER_ERROR;
    };
    unsafe {
        crate::launch::ffi::pie_k_vision_gemma4_audio_encode(
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_proj,
            out_w,
            out_b,
            embed,
            table.as_ptr(),
            ac.num_hidden_layers,
            ac.hidden_size,
            ac.num_attention_heads,
            ac.conv_kernel_size,
            ac.feature_size,
            ac.subsampling_conv_channels0,
            ac.subsampling_conv_channels1,
            ac.output_proj_dims,
            i32::try_from(text_hidden).unwrap_or(0),
            ac.attention_chunk_size,
            ac.attention_context_left,
            ac.attention_context_right,
            ac.attention_logit_cap,
            ac.residual_weight,
            ac.rms_norm_eps,
            desc.audio_features.ptr.cast(),
            desc.audio_feature_indptr.ptr,
            desc.audio_anchor_rows.ptr,
            i32::try_from(desc.audio_anchor_rows.len).unwrap_or(0),
            out_ptr.cast(),
            out_bytes,
            indptr_ptr,
            stream.as_ref().as_raw().cast(),
        );
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    PIE_STATUS_OK
}

/// The MULTIMODAL encode: image/audio media in, embedding rows out —
/// the towers behind `vision::gemma4_*_encode`. One media kind per call
/// today; mixed batches await the offset plumbing.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_encode(
    driver: *mut PieDriver,
    encode: *const PieEncodeDesc,
    completion: PieCompletion,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { encode.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(model) = state.model.as_ref() else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let num_images = desc.image_anchor_rows.len;
    let num_clips = desc.audio_anchor_rows.len;
    if num_images == 0 && num_clips == 0 {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if desc.output_row_indptr.len < num_images + num_clips + 1 {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let notify_done = |state: &Shell| {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if let Some(notify) = state.notify {
            unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
        }
    };
    if num_images == 0 {
        // Audio only: the helper writes the whole CSR itself.
        let st = encode_gemma4_audio_arm(
            model,
            desc,
            desc.output_rows.ptr.cast(),
            desc.output_rows.len,
            desc.output_row_indptr.ptr,
        );
        if st != PIE_STATUS_OK {
            return st;
        }
        notify_done(state);
        return PIE_STATUS_OK;
    }
    let Some(vc) = model.hf.gemma_vision.as_ref() else {
        eprintln!("[driver-cuda-new] encode: this deployment carries no vision tower");
        return PIE_STATUS_UNSUPPORTED;
    };
    // The vision table, `vision/gemma4_towers_c.hpp`'s stride-41 layout,
    // built per call from the loaded weights — name lookups, no stored
    // pointers. The binder mapping is `bind_gemma4_vision`'s.
    let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
        model.weights.get(n).map(|b| b.as_ptr().cast_const()).ok_or_else(|| {
            eprintln!("[driver-cuda-new] encode: missing vision weight {n}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let opt = |n: String| -> *const std::ffi::c_void {
        model.weights.get(&n).map_or(core::ptr::null(), |b| b.as_ptr().cast_const())
    };
    let vp = "model.vision_tower";
    let patch_w = match need(&format!("{vp}.patch_embedder.input_proj.weight")) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let pos_table = match need(&format!("{vp}.patch_embedder.position_embedding_table")) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let embed_proj = match need("model.embed_vision.embedding_projection.weight") {
        Ok(p) => p,
        Err(e) => return e,
    };
    let depth = usize::try_from(vc.num_hidden_layers).unwrap_or(0);
    let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(depth * 41);
    for l in 0..depth {
        let lp = format!("{vp}.encoder.layers.{l}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            match need(&format!("{lp}.{norm}.weight")) {
                Ok(p) => table.push(p),
                Err(e) => return e,
            }
        }
        for norm in ["self_attn.q_norm", "self_attn.k_norm"] {
            match need(&format!("{lp}.{norm}.weight")) {
                Ok(p) => table.push(p),
                Err(e) => return e,
            }
        }
        for clip in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            match need(&format!("{lp}.{clip}.linear.weight")) {
                Ok(p) => table.push(p),
                Err(e) => return e,
            }
            for m in ["input_min", "input_max", "output_min", "output_max"] {
                table.push(opt(format!("{lp}.{clip}.{m}")));
            }
        }
    }
    // pos_table is `[2, S, hidden]` bf16 — S from the buffer itself; the
    // media row width from the projection (`[text_hidden, hidden]`).
    let hidden = usize::try_from(vc.hidden_size.max(1)).unwrap_or(1);
    let pos_table_size = model
        .weights
        .get(&format!("{vp}.patch_embedder.position_embedding_table"))
        .map_or(0, |b| b.len() / (2 * hidden * 2));
    let text_hidden = model
        .weights
        .get("model.embed_vision.embedding_projection.weight")
        .map_or(0, |b| b.len() / (hidden * 2));

    let Ok(stream) = crate::cuda::OwnedStream::new(0) else {
        return PIE_STATUS_DRIVER_ERROR;
    };
    let mut vis_bounds = vec![0u32; num_images + 1];
    unsafe {
        crate::launch::ffi::pie_k_vision_gemma4_vision_encode(
            patch_w,
            pos_table,
            embed_proj,
            table.as_ptr(),
            vc.num_hidden_layers,
            vc.hidden_size,
            vc.num_attention_heads,
            vc.intermediate_size,
            i32::try_from(pos_table_size).unwrap_or(0),
            i32::try_from(text_hidden).unwrap_or(0),
            vc.pooling_kernel_size,
            vc.rms_norm_eps,
            vc.rope_theta,
            desc.image_pixels.ptr.cast(),
            desc.image_pixel_indptr.ptr,
            desc.image_patch_positions.ptr,
            desc.image_anchor_rows.ptr,
            i32::try_from(num_images).unwrap_or(0),
            desc.output_rows.ptr.cast(),
            desc.output_rows.len,
            vis_bounds.as_mut_ptr(),
            stream.as_ref().as_raw().cast(),
        );
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    // Compose the shared CSR the C++ `Context::encode` writes: the
    // vision segment's boundaries verbatim, then the audio segment's
    // shifted by the vision row count.
    let indptr = desc.output_row_indptr.ptr;
    unsafe {
        for (i, b) in vis_bounds.iter().enumerate() {
            *indptr.add(i) = *b;
        }
    }
    if num_clips > 0 {
        let row_offset = *vis_bounds.last().unwrap_or(&0) as usize;
        let consumed = row_offset * text_hidden * 2;
        if consumed > desc.output_rows.len {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let mut audio_bounds = vec![0u32; num_clips + 1];
        let st = encode_gemma4_audio_arm(
            model,
            desc,
            unsafe { desc.output_rows.ptr.add(consumed) }.cast(),
            desc.output_rows.len - consumed,
            audio_bounds.as_mut_ptr(),
        );
        if st != PIE_STATUS_OK {
            return st;
        }
        unsafe {
            for c in 0..num_clips {
                *indptr.add(num_images + 1 + c) =
                    u32::try_from(row_offset).unwrap_or(u32::MAX) + audio_bounds[c + 1];
            }
        }
    }
    notify_done(state);
    PIE_STATUS_OK
}

/// KV copies within the device domain: whole-page copies (`src_page_ids`
/// → `dst_page_ids`, every layer, both planes) and beam-repair CELL
/// moves through the bridged `copy_kv_cells_bf16`. Host-pinned domains
/// refuse until the swap pool wires in — a swap, not a copy, and its
/// store is ported but not yet mounted here.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_copy_kv(
    driver: *mut PieDriver,
    copy: *const PieKvCopyDesc,
    completion: PieCompletion,
) -> i32 {
    use driver_abi::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE;

    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { copy.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let host_src = desc.src_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    let host_dst = desc.dst_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    if host_src && host_dst {
        eprintln!("[driver-cuda-new] copy_kv: host-to-host moves have no device leg");
        return PIE_STATUS_UNSUPPORTED;
    }
    let (Some(model), Some(_kv)) = (state.model.as_ref(), state.kv.as_ref()) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let src_pages = slice_of(desc.src_page_ids.ptr, desc.src_page_ids.len);
    let dst_pages = slice_of(desc.dst_page_ids.ptr, desc.dst_page_ids.len);
    if src_pages.len() != dst_pages.len() {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let cells = slice_of(desc.cells.ptr, desc.cells.len);
    if (host_src || host_dst) && !cells.is_empty() {
        return PIE_STATUS_INVALID_ARGUMENT; // cell moves are device-only
    }
    let (kv_heads, head_dim) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let page_size: i32 = 16;
    let page_bytes = page_size as usize * kv_heads as usize * head_dim as usize * 2;
    let layers_n = model.hf.num_hidden_layers as usize;

    // Non-uniform pools (gemma-4: per-layer head dims, shared layers
    // owning no pages) fit the DEVICE legs below — the per-layer bytes
    // derive from each pool — but the host swap pool is a uniform-stride
    // store. Refuse the host domains there until it learns per-layer
    // strides; a wrong-sized host page is corruption, not degradation.
    {
        let kv_ref = state.kv.as_ref().expect("checked");
        let uniform = kv_ref
            .pools
            .iter()
            .all(|p| p.as_ref().is_some_and(|(k, _)| k.len() == page_bytes * kv_ref.num_pages as usize));
        if !uniform && (host_src || host_dst) {
            eprintln!("[driver-cuda-new] copy_kv: host swap awaits per-layer strides");
            return PIE_STATUS_UNSUPPORTED;
        }
    }

    // The swap pool: ensured to cover the highest HOST page id touched.
    if host_src || host_dst {
        use crate::model::attention_workspace::{LiveStagingOps, StagingOps};
        let host_ids = if host_src { src_pages } else { dst_pages };
        let need = host_ids.iter().copied().max().map_or(1, |m| m + 1);
        let grow = !matches!(&state.swap, Some(sp) if sp.num_pages >= need
            && sp.page_bytes == page_bytes);
        if grow {
            let mut ops = LiveStagingOps;
            let mut blocks = Vec::new();
            for _ in 0..layers_n {
                let Some(b) = ops.malloc_host(2 * need as usize * page_bytes) else {
                    for &p in &blocks {
                        ops.free_host(p);
                    }
                    return PIE_STATUS_EXHAUSTED;
                };
                blocks.push(b);
            }
            if let Some(old) = state.swap.take() {
                // Migrate what fits: the retained host pages keep their ids.
                let keep = old.num_pages.min(need) as usize * old.page_bytes;
                if old.page_bytes == page_bytes {
                    for (l, &nb) in blocks.iter().enumerate() {
                        for plane in 0..2usize {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    old.page(l, plane, 0),
                                    nb.cast::<u8>()
                                        .add(plane * need as usize * page_bytes),
                                    keep,
                                );
                            }
                        }
                    }
                }
                old.free();
            }
            state.swap = Some(SwapPool { blocks, num_pages: need, page_bytes });
        }
    }

    let stream = match crate::cuda::OwnedStream::new(0) {
        Ok(s) => s,
        Err(_) => return PIE_STATUS_DRIVER_ERROR,
    };
    use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
    let kv_ref = state.kv.as_ref().expect("checked");
    for (s_id, d_id) in src_pages.iter().zip(dst_pages) {
        if (!host_src && *s_id >= kv_ref.num_pages)
            || (!host_dst && *d_id >= kv_ref.num_pages)
        {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        for (l, pools) in kv_ref.pools.iter().enumerate() {
            // A layer that owns no pages has none to move.
            let Some((k, v)) = pools.as_ref() else { continue };
            // THIS layer's page bytes — the pool's own stride, so the
            // two-head-dim families move the right amount per layer.
            let pb = k.len() / kv_ref.num_pages as usize;
            for (plane, pool) in [k, v].into_iter().enumerate() {
                let dev = pool.as_ptr().cast::<u8>();
                let (dst, src, kind) = if host_dst {
                    (
                        state.swap.as_ref().expect("ensured").page(l, plane, *d_id),
                        unsafe { dev.add(*s_id as usize * pb) },
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                } else if host_src {
                    (
                        unsafe { dev.add(*d_id as usize * pb) },
                        state.swap.as_ref().expect("ensured").page(l, plane, *s_id),
                        cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )
                } else {
                    (
                        unsafe { dev.add(*d_id as usize * pb) },
                        unsafe { dev.add(*s_id as usize * pb) },
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    )
                };
                let code = unsafe {
                    cudaMemcpyAsync(dst.cast(), src.cast_const().cast(), pb, kind,
                        stream.as_ref().as_raw())
                };
                if code != cudaError::cudaSuccess {
                    return PIE_STATUS_DRIVER_ERROR;
                }
            }
        }
    }
    // Cell moves: the bridged beam-repair launcher, per layer. Disjoint
    // spans are the CALLER's contract, as the kernel's header states.
    if !cells.is_empty() {
        let alloc = crate::cuda::Allocator::new();
        let up = |vals: &[u32]| -> Result<crate::cuda::DeviceBuffer, i32> {
            let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            let mut b = alloc.alloc(bytes.len()).map_err(|_| PIE_STATUS_EXHAUSTED)?;
            b.copy_from_host(&bytes, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            Ok(b)
        };
        let dp: Vec<u32> = cells.iter().map(|c| c.dst_page_id).collect();
        let doff: Vec<u32> = cells.iter().map(|c| c.dst_token_offset).collect();
        let sp: Vec<u32> = cells.iter().map(|c| c.src_page_id).collect();
        let soff: Vec<u32> = cells.iter().map(|c| c.src_token_offset).collect();
        let (d_dp, d_doff, d_sp, d_soff) = match (up(&dp), up(&doff), up(&sp), up(&soff)) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => return PIE_STATUS_EXHAUSTED,
        };
        for (i, pools) in kv_ref.pools.iter().enumerate() {
            let Some((k, v)) = pools.as_ref() else { continue };
            // THIS layer's head dim, derived from its own pool — the
            // two-head-dim families' rows disagree and the stride must
            // follow the pool, not the config's single number.
            let d = (k.len()
                / kv_ref.num_pages as usize
                / page_size as usize
                / kv_heads.max(1) as usize
                / 2) as i32;
            let layer = crate::launch::KvCacheLayerView {
                layer: i as i32,
                source_layer: i as i32,
                num_pages: kv_ref.num_pages as i32,
                page_size,
                num_kv_heads: kv_heads,
                head_dim: d,
                scheme: crate::launch::KvCacheScheme::Native,
                storage_dtype: crate::dtype::DType::Bf16,
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
            };
            unsafe {
                crate::launch::ffi::pie_k_attn_copy_kv_cells_bf16(
                    layer,
                    d_dp.as_ptr().cast(),
                    d_doff.as_ptr().cast(),
                    d_sp.as_ptr().cast(),
                    d_soff.as_ptr().cast(),
                    i32::try_from(cells.len()).unwrap_or(i32::MAX),
                    stream.as_ref().as_raw().cast(),
                );
            }
        }
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    if let Some(notify) = state.notify {
        unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
    }
    PIE_STATUS_OK
}

/// Direct recurrent-state copies: WHOLE-SLOT d2d over the hybrid's GDN
/// slabs (conv + recurrent, every linear layer), the C++ shape
/// (`context.cpp` ignores the token fields — those ride for the rs
/// BUFFER pool, spec-decode machinery). Slot ids are the engine's; the
/// slabs grow with migration to cover them.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_copy_state(
    driver: *mut PieDriver,
    copy: *const PieStateCopyDesc,
    _completion: PieCompletion,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { copy.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(gdn) = state.gdn.as_mut() else {
        // No recurrent family is loaded — the C++ shape: state copies
        // only mean something once the rs cache exists.
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let ranges = slice_of(desc.slot_ranges.ptr, desc.slot_ranges.len);
    let Ok(stream) = crate::cuda::OwnedStream::new(0) else {
        return PIE_STATUS_DRIVER_ERROR;
    };
    let alloc = crate::cuda::Allocator::new();
    let need = ranges
        .iter()
        .map(|r| r.src_slot_id.max(r.dst_slot_id) + 1)
        .max()
        .unwrap_or(0);
    if let Err(code) = gdn.ensure_slots(need, &alloc, &stream) {
        return code;
    }
    use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
    for range in ranges {
        // The C++ (`context.cpp::copy_state`) copies WHOLE SLOTS
        // (`rs_cache->copy_slot_d2d(src, dst)`) — the token fields ride
        // for the rs BUFFER pool, which is spec-decode machinery.
        let conv_bytes = gdn.conv_stride_elems as usize * 2;
        let st_bytes = gdn.state_stride_elems as usize * gdn.state_elem_bytes;
        for slab in gdn.slabs.iter().flatten() {
            for (buf, stride) in [(&slab.0, conv_bytes), (&slab.1, st_bytes)] {
                let code = unsafe {
                    cudaMemcpyAsync(
                        buf.as_ptr()
                            .cast::<u8>()
                            .add(range.dst_slot_id as usize * stride)
                            .cast(),
                        buf.as_ptr()
                            .cast::<u8>()
                            .add(range.src_slot_id as usize * stride)
                            .cast_const()
                            .cast(),
                        stride,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream.as_ref().as_raw().cast(),
                    )
                };
                if code != cudaError::cudaSuccess {
                    return PIE_STATUS_DRIVER_ERROR;
                }
            }
        }
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    PIE_STATUS_OK
}

/// Resize the KV pool to `target_pages`, MIGRATING the surviving pages —
/// the migration the launch-time growth deliberately skipped. Shrinks
/// drop the tail; `map_ranges`/`unmap_ranges` (the elastic-VMM form) are
/// accepted but the shell's pools are plain allocations, so the target
/// page count is the whole contract here — stated, not hidden.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_resize_pool(
    driver: *mut PieDriver,
    resize: *const PiePoolResizeDesc,
    completion: PieCompletion,
) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Some(desc) = (unsafe { resize.as_ref() }) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let Ok(target) = u32::try_from(desc.target_pages) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    if target == 0 {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let Some(model) = state.model.as_ref() else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    let (kv_heads, head_dim) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let page_size: usize = 16;
    let page_bytes = page_size * kv_heads as usize * head_dim as usize * 2;

    let stream = match crate::cuda::OwnedStream::new(0) {
        Ok(s) => s,
        Err(_) => return PIE_STATUS_DRIVER_ERROR,
    };
    let alloc = crate::cuda::Allocator::new();
    let old = state.kv.take();
    // Per-layer page bytes: an existing pool states its own stride (the
    // two-head-dim families' rows disagree); before any pool exists the
    // config decides — gemma-4 by its layer kinds and shared tail, every
    // other family uniformly.
    let is_gemma4 =
        model.weights.contains_key("model.language_model.embed_tokens_per_layer.weight");
    let n_layers = model.hf.num_hidden_layers as usize;
    let first_shared =
        n_layers.saturating_sub(usize::try_from(model.hf.num_kv_shared_layers).unwrap_or(0));
    let layer_page_bytes = |i: usize| -> Option<usize> {
        if let Some(old_kv) = &old {
            return old_kv.pools[i]
                .as_ref()
                .map(|(k, _)| k.len() / old_kv.num_pages.max(1) as usize);
        }
        if !is_gemma4 {
            return Some(page_bytes);
        }
        if i >= first_shared {
            return None;
        }
        let full = model.hf.layer_types.get(i).is_some_and(|t| t == "full_attention");
        let d = if full {
            model.hf.gemma4_global_head_dim.max(model.hf.head_dim)
        } else {
            model.hf.head_dim
        };
        Some(page_size * kv_heads as usize * d as usize * 2)
    };
    let mut pools = Vec::new();
    for i in 0..n_layers {
        let Some(pb) = layer_page_bytes(i) else {
            pools.push(None);
            continue;
        };
        let Ok(mut k) = alloc.alloc(target as usize * pb) else {
            return PIE_STATUS_EXHAUSTED;
        };
        let Ok(mut v) = alloc.alloc(target as usize * pb) else {
            return PIE_STATUS_EXHAUSTED;
        };
        if k.memset(0, stream.as_ref()).is_err() || v.memset(0, stream.as_ref()).is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        if let Some(old_kv) = &old
            && let Some((ok_, ov)) = &old_kv.pools[i]
        {
            let keep = old_kv.num_pages.min(target) as usize * pb;
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            for (dst, src) in [(&mut k, ok_), (&mut v, ov)] {
                let code = unsafe {
                    cudaMemcpyAsync(
                        dst.as_ptr(),
                        src.as_ptr().cast_const(),
                        keep,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream.as_ref().as_raw(),
                    )
                };
                if code != cudaError::cudaSuccess {
                    return PIE_STATUS_DRIVER_ERROR;
                }
            }
        }
        pools.push(Some((k, v)));
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    state.kv = Some(KvState { pools, num_pages: target });
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    if let Some(notify) = state.notify {
        unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
    }
    PIE_STATUS_OK
}

/// Close an instance — idempotently, the C++'s reading: closing what is
/// not open is not an error.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_close_instance(driver: *mut PieDriver, instance_id: u64) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    state.instances.remove(&instance_id);
    PIE_STATUS_OK
}

/// Close a channel — idempotently, freeing its pinned endpoint.
#[unsafe(no_mangle)]
pub extern "C" fn pie_cuda_close_channel(driver: *mut PieDriver, channel_id: u64) -> i32 {
    let Some(state) = shell(driver) else {
        return PIE_STATUS_INVALID_ARGUMENT;
    };
    if let Some(ch) = state.channels.remove(&channel_id) {
        ch.free();
    }
    PIE_STATUS_OK
}

#[cfg(test)]
mod tests {
    /// The boot-TOML extraction, isolated: the exact chain `create` runs.
    #[test]
    fn the_boot_descriptor_extracts() {
        let boot = "[model]\ndescriptor = \"/tmp/x.json\"\n";
        let v = boot.parse::<toml::Table>().expect("parses");
        let path = v
            .get("model")
            .and_then(|m| m.get("descriptor"))
            .and_then(|d| d.as_str())
            .expect("extracts");
        assert_eq!(path, "/tmp/x.json");
    }
}
