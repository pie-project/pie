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
    /// The unionized supergraph's instantiated graphs, one per (R, N)
    /// bucket. Empty unless `PIE_CUDA_SUPERGRAPH` armed it.
    ///
    /// **Declared BEFORE [`Self::fire_arrays`], and that ordering is
    /// load-bearing.** Struct fields drop in declaration order, an exec
    /// holds the addresses it recorded, and those addresses are the fire
    /// arrays — so freeing the arrays first leaves live graph execs
    /// pointing at returned memory, and destroying them then faults.
    ///
    /// Nothing about the types says this; the only thing that says it is
    /// this comment and the order. Which is why it is a comment and not a
    /// convention.
    supergraph: crate::model::supergraph::SupergraphCache,
    /// The per-fire device arrays, pooled so a capture can outlive the
    /// fire that recorded it. See [`FireArrays`]. Dropped AFTER the execs
    /// that address it — see above.
    fire_arrays: FireArrays,
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
    /// THE FIRE STREAM AND ITS ALLOCATOR, held per driver.
    ///
    /// Both were built per fire — `OwnedStream::new(0)` and
    /// `Allocator::new()` at the top of every `step_impl` — which is two
    /// costs and one impossibility.
    ///
    /// The costs: a stream create/destroy per fire, and an allocator that
    /// POOLS discarding its pool every fire, so every buffer a fire wants
    /// is a fresh `cudaMalloc`.
    ///
    /// The impossibility is run-ahead. A second fire cannot be enqueued
    /// behind the first if there is no stream that outlives the first,
    /// and `pie_cuda_launch` cannot return before its work retires if the
    /// stream it queued onto dies with the call. Everything about
    /// n+1-while-n-runs starts here.
    ///
    /// `None` until the first launch, because a driver that never fires
    /// should not hold a stream.
    fire_stream: Option<crate::cuda::OwnedStream>,
    /// The allocator every fire's transient device memory comes from.
    /// Held for the pool, and dropped with the shell.
    fire_alloc: Option<crate::cuda::Allocator>,
    /// The PTIR plane: what a registered program adopted to, and what its
    /// generated regions compiled to.
    ///
    /// Two fields rather than one because they have different lifetimes.
    /// [`crate::ptir::Runtime`] is the CACHE — it outlives any one program
    /// and is what makes the second registration of a shared stage free —
    /// while `ptir_programs` is this shell's OWNERSHIP of the compiled
    /// modules, so closing the last user of a program can drop its
    /// `CUmodule`s at a point the shell chose.
    ptir: crate::ptir::Runtime,
    /// The compiled form of each registered program, by program id.
    ptir_programs: crate::ptir::Programs,
    /// The adopted plans, by program id. Separate from the compiled
    /// modules because a program can be adopted and REJECTED — an
    /// unexecutable plan is still a plan, and the reason it was rejected
    /// is what the launch that needs it has to report — while a
    /// compilation only exists for a program that got that far.
    ptir_plans: std::collections::BTreeMap<u64, driver_pipeline::ExecPlan>,
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
/// The per-fire device arrays, POOLED across fires.
///
/// They used to be allocated and dropped every launch — `step_impl`'s own
/// comment said "KV pools (persistent), fire arrays (per launch)" — and
/// that is the one thing standing between the supergraph and the live
/// path. A captured exec bakes the addresses it recorded, so an arena
/// freed at the end of its fire can never be replayed into.
///
/// So they are kept and reused, and grown when a fire needs more than the
/// last one did. Growth MOVES a base address, which invalidates every
/// capture that recorded it — hence [`Self::epoch`], which is the
/// `PlanEpoch` `model::supergraph::SupergraphCache` keys its execs on. A
/// bump means stale, and stale means recapture rather than a wrong
/// answer.
#[derive(Default)]
struct FireArrays {
    arena: Option<crate::cuda::DeviceBuffer>,
    named: std::collections::BTreeMap<model_compiler::trace::ValueId, crate::cuda::DeviceBuffer>,
    /// The small per-fire u32 descriptor arrays, by slot.
    slots: Vec<Option<crate::cuda::DeviceBuffer>>,
    epoch: u64,
}

impl FireArrays {
    /// The activation arena, at least `bytes` wide.
    fn arena(
        &mut self,
        alloc: &crate::cuda::Allocator,
        bytes: usize,
    ) -> Result<*mut std::ffi::c_void, i32> {
        if self.arena.as_ref().is_none_or(|b| b.len() < bytes) {
            self.arena = Some(alloc.alloc(bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        Ok(self.arena.as_ref().expect("just ensured").as_ptr())
    }

    /// One per-fire u32 descriptor array, by SLOT.
    ///
    /// The same discipline the arena gets, for the small arrays: the
    /// buffer is kept and its CONTENTS refreshed, so a capture that
    /// recorded the address keeps addressing something real. Slots are
    /// positional because these are a fixed list — see the constants
    /// beside the call site.
    ///
    /// Returns the device pointer rather than the buffer, so a caller
    /// holds no borrow and the next slot can be uploaded on the next line.
    fn upload_u32(
        &mut self,
        alloc: &crate::cuda::Allocator,
        slot: usize,
        vals: &[u32],
        stream: crate::cuda::StreamRef<'_>,
    ) -> Result<*const u32, i32> {
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, || None);
        }
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let need = bytes.len().max(4);
        if self.slots[slot].as_ref().is_none_or(|b| b.len() < need) {
            self.slots[slot] = Some(alloc.alloc(need).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        let b = self.slots[slot].as_mut().expect("just ensured");
        b.copy_from_host(&bytes, stream).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        Ok(b.as_ptr().cast_const().cast::<u32>())
    }

    /// One named seam buffer, at least `bytes` wide, zeroed.
    ///
    /// Zeroed on every fire rather than only on allocation: the pin is
    /// per-fire state whatever its storage is, and a reused buffer still
    /// holds the last fire's values.
    fn named(
        &mut self,
        alloc: &crate::cuda::Allocator,
        v: model_compiler::trace::ValueId,
        bytes: usize,
        stream: crate::cuda::StreamRef<'_>,
    ) -> Result<(), i32> {
        let grow = self.named.get(&v).is_none_or(|b| b.len() < bytes);
        if grow {
            self.named
                .insert(v, alloc.alloc(bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        self.named
            .get_mut(&v)
            .expect("just ensured")
            .memset(0, stream)
            .map_err(|_| PIE_STATUS_DRIVER_ERROR)
    }
}

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
        fire_arrays: FireArrays::default(),
        supergraph: crate::model::supergraph::SupergraphCache::new(),
        kv: None,
        gdn: None,
        channels: std::collections::BTreeMap::new(),
        swap: None,
        scratch: None,
        fire_stream: None,
        fire_alloc: None,
        ptir: crate::ptir::Runtime::default(),
        ptir_programs: crate::ptir::Programs::new(),
        ptir_plans: std::collections::BTreeMap::new(),
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
        // WHAT A LOAD CAN HAND THE KERNELS, and nothing else.
        //
        // Two answers, and the difference is whether the driver has to
        // change the bytes.
        //
        // RAW bf16/f32 is what most of a checkpoint is. (f32 is the GDN
        // parameter side of the `gdn_fp32_parameters` contract — `A_log`,
        // the gate norm — consumed as f32 by its kernels.)
        //
        // A BYTE PAYLOAD is also loadable, and finding out why took
        // reading the file. gpt-oss's MXFP4 expert banks are not a
        // `Quant` encoding at all — safetensors stores them as `U8`
        // tensors (`…experts.down_proj_blocks`, `U8 [32, 2880, 90, 16]`,
        // beside a `_scales` companion), and the MXFP4 MEANING lives in
        // the checkpoint's `quantization_config`, which the contract
        // reads. The tensor's dtype says only "bytes".
        //
        // That is the right division and it makes the load's job small:
        // get the bytes on the device unchanged. What they MEAN is the
        // binder's business, and `quant::mxfp4_moe_gate_up_decode_bf16`
        // indexes the stored layout directly.
        //
        // What still refuses is an encoding whose kernels want a
        // DIFFERENT layout than the file has — a Marlin repack, an FP8
        // re-encode, a GGUF block unpack. That is `transcode_engine`'s
        // work in the retired C++ tree and it is not ported, so a
        // checkpoint needing it is turned away at load rather than
        // mis-bound at launch.
        match &t.encoding {
            Encoding::Raw(d)
                if matches!(format!("{d:?}").as_str(), "BF16" | "F32" | "U8") => {}
            Encoding::Quant(spec) if reads_its_stored_form(spec.scheme) => {}
            other => {
                eprintln!(
                    "[driver-cuda-new] load_model: {}: unsupported encoding {other:?}. \
                     Raw bf16/f32 and packed schemes the kernels read as stored \
                     load; anything needing a transcode does not.",
                    t.name
                );
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

/// Register a program: adopt its launch package, compile its generated
/// regions, and answer an id.
///
/// The C3 hash is the dedup key — re-registering answers the existing id
/// without recompiling — which is what makes a program that is bound a
/// thousand times compiled once.
///
/// # What a failure here means, and why it is not always one
///
/// Four outcomes, and only two of them are errors:
///
/// * The descriptor carries NO launch package — an empty stage list. That
///   is a forward-only deployment: the model runs and the logits come
///   back through the instance's reader channel, with no user program
///   around the fire. An id is issued and nothing is adopted. `OK`.
/// * The package adopts and every generated region compiles. `OK`.
/// * The package adopts and the plan is UNEXECUTABLE — a per-layer tap
///   stage, an op this driver does not implement. That is not a driver
///   failure and not a registration failure either: the plan is recorded
///   with its reason, and the refusal surfaces at the launch that needs
///   it, where the caller can see which fire it lost.
/// * A region NVRTC rejects, or an emitted table with a hole in it.
///   `UNSUPPORTED`, and remembered: this driver carries no emitter, so a
///   generated region with no host source has no slower path to fall
///   back to.
///
/// A compile needs a device — the architecture comes from the GPU that
/// will run the code, never a guess — so a shell with no model loaded has
/// not bound one yet and defers the compile to the first launch rather
/// than compiling for an architecture it made up.
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
    if let Some(id) = state
        .programs
        .iter()
        .find(|(_, p)| p.program_hash == desc.program_hash)
        .map(|(&id, _)| id)
    {
        if let Some(out) = unsafe { program_id.as_mut() } {
            *out = id;
        }
        return PIE_STATUS_OK;
    }

    // SAFETY: the engine's contract for `register_program` is that every
    // array reachable from the descriptor is live for the duration of the
    // call. Adoption COPIES, so nothing here outlives that window --
    // which is the reason it is done now rather than by holding the
    // descriptor: `PieProgramDesc` is the caller's transient memory.
    let package = unsafe { driver_abi::adopt_package(&desc.launch) };
    let kernels = unsafe { driver_abi::adopt_emitted_kernels(desc.emitted_kernels) };

    let id = state.next_id;
    state.next_id += 1;

    // A package with NO STAGES is not a malformed program; it is the
    // absence of one. The engine registers such a descriptor for a
    // forward-only deployment — the model runs, the logits come back
    // through the instance's reader channel, and no user program sits
    // around the fire. `adopt_launch_package` refuses an empty stage list
    // because an ExecPlan with nothing to execute is not a plan, and it is
    // right to; the judgement that this is not an ERROR belongs here,
    // where the difference between "the host sent a broken program" and
    // "the host sent no program" is visible.
    if !package.stages.is_empty() {
        if let Err(code) = adopt_and_compile(state, id, desc, package, &kernels) {
            return code;
        }
    }

    state.programs.insert(
        id,
        ProgramEntry {
            program_hash: desc.program_hash,
            emitter_version: desc.emitter_version,
        },
    );
    if let Some(out) = unsafe { program_id.as_mut() } {
        *out = id;
    }
    PIE_STATUS_OK
}

/// Adopt one non-empty launch package and compile what it generates.
///
/// Split out so the id lifecycle above reads as the lifecycle: the empty
/// case, the dedup case and the id assignment are all one paragraph, and
/// the thing that can fail is one call.
fn adopt_and_compile(
    state: &mut Shell,
    id: u64,
    desc: &PieProgramDesc,
    package: driver_pipeline::driver_abi::plan::LaunchPackage,
    kernels: &[driver_pipeline::EmittedKernel],
) -> Result<(), i32> {
    let plan = match driver_pipeline::adopt_launch_package(package) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("[driver-cuda-new] register_program: {error}");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
    };

    // The compile, when there is a device to compile FOR. `load_model`
    // binds it; a registration that arrives first is not an error, and
    // guessing an architecture would produce a cubin for the wrong GPU
    // rather than a diagnostic.
    if plan.executable && state.model.is_some() {
        let target = ptir_target()?;
        let versions = driver_pipeline::Versions::mirrored(desc.emitter_version);
        match state
            .ptir
            .compile(desc.program_hash, &plan, kernels, versions, target)
        {
            Ok(compiled) => state.ptir_programs.insert(id, compiled),
            Err(failure) => {
                eprintln!(
                    "[driver-cuda-new] register_program: cannot compile program \
                     {:#018x}: {}",
                    desc.program_hash,
                    failure.reason()
                );
                return Err(PIE_STATUS_UNSUPPORTED);
            }
        }
    } else if !plan.executable {
        // Recorded rather than refused: an unexecutable plan is a fact
        // about the program that the launch needing it must be able to
        // report, and losing the reason here would leave that launch with
        // nothing to say.
        eprintln!(
            "[driver-cuda-new] register_program: program {:#018x} adopted but is \
             not executable by this driver: {}",
            desc.program_hash,
            plan.reject_reason.as_deref().unwrap_or("no reason given")
        );
    }

    state.ptir_plans.insert(id, plan);
    Ok(())
}

/// What the compile cache needs to know about the GPU it is compiling for.
///
/// Read per registration rather than cached on the shell because the two
/// numbers that matter are cheap and the one that is not — the NVRTC
/// version — is a `dlopen`'d call the loader has already resolved by the
/// second registration. Caching it would trade nothing for a field that
/// can go stale against a runtime swap.
fn ptir_target() -> Result<crate::ptir::Target, i32> {
    let device = crate::cuda::Device::bind(0).map_err(|error| {
        eprintln!("[driver-cuda-new] register_program: no device to compile for: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    let (major, minor) = device.compute_capability().map_err(|error| {
        eprintln!("[driver-cuda-new] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    let nvrtc = crate::ptir::nvrtc::version().map_err(|error| {
        eprintln!("[driver-cuda-new] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    Ok(crate::ptir::Target {
        major,
        minor,
        // The ordinal, widened. A stable per-GPU id is what the identity
        // wants and what stops one machine's cache answering for another
        // family; with one device bound per process the ordinal IS that
        // id, and it is the number the C++ used.
        device: u64::try_from(device.ordinal()).unwrap_or(0),
        nvrtc,
    })
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

/// What the SHELL needs to ask a loaded model, and the whole of it.
///
/// `cuda.md` §5.B calls deleting `FamilyFacts` the real half of B's exit,
/// and the reason is this list: a shell that claims not to know which
/// families there are was matching a three-armed enum at eleven sites,
/// each asking a different question. The questions were never the
/// problem — every one of them is a legitimate thing a driver must know
/// before it can plan. Naming the families to answer them was.
///
/// So the questions became the trait, and the shape of the old matches
/// became the defaults: almost every site read `Gemma4(..) => …, _ => …`,
/// one family answering and the rest falling through. A fall-through IS a
/// default body, and writing it here means a family that never mentions
/// `head_dim_of` is *stating* that its layers agree about head dim rather
/// than being lumped in with everything else that never came up.
///
/// A new family implements this and appears in [`FACTS_ROWS`]. Nothing
/// else in the shell learns its name.
trait PlannedFamily {
    /// This family's text, traced and lowered for one fire class.
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan;

    /// Layers in the backbone — the length of every per-layer answer below.
    fn layers(&self) -> u32;

    /// Layer `l`'s head dim. Uniform unless a family says otherwise;
    /// gemma-4's two layer kinds disagree (256 vs 512), which is the only
    /// reason this is per-layer at all.
    fn head_dim_of(&self, _l: u32, uniform: u32) -> u32 {
        uniform
    }

    /// The layer whose KV pages `l` attends through, or `None` when `l`
    /// owns its own. A `Some` layer projects and writes nothing.
    fn kv_source(&self, _l: u32) -> Option<u32> {
        None
    }

    /// Layer `l`'s sliding window, `-1` for the whole context. An empty
    /// answer from [`Self::window_by_layer`] means the fire's single
    /// window applies to every layer.
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        Vec::new()
    }

    /// The attention softmax scale. `1/sqrt(head_dim)` unless the
    /// family's q/k norms already carry it (gemma-4 runs 1.0).
    fn sm_scale(&self, head_dim: u32) -> f32 {
        1.0 / (head_dim as f32).sqrt()
    }

    /// Whether this family carries RECURRENT STATE. Such a fire is not
    /// replayable — a captured body bakes one instance's slots — so it
    /// stays eager, and it is the only family that may be handed an MTP
    /// service class.
    fn recurrent(&self) -> bool {
        false
    }

    /// The two head dims a family's layer kinds decode at, when it needs
    /// SEPARATE decode plans for them — `(sliding, full)`. `None` for a
    /// family whose layers agree, which is why one plan serves them: the
    /// planner bakes the head dim in.
    fn decode_plan_head_dims(&self) -> Option<(u32, u32)> {
        None
    }

    /// Whether the family's PREFILL plans internally, per fire, off the
    /// host CSR mirrors — so there is nothing to pre-plan and the mirrors
    /// must be uploaded.
    fn planless_prefill(&self) -> bool {
        false
    }

    /// Whether both attention forms state `[q, o]` as SSA args, so the
    /// guard-owned attention pins stay null. Only gemma-4 does.
    fn pins_attention_values(&self) -> bool {
        true
    }

    /// Per-layer rope tables, softcap, PLE width and named scalar
    /// constants — everything the prologue reads that is not a shape.
    /// Empty for a family whose rope is one theta and whose epilogue caps
    /// nothing.
    fn tables(&self, _model: &LoadedModel) -> FamilyTables {
        FamilyTables::default()
    }

    /// The family's recurrent geometry, when it has one.
    fn gdn_shape(&self) -> Option<GdnShape> {
        None
    }
}

impl PlannedFamily for model::gemma_2::forward::facts::Gemma2Facts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma_2::forward::gemma2_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
    fn tables(&self, _model: &LoadedModel) -> FamilyTables {
        FamilyTables {
            softcap: if self.final_logit_softcap { 30.0 } else { 0.0 },
            ..FamilyTables::default()
        }
    }
}

impl PlannedFamily
    for (
        model::gpt_oss::forward::facts::GptOssFacts,
        model::gpt_oss::forward::facts::GptOssCudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gpt_oss::forward::gpt_oss_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.0.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.1.window_left.clone()
    }
}

impl PlannedFamily for model::glm5::forward::facts::Glm5Facts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::glm5::forward::glm5_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        // MLA's pages hold the LATENT, not a head-split key, and
        // `kv_a_width` is that row — the shared `MlaFacts` says it once
        // for every family in this lineage.
        self.attn.kv_a_width()
    }
}

impl PlannedFamily
    for (
        model::kimi_k2::forward::facts::KimiFacts,
        model::kimi_k2::forward::facts::KimiCudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::kimi_k2::forward::kimi_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.0.attn.kv_a_width()
    }
}

impl PlannedFamily for model::kimi_k3::forward::facts::KimiK3Facts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::kimi_k3::forward::kimi_k3_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.kv_a_width()
    }
    fn recurrent(&self) -> bool {
        // KDA carries per-request recurrent state, so a fire of this
        // family stays eager for the rule the hybrid states.
        true
    }
}

impl PlannedFamily for model::deepseek_v4::forward::facts::Dsv4Facts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::deepseek_v4::forward::dsv4_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        let w = i32::try_from(self.attn.sliding_window).unwrap_or(0);
        (0..self.layers).map(|_| if w > 0 { w } else { -1 }).collect()
    }
}

impl PlannedFamily for model::nemotron_h::forward::facts::NemotronHFacts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::nemotron_h::forward::nemotron_h_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        u32::try_from(self.layer_types.len()).unwrap_or(0)
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
    fn recurrent(&self) -> bool {
        // The mamba layers' selective-scan state is per request.
        true
    }
}

impl PlannedFamily for model::gemma3n::forward::facts::Gemma3nFacts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma3n::forward::gemma3n_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        u32::try_from(self.per_layer_intermediate.len()).unwrap_or(0)
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
}

/// The per-layer tables and named constants a family's prologue reads.
#[derive(Default)]
struct FamilyTables {
    /// Rope base per layer; empty means the one `rope_theta` applies.
    theta_by_layer: Vec<f32>,
    /// Rotary width per layer; empty means full rotation at head dim.
    rotary_by_layer: Vec<u32>,
    /// Final-logit softcap, 0 for none.
    softcap: f32,
    /// Per-layer-embedding width, 0 for a family without one.
    ple_dim: i32,
    /// Named scalar constants the trace refers to by name.
    scales: std::collections::BTreeMap<String, f32>,
}

/// A recurrent family's slab geometry — what the shell must allocate and
/// stride before it can hand the executor a `GdnCtx`.
struct GdnShape {
    layers: u32,
    linear_layers: Vec<u32>,
    conv_stride: usize,
    state_stride: usize,
    state_elem: usize,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
    conv_k: i32,
}

/// The three implementations. Each is the set of answers its family
/// PREVIOUSLY contributed to eleven scattered matches, gathered into one
/// place where the family's own name is the last time it is mentioned.

impl PlannedFamily
    for (
        model::families::llama_like::forward::facts::LlamaLikeFacts,
        model::families::llama_like::forward::facts::LlamaLikeCudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::families::llama_like::forward::llama_like_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    // Every other answer is the default: uniform head dim, no KV sharing,
    // one window, the standard scale, no recurrence, one decode plan, no
    // per-layer tables. The lineage is the family the defaults were
    // written from.
}

impl PlannedFamily
    for (
        model::qwen_3_5::forward::facts::Qwen35HybridFacts,
        model::qwen_3_5::forward::facts::Qwen35CudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::qwen_3_5::forward::qwen3_5_hybrid_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn recurrent(&self) -> bool {
        true
    }
    fn gdn_shape(&self) -> Option<GdnShape> {
        let g = &self.0.gdn;
        Some(GdnShape {
            layers: self.0.layers,
            linear_layers: (0..self.0.layers).filter(|&l| !self.0.is_full_attn(l)).collect(),
            conv_stride: (g.conv_kernel * g.conv_dim()) as usize,
            state_stride: (g.value_heads * g.key_head_dim * g.value_head_dim) as usize,
            state_elem: if self.1.state_bf16 { 2 } else { 4 },
            k_h: g.key_heads as i32,
            v_h: g.value_heads as i32,
            k_d: g.key_head_dim as i32,
            v_d: g.value_head_dim as i32,
            conv_dim: g.conv_dim() as i32,
            conv_k: g.conv_kernel as i32,
        })
    }
}

impl PlannedFamily
    for (
        model::gemma_4::forward::facts::Gemma4Facts,
        model::gemma_4::forward::facts::Gemma4CudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma_4::forward::gemma4_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, l: u32, _uniform: u32) -> u32 {
        self.0.head_dim_of(l)
    }
    fn kv_source(&self, l: u32) -> Option<u32> {
        self.0.kv_source(l)
    }
    fn window_by_layer(&self, sliding_window: i32) -> Vec<i32> {
        (0..self.0.layers)
            .map(|l| if self.0.is_full_attn(l) { -1 } else { sliding_window.max(0) })
            .collect()
    }
    fn sm_scale(&self, _head_dim: u32) -> f32 {
        // The q/k norms carry the scaling.
        1.0
    }
    fn decode_plan_head_dims(&self) -> Option<(u32, u32)> {
        Some((self.0.head_dim, self.0.global_head_dim))
    }
    fn planless_prefill(&self) -> bool {
        true
    }
    fn pins_attention_values(&self) -> bool {
        // Both attention forms state [q, o] as SSA args.
        false
    }
    fn tables(&self, model: &LoadedModel) -> FamilyTables {
        let (facts, hf) = (&self.0, &model.hf);
        let theta: Vec<f32> = (0..facts.layers as usize)
            .map(|l| {
                hf.gemma_per_layer_rope_theta.get(l).copied().unwrap_or({
                    // The C++ parse fallback: full layers (and configs
                    // without a local base) ride `rope_theta`.
                    if facts.is_full_attn(l as u32) || hf.gemma3n_rope_local_base_freq <= 0.0 {
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
        for (n, sc) in model.gemma_layer_scalars.iter().enumerate() {
            scales.insert(format!("layer.{n}.ple_norm"), *sc);
        }
        FamilyTables {
            theta_by_layer: theta,
            rotary_by_layer: rotary,
            softcap: facts.logit_softcap,
            ple_dim: facts.ple_dim as i32,
            scales,
        }
    }
}

/// gemma-4's facts off the checkpoint's config — the layer schedule
/// reduced to the interval (irregular arrays refuse, qwen3_5's rule),
/// the FULL layers' rotary width by the driver's derivation, the
/// double-wide-MLP and KV-shared counts as stated. The E2B anchor's
/// legs only: `k_eq_v` (26B-A4B's V-from-K mode) and the MoE block
/// refuse until a deployment anchors them.
fn gemma4_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
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
    Ok(Box::new((facts, cuda)))
}

/// The qwen3_5 hybrid's facts, read off the checkpoint's own config —
/// the layer schedule from `layer_types` (reduced to the interval, the
/// Metal driver's reduction; irregular arrays refuse), the GDN geometry
/// from the `linear_*` fields, the rotary width by the driver's
/// `max(2, 2·int(0.5·factor·head_dim))` derivation. Dense MLP only —
/// a MoE config refuses until a MoE deployment anchors that leg.
fn qwen35_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::qwen_3_5::forward::facts::{
        Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind,
        Qwen35MoeMlpFacts,
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
        // THE MLP KIND, off the config rather than assumed dense.
        //
        // `n_experts > 0` IS the mixture — the same reading
        // `LlamaLikeFacts::n_experts` documents, and the reason a routed
        // FFN is a fact and not a family: the attention is unchanged and
        // only the block between the two norms differs. The hybrid's own
        // text already branches on `Qwen35MlpKind`, so this derivation
        // was the only thing making every qwen3_5 deployment dense.
        //
        // Qwen3.5-35B-A3B is what it opens: 256 routed experts, top-k 8,
        // `moe_intermediate` 512 beside a shared expert of the same
        // width. Those numbers were PINNED as a fixture from the C++
        // driver's measured notes because no config was committed; the
        // checkpoint's own config agrees with the fixture on every one.
        mlp: if to_u32(hf.num_experts) > 0 {
            Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
                hidden: to_u32(hf.hidden_size),
                num_experts: to_u32(hf.num_experts),
                top_k: to_u32(hf.num_experts_per_tok),
                moe_intermediate: to_u32(hf.moe_intermediate_size),
                shared_expert_intermediate: to_u32(hf.shared_expert_intermediate_size),
                norm_variant: NormVariant::Gemma,
            })
        } else {
            Qwen35MlpKind::Dense { intermediate: to_u32(hf.intermediate_size) }
        },
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
    Ok(Box::new((facts, cuda)))
}

/// The loaded model's facts, family-dispatched: the qwen3_5 hybrid by
/// its `linear_*` geometry + layer schedule, else the llama-like
/// mapping. Only the qwen3-family pre-norm shape is claimed on the
/// llama-like side; anything else refuses rather than mis-executes.
/// The `FireArrays::named` key the SCORE pin is pooled under.
///
/// A reserved id rather than a traced one: no statement names this value,
/// the driver publishes it, and the pool is keyed by `ValueId` because
/// every other thing in it is a traced seam. `u32::MAX` cannot collide
/// with a trace value — a plan with four billion values would have failed
/// long before.
const SCORE_PIN: model_compiler::trace::ValueId = model_compiler::trace::ValueId::MAX;

/// Is the unionized supergraph armed for this process?
///
/// **ON by default now**, and `PIE_CUDA_SUPERGRAPH=0` turns it off.
///
/// It was off, deliberately, with this reason: every A/B in the tree pins
/// the EAGER leg, and a capture is an optimisation that has to prove
/// itself against that rather than replace it silently. It has now proved
/// it, on the three claims that were the actual doubt:
///
/// - the whole ABI suite records and replays (19/19 with the gate on),
///   which is every family this shell opens and every fire shape it
///   serves;
/// - one exec runs two structurally distinct KV-write programs and
///   returns byte-identical logits, selected by a byte of device memory
///   (`bridge_smoke::the_union_captures_and_replays_the_same_decode`);
/// - and one exec serves a SECOND fire's tokens
///   (`a_cached_exec_serves_the_next_fire`), which is the property that
///   makes a cached exec worth caching and the only one that can tell a
///   baked address from baked contents.
///
/// What cannot be replayed still refuses rather than being captured
/// wrong: recurrent-state families stay eager at the LOWERING decision,
/// and an arm whose prepared state the fire declines to build is refused.
/// So default-on changes which leg runs, not which answers are possible.
///
/// The env var inverts rather than disappears, because a default is a
/// judgement and a judgement should stay reversible without a rebuild.
fn supergraph_enabled() -> bool {
    !std::env::var_os("PIE_CUDA_SUPERGRAPH")
        .is_some_and(|v| v == "0" || v == "false" || v == "off")
}

/// What a row dispatches to: this family's facts, off the checkpoint.
type FactsFrom = fn(&LoadedModel) -> Result<Box<dyn PlannedFamily>, i32>;

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
    // gemma-3 is a llama-lineage decoder with per-head qk-norm and an
    // alternating window; the derivation reads both off the checkpoint.
    ("gemma3", llama_like_facts_from_hf),
    ("gemma3_text", llama_like_facts_from_hf),
    // A ROUTED FFN is a fact, not a family: `n_experts > 0` selects the
    // mixture and the attention is unchanged, which is why these two
    // reach the same derivation as every dense deployment above.
    ("mixtral", llama_like_facts_from_hf),
    ("qwen3_moe", llama_like_facts_from_hf),
    // ── Gemma-4: nested decoder, PLE, two layer kinds.
    ("gemma4", gemma4_facts_from_hf),
    ("gemma4_text", gemma4_facts_from_hf),
    // ── Qwen3.5 hybrids: GDN linear attention beside full attention.
    ("qwen3_5", qwen35_facts_from_hf),
    ("qwen3_5_text", qwen35_facts_from_hf),
    ("qwen3_5_moe", qwen35_facts_from_hf),
    ("qwen3_5_moe_text", qwen35_facts_from_hf),
    // ── gemma-2: alternating local/global attention, softcapped twice.
    ("gemma2", gemma2_facts_from_hf),
    // ── gpt-oss: MXFP4 mixture, attention sinks, alternating window.
    ("gpt_oss", gpt_oss_facts_from_hf),
    // ── The MLA lineage: latent q/kv, a dense prefix, then the mixture.
    ("glm_moe_dsa", glm5_facts_from_hf),
    ("deepseek_v2", kimi_k2_facts_from_hf),
    ("deepseek_v3", kimi_k2_facts_from_hf),
    ("kimi_k2", kimi_k2_facts_from_hf),
    ("kimi_k3", kimi_k3_facts_from_hf),
    ("deepseek_v4", dsv4_facts_from_hf),
    // ── Hybrids and the per-layer-embedding gemma.
    ("nemotron_h", nemotron_h_facts_from_hf),
    ("gemma3n", gemma3n_facts_from_hf),
    ("gemma3n_text", gemma3n_facts_from_hf),
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

/// The fire's CLASS, read off the descriptor rather than guessed from its
/// shape.
///
/// The shape alone answers only the first question — one row per request
/// is a decode, anything else is prefill-shaped. The MTP service passes
/// are not shapes at all: they are *what the pass is for*, and the wire
/// has said so since v23 in the recurrent-state flags. The C++ driver
/// read exactly these bits into `RsExecutionMode`
/// (`pipeline/batch_compose.hpp`) and this is that derivation, with the
/// mode's name replaced by the class it selects:
///
/// | rows                                   | C++ mode      | class           |
/// |----------------------------------------|---------------|-----------------|
/// | every row replays buffered tokens      | `BufferFold`  | `CommitAdvance` |
/// | some row writes buffered slabs         | `BufferWrite` | `FrozenVerify`  |
/// | recurrent, no readout rows             | `Forward`     | `StateOnly`     |
/// | otherwise                              | `Forward`     | shape decides   |
///
/// The mixed case is REFUSED for the same reason the C++ composer
/// refuses it: a replay row gathers its activations out of the slabs and
/// a computing row does not, so the two cannot share one op list. That is
/// a property of the pass, so it is answered here, once, rather than by
/// every arm that would otherwise find half a fire.
pub fn fire_class_of(
    step: &driver_abi::local::PieStepDesc,
    rows: usize,
    requests: usize,
) -> Result<model_compiler::trace::FireClass, i32> {
    use driver_abi::local::{PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD};
    use model_compiler::trace::FireClass;

    let flags = slice_of(step.rs_slot_flags.ptr, step.rs_slot_flags.len);
    let buf_indptr = slice_of(step.rs_buffer_slot_indptr.ptr, step.rs_buffer_slot_indptr.len);
    // A row's buffer span, and 0 when the fire carries no CSR at all.
    let span = |r: usize| -> u32 {
        match (buf_indptr.get(r + 1), buf_indptr.get(r)) {
            (Some(&hi), Some(&lo)) => hi.saturating_sub(lo),
            _ => 0,
        }
    };
    let mut any_buffer = false;
    let mut any_replay = false;
    let mut all_replay = requests > 0;
    for r in 0..requests {
        let f = flags.get(r).copied().unwrap_or(0);
        let buffered = span(r) > 0;
        // `BUFFER_WRITE` marks a row that writes AND folds; a pure replay
        // folds without writing.
        let replays = buffered && f & PIE_RS_FLAG_FOLD != 0 && f & PIE_RS_FLAG_BUFFER_WRITE == 0;
        any_buffer |= buffered;
        any_replay |= replays;
        all_replay &= replays;
    }
    if any_replay && !all_replay {
        eprintln!(
            "[driver-cuda-new] launch: a row that only replays buffered tokens \
             cannot share a fire with one that computes new ones"
        );
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    if all_replay {
        return Ok(FireClass::CommitAdvance);
    }
    if any_buffer {
        return Ok(FireClass::FrozenVerify);
    }
    // `StateOnly` — the backbone with the epilogue cut off — is the one
    // class this function CANNOT yet read, and the reason is worth
    // stating rather than working around.
    //
    // Its wire signal is "no readout rows", i.e. an empty
    // `sampling_indices`. But the shell below does not read that field at
    // all: it builds `Row { samples: true }` for every row, unconditionally.
    // So the field is empty on every fire that reaches here, and keying a
    // class on it would classify every hybrid prefill as a service pass —
    // which is exactly what it did, and the hybrid's logits went missing.
    //
    // Reading a fact off where a statement SITS rather than what it SAYS
    // is the defect this port keeps re-finding. `sampling_indices` will
    // say it once the readout rows are plumbed through to `Row::samples`;
    // until then the shape answers, and `StateOnly` is reachable only
    // through the lowering tests.
    Ok(if rows == requests { FireClass::Decode } else { FireClass::Prefill })
}

/// gemma-2's facts off the checkpoint's config.
///
/// The window list is the family's own shape: gemma-2 ALTERNATES local
/// and global attention, odd layers seeing the whole context. `layer_types`
/// states it when the config ships one; the parity is the fallback the
/// C++ parse used.
fn gemma2_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gemma_2::forward::facts::{Gemma2AttnFacts, Gemma2Facts};
    let hf = &model.hf;
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = to_u32(hf.num_hidden_layers);
    let window_left: Vec<i32> = (0..layers)
        .map(|l| {
            let global = hf
                .layer_types
                .get(l as usize)
                .map_or(l % 2 == 1, |t| t == "full_attention");
            if global { -1 } else { hf.sliding_window.max(0) }
        })
        .collect();
    Ok(Box::new(Gemma2Facts {
        layers,
        vocab: to_u32(hf.vocab_size),
        hidden: to_u32(hf.hidden_size),
        intermediate: to_u32(hf.intermediate_size),
        tied_embeddings: hf.tie_word_embeddings,
        final_logit_softcap: hf.gemma_final_logit_softcap > 0.0,
        window_left,
        attn: Gemma2AttnFacts {
            heads: to_u32(hf.num_attention_heads),
            kv_heads: to_u32(hf.num_key_value_heads),
            head_dim: to_u32(hf.head_dim),
            qk_norm: false,
            query_pre_attn_scale: true,
            attn_logit_softcap: true,
        },
    }))
}

/// gpt-oss's facts. The sliding schedule is `layer_types`' when the
/// config ships one — gpt-oss alternates from layer 0 — and the fused
/// MXFP4 decode leg is the engine default this text states.
fn gpt_oss_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    let hf = &model.hf;
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = to_u32(hf.num_hidden_layers);
    let experts = to_u32(hf.num_experts);
    let facts = GptOssFacts {
        hidden: to_u32(hf.hidden_size),
        layers,
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        intermediate: to_u32(hf.intermediate_size),
        experts,
        top_k: to_u32(hf.num_experts_per_tok),
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        swiglu_limit: 7.0,
        attention_bias: true,
        rope_yarn_original: true,
        attn_sinks: true,
    };
    let cuda = GptOssCudaFacts {
        mxfp4_decode_gemv: true,
        mxfp4_decode_max_routes: 32 * experts.max(1),
        streamed_experts: false,
        window_left: (0..layers)
            .map(|l| {
                let sliding = hf
                    .layer_types
                    .get(l as usize)
                    .map_or(l % 2 == 0, |t| t == "sliding_attention");
                if sliding { hf.sliding_window.max(0) } else { -1 }
            })
            .collect(),
    };
    Ok(Box::new((facts, cuda)))
}

/// The MLA lineage's shared reading: a dense PREFIX then the mixture,
/// latent q/kv projections, and the rope half carried beside the nope
/// half. `first_k_dense_replace` is the prefix length in every config
/// that ships one.
fn glm5_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::glm5::forward::facts::{Glm5DsaFacts, Glm5Facts, Glm5MlaFacts, Glm5MoeFacts};
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    Ok(Box::new(Glm5Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: Glm5MlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            // Only kimi-k3 gates the MLA output.
            output_gate: false,
        },
        dsa: Glm5DsaFacts {
            index_n_heads: u(hf.dsv4_index_n_heads),
            index_head_dim: u(hf.dsv4_index_head_dim),
            index_topk: u(hf.dsv4_index_topk),
        },
        moe: Glm5MoeFacts {
            hidden: u(hf.hidden_size),
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
            aligned_block: 16,
        },
    }))
}

/// kimi-k2: the same MLA reading as glm5, without the DSA indexer.
fn kimi_k2_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::kimi_k2::forward::facts::{KimiCudaFacts, KimiFacts, KimiMlaFacts, KimiMoeFacts};
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let facts = KimiFacts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: KimiMlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            // Only kimi-k3 gates the MLA output.
            output_gate: false,
        },
        moe: KimiMoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
        },
    };
    // The BINDING facts: one fused q/kv latent GEMM when the load joined
    // them, and YaRN when the config asked for it.
    let cuda = KimiCudaFacts {
        q_kv_a_fused: model.aliases.contains_key("layer.0.q_kv_a_fused"),
        rope_yarn_original: matches!(
            hf.rope_scaling_kind,
            crate::model::config::RopeScaling::OriginalYarn
        ),
    };
    Ok(Box::new((facts, cuda)))
}

/// kimi-k3: MLA beside KDA linear attention, on the periodic schedule
/// `full_attn_at` states.
fn kimi_k3_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::kimi_k3::forward::facts::{
        KimiK3Facts, KimiK3KdaFacts, KimiK3MlaFacts, KimiK3MoeFacts,
    };
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    Ok(Box::new(KimiK3Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        full_attn_interval: interval,
        attn_res_block: 0,
        attn: KimiK3MlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            output_gate: true,
        },
        kda: KimiK3KdaFacts {
            value_heads: u(hf.linear_num_value_heads),
            value_head_dim: u(hf.linear_value_head_dim),
            conv_kernel: u(hf.linear_conv_kernel_dim),
            gate_lower_bound_milli: 0,
        },
        moe: KimiK3MoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
        },
    }))
}

/// deepseek-v4: the DSA indexer, hyper-connections, and a routed MLP
/// whose activation clamps.
fn dsv4_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::deepseek_v4::forward::facts::{
        Dsv4AttnFacts, Dsv4Facts, Dsv4HcFacts, Dsv4MoeFacts,
    };
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    Ok(Box::new(Dsv4Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: Dsv4AttnFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            head_dim: u(hf.head_dim),
            q_lora_rank: u(hf.q_lora_rank),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            sliding_window: u(hf.sliding_window.max(0)),
            o_lora_rank: 0,
            o_groups: 1,
        },
        hc: Dsv4HcFacts { mult: 1 },
        moe: Dsv4MoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            swiglu_limit_milli: 0,
            hash_routed: false,
        },
    }))
}

/// nemotron-h: three layer kinds, and the schedule is the LIST rather
/// than an interval — the family has an MLP-only layer no period spells.
fn nemotron_h_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::nemotron_h::forward::facts::{
        NemotronAttnFacts, NemotronHFacts, NemotronLayerKind, NemotronMambaFacts,
        NemotronMoeFacts,
    };
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let layer_types: Vec<NemotronLayerKind> = hf
        .layer_types
        .iter()
        .map(|t| match t.as_str() {
            "attention" | "full_attention" => NemotronLayerKind::Attention,
            "mlp" => NemotronLayerKind::Mlp,
            _ => NemotronLayerKind::Mamba,
        })
        .collect();
    if layer_types.is_empty() {
        eprintln!("[driver-cuda-new] launch: nemotron-h states no layer_types");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let window_left = vec![-1; layer_types.len()];
    Ok(Box::new(NemotronHFacts {
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        layer_types,
        mamba: NemotronMambaFacts {
            num_heads: u(hf.mamba_num_heads),
            head_dim: u(hf.mamba_head_dim),
            state_size: u(hf.mamba_state_size),
            n_groups: u(hf.mamba_n_groups),
            conv_kernel: u(hf.mamba_conv_kernel),
        },
        attn: NemotronAttnFacts {
            heads: u(hf.num_attention_heads),
            kv_heads: u(hf.num_key_value_heads),
            head_dim: u(hf.head_dim),
        },
        moe: NemotronMoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.shared_expert_intermediate_size),
        },
        window_left,
    }))
}

/// gemma-3n: altUp streams, laurel, per-layer embeddings and a per-layer
/// MLP width the config states as a list.
fn gemma3n_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gemma3n::forward::facts::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts};
    let hf = &model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = u(hf.num_hidden_layers) as usize;
    Ok(Box::new(Gemma3nFacts {
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        per_layer_intermediate: vec![u(hf.intermediate_size); layers],
        laurel_rank: u(hf.laurel_rank),
        ple_width: u(hf.gemma_hidden_size_per_layer_input),
        sparsity_layers: u32::try_from(
            hf.gemma3n_activation_sparsity.iter().filter(|&&s| s > 0.0).count(),
        )
        .unwrap_or(0),
        altup: Gemma3nAltUpFacts {
            num_streams: u(hf.altup_num_inputs),
            active: u(hf.altup_active_idx),
        },
        attn: Gemma3nAttnFacts {
            heads: u(hf.num_attention_heads),
            kv_heads: u(hf.num_key_value_heads),
            head_dim: u(hf.head_dim),
        },
        window_left: (0..layers)
            .map(|l| {
                if hf.layer_types.get(l).is_some_and(|t| t == "full_attention") {
                    -1
                } else {
                    hf.sliding_window.max(0)
                }
            })
            .collect(),
    }))
}

/// Whether a quantized scheme's STORED bytes are what its kernels read.
///
/// The dividing line for what a load can accept without a transcode
/// engine. A scheme here is uploaded verbatim; anything else needs its
/// layout changed on the way in, which is `transcode_engine.hpp`'s job in
/// the retired C++ tree and is not ported.
///
/// Note this is the arm for a checkpoint that DECLARES its scheme in the
/// tensor encoding. gpt-oss does not — its MXFP4 banks arrive as plain
/// `U8` and the scheme is in `quantization_config` — so the live MXFP4
/// path is the `Raw(U8)` arm above. This one is for the encodings the
/// loader does tag, and it is a MATCH rather than a default-allow for the
/// same reason: a scheme nobody has checked should refuse, because
/// guessing hands a kernel a layout it was not compiled for, which is not
/// a crash but wrong numbers.
///
/// Deliberately a MATCH rather than a default-allow: a scheme nobody has
/// checked should refuse, because the failure mode of guessing is a
/// kernel reading a layout it was not compiled for — which is not a
/// crash, it is wrong numbers.
const fn reads_its_stored_form(scheme: model_loader::types::QuantScheme) -> bool {
    use model_loader::types::QuantScheme as Q;
    matches!(scheme, Q::Mxfp4E2M1E8M0 | Q::MlxAffineU4)
}

/// THE GQA RATIO, refused at LOAD rather than discovered at launch.
///
/// FlashInfer's decode instantiates group sizes {1, 2, 3, 4, 8} and
/// reports anything else by THROWING. A throw crossing the C ABI is
/// undefined behaviour; the generated shim prints the message before it
/// dies, but printing is all it can do — the launcher signatures have
/// nowhere to put a failure. A load DOES: it returns a status code.
///
/// This lived inside the llama lineage's derivation, which made it a
/// property of that lineage rather than of the BUILD. It is the build's:
/// every family whose attention reaches the same dispatch is subject to
/// the same instantiation set, and the hybrid is the live proof —
/// Qwen3.6-27B declares `qwen3_5_text`, so it is already openable, and
/// its 24 query heads over 4 kv heads is a group size of six.
///
/// Qwen2.5-1.5B is the other live example, twelve over two.
fn refuse_unservable_gqa(hf: &crate::model::config::HfConfig) -> Result<(), i32> {
    let kv_heads = hf.num_key_value_heads.max(1);
    let group_size = hf.num_attention_heads / kv_heads;
    if hf.num_attention_heads % kv_heads != 0 || !matches!(group_size, 1 | 2 | 3 | 4 | 8) {
        eprintln!(
            "[driver-cuda-new] load: this build's decode does not instantiate \
             GQA group size {group_size} ({} q heads over {kv_heads} kv heads); \
             the supported set is 1, 2, 3, 4, 8",
            hf.num_attention_heads
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok(())
}

/// The facts for a loaded checkpoint, by the model type it declares.
fn facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    refuse_unservable_gqa(&model.hf)?;
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
fn llama_like_facts_from_hf(model: &LoadedModel) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::families::llama_like::forward::facts::{
        LlamaLikeCudaFacts, LlamaLikeFacts, NormPlacement, QkNorm,
    };
    use model_compiler::trace::{NormVariant, RopeKind};
    let hf = &model.hf;
    if !model.weights.contains_key("model.embed_tokens.weight") {
        eprintln!("[driver-cuda-new] launch: only HF llama-like checkpoints execute today");
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
        // A ROUTED FFN is a fact, not a family (the `LlamaLikeFacts` doc's
        // own argument), so these come off the checkpoint like every other
        // width. Zero throughout is a dense deployment, which is what the
        // fields mean rather than a stand-in for "unknown".
        n_experts: to_u32(hf.num_experts),
        experts_per_token: to_u32(hf.num_experts_per_tok),
        moe_intermediate: to_u32(hf.moe_intermediate_size),
        shared_intermediate: to_u32(hf.shared_expert_intermediate_size),
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
    Ok(Box::new((facts, cuda)))
}

/// Replay this fire's bucket if it is captured, and capture it if not.
///
/// The whole supergraph arc, at its one live call site. What it does, in
/// the order the pieces were built:
///
/// 1. **Eligibility.** A fire whose staged LoRA did not group cannot be
///    recorded at all — `apply`'s solo path is a host loop whose launch
///    count follows the adapter set. Ineligible means eager, which is the
///    C++ arc's own device for what cannot be replayed.
/// 2. **The bucket.** `(R, N, fire class, model)` plus the lora group
///    shape. Every `GuardPred` axis is deliberately absent: those are what
///    the conditionals fold.
/// 3. **The epoch.** `FireArrays` bumps it whenever a pool grew, because
///    growth moves a base address out from under a recorded launch. A
///    stale exec is dropped and recaptured rather than replayed.
/// 4. **Dual-prepare.** A capture must be taken warm — a launcher that
///    allocates on first use cannot do so inside a capture — and a warm-up
///    must walk a VALID program, so warm once per variant with its own
///    resolved lowering. A union records arms no single valid program
///    takes, which is why one warm fire is not enough.
/// 5. **The predicates**, uploaded before every launch: this is the fire's
///    own shape, and the only thing that differs between two replays of
///    one exec.
#[allow(clippy::too_many_arguments)]
fn capture_or_replay<R: crate::model::executor::Resolver>(
    cache: &mut crate::model::supergraph::SupergraphCache,
    epoch: u64,
    model_id: u64,
    plan: &model_compiler::trace::ForwardPlan,
    rows_desc: &[model_compiler::lower::Row],
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::model::executor::DispatchPlan,
    frame: crate::model::executor::Frame,
    resolver: &mut R,
    ctx: &crate::model::executor::DispatchCtx,
    regions: crate::model::executor::AttnRegions<'_>,
    gdn: Option<&crate::model::executor::GdnCtx>,
    alloc: &crate::cuda::Allocator,
    stream: crate::cuda::StreamRef<'_>,
    requests: usize,
    rows: usize,
    class: model_compiler::trace::FireClass,
) -> Result<usize, crate::model::executor::RunRefusal> {
    use crate::model::executor::{DispatchPlan, run};
    use crate::model::supergraph::{BucketKey, fire_predicates, union_eligibility};

    let eligibility = union_eligibility(None);
    let key = BucketKey::new(
        u32::try_from(requests).unwrap_or(0),
        u32::try_from(rows).unwrap_or(0),
        class,
        model_id,
    );

    // The fire's own bits, and the only thing that differs between two
    // replays of one exec.
    let mut preds = match crate::cuda::PredicateWord::new(alloc) {
        Ok(p) => p,
        Err(_) => return run(lowered, dplan, frame, resolver, ctx, regions, gdn),
    };
    if fire_predicates(rows_desc, &lowered.conds, &mut preds).is_err()
        || preds.upload(stream).is_err()
        || stream.synchronize().is_err()
    {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }

    if cache.replay(key, epoch, stream).unwrap_or(false) {
        return Ok(lowered.launches.len());
    }

    // DUAL-PREPARE: one warm fire per variant, each a resolved program.
    // Only variants this fire can PREPARE. A `wants_scores` warm-up would
    // lower the score-capturing dispatch, which refuses without a score
    // sink — and warming is not the place to discover that. It is also
    // why scores are not a union axis: the north star's list is "hook
    // attachment, mask kind, correction arm, depth, LoRA rank", and every
    // one of those is a branch rather than a different prepared state.
    for marks in [
        model_compiler::lower::Row { samples: true, ..Default::default() },
        model_compiler::lower::Row { samples: true, write_desc: true, ..Default::default() },
    ] {
        let warm_rows = vec![marks; rows];
        let Ok(warm) = model_compiler::lower::lower_with(
            plan,
            &warm_rows,
            model_compiler::lower::Fire { captures_across_splits: false },
            model_compiler::lower::GuardMode::Resolve,
        ) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        let warm_dplan = DispatchPlan::new(plan, &warm);
        run(&warm, &warm_dplan, frame, resolver, ctx, regions, gdn)?;
        let _ = stream.synchronize();
    }

    let captured = {
        let mut a = crate::cuda::Allocator::new();
        let Ok(scope) = a.begin_capture(stream) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        let mut b = crate::cuda::SupergraphBuilder::new(scope.stream(), &preds);
        let ran = crate::model::executor::run_captured(
            lowered, dplan, frame, resolver, ctx, regions, gdn, &mut b,
        );
        drop(b);
        // A REFUSED CAPTURE IS NOT A REFUSED FIRE.
        //
        // Some arms cannot be recorded at all, and the reason is always
        // the same shape: their prepared state is something the fire
        // declined to build. The score-capturing prefill dispatch wants a
        // plan raised for the full-attention variant, buffers laid out for
        // an observation window, and a positive window — none of which a
        // fire that wants no scores has any reason to prepare.
        //
        // So the capture is abandoned and the fire runs eagerly. That is
        // the same answer ungrouped LoRA gets from `union_eligibility`,
        // and the same one the C++ arc gives mixed peels: what cannot be
        // replayed stays eager. The alternative — failing the fire — would
        // make an optimisation into a correctness requirement.
        match (ran, scope.end()) {
            (Ok(n), Ok(g)) => Some((n, g)),
            (Err(_) | Ok(_), _) => None,
        }
    };
    let Some((ran, graph)) = captured else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    let Ok(exec) = graph.instantiate() else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    if exec.launch(stream).is_err() {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }
    let _ = cache.insert(key, exec, epoch, eligibility);
    Ok(ran)
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
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, DispatchPlan, Frame, GdnCtx, PrefillPlan, Resolver,
        run,
    };
    use model_compiler::lower::{Arg, Fire, GuardMode, Row, lower_with};
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
    let class = fire_class_of(step, rows, requests)?;

    // A family that does not DECLARE a service class must be turned away
    // rather than traced: its text answers the three with `unreachable!`,
    // and a panic crossing the C ABI aborts the process instead of
    // returning the status this call has a slot for. Only the MTP family
    // composes those passes.
    if !matches!(class, FireClass::Decode | FireClass::Prefill) && !family.recurrent() {
        eprintln!(
            "[driver-cuda-new] launch: {class:?} is an MTP service pass and \
             this family declares no trace for it"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }

    // ── The lowering. ──
    let plan = family.trace(class);
    let fire_rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    // THE SUPERGRAPH GATE, off unless asked. `Union` keeps every guard so
    // the arms can be recorded into conditional bodies and decided at
    // replay; `Resolve` answers them here and produces one program. Off by
    // default because the eager leg is what every A/B in the tree pins,
    // and a capture is an optimisation that must prove itself against it.
    // A family carrying RECURRENT STATE is not replayable, and the rule is
    // decided HERE rather than at the capture, because a fire built
    // against a union lowering cannot fall back to an eager one — it would
    // run the union's program, both sides of every guard, over the same
    // rows.
    //
    // The hybrid's GDN slabs are per-fire mutable state reached through a
    // slot indirection the host rewrites, so a captured body bakes one
    // fire's slots and a replay would update another instance's
    // recurrence. Found the hard way: the corruption surfaced as a fault
    // inside `cudaGraphDestroy`, about as far from the cause as a symptom
    // gets.
    //
    // Third instance of one rule — ungrouped LoRA, an arm whose prepared
    // state the fire declines to build, and now recurrent state. What
    // cannot be replayed stays eager.
    let mut union =
        supergraph_enabled() && !family.recurrent();
    let lower_as = |g: GuardMode| {
        lower_with(&plan, &fire_rows, Fire { captures_across_splits: false }, g).map_err(|e| {
            eprintln!("[driver-cuda-new] launch: uncovered: {e:?}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let mut lowered = lower_as(if union { GuardMode::Union } else { GuardMode::Resolve })?;

    // A union this fire cannot record is not a union worth building, and
    // the decision has to be made HERE — before the arena, the pins and
    // the attention context are sized against it — because falling back
    // later would run one lowering's program against another's offsets.
    //
    // The test is narrow on purpose: a SCORE-capturing dispatch needs a
    // plan raised for the full-attention variant, a folded-row layout and
    // an observation window, none of which a fire that wants no scores
    // prepares. Its launcher answers by throwing, which the shim can only
    // turn into a message and an abort. So a union that names one is
    // abandoned for this fire and the guards are answered instead.
    if union {
        let d = DispatchPlan::new(&plan, &lowered);
        // Two things make a union unservable for THIS fire, and both are
        // about prepared state rather than about the graph.
        //
        // A `_capture` dispatch wants a plan raised for the full-attention
        // variant, a folded-row layout and an observation window, none of
        // which a fire that wants no scores builds; its launcher answers
        // by throwing, which the shim can only turn into an abort.
        //
        // And the attention output slot has to be findable from the op
        // JOIN. The neighbour trick — "the launch after the dispatch is
        // the o_proj" — is what `Resolve` allows and `Union` does not,
        // because every arm is present and the neighbour belongs to some
        // other body. A deployment that states its attention as [q, o]
        // records no output in the join and has only the neighbour.
        let servable = !lowered.kernels.iter().any(|k| k.contains("_capture"))
            && {
                let name = if lowered
                    .kernels
                    .iter()
                    .any(|k| k == "attn::dispatch_attention_flashinfer_decode")
                {
                    "attn::dispatch_attention_flashinfer_decode"
                } else {
                    "attn::dispatch_attention_flashinfer_prefill_bf16"
                };
                lowered
                    .launches
                    .iter()
                    .position(|x| lowered.kernels[x.kernel as usize] == name)
                    .is_some_and(|fi| matches!(d.spec(fi).outs.first(), Some(Arg::Arena { .. })))
            };
        if !servable {
            union = false;
            lowered = lower_as(GuardMode::Resolve)?;
        }
    }
    let dplan = DispatchPlan::new(&plan, &lowered);

    // ── Device state, and all of it PERSISTENT now. ──
    //
    // The stream and the allocator used to be built here, per fire. The
    // stream because nothing needed it to outlive the call, and the
    // allocator because it was convenient — but an allocator that POOLS
    // and is rebuilt every fire has no pool, so every buffer a fire
    // wanted was a fresh `cudaMalloc`.
    //
    // Both are the shell's now. That is a saving on its own and it is the
    // precondition for run-ahead: a second fire cannot queue behind the
    // first onto a stream that dies with the first call.
    if state.fire_stream.is_none() {
        state.fire_stream =
            Some(crate::cuda::OwnedStream::new(0).map_err(|_| PIE_STATUS_DRIVER_ERROR)?);
    }
    if state.fire_alloc.is_none() {
        state.fire_alloc = Some(crate::cuda::Allocator::new());
    }
    let stream = state.fire_stream.as_ref().expect("just ensured");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = state.fire_alloc.as_ref().expect("just ensured");

    let need_pages = frame.required_kv_pages.max(
        kv_indices.iter().copied().max().map_or(1, |m| m + 1),
    );
    let page_size: i32 = 16;
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let head_dim_u = u32::try_from(head_dim_i).unwrap_or(0);
    // Per-layer pool geometry, family-decided: gemma-4's two layer kinds
    // disagree on head dim and its trailing layers own NO pool (they
    // attend through their source's pages — the load-time decision). A
    // `None` row is a shared layer; its VIEW mirrors the source below.
    let layer_geom: Vec<Option<(i32, u32)>> = (0..family.layers())
        .map(|l| {
            // A layer that attends through another's pages owns no pool;
            // its VIEW mirrors the source below.
            family
                .kv_source(l)
                .is_none()
                .then(|| (family.head_dim_of(l, head_dim_u) as i32, l))
        })
        .collect();
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
        family.kv_source(u32::try_from(i).unwrap_or(0)).map_or(i, |s| s as usize)
    };
    let layers: Vec<crate::launch::KvCacheLayerView> = (0..kv.pools.len())
        .map(|i| {
            let src = kv_source_of(i);
            let (k, v) = kv.pools[src].as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            let d = family.head_dim_of(u32::try_from(i).unwrap_or(0), head_dim_u) as i32;
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

    // The fire's descriptor arrays, POOLED like the arena and for the same
    // reason: a capture bakes an address, so the buffer has to be the same
    // one next fire with only its contents refreshed. Slots are positional
    // and this is the whole list of them.
    const S_IDS: usize = 0;
    const S_POS: usize = 1;
    const S_KV_INDICES: usize = 2;
    const S_KV_INDPTR: usize = 3;
    const S_KV_LENS: usize = 4;
    const S_QO: usize = 5;
    const S_W_PAGE: usize = 6;
    const S_W_OFF: usize = 7;
    let d_ids = state.fire_arrays.upload_u32(&alloc, S_IDS, token_ids, stream.as_ref())?;
    let d_pos = state.fire_arrays.upload_u32(&alloc, S_POS, position_ids, stream.as_ref())?;
    let d_kv_indices =
        state.fire_arrays.upload_u32(&alloc, S_KV_INDICES, kv_indices, stream.as_ref())?;
    let d_kv_indptr =
        state.fire_arrays.upload_u32(&alloc, S_KV_INDPTR, kv_indptr, stream.as_ref())?;
    let d_kv_lens =
        state.fire_arrays.upload_u32(&alloc, S_KV_LENS, kv_lens, stream.as_ref())?;
    let d_qo = state.fire_arrays.upload_u32(&alloc, S_QO, qo_indptr, stream.as_ref())?;

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
    let d_w_page = state.fire_arrays.upload_u32(&alloc, S_W_PAGE, &w_page, stream.as_ref())?;
    let d_w_off = state.fire_arrays.upload_u32(&alloc, S_W_OFF, &w_off, stream.as_ref())?;
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
    let decode_plan_full_ptr = if let Some((d_sliding, d_full)) = family.decode_plan_head_dims() {
        if states_decode_dispatch {
            // TWO decode plans, one per layer kind — the C++'s
            // `decode_plan_sliding` / `decode_plan_full` pair, because
            // the kinds disagree on head dim and the planner bakes it in.
            decode_plan.plan_decode_variant(
                kv_indptr,
                model.hf.num_attention_heads,
                kv_heads_i,
                d_sliding as i32,
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
                d_full as i32,
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

    let arena_bytes = lowered.arena_bytes.max(64);
    let arena_ptr = state.fire_arrays.arena(&alloc, arena_bytes)?;
    let exec_frame = Frame { arena: arena_ptr, arena_bytes };

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
    for (&v, &w) in &named_widths {
        // fp32-wide: the GDN seam pins are f32; llama-like's are bf16 and
        // simply leave half the pin unread.
        state.fire_arrays.named(&alloc, v, rows * w as usize * 4, stream.as_ref())?;
    }
    // NO SCORE SINK, deliberately, and this is what makes the capture
    // path safe rather than merely optional.
    //
    // A fire that wants no scores prepares no score path: no plan raised
    // for the full-attention variant, no folded-row layout, no
    // observation window. The score-capturing dispatch needs all three,
    // and its launcher REFUSES by throwing — which the generated shim
    // prints and then aborts on, because an exception crossing the C ABI
    // has nowhere else to go.
    //
    // So the arm has to refuse before the launcher is reached, and a null
    // sink is how it knows to. `run_captured` then returns a refusal, the
    // capture is abandoned, and the fire runs eagerly — the same answer
    // ungrouped LoRA gets. Publishing a plausible-looking empty sink
    // instead would put the decision inside a launcher that can only
    // answer by killing the process.
    let d_score_indptr: *const u32 = core::ptr::null();
    let d_scores: *mut std::ffi::c_void = core::ptr::null_mut();

    let named_bufs = &state.fire_arrays.named;

    // ── The hybrid's GDN context: driver-owned slabs, instance slots. ──
    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::cuda::DeviceBuffer> = None;
    if let Some(shape) = family.gdn_shape() {
        let (conv_stride, state_stride, state_elem) =
            (shape.conv_stride, shape.state_stride, shape.state_elem);
        const GDN_SLOTS: u32 = 8;
        if state.gdn.is_none() {
            let mut slabs = Vec::new();
            for l in 0..shape.layers {
                if !shape.linear_layers.contains(&l) {
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
        let _ = to_i32;
        gdn_ctx = Some(GdnCtx {
            k_h: shape.k_h,
            v_h: shape.v_h,
            k_d: shape.k_d,
            v_d: shape.v_d,
            conv_dim: shape.conv_dim,
            conv_k: shape.conv_k,
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
    let (q_pin, o_off) = if !family.pins_attention_values() {
        (None, None)
    } else {
        let dispatch_name = if states_decode_dispatch {
            "attn::dispatch_attention_flashinfer_decode"
        } else {
            "attn::dispatch_attention_flashinfer_prefill_bf16"
        };
        let Some(fi) = lowered
            .launches
            .iter()
            .position(|x| lowered.kernels[x.kernel as usize] == dispatch_name)
        else {
            eprintln!(
                "[driver-cuda-new] launch: the lowering states no {dispatch_name}"
            );
            return Err(PIE_STATUS_UNSUPPORTED);
        };
        let q_pin = lowered.launches[fi]
            .args
            .clone()
            .find_map(|ai| match &lowered.args[ai as usize] {
                Arg::Named { value, .. } => Some(*value),
                _ => None,
            });
        // The dispatch's OUTPUT, read off its own op join.
        //
        // This used to be `launches[fi + 1]`'s first operand — "the launch
        // after the dispatch is the o_proj, and its input is the slot the
        // dispatch wrote". True under `Resolve`, where the guard has
        // already deleted every arm the fire did not take, and false under
        // `Union`, where every arm is present and the next launch belongs
        // to some other guard's body.
        //
        // A value found by counting launches is a fact derived from where
        // a statement SITS. The join says it: the attention statement
        // carries one arg (q) and its output placement, which is exactly
        // the slot wanted. Same read the executor's arms make.
        // Prefer the join, fall back to the neighbour.
        //
        // The join is the STATED read: the attention statement carries its
        // output placement, which is the slot the o_proj goes on to read.
        // Where a deployment spells the attention with [q, o] as SSA args
        // the join records no output of its own, and there the old
        // positional read is still the only answer available.
        //
        // Positional is what breaks under `Union` — every guard arm is
        // present, so the launch after the dispatch belongs to some other
        // body — which is why the join is tried first rather than second.
        let o_off = match dplan.spec(fi).outs.first() {
            Some(Arg::Arena { at, .. }) => *at,
            _ => match lowered.launches.get(fi + 1).map(|n| &lowered.args[n.args.start as usize]) {
                Some(Arg::Arena { at, .. }) => *at,
                _ => {
                    eprintln!(
                        "[driver-cuda-new] launch: {dispatch_name} states no arena \
                         output, and the launch after it is not one either"
                    );
                    return Err(PIE_STATUS_UNSUPPORTED);
                }
            },
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
    let sm_scale = family.sm_scale(u32::try_from(model.hf.head_dim).unwrap_or(1));
    let window_by_layer = family.window_by_layer(model.hf.sliding_window);
    let is_gemma4 = family.planless_prefill();
    let attn = AttnCtx {
        decode_plan: decode_plan.as_ptr(),
        decode_plan_full: decode_plan_full_ptr,
        prefill_plan: prefill_plan.as_ptr(),
        workspace: ws.view(),
        layers,
        q_out: q_pin
            .and_then(|v| named_bufs.get(&v).map(|b| b.as_ptr()))
            .unwrap_or(core::ptr::null_mut()),
        score_out: d_scores.cast(),
        score_indptr_d: d_score_indptr.cast(),
        o_out: o_off
            .map_or(core::ptr::null_mut(), |off| unsafe {
                arena_ptr.cast::<u8>().add(off)
            }
            .cast()),
        kv_page_indices_d: d_kv_indices.cast(),
        kv_page_indptr_d: d_kv_indptr.cast(),
        kv_last_page_lens_d: d_kv_lens.cast(),
        qo_indptr_d: d_qo.cast(),
        qo_indptr_h: if is_gemma4 { qo_indptr.as_ptr() } else { core::ptr::null() },
        kv_page_indptr_h: if is_gemma4 { kv_indptr.as_ptr() } else { core::ptr::null() },
        num_requests: requests as i32,
        num_pages_in_batch: kv_indices.len() as i32,
        first_token: 0,
        w_page_d: d_w_page.cast(),
        w_off_d: d_w_off.cast(),
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
    // The family's per-layer tables and named constants — the C++
    // parse-time vectors (`per_layer_rope_theta`, `rotary_of`) and the
    // prologue's `scale.*` values plus the load-read layer scalars. A
    // family whose rope is one theta and whose epilogue caps nothing
    // answers with empties.
    let FamilyTables { theta_by_layer, rotary_by_layer, softcap, ple_dim, scales } =
        family.tables(model);
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
        token_ids: d_ids.cast_mut().cast(),
        positions: d_pos.cast_mut().cast(),
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
    let regions = AttnRegions::whole(Some(&attn));
    let result = if union {
        capture_or_replay(
            &mut state.supergraph,
            state.fire_arrays.epoch,
            u64::from(model.hf.num_hidden_layers.unsigned_abs()),
            &plan, &fire_rows, &lowered, &dplan, exec_frame, &mut resolver, &ctx,
            regions, gdn_ctx.as_ref(), &alloc, stream.as_ref(), requests, rows, class,
        )
    } else {
        run(&lowered, &dplan, exec_frame, &mut resolver, &ctx, regions, gdn_ctx.as_ref())
    };
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
