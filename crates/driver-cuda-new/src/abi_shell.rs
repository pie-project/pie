//! The thirteen `pie_cuda_*` exports — the cutover's door (retirement
//! plan phase D).
//!
//! The engine consumes a driver through `pie_driver_abi.h`, whose Rust
//! source of truth is `driver_api::local`. This module DEFINES the symbols
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
// caller-side contract is `driver_api::local`'s `unsafe extern` block;
// marking the DEFINITIONS `unsafe fn` would change their ABI type for a
// fact the boundary already states.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use driver_api::local::{
    PIE_DRIVER_ABI_VERSION, PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_OK,
    PIE_STATUS_UNSUPPORTED, PieChannelDesc, PieChannelEndpointBinding, PieCompletion,
    PieDriver, PieDriverCaps, PieDriverCreateDesc, PieEncodeDesc, PieFrameDesc,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieModelLoadDesc,
    PiePoolResizeDesc, PieProgramDesc, PieStateCopyDesc, validate_create_out_params,
};

/// The shell's state — what `PieDriver` points at.
struct Shell {
    /// The capabilities JSON `create` hands back; owned here so the
    /// pointer in [`PieDriverCaps`] lives as long as the driver.
    caps: Vec<u8>,
    /// `[model] descriptor` from the boot TOML, for HF snapshots whose
    /// descriptor does not ride inside the checkpoint.
    boot_descriptor: Option<std::path::PathBuf>,
    /// Does this driver hand its completions to a stream callback and
    /// return with the fire still queued? See `runahead_env`.
    runahead: bool,
    /// How the KV pages are stored — `[driver] kv_cache_dtype`.
    ///
    /// `store/kv_format.rs` has had nine spellings since the port, the
    /// layout plans their scale planes and `kv_paged.cu` switches on the
    /// scheme to write them. What was missing was any way to ASK for one:
    /// the shell built its pages by hand and could only build bf16.
    kv_format: crate::store::KvCacheFormat,
    /// The traced-and-lowered program for a fire SHAPE, kept.
    ///
    /// Tracing the forward, lowering it and joining the ops back onto the
    /// launches is ~3.3 ms per fire on a 0.6B decode — 1.25 ms of trace,
    /// 1.85 ms of lowering, 0.17 ms of dispatch plan — and it is a pure
    /// function of the key below. It was being redone on every launch, which
    /// on a decode is most of the time the call takes.
    ///
    /// Keyed by what the answer can depend on: which model, which fire class,
    /// how many rows, and whether a union was ASKED for. The union may still
    /// be declined after the servability test, so the answer records what was
    /// actually built rather than what was requested.
    lowerings: std::collections::BTreeMap<LoweringKey, LoweredFire>,
    /// The cuBLAS handle, created once.
    ///
    /// It used to be created and DESTROYED inside every fire, and
    /// `cublasDestroy` costs **3.2 ms** — measured, and it was three quarters
    /// of what a warm decode spent being issued. Creating one per fire also
    /// meant a fresh workspace allocation each time, which is the part that
    /// actually takes the time.
    ///
    /// The stream is rebound per fire instead, which is what `cublasSetStream`
    /// is for.
    cublas: Option<crate::cuda::cublas::CublasHandle<cudarc::cublas::sys::cublasHandle_t>>,
    /// The fire's predicate word, allocated once.
    ///
    /// PERSISTENT for two reasons, and the second is correctness. It used to
    /// be built and dropped inside every `capture_or_replay`, which is a
    /// `cudaMalloc` and a `cudaFree` per fire — and `cudaFree` SYNCHRONIZES
    /// THE DEVICE, so the run-ahead the rest of this file is built around was
    /// being undone by the graph path. It is also the address a captured
    /// graph BAKES: a word that is freed and reallocated between two replays
    /// is one whose address the exec has no reason to still be right about.
    preds: Option<crate::cuda::PredicateWord>,
    /// The fire's peel-window word, allocated once. Same reasoning as
    /// [`Shell::preds`]: it was a `cudaMalloc` and a `cudaFree` per fire, and
    /// `cudaFree` synchronizes the device.
    peel_win: Option<crate::cuda::PeelWindowWord>,
    /// The pinned host buffer the logits D2H lands in, grown to fit and
    /// reused. The shell's rather than the fire's because a stream
    /// callback may not free it — see `FireDebt::staging`.
    logits_staging: Option<crate::cuda::PinnedBuf>,
    /// Is this boot MEASURING rather than serving? `[driver]
    /// calibrate_planner`.
    ///
    /// READ BY NOTHING YET, and the reason is recorded in
    /// `.wiki/new-driver/next.md`: the sweep works — it measures a ladder and
    /// stores a winner — but only below the shape this driver ADVERTISES,
    /// because a fire it cannot bind aborts inside a CUDA seam
    /// (`attention_workspace`'s `assert!` on the runtime status) instead of
    /// returning one. A probe that cannot fail safely cannot probe, so the
    /// flag is parsed and nothing acts on it until the fire path refuses.
    #[allow(dead_code)]
    calibrating: bool,
    /// The CUDA device ordinal this driver binds, from `[driver] device`.
    ///
    /// It was hardwired to 0 in both places that bind, which is wrong on any
    /// box with more than one GPU: an operator who asks for device 1 gets
    /// device 0 and no diagnostic. A rank of a tensor-parallel group would
    /// need one per rank; that is refused for other reasons, so this is the
    /// single-device answer and honest about being one.
    device_ordinal: i32,
    /// This driver's place in its tensor-parallel group, from
    /// `[driver] tp_rank` / `tp_size`. One rank, rank zero, unless told
    /// otherwise — and the two numbers travel together into both the load
    /// plan and the KV geometry, so a rank cannot be one width for its
    /// weights and another for its cache.
    tp_rank: u32,
    tp_size: u32,
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
    notify: driver_api::local::PieRuntimeNotifyFn,
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
    /// The fire that is still running, if any — see [`InFlight`]. One
    /// slot, so the driver runs exactly one fire ahead.
    in_flight: std::collections::VecDeque<InFlight>,
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
    /// The control kernels, compiled once for this device's architecture.
    ///
    /// Not per instance and not per program: `readiness` and `commit`
    /// depend on the architecture and nothing else, so a second instance
    /// finds them built. Built lazily because a driver that never fires a
    /// program should not pay NVRTC for two kernels it will not call.
    ptir_control: Option<crate::ptir::Control>,
    /// One instance's device rings, by instance id.
    ///
    /// Per INSTANCE by necessity rather than by choice: the rings are
    /// sized from that instance's own `channel_ids`, in that order,
    /// because a program names a channel by index into it. And they must
    /// outlive a fire — rebuilding zeroes the cursors, and a cursor is the
    /// only record of what a previous fire published.
    ptir_sessions: std::collections::BTreeMap<u64, crate::ptir::session::Session>,
    /// The adopted plans, by program id. Separate from the compiled
    /// modules because a program can be adopted and REJECTED — an
    /// unexecutable plan is still a plan, and the reason it was rejected
    /// is what the launch that needs it has to report — while a
    /// compilation only exists for a program that got that far.
    ptir_plans: std::collections::BTreeMap<u64, driver::ExecPlan>,
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
#[derive(Clone, Copy)]
struct ChannelState {
    mirror: *mut std::ffi::c_void,
    words: *mut std::ffi::c_void,
    mirror_bytes: usize,
    /// WIRE bytes per cell — bit-packed for bools.
    cell_bytes: usize,
    /// `capacity + 1` — the ring modulus.
    ring: u32,
    host_role: u8,
    /// Lanes in one cell, and the cell's element type.
    ///
    /// Kept because `cell_bytes` cannot be inverted: a bool cell is
    /// `numel.div_ceil(8)` wire bytes, so eight lanes and one lane are
    /// both one byte. The DEVICE ring is sized in NATIVE bytes
    /// (`ptir::ring::ChannelShape`), and deriving one from the other needs
    /// the shape rather than the width — which is why a driver that only
    /// ever wrote to the host mirror could get away without these and one
    /// that fires a program cannot.
    numel: usize,
    dtype: driver::tensor_ir::DType,
}

impl ChannelState {
    /// This channel as the device rings want it.
    fn shape(&self) -> crate::ptir::ring::ChannelShape {
        crate::ptir::ring::ChannelShape {
            numel: self.numel,
            dtype: self.dtype,
            // `ring` is `capacity + 1`, and `ChannelShape` states the
            // capacity — the one place the two vocabularies differ, and
            // the reason this is a method rather than a struct literal at
            // each call.
            capacity: self.ring.saturating_sub(1),
        }
    }
}

/// A fire's transient device memory, kept alive until the fire retires.
///
/// **`cudaFree` synchronizes the device.** So a fire that returns with
/// work still queued and then drops its scratch drains its own stream
/// inside the call — which is exactly what the first run-ahead attempt
/// did, and `a_launch_returns_before_its_fire_retires` caught it: the
/// completion fired on the calling thread because freeing the LSE buffer
/// waited for the fire that was still using it.
///
/// Freeing on the CALLBACK thread is not the fix either: CUDA forbids
/// calling into the runtime from a host callback, and `cudaFree` is the
/// runtime.
///
/// So the buffers outlive the call and the NEXT launch reclaims them,
/// after waiting on the event that says the fire they belong to is done.
/// That wait is where the run-ahead depth lives: with one holder the
/// driver runs one fire ahead, which is the whole of the property and the
/// smallest thing that has it.
struct InFlight {
    done: crate::cuda::Event,
    /// Ordinary scratch. Named for what it is rather than listed, because
    /// the point is that nothing here is read again — it is held only so
    /// that dropping it does not synchronize at the wrong moment.
    scratch: Vec<crate::cuda::DeviceBuffer>,
    /// Channels closed while this fire was queued, freed when it retires.
    ///
    /// A fire's debt COPIES the `ChannelState` it will publish into, so a
    /// `close_channel` that freed the mirror immediately would leave the
    /// stream-ordered callback writing into memory the allocator had already
    /// taken back. Run-ahead is what made that reachable: before it, the
    /// publish happened on the calling thread and was over before any close
    /// could be processed.
    ///
    /// Deferring rather than refcounting because the lifetime is already
    /// modelled here -- an in-flight fire is exactly the thing that might
    /// still be holding it, and this queue already knows when one retires.
    closed_channels: Vec<ChannelState>,
}

/// Give back what a retired fire was holding.
///
/// The scratch drops on its own; the channels do not, because a
/// `ChannelState` is a pair of raw host allocations this shell owns rather
/// than a Rust value with a destructor. Freeing them HERE rather than in
/// `close_channel` is the whole point -- see `InFlight::closed_channels`.
fn retire(fire: InFlight) {
    for ch in &fire.closed_channels {
        ch.free();
    }
}

/// How many fires the driver may have queued ahead of the GPU.
///
/// Backpressure by SCRATCH, not by time: each in-flight fire holds the
/// buffers it is still writing, so the bound is on how much the driver is
/// carrying rather than on how far ahead it has run.
///
/// Two, which is the C++'s `kSchedulerMaxInFlight` minus the frame the
/// engine itself is holding. One is what this had, and one is not run-ahead:
/// the call that would queue fire n+1 waited for fire n.
const RUNAHEAD_DEPTH: usize = 2;

/// What a lowering can depend on: see [`Shell::lowerings`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct LoweringKey {
    model_id: u64,
    class: model_compiler::trace::FireClass,
    rows: u32,
    /// A digest of the fire's ROWS, not just how many.
    ///
    /// The lowering resolves five guards off the rows — `HasLora`,
    /// `HasCustomMask`, `HasStageHooks`, `HasWriteDesc`,
    /// `WantsAttnScore` — and sizes the epilogue off how many of them
    /// sample. All of that was invariant while every row was
    /// `Row { samples: true, ..default() }`, so the count was a complete
    /// key. It is not one now: a fire carrying an adapter and a fire
    /// without one have the same shape and different launch lists, and a
    /// key that could not tell them apart would serve the first
    /// lowering to arrive to both.
    rows_digest: u64,
    union_asked: bool,
}

/// FNV-1a over the rows' axes.
///
/// Not a hash of the struct: `Row` is not `Hash`, and adding the derive
/// would make every future field silently part of a cache key. Naming
/// the axes here means a new one has to be added deliberately.
fn digest_rows(rows: &[model_compiler::lower::Row]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u64| {
        h ^= b;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for r in rows {
        eat(u64::from(r.multi_token));
        eat(u64::from(r.custom_mask));
        eat(u64::from(r.hooked));
        eat(u64::from(r.lora));
        eat(u64::from(r.write_desc));
        eat(u64::from(r.wants_scores));
        eat(u64::from(r.samples));
        eat(r.depth_k.map_or(u64::MAX, u64::from));
    }
    h
}

/// A traced, lowered and joined program, and whether it kept its union.
struct LoweredFire {
    plan: model_compiler::trace::ForwardPlan,
    lowered: model_compiler::lower::Lowered,
    dplan: crate::model::executor::DispatchPlan,
    union: bool,
}

/// EVERYTHING A FIRE STILL OWES WHEN ITS WORK IS ENQUEUED.
///
/// The driver used to pay these debts on the calling thread, after a
/// `stream.synchronize()`: widen the logits, publish them, mark the
/// terminal cells, notify. That is why `pie_cuda_launch` was synchronous
/// and why the engine's `frame_dispatch_depth` was serialized by the
/// driver — the call that would enqueue fire n+1 had not returned.
///
/// So the debts move into a stream-ordered HOST CALLBACK. It runs when
/// everything queued before it has retired, on a CUDA-owned thread, and
/// everything it needs is here because a callback cannot borrow.
///
/// The staging buffer is the reason this is a struct rather than a
/// closure: the D2H used to land in a `Vec` on the stack, which is a
/// use-after-free the moment the call returns before the copy does.
///
/// **The debt BORROWS that buffer; it does not own it.** A callback may
/// not make CUDA runtime calls, and freeing pinned memory is one — an
/// owned `PinnedBuf` here means `cudaFreeHost` on a CUDA-owned thread,
/// which is undefined and shows up as heap corruption several fires
/// later. It is the same rule that put the device scratch in `InFlight`.
/// The buffer is the shell's, reused every fire.
struct FireDebt {
    /// The bf16 logits, D2H'd into the SHELL's pinned staging by a
    /// stream-ordered copy, as (pointer, length).
    ///
    /// PINNED, and that is the whole reason the shell keeps one:
    /// `cudaMemcpyAsync` into pageable host memory blocks until the copy
    /// completes, so a `Vec` here drains the stream inside
    /// `pie_cuda_launch` and undoes the run-ahead.
    staging: Option<(*const u8, usize)>,
    /// One `(reader channel, logits row)` per request in the frame.
    ///
    /// A LIST, because a frame carries a roster and every request in it
    /// is owed an answer. This was one channel and one row — the roster's
    /// FIRST instance and `rows - 1` — so a two-request batch published
    /// request 0's vocabulary and returned request 1 nothing.
    ///
    /// The row is not the request's index: request `r` owns `qo_indptr[r]
    /// ..qo_indptr[r + 1]`, so its answer is at `qo_indptr[r + 1] - 1`. On
    /// a decode that equals `r`; on a prefill it does not.
    readouts: Vec<(ChannelState, usize)>,
    vocab: usize,
    /// The terminal cells this frame publishes, and the completion the
    /// runtime is waiting on.
    cells: Vec<*mut driver_api::local::PieTerminalCell>,
    completion: PieCompletion,
    notify: driver_api::local::PieRuntimeNotifyFn,
    notify_ctx: *mut std::ffi::c_void,
}

// The debt crosses to a CUDA callback thread. Every field is either owned
// bytes or a raw pointer into memory the runtime keeps alive for the
// driver's lifetime — the channel's mirror and words are the engine's
// mapping, the terminal cells are the frame's.
unsafe impl Send for FireDebt {}

/// The stream-ordered callback: pay the debt, then drop it.
///
/// # Safety
///
/// `data` is a `Box<FireDebt>` leaked by the enqueuing side. CUDA forbids
/// calling back into the runtime from here, and nothing below does: this
/// is host memory, a volatile write and a function pointer.
unsafe extern "C" fn retire_fire(data: *mut std::ffi::c_void) {
    if data.is_null() {
        return;
    }
    let debt = unsafe { Box::from_raw(data.cast::<FireDebt>()) };

    // The logits, widened bf16 -> f32 and published. The widening is the
    // wire's, not the model's: the ring's cell is f32 and the device
    // wrote bf16, so the shift is the conversion.
    if let Some(&(ptr, len)) = debt.staging.as_ref()
        && debt.vocab > 0
    {
        // SAFETY: the shell's staging buffer, alive for the driver's
        // lifetime, and the D2H that filled it is ordered before this
        // callback on the same stream. ONE copy carries every row, so
        // each request reads its own out of the same staging.
        let staged = unsafe { std::slice::from_raw_parts(ptr, len) };
        for (ch, row) in &debt.readouts {
            let mut cell = vec![0u8; debt.vocab * 4];
            for t in 0..debt.vocab {
                let off = (row * debt.vocab + t) * 2;
                if off + 1 < staged.len() {
                    let bits = u16::from_le_bytes([staged[off], staged[off + 1]]);
                    cell[t * 4..t * 4 + 4]
                        .copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
                }
            }
            if !ch.publish(&cell) {
                eprintln!(
                    "[driver-cuda-new] launch: logits ring full; a request dropped its output"
                );
            }
        }
    }

    // Then the terminal cells, then the notify — in that order, with a
    // release between, because the runtime reads the cells the moment the
    // notify lands.
    for &cell in &debt.cells {
        if !cell.is_null() {
            unsafe {
                std::ptr::addr_of_mut!((*cell).outcome)
                    .write_volatile(driver_api::local::PIE_TERMINAL_OUTCOME_SUCCESS);
            }
        }
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    if let Some(notify) = debt.notify {
        unsafe { notify(debt.notify_ctx, debt.completion.wait_id, debt.completion.target_epoch) };
    }
}

impl ChannelState {
    /// Publish one wire cell: write it at `tail % ring`, then advance the
    /// tail word with release ordering — the reader (the engine) consumes
    /// from the head. The writer side of the C++ ring, host-resident.
    /// The host plane of this channel, as `ptir::bridge` sees it.
    ///
    /// One definition of the ring, and it is that one. This used to carry
    /// its own publish — the same cursor arithmetic, the same release
    /// fence — beside the bridge's, which is two statements of a layout
    /// that has to agree byte for byte with the engine's poller.
    ///
    /// # Safety
    ///
    /// The returned view borrows the mirror and the words this channel
    /// owns; `close_channel` defers the free onto an in-flight fire, so
    /// they outlive any view a fire holds.
    fn host_plane(&self) -> crate::ptir::bridge::HostChannel {
        debug_assert!(self.cell_bytes * self.ring as usize <= self.mirror_bytes);
        unsafe {
            crate::ptir::bridge::HostChannel::new(
                self.mirror,
                self.words,
                self.cell_bytes,
                self.ring,
            )
        }
    }

    fn publish(&self, cell: &[u8]) -> bool {
        self.host_plane().publish(cell)
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
    /// The unconditional attention-score sink — see [`Self::score`].
    score: Option<crate::cuda::DeviceBuffer>,
    /// The unconditional custom attention mask — see [`Self::mask`].
    mask: Option<crate::cuda::DeviceBuffer>,
    /// The driver-owned attention landing buffer — see [`Self::attn_out`].
    attn_out: Option<crate::cuda::DeviceBuffer>,
    named: std::collections::BTreeMap<model_compiler::trace::ValueId, crate::cuda::DeviceBuffer>,
    /// The small per-fire u32 descriptor arrays, by slot.
    slots: Vec<Option<crate::cuda::DeviceBuffer>>,
    /// PINNED host staging for those uploads, ONE PER SLOT.
    ///
    /// A `cudaMemcpyAsync` out of PAGEABLE host memory is SYNCHRONOUS — it
    /// blocks until the copy lands — and there are eight of these per fire.
    /// A `Vec` here therefore drained the stream eight times inside
    /// `pie_cuda_launch`, which is the same trap the logits D2H fell into.
    ///
    /// Per slot and not one shared buffer, because pinning is what makes the
    /// copy genuinely ASYNCHRONOUS: a single buffer would be overwritten by
    /// the next slot's upload while the previous copy was still queued. Two
    /// fires cannot collide on one slot either — the next launch waits on the
    /// previous fire's event before it reclaims anything (`InFlight`).
    staging: Vec<Option<crate::cuda::PinnedBuf>>,
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

    /// The score sink, at least `bytes` wide — the same pooling discipline
    /// the arena gets, for the same reason: a captured exec bakes the
    /// address, so the sink outlives its fire and grows rather than moving
    /// every launch. See [`crate::model::attn_score::plan_score_sink`].
    ///
    /// Returns the base and writes the CSR into its own carved span, so a
    /// replay whose KV lengths moved still scores against this fire's
    /// truth rather than the recording fire's.
    fn score(
        &mut self,
        alloc: &crate::cuda::Allocator,
        plan: &crate::model::attn_score::ScoreSinkPlan,
        stream: crate::cuda::StreamRef<'_>,
    ) -> Result<*mut std::ffi::c_void, i32> {
        if self.score.as_ref().is_none_or(|b| b.len() < plan.bytes) {
            self.score = Some(alloc.alloc(plan.bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        let b = self.score.as_mut().expect("just ensured");
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)
            .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        Ok(b.as_ptr())
    }

    /// The attention output slot, for a fire whose op join names none.
    ///
    /// `AttnCtx::o_out` is driver-owned by design -- a guard region's
    /// launches record no SSA output of their own -- so this is where that
    /// design finally has storage instead of a refusal. Pooled like the
    /// arena, because a capture bakes the address.
    fn attn_out(
        &mut self,
        alloc: &crate::cuda::Allocator,
        bytes: usize,
    ) -> Result<*mut std::ffi::c_void, i32> {
        let bytes = bytes.max(64);
        if self.attn_out.as_ref().is_none_or(|b| b.len() < bytes) {
            self.attn_out = Some(alloc.alloc(bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        Ok(self.attn_out.as_ref().expect("just ensured").as_ptr())
    }

    /// The custom attention mask, pooled like the score sink and for the
    /// same reason. See [`crate::model::page_mask::element_mask`].
    fn mask(
        &mut self,
        alloc: &crate::cuda::Allocator,
        plan: &crate::model::page_mask::element_mask::ElementMaskPlan,
        stream: crate::cuda::StreamRef<'_>,
    ) -> Result<*mut std::ffi::c_void, i32> {
        if self.mask.as_ref().is_none_or(|b| b.len() < plan.bytes) {
            self.mask = Some(alloc.alloc(plan.bytes).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        let b = self.mask.as_mut().expect("just ensured");
        b.write_at(0, &plan.mask, stream).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)
            .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        Ok(b.as_ptr())
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
        if self.staging.len() <= slot {
            self.staging.resize_with(slot + 1, || None);
        }
        let live = vals.len() * 4;
        let need = live.max(4);
        if self.staging[slot].as_ref().is_none_or(|p| p.len() < need) {
            self.staging[slot] =
                Some(crate::cuda::PinnedBuf::new(need).map_err(|_| PIE_STATUS_EXHAUSTED)?);
        }
        let pin = self.staging[slot].as_mut().expect("just sized");
        for (dst, v) in pin.as_mut_slice()[..live].chunks_exact_mut(4).zip(vals) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        if self.slots[slot].as_ref().is_none_or(|b| b.len() < need) {
            self.slots[slot] = Some(alloc.alloc(need).map_err(|_| PIE_STATUS_EXHAUSTED)?);
            self.epoch += 1;
        }
        let src = &self.staging[slot].as_ref().expect("just sized").as_slice()[..live];
        let b = self.slots[slot].as_mut().expect("just ensured");
        b.copy_from_host(src, stream).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
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

/// The KV cache, and the buffers backing it.
///
/// The geometry is [`store::kv_cache_live::KvCache`](crate::store::kv_cache_live::KvCache)'s:
/// it plans one slot per layer, allocates only for the layers that OWN
/// pages, and hands out the `KvCacheLayerView` a kernel is launched with.
/// The shell built all of that by hand until now — two `cudaMalloc`s and
/// fifteen literal fields per layer — and the hand-built version could
/// not express a format, a scale plane or an envelope, which is why
/// `kv_cache_dtype` was hardwired `"bf16"`.
///
/// `held` is what makes the port safe in Rust. `KvCache` keeps raw
/// pointers and has no `Drop`, because the C++ it came from had an arena
/// that outlived it; here the `DeviceBuffer`s its ops allocated are kept
/// beside it, so replacing the pair on a resize frees the old one.
///
/// # Why the accessors
///
/// Four call sites in `copy_kv` and `resize_pool` used to recover the
/// same two facts from a buffer's LENGTH: a layer's page stride and,
/// from that, its head dim. Both are things the layout states.
struct KvState {
    cache: crate::store::kv_cache_live::KvCache<crate::store::kv_cache_live::AllResident>,
    /// Backing store for `cache`; dropping this frees the pages.
    _held: Vec<crate::cuda::DeviceBuffer>,
    num_pages: u32,
}

impl KvState {
    /// The pages `layer` owns, or `None` if it reads through another's.
    fn owned(&self, layer: usize) -> Option<(*mut core::ffi::c_void, *mut core::ffi::c_void)> {
        let l = i32::try_from(layer).ok()?;
        let slot = self.cache.layout().slots().get(layer)?;
        if slot.is_alias() {
            return None;
        }
        Some((self.cache.k(l), self.cache.v(l)))
    }

    /// Bytes of one page at `layer` — its own stride, so the two-head-dim
    /// families move the right amount per layer.
    fn page_bytes(&self, layer: usize) -> Option<usize> {
        let slot = self.cache.layout().slots().get(layer)?;
        let k = slot.k.as_ref()?;
        Some(usize::try_from(k.nbytes()).unwrap_or(0) / self.num_pages.max(1) as usize)
    }

    /// How many layers the cache describes.
    fn layers(&self) -> usize {
        self.cache.layout().slots().len()
    }

    /// `layer`'s head dim.
    ///
    /// The config's single number is wrong for the families whose layers
    /// disagree, and this is the extent the copy will actually stride
    /// through.
    fn head_dim(&self, layer: usize) -> Option<i32> {
        let slot = self.cache.layout().slots().get(layer)?;
        slot.k.as_ref()?;
        Some(self.cache.layout().head_dim_at(i32::try_from(layer).ok()?))
    }

    /// What a kernel is handed for each layer.
    fn views(&self) -> Vec<crate::launch::KvCacheLayerView> {
        (0..self.layers())
            .map(|l| self.cache.layer_view(i32::try_from(l).unwrap_or(0)))
            .collect()
    }

    /// Every layer owns pages, and every page is `page_bytes`.
    ///
    /// The host swap path needs both halves: it has ONE page stride for
    /// all layers, so a per-layer head dim breaks it, and it has no way
    /// to skip a layer that owns nothing. A wrong-sized host page is
    /// corruption rather than degradation, so a shared-page stack is
    /// refused rather than approximated.
    fn uniform_stride(&self, page_bytes: usize) -> bool {
        (0..self.layers()).all(|l| self.page_bytes(l) == Some(page_bytes))
    }
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
    /// Every tensor the plan named, as a span of the arena. A SPAN and not
    /// an allocation: a resident plan lays the whole model out contiguously,
    /// so a weight is an offset into one buffer rather than one of a thousand
    /// `cudaMalloc`s.
    weights: std::collections::BTreeMap<String, crate::loader::stage::WeightSpan>,
    /// The arena, and anything the plan published outside it. Held so the
    /// spans above stay valid; never indexed.
    #[allow(dead_code)]
    owned: Vec<crate::cuda::DeviceBuffer>,
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
    /// The group this rank's weights were sharded for, carried from the
    /// shell so a family's facts and its load plan cannot disagree about
    /// how wide a rank is. A forward derivation reads it to decide whether
    /// its landing needs a collective.
    tp_size: u32,
}

impl LoadedModel {
    /// The device pointer for a name — the live half of the executor's
    /// `Resolver::weight`. Checkpoint names, fused names and aliases all
    /// answer. `launch` is its caller; until that arm lands it is only
    /// the load test's assertion surface.
    #[allow(dead_code)]
    fn weight(&self, name: &str) -> Option<*const std::ffi::c_void> {
        if let Some(b) = self.weights.get(name) {
            return Some(b.ptr.cast_const());
        }
        let target = self.aliases.get(name)?;
        self.weights.get(target).map(|b| b.ptr.cast_const())
    }
}

/// The capabilities this shell can honestly claim today.
const CAPS_JSON: &str =
    r#"{"driver":"driver-cuda-new","status":"phase-d-shell","abi":24}"#;

/// The device-ring shapes of one instance's channels, in the order the
/// program indexes them.
///
/// ORDER IS THE CONTRACT and it is the whole reason this is a function.
/// A compiled program refers to a channel by INDEX — `Op::ChanRead(0)`,
/// `Op::ChanPut { chan: 1, .. }` — and that index is a position in the
/// instance's `channel_ids`, not an id and not a registration order. So
/// building the rings means walking that list, and a missing channel is a
/// refusal rather than a gap to skip: skipping would renumber every
/// channel after it, and the program would read someone else's.
///
/// Returns `None` when the instance names a channel this shell does not
/// hold, which is the same drift `Resolver::weight` refuses for weights.
#[cfg(feature = "abi")]
fn instance_ring_shapes(
    instance: &InstanceEntry,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
) -> Option<Vec<crate::ptir::ring::ChannelShape>> {
    instance
        .channel_ids
        .iter()
        .map(|id| channels.get(id).map(ChannelState::shape))
        .collect()
}

/// The wire dtype byte, as the tensor IR names it.
///
/// The two vocabularies AGREE on 0..=3 by construction —
/// `PIE_CHANNEL_DTYPE_F32/I32/U32/BOOL` are `DType::from_wire`'s first
/// four — and `declare_dtypes!` asserts the ordering, so this is a lookup
/// rather than a translation.
///
/// `PIE_CHANNEL_DTYPE_ACT` (4) is the exception and it is not a dtype: it
/// names an ACTIVATION channel, whose element width is the deployment's
/// rather than the wire's. Nothing in this shell rings an activation
/// channel yet, so it reads as `F32` — the width `register_channel`
/// already sizes it at — and the day one does, that is a decision to make
/// rather than a default to inherit.
fn channel_dtype(byte: u8) -> driver::tensor_ir::DType {
    driver::tensor_ir::DType::from_wire(byte).unwrap_or(driver::tensor_ir::DType::F32)
}

/// A descriptor pointer, dereferenced ONLY through its validator.
///
/// North star rule 4 says a shared capability must not be optional: if it
/// can be skipped, it will be. `driver-api` ships seventeen `validate_*`
/// functions; `driver-dummy` — the reference implementation of this
/// contract — calls them, and this shell called NONE, re-deriving similar
/// checks by hand at 51 sites. The capability was built, shipped, and
/// routed around.
///
/// Calling them would not have fixed that, only postponed it: a helper
/// that must be remembered is the same shape as a validator that must be
/// remembered. So the dereference and the validation are ONE operation.
/// There is no way to obtain a `&PieKvCopyDesc` in this file without
/// having run `validate_kv_copy_desc` over it, because the only thing
/// that turns the pointer into a reference is this function and it takes
/// the validator as an argument.
///
/// `tests/entry_validation.rs` holds the other half — that no entry point
/// reaches a descriptor any other way — because a rule the compiler
/// cannot state is one a reviewer has to, and this one can at least be
/// stated once rather than at every call.
///
/// # Why the validators are `unsafe` and this is not
///
/// Four of them (`frame`, `encode`, `instance`, `channel`) walk slices the
/// descriptor points at, so they carry the same obligation this function
/// does: the pointer is the caller's to vouch for. Wrapping the null test
/// and the deref together is what discharges it — a non-null descriptor
/// from the engine is a well-formed one, and a null is the error this
/// returns rather than the fault it would otherwise be.
#[cfg(feature = "abi")]
fn checked<'a, T>(
    p: *const T,
    validate: impl FnOnce(&T) -> driver_api::local::PieAbiValidationResult,
    what: &str,
) -> Result<&'a T, i32> {
    let Some(desc) = (unsafe { p.as_ref() }) else {
        eprintln!("[driver-cuda-new] {what}: the descriptor pointer is null");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };
    match validate(desc) {
        Ok(()) => Ok(desc),
        Err(e) => {
            // The MESSAGE is the point. A hand-rolled check returns
            // `INVALID_ARGUMENT` and the caller learns which CALL failed;
            // the validators say which FIELD, and that difference is most
            // of what they are worth.
            eprintln!("[driver-cuda-new] {what}: {}", e.message());
            Err(e.status())
        }
    }
}

fn shell(driver: *mut PieDriver) -> Option<&'static mut Shell> {
    // SAFETY: the only non-null `PieDriver` values in circulation are the
    // boxes `pie_cuda_create` leaked; the engine's contract is to pass
    // them back unmodified.
    unsafe { driver.cast::<Shell>().as_mut() }
}

/// Run a driver entry point, turning a panic into a status rather than into
/// the caller's problem.
///
/// # Why this is here now and was not before
///
/// The thirteen entry points used to be `extern "C"`, where a Rust panic is
/// not a recoverable event: unwinding out of one was undefined behaviour and
/// then, from Rust 1.81, an abort. So the C++ shell's `try/catch` on every
/// export had no counterpart on this side and could not have one -- catching
/// would have to happen INSIDE the frame, which is what this does.
///
/// They are plain Rust functions now, so a panic unwinds normally into the
/// engine. That makes catching possible; this makes it the contract. Every one
/// of these already answers a status code, and "the driver hit a bug" is a
/// status, not a reason to take the process down with the other requests it
/// was serving.
///
/// **This does not make a panic safe to ignore.** A caught panic means the
/// shell's invariants are in an unknown state -- a half-built plan, a stream
/// with work queued, an arena mid-resize -- so it is reported loudly and the
/// driver should be considered untrustworthy afterwards. What it buys is that
/// the OTHER requests in flight get to finish, and that the operator gets a
/// message naming the entry point instead of a bare SIGABRT.
fn guard<T>(what: &str, on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let why = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_owned());
            eprintln!(
                "[driver-cuda-new] {what}: PANICKED: {why}. The request is failed \
                 rather than the process; this driver's state is no longer \
                 trustworthy and it should be recreated."
            );
            on_panic
        }
    }
}

/// Create the driver. Refuses a null descriptor or a mismatched ABI
/// version by returning null, as the C++ shell does.
pub fn pie_cuda_create(
    desc: *const PieDriverCreateDesc,
    caps: *mut PieDriverCaps,
) -> *mut PieDriver {
    guard("create", std::ptr::null_mut(), || create_impl(desc, caps))
}

fn create_impl(desc: *const PieDriverCreateDesc, caps: *mut PieDriverCaps) -> *mut PieDriver {
    // `create` RETURNS A POINTER, not a status, so it cannot pass the
    // validator's message back — it can only refuse. The message still
    // reaches the log, which is the whole reason this call is here and not
    // an `abi_version` test: a caller that got a null handle learns WHY
    // from the line `checked` prints rather than from three candidate
    // causes.
    let Ok(desc) = checked(desc, driver_api::local::validate_driver_create_desc, "create") else {
        return std::ptr::null_mut();
    };
    if validate_create_out_params(caps).is_err() {
        eprintln!("[driver-cuda-new] create: the caps out-parameter must be non-null");
        return std::ptr::null_mut();
    }
    // The boot TOML rides in `config_bytes`. Two keys are read today:
    // `[model] descriptor` and `[driver] runahead`.
    let boot = (!desc.config_bytes.ptr.is_null())
        .then(|| unsafe {
            std::slice::from_raw_parts(desc.config_bytes.ptr, desc.config_bytes.len)
        })
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| text.parse::<toml::Table>().ok())
        .unwrap_or_default();
    let boot_descriptor = boot
        .get("model")
        .and_then(|m| m.get("descriptor")?.as_str())
        .map(std::path::PathBuf::from);
    // PER-DRIVER, not per-process, so a caller that wants asynchronous
    // completions can ask for them without deciding for every other
    // driver alive in the same process. The env var is the default the
    // boot key overrides.
    let runahead = boot
        .get("driver")
        .and_then(|d| d.get("runahead")?.as_bool())
        .unwrap_or_else(runahead_env);
    // An unrecognised spelling is refused rather than defaulted: silently
    // giving bf16 to a caller who asked for fp8 is the kind of wrong
    // answer that reads as a slightly worse model.
    let kv_format = match crate::store::KvCacheFormat::from_name(
        boot.get("driver").and_then(|d| d.get("kv_cache_dtype")?.as_str()).unwrap_or("auto"),
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[driver-cuda-new] create: {e:?}");
            return std::ptr::null_mut();
        }
    };
    // The pages can be written in every catalogued format — `kv_paged.cu`
    // switches on the scheme — but only `attn::attention_naive_paged`
    // READS one back. The fire path's prefill and decode are FlashInfer's
    // `_bf16` entry points, which take the view and ignore its scheme, so
    // a non-native format today would be appended correctly and attended
    // to as though the bytes were bf16.
    //
    // That is a wrong answer rather than a crash, so it is refused here.
    // Lifting it is a kernel change (a scheme-aware attention fast path),
    // not a plumbing one, and the plumbing should not wait for it.
    if !kv_format.is_native_bf16() {
        eprintln!(
            "[driver-cuda-new] create: kv_cache_dtype '{}' can be written but not \
             read back — the fire path's attention is FlashInfer's bf16 entry \
             point, which ignores the scheme. Refusing rather than returning \
             garbage logits.",
            kv_format.name()
        );
        return std::ptr::null_mut();
    }
    let driver_u32 = |key: &str, default: u32| {
        boot.get("driver")
            .and_then(|d| d.get(key)?.as_integer())
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(default)
    };
    let calibrating = boot
        .get("driver")
        .and_then(|d| d.get("calibrate_planner")?.as_bool())
        .unwrap_or(false);
    let device_ordinal = boot
        .get("driver")
        .and_then(|d| d.get("device")?.as_integer())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(0);
    let tp_size = driver_u32("tp_size", 1).max(1);
    let tp_rank = driver_u32("tp_rank", 0).min(tp_size - 1);
    // BIND THE DEVICE HERE, on the thread that will fire.
    //
    // `cudaSetDevice` is per-THREAD, so binding it only inside `load_model`
    // would leave every later call on whatever device the thread last had --
    // which is device 0, which is why the hardwiring was invisible. Doing it
    // at create is what makes `[driver] device` mean anything.
    if let Err(e) = crate::cuda::Device::bind(device_ordinal) {
        eprintln!(
            "[driver-cuda-new] create: cannot bind CUDA device {device_ordinal}: {e}"
        );
        return std::ptr::null_mut();
    }

    // A GROUP OF MORE THAN ONE IS REFUSED, and refusing is the whole point.
    //
    // The LAYOUT half of tensor parallelism works: `tp_rank`/`tp_size` reach
    // `cuda_storage_target`, so a rank compiles a plan that reads only its own
    // bands, and `store::kv_geometry` divides the cache the same way. A rank
    // therefore holds a SHARD of every projection.
    //
    // The COLLECTIVE that puts the shards back together does not exist. The
    // three `comm::`/`dist::` all-reduce rows are declared in
    // `kernels-cuda/src/gemm.rs` and no launch reaches them; there is no NCCL
    // in this tree and no `CustomAllReduce` handle to pass. So a rank that
    // accepted `tp_size > 1` would run its own shard, skip the reduction, and
    // return an answer that is a fraction of the real one -- with no error
    // anywhere, which is the worst failure a driver has.
    //
    // Silence is the bug. Until a collective lands, this is a refusal.
    if tp_size > 1 {
        eprintln!(
            "[driver-cuda-new] create: [driver] tp_size = {tp_size} is refused. \
             This driver shards a rank's WEIGHTS and its KV cache correctly, and \
             has no all-reduce to combine the shards -- so serving would return \
             one rank's partial answer as if it were the whole one. See \
             .wiki/new-driver/next.md, Priority 3."
        );
        return std::ptr::null_mut();
    }
    let boxed = Box::new(Shell {
        caps: CAPS_JSON.as_bytes().to_vec(),
        boot_descriptor,
        runahead,
        kv_format,
        cublas: None,
        lowerings: std::collections::BTreeMap::new(),
        calibrating,
        device_ordinal,
        preds: None,
        peel_win: None,
        logits_staging: None,
        tp_rank,
        tp_size,
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
        in_flight: std::collections::VecDeque::new(),
        fire_alloc: None,
        ptir: crate::ptir::Runtime::default(),
        ptir_control: None,
        ptir_sessions: std::collections::BTreeMap::new(),
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
pub fn pie_cuda_destroy(driver: *mut PieDriver) {
    guard("destroy", (), || destroy_impl(driver));
}

fn destroy_impl(driver: *mut PieDriver) {
    if !driver.is_null() {
        let mut shell = unsafe { Box::from_raw(driver.cast::<Shell>()) };
        // EVERY QUEUED FIRE FIRST, because they may still be writing.
        //
        // A fire that is on the stream when the driver is destroyed will run
        // its stream-ordered callback against a `ChannelState` copy, and the
        // frees below would take that memory back underneath it. Waiting is
        // the only correct answer here: unlike the reclaim in `step_impl`
        // there is no later call to defer to.
        //
        // It also frees the channels those fires were holding for a
        // `close_channel` that arrived while they were queued.
        for fire in std::mem::take(&mut shell.in_flight) {
            let _ = fire.done.synchronize();
            retire(fire);
        }
        for ch in shell.channels.values() {
            ch.free();
        }
        if let Some(swap) = &shell.swap {
            swap.free();
        }
        // The handle is the DRIVER's now, so its destructor is the driver's
        // too — `CublasHandle` asserts it was released rather than dropped.
        if let Some(mut h) = shell.cublas.take() {
            h.release(&mut crate::cuda::cublas::LiveCublas);
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
pub fn pie_cuda_load_model(
    driver: *mut PieDriver,
    load: *const PieModelLoadDesc,
    caps: *mut PieDriverCaps,
) -> i32 {
    guard("pie_cuda_load_model", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let load = match checked(load, driver_api::local::validate_model_load_desc, "load_model") {
            Ok(d) => d,
            Err(status) => return status,
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
    })
}

/// The load itself; `i32` errors are the ABI's status codes.
fn load_impl(state: &mut Shell, snapshot: &std::path::Path) -> Result<(), i32> {
    use model_loader::checkpoint::read::{parse_checkpoint_metadata, read_meta};

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

    // THE LOAD IS `model-loader`'s PLAN, EXECUTED ONTO THE DEVICE.
    //
    // What this replaced: a loop that read each checkpoint tensor into a
    // host `Vec`, uploaded it, and then read three of them BACK off the
    // device to concatenate a `qkv` and uploaded that — three round trips
    // for bytes a plan lays out once, and a thousand `cudaMalloc`s where
    // a resident plan wants one arena.
    //
    // The plan also decides what the driver used to decide by hand: which
    // encodings are loadable (a transform outside `CUDA_TILE_MAP_MASK` is
    // refused when the plan COMPILES, with the tensor named, rather than
    // mis-bound at launch), and which projections are fused.
    let target = crate::loader::plan::cuda_storage_target(state.tp_rank, state.tp_size);
    let (plan, _moe) =
        crate::loader::plan::compile_load_plan(snapshot, &meta, &target, &descriptor_json)
            .map_err(|e| {
                eprintln!("[driver-cuda-new] load_model: {e}");
                PIE_STATUS_UNSUPPORTED
            })?;
    let alloc = crate::cuda::Allocator::new();
    let staged = crate::loader::stage::stage_plan_weights(&plan, snapshot, &alloc).map_err(
        |e| {
            eprintln!("[driver-cuda-new] load_model: staging: {e:?}");
            PIE_STATUS_EXHAUSTED
        },
    )?;

    let mut model = LoadedModel {
        hf,
        load_caps: Vec::new(),
        weights: staged.spans,
        owned: staged.owned,
        aliases: std::collections::BTreeMap::new(),
        gemma_layer_scalars: Vec::new(),
        tp_size: state.tp_size,
    };
    wire_trace_names(&mut model);
    state.model = Some(model);
    // AFTER the model is stored, because a calibration boot fires through the
    // ordinary path and that path reads `state.model`.
    let caps = capabilities_json(state, snapshot)?;
    state.model.as_mut().expect("just stored").load_caps = caps;
    Ok(())
}

/// Answer the trace names a launch will ask for, from `model`'s tables.
///
/// The driver's whole part in naming, and it is deliberately small: which
/// trace name means which published tensor is FAMILY knowledge and lives in
/// `model::weight_names`, beside the DSL that invents the trace names and the
/// contract author that invents the published ones. What is left here is the
/// two things only a driver can answer.
///
/// **Whether a join is a rename or nothing.** A checkpoint that ships its
/// projections pre-joined (Phi-3) has its contract SPLIT them, so
/// `Projections::Fused` has nothing to fuse and the halves are merely
/// adjacent. They are adjacent IN THE ARENA — the plan wrote them once, in
/// file order — so the fused operand exists and only wants a name. Only the
/// driver holds the addresses that decide it, and it checks rather than
/// assumes: a GEMM handed a discontiguous operand reads what lies between.
///
/// **Reading a load-time scalar to the host.** gemma-4's per-layer
/// `layer_scalar` is one bf16 on the device; `model` says which tensors they
/// are and in what order, and the copy is CUDA's.
fn wire_trace_names(model: &mut LoadedModel) {
    let published: Vec<String> = model.weights.keys().cloned().collect();
    let set: std::collections::BTreeSet<&str> =
        published.iter().map(String::as_str).collect();
    let has = |n: &str| set.contains(n);
    let wiring = model::weight_names::wire(&model.hf, &has);

    for (trace, name) in wiring.aliases {
        model.aliases.insert(trace, name);
    }
    for (trace, parts) in wiring.joins {
        let mut spans = Vec::with_capacity(parts.len());
        for p in &parts {
            let Some(span) = model.weights.get(p).copied() else {
                spans.clear();
                break;
            };
            spans.push(span);
        }
        if spans.len() != parts.len() {
            continue;
        }
        let abut = spans.windows(2).all(|p| {
            std::ptr::eq(
                p[0].ptr.wrapping_byte_add(p[0].bytes).cast_const(),
                p[1].ptr.cast_const(),
            )
        });
        if abut {
            model.weights.insert(trace, crate::loader::stage::WeightSpan {
                ptr: spans[0].ptr,
                bytes: spans.iter().map(|s| s.bytes).sum(),
            });
        }
    }
    model.gemma_layer_scalars = wiring
        .scalars
        .iter()
        .map(|n| {
            model.weights.get(n).map_or(1.0f32, |b| {
                match crate::loader::stage::read_span(*b) {
                    Ok(back) if back.len() == 2 => {
                        f32::from_bits(u32::from(u16::from_le_bytes([back[0], back[1]])) << 16)
                    }
                    _ => 1.0,
                }
            })
        })
        .collect();
}

/// What `load_model` answers: a `driver_api::DriverCapabilities` document.
///
/// **This used to be a five-field summary of the checkpoint** —
/// `{"model_type":…,"hidden":…,"layers":…,"vocab":…,"weights":…}` — which is
/// not a capability payload and which `DriverCapabilities` rejects outright,
/// field by field, at `unknown field \`model_type\``. So no engine could load
/// a model through this driver; the ABI tests call `pie_cuda_load_model`
/// directly and pass `null` for the caps, so nothing here noticed.
///
/// # The KV pool is SIZED HERE, and that is the substance
///
/// `total_pages` is what a scheduler admits against, so answering it is
/// answering how much context this device holds. The budget comes from
/// `store::memory_planner::budget_for` — the ported planner's own reserve
/// arithmetic, which until now had no live caller — measured AFTER the
/// weights are resident, so `cudaMemGetInfo`'s free figure already has them
/// subtracted.
///
/// What the budget then has to cover is the KV pool and the fire's
/// activations. The activation share is a fraction rather than a computed
/// arena, because the arena is a property of the LOWERING and no fire has
/// been lowered yet; a fifth is the C++'s own rule of thumb and it is stated
/// here rather than hidden in a constant.
fn capabilities_json(state: &mut Shell, snapshot: &std::path::Path) -> Result<Vec<u8>, i32> {
    use crate::store::memory_planner::{
        DeviceMemory, DeviceProps, Family, ModelCosts, ModelShape, NoProfiles, PlannerConfig,
        ProfileSource, plan,
    };
    use crate::store::model_costs::{CheckpointCosts, DiskProfiles};

    let model = state.model.as_ref().expect("the model is stored");
    let hf = model.hf.clone();
    let hf = &hf;
    let model_tp = model.tp_size;
    let device =
        crate::cuda::Device::bind(state.device_ordinal).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let (free, total) = device.memory_info().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let (major, minor) = device.compute_capability().unwrap_or((0, 0));
    let cfg = PlannerConfig {
        gpu_mem_utilization: 0.90,
        memory_profile: "auto".to_owned(),
        max_forward_tokens: 0,
        max_forward_requests: 0,
        // PINNED, and this is a coupling rather than a preference: the fire
        // path builds 16-token pages by construction (`page_size: usize = 16`
        // in `step_impl` and in `resize_pool`). Letting the lattice sweep page
        // sizes would have it answer a geometry the driver does not build.
        kv_page_size: 16,
        // The driver's OWN format, so the planner sizes the pages the
        // driver will actually allocate. It was hardwired while the shell
        // could only build bf16; now that the format is a boot key, a
        // hardwired planner would under-count a quantized cache by up to
        // 4x and hand back a page budget the device cannot hold.
        kv_cache_dtype: state.kv_format.name().to_owned(),
        tp_size: i32::try_from(model_tp).unwrap_or(1),
        mtp_num_drafts: 0,
        // FALSE EVEN WHEN CALIBRATING, and the divergence is deliberate.
        //
        // `calibrating` makes the planner build the CEILING of the feasible
        // region — the largest rectangle whose `arena + persistent` fits the
        // budget — on the reasoning that a bigger arena can run a smaller
        // shape. That holds when the arena is the only limit. It is not here:
        // the attention workspace is a FIXED 32 MB the shell allocates, and a
        // fire wider than it supports fails inside CUDA rather than returning
        // a status. On an L40S the ceiling is N=65536, whose logits buffer
        // alone is twenty gigabytes and whose fire aborts.
        //
        // So the sweep explores at or below the shape the driver is built to
        // fire, which is the scored pick. It measures which of the REACHABLE
        // shapes is fastest — a smaller claim than the C++'s and a true one.
        calibrating: false,
        rs_slot_mult: 1,
        nccl_unique_id_hex: String::new(),
    };
    let costs = CheckpointCosts::new(hf, model_tp);
    let shape = ModelShape {
        hidden_size: hf.hidden_size,
        num_hidden_layers: hf.num_hidden_layers,
        num_attention_heads: hf.num_attention_heads,
        num_key_value_heads: hf.num_key_value_heads,
        head_dim_kernel: hf.head_dim_kernel.max(hf.head_dim),
        model_type: hf.model_type.clone(),
    };
    let props = DeviceProps {
        name: String::new(),
        major,
        minor,
        sm_count: device.sm_count().unwrap_or(0),
    };
    let mem = DeviceMemory {
        free_bytes: free as u64,
        total_bytes: total as u64,
    };
    // MEASURED AFTER THE WEIGHTS ARE RESIDENT, so `cudaMemGetInfo`'s free
    // figure already has them subtracted and the budget is what is left.
    // A MEASUREMENT BEATS THE SCORE, when there is one. `DiskProfiles` reads
    // the cache a calibration boot writes; a machine that has never
    // calibrated has no file, which is a miss and not an error, and the
    // planner falls back to the analytic pick. `NoProfiles` is the honest
    // stand-in when no cache directory can even be derived.
    let disk = DiskProfiles::discover("").ok();
    let profiles: &dyn ProfileSource = disk.as_ref().map_or(&NoProfiles, |d| d);
    let planned = plan(&cfg, &shape, &props, mem, Family::Generic, &costs, profiles)
        .map_err(|e| {
            eprintln!("[driver-cuda-new] load_model: memory planner: {e:?}");
            PIE_STATUS_EXHAUSTED
        })?;
    for note in &planned.notes {
        eprintln!("[driver-cuda-new] {note}");
    }

    // PAGES AGAINST WHAT THE ARENA LEAVES, and this is a deliberate
    // divergence from the planner's own rule.
    //
    // The planner sizes pages against the FULL budget, and says why: "the
    // arena is a transient graph workspace that is freed between fires, while
    // the KV pool is the resident one". That was true of the C++. It is not
    // true here — `FireArrays` keeps the arena, the named seam buffers and
    // the descriptor arrays for the life of the driver, because a captured
    // graph bakes the addresses it recorded and a freed arena can never be
    // replayed into. Persistence is the precondition for the supergraph and
    // for run-ahead, and it is not negotiable.
    //
    // So the two resident allocations share one budget, and charging only one
    // of them would advertise a page count whose pool cannot be built: on
    // qwen3-0.6B the full-budget figure is 22,016 pages against a budget the
    // arena also has to come out of.
    let per_page = planned.plan.kv_page_bytes.max(1);
    let resident_arena = costs
        .arena_bytes(
            planned.plan.capacity.max_forward_tokens,
            planned.plan.capacity.max_logit_rows,
            0,
        )
        .saturating_add(planned.plan.attn_float_workspace_bytes)
        .saturating_add(planned.plan.persistent_input_bytes)
        .saturating_add(planned.plan.runtime_quant_scratch_bytes);
    let total_pages = planned.budget.saturating_sub(resident_arena) / per_page;

    let caps = driver_api::DriverCapabilities {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        total_pages: u32::try_from(total_pages).unwrap_or(u32::MAX),
        kv_page_size: u32::try_from(planned.plan.kv_page_size).unwrap_or(16),
        // THE LATTICE'S ANSWER, not a stated ceiling. These are what a
        // scheduler batches under, and the arena the planner chose is sized
        // for exactly this rectangle — a fire wider than it has no workspace.
        max_forward_tokens: u32::try_from(planned.plan.capacity.max_forward_tokens).unwrap_or(0),
        max_forward_requests: u32::try_from(planned.plan.capacity.max_forward_requests)
            .unwrap_or(0),
        max_page_refs: u32::try_from(planned.plan.capacity.max_page_refs).unwrap_or(0),
        arch_name: hf.model_type.clone(),
        vocab_size: u32::try_from(hf.vocab_size).unwrap_or(0),
        max_model_len: u32::try_from(hf.max_position_embeddings).unwrap_or(0),
        activation_dtype: "bf16".to_owned(),
        hidden_size: u32::try_from(hf.hidden_size).unwrap_or(0),
        rs_cache_required: costs.has_linear_state(),
        snapshot_dir: snapshot.display().to_string(),
        // No swap pool, no elastic accounting, no MTP or value head, and no
        // sink this shell honours yet. Every one of these is a claim a
        // program BINDS against, so a false advertisement is a program that
        // runs as a silent no-op rather than one that is refused.
        swap_pool_size: 0,
        kv_copy_domain_mask: 0,
        rs_cache_slots: 0,
        rs_cache_slot_bytes: costs.state_slot_bytes(),
        elastic_page_bytes: 0,
        elastic_budget_pages: 0,
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        has_kv_envelopes: false,
        has_attn_score: false,
        has_attn_page_mask: false,
        has_lora: false,
        model_site_summary: driver_api::ModelSiteSummary::default(),
        device_geometry_port_mask: 0,
        supports_media_encode: false,
        kv_handle: None,
        // This driver compiles its own PTIR through NVRTC; nothing upstream
        // needs to generate a kernel for it.
        codegen_backend: String::new(),
    };
    serde_json::to_vec(&caps).map_err(|_| PIE_STATUS_DRIVER_ERROR)
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
pub fn pie_cuda_register_program(
    driver: *mut PieDriver,
    program: *const PieProgramDesc,
    program_id: *mut u64,
) -> i32 {
    guard("pie_cuda_register_program", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(program, driver_api::local::validate_program_desc, "register_program") {
                Ok(d) => d,
                Err(status) => return status,
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
        let package = unsafe { driver_api::adopt_package(&desc.launch) };
        let kernels = unsafe { driver_api::adopt_emitted_kernels(desc.emitted_kernels) };

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
    })
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
    package: driver::driver_api::plan::LaunchPackage,
    kernels: &[driver_api::EmittedKernel],
) -> Result<(), i32> {
    let plan = match driver::adopt_launch_package(package) {
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
        let target = ptir_target(state.device_ordinal)?;
        let versions = driver::Versions::mirrored(desc.emitter_version);
        match state
            .ptir
            .compile(desc.program_hash, &plan, kernels, versions, target)
        {
            Ok(compiled) => {
                sg_trace(|| {
                    format!(
                        "  ptir program {:#018x}: {} stage(s) compiled",
                        desc.program_hash,
                        plan.stages.len()
                    )
                });
                state.ptir_programs.insert(id, compiled);
            }
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
fn ptir_target(ordinal: i32) -> Result<crate::ptir::Target, i32> {
    let device = crate::cuda::Device::bind(ordinal).map_err(|error| {
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
pub fn pie_cuda_register_channel(
    driver: *mut PieDriver,
    channel: *const PieChannelDesc,
    binding: *mut PieChannelEndpointBinding,
) -> i32 {
    guard("pie_cuda_register_channel", PIE_STATUS_DRIVER_ERROR, move || {
        use crate::model::attention_workspace::{LiveStagingOps, StagingOps};

        const MAX_RING: u64 = 64;
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            channel,
            |d| unsafe { driver_api::local::validate_channel_desc(d) },
            "register_channel",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        if desc.abi_version != PIE_DRIVER_ABI_VERSION
            || state.channels.contains_key(&desc.channel_id)
            || desc.dtype > driver_api::local::PIE_CHANNEL_DTYPE_ACT
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
        let wire_bytes: u64 = if desc.dtype == driver_api::local::PIE_CHANNEL_DTYPE_BOOL {
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
                numel: usize::try_from(numel).unwrap_or(usize::MAX),
                dtype: channel_dtype(desc.dtype),
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
    })
}

/// Bind an instance to a registered program: the id lifecycle, honoring
/// a nonzero `requested_instance_id` and echoing the geometry class.
/// KV-slot and adapter state ride in with the `launch` arm.
pub fn pie_cuda_bind_instance(
    driver: *mut PieDriver,
    instance: *const PieInstanceDesc,
    binding: *mut PieInstanceBinding,
) -> i32 {
    guard("pie_cuda_bind_instance", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            instance,
            |d| unsafe { driver_api::local::validate_instance_desc(d) },
            "bind_instance",
        ) {
            Ok(d) => d,
            Err(status) => return status,
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
    })
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
pub fn pie_cuda_launch(
    driver: *mut PieDriver,
    frame: *const PieFrameDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_launch", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        // NOT `validate_frame_desc` YET, and the reason is written down
        // rather than left as an omission. That validator requires
        // `terminal_cells.len == roster_rows.len` and a
        // `channel_ticket_indptr` whenever ticket values are present;
        // this shell serves frames today that state neither, so adopting
        // it here would refuse traffic that works. Which of the two is
        // wrong — the frames or the rule — is a question for whoever owns
        // the ticket path, and guessing is how a validator gets weakened
        // instead of a caller fixed. See `validators-unskippable`.
        let frame = match checked(
            frame,
            |d| unsafe { driver_api::local::validate_frame_desc(d) },
            "launch",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        // THE CALL RETURNS WITH THE WORK STILL ON THE STREAM.
        //
        // Publishing the terminal cells and notifying used to happen here,
        // after `launch_impl` had synchronized — which made the driver
        // serialize the engine's `frame_dispatch_depth`, because the call
        // that would enqueue fire n+1 could not start until fire n had
        // retired on the GPU.
        //
        // Both debts moved into a stream-ordered host callback
        // (`retire_fire`), so an `Ok` here means ENQUEUED rather than DONE.
        // An error still returns synchronously: a fire that could not be
        // built owes nothing and the runtime must hear so on this thread.
        match launch_impl(state, frame, completion) {
            Ok(()) => PIE_STATUS_OK,
            Err(code) => code,
        }
    })
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

/// Does a fire's completion ride a stream callback? **YES unless told not to**
/// (`PIE_CUDA_RUNAHEAD=0`, or `[driver] runahead = false` per driver).
///
/// It was off, and the reason was honest: `pie_cuda_launch` used to finish the
/// fire before it returned, and a caller reading the ring on the next line was
/// asserting that. The gate protected a CONTRACT CHANGE.
///
/// It is on because the contract is now the ABI's: the notify says the fire
/// retired, the engine waits for it, and this tree's tests do too. And because
/// there is finally something to gain — a warm decode issues in 0.68 ms and
/// retires 3.75 ms later, so the call returns with 3 ms of GPU work queued
/// behind it. When the gate was written those two numbers were the same, and
/// run-ahead bought nothing.
fn runahead_env() -> bool {
    !std::env::var_os("PIE_CUDA_RUNAHEAD").is_some_and(|v| v == "0" || v == "false" || v == "off")
}

/// The arena offset the attention dispatch at `fi` WRITES.
///
/// Two readings, and only the first is right under a union lowering.
///
/// 1. **The dispatch's own op join.** The attention statement carries its
///    output placement, which is exactly the slot the o_proj goes on to read.
/// 2. **The next launch's first operand** — "the launch after the dispatch is
///    the o_proj". True under `Resolve`, where the guard has already deleted
///    every arm the fire did not take. False under `Union`, where every arm is
///    present and the next launch belongs to some other body, which is why a
///    union that has to fall back here declines instead.
///
/// # And why every DECODE declines
///
/// `dsl::seam::attn_at` records an output only when the statement is NOT
/// inside a value-producing region:
///
/// ```ignore
/// let out = q.t.inner.borrow().inside_value_region();
/// let shape = (!out).then(|| …);   // None inside a region
/// ```
///
/// The decode arm states its attention inside one and the prefill arm does
/// not. So reading 1 is empty for every decode, the union is declined, and a
/// one-token fire walks all 396 launches — ~9 ms on a 0.6B model, which is
/// most of what `.wiki/new-driver/next.md`'s table measures.
///
/// **Recovering it from the enclosing region here does not work, and the
/// failure is quiet.** Two attempts, both green on 20 of 21 ABI tests and
/// both wrong on `multi_step_resize_and_copy_preserve_the_kv`: the nearest
/// preceding region is as often a closed guard from an earlier layer as the
/// enclosing one; requiring coverage and matching the output's shape to q's
/// still picks the wrong construct for some fires, because several covering
/// regions can carry a q-shaped value. A wrong offset here is not a refusal —
/// it binds the attention plan over another activation.
///
/// The fix belongs at the STATEMENT, not here: `attn_at` should record its
/// landing in the join even inside a region, so the value has one stated
/// producer instead of being reverse-engineered from op adjacency.
fn attention_landing(
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::model::executor::DispatchPlan,
    fi: usize,
) -> Option<usize> {
    use model_compiler::lower::Arg;
    match dplan.spec(fi).outs.first() {
        Some(Arg::Arena { at, .. }) => Some(*at),
        _ => match lowered.launches.get(fi + 1).map(|n| &lowered.args[n.args.start as usize]) {
            Some(Arg::Arena { at, .. }) => Some(*at),
            _ => None,
        },
    }
}

/// `PIE_CUDA_TRACE_SUPERGRAPH=1`: say what each fire did with the graph.
///
/// The one instrument that answers the question the measurements raise —
/// whether a fire REPLAYED or walked its launches, and if it walked, which
/// clause of the servability test turned it away. Lazy, so an unset variable
/// costs a `getenv` and no formatting.
fn sg_trace(what: impl FnOnce() -> String) {
    if std::env::var_os("PIE_CUDA_TRACE_SUPERGRAPH").is_some() {
        eprintln!("[sg] {}", what());
    }
}

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

/// The fire's CLASS, read off its SHAPE — one row per request is a
/// decode, anything else is prefill-shaped.
///
/// It used to read the recurrent-state flags too, and derive three MTP
/// service classes from them (`CommitAdvance` where every row replayed
/// buffered tokens, `FrozenVerify` where any row wrote buffered slabs,
/// `StateOnly` where a recurrent fire had no readout rows). Those classes
/// are gone — `.wiki/driver/graph.md` §4.2: a speculative decode buffers
/// its tokens and folds only the accepted prefix, so a rejected token is
/// never folded and there is nothing to repair. The driver executes the
/// flags now (see the fold in `gdn_context`) instead of classifying on
/// them.
///
/// What remains is a shape question, and it is on its way out too: the
/// window class is `GuardPred::WindowOne` now, so the two surviving
/// values pick nothing the trace does not already guard. See §4.1.
pub fn fire_class_of(
    _step: &driver_api::local::PieStepDesc,
    rows: usize,
    requests: usize,
) -> Result<model_compiler::trace::FireClass, i32> {
    use model_compiler::trace::FireClass;
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
        Some(model.weights.get(ckpt)?.bytes / 2)
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
        // The group the load sharded for. A rank whose weights are a band
        // of the real ones has to land its projections with a collective,
        // and this is where the trace learns that.
        tp_size: model.tp_size,
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
    alloc: &mut crate::cuda::Allocator,
    preds: &mut crate::cuda::PredicateWord,
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
    // replays of one exec. NOT synchronized after: the upload and the replay
    // are ordered on the same stream, so waiting here only made the call
    // block on work it had just enqueued.
    if fire_predicates(rows_desc, &lowered.conds, preds).is_err()
        || preds.upload(stream).is_err()
    {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }

    if cache.replay(key, epoch, stream).unwrap_or(false) {
        sg_trace(|| format!("replay {key:?}"));
        return Ok(lowered.launches.len());
    }
    sg_trace(|| format!("miss {key:?} launches={}", lowered.launches.len()));

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
        // THE CAPTURE OPENS ON THE FIRE'S OWN ALLOCATOR, and it used to open
        // on a throwaway one made right here. That looked harmless — nothing
        // allocates during a capture — but the flag it raises is what makes a
        // `cudaFree` DEFER, and the deferral has to happen on the allocator
        // that OWNS the buffer. A temporary dropped inside `run_captured`
        // therefore freed immediately, in the middle of an open stream
        // capture, and the graph that came out faulted when it was destroyed.
        //
        // Phi-3's decode found it, because no decode had ever been captured.
        let Ok(scope) = alloc.begin_capture(stream) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        let mut b = crate::cuda::SupergraphBuilder::new(scope.stream(), preds);
        let ran = crate::model::executor::run_captured(
            lowered, dplan, frame, resolver, ctx, regions, gdn, &mut b,
        );
        // The nodes the capture retained, taken BEFORE the builder is
        // dropped: one per launch, and what lets a later fire of a
        // different row count retune this exec's rectangles instead of
        // recapturing (`.wiki/driver/graph.md` §6.2).
        let nodes = b.nodes().to_vec();
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
        let ended = scope.end();
        sg_trace(|| format!("capture ran={ran:?} ended_ok={}", ended.is_ok()));
        match (ran, ended) {
            (Ok(n), Ok(g)) => Some((n, g, nodes)),
            (Err(_), Ok(g)) => {
                // AN ABANDONED CAPTURE IS NOT DESTROYED, it is forgotten.
                //
                // A run that refused part-way leaves a recording whose nodes
                // the builder has already dropped, and `cudaGraphDestroy` on
                // that faults inside the CUDA driver — no device error, a
                // host segfault. Found through phi-3, where an unservable
                // arm made every decode capture abandon.
                //
                // Leaking one graph template per abandoned capture is the
                // cheaper wrong answer: captures are per (bucket, epoch) and
                // a refusal is meant to be rare. The refusals that are NOT
                // rare belong in the servability test above, which is where
                // phi-3's went. `ManuallyDrop` rather than `mem::forget`
                // because the leak is the POINT and should read as one.
                let _leaked = std::mem::ManuallyDrop::new(g);
                None
            }
            (_, Err(_)) => None,
        }
    };
    let Some((ran, graph, nodes)) = captured else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    let Ok(exec) = graph.instantiate() else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    if exec.launch(stream).is_err() {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }
    let _ = cache.insert_with_nodes(key, exec, epoch, nodes, eligibility);
    Ok(ran)
}

/// The fire itself. Everything here is the proven smoke assembly, run
/// against the shell's own state.
#[allow(clippy::too_many_lines)]
fn launch_impl(
    state: &mut Shell,
    frame: &PieFrameDesc,
    completion: PieCompletion,
) -> Result<(), i32> {
    let steps = slice_of(frame.steps.ptr, frame.steps.len);
    if steps.is_empty() {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // Steps run SEQUENTIALLY, each a fire of its own — the frame's
    // producer→consumer ordering. One shared KV, per-step everything else.
    for step in &steps[..steps.len() - 1] {
        step_impl(state, frame, step, None)?;
    }
    // The LAST step carries the frame's debt: its terminal cells and the
    // completion the runtime waits on. Only it enqueues an asynchronous
    // retire, because a frame completes once.
    let step = steps.last().expect("nonempty");
    let cells = slice_of(step.terminal_cells.ptr, step.terminal_cells.len).to_vec();
    step_impl(state, frame, step, Some((completion, cells)))
}

/// Trace a family's forward for one fire shape, lower it, and join the ops
/// back onto the launches.
///
/// Split out of `step_impl` so its result can be CACHED — see
/// [`Shell::lowerings`]. Nothing here reads the fire's data; it reads the
/// shape, which is what makes the answer reusable.
fn build_lowering(
    family: &dyn PlannedFamily,
    class: model_compiler::trace::FireClass,
    fire_rows: &[model_compiler::lower::Row],
    union_asked: bool,
) -> Result<LoweredFire, i32> {
    use crate::model::executor::DispatchPlan;
    use model_compiler::lower::{Arg, Fire, GuardMode, lower_with};

    let plan = family.trace(class);
    let lower_as = |g: GuardMode| {
        lower_with(&plan, fire_rows, Fire { captures_across_splits: false }, g).map_err(|e| {
            eprintln!("[driver-cuda-new] launch: uncovered: {e:?}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let mut union = union_asked;
    if !union {
        sg_trace(|| "union off at the gate".into());
    }
    let mut lowered = lower_as(if union { GuardMode::Union } else { GuardMode::Resolve })?;

    // NOTHING DECLINES THE UNION ANY MORE, and the two clauses that used
    // to are worth naming because their removal is the point of §5 (1)
    // and (2) in `.wiki/driver/graph.md`.
    //
    // The first refused any lowering mentioning `_capture` or `_custom` --
    // the exact case a folded `WantsAttnScore` / `HasCustomMask` predicate
    // exists for. Its cause was PER-ARM PREPARED STATE: under `Union`
    // every arm is recorded whether this fire takes it or not, so the
    // capture walked the arm whose state the fire declined to build and
    // abandoned the whole recording. Prepared state belongs to the BUCKET
    // now -- the score sink is published every fire, every plan the
    // geometry permits is raised, a causal element mask stays resident.
    //
    // The second refused a fire whose attention output slot the op JOIN
    // could not name. The old fallback -- "the launch after the dispatch
    // is the o_proj, so its input is the slot" -- is a fact read off where
    // a statement SITS, true under `Resolve` (the guard has deleted every
    // arm the fire did not take) and false under `Union` (every arm is
    // present and the neighbour belongs to some other body). But declining
    // was never the only answer: `AttnCtx::o_out` is a DRIVER-owned
    // pointer, so a fire whose join names no slot gets a driver-owned
    // buffer instead of losing its graph. See `o_off` at the fire site.

    let dplan = DispatchPlan::new(&plan, &lowered);
    sg_trace(|| format!("built: launches={} union={union}", lowered.launches.len()));
    Ok(LoweredFire { plan, lowered, dplan, union })
}

/// One step's fire — the former single-step body.
#[allow(clippy::too_many_lines)]
/// What a step must satisfy before anything is traced, lowered or
/// allocated for it — and the handful of facts that survive the asking.
///
/// The FIRST of `step_impl`'s phases, promoted out of it. Its boundary is
/// the one place in the fire path where "this driver cannot serve this"
/// is still a cheap answer: nothing has been built, nothing has been
/// bound, and no device memory has moved. Every refusal below is
/// therefore an early return and not a rollback, which is the north
/// star's third rule read forwards — decide, then move.
///
/// The borrow shape is why this returns indices into `step` rather than
/// the slices themselves: `state` is `&mut` for the rest of the fire, so
/// a phase that handed back `&[u32]` borrowed from `state.model` would
/// pin it. The caller re-slices from `step`, which it owns.
struct Admitted {
    /// The service class the row/request ratio implies.
    class: model_compiler::trace::FireClass,
    /// Token rows in this step.
    rows: usize,
    /// Requests the step's CSR partitions those rows into.
    requests: usize,
    /// The rows the lowering will resolve its guards against, read from
    /// the step's REGION TABLE.
    ///
    /// This shell used to build `vec![Row { samples: true, ..default() };
    /// rows]` and never look at `region_sig` — zero reads in the whole
    /// file. So `HasLora`, `HasCustomMask`, `HasStageHooks` and the depth
    /// truncation could not hold no matter what the engine sent, and
    /// every fire claimed to sample every row. LoRA looked like a missing
    /// feature; the wire had been carrying it the whole time.
    fire_rows: Vec<model_compiler::lower::Row>,
}

/// See [`Admitted`].
#[cfg(feature = "abi")]
fn admit(
    state: &Shell,
    step: &driver_api::local::PieStepDesc,
) -> Result<(Admitted, Box<dyn PlannedFamily>), i32> {
    use model_compiler::trace::FireClass;

    // A USER MASK IS REFUSED, not quietly replaced by a causal one.
    //
    // The engine encodes a program's attention mask as BRLE runs
    // (`driver_api::plan::EncodedMask`), packs them into `step.masks`, and sets
    // this flag. This shell reads NEITHER: `brle` is not ported and no
    // custom-mask kernel is launched, so a fire that was asked to attend over
    // a caller's mask attended causally instead and returned an answer that
    // looks exactly like a correct one.
    //
    // Same class as the tensor-parallel bug and the same remedy. A mask the
    // driver cannot honour is a fire it cannot serve.
    if step.has_user_mask != 0 {
        eprintln!(
            "[driver-cuda-new] launch: this frame carries a user attention mask \
             and this driver has no kernel that reads one. Serving it would \
             attend CAUSALLY over a mask the caller supplied, which is a wrong \
             answer that looks like a right one. See `custom-attention-mask`."
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
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
    // THE REGION TABLE, which is the seriation's output stated once. An
    // empty one is the legacy discipline — no seriation ran, so the fire
    // is one region of the default point — and not a refusal.
    let mut fire_rows = model_compiler::lower::rows_from_regions(
        rows,
        slice_of(step.sampling_indices.ptr, step.sampling_indices.len),
        slice_of(step.region_row_indptr.ptr, step.region_row_indptr.len),
        slice_of(step.region_sig.ptr, step.region_sig.len),
        slice_of(step.region_k.ptr, step.region_k.len),
    )
    .map_err(|drift| {
        eprintln!(
            "[driver-cuda-new] launch: the step's region table does not describe \
             its rows: {drift:?}"
        );
        PIE_STATUS_INVALID_ARGUMENT
    })?;
    // EVERY ROW SAMPLES, and this override is the one thing above that is
    // not the wire's answer.
    //
    // `lower::epilogue` states a gather when `sampled < window.len()` —
    // a prefill reads one distribution per request out of a stream of one
    // row per token, and those rows are not contiguous. But it emits the
    // gather, the final norm and the head over ONE op's operands, so the
    // gather's output IS the logits buffer: it would write `[sampled, H]`
    // rows into a `[sampled, vocab]` allocation at the wrong stride and
    // the head would read what it wrote. The intermediate the C++
    // epilogue uses (`ws.norm_x`) has no SSA value for the lowering to
    // hand out.
    //
    // So the compaction waits on a lowering that can state that temp, and
    // until then every row claims its readout — which is what this shell
    // did unconditionally, and is exactly right for a decode. The cost is
    // a prefill computing the head over every row instead of one per
    // request: slower, and the same numbers.
    //
    // The five AXES above are the wire's, and they are the point: `lora`,
    // `custom_mask`, `hooked`, `multi_token` and `depth_k` now hold when
    // the engine says they do. `samples` is the only field a shell can
    // over-claim without changing an answer.
    for r in &mut fire_rows {
        r.samples = true;
    }
    // AN ADAPTER THIS DRIVER CANNOT APPLY IS REFUSED.
    //
    // Now that the region bit is read, a marked row reaches the
    // `HasLora` guard and the trace states `pie_lora_qkv_correction`.
    // The executor's arm for it returns `Ok(())` when `ctx.lora` is
    // `None` — which it always is, because nothing stages the table
    // yet — and that no-op is LOAD-BEARING for union captures: under
    // `GuardMode::Union` every arm lowers and the predicate is decided
    // at replay, so the arm has to be issuable with nothing to correct.
    //
    // So the refusal cannot live in the arm. It lives here, where the
    // question is whether this FIRE asked for something the driver
    // cannot do. Running it would apply no correction and return tokens
    // that look like a slightly worse model, which is the one failure
    // mode worth refusing over.
    //
    // What lifts it: the adapter arrives as a pass-wide `lora` SINK in
    // the program's PROLOGUE (`fwd.adapter` lowers LoRA, IA3 and DoRA
    // into it; the A/B are channel cells). Reading that sink's operands
    // into a `LoraTable` and staging it with
    // `model::lora::llama_like_lora_stage` is what fills `ctx.lora` —
    // and `caps.has_lora` turns true with it.
    if fire_rows.iter().any(|r| r.lora) {
        eprintln!(
            "[driver-cuda-new] launch: a region carries an adapter \
             (PIE_REGION_SIG_LORA) and this driver stages none — the \
             prologue's `lora` sink has no reader, so the correction would \
             be a no-op. Refusing rather than answering without it; \
             `caps.has_lora` is false."
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }

    // A family that does not DECLARE a service class must be turned away
    // rather than traced: its text answers the three with `unreachable!`,
    // and a panic crossing the entry point is caught but costs the whole
    // request. Only the MTP family composes those passes.
    if !matches!(class, FireClass::Decode | FireClass::Prefill) && !family.recurrent() {
        eprintln!(
            "[driver-cuda-new] launch: {class:?} is an MTP service pass and \
             this family declares no trace for it"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok((Admitted { class, rows, requests, fire_rows }, family))
}

/// Run the instance's registered program over the fire's logits.
///
/// `step_impl`'s SAMPLING phase, and the reason `ptir_programs` had a
/// writer and no reader until now. Sampling is a PTIR stage and not a
/// driver flag — top-p, top-k, temperature and argmax are ops a caller's
/// program states — so a fire that skipped this returned raw logits and
/// ignored every sampling parameter the request carried.
///
/// Returns `Ok(false)` when there is nothing to run, which is the common
/// case and not a failure: an instance with no program, a program that
/// compiled to nothing, or an instance whose channels this shell does not
/// hold. The caller then delivers logits the old way.
///
/// # Why a refusal here is not a failed request
///
/// A program that DECLINES a fire has said so deliberately, and one whose
/// inputs are not ready is waiting on the engine rather than broken.
/// Neither is a reason to fail the step: the cursors are left where they
/// were, so the next fire sees the same inputs and the same decision.
/// Only a device error propagates.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn run_program(
    // THE FIVE FIELDS THIS PHASE TOUCHES, not `&mut Shell` — the caller
    // has already borrowed `model`, `named_bufs`, `stream` and `alloc` out
    // of the shell, so a whole-shell borrow here is a conflict. Same wall
    // `deliver_logits` and `gdn_context` hit, same answer.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    programs: &crate::ptir::Programs,
    control: &mut Option<crate::ptir::Control>,
    sessions: &mut std::collections::BTreeMap<u64, crate::ptir::session::Session>,
    disk: &crate::ptir::Disk,
    device_ordinal: i32,
    instance_id: u64,
    logits: (u64, u32, u32),
    rows: usize,
    // The row of `logits` this instance's program reads — the last row of
    // its token span, not its index.
    row: usize,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
) -> Result<bool, i32> {
    use crate::ptir::session::{Fired, Session};

    let Some(instance) = instances.get(&instance_id) else {
        return Ok(false);
    };
    let Some(compiled) = programs.get(instance.program_id) else {
        return Ok(false);
    };
    // THE EPILOGUE'S plan, not the first one.
    //
    // Sampling is an epilogue stage. `plans.first()` was the epilogue
    // only by the accident that no program in the tree has a prologue —
    // and `fwd.adapter` puts its `lora` sink in one, so the first program
    // that carries an adapter would have had its ADAPTER fired here and
    // its sampler never run, with the fire reporting a successful publish
    // either way.
    //
    // Falling back to the first stage keeps a package that states no
    // kinds working, which is what every fixture in the tree is.
    let stage = compiled
        .stage_of_kind(crate::ptir::runtime::stage_kind::EPILOGUE)
        .unwrap_or(0);
    let Some(plan) = compiled.plans.get(stage).cloned() else {
        return Ok(false);
    };
    let Some(shapes) = instance_ring_shapes(instance, channels) else {
        eprintln!(
            "[driver-cuda-new] launch: instance {instance_id} names a channel this \
             driver does not hold; its program cannot be given rings"
        );
        return Ok(false);
    };
    let channel_ids = instance.channel_ids.clone();
    let compiled = compiled.clone();

    // THE CONTROL KERNELS, ONCE. Same disk as the program runtime's on
    // purpose: the two share a key scheme, so a second cache directory
    // would recompile both every boot and neither would ever hit.
    if control.is_none() {
        let target = ptir_target(device_ordinal)?;
        let architecture = crate::ptir::nvrtc::arch_flag(target.major, target.minor);
        match crate::ptir::Control::compile(disk, &architecture, "pie-cuda") {
            Ok(built) => *control = Some(built),
            Err(failure) => {
                eprintln!(
                    "[driver-cuda-new] launch: the PTIR control kernels will not \
                     compile ({}); this fire delivers raw logits",
                    failure.reason()
                );
                return Ok(false);
            }
        }
    }

    if !sessions.contains_key(&instance_id) {
        let session = Session::new(alloc, &shapes, stream.as_ref()).map_err(|error| {
            eprintln!("[driver-cuda-new] launch: cannot ring instance {instance_id}: {error}");
            PIE_STATUS_EXHAUSTED
        })?;
        sessions.insert(instance_id, session);
    }

    // The host planes, in the instance's channel ORDER — which is the
    // order a program indexes them by, and not the map's.
    let mut host: Vec<crate::ptir::bridge::HostChannel> = Vec::with_capacity(channel_ids.len());
    for id in &channel_ids {
        let Some(channel) = channels.get(id) else {
            return Ok(false);
        };
        host.push(channel.host_plane());
    }

    let control = control.as_ref().expect("just ensured");
    let session = sessions.get_mut(&instance_id).expect("just ensured");
    let extents = driver::Extents {
        row_count: u32::try_from(rows).unwrap_or(1),
        token_count: u32::try_from(rows).unwrap_or(1),
        sampled_rows: 1,
        ..driver::Extents::default()
    };
    match session.fire(
        &compiled,
        &plan,
        control,
        &mut host,
        logits,
        // ONE LANE per fire — `ptir::fire` says grouping is unbuilt — but
        // no longer one ROW per frame: the caller fires once per request
        // and hands each its own. When grouping lands, the lane index
        // selects among the group's rows and nothing else here changes.
        |_lane| u32::try_from(row).unwrap_or(0),
        extents,
        alloc,
        stream.as_ref(),
    ) {
        Ok(Fired::Committed { published }) => Ok(published > 0),
        Ok(Fired::Declined) => {
            sg_trace(|| format!("  ptir instance {instance_id} declined the fire"));
            Ok(false)
        }
        Ok(Fired::NotReady) => {
            sg_trace(|| format!("  ptir instance {instance_id} was not ready"));
            Ok(false)
        }
        Err(error) => {
            eprintln!("[driver-cuda-new] launch: the program refused: {error}");
            Err(PIE_STATUS_DRIVER_ERROR)
        }
    }
}

/// Publish the fire's readout: the LAST row's logits, out through the
/// instance's reader channel.
///
/// `step_impl`'s DELIVERY phase, promoted out of it. The convention until
/// the launch package's channel table is parsed: the roster's first
/// instance, its first registered channel with `host_role == READER`
/// whose cell is `[vocab]` f32. Device bf16 widens to the f32 wire on the
/// host.
///
/// Takes `debt` by `&mut Option` rather than returning a copy because the
/// two paths through here differ in WHO waits, not in what is produced: a
/// step that owes a completion hands the D2H's destination to the debt
/// and returns with the copy still queued, and a step that owes nothing
/// has already synchronized and can widen on this stack. Splitting those
/// into two functions would duplicate the channel search, which is the
/// part that is actually shared.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn deliver_logits(
    // THE THREE FIELDS OF `Shell` THIS PHASE TOUCHES, and not `&mut
    // Shell`. Not a style choice: `model` and `named_bufs` below are
    // borrowed OUT of the shell by the caller, so a `&mut Shell` here is
    // a borrow conflict, and the fix that keeps working is to name the
    // disjoint fields rather than to widen the borrow. It also documents
    // the phase — delivery reads the roster and the channel table and
    // writes exactly one buffer.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    logits_staging: &mut Option<crate::cuda::PinnedBuf>,
    frame: &PieFrameDesc,
    model: &LoadedModel,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::model::executor::DispatchPlan,
    named_bufs: &std::collections::BTreeMap<
        model_compiler::trace::ValueId,
        crate::cuda::DeviceBuffer,
    >,
    stream: crate::cuda::StreamRef<'_>,
    rows: usize,
    // Where each request's token span ends, so its answer row can be
    // found. `qo_indptr[r + 1] - 1` is request `r`'s last row.
    qo_indptr: &[u32],
    // The requests this fallback is for — the ones whose PTIR program did
    // not publish. A frame can be mixed, and a request that already has a
    // sampled answer must not also get a vocabulary.
    serve: &[usize],
    debt: &mut Option<FireDebt>,
) -> Result<(), i32> {
    use model_compiler::lower::Arg;
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
let vocab = usize::try_from(model.hf.vocab_size).unwrap_or(0);

// EVERY REQUEST, each its OWN reader channel and its OWN row.
//
// This used to take `instance_ids.first()` and `rows - 1`, so a frame
// with a roster of two published request 0's vocabulary and returned
// request 1 nothing at all. The row is not the index: request `r` owns
// `qo_indptr[r]..qo_indptr[r + 1]`, so its answer is at
// `qo_indptr[r + 1] - 1` — equal to `r` on a decode, and not on a
// prefill.
let readouts: Vec<(ChannelState, usize)> = serve
    .iter()
    .filter_map(|&r| {
        let iid = instance_ids.get(r)?;
        let inst = instances.get(iid)?;
        let end = qo_indptr.get(r + 1).copied()? as usize;
        let row = end.saturating_sub(1).min(rows.saturating_sub(1));
        let ch = inst.channel_ids.iter().find_map(|cid| {
            channels.get(cid).filter(|ch| {
                ch.host_role == driver_api::local::PIE_CHANNEL_HOST_ROLE_READER
                    && ch.cell_bytes == vocab * 4
            })
        })?;
        Some((*ch, row))
    })
    .collect();

// THE D2H IS ENQUEUED, NOT AWAITED. Its destination belongs to
// the debt rather than to this stack frame — a `Vec` here would
// be freed the moment this call returns, which with an
// asynchronous completion is before the copy lands.
//
// ONE copy carries every row, so N requests cost one D2H and N
// widenings rather than N copies.
if let (Some(lv), false) = (logits_value, readouts.is_empty())
    && let Some(buf) = named_bufs.get(&lv)
{
    match debt.as_mut() {
        Some(d) => {
            // The shell's buffer, grown to fit and reused. Not the
            // debt's: see `FireDebt::staging`.
            if logits_staging.as_ref().is_none_or(|p| p.len() < buf.len()) {
                *logits_staging = Some(
                    crate::cuda::PinnedBuf::new(buf.len())
                        .map_err(|_| PIE_STATUS_EXHAUSTED)?,
                );
            }
            let pin = logits_staging.as_mut().expect("just sized");
            let view = (pin.as_slice().as_ptr(), buf.len());
            buf.copy_to_host(&mut pin.as_mut_slice()[..buf.len()], stream)
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            d.staging = Some(view);
            d.readouts = readouts;
        }
        None => {
            // A step that owes nothing has already synchronized,
            // so the old shape is still correct for it.
            let mut bf16 = vec![0u8; buf.len()];
            buf.copy_to_host(&mut bf16, stream)
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            stream.synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            for (ch, row) in &readouts {
                let mut cell = vec![0u8; vocab * 4];
                for t in 0..vocab {
                    let off = (row * vocab + t) * 2;
                    if off + 1 < bf16.len() {
                        let bits = u16::from_le_bytes([bf16[off], bf16[off + 1]]);
                        cell[t * 4..t * 4 + 4]
                            .copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
                    }
                }
                if !ch.publish(&cell) {
                    eprintln!(
                        "[driver-cuda-new] launch: logits ring full; a request dropped \
                         its output"
                    );
                }
            }
        }
    }
}
    Ok(())
}


/// Make the shell's persistent device state ready, and reclaim what the
/// last fires finished with.
///
/// `step_impl`'s DEVICE STATE phase, promoted out of it — and the reason
/// it could be is that it is the only phase before the launch that takes
/// `&mut Shell` and borrows nothing else. Extracting it is what lets
/// every shared borrow below it (`model`, the lowering, the stream, the
/// allocator) be taken AFTER the mutation rather than across it, which is
/// the borrow conflict that stopped `deliver_logits` from taking a
/// `&mut Shell` and would have stopped each of these too.
///
/// The stream and the allocator used to be built per fire. The stream
/// because nothing needed it to outlive the call, and the allocator
/// because it was convenient — but an allocator that POOLS and is rebuilt
/// every fire has no pool, so every buffer a fire wanted was a fresh
/// `cudaMalloc`. Both are the shell's now, which is a saving on its own
/// and is the precondition for run-ahead: a second fire cannot queue
/// behind the first onto a stream that dies with the first call.
///
/// # Reclaim, and when it waits
///
/// A fire's scratch cannot be freed while it runs and cannot be freed
/// from the callback — CUDA forbids calling the runtime from a host
/// function, and `cudaFree` is the runtime — so it is freed here. What
/// matters is WHEN.
///
/// This used to hold exactly one holder and `synchronize()` on it, which
/// is a run-ahead that never runs ahead: the call that would queue fire
/// n+1 blocked until fire n had finished. It cost nothing to notice while
/// issuing a fire took longer than running one, and it is the whole game
/// now that issue is 0.81 ms against 2.9 ms of work.
///
/// So: drop everything already retired without asking the driver to wait,
/// and wait only when the queue is at depth. `RUNAHEAD_DEPTH` is the
/// backpressure — the driver runs at most that many fires ahead of the
/// GPU, which bounds the SCRATCH it is holding rather than the time.
#[cfg(feature = "abi")]
fn ready_device_state(state: &mut Shell) -> Result<(), i32> {
    if state.fire_stream.is_none() {
        state.fire_stream =
            Some(crate::cuda::OwnedStream::new(0).map_err(|_| PIE_STATUS_DRIVER_ERROR)?);
    }
    if state.fire_alloc.is_none() {
        state.fire_alloc = Some(crate::cuda::Allocator::new());
    }
    while state
        .in_flight
        .front()
        .is_some_and(|f| f.done.is_complete().unwrap_or(true))
    {
        let done = state.in_flight.pop_front().expect("just checked");
        retire(done);
    }
    while state.in_flight.len() >= RUNAHEAD_DEPTH {
        let oldest = state.in_flight.pop_front().expect("nonempty");
        oldest.done.synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        retire(oldest);
    }
    Ok(())
}

/// The hybrids' recurrent context: driver-owned slabs, instance slots.
///
/// `step_impl`'s GDN phase, promoted out of it — and the LARGEST of its
/// phases at a hundred lines, which is why it is worth the six
/// parameters. It also settles the design question the cut left open.
///
/// # Named fields, not a carrier
///
/// The middle phases each read the shell one way and write it another,
/// so `&mut Shell` is a borrow conflict against the `model`, `stream`
/// and `alloc` the caller already holds — the same wall `deliver_logits`
/// hit. Two answers were on the table: a `Fire<'a>` struct holding the
/// borrowed halves, or naming the fields each phase touches.
///
/// Naming them wins, and doing it three times is what says so. This
/// phase touches exactly ONE field of the shell (`gdn`) and reads four
/// things the caller has already resolved; a carrier would have handed
/// it the whole fire so it could reach one `Option`. The parameter list
/// IS the phase's contract, and a reader who wants to know whether the
/// recurrent slabs can affect the attention plan can answer it from the
/// signature.
///
/// The `alloc`/`stream` pair is shared and `gdn` is mutable, which the
/// compiler accepts at the call site because they are disjoint fields —
/// and that is the property a carrier would have hidden rather than
/// removed.
///
/// Returns the context the dispatch reads, and the slot-id buffer it
/// points into: the buffer is returned rather than dropped because the
/// context holds a raw pointer to it, and a fire that let it go would
/// bind a freed address.
#[cfg(feature = "abi")]
fn gdn_context(
    gdn: &mut Option<GdnState>,
    family: &dyn PlannedFamily,
    step: &driver_api::local::PieStepDesc,
    requests: usize,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
) -> Result<(Option<crate::model::executor::GdnCtx>, Option<crate::cuda::DeviceBuffer>), i32> {
        use crate::model::executor::GdnCtx;

    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::cuda::DeviceBuffer> = None;
    if let Some(shape) = family.gdn_shape() {
        let (conv_stride, state_stride, state_elem) =
            (shape.conv_stride, shape.state_stride, shape.state_elem);
        const GDN_SLOTS: u32 = 8;
        if (*gdn).is_none() {
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
            (*gdn) = Some(GdnState {
                slabs,
                num_slots: GDN_SLOTS,
                conv_stride_elems: i64::try_from(conv_stride).unwrap_or(0),
                state_stride_elems: i64::try_from(state_stride).unwrap_or(0),
                state_elem_bytes: state_elem,
            });
        }
        let gdn_state = (*gdn).as_mut().expect("just ensured");
        // The ENGINE assigns slots: `rs_slot_ids`, one per request. RESET
        // zeroes a slot before the fire; BUFFER_WRITE routes the pass's
        // state into a buffer slot instead of the live one; FOLD copies
        // the accepted prefix back afterwards. See the fold below.
        let rs_slot_ids = slice_of(step.rs_slot_ids.ptr, step.rs_slot_ids.len);
        let rs_flags = slice_of(step.rs_slot_flags.ptr, step.rs_slot_flags.len);
        if rs_slot_ids.len() != requests {
            eprintln!("[driver-cuda-new] launch: hybrid fire without rs_slot_ids");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        // THE BUFFER/FOLD FLAGS ARE ACCEPTED NOW, and that is directive
        // 4.2 of `.wiki/driver/graph.md`.
        //
        // They used to be refused wholesale — "rs fold/buffer flags await
        // spec-decode" — which left the tree carrying TWO mechanisms for
        // one job: this one, complete on the ABI side since v23/v24, and
        // the repair fire classes (`CommitAdvance`, `StateOnly`,
        // `FrozenVerify`) that existed only because this one was off.
        //
        // The mechanism is the whole argument for deleting them. A
        // speculative decode writes its tokens into a BUFFER slot and
        // folds only the accepted prefix into the live slot; a rejected
        // token is simply never folded, so there is nothing to repair.
        // `FrozenVerify` is "prefill plus a verify-stash store" — the
        // buffer IS the stash. `CommitAdvance` is "replay the confirmed
        // prefix" — the fold length IS that prefix.
        let rs_fold_lens = slice_of(step.rs_fold_lens.ptr, step.rs_fold_lens.len);
        let rs_buffer_slot_ids =
            slice_of(step.rs_buffer_slot_ids.ptr, step.rs_buffer_slot_ids.len);
        let rs_buffer_indptr =
            slice_of(step.rs_buffer_slot_indptr.ptr, step.rs_buffer_slot_indptr.len);
        let need_slots = rs_slot_ids.iter().copied().max().map_or(1, |m| m + 1);
        gdn_state.ensure_slots(need_slots, &alloc, &stream)?;
        for (r, &slot) in rs_slot_ids.iter().enumerate() {
            if rs_flags.get(r).copied().unwrap_or(0) & driver_api::local::PIE_RS_FLAG_RESET
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
        let slot_ids_h: Vec<i32> = rs_slot_ids
            .iter()
            .enumerate()
            .map(|(r, &live)| {
                // A BUFFER_WRITE row's pass writes the BUFFER slot, not
                // the live one — that is the whole of "the pass scatters
                // its own tokens". The buffer CSR names one slot per
                // buffered token; the pass writes the row's head, which
                // is its first entry.
                let f = rs_flags.get(r).copied().unwrap_or(0);
                let slot = if f & driver_api::local::PIE_RS_FLAG_BUFFER_WRITE != 0 {
                    rs_buffer_indptr
                        .get(r)
                        .and_then(|&lo| rs_buffer_slot_ids.get(lo as usize))
                        .copied()
                        .unwrap_or(live)
                } else {
                    live
                };
                i32::try_from(slot).unwrap_or(0)
            })
            .collect();
        // Every slot the fire names has to exist, buffer slots included.
        let need_buffer = rs_buffer_slot_ids.iter().copied().max().map_or(0, |m| m + 1);
        gdn_state.ensure_slots(need_buffer.max(need_slots), &alloc, &stream)?;
        // THE FOLD, recorded on the fire's stream so it lands after the
        // pass that filled the buffer. Copying the accepted prefix's LAST
        // state into the live slot is the whole operation: a linear
        // state is a running summary, so the state after token `k` is the
        // state the next fire continues from, and the tokens past `k`
        // were rejected and are simply never folded.
        for (r, &live) in rs_slot_ids.iter().enumerate() {
            let f = rs_flags.get(r).copied().unwrap_or(0);
            if f & driver_api::local::PIE_RS_FLAG_FOLD == 0 {
                continue;
            }
            let (lo, hi) = match (rs_buffer_indptr.get(r), rs_buffer_indptr.get(r + 1)) {
                (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
                _ => continue,
            };
            // A device-resolved length is CLAMPED to the row's replay
            // length, which the ABI names as the bound. The port itself
            // is not read yet, so a device row folds its whole replay —
            // the conservative answer, and the one a non-speculative
            // fire produces anyway.
            let span = hi.saturating_sub(lo);
            let want = if f & driver_api::local::PIE_RS_FLAG_FOLD_LEN_DEVICE != 0 {
                span
            } else {
                rs_fold_lens.get(r).copied().unwrap_or(0) as usize
            };
            let take = want.min(span);
            if take == 0 {
                continue;
            }
            let Some(&src_slot) = rs_buffer_slot_ids.get(lo + take - 1) else { continue };
            for slab in gdn_state.slabs.iter().flatten() {
                use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
                let conv_bytes = gdn_state.conv_stride_elems as usize * 2;
                let st_bytes =
                    gdn_state.state_stride_elems as usize * gdn_state.state_elem_bytes;
                for (buf, stride) in [(&slab.0, conv_bytes), (&slab.1, st_bytes)] {
                    let code = unsafe {
                        cudaMemcpyAsync(
                            buf.as_ptr().cast::<u8>().add(live as usize * stride).cast(),
                            buf.as_ptr()
                                .cast::<u8>()
                                .add(src_slot as usize * stride)
                                .cast_const()
                                .cast(),
                            stride,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_ref().as_raw().cast(),
                        )
                    };
                    if code != cudaError::cudaSuccess {
                        return Err(PIE_STATUS_DRIVER_ERROR);
                    }
                }
            }
        }
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
    Ok((gdn_ctx, _slot_ids_buf))
}


/// Size the KV pools for this fire and describe them per layer.
///
/// `step_impl`'s KV phase. It touches ONE field of the shell — `kv` — and
/// bumps the array epoch when it grows, which is the second reason it is
/// worth being a function: growth MOVES base addresses, so every capture
/// that recorded one is stale, and the bump is what says so. Keeping that
/// pair in one place is what keeps a reader from having to notice it.
///
/// # Per-layer geometry, and the layers that own no pool
///
/// A family may share one layer's cache with another — gemma-4's trailing
/// layers project no KV of their own — so `kv_source(l)` says whose pool a
/// layer reads and only the SOURCES get an allocation. The views then
/// point every layer at its source's pages, which is why the returned
/// vector is as long as the layer count and the pool vector is not.
///
/// # What growth does not do
///
/// It REPLACES the pools without migrating pages. Decode continuity holds
/// while page demand is stable, which is the single-frame world; page
/// migration rides with `resize_pool`.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn kv_pools_for(
    kv: &mut Option<KvState>,
    epoch: &mut u64,
    family: &dyn PlannedFamily,
    model: &LoadedModel,
    need_pages: u32,
    page_size: i32,
    alloc: &crate::cuda::Allocator,
    stream: &crate::cuda::OwnedStream,
    format: crate::store::KvCacheFormat,
) -> Result<Vec<crate::launch::KvCacheLayerView>, i32> {
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let head_dim_u = u32::try_from(head_dim_i).unwrap_or(0);
    let n = family.layers();
    // Per-layer geometry, family-decided: gemma-4's two layer kinds
    // disagree on head dim, and its trailing layers own NO pages (they
    // attend through their source's — the load-time decision).
    let per_layer = crate::store::kv_cache::PerLayer {
        head_dim: (0..n).map(|l| family.head_dim_of(l, head_dim_u) as i32).collect(),
        kv_source_layer: (0..n).map(|l| family.kv_source(l).unwrap_or(l) as i32).collect(),
        num_kv_heads: vec![kv_heads_i; n as usize],
    };
    // One set of pages has ONE shape, so a layer that reads through
    // another's must have that layer's dims. gemma-4 holds this without
    // trying — `kv_source` searches by the same predicate `head_dim_of`
    // keys on — but that is an invariant spread across two functions in
    // another crate. A violation would not crash: every shared layer
    // would read its source's pages at its own stride and emit plausible
    // tokens. It matters HERE because `layer_view` reports an aliased
    // layer's dims as its SOURCE's.
    per_layer.check_sharing().map_err(|_| PIE_STATUS_INVALID_ARGUMENT)?;

    let grow = !matches!(&(*kv), Some(kv) if kv.num_pages >= need_pages);
    if grow {
        let layout = crate::store::kv_cache::KvCacheLayout::plan_per_layer(
            n as i32,
            need_pages as i32,
            page_size,
            kv_heads_i,
            per_layer,
            format,
            false,
        )
        .map_err(|_| PIE_STATUS_INVALID_ARGUMENT)?;

        let mut ops = crate::store::kv_cache_live::LiveKvCacheOps::new(
            stream.as_ref().as_raw().cast(),
            alloc,
        );
        let cache = crate::store::kv_cache_live::KvCache::materialize(layout, &mut ops)
            .map_err(|_| PIE_STATUS_EXHAUSTED)?;
        let mut held = ops.into_held();
        // `materialize` does not zero, and the C++ did not either — but
        // the shell's hand-built pools did, and a page read before its
        // first write is otherwise whatever the allocator last had.
        for b in &mut held {
            b.memset(0, stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        }

        // NOTE: growth REPLACES the pages without migrating them — decode
        // continuity holds while the page demand is stable, which is the
        // single-frame smoke's world. Page migration rides with resize_pool.
        //
        // AND IT MOVES BASE ADDRESSES, so every capture that recorded one is
        // stale. Same rule as `FireArrays`' own growth, and the same bump:
        // stale means recapture rather than a replay into freed pages.
        (*kv) = Some(KvState { cache, _held: held, num_pages: need_pages });
        *epoch += 1;
    }
    Ok((*kv).as_ref().expect("just ensured").views())
}


#[cfg(feature = "abi")]
#[allow(clippy::too_many_lines)]
fn step_impl(
    state: &mut Shell,
    frame: &PieFrameDesc,
    step: &driver_api::local::PieStepDesc,
    // `owes` is the debt this step carries when it is the frame's LAST:
    // `None` for the earlier steps, which owe nothing because a frame
    // completes once. A step handed one enqueues an asynchronous
    // completion and does NOT synchronize; a step handed `None`
    // synchronizes, because the next step's work depends on it and the
    // producer→consumer ordering inside a frame is what makes steps
    // sequential in the first place.
    owes: Option<(PieCompletion, Vec<*mut driver_api::local::PieTerminalCell>)>,
) -> Result<(), i32> {
    use crate::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use crate::model::executor::{
        AttnCtx, AttnRegions, DecodePlan, DispatchCtx, Frame, PrefillPlan, Resolver,
        run,
    };
    use model_compiler::lower::Arg;
    use model_compiler::trace::ValueId;

    let t_head = std::time::Instant::now();
    let (Admitted { class, rows, requests, fire_rows }, family) = admit(state, step)?;
    // THE MUTATION FIRST, AND THEN THE BORROWS. `ready_device_state` takes
    // `&mut Shell`, so every shared borrow this function goes on to hold —
    // `model`, the lowering, the stream, the allocator — has to be taken
    // after it. Reading the layer count out as a NUMBER rather than
    // keeping `model` alive across the call is the whole of what that
    // costs, and it is what lets the phase be a function at all.
    let layers = state
        .model
        .as_ref()
        .ok_or(PIE_STATUS_INVALID_ARGUMENT)?
        .hf
        .num_hidden_layers;
    ready_device_state(state)?;
    let model = state.model.as_ref().ok_or(PIE_STATUS_INVALID_ARGUMENT)?;
    let token_ids = slice_of(step.token_ids.ptr, step.token_ids.len);
    let position_ids = slice_of(step.position_ids.ptr, step.position_ids.len);
    let kv_indices = slice_of(step.kv_page_indices.ptr, step.kv_page_indices.len);
    let kv_indptr = slice_of(step.kv_page_indptr.ptr, step.kv_page_indptr.len);
    let kv_lens = slice_of(step.kv_last_page_lens.ptr, step.kv_last_page_lens.len);
    let qo_indptr = slice_of(step.qo_indptr.ptr, step.qo_indptr.len);

    sg_trace(|| format!("  head {:?}", t_head.elapsed()));
    let t_low = std::time::Instant::now();
    // ── The lowering, or the one this shape already has. ──
    //
    // Everything between here and `DispatchPlan` is a pure function of the
    // key, and it costs ~3.3 ms on a 0.6B decode. See `Shell::lowerings`.
    let key = LoweringKey {
        model_id: u64::from(layers.unsigned_abs()),
        class,
        rows: u32::try_from(rows).unwrap_or(0),
        rows_digest: digest_rows(&fire_rows),
        union_asked: supergraph_enabled() && !family.recurrent(),
    };
    if !state.lowerings.contains_key(&key) {
        let built = build_lowering(family.as_ref(), class, &fire_rows, key.union_asked)?;
        state.lowerings.insert(key, built);
    }
    let LoweredFire { plan, lowered, dplan, union } =
        state.lowerings.get(&key).expect("just built");
    let union = *union;
    sg_trace(|| format!("  lowering {:?}", t_low.elapsed()));
    let mut phase = std::time::Instant::now();
    let mut lap = |what: &str| {
        sg_trace(|| format!("  {what} {:?}", phase.elapsed()));
        phase = std::time::Instant::now();
    };

    let stream = state.fire_stream.as_ref().expect("just ensured");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = state.fire_alloc.as_ref().expect("just ensured");

    let need_pages = frame.required_kv_pages.max(
        kv_indices.iter().copied().max().map_or(1, |m| m + 1),
    );
    let page_size: i32 = 16;
    // Re-derived here as well as inside `kv_pools_for`, because the
    // attention plans below want the same two numbers and returning them
    // would make the phase's result a tuple whose second half is a
    // restatement of its arguments.
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let kv_format = state.kv_format;
    let layers = kv_pools_for(
        &mut state.kv,
        &mut state.fire_arrays.epoch,
        family.as_ref(),
        model,
        need_pages,
        page_size,
        alloc,
        stream,
        kv_format,
    )?;

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
    const S_SAMPLED: usize = 8;
    let d_ids = state.fire_arrays.upload_u32(&alloc, S_IDS, token_ids, stream.as_ref())?;
    let d_pos = state.fire_arrays.upload_u32(&alloc, S_POS, position_ids, stream.as_ref())?;
    let d_kv_indices =
        state.fire_arrays.upload_u32(&alloc, S_KV_INDICES, kv_indices, stream.as_ref())?;
    let d_kv_indptr =
        state.fire_arrays.upload_u32(&alloc, S_KV_INDPTR, kv_indptr, stream.as_ref())?;
    let d_kv_lens =
        state.fire_arrays.upload_u32(&alloc, S_KV_LENS, kv_lens, stream.as_ref())?;
    let d_qo = state.fire_arrays.upload_u32(&alloc, S_QO, qo_indptr, stream.as_ref())?;
    // WHICH ROWS the epilogue gathers, from the rows the step described.
    // Derived here rather than taken from `sampling_indices` directly, so
    // the pointer and the guard that produced it cannot disagree: the
    // lowering states a gather exactly when `sampled < window.len()`, and
    // it counts the same `Row::samples` this reads.
    let sampled_rows: Vec<u32> = fire_rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.samples.then_some(u32::try_from(i).unwrap_or(0)))
        .collect();
    let d_sampled = if sampled_rows.len() == rows {
        // Every row sampled means no gather is stated, and a pointer for
        // a launch nobody makes is one more thing to keep in step.
        core::ptr::null()
    } else {
        state.fire_arrays.upload_u32(&alloc, S_SAMPLED, &sampled_rows, stream.as_ref())?
    };

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

    lap("kv+arrays");
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
    // EVERY PLAN THE GEOMETRY PERMITS, not just the one this fire's text
    // states. Under `GuardMode::Union` both arms of an attention guard are
    // recorded, and a capture that walks an arm whose plan was never
    // raised is abandoned — so prepared state belongs to the bucket rather
    // than to the arm (`.wiki/driver/graph.md` §5 ①). It is also the
    // precondition for §4.1: a merged Decode/Prefill graph needs BOTH
    // classes' plans standing whenever either is captured.
    //
    // The cost is plan-raise time. It is not runtime waste — the arm a
    // fire does not take is skipped by the conditional, not executed.
    let planless_prefill = family.planless_prefill();
    let decode_plan_full_ptr = if let Some((d_sliding, d_full)) = family.decode_plan_head_dims() {
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
    };
    // gemma-4's prefill is PLANLESS (it plans internally per fire, off the
    // host CSR mirrors) and its 512-wide layers take the naive kernel —
    // there is nothing to pre-plan, so it is the one plan that stays
    // unraised.
    if !planless_prefill {
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
    }
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
    // THE SCORE SINK IS UNCONDITIONAL, and that is a reversal.
    //
    // It used to be null on purpose: a fire that wants no scores prepares
    // no score path, the capturing dispatch refuses without one, and
    // refusing before the launcher is reached beats an exception crossing
    // the C ABI with nowhere to go. Sound — but it makes the union decline
    // every lowering that so much as MENTIONS `_capture`, which is the one
    // case the union was built for. `WantsAttnScore` is a FOLDED
    // predicate: one exec is supposed to serve the fire that wants scores
    // and the fire that does not, and under `GuardMode::Union` both arms
    // are recorded whether or not this fire takes either.
    //
    // So the question is not "what does this fire need" but "what could
    // any arm need", and the answer is published every time. The cost is
    // resident memory and plan-raise time; it is NOT runtime waste,
    // because the arm a fire does not take is skipped by the conditional
    // rather than executed. `.wiki/driver/graph.md` §5 ①.
    let score_window = if states_decode_dispatch {
        1
    } else {
        crate::model::attn_score::default_attn_score_window()
    };
    let sink = crate::model::attn_score::plan_score_sink(
        kv_indptr,
        kv_lens,
        page_size,
        u32::try_from(model.hf.num_attention_heads).unwrap_or(0),
        score_window,
    );
    let (d_scores, d_folded, d_score_indptr) = match sink {
        // A sink too large to publish (the prefill window grows with the
        // context) keeps the old answer: null, and the capturing arm
        // declines exactly as it did.
        None => (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
        Some(p) => {
            let base = state.fire_arrays.score(&alloc, &p, stream.as_ref())?;
            (
                base,
                unsafe { base.cast::<u8>().add(p.folded_offset) }.cast::<std::ffi::c_void>(),
                unsafe { base.cast::<u8>().add(p.indptr_offset) }.cast::<i32>().cast_const(),
            )
        }
    };

    // THE CUSTOM MASK, likewise unconditional. `HasCustomMask` is folded
    // too, so the `_custom` arm has to be recordable whether or not this
    // fire stages a mask. The resident form is plain causal — the same
    // answer the unmasked arm computes — so taking the arm with nothing
    // staged is correct rather than merely safe. A real staged mask
    // overwrites these bytes; the addresses do not move.
    let (d_mask, d_mask_indptr) = match crate::model::page_mask::element_mask::plan_causal(
        qo_indptr,
        kv_indptr,
        kv_lens,
        page_size,
    ) {
        None => (core::ptr::null(), core::ptr::null()),
        Some(p) => {
            let base = state.fire_arrays.mask(&alloc, &p, stream.as_ref())?;
            (
                base.cast::<u8>().cast_const(),
                unsafe { base.cast::<u8>().add(p.indptr_offset) }.cast::<i32>().cast_const(),
            )
        }
    };

    // The driver-owned attention landing, resolved BEFORE `named_bufs`
    // borrows the map: a fire whose op join names no output slot lands
    // here instead of losing its graph. Null when the family states its
    // attention output as an SSA arg (gemma-4 does).
    let d_attn_out = if family.pins_attention_values() {
        state.fire_arrays.attn_out(
            &alloc,
            rows * model.hf.num_attention_heads as usize
                * usize::try_from(model.hf.head_dim).unwrap_or(0)
                * 2,
        )?
    } else {
        core::ptr::null_mut()
    };

    let named_bufs = &state.fire_arrays.named;

    lap("attn-plan");
    // ── The hybrid's GDN context: driver-owned slabs, instance slots. ──
    let (gdn_ctx, _slot_ids_buf) =
        gdn_context(&mut state.gdn, family.as_ref(), step, requests, alloc, stream)?;

    let lse = alloc
        .alloc(rows * model.hf.num_attention_heads as usize * 4)
        .map_err(|_| PIE_STATUS_EXHAUSTED)?;


    // The guard-owned attention values, discovered from the lowering as
    // the smokes discovered them. gemma-4 has NONE: both its attention
    // forms state [q, o] as SSA args, so the pins stay null.
    let (q_pin, o_off): (Option<ValueId>, Option<usize>) = if !family.pins_attention_values() {
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
        // A JOIN THAT NAMES NO SLOT IS NOT A REFUSAL any more.
        //
        // This used to return `PIE_STATUS_UNSUPPORTED`, and before the
        // union it also fell back to the neighbour read -- "the launch
        // after the dispatch is the o_proj, so its input is the slot the
        // dispatch wrote". That is a fact read off where a statement SITS,
        // and it is false under `GuardMode::Union`, where every arm is
        // present and the neighbour belongs to some other body.
        //
        // But `AttnCtx::o_out` is a DRIVER-owned pointer -- the whole
        // reason it exists is that the region's launches record no SSA
        // output of their own. So a fire whose join names no slot gets a
        // driver-owned landing buffer, and keeps its graph
        // (`.wiki/driver/graph.md` §5 (2)).
        (q_pin, attention_landing(&lowered, &dplan, fi))
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
        folded_out: d_folded.cast(),
        score_indptr_d: d_score_indptr.cast(),
        mask_d: d_mask,
        mask_indptr_d: d_mask_indptr,
        o_out: match o_off {
            Some(off) => unsafe { arena_ptr.cast::<u8>().add(off) }.cast(),
            // No stated slot: the driver's own landing buffer, sized to
            // the fire's attention output and pooled like the rest so a
            // capture that baked its address keeps addressing something.
            None => d_attn_out,
        },
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

    // ONE HANDLE FOR THE DRIVER, its stream rebound per fire. See
    // `Shell::cublas`: creating and destroying one per fire cost 3.2 ms.
    let mut cublas_ops = crate::cuda::cublas::LiveCublas;
    if state.cublas.is_none() {
        state.cublas = Some(
            crate::cuda::cublas::CublasHandle::create(&mut cublas_ops, raw_stream)
                .map_err(|_| PIE_STATUS_DRIVER_ERROR)?,
        );
    }
    let cublas = state.cublas.as_mut().expect("just ensured");
    cublas
        .set_stream(&mut cublas_ops, raw_stream)
        .map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    // The family's per-layer tables and named constants — the C++
    // parse-time vectors (`per_layer_rope_theta`, `rotary_of`) and the
    // prologue's `scale.*` values plus the load-read layer scalars. A
    // family whose rope is one theta and whose epilogue caps nothing
    // answers with empties.
    let FamilyTables { theta_by_layer, rotary_by_layer, softcap, ple_dim, scales } =
        family.tables(model);
    // THE PEEL WINDOW, and this is where layer 3 stops being vocabulary.
    //
    // It used to be `set(0, rows)` on every fire — "start 0, count ALL",
    // which is the word for NO SPLIT. Nothing ever computed a boundary, so
    // the per-row polymorphism the `_devwin` kernels exist for had never
    // once executed (`.wiki/driver/graph.md` §2, §5 ③).
    //
    // Worse than unused: WRONG whenever a peel did lower. `lower::split_at`
    // computes the real boundary from the fire's rows and gives the tail
    // region the rectangle `[split, N)` — but a `_devwin` launch ignores
    // `bound.rows.start` by contract and reads this word instead. Saying
    // "all rows" therefore ran the tail's program over the PREFIX rows
    // too, silently, on every peeled fire.
    //
    // The window is read off the lowering rather than re-derived, because
    // the lowering already answered: the tail rectangle IS the window its
    // launches want, and two derivations of one split is how they drift.
    if state.peel_win.is_none() {
        state.peel_win = Some(
            crate::cuda::PeelWindowWord::new(alloc).map_err(|_| PIE_STATUS_EXHAUSTED)?,
        );
    }
    let peel_win = state.peel_win.as_mut().expect("just ensured");
    let (peel_start, peel_count) = lowered
        .launches
        .iter()
        .find(|l| l.peel.is_some_and(|p| p.tail))
        .map_or_else(
            // No peel lowered: the whole fire is one region, which is what
            // an unpeeled fire means.
            || (0, u32::try_from(rows).unwrap_or(0)),
            |l| (l.rows.start, l.rows.end.saturating_sub(l.rows.start)),
        );
    peel_win.set(peel_start, peel_count);
    peel_win.upload(stream.as_ref()).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let peel_window_ptr = peel_win.device_ptr();

    let ctx = DispatchCtx {
        sampling_indices: d_sampled.cast::<i32>(),
        sampled_rows: i32::try_from(sampled_rows.len()).unwrap_or(0),
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
        peel_window: peel_window_ptr,
        rows_total: i32::try_from(rows).unwrap_or(0),
    };

    lap("bind");
    let mut resolver = LiveResolver { model, named: &named_bufs };
    let regions = AttnRegions::whole(Some(&attn));
    // The last use of `alloc` is above, so the shared borrow is dead and the
    // capture can take the same allocator mutably — which is the point: a
    // capture has to be opened on the allocator that owns what the fire
    // frees, or the frees are not deferred.
    if state.preds.is_none() {
        state.preds = crate::cuda::PredicateWord::new(
            state.fire_alloc.as_ref().expect("the fire allocator exists"),
        )
        .ok();
    }
    let (capture_alloc, capture_preds) = match (&mut state.fire_alloc, &mut state.preds) {
        (Some(a), Some(p)) => (a, p),
        _ => return Err(PIE_STATUS_EXHAUSTED),
    };
    lap("ctx");
    let result = if union {
        capture_or_replay(
            &mut state.supergraph,
            state.fire_arrays.epoch,
            u64::from(model.hf.num_hidden_layers.unsigned_abs()),
            &plan, &fire_rows, &lowered, &dplan, exec_frame, &mut resolver, &ctx,
            regions, gdn_ctx.as_ref(), capture_alloc, capture_preds, stream.as_ref(),
            requests, rows, class,
        )
    } else {
        run(&lowered, &dplan, exec_frame, &mut resolver, &ctx, regions, gdn_ctx.as_ref())
    };
    lap("run");
    // A step that owes nothing SYNCHRONIZES, because the next step in the
    // frame reads what this one wrote. A step that owes the frame's
    // completion does not: its debt rides a stream-ordered callback and
    // this call returns with the work still queued, which is the whole
    // point.
    let sync = if owes.is_some() && state.runahead {
        Ok(())
    } else {
        stream.as_ref().synchronize()
    };
    lap("sync");
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

    lap("post-match");
    // THE FRAME'S DEBT, built before the delivery below because it is
    // owed whether or not this fire has logits to deliver. Paying it
    // inside the delivery block meant a fire with no readout channel
    // never published its terminal cells and never notified — the
    // runtime waited forever on a frame the driver had finished.
    let mut debt = owes.map(|(completion, cells)| FireDebt {
        staging: None,
        readouts: Vec::new(),
        vocab: usize::try_from(model.hf.vocab_size).unwrap_or(0),
        cells,
        completion,
        notify: state.notify,
        notify_ctx: state.notify_ctx,
    });

    // ── Sampling: the instance's PROGRAM, if it has one. ──
    //
    // Before the delivery below and not after, because a program that
    // published is a fire whose answer has already gone out — top-p,
    // top-k, temperature and argmax are its stages, and handing the
    // caller raw logits beside them would deliver twice and disagree
    // with itself.
    //
    // Everything about this degrades to the old behaviour: no program, a
    // program that declines, inputs not ready, or channels this shell
    // does not hold all return `false` and fall through to the raw
    // logits. That is what the driver did before this existed, so the
    // worst case is the status quo rather than a broken fire.
    // A FRESH borrow of the allocator, not the one from the top of the
    // fire. The capture above takes `&mut state.fire_alloc` — deliberately,
    // since a capture must be opened on the allocator that owns what the
    // fire frees — and reusing the earlier binding here would extend a
    // shared borrow across it. Re-deriving is one line and says where the
    // mutable window ended.
    let alloc = state.fire_alloc.as_ref().expect("the fire allocator exists");
    let vocab = u32::try_from(model.hf.vocab_size).unwrap_or(0);
    let readout = (0..lowered.launches.len()).rev().find_map(|i| {
        dplan.spec(i).outs.first().and_then(|a| match a {
            Arg::Named { value, .. } => Some(*value),
            Arg::Arena { .. } | Arg::Weight(_) => None,
        })
    });
    let logits_base = readout
        .and_then(|v| named_bufs.get(&v))
        .map_or(0, |b| b.as_ptr() as u64);
    // EVERY REQUEST, each over its OWN row.
    //
    // This used to fire only `instance_ids.first()`, and then let a
    // successful publish suppress `deliver_logits` for the whole frame —
    // so a two-request batch sampled request 0 and returned request 1
    // NOTHING AT ALL: no sample, because its program never ran, and no
    // logits, because request 0's had.
    //
    // A request's logits row is the last row of its token span, which is
    // what `qo_indptr` states: request `r` owns `qo_indptr[r]
    // ..qo_indptr[r + 1]`, so its row is `qo_indptr[r + 1] - 1`. On a
    // decode that is `r`; on a prefill it is not, and the difference is
    // the whole reason to read the indptr rather than count.
    //
    // Still one lane per fire — `ptir::fire`'s grouping is unbuilt — so
    // this is N single-lane fires rather than one N-lane fire. Slower and
    // correct, which is the right order.
    let instance_ids = slice_of(frame.instance_ids.ptr, frame.instance_ids.len);
    // Which requests still need raw logits: the ones whose program did
    // not publish. A frame can be MIXED — one request bound to a sampling
    // program and another not — and each half has to be served, which is
    // why this is a set rather than a flag.
    let mut unsampled: Vec<usize> = Vec::new();
    for (r, &iid) in instance_ids.iter().enumerate() {
        let Some(&end) = qo_indptr.get(r + 1) else { break };
        let row = (end as usize).saturating_sub(1).min(rows.saturating_sub(1));
        if run_program(
            &state.instances,
            &state.channels,
            &state.ptir_programs,
            &mut state.ptir_control,
            &mut state.ptir_sessions,
            state.ptir.disk(),
            state.device_ordinal,
            iid,
            (logits_base, vocab, vocab),
            rows,
            row,
            alloc,
            stream,
        )? {
            continue;
        }
        unsampled.push(r);
    }
    lap("sample");

    if !unsampled.is_empty() {
        deliver_logits(
            &state.instances,
            &state.channels,
            &mut state.logits_staging,
            frame,
            model,
            lowered,
            dplan,
            named_bufs,
            stream.as_ref(),
            rows,
            qo_indptr,
            &unsampled,
            &mut debt,
        )?;
    }

    // The debt goes last in stream order, so it runs after every
    // launch and after the D2H above.
    if let Some(d) = debt {
        let raw = Box::into_raw(Box::new(d)).cast::<std::ffi::c_void>();
        // ONE SET OF DEBTS, TWO WAYS TO PAY THEM. Gated off, this
        // thread pays after the synchronize above — which is what the
        // driver always did, and what every caller reading a result
        // on the next line still expects. Gated on, a stream-ordered
        // callback pays them and this call returns with the work
        // still queued.
        if !state.runahead {
            // The D2H above was ENQUEUED, so paying here means waiting
            // here. Without this the callback's staging is read before
            // the copy into it has landed.
            stream.as_ref().synchronize().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
            unsafe { retire_fire(raw) };
            return Ok(());
        }
        if let Err(e) = unsafe { stream.as_ref().host_fn(retire_fire, raw) } {
            // The callback never ran, so nothing will reclaim the box
            // or pay the debt — do both here rather than leak a frame
            // the runtime is waiting on.
            eprintln!("[driver-cuda-new] launch: cannot enqueue completion: {e:?}");
            let _ = stream.as_ref().synchronize();
            unsafe { retire_fire(raw) };
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
        // AND THE SCRATCH SURVIVES THE CALL. Dropping it here would
        // `cudaFree` while the fire runs, which synchronizes the
        // device and undoes everything above. The next launch
        // reclaims it — see `InFlight`.
        lap("debt");
        let done = crate::cuda::Event::new().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        stream.as_ref().record(&done).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
        state.in_flight.push_back(InFlight {
            done,
            scratch: [Some(lse), Some(d_valid), _slot_ids_buf]
                .into_iter()
                .flatten()
                .collect(),
            closed_channels: Vec::new(),
        });
    }
    lap("tail");
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
        model.weights.get(n).map(|b| b.ptr.cast_const()).ok_or_else(|| {
            eprintln!("[driver-cuda-new] encode: missing audio weight {n}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let opt = |n: String| -> *const std::ffi::c_void {
        model.weights.get(&n).map_or(core::ptr::null(), |b| b.ptr.cast_const())
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
        .map_or(0, |b| b.bytes / (usize::try_from(ac.output_proj_dims.max(1)).unwrap_or(1) * 2));
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
pub fn pie_cuda_encode(
    driver: *mut PieDriver,
    encode: *const PieEncodeDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_encode", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            encode,
            |d| unsafe { driver_api::local::validate_encode_desc(d) },
            "encode",
        ) {
            Ok(d) => d,
            Err(status) => return status,
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
            model.weights.get(n).map(|b| b.ptr.cast_const()).ok_or_else(|| {
                eprintln!("[driver-cuda-new] encode: missing vision weight {n}");
                PIE_STATUS_UNSUPPORTED
            })
        };
        let opt = |n: String| -> *const std::ffi::c_void {
            model.weights.get(&n).map_or(core::ptr::null(), |b| b.ptr.cast_const())
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
            .map_or(0, |b| b.bytes / (2 * hidden * 2));
        let text_hidden = model
            .weights
            .get("model.embed_vision.embedding_projection.weight")
            .map_or(0, |b| b.bytes / (hidden * 2));

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
    })
}

/// KV copies within the device domain: whole-page copies (`src_page_ids`
/// → `dst_page_ids`, every layer, both planes) and beam-repair CELL
/// moves through the bridged `copy_kv_cells_bf16`. Host-pinned domains
/// refuse until the swap pool wires in — a swap, not a copy, and its
/// store is ported but not yet mounted here.
pub fn pie_cuda_copy_kv(
    driver: *mut PieDriver,
    copy: *const PieKvCopyDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_copy_kv", PIE_STATUS_DRIVER_ERROR, move || {
        use driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE;

        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(copy, driver_api::local::validate_kv_copy_desc, "copy_kv") {
            Ok(d) => d,
            Err(status) => return status,
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
            if !kv_ref.uniform_stride(page_bytes) && (host_src || host_dst) {
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
            for l in 0..kv_ref.layers() {
                // A layer that owns no pages has none to move.
                let (Some((k, v)), Some(pb)) = (kv_ref.owned(l), kv_ref.page_bytes(l)) else {
                    continue;
                };
                for (plane, pool) in [k, v].into_iter().enumerate() {
                    let dev = pool.cast::<u8>();
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
            // Only the layers that OWN pages: a shared layer's cells live
            // in its source's pool, so visiting it too would copy them
            // twice.
            for i in 0..kv_ref.layers() {
                let (Some((k, v)), Some(d)) = (kv_ref.owned(i), kv_ref.head_dim(i)) else {
                    continue;
                };
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
    })
}

/// Direct recurrent-state copies: WHOLE-SLOT d2d over the hybrid's GDN
/// slabs (conv + recurrent, every linear layer), the C++ shape
/// (`context.cpp` ignores the token fields — those ride for the rs
/// BUFFER pool, spec-decode machinery). Slot ids are the engine's; the
/// slabs grow with migration to cover them.
pub fn pie_cuda_copy_state(
    driver: *mut PieDriver,
    copy: *const PieStateCopyDesc,
    _completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_copy_state", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(copy, driver_api::local::validate_state_copy_desc, "copy_state") {
                Ok(d) => d,
                Err(status) => return status,
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
    })
}

/// Resize the KV pool to `target_pages`, MIGRATING the surviving pages —
/// the migration the launch-time growth deliberately skipped. Shrinks
/// drop the tail; `map_ranges`/`unmap_ranges` (the elastic-VMM form) are
/// accepted but the shell's pools are plain allocations, so the target
/// page count is the whole contract here — stated, not hidden.
pub fn pie_cuda_resize_pool(
    driver: *mut PieDriver,
    resize: *const PiePoolResizeDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_resize_pool", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(resize, driver_api::local::validate_pool_resize_desc, "resize_pool") {
                Ok(d) => d,
                Err(status) => return status,
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

        // A resize changes the page COUNT and nothing else, so an existing
        // cache states its own geometry and the config is never re-read.
        // Before any cache exists there is nothing to ask, and the config
        // decides: gemma-4 by its layer kinds and shared tail, every other
        // family uniformly.
        let layout = match old.as_ref().map(|o| o.cache.layout().with_num_pages(target as i32)) {
            Some(Ok(l)) => l,
            Some(Err(_)) => return PIE_STATUS_INVALID_ARGUMENT,
            None => {
                let per = crate::store::kv_cache::PerLayer {
                    head_dim: (0..n_layers)
                        .map(|i| {
                            if !is_gemma4 {
                                return head_dim as i32;
                            }
                            let full = model
                                .hf
                                .layer_types
                                .get(i)
                                .is_some_and(|t| t == "full_attention");
                            if full {
                                model.hf.gemma4_global_head_dim.max(model.hf.head_dim)
                            } else {
                                model.hf.head_dim
                            }
                        })
                        .collect(),
                    // gemma-4's tail attends through the last earlier layer
                    // of its own kind; every other family owns its pages.
                    kv_source_layer: (0..n_layers)
                        .map(|i| {
                            if !is_gemma4 || i < first_shared {
                                return i as i32;
                            }
                            let mine = model
                                .hf
                                .layer_types
                                .get(i)
                                .is_some_and(|t| t == "full_attention");
                            (0..first_shared)
                                .rev()
                                .find(|&j| {
                                    model
                                        .hf
                                        .layer_types
                                        .get(j)
                                        .is_some_and(|t| t == "full_attention")
                                        == mine
                                })
                                .unwrap_or(i) as i32
                        })
                        .collect(),
                    num_kv_heads: vec![kv_heads; n_layers],
                };
                if per.check_sharing().is_err() {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                let f = state.kv_format;
                match crate::store::kv_cache::KvCacheLayout::plan_per_layer(
                    n_layers as i32,
                    target as i32,
                    page_size as i32,
                    kv_heads,
                    per,
                    f,
                    false,
                ) {
                    Ok(l) => l,
                    Err(_) => return PIE_STATUS_INVALID_ARGUMENT,
                }
            }
        };

        let mut ops = crate::store::kv_cache_live::LiveKvCacheOps::new(
            stream.as_ref().as_raw().cast(),
            &alloc,
        );
        let Ok(cache) = crate::store::kv_cache_live::KvCache::materialize(layout, &mut ops) else {
            return PIE_STATUS_EXHAUSTED;
        };
        let mut held = ops.into_held();
        for b in &mut held {
            if b.memset(0, stream.as_ref()).is_err() {
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        let fresh = KvState { cache, _held: held, num_pages: target };

        // Carry over what still fits. A layer that owns no pages has none
        // to carry, and a shrink keeps the low pages.
        if let Some(old_kv) = &old {
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            for i in 0..fresh.layers() {
                let (Some((nk, nv)), Some((ok_, ov)), Some(pb)) =
                    (fresh.owned(i), old_kv.owned(i), fresh.page_bytes(i))
                else {
                    continue;
                };
                let keep = old_kv.num_pages.min(target) as usize * pb;
                for (dst, src) in [(nk, ok_), (nv, ov)] {
                    let code = unsafe {
                        cudaMemcpyAsync(
                            dst,
                            src.cast_const(),
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
        }
        if stream.as_ref().synchronize().is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        state.kv = Some(fresh);
        // A RESIZE MOVES THE KV PAGES, and a captured graph baked their old
        // addresses into every attention launch. Bumping the epoch is what tells
        // `SupergraphCache` to recapture instead of replaying into memory the
        // pool no longer owns — which showed up as a segfault the moment decode
        // fires became capturable at all.
        state.fire_arrays.epoch += 1;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if let Some(notify) = state.notify {
            unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
        }
        PIE_STATUS_OK
    })
}

/// Close an instance — idempotently, the C++'s reading: closing what is
/// not open is not an error.
pub fn pie_cuda_close_instance(driver: *mut PieDriver, instance_id: u64) -> i32 {
    guard("pie_cuda_close_instance", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        state.instances.remove(&instance_id);
        // The rings go with it. They are the instance's, and holding them
        // past its close is a device allocation nothing can reach — the
        // channel ids that named it are gone with the entry above.
        state.ptir_sessions.remove(&instance_id);
        PIE_STATUS_OK
    })
}

/// Close a channel — idempotently, freeing its pinned endpoint.
pub fn pie_cuda_close_channel(driver: *mut PieDriver, channel_id: u64) -> i32 {
    guard("pie_cuda_close_channel", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        if let Some(ch) = state.channels.remove(&channel_id) {
            // UNREGISTERED NOW, FREED LATER. See `InFlight::closed_channels`:
            // a queued fire's debt holds a copy of this state and will publish
            // into it from a stream callback, so the mirror cannot go back to
            // the allocator until every fire that could name it has retired.
            //
            // No fire in flight means nothing can be holding it, and the free
            // happens here as it always did.
            match state.in_flight.back_mut() {
                Some(last) => last.closed_channels.push(ch),
                None => ch.free(),
            }
        }
        PIE_STATUS_OK
    })
}

#[cfg(test)]
mod tests {
    use super::{ChannelState, InstanceEntry, instance_ring_shapes};

    /// A registered channel becomes the ring shape the device wants, and
    /// the bool case is the one that could not be derived any other way.
    ///
    /// `cell_bytes` is WIRE bytes, and for bools that is
    /// `numel.div_ceil(8)` — so one lane and eight lanes are both one
    /// byte, and a device ring sized from the width would be eight times
    /// too small for the eight-lane case. That is why `numel` and `dtype`
    /// are stored rather than recomputed, and this is the test that says
    /// so out loud.
    #[test]
    fn a_bool_channels_ring_shape_cannot_be_derived_from_its_wire_width() {
        use driver::tensor_ir::DType;

        let chan = |numel: usize, dtype: DType, capacity: u32| ChannelState {
            mirror: std::ptr::null_mut(),
            words: std::ptr::null_mut(),
            mirror_bytes: 0,
            cell_bytes: crate::ptir::bridge::wire_cell_bytes(dtype, numel),
            ring: capacity + 1,
            host_role: 0,
            numel,
            dtype,
        };

        let one = chan(1, DType::Bool, 1);
        let eight = chan(8, DType::Bool, 1);
        assert_eq!(one.cell_bytes, eight.cell_bytes, "one wire byte either way");
        assert_eq!(one.shape().numel, 1);
        assert_eq!(eight.shape().numel, 8, "the shape knows what the width cannot");
        assert_eq!(eight.shape().cell_bytes(), 8, "native is a byte per lane");

        // And the capacity survives the `ring = capacity + 1` round trip,
        // which is the one place the two vocabularies disagree.
        assert_eq!(chan(4, DType::F32, 7).shape().capacity, 7);
    }

    /// An instance's ring shapes come back in the order the program
    /// indexes them, and a channel it names but this shell lacks is a
    /// refusal rather than a shorter list.
    ///
    /// Skipping would RENUMBER every channel after the gap, so a program
    /// that meant `chan 1` would read channel 2 — a wrong answer that
    /// looks like a working fire, which is the class this whole driver
    /// keeps choosing to refuse instead.
    #[test]
    fn a_channel_an_instance_names_and_this_shell_lacks_refuses_the_whole_list() {
        use driver::tensor_ir::DType;

        let mut channels = std::collections::BTreeMap::new();
        for (id, numel) in [(10u64, 4usize), (20, 9)] {
            channels.insert(id, ChannelState {
                mirror: std::ptr::null_mut(),
                words: std::ptr::null_mut(),
                mirror_bytes: 0,
                cell_bytes: numel * 4,
                ring: 2,
                host_role: 0,
                numel,
                dtype: DType::F32,
            });
        }
        let inst = |ids: Vec<u64>| InstanceEntry {
            program_id: 1,
            geometry_class: 0,
            channel_ids: ids,
        };

        // The list is the INSTANCE's order, not the map's.
        let shapes = instance_ring_shapes(&inst(vec![20, 10]), &channels).expect("all present");
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].numel, 9, "channel 20 is index 0 because it is listed first");
        assert_eq!(shapes[1].numel, 4);

        assert!(
            instance_ring_shapes(&inst(vec![10, 99]), &channels).is_none(),
            "an unknown channel refuses rather than shortening the list"
        );
    }
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
