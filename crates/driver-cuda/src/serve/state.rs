//! The shell's state, and the device-lifetime
//! things hung off it.
//!
//! A leaf. Nothing here calls into `load`, `launch`, `encode` or `transfer`
//! — they all call in, which is the direction that makes this the file to
//! read first. The types are the driver's nouns; the verbs are next door.

use crate::fire::scratch::Scratch;
use driver_api::completion::{CompletionBroker, CompletionTarget};
use driver_api::local::PIE_STATUS_DRIVER_ERROR;

/// The shell's state.
///
/// It was what a `*mut PieDriver` pointed at, reached through a
/// `cast::<Shell>().as_mut()` on every one of thirteen entry points. The
/// entry points are methods now, so the receiver IS this.
pub struct Shell {
    // `caps: Vec<u8>` STOOD HERE, and its own doc had already retired it:
    // *"It was owned here so the `{ptr, len}` in a `PieDriverCaps`
    // out-parameter would outlive the `create` call. Nothing hands out a
    // pointer to it now."* An owning field exists to outlive something, and
    // once nothing borrows from it there is no third reading — it was written
    // once from `CAPS_JSON` and read never. `LoadedModel::load_caps` is the
    // one that is still live, and it is live because `serve::mod` parses it.
    /// What this device is, parsed once at create.
    ///
    /// The engine used to parse the capabilities JSON itself, from a
    /// `{ptr, len}` it was handed back. One parse, on the side that authored
    /// the JSON — which is why `CAPS_JSON` no longer needs an owner here.
    pub(crate) facts: driver_api::DeviceFacts,
    /// `[model] config` from the boot TOML, for HF snapshots whose
    /// config does not ride inside the checkpoint.
    ///
    /// One field is read out of it — the declared quantization — which is
    /// the one thing a catalog row cannot state, because the same model
    /// is published at four bits and at eight.
    pub(crate) boot_config: Option<std::path::PathBuf>,
    /// `[model] id` from the boot TOML: the operator's answer to "which
    /// model is this", when they have one.
    ///
    /// `None` is the ordinary case and means "read the tensors". It is
    /// an OVERRIDE and not a replacement for the check — a named row's
    /// manifest is still matched, so this cannot be used to load a
    /// checkpoint as something it is not.
    pub(crate) boot_model_id: Option<String>,
    /// Does this driver hand its completions to a stream callback and
    /// return with the fire still queued? Copied from
    /// [`boot::Boot::runahead`](crate::boot::Boot::runahead), which is where
    /// the reasoning for the default lives.
    pub(crate) runahead: bool,
    /// THE parse of this driver's knobs. See [`crate::boot`] for why one
    /// struct rather than ten `env::var` reads scattered across six
    /// modules — three of which disagreed about how to spell "false".
    pub(crate) boot: crate::boot::Boot,
    /// How the KV pages are stored — `[driver] kv_cache_dtype`.
    ///
    /// `store/kv_format.rs` has had nine spellings since the port, the
    /// layout plans their scale planes and `kv_paged.cu` switches on the
    /// scheme to write them. What was missing was any way to ASK for one:
    /// the shell built its pages by hand and could only build bf16.
    pub(crate) kv_format: crate::layout::KvCacheFormat,
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
    pub(crate) lowerings: std::collections::BTreeMap<LoweringKey, LoweredFire>,
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
    pub(crate) cublas:
        Option<crate::device::cublas::CublasHandle<cudarc::cublas::sys::cublasHandle_t>>,
    /// The fire's predicate word, allocated once.
    ///
    /// PERSISTENT for two reasons, and the second is correctness. It used to
    /// be built and dropped inside every `capture_or_replay`, which is a
    /// `cudaMalloc` and a `cudaFree` per fire — and `cudaFree` SYNCHRONIZES
    /// THE DEVICE, so the run-ahead the rest of this file is built around was
    /// being undone by the graph path. It is also the address a captured
    /// graph BAKES: a word that is freed and reallocated between two replays
    /// is one whose address the exec has no reason to still be right about.
    pub(crate) preds: Option<crate::device::PredicateWord>,
    /// The fire's peel-window word, allocated once. Same reasoning as
    /// [`Shell::preds`]: it was a `cudaMalloc` and a `cudaFree` per fire, and
    /// `cudaFree` synchronizes the device.
    pub(crate) peel_win: Option<crate::device::PeelWindowWord>,
    /// The pinned host buffer the logits D2H lands in, grown to fit and
    /// reused. The shell's rather than the fire's because a stream
    /// callback may not free it — see `FireDebt::staging`.
    pub(crate) logits_staging: Option<crate::device::PinnedBuf>,
    /// STAGING BUFFERS A WIDER FIRE REPLACED, held until nothing is in
    /// flight.
    ///
    /// `PinnedBuf::drop` calls `cudaFreeHost` on the calling thread, and
    /// it is NOT stream-ordered. Reusing one buffer across fires is safe
    /// because everything rides one stream; REPLACING it is not — with
    /// `RUNAHEAD_DEPTH = 2`, fire N's D2H is still queued into that exact
    /// buffer when fire N+1 widens it, and fire N's `FireDebt::staging`
    /// holds a `(ptr, len)` that `retire_fire` reads from a CUDA callback
    /// thread.
    ///
    /// `FireDebt::staging`'s own doc says the debt "BORROWS that buffer;
    /// it does not own it… The buffer is the shell's, reused every fire".
    /// Reuse holds. Replacement is what it does not cover.
    ///
    /// Bounded without any policy: `buf.len()` is the pooled logits value's
    /// length, which only ever grows, so this collects one entry per
    /// widening and then stops.
    pub(crate) retired_staging: Vec<crate::device::PinnedBuf>,
    /// Is this boot MEASURING rather than serving? `[driver]
    /// calibrate_planner`.
    ///
    /// Set, `load_model` ends by sweeping the ladder below the shape this
    /// driver ADVERTISES and writing the fastest point to the profile cache
    /// the next boot reads (`serve::load::calibrate_planner`).
    ///
    /// It stays BELOW the advertisement rather than above it. The planner
    /// can build a larger arena than the shell can fire — the attention
    /// workspace is a fixed allocation the lattice does not model — so the
    /// ceiling that means anything is the one `capabilities_json` published.
    pub(crate) calibrating: bool,
    /// The CUDA device ordinal this driver binds, from `[driver] device`.
    ///
    /// It was hardwired to 0 in both places that bind, which is wrong on any
    /// box with more than one GPU: an operator who asks for device 1 gets
    /// device 0 and no diagnostic.
    ///
    /// **A tensor-parallel group needs every rank's ordinal, and it does not
    /// get them from here.** `CustomAllReduce::Config::group_devices` is
    /// indexed by rank and `enable_peer_access` reads real ORDINALS out of
    /// it, so `super::load::build_tp_plane` ALL-GATHERS this field rather
    /// than reading a second config key: what crosses is the ordinal each
    /// rank actually bound. That is one field to keep true instead of two,
    /// and it makes the group's map impossible to fill with rank indices —
    /// a mistake that works on every single-group box and corrupts the
    /// second group on a four-GPU one.
    pub(crate) device_ordinal: i32,
    /// This driver's place in its tensor-parallel group, from
    /// `[driver] tp_rank` / `tp_size`. One rank, rank zero, unless told
    /// otherwise — and the two numbers travel together into both the load
    /// plan and the KV geometry, so a rank cannot be one width for its
    /// weights and another for its cache.
    pub(crate) tp_rank: u32,
    pub(crate) tp_size: u32,
    /// The loaded model, once `load_model` succeeds.
    pub(crate) model: Option<LoadedModel>,
    /// WHICH LOAD this is, counting from one.
    ///
    /// The identity every cache keyed on a "model" actually needs.
    /// `LoweringKey::model_id` and `BucketKey::model` were both the LAYER
    /// COUNT — so a shell that loaded a second 32-layer checkpoint would
    /// hit the first one's lowering and replay the first one's captured
    /// graph, whose baked addresses point into an arena the first
    /// `LoadedModel` freed on being replaced.
    ///
    /// A counter rather than a path hash because the question is not
    /// "which checkpoint" but "which residency": reloading the SAME
    /// checkpoint frees and reallocates just as thoroughly, so the same
    /// path must not answer the same id.
    pub(crate) load_generation: u64,
    /// Registered programs by id — the C3 hash is the dedup key, so
    /// re-registering a program answers the id it already has.
    pub(crate) programs: std::collections::BTreeMap<u64, ProgramEntry>,
    /// Bound instances by id.
    pub(crate) instances: std::collections::BTreeMap<u64, InstanceEntry>,
    /// The next never-used id (programs and instances share the counter —
    /// simpler, and nothing in the ABI wants them dense).
    pub(crate) next_id: u64,
    /// Who to tell when work finishes, from `create`.
    ///
    /// It was a `{notify: PieRuntimeNotifyFn, ctx: *mut c_void}` pair — a C
    /// function pointer and an erased context — because the engine on the
    /// other side had to be reachable from a C++ shell. The engine is Rust
    /// and so is this, so the pair is the handle it was erasing: a
    /// `CompletionBroker` is `Clone + Send`, and a stream callback publishes
    /// through it by name.
    pub(crate) broker: CompletionBroker,
    /// The hybrid's GDN state slabs, allocated on first hybrid launch.
    pub(crate) gdn: Option<GdnState>,
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
    pub(crate) supergraph: crate::fire::recordings::Recordings,
    /// The per-fire device arrays, pooled so a capture can outlive the
    /// fire that recorded it. See [`Scratch`]. Dropped AFTER the execs
    /// that address it — see above.
    pub(crate) fire_arrays: Scratch,
    /// This rank's custom P2P all-reduce plane, when the group has one.
    ///
    /// A device-lifetime handle, held here for [`Self::cublas`]'s reason and
    /// with [`Self::preds`]' one on top: the plane's signal slab, its
    /// `RankData` table and its fusion buffers are `cudaMalloc`'d ONCE and
    /// their addresses are what every peer's IPC mapping points at, so a
    /// per-fire plane would be a peer exchange per fire and a set of
    /// addresses no rank could agree on twice.
    ///
    /// **Declared AFTER [`Self::supergraph`], for that field's reason read
    /// the other way.** A captured graph bakes the addresses its launches
    /// used, and a fired all-reduce's are these; freeing them while an exec
    /// that recorded them is still alive is the same fault, so the execs go
    /// first.
    ///
    /// `None` on a single-rank driver, and that used to be every driver this
    /// build served: `super::load::tp_serving_refusal` refused `tp_size > 1`
    /// before `build_tp_plane` was reached, because
    /// `kernels_cuda::comm::CAN_LAUNCH` was `false`. Both all-reduce headers
    /// are internalised now and that constant is `true`, so a group with a
    /// key and a world size in `{2, 4, 6, 8}` reaches `build_tp_plane` and
    /// this field is `Some`. It is `Option` rather than absent because a rank
    /// of a group is the case the type exists for, not an error case.
    ///
    /// # This field is what makes `Shell` `!Send`
    ///
    /// `crate::fire::all_reduce::CustomAllReduce` holds raw device pointers
    /// and peer mappings that belong to the thread that created them, and
    /// `ResidentPlane` publishes its address into a THREAD-LOCAL for the bind
    /// arms to find. A `Shell` that could move threads would leave that
    /// publication behind on the old one. Nothing moves a `Shell` between
    /// threads today — `create` binds the device on the thread that fires —
    /// and this is what stops that from silently changing.
    pub(crate) all_reduce: Option<crate::fire::all_reduce::ResidentPlane>,
    /// The key this rank's tensor-parallel group rendezvouses on —
    /// `[driver] tp_group_id`, empty when unstated.
    ///
    /// `layout::rendezvous` needs a string to find the other ranks by, and
    /// this driver had none: `capabilities_json` passed
    /// `nccl_unique_id_hex: String::new()` to the memory planner, which is
    /// exactly the value `tp_min_plan` treats as *"no group to reconcile
    /// with"* — so the cross-rank plan agreement was wired up and inert.
    /// One key serves both rendezvous.
    pub(crate) tp_group_id: String,
    /// The driver-owned KV pools, allocated on first launch and grown on
    /// demand — decode continuity across launches lives here.
    pub(crate) kv: Option<KvState>,
    /// Registered channels: the pinned host ring endpoints the engine
    /// maps. Device-side rings and fire delivery ride with the launch
    /// integration.
    pub(crate) channels: std::collections::BTreeMap<u64, ChannelState>,
    /// The host-pinned KV swap pool: page-granular, per layer, both
    /// planes — where `copy_kv`'s host-pinned domain lands. Grown on
    /// demand by highest page id touched.
    pub(crate) swap: Option<SwapPool>,
    /// The adapter staging's bump arena, driver-lifetime.
    ///
    /// Per FIRE in what it holds and per DRIVER in what it owns: it is
    /// reset each fire and grown when a batch's adapters need more than
    /// the last one did, because retiring a block an in-flight fire may
    /// still read is the one thing a bump arena must not do.
    pub(crate) lora_arena: crate::fire::lora::LoraStageArena,
    /// The fire scratch held PER DRIVER, as the C++ holds it: the
    /// attention workspace and both FlashInfer plan caches. Created on
    /// first launch. This is also what the 711-fire soak enforced: the
    /// per-fire version leaked its 48 MB workspace every fire.
    pub(crate) scratch: Option<FireScratch>,
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
    pub(crate) fire_stream: Option<crate::device::OwnedStream>,
    /// The fire that is still running, if any — see [`InFlight`]. One
    /// slot, so the driver runs exactly one fire ahead.
    pub(crate) in_flight: std::collections::VecDeque<InFlight>,
    /// The allocator every fire's transient device memory comes from.
    /// Held for the pool, and dropped with the shell.
    pub(crate) fire_alloc: Option<crate::device::Allocator>,
    /// The PTIR plane: what a registered program adopted to, and what its
    /// generated regions compiled to.
    ///
    /// Two fields rather than one because they have different lifetimes.
    /// [`crate::program::Runtime`] is the CACHE — it outlives any one program
    /// and is what makes the second registration of a shared stage free —
    /// while `ptir_programs` is this shell's OWNERSHIP of the compiled
    /// modules, so closing the last user of a program can drop its
    /// `CUmodule`s at a point the shell chose.
    pub(crate) ptir: crate::program::Runtime,
    /// The compiled form of each registered program, by program id.
    pub(crate) ptir_programs: crate::program::Programs,
    /// The control kernels, compiled once for this device's architecture.
    ///
    /// Not per instance and not per program: `readiness` and `commit`
    /// depend on the architecture and nothing else, so a second instance
    /// finds them built. Built lazily because a driver that never fires a
    /// program should not pay NVRTC for two kernels it will not call.
    pub(crate) ptir_control: Option<crate::program::Control>,
    /// Every registered channel's device ring, by driver-wide SLOT.
    ///
    /// One registry rather than one per instance, which is the correction
    /// [`ChannelState::is_extern`] describes the cost of: a channel two
    /// instances name is ONE ring, so the exporter's publish is what the
    /// importer reads. Built lazily, because it needs the fire allocator and
    /// a driver that never fires a program should not allocate for one.
    pub(crate) ptir_rings: Option<crate::program::channel::Rings>,
    /// Which slot each registered channel's ring lives at, by channel id.
    ///
    /// Assigned on first use rather than at registration, because
    /// `register_channel` has no allocator: the shell's device state is
    /// readied by the first fire.
    pub(crate) ptir_channel_slots: std::collections::BTreeMap<u64, u32>,
    /// One instance's dense channel index → registry slot, by instance id.
    ///
    /// Per INSTANCE by necessity: a program names a channel by index into
    /// its own `channel_ids`, and that numbering is the instance's. It must
    /// outlive a fire — rebuilding renumbers, and a slot is what ties a
    /// program's channel to the cursors that record what a previous fire
    /// published.
    pub(crate) ptir_sessions: std::collections::BTreeMap<u64, crate::program::session::Session>,
    /// The adopted plans, by program id. Separate from the compiled
    /// modules because a program can be adopted and REJECTED — an
    /// unexecutable plan is still a plan, and the reason it was rejected
    /// is what the launch that needs it has to report — while a
    /// compilation only exists for a program that got that far.
    pub(crate) ptir_plans: std::collections::BTreeMap<u64, driver::ExecPlan>,
}

/// Driver-lifetime fire scratch.
pub(crate) struct FireScratch {
    pub ws: crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
    /// The PREFILL plan's own workspace, and it has to be its own.
    ///
    /// A FlashInfer plan writes its schedule into the workspace it was
    /// raised against, so two plans sharing one workspace is one plan
    /// clobbering the other — which is why the C++ carried the axiom "no
    /// mutable plan sharing across fire classes" and why the shell used to
    /// raise exactly the plan this fire's text named.
    ///
    /// `.wiki/driver/graph.md` §5 ① wants EVERY plan the geometry permits
    /// raised, so that a union capture can walk an arm this fire does not
    /// take. That is a memory cost, not a correctness one — as long as the
    /// plans stop sharing storage. This is that separation.
    pub prefill_ws:
        crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
    pub decode_plan: crate::bind::DecodePlan,
    /// gemma-4's SECOND decode plan — the FULL layers' 512-wide
    /// geometry; single-kind families never plan it.
    pub decode_plan_full: crate::bind::DecodePlan,
    pub prefill_plan: crate::bind::PrefillPlan,
    /// A peel TAIL's decode plan, and its own workspace.
    ///
    /// Its own for the reason `prefill_ws` is its own: a FlashInfer plan
    /// writes its schedule into the workspace it was raised against, so a
    /// tail planned into the fire's would clobber the plan the PREFIX is
    /// about to use — and a peel launches both regions.
    ///
    /// `Launch::peel`'s doc says a prepared plan "is found by the
    /// rectangle's ROW COUNT". A tail serves `[split, N)`, which is a
    /// different request count and therefore a different schedule.
    pub tail_plan: crate::bind::DecodePlan,
    pub tail_ws:
        crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
}

/// The pinned swap pool: `layers × [pages × page_bytes]` per plane.
pub(crate) struct SwapPool {
    /// One pinned region per `(layer, buffer)`, in `plan.buffers()` order.
    ///
    /// PER BUFFER and not per layer, which is the whole reason this stopped
    /// being a hand-rolled `[k | v]` block: a quantized cache carries FOUR
    /// buffers per layer (`k`, `v`, `k_scale`, `v_scale`) and gemma-4's
    /// layers disagree on head dim, so neither the count nor the width is
    /// a constant. `pools::swap_pool::SwapPool` says what to allocate;
    /// this holds what was allocated.
    pub regions: Vec<*mut std::ffi::c_void>,
    /// The plan those regions were allocated against — the geometry a
    /// `SwapPlan` is built from, and the record of what this pool can
    /// serve.
    pub plan: crate::pools::swap_pool::SwapPoolLayout,
    /// The two stream roles the plan asked for. Kept for the driver's
    /// life: the C++ creates them once, and an eviction queued behind a
    /// restore is the stall the second stream exists to avoid.
    pub evict: Option<crate::device::OwnedStream>,
    pub restore: Option<crate::device::OwnedStream>,
}

impl SwapPool {
    pub(crate) fn free(&self) {
        use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        for &b in &self.regions {
            ops.free_host(b);
        }
    }

    /// The host base of one `(layer, buffer)` region.
    pub(crate) fn region(&self, layer: u32, buffer: u32) -> Option<*mut u8> {
        let i = self
            .plan
            .buffers()
            .iter()
            .position(|b| b.layer == layer && b.buffer == buffer)?;
        Some(self.regions.get(i)?.cast::<u8>())
    }
}

/// One channel's host endpoint: the pinned mirror and the four control
/// words, exactly the C++ registry's binding contract.
#[derive(Clone, Copy)]
pub(crate) struct ChannelState {
    pub mirror: *mut std::ffi::c_void,
    pub words: *mut std::ffi::c_void,
    pub mirror_bytes: usize,
    /// WIRE bytes per cell — bit-packed for bools.
    pub cell_bytes: usize,
    /// `capacity + 1` — the ring modulus.
    pub ring: u32,
    pub host_role: u8,
    /// Lanes in one cell, and the cell's element type.
    ///
    /// Kept because `cell_bytes` cannot be inverted: a bool cell is
    /// `numel.div_ceil(8)` wire bytes, so eight lanes and one lane are
    /// both one byte. The DEVICE ring is sized in NATIVE bytes
    /// (`program::channel::ChannelShape`), and deriving one from the other needs
    /// the shape rather than the width — which is why a driver that only
    /// ever wrote to the host mirror could get away without these and one
    /// that fires a program cannot.
    pub numel: usize,
    pub dtype: driver::tensor_ir::DType,
    /// `PIE_CHANNEL_EXTERN_*`: is this channel private to one instance,
    /// or does it cross between programs?
    ///
    /// Recorded rather than acted on, and `bind_instance` refuses an
    /// instance that names one. See [`ChannelState::is_extern`].
    pub extern_dir: u8,
}

impl ChannelState {
    /// Does this channel cross a program boundary?
    ///
    /// An extern channel is one two DIFFERENT programs share — one
    /// exports, the other imports — and sharing it means sharing ONE
    /// ring: the same cells and, just as much, the same head/tail
    /// cursors, because the cursors are how a producer tells a consumer
    /// that a cell is full.
    ///
    /// THE RING IS SHARED NOW, and this paragraph used to say it was not.
    ///
    /// It read: "this driver allocates a `Rings` PER SESSION and a session
    /// per instance, so two instances naming one extern channel get two
    /// rings", and it went on to describe the fix — a driver-owned registry
    /// keyed by channel id, with all five arrays per channel. That registry
    /// exists ([`crate::program::channel::Rings`]): a channel is registered
    /// once, at one slot, and every instance that names it holds the same
    /// slot. So two programs sharing a channel now share its cells AND its
    /// cursors, which is what an extern channel needs.
    ///
    /// `bind_instance` still refuses one, and the refusal is now about what
    /// has NOT been done rather than about what cannot be: nothing in this
    /// driver reads `PIE_CHANNEL_EXTERN_IMPORT`/`EXPORT` to decide who may
    /// publish and who may consume, no test in the tree binds one, and the
    /// engine's own path for them is untried here. Serving one would be a
    /// guess about a direction nobody has checked. The mechanical blocker is
    /// gone; the unmeasured half is not.
    pub const fn is_extern(&self) -> bool {
        self.extern_dir != driver_api::local::PIE_CHANNEL_EXTERN_NONE
    }

    /// This channel as the device rings want it.
    pub(crate) fn shape(&self) -> crate::program::channel::ChannelShape {
        crate::program::channel::ChannelShape {
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
pub(crate) struct InFlight {
    pub done: crate::device::Event,
    /// Ordinary scratch. Named for what it is rather than listed, because
    /// the point is that nothing here is read again — it is held only so
    /// that dropping it does not synchronize at the wrong moment.
    ///
    /// `expect` rather than `allow`: never-read is this field's JOB, and
    /// the paragraph above is why, so the lint is right about the fact and
    /// wrong about what to do. `expect` also turns the day someone gives
    /// it a reader into a build error, which is the day this reasoning
    /// needs revisiting rather than the day it silently stops applying.
    #[expect(dead_code, reason = "owned to defer the drop; see the type's doc")]
    pub(crate) scratch: Vec<crate::device::DeviceBuffer>,
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
    pub closed_channels: Vec<ChannelState>,
}

/// Give back what a retired fire was holding.
///
/// The scratch drops on its own; the channels do not, because a
/// `ChannelState` is a pair of raw host allocations this shell owns rather
/// than a Rust value with a destructor. Freeing them HERE rather than in
/// `close_channel` is the whole point -- see `InFlight::closed_channels`.
pub(crate) fn retire(fire: InFlight) {
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
pub(crate) const RUNAHEAD_DEPTH: usize = 2;

/// What a lowering can depend on: see [`Shell::lowerings`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct LoweringKey {
    pub model_id: u64,
    pub class: model_ir::trace::FireClass,
    pub rows: u32,
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
    pub rows_digest: u64,
    pub union_asked: bool,
}

/// FNV-1a over the rows' axes.
///
/// Not a hash of the struct: `Row` is not `Hash`, and adding the derive
/// would make every future field silently part of a cache key. Naming
/// the axes here means a new one has to be added deliberately.
pub(crate) fn digest_rows(rows: &[model_compiler::lower::Row]) -> u64 {
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
pub(crate) struct LoweredFire {
    pub plan: model_ir::trace::ForwardPlan,
    pub lowered: model_compiler::lower::Lowered,
    pub dplan: crate::bind::DispatchPlan,
    pub union: bool,
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
pub(crate) struct FireDebt {
    /// The bf16 logits, D2H'd into the SHELL's pinned staging by a
    /// stream-ordered copy, as (pointer, length).
    ///
    /// PINNED, and that is the whole reason the shell keeps one:
    /// `cudaMemcpyAsync` into pageable host memory blocks until the copy
    /// completes, so a `Vec` here drains the stream inside
    /// `pie_cuda_launch` and undoes the run-ahead.
    pub staging: Option<(*const u8, usize)>,
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
    pub readouts: Vec<(ChannelState, usize)>,
    pub vocab: usize,
    /// The terminal cells this frame publishes, and the completion the
    /// runtime is waiting on.
    pub cells: Vec<*mut driver_api::local::TerminalCell>,
    pub completion: CompletionTarget,
    pub(crate) broker: CompletionBroker,
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
pub(crate) unsafe extern "C" fn retire_fire(data: *mut std::ffi::c_void) {
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
                    cell[t * 4..t * 4 + 4].copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
                }
            }
            if !ch.publish(&cell) {
                eprintln!("[driver-cuda] launch: logits ring full; a request dropped its output");
            }
        }
    }

    // Then the terminal cells, then the notify — in that order, because the
    // runtime reads the cells the moment the notify lands.
    //
    // `publish` is a RELEASE STORE on the cell's own `AtomicU32`, which is
    // what pairs with the runtime's `load(Acquire)` on that same word. It
    // replaces a non-atomic `write_volatile` followed by a `fence(Release)`:
    // the write raced the reader's atomic load, and the fence was on the
    // wrong side of the store to give that store a release anyway.
    for &cell in &debt.cells {
        if !cell.is_null() {
            unsafe {
                (*cell).publish(driver_api::local::PIE_TERMINAL_OUTCOME_SUCCESS);
            }
        }
    }
    // Still fenced before the notify: the channel publishes above are a
    // different plane, and the runtime reads those on the notify too.
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    debt.broker
        .notify(debt.completion.wait_id, debt.completion.target_epoch);
}

impl ChannelState {
    /// Publish one wire cell: write it at `tail % ring`, then advance the
    /// tail word with release ordering — the reader (the engine) consumes
    /// from the head. The writer side of the C++ ring, host-resident.
    /// The host plane of this channel, as `program::channel` sees it.
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
    pub(crate) fn host_plane(&self) -> crate::program::channel::HostChannel {
        debug_assert!(self.cell_bytes * self.ring as usize <= self.mirror_bytes);
        unsafe {
            crate::program::channel::HostChannel::new(
                self.mirror,
                self.words,
                self.cell_bytes,
                self.ring,
                self.host_role,
            )
        }
    }

    pub(crate) fn publish(&self, cell: &[u8]) -> bool {
        self.host_plane().publish(cell)
    }
}

impl ChannelState {
    pub(crate) fn free(&self) {
        use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        ops.free_host(self.mirror);
        ops.free_host(self.words);
    }
}

/// The shell's KV: one (k, v) pool per layer, plus the capacity in
/// pages. A `None` row is a layer that owns no pages — gemma-4's
/// KV-shared trailing layers, whose views ride their source's pool.

/// The KV cache, and the buffers backing it.
///
/// The geometry is [`pools::kv_cache_live::KvCache`](crate::pools::kv_cache_live::KvCache)'s:
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
pub(crate) struct KvState {
    pub cache: crate::pools::kv_cache_live::KvCache<crate::pools::kv_cache_live::AllResident>,
    /// Backing store for `cache`; dropping this frees the pages.
    pub _held: Vec<crate::device::DeviceBuffer>,
    pub num_pages: u32,
}

impl KvState {
    /// The pages `layer` owns, or `None` if it reads through another's.
    pub(crate) fn owned(
        &self,
        layer: usize,
    ) -> Option<(*mut core::ffi::c_void, *mut core::ffi::c_void)> {
        let l = i32::try_from(layer).ok()?;
        let slot = self.cache.layout().slots().get(layer)?;
        if slot.is_alias() {
            return None;
        }
        Some((self.cache.k(l), self.cache.v(l)))
    }

    /// Bytes of one page at `layer` — its own stride, so the two-head-dim
    /// families move the right amount per layer.
    pub(crate) fn page_bytes(&self, layer: usize) -> Option<usize> {
        let slot = self.cache.layout().slots().get(layer)?;
        let k = slot.k.as_ref()?;
        Some(usize::try_from(k.nbytes()).unwrap_or(0) / self.num_pages.max(1) as usize)
    }

    /// How many layers the cache describes.
    pub(crate) fn layers(&self) -> usize {
        self.cache.layout().slots().len()
    }

    /// `layer`'s head dim.
    ///
    /// The config's single number is wrong for the families whose layers
    /// disagree, and this is the extent the copy will actually stride
    /// through.
    pub(crate) fn head_dim(&self, layer: usize) -> Option<i32> {
        let slot = self.cache.layout().slots().get(layer)?;
        slot.k.as_ref()?;
        Some(self.cache.layout().head_dim_at(i32::try_from(layer).ok()?))
    }

    /// What a kernel is handed for each layer.
    pub(crate) fn views(&self) -> Vec<crate::bind::abi::KvCacheLayerView> {
        (0..self.layers())
            .map(|l| self.cache.layer_view(i32::try_from(l).unwrap_or(0)))
            .collect()
    }

    // `uniform_stride` STOOD HERE. It asked whether every layer owns pages
    // and every page is one size, for a host swap path that had ONE stride
    // for the whole stack. That path asks per layer now — `serve::transfer`
    // takes `fresh.page_bytes(i)` inside the loop — so the question has no
    // asker, and a whole-stack answer would be the weaker one anyway: it can
    // only refuse a stack a per-layer walk would serve correctly.
}

/// The hybrid's driver-owned GDN state, addressed through the ported
/// [`RecurrentStateCache`](crate::pools::recurrent_state_cache).
///
/// It used to be one `(conv, recurrent)` allocation PER LAYER with the
/// reset, the growth and the slot copy all hand-written beside it -- the
/// same three jobs `pools::recurrent_state_cache` was ported to do, and
/// which it had no caller for. The port pools every linear layer into ONE
/// allocation per buffer kind and addresses with
/// `linear_index * max_slots * stride + slot * stride`, which is why
/// wiring it was an allocation change rather than a call substitution.
///
/// The fire path did not have to move: `GdnCtx::conv_state` is still a
/// per-model-layer `Vec<u64>` of bases, each one now an offset into the
/// pool rather than its own allocation.
///
/// Slot ids are the ENGINE's (`rs_slot_ids` on the step,
/// `StateCopyRange` on state copies); the shell only stores.
pub(crate) struct GdnState {
    /// The cache: the layout, the strides, and what to do to the buffers.
    pub cache: crate::pools::recurrent_state_cache::RecurrentStateCache,
    /// The two pooled allocations, in `Buffer` order. `mtp` is absent
    /// until the MTP pending-hidden row has a writer.
    pub conv: crate::device::DeviceBuffer,
    pub recurrent: crate::device::DeviceBuffer,
    /// Which MODEL layers are linear, so a caller can map a model layer to
    /// the pool's linear index.
    pub is_linear: Vec<bool>,
    pub num_slots: u32,
    pub conv_stride_elems: i64,
    pub state_stride_elems: i64,
    // `state_elem_bytes` STOOD HERE — "bytes per element of the recurrent
    // store (2 = bf16 state)". It was set from `shape.state_elem` at
    // construction and read by nothing: every consumer of this pool addresses
    // it in ELEMENTS, through `conv_stride_elems` and `state_stride_elems`
    // beside it, which is the whole reason those two carry the unit in their
    // names. A width nothing multiplies by is a width nobody agreed on.
}

impl GdnState {
    /// The device base of one MODEL layer's conv window, or 0 where the
    /// layer is full-attention.
    ///
    /// This is what `GdnCtx::conv_state` carries, and it is why the fire
    /// path is unchanged by the pooling: a per-layer base is still a
    /// per-layer base, it just comes out of one allocation now.
    pub(crate) fn conv_base(&self, layer: usize) -> u64 {
        self.base(layer, true)
    }

    /// The same for the recurrent state.
    pub(crate) fn recurrent_base(&self, layer: usize) -> u64 {
        self.base(layer, false)
    }

    fn base(&self, layer: usize, conv: bool) -> u64 {
        let l = match u32::try_from(layer) {
            Ok(l) => l,
            Err(_) => return 0,
        };
        let addr = if conv {
            self.cache.layout().conv_state(l, 0)
        } else {
            self.cache.layout().recurrent_state(l, 0)
        };
        let Some(addr) = addr else { return 0 };
        let pool = if conv { &self.conv } else { &self.recurrent };
        (pool.as_ptr() as u64).wrapping_add(addr.offset)
    }

    /// Run the ops a `RecurrentStateCache` routine asked for.
    ///
    /// The cache returns [`StateOp`]s rather than calling CUDA, which is
    /// what makes its reset and copy semantics checkable without a GPU --
    /// and this is the one place that turns them back into calls.
    pub(crate) fn apply(
        &self,
        ops: &[crate::pools::recurrent_state_cache::StateOp],
        stream: crate::device::StreamRef<'_>,
    ) -> Result<(), i32> {
        use crate::pools::recurrent_state_cache::{Buffer, StateOp};
        use cudarc::runtime::sys::{
            cudaError, cudaMemcpy2DAsync, cudaMemcpyAsync, cudaMemcpyKind, cudaMemset2DAsync,
            cudaMemsetAsync,
        };
        let base = |b: Buffer| -> Option<*mut u8> {
            match b {
                Buffer::Conv => Some(self.conv.as_ptr().cast::<u8>()),
                Buffer::Recurrent => Some(self.recurrent.as_ptr().cast::<u8>()),
                // No writer yet, so no allocation -- and an op against a
                // buffer that does not exist is skipped rather than
                // faulted, which is what `has_mtp_hidden` already gates.
                Buffer::MtpHidden => None,
            }
        };
        let at = |p: *mut u8, off: u64| unsafe { p.add(usize::try_from(off).unwrap_or(0)) };
        let n = |v: u64| usize::try_from(v).unwrap_or(0);
        for op in ops {
            let code = match *op {
                StateOp::Memset {
                    buffer,
                    offset,
                    len,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe { cudaMemsetAsync(at(p, offset).cast(), 0, n(len), stream.as_raw()) }
                }
                StateOp::Memset2D {
                    buffer,
                    offset,
                    pitch,
                    width,
                    rows,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemset2DAsync(
                            at(p, offset).cast(),
                            n(pitch),
                            0,
                            n(width),
                            n(rows),
                            stream.as_raw(),
                        )
                    }
                }
                StateOp::Memcpy {
                    buffer,
                    dst,
                    src,
                    len,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemcpyAsync(
                            at(p, dst).cast(),
                            at(p, src).cast_const().cast(),
                            n(len),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_raw(),
                        )
                    }
                }
                StateOp::Memcpy2D {
                    buffer,
                    dst,
                    src,
                    pitch,
                    width,
                    rows,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemcpy2DAsync(
                            at(p, dst).cast(),
                            n(pitch),
                            at(p, src).cast_const().cast(),
                            n(pitch),
                            n(width),
                            n(rows),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_raw(),
                        )
                    }
                }
                // `reset_slots_if_fresh` reads the fire's slot ids and a
                // freshness flag OFF THE DEVICE and zeroes what it finds,
                // which needs two device arrays this shell does not build:
                // it resets from the HOST instead, off `PIE_RS_FLAG_RESET`,
                // one `reset_slot` per marked row. Same effect, decided a
                // fire earlier. The op is skipped rather than approximated.
                StateOp::ZeroSlotsIfFresh { .. } => continue,
            };
            if code != cudaError::cudaSuccess {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
        }
        Ok(())
    }

    /// Grow to cover `need` slots, MIGRATING the surviving ones.
    ///
    /// The migration is per LINEAR LAYER and not one block, because the
    /// slot axis sits INSIDE the layer axis: growing `max_slots` restrides
    /// every layer, so a straight copy of the old pool would land layer
    /// `l`'s slots inside layer `l`'s new stride only for `l == 0`.
    ///
    /// Returns whether it GREW, which the caller must turn into a plan
    /// epoch bump: both pool bases move and a capture bakes them, so an
    /// exec recorded against the old ones would address freed memory.
    /// This was true of the per-layer allocations too and nothing said so.
    /// Grow the recurrent slabs to hold `need` slots, bumping `epoch` if
    /// they moved.
    ///
    /// THE EPOCH IS A PARAMETER, not the caller's afterthought. Both
    /// call sites used to write `if ensure_slots(..)? { *epoch += 1; }`,
    /// and a third that forgot would be a captured graph replaying
    /// against a freed slab — a wrong answer, not a fault. Same
    /// relocation `Scratch::grow` makes: growing is what bumps, so the
    /// bump cannot be forgotten by anyone who did not grow.
    pub(crate) fn ensure_slots(
        &mut self,
        need: u32,
        epoch: &mut crate::fire::recordings::PlanEpoch,
        alloc: &crate::device::Allocator,
        stream: &crate::device::OwnedStream,
    ) -> Result<bool, i32> {
        if self.num_slots >= need {
            return Ok(false);
        }
        let grown =
            crate::pools::recurrent_state_cache::RecurrentStateCache::allocate_bf16_recurrent(
                &self.is_linear,
                self.cache.conv_dim(),
                self.cache.conv_kernel(),
                self.cache.v_heads(),
                self.cache.head_k_dim(),
                self.cache.head_v_dim(),
                i32::try_from(need).unwrap_or(i32::MAX),
            );
        let (conv_n, rec_n) = (
            usize::try_from(grown.layout().conv_total_bytes())
                .unwrap_or(0)
                .max(1),
            usize::try_from(grown.layout().recurrent_total_bytes())
                .unwrap_or(0)
                .max(1),
        );
        let mut conv = alloc.alloc(conv_n)?;
        let mut recurrent = alloc.alloc(rec_n)?;
        conv.memset(0, stream.as_ref())?;
        recurrent.memset(0, stream.as_ref())?;
        {
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            let keep = self.num_slots;
            let copy = |dst: *mut u8, src: *const u8, bytes: u64| -> Result<(), i32> {
                if bytes == 0 {
                    return Ok(());
                }
                let code = unsafe {
                    cudaMemcpyAsync(
                        dst.cast(),
                        src.cast(),
                        usize::try_from(bytes).unwrap_or(0),
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream.as_ref().as_raw(),
                    )
                };
                (code == cudaError::cudaSuccess)
                    .then_some(())
                    .ok_or(PIE_STATUS_DRIVER_ERROR)
            };
            for l in 0..self.is_linear.len() {
                let lu = u32::try_from(l).unwrap_or(0);
                for conv_side in [true, false] {
                    let (old_a, new_a) = if conv_side {
                        (
                            self.cache.layout().conv_state(lu, 0),
                            grown.layout().conv_state(lu, 0),
                        )
                    } else {
                        (
                            self.cache.layout().recurrent_state(lu, 0),
                            grown.layout().recurrent_state(lu, 0),
                        )
                    };
                    let (Some(old_a), Some(new_a)) = (old_a, new_a) else {
                        continue;
                    };
                    let (old_pool, new_pool) = if conv_side {
                        (self.conv.as_ptr().cast::<u8>(), conv.as_ptr().cast::<u8>())
                    } else {
                        (
                            self.recurrent.as_ptr().cast::<u8>(),
                            recurrent.as_ptr().cast::<u8>(),
                        )
                    };
                    copy(
                        unsafe { new_pool.add(usize::try_from(new_a.offset).unwrap_or(0)) },
                        unsafe { old_pool.add(usize::try_from(old_a.offset).unwrap_or(0)) }
                            .cast_const(),
                        new_a.len * u64::from(keep),
                    )?;
                }
            }
        }
        stream.as_ref().synchronize()?;
        self.cache = grown;
        self.conv = conv;
        self.recurrent = recurrent;
        self.num_slots = need;
        // GREW, so every capture that recorded a slab base is stale.
        epoch.bump();
        Ok(true)
    }
}

/// Install a rebuilt KV pool, bumping the generation because the pages
/// moved.
///
/// TWO PATHS REBUILD THIS POOL — `kv_pools_for` when a fire needs more
/// pages than the last one did, and `pie_cuda_resize_pool` when the
/// engine asks — and both used to assign the field and then bump by
/// hand. A third that forgot would be a captured attention launch
/// replaying against pages the pool no longer owns, which showed up
/// once already as a segfault the moment decode fires became capturable.
///
/// One line, and it is a line the type system cannot enforce; what it
/// can do is make there be only one of it.
pub(crate) fn install_kv(
    kv: &mut Option<KvState>,
    epoch: &mut crate::fire::recordings::PlanEpoch,
    next: KvState,
) {
    *kv = Some(next);
    epoch.bump();
}

/// What registration keeps of a program today: the identity the engine
/// dedups on. The launch package itself is deep-copied when the `launch`
/// arm lands — it is the caller's transient memory, and copying an IR
/// nothing can execute yet would be bytes without a reader.
pub(crate) struct ProgramEntry {
    pub program_hash: u64,
    #[allow(dead_code)] // read when launch's compile cache lands
    pub emitter_version: u32,
}

/// A bound instance: which program, the geometry the binding echoed, and
/// the channels the instance attached.
pub(crate) struct InstanceEntry {
    #[allow(dead_code)] // read when launch resolves frames to instances
    pub program_id: u64,
    #[allow(dead_code)]
    pub geometry_class: u32,
    pub channel_ids: Vec<u64>,
    /// The value each seeded channel starts with, in WIRE form, by channel id.
    ///
    /// Held rather than applied at bind because a cell's home is a device ring
    /// and `bind_instance` has no allocator — the shell's device state is
    /// readied by the first fire. `launch::ensure_sessions` applies them when
    /// it registers the channel, which is the one moment the ring is known
    /// empty.
    ///
    /// They were DROPPED. `InstanceBindingPlan::seed_values` arrived and
    /// nothing read it, so a seeded channel held nothing in either plane: the
    /// shared `driver::registry` pushes each one into the ring at bind and this
    /// shell did not, which is why an epilogue that reads a seeded `rng` or a
    /// decode whose first `Positions` is a seed found an empty cell and the
    /// fire declined.
    pub seeds: Vec<(u64, Vec<u8>)>,
}

/// What a successful `load_model` leaves behind: the parsed config and
/// every weight resident on the device, keyed by BOTH its checkpoint name
/// and (for the llama-like family) the fused trace name the executor asks
/// by.
pub(crate) struct LoadedModel {
    /// The catalog row this checkpoint matched, by id.
    ///
    /// `&'static str` because it is borrowed from the `const` table, not
    /// parsed from anything. This is what leaves the driver in
    /// `DriverCapabilities::model_id` and reaches the host's chat
    /// template — the same row, named once, rather than a `model_type`
    /// string re-interpreted by a second table on the far side.
    pub id: &'static str,
    /// What this checkpoint IS, derived ONCE at load.
    ///
    /// It used to be a `Box<dyn PlannedFamily>` built from the
    /// admission of every fire — allocating and cloning per-layer
    /// `Vec`s each time, while the lowering it feeds is cached
    /// precisely because it costs 3.3 ms. The expensive answer was
    /// cached and its input was rederived.
    ///
    /// And it carries no family name, so the fire path cannot recover
    /// one. `let is_gemma4 = family.planless_prefill()` appeared three
    /// times in this shell; a `Deployment` has nothing to ask.
    pub deployment: model::deployment::Deployment,
    /// The caps JSON `load_model` answered with; owned like `Shell::caps`.
    pub load_caps: Vec<u8>,
    /// Every tensor the plan named, as a span of the arena. A SPAN and not
    /// an allocation: a resident plan lays the whole model out contiguously,
    /// so a weight is an offset into one buffer rather than one of a thousand
    /// `cudaMalloc`s.
    pub weights: std::collections::BTreeMap<String, crate::weights::stage::WeightSpan>,
    /// The arena, and anything the plan published outside it. Held so the
    /// spans above stay valid; never indexed.
    #[allow(dead_code)]
    pub owned: Vec<crate::device::DeviceBuffer>,
    /// Trace-name RENAMES onto checkpoint names (`layer.3.attn_norm` →
    /// `model.layers.3.input_layernorm.weight`); concats get buffers of
    /// their own in `weights`, renames get a row here — no second copy of
    /// a tensor that already sits on the device.
    pub aliases: std::collections::BTreeMap<String, String>,
    /// The per-layer `layer_scalar` [1] tensors a deployment names, read to
    /// host once at load (the C++ `read_bf16_scalar_once`) — the fused
    /// sandwich norm's whole-stream multiplier, carried into
    /// `DispatchCtx::scales` per fire.
    ///
    /// Empty for a deployment whose wiring names none, which is most of them.
    /// WHICH deployments name them is not asked here: `wiring.scalars` is the
    /// list, and this reads it.
    pub layer_scalars: Vec<f32>,
    /// The group this rank's weights were sharded for, carried from the
    /// shell so a family's facts and its load plan cannot disagree about
    /// how wide a rank is. A forward derivation reads it to decide whether
    /// its landing needs a collective.
    pub(crate) tp_size: u32,
}

impl LoadedModel {
    /// The device pointer for a name — the live half of the executor's
    /// `Resolver::weight`. Checkpoint names, fused names and aliases all
    /// answer. `launch` is its caller; until that arm lands it is only
    /// the load test's assertion surface.
    #[allow(dead_code)]
    pub(crate) fn weight(&self, name: &str) -> Option<*const std::ffi::c_void> {
        if let Some(b) = self.weights.get(name) {
            return Some(b.ptr.cast_const());
        }
        let target = self.aliases.get(name)?;
        self.weights.get(target).map(|b| b.ptr.cast_const())
    }
}

// `CAPS_JSON` STOOD HERE, and `serve::load` parsed it into a
// `driver_api::DeviceFacts`. It was
//
//     {"driver":"driver-cuda","status":"phase-d-shell","abi":24}
//
// which is the C-era status blob and not a `DeviceFacts` at all: none of its
// three keys is one of that struct's nine, and `DeviceFacts` is
// `#[serde(deny_unknown_fields)]`. **So the parse failed, `Shell::open`
// returned `PIE_STATUS_DRIVER_ERROR`, and this driver could not be created —
// unconditionally, on every path, since `2ef431d02`.** `tests/serve.rs`'s
// twenty-eight tests all fail on their first line for this reason, and so
// does `pie` itself: the engine reports `cuda rank 0 create failed with
// status -5`.
//
// The facts are STATED now, which is what the other three drivers already
// did. `driver-metal`'s says why in one sentence — *"stated from what this
// backend IS rather than parsed out of a config; a config that disagreed with
// the hardware would simply be believed"* — and that is the whole argument. A
// JSON literal cannot disagree with a struct at compile time; a struct
// literal cannot disagree with itself.

/// The facts a scheduler reads, stated from what this backend IS.
///
/// Four of the nine are not this file's to choose: they are the STORAGE
/// compiler's, and `model_loader::plan::StorageTarget::for_backend` states
/// them for `BackendKind::Cuda` already. Three are repeated with the reason
/// their author gave, because they are plain numbers a reader here needs to
/// see; the fourth is IMPORTED, because it is a mask and a transcribed mask
/// is a bug waiting for a fifth bit:
///
/// * `storage_alignment` 256 — what cuBLAS wants for a matrix operand and
///   what `cudaMalloc` itself guarantees, so a view into the arena is as
///   aligned as its own allocation would have been.
/// * `storage_max_tile_bytes` 64 MiB — how much host staging one load-time
///   transform may take at once.
/// * `storage_tile_map_mask` — `CAST | ENCODE | SCALE`, which is
///   `passes::tile::CUDA_TILE_MAP_MASK`. `REBLOCK` and `DECODE` are not
///   there: the first has no device kernel in this tree and the second reads
///   scales that live inside the payload, which no device kernel reads.
/// * `native_mxfp4_moe` false — **and the name is the trap.** It does not
///   mean "reads MXFP4"; it means "has a native MXFP4 *GEMM*", which in
///   gpt-oss's contract selects a Marlin REPACK of the expert banks, work
///   this tree did not port. A driver whose GEMM reads the stored banks
///   directly wants the other branch, which is this one.
///
/// The remaining five are this backend's own:
pub(crate) fn device_facts() -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: "cuda".to_string(),
        // FALSE, and it is the one that changes SCHEDULING. On a discrete
        // card the KV pool and the host do not share physical memory, so
        // "the device is full" is a question about the card alone —
        // the opposite of `driver-metal`, which answers `true` for exactly
        // that reason and says so.
        unified_memory: false,
        // TRUE, on the same rule `driver-metal` applies to reach `false`:
        // the table says which kernels exist. This one has
        // `quant::quantize_bf16_to_fp8_e4m3_per_channel` and
        // `quant::quantize_bf16_to_fp8_e4m3_per_token_group`, and `KvDType`
        // names `Fp8E4M3` and `Fp8E5M2` as page storage the paged-attention
        // kernels read. Metal has neither and answers `false`.
        fp8_native: true,
        native_mxfp4_moe: false,
        storage_alignment: 256,
        storage_max_tile_bytes: 64 * 1024 * 1024,
        // NOT transcribed. `CUDA_TILE_MAP_MASK` is
        // `TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_SCALE`, and those bits
        // are 1, 4 and 128 rather than the 1, 2, 4 a reader would guess — so
        // writing the number here would have been wrong on the first try and
        // silently wrong afterwards. `serve` is `feature = "abi"` and that
        // feature brings `model-loader`, so the constant is simply in scope.
        storage_tile_map_mask: model_loader::plan::passes::tile::CUDA_TILE_MAP_MASK,
        // The paged KV pool's rows per page, which every `kv_translation`
        // index is in units of. `boot::KV_PAGE_SIZE` is the same sixteen and
        // is not a preference: the paged-attention kernels are compiled for
        // it.
        page_size: crate::boot::KV_PAGE_SIZE.unsigned_abs(),
    }
}

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
pub(crate) fn instance_ring_shapes(
    instance: &InstanceEntry,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
) -> Option<Vec<crate::program::channel::ChannelShape>> {
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
pub(crate) fn channel_dtype(byte: u8) -> driver::tensor_ir::DType {
    driver::tensor_ir::DType::from_wire(byte).unwrap_or(driver::tensor_ir::DType::F32)
}

// `slice_of` STOOD HERE — a `{ptr, len}` pair as a `&[T]`, empty for null.
// It was the marshalling half of a C descriptor surface (`PieU32Slice`,
// `PieBytes` and the rest), and `50fa127a3` deleted the descriptors: the
// verbs take Rust values now, so there is no borrowed ABI slice to reconstruct
// and no unsafe block to justify.
