//! The shell's nouns: everything this driver holds between calls, and the
//! two verbs that create and describe it.
//!
//! A leaf. `load`, `launch`, `register` and `control` all call in here and
//! nothing here calls them, which is the same partition `driver-cuda`'s
//! `serve/state.rs` has for the same reason: the state is nobody's subject,
//! so it can be edited without meeting the modules that use it.
//!
//! # Why this is in the driver and not in the engine
//!
//! It was in the engine — `crates/engine/src/driver/backend/metal.rs`, which
//! assembled the seven-field `Machine` by hand, computed KV write pages,
//! staged nine fire tables and built the KV page resolver, against 66 names
//! re-exported flat from `lib.rs`. All of that is the driver's work, done on
//! the other side of the crate boundary, where every internal change to it
//! was a breaking change.
//!
//! `driver-cuda` never had the problem because the engine reaches it through
//! a C ABI: its shell holds the same nouns this one does (a model, a KV pool,
//! a scratch, recordings, a registry) and they live in the driver because
//! `PieDriver` is opaque. **CUDA has the right side of that boundary and the
//! wrong shape for it; this crate had the right shape and the wrong side.**
//!
//! What is taken from `serve/` is its decomposition and not its FFI: the C
//! ABI was deleted for Metal deliberately (`2cc4e5e4d`), so the facade is a
//! Rust type with methods (`.wiki/driver/real-metal-north-star.md` §9).

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, Result};

/// Everything the driver holds between calls.
///
/// The engine's seam is this plus a completion broker. The broker stays over
/// there because a completion is the engine's object — the driver reports
/// that work was submitted, and what a scheduler waits on is not its concern.
pub struct Shell {
    pub(crate) context: Arc<crate::device::Context>,
    /// The command timeline, held ACROSS frames.
    ///
    /// This is what makes run-ahead run-ahead rather than within-frame
    /// pipelining. The timeline and the two-allocator ring live on the
    /// stepper, so a fresh one per frame has no previous value to compare
    /// against and no allocator to alternate: frame n+1 could not be queued
    /// while frame n ran, however the steps inside each were arranged.
    ///
    /// `Stepper::shared` rather than `Stepper::new` because a borrowing
    /// stepper beside the `Context` it borrows is a self-reference; sharing
    /// the context is what lets one outlive a call.
    pub(crate) stepper: crate::device::Stepper<'static>,
    /// Reusable fire regions, held ACROSS frames for the same reason the
    /// stepper is. A fresh region per fire leaks it into the residency set
    /// permanently -- nothing removes -- and moves an address that is one of
    /// only three things differing between two fires of one shape.
    pub(crate) scratch: crate::fire::Scratch,
    /// Which buffer each address belongs to. A recorded command binds a
    /// BUFFER where this driver otherwise binds an address, so recording
    /// needs the inverse of what a fire computes.
    pub(crate) regions: crate::device::Regions,
    /// Fires already recorded, by what they are valid for. Replaying one
    /// costs 39.8 us where encoding the same fire costs 14.87 ms.
    pub(crate) recordings: crate::fire::Recordings,
    /// Graphs already lowered, by the fire shape that produced them. A
    /// decode's is a constant of the deployment and this driver was deriving
    /// it once per token -- 0.81 ms of a 4.9 ms step.
    pub(crate) lowerings: crate::lowering::cached::Lowerings,
    pub(crate) registry: crate::channel::Registry,
    pub(crate) device_facts: driver_api::DeviceFacts,
    /// The checkpoint, once one is loaded. Held because every address in its
    /// tensor map points into the region it owns.
    pub(crate) model: Option<crate::weights::load::Loaded>,
    /// WHICH ROW this checkpoint matched, by id.
    ///
    /// `&'static str` because a row is a `const` and its id is the row's own
    /// spelling — there is nothing to allocate and nothing to keep in step.
    /// What was here was a `String` read out of a `config.json`'s
    /// `architectures[0]`, lowercased and stripped of its `ForCausalLM`
    /// tail, and it was a DISPATCH KEY: three tables were consulted with it
    /// and they disagreed in production. It is an identity now, and the
    /// only thing that reads it is the capability report an operator sees.
    pub(crate) id: &'static str,
    /// The paged KV pool, allocated at load.
    ///
    /// Declared BEFORE `arena`, because fields drop in declaration order and
    /// an elastic buffer charges its tiles back to the arena on the way out.
    /// The other order leaves the arena with nothing to charge and the
    /// accounting permanently short.
    pub(crate) pool: Option<crate::pools::kv::Pool>,
    /// What the KV pool's memory is charged against.
    ///
    /// Held on the driver rather than made per-load, because it is the thing
    /// that knows how much of the machine is already spoken for -- a fresh
    /// arena per load would let two pools each believe they had the whole
    /// budget.
    pub(crate) arena: crate::device::Arena,
    /// `[model] config` from the boot TOML, parsed by the caller.
    ///
    /// The one key this seam reads out of the boot config, and the same one
    /// `driver-cuda`'s shell reads. The PARSE stays with the caller: a boot
    /// TOML is the engine's format, and a driver that read it would be the
    /// second thing entitled to an opinion about the file's shape.
    ///
    /// WHAT IT POINTS AT CHANGED, and the change is the whole refactor in one
    /// field. It named a `pie.model/1` document — ~40 numbers a 845-line
    /// normalizer had projected out of a 136-field schema — which this driver
    /// parsed back into a private `ModelFacts` and read the model out of.
    /// It names the checkpoint's own `config.json` now, it is the FALLBACK
    /// for when the artifact does not carry one embedded, and ONE FIELD IS
    /// READ OUT OF IT: the declared quantization. Everything else a driver
    /// used to read here is a catalog row's, matched from the tensors.
    pub(crate) boot_config: Option<PathBuf>,
    /// `[model] id` from the boot TOML, when the operator named one.
    ///
    /// An OVERRIDE and not a selector. `catalog::identify` matches a
    /// checkpoint by its tensors and that is the answer in every ordinary
    /// case; this exists for the one case tensors cannot settle, where two
    /// rows are shape-identical and an operator knows which download this
    /// is. `driver-cuda` reads the same key into the same `Override`.
    pub(crate) boot_model_id: Option<String>,
    /// Whether the loaded checkpoint has GDN / linear-attention layers.
    ///
    /// A control-op capability rather than a shape: the recurrent state only
    /// exists if it does, so `copy_state` and `copy_kv` ask it before planning.
    pub(crate) has_linear_attn: bool,
    /// The rotary ladder, derived ONCE at load.
    ///
    /// A load-time derivation and not a per-fire one: a deployment that
    /// rescales its frequencies (llama-3, YaRN) states the rescaling in its
    /// config, and the config does not change between fires. Held as f32 bits
    /// because that is the channel it rides.
    pub(crate) inv_freq: Vec<u32>,
    /// The row's DEPLOYMENT, projected ONCE at load.
    ///
    /// The value `driver-cuda` holds under the same name, and held for the
    /// reason its own docs give: a launch that re-derived its geometry read
    /// the checkpoint through a DIFFERENT reading than the one its trace was
    /// built from, so a fire's head count and a trace's head count came from
    /// two readers of one document with nobody holding them together. There
    /// is one projection and every consumer reads it.
    pub(crate) deployment: Option<model::deployment::Deployment>,
    /// The identified ROW, and what this load observed that no row can state.
    ///
    /// The field this replaced held `(LlamaLikeFacts, LlamaLikeMetalFacts)` —
    /// twenty-nine model facts this driver had rebuilt for itself out of the
    /// projected deployment plus nine `has_tensor` probes — and its doc
    /// explained that it was synthesized here because "`catalog::Variant::trace`
    /// traces the row's CUDA text; there is no backend parameter on it".
    /// There is one now. `catalog::Deployed::backend` names which driver is
    /// asking, so the Metal text is the row's own answer and the facts are
    /// the row's own facts, stated once in `crates/model` for both backends
    /// instead of derived twice.
    ///
    /// What survives is the pair: the row, and the [`MetalBinding`] holding
    /// the six things a row genuinely cannot know — the affine group and bit
    /// width the bytes arrived in (`mlx-community` publishes one model at
    /// g64/b4 and at g128/b8, and the two pack to identical extents), whether
    /// the expert bank reached the device still in MXFP4, and the three
    /// kernel capabilities of `crates/kernels-metal` as compiled into this
    /// binary. [`crate::model::binding`] builds it and is the only thing that
    /// may.
    ///
    /// Both halves are `Copy`, so a fire takes them by copy where the old
    /// pair needed a `clone()` of two heap-allocated fact structs per launch.
    ///
    /// Held rather than re-derived per fire, for the same reason as the
    /// deployment above — and here the reason is sharper. A launch that
    /// re-identified would ask the tensors a second time, and the tensors are
    /// gone: staging consumed them. The row is the load's answer, carried.
    ///
    /// [`MetalBinding`]: model::catalog::MetalBinding
    pub(crate) text_row: Option<(
        &'static dyn model::catalog::Variant,
        model::catalog::MetalBinding,
    )>,
    /// The runtime shader compiler, and the pipelines a fire's symbols have
    /// compiled to. Held across fires: a model's symbol set is bounded by its
    /// text, so a driver that recompiled per fire would spend more time in the
    /// compiler than on the GPU.
    pub(crate) compiler: crate::program::Compiler,
    pub(crate) pipelines: crate::bind::encode::Pipelines,
}

// The context holds Objective-C objects, which are not `Send` by declaration.
// The engine owns the shell exclusively and the scheduler drives it from one
// place, which is the same reason `DummyDriver` asserts this.
//
// The assertion lives HERE now rather than on the engine's wrapper, and that
// is the point of moving it: a crate that hands out a type is the crate that
// knows what makes it safe to send. The engine was asserting a property of
// somebody else's fields.
unsafe impl Send for Shell {}
unsafe impl Sync for Shell {}

impl Shell {
    /// Open the default Metal 4 device.
    ///
    /// `boot_config` is `[model] config` and `boot_model_id` is
    /// `[model] id`, both from the caller's boot config and both already
    /// parsed — see the fields' docs for why the parse is not in here.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created. Both
    /// are boot conditions, not runtime ones.
    pub fn open(boot_config: Option<PathBuf>, boot_model_id: Option<String>) -> Result<Self> {
        // `Arc` over a type that is neither `Send` nor `Sync`, deliberately.
        //
        // `Stepper::shared` exists because a stepper that BORROWS the context
        // beside it is a self-reference, and the timeline has to outlive a
        // call for run-ahead to be run-ahead. Sharing is what makes that
        // possible, and the sharing is within one thread: the shell is owned
        // exclusively by whoever holds it and driven from one place, which is
        // the `unsafe impl Send` above stating exactly that.
        //
        // `Rc` is what clippy suggests and it is not available: `Stepper`'s
        // signature is `Arc<Context>`, and the whole point of the type is to
        // be held across frames by a caller that is itself `Send`.
        #[allow(clippy::arc_with_non_send_sync)]
        let context = Arc::new(crate::device::Context::new()?);
        let stepper = crate::device::Stepper::shared(context.clone())?;
        let compiler = crate::program::Compiler::new(&context)?;
        Ok(Self {
            arena: elastic_arena(&context),
            context,
            stepper,
            scratch: crate::fire::Scratch::new(),
            regions: crate::device::Regions::new(),
            recordings: crate::fire::Recordings::new(),
            lowerings: crate::lowering::cached::Lowerings::new(),
            registry: crate::channel::Registry::new(),
            device_facts: device_facts(),
            model: None,
            // No checkpoint is loaded, so no row has been matched. Empty
            // rather than a placeholder id: a placeholder here would be a
            // name the capability report could publish for a model that does
            // not exist.
            id: "",
            pool: None,
            boot_config,
            boot_model_id,
            inv_freq: Vec::new(),
            deployment: None,
            text_row: None,
            has_linear_attn: false,
            compiler,
            pipelines: crate::bind::encode::Pipelines::new(shader_tree()),
        })
    }

    /// The device's stated facts.
    #[must_use]
    pub fn device_facts(&self) -> &driver_api::DeviceFacts {
        &self.device_facts
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<driver_api::KvHandle> {
        None
    }

    /// The device this driver runs on.
    #[must_use]
    pub fn context(&self) -> &crate::device::Context {
        &self.context
    }

    /// The program/instance/channel registry.
    #[must_use]
    pub fn registry(&self) -> &crate::channel::Registry {
        &self.registry
    }

    /// The loaded checkpoint, if `load_model` has run.
    #[must_use]
    pub fn model(&self) -> Option<&crate::weights::load::Loaded> {
        self.model.as_ref()
    }

    /// The KV pool the checkpoint's geometry was allocated at.
    #[must_use]
    pub fn pool(&self) -> Option<&crate::pools::kv::Pool> {
        self.pool.as_ref()
    }

    /// The pool, or the refusal that names what would create one.
    pub(crate) fn need_pool(&self, what: &'static str) -> Result<&crate::pools::kv::Pool> {
        self.pool.as_ref().ok_or_else(|| Error::Unserved {
            what,
            message: "called before load_model, which is what allocates the KV pool. \
                      This is an order that was broken rather than something this \
                      backend cannot do"
                .to_string(),
        })
    }
}

/// The facts a scheduler reads, stated from what this backend IS rather than
/// parsed out of a config — a config that disagreed with the hardware would
/// simply be believed.
///
/// `unified_memory` is the one that changes scheduling: on Apple silicon the
/// KV pool and the host share physical memory, so a "device is full" question
/// is a different question here.
fn device_facts() -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: "metal".to_string(),
        unified_memory: true,
        // Metal has no native fp8 path and no MXFP4 MoE kernel; the table
        // says which kernels exist and neither is among them.
        fp8_native: false,
        native_mxfp4_moe: false,
        storage_alignment: 256,
        storage_max_tile_bytes: 0,
        storage_tile_map_mask: 0,
        // The paged KV pool's rows per page, which every `kv_translation`
        // index is in units of.
        page_size: 16,
    }
}

/// What the KV pool is allowed to hold, in bytes.
///
/// `recommendedMaxWorkingSetSize` is what the device will keep resident, and
/// on Apple silicon it is a share of the same physical memory the host is
/// using -- so the whole of it is not available and asking for the whole of
/// it is how a machine starts swapping mid-fire. Three quarters leaves room
/// for the weights, the scratch ring, and whatever else the box is doing,
/// which on a laptop is not nothing.
///
/// A device that reports zero declines to answer rather than promising
/// nothing; `Context::check_working_set` reads it the same way. Advertising a
/// budget there would be inventing one, so the answer is zero and the
/// scheduler sees a backend that cannot resize.
pub(crate) fn elastic_budget_bytes(context: &crate::device::Context) -> u64 {
    let working_set = context.working_set_bytes();
    if working_set == 0 {
        0
    } else {
        working_set / 4 * 3
    }
}

/// The arena the KV pool's pages are charged against.
///
/// The floor is zero: under critical pressure nothing new is mapped, and what
/// is already mapped stays -- see `Need`. A pool that could be taken back out
/// from under a bound address by the pressure probe would be a pool no fire
/// could rely on.
fn elastic_arena(context: &crate::device::Context) -> crate::device::Arena {
    crate::device::Arena::new(elastic_budget_bytes(context), 0)
}

/// Where the Metal shader tree lives.
///
/// Metal compiles at run time from `(path, entry name)`, so a driver needs the
/// `.metal` sources on disk. `PIE_METAL_KERNELS` overrides; the default is the
/// checkout's own tree, which is what a development run wants and what every
/// device test already uses.
///
/// `CARGO_MANIFEST_DIR` is THIS crate's now rather than the engine's, which is
/// the same directory one level up and the right one to read from: the tree
/// belongs to `kernels-metal`, which is this crate's sibling and not the
/// engine's.
fn shader_tree() -> PathBuf {
    std::env::var_os("PIE_METAL_KERNELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|crates| crates.join("kernels-metal/kernels"))
                .unwrap_or_default()
        })
}
