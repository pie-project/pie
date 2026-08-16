//! The shell's nouns: everything this driver holds between calls, and the
//! two verbs that create and describe it.
//!
//! A leaf. `load`, `launch`, `register` and `control` all call in here and
//! nothing here calls them, so the state can be edited without meeting the
//! modules that use it. The facade is a Rust type with methods rather than a
//! C ABI, which was deleted for Metal deliberately.

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, Result};

/// Everything the driver holds between calls.
///
/// The engine's seam is this plus a completion broker. The broker stays over
/// there because a completion is the engine's object.
pub struct Shell {
    pub(crate) context: Arc<crate::device::Context>,
    /// The command timeline, held ACROSS frames — what makes run-ahead
    /// run-ahead rather than within-frame pipelining, since a fresh stepper
    /// per frame has no previous value to compare against and no allocator to
    /// alternate. `Stepper::shared` rather than `Stepper::new` because a
    /// stepper borrowing the `Context` beside it is a self-reference.
    pub(crate) stepper: crate::device::Stepper<'static>,
    /// Reusable fire regions, held ACROSS frames for the same reason the
    /// stepper is. A fresh region per fire leaks it into the residency set
    /// permanently — nothing removes.
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
    /// WHICH ROW this checkpoint matched, by id. `&'static str` because a row
    /// is a `const` and its id is the row's own spelling; it is an identity,
    /// not a dispatch key.
    pub(crate) id: &'static str,
    /// The paged KV pool, allocated at load.
    ///
    /// Declared BEFORE `arena`, because fields drop in declaration order and
    /// an elastic buffer charges its tiles back to the arena on the way out.
    pub(crate) pool: Option<crate::pools::kv::Pool>,
    /// The recurrent stack's state planes, for a hybrid that has any.
    ///
    /// `None` for every pure-attention checkpoint, which is most of them, and
    /// the reason `Resolver::slab` may decline rather than having to answer.
    ///
    /// Declared beside `pool` and before `arena` for the same drop-order
    /// reason, though these are fixed allocations and charge nothing back.
    pub(crate) recurrent: Option<crate::pools::recurrent::Pool>,
    /// What the KV pool's memory is charged against.
    ///
    /// Held on the driver rather than made per-load: a fresh arena per load
    /// would let two pools each believe they had the whole budget.
    pub(crate) arena: crate::device::Arena,
    /// `[model] config` from the boot TOML, parsed by the caller: a boot TOML
    /// is the engine's format, and a driver that read it would be the second
    /// thing entitled to an opinion about the file's shape. It points at the
    /// checkpoint's own `config.json`, is the FALLBACK for when the artifact
    /// carries none embedded, and ONE FIELD is read out of it — the declared
    /// quantization.
    pub(crate) boot_config: Option<PathBuf>,
    /// `[model] id` from the boot TOML: an OVERRIDE, not a selector.
    /// `catalog::identify` matches a checkpoint by its tensors; this exists
    /// for the one case tensors cannot settle, where two rows are
    /// shape-identical. `driver-cuda` reads the same key the same way.
    pub(crate) boot_model_id: Option<String>,
    /// Whether the loaded checkpoint has GDN / linear-attention layers.
    ///
    /// A control-op capability rather than a shape: the recurrent state only
    /// exists if it does, so `copy_state` and `copy_kv` ask it before planning.
    pub(crate) has_linear_attn: bool,
    /// The rotary ladder, derived ONCE at load: a deployment that rescales
    /// its frequencies (llama-3, YaRN) states the rescaling in its config,
    /// and the config does not change between fires. Held as f32 bits.
    pub(crate) inv_freq: Vec<u32>,
    /// The row's DEPLOYMENT, projected ONCE at load. A launch that re-derived
    /// its geometry would read the checkpoint through a DIFFERENT reading
    /// than the one its trace was built from, so a fire's head count and a
    /// trace's head count would disagree with nobody holding them together.
    pub(crate) deployment: Option<model::deployment::Deployment>,
    /// The identified ROW, and what this load observed that no row can state.
    ///
    /// `catalog::Deployed::backend` names which driver is asking, so the
    /// facts are stated once in `crates/model` for both backends.
    ///
    /// The [`MetalBinding`] half holds the six things a row cannot know: the
    /// affine group and bit width the bytes arrived in (`mlx-community`
    /// publishes one model at g64/b4 and at g128/b8, and the two pack to
    /// identical extents), whether the expert bank reached the device still
    /// in MXFP4, and the three kernel capabilities of `crates/kernels-metal`
    /// as compiled into this binary. [`crate::model::binding`] is the only
    /// thing that may build it. Held rather than re-derived per fire: a
    /// launch that re-identified would ask the tensors again, and staging has
    /// consumed them.
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
// place. The assertion lives here because a crate that hands out a type is
// the crate that knows what makes it safe to send.
unsafe impl Send for Shell {}
unsafe impl Sync for Shell {}

impl Shell {
    /// Open the default Metal 4 device. `boot_config` is `[model] config` and
    /// `boot_model_id` is `[model] id`, both already parsed by the caller.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created.
    pub fn open(boot_config: Option<PathBuf>, boot_model_id: Option<String>) -> Result<Self> {
        // `Arc` over a type that is neither `Send` nor `Sync`, deliberately: a
        // stepper that BORROWS the context beside it is a self-reference, and
        // the timeline has to outlive a call for run-ahead to be run-ahead.
        // The sharing is within one thread, which the `unsafe impl Send`
        // above states. `Rc` is what clippy suggests and is unavailable — the
        // signature is `Arc<Context>`.
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
            // rather than a placeholder the capability report could publish.
            id: "",
            pool: None,
            boot_config,
            boot_model_id,
            inv_freq: Vec::new(),
            deployment: None,
            text_row: None,
            recurrent: None,
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
/// parsed out of a config that could disagree with the hardware.
/// `unified_memory` is the one that changes scheduling: on Apple silicon the
/// KV pool and the host share physical memory.
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

/// What the KV pool is allowed to hold, in bytes. On Apple silicon
/// `recommendedMaxWorkingSetSize` is a share of the same physical memory the
/// host is using, so asking for the whole of it is how a machine starts
/// swapping mid-fire. A device reporting zero declines to answer, so the
/// answer is zero and the scheduler sees a backend that cannot resize.
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
/// The floor is zero: under critical pressure nothing new is mapped and what
/// is already mapped stays, so a bound address cannot be taken back.
fn elastic_arena(context: &crate::device::Context) -> crate::device::Arena {
    crate::device::Arena::new(elastic_budget_bytes(context), 0)
}

/// Where the Metal shader tree lives. Metal compiles at run time from
/// `(path, entry name)`, so a driver needs the `.metal` sources on disk.
/// `PIE_METAL_KERNELS` overrides; the default resolves from THIS crate's
/// `CARGO_MANIFEST_DIR`, because the tree belongs to `kernels-metal`.
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
