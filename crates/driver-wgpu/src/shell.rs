//! One assembled server: a device, a model, a cache, and a way to take turns.
//!
//! # Why this exists, given that everything in it already existed
//!
//! Every layer below is complete and tested, and none of them is a SERVER.
//! [`Serving::step`] takes six arguments — a device, a pipeline cache, a module
//! store, three borrowed resources and the turns — because it is deliberately
//! about one step and holds no state. Somebody has to own the other four, and
//! until this module every caller did it by hand: open a device, open a pool,
//! ask it for a stand-in and a rope ladder, open a book at the same shape, make
//! a [`Weights`] and give it a seam, trace two plans at two fire classes, build
//! a [`Geometry`] out of the facts, pick a tier. Thirty lines that must all
//! agree, of which perhaps four are interesting.
//!
//! That is a defect waiting to be written rather than a tidiness problem. The
//! shape a [`Pool`] is opened at and the shape a [`Book`] is opened at are the
//! same shape and nothing checked it; the [`Geometry`] a step is served with is
//! derived from the same facts the plans were traced from and nothing checked
//! that either. A caller who got one of those pairs out of step got a server
//! that ran, returned finite numbers, and was wrong.
//!
//! # `open` takes no modules, and that is the headline
//!
//! `driver-vulkan`'s counterpart takes a `BTreeMap<String, Vec<u8>>` of SPIR-V
//! and `driver-metal`'s finds a `.metallib`. **This one takes none**, because
//! `kernels-wgpu` embeds every shader source in the rlib and `naga` compiles
//! them in this process. There is no directory to ship beside the binary, no
//! `OUT_DIR` to relay through a build script, and no way for a deployment to be
//! given a driver and not its kernels. See [`crate::serve`]'s docs for the seam
//! that survives and why.
//!
//! # What it is not
//!
//! Not the engine's seam. `driver-metal`'s `serve::Shell` answers fourteen verbs
//! because it is the whole of what a driver owes the runtime. This is the part
//! of that which is about running a model: open, hold weights, take turns, plus
//! the five registration verbs [`crate::programs`] already serves. The rest is a
//! scheduler's vocabulary.
//!
//! Not a sampler, and not a tokenizer. A [`Step`] comes back with its
//! distributions, as it does one layer down.

use kernels_wgpu::Capability;
use model_compiler::trace::ForwardPlan;

use crate::device::{Device, Failed, Pipelines, Unavailable};
use crate::dispatch::Geometry;
use crate::pages::Book;
use crate::programs::{Programs, Unregistered};
use crate::resources::{Pool, Shape, Weights};
use crate::serve::Embedded;
use crate::turns::{Held, Serving, Step, Turn};

/// What a deployment decides that a model does not.
///
/// Every field here is absent from the checkpoint on purpose: a text states what
/// the model computes, and how much cache a server keeps or how big a staging
/// seam it wants are properties of the machine it runs on.
#[derive(Debug, Clone, Copy)]
pub struct Deployment {
    /// How many KV pages the pool holds.
    pub pages: u32,
    /// Bytes per cache element, 2 for bf16.
    pub bytes: u32,
    /// The rotary base.
    ///
    /// Not read from the facts, because a driver that guessed 10000 would be
    /// quietly wrong for every model that was trained longer.
    pub theta: f32,
    /// The rescaling a long-context model asks for, if any.
    pub rescale: Option<crate::rope::Rescale>,
    /// The stand-in buffer's size, which bounds the largest scalar block a fire
    /// can stage.
    pub seam: u64,
}

impl Default for Deployment {
    /// A deployment big enough to serve a small model on a desktop card.
    ///
    /// Stated as numbers rather than derived, because deriving them would
    /// require knowing the vocabulary and the widest launch, and a default that
    /// is wrong in a way nobody notices is worse than one that is obviously
    /// arbitrary.
    fn default() -> Self {
        Self {
            pages: 64,
            bytes: 2,
            theta: 1_000_000.0,
            rescale: None,
            seam: 1 << 22,
        }
    }
}

/// The model, as this driver receives it: two plans and the shape they were
/// traced for.
///
/// # Why the shell does not trace this itself
///
/// `crates/model` is not a dependency of this crate. A driver that traced its
/// own text would be a driver that had an opinion about which models exist; this
/// one executes a plan somebody else authored.
///
/// The cost is that the caller assembles the pieces, and can assemble them
/// wrongly — a geometry from one model beside a plan from another. So
/// [`Shell::on`] CHECKS them against each other rather than trusting them, which
/// is a stronger guarantee than deriving would have given: deriving assumes one
/// set of facts went in and cannot notice when two did.
pub struct Text {
    /// The text a one-row step lowers, traced at
    /// [`FireClass::Decode`](model_compiler::trace::FireClass::Decode).
    pub decode: ForwardPlan,
    /// The text a wider step lowers, traced at
    /// [`FireClass::Prefill`](model_compiler::trace::FireClass::Prefill). See
    /// [`Serving::prefill`] for the measurement that says why one plan will not
    /// do.
    pub prefill: ForwardPlan,
    /// The model's shape, for the launch rules that need it.
    pub geometry: Geometry,
    /// How many layers of cache the model needs.
    pub layers: u16,
}

impl Text {
    /// Do the four pieces describe one model this driver can serve?
    ///
    /// Every check here is between two things the caller supplied separately,
    /// which is the only kind worth making: a claim about one field alone would
    /// be a claim the caller could not have got wrong.
    ///
    /// # Errors
    ///
    /// [`Unopened::Unservable`], naming which two disagreed.
    pub fn servable(&self) -> Result<(), Unopened> {
        let no = |why: String| Err(Unopened::Unservable(why));

        // THE TWO PLANS, by their load-time constants.
        //
        // NOT by `family`, which reads `..decode` and `..prefill`, so the two
        // plans of ONE model never agree on it: its doc calls it a facts digest,
        // and for these texts the digest is the fire class.
        //
        // A `Dim::Const` is a load-time extent -- the hidden size, the head
        // count times the head dimension, the vocabulary. Every one of them is a
        // property of the model and none is a property of the fire, so the two
        // classes of one text state the same set and two models do not.
        let constants = |p: &ForwardPlan| -> std::collections::BTreeSet<u32> {
            p.values
                .iter()
                .flat_map(|v| v.shape.0.iter())
                .filter_map(|d| match d {
                    model_compiler::trace::Dim::Const(n) => Some(*n),
                    _ => None,
                })
                .collect()
        };
        let (d, f) = (constants(&self.decode), constants(&self.prefill));
        if d != f {
            let only = |a: &std::collections::BTreeSet<u32>,
                        b: &std::collections::BTreeSet<u32>| {
                a.difference(b).take(4).copied().collect::<Vec<_>>()
            };
            return no(format!(
                "the two plans state different load-time widths, so they are two models: \
                 {:?} appears only in the decode plan and {:?} only in the prefill one",
                only(&d, &f),
                only(&f, &d)
            ));
        }

        // THE PLANS AND THE CACHE. A plan tags each op with its layer, so the
        // depth is the plan's to state and the caller's to get wrong. A cache
        // one layer short is not refused by anything downstream: layer `L-1`
        // reads and writes a region that belongs to no layer, and the answer
        // stays finite.
        let depth = |p: &ForwardPlan| p.ops.iter().filter_map(|o| o.layer).max().map(|l| l + 1);
        for (which, plan) in [("decode", &self.decode), ("prefill", &self.prefill)] {
            match depth(plan) {
                Some(deep) if deep == u32::from(self.layers) => {}
                Some(deep) => {
                    return no(format!(
                        "the {which} plan states {deep} layers and the cache is opened for {}",
                        self.layers
                    ));
                }
                None => return no(format!("the {which} plan has no layer-tagged op")),
            }
        }

        // THE GEOMETRY, against itself and against the kernel table. Grouped
        // attention divides the query heads among the key heads, a rotation
        // cannot be wider than the head it turns, and a router cannot pick more
        // experts than exist.
        let g = self.geometry;
        if g.kv_heads == 0 || !g.q_heads.is_multiple_of(g.kv_heads) {
            return no(format!(
                "{} query heads do not divide among {} key heads",
                g.q_heads, g.kv_heads
            ));
        }
        if g.rotary_dims > g.head_dim {
            return no(format!(
                "a rotation of {} dimensions over a head of {}",
                g.rotary_dims, g.head_dim
            ));
        }
        if g.experts_per_token > g.n_experts {
            return no(format!(
                "{} experts picked per token out of {}",
                g.experts_per_token, g.n_experts
            ));
        }
        // Asked of the table a fire would ask, so it cannot drift from the
        // kernels that exist. A head dimension with no attention kernel is
        // refused HERE rather than at the first fire: a server that starts and
        // then cannot answer is worse than one that will not start.
        let width = format!("_d_{}", g.head_dim);
        if !kernels_wgpu::entrypoints()
            .iter()
            .any(|e| e.contains(&width))
        {
            return no(format!(
                "no attention kernel for a head dimension of {}",
                g.head_dim
            ));
        }
        Ok(())
    }
}

/// Why a shell did not open.
///
/// Four kinds and not one string, because the caller's next move differs: no
/// adapter is a machine fact, a failed allocation is a size to reduce, a model
/// this driver cannot serve is a configuration to change, and an adapter that
/// cannot build a kernel this model needs is a different machine.
#[derive(Debug)]
pub enum Unopened {
    /// There is no adapter to open.
    Absent(Unavailable),
    /// The adapter is there and would not give up the memory.
    Device(Failed),
    /// The facts state a model no plan here serves.
    Unservable(String),
    /// The adapter cannot bind everything this table needs.
    ///
    /// **Named at open, and this is the whole reason it is a variant of its
    /// own.** WebGPU's guaranteed floor is 8 storage buffers per shader stage
    /// and `sdpa_paged_decode` binds eleven, so an adapter at the floor can
    /// build most of this tree and cannot build attention. Left to `wgpu` that
    /// arrives at the first decode as a validation message about a number; here
    /// it arrives before a model is loaded, with the kernels named.
    Unreachable {
        /// The adapter that cannot.
        adapter: String,
        /// Storage buffers it allows one compute stage.
        limit: u32,
        /// The rows that need more, by name.
        kernels: Vec<&'static str>,
    },
}

impl std::fmt::Display for Unopened {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absent(e) => write!(f, "no device: {e}"),
            Self::Device(e) => write!(f, "the device refused: {e}"),
            Self::Unservable(why) => write!(f, "unservable: {why}"),
            Self::Unreachable {
                adapter,
                limit,
                kernels,
            } => write!(
                f,
                "`{adapter}` allows a compute stage {limit} storage buffers and \
                 {} kernels need more, starting with {:?}",
                kernels.len(),
                kernels.iter().take(4).collect::<Vec<_>>()
            ),
        }
    }
}

impl std::error::Error for Unopened {}

/// Why a fork did not happen.
#[derive(Debug)]
pub enum Unforked {
    /// The book refused: no pages, or a seat that may not be taken.
    Unhoused(crate::pages::Unhoused),
    /// The copy failed partway. The destination was released again.
    Device(Failed),
}

impl std::fmt::Display for Unforked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unhoused(e) => write!(f, "{e}"),
            Self::Device(e) => write!(f, "the copy failed: {e}"),
        }
    }
}

impl std::error::Error for Unforked {}

/// Why a pool did not change size.
#[derive(Debug)]
pub enum Unresized {
    /// Somebody holds a page the shrink would drop.
    Stranded(crate::pages::Unhoused),
    /// The reallocation failed. The pool still holds what it did.
    Device(Failed),
}

impl std::fmt::Display for Unresized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stranded(e) => write!(f, "{e}"),
            Self::Device(e) => write!(f, "the cache could not be rebuilt: {e}"),
        }
    }
}

impl std::error::Error for Unresized {}

/// A device, a model's two plans, its cache, and its weights.
///
/// Owns everything a step needs except the turns. The fields are private because
/// the invariant this module exists for is that they agree with each other — see
/// [`Shell::on`] for which pairs, and what being out of step looks like.
///
/// # There is no `Drop`, and that is a conclusion rather than an omission
///
/// `driver-vulkan`'s `Shell` has one, and its comment records a real defect: the
/// first `Shell` to own a device rather than borrow a shared static one
/// destroyed that device with buffers still on it, and the validation layer said
/// `vkDestroyDevice(): VkBuffer 0x97 has not been destroyed`. Without the layer
/// it is a leak that grows one model's worth per shell. Its `Drop` releases the
/// weights, the pool and the pipelines before the device field is reached.
///
/// **The concern does not survive to `wgpu`, and the reason is ownership rather
/// than ordering.** `wgpu::Device`, `Buffer`, `ComputePipeline` and
/// `ShaderModule` are all handles onto `Arc`-backed resources, and every
/// resource holds a strong reference to the device that made it. Dropping the
/// last `Device` handle while buffers are alive does not destroy anything: the
/// internal device outlives them and is torn down when the last resource goes.
/// So there is no use-after-free to order against, no validation message to
/// provoke, and no leak — the memory returns when the last handle drops,
/// whichever order that happens in.
///
/// What a `Drop` here could still do is make the release PROMPT. It is not
/// written, because the only thing it could do is drop fields that are about to
/// be dropped anyway, and a `Drop` that restates the compiler's own order is a
/// thing a reader has to check for a difference that is not there.
///
/// `device` is nonetheless the LAST field, so declaration order matches the
/// intuition a reader arrives with from the sibling. That costs nothing and the
/// day this crate holds something that is NOT `Arc`-backed — a mapped buffer, a
/// raw handle borrowed through `wgpu-hal` — it is already right.
/// `a_shell_can_be_dropped_with_its_buffers_still_alive` in `tests/device.rs`
/// is the evidence for the paragraph above rather than for this one.
pub struct Shell {
    pipelines: Pipelines,
    pool: Pool,
    book: Book,
    weights: Weights,
    text: Text,
    /// The shader tree, which is a unit struct: see this module's own docs for
    /// why there is nothing to load and nothing to configure.
    modules: Embedded,
    /// The PTIR programs, channels and instances the engine has registered.
    ///
    /// Here rather than beside them because the engine's seam wants ONE object
    /// to call fourteen verbs on, and five of those fourteen are this. It shares
    /// nothing with the fields above it — no device, no cache — so a
    /// registration cannot disturb a conversation and a fire cannot disturb a
    /// ring.
    programs: Programs,
    tier: Capability,
    /// LAST. See the type's own docs for what that is worth here, which is less
    /// than it is worth next door and is not nothing.
    device: Device,
}

impl Shell {
    /// Open an adapter and assemble a server for `text`.
    ///
    /// **No module argument.** See this module's docs: the shaders are in the
    /// rlib.
    ///
    /// The four pairs this exists to keep in step, each of which used to be a
    /// caller's job and none of which anything checked:
    ///
    /// * the pool's [`Shape`] and the book's, which address the same pages;
    /// * the [`Geometry`] a step is served with and the plans it serves, which
    ///   describe the same model;
    /// * the two plans, which must be one text at two fire classes rather than
    ///   two texts;
    /// * the rope ladder's width and the head dimension, since a ladder built at
    ///   the wrong width rotates part of a head and leaves the rest alone.
    ///
    /// The tier is the best the adapter's features allow, from
    /// [`crate::device::Device::tiers`].
    ///
    /// # Errors
    ///
    /// [`Unopened`].
    pub fn open(text: Text, deployment: Deployment) -> Result<Self, Unopened> {
        let device = Device::open().map_err(Unopened::Absent)?;
        Self::on(device, text, deployment)
    }

    /// As [`Shell::open`], on a device the caller already has.
    ///
    /// Separate because opening an instance and an adapter twice in one process
    /// is legal and slow, and a suite that tests this has one device it shares.
    ///
    /// # Errors
    ///
    /// As [`Shell::open`], minus [`Unopened::Absent`].
    pub fn on(device: Device, text: Text, deployment: Deployment) -> Result<Self, Unopened> {
        text.servable()?;
        // Before a byte is allocated. An adapter that cannot bind the widest row
        // in the table can still build most of it, so this refusal is about the
        // MACHINE and arrives at the only moment a deployment can act on it --
        // which is the whole argument in `Unopened::Unreachable`.
        if !device.unreachable().is_empty() {
            return Err(Unopened::Unreachable {
                adapter: device.name().to_owned(),
                limit: device.limits().storage_buffers,
                kernels: device.unreachable().to_vec(),
            });
        }
        let shape = Shape {
            layers: text.layers,
            kv_heads: text.geometry.kv_heads,
            head_dim: text.geometry.head_dim,
            page_size: crate::facts::PAGE_SIZE,
            pages: deployment.pages,
            bytes: deployment.bytes,
        };
        let mut pool = Pool::open(&device, shape).map_err(Unopened::Device)?;
        pool.stand_in(&device, deployment.seam)
            .map_err(Unopened::Device)?;
        // From the TEXT's rotary width, so a ladder cannot disagree with the
        // rotation the plan states.
        pool.ladder(
            &device,
            text.geometry.rotary_dims,
            deployment.theta,
            deployment.rescale,
        )
        .map_err(Unopened::Device)?;
        let mut weights = Weights::new();
        weights
            .seam(&device, deployment.seam)
            .map_err(Unopened::Device)?;

        Ok(Self {
            pipelines: Pipelines::new(),
            book: Book::over(shape),
            pool,
            weights,
            text,
            modules: Embedded,
            programs: Programs::new(),
            // Best first, and an adapter always reports at least `Baseline`,
            // which requires no feature at all.
            tier: device
                .tiers()
                .first()
                .copied()
                .unwrap_or(Capability::Baseline),
            device,
        })
    }

    /// Hold one weight, under the name a PLAN uses for it.
    ///
    /// Bytes, not a path: a driver that depended on a checkpoint format would be
    /// a driver that could not be handed bytes. [`crate::names`] is what turns a
    /// loader's names into these.
    ///
    /// # Errors
    ///
    /// [`Failed`] if the device will not allocate.
    pub fn hold(&mut self, name: &str, bytes: &[u8]) -> Result<(), Failed> {
        self.weights.hold(&self.device, name, bytes)
    }

    /// Take one step over `turns`.
    ///
    /// # Errors
    ///
    /// [`crate::turns::Unstepped`], unchanged from the layer below: this adds no
    /// refusal of its own, because everything it owns was checked when it was
    /// opened.
    pub fn step(&mut self, turns: &[Turn]) -> Result<Step, crate::turns::Unstepped> {
        let serving = Serving {
            plan: &self.text.decode,
            prefill: &self.text.prefill,
            geometry: self.text.geometry,
            tier: self.tier,
        };
        let mut held = Held {
            book: &mut self.book,
            pool: &mut self.pool,
            weights: &self.weights,
        };
        serving.step(
            &self.device,
            &mut self.pipelines,
            &self.modules,
            &mut held,
            turns,
        )
    }

    /// Give `to` a copy of `from`'s history.
    ///
    /// The two halves of a fork live in two places on purpose — the book owns
    /// who holds which page, the pool owns what is in it — and this is the only
    /// place that has both. [`Book::fork`] hands back the moves rather than
    /// performing them precisely so that a caller cannot do one half; here the
    /// list is consumed immediately.
    ///
    /// Returns how many pages were copied.
    ///
    /// # Errors
    ///
    /// [`crate::pages::Unhoused`] from the book, or [`Failed`] from a copy. A
    /// refusal from the book leaves nothing taken; a failure DURING the copy
    /// leaves `to` seated on pages holding a partial history, which is why the
    /// pages are released again before returning.
    pub fn fork(&mut self, from: u64, to: u64) -> Result<usize, Unforked> {
        let moves = self.book.fork(from, to).map_err(Unforked::Unhoused)?;
        for (source, page) in &moves {
            if let Err(e) = self.pool.copy_page(&self.device, *source, *page) {
                // Not left half-copied. A seat over pages holding some of
                // another conversation's history is worse than no seat: it
                // answers, and the answer is a blend of two conversations.
                self.book.release(to);
                return Err(Unforked::Device(e));
            }
        }
        Ok(moves.len())
    }

    /// Serve the engine's `copy_kv`: a list of whole-page moves and a list of
    /// single-row cells.
    ///
    /// The work is [`crate::resources::Pool::copy_plan`]'s, so that a test of
    /// the arithmetic does not need a model.
    ///
    /// # Errors
    ///
    /// See [`crate::resources::Pool::copy_plan`].
    pub fn copy_kv(&mut self, plan: &driver_api::KvCopyPlan) -> Result<usize, Failed> {
        self.pool.copy_plan(&self.device, plan)
    }

    /// Serve the engine's `resize_pool`: hold `target_pages` pages.
    ///
    /// # Which pool
    ///
    /// The KV one, and only it. The engine's trim task asks about three — KV,
    /// recurrent state and workspace — on every tick, and the other two have no
    /// storage here. They are ANSWERED rather than refused, because "resize the
    /// thing that holds nothing" is satisfied by doing nothing, and a refusal
    /// would make the trim task log a failure every tick for a question it was
    /// right to ask. Ignoring the id instead would resize the KV pool to the
    /// state pool's target, which is a high-water mark of zero.
    ///
    /// The plan's `map_ranges` and `unmap_ranges` are not read. They describe a
    /// sparse pool's commits, and **WebGPU has no sparse binding at all** — so
    /// unlike `driver-vulkan`, which declines the optional feature on purpose,
    /// this backend could not act on them if it wanted to. `target_pages` is the
    /// whole of what it can serve.
    ///
    /// # Errors
    ///
    /// [`Unresized::Stranded`] if a conversation holds a page the shrink would
    /// drop — checked BEFORE anything moves, so a refusal leaves the pool and
    /// the book exactly as they were. [`Unresized::Device`] if the allocation
    /// fails, which also leaves the pool unchanged, and the book is put back to
    /// match it.
    pub fn resize_pool(&mut self, plan: &driver_api::PoolResizePlan) -> Result<(), Unresized> {
        if plan.pool_id != driver_api::PIE_ELASTIC_POOL_KV {
            return Ok(());
        }
        let target = u32::try_from(plan.target_pages).map_err(|_| {
            Unresized::Device(Failed::Wgpu(format!(
                "{} pages is not a cache this device could hold",
                plan.target_pages
            )))
        })?;
        let was = self.pool.shape().pages;
        // The book first, because it is the half that can refuse and the half
        // that can be put back.
        self.book.resize(target).map_err(Unresized::Stranded)?;
        if let Err(e) = self.pool.resize(&self.device, target) {
            self.book
                .resize(was)
                .expect("a book cannot strand a page going back to where it was");
            return Err(Unresized::Device(e));
        }
        Ok(())
    }

    /// What this driver tells the engine about the device it opened.
    #[must_use]
    pub fn device_facts(&self) -> driver_api::DeviceFacts {
        crate::facts::of(
            u32::try_from(self.device.min_storage_offset()).unwrap_or(u32::MAX),
            self.device.unified(),
        )
    }

    /// WebGPU exports no KV handle: nothing here shares a pool across processes,
    /// and both siblings answer the same way for the same reason. It is more
    /// firmly true here — WebGPU has no external-memory extension at all, where
    /// Vulkan has one this driver declines.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<driver_api::KvHandle> {
        None
    }

    /// The device this shell runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The tier every pipeline is built at.
    #[must_use]
    pub fn tier(&self) -> Capability {
        self.tier
    }

    /// Register a PTIR program.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a package the registry refuses.
    pub fn register_program(
        &mut self,
        desc: &driver_api::ProgramRegistration,
    ) -> Result<u64, Unregistered> {
        self.programs.register_program(desc)
    }

    /// Register a channel and say where its ring is.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a shape, dtype or id the registry will not serve.
    pub fn register_channel(
        &mut self,
        desc: &driver_api::ChannelRegistrationPlan,
    ) -> Result<driver_api::PieChannelEndpointBinding, Unregistered> {
        self.programs.register_channel(desc)
    }

    /// Bind an instance of a registered program to its channels.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for an unknown program, an unbindable channel, or a
    /// geometry class this driver does not serve.
    pub fn bind_instance(
        &mut self,
        program_id: u64,
        requested: Option<u64>,
        geometry_class: u32,
        channel_ids: &[u64],
        seeds: &[(u64, Vec<u8>)],
    ) -> Result<driver_api::PieInstanceBinding, Unregistered> {
        self.programs
            .bind_instance(program_id, requested, geometry_class, channel_ids, seeds)
    }

    /// Release an instance. Idempotent.
    pub fn close_instance(&mut self, id: u64) {
        self.programs.close_instance(id);
    }

    /// Release a channel. Idempotent.
    pub fn close_channel(&mut self, id: u64) {
        self.programs.close_channel(id);
    }

    /// The registry, for a caller that runs a program's stages.
    #[must_use]
    pub fn programs(&self) -> &Programs {
        &self.programs
    }

    /// The cache's shape.
    #[must_use]
    pub fn shape(&self) -> Shape {
        // ASKED, not remembered. A `Shape` field here was stale the moment
        // `resize_pool` existed, and it reported the old page count while the
        // pool held the new one -- a caller who sized a frame from it would have
        // addressed pages that were no longer there.
        self.pool.shape()
    }

    /// Who owns which page.
    #[must_use]
    pub fn book(&self) -> &Book {
        &self.book
    }

    /// How many pipelines have been built.
    ///
    /// A server's pipeline cache must stop growing; `tests/device.rs` holds that
    /// over a run of steps and needs to see the number.
    #[must_use]
    pub fn built(&self) -> usize {
        self.pipelines.built()
    }
}
