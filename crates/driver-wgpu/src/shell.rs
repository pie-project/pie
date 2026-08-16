//! One assembled server: a device, a model, a cache, and a way to take turns.
//!
//! Every layer below is complete and tested, and none of them is a SERVER.
//! [`Serving::step`] takes six arguments — a device, a pipeline cache, a module
//! store, three borrowed resources and the turns — because it is deliberately
//! about one step and holds no state. This module owns the other four: a
//! [`Pool`] and a [`Book`] opened at the same shape, a [`Weights`] with a
//! seam, two plans traced at two fire classes, and a [`Geometry`] derived
//! from the same facts — pairs that a caller assembling them by hand could
//! get out of step, producing a server that runs, returns finite numbers,
//! and is wrong. [`Shell::on`] checks each pair against the other instead of
//! trusting them.
//!
//! `open` takes no module argument, unlike `driver-vulkan`'s SPIR-V map or
//! `driver-metal`'s `.metallib`: `kernels-wgpu` embeds every shader source in
//! the rlib and `naga` compiles them in this process, so there is no
//! directory to ship beside the binary and no way to hand out a driver
//! without its kernels. See [`crate::serve`]'s docs for the seam that
//! survives this.
//!
//! This is not the engine's seam — `driver-metal`'s `serve::Shell` answers
//! fourteen verbs for the whole of what a driver owes the runtime, and this
//! is the part of that about running a model, plus the five registration
//! verbs [`crate::programs`] already serves. Nor is it a sampler or a
//! tokenizer: a [`Step`] comes back with its distributions, as it does one
//! layer down.
//!
//! [`Shell::launch`] keeps two allocators apart: it fires over the PHYSICAL
//! pages a `FrameSubmission` names and never asks the [`Book`], because the
//! engine's scheduler already handed them out. The book belongs to
//! [`Shell::step`], the entry point for a server built on this crate alone —
//! firing a frame through `step` instead would let attention read another
//! conversation's keys while the model answers fluently.

use kernels_wgpu::Capability;
use model_ir::trace::ForwardPlan;

use crate::device::{Device, Failed, Pipelines, Unavailable};
use crate::dispatch::Geometry;
use crate::frames::{Launched, Unlaunched, pages_named, requests_of, tokens_of};
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
    /// The gated DeltaNet's carry, for a HYBRID text.
    ///
    /// `None` for every model this backend serves today, and the shell then
    /// opens no [`crate::resources::RecurrentPool`] and a GDN arm declines by
    /// name. It is stated
    /// by the CALLER rather than derived from the text because a plan does not
    /// say how many of its layers are linear, nor what a conv window is wide;
    /// those are the deployment's own facts, exactly as `pages` and `bytes`
    /// are, and a shell that guessed them would allocate the wrong slab and
    /// index off the end of it.
    pub recurrent: Option<crate::resources::Recurrent>,
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
            recurrent: None,
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
    /// [`FireClass::Decode`](model_ir::trace::FireClass::Decode).
    pub decode: ForwardPlan,
    /// The text a wider step lowers, traced at
    /// [`FireClass::Prefill`](model_ir::trace::FireClass::Prefill). See
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
                    model_ir::trace::Dim::Const(n) => Some(*n),
                    _ => None,
                })
                .collect()
        };
        // A SUBSET and not equality, and the difference is a HYBRID.
        //
        // This was equality, on the reasoning above: every `Dim::Const` is a
        // model extent, so the two classes state the same set. That holds for
        // a llama-like text and is FALSE for a gated DeltaNet, because the two
        // classes do not compute the same graph. A prefill SPLITS the scan --
        // `gdn_prep` stages the q/k path once and `gdn_core_recurrent` walks
        // it -- so it materialises `pre_q`, `pre_k` and `pre_gate` as traced
        // values, and `pre_gate`'s width is `2 * v_heads`. A decode FUSES the
        // same work into `gdn_core` and never names them. Qwen3.5-0.8B's
        // prefill states a `Const(32)` its decode does not, and it is one
        // model.
        //
        // So the prefill may state MORE: a split materialises intermediates
        // the fused form has no value for. It may not state FEWER, and that
        // direction is what the check was for -- a constant only the decode
        // has is a model extent the prefill's text lacks, which is two texts.
        let (d, f) = (constants(&self.decode), constants(&self.prefill));
        if !d.is_subset(&f) {
            let only = d.difference(&f).take(4).copied().collect::<Vec<_>>();
            return no(format!(
                "the decode plan states load-time widths the prefill one does not, so they \
                 are two models: {only:?}. A prefill may state MORE -- a split scan \
                 materialises intermediates a fused one never names -- but never fewer."
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
/// # There is no `Drop`
///
/// `wgpu::Device`, `Buffer`, `ComputePipeline` and `ShaderModule` are all
/// `Arc`-backed handles, and every resource holds a strong reference to the
/// device that made it, so dropping these fields in any order is safe —
/// unlike `driver-vulkan`'s `Shell`, which owns a raw device handle and needs
/// an explicit `Drop`. `device` is still declared LAST, matching that
/// sibling's field order; `tests/device.rs`'s
/// `a_buffer_outlives_the_device_handle_that_made_it` checks the claim.
pub struct Shell {
    pipelines: Pipelines,
    pool: Pool,
    /// The gated DeltaNet's carry, when the deployment stated one.
    ///
    /// `None` for every model this backend serves today. It is held beside the
    /// KV pool because it is the same KIND of thing — per-layer state no plan
    /// mentions and no arena holds — and is handed to `Model` so an arm can
    /// reach it.
    recurrent: Option<crate::resources::RecurrentPool>,
    book: Book,
    weights: Weights,
    text: Text,
    /// Lowerings already derived, by fire shape.
    ///
    /// Beside `text` because it is derived from it and from nothing else the
    /// shell can change: `text` is set once at [`Shell::on`] and never
    /// replaced (`a_shell_never_replaces_its_text` is what makes that a
    /// checked claim), so no path here has to remember to clear this.
    lowerings: crate::lowering::cached::Lowerings,
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
    /// Whether each step reads its arena back onto [`crate::turns::Step::arena`].
    keep_arena: bool,
    /// Whether each step submits every dispatch on its own command buffer.
    one_at_a_time: bool,
    /// How many rectangles of each fire to record, or all of them.
    prefix: Option<usize>,
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
    /// The four pairs this exists to keep in step:
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
        // Opened only for a deployment that states one. Its planes are keyed on
        // the layer NUMBER, and a hybrid interleaves, so every layer of the
        // stack is offered a seat and the ones that are not linear simply hold
        // none -- which is what `RecurrentPool::slab` answering `None` means,
        // and is right: nothing should be asking, and handing back another
        // layer's carry is worse than refusing.
        let recurrent = match deployment.recurrent {
            Some(shape) => Some(
                crate::resources::RecurrentPool::open(&device, shape, 0..text.layers)
                    .map_err(Unopened::Device)?,
            ),
            None => None,
        };
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
            recurrent,
            weights,
            text,
            lowerings: crate::lowering::cached::Lowerings::new(),
            modules: Embedded,
            programs: Programs::new(),
            // Best first, and an adapter always reports at least `Baseline`,
            // which requires no feature at all.
            tier: device
                .tiers()
                .first()
                .copied()
                .unwrap_or(Capability::Baseline),
            keep_arena: false,
            one_at_a_time: false,
            prefix: None,
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

    /// Read every fire's whole ARENA back onto [`crate::turns::Step::arena`].
    ///
    /// A DIAGNOSTIC and off by default: the arena is megabytes and the
    /// readback is a device stall, so a server wants none of it. What it is
    /// for is the question a driver otherwise cannot answer — WHERE a
    /// computation first went wrong — because a fire's activations exist only
    /// between the allocation and the drop inside one function, and the logits
    /// are the same sentence for a bad embedding and a bad lm head.
    ///
    /// `Lowered::value_offset` says where each traced value lives; with the
    /// bytes beside it, a caller can name the first value that is not finite.
    pub fn keep_arena(&mut self, keep: bool) {
        self.keep_arena = keep;
    }

    /// The gated DeltaNet's carry, for a caller that wants to read it.
    ///
    /// `None` for a deployment that opened no pool. A DIAGNOSTIC, and the one
    /// this backend's numerical hunts kept wanting last: a fire's recurrent
    /// state is a slab the pool owns, so it never appears in `Lowered::args`
    /// and a walk over a rectangle's operands cannot see it. A scan loads its
    /// state before the first token and writes it back after the last, which
    /// means a NaN already IN the slab comes out looking exactly like one the
    /// scan made.
    ///
    /// Read-only. What a caller does with it is
    /// [`crate::device::Device::read_at`] over
    /// [`crate::resources::RecurrentPool::slab`].
    #[must_use]
    pub const fn recurrent(&self) -> Option<&crate::resources::RecurrentPool> {
        self.recurrent.as_ref()
    }

    /// Record only the first `n` rectangles of each fire, or all of them.
    ///
    /// A DIAGNOSTIC, and the answer is not a fire: a truncated plan computes
    /// nothing anybody wants and its readout is whatever the arena held. What
    /// it is for is the one question a whole-fire readback cannot answer —
    /// WHERE a value first went wrong — because stopping at `n` makes the
    /// arena's end state the state at `n`, and a caller that walks `n` finds
    /// the rectangle rather than the subsystem.
    pub fn fire_prefix(&mut self, n: Option<usize>) {
        self.prefix = n;
    }

    /// Submit every dispatch on its own command buffer, with a device wait
    /// between them.
    ///
    /// Ruinously slow — a real plan is hundreds of rectangles and this is
    /// hundreds of submissions — and a DIAGNOSTIC. What it is for is the one
    /// claim this driver makes that it cannot otherwise check:
    /// `crate::serve::Fire::one_at_a_time`'s doc says `wgpu` inserts the
    /// barrier itself at every encoding granularity, so a disagreement between
    /// the two paths cannot be a missing barrier. That is the only thing in
    /// the device half taken from reading somebody else's source rather than
    /// from running something, and until this existed nothing could reach the
    /// flag to test it.
    pub fn one_at_a_time(&mut self, one: bool) {
        self.one_at_a_time = one;
    }

    /// How many lowerings this shell has derived.
    ///
    /// The number that says whether [`crate::lowering::cached`] is doing
    /// anything: a hundred decodes over one deployment should move it by one.
    /// Exposed because the claim is about a RUN and cannot be made from
    /// inside the cache's own unit tests, which have no device and no text.
    #[must_use]
    pub fn lowerings_derived(&self) -> usize {
        self.lowerings.derived()
    }

    /// Forget every lowering derived so far.
    ///
    /// **For the control**, and for a path that one day replaces a text.
    /// `a_run_of_decodes_derives_one_lowering_and_says_the_same_thing` runs a
    /// generation twice — once normally and once calling this before every
    /// step — and compares the tokens. The second run is this driver with the
    /// cache switched off, taking the same kernels down the same path, which
    /// is the only comparison that isolates the cache: re-prefilling a
    /// growing history instead would swap the matvec for the tiled GEMM and
    /// compare two different kernel families.
    pub fn forget_lowerings(&mut self) {
        self.lowerings.clear();
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
            keep_arena: self.keep_arena,
            one_at_a_time: self.one_at_a_time,
            prefix: self.prefix,
        };
        let mut held = Held {
            book: &mut self.book,
            pool: &mut self.pool,
            recurrent: self.recurrent.as_ref(),
            weights: &self.weights,
            lowerings: &mut self.lowerings,
        };
        let out = serving.step(
            &self.device,
            &mut self.pipelines,
            &self.modules,
            &mut held,
            turns,
        );
        // The convolution windows, carried back. See
        // [`crate::resources::RecurrentPool::carry_back`] -- and note that
        // BOTH ways into a fire need it, which is why this reads twice in this
        // file. Patching only [`Self::serve`] left every test in the tree
        // untouched, because they all come through here.
        if let Some(pool) = self
            .recurrent
            .as_ref()
            .filter(|_| out.is_ok() && self.prefix.is_none())
        {
            pool.carry_back(&self.device)
                .map_err(crate::turns::Unstepped::Failed)?;
        }
        out
    }

    /// Serve one frame from the engine.
    ///
    /// # What this does that [`Self::step`] does not
    ///
    /// Nothing to the book. The engine's scheduler owns page allocation —
    /// eviction, prefix sharing, the copy plans — and hands down the physical
    /// pages it chose; running those through this driver's own allocator would
    /// give two allocators one page and no way to notice. See [`crate::frames`].
    ///
    /// # The order, and why it is this one
    ///
    /// * admit first, WITHOUT side effects, so a refused frame can be re-posted
    ///   rather than undone;
    /// * grow the pool to the highest page the frame NAMES, since the pool may
    ///   have been trimmed below a mark the scheduler was right to hand out;
    /// * convert every step's CSRs BEFORE firing any of them, so a frame with a
    ///   malformed third step does not append the first two;
    /// * then fire, in the frame's own execution order, because step `n + 1`
    ///   reads the cache step `n` appended.
    ///
    /// # Errors
    ///
    /// [`Unlaunched`]. A frame the pool cannot hold is an `Ok` answer —
    /// [`Launched::Exhausted`] or [`Launched::Impossible`] — and not an error,
    /// because a full cache is a scheduling fact rather than a fault.
    pub fn launch(&mut self, frame: &driver_api::FrameSubmission) -> Result<Launched, Unlaunched> {
        if let Some(refused) = self.admit(frame)? {
            return Ok(refused);
        }

        // Every step converted before any is fired. A frame whose third step
        // does not close its CSR would otherwise have appended the first two
        // steps' keys, and the scheduler's retry of the same frame would append
        // them twice.
        let mut work = Vec::with_capacity(frame.steps.len());
        for step in &frame.steps {
            // A wire frame states no write descriptor: the class that does
            // reaches this driver through `envelope::fill`, which is a
            // different entry (`crates/engine/src/driver/backend/wgpu.rs`).
            work.push(self.prepare(&step.plan, &[])?);
        }

        let mut out = Vec::with_capacity(work.len());
        for (requests, tokens) in &work {
            out.push(self.serve(requests, tokens)?);
        }
        Ok(Launched::Ran(out))
    }

    /// Make room for a frame, or say why there is none.
    ///
    /// `Ok(None)` is the admitted case. `Ok(Some(..))` is one of the two
    /// refusals, which are answers rather than errors -- see [`Self::launch`],
    /// whose first half this is.
    ///
    /// # Errors
    ///
    /// A frame of no steps, or a pool that will not grow.
    pub fn admit(
        &mut self,
        frame: &driver_api::FrameSubmission,
    ) -> Result<Option<Launched>, Unlaunched> {
        if frame.steps.is_empty() {
            return Err(Unlaunched::Malformed("a frame of no steps".to_string()));
        }
        let need = pages_named(frame);
        // Against what the pool COULD hold rather than what it holds now: the
        // pool gives pages back down to a high-water mark, and a frame past
        // that mark is one it can serve after growing. Calling that impossible
        // would have the scheduler permanently drop work it had correctly
        // admitted, because the pool had been idle.
        if need > self.pool.ceiling(&self.device) {
            return Ok(Some(Launched::Impossible));
        }
        if need > self.pool.shape().pages {
            // A resize the device would not give memory for is `Exhausted`,
            // not a fault: the ceiling above is measured against
            // `Device::budget`, a heap's SIZE, so it admits any frame the
            // device could hold if it were EMPTY — and it never is, because
            // the weights are in it. Faulting here would refuse a request
            // the scheduler could have served by evicting first.
            if let Err(e) = self.pool.resize(&self.device, need) {
                if e.is_out_of_memory() {
                    return Ok(Some(Launched::Exhausted));
                }
                return Err(Unlaunched::Unstepped(crate::turns::Unstepped::Failed(e)));
            }
        }
        Ok(None)
    }

    /// One admitted step's plan, checked and converted but not fired.
    ///
    /// # Errors
    ///
    /// [`Unlaunched`] naming the CSR that does not close, or the field this
    /// driver does not serve.
    pub fn prepare(
        &self,
        plan: &driver_api::LaunchPlan,
        writes: &[Option<(u32, u32)>],
    ) -> Result<(Vec<crate::resources::Request>, Vec<Vec<u32>>), Unlaunched> {
        plan.validate_geometry()
            .map_err(|e| Unlaunched::Malformed(format!("this frame's geometry: {e}")))?;
        plan.validate_kv_writes(self.pool.shape().page_size)
            .map_err(|e| Unlaunched::Malformed(format!("this frame's KV writes: {e}")))?;
        // Before any conversion: a plan naming something this driver does
        // not implement is refused by the field's own name rather than
        // served without it. See `frames::unserved_in`.
        if let Some(what) = crate::frames::unserved_in(plan, self.recurrent.is_some()) {
            return Err(Unlaunched::Unserved(what));
        }
        let mut requests = requests_of(plan)?;
        crate::frames::writes_stay_in_the_declared_span(
            plan,
            &requests,
            self.pool.shape().page_size,
        )?;
        // The STATED write descriptor, cut into the requests it belongs to.
        // Flat and row-indexed on the way in, because that is how the program
        // states it and how `envelope::fill` translates it; per request from
        // here, because that is what a `Request` is.
        if !writes.is_empty() {
            let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
            if writes.len() != rows {
                return Err(Unlaunched::Malformed(format!(
                    "this step states {} write slot(s) for {rows} row(s)",
                    writes.len()
                )));
            }
            let mut at = 0;
            for request in &mut requests {
                let end = at + request.positions.len();
                request.writes = writes[at..end].to_vec();
                at = end;
            }
        }
        Ok((requests, tokens_of(plan)?))
    }

    /// Fire one prepared step.
    ///
    /// # Why this half is separate
    ///
    /// A frame whose steps are DEVICE-RESOLVED cannot be converted all at
    /// once: step `n + 1`'s tokens are what step `n`'s program puts on a
    /// channel, and they do not exist until step `n` has both fired and had
    /// its program run. Such a frame is driven a step at a time --
    /// prepare, fire, run the program, prepare the next -- which is what
    /// these two halves are for. A frame of ordinary host-wire steps keeps
    /// the stronger order [`Self::launch`] states.
    ///
    /// # Errors
    ///
    /// [`Unlaunched::Unstepped`] for a fire the device refused.
    pub fn serve(
        &mut self,
        requests: &[crate::resources::Request],
        tokens: &[Vec<u32>],
    ) -> Result<Step, Unlaunched> {
        let borrowed: Vec<&[u32]> = tokens.iter().map(Vec::as_slice).collect();
        let serving = Serving {
            plan: &self.text.decode,
            prefill: &self.text.prefill,
            geometry: self.text.geometry,
            tier: self.tier,
            keep_arena: self.keep_arena,
            one_at_a_time: self.one_at_a_time,
            prefix: self.prefix,
        };
        let mut held = Held {
            book: &mut self.book,
            pool: &mut self.pool,
            recurrent: self.recurrent.as_ref(),
            weights: &self.weights,
            lowerings: &mut self.lowerings,
        };
        let out = serving
            .over(
                &self.device,
                &mut self.pipelines,
                &self.modules,
                &mut held,
                requests,
                &borrowed,
            )
            .map_err(Unlaunched::Unstepped);
        // THE CONVOLUTION WINDOWS, CARRIED BACK. The gated-DeltaNet kernels
        // always read `conv_state` and always write `new_conv_state`, so the
        // live windows end a fire in the plane nothing reads. See
        // `RecurrentPool::carry_back` for why this is a copy and not a swap.
        //
        // Only after a fire that RAN, and not after a truncated one: a
        // `fire_prefix` stops partway and leaves `new_conv_state` holding some
        // layers' new windows beside others' old, which is not a state to carry
        // anywhere.
        if let Some(pool) = self
            .recurrent
            .as_ref()
            .filter(|_| out.is_ok() && self.prefix.is_none())
        {
            pool.carry_back(&self.device)
                .map_err(|e| Unlaunched::Unstepped(crate::turns::Unstepped::Failed(e)))?;
        }
        out
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
    /// # Why this grows the pool first, and only for destinations
    ///
    /// This pool is elastic and grows ON DEMAND: [`Self::admit`] raises it to
    /// the highest page a frame NAMES, so it holds what the frames so far
    /// have needed and not what the scheduler is entitled to hand out. A copy
    /// plan is the other door a page number arrives through, so it must carry
    /// the same reasoning or a destination one page above the high-water
    /// mark is wrongly refused.
    ///
    /// DESTINATIONS only -- see [`crate::frames::copy_pages_named`] for why a
    /// source above the pool stays a refusal.
    ///
    /// # Errors
    ///
    /// See [`crate::resources::Pool::copy_plan`], plus a [`Failed`] from the
    /// growth itself. There is no `Exhausted` answer on this verb: a copy the
    /// device cannot find memory for is a failure of a copy the engine had
    /// already committed to, not a scheduling fact it can act on.
    pub fn copy_kv(&mut self, plan: &driver_api::KvCopyPlan) -> Result<usize, Failed> {
        let need = crate::frames::copy_pages_named(plan);
        // Checked BEFORE the growth: without it, a page number past what the
        // adapter can hold reaches `Pool::resize` and returns the allocator's
        // own `PastLimit` error, naming a buffer size rather than a page.
        if need > self.pool.ceiling(&self.device) {
            return Err(Failed::Wgpu(format!(
                "this copy plan writes into page {} and the pool could hold at \
                 most {} pages on this adapter",
                need.saturating_sub(1),
                self.pool.ceiling(&self.device),
            )));
        }
        if need > self.pool.shape().pages {
            // The pool and not the book, exactly as `admit` does it: a copy
            // plan arrives on the engine-driven path, where the scheduler owns
            // page allocation and the book is not the allocator. Handing these
            // pages to the book as well would put two allocators on them.
            self.pool.resize(&self.device, need)?;
        }
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
        // The CEILING first, before either half moves: `Book::resize` builds
        // a free list with one entry per page grown, so an unchecked target
        // between `u32::MAX` and the adapter's actual limit would allocate
        // gigabytes of HOST memory before the device ever refused it.
        let ceiling = self.pool.ceiling(&self.device);
        if target > ceiling {
            return Err(Unresized::Device(Failed::Wgpu(format!(
                "a target of {target} pages is past the {ceiling} this adapter \
                 could hold"
            ))));
        }
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

    /// The KV pool this shell holds, for a caller that needs to READ it.
    ///
    /// Shared, never handed out mutably: everything that changes the pool goes
    /// through a verb on this type, so that the book and the pool cannot get
    /// different ideas of how many pages there are. `driver-vulkan` exposes
    /// the same accessor for the same reason -- a test that asks whether a
    /// growth kept the cache has to be able to look at the cache.
    #[must_use]
    pub const fn pool(&self) -> &Pool {
        &self.pool
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
    ) -> Result<driver_api::ChannelBinding, Unregistered> {
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
    ) -> Result<driver_api::InstanceBinding, Unregistered> {
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

    /// What this shell can do, now that a model is loaded.
    ///
    /// Every field below except the six in [`driver_api::ModelFacts`] is a
    /// statement about the DEVICE — pages the pool holds, copy directions it
    /// serves, sinks the kernels honour, how wide a fire the scratch can run.
    ///
    /// The checkpoint half is handed in rather than answered here: this crate
    /// keeps `model` and `model-loader` as dev-dependencies (`tests/pure.rs`
    /// asserts the closure), since a driver that depended on a checkpoint
    /// FORMAT could not be handed bytes. `driver-metal` answers both halves
    /// because it identifies the checkpoint itself.
    #[must_use]
    pub fn capabilities(&self, model: &driver_api::ModelFacts) -> driver_api::DriverCapabilities {
        let shape = self.shape();
        driver_api::DriverCapabilities {
            abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
            total_pages: shape.pages,
            kv_page_size: shape.page_size,
            // No swap pool and no recurrent-state cache: this driver has
            // neither, and `copy_state` refuses by name for the same reason.
            swap_pool_size: 0,
            // Device to device, and only that. `Pool::copy_plan` moves whole
            // pages and single rows inside the one KV buffer -- which is what
            // a prefix-cache hit is -- and refuses any plan whose ends are not
            // both this driver's own domain. Host directions stay off: there
            // is no swap pool, so a device-to-host copy has nowhere to land.
            kv_copy_domain_mask: driver_api::KV_COPY_DEVICE_TO_DEVICE,
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // Not elastic. `resize_pool` here reallocates the KV buffer
            // whole, so nothing can be given back page-wise, and both numbers
            // are zero together -- which is the condition `bootstrap` reads
            // before it starts a trim task at all. WebGPU has no sparse
            // binding, so unlike Vulkan this is not a declined optional
            // feature: there is nothing to decline.
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Sinks this backend cannot honour. Every one of them would bind
            // and then run as a silent no-op, which is worse than a refusal
            // at the door.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: driver_api::ModelSiteSummary::default(),
            device_geometry_port_mask: 0,
            // FALSE, unlike the engine seam's `true`: this describes the
            // HOST-WIRE path (`Self::launch`), which converts every step of a
            // frame up front rather than interleaving per step -- see
            // `Self::launch` for why. With zero device-geometry ports
            // advertised above, no frame reaching this path ever needs a
            // per-step resolve.
            resolves_geometry_per_step: false,
            // The ceilings a batch is formed under. `Shell::open` sizes one
            // fire's scratch from `Deployment::seam`, and a fire wider than
            // this has nothing to run in.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: shape.pages,
            // The row's answers, which this crate cannot read for itself.
            arch_name: model.arch_name.clone(),
            model_id: model.model_id.clone(),
            vocab_size: model.vocab_size,
            max_model_len: model.max_model_len,
            hidden_size: model.hidden_size,
            snapshot_dir: model.snapshot_dir.clone(),
            activation_dtype: "bf16".to_string(),
            // False about the BACKEND rather than about the row: there is no
            // encode entry point here at all, so a model with a vision tower
            // is served as its text half. `Shell::encode` refuses by name.
            supports_media_encode: false,
            kv_handle: None,
            // The shaders are in the rlib and `naga` compiles them; nothing
            // upstream generates a kernel for this driver.
            codegen_backend: String::new(),
        }
    }

    /// The cache's shape.
    #[must_use]
    pub fn shape(&self) -> Shape {
        // Asked live, not cached, so it can't go stale relative to `resize_pool`.
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

    /// How many distinct shader modules have been expanded and reflected.
    ///
    /// See [`Pipelines::modules_read`]. Public for the same reason
    /// [`Self::built`] is: it is a number a test can hold across a run of
    /// steps, and the claim it holds is worth 25x on a decode.
    #[must_use]
    pub const fn modules_read(&self) -> usize {
        self.pipelines.modules_read()
    }

    /// How many times the module cache has been consulted.
    ///
    /// See [`Pipelines::modules_asked`].
    #[must_use]
    pub const fn modules_asked(&self) -> usize {
        self.pipelines.modules_asked()
    }
}
