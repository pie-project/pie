//! One assembled server: a device, a model, a cache, and a way to take turns.
//!
//! # Why this exists, given that everything in it already existed
//!
//! Every layer below is complete and tested, and none of them is a SERVER.
//! `Serving::step` takes five arguments -- a device, a pipeline cache, a module
//! store, three borrowed resources and the turns -- because it is deliberately
//! about one step and holds no state. Somebody has to own the other four, and
//! until this module every caller did it by hand: open a device, read the
//! modules off disk, open a pool, ask it for a stand-in and a rope ladder,
//! open a book at the same shape, make a `Weights` and give it a seam, trace
//! two plans at two fire classes, build a `Geometry` out of the facts, pick a
//! tier. Thirty lines that must all agree, of which perhaps four are
//! interesting.
//!
//! That is a defect waiting to be written rather than a tidiness problem. The
//! shape a `Pool` is opened at and the shape a `Book` is opened at are the same
//! shape and nothing checked it; the `Geometry` a step is served with is
//! derived from the same facts the plans were traced from and nothing checked
//! that either. A caller who got one of those pairs out of step got a server
//! that ran, returned finite numbers, and was wrong.
//!
//! # What it is not
//!
//! Not the engine's seam. `driver-metal`'s `serve::Shell` answers fourteen
//! verbs -- programs, channels, instances, a `FrameSubmission`, KV copies, pool
//! resizes -- because it is the whole of what a driver owes the runtime. This
//! is the part of that which is about running a model: open, hold weights,
//! take turns. The rest is a registry and a scheduler's vocabulary, and
//! building it before this composed would be building the door before the
//! room.
//!
//! Not a sampler, and not a tokenizer. A `Step` comes back with its
//! distributions, as it does one layer down.

use std::collections::BTreeMap;

use kernels_vulkan::Capability;
use model_ir::trace::ForwardPlan;

use crate::device::{Device, Failed, Pipelines, Unavailable};
use crate::dispatch::Geometry;
use crate::frames::{Launched, Unlaunched, pages_named, requests_of, tokens_of};
use crate::pages::Book;
use crate::resources::{Pool, Shape, Weights};
use crate::turns::{Held, Serving, Step, Turn};

/// What a deployment decides that a model does not.
///
/// Every field here is absent from the checkpoint on purpose: a text states
/// what the model computes, and how much cache a server keeps or how big a
/// staging seam it wants are properties of the machine it runs on. See
/// [`crate::resources`] for the same argument at more length.
#[derive(Debug, Clone, Copy)]
pub struct Deployment {
    /// How many KV pages the pool holds.
    pub pages: u32,
    /// Bytes per cache element, 2 for bf16.
    pub bytes: u32,
    /// The rotary base.
    ///
    /// Not read from the facts, because `LlamaLikeFacts` does not carry it and
    /// a driver that guessed 10000 would be quietly wrong for every model that
    /// was trained longer.
    pub theta: f32,
    /// The rescaling a long-context model asks for, if any.
    pub rescale: Option<crate::rope::Rescale>,
    /// The stand-in buffer's size, which bounds the largest scalar block a
    /// fire can stage.
    pub seam: u64,
}

impl Default for Deployment {
    /// A deployment big enough to serve a small model on a desktop card.
    ///
    /// Stated as numbers rather than derived, because deriving them would
    /// require knowing the vocabulary and the widest launch, and a default
    /// that is wrong in a way nobody notices is worse than one that is
    /// obviously arbitrary.
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
/// `crates/model` is a DEV-dependency of this crate, deliberately, and
/// `driver-metal` takes it as a real one. The difference is a claim about what
/// a driver is: this one executes a plan somebody else authored, and a driver
/// that traced its own text would be a driver that had an opinion about which
/// models exist. `tests/` traces, because a test needs a real text to run
/// against; the library never does.
///
/// The cost is that the caller assembles the pieces, and can assemble them
/// wrongly -- a geometry from one model beside a plan from another. So
/// [`Shell::on`] CHECKS them against each other rather than trusting them,
/// which is a stronger guarantee than deriving would have given: deriving
/// assumes one set of facts went in and cannot notice when two did.
pub struct Text {
    /// The text a one-row step lowers, traced at
    /// [`FireClass::Decode`](model_ir::trace::FireClass::Decode).
    pub decode: ForwardPlan,
    /// The text a wider step lowers, traced at
    /// [`FireClass::Prefill`](model_ir::trace::FireClass::Prefill). See
    /// [`Serving::prefill`] for the measurement that says why one plan will
    /// not do.
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
    /// which is the only kind worth making: a claim about one field alone
    /// would be a claim the caller could not have got wrong.
    ///
    /// # Errors
    ///
    /// [`Unopened::Unservable`], naming which two disagreed.
    pub fn servable(&self) -> Result<(), Unopened> {
        let no = |why: String| Err(Unopened::Unservable(why));

        // THE TWO PLANS, by their load-time constants.
        //
        // NOT by `family`, which was the first thing tried and is wrong: it
        // reads `llama_like.metal.decode` and `llama_like.metal.prefill`, so
        // the two plans of ONE model never agree on it. Its doc calls it a
        // facts digest, and for these texts the digest is the fire class.
        //
        // A `Dim::Const` is a load-time extent -- the hidden size, the head
        // count times the head dimension, the vocabulary. Every one of them is
        // a property of the model and none is a property of the fire, so the
        // two classes of one text state the same set and two models do not.
        // Measured across the two real checkpoints this suite serves:
        // qwen3-0.6B states {1024, 128, 151936, ...} and qwen2.5-1.5B states
        // {1536, 8960, ...}, which share nothing but the vocabulary.
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
        // cannot be wider than the head it turns, and a router cannot pick
        // more experts than exist.
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
        if !kernels_vulkan::entrypoints()
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
/// Three kinds and not one string, because the caller's next move differs: no
/// device is a machine fact, a failed allocation is a size to reduce, and a
/// model this driver cannot serve is a configuration to change.
#[derive(Debug)]
pub enum Unopened {
    /// There is no Vulkan device to open.
    Absent(Unavailable),
    /// The device is there and would not give up the memory.
    Device(Failed),
    /// The facts state a model no plan here serves.
    Unservable(String),
}

impl std::fmt::Display for Unopened {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absent(e) => write!(f, "no device: {e:?}"),
            Self::Device(e) => write!(f, "the device refused: {e:?}"),
            Self::Unservable(why) => write!(f, "unservable: {why}"),
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
            Self::Device(e) => write!(f, "the copy failed: {e:?}"),
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
    /// A pool this driver does not have was asked to hold pages. Sizing it to
    /// zero is `Ok`; sizing it to anything else cannot be honoured, and
    /// saying otherwise would have the engine stop asking.
    Absent {
        /// The id that was asked, which may be one this driver has never
        /// heard of rather than one of the three the engine defines.
        pool_id: u64,
        /// What it was asked to hold. Never zero: zero is `Ok`.
        target_pages: u64,
    },
}

impl std::fmt::Display for Unresized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stranded(e) => write!(f, "{e}"),
            Self::Device(e) => write!(f, "the cache could not be rebuilt: {e:?}"),
            Self::Absent {
                pool_id,
                target_pages,
            } => {
                // The id by NAME where there is one. A trim task's log is
                // read by somebody asking which pool went wrong, and `2`
                // answers that only for a reader holding `local.rs` open.
                let which = match *pool_id {
                    driver_api::PIE_ELASTIC_POOL_STATE => "the recurrent-state pool".to_string(),
                    driver_api::PIE_ELASTIC_POOL_WORKSPACE => "the workspace pool".to_string(),
                    other => format!("pool {other}"),
                };
                write!(
                    f,
                    "{which} cannot hold {target_pages} pages: this backend has no such pool, \
                     and only a target of 0 is true of a pool that does not exist"
                )
            }
        }
    }
}

impl std::error::Error for Unresized {}

/// A device, a model's two plans, its cache, and its weights.
///
/// Owns everything a step needs except the turns. The fields are private
/// because the invariant this module exists for is that they agree with each
/// other -- see [`Shell::open`] for which pairs, and what being out of step
/// looks like.
pub struct Shell {
    modules: BTreeMap<String, Vec<u8>>,
    pipelines: Pipelines,
    pool: Pool,
    book: Book,
    weights: Weights,
    /// Lowerings kept between steps. Not `Drop`-ordered with the fields
    /// above because it holds no device object: a lowering is launches,
    /// symbols and offsets.
    lowerings: crate::turns::Lowerings,
    text: Text,
    tier: Capability,
    /// LAST, so that it outlives every buffer above it even if the [`Drop`]
    /// below is ever removed. Fields drop in declaration order.
    device: Device,
}

impl Shell {
    /// Open a device and assemble a server for `text`.
    ///
    /// The four pairs this exists to keep in step, each of which used to be a
    /// caller's job and none of which anything checked:
    ///
    /// * the pool's [`Shape`] and the book's, which address the same pages;
    /// * the [`Geometry`] a step is served with and the plans it serves,
    ///   which describe the same model;
    /// * the two plans, which must be one text at two fire classes rather than
    ///   two texts;
    /// * the rope ladder's width and the head dimension, since a ladder built
    ///   at the wrong width rotates part of a head and leaves the rest alone.
    ///
    /// The tier is the best the device loads, from
    /// [`Device::tiers`](crate::device::Device::tiers).
    ///
    /// # Errors
    ///
    /// [`Unopened`].
    pub fn open(
        text: Text,
        deployment: Deployment,
        modules: BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, Unopened> {
        let device = Device::open().map_err(Unopened::Absent)?;
        Self::on(device, text, deployment, modules)
    }

    /// As [`Shell::open`], on a device the caller already has.
    ///
    /// Separate because opening a Vulkan instance twice in one process is
    /// legal and slow, and the suite that tests this has one device it shares.
    ///
    /// # Errors
    ///
    /// As [`Shell::open`], minus [`Unopened::Absent`].
    pub fn on(
        device: Device,
        text: Text,
        deployment: Deployment,
        modules: BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, Unopened> {
        text.servable()?;
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
            modules,
            pipelines: Pipelines::new(),
            book: Book::over(shape),
            pool,
            weights,
            lowerings: crate::turns::Lowerings::default(),
            text,
            // Best first, and a device always reports at least `Baseline`.
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
    /// Bytes, not a path: see `Cargo.toml` for why a driver that depended on a
    /// checkpoint format would be a driver that could not be handed bytes.
    /// [`crate::names`] is what turns a loader's names into these.
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
    /// [`crate::turns::Unstepped`], unchanged from the layer below: this adds
    /// no refusal of its own, because everything it owns was checked when it
    /// was opened.
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
            lowerings: &mut self.lowerings,
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
    /// The two halves of a fork live in two places on purpose -- the book owns
    /// who holds which page, the pool owns what is in it -- and this is the
    /// only place that has both. `Book::fork` hands back the moves rather than
    /// performing them precisely so that a caller cannot do one half; here the
    /// list is consumed immediately.
    ///
    /// Returns how many pages were copied.
    ///
    /// # What it is for
    ///
    /// The engine's `copy_kv`. A conversation that branches -- a beam, a
    /// retry, a shared system prompt continued two ways -- otherwise pays a
    /// second prefill over tokens the cache already holds.
    ///
    /// # Errors
    ///
    /// [`Unhoused`] from the book, or [`Failed`] from a copy. A refusal from
    /// the book leaves nothing taken; a failure DURING the copy leaves `to`
    /// seated on pages holding a partial history, which is why the pages are
    /// released again before returning.
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
    /// This is the SHAPE the engine speaks, and [`Shell::fork`] is the shape a
    /// conversation has; they are different verbs on purpose. The engine's
    /// prefix cache knows which physical page it wants where and has no
    /// conversation id to name; a fork knows the conversation and not the
    /// pages. Both end at [`crate::resources::Pool::copy_rows`].
    ///
    /// The work is [`crate::resources::Pool::copy_plan`]'s, so that a test of
    /// the arithmetic does not need a model.
    ///
    /// # Why this grows the pool first, and only for destinations
    ///
    /// Because this pool is elastic and grows ON DEMAND: [`Self::admit`]
    /// raises it to the highest page a frame NAMES, so the pool holds what
    /// the frames so far have needed and not what the scheduler is entitled
    /// to hand out. A copy plan is the other way a page number arrives, and
    /// it did not carry that reasoning -- so a plan whose destination sat one
    /// page above the last frame's high-water mark was REFUSED, by a check
    /// that was right about the pool and wrong about what the pool could be.
    ///
    /// `prefix-tree-kv-cache` is what found it, in the curated sweep and only
    /// there: it needs a destination past the pages its own prefills had
    /// grown the pool to, and it failed with "page move 0's destination names
    /// page 3 row 0, and the pool has 3 pages of 16 rows" while passing
    /// whenever it ran alone. A driver that answers differently depending on
    /// which requests preceded it is the shape of defect a per-test suite
    /// cannot see.
    ///
    /// DESTINATIONS only. A source above the pool is still refused, and must
    /// be: this pool only ever grows on demand, so a page number the pool has
    /// never held is a page nothing has ever written. Growing for it would
    /// turn a refusal into a copy of freshly zeroed memory -- the same
    /// history-shaped silence the `Stranded` check exists to prevent, arrived
    /// at from the other side.
    ///
    /// # Errors
    ///
    /// See [`crate::resources::Pool::copy_plan`], plus a [`Failed`] from the
    /// growth itself. There is no `Exhausted` answer on this verb: a copy the
    /// device cannot find memory for is a failure of a copy the engine had
    /// already committed to, not a scheduling fact it can act on.
    pub fn copy_kv(&mut self, plan: &driver_api::KvCopyPlan) -> Result<usize, Failed> {
        let need = plan
            .dst_page_ids
            .iter()
            .copied()
            .chain(plan.cells.iter().map(|cell| cell.dst_page_id))
            .max()
            .map_or(0, |page| page.saturating_add(1));
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
    /// The KV one, and only it. The engine's trim task asks about three --
    /// KV, recurrent state and workspace -- on every tick, and the other two
    /// have no storage here. They are ANSWERED rather than refused, because
    /// "resize the thing that holds nothing" is satisfied by doing nothing,
    /// and a refusal would make the trim task log a failure every tick for a
    /// question it was right to ask. `driver-metal` says the same, for the
    /// same reason, and its note adds the one that matters: ignoring the id
    /// instead would resize the KV pool to the state pool's target, which is
    /// a high-water mark of zero.
    ///
    /// # But only down to nothing
    ///
    /// "Satisfied by doing nothing" is only true when nothing is what was
    /// asked for. This answered EVERY target on those ids with `Ok(())`,
    /// including a request for storage, and the engine does not treat that
    /// answer as advisory: `bootstrap`'s trim task records the target in
    /// `applied` on success and then SKIPS that pool on every later tick,
    /// because a target it has already reached is not worth re-sending. So a
    /// blanket `Ok` did not merely mislay one request, it permanently
    /// convinced the engine that a pool with no bytes behind it was holding
    /// the pages it asked for.
    ///
    /// That is the failure the capability literal refuses one seam away, in
    /// those same words -- a sink that would "bind and then run as a silent
    /// no-op, which is worse than a refusal at the door". It is worth no less
    /// here. A target of zero is still `Ok`, because zero is what this
    /// backend genuinely holds in both of those pools and what the trim task
    /// actually asks for -- workspace is asked for `0` on every tick, and
    /// state is not asked at all while `rs_cache_slot_bytes` is zero. Anything
    /// above zero, on those ids or on an id this driver has never heard of,
    /// is [`Unresized::Absent`]: a refusal the trim task retries next tick
    /// rather than a success it stops questioning.
    ///
    /// The plan's `map_ranges` and `unmap_ranges` are not read. They describe
    /// a sparse pool's commits, and this pool is not sparse -- see
    /// [`crate::resources::Pool::resize`] for why it does not need to be.
    /// `target_pages` is the whole of what this backend can act on.
    ///
    /// # Errors
    ///
    /// [`Unresized::Stranded`] if a conversation holds a page the shrink
    /// would drop -- checked BEFORE anything moves, so a refusal leaves the
    /// pool and the book exactly as they were. [`Unresized::Device`] if the
    /// allocation fails, which also leaves the pool unchanged, and the book
    /// is put back to match it. [`Unresized::Absent`] if a pool this driver
    /// does not have was asked to hold pages.
    pub fn resize_pool(&mut self, plan: &driver_api::PoolResizePlan) -> Result<(), Unresized> {
        if plan.pool_id != driver_api::PIE_ELASTIC_POOL_KV {
            if plan.target_pages == 0 {
                return Ok(());
            }
            return Err(Unresized::Absent {
                pool_id: plan.pool_id,
                target_pages: plan.target_pages,
            });
        }
        let target = u32::try_from(plan.target_pages).map_err(|_| {
            Unresized::Device(Failed::Vulkan(format!(
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
        crate::facts::of(&self.device)
    }

    /// Vulkan exports no KV handle: nothing here shares a pool across
    /// processes, and `driver-metal` answers the same way for the same reason.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<driver_api::KvHandle> {
        None
    }

    /// The device this shell runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Serve one frame from the engine.
    ///
    /// # What this does that [`Self::step`] does not
    ///
    /// Nothing to the book. The engine's scheduler owns page allocation --
    /// eviction, prefix sharing, the copy plans -- and hands down the physical
    /// pages it chose; running those through this driver's own allocator would
    /// give two allocators one page and no way to notice. See
    /// [`crate::frames`].
    ///
    /// # The order, and why it is this one
    ///
    /// * admit first, WITHOUT side effects, so a refused frame can be
    ///   re-posted rather than undone;
    /// * grow the pool to the highest page the frame NAMES, since the pool
    ///   may have been trimmed below a mark the scheduler was right to hand
    ///   out;
    /// * convert every step's CSRs BEFORE firing any of them, so a frame with
    ///   a malformed third step does not append the first two;
    /// * then fire, in the frame's own execution order, because step `n + 1`
    ///   reads the cache step `n` appended.
    ///
    /// # Errors
    ///
    /// [`Unlaunched`]. A frame the pool cannot hold is an `Ok` answer --
    /// [`Launched::Exhausted`] or [`Launched::Impossible`] -- and not an
    /// error, because a full cache is a scheduling fact rather than a fault.
    /// The two are different questions and are answered in different places:
    /// `Impossible` is the ceiling, decided before anything is attempted;
    /// `Exhausted` is the growth itself refusing, which cannot be known
    /// without trying.
    pub fn launch(&mut self, frame: &driver_api::FrameSubmission) -> Result<Launched, Unlaunched> {
        if let Some(refused) = self.admit(frame)? {
            return Ok(refused);
        }

        // Every step converted before any is fired. A frame whose third step
        // does not close its CSR would otherwise have appended the first two
        // steps' keys, and the scheduler's retry of the same frame would
        // append them twice.
        let mut work = Vec::with_capacity(frame.steps.len());
        for step in &frame.steps {
            work.push(self.prepare(&step.plan, &[], &[])?);
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
            // A resize that the device would not give memory for is
            // `Exhausted`, not a fault. The distinction is the whole reason
            // that variant exists, and until this it was produced nowhere:
            // the ceiling above is measured against a heap's SIZE, so a frame
            // that clears it can still be more than the device has FREE once
            // the weights are resident. Faulting there would fail a request
            // the scheduler could have served by evicting first -- and since
            // a driver lane that faults now answers the token with the
            // failure rather than hanging, that fault reaches the user.
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
    /// `traced` is one byte per request of `plan`, non-zero where the program
    /// resolved its own pages; see [`crate::resources::Request::traced`]. An
    /// empty slice means none did, which is every host-lowered fire.
    pub fn prepare(
        &self,
        plan: &driver_api::LaunchPlan,
        traced: &[u8],
        writes: &[Vec<(u32, u32)>],
    ) -> Result<(Vec<crate::resources::Request>, Vec<Vec<u32>>), Unlaunched> {
        plan.validate_geometry()
            .map_err(|e| Unlaunched::Malformed(format!("this frame's geometry: {e}")))?;
        plan.validate_kv_writes(self.pool.shape().page_size)
            .map_err(|e| Unlaunched::Malformed(format!("this frame's KV writes: {e}")))?;
        // Before any conversion: a plan naming something this driver does not
        // implement is refused by the field's own name rather than served
        // without it. See `frames::unserved_in`.
        if let Some(what) = crate::frames::unserved_in(plan) {
            return Err(Unlaunched::Unserved(what));
        }
        let mut requests = requests_of(plan)?;
        if !traced.is_empty() {
            if traced.len() != requests.len() {
                return Err(Unlaunched::Malformed(format!(
                    "this frame states {} traced flag(s) for {} request(s)",
                    traced.len(),
                    requests.len()
                )));
            }
            for (request, &flag) in requests.iter_mut().zip(traced) {
                request.traced = flag != 0;
            }
        }
        if !writes.is_empty() {
            if writes.len() != requests.len() {
                return Err(Unlaunched::Malformed(format!(
                    "this frame states write targets for {} request(s) and carries {}",
                    writes.len(),
                    requests.len()
                )));
            }
            for (request, stated) in requests.iter_mut().zip(writes) {
                request.writes.clone_from(stated);
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
        };
        let mut held = Held {
            book: &mut self.book,
            pool: &mut self.pool,
            weights: &self.weights,
            lowerings: &mut self.lowerings,
        };
        serving
            .over(
                &self.device,
                &mut self.pipelines,
                &self.modules,
                &mut held,
                requests,
                &borrowed,
            )
            .map_err(Unlaunched::Unstepped)
    }

    /// Encode a program's stages without firing them.
    ///
    /// Refused by name, as `driver-cuda` and `driver-metal` refuse it. There
    /// is no separate encode step in this driver: a fire records its own
    /// command buffer inside [`Self::step`], and there is nothing for a caller
    /// to hold between the two halves.
    ///
    /// # Errors
    ///
    /// Always [`Unlaunched::Unserved`].
    pub fn encode(&mut self) -> Result<(), Unlaunched> {
        Err(Unlaunched::Unserved(
            "encode: a fire records and submits in one call, so there is no \
             encoded frame to hand back",
        ))
    }

    /// Move a recurrent state between slots.
    ///
    /// Refused by name. This driver serves attention models only -- there is
    /// no recurrent-state pool to move anything between, and the plans it
    /// lowers name none.
    ///
    /// # Errors
    ///
    /// Always [`Unlaunched::Unserved`].
    pub fn copy_state(&mut self) -> Result<(), Unlaunched> {
        Err(Unlaunched::Unserved(
            "copy_state: no model this driver serves holds a recurrent state",
        ))
    }

    /// What this shell can do, now that a model is loaded.
    ///
    /// # Why the driver answers this and the engine used to
    ///
    /// Every field below except the six in [`ModelFacts`] is a statement
    /// about the DEVICE — how many pages the pool holds, which copy
    /// directions the pool serves, which sinks the kernels honour, how wide a
    /// fire the scratch can run. `engine`'s seam built the whole struct on
    /// this driver's behalf, which put a fact about Vulkan in the crate that
    /// dispatches to Vulkan, in a copy that had already drifted from the
    /// `driver-wgpu` one beside it.
    ///
    /// The checkpoint half cannot be answered here and is handed in: this
    /// crate keeps `model` and `model-loader` as dev-dependencies and
    /// `tests/pure.rs` asserts that closure, so identifying a checkpoint is
    /// not something it can do. `driver-metal` answers both halves because it
    /// identifies the checkpoint itself.
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
            // pages inside the one KV buffer -- which is what a prefix-cache
            // hit is -- and refuses any plan whose ends are not both this
            // driver's own domain. Host directions stay off: there is no swap
            // pool here, so a device-to-host copy has nowhere to land.
            kv_copy_domain_mask: driver_api::KV_COPY_DEVICE_TO_DEVICE,
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // Not elastic. `resize_pool` here restages the whole KV buffer --
            // `Pool::resize` -- so nothing can be given back page-wise, and
            // both numbers are zero together, which is the condition
            // `bootstrap` reads before it starts a trim task at all.
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
            device_geometry_port_mask: driver_api::PIE_DECODE_ENVELOPE_PORTS,
            // True here and false almost everywhere, and the difference is
            // real rather than an exemption taken to get a frame through.
            // `launch` converts ONE step, fires it, lets its program run, and
            // only then converts the next -- `envelope::fill` is called inside
            // the per-step loop, and answers `Filled::Early` for a channel
            // that is still empty. So a slot chained behind an earlier slot of
            // the same frame reads a cell that exists by the time it reads it.
            // CUDA's `FramePrepare` does every step's host work at frame entry
            // and cannot say this.
            resolves_geometry_per_step: true,
            // The ceilings a batch is formed under, and they are the arena's:
            // `Shell::open` sizes one fire's scratch, and a fire wider than
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
            // The modules are read from disk already built; nothing upstream
            // generates a kernel for this driver.
            codegen_backend: String::new(),
        }
    }

    /// The cache's shape.
    #[must_use]
    pub fn shape(&self) -> Shape {
        // ASKED, not remembered. A `Shape` field here was stale the moment
        // `resize_pool` existed, and it reported the old page count while the
        // pool held the new one -- a caller who sized a frame from it would
        // have addressed pages that were no longer there. Found by
        // `a_cache_resized_under_a_conversation_does_not_change_its_answer`.
        self.pool.shape()
    }

    /// The cache itself, read-only.
    ///
    /// The counterpart of [`Shell::book`]: that one says who holds which
    /// page, and this one is what the pages are IN. For a caller checking the
    /// cache -- a test reading a row back, an eviction proving it moved what
    /// it named -- rather than dispatching against it.
    #[must_use]
    pub fn pool(&self) -> &crate::resources::Pool {
        &self.pool
    }

    /// Who owns which page.
    #[must_use]
    pub fn book(&self) -> &Book {
        &self.book
    }

    /// How many pipelines have been built.
    ///
    /// A server's pipeline cache must stop growing; `tests/device.rs` holds
    /// that over a run of steps and needs to see the number.
    #[must_use]
    pub fn built(&self) -> usize {
        self.pipelines.built()
    }
}

impl Drop for Shell {
    /// Give the device its buffers back before destroying it.
    ///
    /// The order is the content: every resource is freed while the device is
    /// still alive, and `device` is the last field so it is dropped last
    /// anyway -- this runs before any of that.
    ///
    /// A defect found by owning a device rather than borrowing one. Every
    /// caller in this repository shared one static device that outlived the
    /// process, so nothing had ever destroyed a device with buffers still on
    /// it. The first `Shell` to drop did, and the validation layer said so:
    /// `vkDestroyDevice(): VkBuffer 0x97 has not been destroyed`. Without the
    /// layer it is a leak that grows one model's worth per shell.
    fn drop(&mut self) {
        self.weights.release(&self.device);
        self.pool.release(&self.device);
        self.pipelines.clear(&self.device);
    }
}
