//! `LoadRequest` in, `Loaded` out — the one door a model comes through.
//!
//! # The `Trace` crosses; `CompiledModel` never does (decision 18)
//!
//! The RUNTIME traces the model. It links `model` anyway — `Classify::of` runs
//! per lane on the fire path to compute [`Lane::word`](crate::fire::Lane::word)
//! — so the supergraph is already in its address space, and handing it across
//! costs a `serde` round trip a remote engine was going to need regardless.
//!
//! What does NOT cross is `model_compiler::CompiledModel`: the region template, the
//! class table, the arena carve. Those are the SHELL's, produced by
//! `compile(trace, budgets, profile)` on the far side of this boundary, because
//! they are answers about a device the runtime does not have. The consequence
//! is the one the design wants: a remote engine is a `serde`-able `Trace` on a
//! socket and nothing else, and the compiler never has to be portable.
//!
//! ```text
//!  runtime                         |  shell
//!  -------                         |  -----
//!  model/ forward -> Trace  -------|--> compile(trace, budgets, profile) -> CompiledModel
//!  Classify::of(req) -> word       |    record one graph per bucket
//!                                  |    land the checkpoint
//!  Step{lanes} ----------|--> compose -> walk -> replay
//! ```
//!
//! # What died here
//!
//! `ModelLoadDesc` was `{ snapshot_dir: PathBuf, runtime_quant: String,
//! mxfp4_moe: Mxfp4MoeRequest, component: ModelComponent }` and `load_model`
//! took a `Vec` of them. Every field of it was a way of saying something the
//! plan now says outright:
//!
//! * `snapshot_dir` -> [`LoadRequest::checkpoint`], unchanged in substance.
//! * `runtime_quant: String` — a backend-parsed quantization *word*. The
//!   plan's params carry their own dtypes; a string the shell string-matched
//!   is not a second opinion worth keeping.
//! * `mxfp4_moe: Mxfp4MoeRequest` — four ways to run one op, chosen at load by
//!   name. Which kernel answers an op is the dispatch arm's decision (design
//!   §6), and the axis it was really selecting is a model-declared one.
//! * `component: ModelComponent{Full,Text,Encode}` — WHICH graph to load, by
//!   enum. It is now which `Trace` you hand over: the encoder is a traced plan
//!   like any other, and `Vec<ModelLoadDesc>` collapses to one request per
//!   plan.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

/// The ceilings a load is baked against.
///
/// The same four numbers `model_compiler::Budget` states, carried across the
/// boundary because the compile happens on the far side of it and a shell that
/// invented its own ceilings would bake a graph the runtime cannot fill.
///
/// A duplicate spelling is exactly what decision 1 kills, so this is written
/// once here and converted by the shell in one place — the alternative,
/// `engine-api` depending on `model-compiler`, would put `CompiledModel` in the
/// dependency graph of `transport` and `controller-api`, which is the edge
/// decision 18 exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// The most requests one fire may carry. `Dim::Lanes` is this number.
    pub max_lanes: u32,
    /// The most token rows one fire may carry. `Dim::Tokens` is this number.
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to — one immutable
    /// graph per entry. Ascending, each entry at most `max_tokens`.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds (design §8).
    pub max_adapters: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub max_context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
}

impl Default for Budgets {
    /// 256 lanes, 8192 rows, 16-token pages, 4096 tokens of context, 256
    /// slots: a deployment that runs, for a caller who has measured nothing.
    fn default() -> Budgets {
        Budgets {
            max_lanes: 256,
            max_tokens: 8192,
            buckets: Vec::new(),
            max_adapters: 0,
            page_size: 16,
            max_context: 4096,
            slots: 256,
        }
    }
}

/// Where a load's weights come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    /// A snapshot directory, or one container file.
    Path(PathBuf),
    /// No weights: bind the device and bake the plan, but land nothing. What a
    /// shape-only smoke test loads, and what makes a `CompiledModel` inspectable
    /// without a checkpoint on the machine.
    None,
}

/// **How much of the weight table this load may keep, stated as two budgets
/// rather than a mode** (alto design §7).
///
/// Weights are a cache hierarchy and not a setting:
///
/// ```text
/// T2 SSD      the checkpoint on disk, plus the warm-boot artifact cache
/// T1 pinned   host cache, budgeted   <- `host_weight_budget`
/// T0 device   slab, budgeted         <- `device_weight_budget`
/// ```
///
/// "Full residency" is the DEGENERATE CASE of that hierarchy — the T0 budget
/// covers everything — and not a separate arm, which is why there is no
/// `enum ResidencyMode` here. A mode would have had to name every combination
/// the tiers can be in, and the tiers are two numbers; two numbers name them
/// all, including the ones nobody has built yet.
///
/// `None` is UNCAPPED, and uncapped on both is exactly today's behaviour: the
/// whole table lands on the device at load and never moves. A caller that
/// states neither budget gets the engine it had before this field existed,
/// which is why [`Default`] is both-`None` and why the field is
/// `#[serde(default)]` on [`LoadRequest`].
///
/// # A budget an engine cannot meet by holding less refuses; it never
/// silently holds more
///
/// Which planes an engine can hold LESS of is a property of the planes, not a
/// setting. The CUDA shell streams the demand shape design §7 calls dynamic —
/// ROUTED EXPERT BANKS, whose residency is a performance promotion because
/// routing is computed on device and no host decision precedes a fire — by
/// keeping a device slab of a few experts over a pinned host copy of all of
/// them, behind a device-resident indirection table the kernels read. Under
/// such a budget the load SERVES, [`LoadFacts::weights_resident`] answers
/// `false`, and the numbers are the numbers full residency would have
/// produced.
///
/// Everything else still refuses. The static demand shape — dense overflow,
/// whose prefetch schedule the compiler emits — is design §7's other half and
/// is not built; the Metal shell holds one tier and streams nothing. So a
/// budget under a plan's dense planes, or under a plan with no routed bank at
/// all, is [`Error::Impossible`](crate::Error::Impossible), naming BOTH
/// numbers: what was asked for and what must stay resident. That is F1's
/// doctrine — an unbuilt combination refuses loudly rather than being rounded
/// to the nearest built one — and it is why the noun landed before the
/// machinery: the same field that refused every budget in wave D1 now admits
/// the ones a tier can meet, and nothing above this line changed to let it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residency {
    /// The most weight bytes this load may hold on the DEVICE (tier T0).
    ///
    /// `None` is uncapped: land the whole table. `Some(n)` is a ceiling met
    /// by holding fewer ROUTED EXPERTS, which is the one thing an engine here
    /// can hold fewer of; an engine that cannot reach `n` that way refuses the
    /// load rather than pretending.
    pub device_weight_budget: Option<u64>,
    /// The most weight bytes this load may hold in the PINNED HOST cache
    /// (tier T1) — the tier a device miss reads over UVA instead of stalling
    /// on the checkpoint.
    ///
    /// `None` is uncapped. An engine that holds everything on the device
    /// keeps zero host-resident weight bytes and every host budget admits it;
    /// one that streams routed experts pins EVERY expert of every streamed
    /// bank — the pinned copy is authoritative and the device slab is a cache
    /// over it, which is what makes a demotion a table entry rather than a
    /// write-back — so its host demand is those banks whole, and a budget
    /// under that refuses like any other.
    pub host_weight_budget: Option<u64>,
}

impl Residency {
    /// Both budgets uncapped — the whole table resident, which is what every
    /// load in this workspace does today.
    #[must_use]
    pub const fn uncapped() -> Residency {
        Residency {
            device_weight_budget: None,
            host_weight_budget: None,
        }
    }

    /// True when neither tier is capped, i.e. the degenerate full-residency
    /// case a shell may serve by doing nothing special.
    #[must_use]
    pub const fn is_uncapped(&self) -> bool {
        self.device_weight_budget.is_none() && self.host_weight_budget.is_none()
    }

    /// **Does this policy admit a checkpoint that demands these bytes
    /// resident?**
    ///
    /// The one gate a shell calls, with the demand it has ALREADY PLANNED —
    /// what the tiers it has will actually hold, not what the checkpoint
    /// contains. A shell that streams routed experts asks about the slab plus
    /// the dense planes and about the pinned tier beside it; a shell that
    /// streams nothing asks about the whole table and zero. Uncapped admits
    /// everything; a stated budget admits what fits under it and refuses what
    /// does not, by name and with both numbers in the message.
    ///
    /// # Errors
    ///
    /// [`Error::Impossible`](crate::Error::Impossible) when either demand is
    /// past its budget. `Impossible` and not
    /// [`Exhausted`](crate::Error::Exhausted) on purpose: nothing the
    /// deployment frees changes the answer, because the refusal is about a
    /// tier this build does not have rather than about a pool that is full.
    pub fn admit(&self, device_demand: u64, host_demand: u64) -> crate::Result<()> {
        for (budget, demand, tier, field) in [
            (
                self.device_weight_budget,
                device_demand,
                "device",
                "device_weight_budget",
            ),
            (
                self.host_weight_budget,
                host_demand,
                "pinned host",
                "host_weight_budget",
            ),
        ] {
            if let Some(budget) = budget {
                if demand > budget {
                    return Err(crate::Error::Impossible(format!(
                        "weight residency: `{field}` is {budget} bytes and this load demands \
                         {demand} bytes on the {tier} tier. That demand is what the engine \
                         has already reduced to as far as its tiers allow — routed expert \
                         banks stream, dense planes do not (alto design §7) — so the budget \
                         cannot be met by holding less of it. Raise the budget, or state \
                         `None` for uncapped."
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Everything a load states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    /// The traced supergraph. The runtime traces it (decision 18).
    pub trace: model_ir::Trace,
    /// Where the weights are.
    pub checkpoint: Checkpoint,
    /// The ceilings every fire is baked against.
    pub budgets: Budgets,
    /// **How much of the weight table this load may keep resident**, per tier
    /// (alto design §7). Both budgets `None` — the [`Default`] — is the
    /// uncapped, whole-table-on-the-device load every caller had before the
    /// field existed, which is why it is `#[serde(default)]`: an older
    /// caller's request still parses and still means what it meant.
    #[serde(default)]
    pub residency: Residency,
    /// Which device to bind, when the shell serves more than one.
    pub ordinal: i32,
    /// **How many frames the caller will keep in flight** — the one run-ahead
    /// number, crossing once (article 8).
    ///
    /// `[runtime] frame_dispatch_depth` at the deployment's end,
    /// `engine::runahead::Runahead` at the shell's, and every pool the shell
    /// carves for run-ahead derives from it rather than re-declaring a depth
    /// of its own. It is stated at LOAD and not per frame because what it
    /// sizes — the staging ring — is carved once and never grows (article 7:
    /// the fire path allocates nothing).
    ///
    /// A shell clamps what it cannot serve rather than refusing the load; the
    /// deployment's config layer is where an out-of-range depth is named
    /// (`engine::runahead::Runahead::MAX_FRAMES`). Zero reads as one.
    pub frames_in_flight: u8,
}

/// What a load answers with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    /// The facts this load carries — the plan's own name and the residency it
    /// achieved.
    pub facts: LoadFacts,
    /// What it can do.
    pub caps: Capabilities,
}

/// What came of a load, as numbers a caller can log and act on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadFacts {
    /// The trace's own name, as the model text declared it.
    pub trace_name: String,
    /// Bytes the weight tables occupy on the device.
    pub weight_bytes: u64,
    /// **Is the WHOLE weight table device-resident?** (alto design §7.)
    ///
    /// The residency half of the answer
    /// [`weight_bytes`](LoadFacts::weight_bytes) gives the size of. `true`
    /// says every param this plan names is on the device and no fire will
    /// ever wait on a promotion; `false` says the load is streaming some tier
    /// of it, and `weight_bytes` is then what is RESIDENT rather than what the
    /// checkpoint holds.
    ///
    /// `true` whenever the request's [`Residency`] budgets are uncapped, or
    /// cover the whole table — the degenerate case design §7 calls full
    /// residency. `false` is what the CUDA shell answers for a load whose
    /// `device_weight_budget` it met by streaming routed expert banks: the
    /// logits are the logits full residency would have produced, and what
    /// differs is only how much of the table is on the device at once.
    pub weights_resident: bool,
    /// **Did this load's weight table come off a warm-boot artifact cache
    /// instead of the checkpoint?** (alto design §7's T2 tier.)
    ///
    /// The materialized device weight table is a deterministic function of
    /// the checkpoint, the load contract compiled against it, and the layout
    /// the shell chose — so an engine may snapshot it after a cold load and
    /// read it straight back on the next boot, skipping the host-side
    /// transform pipeline entirely. `true` says that is what happened.
    ///
    /// **`false` IS ALSO WHAT AN ENGINE WITH NO CACHE SAYS**, and it says it
    /// honestly rather than by omission: a shell that has no artifact cache
    /// answers `false` on every load, which is the same answer a shell with
    /// one gives on a cold boot. A caller reading this learns what happened
    /// to THIS load, not what the engine is capable of — which is the
    /// division [`Capabilities`] and [`LoadFacts`] have from the start.
    ///
    /// `#[serde(default)]` so an engine written before the tier existed still
    /// deserializes, reporting the truth about itself.
    #[serde(default)]
    pub weights_from_cache: bool,
    /// Bytes the activation arena occupies.
    pub arena_bytes: u64,
    /// Bytes the pools occupy.
    pub pool_bytes: u64,
    /// Bytes the resident fire inputs occupy.
    pub input_bytes: u64,
    /// **Bytes of the pools actually under a physical mapping** (alto design
    /// §8, article 8: one number, one owner).
    ///
    /// `pool_bytes` is the CEILING the pools' address space was reserved at;
    /// this is what is backed right now. On an engine whose pools are elastic
    /// the two differ by design — the load commits to demand and grows — and
    /// on one whose pools are a single reservation they are equal.
    ///
    /// The engine owns physical commit and trim, so the engine is what
    /// answers this. It replaced a runtime-side scan of its own page free
    /// list, which was a policy structure being asked a supply question.
    #[serde(default)]
    pub pool_committed_bytes: u64,
    /// **The most that has ever been mapped**, since load. What a trim is
    /// measured against, and what an operator sizing the next machine wants
    /// rather than either the ceiling or the instantaneous figure.
    #[serde(default)]
    pub pool_high_water_bytes: u64,
}
