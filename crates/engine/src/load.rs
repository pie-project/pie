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
/// `engine` depending on `model-compiler`, would put `CompiledModel` in the
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
    /// **THE SECOND ROW AXIS'S CEILING, OR "DERIVE ONE"** (multimodal §5.5).
    ///
    /// The most patch rows one fire may carry. `None` — the default — means
    /// the shell derives a ladder from the loaded TEXT: a plan that states no
    /// `Dim::Patches` gets no ladder at all (which is G4's invariant, and the
    /// literal `None` every pre-tower SKU has always had), and a plan that
    /// states one gets rungs at whole images from `PATCH_LATTICE_FLOOR` up to
    /// a ceiling read off the token rectangle. So a vision SKU serves with
    /// zero configuration and a text-only SKU is untouched.
    ///
    /// **A PLAIN INTEGER AND NOT A `PatchLadder`**, because this crate deps
    /// `model-ir` and not `model-compiler`: the ladder is the compiler's type
    /// and the ceiling is the operator's number, which is the same split
    /// [`max_tokens`](Budgets::max_tokens) and `Budget::buckets` already have.
    /// A shell turns the two numbers below into a ladder the way
    /// `bake_budgets` turns the six above into a `Budget`.
    #[serde(default)]
    pub max_patches: Option<u32>,
    /// **THE PATCH AXIS'S `max_lanes`**, or `None` to derive it (multimodal
    /// §5.5): the most IMAGES one fire may carry, over every lane in it.
    ///
    /// Not derivable from [`max_patches`](Budgets::max_patches) by the shell's
    /// own doctrine — an image contributes at least one patch row, so reading
    /// it off that number would size the `images + 1` indptr at the patch
    /// ceiling — but a DEFAULT may be argued from it, and the shell argues
    /// one. An operator who has measured their traffic states this instead.
    #[serde(default)]
    pub max_images: Option<u32>,
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
            // **DERIVE, WHICH FOR A TEXT-ONLY PLAN IS `None`** — the same
            // answer this field's absence has always given, and the reason
            // `#[serde(default)]` keeps an older caller's request meaning
            // what it meant.
            max_patches: None,
            max_images: None,
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
    /// write-back — so its host demand is those banks whole.
    ///
    /// **AND A BUDGET UNDER THAT NO LONGER ENDS THE CONVERSATION** (alto
    /// streaming §2, wave W-6). What a host budget cannot hold may still be
    /// SERVED, out of the third tier: the warm-boot artifact, mapped, whose
    /// pages a GPU touch faults in from NVMe. So "over the host budget" is a
    /// refusal only when this load has no such source —
    /// [`Residency::admit_tiers`] is where that distinction lives, and
    /// [`Tiers::sourced`] is the bit that decides it.
    pub host_weight_budget: Option<u64>,
}

/// **What a PLANNED load will hold, tier by tier** — what
/// [`Residency::admit_tiers`] is asked with (alto streaming §2).
///
/// Three numbers and one bit, because the tiers are three and the third one
/// exists conditionally:
///
/// ```text
/// T0 device   the slab and every dense plane            `device_weight_budget`
/// T1 pinned   what the device does not hold, page-locked `host_weight_budget`
/// T2 mapped   what NEITHER holds, in the artifact        (no budget: it is a file)
/// ```
///
/// T2 has no budget because it is not a reservation — it is a mapping of a
/// file that already exists, and the only question about it is whether it
/// exists at all. That is [`Tiers::sourced`], and it is the difference
/// between a load this machine can serve slowly and one it cannot serve.
///
/// # What this type does NOT account for, said here so it is not assumed
///
/// **The elastic pool and the safety floor**, and that is still true and is
/// no longer a gap. Alto streaming §3 item 5 wants one sentence — *weight
/// tiers + elastic pool + safety floor = the card* — and these three numbers
/// are only its first term. The sentence is written down, but in the SHELL
/// and not here: `engine_cuda::store::Accounting` states *card, ceiling,
/// weights, floor, pool, minimum* and refuses ahead of every allocation, and
/// `[engine] gpu_mem_utilization` reaches `PhysicalPool::open` (`next.md` B1
/// and B2). It belongs there rather than here for the reason this statute is
/// three numbers and a bit: what a card has, what a driver needs held back
/// and what a pool must cover are all device facts, and a portable statute
/// that named them would be naming a machine it cannot see. What crosses is
/// what always crossed — the two budgets — and the refusal the shell adds is
/// `Error::Impossible` beside these, drawn on the same statute/physics line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tiers {
    /// T0: the weight bytes this load will hold on the device.
    pub device: u64,
    /// T1: the weight bytes it will hold in pinned host memory.
    pub host: u64,
    /// T2: the weight bytes NEITHER budget holds, to be read from a mapped
    /// source over HMM. Zero for every load that fits its two budgets, which
    /// is every load in this workspace that states no budget at all.
    pub spilled: u64,
    /// **Is there a T2 source for `spilled` to come out of?**
    ///
    /// Answered by the SHELL, because only the shell knows whether the
    /// artifact for this recipe is on disk — the statute here does not know
    /// what a file is. `false` with a non-zero `spilled` is the load that
    /// cannot be served, and it is the one new refusal this type exists for.
    pub sourced: bool,
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
    ///
    /// **THE TWO-TIER DOOR, KEPT.** A shell that plans onto T0 and T1 and
    /// nothing else asks here and reads the same answer it always did; the
    /// third tier arrives through [`Residency::admit_tiers`], which this
    /// delegates to with nothing spilled.
    pub fn admit(&self, device_demand: u64, host_demand: u64) -> crate::Result<()> {
        self.admit_tiers(Tiers {
            device: device_demand,
            host: host_demand,
            spilled: 0,
            sourced: false,
        })
    }

    /// **Does this policy admit a load planned across all three tiers?**
    /// (alto streaming §2, wave W-6.)
    ///
    /// [`Residency::admit`]'s pair, grown by one answer. The two budget
    /// refusals are word for word the ones above — a demand past a budget is
    /// `Impossible`, because the plan handed here is already what the engine
    /// reduced to. What is new is the THIRD outcome, and it is the whole of
    /// why streaming §2 exists:
    ///
    /// ```text
    /// spilled == 0                  the two-tier load; the pair decides
    /// spilled > 0 &&  sourced       ADMITTED — the artifact holds them, and
    ///                               a GPU touch faults each page in from NVMe
    /// spilled > 0 && !sourced       `Impossible`, naming the bytes and the
    ///                               one thing that would change the answer
    /// ```
    ///
    /// The third arm is `Impossible` and not
    /// [`Exhausted`](crate::Error::Exhausted) for the same reason the first
    /// two are: freeing device or host memory does not conjure a file. What
    /// WOULD change it is running `pie model import --prepare-only` on this
    /// box, and the sentence says so rather than leaving the operator to infer
    /// it. **A SECOND ROAD COUNTS TOO**: a shell whose weight cache holds this
    /// deployment's serving artifact is sourced by it at any budget (§M.3), so
    /// a prepared deployment carries its own source from then on. Which files
    /// a shell will look at is the shell's to decide; this statute only asks
    /// whether it found one.
    ///
    /// **AND IT STILL ADMITS ON THE BOOTSTRAP ALONE, DELIBERATELY** (§M-3).
    /// The whole-table artifact an uncapped boot wrote is not something a
    /// warm-only SERVE can be built from — the shell refuses one screen later,
    /// by name, with the same remedy — but it is exactly what a PREPARE of a
    /// spilled deployment reads its spilled planes out of, and `Cuda::prepare`
    /// is admitted by this statute too. A rule that demanded the serving
    /// artifact here would refuse the one run that creates it.
    ///
    /// **AND `Exhausted` IS STILL REACHABLE, ELSEWHERE AND CORRECTLY.** A
    /// budget this statute admits and an allocation the DEVICE then refuses is
    /// `Fault::OutOfMemory` -> `Error::Exhausted` with both numbers — a pool
    /// that is full, which another deployment's exit can fix. The pair is
    /// exact: statute is `Impossible` here, physics is `Exhausted` there, and
    /// no path answers both.
    ///
    /// # Errors
    ///
    /// [`Error::Impossible`](crate::Error::Impossible) for a demand past
    /// either budget, or for spilled bytes with no source to spill to.
    pub fn admit_tiers(&self, tiers: Tiers) -> crate::Result<()> {
        for (budget, demand, tier, field) in [
            (
                self.device_weight_budget,
                tiers.device,
                "device",
                "device_weight_budget",
            ),
            (
                self.host_weight_budget,
                tiers.host,
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
        // ── THE THIRD ANSWER. Bytes that fit neither budget are not a
        //    refusal by themselves: streaming §2's whole claim is that a model
        //    larger than device AND host memory combined can still be served,
        //    slowly, out of a mapping. What refuses is spilled bytes with
        //    nowhere to spill FROM.
        if tiers.spilled > 0 && !tiers.sourced {
            return Err(crate::Error::Impossible(format!(
                "weight residency: this load plans {} bytes onto the third tier — \
                 weights that neither `device_weight_budget` nor `host_weight_budget` \
                 holds — and it has no T2 source to read them out of. The source is a \
                 weight artifact, which is a snapshot of plane images and therefore \
                 needs no conversion to serve from (alto streaming §0); this \
                 deployment either states no weight cache directory or has never been \
                 prepared on this machine. TWO FILES COUNT AS A SOURCE: the SERVING \
                 artifact `pie model import` writes, which carries every plane of the \
                 trace at a budget-free ranking and is what a serve reads; and the \
                 whole-table artifact an UNCAPPED boot leaves behind, which is what a \
                 PREPARE of a spilled deployment reads its spilled planes out of. So: \
                 run `pie model import --prepare-only <checkpoint>` on this box — \
                 booting it once uncapped first if this deployment has never had a \
                 machine large enough to hold it whole — or raise one of the budgets, \
                 or state `None`.",
                tiers.spilled,
            )));
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
    /// `crate::runahead::Runahead` at the shell's, and every pool the shell
    /// carves for run-ahead derives from it rather than re-declaring a depth
    /// of its own. It is stated at LOAD and not per frame because what it
    /// sizes — the staging ring — is carved once and never grows (article 7:
    /// the fire path allocates nothing).
    ///
    /// A shell clamps what it cannot serve rather than refusing the load; the
    /// deployment's config layer is where an out-of-range depth is named
    /// (`crate::runahead::Runahead::MAX_FRAMES`). Zero reads as one.
    pub frames_in_flight: u8,
    /// **THE TENSOR-PARALLEL DEGREE THIS LOAD IS FOR**, which a serving
    /// artifact states and a shell must be able to compare against.
    ///
    /// A catalog fact — the row's own width — and not a fact the shell can
    /// reach: a shell must not know a model family (design §7, decision 18),
    /// which is why `engine-cuda`'s and `engine-metal`'s edges to `models` are
    /// both DEV. So the runtime, which does see the catalog, states it here.
    ///
    /// `#[serde(default)]` reads as 1, which is what every request meant
    /// before the field existed and what an unsharded load means now.
    #[serde(default = "one_rank")]
    pub tp_size: u64,
    /// **THE SERVED NUMERIC FORM THIS LOAD IS FOR** — the field that makes one
    /// model at two quantizations two artifacts.
    ///
    /// Here for [`tp_size`](LoadRequest::tp_size)'s reason and no other: it is
    /// `models::precision_of`'s answer, and a shell cannot ask. A shell builds
    /// `checkpoint::serving::Stamp::of` out of this, `tp_size`, the trace's
    /// `name` and `platform`, and checks the artifact against it before a
    /// plane lands.
    ///
    /// Empty is not a precision. A request that carries one is a request the
    /// runtime could not assemble, and a shell should refuse rather than skip
    /// the check — an artifact that passes because nobody stated what to
    /// compare it to is the silent failure the stamp exists to end.
    #[serde(default)]
    pub precision: String,
}

/// [`LoadRequest::tp_size`]'s serde default: one rank.
fn one_rank() -> u64 {
    1
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

#[cfg(test)]
mod residency_tests {
    use super::{Residency, Tiers};

    fn capped(device: u64, host: u64) -> Residency {
        Residency {
            device_weight_budget: Some(device),
            host_weight_budget: Some(host),
        }
    }

    #[test]
    fn the_two_tier_door_answers_exactly_what_it_always_did() {
        let policy = capped(1_000, 500);
        assert!(policy.admit(1_000, 500).is_ok(), "a demand at both budgets lands");
        assert!(policy.admit(1_001, 0).is_err(), "one byte past the device budget");
        assert!(policy.admit(0, 501).is_err(), "one byte past the host budget");
        assert!(
            Residency::uncapped().admit(u64::MAX, u64::MAX).is_ok(),
            "uncapped admits everything"
        );
    }

    #[test]
    fn spilled_bytes_with_a_source_are_admitted_and_without_one_are_impossible() {
        let policy = capped(1_000, 500);
        let planned = |spilled, sourced| Tiers {
            device: 1_000,
            host: 500,
            spilled,
            sourced,
        };
        // THE THIRD ANSWER. Both budgets are met exactly and there are bytes
        // left over; whether that is a load or a refusal is one bit.
        assert!(
            policy.admit_tiers(planned(4_000, true)).is_ok(),
            "bytes neither budget holds are SERVED when a source holds them — \
             that sentence is streaming §2's reason to exist"
        );
        let refused = policy
            .admit_tiers(planned(4_000, false))
            .expect_err("and refused when nothing does");
        let said = format!("{refused}");
        assert!(said.contains("4000"), "the refusal names the bytes: {said}");
        assert!(
            said.contains("third tier"),
            "and which tier they wanted: {said}"
        );
        assert!(
            said.contains("pie model import --prepare-only"),
            "and the one thing that would change the answer — which since §M-3 \
             is a command and not a differently-configured boot: {said}"
        );
        assert!(
            said.contains("uncapped"),
            "and the one case where a boot still comes into it, because a \
             deployment that has never been held whole has nothing for a \
             prepare to read its spilled planes out of: {said}"
        );
        assert!(
            matches!(refused, crate::Error::Impossible(_)),
            "statute, not exhaustion: freeing memory does not conjure a file"
        );
    }

    #[test]
    fn a_budget_refusal_beats_a_spill_refusal_to_the_answer() {
        // Order matters for the sentence an operator reads: a load that is
        // over its DEVICE budget is over its device budget, and telling it
        // about the third tier first would send it looking for a file it does
        // not need.
        let policy = capped(1_000, 500);
        let why = policy
            .admit_tiers(Tiers {
                device: 9_999,
                host: 0,
                spilled: 4_000,
                sourced: false,
            })
            .expect_err("over the device budget");
        assert!(
            format!("{why}").contains("device_weight_budget"),
            "{why}"
        );
    }

    #[test]
    fn nothing_spilled_is_never_a_refusal_however_sourceless() {
        assert!(
            capped(1_000, 500)
                .admit_tiers(Tiers {
                    device: 1_000,
                    host: 500,
                    spilled: 0,
                    sourced: false,
                })
                .is_ok(),
            "a two-tier load is not asked about a tier it does not use"
        );
    }
}
