//! **The routed-expert tier, on unified memory**: a wired slab smaller than
//! the bank, a host band table beside it, and a SEGMENT CUT between the
//! router that decides and the matmuls that read (alto design §7, wave W-a).
//!
//! ```text
//! T0 wired    `slots` expert seats of every streamed band, budgeted  `device_weight_budget`
//! source      every expert of every streamed band, host bytes        a mapping the kernel
//!                                                                    may reclaim
//!                                                                    (`crate::host_source`)
//! ```
//!
//! # There is no T1 here, and that is the platform rather than a shortcut
//!
//! The CUDA sibling's ladder has three rungs because a device address and a
//! host address are different addresses: an expert the slab does not hold is
//! read over UVA out of PINNED memory, and the pinned copy is a real tier
//! with a real budget. On Apple silicon the GPU and the CPU address the same
//! bytes, so a "pinned host copy the device can read" is not a second tier —
//! it is the same tier under another name. The honest host demand of a
//! streamed load on this plane is therefore **zero**, and
//! [`Plan::host_demand`] answers zero for every plan, streamed or not.
//!
//! What replaces the missing rung is a SWAP. The kernels read
//! `base + e * stride` and nothing else (`moe_select.metal`'s
//! `bank + (e * out_width + out_row) * in_width`, `quant_qmv.metal`'s
//! `ws += e * out_vec_size * in_vec_size_w`), so a slab of `slots` seats
//! serves a bank of `experts` the moment `e` names a SEAT rather than an
//! EXPERT — which is a rewrite of the router's own output vector, on the
//! host, between two command buffers. No kernel changes. That is dev's
//! `expert_slab.hpp` / `expert_paging.hpp` mechanism, restated for a shell
//! whose banks are split-plane.
//!
//! # A SEAT IS FOUR BANDS, NOT ONE PLANE
//!
//! This shell seats `WeightRow::Planes` — a quantized bank is codes, scales
//! and (for an affine codec) zero points, three device planes under one
//! `Def::Weight` — and a routed matmul with a bias reads a FOURTH plane the
//! same way (`bias_row += expert_ids[sel] * out_vec_size`). Every one of them
//! is indexed by the number in the routing vector, so rewriting that number
//! moves all four or it corrupts three of them. A band is therefore not the
//! unit of residency: the unit is the GROUP — every expert-indexed param read
//! against ONE router's `routes` vector — and a group's seat `s` means seat
//! `s` of every band in it.
//!
//! Holding the scale planes whole while streaming the codes is not a cheaper
//! version of this. It is wrong: `scales += e * …` with `e` a seat number
//! reads another expert's factors, and the model computes and is wrong.
//!
//! # Where the swap can happen, and why it is not a frame boundary
//!
//! Routing is computed ON DEVICE. The host learns which experts a fire wants
//! only by reading the router's output, and that output does not exist until
//! the router has run — so the swap cannot ride a frame boundary the way a
//! promotion does on the CUDA plane. It rides a SEGMENT boundary: the walk of
//! a streamed load is cut into `N + 1` command buffers, one cut immediately
//! after each mixture layer's router, each segment closed with a BLOCKING
//! [`Frame::commit`](crate::device::ctx::Frame::commit). Between two segments
//! the host reads the routing vector out of the arena (`StorageModeShared`,
//! so it is a `memcpy` and not a transfer), seats what the segment ahead will
//! read, and rewrites the vector in place.
//!
//! **THE BLOCKING COMMIT IS WHAT MAKES THE MEMCPY LEGAL.** A wired slab is
//! bytes some already-committed command buffer may still be reading; there is
//! no fence in this shell and no second copy of the weight store, so the only
//! thing that can prove "nothing is reading seat `s`" is that everything
//! committed before this instant has COMPLETED. `commit()` proves exactly
//! that. It also prices the whole mechanism, and the price is named in
//! [`serve`](crate::serve)'s header: on a streamed load the run-ahead
//! collapses to one, and articles 1 and 2 are false there by construction.
//!
//! # What a segment may ask for, and what it is refused for
//!
//! Every distinct expert one segment routes to must be seated AT ONCE, because
//! the segment's matmuls all run after its cut. So the slab must seat at least
//! the distinct expert count of one fire's routing for that layer — top_k for
//! a one-token decode, and up to the whole bank for a wide prefill. A segment
//! that asks for more seats than the slab has is [`Fault::Residency`] naming
//! both numbers; the fix is a larger budget or fewer tokens per fire, and the
//! mechanism that would make it neither — splitting one segment's TOKENS into
//! sub-batches, dev's `expert_paging` — is not this wave's.
//!
//! # What this wave does NOT do, named so that the next one is not a surprise
//!
//! * **The source is a mapping this shell writes, not the artifact itself.**
//!   Every streamed band is still held whole on the host — what changed is
//!   WHERE: [`crate::host_source`] backs it with an unlinked temporary file,
//!   so the kernel may write the pages back and reclaim them instead of the
//!   process pinning a whole routed bank in anonymous memory. That bounds the
//!   term; it does not remove it, and it is not yet the artifact. Mapping the
//!   `.zt` directly needs the load's own transforms out of the way — the
//!   checkpoint's Metal tile mask admits `CAST|SCALE|DECODE|BIAS`
//!   (`checkpoint`'s `plan::passes::tile`), so what a band LANDS as is not in
//!   general what the file HOLDS, and a mapping of the artifact would be a
//!   mapping of the wrong bytes. Hoisting the mmap door into `checkpoint` and
//!   planning for an untransformed band is the rest of W-b.
//! * **The admission arithmetic asks the machine, but the tier still does
//!   not.** `Context::working_set` IS plumbed now — `api.rs` binds a
//!   throwaway context at admission, prices the kv pool and the dense floor
//!   against the card's own working set, and shrinks the budget this plan is
//!   formed at when the stated one is over the ceiling. What is still open is
//!   origin/dev's other half: its rule for the transient window is "one
//!   window, not one model" (`forward.cpp`), and nothing here bounds a
//!   segment's window against what is RECLAIMABLE at the instant it runs.
//! * **One segment's tokens are not split.** Every distinct expert a segment
//!   routes to is pinned at once, so a wide prefill over a small slab is
//!   refused rather than served in pieces. dev's `expert_paging` splits the
//!   TOKEN batch to bound that set; the refusal below names the two numbers
//!   and the two fixes a deployment has today.
//! * **A cut region may not split its window.** `Fallback::Split` runs a
//!   region once per interval, and each firing would rewrite rows an earlier
//!   interval already rewrote. Refused by name in
//!   [`Shell::walk_streamed`](crate::serve::Shell).
//!
//! # Full residency is the degenerate case and costs nothing
//!
//! [`Plan::of`] with an uncapped budget, or with a budget that covers the
//! whole table, answers the empty plan: [`Plan::streams`] is false, no band
//! is diverted, no [`Tier`] is opened, `walk_once` takes the path it always
//! took and the fire path is byte for byte the one that fired before this
//! file existed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kernels_metal::Tensor;
use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

use crate::device::{Buffer, Handles};
use crate::error::{Fault, Result};
use crate::host_source::HostSource;
use crate::mapping::Mapping;

/// A param's other device planes, by `Trace::params` index — the pairing
/// `weights::pairings` reads off the load plan, in the one shape this module
/// needs it: "which OTHER params move when this one's expert moves".
///
/// Empty for a dense bank, one entry for a symmetric quantized one, two for
/// an affine one. Keyed by the CODE plane's index, which is the param a
/// routed matmul's `bank` port names.
pub type Attachments = BTreeMap<usize, Vec<usize>>;

/// **What a plan wants and what a budget allows**, decided off the trace and
/// the load plan's pairings alone — before the device store is reserved and
/// before a byte is landed.
///
/// The empty plan is full residency and is what an uncapped
/// [`Residency`](engine::load::Residency) produces. Everything downstream
/// reads [`Plan::streams`] and does nothing when it is false.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// One entry per streamed BAND, in param order.
    bands: Vec<BandPlan>,
    /// One entry per streamed GROUP, in router order.
    groups: Vec<GroupPlan>,
    /// `param index -> how many experts of it the slab seats`. The one map
    /// [`weights::places`](crate::weights) consults to reserve a bank at less
    /// than its declared size.
    resident_of: BTreeMap<usize, u32>,
    /// `param index -> where its whole plane lives in the host band table`.
    host_of: BTreeMap<usize, u64>,
    /// How many seats every group's slab has. Zero for the empty plan.
    slots: u32,
    device_bytes: u64,
    host_bytes: u64,
}

/// One streamed band, as the plan sees it: a param whose leading axis is the
/// expert count, sliced by the number a routing vector carries.
#[derive(Debug, Clone)]
pub struct BandPlan {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The param's own name, which is the plan's and the contract's.
    pub name: String,
    /// The bank's leading axis.
    pub experts: u32,
    /// How many of them the wired slab seats.
    pub slots: u32,
    /// One expert's bytes — the seat stride, uniform across the band.
    pub stride: u64,
    /// Which [`GroupPlan`] this band moves with.
    pub group: usize,
}

/// One streamed group: a router, and every expert-indexed param read against
/// the vector it writes.
///
/// **THE GROUP IS THE UNIT OF RESIDENCY AND THE BAND IS NOT**, for the reason
/// this module's header gives: one rewrite of `routes` re-indexes every band
/// in the group at once, so they seat and evict together or three of them
/// read somebody else's expert.
#[derive(Debug, Clone)]
pub struct GroupPlan {
    /// The routing vector this group's bands are indexed by — the `routes`
    /// output of the `Linear::MoeTopk*` node that decides them, and the
    /// buffer the segment cut rewrites.
    pub routes: ValueId,
    /// How many experts the router declares.
    pub experts: u32,
    /// How many of them the slab seats.
    pub slots: u32,
    /// Indices into [`Plan::bands`], ascending by param.
    pub bands: Vec<usize>,
}

impl Plan {
    /// **The residency plan for `trace` under `budget`.**
    ///
    /// `None` — uncapped — answers the empty plan: land everything, open no
    /// tier, cut no segment. A stated budget is met by seating fewer EXPERTS
    /// and never by holding fewer dense planes.
    ///
    /// `planes` is the load plan's pairing, in [`Attachments`]' shape: it is
    /// what makes a quantized bank's scales and zero points part of the seat
    /// rather than a plane left behind. It is READ rather than derived,
    /// because a suffix convention is exactly what pairs a weight with
    /// somebody else's scales the day a checkpoint ships a tensor named
    /// `…scales` (`weights::pairings` argues it whole).
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a param whose dtype has no element size or whose
    /// band shape breaks the uniformity proof; [`Fault::Residency`] for a
    /// budget that cannot hold the dense planes plus one seat of every band,
    /// naming both numbers, and for a capped budget over a plan with nothing
    /// routed to hold less of.
    pub fn of(trace: &Trace, planes: &Attachments, budget: Option<u64>) -> Result<Plan> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full: u64 = bytes
            .iter()
            .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let Some(budget) = budget else {
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        };
        if budget >= full {
            // The budget covers the whole table: the degenerate case, and it
            // is answered as the degenerate case rather than as a streamed
            // load that happens to seat everything.
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        }

        let (mut bands, mut groups) = found(trace, planes, &bytes)?;
        if bands.is_empty() {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes and this plan's weight table \
                 demands {full}. Nothing in it is a routed-expert bank, so there is no \
                 tier to hold less of: alto design §7 streams the DYNAMIC demand shape \
                 (routed experts, whose seat is chosen after the router has run) and the \
                 static one (dense overflow, a compiler-emitted prefetch schedule) is \
                 not built. Raise the budget, or state `None` for uncapped."
            )));
        }

        // ── THE DENSE FLOOR. Every plane that is not a streamed band is held
        //    whole, so its reserved bytes are the floor under any budget this
        //    plan can serve. A budget below it is refused with both numbers,
        //    and it is `Residency` — nothing the deployment frees changes the
        //    answer.
        let streamed: BTreeSet<usize> = bands.iter().map(|band| band.param).collect();
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !streamed.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        // The seat strides, lifted out of `bands` so that the sweep below can
        // read them while the plan is being written back into the bands
        // themselves.
        let strides: Vec<u64> = bands.iter().map(|band| band.stride).collect();
        let seats = |n: u32| -> u64 {
            strides
                .iter()
                .map(|stride| (u64::from(n) * stride).next_multiple_of(crate::weights::ALIGN))
                .sum()
        };
        let floor = dense + seats(1);
        if budget < floor {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes; this plan's DENSE planes \
                 demand {dense} resident and its {} routed bands need one expert seat \
                 each on top, which is {floor} before a second expert is seated. Dense \
                 planes do not stream in this build (alto design §7: their demand is \
                 static and its prefetch schedule is a later wave), so the budget cannot \
                 be met by holding less. Raise it to at least {floor}, or state `None`.",
                bands.len(),
            )));
        }

        // How many experts every group seats, ONE number for the plan.
        // Monotone in `n`, and `experts` is at most a few hundred, so it is
        // walked rather than searched — and walked DOWN from the whole bank so
        // that a budget one byte under full residency still lands on the
        // largest count it can hold.
        let experts = groups[0].experts;
        let slack = budget - dense;
        let mut slots = 0u32;
        for n in (1..=experts).rev() {
            if seats(n) <= slack {
                slots = n;
                break;
            }
        }
        debug_assert!(slots >= 1, "the floor check above proved one seat fits");

        for band in &mut bands {
            band.slots = slots;
        }
        for group in &mut groups {
            group.slots = slots;
        }
        let resident_of = bands.iter().map(|band| (band.param, slots)).collect();
        // The host band table's layout, decided here so that the landing sink
        // and the tier read one arithmetic. Bands in param order, each whole.
        let mut host_bytes = 0u64;
        let mut host_of = BTreeMap::new();
        for band in &bands {
            host_of.insert(band.param, host_bytes);
            host_bytes += u64::from(band.experts) * band.stride;
        }
        Ok(Plan {
            device_bytes: dense + seats(slots),
            bands,
            groups,
            resident_of,
            host_of,
            slots,
            host_bytes,
        })
    }

    /// Does this load stream any band?
    #[must_use]
    pub fn streams(&self) -> bool {
        !self.bands.is_empty()
    }

    /// The bands it streams, in param order.
    #[must_use]
    pub fn bands(&self) -> &[BandPlan] {
        &self.bands
    }

    /// The groups it streams, in router order.
    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    /// How many seats every group's slab has — `0` for a full-residency plan.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// How many experts of `param` the slab seats, or `None` for a param held
    /// whole — which is every param of a full-residency load and every dense
    /// plane of a streamed one.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    /// Where `param`'s whole plane lives in the host band table, or `None`
    /// for a param that lands on the device.
    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.host_of.get(&param).copied()
    }

    /// **What this plan demands of the device**, in bytes — what
    /// [`Residency::admit`](engine::load::Residency::admit) is asked with, and
    /// what the weight store will actually occupy.
    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes
    }

    /// **What this plan demands of a host tier: zero, always.**
    ///
    /// Not "not measured" and not "not implemented" — zero is the true
    /// number on unified memory, and this module's header is where the
    /// sentence lives. The source bytes a seat is copied FROM are host bytes
    /// the process would hold either way — and since W-b they are a mapping
    /// the kernel may take back ([`crate::host_source`]) rather than a
    /// `Vec<u8>` it cannot — and no byte of them is a second copy the device
    /// reads through.
    #[must_use]
    pub fn host_demand(&self) -> u64 {
        let _ = self.host_bytes;
        0
    }

    /// How many bytes the host band table holds — the size
    /// [`HostSource::open`](crate::host_source::HostSource::open) backs with a
    /// file for the COLD arm.
    ///
    /// **IT IS THE STAGING FILE'S SIZE AND NOT "WHAT THE SEAT COPIES READ",**
    /// and the two parted company at §M-6. A warm load streams out of the
    /// serving artifact it already mapped ([`Source::artifact`]) and opens no
    /// staging file at all, so this number is what it WOULD have written and
    /// nothing it did. [`Tier::source`] is what a load's actual source
    /// answers for.
    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.host_bytes
    }
}

/// **The streamed bands and groups a trace declares, with the uniformity
/// proof checked.**
///
/// A band is a param some routed op reads at an expert-indexed port — stated
/// by the OP and not by a naming convention — plus, for a quantized bank, the
/// planes the load plan pairs with it. The scan is over `Trace::nodes`
/// because that is where the reading is; `Trace::params` says only what
/// exists.
fn found(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BandPlan>, Vec<GroupPlan>)> {
    // ── The routers, first: a group cannot exist without one, and the
    //    expert count is the ROUTER's field. No operand of a select op
    //    carries it (`crate::scratch::routers` states the same fact for the
    //    same reason), so the routing vector is what the two halves are
    //    joined on.
    let mut arity: BTreeMap<u32, u32> = BTreeMap::new();
    let mut order: Vec<ValueId> = Vec::new();
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, experts) = match op {
            Linear::MoeTopkSoftmax {
                routes, experts, ..
            }
            | Linear::MoeTopkSoftmaxScaled {
                routes, experts, ..
            }
            | Linear::MoeTopkSigmoid {
                routes, experts, ..
            }
            | Linear::MoeTopkSqrtSoftplus {
                routes, experts, ..
            }
            // **AND THE LOOKUP ROUTER IS ONE OF THEM.** It reads no logits, so
            // it is not a `MoeTopk*` by name — but it lands the same `routes`
            // vector the selects behind it are indexed by, and dsv4-flash's
            // first `num_hash_layers` layers route ONLY this way. Omitted, its
            // groups are routed reads whose router this scan does not state,
            // and `Plan::of` refuses the whole model rather than the layer:
            // no DeepSeek-V4-Flash load could stream at any budget.
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if arity.insert(routes.0, experts).is_none() {
            order.push(routes);
        }
    }

    // ── The expert-indexed reads, joined onto the router that wrote their
    //    routing vector. `MoeBiasSum` is on the list beside the three select
    //    ops because it indexes its `bias` by the SAME vector, after the fold
    //    — a plane left un-swapped there is the routed bias of another
    //    expert, added to a correct sum.
    let mut of_group: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, indexed) = match op {
            Linear::MoeMatmulSelect { bank, routes, .. }
            | Linear::MoeMatmulSelectQuant { bank, routes, .. } => (*routes, vec![*bank]),
            Linear::MoeMatmulSelectBias {
                bank, bias, routes, ..
            } => (*routes, vec![*bank, *bias]),
            Linear::MoeBiasSum { bias, routes, .. } => (*routes, vec![*bias]),
            _ => continue,
        };
        if !arity.contains_key(&routes.0) {
            return Err(Fault::Residency(format!(
                "value {} is read as a routing vector by `{}` and no router node of this \
                 plan writes it; the expert count a seat is divided out of is the \
                 ROUTER's field, so a routed read whose router this plan does not state \
                 cannot be seated at less than its declared size",
                routes.0,
                op.name(),
            )));
        }
        let seats = of_group.entry(routes.0).or_default();
        for id in indexed {
            let at = weight_of(trace, id)?;
            if !seats.contains(&at) {
                seats.push(at);
            }
            for &plane in planes.get(&at).into_iter().flatten() {
                if !seats.contains(&plane) {
                    seats.push(plane);
                }
            }
        }
    }

    let mut bands: Vec<BandPlan> = Vec::new();
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut owner: BTreeMap<usize, ValueId> = BTreeMap::new();
    let mut declared: Option<u32> = None;
    for routes in order {
        let Some(mut params) = of_group.remove(&routes.0) else {
            // A router whose vector nothing expert-indexed reads is not a
            // group: there is no band to hold less of, and the vector is
            // still written and still read by the weighted sum.
            continue;
        };
        params.sort_unstable();
        let experts = arity[&routes.0];
        // ── THE UNIFORMITY PROOF, RESTATED FOR A PLANE (dev's, off the
        //    plan): every band of one group divides evenly into its experts,
        //    and every group of one plan states the same expert count — so
        //    one residency decision covers the plan rather than one per layer.
        match declared {
            None => declared = Some(experts),
            Some(first) if first != experts => {
                return Err(Fault::Param {
                    name: trace.params[params[0]].name.clone(),
                    why: "is a routed band whose expert count differs from an earlier \
                          group of the same plan; one residency decision covers the plan, \
                          and two arities would make it two decisions",
                });
            }
            Some(_) => {}
        }
        let mut of_this = Vec::with_capacity(params.len());
        for at in params {
            if let Some(other) = owner.insert(at, routes) {
                if other != routes {
                    return Err(Fault::Param {
                        name: trace.params[at].name.clone(),
                        why: "is expert-indexed by two different routing vectors; a seat \
                              number means one group's seat, and a band shared between two \
                              groups would be re-indexed twice",
                    });
                }
            }
            let param = &trace.params[at];
            let leading = u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
            if leading != experts || param.shape.len() < 2 {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is read as a routed expert band and does not declare \
                          `[experts, ...]` at the router's own expert count; a seat stride \
                          cannot be divided out of it",
                });
            }
            let plane = bytes[at];
            if plane == 0 || plane % u64::from(experts) != 0 {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert band whose bytes do not divide by its expert \
                          count — the experts of one band are not equal, and the seat \
                          arithmetic the tier does would be wrong rather than refused",
                });
            }
            of_this.push(bands.len());
            bands.push(BandPlan {
                param: at,
                name: param.name.clone(),
                experts,
                slots: experts,
                stride: plane / u64::from(experts),
                group: groups.len(),
            });
        }
        groups.push(GroupPlan {
            routes,
            experts,
            slots: experts,
            bands: of_this,
        });
    }
    Ok((bands, groups))
}

/// The `Trace::params` row a value id names, or a refusal.
fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read at a routed matmul's expert-indexed port and is not a weight; a \
                  band is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

/// **Where the walk is cut**, one entry per region of the compiled template:
/// the routing vector the router in that region writes, or `None`.
///
/// **THE CUT IS DERIVED FROM THE TRACE AND NOT FROM A KERNEL NAME.** What a
/// segment boundary has to be is "after the node that decides, before the
/// nodes that read", and the node that decides is a `Linear::MoeTopk*` — an
/// IR fact, stable across every codec and every family. The region index is
/// how the encode sink asks the question at fire time, because that is the
/// one coordinate `crate::window::Cursor` puts in its hand.
///
/// # Errors
///
/// [`Fault::Residency`] when `streams` and a region holds two routers: the
/// cut would then fall after BOTH of them and the first mixture's matmuls
/// would already be encoded against un-swapped seats. The refusal names the
/// region rather than serving it wrong.
pub fn cuts(
    trace: &Trace,
    compiled: &CompiledModel,
    streams: bool,
) -> Result<Vec<Option<ValueId>>> {
    let mut out = Vec::with_capacity(compiled.template().len());
    for (at, region) in compiled.template().iter().enumerate() {
        let mut here: Option<ValueId> = None;
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            let routes = match op {
                Linear::MoeTopkSoftmax { routes, .. }
                | Linear::MoeTopkSoftmaxScaled { routes, .. }
                | Linear::MoeTopkSigmoid { routes, .. }
                | Linear::MoeTopkSqrtSoftplus { routes, .. }
                // The lookup router decides a mixture exactly as the ranked
                // four do, so the cut falls after it for the same reason.
                | Linear::MoeHashRoute { routes, .. } => *routes,
                _ => continue,
            };
            if let Some(first) = here {
                if first != routes && streams {
                    return Err(Fault::Residency(format!(
                        "region {at} holds two routers (values {} and {}), and a streamed \
                         load cuts its command buffer after EACH one — a single cut behind \
                         both would encode the first mixture's matmuls against seats the \
                         host had not swapped yet. Raise `device_weight_budget` to hold \
                         this plan whole, or bake an artifact whose regions carry one \
                         mixture each.",
                        first.0, routes.0
                    )));
                }
                continue;
            }
            here = Some(routes);
        }
        out.push(here);
    }
    Ok(out)
}

/// One band, seated: where its experts are on both sides, and how wide a seat
/// is.
#[derive(Debug)]
struct Band {
    /// The param's own name, for a refusal that names it.
    name: String,
    /// Byte offset of seat 0 inside the device weight store.
    at: u64,
    /// Byte offset of expert 0 inside the host band table.
    from: u64,
    /// One expert's bytes.
    stride: u64,
}

/// One group's wired slab: which expert is in which seat, and which seats the
/// segment in flight may not lose.
#[derive(Debug)]
struct Slab {
    experts: u32,
    slots: u32,
    bands: Vec<Band>,
    /// `expert -> seat`, or `None` for an expert the slab does not hold.
    seat_of: Vec<Option<u32>>,
    /// `seat -> expert`, or `None` for a seat nothing has been copied into.
    in_seat: Vec<Option<u32>>,
    /// **PINNED: a seat the segment being built will read.** Pins are taken
    /// as a segment's routing is seated and released at the NEXT cut, which
    /// is after a blocking commit — so a pin's life is exactly "the device
    /// may still be reading this".
    pinned: Vec<bool>,
    /// The clock's reference bit: set when a seat is hit, cleared as the hand
    /// sweeps past it.
    used: Vec<bool>,
    /// The clock hand.
    hand: u32,
}

/// **What one group's residency looks like from outside** — the accessor a
/// gate reads, and the only observable a swap has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupResidency {
    /// The group's first band, by name — what a reader recognises it as.
    pub name: String,
    /// How many experts the router declares.
    pub experts: u32,
    /// How many of them the slab seats.
    pub slots: u32,
    /// Which expert is in which seat, ascending by seat; `None` for a seat
    /// nothing has been copied into yet.
    pub in_seat: Vec<Option<u32>>,
}

/// The host bytes themselves, under whichever arm produced them.
#[derive(Debug)]
enum Bytes {
    /// The cold arm's staging: a `MAP_SHARED` mapping of an unlinked
    /// temporary file, written once by the landing sink.
    Landed(HostSource),
    /// The warm arm's: the serving artifact's own `PROT_READ` mapping — the
    /// same `Arc<Mapping>` the resident planes are bound through.
    Artifact(Arc<Mapping>),
}

/// **WHERE A SEAT COPY READS FROM**, and the whole of the difference between
/// the two load arms.
///
/// A seat copy is `store.write(seat, &source[expert])` and nothing else, so
/// what the tier needs of a source is exactly two things: a `&[u8]` the CPU
/// may read, and the byte offset of expert 0 of every streamed band inside
/// it. The two arms answer them differently and share nothing else:
///
/// * [`Source::landed`] — **the COLD arm.** The landing sink wrote every
///   expert of every band into a [`HostSource`], laid out in the plan's own
///   host-table order: bands in param order, each whole, packed contiguously
///   from zero. The offsets are [`Plan::host_at`]'s, which is the one
///   arithmetic the sink and the tier share.
/// * [`Source::artifact`] — **the WARM arm.** The bytes are already on this
///   machine, in the serving artifact the load has just mapped, and staging
///   them into a second file to read them back out of it would be the copy
///   the warm arm exists to remove. So the source IS the artifact's mapping,
///   and the offsets come per-band out of the serving manifest.
///
/// # The per-band table IS the translation, and it is why there is no base
///
/// The two layouts are different on purpose. The host table packs bands
/// CONTIGUOUSLY in param order because nothing else ever reads it; the
/// artifact lays its planes out in the SERVING order — hotness-ranked, at the
/// writer's own alignment, after a manifest — with every other plane of the
/// model interleaved between them. There is no offset, no scale and no
/// permutation that carries one into the other, and a tier that walked either
/// with one base and a running sum would, on the other, read somebody else's
/// experts and answer finite deterministic nonsense.
///
/// So NEITHER arm gets a base. Both state `param -> byte offset of expert 0`
/// outright, and what survives as shared arithmetic is only the step INSIDE a
/// band — `offset + expert * stride` — which is a fact about the band's own
/// rectangle and is true wherever the band lies.
///
/// # And the GPU touches neither
///
/// Both are CPU-read-only sources of the copy INTO the wired slab, which is
/// what `.wiki/alto/streaming.md`'s measurement demands: a GPU-touched shared
/// page WIRES and the pager takes none of it back, so a band the device read
/// in place would convert the whole bank into wired memory and defeat the
/// budget that made this load stream.
///
/// That the artifact's mapping is ALSO bound to an `MTLBuffer` does not
/// weaken it, because binding is not touching (`crate::mapping`'s header
/// carries the measurement, and §M-5 measured +0.001 GiB wired at the bind).
/// What wires a page is a kernel addressing it, and no weight row of a warm
/// streamed load names a band's file offset: a band's `Tensor` is a view into
/// the STORE, at its seats. The pages this enum reads are read by `memcpy`
/// and by nothing else.
#[derive(Debug)]
pub struct Source {
    bytes: Bytes,
    /// `param index -> byte offset of expert 0 of that band`, in `bytes`.
    bands: BTreeMap<usize, u64>,
}

impl Source {
    /// The cold arm's source: the staging file the landing sink filled, at the
    /// plan's own host-table offsets.
    #[must_use]
    pub fn landed(plan: &Plan, host: HostSource) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands: plan.host_of.clone(),
        }
    }

    /// The warm arm's source: the mapped artifact, at the per-band offsets
    /// `bands` recovered from its serving manifest.
    ///
    /// `map` is the SAME `Arc` the resident planes are bound through — one
    /// mapping of one file, and no second copy of the bands anywhere.
    #[must_use]
    pub fn artifact(map: Arc<Mapping>, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Artifact(map),
            bands,
        }
    }

    /// **What this source is made of**, as one word — the observable half of
    /// the two arms, and the reason [`Source::backing`]'s second number reads
    /// differently under each.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.bytes {
            Bytes::Landed(_) => "landed",
            Bytes::Artifact(_) => "artifact",
        }
    }

    /// **What is actually behind these bytes**: `(the backing file's size, its
    /// link count)`, or `None` for a source with no file at all.
    ///
    /// The two arms answer the SECOND number differently and both answers are
    /// the truth about a different file. A `landed` source says `(source
    /// bytes, 0)`: a mapping of a real file this process created and unlinked,
    /// which is what makes its pages reclaimable and what nothing outside this
    /// process can reach. An `artifact` source says `(the artifact's bytes,
    /// its links)` — at least one, because the artifact is a file the operator
    /// named and this load did not create — and its pages are reclaimable for
    /// the plainer reason that they are clean: this arm never wrote one.
    ///
    /// A gate that wants "the source is not anonymous memory" reads the pair;
    /// one that wants to know WHICH claim it is holding reads [`Source::kind`]
    /// first. Asserting `0` links without asking the kind is asserting that
    /// the load was cold.
    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        match &self.bytes {
            Bytes::Landed(host) => host.backing(),
            Bytes::Artifact(map) => Some((map.backing()?, map.links()?)),
        }
    }

    /// Where expert 0 of `param`'s band lies in these bytes.
    fn at(&self, param: usize) -> Option<u64> {
        self.bands.get(&param).copied()
    }

    /// `len` bytes at `from`, or `None` for a span that leaves the source.
    fn get(&self, from: usize, len: usize) -> Option<&[u8]> {
        let all: &[u8] = match &self.bytes {
            Bytes::Landed(host) => host,
            Bytes::Artifact(map) => map,
        };
        all.get(from..from.checked_add(len)?)
    }

    /// How many bytes the source holds — the bound a refusal names.
    fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Landed(host) => host.len() as u64,
            Bytes::Artifact(map) => map.len(),
        }
    }

    /// **Hand the source to the pager**, once, after the landing and the
    /// identity prefix have finished with it.
    ///
    /// For a `landed` source this is [`HostSource::settle`]'s two calls, and
    /// they are argued there.
    ///
    /// For an `artifact` source it is deliberately NOTHING, and that is not an
    /// omission. `msync` schedules writeback for DIRTY pages and this arm
    /// dirtied none — the mapping is `PROT_READ`, so its pages are clean the
    /// instant they fault in and dropping one already costs nothing. And a
    /// `MADV_DONTNEED` here would be advice about an address range this
    /// process has ALSO bound to an `MTLBuffer` for the resident planes: the
    /// bands' pages are not that buffer's business, but the advice is a range
    /// and the wired frames beneath the resident planes are inside it. There
    /// is no reclaim to buy — the pages are already clean — so there is
    /// nothing worth aiming a range call at a live buffer for.
    fn settle(&mut self) {
        match &mut self.bytes {
            Bytes::Landed(host) => host.settle(),
            Bytes::Artifact(_) => {}
        }
    }
}

/// **The tier**: the host band table, a wired slab per group, and the seat
/// bookkeeping between them.
///
/// It holds a RETAIN of the weight store rather than a borrow of it, which is
/// the one place this file reaches around Rust's aliasing and is worth naming.
/// A `Buffer` is a retained `MTLBuffer` and cloning one is a retain, not a
/// copy — the same bytes under a second owner. The tier writes SEAT spans and
/// the weight table reads HANDLE rows that name the same reservation, and what
/// keeps them from colliding is not the borrow checker: it is that every seat
/// write happens between two command buffers, after a blocking commit, with no
/// encode in flight.
#[derive(Debug)]
pub struct Tier {
    /// A retain of the weight store — where seats live.
    store: Buffer,
    /// Every expert of every streamed band, in a file-backed mapping — the
    /// cold arm's staging file or the warm arm's artifact ([`Source`] argues
    /// the two and the per-band offsets that separate them).
    ///
    /// **THE CPU IS THE ONLY THING THAT READS IT.** It is the `&[u8]` source
    /// of the seat copies in [`Tier::copy`] and no kernel addresses a byte of
    /// it, because a mapped page the GPU touches WIRES and stops being
    /// reclaimable at all (`.wiki/alto/streaming.md`, measured); the module
    /// header of [`crate::host_source`] states the whole rule and
    /// [`Source`]'s says why the warm arm's artifact does not break it.
    source: Source,
    slabs: Vec<Slab>,
    /// `routes value -> slab index`.
    of_routes: BTreeMap<u32, usize>,
    /// How many seat copies this load has done.
    swaps: u64,
    /// How many segment cuts this load has taken.
    segments: u64,
}

impl Tier {
    /// Open the tier `plan` describes, over a weight store whose streamed
    /// bands are reserved at `slots` seats, reading experts out of `source`.
    ///
    /// `offsets` is `param index -> byte offset of seat 0 inside the store`,
    /// which is the layout of whichever reservation the calling arm made;
    /// `source` states the other side of every copy and where each band's
    /// expert 0 lies in it. **THE TWO ARE INDEPENDENT AND ARE MEANT TO BE** —
    /// that is the whole of what lets one tier serve a cold load out of its
    /// staging file and a warm one out of the artifact it is already mapping.
    ///
    /// **THE INITIAL SEATING IS THE IDENTITY PREFIX** — seat `i` holds expert
    /// `i` — and it is copied in here rather than left to the first segment,
    /// so that a fire whose routing happens to land inside the prefix costs no
    /// copy at all. The alternative, an empty slab, would make the FIRST
    /// segment of every load its most expensive one for no gain.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a seat span that leaves the store or a source
    /// span that leaves the source, [`Fault::Residency`] for a band the
    /// source states no offset for, [`Fault::Deviceless`] off Apple.
    pub fn open(plan: &Plan, store: &Buffer, source: Source, offsets: &[u64]) -> Result<Tier> {
        let mut tier = Tier {
            store: store.clone(),
            source,
            slabs: Vec::with_capacity(plan.groups.len()),
            of_routes: BTreeMap::new(),
            swaps: 0,
            segments: 0,
        };
        for (at, group) in plan.groups.iter().enumerate() {
            let bands = group
                .bands
                .iter()
                .map(|&band| {
                    let band = &plan.bands[band];
                    // A band the source cannot place is a source and a plan
                    // that were not built from each other, and seating it at
                    // a guessed offset is the failure this whole type is
                    // arranged to make impossible. It is refused by name.
                    let from = tier.source.at(band.param).ok_or_else(|| {
                        Fault::Residency(format!(
                            "the seat source states no offset for band `{}` (param {}), \
                             which this plan streams — the residency plan and the bytes \
                             behind it were not built from each other",
                            band.name, band.param,
                        ))
                    })?;
                    Ok(Band {
                        name: band.name.clone(),
                        at: offsets[band.param],
                        from,
                        stride: band.stride,
                    })
                })
                .collect::<Result<Vec<Band>>>()?;
            tier.of_routes.insert(group.routes.0, at);
            tier.slabs.push(Slab {
                experts: group.experts,
                slots: group.slots,
                bands,
                seat_of: vec![None; group.experts as usize],
                in_seat: vec![None; group.slots as usize],
                pinned: vec![false; group.slots as usize],
                used: vec![false; group.slots as usize],
                hand: 0,
            });
        }
        for at in 0..tier.slabs.len() {
            for seat in 0..tier.slabs[at].slots {
                tier.copy(at, seat, seat)?;
            }
        }
        // **THE COUNTER STARTS AFTER THE PREFIX.** [`Tier::motion`] answers
        // "did the mechanism move anything while SERVING", and a load-time
        // fill counted there would make that question answer yes on a load
        // that never routed outside its prefix.
        tier.swaps = 0;
        // ── **AND THE SOURCE IS HANDED TO THE PAGER HERE**, at the one instant
        //    where it is fully written and fully read: the landing filled it,
        //    the prefix above has just copied out of it, and nothing will read
        //    it again until a segment routes outside the prefix. Before this
        //    point every page is DIRTY and a dirty page cannot be reclaimed
        //    without a writeback the pager would have to schedule under
        //    pressure; after it they are clean and deactivated, and a seat copy
        //    that wants one faults it back from the file it already lives in.
        //    [`Source::settle`] argues the two calls, and why the warm arm's
        //    artifact needs neither of them.
        tier.source.settle();
        Ok(tier)
    }

    /// **One segment cut**: read the routing vector the region just wrote,
    /// seat every expert it names, and rewrite it in place to name seats.
    ///
    /// `rect` is the routing vector's FIRE-WIDE rectangle as the carve
    /// placed it, `span` the window the region ran at — so the rows touched
    /// are exactly the rows the router just wrote and the rows the matmuls
    /// behind this cut will read. A row outside the window still holds the
    /// PREVIOUS segment's seat numbers and is deliberately left alone:
    /// rewriting it would read a seat number as an expert number.
    ///
    /// The caller has already committed and waited for everything encoded
    /// before this point; that is what makes the seat copies below legal and
    /// what makes releasing the previous segment's pins the first thing this
    /// does.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a routing vector this fire minted no handle
    /// row for, [`Fault::Residency`] for a segment that routes to more
    /// distinct experts than the slab seats or to an expert the router does
    /// not declare, [`Fault::Ceiling`] for a span that leaves a reservation.
    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        span: MaskSpan,
    ) -> Result<()> {
        let Some(&at) = self.of_routes.get(&routes.0) else {
            return Ok(());
        };
        // ── The previous segment has completed (the caller's blocking
        //    commit), so nothing on the device is reading a seat any more.
        for seat in &mut self.slabs[at].pinned {
            *seat = false;
        }
        self.segments += 1;
        if span.rows == 0 {
            return Ok(());
        }
        let width = u64::from(rect.width);
        let base = {
            let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the routing vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, routes.0
                ),
            })?;
            row.offset()
        };
        let first = base + u64::from(span.row_offset) * width * 4;
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let mut raw = vec![0u8; count * 4];
        arena.read(first, &mut raw)?;
        // Seat, then rewrite, in one pass: an id repeated inside the segment
        // hits the seat the earlier occurrence took, which is what makes a
        // wide fire cost `distinct experts` copies rather than `routes`.
        for entry in raw.chunks_exact_mut(4) {
            let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id < 0 {
                continue;
            }
            let expert = id as u32;
            if expert >= self.slabs[at].experts {
                return Err(Fault::Residency(format!(
                    "a routing vector names expert {expert} and `{}` declares {} of them; \
                     a seat cannot be found for an expert the router does not have. This \
                     is a routing vector read at the wrong instant — the segment cut ran \
                     against bytes some other segment wrote.",
                    self.slabs[at].bands[0].name, self.slabs[at].experts
                )));
            }
            let seat = self.seat(at, expert)?;
            entry.copy_from_slice(&(seat as i32).to_le_bytes());
        }
        arena.write(first, &raw)?;
        Ok(())
    }

    /// Where expert `expert` of slab `at` sits, seating it if it is not
    /// seated — and pinning it either way, because the segment being built is
    /// about to read it.
    fn seat(&mut self, at: usize, expert: u32) -> Result<u32> {
        if let Some(seat) = self.slabs[at].seat_of[expert as usize] {
            let slab = &mut self.slabs[at];
            slab.used[seat as usize] = true;
            slab.pinned[seat as usize] = true;
            return Ok(seat);
        }
        let seat = self.evict(at)?;
        if let Some(held) = self.slabs[at].in_seat[seat as usize] {
            self.slabs[at].seat_of[held as usize] = None;
        }
        self.copy(at, seat, expert)?;
        let slab = &mut self.slabs[at];
        slab.used[seat as usize] = true;
        slab.pinned[seat as usize] = true;
        Ok(seat)
    }

    /// **The clock**, over the unpinned seats of one slab.
    ///
    /// A second-chance sweep rather than a true LRU: the ordering an LRU
    /// needs is a per-seat timestamp updated on every route, and what the
    /// eviction actually has to answer is only "which seat is this segment
    /// least likely to want again". The reference bit answers that at one
    /// bool per seat, and the hand is where the last sweep stopped — so a
    /// slab whose working set fits stops moving entirely, which is the
    /// property that matters on a decode stream.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] when every seat is pinned: this segment routes to
    /// more distinct experts than the slab has room for, which is the one
    /// shape of this mechanism that cannot be served by copying harder.
    fn evict(&mut self, at: usize) -> Result<u32> {
        let slots = self.slabs[at].slots;
        // Two sweeps at most: the first clears reference bits, the second
        // finds a seat whose bit the first cleared.
        for _ in 0..(2 * slots) {
            let seat = self.slabs[at].hand;
            self.slabs[at].hand = (seat + 1) % slots.max(1);
            let slab = &mut self.slabs[at];
            if slab.pinned[seat as usize] {
                continue;
            }
            if slab.used[seat as usize] {
                slab.used[seat as usize] = false;
                continue;
            }
            return Ok(seat);
        }
        Err(Fault::Residency(format!(
            "one segment of this fire routes to more than {slots} distinct experts of \
             `{}`, and the wired slab seats {slots}: every seat is pinned by a matmul \
             this same segment will run, so no seat can be reused. Every expert one \
             segment reads must be resident at once — raise `device_weight_budget`, or \
             fire fewer tokens per step. Splitting one segment's tokens into sub-batches \
             is the mechanism that would make this neither, and it is not in this build.",
            self.slabs[at].bands[0].name
        )))
    }

    /// Copy expert `expert` into seat `seat` of slab `at` — every band of it,
    /// which is what makes one seat number mean one expert across the codes,
    /// the factors, the zero points and the routed bias.
    ///
    /// **THE OVERWRITE IN PLACE IS WHAT BOUNDS WIRED RESIDENCY, AND IT IS LOAD-
    /// BEARING.** The measurement in `.wiki/alto/streaming.md` proved that a
    /// GPU-touched `StorageModeShared` page stays WIRED until its buffer is
    /// released or explicitly `madvise(MADV_DONTNEED)`'d — the pager will not
    /// evict it under pressure. This shell needs neither of those because it
    /// never GROWS the wired set: the slab is one fixed reservation of `slots`
    /// seats inside `self.store`, carved to the budget at load, and a segment's
    /// eviction is a `write` OVER an existing seat's bytes (`band.at + seat *
    /// stride`), not a new allocation and not a re-created buffer. The seat's
    /// pages were wired on their first touch and are reused for every expert
    /// that lands in that seat thereafter, so the wired footprint is exactly the
    /// slab — the budget — and never the cumulative set of experts the load has
    /// touched. A future change that reallocated or extended the slab per
    /// segment would silently unbound this; the whole point of the seat clock
    /// (`evict`) is to stay inside the fixed berths instead.
    fn copy(&mut self, at: usize, seat: u32, expert: u32) -> Result<()> {
        for band in 0..self.slabs[at].bands.len() {
            let (into, from, stride) = {
                let band = &self.slabs[at].bands[band];
                (
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(expert) * band.stride,
                    band.stride,
                )
            };
            let from = usize::try_from(from).unwrap_or(usize::MAX);
            let len = usize::try_from(stride).unwrap_or(usize::MAX);
            // **ONE COPY, NOT TWO.** The band used to be `.to_vec()`'d out of
            // the host table and the temporary handed to `write` — a whole
            // expert band allocated, filled and freed per seat copy, for no
            // reason but to end a borrow of `self.source` before `self.store`
            // was borrowed mutably. The two fields are disjoint, so the
            // reborrow below says that directly and the allocation is gone:
            // what remains is exactly the copy that IS the mechanism, from
            // the reclaimable mapping into the wired slab.
            let source = self.source.get(from, len).ok_or_else(|| Fault::Ceiling {
                what: "bytes of the seat source",
                need: (from + len) as u64,
                have: self.source.len(),
            })?;
            self.store.write(into, source)?;
            self.swaps += 1;
        }
        self.slabs[at].in_seat[seat as usize] = Some(expert);
        self.slabs[at].seat_of[expert as usize] = Some(seat);
        Ok(())
    }

    /// Every group's occupancy, in plan order — what a gate asserts on.
    #[must_use]
    pub fn residency(&self) -> Vec<GroupResidency> {
        self.slabs
            .iter()
            .map(|slab| GroupResidency {
                name: slab.bands[0].name.clone(),
                experts: slab.experts,
                slots: slab.slots,
                in_seat: slab.in_seat.clone(),
            })
            .collect()
    }

    /// `(band copies, segment cuts)` since the load — the two numbers that
    /// say whether the mechanism moved.
    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.swaps, self.segments)
    }

    /// **What the source is actually made of**: `(the backing file's size, its
    /// link count)`, or `None` when this tier has no file behind it at all.
    ///
    /// The observable half of the reclaimability claim, and the only one a
    /// gate can hold honestly — but it is TWO claims now and the pair alone
    /// does not say which, so read [`Tier::source_kind`] beside it.
    /// `("landed", source_bytes, 0)` is the cold arm: a mapping of a real,
    /// unlinked file nothing outside this process can reach.
    /// `("artifact", artifact_bytes, links >= 1)` is the warm one: the
    /// serving artifact itself, which this load did not create and will not
    /// unlink, and whose pages are reclaimable because no arm ever dirties
    /// them. Whether the kernel TAKES them is the kernel's business under a
    /// pressure this box cannot stage without measuring the box instead of the
    /// mechanism.
    #[must_use]
    pub fn source(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }

    /// **Which source this tier's seat copies read from** — `"landed"` for
    /// the cold arm's staging file, `"artifact"` for the warm arm's mapped
    /// serving artifact ([`Source`]).
    ///
    /// The word that makes [`Tier::source`]'s pair readable: the link count
    /// means opposite things under the two arms, and a gate that asserts one
    /// without asking this is asserting which arm ran.
    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }
}
