//! The routed-expert residency tier: a wired slab of expert seats, smaller
//! than the bank, swapped in from a host band table between fire segments.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kernels_metal::Tensor;
use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

use crate::device::{Buffer, Handles};
use crate::weight_store::Store;
use crate::error::{Fault, Result};
use crate::host_source::HostSource;
use crate::mapping::Mapping;

/// A param's other device planes, by `Trace::params` index — which move together.
pub type Attachments = BTreeMap<usize, Vec<usize>>;

/// What a plan wants and what a budget allows, decided from the trace and
/// the load plan's pairings alone. The empty plan means full residency.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// One entry per streamed BAND, in param order.
    bands: Vec<BandPlan>,
    /// One entry per streamed GROUP, in router order.
    groups: Vec<GroupPlan>,
    /// `param index -> how many experts of it the slab seats`, consulted by
    /// [`weights::places`](crate::weights) to reserve a bank smaller than declared.
    resident_of: BTreeMap<usize, u32>,
    /// `param index -> where its whole plane lives in the host band table`.
    host_of: BTreeMap<usize, u64>,
    /// How many seats every group's slab has. Zero for the empty plan.
    slots: u32,
    device_bytes: u64,
    host_bytes: u64,
    /// The gathered class's half of this plan (`crate::gather`), sized in
    /// the same pass before the expert slab is sized against what's left.
    gathered: crate::gather::Plan,
}

/// One streamed band: a param sliced by a routing vector's expert count.
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

/// One streamed group: a router, and every expert-indexed param read
/// against the vector it writes — the unit of residency, since one
/// rewrite of `routes` re-indexes every band in the group at once.
/// The experts one expert-major pass seats: half the slab, so the other
/// half can be filled for the next pass while this one runs on the device.
#[must_use]
pub fn pass_group(slots: u32) -> u32 {
    if std::env::var_os("PIE_PASS_HALF").is_some_and(|v| v == "0") {
        return slots.max(1);
    }
    (slots / 2).max(1)
}

/// A run being walked in expert-major passes (`Tier::pass_at`).
#[derive(Clone, Debug)]
struct Passing {
    row_offset: u32,
    rows: u32,
    /// The routing vector as the router wrote it: expert ids.
    ids: Vec<i32>,
    /// The distinct experts, in first-appearance order, cut into groups of
    /// at most the slab's seats — one per pass.
    groups: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    /// The routing vector this group's bands are indexed by, rewritten at the segment cut.
    pub routes: ValueId,
    /// How many experts the router declares.
    pub experts: u32,
    /// How many of them the slab seats.
    pub slots: u32,
    /// Indices into [`Plan::bands`], ascending by param.
    pub bands: Vec<usize>,
    /// The route prediction as `hint`: `[tokens, k]` ranked experts for
    /// the NEXT group. `None` if none.
    pub hint: Option<ValueId>,
}

impl Plan {
    /// The residency plan for `trace` under `budget`. `None` (uncapped)
    /// answers the empty plan; a stated budget is met by seating fewer
    /// experts, never by holding fewer dense planes. `planes` pairs a
    /// quantized bank's scales and zero points to stay seated with its codes.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a param whose dtype has no element size or whose
    /// band shape breaks the uniformity proof; [`Fault::Residency`] for a
    /// budget that cannot hold the dense planes plus one seat of every band,
    /// or a capped budget over a plan with nothing routed to hold less of.
    pub fn of(trace: &Trace, planes: &Attachments, budget: Option<u64>) -> Result<Plan> {
        Plan::beside(trace, planes, budget, crate::gather::Plan::default())
    }

    /// The same plan, beside a gathered class that already holds some
    /// planes CPU-side (`crate::gather::Plan::params`). `gathered` is an
    /// exclusion, not a second budget: those bytes are in neither `full`
    /// nor the dense floor. [`Plan::of`] is this with an empty set.
    ///
    /// # Errors
    ///
    /// [`Plan::of`]'s.
    pub fn beside(
        trace: &Trace,
        planes: &Attachments,
        budget: Option<u64>,
        gathered: crate::gather::Plan,
    ) -> Result<Plan> {
        let held = gathered.params();
        let bytes = crate::weights::plane_bytes(trace)?;
        let full: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !held.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let Some(budget) = budget else {
            return Ok(Plan {
                device_bytes: full,
                gathered,
                ..Plan::default()
            });
        };
        if budget >= full {
            // Budget covers everything: no streaming.
            return Ok(Plan {
                device_bytes: full,
                gathered,
                ..Plan::default()
            });
        }

        let (mut bands, mut groups) = found(trace, planes, &bytes)?;
        if bands.is_empty() {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes and this plan's weight table \
                 demands {full}. Nothing in it is a routed-expert bank, so there is no \
                 tier to hold less of: only routed experts stream (their seat is \
                 chosen after the router has run); dense planes do not. Raise the budget, or state `None` for uncapped."
            )));
        }

        // Dense floor: planes that aren't streamed bands are held whole.
        let streamed: BTreeSet<usize> = bands.iter().map(|band| band.param).collect();
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !streamed.contains(at) && !held.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        // Lifted out so the sweep below can read `bands` while writing it back into.
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
                 planes do not stream in this build, so the budget cannot be met by \
                 holding less. Raise it to at least {floor}, or state `None`.",
                bands.len(),
            )));
        }

        // One seat count for the whole plan, walked down from the full bank.
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
        // Host band table layout: bands in param order, each whole.
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
            gathered,
        })
    }

    /// The gathered half of this plan — empty except for a capped Flash-Next load.
    #[must_use]
    pub fn gathered(&self) -> &crate::gather::Plan {
        &self.gathered
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

    /// How many experts of `param` the slab seats, or `None` if held whole.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    /// Where `param`'s whole plane lives in the host band table, or `None` if it lands on the device.
    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.host_of.get(&param).copied()
    }

    /// What this plan demands of the device, in bytes — what
    /// [`Residency::admit`](engine::load::Residency::admit) is asked with.
    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes + self.gathered.device_demand()
    }

    /// What this plan demands of a host tier: zero, always, on unified memory.
    #[must_use]
    pub fn host_demand(&self) -> u64 {
        let _ = self.host_bytes;
        0
    }

    /// How many bytes the host band table holds, for the cold arm's
    /// [`HostSource::open`](crate::host_source::HostSource::open) — unread by a warm load.
    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.host_bytes
    }
}

/// The streamed bands and groups a trace declares, with the uniformity
/// proof checked. A band is a param some routed op reads at an
/// expert-indexed port, determined by the op, not a naming convention.
fn found(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BandPlan>, Vec<GroupPlan>)> {
    // Routers first: expert count is the router's field.
    let mut arity: BTreeMap<u32, u32> = BTreeMap::new();
    let mut hints: BTreeMap<u32, ValueId> = BTreeMap::new();
    let mut order: Vec<ValueId> = Vec::new();
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        if let Linear::MoeTopkSqrtSoftplus {
            routes,
            hint: Some(hint),
            ..
        }
        | Linear::MoeTopkSigmoid {
            routes,
            hint: Some(hint),
            ..
        } = op
        {
            hints.insert(routes.0, *hint);
        }
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
            // The lookup router reads no logits, so it's not a `MoeTopk*` by
            // name, but it writes the same `routes` vector.
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if arity.insert(routes.0, experts).is_none() {
            order.push(routes);
        }
    }

    // Expert-indexed reads, joined to the router that wrote their routing vector.
    let mut of_group: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut routers_of: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
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
            routers_of.entry(at).or_default().insert(routes.0);
            for &plane in planes.get(&at).into_iter().flatten() {
                if !seats.contains(&plane) {
                    seats.push(plane);
                }
                routers_of.entry(plane).or_default().insert(routes.0);
            }
        }
    }
    // **A BANK TWO ROUTERS INDEX STAYS RESIDENT.** A seat number means one
    // group's seat, and a band shared between two groups would be re-indexed
    // twice — so such a band (a draft head's experts, routed once per chain
    // step) is not a streamed band at all: it is held whole, as a dense plane
    // is, and costs the budget its full size.
    let shared: BTreeSet<usize> = routers_of
        .iter()
        .filter(|(_, routers)| routers.len() > 1)
        .map(|(&at, _)| at)
        .collect();

    let mut bands: Vec<BandPlan> = Vec::new();
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut owner: BTreeMap<usize, ValueId> = BTreeMap::new();
    let mut declared: Option<u32> = None;
    for routes in order {
        let Some(mut params) = of_group.remove(&routes.0) else {
            // A router nothing expert-indexed reads isn't a group.
            continue;
        };
        params.retain(|at| !shared.contains(at));
        if params.is_empty() {
            continue;
        }
        params.sort_unstable();
        let experts = arity[&routes.0];
        // Every group in a plan states the same expert count.
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
            hint: hints.get(&routes.0).copied(),
        });
    }
    Ok((bands, groups))
}

/// How many experts one row of the router writing `routes` picks — the
/// bound on distinct experts `rows` rows can name, which is what sizes a
/// streamed segment's sub-batch (`slots / fan_out` rows fit the slab).
#[must_use]
pub fn fan_out(trace: &Trace, routes: ValueId) -> Option<u32> {
    trace.nodes.iter().find_map(|node| match &node.op {
        Operation::Linear(Linear::MoeTopkSoftmax { routes: r, top_k, .. })
        | Operation::Linear(Linear::MoeTopkSoftmaxScaled { routes: r, top_k, .. })
        | Operation::Linear(Linear::MoeTopkSigmoid { routes: r, top_k, .. })
        | Operation::Linear(Linear::MoeTopkSqrtSoftplus { routes: r, top_k, .. })
        | Operation::Linear(Linear::MoeHashRoute { routes: r, top_k, .. })
            if *r == routes =>
        {
            Some(*top_k)
        }
        _ => None,
    })
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

/// Where the walk is cut, one entry per region of the compiled template:
/// the routing vector the router in that region writes, or `None`. The cut
/// falls after the deciding node and before the nodes that read.
///
/// Only a router whose bands `plan` STREAMS cuts anything: a mixture over a
/// resident bank (a draft head's experts, held whole because two routers
/// index them) needs no seat swapped, and a cut is a blocking commit — two
/// of them a fire, for nothing, on every plain decode.
///
/// # Errors
///
/// [`Fault::Residency`] when the plan streams and a region holds two
/// streamed routers: the cut would fall after both, encoding the first
/// mixture against un-swapped seats.
pub fn cuts(
    trace: &Trace,
    compiled: &CompiledModel,
    plan: &Plan,
) -> Result<Vec<Option<ValueId>>> {
    let streams = plan.streams();
    let streamed: BTreeSet<u32> = plan.groups.iter().map(|group| group.routes.0).collect();
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
                // The lookup router decides a mixture the same way the ranked
                // four do.
                | Linear::MoeHashRoute { routes, .. } => *routes,
                _ => continue,
            };
            if !streamed.contains(&routes.0) {
                continue;
            }
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

/// One band, seated: where its experts are on both sides, and the seat's width.
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

/// One group's wired slab: which expert is in which seat, and which seats
/// the segment in flight may not lose.
#[derive(Debug)]
struct Slab {
    experts: u32,
    slots: u32,
    bands: Vec<Band>,
    /// `expert -> seat`, or `None` for an expert the slab does not hold.
    seat_of: Vec<Option<u32>>,
    /// `seat -> expert`, or `None` for a seat nothing has been copied into.
    in_seat: Vec<Option<u32>>,
    /// A seat the segment being built will read; released at the next cut.
    pinned: Vec<bool>,
    /// The tick each seat was last routed to or filled at — [`Tier::evict`]
    /// takes the least recent unpinned seat (true LRU).
    last_used: Vec<u64>,
}

/// What one group's residency looks like from outside — the only observable a swap has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupResidency {
    /// The group's first band, by name — what a reader recognises it as.
    pub name: String,
    /// How many experts the router declares.
    pub experts: u32,
    /// How many of them the slab seats.
    pub slots: u32,
    /// Which expert is in which seat, ascending by seat; `None` if uncopied.
    pub in_seat: Vec<Option<u32>>,
}

/// The host bytes themselves, under whichever arm produced them.
#[derive(Debug)]
enum Bytes {
    /// The cold arm's staging: an unlinked temporary file, written once.
    Landed(HostSource),
    /// The warm arm's: the serving artifact's own `PROT_READ` mapping.
    Artifact(Arc<Mapping>),
}

/// Where a seat copy reads from: [`Source::landed`] (cold) reads the
/// landing sink's host table, [`Source::artifact`] (warm) reads the
/// serving artifact's mapping. Both are CPU-read-only.
#[derive(Debug)]
pub struct Source {
    bytes: Bytes,
    /// `param index -> byte offset of expert 0 of that band`, in `bytes`.
    bands: BTreeMap<usize, u64>,
}

impl Source {
    /// The cold arm's source: the staging file, at the plan's host-table offsets.
    #[must_use]
    pub fn landed(plan: &Plan, host: HostSource) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands: plan.host_of.clone(),
        }
    }

    /// The cold arm's source, at caller-stated offsets — used by `crate::gather`.
    #[must_use]
    pub fn from_host(host: HostSource, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands,
        }
    }

    /// The warm arm's source: the mapped artifact, at per-band offsets
    /// `bands` recovered from its serving manifest.
    #[must_use]
    pub fn artifact(map: Arc<Mapping>, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Artifact(map),
            bands,
        }
    }

    /// What this source is made of, as one word.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.bytes {
            Bytes::Landed(_) => "landed",
            Bytes::Artifact(_) => "artifact",
        }
    }

    /// What's actually behind these bytes: `(file size, link count)`, or
    /// `None` for no file at all — `(bytes, 0)` for `landed` (unlinked),
    /// `(bytes, links >= 1)` for `artifact`.
    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        match &self.bytes {
            Bytes::Landed(host) => host.backing(),
            Bytes::Artifact(map) => Some((map.backing()?, map.links()?)),
        }
    }

    /// Where expert 0 of `param`'s band lies in these bytes.
    pub(crate) fn at(&self, param: usize) -> Option<u64> {
        self.bands.get(&param).copied()
    }

    /// The file a seat copy may `pread` out of, if this source has one.
    /// `None` only for an empty cold source.
    pub(crate) fn file(&self) -> Option<&std::fs::File> {
        match &self.bytes {
            Bytes::Landed(host) => host.file(),
            Bytes::Artifact(map) => Some(map.file()),
        }
    }

    /// `len` bytes at `from`, or `None` for a span that leaves the source.
    pub(crate) fn get(&self, from: usize, len: usize) -> Option<&[u8]> {
        let all: &[u8] = match &self.bytes {
            Bytes::Landed(host) => host,
            Bytes::Artifact(map) => map,
        };
        all.get(from..from.checked_add(len)?)
    }

    /// How many bytes the source holds — the bound a refusal names.
    pub(crate) fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Landed(host) => host.len() as u64,
            Bytes::Artifact(map) => map.len(),
        }
    }

    /// Hand the source to the pager, once, after landing and the identity
    /// prefix have finished reading it. A no-op for the artifact arm.
    pub(crate) fn settle(&mut self) {
        match &mut self.bytes {
            Bytes::Landed(host) => host.settle(),
            Bytes::Artifact(_) => {}
        }
    }
}

/// The tier: the host band table, a wired slab per group, and the seat
/// bookkeeping between them. Holds a retain of the weight store, not a
/// borrow; safe because every seat write happens between two command
/// buffers, after a blocking commit, with no encode in flight.
#[derive(Debug)]
pub struct Tier {
    /// A retain of the weight store — where seats live.
    store: Store,
    /// Every expert of every streamed band, in a file-backed mapping — the
    /// cold arm's staging file or the warm arm's artifact. CPU-read-only:
    /// no kernel addresses this, since a GPU-touched mapped page wires.
    source: Source,
    slabs: Vec<Slab>,
    /// `routes value -> slab index`.
    of_routes: BTreeMap<u32, usize>,
    /// How many seat copies this load has done.
    swaps: u64,
    /// How many segment cuts this load has taken.
    segments: u64,
    /// Threads a segment's seat copies spread over (`PIE_SEAT_THREADS`).
    threads: usize,
    /// Seats decided but not yet filled — `(slab, seat, expert)` — between a
    /// segment's rewrite pass and its [`Tier::flush`].
    pending: Vec<(usize, u32, u32)>,
    /// The recency tick: one per seat touch, monotone, never zero.
    tick: u64,
    /// How many distinct `(segment, expert)` lookups found the expert seated.
    hits: u64,
    /// How many had to seat it.
    misses: u64,
    /// Wall time inside [`Tier::segment`], and inside [`Tier::flush`] alone.
    cut_ns: u64,
    copy_ns: u64,
    /// Wall time the cut spent waiting on its blocking commit.
    wait_ns: u64,
    /// `routes value -> the prediction its router carries` ([`GroupPlan::hint`]).
    hint_of: BTreeMap<u32, ValueId>,
    /// Per slab: the prediction read at the previous group's cut for this
    /// one, or `None`.
    predicted: Vec<Option<Vec<Vec<u32>>>>,
    /// Per slab: the run an expert-major walk is in the middle of — its
    /// original routing vector and the expert groups its passes seat.
    passing: Vec<Option<Passing>>,
    /// The route prefetch in flight: the next group's predicted experts
    /// being read on another thread. Joined at the next cut.
    inflight: Option<std::thread::JoinHandle<Result<()>>>,
    /// The source's file, shared with the prefetch thread.
    file: Option<std::sync::Arc<std::fs::File>>,
    /// Whether predicted experts are prefetched, or only scored.
    prefetch: bool,
    /// How many predicted experts per row the prefetch reads ([`PREFETCH_K`]).
    prefetch_k: usize,
    /// `PIE_ROUTE_DUMP=path`: every cut's true routes appended as one line
    /// `slab<TAB>id id …` per token row.
    dump: Option<std::io::BufWriter<std::fs::File>>,
    /// The prediction's score, over every cut that had one to check.
    prediction: Prediction,
}

/// How good a route prediction is, counted at the cuts — [`Tier::prediction`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Prediction {
    /// True experts checked against a prediction.
    pub total: u64,
    /// Of them, how many the prediction's top 6, 8, 12 and 16 contained.
    pub covered: [u64; 4],
    /// True experts that were not seated at the cut — the misses.
    pub misses: u64,
    /// Of the misses, how many the prediction's top 6, 8, 12 and 16 contained.
    pub saved: [u64; 4],
    /// How many experts the prefetch read ahead, since the load.
    pub prefetched: u64,
}

/// How many predicted experts per token row the prefetch reads ahead by
/// default (`PIE_PREFETCH_K` overrides) — fewer than the router's fan-out,
/// since a wrong pick is a whole expert read for nothing.
const PREFETCH_K: usize = 4;

/// The prediction prefixes [`Prediction`] scores.
pub const PREDICTION_PREFIXES: [usize; 4] = [6, 8, 12, 16];

/// How many threads seat copies are spread over by default: the measured
/// plateau of `pread` on this box is reached at four and flat past eight.
const SEAT_THREADS: usize = 8;

impl Tier {
    /// Open the tier `plan` describes, reading experts out of `source`;
    /// `offsets` is `param index -> byte offset of seat 0 inside the store`.
    /// Initial seating is the identity prefix (seat `i` holds expert `i`).
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a seat or source span that leaves its bound,
    /// [`Fault::Residency`] for a band the source states no offset for,
    /// [`Fault::Deviceless`] off Apple.
    pub fn open(plan: &Plan, store: &Store, source: Source, offsets: &[u64]) -> Result<Tier> {
        let mut tier = Tier {
            store: store.clone(),
            source,
            slabs: Vec::with_capacity(plan.groups.len()),
            of_routes: BTreeMap::new(),
            swaps: 0,
            segments: 0,
            tick: 1,
            hits: 0,
            misses: 0,
            cut_ns: 0,
            copy_ns: 0,
            wait_ns: 0,
            hint_of: plan
                .groups
                .iter()
                .filter_map(|group| group.hint.map(|hint| (group.routes.0, hint)))
                .collect(),
            predicted: vec![None; plan.groups.len()],
            passing: vec![None; plan.groups.len()],
            prediction: Prediction::default(),
            inflight: None,
            file: None,
            prefetch: std::env::var_os("PIE_ROUTE_PREFETCH").is_none_or(|v| v != "0"),
            dump: std::env::var_os("PIE_ROUTE_DUMP")
                .and_then(|path| std::fs::File::create(path).ok())
                .map(std::io::BufWriter::new),
            prefetch_k: std::env::var("PIE_PREFETCH_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(PREFETCH_K),
            threads: std::env::var("PIE_SEAT_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n > 0)
                .unwrap_or(SEAT_THREADS),
            pending: Vec::new(),
        };
        for (at, group) in plan.groups.iter().enumerate() {
            let bands = group
                .bands
                .iter()
                .map(|&band| {
                    let band = &plan.bands[band];
                    // A band the source can't place means the source and
                    // plan weren't built from each other — refused by name.
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
                last_used: vec![0; group.slots as usize],
            });
        }
        // The prefix goes through the same batched `pread` path a segment uses.
        for at in 0..tier.slabs.len() {
            for seat in 0..tier.slabs[at].slots {
                tier.slabs[at].in_seat[seat as usize] = Some(seat);
                tier.slabs[at].seat_of[seat as usize] = Some(seat);
                tier.pending.push((at, seat, seat));
            }
        }
        tier.flush()?;
        tier.file = tier
            .source
            .file()
            .and_then(|file| file.try_clone().ok())
            .map(std::sync::Arc::new);
        // Counter starts after the prefix fill, so `Tier::motion` counts only serving.
        tier.swaps = 0;
        // Source handed to the pager once fully written and read; pages go
        // dirty -> clean/reclaimable, and a later seat copy faults them back.
        tier.source.settle();
        Ok(tier)
    }

    /// One segment cut: read the routing vector the region just wrote, seat
    /// every expert it names, and rewrite it in place to name seats.
    /// Callers must have already committed and waited for prior encodes.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a routing vector this fire minted no handle
    /// row for, [`Fault::Residency`] for a segment routing to more distinct
    /// experts than the slab seats or to an undeclared one,
    /// [`Fault::Ceiling`] for a span that leaves a reservation.
    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
        pass: (u32, u32),
    ) -> Result<u32> {
        let Some(&at) = self.of_routes.get(&routes.0) else {
            return Ok(1);
        };
        let started = std::time::Instant::now();
        let out = self.segment_at(at, arena, handles, routes, rect, hint, span, pass);
        self.cut_ns += started.elapsed().as_nanos() as u64;
        out
    }

    /// The prediction the router writing `routes` carries, if any ([`GroupPlan::hint`]).
    #[must_use]
    pub fn hint_for(&self, routes: ValueId) -> Option<ValueId> {
        self.hint_of.get(&routes.0).copied()
    }

    /// Read one `[rows, width]` i32 rectangle's window out of the arena.
    fn read_rows(
        arena: &mut Buffer,
        handles: &Handles,
        rect: Tensor,
        span: MaskSpan,
        what: &str,
    ) -> Result<Vec<Vec<i32>>> {
        let width = usize::try_from(rect.width).unwrap_or(usize::MAX);
        let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!("handle {}, {what}, which this fire minted no row for", rect.buf),
        })?;
        let first = row.offset() + u64::from(span.row_offset) * rect.width as u64 * 4;
        let mut raw = vec![0u8; span.rows as usize * width * 4];
        arena.read(first, &mut raw)?;
        Ok(raw
            .chunks_exact(width * 4)
            .map(|row| {
                row.chunks_exact(4)
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                    .collect()
            })
            .collect())
    }

    /// Score the prediction made for this group, against what the router
    /// chose and what the slab held, before this cut seats anything — then
    /// read the prediction this router makes for the next group.
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        if let Some(predicted) = self.predicted[at].take() {
            let truth = Self::read_rows(arena, handles, rect, span, "a routing vector")?;
            for (row, ranked) in truth.iter().zip(predicted.iter()) {
                for &id in row {
                    if id < 0 {
                        continue;
                    }
                    let expert = id as u32;
                    if expert >= self.slabs[at].experts {
                        continue;
                    }
                    let seated = self.slabs[at].seat_of[expert as usize].is_some();
                    self.prediction.total += 1;
                    if !seated {
                        self.prediction.misses += 1;
                    }
                    for (i, &k) in PREDICTION_PREFIXES.iter().enumerate() {
                        if ranked.iter().take(k).any(|&p| p == expert) {
                            self.prediction.covered[i] += 1;
                            if !seated {
                                self.prediction.saved[i] += 1;
                            }
                        }
                    }
                }
            }
        }
        if let (Some(hint), true) = (hint, at + 1 < self.predicted.len()) {
            let _ = routes;
            let rows = Self::read_rows(arena, handles, hint, span, "a route prediction")?;
            self.predicted[at + 1] = Some(
                rows.into_iter()
                    .map(|row| row.into_iter().filter(|&id| id >= 0).map(|id| id as u32).collect())
                    .collect(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn segment_at(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
        pass: (u32, u32),
    ) -> Result<u32> {
        // ── Whatever the prefetch was still reading into the seats this
        //    segment is about to name has to have landed first.
        self.join_inflight()?;
        if pass.1 > 1 {
            return self.pass_at(at, arena, handles, routes, rect, span, pass);
        }
        self.segment_rows(at, arena, handles, routes, rect, hint, span)?;
        Ok(1)
    }

    /// One row-cut segment (the legacy piece): seat every expert the span
    /// routes to and rewrite the vector to seats.
    #[allow(clippy::too_many_arguments)]
    fn segment_rows(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        if span.rows > 0 && (hint.is_some() || self.predicted[at].is_some()) {
            self.predict(at, arena, handles, routes, rect, hint, span)?;
        }
        // Previous segment completed (caller's blocking commit), so nothing
        // is reading a seat any more.
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
        if let Some(dump) = &mut self.dump {
            use std::io::Write;
            for row in raw.chunks_exact(width as usize * 4) {
                let ids: Vec<String> = row
                    .chunks_exact(4)
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]).to_string())
                    .collect();
                let _ = writeln!(dump, "{at}\t{}", ids.join(" "));
            }
        }
        // Seat then rewrite in one pass, so a repeated id costs one copy.
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
        // The decisions were serial (the clock is one hand); the copies they
        // decided are independent and go out together.
        self.flush()?;
        arena.write(first, &raw)?;
        // ── And the NEXT group's predicted experts start reading now, while
        //    the device runs this segment.
        if self.prefetch {
            if let Some(rows) = self.predicted.get(at + 1).cloned().flatten() {
                self.prefetch(at + 1, &rows)?;
            }
        }
        Ok(())
    }

    /// One expert-major pass over a whole run (`compose::pass_spans`): pass
    /// 0 reads the run's routing vector and cuts its distinct experts into
    /// groups of at most the slab's seats; pass `p` seats group `p` and
    /// writes the vector as seat indices for that group's experts and `-1`
    /// for every other entry, which the routed kernels skip (`route_sort`
    /// drops a negative pair, `route_scatter` leaves its row, the matvec
    /// returns). Each expert is copied once per run, not once per piece.
    #[allow(clippy::too_many_arguments)]
    fn pass_at(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        span: MaskSpan,
        (pass, passes): (u32, u32),
    ) -> Result<u32> {
        for seat in &mut self.slabs[at].pinned {
            *seat = false;
        }
        self.segments += 1;
        if span.rows == 0 {
            return Ok(0);
        }
        let width = u64::from(rect.width);
        let base = handles
            .get(rect.buf)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the routing vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, routes.0
                ),
            })?
            .offset();
        let first = base + u64::from(span.row_offset) * width * 4;
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let fresh = pass == 0
            || self.passing[at]
                .as_ref()
                .is_none_or(|p| p.row_offset != span.row_offset || p.rows != span.rows);
        if fresh {
            let mut raw = vec![0u8; count * 4];
            arena.read(first, &mut raw)?;
            let ids: Vec<i32> = raw
                .chunks_exact(4)
                .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                .collect();
            if let Some(dump) = &mut self.dump {
                use std::io::Write;
                for row in ids.chunks_exact(width as usize) {
                    let ids: Vec<String> = row.iter().map(ToString::to_string).collect();
                    let _ = writeln!(dump, "{at}\t{}", ids.join(" "));
                }
            }
            let mut order: Vec<u32> = Vec::new();
            for &id in &ids {
                if id < 0 {
                    continue;
                }
                let expert = id as u32;
                if expert >= self.slabs[at].experts {
                    return Err(Fault::Residency(format!(
                        "a routing vector names expert {expert} and `{}` declares {} of them; \
                         a seat cannot be found for an expert the router does not have.",
                        self.slabs[at].bands[0].name, self.slabs[at].experts
                    )));
                }
                if !order.contains(&expert) {
                    order.push(expert);
                }
            }
            let seats = pass_group(self.slabs[at].slots) as usize;
            let groups = order.chunks(seats).map(<[u32]>::to_vec).collect();
            self.passing[at] = Some(Passing {
                row_offset: span.row_offset,
                rows: span.rows,
                ids,
                groups,
            });
        }
        let (ids, group, next) = {
            let state = self.passing[at].as_ref().expect("stated just above");
            (
                state.ids.clone(),
                state.groups.get(pass as usize).cloned().unwrap_or_default(),
                state.groups.get(pass as usize + 1).cloned(),
            )
        };
        let _ = passes;
        // Seat this pass's group, then write the vector: seats for the
        // group, `-1` for the rest.
        let mut seat_of: BTreeMap<u32, i32> = BTreeMap::new();
        for &expert in &group {
            let seat = self.seat(at, expert)?;
            seat_of.insert(expert, seat as i32);
        }
        let mut raw = Vec::with_capacity(count * 4);
        let mut assigned = 0usize;
        for id in ids {
            let entry = if id < 0 {
                -1
            } else {
                seat_of.get(&(id as u32)).copied().unwrap_or(-1)
            };
            if entry >= 0 {
                assigned += 1;
            }
            raw.extend_from_slice(&entry.to_le_bytes());
        }
        if std::env::var_os("PIE_CUT_TRACE").is_some() {
            let seats: Vec<i32> = seat_of.values().copied().collect();
            eprintln!(
                "pass {pass} of {passes} on slab {at}: group of {} experts (seats {:?}), {assigned} of {count} entries assigned, groups {}",
                group.len(),
                seats,
                self.passing[at].as_ref().map_or(0, |p| p.groups.len())
            );
        }
        self.flush()?;
        arena.write(first, &raw)?;
        // The NEXT pass's group starts reading now, into the half of the
        // slab this pass does not touch, while the device runs this one.
        if self.prefetch {
            if let Some(next) = next {
                self.prefetch_group(at, &next)?;
            }
        }
        Ok(self.passing[at].as_ref().map_or(0, |p| p.groups.len() as u32))
    }

    /// Read `experts` of slab `at` ahead on a thread, into unpinned seats
    /// only — the current pass's seats stay pinned and untouched. Joined at
    /// the next cut (`join_inflight`), where the seats are then hits.
    fn prefetch_group(&mut self, at: usize, experts: &[u32]) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        let mut jobs: Vec<(u64, u64, u64)> = Vec::new();
        for &expert in experts {
            if self.slabs[at].seat_of[expert as usize].is_some() {
                continue;
            }
            let Ok(seat) = self.evict(at) else {
                break;
            };
            if let Some(held) = self.slabs[at].in_seat[seat as usize] {
                self.slabs[at].seat_of[held as usize] = None;
            }
            let slab = &mut self.slabs[at];
            slab.in_seat[seat as usize] = Some(expert);
            slab.seat_of[expert as usize] = Some(seat);
            slab.last_used[seat as usize] = self.tick;
            self.tick += 1;
            slab.pinned[seat as usize] = true;
            for band in &slab.bands {
                jobs.push((
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(expert) * band.stride,
                    band.stride,
                ));
            }
            self.prediction.prefetched += 1;
        }
        if jobs.is_empty() {
            return Ok(());
        }
        self.swaps += jobs.len() as u64;
        let writers = self.store.file_writers(&jobs)?;
        let threads = self.threads;
        self.inflight = Some(std::thread::spawn(move || {
            for (writer, jobs) in &writers {
                writer.pread(&file, jobs, threads)?;
            }
            Ok(())
        }));
        Ok(())
    }

    /// Wait for the prefetch in flight, if any, and surface its refusal.
    fn join_inflight(&mut self) -> Result<()> {
        match self.inflight.take() {
            Some(handle) => handle.join().unwrap_or_else(|_| {
                Err(Fault::Residency(
                    "the route prefetch thread panicked".to_string(),
                ))
            }),
            None => Ok(()),
        }
    }

    /// Read the predicted experts of slab `at` ahead, on a thread: each
    /// unseated expert in the top [`PREFETCH_K`] of a row's prediction is
    /// seated and pinned, then `pread` on a thread joined at the next cut.
    fn prefetch(&mut self, at: usize, rows: &[Vec<u32>]) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        // Previous segment's pins release here: this cut's commit proves it completed.
        for pin in &mut self.slabs[at].pinned {
            *pin = false;
        }
        let mut wanted: Vec<u32> = Vec::new();
        for row in rows {
            for &expert in row.iter().take(self.prefetch_k) {
                if expert < self.slabs[at].experts && !wanted.contains(&expert) {
                    wanted.push(expert);
                }
            }
        }
        let mut jobs: Vec<(u64, u64, u64)> = Vec::new();
        for expert in wanted {
            if let Some(seat) = self.slabs[at].seat_of[expert as usize] {
                self.slabs[at].last_used[seat as usize] = self.tick;
                self.tick += 1;
                continue;
            }
            // A slab with no free seat prefetches nothing rather than
            // refusing: the sync cut will read what it needs.
            let Ok(seat) = self.evict(at) else {
                break;
            };
            if let Some(held) = self.slabs[at].in_seat[seat as usize] {
                self.slabs[at].seat_of[held as usize] = None;
            }
            let slab = &mut self.slabs[at];
            slab.in_seat[seat as usize] = Some(expert);
            slab.seat_of[expert as usize] = Some(seat);
            slab.last_used[seat as usize] = self.tick;
            self.tick += 1;
            slab.pinned[seat as usize] = true;
            for band in &slab.bands {
                jobs.push((
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(expert) * band.stride,
                    band.stride,
                ));
            }
            self.prediction.prefetched += 1;
        }
        if jobs.is_empty() {
            return Ok(());
        }
        self.swaps += jobs.len() as u64;
        let writers = self.store.file_writers(&jobs)?;
        let threads = self.threads;
        self.inflight = Some(std::thread::spawn(move || {
            for (writer, jobs) in &writers {
                writer.pread(&file, jobs, threads)?;
            }
            Ok(())
        }));
        Ok(())
    }

    /// Where expert `expert` of slab `at` sits, seating it if needed, and
    /// pinning it either way since the segment being built reads it.
    fn seat(&mut self, at: usize, expert: u32) -> Result<u32> {
        if let Some(seat) = self.slabs[at].seat_of[expert as usize] {
            let slab = &mut self.slabs[at];
            // A repeat inside one segment is neither a hit nor a miss: only
            // the first lookup of a segment counts.
            if !slab.pinned[seat as usize] {
                self.hits += 1;
            }
            slab.last_used[seat as usize] = self.tick;
            self.tick += 1;
            slab.pinned[seat as usize] = true;
            return Ok(seat);
        }
        self.misses += 1;
        let seat = self.evict(at)?;
        if let Some(held) = self.slabs[at].in_seat[seat as usize] {
            self.slabs[at].seat_of[held as usize] = None;
        }
        // Claimed now, so a repeat in this segment hits it and eviction
        // can't hand it out twice; filled later at `flush`.
        let slab = &mut self.slabs[at];
        slab.in_seat[seat as usize] = Some(expert);
        slab.seat_of[expert as usize] = Some(seat);
        slab.last_used[seat as usize] = self.tick;
        self.tick += 1;
        slab.pinned[seat as usize] = true;
        self.pending.push((at, seat, expert));
        Ok(seat)
    }

    /// Fill every pending seat, every band of each — `pread` out of the
    /// source's file across [`Tier::threads`] threads where the source has
    /// one, the mapping `memcpy` otherwise.
    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let pending = std::mem::take(&mut self.pending);
        let mut jobs: Vec<(u64, u64, u64)> = Vec::with_capacity(pending.len() * 3);
        for &(at, seat, expert) in &pending {
            for band in &self.slabs[at].bands {
                jobs.push((
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(expert) * band.stride,
                    band.stride,
                ));
            }
        }
        match self.source.file() {
            Some(file) => self.store.write_from_file(file, &jobs, self.threads)?,
            None => {
                for &(into, from, len) in &jobs {
                    let from = usize::try_from(from).unwrap_or(usize::MAX);
                    let len = usize::try_from(len).unwrap_or(usize::MAX);
                    let source = self.source.get(from, len).ok_or_else(|| Fault::Ceiling {
                        what: "bytes of the seat source",
                        need: (from + len) as u64,
                        have: self.source.len(),
                    })?;
                    self.store.write(into, source)?;
                }
            }
        }
        self.swaps += jobs.len() as u64;
        self.copy_ns += started.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Least recently used, over the unpinned seats of one slab.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] when every seat is pinned: this segment routes to
    /// more distinct experts than the slab has room for.
    fn evict(&mut self, at: usize) -> Result<u32> {
        let slab = &self.slabs[at];
        let victim = (0..slab.slots)
            .filter(|&seat| !slab.pinned[seat as usize])
            .min_by_key(|&seat| slab.last_used[seat as usize]);
        victim.ok_or_else(|| {
            Fault::Residency(format!(
                "one segment of this fire routes to more than {} distinct experts of \
                 `{}`, and the wired slab seats {}: every seat is pinned by a matmul \
                 this same segment will run, so no seat can be reused. Every expert one \
                 segment reads must be resident at once — raise `device_weight_budget`, or \
                 fire fewer tokens per step. Splitting one segment's tokens into sub-batches \
                 is the mechanism that would make this neither, and it is not in this build.",
                slab.slots, slab.bands[0].name, slab.slots
            ))
        })
    }

    // A seat copy writes OVER an existing seat's bytes, never a new
    // allocation, so the wired footprint stays exactly the slab's `slots`.

    /// Every group's occupancy, in plan order.
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

    /// `(band copies, segment cuts)` since the load.
    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.swaps, self.segments)
    }

    /// `(hits, misses)` of the seat cache since the load. The prefix fill counts as neither.
    #[must_use]
    pub fn hits(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// How the route predictions scored, over every cut that checked one.
    #[must_use]
    pub fn prediction(&self) -> Prediction {
        self.prediction
    }

    /// `(cut ns, seat-copy ns, blocking-commit wait ns)` since the load.
    #[must_use]
    pub fn host_time(&self) -> (u64, u64, u64) {
        (self.cut_ns, self.copy_ns, self.wait_ns)
    }

    /// The cut's blocking commit took `ns`.
    pub fn note_wait(&mut self, ns: u64) {
        self.wait_ns += ns;
    }

    /// `(backing file size, link count)`, or `None` with no file behind it.
    /// Read [`Tier::source_kind`] to tell the two arms apart.
    #[must_use]
    pub fn source(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }

    /// Which source this tier's seat copies read from: `"landed"` or `"artifact"`.
    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }
}

impl Drop for Tier {
    fn drop(&mut self) {
        // A prefetch still writing into a slab that is about to be released
        // must land (or fail) first; its verdict has nobody left to hear it.
        let _ = self.join_inflight();
    }
}
