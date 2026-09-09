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
use crate::weight_store::Store;

pub type Attachments = BTreeMap<usize, Vec<usize>>;

#[derive(Debug, Clone, Default)]
pub struct Plan {
    bands: Vec<BandPlan>,
    groups: Vec<GroupPlan>,
    resident_of: BTreeMap<usize, u32>,
    host_of: BTreeMap<usize, u64>,
    slots: u32,
    device_bytes: u64,
    host_bytes: u64,
    gathered: crate::gather::Plan,
}

#[derive(Debug, Clone)]
pub struct BandPlan {
    pub param: usize,
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub stride: u64,
    pub group: usize,
}

#[must_use]
pub fn pass_group(slots: u32) -> u32 {
    if !crate::diag::on().pass_half {
        return slots.max(1);
    }
    (slots / 2).max(1)
}

#[derive(Clone, Debug)]
struct Passing {
    row_offset: u32,
    rows: u32,
    ids: Vec<i32>,
    groups: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub routes: ValueId,
    pub experts: u32,
    pub slots: u32,
    pub bands: Vec<usize>,
    pub hint: Option<ValueId>,
}

impl Plan {
    pub fn of(trace: &Trace, planes: &Attachments, budget: Option<u64>) -> Result<Plan> {
        Plan::beside(trace, planes, budget, crate::gather::Plan::default())
    }

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

        let streamed: BTreeSet<usize> = bands.iter().map(|band| band.param).collect();
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !streamed.contains(at) && !held.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let strides: Vec<u64> = bands.iter().map(|band| band.stride).collect();
        let seats = |n: u32| -> u64 {
            strides
                .iter()
                .map(|stride| (u64::from(n) * stride).next_multiple_of(crate::weights::ALIGN))
                .sum()
        };
        let fan = groups
            .iter()
            .filter_map(|group| fan_out(trace, group.routes))
            .max()
            .unwrap_or(1)
            .max(1);
        let need = (1..=u32::MAX)
            .find(|&n| pass_group(n) >= fan)
            .unwrap_or(fan);
        let floor = dense + seats(need);
        if budget < floor {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes; this plan's DENSE planes \
                 demand {dense} resident and its {} routed bands need {need} expert \
                 seats each on top (a row routes to {fan} experts and a pass seats half \
                 the slab), which is {floor}. Dense planes do not stream in this build, \
                 so the budget cannot be met by holding less. Raise it to at least \
                 {floor}, or state `None`.",
                bands.len(),
            )));
        }

        let experts = groups[0].experts;
        let slack = budget - dense;
        let mut slots = 0u32;
        for n in (need..=experts.max(need)).rev() {
            if seats(n) <= slack {
                slots = n;
                break;
            }
        }
        debug_assert!(slots >= need, "the floor check above proved the seats fit");

        for band in &mut bands {
            band.slots = slots;
        }
        for group in &mut groups {
            group.slots = slots;
        }
        let resident_of = bands.iter().map(|band| (band.param, slots)).collect();
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

    #[must_use]
    pub fn gathered(&self) -> &crate::gather::Plan {
        &self.gathered
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        !self.bands.is_empty()
    }

    #[must_use]
    pub fn bands(&self) -> &[BandPlan] {
        &self.bands
    }

    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.host_of.get(&param).copied()
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes + self.gathered.device_demand()
    }

    #[must_use]
    pub fn host_demand(&self) -> u64 {
        let _ = self.host_bytes;
        0
    }

    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.host_bytes
    }
}

fn found(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BandPlan>, Vec<GroupPlan>)> {
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
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if arity.insert(routes.0, experts).is_none() {
            order.push(routes);
        }
    }

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
            continue;
        };
        params.retain(|at| !shared.contains(at));
        if params.is_empty() {
            continue;
        }
        params.sort_unstable();
        let experts = arity[&routes.0];
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
            if let Some(other) = owner.insert(at, routes)
                && other != routes
            {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is expert-indexed by two different routing vectors; a seat \
                              number means one group's seat, and a band shared between two \
                              groups would be re-indexed twice",
                });
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
            if plane == 0 || !plane.is_multiple_of(u64::from(experts)) {
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

#[must_use]
pub fn fan_out(trace: &Trace, routes: ValueId) -> Option<u32> {
    trace.nodes.iter().find_map(|node| match &node.op {
        Operation::Linear(Linear::MoeTopkSoftmax {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSoftmaxScaled {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSigmoid {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSqrtSoftplus {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeHashRoute {
            routes: r, top_k, ..
        }) if *r == routes => Some(*top_k),
        _ => None,
    })
}

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

pub fn cuts(trace: &Trace, compiled: &CompiledModel, plan: &Plan) -> Result<Vec<Option<ValueId>>> {
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

#[derive(Debug)]
struct Band {
    name: String,
    at: u64,
    from: u64,
    stride: u64,
}

#[derive(Debug)]
struct Slab {
    experts: u32,
    slots: u32,
    bands: Vec<Band>,
    seat_of: Vec<Option<u32>>,
    in_seat: Vec<Option<u32>>,
    pinned: Vec<bool>,
    last_used: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupResidency {
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub in_seat: Vec<Option<u32>>,
}

#[derive(Debug)]
enum Bytes {
    Landed(HostSource),
    Artifact(Arc<Mapping>),
}

#[derive(Debug)]
pub struct Source {
    bytes: Bytes,
    bands: BTreeMap<usize, u64>,
}

impl Source {
    #[must_use]
    pub fn landed(plan: &Plan, host: HostSource) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands: plan.host_of.clone(),
        }
    }

    #[must_use]
    pub fn from_host(host: HostSource, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands,
        }
    }

    #[must_use]
    pub fn artifact(map: Arc<Mapping>, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Artifact(map),
            bands,
        }
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.bytes {
            Bytes::Landed(_) => "landed",
            Bytes::Artifact(_) => "artifact",
        }
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        match &self.bytes {
            Bytes::Landed(host) => host.backing(),
            Bytes::Artifact(map) => Some((map.backing()?, map.links()?)),
        }
    }

    pub(crate) fn at(&self, param: usize) -> Option<u64> {
        self.bands.get(&param).copied()
    }

    pub(crate) fn file(&self) -> Option<&std::fs::File> {
        match &self.bytes {
            Bytes::Landed(host) => host.file(),
            Bytes::Artifact(map) => Some(map.file()),
        }
    }

    pub(crate) fn get(&self, from: usize, len: usize) -> Option<&[u8]> {
        let all: &[u8] = match &self.bytes {
            Bytes::Landed(host) => host,
            Bytes::Artifact(map) => map,
        };
        all.get(from..from.checked_add(len)?)
    }

    pub(crate) fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Landed(host) => host.len() as u64,
            Bytes::Artifact(map) => map.len(),
        }
    }

    pub(crate) fn settle(&mut self) {
        match &mut self.bytes {
            Bytes::Landed(host) => host.settle(),
            Bytes::Artifact(_) => {}
        }
    }
}

#[derive(Debug)]
pub struct Tier {
    store: Store,
    source: Source,
    slabs: Vec<Slab>,
    of_routes: BTreeMap<u32, usize>,
    swaps: u64,
    segments: u64,
    threads: usize,
    pending: Vec<(usize, u32, u32)>,
    tick: u64,
    hits: u64,
    misses: u64,
    cut_ns: u64,
    copy_ns: u64,
    wait_ns: u64,
    hint_of: BTreeMap<u32, ValueId>,
    predicted: Vec<Option<Vec<Vec<u32>>>>,
    passing: Vec<Option<Passing>>,
    inflight: Option<std::thread::JoinHandle<Result<()>>>,
    file: Option<std::sync::Arc<std::fs::File>>,
    prefetch: bool,
    prefetch_k: usize,
    dump: Option<std::io::BufWriter<std::fs::File>>,
    prediction: Prediction,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Prediction {
    pub total: u64,
    pub covered: [u64; 4],
    pub misses: u64,
    pub saved: [u64; 4],
    pub prefetched: u64,
}

const PREFETCH_K: usize = 4;

pub const PREDICTION_PREFIXES: [usize; 4] = [6, 8, 12, 16];

const SEAT_THREADS: usize = 8;

impl Tier {
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
            prefetch: crate::diag::on().route_prefetch,
            dump: crate::diag::on()
                .route_dump
                .as_ref()
                .and_then(|path| std::fs::File::create(path).ok())
                .map(std::io::BufWriter::new),
            prefetch_k: crate::diag::on().prefetch_k.unwrap_or(PREFETCH_K),
            threads: crate::diag::on().seat_threads.unwrap_or(SEAT_THREADS),
            pending: Vec::new(),
        };
        for (at, group) in plan.groups.iter().enumerate() {
            let bands = group
                .bands
                .iter()
                .map(|&band| {
                    let band = &plan.bands[band];
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
        tier.swaps = 0;
        tier.source.settle();
        Ok(tier)
    }

    #[allow(clippy::too_many_arguments)]
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

    #[must_use]
    pub fn hint_for(&self, routes: ValueId) -> Option<ValueId> {
        self.hint_of.get(&routes.0).copied()
    }

    fn read_rows(
        arena: &mut Buffer,
        handles: &Handles,
        rect: Tensor,
        span: MaskSpan,
        what: &str,
    ) -> Result<Vec<Vec<i32>>> {
        let width = usize::try_from(rect.width).unwrap_or(usize::MAX);
        let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, {what}, which this fire minted no row for",
                rect.buf
            ),
        })?;
        let first = row.offset() + u64::from(span.row_offset) * rect.width as u64 * 4;
        let mut raw = vec![0u8; span.rows as usize * width * 4];
        arena.read(first, &mut raw)?;
        Ok(raw
            .chunks_exact(width * 4)
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                    .collect()
            })
            .collect())
    }

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
                    .map(|row| {
                        row.into_iter()
                            .filter(|&id| id >= 0)
                            .map(|id| id as u32)
                            .collect()
                    })
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
        self.join_inflight()?;
        if pass.1 > 1 {
            return self.pass_at(at, arena, handles, routes, rect, span, pass);
        }
        self.segment_rows(at, arena, handles, routes, rect, hint, span)?;
        Ok(1)
    }

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
        self.slabs[at].pinned.fill(false);
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
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]).to_string())
                    .collect();
                let _ = writeln!(dump, "{at}\t{}", ids.join(" "));
            }
        }
        for entry in raw.as_chunks_mut::<4>().0 {
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
        self.flush()?;
        arena.write(first, &raw)?;
        if self.prefetch
            && let Some(rows) = self.predicted.get(at + 1).cloned().flatten()
        {
            self.prefetch(at + 1, &rows)?;
        }
        Ok(())
    }

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
        self.slabs[at].pinned.fill(false);
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
                .as_chunks::<4>()
                .0
                .iter()
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
            let seat_of = &self.slabs[at].seat_of;
            let (mut leading, trailing): (Vec<u32>, Vec<u32>) = order
                .into_iter()
                .partition(|&e| seat_of[e as usize].is_some());
            leading.extend(trailing);
            let order = leading;
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
        if crate::diag::on().cut_trace {
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
        if self.prefetch
            && let Some(next) = next
        {
            self.prefetch_group(at, &next)?;
        }
        Ok(self.passing[at]
            .as_ref()
            .map_or(0, |p| p.groups.len() as u32))
    }

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

    fn prefetch(&mut self, at: usize, rows: &[Vec<u32>]) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        self.slabs[at].pinned.fill(false);
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

    fn seat(&mut self, at: usize, expert: u32) -> Result<u32> {
        if let Some(seat) = self.slabs[at].seat_of[expert as usize] {
            let slab = &mut self.slabs[at];
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
        let slab = &mut self.slabs[at];
        slab.in_seat[seat as usize] = Some(expert);
        slab.seat_of[expert as usize] = Some(seat);
        slab.last_used[seat as usize] = self.tick;
        self.tick += 1;
        slab.pinned[seat as usize] = true;
        self.pending.push((at, seat, expert));
        Ok(seat)
    }

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

    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.swaps, self.segments)
    }

    #[must_use]
    pub fn hits(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    #[must_use]
    pub fn prediction(&self) -> Prediction {
        self.prediction
    }

    #[must_use]
    pub fn host_time(&self) -> (u64, u64, u64) {
        (self.cut_ns, self.copy_ns, self.wait_ns)
    }

    pub fn note_wait(&mut self, ns: u64) {
        self.wait_ns += ns;
    }

    #[must_use]
    pub fn source(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }

    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }
}

impl Drop for Tier {
    fn drop(&mut self) {
        let _ = self.join_inflight();
    }
}
