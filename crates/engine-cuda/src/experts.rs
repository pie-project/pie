#![allow(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet};

use poem_ir::{Def, Linear, Operation, ParamSource, Trace, ValueId};

use crate::device::Pinned;
use crate::error::{Fault, Result};

pub type Attachments = BTreeMap<usize, Vec<usize>>;

#[derive(Debug)]
pub enum Spill {
    Serving(crate::checkpoint_serving::Serving),
}

impl Spill {
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        let Spill::Serving(serving) = self;
        serving.path()
    }

    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let Spill::Serving(serving) = self;
        serving.plane(id)
    }

    fn remedy(&self) -> &'static str {
        "The artifact is the model's own `.zt`, which IS the serving file: \
         its objects are this trace's plane names, so a hole in it is a name this build \
         declares and that file does not hold. The stamp was checked at open, so this \
         is not a foreign recipe — it is a trace that has gained a plane since the \
         artifact was written. Run `pie model import --force` on the source it names."
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    Pinned,
    Mapped,
}

#[must_use]
pub fn pageable_access() -> bool {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

        let mut ordinal = 0i32;
        // SAFETY: `ordinal` is a live out-parameter.
        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return false;
        }
        let mut value = 0i32;
        // SAFETY: `value` is a live out-parameter and `ordinal` came from the runtime one line above.
        let status = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut value,
                rt::cudaDeviceAttr::cudaDevAttrPageableMemoryAccess,
                ordinal,
            )
        };
        status == rt::cudaError::cudaSuccess && value != 0
    }
    #[cfg(not(feature = "device"))]
    {
        false
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    groups: Vec<GroupPlan>,
    pinned_of: BTreeSet<usize>,
    mapped_of: BTreeSet<usize>,
    device_bytes: u64,
    host_bytes: u64,
    spill_bytes: u64,
    /// The routed banks the weight cache streams an expert at a time
    /// ([`Plan::host_experts`]).
    cached: Vec<usize>,
    /// The planes left in the serving file's mapping and never read by the
    /// device in place ([`Plan::file_weights`]).
    filed: BTreeSet<usize>,
    /// The routed banks the weight cache takes, which `device_bytes` never
    /// counted: their experts lie in the pool, not on the weight tier.
    pooled: BTreeSet<usize>,
}

#[derive(Debug, Clone)]
pub struct BankPlan {
    pub param: usize,
    pub experts: u32,
    pub stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPlane {
    pub param: usize,
    pub bytes: u64,
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub param: usize,
    pub name: String,
    pub planes: Vec<GroupPlane>,
    pub bytes: u64,
    pub held: Held,
}

/// The weights a load's plan sizes: every plane's bytes, and the routed
/// banks among them (dense banks, and packed ones as a group of planes).
#[derive(Debug, Clone)]
pub struct Ranking {
    full: u64,
    banks: Vec<BankPlan>,
    packed: Vec<GroupPlan>,
    /// The routed banks the weight cache takes ([`Ranking::caching`]) and
    /// each one's bytes: on no tier this plan cuts.
    cached: Vec<(usize, u64)>,
}

impl Ranking {
    pub fn of(trace: &Trace, planes: &Attachments) -> Result<Ranking> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes
            .iter()
            .map(|b| b.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let (banks, packed) = routed(trace, planes, &bytes)?;
        Ok(Ranking {
            full,
            banks,
            packed,
            cached: Vec::new(),
        })
    }

    /// The ranking of a load whose weight cache streams the routed banks
    /// `cached` an expert at a time: the plan seats no expert itself. A
    /// cached bank is on no tier the plan cuts (its experts lie in the
    /// pool's slots, or in the copy a miss is read from), and a dense
    /// routed bank the cache does not take stays whole on the device,
    /// beside the planes that cannot move.
    #[must_use]
    pub fn caching(mut self, cached: &[usize]) -> Ranking {
        for bank in &self.banks {
            if cached.contains(&bank.param) {
                let whole =
                    (u64::from(bank.experts) * bank.stride).next_multiple_of(crate::weights::ALIGN);
                self.cached.push((bank.param, whole));
            }
        }
        // A packed bank the cache takes, each plane of it.
        for group in &self.packed {
            if cached.contains(&group.param) {
                self.cached.extend(
                    group
                        .planes
                        .iter()
                        .map(|plane| (plane.param, plane.reserved)),
                );
            }
        }
        self
    }
}

impl Plan {
    pub fn of(trace: &Trace, planes: &Attachments) -> Result<Plan> {
        Ok(Plan::cut(&Ranking::of(trace, planes)?))
    }

    /// The plan of a load: every dense plane on the device, where dense
    /// weights always live. The routed banks the weight cache takes
    /// ([`Ranking::caching`]) are the pool's, not the weight tier's; the
    /// planes a weight cache streams are moved off the device after
    /// ([`Plan::host_rows`], [`Plan::host_experts`], [`Plan::file_weights`]),
    /// as are the planes the package declares host ([`Plan::host_declared`]).
    #[must_use]
    pub fn cut(ranking: &Ranking) -> Plan {
        let pooled: BTreeSet<usize> = ranking.cached.iter().map(|(param, _)| *param).collect();
        let cached_bytes: u64 = ranking.cached.iter().map(|(_, bytes)| *bytes).sum();
        Plan {
            device_bytes: ranking.full.saturating_sub(cached_bytes),
            pooled,
            ..Plan::default()
        }
    }

    /// Moves every plane the trace declares `ParamPlacement::Host` that this
    /// plan still seats on the device to the pinned host tier (served through
    /// its mapped device address), and returns the device bytes it freed.
    pub fn host_declared(&mut self, trace: &Trace) -> Result<u64> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let mut freed = 0u64;
        for (at, param) in trace.params.iter().enumerate() {
            if param.placement != poem_ir::ParamPlacement::Host
                || param.source != ParamSource::Checkpoint
                || self.streamed_whole(at)
            {
                continue;
            }
            let reserved = bytes[at].next_multiple_of(crate::weights::ALIGN);
            self.groups.push(GroupPlan {
                param: at,
                name: param.name.clone(),
                planes: vec![GroupPlane {
                    param: at,
                    bytes: bytes[at],
                    reserved,
                }],
                bytes: reserved,
                held: Held::Pinned,
            });
            self.pinned_of.insert(at);
            // A bank the weight cache takes was never counted on the device.
            if !self.pooled.contains(&at) {
                self.device_bytes = self.device_bytes.saturating_sub(reserved);
            }
            self.host_bytes += reserved;
            freed += reserved;
        }
        Ok(freed)
    }

    /// Moves the planes `params` (the tables the weight cache streams by
    /// their rows) to the pinned host tier, each served through its mapped
    /// device address: the copy a row not in the cache is read from, and
    /// the source a row promoted into it is filled from. Returns the device
    /// bytes it freed.
    pub fn host_rows(&mut self, trace: &Trace, params: &[usize]) -> Result<u64> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let mut freed = 0u64;
        for &at in params {
            let param = &trace.params[at];
            if param.source != ParamSource::Checkpoint || self.streamed_whole(at) {
                continue;
            }
            let reserved = bytes[at].next_multiple_of(crate::weights::ALIGN);
            self.groups.push(GroupPlan {
                param: at,
                name: param.name.clone(),
                planes: vec![GroupPlane {
                    param: at,
                    bytes: bytes[at],
                    reserved,
                }],
                bytes: reserved,
                held: Held::Pinned,
            });
            self.pinned_of.insert(at);
            // A bank the weight cache takes was never counted on the device.
            if !self.pooled.contains(&at) {
                self.device_bytes = self.device_bytes.saturating_sub(reserved);
            }
            self.host_bytes += reserved;
            freed += reserved;
        }
        Ok(freed)
    }

    /// [`Plan::host_rows`] for the routed banks `params` the weight cache
    /// streams an expert at a time, which the plan then names
    /// ([`Plan::cached_experts`]).
    pub fn host_experts(&mut self, trace: &Trace, params: &[usize]) -> Result<u64> {
        let freed = self.host_rows(trace, params)?;
        self.cached = params
            .iter()
            .copied()
            .filter(|param| self.pinned(*param))
            .collect();
        Ok(freed)
    }

    /// The routed banks the weight cache streams an expert at a time.
    #[must_use]
    pub fn cached_experts(&self) -> &[usize] {
        &self.cached
    }

    /// Leaves the planes `params` (weights the weight cache takes) in the
    /// serving file's mapping, neither on the device nor pinned. `experts`
    /// are the routed banks among them ([`Plan::cached_experts`]).
    ///
    /// Not `in_place`, the runtime holds them whole in the pool for the
    /// load's life: each is copied into its pool slots once, at the first
    /// weight residency, and never read where it lies. `in_place`, a
    /// deployment that streams reads a miss out of the mapping, which needs
    /// a device that faults mapped pages in ([`pageable_access`]).
    ///
    /// Returns the device bytes it freed.
    pub fn file_weights(
        &mut self,
        trace: &Trace,
        params: &[usize],
        experts: bool,
        in_place: bool,
    ) -> Result<u64> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let mut freed = 0u64;
        for &at in params {
            let param = &trace.params[at];
            if param.source != ParamSource::Checkpoint || self.streamed_whole(at) {
                continue;
            }
            let reserved = bytes[at].next_multiple_of(crate::weights::ALIGN);
            self.groups.push(GroupPlan {
                param: at,
                name: param.name.clone(),
                planes: vec![GroupPlane {
                    param: at,
                    bytes: bytes[at],
                    reserved,
                }],
                bytes: reserved,
                held: Held::Mapped,
            });
            self.mapped_of.insert(at);
            if !in_place {
                self.filed.insert(at);
            }
            // A bank the weight cache takes was never counted on the device.
            if !self.pooled.contains(&at) {
                self.device_bytes = self.device_bytes.saturating_sub(reserved);
            }
            self.spill_bytes += reserved;
            freed += reserved;
            if experts {
                self.cached.push(at);
            }
        }
        Ok(freed)
    }

    /// Whether `param` is left in the serving file ([`Plan::file_weights`]).
    #[must_use]
    pub fn filed(&self, param: usize) -> bool {
        self.filed.contains(&param)
    }

    /// Whether any plane is left in the serving file.
    #[must_use]
    pub fn files(&self) -> bool {
        !self.filed.is_empty()
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        !self.groups.is_empty()
    }

    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    #[must_use]
    pub fn pinned(&self, param: usize) -> bool {
        self.pinned_of.contains(&param)
    }

    #[must_use]
    pub fn mapped(&self, param: usize) -> bool {
        self.mapped_of.contains(&param)
    }

    #[must_use]
    pub fn streamed_whole(&self, param: usize) -> bool {
        self.pinned(param) || self.mapped(param)
    }

    #[must_use]
    pub fn spill_demand(&self) -> u64 {
        self.spill_bytes
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes
    }

    #[must_use]
    pub fn host_demand(&self) -> u64 {
        self.host_bytes
    }

    #[must_use]
    pub fn host_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.host_walk().0
    }

    #[must_use]
    pub fn host_image(&self) -> u64 {
        self.host_walk().1
    }

    fn host_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::with_capacity(self.groups.len());
        let mut at = 0u64;
        for group in &self.groups {
            if group.held != Held::Pinned {
                continue;
            }
            for plane in &group.planes {
                out.push((plane.param as u64, at, plane.bytes, plane.reserved));
                at += plane.reserved;
            }
        }
        (out, at)
    }

    #[must_use]
    pub fn mapped_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.mapped_walk().0
    }

    fn mapped_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::new();
        let mut at = 0u64;
        for group in &self.groups {
            if group.held != Held::Mapped {
                continue;
            }
            for plane in &group.planes {
                out.push((plane.param as u64, at, plane.bytes, plane.reserved));
                at += plane.reserved;
            }
        }
        (out, at)
    }
}

fn routed(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BankPlan>, Vec<GroupPlan>)> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut banks: Vec<BankPlan> = Vec::new();
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut arity: Option<u32> = None;
    // On ROCm a fused routed MLP's selects read their banks as they would
    // apart.
    #[cfg(feature = "rocm")]
    let selects = trace.nodes.iter().flat_map(|node| match &node.op {
        Operation::Linear(op) => vec![op],
        Operation::Fused(
            poem_ir::Fused::MoeExperts { select, down, .. }
            | poem_ir::Fused::MoeMega { select, down, .. },
        ) => vec![&**select, &**down],
        Operation::Fused(poem_ir::Fused::MoeSelectGlu { select, .. }) => vec![&**select],
        _ => Vec::new(),
    });
    #[cfg(not(feature = "rocm"))]
    let selects = trace.nodes.iter().filter_map(|node| match &node.op {
        Operation::Linear(op) => Some(op),
        _ => None,
    });
    for op in selects {
        let (bank, dense) = match op {
            Linear::MoeSelect {
                bank,
                bias: None,
                quantized: false,
                quant: None,
                ..
            } => (*bank, true),
            Linear::MoeSelect {
                bank,
                bias: Some(_),
                quantized: false,
                quant: None,
                ..
            }
            | Linear::MoeSelect {
                bank,
                bias: None,
                quantized: true,
                quant: None,
                ..
            }
            | Linear::MoeSelect {
                bank,
                bias: None,
                quantized: true,
                quant: Some(_),
                ..
            } => (*bank, false),
            _ => continue,
        };
        let at = weight_of(trace, bank)?;
        if !seen.insert(at) {
            continue;
        }
        let param = &trace.params[at];
        let experts = u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
        if experts == 0 || param.shape.len() < 2 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is read as a routed expert bank and does not declare \
                      `[experts, ...]`; a slot stride cannot be divided out of it",
            });
        }
        match arity {
            None => arity = Some(experts),
            Some(first) if first != experts => {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert bank whose expert count differs from an \
                          earlier bank of the same plan; one residency decision covers \
                          the plan, and two arities would make it two decisions",
                });
            }
            Some(_) => {}
        }
        if dense {
            let plane = bytes[at];
            if plane == 0 || !plane.is_multiple_of(u64::from(experts)) {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert bank whose bytes do not divide by its expert \
                          count — the experts of one bank are not equal, and the slot \
                          arithmetic the tier does would be wrong rather than refused",
                });
            }
            banks.push(BankPlan {
                param: at,
                experts,
                stride: plane / u64::from(experts),
            });
            continue;
        }
        let Some(companions) = planes.get(&at) else {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is read at a QUANTIZED routed matmul's `bank` port and the load \
                      plan pairs no scales plane with it. A split-plane bank is codes \
                      AND factors, both indexed by the same expert id, and this shell \
                      seats them as one group or not at all — a contract states the \
                      pair with `TensorContract::scaling`",
            });
        };
        let mut all: Vec<usize> = std::iter::once(at)
            .chain(companions.iter().copied())
            .collect();
        all.sort_unstable();
        all.dedup();
        let planes: Vec<GroupPlane> = all
            .iter()
            .map(|at| GroupPlane {
                param: *at,
                bytes: bytes[*at],
                reserved: bytes[*at].next_multiple_of(crate::weights::ALIGN),
            })
            .collect();
        let total = planes.iter().map(|plane| plane.reserved).sum();
        if total == 0 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is a split-plane quantized expert bank whose planes reserve no \
                      bytes at all; a group with nothing in it is a bank the store \
                      would seat at address zero",
            });
        }
        groups.push(GroupPlan {
            param: at,
            name: param.name.clone(),
            planes,
            bytes: total,
            held: Held::Pinned,
        });
    }
    banks.sort_by_key(|bank| bank.param);
    groups.sort_by_key(|group| group.param);
    Ok((banks, groups))
}

fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read at a routed matmul's `bank` port and is not a weight; a bank \
                  is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

#[derive(Debug)]
struct Whole {
    param: usize,
    host_at: u64,
    serving_at: u64,
}

#[derive(Debug)]
struct Mapped {
    param: usize,
    at: u64,
}

/// How the pinned host image starts: zeroed, to be landed cold, or
/// uninitialized, to be read back from the serving file's own image.
#[derive(Debug)]
pub enum Fill {
    Cold,
    Restored,
}

/// The weights a load keeps off the device, each read where it lies: the
/// planes the plan pins (the weight cache's host copies, the planes a
/// package declares host) in one page-locked image the device maps, and
/// the planes left in the serving file's mapping.
#[derive(Debug)]
pub struct Tier {
    plan: Plan,
    host: Pinned,
    wholes: Vec<Whole>,
    /// The spill the mapped planes point into, held mapped for the tier's
    /// life.
    _source: Option<Spill>,
    mapped: Vec<Mapped>,
}

impl Tier {
    pub fn open(plan: Plan, source: Option<Spill>, fill: Fill) -> Result<Tier> {
        let layout = plan.host_layout();
        let want = usize::try_from(plan.host_image()).unwrap_or(usize::MAX);
        let host = match fill {
            Fill::Cold => Pinned::mapped(want)?,
            Fill::Restored => Pinned::mapped_uninit(want)?,
        };
        let wholes = layout
            .iter()
            .map(|&(param, host_at, _, _)| Whole {
                param: usize::try_from(param).unwrap_or(usize::MAX),
                host_at,
                serving_at: host.device().saturating_add(host_at),
            })
            .collect();
        // A plane left in the file for the weight cache is copied out of
        // it, never touched by the device where it lies.
        let in_place = plan
            .groups()
            .iter()
            .any(|group| group.held == Held::Mapped && !plan.filed(group.param));
        if in_place && !pageable_access() {
            return Err(Fault::Residency(format!(
                "this load plans {} bytes onto the mapped tier and this device does \
                 not report `pageableMemoryAccess` (CUDA 12.2+ HMM), so a GPU touch of a \
                 mapped page cannot fault it in. Registering the mapping instead would \
                 page-lock every byte of it, which is the pinned tier under another name. \
                 Run on a device that reports the attribute.",
                plan.spill_demand(),
            )));
        }
        let mut mapped = Vec::new();
        for group in plan.groups() {
            if group.held != Held::Mapped {
                continue;
            }
            let Some(artifact) = source.as_ref() else {
                return Err(Fault::Residency(format!(
                    "`{}` is planned onto the mapped tier and this load opened no \
                     artifact to map it out of: a shell that planned a plane off the \
                     device it never sourced",
                    group.name,
                )));
            };
            for seat in &group.planes {
                let id = u32::try_from(seat.param).unwrap_or(u32::MAX);
                let Some(bytes) = artifact.plane(id) else {
                    return Err(Fault::Residency(format!(
                        "`{}` is planned onto the mapped tier and the artifact at {} \
                         carries no plane {id}. {}",
                        group.name,
                        artifact.path().display(),
                        artifact.remedy(),
                    )));
                };
                if bytes.len() as u64 != seat.bytes {
                    return Err(Fault::Residency(format!(
                        "`{}`'s plane {id} is {} bytes in the artifact and {} in this \
                         plan; the two were written from different traces",
                        group.name,
                        bytes.len(),
                        seat.bytes,
                    )));
                }
                mapped.push(Mapped {
                    param: seat.param,
                    at: bytes.as_ptr() as u64,
                });
            }
        }
        if !mapped.is_empty() {
            let bytes: u64 = plan.mapped_layout().iter().map(|plane| plane.2).sum();
            let groups = plan
                .groups()
                .iter()
                .filter(|group| group.held == Held::Mapped)
                .count();
            let filed = mapped
                .iter()
                .filter(|plane| plan.filed(plane.param))
                .count();
            eprintln!(
                "engine-cuda: the MAPPED tier holds {groups} group(s), {} plane(s), {bytes} \
                 byte(s) left in {}: {filed} plane(s) the pool will hold, copied into their \
                 slots at the first weight residency and never read where they lie; the \
                 rest read in place, a GPU touch faulting the page in over HMM",
                mapped.len(),
                source.as_ref().map_or_else(
                    || "<no artifact>".to_string(),
                    |artifact| artifact.path().display().to_string()
                ),
            );
        }
        Ok(Tier {
            plan,
            host,
            wholes,
            _source: source,
            mapped,
        })
    }

    /// Where `param` lies in the pinned image, from its start.
    #[must_use]
    pub fn host_offset(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| whole.host_at)
    }

    #[must_use]
    pub fn pinned_at(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| whole.serving_at)
    }

    #[must_use]
    pub fn mapped_at(&self, param: usize) -> Option<u64> {
        self.mapped
            .iter()
            .find(|plane| plane.param == param)
            .map(|plane| plane.at)
    }

    #[must_use]
    pub fn offloaded_at(&self, param: usize) -> Option<u64> {
        self.pinned_at(param).or_else(|| self.mapped_at(param))
    }

    #[must_use]
    pub fn host(&self) -> &Pinned {
        &self.host
    }

    /// Frees the pinned host image where it holds only `held`: the copies
    /// of weights the runtime holds whole in the pool for the load's life,
    /// which no fire reads. Returns the bytes freed; 0 where the image also
    /// holds a plane read in place.
    pub fn release_host(&mut self, held: &[usize]) -> u64 {
        if held.is_empty()
            || self.host.bytes() == 0
            || !self.wholes.iter().all(|whole| held.contains(&whole.param))
        {
            return 0;
        }
        let bytes = self.host.bytes() as u64;
        self.host = Pinned::empty();
        bytes
    }

    pub fn zero_host(&self) {
        self.host.zero();
    }

    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
}

#[cfg(test)]
mod tests {
    use poem::Platform;

    use super::*;

    fn gpt_oss() -> Trace {
        poem_compiler::catalog::deployment("gptoss-20b-bf16-mxfp4-kv-bf16")
            .expect("the catalog ships the deployment")
            .trace(Platform::Cuda)
    }

    fn scales_of(trace: &Trace) -> Attachments {
        let at: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        trace
            .params
            .iter()
            .enumerate()
            .filter(|(_, param)| param.dtype == poem_ir::Dtype::Mxfp4)
            .map(|(codes, param)| {
                let scales = poem::scales_name(&param.name);
                let scales = *at
                    .get(scales.as_str())
                    .unwrap_or_else(|| panic!("`{}` declares no scales plane", param.name));
                (codes, vec![scales])
            })
            .collect()
    }

    #[test]
    fn experts_every_case() {
        a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name();
        a_host_declared_table_leaves_the_device_and_nothing_else_does();
    }

    fn a_host_declared_table_leaves_the_device_and_nothing_else_does() {
        let trace = poem_compiler::catalog::deployment("qwen38-flash-next-mini-bf16-kv-bf16")
            .expect("the catalog ships the deployment")
            .trace(Platform::Cuda);
        let at = |name: &str| {
            trace
                .params
                .iter()
                .position(|param| param.name == name)
                .unwrap_or_else(|| panic!("the trace declares `{name}`"))
        };
        let table = at("ple.table");
        let mut plan = Plan::of(&trace, &Attachments::new()).expect("plans");
        let full = plan.device_demand();
        let freed = plan.host_declared(&trace).expect("the table moves");
        let bytes = crate::weights::plane_bytes(&trace).expect("sized")[table]
            .next_multiple_of(crate::weights::ALIGN);
        assert_eq!(freed, bytes, "the codes of the table and nothing else move");
        assert!(plan.pinned(table));
        assert_eq!(
            (0..trace.params.len())
                .filter(|&p| plan.streamed_whole(p))
                .count(),
            1,
            "only the table leaves the device"
        );
        assert_eq!(plan.device_demand(), full - bytes);
        assert_eq!(plan.host_demand(), bytes);
        assert_eq!(
            plan.host_declared(&trace).expect("again"),
            0,
            "a second call moves nothing"
        );

        let trace = gpt_oss();
        let mut plan = Plan::of(&trace, &scales_of(&trace)).expect("plans");
        assert_eq!(plan.host_declared(&trace).expect("nothing to move"), 0);
        assert!(
            !plan.streams(),
            "a trace that declares no host plane plans as before"
        );
    }

    fn a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name() {
        let trace = gpt_oss();
        let why = Plan::of(&trace, &Attachments::new())
            .expect_err("a packed bank with no pairing is not a bank this shell seats");
        let said = why.to_string();
        assert!(
            said.contains("pairs no scales plane"),
            "the refusal names what is missing: {said}"
        );
        assert!(
            said.contains("one group or not at all"),
            "and says what the group is for: {said}"
        );
    }
}
