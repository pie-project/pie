//! The weight cache's routed experts (SPEC "Weight cache"): every routed
//! bank streams an expert at a time through read-only weight slots of the
//! chunk pool, as the runtime names them.
//!
//! The recipe lays the banks one MoE layer's selects read (gate/up and
//! down, each one's scales and zero points, the expert biases) in one
//! weight shape: an atom is one expert of one layer, its plane of every
//! bank back to back (each plane from a whole word), a slot one atom. The
//! banks themselves stay on the host (the artifact, mapped): the runtime
//! names which atoms are resident and in which slot
//! ([`ExpertCaches::residency`]); the engine fills a newly named slot from
//! the host and points the atom's entry at it. Every routed kernel reads an
//! expert through its atom's entry
//! ([`kernels_slang::linear::moe::select::ExpertTable`]) on every arm.
//!
//! **An expert no slot holds is staged.** Before the first select of a
//! layer whose experts are not all resident reads them, the frame so far
//! lands, the routes are read back, and each routed expert that no slot
//! holds is copied from the host into a seat of a host-visible stage the
//! kernels bind (Vulkan: host-visible memory read in place; wgpu: written
//! through the queue). The stage is a copy of the experts one layer of one
//! fire reads, never the bank: the host copy is the mapped artifact. Binding
//! the whole bank host-visible instead would pin a second host copy of
//! every expert (wgpu cannot bind mapped memory to a kernel at all), so
//! staging is the one road on both engines. A fire with every expert
//! resident flushes and stages nothing, and the stage is given back.
//!
//! Each route counts a use of its expert (at the layer's first bank), which
//! [`ExpertCaches::uses`] reports.

use std::collections::BTreeMap;

use engine::chunk::{ShapeId, WeightAtom, WeightHeld, WeightKind, WeightResidency, WeightShape};
use kernels_slang::Tensor;
use kernels_slang::linear::moe::select::ExpertTable;
use poem_compiler::chunk::{ChunkRecipe, WeightKind as RecipeKind};
use poem_ir::{Def, Dtype, Linear, Operands, Operation, ParamSource, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{HostPlane, HostTier};
use crate::store::Pools;
use crate::store::rows::{NOT_HELD, Uses};

/// Bytes one fill moves through its host-visible upload at most.
const UPLOAD: u64 = 64 << 20;

/// The routed planes of `trace` the cache streams: every plane of the
/// checkpoint the residency census reads an expert at a time, where every
/// op that reads it is a routed select (its bank or its expert bias) or a
/// routed bias sum, the ops whose kernels read an expert table. A bank's
/// companions (scales, zero points) stream with it.
#[must_use]
pub fn streamed(trace: &Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, leads, residency};
    let classes = residency(trace);
    let lead = leads(trace);
    let weight = |v: poem_ir::ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    };
    let mut leads_read = vec![false; trace.params.len()];
    let mut refused = vec![false; trace.params.len()];
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        let tabled: Vec<poem_ir::ValueId> = match &node.op {
            Operation::Linear(Linear::MoeSelect {
                bank,
                bias,
                quant: None,
                ..
            }) => std::iter::once(*bank).chain(*bias).collect(),
            Operation::Linear(Linear::MoeBiasSum { bias, .. }) => vec![*bias],
            _ => Vec::new(),
        };
        for v in &inputs {
            let Some(at) = weight(*v) else {
                continue;
            };
            if tabled.contains(v) {
                leads_read[at] = true;
            } else {
                refused[at] = true;
            }
        }
    }
    (0..trace.params.len())
        .filter(|&at| {
            let p = lead[at];
            matches!(classes[at], Residency::Expert(_))
                && trace.params[at].source == ParamSource::Checkpoint
                && trace.params[at].shape.len() >= 2
                && leads_read[p]
                && !refused[p]
        })
        .collect()
}

/// One plane of a layer's atom: its param, where it lies in the atom and
/// the bytes an expert of it takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub param: usize,
    pub origin: u64,
    pub bytes: u64,
    pub name: String,
}

/// One MoE layer's banks in a shape: their planes, its first atom and its
/// experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub planes: Vec<Plane>,
    pub first: u32,
    pub experts: u32,
}

/// The layers whose experts are alike, streamed an expert a slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertShape {
    pub id: ShapeId,
    pub layers: Vec<Layer>,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub count: u32,
    pub floor: u32,
    /// Experts one layer of a fire may stage at most.
    pub seats: u32,
    pub name: String,
}

impl ExpertShape {
    /// The recipe's expert slots of one piece whose planes are all
    /// `streamed`, a shape each, ids from `first`.
    pub fn of(
        trace: &Trace,
        recipe: &ChunkRecipe,
        streamed: &[usize],
        first: u32,
        max_tokens: u32,
    ) -> Result<Vec<ExpertShape>> {
        use poem_ir::residency::{Residency, residency};
        let classes = residency(trace);
        let bytes = crate::weights::plane_bytes(trace)?;
        let mut shapes = Vec::new();
        for slot in &recipe.weights {
            if slot.kind != RecipeKind::Expert
                || slot.pieces != 1
                || slot.banks.is_empty()
                || !slot.banks.iter().all(|bank| {
                    !bank.params.is_empty()
                        && bank
                            .params
                            .iter()
                            .all(|&p| streamed.contains(&(p as usize)))
                })
            {
                continue;
            }
            let mut layers = Vec::with_capacity(slot.banks.len());
            let mut atoms = 0u32;
            let mut top_k = 1u32;
            for bank in &slot.banks {
                let experts = u32::try_from(bank.atoms).unwrap_or(u32::MAX);
                let mut origin = 0u64;
                let mut planes = Vec::with_capacity(bank.params.len());
                for &p in &bank.params {
                    let p = p as usize;
                    let held = trace.params[p].shape.first().copied().unwrap_or(1).max(1);
                    if held != u64::from(experts) || !bytes[p].is_multiple_of(held) {
                        return Err(Fault::Unbound {
                            what: format!(
                                "routed plane `{}`, whose {} bytes are not {experts} experts",
                                trace.params[p].name, bytes[p]
                            ),
                        });
                    }
                    if let Residency::Expert(atom) = classes[p] {
                        top_k = top_k.max(atom.top_k);
                    }
                    let per = bytes[p] / held;
                    planes.push(Plane {
                        param: p,
                        origin,
                        bytes: per,
                        name: trace.params[p].name.clone(),
                    });
                    origin = (origin + per).next_multiple_of(4);
                }
                if origin > slot.bytes {
                    return Err(Fault::Unbound {
                        what: format!(
                            "an expert atom of {origin} bytes laid a plane from a word, where \
                             the recipe's slot holds {}",
                            slot.bytes
                        ),
                    });
                }
                layers.push(Layer {
                    planes,
                    first: atoms,
                    experts,
                });
                atoms = atoms.saturating_add(experts);
            }
            let widest = layers.iter().map(|l| l.experts).max().unwrap_or(0);
            let seats = widest.min(max_tokens.max(1).saturating_mul(top_k)).max(1);
            let id = ShapeId(first + u32::try_from(shapes.len()).unwrap_or(u32::MAX));
            shapes.push(ExpertShape {
                id,
                name: trace.params[layers[0].planes[0].param].name.clone(),
                layers,
                slot_bytes: slot.bytes,
                per_chunk: u32::try_from(slot.per_chunk(recipe.chunk_bytes)).unwrap_or(u32::MAX),
                count: atoms,
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX),
                seats,
            });
        }
        Ok(shapes)
    }

    #[must_use]
    pub fn facts(&self) -> WeightShape {
        WeightShape {
            id: self.id,
            kind: WeightKind::Expert,
            slot_bytes: self.slot_bytes,
            slots_per_chunk: self.per_chunk,
            count: self.count,
            floor_slots: self.floor,
            name: self.name.clone(),
            // The runtime places every expert; the load seats none.
            seated: None,
        }
    }

    /// Every param of this shape.
    pub fn params(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers
            .iter()
            .flat_map(|l| l.planes.iter().map(|p| p.param))
    }
}

/// One expert shape's cache: which slot each atom lies in, the device's
/// view of that, the uses counted and the experts staged a fire.
#[derive(Debug)]
struct Cache {
    shape: ExpertShape,
    /// Each atom's slot ref, [`NOT_HELD`] where none.
    slots: Vec<u32>,
    /// Each atom's pool unit as kernels read it.
    units: Buffer,
    /// A use a route read, an atom each.
    hits: Buffer,
    /// Each atom's seat of the stage this fire, where staged.
    staged: Buffer,
    /// The experts staged for the layer a fire reads now; none while every
    /// atom is resident.
    stage: Option<Buffer>,
    /// The set is the load's for life.
    held: bool,
    /// The bytes a pool unit of this shape counts: `gcd(chunk, slot)`.
    unit: u64,
}

impl Cache {
    fn resident(&self) -> u32 {
        u32::try_from(self.slots.iter().filter(|&&s| s != NOT_HELD).count()).unwrap_or(u32::MAX)
    }

    fn stage_bytes(&self) -> u64 {
        u64::from(self.shape.seats) * self.shape.slot_bytes
    }
}

/// The routes a fire's layer read back and staged: `(routes handle, rows,
/// width, cache, layer, region, run)`.
pub type StageKey = (u32, u32, u32, usize, usize, u32, u32);

/// The load's routed experts.
#[derive(Debug)]
pub struct ExpertCaches {
    caches: Vec<Cache>,
    /// Each streamed plane's cache, layer and place among the layer's.
    by_param: BTreeMap<usize, (usize, usize, usize)>,
    /// What a binding with nothing to read binds.
    spare: Buffer,
}

impl ExpertCaches {
    /// The caches of `shapes`, nothing resident.
    pub fn new(device: &Context, shapes: Vec<ExpertShape>, chunk_bytes: u64) -> Result<Self> {
        let mut caches = Vec::with_capacity(shapes.len());
        let mut by_param = BTreeMap::new();
        for shape in shapes {
            let atoms = shape.count.max(1);
            for (l, layer) in shape.layers.iter().enumerate() {
                for (k, plane) in layer.planes.iter().enumerate() {
                    by_param.insert(plane.param, (caches.len(), l, k));
                }
            }
            let mut units = Buffer::host(device, u64::from(atoms) * 4)?;
            units.write(0, &vec![0xffu8; atoms as usize * 4])?;
            let mut staged = Buffer::host(device, u64::from(atoms) * 4)?;
            staged.write(0, &vec![0xffu8; atoms as usize * 4])?;
            let mut cache = Cache {
                slots: vec![NOT_HELD; shape.count as usize],
                units,
                hits: Buffer::host(device, u64::from(atoms) * 4)?,
                staged,
                stage: None,
                held: false,
                unit: crate::store::layout::gcd(chunk_bytes, shape.slot_bytes),
                shape,
            };
            cache.stage = Some(Buffer::host(device, cache.stage_bytes().max(4))?);
            caches.push(cache);
        }
        Ok(ExpertCaches {
            caches,
            by_param,
            spare: Buffer::host(device, 16)?,
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    /// The weight shapes the runtime counts.
    #[must_use]
    pub fn facts(&self) -> Vec<WeightShape> {
        self.caches.iter().map(|c| c.shape.facts()).collect()
    }

    /// Whether every expert of every shape is resident: no fire stages.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.caches.iter().all(|c| !c.slots.contains(&NOT_HELD))
    }

    #[must_use]
    pub fn streams(&self, param: usize) -> bool {
        self.by_param.contains_key(&param)
    }

    /// Whether any slot of these caches lies in logical chunks `chunks` (a
    /// seat run an engine's own fire writes).
    #[must_use]
    pub fn holds_in(&self, chunks: std::ops::Range<u32>) -> bool {
        self.caches.iter().any(|c| {
            c.slots
                .iter()
                .any(|&s| s != NOT_HELD && chunks.contains(&(s / c.shape.per_chunk.max(1))))
        })
    }

    /// The planes a routed op reads, by param: `bank`'s codes (or its
    /// dense plane), scales and zero points, and the expert bias `bias`;
    /// `None` where `bank` (else `bias`) is not streamed.
    #[must_use]
    pub fn planes_of(
        &self,
        bank: Option<usize>,
        bias: Option<usize>,
    ) -> Option<[Option<usize>; 4]> {
        let lead = bank.or(bias)?;
        let &(c, l, _) = self.by_param.get(&lead)?;
        let layer = &self.caches[c].shape.layers[l];
        let companion = |suffix: &str| {
            let bank = bank?;
            let name = format!(
                "{}{suffix}",
                layer.planes.iter().find(|p| p.param == bank)?.name
            );
            layer
                .planes
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.param)
        };
        Some([
            bank,
            companion(poem_ir::SCALES),
            companion(poem_ir::BIASES),
            bias,
        ])
    }

    /// The host bytes a fire's staging holds now.
    #[must_use]
    pub fn stage_bytes(&self) -> u64 {
        self.caches
            .iter()
            .filter_map(|c| c.stage.as_ref())
            .map(Buffer::bytes)
            .sum()
    }

    /// Takes `residency` as the whole set (its atoms of these shapes): fills
    /// each slot newly named from the host (`host`, the planes by param in
    /// `planes`), rewrites the atoms' entries, and gives the stage back once
    /// every atom is resident. The caller has drained every fire. The slots
    /// it filled.
    pub fn residency(
        &mut self,
        device: &Context,
        pools: &Pools,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        residency: &WeightResidency,
    ) -> Result<usize> {
        let mut named: Vec<Vec<u32>> = self
            .caches
            .iter()
            .map(|c| vec![NOT_HELD; c.shape.count as usize])
            .collect();
        for resident in &residency.resident {
            let Some(at) = self
                .caches
                .iter()
                .position(|c| c.shape.id == resident.atom.shape)
            else {
                continue;
            };
            let cache = &self.caches[at];
            let atom = resident.atom.atom as usize;
            if atom >= named[at].len() {
                return Err(Fault::Ceiling {
                    what: "atoms of an expert shape",
                    need: u64::from(resident.atom.atom) + 1,
                    have: u64::from(cache.shape.count),
                });
            }
            pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, resident.slot)?;
            named[at][atom] = resident.slot;
        }
        for (cache, named) in self.caches.iter().zip(&named) {
            if cache.held
                && (!residency.held || named.iter().zip(&cache.slots).any(|(n, s)| n != s))
            {
                return Err(Fault::Unbound {
                    what: format!(
                        "a weight residency that moves the experts of `{}`, held for the \
                         load's life",
                        cache.shape.name
                    ),
                });
            }
        }
        let mut fills: Vec<(u64, usize, u32)> = Vec::new();
        for (at, (cache, named)) in self.caches.iter().zip(&named).enumerate() {
            for (atom, (&want, &have)) in named.iter().zip(&cache.slots).enumerate() {
                if want != NOT_HELD && want != have {
                    let to =
                        pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, want)?;
                    fills.push((to, at, u32::try_from(atom).unwrap_or(u32::MAX)));
                }
            }
        }
        self.fill(device, pools, host, planes, &mut fills)?;
        let filled = fills.len();
        for (cache, named) in self.caches.iter_mut().zip(named) {
            cache.slots = named;
            cache.held = residency.held;
            // The stage is a fire's copy of what no slot holds: none is
            // needed once every atom is resident.
            if !cache.slots.contains(&NOT_HELD) {
                cache.stage = None;
            } else if cache.stage.is_none() {
                cache.stage = Some(Buffer::host(device, cache.stage_bytes().max(4))?);
            }
        }
        self.restage(pools)?;
        Ok(filled)
    }

    /// Atom `atom` of `shape` laid as its slot holds it: each plane's
    /// expert row from its origin.
    fn image(
        shape: &ExpertShape,
        atom: u32,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        into: &mut [u8],
    ) -> Result<()> {
        let layer = shape
            .layers
            .iter()
            .rfind(|l| l.first <= atom)
            .ok_or(Fault::Ceiling {
                what: "atoms of an expert shape",
                need: u64::from(atom) + 1,
                have: u64::from(shape.count),
            })?;
        let expert = u64::from(atom - layer.first);
        for plane in &layer.planes {
            let host_plane = planes.get(&plane.param).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "routed plane param {}, which the host tier does not hold",
                    plane.param
                ),
            })?;
            let from = usize::try_from(plane.origin).unwrap_or(usize::MAX);
            let len = usize::try_from(plane.bytes).unwrap_or(0);
            host_plane.row_into(host.map(), expert, &mut into[from..from + len])?;
        }
        Ok(())
    }

    /// Fills the slots `fills` names (virtual byte, cache, atom) with their
    /// experts from the host, in uploads of [`UPLOAD`] bytes at most.
    fn fill(
        &self,
        device: &Context,
        pools: &Pools,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        fills: &mut [(u64, usize, u32)],
    ) -> Result<()> {
        fills.sort_unstable();
        let mut start = 0usize;
        while start < fills.len() {
            let mut bytes = 0u64;
            let mut end = start;
            while end < fills.len() {
                let slot = self.caches[fills[end].1].shape.slot_bytes;
                if bytes > 0 && bytes + slot > UPLOAD {
                    break;
                }
                bytes += slot;
                end += 1;
            }
            let mut upload = Buffer::host(device, bytes.max(4))?;
            let mut at = 0u64;
            let mut image = Vec::new();
            for &(_, cache, atom) in &fills[start..end] {
                let shape = &self.caches[cache].shape;
                image.clear();
                image.resize(usize::try_from(shape.slot_bytes).unwrap_or(0), 0);
                Self::image(shape, atom, host, planes, &mut image)?;
                upload.write(at, &image)?;
                at += shape.slot_bytes;
            }
            // One copy of many regions a segment, slots side by side merged.
            let mut frame = device.frame()?;
            let mut at = 0u64;
            let mut regions: Vec<(u64, u64, u64)> = Vec::new();
            let mut into: Option<(&Buffer, u64)> = None;
            for &(to, cache, _) in &fills[start..end] {
                let slot = self.caches[cache].shape.slot_bytes;
                let (buffer, local) = pools.locate(to)?;
                let segment = to - local;
                if into.is_some_and(|(_, held)| held != segment) {
                    let (buffer, _) = into.take().expect("checked");
                    for &(from, local, len) in &regions {
                        frame.copy(&upload, from, buffer, local, len)?;
                    }
                    regions.clear();
                }
                into = Some((buffer, segment));
                match regions.last_mut() {
                    Some(last) if last.0 + last.2 == at && last.1 + last.2 == local => {
                        last.2 += slot;
                    }
                    _ => regions.push((at, local, slot)),
                }
                at += slot;
            }
            if let Some((buffer, _)) = into {
                for &(from, local, len) in &regions {
                    frame.copy(&upload, from, buffer, local, len)?;
                }
            }
            frame.commit()?;
            start = end;
        }
        Ok(())
    }

    /// Rewrites every atom's entry from its slot ref through the pool's
    /// chunk table: after a residency, and after an evacuation moves a
    /// chunk. The caller has drained every fire that reads the entries.
    pub fn restage(&mut self, pools: &Pools) -> Result<()> {
        for cache in &mut self.caches {
            let mut units = Vec::with_capacity(cache.slots.len() * 4);
            for &slot in &cache.slots {
                let unit = if slot == NOT_HELD {
                    NOT_HELD
                } else {
                    let at =
                        pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, slot)?;
                    u32::try_from(at / cache.unit).map_err(|_| Fault::Ceiling {
                        what: "pool units an expert table's entry names",
                        need: at / cache.unit,
                        have: u64::from(u32::MAX),
                    })?
                };
                units.extend_from_slice(&unit.to_le_bytes());
            }
            cache.units.write(0, &units)?;
        }
        Ok(())
    }

    /// The uses counted since the last call, and where each shape's atoms
    /// lie; the counts start over. The caller has drained every fire.
    pub fn uses(&mut self) -> Result<Uses> {
        let mut uses = Vec::new();
        let mut held = Vec::with_capacity(self.caches.len());
        for cache in &mut self.caches {
            let atoms = cache.shape.count as usize;
            let mut raw = vec![0u8; atoms * 4];
            cache.hits.read(0, &mut raw)?;
            for (atom, word) in raw.as_chunks::<4>().0.iter().enumerate() {
                let n = u32::from_le_bytes(*word);
                if n > 0 {
                    uses.push((
                        WeightAtom {
                            shape: cache.shape.id,
                            atom: u32::try_from(atom).unwrap_or(u32::MAX),
                        },
                        n,
                    ));
                }
            }
            cache.hits.zero_span(0, raw.len() as u64)?;
            let resident = cache.resident();
            held.push(WeightHeld {
                shape: cache.shape.id,
                resident,
                host: cache.shape.count - resident,
            });
        }
        Ok((uses, held))
    }

    /// The expert table a routed op reading `planes` (`[codes or the dense
    /// plane, scales, zero points, the expert bias]`, by param) through
    /// `routes` reads, the pool as `pool` binds it; `count`: its routes
    /// count uses (the layer's first bank). Where some routed expert is in
    /// no slot, the frame so far lands, the routes are read back and those
    /// experts are staged, once a layer of a fire (`key`, held in `read`).
    /// `None` where the planes are not streamed.
    #[allow(clippy::too_many_arguments)]
    pub fn table(
        &self,
        handles: &Handles,
        host: &HostTier,
        host_planes: &BTreeMap<usize, &HostPlane>,
        planes: [Option<usize>; 4],
        routes: Tensor,
        at: (u32, u32),
        read: &std::cell::RefCell<Option<StageKey>>,
        pool: kernels_slang::Segments,
    ) -> Result<Option<ExpertTable>> {
        let Some(lead) = planes.iter().flatten().next() else {
            return Ok(None);
        };
        let Some(&(c, l, _)) = self.by_param.get(lead) else {
            return Ok(None);
        };
        let cache = &self.caches[c];
        let layer = &cache.shape.layers[l];
        let mut origins = [0u32; 4];
        for (k, param) in planes.iter().enumerate() {
            let Some(param) = param else {
                continue;
            };
            let Some(&(pc, pl, pk)) = self.by_param.get(param) else {
                return Err(Fault::Unbound {
                    what: format!(
                        "routed plane param {param}, read beside a streamed bank and not \
                         streamed with it"
                    ),
                });
            };
            if (pc, pl) != (c, l) {
                return Err(Fault::Unbound {
                    what: format!("routed plane param {param}, streamed in another layer's atom"),
                });
            }
            origins[k] = u32::try_from(layer.planes[pk].origin / 4).unwrap_or(u32::MAX);
        }
        let count = planes[0].is_some() && planes[0] == Some(layer.planes[0].param);
        let first = layer.first as usize;
        let missing = cache.slots[first..first + layer.experts as usize].contains(&NOT_HELD);
        if missing {
            let key: StageKey = (routes.buf, routes.rows, routes.width, c, l, at.0, at.1);
            if *read.borrow() != Some(key) {
                self.stage(handles, host, host_planes, cache, layer, routes)?;
                *read.borrow_mut() = Some(key);
            }
        }
        let bind = |buffer: &Buffer, len: u64| -> Result<Tensor> {
            let handle = handles.bind(buffer, 0, len.max(4))?;
            Ok(Tensor::new(
                handle,
                u32::try_from(len.max(4) / 4).unwrap_or(u32::MAX),
                1,
                Dtype::U32,
            ))
        };
        let atoms = u64::from(cache.shape.count.max(1)) * 4;
        let stage = match &cache.stage {
            Some(stage) => bind(stage, stage.bytes())?,
            None => bind(&self.spare, self.spare.bytes())?,
        };
        Ok(Some(ExpertTable {
            units: bind(&cache.units, atoms)?,
            staged: bind(&cache.staged, atoms)?,
            stage,
            hits: bind(&cache.hits, atoms)?,
            pool,
            unit_words: u32::try_from(cache.unit / 4).unwrap_or(u32::MAX),
            atom_words: u32::try_from(cache.shape.slot_bytes / 4).unwrap_or(u32::MAX),
            atom_first: layer.first,
            origins,
            count,
        }))
    }

    /// Lands the frame so far, reads `routes` back and stages each expert
    /// of `layer` they name that no slot holds, a seat each.
    fn stage(
        &self,
        handles: &Handles,
        host: &HostTier,
        host_planes: &BTreeMap<usize, &HostPlane>,
        cache: &Cache,
        layer: &Layer,
        routes: Tensor,
    ) -> Result<()> {
        let stage = cache.stage.as_ref().ok_or_else(|| Fault::Unbound {
            what: "the stage of an expert shape not wholly resident".into(),
        })?;
        crate::probe::with_frame(Frame::flush_settled)
            .ok_or_else(|| Fault::Residency("no frame is recording on this thread".into()))??;
        let n = u64::from(routes.rows) * u64::from(routes.width);
        let raw = handles.read(routes.buf, n * 4)?;
        let mut seated: BTreeMap<u32, u32> = BTreeMap::new();
        for word in raw.as_chunks::<4>().0 {
            let e = i32::from_le_bytes(*word);
            let Ok(e) = u32::try_from(e) else {
                continue;
            };
            if e >= layer.experts || cache.slots[(layer.first + e) as usize] != NOT_HELD {
                continue;
            }
            let next = u32::try_from(seated.len()).unwrap_or(u32::MAX);
            seated.entry(e).or_insert(next);
        }
        if seated.len() as u64 > u64::from(cache.shape.seats) {
            return Err(Fault::Ceiling {
                what: "experts one layer of a fire stages",
                need: seated.len() as u64,
                have: u64::from(cache.shape.seats),
            });
        }
        let mut image = vec![0u8; usize::try_from(cache.shape.slot_bytes).unwrap_or(0)];
        let mut entries = vec![0xffu8; layer.experts as usize * 4];
        for (&e, &seat) in &seated {
            image.fill(0);
            Self::image(&cache.shape, layer.first + e, host, host_planes, &mut image)?;
            stage.write_shared(u64::from(seat) * cache.shape.slot_bytes, &image)?;
            entries[e as usize * 4..e as usize * 4 + 4].copy_from_slice(&seat.to_le_bytes());
        }
        cache
            .staged
            .write_shared(u64::from(layer.first) * 4, &entries)?;
        Ok(())
    }
}
