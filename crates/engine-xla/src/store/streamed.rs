//! The weight cache on XLA (SPEC "Weight cache", WEIGHT-CACHE-CUDA.md): the
//! weights a load streams live as read-only slots of the one chunk pool,
//! and a fire reads them through tables the host writes.
//!
//! Two kinds, as on CUDA and Vulkan:
//!
//! - **Rows**: a bf16 table every reader of which is a plain token-id
//!   `Gather::Embed` (per-layer embeddings, an untied embedding). An atom
//!   is a block of `block` rows. The host knows every row a fire reads (its
//!   token ids; ids the device fed are read back first), so each row's
//!   place in the pool is a per-fire index input: a resident block's slot,
//!   or, for a block in no slot, a staging slot the host fills from its
//!   copy before the fire. The program gathers the rows from a view of the
//!   pool; it never reads a block table.
//! - **Experts**: dense bf16 banks every reader of which is a plain routed
//!   select (no bias, not quantized). An atom is one expert of one MoE
//!   layer, its plane of every bank of the layer back to back. A select
//!   reads its bank as the dense `[E·rows, cols]` value the base build
//!   reads, gathered from the pool by a per-fire input of each expert's
//!   first row (`Run::streamed_bank`), so the kernels are the base build's.
//!   Routes are decided inside the fire, so a fire whose layers are not
//!   all resident is cut before each such layer's first select
//!   (`crate::trace::Tracer::cut`): the segment before the cut lands, the
//!   routes come back, the experts they name that lie in no slot are
//!   staged, and the next segment reads them.
//!
//! The load decides: a load told its weights are held
//! (`Residency::weights_held`, a deployment that does not stream) states
//! no weight shape and lands every plane as the base build does, so its
//! programs are the base build's. A streaming load keeps each streamed
//! plane's bytes on the host (the miss copy) and seats every atom in the
//! pool at load (`WeightShape::seated`): the top chunks below the sink, the
//! shapes' alone. A held residency naming every atom drops the miss copy;
//! after it a residency that moves an atom is refused (refilling from the
//! serving file is not built here).

use std::collections::{BTreeMap, BTreeSet};

use engine::chunk::{ShapeId, WeightAtom, WeightHeld, WeightKind, WeightResidency, WeightShape};
use poem_ir::{Def, Dtype, Fused, Gather, Linear, Operands, Operation, ParamSource, Trace};

use crate::error::{Fault, Result};
use crate::store::layout::ROW_BYTES;

/// Expert layers laid alike: each bank's `(rows, cols)`, the experts a
/// layer, and each layer's trace layer and banks.
type Group = (Vec<(u32, u32)>, u32, Vec<(Option<u32>, Vec<usize>)>);

/// An atom in no slot.
pub const NOT_HELD: u32 = u32::MAX;

/// Bytes a row block aims at.
const ROW_BLOCK: u64 = 64 << 10;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

fn align_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

/// One plane of an atom: `rows` rows of `width` bf16 at `offset` bytes
/// into the slot, read through the pool's `view`-wide bf16 view (a table
/// a row at a time; an expert's plane in pool rows of 512 bytes, so the
/// chunk needs no other multiple).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pub offset: u64,
    pub rows: u32,
    pub width: u32,
    pub view: u32,
}

impl Plane {
    /// Bytes a row of the view it is read through takes.
    #[must_use]
    pub fn view_row_bytes(&self) -> u64 {
        u64::from(self.view) * 2
    }

    /// Rows of its view the plane takes.
    #[must_use]
    pub fn view_rows(&self) -> u32 {
        u32::try_from(self.bytes() / self.view_row_bytes().max(1)).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.width) * 2
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.rows) * self.row_bytes()
    }
}

/// What a weight shape's atoms are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Form {
    /// One table `[vocab, width]`, `block` rows an atom.
    Rows {
        param: usize,
        block: u32,
        vocab: u32,
    },
    /// Routed experts, `experts` a layer: each layer's trace layer and its
    /// banks' params in plane order (atom `layer · experts + expert`).
    Experts {
        experts: u32,
        layers: Vec<(Option<u32>, Vec<usize>)>,
    },
}

/// One weight shape and its cache.
#[derive(Debug, Clone)]
pub struct Shape {
    pub id: ShapeId,
    pub name: String,
    pub form: Form,
    /// Each plane's place in a slot (one for rows).
    pub planes: Vec<Plane>,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub count: u32,
    pub floor: u32,
    /// The first logical chunk the load seated the shape in.
    pub first_chunk: u32,
    /// Each atom's slot ref, [`NOT_HELD`] where none.
    pub slots: Vec<u32>,
    /// The staging slots the runtime named.
    pub staging: Vec<u32>,
    /// Reads since the uses were last taken, an atom each.
    pub uses: Vec<u32>,
}

impl Shape {
    #[must_use]
    pub fn kind(&self) -> WeightKind {
        match self.form {
            Form::Rows { .. } => WeightKind::Rows,
            Form::Experts { .. } => WeightKind::Expert,
        }
    }

    #[must_use]
    pub fn facts(&self) -> WeightShape {
        WeightShape {
            id: self.id,
            kind: self.kind(),
            slot_bytes: self.slot_bytes,
            slots_per_chunk: self.per_chunk,
            count: self.count,
            floor_slots: self.floor,
            name: self.name.clone(),
            seated: Some(self.first_chunk * self.per_chunk),
        }
    }

    /// The chunks the load seated the shape in.
    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.count.div_ceil(self.per_chunk.max(1))
    }

    fn resident(&self) -> u32 {
        u32::try_from(self.slots.iter().filter(|&&s| s != NOT_HELD).count()).unwrap_or(u32::MAX)
    }
}

/// The streamed weights of a load.
#[derive(Debug, Default)]
pub struct Streamed {
    pub shapes: Vec<Shape>,
    /// Each streamed param's shape and plane.
    by_param: BTreeMap<usize, (usize, usize)>,
    /// Each streamed param's bytes, row-major as declared: the miss copy.
    host: BTreeMap<usize, Vec<u8>>,
    /// A held residency named every atom: the miss copy is gone.
    held: bool,
    /// What every slot start and chunk is a multiple of.
    pub unit: u64,
    /// The largest slot.
    pub least: u64,
}

/// A trace param read as a weight value.
fn param_of(trace: &Trace, v: poem_ir::ValueId) -> Option<usize> {
    match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Weight(p)) => Some(*p as usize),
        _ => None,
    }
}

/// A plain dense routed select: its bank.
fn plain_select(op: &Linear) -> Option<poem_ir::ValueId> {
    match op {
        Linear::MoeSelect {
            bank,
            bias: None,
            quantized: false,
            quant: None,
            input_scales: None,
            ..
        } => Some(*bank),
        _ => None,
    }
}

/// The tables a plain token-id gather alone reads: bf16, two axes, from
/// the checkpoint, every reader a `Gather::Embed` with no weights, scales or
/// concat over the table's own rows (engine-vulkan `rows::streamed`).
#[must_use]
pub fn streamed_rows(trace: &Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    let classes = residency(trace);
    let mut readers: BTreeMap<usize, bool> = BTreeMap::new();
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for &v in &inputs {
            let Some(p) = param_of(trace, v) else {
                continue;
            };
            // The host reads the rows a fire gathers off its own token
            // ids, so the gather reads the fire's tokens.
            let ok = match &node.op {
                Operation::Gather(Gather::Embed {
                    ids,
                    table,
                    weights: None,
                    scales: None,
                    concat: false,
                    vocab,
                    ..
                }) => {
                    *table == v
                        && u64::from(*vocab) == trace.params[p].shape[0]
                        && matches!(
                            trace.values.get(ids.0 as usize).map(|d| &d.def),
                            Some(Def::Input(poem_ir::RuntimeInput::Tokens))
                        )
                }
                _ => false,
            };
            let entry = readers.entry(p).or_insert(true);
            *entry &= ok;
        }
    }
    for decl in &trace.values {
        if let Def::Merge(arms) = &decl.def {
            for (v, _) in arms {
                if let Some(p) = param_of(trace, *v) {
                    readers.insert(p, false);
                }
            }
        }
    }
    readers
        .into_iter()
        .filter(|&(p, ok)| {
            let param = &trace.params[p];
            ok && matches!(classes[p], Residency::Rows(_))
                && param.dtype == Dtype::Bf16
                && param.shape.len() == 2
                && param.source == ParamSource::Checkpoint
        })
        .map(|(p, _)| p)
        .collect()
}

/// The routed banks every reader of which is a plain dense select at its
/// bank port (alone or inside a fused routed MLP): bf16 `[E, rows, cols]`
/// from the checkpoint. Each with its expert count and layer.
#[must_use]
pub fn streamed_experts(trace: &Trace) -> Vec<(usize, u32, Option<u32>)> {
    use poem_ir::residency::{Residency, residency};
    let classes = residency(trace);
    let mut readers: BTreeMap<usize, bool> = BTreeMap::new();
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        let banks: Vec<poem_ir::ValueId> = match &node.op {
            Operation::Linear(op) => plain_select(op).into_iter().collect(),
            Operation::Fused(Fused::MoeExperts { select, down, .. }) => {
                [plain_select(select), plain_select(down)]
                    .into_iter()
                    .flatten()
                    .collect()
            }
            _ => Vec::new(),
        };
        for &v in &inputs {
            let Some(p) = param_of(trace, v) else {
                continue;
            };
            *readers.entry(p).or_insert(true) &= banks.contains(&v);
        }
    }
    for decl in &trace.values {
        if let Def::Merge(arms) = &decl.def {
            for (v, _) in arms {
                if let Some(p) = param_of(trace, *v) {
                    readers.insert(p, false);
                }
            }
        }
    }
    readers
        .into_iter()
        .filter_map(|(p, ok)| {
            let param = &trace.params[p];
            let Residency::Expert(atom) = classes[p] else {
                return None;
            };
            (ok && param.dtype == Dtype::Bf16
                && param.shape.len() == 3
                && param.shape[0] == u64::from(atom.experts)
                && param.experts.is_none()
                && param.source == ParamSource::Checkpoint)
                .then_some((p, atom.experts, atom.layer))
        })
        .collect()
}

impl Streamed {
    /// The shapes of `trace`'s streamed weights, ids from `first`; a fire
    /// stages up to `rows` token rows.
    pub fn plan(trace: &Trace, first: u32, rows: u32) -> Result<Streamed> {
        let mut shapes = Vec::new();
        let mut by_param = BTreeMap::new();
        let mut unit = ROW_BYTES;
        let mut least = 0u64;
        let mut next = first;
        for p in streamed_rows(trace) {
            let param = &trace.params[p];
            let vocab = u32::try_from(param.shape[0]).unwrap_or(u32::MAX);
            let width = u32::try_from(param.shape[1]).unwrap_or(u32::MAX);
            let row = u64::from(width) * 2;
            // Whole pool rows a block, near `ROW_BLOCK`.
            let step = ROW_BYTES / gcd(ROW_BYTES, row);
            let block = align_up((ROW_BLOCK / row).max(1), step);
            let block = u32::try_from(block).unwrap_or(u32::MAX);
            let plane = Plane {
                offset: 0,
                rows: block,
                width,
                view: width,
            };
            let slot_bytes = plane.bytes();
            unit = lcm(unit, row);
            least = least.max(slot_bytes);
            let count = vocab.div_ceil(block);
            by_param.insert(p, (shapes.len(), 0));
            shapes.push(Shape {
                id: ShapeId(next),
                name: param.name.clone(),
                form: Form::Rows {
                    param: p,
                    block,
                    vocab,
                },
                planes: vec![plane],
                slot_bytes,
                per_chunk: 0,
                count,
                floor: count.min(rows.max(1)),
                first_chunk: 0,
                slots: Vec::new(),
                staging: Vec::new(),
                uses: Vec::new(),
            });
            next += 1;
        }
        // Experts: one shape a layout of a layer's banks, the layers in
        // trace order of their first bank.
        let mut layers: Vec<(Option<u32>, u32, Vec<usize>)> = Vec::new();
        for (p, experts, layer) in streamed_experts(trace) {
            match layers
                .iter_mut()
                .find(|(l, e, _)| *l == layer && *e == experts && layer.is_some())
            {
                Some((_, _, banks)) => banks.push(p),
                None => layers.push((layer, experts, vec![p])),
            }
        }
        let mut groups: Vec<Group> = Vec::new();
        for (layer, experts, banks) in layers {
            let dims: Vec<(u32, u32)> = banks
                .iter()
                .map(|&p| {
                    let s = &trace.params[p].shape;
                    (
                        u32::try_from(s[1]).unwrap_or(u32::MAX),
                        u32::try_from(s[2]).unwrap_or(u32::MAX),
                    )
                })
                .collect();
            match groups
                .iter_mut()
                .find(|(d, e, _)| *d == dims && *e == experts)
            {
                Some((_, _, held)) => held.push((layer, banks)),
                None => groups.push((dims, experts, vec![(layer, banks)])),
            }
        }
        for (dims, experts, held) in groups {
            // An expert plane is read in whole pool rows; one that is not
            // whole rows stays resident.
            if dims
                .iter()
                .any(|&(r, c)| !(u64::from(r) * u64::from(c) * 2).is_multiple_of(ROW_BYTES))
            {
                continue;
            }
            let mut offset = 0u64;
            let mut planes = Vec::with_capacity(dims.len());
            let mut shape_unit = ROW_BYTES;
            for &(rows, width) in &dims {
                let plane = Plane {
                    offset: 0,
                    rows,
                    width,
                    view: u32::try_from(ROW_BYTES / 2).unwrap_or(256),
                };
                offset = align_up(offset, plane.view_row_bytes());
                planes.push(Plane { offset, ..plane });
                offset += align_up(plane.bytes(), plane.view_row_bytes());
                shape_unit = lcm(shape_unit, plane.view_row_bytes());
            }
            let slot_bytes = align_up(offset, shape_unit);
            unit = lcm(unit, shape_unit);
            least = least.max(slot_bytes);
            let at = shapes.len();
            for (layer_at, (_, banks)) in held.iter().enumerate() {
                for (plane, &p) in banks.iter().enumerate() {
                    by_param.insert(p, (at, layer_at * dims.len() + plane));
                }
            }
            let name = trace.params[held[0].1[0]].name.clone();
            let count = experts * u32::try_from(held.len()).unwrap_or(u32::MAX);
            shapes.push(Shape {
                id: ShapeId(next),
                name,
                form: Form::Experts {
                    experts,
                    layers: held,
                },
                planes,
                slot_bytes,
                per_chunk: 0,
                count,
                floor: experts,
                first_chunk: 0,
                slots: Vec::new(),
                staging: Vec::new(),
                uses: Vec::new(),
            });
            next += 1;
        }
        Ok(Streamed {
            shapes,
            by_param,
            host: BTreeMap::new(),
            held: false,
            unit,
            least,
        })
    }

    /// Numbers the shapes from `first` on (past the state shapes).
    pub fn renumber(&mut self, first: u32) {
        for (at, shape) in self.shapes.iter_mut().enumerate() {
            shape.id = ShapeId(first + at as u32);
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Whether `param` streams.
    #[must_use]
    pub fn streams(&self, param: usize) -> bool {
        self.by_param.contains_key(&param)
    }

    #[must_use]
    pub fn params(&self) -> BTreeSet<usize> {
        self.by_param.keys().copied().collect()
    }

    /// The shape and plane a streamed param lies in.
    #[must_use]
    pub fn place(&self, param: usize) -> Option<(usize, usize)> {
        self.by_param.get(&param).copied()
    }

    /// Lays the shapes into chunks of `chunk_bytes` from logical chunk
    /// `first` on, every atom seated in order: the chunks this takes.
    pub fn seat(&mut self, chunk_bytes: u64, first: u32) -> Result<u32> {
        let mut at = first;
        for shape in &mut self.shapes {
            if !chunk_bytes.is_multiple_of(self.unit) || shape.slot_bytes > chunk_bytes {
                return Err(Fault::Unbound {
                    what: format!(
                        "weight shape {} ({} bytes a slot) in chunks of {chunk_bytes} bytes, \
                         which are no multiple of its {}-byte unit",
                        shape.name, shape.slot_bytes, self.unit
                    ),
                });
            }
            shape.per_chunk = u32::try_from(chunk_bytes / shape.slot_bytes).unwrap_or(u32::MAX);
            shape.first_chunk = at;
            shape.slots = (0..shape.count).map(|a| at * shape.per_chunk + a).collect();
            shape.uses = vec![0; shape.count as usize];
            at += shape.chunks();
        }
        Ok(at - first)
    }

    /// The chunks every shape takes seated whole.
    #[must_use]
    pub fn chunks(&self, chunk_bytes: u64) -> u32 {
        self.shapes
            .iter()
            .map(|s| {
                let per = (chunk_bytes / s.slot_bytes.max(1)).max(1);
                u32::try_from(u64::from(s.count).div_ceil(per)).unwrap_or(u32::MAX)
            })
            .sum()
    }

    /// Keeps `bytes`, param `param`'s landed bytes, as its miss copy.
    pub fn keep(&mut self, param: usize, bytes: &[u8]) {
        self.host.insert(param, bytes.to_vec());
    }

    #[must_use]
    pub fn facts(&self) -> Vec<WeightShape> {
        self.shapes.iter().map(Shape::facts).collect()
    }

    /// Atom `atom` of shape `at`, laid as its slot holds it.
    pub fn atom_bytes(&self, at: usize, atom: u32) -> Result<Vec<u8>> {
        let shape = &self.shapes[at];
        let mut out = vec![0u8; shape.slot_bytes as usize];
        let copy_of =
            |p: usize| {
                self.host.get(&p).ok_or_else(|| Fault::Residency(format!(
                "weight `{}` keeps no host copy (a held set dropped it), so atom {atom} cannot \
                 be filled again",
                shape.name
            )))
            };
        match &shape.form {
            Form::Rows { param, block, .. } => {
                let host = copy_of(*param)?;
                let row = shape.planes[0].row_bytes() as usize;
                let from = atom as usize * *block as usize * row;
                let to = (from + *block as usize * row).min(host.len());
                if from < to {
                    out[..to - from].copy_from_slice(&host[from..to]);
                }
            }
            Form::Experts { experts, layers } => {
                let layer = (atom / experts) as usize;
                let expert = (atom % experts) as usize;
                let (_, banks) = layers.get(layer).ok_or_else(|| Fault::Unbound {
                    what: format!("expert atom {atom} of {}", shape.name),
                })?;
                for (plane, &p) in shape.planes.iter().zip(banks) {
                    let host = copy_of(p)?;
                    let n = plane.bytes() as usize;
                    let from = expert * n;
                    out[plane.offset as usize..plane.offset as usize + n]
                        .copy_from_slice(&host[from..from + n]);
                }
            }
        }
        Ok(out)
    }

    /// Takes `residency` as the whole set: each atom's slot from here on.
    /// Answers the atoms newly named a slot, `(shape, atom, slot)`, which
    /// the caller fills ([`Streamed::atom_bytes`]) before any fire reads
    /// them.
    pub fn residency(&mut self, residency: &WeightResidency) -> Result<Vec<(usize, u32, u32)>> {
        if self.held && !residency.held {
            return Err(Fault::Residency(
                "a weight residency that is not held after a held one".to_string(),
            ));
        }
        let mut named: Vec<Vec<u32>> = self
            .shapes
            .iter()
            .map(|s| vec![NOT_HELD; s.count as usize])
            .collect();
        for r in &residency.resident {
            let Some(at) = self.shapes.iter().position(|s| s.id == r.atom.shape) else {
                return Err(Fault::Residency(format!(
                    "weight shape {}, which this load does not state",
                    r.atom.shape
                )));
            };
            let Some(entry) = named[at].get_mut(r.atom.atom as usize) else {
                return Err(Fault::Residency(format!(
                    "atom {} of weight shape {}, which holds {}",
                    r.atom.atom, r.atom.shape, self.shapes[at].count
                )));
            };
            *entry = r.slot;
        }
        let mut fills = Vec::new();
        for (at, (shape, named)) in self.shapes.iter().zip(&named).enumerate() {
            for (atom, (&now, &was)) in named.iter().zip(&shape.slots).enumerate() {
                if now != NOT_HELD && now != was {
                    fills.push((at, atom as u32, now));
                }
            }
        }
        if self.host.is_empty() && !fills.is_empty() {
            return Err(Fault::Residency(format!(
                "the residency moves {} weight atom(s) of a held set, which keeps no host \
                 copy to fill them from",
                fills.len()
            )));
        }
        for (shape, named) in self.shapes.iter_mut().zip(named) {
            shape.slots = named;
            shape.staging = residency
                .staging
                .iter()
                .find(|(id, _)| *id == shape.id)
                .map(|(_, slots)| slots.clone())
                .unwrap_or_default();
        }
        if residency.held {
            self.held = true;
        }
        Ok(fills)
    }

    /// After the fills of a held residency landed: with every atom named,
    /// the miss copy goes (held = no host copy kept).
    pub fn settle_held(&mut self) {
        if self.held && self.whole() && !self.host.is_empty() {
            let bytes: usize = self.host.values().map(Vec::len).sum();
            self.host.clear();
            tracing::info!(
                host_mib = bytes >> 20,
                "xla weights held: every atom resident, the host copy dropped"
            );
        }
    }

    /// Every atom of every shape in a slot.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.shapes.iter().all(|s| !s.slots.contains(&NOT_HELD))
    }

    /// Whether every expert of `layer` (a trace layer) lies in a slot.
    #[must_use]
    pub fn layer_whole(&self, at: usize, layer_at: usize) -> bool {
        let shape = &self.shapes[at];
        let Form::Experts { experts, .. } = shape.form else {
            return true;
        };
        let from = layer_at * experts as usize;
        shape.slots[from..from + experts as usize]
            .iter()
            .all(|&s| s != NOT_HELD)
    }

    /// Counts one read of each of `atoms` of shape `at`.
    pub fn count(&mut self, at: usize, atoms: impl IntoIterator<Item = u32>) {
        let uses = &mut self.shapes[at].uses;
        for atom in atoms {
            if let Some(n) = uses.get_mut(atom as usize) {
                *n = n.saturating_add(1);
            }
        }
    }

    /// The uses counted since the last call, zeroed, and where each shape's
    /// atoms lie.
    pub fn uses(&mut self) -> (Vec<(WeightAtom, u32)>, Vec<WeightHeld>) {
        let mut uses = Vec::new();
        let mut held = Vec::with_capacity(self.shapes.len());
        for shape in &mut self.shapes {
            for (atom, n) in shape.uses.iter_mut().enumerate() {
                if *n > 0 {
                    uses.push((
                        WeightAtom {
                            shape: shape.id,
                            atom: atom as u32,
                        },
                        std::mem::take(n),
                    ));
                }
            }
            let resident = shape.resident();
            held.push(WeightHeld {
                shape: shape.id,
                resident,
                host: shape.count - resident,
            });
        }
        (uses, held)
    }

    /// Whether any atom lies in logical chunks `chunks`.
    #[must_use]
    pub fn holds_in(&self, chunks: std::ops::Range<u32>) -> bool {
        self.shapes.iter().any(|s| {
            s.slots
                .iter()
                .any(|&r| r != NOT_HELD && chunks.contains(&(r / s.per_chunk.max(1))))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_block_is_whole_pool_rows_and_whole_table_rows() {
        for width in [256u64, 2048, 160, 96, 2880] {
            let row = width * 2;
            let step = ROW_BYTES / gcd(ROW_BYTES, row);
            let block = align_up((ROW_BLOCK / row).max(1), step);
            assert_eq!((block * row) % ROW_BYTES, 0, "width {width}");
        }
    }
}

/// What one fire's walk reads the streamed weights through.
#[derive(Debug, Default)]
pub struct Fire {
    /// Each streamed table: its param, the pool view its rows are read
    /// from, and each token row's row of that view.
    pub rows: Vec<(usize, crate::handle::Tensor, crate::handle::Tensor)>,
    /// Each streamed bank: its param and how a select reads it.
    pub banks: Vec<(usize, Bank)>,
    /// The routes each cut brought back, in cut order (`Cut`).
    pub cuts: std::sync::Mutex<Vec<Cut>>,
    /// The `(routes, span)` already cut for.
    pub cut_for: std::sync::Mutex<Vec<(poem_ir::ValueId, (u32, u32))>>,
}

/// How a select reads a streamed bank: expert `e`'s plane is rows
/// `[loc[e], loc[e] + rows)` of `view`.
#[derive(Debug, Clone, Copy)]
pub struct Bank {
    pub view: crate::handle::Tensor,
    pub loc: crate::handle::Tensor,
    /// The bank's rows an expert, and the view's rows an expert takes.
    pub rows: u32,
    pub view_rows: u32,
    pub experts: u32,
    /// The shape and its layer (`Form::Experts::layers`).
    pub shape: usize,
    pub layer: usize,
    /// Every expert of the layer lies in a slot: no cut.
    pub whole: bool,
}

/// Where a cut fire's routes for one layer came back: the handle the
/// routes value had, and the layer.
#[derive(Debug, Clone, Copy)]
pub struct Cut {
    pub routes: crate::handle::Tensor,
    pub shape: usize,
    pub layer: usize,
}

impl Fire {
    #[must_use]
    pub fn rows_of(&self, param: usize) -> Option<(crate::handle::Tensor, crate::handle::Tensor)> {
        self.rows
            .iter()
            .find(|(p, ..)| *p == param)
            .map(|&(_, view, idx)| (view, idx))
    }

    #[must_use]
    pub fn bank_of(&self, param: usize) -> Option<Bank> {
        self.banks
            .iter()
            .find(|(p, _)| *p == param)
            .map(|&(_, bank)| bank)
    }
}
