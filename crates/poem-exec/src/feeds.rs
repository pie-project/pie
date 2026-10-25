//! The inputs a model's guest feeds, as a loaded plan reads them: each guest
//! input's seat and the classes that read it, the row selections its packings
//! are cut by, and the merges whose arm is a guest input.

use poem_compiler::CompiledModel;
use poem_ir::{
    ClassSet, Def, Dtype, GeomKind, InputAxis, InputDecl, LaneClass, LaneClasses, Operands as _,
    Readout, ReadoutKind, RuntimeInput, Selection, Trace, ValueId,
};

/// A guest input as a fire lands it: `width` values of `dtype` a lane, a row
/// or a voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seat {
    pub id: u32,
    pub axis: InputAxis,
    pub width: u32,
    pub dtype: Dtype,
}

impl Seat {
    #[must_use]
    pub fn per_lane(&self) -> bool {
        self.axis == InputAxis::Lane
    }

    #[must_use]
    pub fn voxel(&self) -> bool {
        self.axis == InputAxis::Voxel
    }

    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.width) * poem_compiler::arena::elem_bytes(self.dtype).unwrap_or(0)
    }
}

/// A merge whose arm is a guest input, landed in the merge's column before the
/// walk on the rows `select` admits.
#[derive(Debug, Clone, Copy)]
pub struct MergedInput {
    pub merge: ValueId,
    pub seat: Seat,
    pub select: Selection,
}

#[derive(Debug, Clone, Default)]
pub struct Feeds {
    /// Each guest input the plan reads, and the classes reading it.
    pub inputs: Vec<(Seat, ClassSet)>,
    pub selections: Vec<Selection>,
    /// The (query, key) packings each group-packed `attention.ragged` reads:
    /// the pairs a stated attention mask is staged for.
    pub ragged: Vec<(Selection, Selection)>,
    pub merged: Vec<MergedInput>,
    /// Merges over a runtime input no fire can land.
    pub unlanded: Vec<ValueId>,
}

impl Feeds {
    #[must_use]
    pub fn of(trace: &Trace, compiled: &CompiledModel) -> Feeds {
        let mut feeds = Feeds::default();
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            match *input {
                RuntimeInput::Guest {
                    id,
                    axis,
                    width,
                    dtype,
                } => {
                    let seat = Seat {
                        id,
                        axis,
                        width,
                        dtype,
                    };
                    let readers = reader_classes(trace, compiled, ValueId(at as u32));
                    feeds.inputs.push((seat, readers));
                }
                RuntimeInput::RowPermutation { select }
                | RuntimeInput::Geometry {
                    kind: GeomKind::GroupIndptr { select } | GeomKind::LaneIndptr { select },
                    ..
                } if !feeds.selections.contains(&select) => feeds.selections.push(select),
                _ => {}
            }
        }
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Merge(arms) = &decl.def else {
                continue;
            };
            for (arm, guard) in arms {
                let Def::Input(input) = &trace.values[arm.0 as usize].def else {
                    continue;
                };
                let seat = match *input {
                    RuntimeInput::Guest { id, axis, .. } if axis != InputAxis::Voxel => {
                        feeds.seat(id)
                    }
                    _ => None,
                };
                let (Some(seat), Some(select)) = (seat, Selection::of(guard)) else {
                    feeds.unlanded.push(ValueId(at as u32));
                    continue;
                };
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                if let Some((_, classes)) = feeds.inputs.iter_mut().find(|(s, _)| s.id == seat.id) {
                    for class in readers.iter() {
                        if select.holds(compiled.classes.classes[class].word()) {
                            classes.insert(class);
                        }
                    }
                }
                feeds.merged.push(MergedInput {
                    merge: ValueId(at as u32),
                    seat,
                    select,
                });
            }
        }
        let packing = |id: ValueId| match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
            Some(Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::GroupIndptr { select },
                ..
            })) => Some(*select),
            _ => None,
        };
        for node in &trace.nodes {
            if let poem_ir::Operation::Attn(poem_ir::Attn::Ragged {
                q_indptr,
                kv_indptr,
                ..
            }) = &node.op
                && let (Some(q), Some(kv)) = (packing(*q_indptr), packing(*kv_indptr))
                && !feeds.ragged.contains(&(q, kv))
            {
                feeds.ragged.push((q, kv));
            }
        }
        feeds
    }

    /// The seat of guest input `id`, if the plan reads it.
    #[must_use]
    pub fn seat(&self, id: u32) -> Option<Seat> {
        self.inputs
            .iter()
            .find(|(seat, _)| seat.id == id)
            .map(|(seat, _)| *seat)
    }

    /// The classes reading guest input `id`.
    #[must_use]
    pub fn readers(&self, id: u32) -> Option<&ClassSet> {
        self.inputs
            .iter()
            .find(|(seat, _)| seat.id == id)
            .map(|(_, readers)| readers)
    }

    /// The seats a fire stages per lane or per row; the voxel inputs land in
    /// the fire's one voxel rectangle.
    #[must_use]
    pub fn seats(&self) -> Vec<Seat> {
        self.inputs
            .iter()
            .map(|(seat, _)| *seat)
            .filter(|seat| !seat.voxel())
            .collect()
    }

    /// Every guest input the trace names, in id order, with the lane words
    /// whose class reads it.
    #[must_use]
    pub fn decls(&self, trace: &Trace, compiled: &CompiledModel) -> Vec<InputDecl> {
        let table = &compiled.classes;
        trace
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(id, named)| {
                let (seat, readers) = self
                    .inputs
                    .iter()
                    .find(|(seat, _)| seat.id as usize == id)?;
                Some(InputDecl {
                    name: named.name.clone(),
                    axis: seat.axis,
                    width: seat.width,
                    dtype: seat.dtype,
                    rows: named.rows,
                    mask: table.mask,
                    readers: readers
                        .iter()
                        .flat_map(|class| table.classes[class].words.iter().copied())
                        .collect(),
                })
            })
            .collect()
    }
}

/// The classes running a region that reads `value`.
#[must_use]
pub fn reader_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> ClassSet {
    let mut inputs: Vec<ValueId> = Vec::new();
    let mut readers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        if inputs.contains(&value) {
            readers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| readers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}

/// Each class of the plan as a lane landing in it is told of it; a class's
/// readout is its first export seam of `out`, `velocity`, `hidden`, `pixels`.
#[must_use]
pub fn lane_classes(trace: &Trace, compiled: &CompiledModel) -> LaneClasses {
    let kv = classes_reading(trace, compiled, |def| {
        matches!(def, Def::Cache(row) if matches!(
            trace.caches.get(*row as usize),
            Some(poem_ir::CacheRow::Kv { .. })
        ))
    });
    let state = classes_reading(trace, compiled, |def| {
        matches!(def, Def::Cache(row) if matches!(
            trace.caches.get(*row as usize),
            Some(poem_ir::CacheRow::State { .. })
        ))
    });
    let embeds = classes_reading(trace, compiled, |def| {
        matches!(def, Def::Input(RuntimeInput::Tokens))
    });
    let wide = classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Attn(poem_ir::Attn::Prefill { causal: false, .. })
        )
    });
    let [out, _, _, _, velocity, hidden, pixels] = poem_compiler::EXPORT_SEAMS;
    let exports: Vec<(ReadoutKind, Readout, ClassSet)> = [
        (out, ReadoutKind::Logits),
        (velocity, ReadoutKind::Velocity),
        (hidden, ReadoutKind::Hidden),
        (pixels, ReadoutKind::Pixels),
    ]
    .into_iter()
    .flat_map(|(name, kind)| {
        let seams: Vec<_> = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == name)
            .filter_map(|seam| seam.values.first().copied())
            .collect();
        // The last `hidden` a class writes is the one it reads out.
        let seams: Vec<_> = if kind == ReadoutKind::Hidden {
            seams.into_iter().rev().collect()
        } else {
            seams
        };
        seams.into_iter().filter_map(move |value| {
            let width = crate::store::kv::width_of(trace, value).ok()?;
            let readout = Readout {
                kind,
                width: u32::try_from(width).ok()?,
            };
            Some((kind, readout, writer_classes(trace, compiled, value)))
        })
    })
    .collect();
    let patches: Vec<([u32; 3], ClassSet)> = trace
        .values
        .iter()
        .filter_map(|decl| match decl.def {
            Def::Input(RuntimeInput::TokenGrid { p }) => Some(p),
            _ => None,
        })
        .map(|p| {
            let readers = classes_reading(
                trace,
                compiled,
                |def| matches!(def, Def::Input(RuntimeInput::TokenGrid { p: read }) if *read == p),
            );
            (p, readers)
        })
        .collect();
    let table = &compiled.classes;
    LaneClasses {
        mask: table.mask,
        classes: table
            .classes
            .iter()
            .enumerate()
            .map(|(class, landed)| LaneClass {
                words: landed.words.clone(),
                reads_kv: kv.contains(class),
                reads_state: state.contains(class),
                embeds: embeds.contains(class),
                wide: wide.contains(class),
                readout: exports
                    .iter()
                    .find(|(_, _, classes)| classes.contains(class))
                    .map(|(_, readout, _)| *readout),
                patch: patches
                    .iter()
                    .find(|(_, readers)| readers.contains(class))
                    .map(|(p, _)| *p),
            })
            .collect(),
    }
}

/// The classes running a region one of whose nodes `wanted` holds of.
#[must_use]
pub fn classes_running(
    trace: &Trace,
    compiled: &CompiledModel,
    wanted: impl Fn(&poem_ir::Operation) -> bool,
) -> ClassSet {
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        let runs = region.nodes.clone().any(|node| {
            trace
                .nodes
                .get(node as usize)
                .is_some_and(|node| wanted(&node.op))
        });
        if runs {
            for class in region.mask.iter() {
                classes.insert(class);
            }
        }
    }
    classes
}

/// The classes running a region one of whose nodes reads a value whose
/// definition `wanted` holds of.
#[must_use]
pub fn classes_reading(
    trace: &Trace,
    compiled: &CompiledModel,
    wanted: impl Fn(&Def) -> bool,
) -> ClassSet {
    let mut inputs: Vec<ValueId> = Vec::new();
    let mut readers: Vec<bool> = vec![false; trace.nodes.len()];
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        readers[at] = inputs.iter().any(|value| {
            trace
                .values
                .get(value.0 as usize)
                .is_some_and(|decl| wanted(&decl.def))
        });
    }
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        if region
            .nodes
            .clone()
            .any(|node| readers.get(node as usize).copied().unwrap_or(false))
        {
            for class in region.mask.iter() {
                classes.insert(class);
            }
        }
    }
    classes
}

/// The classes running a region that writes `value`, or an arm of it.
#[must_use]
pub fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> ClassSet {
    if let Some(Def::Merge(arms)) = trace.values.get(value.0 as usize).map(|decl| &decl.def) {
        let mut classes = ClassSet::default();
        for (arm, _) in arms {
            for class in writer_classes(trace, compiled, *arm).iter() {
                classes.insert(class);
            }
        }
        return classes;
    }
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}
