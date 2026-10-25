//! The inputs a model's guest feeds, as a loaded plan reads them: each guest
//! input's seat and the classes that read it, the row selections its packings
//! are cut by, and the merges whose arm is a guest input.

use poem_compiler::CompiledModel;
use poem_ir::{
    ClassSet, Def, Dtype, GeomKind, InputAxis, InputDecl, Operands as _, RuntimeInput, Selection,
    Trace, ValueId,
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
                let (seat, readers) = self.inputs.iter().find(|(seat, _)| seat.id as usize == id)?;
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
