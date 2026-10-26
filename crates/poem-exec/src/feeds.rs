use poem_compiler::CompiledModel;
use poem_ir::{
    ClassSet, Def, Dim, Dtype, GeomKind, InputAxis, InputDecl, LaneClass, LaneClasses,
    Operands as _, OutputDecl, RuntimeInput, Selection, Trace, Ty, ValueId,
};

/// A guest input as a fire lands it: `width` values of `dtype` per lane, row or voxel.
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

/// A merge over a guest input, landed in the merge's column before the walk.
#[derive(Debug, Clone, Copy)]
pub struct MergedInput {
    pub merge: ValueId,
    pub seat: Seat,
    pub select: Selection,
}

#[derive(Debug, Clone, Default)]
pub struct Feeds {
    pub inputs: Vec<(Seat, ClassSet)>,
    pub selections: Vec<Selection>,
    /// The (query, key) packings a stated attention mask is staged for.
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

    #[must_use]
    pub fn seat(&self, id: u32) -> Option<Seat> {
        self.inputs
            .iter()
            .find(|(seat, _)| seat.id == id)
            .map(|(seat, _)| *seat)
    }

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

/// An output as the classes in `classes` write it: its values a layer after
/// another (one when it is not written per layer), and the grid laying out a
/// voxel output's own clips.
#[derive(Debug, Clone)]
pub struct ClassOutput {
    pub decl: OutputDecl,
    pub values: Vec<ValueId>,
    pub grid: Option<ValueId>,
    pub classes: ClassSet,
}

/// Every output by the classes writing it. Classes writing the same values
/// of a name share one; a name written at several layers stacks them.
pub fn class_outputs(trace: &Trace, compiled: &CompiledModel) -> Result<Vec<ClassOutput>, String> {
    // A class whose words all set two values of one switch is no lane's.
    let table = &compiled.classes;
    let landed = ClassSet::of((0..table.classes.len()).filter(|&class| {
        table.classes[class]
            .words
            .iter()
            .any(|&word| one_value_a_switch(&trace.facts, word))
    }));
    let mut outs: Vec<ClassOutput> = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    for output in &trace.outputs {
        if !names.contains(&output.name.as_str()) {
            names.push(&output.name);
        }
    }
    for name in names {
        let records: Vec<(&poem_ir::Output, ClassSet)> = trace
            .outputs
            .iter()
            .filter(|output| output.name == name)
            .map(|output| (output, computing_classes(trace, compiled, output.value)))
            .collect();
        for class in landed.iter() {
            let mine: Vec<&poem_ir::Output> = records
                .iter()
                .filter(|(_, writers)| writers.contains(class))
                .map(|(output, _)| *output)
                .collect();
            let Some(first) = mine.first() else {
                continue;
            };
            let values: Vec<ValueId> = mine.iter().map(|output| output.value).collect();
            if let Some(out) = outs
                .iter_mut()
                .find(|out| out.decl.name == name && out.values == values)
            {
                out.classes.insert(class);
                continue;
            }
            let decl = output_decl(trace, name, &values)?;
            outs.push(ClassOutput {
                decl,
                values,
                grid: first.grid,
                classes: ClassSet::of([class]),
            });
        }
    }
    Ok(outs)
}

fn output_decl(trace: &Trace, name: &str, values: &[ValueId]) -> Result<OutputDecl, String> {
    let shape = |value: ValueId| match trace.values.get(value.0 as usize).map(|decl| &decl.ty) {
        Some(Ty::Tensor { shape, dtype }) => Ok((shape.first().copied(), *dtype)),
        _ => Err(format!("output `{name}` reads out no tensor")),
    };
    let (first, dtype) = shape(values[0])?;
    let width = crate::store::kv::width_of(trace, values[0])
        .map_err(|why| format!("output `{name}`: {why}"))?;
    for value in &values[1..] {
        let width_here = crate::store::kv::width_of(trace, *value)
            .map_err(|why| format!("output `{name}`: {why}"))?;
        if shape(*value)? != (first, dtype) || width_here != width {
            return Err(format!(
                "output `{name}` is written at {} layers in different shapes",
                values.len()
            ));
        }
    }
    let axis = match first {
        Some(Dim::Lanes | Dim::LanesPlus(_)) => InputAxis::Lane,
        Some(Dim::Voxels | Dim::VoxelsScaled(..)) => InputAxis::Voxel,
        Some(Dim::Tokens | Dim::TokensTimes(_) | Dim::Readouts) => InputAxis::Row,
        other => {
            return Err(format!(
                "output `{name}` lies along {other:?}, no lane, row or voxel"
            ));
        }
    };
    let layers = if values.len() > 1 {
        if axis != InputAxis::Lane {
            return Err(format!(
                "output `{name}` is written per layer along {axis:?}; only a lane output stacks \
                 its layers"
            ));
        }
        u32::try_from(values.len()).unwrap_or(u32::MAX)
    } else {
        0
    };
    let tail = values
        .iter()
        .filter_map(|value| observed(trace, *value))
        .max()
        .unwrap_or(0);
    Ok(OutputDecl {
        name: name.to_string(),
        axis,
        width: u32::try_from(width).map_err(|_| format!("output `{name}` is {width} wide"))?,
        dtype,
        layers,
        readout: first == Some(Dim::Readouts),
        tail,
    })
}

fn one_value_a_switch(facts: &poem_ir::Facts, word: u64) -> bool {
    facts.switches.iter().all(|decl| {
        facts
            .table
            .iter()
            .enumerate()
            .filter(|(bit, fact)| {
                matches!(fact, poem_ir::Fact::Switch { name, .. } if *name == decl.name)
                    && word & (1 << bit) != 0
            })
            .count()
            <= 1
    })
}

/// The classes whose nodes compute `value`, or an arm of it.
fn computing_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> ClassSet {
    match trace.values.get(value.0 as usize).map(|decl| &decl.def) {
        Some(Def::Op(node)) => compiled
            .classes
            .node_mask
            .get(*node as usize)
            .cloned()
            .unwrap_or_default(),
        Some(Def::Merge(_)) => match compiled.classes.arms_of(value) {
            Some(arms) => ClassSet::of(
                arms.iter()
                    .enumerate()
                    .filter(|(_, arm)| arm.is_some())
                    .map(|(class, _)| class),
            ),
            None => ClassSet::default(),
        },
        _ => ClassSet::default(),
    }
}

/// The rows a score capture `value` (or its gather across ranks) observes.
fn observed(trace: &Trace, value: ValueId) -> Option<u32> {
    let Def::Op(node) = trace.values.get(value.0 as usize)?.def else {
        return None;
    };
    match trace.nodes.get(node as usize)?.op {
        poem_ir::Operation::Attn(poem_ir::Attn::ScoreCapture { observe, .. }) => Some(observe),
        poem_ir::Operation::Collective(poem_ir::Collective::AllGather { x, .. }) => {
            observed(trace, x)
        }
        _ => None,
    }
}

#[must_use]
pub fn lane_classes(
    trace: &Trace,
    compiled: &CompiledModel,
    outputs: &[ClassOutput],
) -> LaneClasses {
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
                outputs: outputs
                    .iter()
                    .filter(|out| out.classes.contains(class))
                    .map(|out| out.decl.clone())
                    .collect(),
                patch: patches
                    .iter()
                    .find(|(_, readers)| readers.contains(class))
                    .map(|(p, _)| *p),
            })
            .collect(),
    }
}

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
