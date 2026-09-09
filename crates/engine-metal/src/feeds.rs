use model_ir::{ClassSet, Def, GeomKind, RuntimeInput, Selection, Trace, Ty, ValueId};

use crate::inputs::PortSeat;

#[derive(Debug, Clone, Default)]
pub(crate) struct Feeds {
    pub(crate) ports: Vec<(PortSeat, ClassSet)>,
    pub(crate) selections: Vec<Selection>,
    pub(crate) merged: Vec<MergedPort>,
    pub(crate) unlanded: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MergedPort {
    pub(crate) merge: ValueId,
    pub(crate) seat: PortSeat,
    pub(crate) select: Selection,
}

impl Feeds {
    pub(crate) fn seats(&self) -> Vec<PortSeat> {
        self.ports
            .iter()
            .map(|(seat, _)| *seat)
            .filter(|seat| seat.kind != engine::fire::PortKind::Voxels)
            .collect()
    }

    pub(crate) fn of(trace: &Trace, compiled: &model_compiler::CompiledModel) -> Feeds {
        let mut feeds = Feeds::default();
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            let Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            let seat = match *input {
                RuntimeInput::Latents { port, width } => Some(PortSeat {
                    kind: engine::fire::PortKind::Latents,
                    port,
                    width,
                    dtype: *dtype,
                }),
                RuntimeInput::LaneVector { port, width } => Some(PortSeat {
                    kind: engine::fire::PortKind::LaneVector,
                    port,
                    width,
                    dtype: *dtype,
                }),
                RuntimeInput::Context { port, width } => Some(PortSeat {
                    kind: engine::fire::PortKind::Context,
                    port,
                    width,
                    dtype: *dtype,
                }),
                RuntimeInput::AxisPositions { port, axes } => Some(PortSeat {
                    kind: engine::fire::PortKind::AxisPositions,
                    port,
                    width: u32::from(axes),
                    dtype: *dtype,
                }),
                RuntimeInput::Voxels { port, channels } => Some(PortSeat {
                    kind: engine::fire::PortKind::Voxels,
                    port,
                    width: channels,
                    dtype: *dtype,
                }),
                RuntimeInput::RowPermutation { select }
                | RuntimeInput::Geometry {
                    kind:
                        GeomKind::GroupIndptr { select }
                        | GeomKind::LaneIndptr { select }
                        | GeomKind::ReferenceTag { select },
                    ..
                } => {
                    if !feeds.selections.contains(&select) {
                        feeds.selections.push(select);
                    }
                    None
                }
                _ => None,
            };
            let Some(seat) = seat else { continue };
            let readers = reader_classes(trace, compiled, ValueId(at as u32));
            match feeds
                .ports
                .iter_mut()
                .find(|(have, _)| have.kind == seat.kind && have.port == seat.port)
            {
                Some((_, classes)) => {
                    for class in readers.iter() {
                        classes.insert(class);
                    }
                }
                None => feeds.ports.push((seat, readers)),
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
                let named = match *input {
                    RuntimeInput::Latents { port, .. } => (engine::fire::PortKind::Latents, port),
                    RuntimeInput::LaneVector { port, .. } => {
                        (engine::fire::PortKind::LaneVector, port)
                    }
                    RuntimeInput::Context { port, .. } => (engine::fire::PortKind::Context, port),
                    RuntimeInput::AxisPositions { port, .. } => {
                        (engine::fire::PortKind::AxisPositions, port)
                    }
                    _ => continue,
                };
                let seat = feeds
                    .ports
                    .iter()
                    .find(|(seat, _)| seat.kind == named.0 && seat.port == named.1)
                    .map(|(seat, _)| *seat);
                let (Some(seat), Some(select)) = (seat, Selection::of(guard)) else {
                    feeds.unlanded.push(ValueId(at as u32));
                    continue;
                };
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                if let Some((_, classes)) = feeds
                    .ports
                    .iter_mut()
                    .find(|(have, _)| have.kind == seat.kind && have.port == seat.port)
                {
                    for class in readers.iter() {
                        if select.holds(compiled.classes.classes[class].word()) {
                            classes.insert(class);
                        }
                    }
                }
                feeds.merged.push(MergedPort {
                    merge: ValueId(at as u32),
                    seat,
                    select,
                });
            }
        }
        feeds
    }
}

pub(crate) fn writer_classes(
    trace: &Trace,
    compiled: &model_compiler::CompiledModel,
    value: ValueId,
) -> ClassSet {
    use model_ir::Operands as _;
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

fn reader_classes(
    trace: &Trace,
    compiled: &model_compiler::CompiledModel,
    value: ValueId,
) -> ClassSet {
    use model_ir::Operands as _;
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
