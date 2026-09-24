use model_compiler::CompiledModel;
use model_ir::{Operands, Trace, ValueId};

use crate::error::{Fault, Result};

pub(crate) const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];
pub(crate) const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];
pub(crate) const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];
pub(crate) const FLOAT_READOUT_SEAMS: [&str; 3] = model_compiler::FLOAT_READOUT_SEAMS;
pub(crate) const PIXELS_SEAM: &str = model_compiler::EXPORT_SEAMS[6];
pub(crate) const DRAFTS_SEAM: &str = model_compiler::EXPORT_SEAMS[3];

#[derive(Debug, Clone)]
pub struct Export {
    pub value: ValueId,
    pub layer: u32,
    pub classes: model_ir::ClassSet,
}

#[derive(Debug, Clone)]
pub(crate) struct Exports {
    pub(crate) out_classes: model_ir::ClassSet,
    pub(crate) out: Option<ValueId>,
    pub(crate) mtp: Option<Export>,
    pub(crate) drafts: Option<Export>,
    pub(crate) drafts_depth: u32,
    pub(crate) scores: Vec<Export>,
    pub(crate) capturing: model_ir::ClassSet,
    pub(crate) velocity: Option<Export>,
    pub(crate) hidden: Vec<Export>,
    pub(crate) pixels: Vec<(Export, ValueId)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadoutSeam {
    pub(crate) seam: engine::fire::ReadoutSeam,
    pub(crate) value: ValueId,
}

impl Exports {
    #[must_use]
    pub(crate) fn pixels_for(&self, class: Option<usize>) -> Option<(ValueId, ValueId)> {
        class
            .and_then(|class| {
                self.pixels
                    .iter()
                    .find(|(export, _)| export.classes.contains(class))
            })
            .or_else(|| (self.pixels.len() == 1).then(|| &self.pixels[0]))
            .map(|(export, grid)| (export.value, *grid))
    }

    pub(crate) fn pixels_widths<'a>(&'a self, trace: &'a Trace) -> impl Iterator<Item = u32> + 'a {
        self.pixels.iter().filter_map(move |(export, _)| {
            crate::store::kv::width_of(trace, export.value)
                .ok()
                .and_then(|width| u32::try_from(width).ok())
        })
    }

    #[must_use]
    pub(crate) fn readout_for(&self, class: usize) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out
            && self.out_classes.contains(class)
        {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = self.velocity_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        if let Some(hidden) = self.hidden_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Hidden,
                value: hidden.value,
            });
        }
        self.readout()
    }

    #[must_use]
    pub(crate) fn velocity_for(&self, class: usize) -> Option<&Export> {
        self.velocity
            .as_ref()
            .filter(|export| export.classes.contains(class))
    }

    #[must_use]
    pub(crate) fn hidden_for(&self, class: usize) -> Option<&Export> {
        self.hidden
            .iter()
            .rev()
            .find(|export| export.classes.contains(class))
    }

    #[must_use]
    pub(crate) fn readout(&self) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = &self.velocity {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        self.hidden.last().map(|hidden| ReadoutSeam {
            seam: Seam::Hidden,
            value: hidden.value,
        })
    }
}

impl Exports {
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let float_readout = trace.seams.iter().any(|seam| {
            FLOAT_READOUT_SEAMS.contains(&seam.seam.as_str()) && !seam.values.is_empty()
        });
        if out.is_none() && !float_readout {
            return Err(Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam and no float readout ({}), so a fire would compute \
                     nothing a reader can take",
                    FLOAT_READOUT_SEAMS.join(", ")
                ),
            });
        }
        let named = |name: &str| -> Vec<Export> {
            trace
                .seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| {
                    let layer = seam.layer.unwrap_or(0);
                    seam.values.iter().map(move |value| (layer, *value))
                })
                .map(|(layer, value)| Export {
                    value,
                    layer,
                    classes: writer_classes(trace, compiled, value),
                })
                .collect()
        };
        let scores = named(SCORES_SEAM);
        let drafts = named(DRAFTS_SEAM).into_iter().next();
        let drafts_depth = match &drafts {
            Some(export) => {
                let width =
                    model_exec::store::kv::width_of(trace, export.value).map_err(|why| {
                        Fault::Unbound {
                            what: format!("the `{DRAFTS_SEAM}` export's width: {why}"),
                        }
                    })?;
                match u32::try_from(width) {
                    Ok(depth) if depth > 0 => depth,
                    _ => {
                        return Err(Fault::Unbound {
                            what: format!("a `{DRAFTS_SEAM}` export {width} wide drafts nothing"),
                        });
                    }
                }
            }
            None => 0,
        };
        let mut capturing = model_ir::ClassSet::default();
        for export in &scores {
            for class in export.classes.iter() {
                capturing.insert(class);
            }
        }
        let pixels = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| match seam.values.as_slice() {
                [plane, grid, ..] => Some((
                    Export {
                        value: *plane,
                        layer: seam.layer.unwrap_or(0),
                        classes: writer_classes(trace, compiled, *plane),
                    },
                    *grid,
                )),
                _ => None,
            })
            .collect();
        Ok(Exports {
            out_classes: out.map_or_else(model_ir::ClassSet::default, |out| {
                writer_classes(trace, compiled, out)
            }),
            out,
            mtp: named(MTP_SEAM).into_iter().next(),
            drafts,
            drafts_depth,
            scores,
            capturing,
            velocity: named(FLOAT_READOUT_SEAMS[0]).into_iter().next(),
            hidden: named(FLOAT_READOUT_SEAMS[1]),
            pixels,
        })
    }
}

fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    if let Some(model_ir::Def::Merge(arms)) =
        trace.values.get(value.0 as usize).map(|decl| &decl.def)
    {
        let mut classes = model_ir::ClassSet::default();
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
    let mut classes = model_ir::ClassSet::default();
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

#[must_use]
pub(crate) fn masked_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Attention(
                model_ir::Attention::Masked { .. } | model_ir::Attention::MaskedLse { .. }
            )
        )
    })
}

#[must_use]
pub(crate) fn corrected_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Linear(model_ir::Linear::LoraCorrect { .. })
        )
    })
}

#[must_use]
pub(crate) fn media_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Layout(
                model_ir::Layout::ScatterRows { .. } | model_ir::Layout::ScatterLiveRows { .. }
            )
        )
    })
}

const READINGS: u8 = 8;

#[must_use]
pub(crate) fn landing_requests(
    classify: model_ir::ClassifyFn,
    classes: &model_ir::ClassTable,
) -> Vec<Vec<model_ir::Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    for reading in 0..READINGS {
        for stream in model_ir::Stream::ALL {
            for bits in 0..128u32 {
                let request =
                    model_ir::Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
                        .adapted(bits & 4 != 0)
                        .drafting(bits & 8 != 0)
                        .capturing_scores(bits & 16 != 0)
                        .with_media(bits & 32 != 0)
                        .denoising(bits & 64 != 0)
                        .on_stream(stream)
                        .in_reading(reading);
                let word = classify(&request) & classes.mask;
                if let Some(class) = classes.class_of(word) {
                    landing[class].push(request);
                }
            }
        }
    }
    for requests in &mut landing {
        requests.sort_by_key(request_flags);
    }
    landing
}

fn request_flags(request: &model_ir::Request) -> u32 {
    u32::from(request.query_len() != 1)
        + u32::from(request.has_custom_mask())
        + u32::from(request.has_adapter())
        + u32::from(request.drafts())
        + u32::from(request.captures_scores())
        + u32::from(request.has_media())
        + u32::from(request.denoise())
        + u32::from(request.stream() != model_ir::Stream::Text)
        + u32::from(request.reading() != 0)
}

#[must_use]
pub(crate) fn plain_of(landing: &[Vec<model_ir::Request>]) -> model_ir::ClassSet {
    model_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| {
                requests
                    .iter()
                    .any(|request| request_flags(request) == u32::from(request.query_len() != 1))
            })
            .map(|(class, _)| class),
    )
}

#[must_use]
pub(crate) fn decoding_of(landing: &[Vec<model_ir::Request>]) -> model_ir::ClassSet {
    model_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| {
                !requests.is_empty() && requests.iter().all(|request| request.query_len() == 1)
            })
            .map(|(class, _)| class),
    )
}

#[must_use]
pub(crate) fn regions_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace.nodes.get(node as usize).is_some_and(|node| {
                    let name = model_ir::Operands::name(&node.op);
                    crate::shifted(name) || crate::PLANNED.contains(&name)
                })
            })
        })
        .collect()
}

#[must_use]
pub(crate) fn regions_lane_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .all(|node| lane_shifting_node(trace, node))
        })
        .collect()
}

fn lane_shifting_node(trace: &Trace, node: u32) -> bool {
    let Some(node) = trace.nodes.get(node as usize) else {
        return false;
    };
    let name = Operands::name(&node.op);
    if crate::lane_shifted(name) || crate::PLANNED.contains(&name) {
        return true;
    }
    let mut operands: Vec<ValueId> = Vec::new();
    node.op.inputs(&mut operands);
    node.op.outputs(&mut operands);
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        if matches!(&decl.def, model_ir::Def::Cache(_)) {
            return false;
        }
        let model_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            return true;
        };
        !matches!(
            shape.first(),
            Some(
                model_ir::Dim::Lanes
                    | model_ir::Dim::LanesPlus(_)
                    | model_ir::Dim::Images
                    | model_ir::Dim::ImagesPlus(_)
            )
        )
    })
}

fn classes_running(
    trace: &Trace,
    compiled: &CompiledModel,
    wanted: impl Fn(&model_ir::Operation) -> bool,
) -> model_ir::ClassSet {
    let mut classes = model_ir::ClassSet::default();
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

#[derive(Debug, Clone, Default)]
pub(crate) struct Feeds {
    pub(crate) ports: Vec<(crate::inputs::PortSeat, model_ir::ClassSet)>,
    pub(crate) selections: Vec<model_ir::Selection>,
    pub(crate) merged: Vec<MergedPort>,
    pub(crate) unlanded: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MergedPort {
    pub(crate) merge: ValueId,
    pub(crate) seat: crate::inputs::PortSeat,
    pub(crate) select: model_ir::Selection,
}

impl Feeds {
    #[must_use]
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Feeds {
        use model_ir::{Def, GeomKind, RuntimeInput, Ty};
        let mut feeds = Feeds::default();
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            let dtype = match &decl.ty {
                Ty::Tensor { dtype, .. } => *dtype,
                Ty::Struct(_) => continue,
            };
            let seat = match *input {
                RuntimeInput::Latents { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Latents,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::LaneVector { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::LaneVector,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::Context { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Context,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::AxisPositions { port, axes } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::AxisPositions,
                    port,
                    width: u32::from(axes),
                    dtype,
                }),
                RuntimeInput::Voxels { port, channels } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Voxels,
                    port,
                    width: channels,
                    dtype,
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
            if let Some(seat) = seat {
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
        }
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Merge(arms) = &decl.def else {
                continue;
            };
            for (arm, guard) in arms {
                let Def::Input(input) = &trace.values[arm.0 as usize].def else {
                    continue;
                };
                let seat = feeds
                    .ports
                    .iter()
                    .find(|(seat, _)| {
                        let (kind, port) = match *input {
                            RuntimeInput::Latents { port, .. } => {
                                (engine::fire::PortKind::Latents, port)
                            }
                            RuntimeInput::LaneVector { port, .. } => {
                                (engine::fire::PortKind::LaneVector, port)
                            }
                            RuntimeInput::Context { port, .. } => {
                                (engine::fire::PortKind::Context, port)
                            }
                            RuntimeInput::AxisPositions { port, .. } => {
                                (engine::fire::PortKind::AxisPositions, port)
                            }
                            _ => return false,
                        };
                        seat.kind == kind && seat.port == port
                    })
                    .map(|(seat, _)| *seat);
                let (Some(seat), Some(select)) = (seat, model_ir::Selection::of(guard)) else {
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
                        let word = compiled.classes.classes[class].word();
                        if select.holds(word) {
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

    #[must_use]
    pub(crate) fn seats(&self) -> Vec<crate::inputs::PortSeat> {
        self.ports
            .iter()
            .map(|(seat, _)| *seat)
            .filter(|seat| seat.kind != engine::fire::PortKind::Voxels)
            .collect()
    }
}

fn reader_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    let mut inputs: Vec<ValueId> = Vec::new();
    let mut readers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        if inputs.contains(&value) {
            readers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = model_ir::ClassSet::default();
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

#[must_use]
pub(crate) fn schedule_streams(trace: &Trace, compiled: &CompiledModel) -> Vec<Option<u32>> {
    let mut streams: Vec<Option<Vec<u32>>> = vec![None; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for region in compiled.template() {
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, model_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let readers = streams[at].get_or_insert_with(Vec::new);
                if !readers.contains(&region.stream) {
                    readers.push(region.stream);
                }
            }
        }
    }
    streams
        .into_iter()
        .map(|readers| match readers.as_deref() {
            Some([stream]) => Some(*stream),
            _ => None,
        })
        .collect()
}

#[must_use]
pub(crate) fn regions_launching_schedules(
    trace: &Trace,
    compiled: &CompiledModel,
) -> Vec<Option<u32>> {
    let mut out: Vec<Option<u32>> = vec![None; trace.values.len()];
    let mut claimed: Vec<bool> = vec![false; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        let here = u32::try_from(at).unwrap_or(u32::MAX);
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, model_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let Some(slot) = out.get_mut(at) else {
                    continue;
                };
                if !claimed[at] {
                    claimed[at] = true;
                    *slot = Some(here);
                }
            }
        }
    }
    out
}
