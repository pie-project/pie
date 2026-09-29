//! The diffusion and video families: the voxel axis (clip grids, the voxel
//! port, the clip slots a frame cache reads), the float ports a lane feeds
//! from its channels (latents, lane vectors, context rows, axis positions,
//! and those merged into a stream), the attention-group packings, the
//! self-conditioning taps, and the export seams besides logits (velocity,
//! hidden, pixels).
//!
//! engine-cuda's `voxels.rs`, `exports.rs` (`Feeds`, `Exports`) and the port
//! half of `serve/prepare.rs`, staged as this shell stages everything else:
//! host tables handed to the traced program as inputs.

use engine::fire::{PortKind, ReadoutSeam};
use kernels_xla::{Emit, Tensor};
use model_compiler::{CompiledModel, VoxelLadder};
use model_exec::fire::{Composition, LaneFacts, LaneRow};
use model_ir::{ClassSet, Def, Dtype, GeomKind, Operands, RuntimeInput, Selection, Trace, Ty, ValueId};

use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{DitBindings, PackingBindings, PortBinding};
use crate::trace::Handles;

/// The seams a denoiser or a decoder exports instead of logits.
const VELOCITY_SEAM: &str = model_compiler::FLOAT_READOUT_SEAMS[0];
const HIDDEN_SEAM: &str = model_compiler::FLOAT_READOUT_SEAMS[1];
const PIXELS_SEAM: &str = model_compiler::EXPORT_SEAMS[6];

/// A lane's clips on the voxel axis: one `[t, h, w]` box per clip and, when
/// the host feeds the voxel port, one port row per voxel in the port's
/// element (engine-cuda `serve::Clips`).
#[derive(Debug, Clone, Copy)]
pub struct Clips<'a> {
    pub lane: u32,
    pub clips: &'a [[u32; 3]],
    pub payload: &'a [u8],
}

impl Clips<'_> {
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.clips
            .iter()
            .map(|[t, h, w]| u64::from(*t) * u64::from(*h) * u64::from(*w))
            .sum()
    }
}

/// One float port's cell for one lane, as f32 (the engine read it off the
/// channel the lane's `PortFeed` names).
#[derive(Debug, Clone, Copy)]
pub struct PortCell<'a> {
    pub kind: PortKind,
    pub port: u8,
    pub values: &'a [f32],
}

/// A lane's self-conditioning taps: `taps` row ids and weights per row.
#[derive(Debug, Clone, Copy)]
pub struct SelfCond<'a> {
    pub taps: u32,
    pub rows: &'a [i32],
    pub weights: &'a [f32],
}

/// What a lane decoded: its pixel rows (f32) and the output box of each of
/// its clips.
pub type Pixels = (Vec<f32>, Vec<[u32; 3]>);

/// The voxel ladder a plan with a voxel axis loads at (engine-cuda
/// `api::voxel_ladder`).
#[must_use]
pub fn voxel_ladder(
    trace: &Trace,
    max_voxels: Option<u32>,
    max_clips: Option<u32>,
    max_lanes: u32,
) -> Option<VoxelLadder> {
    const DERIVED_VOXEL_CEILING: u32 = 65_536;
    let declares_voxels = trace.values.iter().any(|decl| {
        matches!(&decl.ty, Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(model_ir::RowAxis::Voxels))
    });
    if !declares_voxels {
        return None;
    }
    let max_voxels = max_voxels.unwrap_or(DERIVED_VOXEL_CEILING).max(1);
    Some(VoxelLadder::new(
        max_voxels,
        max_clips.unwrap_or(max_lanes).clamp(1, max_voxels),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PortSeat {
    kind: PortKind,
    port: u8,
    width: u32,
    dtype: Dtype,
}

impl PortSeat {
    fn per_lane(&self) -> bool {
        self.kind == PortKind::LaneVector
    }

    fn of(input: &RuntimeInput, dtype: Dtype) -> Option<PortSeat> {
        let (kind, port, width) = match *input {
            RuntimeInput::Latents { port, width } => (PortKind::Latents, port, width),
            RuntimeInput::LaneVector { port, width } => (PortKind::LaneVector, port, width),
            RuntimeInput::Context { port, width } => (PortKind::Context, port, width),
            RuntimeInput::AxisPositions { port, axes } => {
                (PortKind::AxisPositions, port, u32::from(axes))
            }
            RuntimeInput::Voxels { port, channels } => (PortKind::Voxels, port, channels),
            _ => return None,
        };
        Some(PortSeat {
            kind,
            port,
            width,
            dtype,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MergedPort {
    merge: ValueId,
    seat: PortSeat,
    select: Selection,
}

#[derive(Debug, Clone)]
struct VoxelSeat {
    widths: Vec<u32>,
    dtype: Dtype,
    token_patch: Option<[u32; 3]>,
}

#[derive(Debug, Clone)]
struct PixelsExport {
    plane: ValueId,
    grid: ValueId,
    classes: ClassSet,
    width: u32,
}

/// What a load reads off its plan for these families.
#[derive(Debug, Clone, Default)]
pub struct Dit {
    ports: Vec<(PortSeat, ClassSet)>,
    selections: Vec<Selection>,
    merged: Vec<MergedPort>,
    self_cond_taps: u32,
    voxel: Option<VoxelSeat>,
    velocity: Option<(ValueId, u32)>,
    hidden: Option<(ValueId, u32)>,
    pixels: Vec<PixelsExport>,
}

/// One lane's feeds, as the fire hands them to [`Dit::stage`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Feed<'a> {
    pub slot: u32,
    pub stream: u8,
    pub group: Option<u32>,
    pub ports: &'a [PortCell<'a>],
    pub self_cond: Option<SelfCond<'a>>,
    pub clips: Option<Clips<'a>>,
    pub attn_classes: Option<&'a engine::fire::AttnClasses>,
}

/// A merge whose arm is a port: the port's rows land in the merge's slot
/// before the walk (zeros for a lane that feeds none).
#[derive(Debug, Clone, Copy)]
pub struct Land {
    pub merge: ValueId,
    pub first: u32,
    pub rows: u32,
    pub port: Option<Tensor>,
}

/// Where a fire's pixels come back from.
#[derive(Debug, Clone)]
pub struct PixelsSeat {
    pub plane: ValueId,
    pub grid: ValueId,
    /// Per real lane: its first clip and clip count.
    pub lane_clips: Vec<(u32, u32)>,
}

/// One fire's staged inputs for these families.
#[derive(Debug, Default)]
pub struct Staged {
    pub bindings: DitBindings,
    pub clip_slots: Option<Tensor>,
    pub lands: Vec<Land>,
    pub pixels: Option<PixelsSeat>,
    /// What of this staging the program text depends on.
    pub key: String,
}

fn program(why: String) -> Fault {
    Fault::Program {
        at: "serve::dit",
        why,
    }
}

fn width_of(trace: &Trace, value: ValueId) -> u32 {
    match &trace.values[value.0 as usize].ty {
        Ty::Tensor { shape, .. } => shape
            .iter()
            .skip(1)
            .map(|dim| match dim {
                model_ir::Dim::Const(n) => *n,
                _ => 1,
            })
            .product::<u64>()
            .try_into()
            .unwrap_or(u32::MAX),
        Ty::Struct(_) => 0,
    }
}

fn classes_where(
    trace: &Trace,
    compiled: &CompiledModel,
    touches: impl Fn(&model_ir::Operation) -> bool,
) -> ClassSet {
    let nodes: Vec<u32> = trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| touches(&node.op))
        .map(|(at, _)| at as u32)
        .collect();
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        if region.nodes.clone().any(|node| nodes.contains(&node)) {
            for class in region.mask.iter() {
                classes.insert(class);
            }
        }
    }
    classes
}

fn reader_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> ClassSet {
    classes_where(trace, compiled, |op| {
        let mut inputs = Vec::new();
        op.inputs(&mut inputs);
        inputs.contains(&value)
    })
}

fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> ClassSet {
    if let Some(Def::Merge(arms)) = trace.values.get(value.0 as usize).map(|decl| &decl.def) {
        let mut classes = ClassSet::default();
        for (arm, _) in arms {
            for class in writer_classes(trace, compiled, *arm).iter() {
                classes.insert(class);
            }
        }
        return classes;
    }
    classes_where(trace, compiled, |op| {
        let mut outputs = Vec::new();
        op.outputs(&mut outputs);
        outputs.contains(&value)
    })
}

fn bf16_bits(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// `values` in `dtype`'s bytes (the float ports are bf16 or f32).
fn float_bytes(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::Bf16 => values
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    }
}

/// `values` as a port's (or the voxel port's) payload bytes.
pub fn port_bytes(values: &[f32], element: Dtype) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        Dtype::Bf16 | Dtype::F32 => Ok(float_bytes(values, element)),
        _ => Err("a float port whose element is neither `bf16` nor `f32`"),
    }
}

impl Dit {
    /// Reads the plan: the ports it declares and the classes that read
    /// them, the group selections its packed attention reads, the merges
    /// over ports, the voxel port, and the float seams.
    pub fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Dit> {
        let mut dit = Dit::default();
        let mut voxel: Option<VoxelSeat> = None;
        let mut grid = false;
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            let Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            match *input {
                RuntimeInput::RowPermutation { select }
                | RuntimeInput::Geometry {
                    kind:
                        GeomKind::GroupIndptr { select }
                        | GeomKind::LaneIndptr { select }
                        | GeomKind::ReferenceTag { select },
                    ..
                } => {
                    if !dit.selections.contains(&select) {
                        dit.selections.push(select);
                    }
                }
                RuntimeInput::Voxels { channels, .. } => {
                    let seat = voxel.get_or_insert_with(|| VoxelSeat {
                        widths: Vec::new(),
                        dtype: *dtype,
                        token_patch: None,
                    });
                    if !seat.widths.contains(&channels) {
                        seat.widths.push(channels);
                    }
                    seat.dtype = *dtype;
                }
                RuntimeInput::TokenGrid { p } => {
                    voxel
                        .get_or_insert_with(|| VoxelSeat {
                            widths: Vec::new(),
                            dtype: Dtype::Bf16,
                            token_patch: None,
                        })
                        .token_patch = Some(p);
                }
                RuntimeInput::Grid => grid = true,
                RuntimeInput::SelfCondRows => {
                    dit.self_cond_taps = width_of(trace, ValueId(at as u32));
                }
                _ => {}
            }
            if let Some(seat) = PortSeat::of(input, *dtype) {
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                match dit
                    .ports
                    .iter_mut()
                    .find(|(have, _)| have.kind == seat.kind && have.port == seat.port)
                {
                    Some((_, classes)) => {
                        for class in readers.iter() {
                            classes.insert(class);
                        }
                    }
                    None => dit.ports.push((seat, readers)),
                }
            }
        }
        if grid && voxel.is_none() {
            voxel = Some(VoxelSeat {
                widths: Vec::new(),
                dtype: Dtype::Bf16,
                token_patch: None,
            });
        }
        if let Some(seat) = voxel.as_mut() {
            seat.widths.sort_unstable();
        }
        dit.voxel = voxel;

        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Merge(arms) = &decl.def else {
                continue;
            };
            for (arm, guard) in arms {
                let arm_decl = &trace.values[arm.0 as usize];
                let (Def::Input(input), Ty::Tensor { dtype, .. }) = (&arm_decl.def, &arm_decl.ty)
                else {
                    continue;
                };
                let Some(seat) = PortSeat::of(input, *dtype) else {
                    continue;
                };
                if seat.kind == PortKind::Voxels {
                    continue;
                }
                let Some(select) = Selection::of(guard) else {
                    return Err(program(format!(
                        "value {at} merges {:?} port {} under a guard no selection states, so \
                         no lane's rows can be told to land it",
                        seat.kind, seat.port
                    )));
                };
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                if let Some((_, classes)) = dit
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
                dit.merged.push(MergedPort {
                    merge: ValueId(at as u32),
                    seat,
                    select,
                });
            }
        }

        let first = |name: &str| {
            trace
                .seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| seam.values.iter().copied())
                .collect::<Vec<_>>()
        };
        dit.velocity = first(VELOCITY_SEAM)
            .first()
            .map(|&value| (value, width_of(trace, value)));
        dit.hidden = first(HIDDEN_SEAM)
            .last()
            .map(|&value| (value, width_of(trace, value)));
        dit.pixels = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| match seam.values.as_slice() {
                [plane, grid, ..] => Some(PixelsExport {
                    plane: *plane,
                    grid: *grid,
                    classes: writer_classes(trace, compiled, *plane),
                    width: width_of(trace, *plane),
                }),
                _ => None,
            })
            .collect();
        Ok(dit)
    }

    /// The float readout a plan with no `out` seam reads back per token row:
    /// velocity, else the last hidden export.
    #[must_use]
    pub fn float_readout(&self) -> Option<(ReadoutSeam, ValueId)> {
        self.velocity
            .map(|(value, _)| (ReadoutSeam::Velocity, value))
            .or_else(|| self.hidden.map(|(value, _)| (ReadoutSeam::Hidden, value)))
    }

    #[must_use]
    pub fn velocity_width(&self) -> Option<u32> {
        self.velocity.map(|(_, width)| width)
    }

    /// Whether the plan exports pixels, and the widest pixel row.
    #[must_use]
    pub fn pixels_facts(&self) -> (bool, u32) {
        (
            !self.pixels.is_empty(),
            self.pixels.iter().map(|p| p.width).max().unwrap_or(0),
        )
    }

    /// The voxel port's element, when the plan declares one.
    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel
            .as_ref()
            .filter(|seat| !seat.widths.is_empty())
            .map(|seat| seat.dtype)
    }

    /// Whether the plan reads `kind` port `port`, and its element.
    #[must_use]
    pub fn port_element(&self, kind: PortKind, port: u8) -> Option<Dtype> {
        self.ports
            .iter()
            .find(|(seat, _)| seat.kind == kind && seat.port == port)
            .map(|(seat, _)| seat.dtype)
    }

    /// Whether a fire of this plan stages anything here at all.
    /// The float ports (not the voxel port) `class` reads: kind, port,
    /// width, and whether a cell is one row per lane (`crate::serve::dry`).
    pub(crate) fn dry_ports(&self, class: usize) -> Vec<(PortKind, u8, u32, bool)> {
        self.ports
            .iter()
            .filter(|(seat, readers)| seat.kind != PortKind::Voxels && readers.contains(class))
            .map(|(seat, _)| (seat.kind, seat.port, seat.width, seat.per_lane()))
            .collect()
    }

    /// The voxel port's widths, element and token patch, when the plan has
    /// a voxel axis.
    pub(crate) fn dry_voxel(&self) -> Option<(Vec<u32>, Dtype, Option<[u32; 3]>)> {
        self.voxel
            .as_ref()
            .map(|seat| (seat.widths.clone(), seat.dtype, seat.token_patch))
    }

    /// The width of the voxel port `class` reads, if it reads one.
    pub(crate) fn dry_voxel_width(&self, class: usize) -> Option<u32> {
        self.ports
            .iter()
            .find(|(seat, readers)| seat.kind == PortKind::Voxels && readers.contains(class))
            .map(|(seat, _)| seat.width)
    }

    pub(crate) fn dry_self_cond_taps(&self) -> u32 {
        self.self_cond_taps
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ports.is_empty()
            && self.selections.is_empty()
            && self.merged.is_empty()
            && self.self_cond_taps == 0
            && self.voxel.is_none()
    }

    /// Stages one fire: `feeds` holds one entry per submitted lane (padding
    /// lanes past `feeds.len()` feed nothing), `rows` is the fire's token
    /// rows as composed.
    #[allow(clippy::too_many_lines)]
    pub fn stage(
        &self,
        composition: &Composition,
        feeds: &[Feed<'_>],
        inputs: &mut Inputs,
        handles: &Handles,
    ) -> Result<Staged> {
        let mut staged = Staged::default();
        let class_table = self.class_table_of(composition.lanes(), feeds)?;
        if self.is_empty() && composition.voxel_rows() == 0 {
            return Ok(staged);
        }
        let rows = composition.rows();
        let lanes = composition.lanes();
        let lane_count = lanes.len() as u32;
        let feed_of = |row: &LaneRow| feeds.get(row.source as usize).copied();
        let mut key = String::new();

        // ---- the voxel axis
        if composition.voxel_rows() > 0 {
            let (tables, pixels) = self.stage_voxels(composition, feeds)?;
            staged.bindings.grid = Some(inputs.i32s(handles, &tables.grid, 4));
            if !tables.token_grid.is_empty() {
                staged.bindings.token_grid = Some(inputs.i32s(handles, &tables.token_grid, 4));
            }
            if tables.channels > 0 {
                let dtype = self.voxel.as_ref().map_or(Dtype::Bf16, |seat| seat.dtype);
                staged.bindings.voxels = Some(inputs.raw(
                    handles,
                    dtype,
                    composition.voxel_rows(),
                    tables.channels,
                    tables.payload,
                ));
            }
            staged.clip_slots = Some(inputs.i32s(handles, &tables.slots, 1));
            staged.pixels = pixels;
            key.push_str(&format!(
                "vox{}x{}c{}t{};",
                composition.voxel_rows(),
                composition.clips(),
                tables.channels,
                u8::from(!tables.token_grid.is_empty())
            ));
        } else {
            for (at, feed) in feeds.iter().enumerate() {
                if feed.clips.is_some() {
                    return Err(program(format!(
                        "lane {at} carries clips and the fire composed no voxel rows"
                    )));
                }
            }
        }

        // ---- float ports
        let mut planes: Vec<(PortSeat, Vec<f32>)> = self
            .ports
            .iter()
            .filter(|(seat, _)| seat.kind != PortKind::Voxels)
            .map(|(seat, _)| {
                let extent = if seat.per_lane() { lane_count } else { rows };
                (*seat, vec![0f32; extent as usize * seat.width as usize])
            })
            .collect();
        for (fire_lane, row) in lanes.iter().enumerate() {
            let Some(feed) = feed_of(row) else {
                continue;
            };
            for cell in feed.ports {
                if cell.kind == PortKind::Voxels {
                    continue;
                }
                if !self
                    .ports
                    .iter()
                    .any(|(seat, _)| seat.kind == cell.kind && seat.port == cell.port)
                {
                    return Err(program(format!(
                        "lane {} feeds {:?} port {} and this plan declares no such port",
                        row.source, cell.kind, cell.port
                    )));
                }
            }
            for (seat, plane) in &mut planes {
                let reads = self
                    .ports
                    .iter()
                    .any(|(have, readers)| {
                        have.kind == seat.kind
                            && have.port == seat.port
                            && readers.contains(row.class as usize)
                    });
                if !reads {
                    continue;
                }
                let Some(cell) = feed
                    .ports
                    .iter()
                    .find(|cell| cell.kind == seat.kind && cell.port == seat.port)
                else {
                    return Err(program(format!(
                        "lane {} runs a class that reads {:?} port {} and feeds it no \
                         channel; a declared port is fed from a channel cell at every submit \
                         (`Lane::ports`)",
                        row.source, seat.kind, seat.port
                    )));
                };
                let (first, count) = if seat.per_lane() {
                    (fire_lane as u32, 1)
                } else {
                    (row.row_offset, row.rows)
                };
                let want = count as usize * seat.width as usize;
                if cell.values.len() != want {
                    return Err(program(format!(
                        "lane {} feeds {:?} port {} a cell of {} value(s), and the port wants \
                         {count} row(s) x {} = {want} (the lane's rows by the port's width)",
                        row.source,
                        seat.kind,
                        seat.port,
                        cell.values.len(),
                        seat.width
                    )));
                }
                let at = first as usize * seat.width as usize;
                plane[at..at + want].copy_from_slice(cell.values);
            }
        }
        for (seat, plane) in planes {
            let extent = if seat.per_lane() { lane_count } else { rows };
            let tensor = inputs.raw(
                handles,
                seat.dtype,
                extent,
                seat.width,
                float_bytes(&plane, seat.dtype),
            );
            staged.bindings.ports.push(PortBinding {
                kind: seat.kind,
                port: seat.port,
                tensor,
            });
            key.push_str(&format!("p{:?}{}x{};", seat.kind, seat.port, seat.width));
        }

        // ---- merges over ports
        for (fire_lane, row) in lanes.iter().enumerate() {
            for merged in &self.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = feed_of(row).is_some_and(|feed| {
                    feed.ports
                        .iter()
                        .any(|cell| cell.kind == merged.seat.kind && cell.port == merged.seat.port)
                });
                let (first, count) = if merged.seat.per_lane() {
                    (fire_lane as u32, 1)
                } else {
                    (row.row_offset, row.rows)
                };
                let port = fed
                    .then(|| {
                        staged
                            .bindings
                            .ports
                            .iter()
                            .find(|bound| {
                                bound.kind == merged.seat.kind && bound.port == merged.seat.port
                            })
                            .map(|bound| bound.tensor)
                    })
                    .flatten();
                staged.lands.push(Land {
                    merge: merged.merge,
                    first,
                    rows: count,
                    port,
                });
                key.push_str(&format!(
                    "m{}@{first}+{count}{};",
                    merged.merge.0,
                    if port.is_some() { "f" } else { "z" }
                ));
            }
        }

        // ---- self-conditioning taps
        let taps = self.self_cond_taps as usize;
        if taps > 0 {
            let mut ids = vec![0i32; rows as usize * taps];
            let mut weights = vec![0f32; rows as usize * taps];
            for row in lanes {
                let Some(sc) = feed_of(row).and_then(|feed| feed.self_cond) else {
                    continue;
                };
                let cells = row.rows as usize * taps;
                if sc.taps as usize != taps || sc.rows.len() != cells || sc.weights.len() != cells
                {
                    return Err(program(format!(
                        "lane {} states self-conditioning taps of width {} over {} ids, and \
                         this plan reads {taps} taps over the lane's {} rows",
                        row.source,
                        sc.taps,
                        sc.rows.len(),
                        row.rows
                    )));
                }
                let at = row.row_offset as usize * taps;
                ids[at..at + cells].copy_from_slice(sc.rows);
                weights[at..at + cells].copy_from_slice(sc.weights);
            }
            staged.bindings.self_cond_rows = Some(inputs.i32s(handles, &ids, taps as u32));
            staged.bindings.self_cond_weights = Some(inputs.f32s(handles, &weights, taps as u32));
            key.push_str(&format!("sc{taps};"));
        } else if let Some((at, _)) = feeds
            .iter()
            .enumerate()
            .find(|(_, feed)| feed.self_cond.is_some())
        {
            return Err(program(format!(
                "lane {at} states self-conditioning taps and this plan reads none"
            )));
        }

        // ---- attention-group packings
        if !self.selections.is_empty() {
            let facts: Vec<LaneFacts> = (0..lanes
                .iter()
                .map(|row| row.source as usize + 1)
                .max()
                .unwrap_or(0))
                .map(|source| {
                    feeds.get(source).map_or(
                        LaneFacts {
                            stream: 0,
                            group: None,
                        },
                        |feed| LaneFacts {
                            stream: feed.stream,
                            group: feed.group,
                        },
                    )
                })
                .collect();
            let mut groups = model_exec::fire::group_of_lane(lanes, &facts);
            groups.resize(lane_count as usize, -1);
            staged.bindings.group_of_lane = Some(inputs.i32s(handles, &groups, 1));
            let lane_classes: Vec<Option<&[i32]>> = feeds
                .iter()
                .map(|feed| feed.attn_classes.map(|stated| stated.classes.as_slice()))
                .collect();
            if let Some(stated) = class_table {
                let count = stated.count;
                staged.bindings.class_table = Some((
                    inputs.raw(handles, Dtype::U8, count, count, stated.table.clone()),
                    count,
                ));
                key.push_str(&format!("ct{count};"));
            }
            for &select in &self.selections {
                let packed = model_exec::fire::pack_with_classes(
                    select,
                    lanes,
                    &facts,
                    &groups,
                    rows,
                    &lane_classes,
                )
                .map_err(model_exec::Error::Fire)?;
                let lane_table = |table: &[i32]| -> Vec<i32> {
                    let mut out = table.to_vec();
                    let last = table.last().copied().unwrap_or(0);
                    out.resize(lane_count as usize + 1, last);
                    out
                };
                let row_table = |table: &[i32]| -> Vec<i32> {
                    let mut out = table.to_vec();
                    out.resize(rows as usize, -1);
                    out
                };
                let bindings = PackingBindings {
                    group_indptr: inputs.i32s(handles, &lane_table(&packed.group_indptr), 1),
                    lane_indptr: inputs.i32s(handles, &lane_table(&packed.lane_indptr), 1),
                    reference_tag: inputs.i32s(handles, &row_table(&packed.reference_tag), 1),
                    permutation: inputs.i32s(handles, &row_table(&packed.permutation), 1),
                    attn_class: inputs.i32s(handles, &row_table(&packed.attn_class), 1),
                };
                staged.bindings.packings.push((select, bindings));
            }
            key.push_str(&format!("pk{};", self.selections.len()));
        }

        staged.key = key;
        Ok(staged)
    }

    /// The one attention class table a fire's lanes state, checked
    /// (engine-cuda `serve/prepare.rs::class_table_of`).
    fn class_table_of<'a>(
        &self,
        rows: &[LaneRow],
        feeds: &[Feed<'a>],
    ) -> Result<Option<&'a engine::fire::AttnClasses>> {
        let classes = |why: String| Fault::Program {
            at: "serve::classes",
            why,
        };
        let mut table: Option<&'a engine::fire::AttnClasses> = None;
        for (source, feed) in feeds.iter().enumerate() {
            let Some(stated) = feed.attn_classes else {
                continue;
            };
            if self.selections.is_empty() {
                return Err(classes(
                    "a lane states attention classes and this model packs no attention \
                     group; the mask applies to group-packed `attention.ragged` only"
                        .to_string(),
                ));
            }
            let owned = rows
                .iter()
                .find(|r| r.source as usize == source)
                .map_or(0, |r| r.rows);
            if stated.classes.len() as u32 != owned {
                return Err(classes(format!(
                    "a lane states {} attention classes and carries {owned} rows; one \
                     class per row",
                    stated.classes.len()
                )));
            }
            if stated.count == 0 || stated.count > engine::fire::ATTN_CLASSES_MAX {
                return Err(classes(format!(
                    "a lane states {} attention classes; the table holds 1..={}",
                    stated.count,
                    engine::fire::ATTN_CLASSES_MAX
                )));
            }
            if stated.table.len() as u64 != u64::from(stated.count) * u64::from(stated.count) {
                return Err(classes(format!(
                    "a lane's class table is {} bytes for {} classes; it is count x count",
                    stated.table.len(),
                    stated.count
                )));
            }
            if let Some(bad) = stated
                .classes
                .iter()
                .find(|c| **c >= 0 && **c as u32 >= stated.count)
            {
                return Err(classes(format!(
                    "a lane's row names attention class {bad} and the table holds {}",
                    stated.count
                )));
            }
            match table {
                None => table = Some(stated),
                Some(have) if have.count == stated.count && have.table == stated.table => {}
                Some(_) => {
                    return Err(classes(
                        "two lanes of one fire state different attention class tables; a \
                         fire reads one"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(table)
    }

    /// The clip tables (engine-cuda `voxels::Tables::of`).
    fn stage_voxels(
        &self,
        composition: &Composition,
        feeds: &[Feed<'_>],
    ) -> Result<(VoxelTables, Option<PixelsSeat>)> {
        let Some(seat) = self.voxel.as_ref() else {
            return Err(program(
                "a lane carries clips and this plan declares no voxel axis".to_string(),
            ));
        };
        if composition.voxel_classes().present_in_order().count() > 1 {
            return Err(program(
                "the clips of one fire fall in two classes, and a spatial launch runs over the \
                 whole voxel rectangle (one voxel class per fire)"
                    .to_string(),
            ));
        }
        let lanes = composition.lanes();
        let clips_total = composition.clips() as usize;
        let voxels_total = composition.voxel_rows() as usize;
        let elem = model_compiler::arena::elem_bytes(seat.dtype).unwrap_or(0) as usize;
        let mut channels = 0u32;
        for row in lanes {
            let Some(shot) = feeds.get(row.source as usize).and_then(|feed| feed.clips) else {
                continue;
            };
            if shot.payload.is_empty() {
                continue;
            }
            let voxels = shot.voxels() as usize;
            let per_row = shot.payload.len().checked_div(voxels).unwrap_or(0);
            let width = u32::try_from(per_row / elem.max(1)).unwrap_or(u32::MAX);
            if elem == 0
                || voxels == 0
                || per_row * voxels != shot.payload.len()
                || per_row % elem != 0
                || !seat.widths.contains(&width)
            {
                return Err(program(format!(
                    "lane {}'s voxel payload is not one port row per voxel of its clips at a \
                     width the plan reads ({:?})",
                    row.source, seat.widths
                )));
            }
            if channels != 0 && channels != width {
                return Err(program(format!(
                    "lane {} feeds a voxel port of another width than an earlier lane of the \
                     fire (one voxel class per fire)",
                    row.source
                )));
            }
            channels = width;
        }
        if channels == 0 && !seat.widths.is_empty() {
            return Err(program(
                "the fire's clips carry no voxel payload and feed no voxel port, and this plan \
                 reads its voxels off the port"
                    .to_string(),
            ));
        }
        let row_bytes = channels as usize * elem;
        let mut grid = vec![0i32; clips_total * 4];
        let mut token_grid = if seat.token_patch.is_some() {
            vec![0i32; clips_total * 4]
        } else {
            Vec::new()
        };
        let mut payload = vec![0u8; voxels_total * row_bytes];
        let mut slots = vec![0i32; clips_total];
        for row in lanes {
            let Some(feed) = feeds.get(row.source as usize) else {
                continue;
            };
            let Some(shot) = feed.clips else {
                continue;
            };
            if shot.clips.len() != row.clips as usize || shot.voxels() != u64::from(row.voxels) {
                return Err(program(format!(
                    "lane {}: the clips composed are not the clips submitted",
                    row.source
                )));
            }
            if !shot.payload.is_empty() {
                let at = row.voxel_offset as usize * row_bytes;
                payload[at..at + shot.payload.len()].copy_from_slice(shot.payload);
            }
            let mut voxel_row = i64::from(row.voxel_offset);
            let mut token_row = i64::from(row.row_offset);
            for (c, [t, h, w]) in shot.clips.iter().enumerate() {
                let clip = row.clip_offset as usize + c;
                slots[clip] = feed.slot as i32;
                grid[clip * 4..clip * 4 + 4]
                    .copy_from_slice(&[*t as i32, *h as i32, *w as i32, voxel_row as i32]);
                voxel_row += i64::from(*t) * i64::from(*h) * i64::from(*w);
                if let Some(p) = seat.token_patch {
                    if p.contains(&0) || t % p[0] != 0 || h % p[1] != 0 || w % p[2] != 0 {
                        return Err(program(format!(
                            "lane {}: a clip's box {:?} does not divide by the plan's token \
                             patch {p:?}",
                            row.source,
                            [t, h, w]
                        )));
                    }
                    let [tt, th, tw] = [t / p[0], h / p[1], w / p[2]];
                    token_grid[clip * 4..clip * 4 + 4].copy_from_slice(&[
                        tt as i32,
                        th as i32,
                        tw as i32,
                        token_row as i32,
                    ]);
                    token_row += i64::from(tt) * i64::from(th) * i64::from(tw);
                }
            }
            if seat.token_patch.is_some()
                && token_row != i64::from(row.row_offset) + i64::from(row.rows)
            {
                return Err(program(format!(
                    "lane {}: its {} token rows are not its clips' token count",
                    row.source, row.rows
                )));
            }
        }

        let voxel_class = composition
            .voxel_classes()
            .present_in_order()
            .next()
            .map(|class| class as usize);
        let export = voxel_class
            .and_then(|class| self.pixels.iter().find(|p| p.classes.contains(class)))
            .or_else(|| (self.pixels.len() == 1).then(|| &self.pixels[0]));
        let pixels = export.map(|export| {
            let mut lane_clips = vec![(0u32, 0u32); feeds.len()];
            for row in lanes {
                if let Some(slot) = lane_clips.get_mut(row.source as usize) {
                    *slot = (row.clip_offset, row.clips);
                }
            }
            PixelsSeat {
                plane: export.plane,
                grid: export.grid,
                lane_clips,
            }
        });
        Ok((
            VoxelTables {
                grid,
                token_grid,
                payload,
                channels,
                slots,
            },
            pixels,
        ))
    }
}

struct VoxelTables {
    grid: Vec<i32>,
    token_grid: Vec<i32>,
    payload: Vec<u8>,
    channels: u32,
    slots: Vec<i32>,
}

/// Lands each merge-over-port's rows in its slot before the walk: the
/// port's rows for a lane that fed it, zeros for one that did not.
pub fn land(
    ctx: &dyn Emit,
    handles: &Handles,
    slots: &crate::run::SlotTable,
    lands: &[Land],
) -> Result<()> {
    for land in lands {
        let whole = slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, a merge over a port, which the carve gave no rectangle",
                land.merge.0
            ),
        })?;
        let at = handles.cut(whole, land.first, land.rows);
        let from = land.port.map(|port| handles.cut(port, land.first, land.rows));
        ctx.emit(&mut |cx| {
            let v = match from {
                Some(port) => cx.read(port)?,
                None => {
                    let elem = kernels_xla::elem_of("serve.land", at.dtype)?;
                    cx.const_f(elem, 0.0, &[i64::from(at.rows), i64::from(at.width)])
                }
            };
            cx.write(at, v)
        })
        .map_err(|why| program(format!("landing merge {} over a port: {why:?}", land.merge.0)))?;
    }
    Ok(())
}

/// Each lane's pixels, out of the downloaded plane and grid (engine-cuda
/// `serve/settle.rs`).
pub fn pixels_of(
    seat: &PixelsSeat,
    plane: &[u8],
    plane_width: usize,
    grid: &[u8],
    real: usize,
) -> Vec<Pixels> {
    let grid: Vec<i32> = grid
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let row_bytes = plane_width * 4;
    let mut out: Vec<Pixels> = vec![(Vec::new(), Vec::new()); real];
    for (lane, &(first, count)) in seat.lane_clips.iter().enumerate().take(real) {
        let mut values = Vec::new();
        let mut boxes = Vec::with_capacity(count as usize);
        for clip in first..first + count {
            let at = clip as usize * 4;
            let Some(cell) = grid.get(at..at + 4) else {
                continue;
            };
            let [t, h, w, off] = [cell[0], cell[1], cell[2], cell[3]];
            let voxels = t.max(0) as usize * h.max(0) as usize * w.max(0) as usize;
            boxes.push([t as u32, h as u32, w as u32]);
            let from = (off.max(0) as usize * row_bytes).min(plane.len());
            let to = (from + voxels * row_bytes).min(plane.len());
            values.extend(
                plane[from..to]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        out[lane] = (values, boxes);
    }
    out
}
