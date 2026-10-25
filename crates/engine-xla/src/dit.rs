//! The diffusion and video families: the voxel axis (clip grids, the voxel
//! input, the clip slots a frame cache reads), the guest inputs a lane feeds
//! from its channels (and those merged into a stream), the attention-group
//! packings, and the export seams besides logits (velocity, hidden, pixels).
//!
//! engine-cuda's `voxels.rs`, `exports.rs` (`Exports`) and the input half of
//! `serve/prepare.rs`, staged as this shell stages everything else:
//! host tables handed to the traced program as inputs.

use crate::emit::Emit;
use crate::handle::Tensor;
use engine::fire::ReadoutSeam;
use poem_compiler::{CompiledModel, VoxelLadder};
use poem_exec::feeds::{Feeds, Seat};
use poem_exec::fire::{Composition, LaneFacts, LaneRow};
use poem_ir::{ClassSet, Def, Dtype, Operands, RuntimeInput, Trace, Ty, ValueId};

use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{DitBindings, InputBinding, PackingBindings};
use crate::trace::Handles;

/// The seams a denoiser or a decoder exports instead of logits.
const VELOCITY_SEAM: &str = poem_compiler::FLOAT_READOUT_SEAMS[0];
const HIDDEN_SEAM: &str = poem_compiler::FLOAT_READOUT_SEAMS[1];
const PIXELS_SEAM: &str = poem_compiler::EXPORT_SEAMS[6];

/// A lane's clips on the voxel axis: one `[t, h, w]` box per clip and, when
/// the host feeds the voxel input, one input row per voxel in the input's
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

/// One guest input's cell for one lane, in the input's element (the engine
/// read it off the channel the lane's `InputFeed` names, [`cell_bytes`]).
#[derive(Debug, Clone, Copy)]
pub struct InputCell<'a> {
    pub id: u32,
    pub bytes: &'a [u8],
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
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Voxels))
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
    feeds: Feeds,
    voxel: Option<VoxelSeat>,
    velocity: Option<(ValueId, u32)>,
    hidden: Option<(ValueId, u32)>,
    pixels: Vec<PixelsExport>,
}

/// One lane's feeds, as the fire hands them to [`Dit::stage`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Feed<'a> {
    pub slot: u32,
    pub group: Option<u32>,
    pub inputs: &'a [InputCell<'a>],
    pub clips: Option<Clips<'a>>,
    pub attn_classes: Option<&'a engine::fire::AttnClasses>,
}

/// A merge whose arm is a guest input: the input's rows land in the merge's
/// slot before the walk (zeros for a lane that feeds none).
#[derive(Debug, Clone, Copy)]
pub struct Land {
    pub merge: ValueId,
    pub first: u32,
    pub rows: u32,
    pub input: Option<Tensor>,
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
                poem_ir::Dim::Const(n) => *n,
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
    touches: impl Fn(&poem_ir::Operation) -> bool,
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

/// `values` in `dtype`'s bytes (the float inputs are bf16 or f32).
fn float_bytes(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::Bf16 => values
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    }
}

/// `values` as a float input's (or the voxel input's) payload bytes.
pub fn payload_bytes(values: &[f32], element: Dtype) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        Dtype::Bf16 | Dtype::F32 => Ok(float_bytes(values, element)),
        _ => Err("a float input whose element is neither `bf16` nor `f32`"),
    }
}

/// A channel cell as a guest input's bytes in the input's element: an f32
/// cell feeds a bf16 or f32 input, an i32 cell an i32 input.
pub fn cell_bytes(
    cell: &eta_exec::Value,
    element: Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match (cell, element) {
        (eta_exec::Value::F32(values), _) => payload_bytes(values, element),
        (eta_exec::Value::I32(values), Dtype::I32) => {
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        _ => Err("a guest input fed a cell of another dtype than its element"),
    }
}

impl Dit {
    /// Reads the plan: the guest inputs it declares and the classes that
    /// read them, the group selections its packed attention reads, the
    /// merges over inputs, the voxel input, and the float seams.
    pub fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Dit> {
        let mut dit = Dit {
            feeds: Feeds::of(trace, compiled),
            ..Dit::default()
        };
        if let Some(value) = dit.feeds.unlanded.first() {
            return Err(program(format!(
                "value {}, a merge over a runtime input this shell cannot land: only a lane or \
                 row guest input under a conjunction of facts is landed in a merged column \
                 before the walk",
                value.0
            )));
        }
        let mut voxel: Option<VoxelSeat> = None;
        for (seat, _) in dit.feeds.inputs.iter().filter(|(seat, _)| seat.voxel()) {
            let have = voxel.get_or_insert_with(|| VoxelSeat {
                widths: Vec::new(),
                dtype: seat.dtype,
                token_patch: None,
            });
            if !have.widths.contains(&seat.width) {
                have.widths.push(seat.width);
            }
            have.dtype = seat.dtype;
        }
        let mut grid = false;
        for decl in &trace.values {
            match decl.def {
                Def::Input(RuntimeInput::TokenGrid { p }) => {
                    voxel
                        .get_or_insert_with(|| VoxelSeat {
                            widths: Vec::new(),
                            dtype: Dtype::Bf16,
                            token_patch: None,
                        })
                        .token_patch = Some(p);
                }
                Def::Input(RuntimeInput::Grid) => grid = true,
                _ => {}
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

    /// The voxel input's element, when the plan declares one.
    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel
            .as_ref()
            .filter(|seat| !seat.widths.is_empty())
            .map(|seat| seat.dtype)
    }

    /// Guest input `id`'s seat, if the plan reads it.
    #[must_use]
    pub fn seat(&self, id: u32) -> Option<Seat> {
        self.feeds.seat(id)
    }

    /// The feeds the plan reads (`Capabilities::inputs` is their
    /// `decls`).
    #[must_use]
    pub fn feeds(&self) -> &Feeds {
        &self.feeds
    }

    /// The guest inputs (not the voxel input) `class` reads
    /// (`crate::serve::dry`).
    pub(crate) fn dry_inputs(&self, class: usize) -> Vec<Seat> {
        self.feeds
            .inputs
            .iter()
            .filter(|(seat, readers)| !seat.voxel() && readers.contains(class))
            .map(|(seat, _)| *seat)
            .collect()
    }

    /// The voxel input's widths, element and token patch, when the plan has
    /// a voxel axis.
    pub(crate) fn dry_voxel(&self) -> Option<(Vec<u32>, Dtype, Option<[u32; 3]>)> {
        self.voxel
            .as_ref()
            .map(|seat| (seat.widths.clone(), seat.dtype, seat.token_patch))
    }

    /// The width of the voxel input `class` reads, if it reads one.
    pub(crate) fn dry_voxel_width(&self, class: usize) -> Option<u32> {
        self.feeds
            .inputs
            .iter()
            .find(|(seat, readers)| seat.voxel() && readers.contains(class))
            .map(|(seat, _)| seat.width)
    }

    /// Whether a fire of this plan stages anything here at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.feeds.inputs.is_empty()
            && self.feeds.selections.is_empty()
            && self.feeds.merged.is_empty()
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

        // ---- guest inputs
        let mut planes: Vec<(Seat, Vec<u8>)> = self
            .feeds
            .seats()
            .into_iter()
            .map(|seat| {
                let extent = if seat.per_lane() { lane_count } else { rows };
                (seat, vec![0u8; extent as usize * seat.row_bytes() as usize])
            })
            .collect();
        for (fire_lane, row) in lanes.iter().enumerate() {
            let Some(feed) = feed_of(row) else {
                continue;
            };
            for cell in feed.inputs {
                if self.feeds.seat(cell.id).is_none() {
                    return Err(program(format!(
                        "lane {} feeds guest input {} and this plan reads no such input",
                        row.source, cell.id
                    )));
                }
            }
            for (seat, plane) in &mut planes {
                if !self
                    .feeds
                    .readers(seat.id)
                    .is_some_and(|readers| readers.contains(row.class as usize))
                {
                    continue;
                }
                let Some(cell) = feed.inputs.iter().find(|cell| cell.id == seat.id) else {
                    return Err(program(format!(
                        "lane {} runs a class that reads guest input {} and feeds it no \
                         channel; a declared input is fed from a channel cell at every submit \
                         (`Lane::inputs`)",
                        row.source, seat.id
                    )));
                };
                let (first, count) = if seat.per_lane() {
                    (fire_lane as u32, 1)
                } else {
                    (row.row_offset, row.rows)
                };
                let row_bytes = seat.row_bytes() as usize;
                let want = count as usize * row_bytes;
                if cell.bytes.len() != want {
                    return Err(program(format!(
                        "lane {} feeds guest input {} a cell of {} byte(s), and the input wants \
                         {count} row(s) x {} x {:?} = {want} (the lane's rows by the input's \
                         width)",
                        row.source,
                        seat.id,
                        cell.bytes.len(),
                        seat.width,
                        seat.dtype
                    )));
                }
                let at = first as usize * row_bytes;
                plane[at..at + want].copy_from_slice(cell.bytes);
            }
        }
        for (seat, plane) in planes {
            let extent = if seat.per_lane() { lane_count } else { rows };
            let tensor = inputs.raw(handles, seat.dtype, extent, seat.width, plane);
            staged.bindings.inputs.push(InputBinding {
                id: seat.id,
                tensor,
            });
            key.push_str(&format!("i{}x{};", seat.id, seat.width));
        }

        // ---- merges over guest inputs
        for (fire_lane, row) in lanes.iter().enumerate() {
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = feed_of(row)
                    .is_some_and(|feed| feed.inputs.iter().any(|cell| cell.id == merged.seat.id));
                let (first, count) = if merged.seat.per_lane() {
                    (fire_lane as u32, 1)
                } else {
                    (row.row_offset, row.rows)
                };
                let input = fed
                    .then(|| {
                        staged
                            .bindings
                            .inputs
                            .iter()
                            .find(|bound| bound.id == merged.seat.id)
                            .map(|bound| bound.tensor)
                    })
                    .flatten();
                staged.lands.push(Land {
                    merge: merged.merge,
                    first,
                    rows: count,
                    input,
                });
                key.push_str(&format!(
                    "m{}@{first}+{count}{};",
                    merged.merge.0,
                    if input.is_some() { "f" } else { "z" }
                ));
            }
        }

        // ---- attention-group packings
        if !self.feeds.selections.is_empty() {
            let facts: Vec<LaneFacts> = (0..lanes
                .iter()
                .map(|row| row.source as usize + 1)
                .max()
                .unwrap_or(0))
                .map(|source| {
                    feeds
                        .get(source)
                        .map_or(LaneFacts { group: None }, |feed| LaneFacts {
                            group: feed.group,
                        })
                })
                .collect();
            let mut groups = poem_exec::fire::group_of_lane(lanes, &facts);
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
            for &select in &self.feeds.selections {
                let packed =
                    poem_exec::fire::pack_with_classes(select, lanes, &groups, rows, &lane_classes)
                        .map_err(poem_exec::Error::Fire)?;
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
                    permutation: inputs.i32s(handles, &row_table(&packed.permutation), 1),
                    attn_class: inputs.i32s(handles, &row_table(&packed.attn_class), 1),
                };
                staged.bindings.packings.push((select, bindings));
            }
            key.push_str(&format!("pk{};", self.feeds.selections.len()));
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
            if self.feeds.selections.is_empty() {
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
        let elem = poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(0) as usize;
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
                    "lane {}'s voxel payload is not one input row per voxel of its clips at a \
                     width the plan reads ({:?})",
                    row.source, seat.widths
                )));
            }
            if channels != 0 && channels != width {
                return Err(program(format!(
                    "lane {} feeds a voxel input of another width than an earlier lane of the \
                     fire (one voxel class per fire)",
                    row.source
                )));
            }
            channels = width;
        }
        if channels == 0 && !seat.widths.is_empty() {
            return Err(program(
                "the fire's clips carry no voxel payload and feed no voxel input, and this plan \
                 reads its voxels off the input"
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
                grid[clip * 4..clip * 4 + 4].copy_from_slice(&[
                    *t as i32,
                    *h as i32,
                    *w as i32,
                    voxel_row as i32,
                ]);
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

/// Lands each merge-over-input's rows in its slot before the walk: the
/// input's rows for a lane that fed it, zeros for one that did not.
pub fn land(
    ctx: &dyn Emit,
    handles: &Handles,
    slots: &crate::run::SlotTable,
    lands: &[Land],
) -> Result<()> {
    for land in lands {
        let whole = slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, a merge over a guest input, which the carve gave no rectangle",
                land.merge.0
            ),
        })?;
        let at = handles.cut(whole, land.first, land.rows);
        let from = land
            .input
            .map(|input| handles.cut(input, land.first, land.rows));
        ctx.emit(&mut |cx| {
            let v = match from {
                Some(input) => cx.read(input)?,
                None => {
                    let elem = kernels_xla::elem_of("serve.land", at.dtype)?;
                    cx.const_f(elem, 0.0, &[i64::from(at.rows), i64::from(at.width)])
                }
            };
            cx.write(at, v)
        })
        .map_err(|why| {
            program(format!(
                "landing merge {} over a guest input: {why:?}",
                land.merge.0
            ))
        })?;
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
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
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            );
        }
        out[lane] = (values, boxes);
    }
    out
}
