#![allow(unsafe_code)]
use kernels_slang::Tensor;
use poem_compiler::Budget;
use poem_ir::Dtype;

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::store::TableSeat;
use crate::store::kv::{Geometry, Paging};

const ALIGN: u64 = 256;

const AXES: u64 = 3;

#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    pub rows: u64,

    pub row_bytes: u64,

    pub images: u64,

    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelSeat {
    pub rows: u64,
    pub clips: u64,
    pub channels: u64,
    pub dtype: Dtype,
    pub token_patch: Option<[u32; 3]>,
}

#[derive(Debug, Clone, Copy)]
struct VoxelAt {
    payload: u64,
    grid: u64,
    token_grid: u64,
    slots: u64,
    seat: VoxelSeat,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelHandles {
    pub payload: Tensor,
    pub grid: Tensor,
    pub token_grid: Option<Tensor>,
    pub slots: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelFire<'a> {
    pub grid: &'a [i32],
    pub token_grid: &'a [i32],
    pub slots: &'a [i32],
    pub payload: &'a [u8],
}

pub use poem_exec::feeds::Seat;

#[derive(Debug, Clone, Copy)]
struct SeatAt {
    seat: Seat,
    at: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct PackingAt {
    group_indptr: u64,
    lane_indptr: u64,
    permutation: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PackingHandles {
    pub group_indptr: Tensor,
    pub lane_indptr: Tensor,
    pub permutation: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct PackingFire<'a> {
    pub group_indptr: &'a [i32],
    pub lane_indptr: &'a [i32],
    pub permutation: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
struct PatchAt {
    payload: u64,
    segments: u64,
    routes: u64,
    positions: u64,
    grids: u64,
    seat: PatchSeat,
}

#[derive(Debug, Clone, Copy)]
struct TableAt {
    indptr: u64,
    indices: u64,
    last_page_len: u64,
    kv_len: u64,
    write_page: u64,
    write_offset: u64,
}

#[derive(Debug, Clone)]
pub struct Handles {
    pub tokens: Tensor,

    pub positions: Tensor,

    pub readout_rows: Tensor,

    pub windows: u32,

    pub page_tables: Vec<TableHandles>,

    pub slot_ids: Tensor,

    pub slot_of_row: Tensor,

    pub row_valid: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,

    pub patches: Option<PatchHandles>,

    pub mrope_positions: Option<Tensor>,

    pub rs_replay: Option<Tensor>,
    pub rs_commit: Option<Tensor>,

    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<PackingHandles>,
    pub inputs: Vec<(u32, Tensor)>,
    pub voxels: Option<VoxelHandles>,
    pub group_masks: Vec<GroupMaskHandles>,
    /// The snapshot table, where a lane snapshots.
    pub rs_snaps: Option<Tensor>,
}

#[derive(Debug, Clone, Copy)]
pub struct PatchHandles {
    pub patches: Tensor,

    pub segments: Tensor,

    pub routes: Tensor,

    pub positions: Tensor,

    pub grids: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct TableHandles {
    pub indptr: Tensor,

    pub indices: Tensor,

    pub last_page_len: Tensor,

    pub kv_len: Tensor,

    pub write_page: Tensor,

    pub write_offset: Tensor,
}

#[derive(Debug, Clone)]
pub struct Fire<'a> {
    pub tokens: &'a [i32],

    pub positions: &'a [i32],

    pub readout_rows: &'a [i32],

    pub windows: &'a [i32],

    pub slot_ids: &'a [i32],

    pub slot_of_row: &'a [i32],

    pub adapter_routes: Option<&'a [i32]>,

    pub request_of_token: &'a [i32],

    pub page_tables: &'a [Geometry],

    pub mask: Option<&'a crate::mask::Staged>,

    pub patches: Option<PatchFire<'a>>,

    pub mrope_positions: Option<&'a [i32]>,

    pub rs_replay: Option<&'a [i32]>,
    pub rs_commit: Option<&'a [i32]>,

    pub group_of_lane: &'a [i32],
    pub packings: &'a [PackingFire<'a>],
    pub voxels: Option<VoxelFire<'a>>,
    /// Each ragged packing's mask, with the selections it packs.
    pub group_masks: &'a [(
        poem_ir::Selection,
        poem_ir::Selection,
        poem_exec::fire::GroupMask,
    )],
    /// The snapshot table (`kernels_slang::ssm::Snapshots`), where a lane
    /// snapshots.
    pub snapshots: Option<&'a [i32]>,
}

#[derive(Debug, Clone, Copy)]
pub struct PatchFire<'a> {
    pub payload: &'a [u8],

    pub segments: &'a [i32],

    pub routes: &'a [i32],

    pub positions: &'a [i32],

    pub grids: &'a [i32],
}

/// What a load's staged inputs are laid by, beside a rung's rows and lanes.
#[derive(Debug, Clone, Copy)]
pub struct Shape<'a> {
    pub paging: Paging,
    pub page_tables: usize,
    /// The most pages a lane's table takes in any list.
    pub table_pages: u32,
    pub classes: usize,
    pub gathered: usize,
    pub patch: Option<PatchSeat>,
    pub mrope: bool,
    pub seats: &'a [Seat],
    pub selections: usize,
    /// The attention groups the model packs: a group mask each.
    pub ragged: usize,
    pub voxel: Option<VoxelSeat>,
    /// The units a slot entry holds: a record's pieces.
    pub pieces: u32,
    /// Whether a lane may snapshot its recurrent state: a table of them.
    pub snapshots: bool,
}

/// Where each table of a fire of at most `rows` rows and `lanes` lanes lies
/// in its rung's inputs region ([`crate::arena::Span::inputs`]).
#[derive(Debug, Clone)]
struct Layout {
    rows: u32,
    lanes: u32,
    tokens: u64,
    positions: u64,
    readout_rows: u64,
    windows: u64,
    window_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    slot_of_row: u64,
    pieces: u32,
    adapter_routes: u64,
    request_of_token: u64,
    mask_planes: u64,
    mask_plane_bytes: u64,
    mask_enabled: u64,

    mask_stride: u32,
    page_tables: Vec<TableAt>,

    patch: Option<PatchAt>,

    mrope: Option<u64>,

    rs_replay: u64,
    rs_commit: u64,

    group_of_lane: u64,
    packings: Vec<PackingAt>,
    seats: Vec<SeatAt>,
    voxels: Option<VoxelAt>,
    /// The snapshot table (`kernels_slang::ssm::Snapshots`), an entry a lane.
    snapshots: u64,
    snapshot_bytes: u64,
    /// The group masks' room: their size is the fire's groups', bounded by
    /// the rung's rows and lanes ([`group_mask_bytes`]).
    group_masks: u64,
    group_mask_bytes: u64,
    bytes: u64,
}

/// The most a fire's group masks take, one each of `ragged` packings: an
/// offset a group of at most `lanes`, and a bit a query row and key row of
/// one group, `rows` rows of each at most, each group's bits on a byte.
#[must_use]
pub fn group_mask_bytes(ragged: usize, rows: u64, lanes: u64) -> u64 {
    let indptr = (lanes + 1) * 4;
    let bits = (rows * rows).div_ceil(8) + lanes + 4;
    ragged as u64 * (indptr.next_multiple_of(ALIGN) + bits.next_multiple_of(ALIGN))
}

impl Layout {
    fn lay(rows: u32, lanes: u32, shape: &Shape<'_>) -> Layout {
        let (fire_rows, fire_lanes) = (rows, lanes);
        let pieces = shape.pieces.max(1);
        let paging = shape.paging;
        let rows = u64::from(rows);
        let lanes = u64::from(lanes);
        // A lane's table holds as many pages as its narrowest list takes.
        let pages = lanes * u64::from(paging.pages_per_slot.max(shape.table_pages));

        let per_gathered = 3 * rows + shape.page_tables as u64 * (2 * lanes + (lanes + 1) + pages);
        let window_ints = (shape.classes * (shape.classes + 1) / 2 + 1) as u64 * (lanes + 1)
            + shape.gathered as u64 * per_gathered;

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let readout_rows = take(rows * 4);
        let windows = take(window_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * u64::from(pieces) * 4);

        let slot_of_row = take(rows * u64::from(pieces) * 4);

        let adapter_routes = take(rows * 4);
        let request_of_token = take(rows * 4);

        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_stride = u32::try_from(context).unwrap_or(u32::MAX);
        let mask_plane_bytes = rows * context;
        let mask_planes = take(mask_plane_bytes);

        let mask_enabled = take(rows);
        let page_tables: Vec<TableAt> = (0..shape.page_tables)
            .map(|_| TableAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();

        // A rung narrows a fire's tokens and lanes; its patches, images,
        // voxels and clips keep the load's ceilings, as its carve does.
        let patch = shape.patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            positions: take(seat.rows * AXES * 4),
            grids: take(seat.images * AXES * 4),
            seat,
        });
        let mrope = shape.mrope.then(|| take(rows * AXES * 4));
        let rs_replay = take(lanes * 4);
        let rs_commit = take(lanes * 4);
        let group_of_lane = if shape.selections == 0 {
            0
        } else {
            take(lanes * 4)
        };
        let packings: Vec<PackingAt> = (0..shape.selections)
            .map(|_| PackingAt {
                group_indptr: take((lanes + 1) * 4),
                lane_indptr: take((lanes + 1) * 4),
                permutation: take(rows * 4),
            })
            .collect();
        let seats: Vec<SeatAt> = shape
            .seats
            .iter()
            .map(|seat| {
                let bytes = if seat.per_lane() { lanes } else { rows } * seat.row_bytes();
                SeatAt {
                    seat: *seat,
                    at: take(bytes),
                    bytes,
                }
            })
            .collect();
        let voxels = shape.voxel.map(|seat| {
            let element = poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(2);
            VoxelAt {
                payload: take(seat.rows * seat.channels * element),
                grid: take(seat.clips * 4 * 4),
                token_grid: if seat.token_patch.is_some() {
                    take(seat.clips * 4 * 4)
                } else {
                    0
                },
                slots: take(seat.clips * 4),
                seat,
            }
        });
        let snapshot_bytes = if shape.snapshots {
            lanes * u64::from(kernels_slang::ssm::RS_SNAPSHOTS) * (1 + u64::from(pieces)) * 4
        } else {
            0
        };
        let snapshots = take(snapshot_bytes);
        let group_mask_bytes = group_mask_bytes(shape.ragged, rows, lanes);
        let group_masks = take(group_mask_bytes);
        Layout {
            rows: fire_rows,
            lanes: fire_lanes,
            tokens,
            positions,
            readout_rows,
            windows,
            window_ints,
            row_valid,
            slot_ids,
            slot_of_row,
            pieces,
            adapter_routes,
            request_of_token,
            mask_planes,
            mask_plane_bytes,
            mask_enabled,
            mask_stride,
            page_tables,
            patch,
            mrope,
            rs_replay,
            rs_commit,
            group_of_lane,
            packings,
            seats,
            voxels,
            snapshots,
            snapshot_bytes,
            group_masks,
            group_mask_bytes,
            bytes: at,
        }
    }
}

/// An arm's host stage, where the host maps its buffers and cannot write
/// the pool's ([`Backend::MAPS_HOST`]): the fire's tables are laid in it and
/// its frame copies `runs` (`(at, bytes)` of the rung's region) into the
/// span before any kernel reads them.
#[derive(Debug)]
struct HostStage<B: Backend> {
    buffer: B::Buffer,
    runs: Vec<(u64, u64)>,
}

/// A fire's staged inputs: every table it reads lies in its rung's inputs
/// region of the activation span, at the rung's layout. Nothing of theirs
/// is reserved after the load.
#[derive(Debug)]
pub struct Inputs<B: Backend> {
    /// Each rung's layout, as the span's rungs lie.
    layouts: Vec<Layout>,
    /// An arm's each, or none where the queue writes the region itself.
    stages: Vec<HostStage<B>>,
}

#[derive(Debug, Clone, Copy)]
pub struct GroupMaskHandles {
    pub q: poem_ir::Selection,
    pub kv: poem_ir::Selection,
    pub bits: Tensor,
    pub indptr: Tensor,
}

/// Where a fire's tables are laid: through the queue into the rung's
/// region, or in the arm's host stage its frame copies there.
struct Stage<'a, B: Backend> {
    host: Option<&'a mut HostStage<B>>,
    region: &'a B::Buffer,
    base: u64,
    bytes: u64,
}

impl<B: Backend> Stage<'_, B> {
    /// Lays `bytes` at `at` of the region: a table from its start, so the
    /// queue's grain is filled with zeros in the table's own room.
    fn put(&mut self, at: u64, bytes: &[u8]) -> Result<()> {
        let grain = usize::try_from(B::COPY_GRAIN).unwrap_or(1).max(1);
        if self.host.is_some() || bytes.len().is_multiple_of(grain) {
            return self.put_exact(at, bytes);
        }
        let mut whole = bytes.to_vec();
        whole.resize(bytes.len().next_multiple_of(grain), 0);
        self.put_exact(at, &whole)
    }

    fn put_exact(&mut self, at: u64, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len() as u64;
        if at + len > self.bytes {
            return Err(Fault::Ceiling {
                what: "bytes of a rung's staged inputs",
                need: at + len,
                have: self.bytes,
            });
        }
        match &mut self.host {
            Some(stage) => {
                stage.buffer.write(at, bytes)?;
                stage.runs.push((at, len));
                Ok(())
            }
            None => self.region.write_shared(self.base + at, bytes),
        }
    }
}

impl<B: Backend> Inputs<B> {
    /// The layout of every rung of `tokens` (ascending, as the span's): a
    /// rung of so many tokens carries fires of at most so many, and lanes.
    #[must_use]
    pub fn plan(budget: &Budget, tokens: &[u32], shape: &Shape<'_>) -> Inputs<B> {
        Inputs {
            layouts: tokens
                .iter()
                .map(|&tokens| {
                    Layout::lay(
                        tokens.min(budget.max_tokens),
                        tokens.min(budget.max_lanes),
                        shape,
                    )
                })
                .collect(),
            stages: Vec::new(),
        }
    }

    /// Each rung's inputs region, in bytes.
    #[must_use]
    pub fn rung_bytes(&self) -> Vec<u64> {
        self.layouts.iter().map(|layout| layout.bytes).collect()
    }

    /// Reserves each of `arms` arms' host stage, where the host cannot
    /// write the pool's bytes: the widest rung's.
    pub fn arm(&mut self, device: &B::Context, arms: usize) -> Result<()> {
        if !B::MAPS_HOST {
            return Ok(());
        }
        let bytes = self
            .layouts
            .iter()
            .map(|layout| layout.bytes)
            .max()
            .unwrap_or(0);
        self.stages = (0..arms)
            .map(|_| {
                Ok(HostStage {
                    buffer: B::Buffer::host(device, bytes, crate::books::owners::INPUTS)?,
                    runs: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    /// The bytes of the arms' host stages.
    #[must_use]
    pub fn stage_bytes(&self) -> u64 {
        self.stages.iter().map(|stage| stage.buffer.bytes()).sum()
    }

    fn layout(&self, rung: usize) -> Result<&Layout> {
        self.layouts.get(rung).ok_or_else(|| Fault::Unbound {
            what: format!("the staged inputs' layout at rung {rung}, which the load laid none of"),
        })
    }

    /// Where arm `arm`'s fire at rung `rung` lands what the host and the
    /// device feed it: the buffer, the byte its layout starts at and the
    /// layout's bytes.
    pub fn landing<'a>(
        &'a self,
        span: &'a crate::arena::Span<B>,
        rung: usize,
        arm: usize,
    ) -> Result<(&'a B::Buffer, u64, u64)> {
        let bytes = self.layout(rung)?.bytes;
        match self.stages.get(arm) {
            Some(stage) => Ok((&stage.buffer, 0, bytes)),
            None => {
                let (region, base, _) = span.inputs(rung)?;
                Ok((region, base, bytes))
            }
        }
    }

    /// The runs arm `arm`'s frame copies from its host stage into rung
    /// `rung`'s region, `(stage byte, span byte, bytes)`; none where the
    /// queue wrote the region.
    pub fn staged(
        &self,
        span: &crate::arena::Span<B>,
        rung: usize,
        arm: usize,
    ) -> Result<Vec<(u64, u64, u64)>> {
        let Some(stage) = self.stages.get(arm) else {
            return Ok(Vec::new());
        };
        let (_, base, _) = span.inputs(rung)?;
        // Tables a room's padding apart copy as one run: the padding is
        // no table's.
        let mut runs = stage.runs.clone();
        runs.sort_unstable();
        let mut joined: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
        for (at, bytes) in runs {
            match joined.last_mut() {
                Some((first, len)) if at <= (*first + *len).next_multiple_of(ALIGN) => {
                    *len = (*len).max(at + bytes - *first);
                }
                _ => joined.push((at, bytes)),
            }
        }
        Ok(joined
            .into_iter()
            .map(|(at, bytes)| (at, base + at, bytes))
            .collect())
    }

    /// The arm's host stage, which its frame copies the staged runs out of.
    #[must_use]
    pub fn stage(&self, arm: usize) -> Option<&B::Buffer> {
        self.stages.get(arm).map(|stage| &stage.buffer)
    }

    /// States that the device writes `bytes` at `at` of arm `arm`'s
    /// landing before its frame: the frame carries them into the region.
    pub fn fed(&mut self, arm: usize, at: u64, bytes: u64) {
        if let Some(stage) = self.stages.get_mut(arm)
            && bytes > 0
        {
            stage.runs.push((at, bytes));
        }
    }

    /// Where the voxel payload lies in rung `rung`'s layout.
    #[must_use]
    pub fn voxel_payload(&self, rung: usize) -> Option<u64> {
        self.layouts.get(rung)?.voxels.map(|at| at.payload)
    }

    /// Lays a guest input's rows at `at` of rung `rung`'s layout.
    pub fn write_input(
        &mut self,
        span: &crate::arena::Span<B>,
        rung: usize,
        arm: usize,
        at: u64,
        bytes: &[u8],
    ) -> Result<()> {
        let room = self.layout(rung)?.bytes;
        let (region, base, _) = span.inputs(rung)?;
        Stage::<B> {
            host: self.stages.get_mut(arm),
            region,
            base,
            bytes: room,
        }
        .put_exact(at, bytes)
    }

    /// Where guest input `id`'s seat lies in rung `rung`'s layout.
    #[must_use]
    pub fn seat(&self, rung: usize, id: u32) -> Option<u64> {
        self.layouts
            .get(rung)?
            .seats
            .iter()
            .find(|at| at.seat.id == id)
            .map(|at| at.at)
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.layouts.last()?.patch.map(|at| at.seat.dtype)
    }

    /// Where row 0's token id lies in rung `rung`'s layout, for a token the
    /// device writes in place of the host.
    #[must_use]
    pub fn token_seat(&self, rung: usize) -> u64 {
        self.layouts.get(rung).map_or(0, |layout| layout.tokens)
    }

    /// Lays `fire`'s tables in rung `rung`'s region of `span` (arm `arm`'s
    /// host stage, where the host cannot write the pool) and binds them
    /// there. A fire past the rung's bounds is refused by name.
    pub fn write(
        &mut self,
        handles: &crate::handles::Handles<B>,
        span: &crate::arena::Span<B>,
        rung: usize,
        arm: usize,
        fire: &Fire<'_>,
    ) -> Result<Handles> {
        let map = self.layouts.get(rung).ok_or_else(|| Fault::Unbound {
            what: format!("the staged inputs' layout at rung {rung}, which the load laid none of"),
        })?;
        let (region, base, room) = span.inputs(rung)?;
        if room < map.bytes {
            return Err(Fault::Ceiling {
                what: "bytes of a rung's staged inputs in the activation span",
                need: map.bytes,
                have: room,
            });
        }
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32 / map.pieces;
        for (what, need, have) in [
            ("rows of a fire at its rung", rows, map.rows),
            ("lanes of a fire at its rung", lanes, map.lanes),
        ] {
            if need > have {
                return Err(Fault::Ceiling {
                    what,
                    need: u64::from(need),
                    have: u64::from(have),
                });
            }
        }
        let room_of = |bytes: usize| (bytes as u64).next_multiple_of(ALIGN);
        let mask_need: u64 = fire
            .group_masks
            .iter()
            .map(|(_, _, mask)| {
                room_of(mask.indptr.len() * 4) + room_of(mask.bits.len().div_ceil(4).max(1) * 4)
            })
            .sum();
        if mask_need > map.group_mask_bytes {
            return Err(Fault::Ceiling {
                what: "bytes of the group masks a fire stages at its rung",
                need: mask_need,
                have: map.group_mask_bytes,
            });
        }
        let snapshot_need = fire.snapshots.map_or(0, |table| table.len() as u64 * 4);
        if snapshot_need > map.snapshot_bytes {
            return Err(Fault::Ceiling {
                what: "bytes of the snapshot table a fire stages at its rung",
                need: snapshot_need,
                have: map.snapshot_bytes,
            });
        }
        let mut host = self.stages.get_mut(arm);
        if let Some(stage) = host.as_deref_mut() {
            stage.runs.clear();
        }
        let mut stage = Stage::<B> {
            host,
            region,
            base,
            bytes: map.bytes,
        };

        let mut group_masks = Vec::with_capacity(fire.group_masks.len());
        let mut at = map.group_masks;
        for (q, kv, mask) in fire.group_masks {
            let indptr = at;
            stage.put(indptr, bytes_of(&mask.indptr))?;
            at += room_of(mask.indptr.len() * 4);
            // The kernel reads the bits a word at a time.
            let words = mask.bits.len().div_ceil(4).max(1);
            let mut bits = mask.bits.clone();
            bits.resize(words * 4, 0);
            stage.put(at, &bits)?;
            group_masks.push(GroupMaskHandles {
                q: *q,
                kv: *kv,
                bits: Tensor::new(
                    handles.bind(region, base + at, words as u64 * 4)?,
                    u32::try_from(words * 4).unwrap_or(u32::MAX),
                    1,
                    Dtype::U8,
                ),
                indptr: i32s(
                    handles,
                    region,
                    base + indptr,
                    u32::try_from(mask.indptr.len()).unwrap_or(u32::MAX),
                )?,
            });
            at += room_of(words * 4);
        }
        let snapshots = match fire.snapshots {
            None => None,
            Some(table) => {
                stage.put(map.snapshots, bytes_of(table))?;
                Some(i32s(
                    handles,
                    region,
                    base + map.snapshots,
                    u32::try_from(table.len()).unwrap_or(u32::MAX),
                )?)
            }
        };

        let valid = vec![1u8; rows as usize];

        stage.put(map.tokens, bytes_of(fire.tokens))?;
        stage.put(map.positions, bytes_of(fire.positions))?;
        if fire.readout_rows.len() > rows as usize {
            return Err(Fault::Ceiling {
                what: "readout rows in one fire",
                need: fire.readout_rows.len() as u64,
                have: u64::from(rows),
            });
        }
        stage.put(map.readout_rows, bytes_of(fire.readout_rows))?;
        if fire.windows.len() as u64 > map.window_ints {
            return Err(Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: map.window_ints,
            });
        }
        stage.put(map.windows, bytes_of(fire.windows))?;
        stage.put(map.row_valid, &valid)?;
        stage.put(map.slot_ids, bytes_of(fire.slot_ids))?;
        stage.put(map.request_of_token, bytes_of(fire.request_of_token))?;
        stage.put(map.slot_of_row, bytes_of(fire.slot_of_row))?;
        let rs_tables = match (fire.rs_replay, fire.rs_commit) {
            (Some(replay), Some(commit)) => {
                if replay.len() != lanes as usize || commit.len() != lanes as usize {
                    return Err(Fault::Ceiling {
                        what: "recurrent seat tables, one entry per lane",
                        need: replay.len().max(commit.len()) as u64,
                        have: u64::from(lanes),
                    });
                }
                stage.put(map.rs_replay, bytes_of(replay))?;
                stage.put(map.rs_commit, bytes_of(commit))?;
                true
            }
            _ => false,
        };

        let adapter_routes = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                stage.put(map.adapter_routes, bytes_of(routes))?;
                Some(routes.len() as u32)
            }
        };

        let stride = fire.mask.map_or(0, |staged| staged.stride);
        if u64::from(stride) > u64::from(map.mask_stride) {
            return Err(Fault::Ceiling {
                what: "key positions in one mask row",
                need: u64::from(stride),
                have: u64::from(map.mask_stride),
            });
        }
        match fire.mask {
            None => {
                stage.put(map.mask_enabled, &vec![0u8; rows as usize])?;
            }
            Some(staged) => {
                if staged.bytes.len() as u64 > map.mask_plane_bytes {
                    return Err(Fault::Ceiling {
                        what: "mask plane bytes",
                        need: staged.bytes.len() as u64,
                        have: map.mask_plane_bytes,
                    });
                }
                stage.put(map.mask_planes, &staged.bytes)?;
                stage.put(map.mask_enabled, &staged.enabled)?;
            }
        }

        let patches = match (fire.patches, map.patch) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(Fault::Ceiling {
                    what: "patch rows against a load that reserved none",
                    need: 1,
                    have: 0,
                });
            }
            (Some(staged), Some(at)) => {
                let seat = at.seat;

                let rows = (staged.payload.len() as u64)
                    .checked_div(seat.row_bytes)
                    .unwrap_or(0);
                for (what, have, ceiling) in [
                    (
                        "patch payload bytes",
                        staged.payload.len() as u64,
                        seat.rows * seat.row_bytes,
                    ),
                    (
                        "patch segments",
                        staged.segments.len() as u64,
                        seat.images + 1,
                    ),
                    ("patch routes", staged.routes.len() as u64, seat.rows),
                    (
                        "patch positions",
                        staged.positions.len() as u64,
                        seat.rows * AXES,
                    ),
                    ("image grids", staged.grids.len() as u64, seat.images * AXES),
                ] {
                    if have > ceiling {
                        return Err(Fault::Ceiling {
                            what,
                            need: have,
                            have: ceiling,
                        });
                    }
                }
                stage.put(at.payload, staged.payload)?;
                stage.put(at.segments, bytes_of(staged.segments))?;
                stage.put(at.routes, bytes_of(staged.routes))?;
                stage.put(at.positions, bytes_of(staged.positions))?;
                stage.put(at.grids, bytes_of(staged.grids))?;
                let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
                let element = poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(1);
                let width = u32::try_from(seat.row_bytes.checked_div(element).unwrap_or(0))
                    .unwrap_or(u32::MAX);
                Some(PatchHandles {
                    patches: Tensor::new(
                        handles.bind(region, base + at.payload, staged.payload.len() as u64)?,
                        rows32,
                        width,
                        seat.dtype,
                    ),
                    segments: i32s(
                        handles,
                        region,
                        base + at.segments,
                        staged.segments.len() as u32,
                    )?,
                    routes: i32s(
                        handles,
                        region,
                        base + at.routes,
                        staged.routes.len() as u32,
                    )?,

                    positions: Tensor::new(
                        handles.bind(
                            region,
                            base + at.positions,
                            staged.positions.len() as u64 * 4,
                        )?,
                        rows32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                    grids: Tensor::new(
                        handles.bind(region, base + at.grids, staged.grids.len() as u64 * 4)?,
                        staged.grids.len() as u32 / AXES as u32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                })
            }
        };

        let mrope_positions = match (fire.mrope_positions, map.mrope) {
            (None, _) | (_, None) => None,
            (Some(triples), Some(at)) => {
                if triples.len() as u64 > u64::from(rows) * AXES {
                    return Err(Fault::Ceiling {
                        what: "trunk rotation triples",
                        need: triples.len() as u64,
                        have: u64::from(rows) * AXES,
                    });
                }
                stage.put(at, bytes_of(triples))?;
                Some(Tensor::new(
                    handles.bind(region, base + at, triples.len() as u64 * 4)?,
                    triples.len() as u32 / AXES as u32,
                    AXES as u32,
                    Dtype::I32,
                ))
            }
        };

        let group_of_lane = if map.packings.is_empty() {
            None
        } else {
            if fire.group_of_lane.len() as u32 > lanes {
                return Err(Fault::Ceiling {
                    what: "group-of-lane entries",
                    need: fire.group_of_lane.len() as u64,
                    have: u64::from(lanes),
                });
            }
            stage.put(map.group_of_lane, bytes_of(fire.group_of_lane))?;
            Some(i32s(handles, region, base + map.group_of_lane, lanes)?)
        };
        if fire.packings.len() != map.packings.len() {
            return Err(Fault::Ceiling {
                what: "packing tables against the selections this load carved",
                need: fire.packings.len() as u64,
                have: map.packings.len() as u64,
            });
        }
        let mut packings = Vec::with_capacity(map.packings.len());
        let mut scratch: Vec<i32> = Vec::new();
        for (at, staged) in map.packings.iter().zip(fire.packings) {
            for (what, have, ceiling) in [
                (
                    "group indptr entries",
                    staged.group_indptr.len() as u64,
                    u64::from(lanes) + 1,
                ),
                (
                    "lane indptr entries",
                    staged.lane_indptr.len() as u64,
                    u64::from(lanes) + 1,
                ),
                (
                    "packed row permutation",
                    staged.permutation.len() as u64,
                    u64::from(rows),
                ),
            ] {
                if have > ceiling {
                    return Err(Fault::Ceiling {
                        what,
                        need: have,
                        have: ceiling,
                    });
                }
            }
            csr(&mut scratch, staged.group_indptr, lanes + 1);
            stage.put(at.group_indptr, bytes_of(&scratch))?;
            csr(&mut scratch, staged.lane_indptr, lanes + 1);
            stage.put(at.lane_indptr, bytes_of(&scratch))?;
            rows_table(&mut scratch, staged.permutation, rows);
            stage.put(at.permutation, bytes_of(&scratch))?;
            packings.push(PackingHandles {
                group_indptr: i32s(handles, region, base + at.group_indptr, lanes + 1)?,
                lane_indptr: i32s(handles, region, base + at.lane_indptr, lanes + 1)?,
                permutation: i32s(handles, region, base + at.permutation, rows)?,
            });
        }
        let mut inputs = Vec::with_capacity(map.seats.len());
        for at in &map.seats {
            let carved = if at.seat.per_lane() { lanes } else { rows };
            inputs.push((
                at.seat.id,
                Tensor::new(
                    handles.bind(region, base + at.at, at.bytes)?,
                    carved,
                    at.seat.width,
                    at.seat.dtype,
                ),
            ));
        }

        let voxels = match (fire.voxels, map.voxels) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(Fault::Ceiling {
                    what: "voxel rows against a load that reserved none",
                    need: 1,
                    have: 0,
                });
            }
            (Some(staged), Some(at)) => {
                let seat = at.seat;
                let element = poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(2);
                let clips = (staged.grid.len() / 4) as u64;
                for (what, have, ceiling) in [
                    ("voxel clips", clips, seat.clips),
                    (
                        "voxel payload bytes",
                        staged.payload.len() as u64,
                        seat.rows * seat.channels * element,
                    ),
                    (
                        "voxel token grid",
                        (staged.token_grid.len() / 4) as u64,
                        if seat.token_patch.is_some() {
                            seat.clips
                        } else {
                            0
                        },
                    ),
                    ("voxel clip slots", staged.slots.len() as u64, seat.clips),
                ] {
                    if have > ceiling {
                        return Err(Fault::Ceiling {
                            what,
                            need: have,
                            have: ceiling,
                        });
                    }
                }
                stage.put(at.grid, bytes_of(staged.grid))?;
                stage.put(at.slots, bytes_of(staged.slots))?;
                if !staged.token_grid.is_empty() {
                    stage.put(at.token_grid, bytes_of(staged.token_grid))?;
                }
                if !staged.payload.is_empty() {
                    stage.put(at.payload, staged.payload)?;
                }
                let clips32 = u32::try_from(clips).unwrap_or(u32::MAX);
                Some(VoxelHandles {
                    payload: Tensor::new(
                        handles.bind(
                            region,
                            base + at.payload,
                            seat.rows * seat.channels * element,
                        )?,
                        u32::try_from(seat.rows).unwrap_or(u32::MAX),
                        u32::try_from(seat.channels).unwrap_or(u32::MAX),
                        seat.dtype,
                    ),
                    grid: Tensor::new(
                        handles.bind(region, base + at.grid, clips * 16)?,
                        clips32,
                        4,
                        Dtype::I32,
                    ),
                    token_grid: if seat.token_patch.is_some() {
                        Some(Tensor::new(
                            handles.bind(region, base + at.token_grid, clips * 16)?,
                            clips32,
                            4,
                            Dtype::I32,
                        ))
                    } else {
                        None
                    },
                    slots: Tensor::new(
                        handles.bind(region, base + at.slots, clips * 4)?,
                        clips32,
                        1,
                        Dtype::I32,
                    ),
                })
            }
        };

        let mut page_tables = Vec::with_capacity(map.page_tables.len());
        for (at, geometry) in map.page_tables.iter().zip(fire.page_tables) {
            stage.put(at.indptr, bytes_of(&geometry.indptr))?;
            stage.put(at.indices, bytes_of(&geometry.indices))?;
            stage.put(at.last_page_len, bytes_of(&geometry.last_page_len))?;
            stage.put(at.kv_len, bytes_of(&geometry.kv_len))?;
            stage.put(at.write_page, bytes_of(&geometry.write_page))?;
            stage.put(at.write_offset, bytes_of(&geometry.write_offset))?;
            page_tables.push(TableHandles {
                indptr: i32s(handles, region, base + at.indptr, lanes + 1)?,
                indices: i32s(
                    handles,
                    region,
                    base + at.indices,
                    geometry.indices.len() as u32,
                )?,
                last_page_len: i32s(handles, region, base + at.last_page_len, lanes)?,
                kv_len: i32s(handles, region, base + at.kv_len, lanes)?,
                write_page: u32s(handles, region, base + at.write_page, rows)?,
                write_offset: u32s(handles, region, base + at.write_offset, rows)?,
            });
        }

        Ok(Handles {
            tokens: i32s(handles, region, base + map.tokens, rows)?,
            positions: i32s(handles, region, base + map.positions, rows)?,
            readout_rows: i32s(
                handles,
                region,
                base + map.readout_rows,
                fire.readout_rows.len().max(1) as u32,
            )?,

            windows: handles.bind(region, base + map.windows, fire.windows.len() as u64 * 4)?,
            page_tables,
            slot_ids: entries(handles, region, base + map.slot_ids, lanes, map.pieces)?,
            slot_of_row: entries(handles, region, base + map.slot_of_row, rows, map.pieces)?,
            adapter_routes: match adapter_routes {
                None => None,
                Some(rows) => Some(i32s(handles, region, base + map.adapter_routes, rows)?),
            },
            row_valid: Tensor::new(
                handles.bind(region, base + map.row_valid, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            request_of_token: i32s(handles, region, base + map.request_of_token, rows)?,

            mask: Tensor::new(
                handles.bind(
                    region,
                    base + map.mask_planes,
                    u64::from(rows) * u64::from(stride),
                )?,
                rows,
                stride.max(1),
                Dtype::U8,
            ),
            mask_enabled: Tensor::new(
                handles.bind(region, base + map.mask_enabled, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            mask_stride: stride,
            patches,
            mrope_positions,
            rs_replay: if rs_tables {
                Some(i32s(handles, region, base + map.rs_replay, lanes)?)
            } else {
                None
            },
            rs_commit: if rs_tables {
                Some(i32s(handles, region, base + map.rs_commit, lanes)?)
            } else {
                None
            },
            group_of_lane,
            packings,
            inputs,
            voxels,
            group_masks,
            rs_snaps: snapshots,
        })
    }

    #[must_use]
    pub fn seats(
        &self,
        _handles: &crate::handles::Handles<B>,
        views: &Handles,
        pages: u32,
        rows: u32,
        lanes: u32,
    ) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            page_tables: views
                .page_tables
                .iter()
                .map(|table| TableSeat {
                    page_indptr: table.indptr,
                    page_indices: table.indices,
                    last_page_lens: table.last_page_len,
                    row_valid: views.row_valid,
                })
                .collect(),
            slot_ids: views.slot_ids,
            slot_of_row: views.slot_of_row,
        }
    }
}

fn i32s<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
}

/// `rows` entries of `width` i32 units each: a slot table.
fn entries<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
    width: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * u64::from(width) * 4)?;
    Ok(Tensor::new(buf, rows, width, Dtype::I32))
}

fn u32s<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::U32))
}

fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: an `i32` slice is `size_of_val` initialized bytes with no
    // padding, `u8` has alignment 1, and the result borrows `values`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn csr(into: &mut Vec<i32>, staged: &[i32], ceiling: u32) {
    into.clear();
    into.extend_from_slice(staged);
    let last = into.last().copied().unwrap_or(0);
    into.resize(ceiling as usize, last);
}

fn rows_table(into: &mut Vec<i32>, staged: &[i32], rows: u32) {
    into.clear();
    into.extend_from_slice(staged);
    into.resize(rows as usize, -1);
}
