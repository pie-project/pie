#![allow(unsafe_code)]
use kernels_metal::Tensor;
use poem_compiler::Budget;
use poem_ir::Dtype;

use crate::device::{Buffer, Context};
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
    pub windows: u32,
    pub page_tables: Vec<TableHandles>,
    pub slot_ids: Tensor,
    pub slot_of_row: Tensor,
    /// Each band's lane and row tables of record piece ids
    /// (`crate::store::Pools::bands`), `[lanes, width]` and `[rows, width]`.
    pub lane_bands: Vec<Tensor>,
    pub row_bands: Vec<Tensor>,
    pub row_valid: Tensor,
    pub adapter_routes: Option<Tensor>,
    pub request_of_token: Tensor,
    pub readout_rows: Tensor,
    /// Each readout row's position, lane and mask flag: the tables a decode
    /// attention over readout rows reads in place of the tokens'.
    pub readout_positions: Tensor,
    pub readout_request_of_token: Tensor,
    pub readout_mask_enabled: Tensor,
    pub mask: Tensor,
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
    pub patches: Option<PatchHandles>,
    pub mrope_positions: Option<Tensor>,
    pub rs_replay: Option<Tensor>,
    pub rs_commit: Option<Tensor>,
    pub rs_snaps: Option<Tensor>,
    /// Each band's snapshot table, `[lanes, RS_SNAPSHOTS * (1 + width)]`.
    pub rs_snap_bands: Vec<Tensor>,
    pub rs_pages: Option<Tensor>,
    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<PackingHandles>,
    pub inputs: Vec<(u32, Tensor)>,
    pub voxels: Option<VoxelHandles>,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelHandles {
    pub payload: Tensor,
    pub grid: Tensor,
    pub token_grid: Option<Tensor>,
    pub slots: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct PackingHandles {
    pub group_indptr: Tensor,
    pub lane_indptr: Tensor,
    pub permutation: Tensor,
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
    pub windows: &'a [i32],
    pub slot_ids: &'a [i32],
    pub slot_of_row: &'a [i32],
    pub adapter_routes: Option<&'a [i32]>,
    pub request_of_token: &'a [i32],
    pub readout_rows: &'a [i32],
    pub page_tables: &'a [Geometry],
    pub mask: Option<&'a crate::mask::Staged>,
    pub patches: Option<PatchFire<'a>>,
    pub voxels: Option<VoxelFire<'a>>,
    pub mrope_positions: Option<&'a [i32]>,
    pub rs_replay: Option<&'a [i32]>,
    pub rs_commit: Option<&'a [i32]>,
    pub rs_snaps: Option<&'a [i32]>,
    pub rs_pages: Option<&'a [i32]>,
    pub group_of_lane: &'a [i32],
    pub packings: &'a [PackingFire<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct PackingFire<'a> {
    pub group_indptr: &'a [i32],
    pub lane_indptr: &'a [i32],
    pub permutation: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelFire<'a> {
    pub grid: &'a [i32],
    pub token_grid: &'a [i32],
    pub slots: &'a [i32],
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct PatchFire<'a> {
    pub payload: &'a [u8],
    pub segments: &'a [i32],
    pub routes: &'a [i32],
    pub positions: &'a [i32],
    pub grids: &'a [i32],
}

/// The most a fire's group masks take, one each of `ragged` packings: each
/// a word a group of at most `lanes` and its bit a query row and key row of
/// one group, `tokens` rows of each at most ([`Inputs::stage_group_masks`]).
#[must_use]
pub fn group_mask_bytes(ragged: usize, tokens: u32, lanes: u32) -> u64 {
    let indptr = (u64::from(lanes) + 1) * 4;
    let bits = (u64::from(tokens) * u64::from(tokens)).div_ceil(8).max(1);
    ragged as u64 * (indptr.next_multiple_of(ALIGN) + bits.next_multiple_of(ALIGN))
}

#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    slot_of_row: u64,
    /// Each band's width: the piece ids a lane and a row hold for it.
    bands: Vec<u32>,
    adapter_routes: u64,
    request_of_token: u64,
    readout_rows: u64,
    readout_positions: u64,
    readout_request_of_token: u64,
    readout_mask_enabled: u64,
    mask_planes: u64,
    mask_plane_bytes: u64,
    mask_enabled: u64,
    group_of_lane: u64,
    packings: Vec<PackingAt>,
    seats: Vec<SeatAt>,
    voxels: Option<VoxelAt>,
    mask_stride: u32,
    page_tables: Vec<TableAt>,
    patch: Option<PatchAt>,
    mrope: Option<u64>,
    rs_replay: u64,
    rs_commit: u64,
    rs_snaps: u64,
    rs_pages: u64,
    /// Staged apart from the store, sized at the load for the widest fire
    /// where the model packs attention groups ([`group_mask_bytes`]).
    group_masks: Option<Buffer>,
}

#[derive(Debug, Clone, Copy)]
pub struct GroupMaskHandles {
    pub q: poem_ir::Selection,
    pub kv: poem_ir::Selection,
    pub bits: Tensor,
    pub indptr: Tensor,
}

/// The ints one fire's packed window boundaries can fill, at the ceiling.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn window_ints(
    budget: &Budget,
    paging: Paging,
    page_tables: usize,
    classes: usize,
    gathered: usize,
) -> u64 {
    let rows = u64::from(budget.max_tokens);
    let lanes = u64::from(budget.max_lanes);
    let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);
    let intervals = (classes * (classes + 1) / 2) as u64;
    let per_gathered = 3 * rows + page_tables as u64 * (2 * lanes + (lanes + 1) + pages);
    (intervals + 1 + gathered as u64) * (lanes + 1) + gathered as u64 * per_gathered
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: &Context,
        budget: &Budget,
        paging: Paging,
        page_tables: usize,
        classes: usize,
        gathered: usize,
        patch: Option<PatchSeat>,
        voxel: Option<VoxelSeat>,
        mrope: bool,
        seats: &[Seat],
        selections: usize,
        bands: &[u32],
        pages_per_lane: u32,
        ragged: usize,
    ) -> Result<Inputs> {
        let width = u64::from(bands.iter().sum::<u32>().max(1));
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        // A table's page table holds the most pages a lane's table does in
        // any table: a window list of `T_w`-token slots takes more pages
        // than the kv list where `T_w` is the smaller.
        let paged = Paging {
            pages_per_slot: paging.pages_per_slot.max(pages_per_lane),
            ..paging
        };
        let pages = u64::from(budget.max_lanes) * u64::from(paged.pages_per_slot);
        let window_ints = window_ints(budget, paged, page_tables, classes, gathered);

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let windows = take(window_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * width * 4);
        let slot_of_row = take(rows * width * 4);
        let adapter_routes = take(rows * 4);
        let request_of_token = take(rows * 4);
        let readout_rows = take(rows * 4);
        let readout_positions = take(rows * 4);
        let readout_request_of_token = take(rows * 4);
        let readout_mask_enabled = take(rows);
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_stride = u32::try_from(context).unwrap_or(u32::MAX);
        let mask_plane_bytes = rows * context;
        let mask_planes = take(mask_plane_bytes);
        let mask_enabled = take(rows);
        let page_tables: Vec<TableAt> = (0..page_tables)
            .map(|_| TableAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();
        let patch = patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            positions: take(seat.rows * AXES * 4),
            grids: take(seat.images * AXES * 4),
            seat,
        });
        let group_of_lane = if selections == 0 { 0 } else { take(lanes * 4) };
        let packings: Vec<PackingAt> = (0..selections)
            .map(|_| PackingAt {
                group_indptr: take((lanes + 1) * 4),
                lane_indptr: take((lanes + 1) * 4),
                permutation: take(rows * 4),
            })
            .collect();
        let seats: Vec<SeatAt> = seats
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
        let voxels = voxel.map(|seat| {
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
        let mrope = mrope.then(|| take(rows * AXES * 4));
        let rs_replay = take(lanes * 4);
        let rs_commit = take(lanes * 4);
        let snap_ints =
            u64::from(kernels_metal::ssm::RS_SNAPSHOTS) * (bands.len().max(1) as u64 + width);
        let rs_snaps = take(lanes * snap_ints.max(u64::from(SNAP_INTS)) * 4);
        let rs_pages = take(lanes * u64::from(crate::rs::PAGE_INTS) * 4);
        let total = at;

        let store = Buffer::zeroed(device, total)?;
        Ok(Inputs {
            store,
            tokens,
            positions,
            windows,
            window_ints,
            row_valid,
            slot_ids,
            slot_of_row,
            bands: if bands.is_empty() {
                vec![1]
            } else {
                bands.to_vec()
            },
            adapter_routes,
            request_of_token,
            readout_rows,
            readout_positions,
            readout_request_of_token,
            readout_mask_enabled,
            mask_planes,
            mask_plane_bytes,
            mask_enabled,
            group_of_lane,
            packings,
            seats,
            voxels,
            mask_stride,
            page_tables,
            patch,
            mrope,
            rs_replay,
            rs_commit,
            rs_snaps,
            rs_pages,
            group_masks: if ragged == 0 {
                None
            } else {
                Some(Buffer::zeroed(
                    device,
                    group_mask_bytes(ragged, budget.max_tokens, budget.max_lanes),
                )?)
            },
        })
    }

    pub fn stage_group_masks(
        &mut self,
        handles: &crate::device::Handles,
        masks: &[(
            poem_ir::Selection,
            poem_ir::Selection,
            poem_exec::fire::GroupMask,
        )],
    ) -> Result<Vec<GroupMaskHandles>> {
        if masks.is_empty() {
            return Ok(Vec::new());
        }
        let mut image: Vec<u8> = Vec::new();
        let mut placed = Vec::with_capacity(masks.len());
        for (_, _, mask) in masks {
            let indptr = image.len() as u64;
            image.extend_from_slice(bytes_of(&mask.indptr));
            image.resize((image.len() as u64).next_multiple_of(ALIGN) as usize, 0);
            let bits = image.len() as u64;
            image.extend_from_slice(&mask.bits);
            image.resize((image.len() as u64).next_multiple_of(ALIGN) as usize, 0);
            placed.push((indptr, bits));
        }
        let have = self.group_masks.as_ref().map_or(0, Buffer::bytes);
        let Some(buffer) = self
            .group_masks
            .as_mut()
            .filter(|held| held.bytes() >= image.len() as u64)
        else {
            return Err(Fault::Ceiling {
                what: "bytes of the group masks sized at the load",
                need: image.len() as u64,
                have,
            });
        };
        buffer.write(0, &image)?;
        let buffer = &*buffer;
        masks
            .iter()
            .zip(placed)
            .map(|((q, kv, mask), (indptr, bits))| {
                let bytes = mask.bits.len().max(1) as u64;
                Ok(GroupMaskHandles {
                    q: *q,
                    kv: *kv,
                    bits: Tensor::new(
                        handles.bind(buffer, bits, bytes)?,
                        u32::try_from(bytes).unwrap_or(u32::MAX),
                        1,
                        Dtype::U8,
                    ),
                    indptr: i32s(
                        handles,
                        buffer,
                        indptr,
                        u32::try_from(mask.indptr.len()).unwrap_or(u32::MAX),
                    )?,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn input_seat(&self, id: u32) -> Option<(&Buffer, u64)> {
        self.seats
            .iter()
            .find(|at| at.seat.id == id)
            .map(|at| (&self.store, at.at))
    }

    #[must_use]
    pub fn voxel_payload(&self) -> Option<(&Buffer, u64)> {
        self.voxels.map(|at| (&self.store, at.payload))
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch.map(|at| at.seat.dtype)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// The group mask buffer, sized at the load.
    #[must_use]
    pub fn group_mask_bytes(&self) -> u64 {
        self.group_masks.as_ref().map_or(0, Buffer::bytes)
    }

    pub fn write(&mut self, handles: &crate::device::Handles, fire: &Fire<'_>) -> Result<Handles> {
        let rows = fire.tokens.len() as u32;
        let width = self.bands.iter().sum::<u32>().max(1);
        let lanes = fire.slot_ids.len() as u32 / width;

        let valid = vec![1u8; rows as usize];

        self.store.write(self.tokens, bytes_of(fire.tokens))?;
        self.store.write(self.positions, bytes_of(fire.positions))?;
        if fire.windows.len() as u64 > self.window_ints {
            return Err(Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
            });
        }
        self.store.write(self.windows, bytes_of(fire.windows))?;
        self.store.write(self.row_valid, &valid)?;
        self.store.write(self.slot_ids, bytes_of(fire.slot_ids))?;
        self.store
            .write(self.request_of_token, bytes_of(fire.request_of_token))?;
        if fire.readout_rows.len() > rows as usize {
            return Err(Fault::Ceiling {
                what: "readout rows in one fire",
                need: fire.readout_rows.len() as u64,
                have: u64::from(rows),
            });
        }
        self.store
            .write(self.readout_rows, bytes_of(fire.readout_rows))?;
        let enabled = fire.mask.map(|staged| staged.enabled.as_slice());
        let mut readout_positions = Vec::with_capacity(fire.readout_rows.len());
        let mut readout_request_of_token = Vec::with_capacity(fire.readout_rows.len());
        let mut readout_mask_enabled = Vec::with_capacity(fire.readout_rows.len());
        for &row in fire.readout_rows {
            let at = usize::try_from(row)
                .ok()
                .filter(|&at| at < rows as usize)
                .ok_or(Fault::Ceiling {
                    what: "a readout row inside the fire's tokens",
                    need: u64::try_from(row).unwrap_or(u64::MAX).saturating_add(1),
                    have: u64::from(rows),
                })?;
            readout_positions.push(fire.positions[at]);
            readout_request_of_token.push(fire.request_of_token[at]);
            readout_mask_enabled.push(
                enabled
                    .and_then(|flags| flags.get(at))
                    .copied()
                    .unwrap_or(0),
            );
        }
        self.store
            .write(self.readout_positions, bytes_of(&readout_positions))?;
        self.store.write(
            self.readout_request_of_token,
            bytes_of(&readout_request_of_token),
        )?;
        self.store
            .write(self.readout_mask_enabled, &readout_mask_enabled)?;
        self.store
            .write(self.slot_of_row, bytes_of(fire.slot_of_row))?;
        let rs_tables = match (fire.rs_replay, fire.rs_commit) {
            (Some(replay), Some(commit)) => {
                if replay.len() != lanes as usize || commit.len() != lanes as usize {
                    return Err(Fault::Ceiling {
                        what: "recurrent seat tables, one entry per lane",
                        need: replay.len().max(commit.len()) as u64,
                        have: u64::from(lanes),
                    });
                }
                self.store.write(self.rs_replay, bytes_of(replay))?;
                self.store.write(self.rs_commit, bytes_of(commit))?;
                true
            }
            _ => false,
        };
        if let Some(snaps) = fire.rs_snaps {
            if snaps.len() != lanes as usize * SNAP_INTS as usize {
                return Err(Fault::Ceiling {
                    what: "recurrent snapshot tables, one row per lane",
                    need: snaps.len() as u64,
                    have: u64::from(lanes) * u64::from(SNAP_INTS),
                });
            }
            self.store.write(self.rs_snaps, bytes_of(snaps))?;
        }
        if let Some(pages) = fire.rs_pages {
            if pages.len() != lanes as usize * crate::rs::PAGE_INTS as usize {
                return Err(Fault::Ceiling {
                    what: "recurrent buffer page tables, one row per lane",
                    need: pages.len() as u64,
                    have: u64::from(lanes) * u64::from(crate::rs::PAGE_INTS),
                });
            }
            self.store.write(self.rs_pages, bytes_of(pages))?;
        }

        let adapter_routes = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                self.store.write(self.adapter_routes, bytes_of(routes))?;
                Some(routes.len() as u32)
            }
        };

        let stride = fire.mask.map_or(0, |staged| staged.stride);
        if u64::from(stride) > u64::from(self.mask_stride) {
            return Err(Fault::Ceiling {
                what: "key positions in one mask row",
                need: u64::from(stride),
                have: u64::from(self.mask_stride),
            });
        }
        match fire.mask {
            None => {
                self.store.zero_span(self.mask_enabled, u64::from(rows))?;
            }
            Some(staged) => {
                if staged.bytes.len() as u64 > self.mask_plane_bytes {
                    return Err(Fault::Ceiling {
                        what: "mask plane bytes",
                        need: staged.bytes.len() as u64,
                        have: self.mask_plane_bytes,
                    });
                }
                self.store.write(self.mask_planes, &staged.bytes)?;
                self.store.write(self.mask_enabled, &staged.enabled)?;
            }
        }

        let patches = match (fire.patches, self.patch) {
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
                self.store.write(at.payload, staged.payload)?;
                self.store.write(at.segments, bytes_of(staged.segments))?;
                self.store.write(at.routes, bytes_of(staged.routes))?;
                self.store.write(at.positions, bytes_of(staged.positions))?;
                self.store.write(at.grids, bytes_of(staged.grids))?;
                let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
                let element = poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(1);
                let width = u32::try_from(seat.row_bytes.checked_div(element).unwrap_or(0))
                    .unwrap_or(u32::MAX);
                Some(PatchHandles {
                    patches: Tensor::new(
                        handles.bind(&self.store, at.payload, staged.payload.len() as u64)?,
                        rows32,
                        width,
                        seat.dtype,
                    ),
                    segments: i32s(
                        handles,
                        &self.store,
                        at.segments,
                        staged.segments.len() as u32,
                    )?,
                    routes: i32s(handles, &self.store, at.routes, staged.routes.len() as u32)?,
                    positions: Tensor::new(
                        handles.bind(
                            &self.store,
                            at.positions,
                            staged.positions.len() as u64 * 4,
                        )?,
                        rows32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                    grids: Tensor::new(
                        handles.bind(&self.store, at.grids, staged.grids.len() as u64 * 4)?,
                        staged.grids.len() as u32 / AXES as u32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                })
            }
        };

        let mrope_positions = match (fire.mrope_positions, self.mrope) {
            (None, _) | (_, None) => None,
            (Some(triples), Some(at)) => {
                if triples.len() as u64 > u64::from(rows) * AXES {
                    return Err(Fault::Ceiling {
                        what: "trunk rotation triples",
                        need: triples.len() as u64,
                        have: u64::from(rows) * AXES,
                    });
                }
                self.store.write(at, bytes_of(triples))?;
                Some(Tensor::new(
                    handles.bind(&self.store, at, triples.len() as u64 * 4)?,
                    triples.len() as u32 / AXES as u32,
                    AXES as u32,
                    Dtype::I32,
                ))
            }
        };

        let group_of_lane = if self.packings.is_empty() {
            None
        } else {
            if fire.group_of_lane.len() as u32 > lanes {
                return Err(Fault::Ceiling {
                    what: "group-of-lane entries",
                    need: fire.group_of_lane.len() as u64,
                    have: u64::from(lanes),
                });
            }
            self.store
                .write(self.group_of_lane, bytes_of(fire.group_of_lane))?;
            Some(i32s(handles, &self.store, self.group_of_lane, lanes)?)
        };
        if fire.packings.len() != self.packings.len() {
            return Err(Fault::Ceiling {
                what: "packing tables against the selections this load carved",
                need: fire.packings.len() as u64,
                have: self.packings.len() as u64,
            });
        }
        let mut packings = Vec::with_capacity(self.packings.len());
        let mut scratch: Vec<i32> = Vec::new();
        for (at, staged) in self.packings.iter().zip(fire.packings) {
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
            self.store.write(at.group_indptr, bytes_of(&scratch))?;
            csr(&mut scratch, staged.lane_indptr, lanes + 1);
            self.store.write(at.lane_indptr, bytes_of(&scratch))?;
            rows_table(&mut scratch, staged.permutation, rows);
            self.store.write(at.permutation, bytes_of(&scratch))?;
            packings.push(PackingHandles {
                group_indptr: i32s(handles, &self.store, at.group_indptr, lanes + 1)?,
                lane_indptr: i32s(handles, &self.store, at.lane_indptr, lanes + 1)?,
                permutation: i32s(handles, &self.store, at.permutation, rows)?,
            });
        }

        let mut inputs = Vec::with_capacity(self.seats.len());
        for at in &self.seats {
            let seat = at.seat;
            let carved = if seat.per_lane() { lanes } else { rows };
            inputs.push((
                seat.id,
                Tensor::new(
                    handles.bind(&self.store, at.at, at.bytes)?,
                    carved,
                    seat.width,
                    seat.dtype,
                ),
            ));
        }

        let voxels = match (fire.voxels, self.voxels) {
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
                self.store.write(at.grid, bytes_of(staged.grid))?;
                self.store.write(at.slots, bytes_of(staged.slots))?;
                if !staged.token_grid.is_empty() {
                    self.store
                        .write(at.token_grid, bytes_of(staged.token_grid))?;
                }
                if !staged.payload.is_empty() {
                    self.store.write(at.payload, staged.payload)?;
                }
                let clips32 = u32::try_from(clips).unwrap_or(u32::MAX);
                let width = u32::try_from(seat.channels).unwrap_or(u32::MAX);
                let voxel_rows = u32::try_from(seat.rows).unwrap_or(u32::MAX);
                Some(VoxelHandles {
                    payload: Tensor::new(
                        handles.bind(
                            &self.store,
                            at.payload,
                            seat.rows * seat.channels * element,
                        )?,
                        voxel_rows,
                        width,
                        seat.dtype,
                    ),
                    grid: Tensor::new(
                        handles.bind(&self.store, at.grid, clips * 4 * 4)?,
                        clips32,
                        4,
                        Dtype::I32,
                    ),
                    token_grid: seat.token_patch.is_some().then(|| {
                        Tensor::new(
                            handles
                                .bind(&self.store, at.token_grid, clips * 4 * 4)
                                .expect("a handle"),
                            clips32,
                            4,
                            Dtype::I32,
                        )
                    }),
                    slots: i32s(handles, &self.store, at.slots, clips32)?,
                })
            }
        };

        let mut page_tables = Vec::with_capacity(self.page_tables.len());
        for (at, geometry) in self.page_tables.iter().zip(fire.page_tables) {
            self.store.write(at.indptr, bytes_of(&geometry.indptr))?;
            self.store.write(at.indices, bytes_of(&geometry.indices))?;
            self.store
                .write(at.last_page_len, bytes_of(&geometry.last_page_len))?;
            self.store.write(at.kv_len, bytes_of(&geometry.kv_len))?;
            self.store
                .write(at.write_page, bytes_of(&geometry.write_page))?;
            self.store
                .write(at.write_offset, bytes_of(&geometry.write_offset))?;
            page_tables.push(TableHandles {
                indptr: i32s(handles, &self.store, at.indptr, lanes + 1)?,
                indices: i32s(
                    handles,
                    &self.store,
                    at.indices,
                    geometry.indices.len() as u32,
                )?,
                last_page_len: i32s(handles, &self.store, at.last_page_len, lanes)?,
                kv_len: i32s(handles, &self.store, at.kv_len, lanes)?,
                write_page: u32s(handles, &self.store, at.write_page, rows)?,
                write_offset: u32s(handles, &self.store, at.write_offset, rows)?,
            });
        }

        Ok(Handles {
            tokens: i32s(handles, &self.store, self.tokens, rows)?,
            positions: i32s(handles, &self.store, self.positions, rows)?,
            windows: handles.bind(&self.store, self.windows, fire.windows.len() as u64 * 4)?,
            page_tables,
            slot_ids: i32s(handles, &self.store, self.slot_ids, lanes)?,
            slot_of_row: i32s(handles, &self.store, self.slot_of_row, rows)?,
            lane_bands: bands(handles, &self.store, self.slot_ids, &self.bands, lanes)?,
            row_bands: bands(handles, &self.store, self.slot_of_row, &self.bands, rows)?,
            adapter_routes: match adapter_routes {
                None => None,
                Some(rows) => Some(i32s(handles, &self.store, self.adapter_routes, rows)?),
            },
            row_valid: Tensor::new(
                handles.bind(&self.store, self.row_valid, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            request_of_token: i32s(handles, &self.store, self.request_of_token, rows)?,
            readout_rows: i32s(
                handles,
                &self.store,
                self.readout_rows,
                fire.readout_rows.len() as u32,
            )?,
            readout_positions: i32s(
                handles,
                &self.store,
                self.readout_positions,
                fire.readout_rows.len() as u32,
            )?,
            readout_request_of_token: i32s(
                handles,
                &self.store,
                self.readout_request_of_token,
                fire.readout_rows.len() as u32,
            )?,
            readout_mask_enabled: Tensor::new(
                handles.bind(
                    &self.store,
                    self.readout_mask_enabled,
                    fire.readout_rows.len() as u64,
                )?,
                fire.readout_rows.len() as u32,
                1,
                Dtype::U8,
            ),
            group_of_lane,
            packings,
            inputs,
            voxels,
            mask: Tensor::new(
                handles.bind(
                    &self.store,
                    self.mask_planes,
                    u64::from(rows) * u64::from(stride),
                )?,
                rows,
                stride.max(1),
                Dtype::U8,
            ),
            mask_enabled: Tensor::new(
                handles.bind(&self.store, self.mask_enabled, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            mask_stride: stride,
            patches,
            mrope_positions,
            rs_replay: if rs_tables {
                Some(i32s(handles, &self.store, self.rs_replay, lanes)?)
            } else {
                None
            },
            rs_commit: if rs_tables {
                Some(i32s(handles, &self.store, self.rs_commit, lanes)?)
            } else {
                None
            },
            rs_snap_bands: match fire.rs_snaps {
                Some(snaps) => {
                    let max = kernels_metal::ssm::RS_SNAPSHOTS;
                    let per_lane: u32 = self.bands.iter().map(|&w| max * (1 + w)).sum();
                    let lanes = snaps.len() as u32 / per_lane.max(1);
                    let mut at = self.rs_snaps;
                    self.bands
                        .iter()
                        .map(|&w| {
                            let ints = lanes * max * (1 + w);
                            let buf = handles.bind(&self.store, at, u64::from(ints.max(1)) * 4)?;
                            at += u64::from(ints) * 4;
                            Ok(Tensor::new(buf, ints.max(1), 1, Dtype::I32))
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                None => Vec::new(),
            },
            rs_snaps: match fire.rs_snaps {
                Some(snaps) => Some(i32s(
                    handles,
                    &self.store,
                    self.rs_snaps,
                    snaps.len() as u32,
                )?),
                None => None,
            },
            rs_pages: match fire.rs_pages {
                Some(pages) => Some(i32s(
                    handles,
                    &self.store,
                    self.rs_pages,
                    pages.len() as u32,
                )?),
                None => None,
            },
        })
    }

    #[must_use]
    pub fn seats(
        &self,
        _handles: &crate::device::Handles,
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
            lane_bands: views.lane_bands.clone(),
            row_bands: views.row_bands.clone(),
        }
    }
}

/// The ints of one lane's row in the snapshot table: its rows, then its slots.
const SNAP_INTS: u32 = 2 * kernels_metal::ssm::RS_SNAPSHOTS;

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

fn i32s(handles: &crate::device::Handles, store: &Buffer, at: u64, rows: u32) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
}

/// Each band's table, `[count, width]` ints one after another from `at`.
fn bands(
    handles: &crate::device::Handles,
    store: &Buffer,
    mut at: u64,
    widths: &[u32],
    count: u32,
) -> Result<Vec<Tensor>> {
    widths
        .iter()
        .map(|&width| {
            let bytes = u64::from(count) * u64::from(width) * 4;
            let buf = handles.bind(store, at, bytes.max(4))?;
            at += bytes;
            Ok(Tensor::new(buf, count, width, Dtype::I32))
        })
        .collect()
}

fn u32s(handles: &crate::device::Handles, store: &Buffer, at: u64, rows: u32) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::U32))
}

fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy` with no padding/niche; all bytes are init and
    // readable as `u8`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
