use kernels_vulkan::Tensor;
use model_compiler::Budget;
use model_ir::Dtype;

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};
use crate::store::SpaceSeat;
use crate::store::kv::{Geometry, Paging};

const ALIGN: u64 = 256;

const AXES: u64 = 3;

#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    pub rows: u64,

    pub row_bytes: u64,

    pub images: u64,

    pub dtype: Dtype,

    pub embed_taps: u64,

    pub embed_weights: bool,
}

#[derive(Debug, Clone, Copy)]
struct PatchAt {
    payload: u64,
    segments: u64,
    routes: u64,
    positions: u64,
    embed_rows: u64,
    embed_weights: u64,
    seat: PatchSeat,
}

#[derive(Debug, Clone, Copy)]
struct SpaceAt {
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

    pub spaces: Vec<SpaceHandles>,

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
}

#[derive(Debug, Clone, Copy)]
pub struct PatchHandles {
    pub patches: Tensor,

    pub segments: Tensor,

    pub routes: Tensor,

    pub positions: Tensor,

    pub embed_rows: Option<Tensor>,

    pub embed_weights: Option<Tensor>,
}

#[derive(Debug, Clone, Copy)]
pub struct SpaceHandles {
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

    pub spaces: &'a [Geometry],

    pub mask: Option<&'a crate::mask::Staged>,

    pub patches: Option<PatchFire<'a>>,

    pub mrope_positions: Option<&'a [i32]>,

    pub rs_replay: Option<&'a [i32]>,
    pub rs_commit: Option<&'a [i32]>,
}

#[derive(Debug, Clone, Copy)]
pub struct PatchFire<'a> {
    pub payload: &'a [u8],

    pub segments: &'a [i32],

    pub routes: &'a [i32],

    pub positions: &'a [i32],

    pub embed_rows: &'a [i32],

    pub embed_weights: &'a [f32],
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
    adapter_routes: u64,
    request_of_token: u64,
    mask_planes: u64,
    mask_plane_bytes: u64,
    mask_enabled: u64,

    mask_stride: u32,
    spaces: Vec<SpaceAt>,

    patch: Option<PatchAt>,

    mrope: Option<u64>,

    rs_replay: u64,
    rs_commit: u64,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: &Context,
        budget: &Budget,
        paging: Paging,
        spaces: usize,
        classes: usize,
        gathered: usize,
        patch: Option<PatchSeat>,
        mrope: bool,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);

        let per_gathered = 3 * rows + spaces as u64 * (2 * lanes + (lanes + 1) + pages);
        let window_ints =
            (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1) + gathered as u64 * per_gathered;

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
        let slot_ids = take(lanes * 4);

        let slot_of_row = take(rows * 4);

        let adapter_routes = take(rows * 4);
        let request_of_token = take(rows * 4);

        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_stride = u32::try_from(context).unwrap_or(u32::MAX);
        let mask_plane_bytes = rows * context;
        let mask_planes = take(mask_plane_bytes);

        let mask_enabled = take(rows);
        let spaces: Vec<SpaceAt> = (0..spaces)
            .map(|_| SpaceAt {
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
            embed_rows: take(seat.rows * seat.embed_taps * 4),
            embed_weights: if seat.embed_weights {
                take(seat.rows * seat.embed_taps * 4)
            } else {
                0
            },
            seat,
        });
        let mrope = mrope.then(|| take(rows * AXES * 4));
        let rs_replay = take(lanes * 4);
        let rs_commit = take(lanes * 4);
        let total = at;

        let store = Buffer::host(device, total)?;
        Ok(Inputs {
            store,
            tokens,
            positions,
            windows,
            window_ints,
            row_valid,
            slot_ids,
            slot_of_row,
            adapter_routes,
            request_of_token,
            mask_planes,
            mask_plane_bytes,
            mask_enabled,
            mask_stride,
            spaces,
            patch,
            mrope,
            rs_replay,
            rs_commit,
        })
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch.map(|at| at.seat.dtype)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub fn write(&mut self, handles: &crate::device::Handles, fire: &Fire<'_>) -> Result<Handles> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;

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
                    (
                        "patch table rows",
                        staged.embed_rows.len() as u64,
                        seat.rows * seat.embed_taps,
                    ),
                    (
                        "patch table weights",
                        staged.embed_weights.len() as u64,
                        if seat.embed_weights {
                            seat.rows * seat.embed_taps
                        } else {
                            0
                        },
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
                self.store.write(at.payload, staged.payload)?;
                self.store.write(at.segments, bytes_of(staged.segments))?;
                self.store.write(at.routes, bytes_of(staged.routes))?;
                self.store.write(at.positions, bytes_of(staged.positions))?;
                if !staged.embed_rows.is_empty() {
                    self.store
                        .write(at.embed_rows, bytes_of(staged.embed_rows))?;
                }
                if !staged.embed_weights.is_empty() {
                    self.store
                        .write(at.embed_weights, f32_bytes_of(staged.embed_weights))?;
                }
                let taps = u32::try_from(seat.embed_taps).unwrap_or(u32::MAX).max(1);
                let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
                let element = model_compiler::arena::elem_bytes(seat.dtype).unwrap_or(1);
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
                    embed_rows: if staged.embed_rows.is_empty() {
                        None
                    } else {
                        Some(Tensor::new(
                            handles.bind(
                                &self.store,
                                at.embed_rows,
                                staged.embed_rows.len() as u64 * 4,
                            )?,
                            staged.embed_rows.len() as u32 / taps,
                            taps,
                            Dtype::I32,
                        ))
                    },
                    embed_weights: if staged.embed_weights.is_empty() {
                        None
                    } else {
                        Some(Tensor::new(
                            handles.bind(
                                &self.store,
                                at.embed_weights,
                                staged.embed_weights.len() as u64 * 4,
                            )?,
                            staged.embed_weights.len() as u32 / taps,
                            taps,
                            Dtype::F32,
                        ))
                    },
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

        let mut spaces = Vec::with_capacity(self.spaces.len());
        for (at, geometry) in self.spaces.iter().zip(fire.spaces) {
            self.store.write(at.indptr, bytes_of(&geometry.indptr))?;
            self.store.write(at.indices, bytes_of(&geometry.indices))?;
            self.store
                .write(at.last_page_len, bytes_of(&geometry.last_page_len))?;
            self.store.write(at.kv_len, bytes_of(&geometry.kv_len))?;
            self.store
                .write(at.write_page, bytes_of(&geometry.write_page))?;
            self.store
                .write(at.write_offset, bytes_of(&geometry.write_offset))?;
            spaces.push(SpaceHandles {
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
            spaces,
            slot_ids: i32s(handles, &self.store, self.slot_ids, lanes)?,
            slot_of_row: i32s(handles, &self.store, self.slot_of_row, rows)?,
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
            spaces: views
                .spaces
                .iter()
                .map(|space| SpaceSeat {
                    page_indptr: space.indptr,
                    page_indices: space.indices,
                    last_page_lens: space.last_page_len,
                    row_valid: views.row_valid,
                })
                .collect(),
            slot_ids: views.slot_ids,
            slot_of_row: views.slot_of_row,
        }
    }
}

fn i32s(handles: &crate::device::Handles, store: &Buffer, at: u64, rows: u32) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
}

fn u32s(handles: &crate::device::Handles, store: &Buffer, at: u64, rows: u32) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::U32))
}

fn f32_bytes_of(values: &[f32]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn bytes_of(values: &[i32]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
