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

    /// Each row's cache row: attention's causal bound (and its window's).
    pub positions: Tensor,

    /// Each row's stated position, what `RuntimeInput::Positions` reads:
    /// the rotation. With mrope the rotation rides its own table and this
    /// holds the cache row.
    pub rotations: Tensor,

    pub readout_rows: Tensor,

    pub windows: u32,

    pub page_tables: Vec<TableHandles>,

    pub slot_ids: Tensor,

    pub slot_of_row: Tensor,

    pub row_valid: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub request_of_token: Tensor,

    /// The mask plane, bound whole whatever the fire stages: a recorded
    /// fire binds it as one with no mask does.
    pub mask: Tensor,

    /// Two u32 a row (`mask::Staged`'s words): the key the row's columns
    /// start at, `mask::NONE` where it states no mask, and its byte offset.
    pub mask_words: Tensor,

    pub patches: Option<PatchHandles>,

    pub mrope_positions: Option<Tensor>,

    pub rs_replay: Option<Tensor>,
    pub rs_commit: Option<Tensor>,
    /// The fire's staged rs table (`kernels_slang::ssm::Carry`), bound at
    /// its room whatever the fire stages: one view for every fire.
    pub rs_carry: Option<Tensor>,
    /// The fire's beam groups' folds' rs table and slot table (§11.3).
    pub rs_folds: Option<(Tensor, Tensor)>,

    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<PackingHandles>,
    pub inputs: Vec<(u32, Tensor)>,
    pub voxels: Option<VoxelHandles>,
    /// The fire's capture table, where a lane of it captures scores.
    pub capture: Option<Tensor>,
    /// Where the load has windowed rows (CONTRACT-BC 19, 21): four ints a
    /// lane (folded, base, first, folded after; `kernels_slang::Ring::lanes`), and the
    /// fold's job table, bound whole (`kernels_slang::attn::ring::fold`).
    pub ring_lanes: Tensor,
    /// The fold's table (its rings' entries, the snapshots' jobs, the
    /// fold's), where the fold's count lies, the most fold jobs and the
    /// most snapshot jobs it holds.
    pub ring_jobs: Option<(Tensor, u32, u32, u32)>,
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

    pub rotations: &'a [i32],

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
    pub rs_carry: Option<&'a [i32]>,
    /// The beam groups' folds' rs table and slot table, where it folds any.
    pub rs_folds: Option<(&'a [i32], &'a [i32])>,

    pub group_of_lane: &'a [i32],
    pub packings: &'a [PackingFire<'a>],
    pub voxels: Option<VoxelFire<'a>>,
    /// The capture table (`Scored::table`), empty where no lane captures.
    pub capture: &'a [i32],
    /// The windowed rows' lane table and fold jobs ([`crate::ring::Staged`]);
    /// empty where the load has none.
    pub ring_lanes: &'a [i32],
    pub ring_jobs: &'a [i32],
    /// The snapshots' jobs ([`crate::ring::Staged::snaps`]).
    pub ring_snaps: &'a [i32],
    /// Each ring's geometry the fold table opens with
    /// (`Pools::ring_layers`), as many entries as the load has rings.
    pub ring_layers: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct PatchFire<'a> {
    pub payload: &'a [u8],

    pub segments: &'a [i32],

    pub routes: &'a [i32],

    pub positions: &'a [i32],

    pub grids: &'a [i32],
}

/// Where a fire's inputs lie in an arm's store, and the store's bytes: a
/// function of the load's fires alone ([`Laid::lay`]), so a load weighs the
/// store against its largest binding before it lays one.
#[derive(Debug, Clone)]
pub struct Laid {
    /// Each input's room, by where it lies: what a dispatch binds over it,
    /// whatever rows a fire brings, so one view of it serves every fire.
    rooms: std::collections::BTreeMap<u64, u64>,
    tokens: u64,
    positions: u64,
    rotations: u64,
    readout_rows: u64,
    windows: u64,
    window_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    slot_of_row: u64,
    /// The units a slot entry holds: a record's pieces.
    pieces: u32,
    adapter_routes: u64,
    request_of_token: u64,
    mask_planes: u64,
    mask_plane_bytes: u64,
    mask_words: u64,
    /// The leading rows whose words state no mask as the store holds them:
    /// a fire stating none rewrites only past them.
    words_none: u32,
    page_tables: Vec<TableAt>,

    patch: Option<PatchAt>,

    mrope: Option<u64>,

    rs_replay: u64,
    rs_commit: u64,
    /// The rs table's place and ints, where the load buffers rs rows.
    rs_carry: Option<(u64, u64)>,
    /// The beam groups' folds' rs table and slot table: each place and its
    /// ints.
    rs_folds: Option<((u64, u64), (u64, u64))>,

    group_of_lane: u64,
    packings: Vec<PackingAt>,
    seats: Vec<SeatAt>,
    voxels: Option<VoxelAt>,
    /// The capture table's ints, laid where the trace captures scores:
    /// `TABLE_INTS` for each of a fire's lanes at most.
    capture: Option<(u64, u64)>,
    /// A fire's beam groups' records and what they read
    /// ([`crate::beam`]).
    beams: Option<BeamRoom>,
    /// The windowed rows' lane table, the fold's table (its rings'
    /// entries, then its jobs) and the most jobs it holds, and its rings,
    /// where the load has windowed rows.
    ring: Option<(u64, u64, u32, u32, u32)>,
    /// The store's bytes.
    total: u64,
}

/// What a load lays a fire's beam groups for (`crate::beam`): the kv
/// planes a page holds and the rs planes an rs page holds, and the frames
/// in flight a freed page waits out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamFacts {
    pub planes: u32,
    pub rs_planes: u32,
    pub frames: u32,
}

/// Where a fire's beam groups' parts lie in the store, each its first u32
/// word (from the store's first byte) and the words laid for it: a fire's
/// most, its lanes all beams of groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BeamRoom {
    /// Each lane's parent (copied from its group's cell ahead of the step),
    /// and each lane's fault word.
    pub parents: (u32, u32),
    pub faults: (u32, u32),
    /// Each group's common slot (§11.3), right past the fault words: a
    /// fire's readout carries both as one run.
    pub commons: (u32, u32),
    /// The records: groups' steps, lanes' list runs (a table each), their
    /// rs list runs, their kv copies and rs copies.
    pub steps: (u32, u32),
    pub lists: (u32, u32),
    pub rs_lists: (u32, u32),
    pub copies: (u32, u32),
    pub rs_copies: (u32, u32),
    /// The kv planes, then the rs planes: two words a plane.
    pub planes: (u32, u32),
    /// Each group's pool pages' ids, and its set's rs entries' units.
    pub trans: (u32, u32),
    pub rs_bases: (u32, u32),
}

#[derive(Debug)]
pub struct Inputs<B: Backend> {
    store: B::Buffer,
    laid: Laid,
    /// Staged apart from the store: their size is the groups', not the load's.
    group_masks: Option<B::Buffer>,
    /// Smaller ones it outgrew, kept: an earlier fire may still read them.
    outgrown: Vec<B::Buffer>,
}


#[derive(Debug, Clone, Copy)]
pub struct GroupMaskHandles {
    pub q: poem_ir::Selection,
    pub kv: poem_ir::Selection,
    pub bits: Tensor,
    pub indptr: Tensor,
}

impl Laid {
    /// Lays a fire's inputs for fires of `budget`: where each lies in the
    /// store, and the store's bytes; no one input past `binding`, the most
    /// one reservation of the device binds.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn lay(
        budget: &Budget,
        paging: Paging,
        page_tables: usize,
        table_pages: u32,
        classes: usize,
        gathered: usize,
        patch: Option<PatchSeat>,
        mrope: bool,
        seats: &[Seat],
        selections: usize,
        voxel: Option<VoxelSeat>,
        pieces: u32,
        capture: bool,
        rs_page_tokens: Option<u32>,
        rings: Option<(u32, u32)>,
        binding: u64,
        beams: Option<BeamFacts>,
    ) -> Laid {
        let pieces = pieces.max(1);
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        // A lane's table holds as many pages as its narrowest list takes.
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot.max(table_pages));

        // A gathered window's rows, positions, rotations and owners an int a
        // row, its mask entries two.
        let per_gathered = 6 * rows + page_tables as u64 * (2 * lanes + (lanes + 1) + pages);
        let window_ints =
            (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1) + gathered as u64 * per_gathered;

        let mut at = 0u64;
        let mut rooms = std::collections::BTreeMap::new();
        let mut take = |bytes: u64| {
            let here = at;
            // Its room is its bytes in whole words, within the binding the
            // load weighed it against; an input of no bytes lies where the
            // next one does, and has its room.
            let held: &mut u64 = rooms.entry(here).or_default();
            *held = (*held).max(bytes.next_multiple_of(4));
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let rotations = take(rows * 4);
        let readout_rows = take(rows * 4);
        let windows = take(window_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * u64::from(pieces) * 4);

        let slot_of_row = take(rows * u64::from(pieces) * 4);

        let adapter_routes = take(rows * 4);
        let request_of_token = take(rows * 4);

        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        // A row's columns run from its lane's boundary to its own key, at
        // most the context; the plane is one binding, its offsets u32.
        let mask_plane_bytes = (rows * context)
            .min(binding)
            .min(u64::from(u32::MAX))
            / 4
            * 4;
        let mask_planes = take(mask_plane_bytes.max(4));

        let mask_words = take(rows * 8);
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
        let mrope = mrope.then(|| take(rows * AXES * 4));
        let rs_replay = take(lanes * 4);
        let rs_commit = take(lanes * 4);
        // A fire's lanes carry no more buffered rows than its widest
        // bucket's tokens (`engine::fire::rung_holds`, refused past at the
        // stage): the table holds a header a lane, those rows' tokens, and
        // the pages they and the lanes' own rows lie in.
        let rs_carry = rs_page_tokens.map(|tokens| {
            let ints = rs_carry_ints(lanes, rows, tokens);
            (take(ints * 4), ints)
        });
        // A beam group's fold a lane of a table of its own, its rows no more
        // than a fire carries; its record a lane's.
        let rs_folds = rs_page_tokens.map(|tokens| {
            let ints = rs_carry_ints(lanes, rows, tokens);
            let table = (take(ints * 4), ints);
            let slots = lanes * u64::from(pieces);
            (table, (take(slots * 4), slots))
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
        let capture = capture.then(|| {
            let ints = lanes * u64::from(kernels_slang::attn::score::TABLE_INTS);
            (take(ints * 4), ints)
        });
        // A fire's beam groups: each a group of at least one of its lanes,
        // every lane a beam. A group's pool pages staged are at most
        // `B (c + 2) + B F + 1` (BEAM-PAGE-LISTS §3), `c` its column, at
        // most a lane's pages; its set's rs entries cover its pages' slots.
        let beams = beams.map(|facts| {
            use kernels_slang::runtime::beam as kb;
            let tables = page_tables.len() as u64;
            let frames = u64::from(facts.frames.max(1));
            let columns = u64::from(paging.pages_per_slot.max(table_pages)) + 2;
            // The set's rs entries a fire appends to its group's state
            // (§11.1, the state holds the rest): those of the pool pages its
            // step may hand out past what the state holds, a group's at most
            // a page a beam and the frames' and a fire's rows' worth.
            let rs_appends = rs_page_tokens.map_or(0, |tokens| {
                let per_page = u64::from(paging.page_size).div_ceil(u64::from(tokens.max(1)));
                lanes * (frames + 2) * (per_page + 1) + rows.div_ceil(u64::from(tokens.max(1)))
            });
            let mut word = |words: u64| {
                let at = take(words.max(1) * 4);
                (
                    u32::try_from(at / 4).unwrap_or(u32::MAX),
                    u32::try_from(words).unwrap_or(u32::MAX),
                )
            };
            let parents = word(lanes);
            let both = word(2 * lanes);
            let lanes32 = u32::try_from(lanes).unwrap_or(u32::MAX);
            BeamRoom {
                parents,
                faults: (both.0, lanes32),
                commons: (both.0 + lanes32, lanes32),
                steps: word(lanes * u64::from(kb::STEP_WORDS)),
                lists: word(lanes * tables * u64::from(kb::LIST_WORDS)),
                rs_lists: word(if rs_page_tokens.is_some() {
                    lanes * u64::from(kb::RS_LIST_WORDS)
                } else {
                    0
                }),
                copies: word(lanes * u64::from(kb::COPY_WORDS)),
                rs_copies: word(if rs_page_tokens.is_some() {
                    lanes * u64::from(kb::RS_COPY_WORDS)
                } else {
                    0
                }),
                planes: word(2 * u64::from(facts.planes + facts.rs_planes)),
                trans: word(lanes * (columns + frames) + lanes),
                rs_bases: word(rs_appends),
            }
        });
        let ring = rings.map(|(rings, ring_rows)| {
            let most = crate::ring::most_jobs(budget.max_tokens, budget.max_tokens);
            let snaps =
                crate::ring::most_snapshot_jobs(budget.max_lanes, budget.max_tokens, ring_rows);
            // A lane's entry, and its snapshots' records'.
            let entries = lanes * (1 + u64::from(kernels_slang::ssm::RS_SNAPSHOTS));
            let lanes_at = take(entries * (crate::ring::LANE_INTS as u64 + u64::from(pieces)) * 4);
            let layer_ints = u64::from(rings) * u64::from(kernels_slang::attn::ring::LAYER_INTS);
            let jobs_at =
                take((layer_ints + 1 + 4 * u64::from(snaps) + 1 + 4 * u64::from(most)) * 4);
            (lanes_at, jobs_at, most, rings, snaps)
        });
        let total = at;

        Laid {
            ring,
            rooms,
            total,
            tokens,
            positions,
            rotations,
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
            mask_words,
            words_none: 0,
            page_tables,
            patch,
            mrope,
            rs_replay,
            rs_commit,
            rs_carry,
            rs_folds,
            group_of_lane,
            packings,
            seats,
            voxels,
            capture,
            beams,
        }
    }

    /// Where a fire's beam groups' parts lie, where the load lays them.
    #[must_use]
    pub fn beams(&self) -> Option<BeamRoom> {
        self.beams
    }

    /// Where page table `table`'s staged indices, write pages and write
    /// offsets lie: their first u32 words in the store.
    #[must_use]
    pub fn table_words(&self, table: usize) -> Option<(u32, u32, u32)> {
        let at = self.page_tables.get(table)?;
        let word = |bytes: u64| u32::try_from(bytes / 4).unwrap_or(u32::MAX);
        Some((word(at.indices), word(at.write_page), word(at.write_offset)))
    }

    /// Where the beam groups' folds' rs table lies: its first u32 word in
    /// the store.
    #[must_use]
    pub fn rs_folds_word(&self) -> Option<u32> {
        self.rs_folds
            .map(|((at, _), _)| u32::try_from(at / 4).unwrap_or(u32::MAX))
    }

    /// The ints the rs table and the folds' table each hold a fire.
    #[must_use]
    pub fn rs_table_ints(&self) -> (u64, u64) {
        (
            self.rs_carry.map_or(0, |(_, ints)| ints),
            self.rs_folds.map_or(0, |((_, ints), _)| ints),
        )
    }

    /// Where the rs table lies: its first u32 word in the store.
    #[must_use]
    pub fn rs_carry_word(&self) -> Option<u32> {
        self.rs_carry
            .map(|(at, _)| u32::try_from(at / 4).unwrap_or(u32::MAX))
    }

    /// The store's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.total
    }

    /// The room laid at byte `at` of the store, which a view of the input
    /// there binds.
    fn room(&self, at: u64) -> Result<u64> {
        self.rooms.get(&at).copied().ok_or_else(|| Fault::Unbound {
            what: format!("an input at byte {at} of the store, where none was laid"),
        })
    }
}

impl<B: Backend> Inputs<B> {
    /// An arm's inputs, its store laid at `laid`.
    pub fn reserve(device: &B::Context, laid: Laid) -> Result<Inputs<B>> {
        Ok(Inputs {
            store: B::Buffer::host(device, laid.total, crate::books::owners::INPUTS)?,
            laid,
            group_masks: None,
            outgrown: Vec::new(),
        })
    }

    pub fn stage_group_masks(
        &mut self,
        device: &B::Context,
        handles: &crate::handles::Handles<B>,
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
            let indptr = image.len();
            image.extend_from_slice(bytes_of(&mask.indptr));
            image.resize(image.len().next_multiple_of(ALIGN as usize), 0);
            let bits = image.len();
            // The kernel reads the bits a word at a time.
            image.extend_from_slice(&mask.bits);
            image.resize(image.len().next_multiple_of(ALIGN as usize), 0);
            placed.push((indptr, bits));
        }
        if self
            .group_masks
            .as_ref()
            .is_none_or(|held| held.bytes() < image.len() as u64)
        {
            let grown = B::Buffer::host(
                device,
                (image.len() as u64).next_power_of_two(),
                crate::books::owners::INPUTS,
            )?;
            if let Some(old) = self.group_masks.replace(grown) {
                self.outgrown.push(old);
            }
        }
        let buffer = self
            .group_masks
            .as_mut()
            .expect("the group mask buffer was just sized");
        buffer.write(0, &image)?;
        let buffer = &*buffer;
        masks
            .iter()
            .zip(placed)
            .map(|((q, kv, mask), (indptr, bits))| {
                let words = mask.bits.len().div_ceil(4).max(1) as u64;
                Ok(GroupMaskHandles {
                    q: *q,
                    kv: *kv,
                    bits: Tensor::new(
                        handles.bind(buffer, bits as u64, words * 4)?,
                        u32::try_from(words * 4).unwrap_or(u32::MAX),
                        1,
                        Dtype::U8,
                    ),
                    indptr: i32s(
                        handles,
                        buffer,
                        indptr as u64,
                        u32::try_from(mask.indptr.len()).unwrap_or(u32::MAX),
                        mask.indptr.len() as u64 * 4,
                    )?,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn voxel_payload(&self) -> Option<(&B::Buffer, u64)> {
        self.laid.voxels.map(|at| (&self.store, at.payload))
    }

    pub fn write_input(&mut self, at: u64, bytes: &[u8]) -> Result<()> {
        self.store.write(at, bytes)
    }

    #[must_use]
    pub fn store(&self) -> &B::Buffer {
        &self.store
    }

    /// Where its inputs lie.
    #[must_use]
    pub fn laid(&self) -> &Laid {
        &self.laid
    }

    /// Writes `words` from u32 word `at` of the store.
    pub fn write_words(&mut self, at: u32, words: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.store.write(u64::from(at) * 4, &bytes)
    }

    #[must_use]
    pub fn seat(&self, id: u32) -> Option<(&B::Buffer, u64)> {
        self.laid.seats
            .iter()
            .find(|at| at.seat.id == id)
            .map(|at| (&self.store, at.at))
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.laid.patch.map(|at| at.seat.dtype)
    }

    /// The store and the offset of row 0's token id, for a token the device
    /// writes in place of the host.
    #[must_use]
    pub fn token_seat(&self) -> (&B::Buffer, u64) {
        (&self.store, self.laid.tokens)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub fn write(
        &mut self,
        handles: &crate::handles::Handles<B>,
        fire: &Fire<'_>,
    ) -> Result<Handles> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32 / self.laid.pieces;

        // Whole words, its slot laid at `ALIGN`: a sub-word write would have
        // the device read its last word back first.
        let mut valid = vec![1u8; rows as usize];
        valid.resize(valid.len().next_multiple_of(4), 0);

        self.store.write(self.laid.tokens, bytes_of(fire.tokens))?;
        self.store.write(self.laid.positions, bytes_of(fire.positions))?;
        self.store.write(self.laid.rotations, bytes_of(fire.rotations))?;
        if fire.readout_rows.len() > rows as usize {
            return Err(Fault::Ceiling {
                what: "readout rows in one fire",
                need: fire.readout_rows.len() as u64,
                have: u64::from(rows),
            });
        }
        self.store
            .write(self.laid.readout_rows, bytes_of(fire.readout_rows))?;
        if fire.windows.len() as u64 > self.laid.window_ints {
            return Err(Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.laid.window_ints,
            });
        }
        self.store.write(self.laid.windows, bytes_of(fire.windows))?;
        self.store.write(self.laid.row_valid, &valid)?;
        self.store.write(self.laid.slot_ids, bytes_of(fire.slot_ids))?;
        self.store
            .write(self.laid.request_of_token, bytes_of(fire.request_of_token))?;
        self.store
            .write(self.laid.slot_of_row, bytes_of(fire.slot_of_row))?;
        let capture = if fire.capture.is_empty() {
            None
        } else {
            let Some((at, ints)) = self.laid.capture else {
                return Err(Fault::Unbound {
                    what: "the capture table, which this load laid no room for: its trace \
                           captures no scores"
                        .to_string(),
                });
            };
            if fire.capture.len() as u64 > ints {
                return Err(Fault::Ceiling {
                    what: "capture table ints in one fire",
                    need: fire.capture.len() as u64,
                    have: ints,
                });
            }
            self.store.write(at, bytes_of(fire.capture))?;
            Some(i32s(
                handles,
                &self.store,
                at,
                u32::try_from(fire.capture.len()).unwrap_or(u32::MAX),
                self.laid.room(at)?,
            )?)
        };
        let ring_jobs = match self.laid.ring {
            Some((lanes_at, jobs_at, most, rings, snaps)) => {
                let layer_ints = rings as usize * kernels_slang::attn::ring::LAYER_INTS as usize;
                let entry_ints = (crate::ring::LANE_INTS as u32 + self.laid.pieces) as usize;
                let entries = lanes as usize * (1 + kernels_slang::ssm::RS_SNAPSHOTS as usize);
                if fire.ring_lanes.len() > entry_ints * entries
                    || fire.ring_jobs.len() > 1 + 4 * most as usize
                    || fire.ring_snaps.len() > 1 + 4 * snaps as usize
                    || fire.ring_layers.len() != layer_ints
                {
                    return Err(Fault::Ceiling {
                        what: "ints of a fire's windowed lane and fold tables",
                        need: (fire.ring_lanes.len()
                            + fire.ring_jobs.len()
                            + fire.ring_snaps.len()
                            + fire.ring_layers.len()) as u64,
                        have: u64::from(crate::ring::LANE_INTS as u32 + self.laid.pieces) * u64::from(lanes)
                            + layer_ints as u64
                            + 2
                            + 4 * u64::from(snaps)
                            + 4 * u64::from(most),
                    });
                }
                self.store.write(lanes_at, bytes_of(fire.ring_lanes))?;
                let none = [0i32];
                let jobs = if fire.ring_jobs.is_empty() {
                    &none[..]
                } else {
                    fire.ring_jobs
                };
                let snap_jobs = if fire.ring_snaps.is_empty() {
                    &none[..]
                } else {
                    fire.ring_snaps
                };
                // The rings' entries, the snapshots' jobs, then the fold's.
                let fold_at = layer_ints + 1 + 4 * snaps as usize;
                self.store.write(jobs_at, bytes_of(fire.ring_layers))?;
                self.store.write(jobs_at + 4 * layer_ints as u64, bytes_of(snap_jobs))?;
                self.store.write(jobs_at + 4 * fold_at as u64, bytes_of(jobs))?;
                let room = self.laid.room(jobs_at)?;
                Some((
                    i32s(
                        handles,
                        &self.store,
                        jobs_at,
                        u32::try_from(room / 4).unwrap_or(u32::MAX),
                        room,
                    )?,
                    most,
                    u32::try_from(fold_at).unwrap_or(u32::MAX),
                    snaps,
                ))
            }
            None => None,
        };
        let ring_lanes = match self.laid.ring {
            Some((lanes_at, _, _, _, _)) => i32s(
                handles,
                &self.store,
                lanes_at,
                ((crate::ring::LANE_INTS as u32 + self.laid.pieces)
                    * lanes
                    * (1 + kernels_slang::ssm::RS_SNAPSHOTS))
                    .max(1),
                self.laid.room(lanes_at)?,
            )?,
            None => Tensor::new(kernels_slang::ABSENT, 0, 1, Dtype::I32),
        };
        let rs_tables = match (fire.rs_replay, fire.rs_commit) {
            (Some(replay), Some(commit)) => {
                if replay.len() != lanes as usize || commit.len() != lanes as usize {
                    return Err(Fault::Ceiling {
                        what: "recurrent seat tables, one entry per lane",
                        need: replay.len().max(commit.len()) as u64,
                        have: u64::from(lanes),
                    });
                }
                self.store.write(self.laid.rs_replay, bytes_of(replay))?;
                self.store.write(self.laid.rs_commit, bytes_of(commit))?;
                true
            }
            _ => false,
        };
        let rs_carry = match (fire.rs_carry, self.laid.rs_carry) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(Fault::Unbound {
                    what: "the rs table of a fire on a load that buffers no rs rows".to_string(),
                });
            }
            (Some(table), Some((at, ints))) => {
                if table.len() as u64 > ints {
                    return Err(Fault::Ceiling {
                        what: "ints of the staged rs table: lanes' headers, carried tokens \
                               and buffer pages",
                        need: table.len() as u64,
                        have: ints,
                    });
                }
                self.store.write(at, bytes_of(table))?;
                Some(i32s(
                    handles,
                    &self.store,
                    at,
                    u32::try_from(ints).unwrap_or(u32::MAX),
                    self.laid.room(at)?,
                )?)
            }
        };

        let rs_folds = match (fire.rs_folds, self.laid.rs_folds) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(Fault::Unbound {
                    what: "the beam folds' table of a fire on a load that buffers no rs rows"
                        .to_string(),
                });
            }
            (Some((table, slots)), Some(((table_at, table_ints), (slots_at, slots_ints)))) => {
                if table.len() as u64 > table_ints || slots.len() as u64 > slots_ints {
                    return Err(Fault::Ceiling {
                        what: "ints of the beam groups' folds' tables: their headers, rows, \
                               pages and records",
                        need: (table.len().max(slots.len())) as u64,
                        have: table_ints.min(slots_ints),
                    });
                }
                self.store.write(table_at, bytes_of(table))?;
                self.store.write(slots_at, bytes_of(slots))?;
                Some((
                    i32s(
                        handles,
                        &self.store,
                        table_at,
                        u32::try_from(table_ints).unwrap_or(u32::MAX),
                        self.laid.room(table_at)?,
                    )?,
                    i32s(
                        handles,
                        &self.store,
                        slots_at,
                        u32::try_from(slots_ints).unwrap_or(u32::MAX),
                        self.laid.room(slots_at)?,
                    )?,
                ))
            }
        };

        let adapter_routes = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                self.store.write(self.laid.adapter_routes, bytes_of(routes))?;
                Some(routes.len() as u32)
            }
        };

        match fire.mask {
            None => {
                if rows > self.laid.words_none {
                    let from = self.laid.words_none;
                    let none = crate::mask::none((rows - from) as usize);
                    self.store
                        .write(self.laid.mask_words + u64::from(from) * 8, words_of(&none))?;
                    self.laid.words_none = rows;
                }
            }
            Some(staged) => {
                if staged.bytes.len() as u64 > self.laid.mask_plane_bytes {
                    return Err(Fault::Ceiling {
                        what: "mask plane bytes",
                        need: staged.bytes.len() as u64,
                        have: self.laid.mask_plane_bytes,
                    });
                }
                // Whole words, as `valid` above: the plane's body as it is,
                // its last part-word padded with zeros no row's columns reach.
                let (body, tail) = staged.bytes.split_at(staged.bytes.len() & !3);
                self.store.write(self.laid.mask_planes, body)?;
                if !tail.is_empty() {
                    let mut word = [0u8; 4];
                    word[..tail.len()].copy_from_slice(tail);
                    self.store
                        .write(self.laid.mask_planes + body.len() as u64, &word)?;
                }
                self.store.write(self.laid.mask_words, words_of(&staged.words))?;
                self.laid.words_none = 0;
            }
        }

        let patches = match (fire.patches, self.laid.patch) {
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
                        handles.bind(&self.store, at.payload, self.laid.room(at.payload)?)?,
                        rows32,
                        width,
                        seat.dtype,
                    ),
                    segments: i32s(
                        handles,
                        &self.store,
                        at.segments,
                        staged.segments.len() as u32,
                        self.laid.room(at.segments)?,
                    )?,
                    routes: i32s(
                        handles,
                        &self.store,
                        at.routes,
                        staged.routes.len() as u32,
                        self.laid.room(at.routes)?,
                    )?,

                    positions: Tensor::new(
                        handles.bind(&self.store, at.positions, self.laid.room(at.positions)?)?,
                        rows32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                    grids: Tensor::new(
                        handles.bind(&self.store, at.grids, self.laid.room(at.grids)?)?,
                        staged.grids.len() as u32 / AXES as u32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                })
            }
        };

        let mrope_positions = match (fire.mrope_positions, self.laid.mrope) {
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
                    handles.bind(&self.store, at, self.laid.room(at)?)?,
                    triples.len() as u32 / AXES as u32,
                    AXES as u32,
                    Dtype::I32,
                ))
            }
        };

        let group_of_lane = if self.laid.packings.is_empty() {
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
                .write(self.laid.group_of_lane, bytes_of(fire.group_of_lane))?;
            Some(i32s(
                handles,
                &self.store,
                self.laid.group_of_lane,
                lanes,
                self.laid.room(self.laid.group_of_lane)?,
            )?)
        };
        if fire.packings.len() != self.laid.packings.len() {
            return Err(Fault::Ceiling {
                what: "packing tables against the selections this load carved",
                need: fire.packings.len() as u64,
                have: self.laid.packings.len() as u64,
            });
        }
        let mut packings = Vec::with_capacity(self.laid.packings.len());
        let mut scratch: Vec<i32> = Vec::new();
        for (at, staged) in self.laid.packings.iter().zip(fire.packings) {
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
                group_indptr: i32s(
                    handles,
                    &self.store,
                    at.group_indptr,
                    lanes + 1,
                    self.laid.room(at.group_indptr)?,
                )?,
                lane_indptr: i32s(
                    handles,
                    &self.store,
                    at.lane_indptr,
                    lanes + 1,
                    self.laid.room(at.lane_indptr)?,
                )?,
                permutation: i32s(
                    handles,
                    &self.store,
                    at.permutation,
                    rows,
                    self.laid.room(at.permutation)?,
                )?,
            });
        }
        let mut inputs = Vec::with_capacity(self.laid.seats.len());
        for at in &self.laid.seats {
            let carved = if at.seat.per_lane() { lanes } else { rows };
            inputs.push((
                at.seat.id,
                Tensor::new(
                    handles.bind(&self.store, at.at, self.laid.room(at.at)?)?,
                    carved,
                    at.seat.width,
                    at.seat.dtype,
                ),
            ));
        }

        let voxels = match (fire.voxels, self.laid.voxels) {
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
                Some(VoxelHandles {
                    payload: Tensor::new(
                        handles.bind(&self.store, at.payload, self.laid.room(at.payload)?)?,
                        u32::try_from(seat.rows).unwrap_or(u32::MAX),
                        u32::try_from(seat.channels).unwrap_or(u32::MAX),
                        seat.dtype,
                    ),
                    grid: Tensor::new(
                        handles.bind(&self.store, at.grid, self.laid.room(at.grid)?)?,
                        clips32,
                        4,
                        Dtype::I32,
                    ),
                    token_grid: if seat.token_patch.is_some() {
                        Some(Tensor::new(
                            handles.bind(
                                &self.store,
                                at.token_grid,
                                self.laid.room(at.token_grid)?,
                            )?,
                            clips32,
                            4,
                            Dtype::I32,
                        ))
                    } else {
                        None
                    },
                    slots: Tensor::new(
                        handles.bind(&self.store, at.slots, self.laid.room(at.slots)?)?,
                        clips32,
                        1,
                        Dtype::I32,
                    ),
                })
            }
        };

        let mut page_tables = Vec::with_capacity(self.laid.page_tables.len());
        for (at, geometry) in self.laid.page_tables.iter().zip(fire.page_tables) {
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
                indptr: i32s(
                    handles,
                    &self.store,
                    at.indptr,
                    lanes + 1,
                    self.laid.room(at.indptr)?,
                )?,
                indices: i32s(
                    handles,
                    &self.store,
                    at.indices,
                    geometry.indices.len() as u32,
                    self.laid.room(at.indices)?,
                )?,
                last_page_len: i32s(
                    handles,
                    &self.store,
                    at.last_page_len,
                    lanes,
                    self.laid.room(at.last_page_len)?,
                )?,
                kv_len: i32s(
                    handles,
                    &self.store,
                    at.kv_len,
                    lanes,
                    self.laid.room(at.kv_len)?,
                )?,
                write_page: u32s(
                    handles,
                    &self.store,
                    at.write_page,
                    rows,
                    self.laid.room(at.write_page)?,
                )?,
                write_offset: u32s(
                    handles,
                    &self.store,
                    at.write_offset,
                    rows,
                    self.laid.room(at.write_offset)?,
                )?,
            });
        }

        Ok(Handles {
            tokens: i32s(
                handles,
                &self.store,
                self.laid.tokens,
                rows,
                self.laid.room(self.laid.tokens)?,
            )?,
            positions: i32s(
                handles,
                &self.store,
                self.laid.positions,
                rows,
                self.laid.room(self.laid.positions)?,
            )?,
            rotations: i32s(
                handles,
                &self.store,
                self.laid.rotations,
                rows,
                self.laid.room(self.laid.rotations)?,
            )?,
            readout_rows: i32s(
                handles,
                &self.store,
                self.laid.readout_rows,
                fire.readout_rows.len().max(1) as u32,
                self.laid.room(self.laid.readout_rows)?,
            )?,

            windows: handles.bind(
                &self.store,
                self.laid.windows,
                self.laid.room(self.laid.windows)?,
            )?,
            page_tables,
            slot_ids: entries(
                handles,
                &self.store,
                self.laid.slot_ids,
                lanes,
                self.laid.pieces,
                self.laid.room(self.laid.slot_ids)?,
            )?,
            slot_of_row: entries(
                handles,
                &self.store,
                self.laid.slot_of_row,
                rows,
                self.laid.pieces,
                self.laid.room(self.laid.slot_of_row)?,
            )?,
            adapter_routes: match adapter_routes {
                None => None,
                Some(rows) => Some(i32s(
                    handles,
                    &self.store,
                    self.laid.adapter_routes,
                    rows,
                    self.laid.room(self.laid.adapter_routes)?,
                )?),
            },
            row_valid: Tensor::new(
                handles.bind(
                    &self.store,
                    self.laid.row_valid,
                    self.laid.room(self.laid.row_valid)?,
                )?,
                rows,
                1,
                Dtype::U8,
            ),
            request_of_token: i32s(
                handles,
                &self.store,
                self.laid.request_of_token,
                rows,
                self.laid.room(self.laid.request_of_token)?,
            )?,

            mask: Tensor::new(
                handles.bind(
                    &self.store,
                    self.laid.mask_planes,
                    self.laid.room(self.laid.mask_planes)?,
                )?,
                1,
                u32::try_from(self.laid.mask_plane_bytes.max(4)).unwrap_or(u32::MAX),
                Dtype::U8,
            ),
            mask_words: Tensor::new(
                handles.bind(
                    &self.store,
                    self.laid.mask_words,
                    self.laid.room(self.laid.mask_words)?,
                )?,
                rows,
                2,
                Dtype::U32,
            ),
            patches,
            mrope_positions,
            rs_replay: if rs_tables {
                Some(i32s(
                    handles,
                    &self.store,
                    self.laid.rs_replay,
                    lanes,
                    self.laid.room(self.laid.rs_replay)?,
                )?)
            } else {
                None
            },
            rs_commit: if rs_tables {
                Some(i32s(
                    handles,
                    &self.store,
                    self.laid.rs_commit,
                    lanes,
                    self.laid.room(self.laid.rs_commit)?,
                )?)
            } else {
                None
            },
            rs_carry,
            rs_folds,
            group_of_lane,
            packings,
            inputs,
            voxels,
            capture,
            ring_lanes,
            ring_jobs,
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
            ring_lanes: views.ring_lanes,
        }
    }
}

/// The ints a beam fold's header takes in its table: a lane's of the rs
/// table (`kernels_slang::ssm::CARRY_HEADER`).
pub const FOLD_HEADER: u32 = kernels_slang::ssm::CARRY_HEADER;

/// The ints a fire's staged rs table holds for fires of `lanes` lanes and
/// `rows` tokens over rs pages of `page_tokens`: a header a lane, as many
/// carried tokens as the fire's rows, and the pages those and the lanes' own
/// rows lie in (two partial pages a lane at most).
#[must_use]
pub fn rs_carry_ints(lanes: u64, rows: u64, page_tokens: u32) -> u64 {
    let pages = (2 * rows).div_ceil(u64::from(page_tokens.max(1))) + 2 * lanes;
    lanes * u64::from(kernels_slang::ssm::CARRY_HEADER) + rows + pages
}

fn i32s<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
    bound: u64,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, bound)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
}

/// `rows` entries of `width` i32 units each: a slot table.
fn entries<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
    width: u32,
    bound: u64,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, bound)?;
    Ok(Tensor::new(buf, rows, width, Dtype::I32))
}

fn u32s<B: Backend>(
    handles: &crate::handles::Handles<B>,
    store: &B::Buffer,
    at: u64,
    rows: u32,
    bound: u64,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, bound)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::U32))
}

fn words_of(values: &[u32]) -> &[u8] {
    // SAFETY: a `u32` slice is `size_of_val` initialized bytes with no
    // padding, `u8` has alignment 1, and the result borrows `values`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Each input's room, which a view of it binds whatever rows a fire
    /// brings, holds the input's laid bytes and ends where the next input
    /// begins, inside the store: a view confines a kernel to its own input
    /// (a seat a guest fed among them).
    #[test]
    fn a_room_ends_where_the_next_input_begins() {
        let budget = Budget::new(64, 512);
        let paging = Paging::of(16, 512, 8, 1).expect("the paging lays");
        let laid = Laid::lay(
            &budget,
            paging,
            2,
            0,
            3,
            1,
            None,
            true,
            &[],
            2,
            None,
            2,
            true,
            Some(16),
            Some((2, 64)),
            1 << 30,
            Some(BeamFacts {
                planes: 4,
                rs_planes: 2,
                frames: 2,
            }),
        );
        let rooms: Vec<(u64, u64)> = laid.rooms.iter().map(|(&at, &room)| (at, room)).collect();
        assert!(rooms.len() > 20, "the store lays {} inputs", rooms.len());
        for pair in rooms.windows(2) {
            let ((at, room), (next, _)) = (pair[0], pair[1]);
            assert!(
                at + room <= next,
                "the room at {at} of {room} bytes reaches the input at {next}"
            );
        }
        let &(at, room) = rooms.last().expect("the store lays inputs");
        assert!(at + room <= laid.bytes());
        assert_eq!(
            laid.room(laid.tokens).expect("tokens are laid"),
            u64::from(budget.max_tokens) * 4
        );
        assert!(
            laid.room(laid.bytes() + 4).is_err(),
            "no input lies past the store"
        );
    }
}
