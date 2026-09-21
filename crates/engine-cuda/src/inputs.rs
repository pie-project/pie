use kernels_cuda::Tensor;
use kernels_cuda::attn::plan::{Device, Workspace, prefill_graph_padding};
use model_compiler::Budget;
use model_ir::{Dtype, StructKind};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::{Buffer, Pinned};
use crate::error::Result;
use crate::store::SpaceSeat;
use crate::store::kv::{Facts, Geometry, Paging, SpaceFacts};

const GRANT_INT_BYTES: u64 = 1 << 20;

const GRANT_FLOAT_BYTES: u64 = 64 << 20;

fn graph_float_bytes(facts: &SpaceFacts, sms: u32) -> u64 {
    let padded = u64::from(2 * sms.max(1)) / u64::from(facts.kv_heads.max(1)).max(1);
    let tile = if facts.head_dim >= 256 { 64 } else { 128 };
    let heads = u64::from(facts.q_heads);
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

fn prefill_float_bytes(facts: &SpaceFacts, rows: u32, lanes: u32, device: &Device) -> u64 {
    let (tile, padded) = prefill_graph_padding(
        rows,
        lanes,
        facts.q_heads,
        facts.kv_heads,
        facts.head_dim,
        device,
    );
    let heads = u64::from(facts.q_heads);
    let tile = u64::from(tile);
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

fn decode_float_bytes(facts: &SpaceFacts, lanes: u32) -> u64 {
    let padded = u64::from(lanes.max(1));
    let heads = u64::from(facts.q_heads);
    let v = heads * padded * u64::from(facts.head_dim) * 4;
    let s = heads * padded * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

fn latent_float_bytes(rank: u32, sms: u32) -> u64 {
    let rows = 2 * u64::from(sms.max(1)) * 64;
    let partial_o = (rows * 2 * u64::from(rank)).next_multiple_of(16);
    let partial_lse = (rows * 4).next_multiple_of(16);
    (partial_o + partial_lse).next_multiple_of(ALIGN) + 2 * ALIGN
}

const ALIGN: u64 = 256;

const CLAIM_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Clone, Debug)]
struct Grant {
    int_at: Vec<u64>,
    float_at: u64,
    float_bytes: u64,
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
    pub windows: u64,
    pub qo_absolute: Option<u64>,
    pub live_rows: Option<u64>,
    pub spaces: Vec<SpaceHandles>,
    pub slot_ids: Tensor,
    pub row_valid: Tensor,
    pub adapter_routes: Option<Tensor>,
    pub readout_rows: Tensor,
    pub mask: Option<Tensor>,
    pub mask_indptr: Option<Tensor>,
    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<PackingHandles>,
    pub class_table: Option<(Tensor, u32)>,
    pub lane_of_row: Tensor,
}

const AXES: u64 = 3;

const CLASS_TABLE_BYTES: u64 =
    (engine::fire::ATTN_CLASSES_MAX as u64) * (engine::fire::ATTN_CLASSES_MAX as u64);

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
pub struct PatchHandles {
    pub patches: Tensor,
    pub segments: Tensor,
    pub routes: Tensor,
    pub positions: Tensor,
    pub embed_rows: Option<Tensor>,
    pub embed_weights: Option<Tensor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortSeat {
    pub kind: engine::fire::PortKind,
    pub port: u8,
    pub width: u32,
    pub dtype: Dtype,
}

impl PortSeat {
    #[must_use]
    pub fn per_lane(&self) -> bool {
        self.kind == engine::fire::PortKind::LaneVector
    }

    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.width) * model_compiler::arena::elem_bytes(self.dtype).unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy)]
struct PortAt {
    seat: PortSeat,
    at: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PackingFire<'a> {
    pub group_indptr: &'a [i32],
    pub lane_indptr: &'a [i32],
    pub reference_start: &'a [i32],
    pub reference_tag: &'a [i32],
    pub attn_class: &'a [i32],
    pub permutation: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct PackingHandles {
    pub group_indptr: Tensor,
    pub lane_indptr: Tensor,
    pub reference_start: Tensor,
    pub reference_tag: Tensor,
    pub attn_class: Tensor,
    pub permutation: Tensor,
}

#[derive(Debug, Clone, Copy)]
struct PackingAt {
    group_indptr: u64,
    lane_indptr: u64,
    reference_start: u64,
    reference_tag: u64,
    attn_class: u64,
    permutation: u64,
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
    pub qo_absolute: &'a [i32],
    pub live: &'a [u32],
    pub slot_ids: &'a [i32],
    pub adapter_routes: Option<&'a [i32]>,
    pub readout_rows: &'a [i32],
    pub spaces: &'a [Geometry],
    pub mask: Option<&'a crate::mask::Staged>,
    pub live_rows: u32,
    pub lane_of_row: &'a [i32],
    pub lane_reach: u32,
    pub group_of_lane: &'a [i32],
    pub packings: &'a [PackingFire<'a>],
    pub class_table: Option<(&'a [u8], u32)>,
}

#[derive(Debug)]
pub struct Free {
    bits: AtomicU64,
    depth: u32,
}

impl Free {
    #[must_use]
    pub fn of(depth: usize) -> Arc<Free> {
        debug_assert!(depth <= 64, "the free set is one word");
        let bits = if depth >= 64 {
            u64::MAX
        } else {
            (1u64 << depth) - 1
        };
        Arc::new(Free {
            bits: AtomicU64::new(bits),
            depth: depth as u32,
        })
    }

    #[must_use]
    pub fn take(&self) -> Option<u32> {
        let mut seen = self.bits.load(Ordering::Acquire);
        loop {
            if seen == 0 {
                return None;
            }
            let at = seen.trailing_zeros();
            match self.bits.compare_exchange_weak(
                seen,
                seen & !(1u64 << at),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(at),
                Err(now) => seen = now,
            }
        }
    }

    pub fn give(&self, at: u32) {
        self.bits.fetch_or(1u64 << at, Ordering::Release);
    }

    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.depth - self.bits.load(Ordering::Acquire).count_ones()
    }
}

#[derive(Debug)]
pub struct SlotGuard {
    free: Arc<Free>,
    at: u32,
}

impl SlotGuard {
    #[must_use]
    pub fn at(&self) -> u32 {
        self.at
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.free.give(self.at);
    }
}

#[derive(Debug, Clone)]
pub struct Staged {
    rows: u32,
    lanes: u32,
    windows: usize,
    qo_absolute: Option<usize>,
    live: Option<usize>,
    adapter_rows: Option<u32>,
    readouts: u32,
    mask_bytes: Option<u32>,
    space_indices: Vec<u32>,
    space_lanes: u32,
    packed: bool,
    class_count: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TokenInject {
    pub dst_off: u64,
    pub src: u64,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    staging: Vec<Pinned>,
    free: Arc<Free>,
    stage_bytes: u64,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    window_slots: crate::window::Slots,
    qo_absolute: u64,
    qo_absolute_ints: u64,
    live_rows: u64,
    live_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    adapter_routes: u64,
    readout_rows: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    class_table: u64,
    patch: Option<PatchAt>,
    mrope: Option<u64>,
    mrope_bytes: u64,
    self_cond: Option<u64>,
    self_cond_taps: u64,
    self_cond_bytes: u64,
    ports: Vec<PortAt>,
    lane_of_row: u64,
    group_of_lane: u64,
    packings: Vec<PackingAt>,
    spaces: Vec<SpaceAt>,
    max_lanes: u32,
    max_rows: u32,
    plans: Vec<Option<Workspace>>,
    plan_values: usize,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        budget: &Budget,
        paging: Paging,
        spaces: usize,
        facts: &Facts,
        classes: usize,
        regions: usize,
        runs: u32,
        gathered: usize,
        device: Device,
        runahead: engine::runahead::Runahead,
        patch: Option<PatchSeat>,
        mrope: bool,
        self_cond_taps: u64,
        ports: &[PortSeat],
        selections: usize,
        masked: bool,
        schedule_streams: &[Option<u32>],
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(crate::window::lane_capacity(budget.max_lanes));
        let pages = lanes * u64::from(paging.pages_per_slot);
        let window_slots =
            crate::window::Slots::new(classes, lanes, runs, gathered, rows, spaces, pages);
        let window_ints = window_slots.words();

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let windows = take(window_ints * 4);
        let qo_absolute_ints = lanes + 1;
        let qo_absolute = take(qo_absolute_ints * 4);
        let live_seat = crate::window::Seat::new(regions as u64, u64::from(runs.max(1)));
        let live_ints = live_seat.words();
        let live_rows = take(live_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * 4);
        let adapter_routes = take(rows * 4);
        let readout_rows = take(rows * 4);
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_bytes = if masked {
            (rows * context).div_ceil(8) + lanes
        } else {
            0
        };
        let mask_bits = take(mask_bytes);
        let mask_indptr = take((lanes + 1) * 4);
        let class_table = take(CLASS_TABLE_BYTES);
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
        let lane_of_row = take(rows * 4);
        let group_of_lane = take(lanes * 4);
        let packings: Vec<PackingAt> = (0..selections)
            .map(|_| PackingAt {
                group_indptr: take((lanes + 1) * 4),
                lane_indptr: take((lanes + 1) * 4),
                reference_start: take(lanes * 4),
                reference_tag: take(rows * 4),
                attn_class: take(rows * 4),
                permutation: take(rows * 4),
            })
            .collect();
        let stage_bytes = take(0);
        let ports: Vec<PortAt> = ports
            .iter()
            .map(|seat| {
                let bytes = if seat.per_lane() { lanes } else { rows } * seat.row_bytes();
                PortAt {
                    seat: *seat,
                    at: take(bytes),
                    bytes,
                }
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
        let self_cond = (self_cond_taps > 0).then(|| take(rows * self_cond_taps * 4 * 2));
        let runs = runs.max(1);
        let floats_of: Vec<Option<u64>> = facts
            .plans
            .iter()
            .map(|seat| {
                seat.map(|seat| {
                    match seat.kind {
                        StructKind::AttnPrefillPlan => {
                            graph_float_bytes(&seat.reading, device.num_sm).max(
                                prefill_float_bytes(
                                    &seat.reading,
                                    budget.buckets.last().copied().unwrap_or(budget.max_tokens),
                                    budget.max_lanes,
                                    &device,
                                ),
                            )
                        }
                        StructKind::AttnPrefillPlanSm90 => {
                            graph_float_bytes(&seat.reading, device.num_sm)
                        }
                        StructKind::AttnDecodePlan => {
                            graph_float_bytes(&seat.reading, device.num_sm)
                                .max(decode_float_bytes(&seat.reading, budget.max_lanes))
                        }
                        StructKind::MlaPlan => {
                            latent_float_bytes(seat.reading.head_dim, device.num_sm)
                        }
                    }
                    .max(GRANT_FLOAT_BYTES)
                })
            })
            .collect();
        let mut per_stream: std::collections::BTreeMap<u32, u64> =
            std::collections::BTreeMap::new();
        for (plan, floats) in floats_of.iter().enumerate() {
            if let (Some(floats), Some(Some(stream))) = (floats, schedule_streams.get(plan)) {
                let held = per_stream.entry(*stream).or_insert(0);
                *held = (*held).max(*floats);
            }
        }
        let shared: std::collections::BTreeMap<u32, u64> = per_stream
            .iter()
            .map(|(stream, floats)| (*stream, take(*floats)))
            .collect();
        let grants: Vec<Option<Grant>> = floats_of
            .iter()
            .enumerate()
            .map(|(plan, floats)| {
                floats.map(|floats| {
                    let float_at = match schedule_streams.get(plan) {
                        Some(Some(stream)) => shared[stream],
                        _ => take(floats),
                    };
                    Grant {
                        int_at: (0..runs).map(|_| take(GRANT_INT_BYTES)).collect(),
                        float_at,
                        float_bytes: floats,
                    }
                })
            })
            .collect();
        let total = at;

        let store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        let base = store.ptr();
        let depth = runahead.staging_depth();
        let slot_bytes = usize::try_from(stage_bytes).unwrap_or(usize::MAX);
        let mut staging = Vec::with_capacity(depth);
        for _ in 0..depth {
            staging.push(Pinned::mapped(slot_bytes)?);
        }
        Ok(Inputs {
            staging,
            free: Free::of(depth),
            stage_bytes,
            tokens,
            positions,
            windows,
            window_ints,
            window_slots,
            qo_absolute,
            qo_absolute_ints,
            live_rows,
            live_ints,
            row_valid,
            slot_ids,
            adapter_routes,
            readout_rows,
            mask_bits,
            mask_bytes,
            mask_indptr,
            class_table,
            patch,
            mrope,
            mrope_bytes: if mrope.is_some() { rows * AXES * 4 } else { 0 },
            self_cond,
            self_cond_taps,
            self_cond_bytes: if self_cond.is_some() {
                rows * self_cond_taps * 4
            } else {
                0
            },
            ports,
            lane_of_row,
            group_of_lane,
            packings,
            spaces,
            max_lanes: crate::window::lane_capacity(budget.max_lanes),
            max_rows: budget.max_tokens,
            plan_values: grants.len(),
            plans: (0..runs as usize)
                .flat_map(|run| {
                    grants.iter().map(move |grant| {
                        grant.as_ref().map(|grant| Workspace {
                            int_ptr: base + grant.int_at[run],
                            int_bytes: GRANT_INT_BYTES as usize,
                            float_ptr: base + grant.float_at,
                            float_bytes: usize::try_from(grant.float_bytes).unwrap_or(usize::MAX),
                        })
                    })
                })
                .collect(),
            store,
        })
    }

    #[must_use]
    pub fn grant(&self, plan: u32, run: u32) -> Option<Workspace> {
        let at = run as usize * self.plan_values + plan as usize;
        self.plans.get(at).copied().flatten()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    pub fn claim(&self) -> Result<SlotGuard> {
        if let Some(at) = self.free.take() {
            return Ok(SlotGuard {
                free: Arc::clone(&self.free),
                at,
            });
        }
        let until = std::time::Instant::now() + CLAIM_DEADLINE;
        loop {
            std::hint::spin_loop();
            if let Some(at) = self.free.take() {
                return Ok(SlotGuard {
                    free: Arc::clone(&self.free),
                    at,
                });
            }
            if std::time::Instant::now() >= until {
                return Err(crate::error::Fault::Ceiling {
                    what: "staging slots (every one is still in flight; the caller \
                           is running deeper than the frames_in_flight it loaded with)",
                    need: self.staging.len() as u64 + 1,
                    have: self.staging.len() as u64,
                });
            }
        }
    }

    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.free.in_flight()
    }

    #[must_use]
    pub fn window_slots(&self) -> crate::window::Slots {
        self.window_slots
    }

    pub fn write_host(&self, slot: &SlotGuard, fire: &Fire<'_>) -> Result<Staged> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;
        let host = &self.staging[slot.at() as usize];

        if rows > self.max_rows {
            return Err(crate::error::Fault::Ceiling {
                what: "staged token rows",
                need: u64::from(rows),
                have: u64::from(self.max_rows),
            });
        }

        if fire.windows.len() as u64 > self.window_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
            });
        }
        if fire.qo_absolute.len() as u64 > self.qo_absolute_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "absolute qo boundaries",
                need: fire.qo_absolute.len() as u64,
                have: self.qo_absolute_ints,
            });
        }
        if fire.live.len() as u64 > self.live_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "staged live-rows words",
                need: fire.live.len() as u64,
                have: self.live_ints,
            });
        }
        let mut spelled: Option<usize> = None;
        for geometry in fire.spaces {
            let count = geometry.indptr.len().saturating_sub(1);
            if count != geometry.last_page_len.len()
                || count != geometry.kv_len.len()
                || spelled.is_some_and(|first| first != count)
            {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    format!(
                        "this fire's kv spaces state {count} lane(s) of page bounds, {} \
                         last-page length(s) and {} kv length(s), and every space of one fire \
                         is built over one lane vector",
                        geometry.last_page_len.len(),
                        geometry.kv_len.len()
                    ),
                ));
            }
            spelled = Some(count);
        }
        let space_lanes = spelled
            .map_or(lanes, |count| count as u32)
            .max(lanes)
            .max(fire.lane_reach);
        if u64::from(space_lanes) > u64::from(self.max_lanes) {
            return Err(crate::error::Fault::Ceiling {
                what: "staged kv lanes",
                need: u64::from(space_lanes),
                have: u64::from(self.max_lanes),
            });
        }
        if let Some(staged) = fire.mask
            && staged.bits.len() as u64 > self.mask_bytes
        {
            return Err(crate::error::Fault::Ceiling {
                what: "mask bits",
                need: staged.bits.len() as u64,
                have: self.mask_bytes,
            });
        }

        let put = |offset: u64, bytes: &[u8], what: &'static str| -> Result<()> {
            if host.write(usize::try_from(offset).unwrap_or(usize::MAX), bytes) {
                return Ok(());
            }
            Err(crate::error::Fault::Ceiling {
                what,
                need: offset + bytes.len() as u64,
                have: self.stage_bytes,
            })
        };

        put(self.tokens, bytes_of(fire.tokens), "staged tokens")?;
        put(self.positions, bytes_of(fire.positions), "staged positions")?;
        put(
            self.windows,
            bytes_of(fire.windows),
            "staged window boundaries",
        )?;
        let live = if fire.live_rows == 0 {
            rows
        } else {
            fire.live_rows.min(rows)
        };
        let mut valid = vec![1u8; live as usize];
        valid.resize(rows as usize, 0);
        put(self.row_valid, &valid, "staged row_valid")?;
        if fire.lane_of_row.len() != rows as usize {
            return Err(crate::error::Fault::program(
                "inputs::write_host",
                format!(
                    "this fire stages {} lane-of-row entries for {rows} rows; the map is one \
                     lane per staged row, padding included",
                    fire.lane_of_row.len()
                ),
            ));
        }
        put(
            self.lane_of_row,
            bytes_of(fire.lane_of_row),
            "staged lane of row",
        )?;
        put(self.slot_ids, bytes_of(fire.slot_ids), "staged slot ids")?;
        if space_lanes > lanes {
            let inert = vec![-1i32; (space_lanes - lanes) as usize];
            put(
                self.slot_ids + u64::from(lanes) * 4,
                bytes_of(&inert),
                "staged slot id padding",
            )?;
        }

        let qo_absolute = if fire.qo_absolute.is_empty() {
            None
        } else {
            put(
                self.qo_absolute,
                bytes_of(fire.qo_absolute),
                "staged absolute qo bounds",
            )?;
            Some(fire.qo_absolute.len())
        };

        let live = if fire.live.is_empty() {
            None
        } else {
            put(self.live_rows, u32_bytes_of(fire.live), "staged live rows")?;
            Some(fire.live.len())
        };

        let adapter_rows = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                put(
                    self.adapter_routes,
                    bytes_of(routes),
                    "staged adapter routes",
                )?;
                Some(routes.len() as u32)
            }
        };

        put(
            self.readout_rows,
            bytes_of(fire.readout_rows),
            "staged readout rows",
        )?;

        let mask_bytes = match fire.mask {
            None => None,
            Some(staged) => {
                put(self.mask_bits, &staged.bits, "staged mask bits")?;
                put(
                    self.mask_indptr,
                    bytes_of(&staged.indptr),
                    "staged mask indptr",
                )?;
                Some(u32::try_from(staged.bits.len()).unwrap_or(u32::MAX))
            }
        };
        let class_count = match fire.class_table {
            None => None,
            Some((table, count)) => {
                if table.len() as u64 > CLASS_TABLE_BYTES {
                    return Err(crate::error::Fault::Ceiling {
                        what: "attention class table bytes",
                        need: table.len() as u64,
                        have: CLASS_TABLE_BYTES,
                    });
                }
                put(self.class_table, table, "staged attention class table")?;
                Some(count)
            }
        };

        let packed = !fire.group_of_lane.is_empty();
        if packed {
            if fire.packings.len() != self.packings.len() {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    format!(
                        "this fire stages {} packing(s) and the load carved {}",
                        fire.packings.len(),
                        self.packings.len()
                    ),
                ));
            }
            let mut groups = fire.group_of_lane.to_vec();
            groups.resize(space_lanes as usize, -1);
            put(self.group_of_lane, bytes_of(&groups), "staged group ids")?;
            let mut lane_table: Vec<i32> = Vec::with_capacity(space_lanes as usize + 1);
            let mut row_table: Vec<i32> = Vec::with_capacity(rows as usize);
            for (at, packing) in self.packings.iter().zip(fire.packings) {
                for (offset, table, what) in [
                    (at.group_indptr, packing.group_indptr, "staged group bounds"),
                    (
                        at.lane_indptr,
                        packing.lane_indptr,
                        "staged packed lane bounds",
                    ),
                ] {
                    lane_table.clear();
                    lane_table.extend_from_slice(table);
                    let last = table.last().copied().unwrap_or(0);
                    lane_table.resize(space_lanes as usize + 1, last);
                    put(offset, bytes_of(&lane_table), what)?;
                }
                lane_table.clear();
                lane_table.extend_from_slice(packing.reference_start);
                lane_table.resize(space_lanes as usize, 0);
                put(
                    at.reference_start,
                    bytes_of(&lane_table),
                    "staged reference tails",
                )?;
                for (offset, table, what) in [
                    (
                        at.reference_tag,
                        packing.reference_tag,
                        "staged reference tags",
                    ),
                    (
                        at.attn_class,
                        packing.attn_class,
                        "staged attention classes",
                    ),
                    (
                        at.permutation,
                        packing.permutation,
                        "staged row permutation",
                    ),
                ] {
                    row_table.clear();
                    row_table.extend_from_slice(table);
                    row_table.resize(rows as usize, -1);
                    put(offset, bytes_of(&row_table), what)?;
                }
            }
        }

        let mut space_indices = Vec::with_capacity(self.spaces.len());
        for (at, geometry) in self.spaces.iter().zip(fire.spaces) {
            put(at.indptr, bytes_of(&geometry.indptr), "staged kv indptr")?;
            put(
                at.indices,
                bytes_of(&geometry.indices),
                "staged kv page ids",
            )?;
            put(
                at.last_page_len,
                bytes_of(&geometry.last_page_len),
                "staged last page len",
            )?;
            put(at.kv_len, bytes_of(&geometry.kv_len), "staged kv len")?;
            put(
                at.write_page,
                bytes_of(&geometry.write_page),
                "staged write page",
            )?;
            put(
                at.write_offset,
                bytes_of(&geometry.write_offset),
                "staged write offset",
            )?;
            space_indices.push(geometry.indices.len() as u32);
        }

        Ok(Staged {
            rows,
            lanes,
            windows: fire.windows.len(),
            qo_absolute,
            live,
            adapter_rows,
            readouts: fire.readout_rows.len() as u32,
            mask_bytes,
            space_indices,
            space_lanes,
            packed,
            class_count,
        })
    }

    #[must_use]
    pub fn port(&self, kind: engine::fire::PortKind, port: u8, rows: u32) -> Option<Tensor> {
        let at = self
            .ports
            .iter()
            .find(|at| at.seat.kind == kind && at.seat.port == port)?;
        let rows =
            rows.min(u32::try_from(at.bytes / at.seat.row_bytes().max(1)).unwrap_or(u32::MAX));
        Some(Tensor::new(
            self.store.ptr() + at.at,
            rows,
            at.seat.width,
            at.seat.dtype,
        ))
    }

    #[must_use]
    pub fn ports(&self) -> Vec<PortSeat> {
        self.ports.iter().map(|at| at.seat).collect()
    }

    pub fn stage_patches(
        &mut self,
        stream: *mut core::ffi::c_void,
        payload: &[u8],
        segments: &[i32],
        routes: &[i32],
        positions: &[i32],
        embed_rows: &[i32],
        embed_weights: &[f32],
    ) -> Result<PatchHandles> {
        let Some(at) = self.patch else {
            return Err(crate::error::Fault::Ceiling {
                what: "the patch rectangle, which this load reserved none of",
                need: payload.len() as u64,
                have: 0,
            });
        };
        let owed = [
            (payload.len() as u64, at.seat.rows * at.seat.row_bytes),
            (segments.len() as u64 * 4, (at.seat.images + 1) * 4),
            (routes.len() as u64 * 4, at.seat.rows * 4),
            (positions.len() as u64 * 4, at.seat.rows * AXES * 4),
            (
                embed_rows.len() as u64 * 4,
                at.seat.rows * at.seat.embed_taps * 4,
            ),
            (
                embed_weights.len() as u64 * 4,
                if at.seat.embed_weights {
                    at.seat.rows * at.seat.embed_taps * 4
                } else {
                    0
                },
            ),
        ];
        for (need, have) in owed {
            if need > have {
                return Err(crate::error::Fault::Ceiling {
                    what: "bytes of the patch rectangle this load reserved",
                    need,
                    have,
                });
            }
        }
        let base = self.store.ptr();
        self.store.stage(stream, at.payload, payload)?;
        self.store.stage(stream, at.segments, bytes_of(segments))?;
        self.store.stage(stream, at.routes, bytes_of(routes))?;
        self.store
            .stage(stream, at.positions, bytes_of(positions))?;
        if !embed_rows.is_empty() {
            self.store
                .stage(stream, at.embed_rows, bytes_of(embed_rows))?;
        }
        if !embed_weights.is_empty() {
            self.store
                .stage(stream, at.embed_weights, f32_bytes_of(embed_weights))?;
        }
        let rows = if at.seat.row_bytes == 0 {
            0
        } else {
            (payload.len() as u64 / at.seat.row_bytes) as u32
        };
        let width = model_compiler::arena::elem_bytes(at.seat.dtype)
            .filter(|element| *element > 0)
            .map_or(0, |element| (at.seat.row_bytes / element) as u32);
        Ok(PatchHandles {
            patches: Tensor::new(base + at.payload, rows, width, at.seat.dtype),
            segments: i32s(base + at.segments, segments.len() as u32),
            routes: i32s(base + at.routes, routes.len() as u32),
            positions: Tensor::new(
                base + at.positions,
                (positions.len() / AXES as usize) as u32,
                AXES as u32,
                Dtype::I32,
            ),
            embed_rows: (!embed_rows.is_empty()).then(|| {
                let taps = at.seat.embed_taps.max(1) as u32;
                Tensor::new(
                    base + at.embed_rows,
                    embed_rows.len() as u32 / taps,
                    taps,
                    Dtype::I32,
                )
            }),
            embed_weights: (!embed_weights.is_empty()).then(|| {
                let taps = at.seat.embed_taps.max(1) as u32;
                Tensor::new(
                    base + at.embed_weights,
                    embed_weights.len() as u32 / taps,
                    taps,
                    Dtype::F32,
                )
            }),
        })
    }

    pub fn stage_mrope_positions(
        &mut self,
        stream: *mut core::ffi::c_void,
        positions: &[i32],
    ) -> Result<Tensor> {
        let Some(at) = self.mrope else {
            return Err(crate::error::Fault::Ceiling {
                what: "the triple-wide position stream, which this load reserved none of",
                need: positions.len() as u64 * 4,
                have: 0,
            });
        };
        let have = self.mrope_bytes;
        let need = positions.len() as u64 * 4;
        if need > have {
            return Err(crate::error::Fault::Ceiling {
                what: "bytes of the triple-wide position stream this load reserved",
                need,
                have,
            });
        }
        let base = self.store.ptr();
        self.store.stage(stream, at, bytes_of(positions))?;
        Ok(Tensor::new(
            base + at,
            (positions.len() / AXES as usize) as u32,
            AXES as u32,
            Dtype::I32,
        ))
    }

    pub fn stage_self_cond(
        &mut self,
        stream: *mut core::ffi::c_void,
        rows: &[i32],
        weights: &[f32],
    ) -> Result<(Tensor, Tensor)> {
        let Some(at) = self.self_cond else {
            return Err(crate::error::Fault::Ceiling {
                what: "the self-conditioning taps, which this load reserved no seat for",
                need: rows.len() as u64 * 4,
                have: 0,
            });
        };
        let need = rows.len() as u64 * 4;
        if need > self.self_cond_bytes || weights.len() != rows.len() {
            return Err(crate::error::Fault::Ceiling {
                what: "bytes of self-conditioning taps this load reserved",
                need,
                have: self.self_cond_bytes,
            });
        }
        let base = self.store.ptr();
        let weights_at = at + self.self_cond_bytes;
        self.store.stage(stream, at, bytes_of(rows))?;
        self.store
            .stage(stream, weights_at, f32_bytes_of(weights))?;
        let taps = self.self_cond_taps as u32;
        let token_rows = (rows.len() as u64 / self.self_cond_taps.max(1)) as u32;
        Ok((
            Tensor::new(base + at, token_rows, taps, Dtype::I32),
            Tensor::new(base + weights_at, token_rows, taps, Dtype::F32),
        ))
    }

    pub fn commit(
        &mut self,
        stream: *mut core::ffi::c_void,
        slot: &SlotGuard,
        staged: &Staged,
        token_injects: &[TokenInject],
    ) -> Result<Handles> {
        let Staged {
            rows,
            lanes,
            windows,
            qo_absolute,
            live,
            adapter_rows,
            readouts,
            mask_bytes,
            space_indices,
            space_lanes,
            packed,
            class_count,
        } = staged;
        let readouts = *readouts;
        let (rows, lanes) = (*rows, *lanes);
        let packed = *packed;
        let space_lanes = *space_lanes;
        let base = self.store.ptr();
        let (at_tokens, at_positions, at_windows) = (self.tokens, self.positions, self.windows);
        let at_qo_absolute = self.qo_absolute;
        let at_live = self.live_rows;
        let (at_row_valid, at_slot_ids) = (self.row_valid, self.slot_ids);
        let (at_routes, at_mask, at_mask_indptr) =
            (self.adapter_routes, self.mask_bits, self.mask_indptr);
        let at_class_table = self.class_table;
        let class_count = *class_count;
        let at_readouts = self.readout_rows;
        let places: Vec<SpaceAt> = self.spaces.clone();
        let at_lane_of_row = self.lane_of_row;
        let at_group_of_lane = self.group_of_lane;
        let packing_places: Vec<PackingAt> = self.packings.clone();
        let Inputs { store, staging, .. } = self;
        let host = staging[slot.at() as usize].host();

        let mut spans: Vec<(u64, *const u8, usize)> = Vec::with_capacity(24);
        let mut copy = |offset: u64, len: usize| -> Result<()> {
            spans.push((offset, host.wrapping_add(offset as usize), len));
            Ok(())
        };

        copy(at_tokens, rows as usize * 4)?;
        copy(at_positions, rows as usize * 4)?;
        copy(at_windows, windows * 4)?;
        if let Some(bounds) = qo_absolute {
            copy(at_qo_absolute, *bounds * 4)?;
        }
        if let Some(words) = live {
            copy(at_live, *words * 4)?;
        }
        copy(at_row_valid, rows as usize)?;
        copy(at_lane_of_row, rows as usize * 4)?;
        copy(at_readouts, readouts as usize * 4)?;
        copy(at_slot_ids, space_lanes as usize * 4)?;
        if let Some(routes) = adapter_rows {
            copy(at_routes, *routes as usize * 4)?;
        }
        if let Some(bytes) = mask_bytes {
            copy(at_mask, *bytes as usize)?;
            copy(at_mask_indptr, (lanes as usize + 1) * 4)?;
        }
        if let Some(count) = class_count {
            copy(at_class_table, (count as usize) * (count as usize))?;
        }
        let mut packings = Vec::with_capacity(packing_places.len());
        if packed {
            copy(at_group_of_lane, space_lanes as usize * 4)?;
            for at in &packing_places {
                copy(at.group_indptr, (space_lanes as usize + 1) * 4)?;
                copy(at.lane_indptr, (space_lanes as usize + 1) * 4)?;
                copy(at.reference_start, space_lanes as usize * 4)?;
                copy(at.reference_tag, rows as usize * 4)?;
                copy(at.attn_class, rows as usize * 4)?;
                copy(at.permutation, rows as usize * 4)?;
                packings.push(PackingHandles {
                    group_indptr: i32s(base + at.group_indptr, space_lanes + 1),
                    lane_indptr: i32s(base + at.lane_indptr, space_lanes + 1),
                    reference_start: i32s(base + at.reference_start, space_lanes),
                    reference_tag: i32s(base + at.reference_tag, rows),
                    attn_class: i32s(base + at.attn_class, rows),
                    permutation: i32s(base + at.permutation, rows),
                });
            }
        }
        let mut spaces = Vec::with_capacity(places.len());
        for (at, indices) in places.iter().zip(space_indices) {
            copy(at.indptr, (space_lanes as usize + 1) * 4)?;
            copy(at.indices, *indices as usize * 4)?;
            copy(at.last_page_len, space_lanes as usize * 4)?;
            copy(at.kv_len, space_lanes as usize * 4)?;
            copy(at.write_page, rows as usize * 4)?;
            copy(at.write_offset, rows as usize * 4)?;
            spaces.push(SpaceHandles {
                indptr: i32s(base + at.indptr, space_lanes + 1),
                indices: i32s(base + at.indices, *indices),
                last_page_len: i32s(base + at.last_page_len, space_lanes),
                kv_len: i32s(base + at.kv_len, space_lanes),
                write_page: i32s(base + at.write_page, rows),
                write_offset: i32s(base + at.write_offset, rows),
            });
        }

        // SAFETY: the sources are the slot's pinned bytes, held until the `SlotGuard` drops.
        unsafe { store.stage_batch_from(stream, &spans)? };

        for inject in token_injects {
            crate::device::copy_d2d(
                stream,
                base + at_tokens + inject.dst_off,
                inject.src,
                inject.bytes,
            )?;
        }

        Ok(Handles {
            tokens: i32s(base + at_tokens, rows),
            positions: i32s(base + at_positions, rows),
            windows: base + at_windows,
            qo_absolute: qo_absolute.map(|_| base + at_qo_absolute),
            live_rows: live.map(|_| base + at_live),
            spaces,
            slot_ids: i32s(base + at_slot_ids, space_lanes),
            adapter_routes: adapter_rows.map(|rows| i32s(base + at_routes, rows)),
            readout_rows: i32s(base + at_readouts, readouts),
            row_valid: Tensor::new(base + at_row_valid, rows, 1, Dtype::U8),
            mask: mask_bytes.map(|bytes| Tensor::new(base + at_mask, bytes, 1, Dtype::U8)),
            mask_indptr: mask_bytes.map(|_| i32s(base + at_mask_indptr, lanes + 1)),
            group_of_lane: packed.then(|| i32s(base + at_group_of_lane, space_lanes)),
            packings,
            class_table: class_count.map(|count| {
                (
                    Tensor::new(base + at_class_table, count * count, 1, Dtype::U8),
                    count,
                )
            }),
            lane_of_row: i32s(base + at_lane_of_row, rows),
        })
    }

    #[must_use]
    pub fn seats(
        &self,
        handles: &Handles,
        pages: u32,
        rows: u32,
        lanes: u32,
    ) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            spaces: handles
                .spaces
                .iter()
                .map(|space| SpaceSeat {
                    page_indptr: space.indptr,
                    page_indices: space.indices,
                    last_page_lens: space.last_page_len,
                    row_valid: handles.row_valid,
                })
                .collect(),
            slot_ids: handles.slot_ids,
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
        }
    }
}

fn i32s(ptr: u64, rows: u32) -> Tensor {
    Tensor::new(ptr, rows, 1, Dtype::I32)
}

fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy` with no padding/niche, so all `4 * len` bytes are initialized. Result borrows the input, read-only, for one enqueue.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn f32_bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `f32` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn u32_bytes_of(values: &[u32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `u32` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
