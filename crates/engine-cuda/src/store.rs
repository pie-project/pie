#![allow(unsafe_code)]
pub mod chunks;
mod host;
pub use host::{Ring, RingSize};
mod fit;
pub mod kv;
pub mod layout;
pub mod rs;
pub use self::fit::*;
mod weights;
pub use self::weights::*;
mod tier;
pub use self::tier::*;

use engine::transfer::KvRun;
use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Def, Dtype, Operands, Trace};

use crate::device::elastic::{self, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable, PoolSlabs};
use crate::settle::Airborne;
use crate::store::kv::{Facts, Paging};
use crate::store::layout::Kind;

impl From<poem_exec::store::Fault> for Fault {
    fn from(fault: poem_exec::store::Fault) -> Fault {
        match fault {
            poem_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            poem_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            poem_exec::store::Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            } => Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            },
        }
    }
}

const NHD: i32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    pub card: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

impl Accounting {
    #[must_use]
    pub fn of(card: u64, allowance: engine::Allowance, weights: u64, minimum: u64) -> Accounting {
        let ceiling = allowance.ceiling(card);
        let floor = elastic::safety_floor_bytes();
        Accounting {
            card,
            ceiling,
            weights,
            floor,
            pool: ceiling.saturating_sub(weights).saturating_sub(floor),
            minimum,
        }
    }

    pub fn admit(&self) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        // Dense weights always live on the device: past the allowance, no
        // other knob makes room for them.
        let room = self.ceiling.saturating_sub(self.floor);
        if self.weights > room {
            return Err(Fault::Residency(format!(
                "the card does not hold this deployment's weights: they take {weights} bytes \
                 on the device, and of its {card} bytes the load's allowance (`[engine.memory] \
                 budget_device`, or the whole card) gives pie {ceiling}, {room} past the \
                 driver's safety floor. Dense weights always live in device memory (only \
                 routed experts and token-id tables leave it, through the weight cache), so \
                 serve a quantized deployment of this model, split it across devices with \
                 tensor parallelism, or raise `[engine.memory] budget_device`.",
                weights = self.weights,
                card = self.card,
                ceiling = self.ceiling,
            )));
        }
        Err(Fault::Residency(format!(
            "the card does not hold this deployment: {card} bytes on the device, of which \
             the load's allowance (`[engine.memory] budget_device`, or the whole card) \
             gives pie {ceiling}; this load's weight tier takes {weights} and the driver's \
             safety floor holds back {floor}, leaving the elastic pool {pool} bytes — and \
             the cache rows need {minimum}, one sequence at the declared context. weight \
             tier + elastic pool + safety floor must fit inside the allowance, and here \
             they do not. Lower `[engine] max_context`, raise `[engine.memory] \
             budget_device`, or serve a quantized deployment of the model.",
            card = self.card,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

/// What `Pools::fit_the_card` sized the pool to: the kv pages asked, the
/// pages its chunks would hold were every one the kv rows', the chunks it
/// has, the bytes the card had for them, and what was held back from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    pub asked: u64,
    pub fit: u64,
    pub room: u64,
    pub held: u64,
    /// What the bodies are held.
    pub bodies: u64,
    pub spare: u64,
    pub chunks: u32,
}

/// What a load states of the device pool, where it states anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSize {
    /// The pool's bytes, all of it (a test's).
    Bytes(u64),
    /// The kv tokens the pool holds (`[engine.memory] kv_device_tokens`):
    /// exactly the pages they fill, beside what the pool's other shapes
    /// take ([`StatedKv`]).
    Tokens(u64),
}

/// A pool sized by the kv tokens it holds: the pages they fill, the chunks
/// those pages take with the shapes every working set binds beside them
/// (the null chunk, the guest shapes, the windowed pages, the records:
/// [`chunks_at`]), and the chunks the weights take held whole. The fit
/// holds exactly these chunks, so the kv list is exactly these pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatedKv {
    pub tokens: u64,
    pub page_tokens: u32,
    pub pages: u64,
    /// The recurrent records sized beside the pages: one a slot.
    pub records: u32,
    pub kv_chunks: u32,
    pub weight_chunks: u32,
    pub chunk_bytes: u64,
}

impl StatedKv {
    /// The pages `tokens` tokens fill, a page of `page_tokens` at a time.
    #[must_use]
    pub fn pages_for(tokens: u64, page_tokens: u32) -> u64 {
        tokens.div_ceil(u64::from(page_tokens.max(1)))
    }

    fn of(layout: &layout::Layout, paging: &Paging, tokens: u64) -> StatedKv {
        let pages = StatedKv::pages_for(tokens, paging.page_size);
        StatedKv {
            tokens,
            page_tokens: paging.page_size,
            pages,
            records: paging.slots,
            kv_chunks: chunks_at(layout, paging, pages, paging.slots),
            weight_chunks: u32::try_from(layout.weight_chunks()).unwrap_or(u32::MAX),
            chunk_bytes: layout.chunk_bytes,
        }
    }

    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.kv_chunks.saturating_add(self.weight_chunks)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.chunks()).saturating_mul(self.chunk_bytes)
    }

    /// Refuses the load where the `room` the card has for the pool cannot
    /// hold `backing`, what backing these chunks takes of it
    /// ([`Pools::backing_bytes`]) in the range's map units of `unit`.
    fn admit(&self, room: u64, backing: u64, unit: u64) -> Result<()> {
        if backing > room {
            return Err(self.short(room, backing, unit, None));
        }
        Ok(())
    }

    /// The refusal of a pool that cannot be these chunks: `room` bytes
    /// were what the card had for it, `backing` what backing the chunks
    /// takes of it in map units of `unit`, and `fits` the pages it held
    /// where the fit got that far.
    fn short(&self, room: u64, backing: u64, unit: u64, fits: Option<u64>) -> Fault {
        let mib = |bytes: u64| bytes >> 20;
        let held = fits.map_or(String::new(), |fits| {
            format!(", where it backs {fits} pages")
        });
        Fault::Residency(format!(
            "`[engine.memory] kv_device_tokens = {}` asks {} kv pages of {} tokens: {} chunks \
             of {} bytes for them and the shapes every sequence binds beside them, and {} for \
             the weights held whole, {} bytes in all, which the device backs in {} bytes (its \
             map units of {} MiB); the device has {} bytes ({} MiB) for the pool {} once the \
             weight tier, the fires and the graph bodies are down{held}. State fewer tokens or \
             a larger `budget_device`.",
            self.tokens,
            self.pages,
            self.page_tokens,
            self.kv_chunks,
            self.chunk_bytes,
            self.weight_chunks,
            self.bytes(),
            backing,
            mib(unit),
            room,
            mib(room),
            crate::device::card::bound(),
        ))
    }
}

/// A fire's own-row pages (`Pools::own`): `pages` pages laid as
/// window-buffer slots from `start` bytes past the activation's base (which
/// moves with the pool's chunks: a refit moves it), a page id of
/// the buffer's grain each. Above every carve of the activation's span: the
/// load's carves are laid `bytes` lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnRows {
    pub start: u64,
    pub pages: u32,
    pub page_bytes: u64,
    pub grain: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Kv {
        windowed: bool,
        /// The list its pages are slots of: 0 the kv list, `n` window list
        /// `n - 1`.
        list: u8,
        dtype: Dtype,
        keys_width: u64,
        values_width: u64,
        values_plane: usize,
        head_stride: u64,
        live: Option<[u32; 2]>,
        /// Where the key plane and the value plane start in the row's slot
        /// of the state pool.
        offsets: [u64; 2],
    },
    State {
        stride: u64,
        dtype: Dtype,
        /// The row's place in a working set's record: its index among the
        /// fixed shape's rows.
        run: usize,
    },
}

impl Shape {
    /// The bytes one page holds in a kv row's plane `plane` (0 keys, else
    /// values).
    fn page_bytes(&self, plane: usize, page_size: u32) -> u64 {
        match *self {
            Shape::Kv {
                dtype,
                keys_width,
                values_width,
                ..
            } => {
                let width = if plane == 0 { keys_width } else { values_width };
                u64::from(page_size) * width * u64::from(elem_size(dtype))
            }
            Shape::State { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TableSeat {
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub row_valid: Tensor,
}

#[derive(Debug, Clone)]
pub struct Seats {
    pub lanes: u32,
    pub rows: u32,
    pub pages: u32,
    /// One a page table: the kv list's, then the window list's.
    pub page_tables: Vec<TableSeat>,
    pub slot_ids: Tensor,
    pub write_state: bool,
    pub write_state_mask: Tensor,
    pub commit_len: Tensor,
    pub begin_at: Tensor,
    pub snapshots: kernels_cuda::Snapshots,
}

impl Seats {
    #[must_use]
    pub fn rs(mut self, write_state: bool, mask: Tensor, commit_len: Tensor) -> Seats {
        self.write_state = write_state;
        self.write_state_mask = mask;
        self.commit_len = commit_len;
        self
    }

    #[must_use]
    pub fn splitting(mut self, boundary: Tensor) -> Seats {
        self.begin_at = boundary;
        self
    }

    #[must_use]
    pub fn snapshotting(mut self, snapshots: kernels_cuda::Snapshots) -> Seats {
        self.snapshots = snapshots;
        self
    }
}

#[derive(Debug)]
pub struct Pools {
    /// The pages of a fire's own rows its lane folds past its ring, at the
    /// activation's top (rows a fire folds itself take no window-buffer
    /// slot).
    own: Option<OwnRows>,
    pool: PhysicalPool,
    /// The state pool: the kv rows' pages and the working sets' recurrent
    /// records, a slot each.
    chunks: chunks::Chunks,
    /// The row-streamed tables' caches, a shape each ([`Pools::seat_rows`]).
    row_caches: Vec<RowCache>,
    /// The routed experts' caches, a shape each ([`Pools::seat_experts`]).
    expert_caches: Vec<ExpertCache>,
    /// Where the caches' read counters are snapshotted for
    /// [`Pools::weight_uses`].
    use_record: Option<UseRecord>,
    /// Each window a windowed kv row is declared with, and the list its
    /// pages are slots of.
    windows_of: Vec<(u32, u8)>,
    /// Whether the runtime holds the weights' resident set for the load's
    /// life (`WeightResidency::held`).
    weights_held: bool,
    /// Whether the host copies the caches fill from are gone
    /// ([`Pools::drop_weight_hosts`]).
    hosts_dropped: bool,
    /// The caches' host copies are the serving file's mapping, which no
    /// fire may read in place ([`Pools::file_weights`]).
    weights_filed: bool,
    /// The MXFP4 banks ROCm lays in the matrix core's lane order
    /// ([`Pools::lane_banks`]): a slot filled from the serving file, or a
    /// copy pinned from it, is laid in that order once filled.
    lanes: Vec<LaneBank>,
    /// The fp8 banks ROCm puts in the FP8 matrix core's order
    /// ([`Pools::fp8_banks`]): a slot filled from the serving file, or a
    /// copy pinned from it, has its codes put in that order once filled.
    fp8: Vec<Fp8Bank>,
    /// The pinned copy misses are read from, where the load left the
    /// weights in the file and the runtime streams them after all.
    miss_copy: Option<crate::device::Pinned>,
    /// The spans of the serving file's mapping copied out of it since the
    /// pages were last let go ([`Pools::release_file_pages`]).
    file_spans: Vec<(u64, usize)>,
    /// The load seated every weight atom in slots of its own choosing
    /// ([`Pools::preseat_weights`]), until the runtime's first residency.
    preseated: bool,
    /// Arming over a streamed set: an atom on the host copy is read
    /// through a slot of its shape that the pool holds instead
    /// ([`Pools::stand_in`]).
    stand_in: bool,
    /// Each record a lane named by slot refs, by its first piece's ref: the
    /// refs of all its pieces.
    records: std::collections::HashMap<u32, Vec<u32>>,
    /// A record's state's first piece, by the record it names
    /// ([`Pools::record_key`]).
    record_names: std::collections::HashMap<u32, u32>,
    /// The lanes a fire's staged id tables are laid for, and the entries its
    /// snapshot table is ([`Pools::seat_tables`]).
    id_lanes: u32,
    snap_entries: u32,
    shapes: Vec<Shape>,
    pooled: Vec<PoolRow>,
    paging: Paging,
    /// The context is the model's declared one, which the operator did not
    /// state: the fit before arming shortens it to what it seats
    /// ([`Pools::declare_context`]).
    declared: bool,
    asked: u64,
    /// The chunks the range was first reserved for, the asked count's: a
    /// fit may reserve a fresh range anew up to them, whatever an earlier
    /// fit cut it to.
    asked_chunks: u32,
    ceiling: u64,
    /// A load's stated bytes for the pool (a test's, or what
    /// `stated_kv`'s chunks take), never cut to fit; `None` fills what the
    /// card hands out.
    budget: Option<u64>,
    /// The kv tokens the load states the pool holds, and the chunks that
    /// takes: the fit holds exactly those, or refuses the load.
    stated_kv: Option<StatedKv>,
    /// The chunks the fit left the cache rows, of the range's.
    fitted: u32,
    /// A graph body names addresses in the range: it is never reserved
    /// anew ([`Pools::pin`]).
    pinned: bool,
    /// The chunks stated to the runtime where a fit after the load may yet
    /// give the cache rows more than it holds now ([`Pools::state_whole`]):
    /// only a load that arms its bodies after it has stated its pool.
    stated: Option<u32>,
    /// The range is reserved anew to what the card could back even where
    /// it maps on demand: the pool's planes then span about what the fit
    /// holds, not the asked count (reserved for the asked count,
    /// Qwen3.5-0.8B's c64 ran ~1% slower on sm_120).
    whole_fit: bool,
    /// The pool's chunks lie in one run from its base
    /// ([`Pools::keep_contiguous`]).
    contiguous: bool,
    airborne: Option<Airborne>,
    host: Option<HostKv>,
    disk: Option<DiskTier>,
    /// Where the next host<->disk move is told how it ended
    /// ([`Pools::defer_disk_tell`]); taken by the move.
    disk_told: DiskTell,
    /// The shared pool's host tier: chunks of the state
    /// pool's size, a slot of any shape at the place its host ref names, in
    /// pageable memory behind a page-locked staging ring ([`host`]).
    tier: Option<host::HostTier>,
    /// Every declared page and slot is backed ([`Pools::back_whole`]), and
    /// never trimmed since.
    whole: bool,
    /// The moves to or from the host tier still on the swap stream, and
    /// the read-outs on the tap stream, each with the device pages and the
    /// host slots it holds.
    swaps: Vec<Swap>,
}

impl Pools {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: i32,
        ceiling: u64,
        trace: &Trace,
        paging: Paging,
        stated: Option<KvSize>,
        facts: &Facts,
        budgets: &poem_compiler::Budgets,
        activation: u64,
        chunk_bytes: Option<u64>,
        experts: &[usize],
        rows: bool,
        own_rows_tokens: u32,
    ) -> Result<Pools> {
        let pool = PhysicalPool::open(device, ceiling)?;
        let layout = lay(trace, &paging, budgets, chunk_bytes, experts, rows)?;
        // The pages a fire's own rows take where its lane folds them past
        // what its ring holds (`OwnRows`): at the activation's top, above
        // every carve.
        let own_pages = layout.ring.as_ref().map(|ring| &ring.ring).map(|ring| {
            (
                2 * own_rows_tokens.div_ceil(ring.page_rows) + 1,
                ring.page_bytes,
            )
        });
        let grain = layout
            .windows
            .first()
            .map_or(1, |window| window.grain.max(1));
        let own_bytes = own_pages.map_or(0, |(pages, bytes)| u64::from(pages) * bytes + grain);
        let carved = activation;
        let activation = activation + own_bytes;
        let mut shapes = Vec::with_capacity(trace.caches.len());
        let mut windows_of: Vec<(u32, u8)> = Vec::new();

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    window,
                    live,
                    rows: quantized,
                    ..
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let windowed = window.is_some() && paging.window.is_some();
                    // A windowed row pages in the window list the recipe
                    // lays it in: its own window's, or a wider one's where
                    // the windows are merged.
                    let list = match window.filter(|_| windowed) {
                        None => 0u8,
                        Some(held) => layout
                            .windows
                            .iter()
                            .position(|shape| shape.offset_of(index as u32).is_some())
                            .map(|at| at as u8 + 1)
                            .ok_or_else(|| {
                                Fault::program(
                                    "store::reserve",
                                    format!(
                                        "cache `{name}` is read through a window of {held} \
                                         tokens, and the chunk recipe lays it in no window \
                                         list"
                                    ),
                                )
                            })?,
                    };
                    if let Some(held) = window.filter(|_| list != 0) {
                        windows_of.push((held, list));
                    }
                    let (keys_width, values_width, values_plane) = match planes.as_slice() {
                        [plane] => (*plane, *plane, 0),
                        [keys, values] => (*keys, *values, 1),
                        [] => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares no planes at all — one \
                                     token's entry is written as at least one plane"
                                ),
                            });
                        }
                        many => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares {} planes — this shell \
                                     binds a key plane and a value plane, and knows no third",
                                    many.len()
                                ),
                            });
                        }
                    };
                    let restated = facts.row(index);
                    if let Some(seat) = restated.filter(|seat| seat.kv_heads != 0) {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        // DeepSeek's quantized rows hold a 512-wide head in
                        // a u8 plane of the format the trace declares: its
                        // width (a compact pool's share of it) the format's.
                        let ds_row = *dtype == Dtype::U8
                            && heads == 512
                            && quantized.is_some_and(|f| {
                                matches!(
                                    f,
                                    poem_ir::KvRows::V4
                                        | poem_ir::KvRows::Mx8
                                        | poem_ir::KvRows::Nv4
                                ) && keys_width != 0
                                    && f.bytes() % keys_width == 0
                            })
                            && values_width == keys_width;
                        if !ds_row && (heads != keys_width || heads != values_width) {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares the planes {planes:?} while \
                                     the paged launches that read it restate {} heads of {} — a \
                                     {heads}-wide row",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    let at = layout
                        .shape(Kind::paged(list))
                        .and_then(|shape| shape.offset_of(index as u32))
                        .ok_or_else(|| {
                            Fault::program(
                                "store::reserve",
                                format!("cache `{name}` has no run in the chunk recipe's slot"),
                            )
                        })?;
                    let keys = u64::from(paging.page_size) * keys_width * element;
                    shapes.push(Shape::Kv {
                        offsets: [at, at + if values_plane == 0 { 0 } else { keys }],
                        windowed,
                        list,
                        dtype: *dtype,
                        keys_width,
                        values_width,
                        values_plane,
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                        live: *live,
                    });
                }
                CacheRow::State {
                    name, slab, dtype, ..
                } => {
                    let stride: u64 = slab.iter().product();
                    let run = layout
                        .fixed
                        .as_ref()
                        .and_then(|fixed| fixed.rows.iter().position(|run| run.row == index as u32))
                        .ok_or_else(|| {
                            Fault::program(
                                "store::reserve",
                                format!("state `{name}` has no place in the chunk recipe's record"),
                            )
                        })?;
                    shapes.push(Shape::State {
                        stride,
                        dtype: *dtype,
                        run,
                    });
                }
            }
            debug_assert_eq!(shapes.len(), index + 1, "one shape per cache row");
        }
        let mut pooled = Vec::new();
        for (row, width) in poem_compiler::chunk::census::pooled(trace) {
            // A state keyed by a windowed row's pages (`row` is the cache
            // row its gather reads) lies in the windowed pages' slots.
            let list = match shapes.get(row as usize) {
                Some(Shape::Kv { list, .. }) if paging.window.is_some() => *list,
                _ => 0,
            };
            let windowed = list != 0;
            let offset = layout
                .shape(Kind::paged(list))
                .and_then(|shape| shape.pool_of(row))
                .ok_or_else(|| {
                    Fault::program(
                        "store::reserve",
                        format!(
                            "the compressor state keyed by cache row {row} has no run in \
                             the chunk recipe's slot"
                        ),
                    )
                })?;
            pooled.push(PoolRow {
                row,
                width,
                windowed,
                list,
                offset,
            });
        }
        let asked = chunks_at(&layout, &paging, paging.pages(), paging.slots);
        let stated_kv = match stated {
            Some(KvSize::Tokens(tokens)) => Some(StatedKv::of(&layout, &paging, tokens)),
            _ => None,
        };
        let budget = match stated {
            Some(KvSize::Bytes(bytes)) => Some(bytes),
            Some(KvSize::Tokens(_)) => stated_kv.map(|kv| kv.bytes()),
            None => None,
        };
        // The range is every chunk the pool's ceiling backs: the cache
        // rows' demand is `asked`, and what the guests' instances and
        // rings take of the pool is known to no load. A range mapped on
        // demand only reserves the addresses. A stated budget is the
        // pool's whole, so its range is what was asked: the cache rows'
        // chunks, and with stated tokens the weights' beside them.
        let total = match (stated_kv, budget) {
            (Some(kv), _) => kv.chunks(),
            (None, None) if elastic::mapped_on_demand() => {
                let act = u32::try_from(activation.div_ceil(layout.chunk_bytes.max(1)))
                    .unwrap_or(u32::MAX);
                asked.max(chunks::Chunks::ceiling(&pool, &layout, act))
            }
            (None, _) => asked,
        };
        let chunks = chunks::Chunks::reserve(&pool, layout, total, activation)?;
        let own = own_pages.map(|(pages, page_bytes)| OwnRows {
            start: carved.next_multiple_of(grain),
            pages,
            page_bytes,
            grain,
            bytes: own_bytes,
        });
        Ok(Pools {
            own,
            pool,
            chunks,
            row_caches: Vec::new(),
            expert_caches: Vec::new(),
            use_record: None,
            windows_of,
            weights_held: false,
            hosts_dropped: false,
            weights_filed: false,
            lanes: Vec::new(),
            fp8: Vec::new(),
            miss_copy: None,
            preseated: false,
            stand_in: false,
            file_spans: Vec::new(),
            records: std::collections::HashMap::new(),
            record_names: std::collections::HashMap::new(),
            id_lanes: 1,
            snap_entries: 1,
            shapes,
            pooled,
            paging,
            declared: false,
            asked: paging.pages(),
            asked_chunks: asked,
            ceiling: paging.pages(),
            budget,
            stated_kv,
            fitted: total,
            pinned: false,
            stated: None,
            whole_fit: false,
            contiguous: false,
            airborne: None,
            host: None,
            disk: None,
            disk_told: DiskTell::default(),
            tier: None,
            whole: false,
            swaps: Vec::new(),
        })
    }

    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// The kv tokens the load stated the pool holds, and its chunks.
    #[must_use]
    pub fn stated_kv(&self) -> Option<StatedKv> {
        self.stated_kv
    }

    #[must_use]
    pub fn pool_slabs(&self) -> Vec<(u32, PoolSlabs)> {
        let layout = self.chunks.layout();
        self.pooled
            .iter()
            .map(|row| {
                let grain = layout
                    .shape(Kind::paged(row.list))
                    .map_or(0, |shape| shape.grain);
                let rows = u32::try_from(self.chunks.bytes() / row.plane_bytes(1).max(1))
                    .unwrap_or(u32::MAX);
                let plane = |at: u64| {
                    Tensor::new(
                        self.chunks.base()
                            + row.offset
                            + at * row.plane_bytes(self.paging.page_size),
                        rows,
                        narrow(row.width) as u32,
                        POOL_STATE,
                    )
                };
                (
                    row.row,
                    PoolSlabs {
                        state_kv: plane(0),
                        state_score: plane(1),
                        ape: Tensor::ABSENT,
                        stride: wide(grain / u64::from(elem_size(POOL_STATE))),
                    },
                )
            })
            .collect()
    }

    /// The windowed layers a record holds the rings of,
    /// where the load serves them as ring + buffer: each windowed cache row
    /// (its `layer`) with its window and its share of the ring's bytes, the
    /// last row's share padded to the pieces the record takes.
    #[must_use]
    pub fn windowed_layers(&self) -> Vec<engine::chunk::WindowedLayer> {
        let Some((_, ring)) = self.ring() else {
            return Vec::new();
        };
        ring.layers
            .iter()
            .map(|&(layer, window, bytes)| engine::chunk::WindowedLayer {
                layer,
                window,
                bytes,
            })
            .collect()
    }

    /// The record's rings, where the load serves its windowed rows as ring
    /// + buffer.
    #[must_use]
    pub fn ring(&self) -> Option<(&layout::RingShape, &layout::Ring)> {
        let ring = self.chunks.layout().ring.as_ref()?;
        Some((ring, &ring.ring))
    }

    /// The bytes the own-row pages take at the activation's top (none
    /// where the load keeps no ring): every carve is laid this much lower.
    #[must_use]
    pub fn own_rows_bytes(&self) -> u64 {
        self.own.map_or(0, |own| own.bytes)
    }

    /// Own-row page `page`'s id, as a window-buffer page.
    pub fn own_page_id(&self, page: u32) -> Result<i32> {
        let own = self.own.ok_or_else(|| {
            Fault::program(
                "store::ring",
                "an own-row page of a load that keeps no ring",
            )
        })?;
        if page >= own.pages {
            return Err(Fault::Ceiling {
                what: "own-row pages of a fire that folds its rows past its ring",
                need: u64::from(page) + 1,
                have: u64::from(own.pages),
            });
        }
        let at = self.activation_base() - self.chunks.base()
            + own.start
            + u64::from(page) * own.page_bytes;
        let q = at / own.grain;
        i32::try_from(q).map_err(|_| Fault::Ceiling {
            what: "page ids of the state pool",
            need: q,
            have: i32::MAX as u64,
        })
    }

    /// The pool's base, every page id counted from it.
    #[must_use]
    pub fn pool_base(&self) -> u64 {
        self.chunks.base()
    }

    /// The step the window buffer's page ids count in (ring and own-row
    /// pages' too).
    #[must_use]
    pub fn window_grain(&self) -> u64 {
        self.chunks
            .layout()
            .windows
            .first()
            .map_or(1, |window| window.grain.max(1))
    }

    /// Own-row page `page`'s address.
    pub fn own_page_at(&self, page: u32) -> Result<u64> {
        let own = self.own.ok_or_else(|| {
            Fault::program(
                "store::ring",
                "an own-row page of a load that keeps no ring",
            )
        })?;
        Ok(self.activation_base() + own.start + u64::from(page) * own.page_bytes)
    }

    /// The ring pages of the record `key` names, in ring order, by the ids
    /// a kernel reads a window-buffer page by, and their addresses; the
    /// record seated and backed.
    pub fn ring_pages(&mut self, key: u32) -> Result<Vec<(i32, u64)>> {
        let slots = self.ensure_record(key)?;
        let (fixed, ring) = self.ring().ok_or_else(|| {
            Fault::program("store::ring", "ring pages of a load that keeps no ring")
        })?;
        let fixed_grain = fixed.slot.grain.max(1);
        let (pages, page_bytes, ring_offset) = (ring.pages, ring.page_bytes, ring.offset);
        let places: Vec<(u32, u32)> = (0..pages).map(|page| ring.place(page)).collect();
        let grain = self
            .chunks
            .layout()
            .windows
            .first()
            .map_or(1, |window| window.grain.max(1));
        let mut out = Vec::with_capacity(pages as usize);
        for (piece, within) in places {
            let slot = *slots.get(piece as usize).ok_or_else(|| {
                Fault::program("store::ring", "a ring page past its record's pieces")
            })?;
            let offset = u64::try_from(self.chunks.page_id(Kind::Ring, slot)?).unwrap_or(0)
                * fixed_grain
                + ring_offset
                + u64::from(within) * page_bytes;
            let q = offset / grain;
            out.push((
                i32::try_from(q).map_err(|_| Fault::Ceiling {
                    what: "page ids of the state pool",
                    need: q,
                    have: i32::MAX as u64,
                })?,
                self.chunks.base() + offset,
            ));
        }
        Ok(out)
    }

    /// Window-buffer slot `slot` (a slot ref): the page id a kernel reads it
    /// by and its address, seated and backed.
    pub fn window_slot(&mut self, slot: u32) -> Result<(i32, u64)> {
        self.seat_pages(Kind::Window(0), std::iter::once(u64::from(slot)))?;
        let bytes = self
            .chunks
            .layout()
            .windows
            .first()
            .map_or(0, |window| window.slot_bytes);
        Ok((
            self.chunks.page_id(Kind::Window(0), slot)?,
            self.chunks.span(Kind::Window(0), slot, 0, bytes)?,
        ))
    }

    /// Each windowed row's planes in a window-buffer page (and so in a ring
    /// page): (offset, bytes a row), its key plane's then its value plane's.
    #[must_use]
    pub fn ring_planes(&self) -> Vec<(u64, u64)> {
        let Some((_, ring)) = self.ring() else {
            return Vec::new();
        };
        ring.windows
            .iter()
            .filter_map(|&(row, _)| match self.shapes.get(row as usize) {
                Some(&Shape::Kv {
                    dtype,
                    keys_width,
                    values_width,
                    offsets,
                    ..
                }) => {
                    let elem = u64::from(elem_size(dtype));
                    Some([
                        (offsets[0], keys_width * elem),
                        (offsets[1], values_width * elem),
                    ])
                }
                _ => None,
            })
            .flatten()
            // The compressor states the windowed pages key lie in them a
            // row each too (two planes a state): they fold with the rows.
            .chain(
                self.pooled
                    .iter()
                    .filter(|row| row.windowed)
                    .flat_map(|row| {
                        let plane = row.plane_bytes(self.paging.page_size);
                        let per_row = row.width * u64::from(elem_size(POOL_STATE));
                        [(row.offset, per_row), (row.offset + plane, per_row)]
                    }),
            )
            .collect()
    }

    /// The widest window a ring holds the rows of, and the rows a page.
    #[must_use]
    pub fn ring_window(&self) -> Option<(u32, u32)> {
        let (_, ring) = self.ring()?;
        Some((
            ring.windows.iter().map(|&(_, w)| w).max().unwrap_or(0),
            ring.page_rows,
        ))
    }

    /// The bytes a record takes: its recurrent rows' and its rings'.
    #[must_use]
    pub fn record_bytes(&self) -> u64 {
        self.state_slot_bytes()
            + self.ring().map_or(0, |(fixed, _)| {
                fixed.slot.slot_bytes * u64::from(fixed.pieces)
            })
    }

    /// Whether a working set holds a record: its recurrent rows, or its
    /// windowed rows' rings.
    #[must_use]
    pub fn has_record(&self) -> bool {
        self.has_state() || self.ring().is_some()
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    /// The page tables a fire stages: one a kv shape, the list's and then
    /// each window list's, narrowest first. Every kv row of a shape reads
    /// its shape's.
    #[must_use]
    pub fn geometries(&self) -> usize {
        let layout = self.chunks.layout();
        usize::from(layout.list.is_some()) + layout.windows.len()
    }

    /// Whether page table `at` of a fire's is a window list's.
    #[must_use]
    pub fn windowed_geometry(&self, at: usize) -> bool {
        self.geometry_window(at).is_some()
    }

    /// The window list page table `at` of a fire's is of, if it is one's.
    #[must_use]
    pub fn geometry_window(&self, at: usize) -> Option<usize> {
        let layout = self.chunks.layout();
        at.checked_sub(usize::from(layout.list.is_some()))
            .filter(|&window| window < layout.windows.len())
    }

    /// Each window a kv row is declared with and the page table of a
    /// fire's that row's list reads: its own window's list's, or the wider
    /// list's its rows are merged into.
    #[must_use]
    pub fn window_tables(&self) -> Vec<(u32, u32)> {
        let mut tables: Vec<(u32, u32)> = Vec::new();
        for &(window, list) in &self.windows_of {
            let table = self.table_of(list) as u32;
            if !tables.contains(&(window, table)) {
                tables.push((window, table));
            }
        }
        tables
    }

    /// The page table of a fire's that the kv rows paged in the window
    /// list (`windowed`) or in the kv list read. The kv list's is the
    /// first, and the All axis's: a model keeping no full-retention row
    /// pages its All axis in its window list, then the only one.
    #[must_use]
    pub fn table_of(&self, list: u8) -> usize {
        match list {
            0 => 0,
            window => usize::from(self.chunks.layout().list.is_some()) + usize::from(window) - 1,
        }
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }))
    }

    /// Seats the kv pages `pages` (the windowed ones, a window list's) on slots of
    /// the state pool and backs the segments their chunks lie in.
    fn seat_pages(&mut self, kind: Kind, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        // The physical chunks the pages lie in. A lane's pages run through
        // a few chunks, so a chunk just checked is not checked again.
        let mut highest = None;
        let mut unbacked = None;
        let mut checked = None;
        for page in pages {
            let page = u32::try_from(page).map_err(|_| Fault::Ceiling {
                what: kind.name(),
                need: page,
                have: u64::from(u32::MAX),
            })?;
            let chunk = self.chunks.seat(kind, page)?;
            highest = highest.max(Some(chunk));
            if unbacked.is_none() && checked != Some(chunk) {
                checked = Some(chunk);
                if !self.chunks.backed(chunk) {
                    unbacked = Some(chunk);
                }
            }
        }
        let (Some(highest), Some(unbacked)) = (highest, unbacked) else {
            return Ok(());
        };
        if self.chunks.by_refs() {
            // Nothing is backed on a fire's path: the runtime commits a
            // segment before it hands out a chunk of it.
            return Err(Fault::program(
                "store::chunks",
                format!(
                    "a {} lies in physical chunk {unbacked}, which the runtime has not \
                     committed ({} segments are)",
                    kind.name(),
                    self.chunks.committed_segments()
                ),
            ));
        }
        let segments = self.chunks.segments_for(u64::from(highest) + 1);
        match self.chunks.commit_segments(&mut self.pool, segments)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Backs the windowed pages `pages` (and the null page behind a
    /// window): their slots in the state pool, and their rows of the
    /// compressor slabs their pages key.
    pub fn ensure_windowed(&mut self, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        if !self.has_windowed() {
            return Ok(());
        }
        let capacity = if self.chunks.by_refs() {
            u64::MAX
        } else {
            self.paging.window_pages()
        };
        let mut fresh: Vec<u64> = core::iter::once(0).chain(pages).collect();
        fresh.sort_unstable();
        fresh.dedup();
        if let Some(&page) = fresh.iter().find(|&&page| page >= capacity) {
            return Err(Fault::Ceiling {
                what: "windowed kv pages",
                need: page + 1,
                have: capacity,
            });
        }
        // A windowed page is a slot of every window list.
        for window in 0..self.chunks.layout().windows.len() {
            self.seat_pages(Kind::Window(window as u8), fresh.iter().copied())?;
        }
        Ok(())
    }

    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            table.push(match *shape {
                Shape::Kv {
                    windowed,
                    list,
                    dtype,
                    keys_width,
                    values_width,
                    values_plane,
                    head_stride,
                    live,
                    offsets,
                } => {
                    let table = self.table_of(list);
                    let seat = seats.page_tables.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("page table {table}, which this fire staged none of"),
                    })?;
                    // A ringed load's windowed table names ring and own-row
                    // pages anywhere in the pool's range: its planes span it.
                    let pages = if windowed && self.ring().is_some() {
                        self.chunks.bytes() / grain_of(&self.chunks, list).max(1)
                    } else if windowed {
                        self.paging.window_pages()
                    } else {
                        self.paging.pages()
                    };
                    let cells = pages * u64::from(self.paging.page_size);
                    // A plane's pointer carries its offset in a page's
                    // slot; the pages step by the shape's grain.
                    let grain = self
                        .chunks
                        .layout()
                        .shape(Kind::paged(list))
                        .map_or(0, |shape| shape.grain);
                    let _ = values_plane;
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            self.chunks.base() + offsets[at],
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv {
                        table: table as u32,
                        pool: KvPool {
                            keys: plane(0, keys_width),
                            values: plane(1, values_width),
                            bf16_keys: Tensor::new(0, 0, 0, dtype),
                            bf16_values: Tensor::new(0, 0, 0, dtype),
                            key_scales: Tensor::new(0, 0, 0, Dtype::U8),
                            value_scales: Tensor::new(0, 0, 0, Dtype::U8),
                            page_indices: seat.page_indices,
                            page_indptr: seat.page_indptr,
                            last_page_lens: seat.last_page_lens,
                            row_valid: seat.row_valid,
                            env_min: Tensor::new(0, 0, 0, dtype),
                            env_max: Tensor::new(0, 0, 0, dtype),
                            has_envelopes: false,
                            page_size: narrow(u64::from(self.paging.page_size)),
                            seq_stride: wide(keys_width),
                            head_stride: wide(head_stride),
                            layout: NHD,
                            scheme_byte: 0,
                            block_size: 0,
                            max_pages_per_request: narrow(u64::from(self.paging.pages_per_slot)),
                            pages_in_batch: narrow(u64::from(seats.pages)),
                            live: live.unwrap_or([0, 0]),
                            page_stride: [grain, grain],
                        },
                    }
                }
                Shape::State { stride, dtype, run } => {
                    // A lane names each piece's slot a grain at a time, in
                    // the id table the row's pieces share; a piece lies
                    // `offset` into its slot.
                    let fixed = self.chunks.layout().fixed.as_ref().ok_or_else(|| {
                        Fault::program("store::table", "a recurrent row and no fixed shape")
                    })?;
                    let row = &fixed.rows[run];
                    let element = u64::from(elem_size(dtype)).max(1);
                    let step = wide(fixed.slot.grain / element);
                    let slab = Tensor::new(
                        self.chunks.base(),
                        u32::try_from(self.chunks.bytes() / fixed.slot.grain.max(1))
                            .unwrap_or(u32::MAX),
                        narrow(stride) as u32,
                        dtype,
                    );
                    let column = u64::from(row.column);
                    let mut snapshots = seats.snapshots;
                    if !snapshots.slots.is_absent() {
                        snapshots.slots.ptr += column * u64::from(self.snap_entries) * 4;
                    }
                    CachePool::Recurrent(RecurrentPool {
                        write_state: seats.write_state,
                        write_state_mask: seats.write_state_mask,
                        commit_len: seats.commit_len,
                        begin_at: seats.begin_at,
                        fused_decay: false,
                        snapshots,
                        slab,
                        slot_ids: Tensor::new(
                            seats.slot_ids.ptr + column * u64::from(self.id_lanes) * 4,
                            seats.slot_ids.rows,
                            row.pieces,
                            Dtype::I32,
                        ),
                        slot_stride_elems: step,
                        conv_slab: slab,
                        conv_stride: step,
                        segments: kernels_cuda::Segments {
                            table: Tensor::ABSENT,
                            seg_chunks: 0,
                            pieces: row.pieces,
                            piece_off: wide(row.offset / element),
                            piece_atoms: if row.pieces > 1 {
                                u32::try_from(row.atoms_per_piece).unwrap_or(u32::MAX)
                            } else {
                                0
                            },
                        },
                    })
                }
            });
        }
        Ok(CacheTable(table))
    }

    pub fn clear(&mut self, slot: u32) -> Result<()> {
        self.zero_slot(None, slot)
    }

    pub fn clear_on(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<()> {
        self.zero_slot(Some(stream), slot)
    }

    /// Copies the record `src` names onto the one `dst` names, slot for
    /// slot: its recurrent rows and the rs buffer's rows with them.
    pub fn copy_slot(&mut self, stream: *mut core::ffi::c_void, src: u32, dst: u32) -> Result<()> {
        for slot in [src, dst] {
            if !self.state_row(slot)? {
                return Ok(());
            }
        }
        if src == dst {
            return Ok(());
        }
        let (from, to) = (self.ensure_record(src)?, self.ensure_record(dst)?);
        // A record moving to or from the host tier lands first.
        self.fence_all_swaps(stream)?;
        for (piece, (&from, &to)) in from.iter().zip(&to).enumerate() {
            let (kind, bytes) = self.piece_kind(piece)?;
            crate::device::copy_d2d(
                stream,
                self.chunks.span(kind, to, 0, bytes)?,
                self.chunks.span(kind, from, 0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    /// The shapes a record spans, in its pieces' order: its rings', then
    /// its state's, each with its pieces and slot bytes.
    fn record_shapes(&self) -> Vec<(Kind, u32, u64)> {
        let layout = self.chunks.layout();
        layout
            .ring
            .as_ref()
            .map(|ring| (Kind::Ring, ring.pieces, ring.slot.slot_bytes))
            .into_iter()
            .chain(
                layout
                    .fixed
                    .as_ref()
                    .map(|fixed| (Kind::Fixed, fixed.pieces, fixed.slot.slot_bytes)),
            )
            .collect()
    }

    /// The pieces a record takes, every shape's.
    fn record_pieces(&self) -> u32 {
        self.record_shapes()
            .iter()
            .map(|&(_, pieces, _)| pieces)
            .sum()
    }

    /// The record's rings' pieces: the first of its pieces.
    fn ring_pieces(&self) -> u32 {
        self.chunks
            .layout()
            .ring
            .as_ref()
            .map_or(0, |ring| ring.pieces)
    }

    /// Piece `piece` of a record: its shape's kind and slot bytes.
    fn piece_kind(&self, piece: usize) -> Result<(Kind, u64)> {
        let mut at = piece as u64;
        for (kind, pieces, bytes) in self.record_shapes() {
            if at < u64::from(pieces) {
                return Ok((kind, bytes));
            }
            at -= u64::from(pieces);
        }
        Err(Fault::program(
            "store::state",
            format!("piece {piece} past a record's pieces"),
        ))
    }

    fn fixed(&self) -> Result<&layout::FixedShape> {
        self.chunks
            .layout()
            .fixed
            .as_ref()
            .ok_or_else(|| Fault::program("store::state", "a record named and no fixed shape"))
    }

    /// The lanes a fire's id tables are staged for and the entries of its
    /// snapshot table: where each row's ids start in them.
    pub fn seat_tables(&mut self, id_lanes: u32, snap_entries: u32) {
        self.id_lanes = id_lanes.max(1);
        self.snap_entries = snap_entries.max(1);
    }

    /// The ids a fire stages a lane for the recurrent rows.
    #[must_use]
    pub fn ids_a_lane(&self) -> u32 {
        self.chunks
            .layout()
            .fixed
            .as_ref()
            .map_or(1, |fixed| fixed.ids_a_lane().max(1))
    }

    /// The whole chunks of the span under a carve that leaves `under`
    /// bytes below it, as the rungs state them lent: none where the
    /// state's chunks do not end where the activation starts
    /// ([`chunks::Chunks::state_fitted`]), since what is lent and the state's last
    /// chunks would not be one span.
    fn lent_bytes(&self, under: u64) -> u64 {
        if !self.chunks.state_meets_activation() {
            return 0;
        }
        let chunk = self.chunks.layout().chunk_bytes.max(1);
        let act = u64::from(self.chunks.act_chunks());
        (under / chunk).min(act.saturating_sub(1)) * chunk
    }

    /// States the range whole to the runtime: every chunk it reserved, the
    /// fit's and those the fit after the bodies arm may give the cache
    /// rows, which the runtime keeps the pool under by
    /// [`Pools::kept_back_bytes`] meanwhile. For a load that arms at its
    /// starting weight residency, after the runtime read its pool: any
    /// other states what its last fit backed ([`Pools::fit_and_state`]).
    /// The state's chunks then end where the activation starts.
    pub(crate) fn state_whole(&mut self) {
        self.stated = Some(self.chunks.total());
        self.paging.pages = self
            .paging
            .pages
            .max(self.pages_within(self.chunks.total()));
        self.pledge();
    }

    /// The address of a guest item of `bytes` where `item` names it, where
    /// it lies now: a run of a role's slots ([`Pools::guest_span`]), or a
    /// run of adjacent chunks ([`chunks::Chunks::guest_chunks`]).
    pub fn guest_item(&self, item: crate::program::Item<'_>, bytes: u64) -> Result<u64> {
        match item {
            crate::program::Item::Slots(role, refs) => self.guest_span(role, refs, bytes),
            crate::program::Item::Chunks { first, run } => {
                self.chunks.guest_chunks(first, run, bytes)
            }
        }
    }

    /// What a guest holding `per_instance` bytes an instance and
    /// `per_pair` a pair, each in one piece, states (`engine::GuestState`):
    /// the bytes, each a run of `Instance` slots where the runtime lays it
    /// so, else one run of the adjacent chunks that hold the larger, which
    /// a segment must hold.
    pub fn guest_state(
        &self,
        per_instance: u64,
        resident: u64,
        per_pair: u64,
    ) -> Result<engine::GuestState> {
        let layout = self.chunks.layout();
        let chunk = layout.chunk_bytes.max(1);
        // An instance's place holds its resident state, then its scratch:
        // one piece, moved whole at a park.
        let place = resident + per_instance;
        let run = u32::try_from(place.max(per_pair).div_ceil(chunk)).unwrap_or(u32::MAX);
        if run > layout.segment_chunks {
            return Err(Fault::Ceiling {
                what: "chunks of a segment a guest's state lies in, one run",
                need: u64::from(run),
                have: u64::from(layout.segment_chunks),
            });
        }
        Ok(engine::GuestState {
            // What a later fire reads (a beam group's state) lies first in the
            // instance's place, copied with it at a park; its lane's value
            // scratch past it is dead between fires (one arm).
            resident,
            scratch: per_instance,
            scratch_arms: 1,
            per_pair,
            run,
        })
    }

    /// The address of a guest item of `bytes` in `refs`, slot refs of
    /// `role`'s shape, where their chunk lies now.
    pub fn guest_span(
        &self,
        role: engine::chunk::SlotRole,
        refs: &[u32],
        bytes: u64,
    ) -> Result<u64> {
        self.chunks.guest_span(role, refs, bytes)
    }

    /// The slots a guest item of `bytes` takes of `role`'s shape.
    #[must_use]
    pub fn guest_footprint(
        &self,
        role: engine::chunk::SlotRole,
        bytes: u64,
    ) -> engine::program::Footprint {
        self.chunks
            .layout()
            .guest(role)
            .map(|shape| shape.footprint(bytes))
            .unwrap_or_default()
    }

    /// Names the record whose pieces lie in the slot refs `pieces`, for the
    /// fires and copies that name it by its first.
    pub fn name_record(&mut self, pieces: &[u32]) -> Result<u32> {
        let want = self.record_pieces() as usize;
        let Some(&key) = pieces.first().filter(|_| want > 0 && pieces.len() == want) else {
            return Err(Fault::program(
                "store::state",
                format!(
                    "a record named by {} slot refs, and this load's takes {want}",
                    pieces.len()
                ),
            ));
        };
        if want > 1 {
            self.records.insert(key, pieces.to_vec());
        }
        // A record of rings and state is named by its rings' first piece;
        // its state's first names it too.
        if let Some(&state) = pieces
            .get(self.ring_pieces() as usize)
            .filter(|_| self.ring_pieces() > 0)
        {
            self.record_names.insert(state, key);
        }
        Ok(key)
    }

    /// The record a slot ref names: itself, or the record whose state's
    /// first piece it is.
    #[must_use]
    pub fn record_key(&self, slot: u32) -> u32 {
        self.record_names.get(&slot).copied().unwrap_or(slot)
    }

    /// The slots of the record `key` names, in the record's order: the
    /// refs a lane named it by, or, where lanes name state slots, the
    /// slot's own run.
    fn record(&self, key: u32) -> Result<Vec<u32>> {
        let mut slots = Vec::new();
        self.record_into(key, &mut slots)?;
        Ok(slots)
    }

    /// [`Pools::record`] into `slots`, cleared first: a fire's lanes reuse
    /// one buffer.
    fn record_into(&self, key: u32, slots: &mut Vec<u32>) -> Result<()> {
        slots.clear();
        let pieces = self.record_pieces();
        if pieces == 0 {
            return Err(Fault::program(
                "store::state",
                "a record named and no fixed shape",
            ));
        }
        if !self.chunks.by_refs() {
            for (_, n, _) in self.record_shapes() {
                slots.extend((0..n).map(|piece| key * n + piece));
            }
            return Ok(());
        }
        if pieces == 1 {
            slots.push(key);
            return Ok(());
        }
        let held = self.records.get(&key).ok_or_else(|| {
            Fault::program(
                "store::state",
                format!("the record at slot ref {key} is read before a lane names its pieces"),
            )
        })?;
        slots.extend_from_slice(held);
        Ok(())
    }

    /// The record `key` names, its slots seated and backed.
    fn ensure_record(&mut self, key: u32) -> Result<Vec<u32>> {
        let slots = self.record(key)?;
        self.seat_record(&slots)?;
        Ok(slots)
    }

    /// Seats and backs a record's pieces, each on its shape.
    fn seat_record(&mut self, slots: &[u32]) -> Result<()> {
        let rings = self.ring_pieces() as usize;
        let (ring, state) = slots.split_at(rings.min(slots.len()));
        if !ring.is_empty() {
            self.seat_pages(Kind::Ring, ring.iter().map(|&slot| u64::from(slot)))?;
        }
        if !state.is_empty() {
            self.seat_pages(Kind::Fixed, state.iter().map(|&slot| u64::from(slot)))?;
        }
        Ok(())
    }

    /// The record's bytes as the host holds them, `(slot of the record,
    /// offset, bytes)` in order: each recurrent row piece after piece.
    fn record_spans(&self) -> Vec<(u32, u64, u64)> {
        // Its rings' pieces whole (`layout::Ring`), first.
        let mut spans: Vec<(u32, u64, u64)> = self
            .chunks
            .layout()
            .ring
            .as_ref()
            .map(|ring| {
                (0..ring.pieces)
                    .map(|piece| (piece, 0, ring.slot.slot_bytes))
                    .collect()
            })
            .unwrap_or_default();
        let rings = self.ring_pieces();
        let Some(fixed) = self.chunks.layout().fixed.as_ref() else {
            return spans;
        };
        for shape in &self.shapes {
            let Shape::State { run, .. } = *shape else {
                continue;
            };
            let row = &fixed.rows[run];
            for (piece, &slot) in row.slots.iter().enumerate() {
                spans.push((rings + slot, row.offset, row.piece_bytes(piece as u32)));
            }
        }
        spans
    }

    /// The staged id tables of a fire whose lanes name the records `keys`
    /// (under zero: none): each table's block `id_lanes` lanes long, a
    /// lane's ids its pieces' slots as the recurrent kernels read them.
    /// Seats and backs each record.
    pub fn state_tables(&mut self, keys: &[i32]) -> Result<Vec<i32>> {
        if !self.has_state() {
            return Ok(keys.to_vec());
        }
        let lanes = self.id_lanes as usize;
        self.expand(keys, lanes)
    }

    /// A fire's snapshot slots, as [`Pools::state_tables`] lays a fire's
    /// lanes: an entry a record a snapshot lands in.
    pub fn snapshot_tables(&mut self, keys: &[i32]) -> Result<Vec<i32>> {
        if !self.has_state() || keys.is_empty() {
            return Ok(keys.to_vec());
        }
        let entries = self.snap_entries as usize;
        self.expand(keys, entries)
    }

    fn expand(&mut self, keys: &[i32], rows: usize) -> Result<Vec<i32>> {
        if keys.len() > rows {
            return Err(Fault::Ceiling {
                what: "records a fire's id tables name",
                need: keys.len() as u64,
                have: rows as u64,
            });
        }
        // Each record seated and backed first; the ids are read after, the
        // shape borrowed, a buffer of slots reused lane to lane.
        let mut record = Vec::new();
        for &key in keys.iter().filter(|&&key| key >= 0) {
            self.state_row(key as u32)?;
            self.record_into(key as u32, &mut record)?;
            self.seat_record(&record)?;
        }
        let rings = self.ring_pieces() as usize;
        let fixed = self.fixed()?;
        // One table of one piece: the ids as the keys lie, no padding.
        let single = fixed.tables.len() == 1 && fixed.tables[0].len() == 1;
        let mut out = if single {
            vec![-1i32; keys.len()]
        } else {
            vec![-1i32; rows * fixed.ids_a_lane() as usize]
        };
        // A table's block is `rows` lanes long: where it starts.
        let block = if single { 0 } else { rows };
        for (lane, &key) in keys.iter().enumerate() {
            if key < 0 {
                continue;
            }
            self.record_into(key as u32, &mut record)?;
            let mut column = 0usize;
            for table in &fixed.tables {
                for (piece, &slot) in table.iter().enumerate() {
                    out[column * block + lane * table.len() + piece] = self
                        .chunks
                        .page_id(Kind::Fixed, record[rings + slot as usize])?;
                }
                column += table.len();
            }
        }
        Ok(out)
    }

    /// The rs list as the chunk recipe lays it: the tokens a page holds, and
    /// each rs row's run in a page, `(offset, bytes a token)`.
    #[must_use]
    pub fn rs_layout(&self) -> Option<(u32, Vec<(u64, u64)>)> {
        self.chunks
            .layout()
            .rs
            .as_ref()
            .map(|rs| (rs.slot.tokens, rs.layers.clone()))
    }

    /// Seats the rs buffer pages `pages` on slots of the state pool and
    /// backs the segments their chunks lie in.
    pub fn seat_rs(&mut self, pages: &[u32]) -> Result<()> {
        self.seat_pages(Kind::Rs, pages.iter().map(|&page| u64::from(page)))
    }

    /// The address of `bytes` at `offset` of rs buffer page `page`'s slot.
    pub fn rs_span(&self, page: u32, offset: u64, bytes: u64) -> Result<u64> {
        self.chunks.span(Kind::Rs, page, offset, bytes)
    }

    /// Copies whole slots of the shape `shape` names, `(src, dst)`, on
    /// `stream`: a shared tail slot's rows onto the slot a fire then writes.
    pub fn copy_whole_slots(
        &mut self,
        stream: *mut core::ffi::c_void,
        shape: engine::chunk::ShapeId,
        pairs: &[[u32; 2]],
    ) -> Result<()> {
        let kind = Self::kind_of(shape)?;
        let bytes = self
            .chunks
            .layout()
            .shape(kind)
            .map(|shape| shape.slot_bytes)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("a slot copy of a {} in a load that lays none", kind.name()),
                )
            })?;
        self.seat_pages(kind, pairs.iter().flatten().map(|&slot| u64::from(slot)))?;
        self.fence_all_swaps(stream)?;
        for &[src, dst] in pairs {
            if src == dst {
                continue;
            }
            crate::device::copy_d2d(
                stream,
                self.chunks.span(kind, dst, 0, bytes)?,
                self.chunks.span(kind, src, 0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    /// The state rows of `slot`, backed first; none when the load keeps no state.
    pub fn state_spans(&mut self, slot: u32) -> Result<Vec<(u64, u64)>> {
        if !self.state_row(slot)? {
            return Ok(Vec::new());
        }
        let record = self.ensure_record(slot)?;
        self.record_spans()
            .into_iter()
            .map(|(piece, offset, bytes)| {
                let (kind, _) = self.piece_kind(piece as usize)?;
                Ok((
                    self.chunks
                        .span(kind, record[piece as usize], offset, bytes)?,
                    bytes,
                ))
            })
            .collect()
    }

    /// Seats and backs the full-context kv pages `pages` (a lane's working
    /// set's reserve its own page list names, `serve::pages`).
    pub fn seat_kv_pages(&mut self, pages: &[u32]) -> Result<()> {
        self.ensure_kv_pages(pages)
    }

    /// The address kv page `page` of list `list` starts at, seated.
    pub fn page_base(&self, list: u8, page: u32) -> Result<u64> {
        self.chunks.span(Kind::paged(list), page, 0, 1)
    }

    /// The planes a full-context kv page holds, as [`Self::copy_kv`] copies
    /// them: where each lies in the page and the bytes a token holds in it.
    #[must_use]
    pub fn page_planes(&self) -> Vec<kernels_cuda::runtime::channel::PagePlane> {
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: false,
                dtype,
                keys_width,
                values_width,
                values_plane,
                offsets,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            for (at, &offset) in offsets.iter().enumerate().take(values_plane + 1) {
                let width = if at == 0 { keys_width } else { values_width };
                planes.push(kernels_cuda::runtime::channel::PagePlane {
                    offset,
                    cell: width * element,
                });
            }
        }
        planes
    }

    /// Copies `moves` in the full rows and `windowed` (whole pages) in the
    /// windowed ones, whose page ids are their own.
    pub fn copy_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        moves: &[KvRun],
        windowed: &[KvRun],
    ) -> Result<()> {
        if moves.is_empty() && windowed.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u64;
        for span in moves.iter().chain(windowed) {
            for token in [span.src_token, span.dst_token] {
                let end = u64::from(token) + u64::from(span.tokens);
                if end > page_size {
                    return Err(Fault::Ceiling {
                        what: "token slots in one kv page",
                        need: end,
                        have: page_size,
                    });
                }
            }
        }
        for span in moves {
            highest = highest.max(u64::from(span.src_page.max(span.dst_page)) + 1);
        }
        self.kv_within(highest)?;
        self.seat_pages(
            Kind::List,
            moves
                .iter()
                .flat_map(|span| [u64::from(span.src_page), u64::from(span.dst_page)]),
        )?;
        self.ensure_windowed(
            windowed
                .iter()
                .flat_map(|span| [u64::from(span.src_page), u64::from(span.dst_page)]),
        )?;
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: in_window,
                list,
                dtype,
                keys_width,
                values_width,
                values_plane,
                offsets,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            let moves = if in_window { windowed } else { moves };
            for (at, &offset) in offsets.iter().enumerate().take(values_plane + 1) {
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src = offset + u64::from(span.src_token) * cell;
                    let dst = offset + u64::from(span.dst_token) * cell;
                    if span.src_page == span.dst_page && src == dst {
                        continue;
                    }
                    crate::device::copy_d2d(
                        stream,
                        self.chunks
                            .span(Kind::paged(list), span.dst_page, dst, bytes)?,
                        self.chunks
                            .span(Kind::paged(list), span.src_page, src, bytes)?,
                        usize::try_from(bytes).unwrap_or(0),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The paged planes a kv page spans and the bytes it holds in each: the
    /// full-context rows' key and value planes and the compressor slabs
    /// indexed by the same pages. Windowed rows name their own pages.
    fn paged_planes(&self) -> Vec<(PlaneAt, u64)> {
        let mut planes = self.kv_planes(false);
        planes.extend(self.pooled_planes(false));
        planes
    }

    /// The compressor slabs keyed by a windowed row's pages (or not), and
    /// the bytes a page holds in each.
    fn pooled_planes(&self, windowed: bool) -> Vec<(PlaneAt, u64)> {
        let page_size = self.paging.page_size;
        self.pooled
            .iter()
            .filter(|row| row.windowed == windowed)
            .flat_map(|row| {
                let bytes = row.plane_bytes(page_size);
                (0..2).map(move |at| {
                    (
                        PlaneAt {
                            kind: Kind::paged(row.list),
                            offset: row.offset + at * bytes,
                        },
                        bytes,
                    )
                })
            })
            .collect()
    }

    /// The windowed planes a windowed page spans: the windowed kv rows' and
    /// the compressor slabs their pages key.
    fn windowed_planes(&self) -> Vec<(PlaneAt, u64)> {
        let mut planes = self.kv_planes(true);
        planes.extend(self.pooled_planes(true));
        planes
    }

    /// The planes of the kv rows that are (or are not) `windowed`, and the
    /// bytes a page holds in each.
    fn kv_planes(&self, windowed: bool) -> Vec<(PlaneAt, u64)> {
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: in_window,
                list,
                values_plane,
                offsets,
                ..
            } = *shape
            else {
                continue;
            };
            if in_window != windowed {
                continue;
            }
            for (at, &offset) in offsets.iter().enumerate().take(values_plane + 1) {
                planes.push((
                    PlaneAt {
                        kind: Kind::paged(list),
                        offset,
                    },
                    shape.page_bytes(at, self.paging.page_size),
                ));
            }
        }
        planes
    }

    /// The address of page `page`'s `bytes` in `plane`; the page is seated
    /// and backed by the caller.
    fn plane_span(&self, plane: PlaneAt, page: u32, bytes: u64) -> Result<u64> {
        self.chunks.span(plane.kind, page, plane.offset, bytes)
    }

    /// Rewrites the page ids of a fire's table (list `list`'s: 0 the kv
    /// list, `n` window list `n - 1`)
    /// as the ids the paged kernels read their slots by; the fire's commit
    /// has seated them.
    pub fn page_ids(&self, list: u8, ids: &mut [i32]) -> Result<()> {
        self.chunks.page_ids(Kind::paged(list), ids)
    }

    /// Whether the load keeps a recurrent record for state slot `slot`;
    /// refused past the slots it seats, where lanes do not name slot refs.
    fn state_row(&self, slot: u32) -> Result<bool> {
        if self.chunks.by_refs() {
            return Ok(self.has_record());
        }
        slab_row(self.has_record(), self.paging.slots, slot)
    }

    /// From here on lanes name their slots by slot ref
    /// ([`chunks::Chunks::take_refs`]).
    pub fn take_refs(&mut self) {
        self.chunks.take_refs();
    }

    /// Whether lanes name slot refs.
    #[must_use]
    pub fn by_refs(&self) -> bool {
        self.chunks.by_refs()
    }

    /// See `Pool::commit`.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        match self.chunks.commit_segments(&mut self.pool, segments)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// See `Pool::release`: the activation's span stays.
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        // No move still landing reads or writes a chunk that goes.
        self.settle_swaps()?;
        self.chunks.release_segments(&mut self.pool, segments)
    }

    /// The chunks the fit left the cache rows, of the range's.
    fn fitted_chunks(&self) -> u32 {
        self.fitted.min(self.chunks.total())
    }

    /// What `evacuation` takes, checked against the pool with nothing
    /// moved ([`chunks::Chunks::prepare_evacuation`]).
    pub fn prepare_evacuation(
        &self,
        evacuation: &engine::chunk::Evacuation,
    ) -> Result<chunks::Evacuated> {
        self.chunks.prepare_evacuation(evacuation)
    }

    /// `look` of the pool as `evacuated` would leave it: a dry run of what
    /// reads where chunks lie, the pool as it stands after.
    pub fn as_if<R>(
        &mut self,
        evacuated: &mut chunks::Evacuated,
        look: impl FnOnce(&Pools) -> R,
    ) -> R {
        self.chunks.swap_table(evacuated);
        let seen = look(self);
        self.chunks.swap_table(evacuated);
        seen
    }

    /// Moves the chunks `evacuated` names (see `Pool::evacuate`): the
    /// copy enqueued behind every move still landing, then the table. A
    /// copy that cannot be enqueued leaves the pool as it was.
    pub fn move_chunks(
        &mut self,
        stream: *mut core::ffi::c_void,
        evacuated: chunks::Evacuated,
    ) -> Result<()> {
        self.chunks.evacuate(stream, evacuated)
    }

    /// The device's copies of what follows the table, after
    /// [`Pools::move_chunks`]: the copy's own completion held (so
    /// `release_segments` settles it before the evacuated segment goes, and
    /// no later fire need retire first), and each cached atom's table entry
    /// where its chunk moved. It fails only where the device does.
    pub fn restage_after_move(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        self.hold_until(stream, Vec::new(), Vec::new())?;
        self.stage_row_tables(stream)?;
        self.stage_expert_tables(stream)
    }

    /// The chunk table as it stands (`Pool::chunk_table`).
    #[must_use]
    pub fn chunk_table(&self) -> &engine::chunk::ChunkTable {
        self.chunks.table()
    }

    /// The state pool as the runtime counts it, for fires of up to
    /// `max_tokens`: the chunks the fit left the cache rows, and the
    /// activation's right above them, the chunks past the fit skipped
    /// ([`chunks::Chunks::state_fitted`]).
    ///
    /// `rungs`, each smaller rung's tokens and the bytes at the bottom of
    /// the activation's span state may take at it (backed already), each
    /// stated as the activation's chunks less those.
    pub fn chunk_facts(
        &mut self,
        max_tokens: u32,
        rungs: &[(u32, u64)],
    ) -> Result<engine::chunk::ChunkFacts> {
        let fitted = self.stated.map_or(self.fitted_chunks(), |stated| {
            stated.max(self.fitted_chunks())
        });
        self.chunks.state_fitted(fitted, self.contiguous)?;
        // The state's chunks of the segment the stated activation starts in
        // are backed from here on, as the runtime counts them.
        match self.chunks.hold_boundary(&mut self.pool)? {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        let chunk = self.chunks.layout().chunk_bytes.max(1);
        let act = self.chunks.act_chunks();
        let rungs: Vec<(u32, u32)> = rungs
            .iter()
            .map(|&(tokens, under)| {
                let lent = u32::try_from(self.lent_bytes(under) / chunk).unwrap_or(u32::MAX);
                (tokens, act - lent)
            })
            .collect();
        let mut facts = self
            .chunks
            .facts(max_tokens, !elastic::mapped_on_demand(), &rungs);
        // Where the load seated the weights itself, for a runtime to adopt.
        if self.preseated {
            for shape in &mut facts.weights {
                shape.seated = self
                    .row_caches
                    .iter()
                    .map(|cache| (cache.shape.id, &cache.slots))
                    .chain(
                        self.expert_caches
                            .iter()
                            .map(|cache| (cache.shape.id, &cache.slots)),
                    )
                    .find(|(id, _)| *id == shape.id)
                    .and_then(|(_, slots)| slots.first().copied().flatten());
            }
        }
        facts.host_chunks = self.tier_chunks();
        facts.disk_chunks = self.disk_chunks();
        facts.disk_kept = self.disk_kept();
        Ok(facts)
    }

    /// Refuses a page count past the pool's.
    fn kv_within(&self, pages: u64) -> Result<()> {
        let capacity = self.paging.pages();
        if pages > capacity && !self.chunks.by_refs() {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: pages,
                have: capacity,
            });
        }
        Ok(())
    }

    /// Backs every page and slot the pools declare, at load, and keeps them
    /// so: backing a page as a fire first touches it takes
    /// the driver's lock, which a dying process on the host holds for
    /// seconds, and a worker taking a dead peer's turns needs new pages then
    /// above all.
    pub fn back_whole(&mut self) -> Result<()> {
        // Every segment the fitted pool's chunks lie in, the last as far
        // as they reach: the range is reserved for more than the fit left
        // the cache rows.
        // Backed as far as the fit holds; limited to what the runtime is
        // stated, where that is more: what the fit keeps back is backed
        // when the runtime comes to take it.
        let fitted = self.fitted_chunks();
        let stated = self.stated.map_or(fitted, |stated| stated.max(fitted));
        self.chunks.limit(stated);
        // Where the fit keeps chunks back, only the segments wholly under
        // them: the last the fitted chunks reach into would back some of
        // what is kept.
        let segments = if stated > fitted {
            self.chunks.segments_under(u64::from(fitted))
        } else {
            self.chunks.segments_for(u64::from(fitted))
        };
        // The fit that sized these chunks ran in this same step of the
        // load, so the card holds them: a refusal here is a fit that
        // counted wrong, and the load says so.
        match self.chunks.commit_segments(&mut self.pool, segments)? {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        self.whole = true;
        Ok(())
    }

    /// The kv planes as another worker reads them: every declared page, one
    /// region per paged plane, a page `p` at `base + p * page_stride`. A
    /// windowed row is refused: its pages are named by its own window list, not
    /// by the kv pages a peer reads.
    pub fn export_kv(&mut self, ordinal: u32) -> Result<ExportedKv> {
        if !self.windowed_planes().is_empty() {
            return Err(Fault::Residency(
                "a windowed kv row names its pages by its own window list, so it cannot be \
                 handed to another worker by the kv pages it reads"
                    .to_string(),
            ));
        }
        if !self.chunks.in_one_run() {
            return Err(Fault::program(
                "store::chunks",
                "a kv pool exported whose chunks do not lie in one run from its base: say it is \
                 exported before the load's last fit (`Pools::keep_contiguous`)",
            ));
        }
        if !self.whole {
            self.back_whole()?;
        }
        // What the fit backs: the range reserves addresses past it.
        let backed = self.chunk_bytes(self.fitted_chunks());
        let mut regions = Vec::new();
        // A kv plane's region is the pool from the plane's offset in a
        // slot on: the page a slot ref `r` names starts `r` strides in,
        // while its chunk lies where it was opened.
        let grain = self
            .chunks
            .layout()
            .shape(Kind::List)
            .map_or(0, |shape| shape.grain);
        for (plane, _) in self.paged_planes() {
            regions.push(engine::KvRegion {
                base: self.chunks.base() + plane.offset,
                len: backed.saturating_sub(plane.offset),
                page_stride: grain,
                domain: engine::MemoryDomain::Device {
                    ordinal,
                    reach: engine::DeviceReach::Peer,
                },
            });
        }
        let (layers, heads, head_dim, dtype) = self
            .shapes
            .iter()
            .find_map(|shape| match *shape {
                Shape::Kv {
                    windowed: false,
                    dtype,
                    keys_width,
                    head_stride,
                    ..
                } => Some((dtype, keys_width, head_stride)),
                _ => None,
            })
            .map(|(dtype, keys_width, head_stride)| {
                let layers = self
                    .shapes
                    .iter()
                    .filter(|shape| matches!(shape, Shape::Kv { .. }))
                    .count();
                (
                    u32::try_from(layers).unwrap_or(u32::MAX),
                    u32::try_from(keys_width / head_stride.max(1)).unwrap_or(u32::MAX),
                    u32::try_from(head_stride).unwrap_or(u32::MAX),
                    dtype,
                )
            })
            .ok_or_else(|| {
                Fault::Residency("this load pages no full-attention kv row to export".to_string())
            })?;
        Ok(ExportedKv {
            regions,
            layers,
            heads,
            head_dim,
            dtype,
            page_size: self.paging.page_size,
        })
    }

    /// The bytes one windowed kv page holds over every windowed plane.
    #[must_use]
    pub fn window_page_bytes(&self) -> u64 {
        self.windowed_planes().iter().map(|&(_, bytes)| bytes).sum()
    }

    /// The bytes one kv page holds over every paged plane.
    #[must_use]
    pub fn paged_page_bytes(&self) -> u64 {
        self.paged_planes().iter().map(|&(_, bytes)| bytes).sum()
    }

    /// `slot`'s recurrent state, read on `stream` behind the fires that
    /// wrote it.
    pub fn state_bytes(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<Vec<u8>> {
        if !self.state_row(slot)? {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (at, bytes) in self.state_spans(slot)? {
            let from = out.len();
            out.resize(from + usize::try_from(bytes).unwrap_or(0), 0);
            crate::device::copy_d2h_on(stream, at, &mut out[from..])?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if !self.state_row(slot)? {
            return Ok(());
        }
        let record = self.ensure_record(slot)?;
        for (at, piece) in record.into_iter().enumerate() {
            let (kind, bytes) = self.piece_kind(at)?;
            let at = self.chunks.span(kind, piece, 0, bytes)?;
            let len = usize::try_from(bytes).unwrap_or(0);
            match stream {
                Some(stream) => crate::device::zero_span_on(stream, at, len)?,
                None => crate::device::zero_span(at, len)?,
            }
        }
        Ok(())
    }

    /// Seats and backs the kv pages `pages`, and the compressor slabs' rows
    /// under them.
    fn ensure_kv_pages(&mut self, pages: &[u32]) -> Result<()> {
        let highest = pages.iter().map(|&page| u64::from(page) + 1).max();
        let Some(highest) = highest else {
            return Ok(());
        };
        self.kv_within(highest)?;
        self.seat_pages(Kind::List, pages.iter().map(|&page| u64::from(page)))
    }

    pub fn commit_frame(
        &mut self,
        demand: engine::frame::Demand,
        kv_ranges: &[(u64, u64)],
        window_pages: &[Vec<u64>],
    ) -> Result<()> {
        self.seat_pages(
            Kind::List,
            kv_ranges
                .iter()
                .flat_map(|&(first, count)| first..first.saturating_add(count)),
        )?;
        let _ = demand;
        if self.chunks.by_refs() {
            // Each window list's slot refs are its own.
            for (window, pages) in window_pages.iter().enumerate() {
                self.seat_pages(Kind::Window(window as u8), pages.iter().copied())?;
            }
            Ok(())
        } else {
            self.ensure_windowed(window_pages.iter().flatten().copied())
        }
    }
}

/// The step `list`'s page ids count in.
fn grain_of(chunks: &chunks::Chunks, list: u8) -> u64 {
    chunks
        .layout()
        .shape(Kind::paged(list))
        .map_or(1, |shape| shape.grain)
}

/// The ring hook this engine answers the windowed recipe with
/// (`poem_compiler::chunk::RingLayout`): its kernels read a window's rows by
/// one page table of window-buffer slots, so its ring is `ceil(W / P) + 1`
/// pages of the widest window `W`, each laid exactly as a buffer slot, every
/// piece at a chunk's start (a page then has an id of the buffer's grain).
/// A layer's share is its rows of every page, the last padded to the whole.
struct CudaRings;

impl poem_compiler::chunk::RingLayout for CudaRings {
    fn ring_region(
        &self,
        layers: &[poem_compiler::chunk::WindowedRow],
        buffer: &poem_compiler::chunk::WindowBufferSlot<'_>,
    ) -> poem_compiler::chunk::RingRegion {
        let p = u64::from(buffer.tokens.max(1));
        let widest = layers
            .iter()
            .map(|layer| u64::from(layer.window))
            .max()
            .unwrap_or(0);
        let pages = widest.div_ceil(p) + 1;
        let bytes = pages * buffer.bytes;
        let mut layer_bytes: Vec<u64> = layers
            .iter()
            .map(|layer| {
                pages
                    * buffer
                        .rows
                        .iter()
                        .filter(|run| layer.rows.contains(&run.row))
                        .map(|run| p * run.per_token)
                        .sum::<u64>()
            })
            .collect();
        let laid: u64 = layer_bytes.iter().sum();
        if let Some(last) = layer_bytes.last_mut() {
            *last += bytes.saturating_sub(laid);
        }
        poem_compiler::chunk::RingRegion {
            bytes,
            atom: buffer.bytes,
            placement: poem_compiler::chunk::RingPlacement::ChunkStart,
            layer_bytes,
        }
    }
}

/// How `trace`'s kv rows sit in the state pool's chunks: the compiler's
/// recipe for a server card, its token slots this paging's pages. Where the
/// paging keeps no window (the widest spans the context) every row is a list
/// row.
fn lay(
    trace: &Trace,
    paging: &Paging,
    budgets: &poem_compiler::Budgets,
    chunk_bytes: Option<u64>,
    experts: &[usize],
    rows: bool,
) -> Result<layout::Layout> {
    let context = if paging.window.is_some() {
        paging.context()
    } else {
        0
    };
    let mut server = poem_compiler::chunk::ChunkProfile::SERVER_CUDA;
    if let Some(bytes) = chunk_bytes {
        server.target_chunk_bytes = bytes;
    }
    // Where this device's kernels map a table's window, slots lay their
    // planes on the rows those maps index.
    let tma = windows_mapped();
    server.tma_rows = tma;
    let paged = server.paged(paging.page_size, context);
    let profile = poem_compiler::DeviceProfile {
        chunk: paged,
        ..poem_compiler::DeviceProfile::default()
    };
    // Each plane at the bytes its slot holds: laid scales as their atoms.
    let view = crate::weights::laid_view(trace);
    // Windowed rows as ring + buffer: each windowed
    // layer's last folded rows a ring in the lane's record, its unfolded
    // rows in the window buffer.
    let recipe = poem_compiler::chunk::plan_windowed(&view, budgets, &profile, &CudaRings)
        .map_err(|why| Fault::Unbound {
            what: format!("a chunk recipe for this load's cache rows: {why}"),
        })?;
    let banks: Vec<(usize, u64, u64)> = if experts.is_empty() {
        Vec::new()
    } else {
        let bytes = crate::weights::laid_bytes(trace)?;
        // A plane held on the device beside its bank takes no slot bytes.
        let beside = experts
            .iter()
            .flat_map(|&param| layout::beside_planes(trace, param))
            .map(|plane| (plane as usize, 0, 0));
        // Each plane's TMA rows where windows are mapped (the recipe laid
        // the slot by them), else none.
        let rows = |param: usize| {
            if tma {
                poem_compiler::chunk::weights::tma_rows(trace, param, experts)
            } else {
                0
            }
        };
        experts
            .iter()
            .map(|&param| (param, bytes[param], rows(param)))
            .chain(beside)
            .collect()
    };
    let streamed = if rows {
        layout::streamed_rows(trace)
    } else {
        Vec::new()
    };
    layout::Layout::of(&recipe, &streamed, &banks)
}

/// The chunks a pool of `pages` kv pages takes: the null chunk, the pages'
/// and the windowed pages beside them.
fn chunks_at(layout: &layout::Layout, paging: &Paging, pages: u64, slots: u32) -> u32 {
    // The null chunk, and a chunk of each guest shape: an instance's
    // tables and a ring, which every working set served binds.
    let chunks = 1
        + GUEST_SHAPES
        + layout.chunks_for(Kind::List, pages)
        + (0..layout.windows.len())
            .map(|window| {
                layout.chunks_for(Kind::Window(window as u8), paging.window_pages_at(pages))
            })
            .sum::<u64>()
        + layout.chunks_for(
            Kind::Fixed,
            u64::from(slots)
                * layout
                    .fixed
                    .as_ref()
                    .map_or(1, |fixed| u64::from(fixed.pieces)),
        )
        + layout.chunks_for(
            Kind::Ring,
            u64::from(slots)
                * layout
                    .ring
                    .as_ref()
                    .map_or(0, |ring| u64::from(ring.pieces)),
        );
    u32::try_from(chunks).unwrap_or(u32::MAX)
}

/// The most kv pages, from `floor` up to `ceiling`, that `kv_chunks`
/// chunks hold beside the shapes every working set binds and `slots`
/// records ([`chunks_at`]).
fn pages_in(
    layout: &layout::Layout,
    paging: &Paging,
    kv_chunks: u32,
    slots: u32,
    floor: u64,
    ceiling: u64,
) -> u64 {
    longest(floor, ceiling.saturating_add(1), |pages| {
        chunks_at(layout, paging, pages, slots) <= kv_chunks
    })
}

/// The guest shapes: `Instance` and `Channel`.
const GUEST_SHAPES: u64 = 2;

/// Whether this device's kernels map a located table's window by TMA
/// (`kernels_cuda::linear::moe::windows_mapped`); none does off CUDA.
fn windows_mapped() -> bool {
    #[cfg(feature = "cuda")]
    {
        kernels_cuda::linear::moe::windows_mapped()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Whether a slab holds a row for the seat. The runtime seats a kv-only
/// model with no ceiling, so the slot count only binds a model with a slab.
fn slab_row(has_state: bool, slots: u32, slot: u32) -> Result<bool> {
    if !has_state {
        return Ok(false);
    }
    if slot >= slots {
        return Err(Fault::Ceiling {
            what: "recurrent slots",
            need: u64::from(slot) + 1,
            have: u64::from(slots),
        });
    }
    Ok(true)
}

fn refuse(pool: &PhysicalPool, outcome: Commit) -> Fault {
    let page = pool.page_bytes();
    match outcome {
        Commit::Committed => Fault::program(
            "store::commit",
            "a committed outcome reached the refusal path".to_string(),
        ),
        Commit::Exhausted { growth, spare } => Fault::OutOfMemory {
            need: growth.saturating_mul(page),
            have: spare.saturating_mul(page),
        },
        Commit::Impossible { required, ceiling } => Fault::Ceiling {
            what: "bytes of elastic device memory",
            need: required.saturating_mul(page),
            have: ceiling.saturating_mul(page),
        },
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

/// The widest window a windowed kv row is declared with. Every read of such
/// a row must look through a window no wider, since the pages behind it
/// read the null page.
pub fn window_of(trace: &Trace) -> Result<Option<u32>> {
    for node in &trace.nodes {
        let Some(read) = kv::reads(&node.op) else {
            continue;
        };
        let Some(Def::Cache(row)) = trace.values.get(read.cache.0 as usize).map(|v| &v.def) else {
            continue;
        };
        if let Some(CacheRow::Kv {
            name,
            window: Some(held),
            ..
        }) = trace.caches.get(*row as usize)
            && read.window.is_none_or(|read| read > *held)
        {
            return Err(Fault::Unbound {
                what: format!(
                    "cache `{name}`, which holds a window of {held} tokens while `{}` reads \
                     it through {:?}",
                    node.op.name(),
                    read.window
                ),
            });
        }
    }
    Ok(trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { window, .. } => *window,
            CacheRow::State { .. } => None,
        })
        .max())
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn wide(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bodies_are_held_ahead_of_the_kv_fill_and_refused_by_name_past_the_floor() {
        const MIB: u64 = 1 << 20;
        // Qwen3.5-0.8B's check load on the L40S: 39602 MiB live past the
        // weights and the span, one 32768-token sequence's 512 MiB claimed.
        let (live, claim) = (39_602 * MIB, 512 * MIB);
        let room = |(bodies, programs): (u64, u64)| live - bodies - programs;

        // Still to arm: every byte past the claim is the bodies', and the kv
        // pool is fitted to the claim alone until they are captured.
        let measuring = holds(claim, live, MEASURING, 0).expect("one sequence fits");
        assert_eq!(measuring, (live - claim, 0));
        assert_eq!(
            room(measuring),
            claim,
            "the kv fill takes nothing of their room"
        );

        // Measured: what they took is held before the kv pool fills the rest.
        let measured = holds(claim, live, 1010 * MIB, 0).expect("they fit");
        assert_eq!(measured, (1010 * MIB, 0));
        assert_eq!(room(measured), live - 1010 * MIB);
        assert_eq!(
            holds(claim, live, live - claim, 0),
            Ok((live - claim, 0)),
            "a figure that exactly fits is held"
        );

        // Weights, span, bodies and the kv floor past the card: refused,
        // with what the card has past the claim, never held down.
        assert_eq!(holds(claim, live, live - claim + 1, 0), Err(live - claim));
        assert_eq!(
            holds(live + MIB, live, 1, 0),
            Err(0),
            "a floor past the card leaves the bodies nothing"
        );
        assert_eq!(
            holds(live + MIB, live, MEASURING, 0),
            Ok((0, 0)),
            "and still to arm, the fit's own floor check refuses it ([`Pools::fit`])"
        );

        // The guests' programs come after the bodies.
        assert_eq!(
            holds(claim, live, 100 * MIB, live),
            Ok((100 * MIB, live - claim - 100 * MIB))
        );
    }

    #[test]
    fn store_every_case() {
        a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse();
        a_slot_copy_names_the_guest_shapes();
        stated_tokens_are_exactly_their_pages_beside_the_other_shapes();
        stated_tokens_are_backed_whole_at_every_seat_count();
    }

    // The claim is what backing the stated chunks grows the range by, and
    // the room is counted once: what the pool is sized for is what it backs,
    // at every seat count.
    fn stated_tokens_are_backed_whole_at_every_seat_count() {
        const MIB: u64 = 1 << 20;
        // A kv page of 16 bf16 tokens' keys and values over 8 heads of 128.
        let kv_page = |layers: u64| layers * 2 * 8 * 128 * 16 * 2;
        // Qwen3-0.6B (28 layers), nine pages a chunk, with a recurrent
        // record of 3 MiB in two pieces so the records the seats take are
        // chunks too; Qwen3-8B (36 layers), eight pages a chunk of 18 MiB,
        // dense.
        let small = kv_page(28);
        let large = kv_page(36);
        let cases: [(&str, u64, u64, bool, &[u32], &[u64]); 2] = [
            (
                "Qwen3-0.6B",
                small,
                9 * small + 4096,
                true,
                &[1, 4, 16, 32],
                &[16_000, 16_384],
            ),
            ("Qwen3-8B", large, 8 * large, false, &[0, 1, 16], &[53_853]),
        ];
        let mut short_before = 0;
        for (model, page_bytes, chunk, record, seat_counts, token_counts) in cases {
            let shape = |id, tokens, slot_bytes: u64| layout::SlotShape {
                id,
                tokens,
                slot_bytes,
                per_chunk: u32::try_from(chunk / slot_bytes).expect("a count"),
                grain: slot_bytes,
                window: None,
                rows: Vec::new(),
                pools: Vec::new(),
            };
            let layout = layout::Layout {
                chunk_bytes: chunk,
                segment_chunks: 4,
                list: Some(shape(layout::LIST, 16, page_bytes)),
                windows: Vec::new(),
                fixed: record.then(|| layout::FixedShape {
                    slot: shape(layout::FIXED, 0, 3 * MIB / 2),
                    pieces: 2,
                    rows: Vec::new(),
                    tables: Vec::new(),
                }),
                ring: None,
                rs: None,
                instance: shape(layout::INSTANCE, 0, layout::INSTANCE_SLOT),
                channel: shape(layout::CHANNEL, 0, layout::CHANNEL_SLOT),
                rows: Vec::new(),
                experts: Vec::new(),
            };
            // Ranges mapped in units no multiple of the chunk: the last unit
            // of a pool reaches past its last chunk.
            for unit in [32 * MIB, 64 * MIB] {
                for &seats in seat_counts {
                    for &tokens in token_counts {
                        let at = format!(
                            "{model}, {seats} seats, {tokens} tokens, units of {} MiB",
                            unit >> 20
                        );
                        if stated_backs_what_it_asks(&layout, unit, seats, tokens, &at) {
                            short_before += 1;
                        }
                    }
                }
            }
        }
        assert!(short_before > 0, "the chunks' bytes alone came short");
    }

    /// Asserts a pool of `tokens` stated tokens beside `seats` records, in
    /// a range mapped in units of `unit`, is backed whole by the fit's
    /// arithmetic; returns whether its chunks' bytes alone came short.
    fn stated_backs_what_it_asks(
        layout: &layout::Layout,
        unit: u64,
        seats: u32,
        tokens: u64,
        at: &str,
    ) -> bool {
        const MIB: u64 = 1 << 20;
        const PAGE: u64 = elastic::LOGICAL_PAGE_BYTES;
        let units = |bytes: u64| bytes.div_ceil(unit) * unit;
        let paging = Paging::of(16, 4096, seats, 1).expect("a paging");
        let kv = StatedKv::of(layout, &paging, tokens);
        // The pages its chunks hold are the pages asked, counted as the fit
        // counts them: beside the same records.
        assert_eq!(
            pages_in(layout, &paging, kv.kv_chunks, kv.records, 1, kv.pages),
            kv.pages,
            "{at}"
        );
        assert!(
            pages_in(layout, &paging, kv.kv_chunks - 1, kv.records, 1, kv.pages) < kv.pages,
            "{at}: a chunk fewer holds fewer"
        );
        // A fit before arming: the activation's span (a held prefix backed,
        // the rest still to back) and the null chunk's unit backed; the
        // bodies hold every byte past the claim; the card has GiBs free
        // past what the range backs.
        let activation = 40 * MIB;
        let unheld = 24 * MIB;
        let backed = unit + (activation - unheld);
        let free = 24 * 1024 * MIB + 12_345;
        let programs = 96 * MIB;
        let live = free + backed - activation;
        let growth = (units(kv.bytes()) - unit) / PAGE;
        let claim = claim_of(growth, PAGE, backed, activation - unheld);
        for bodies in [MEASURING, 512 * MIB] {
            let (bodies, programs) =
                holds(claim, live, bodies, programs).expect("the card holds them");
            let room = live - bodies - programs;
            assert!(
                kv.admit(room, claim, unit).is_ok(),
                "{at}: the claim is admitted"
            );
            let have = growth_room(free, unheld, bodies + programs, PAGE);
            assert!(growth <= have, "{at}: backing every chunk stated fits");
            if bodies == MEASURING {
                assert_eq!(growth, have, "{at}: and is all it was held");
            }
        }
        // A card short of the claim refuses by the key, with what backing
        // takes.
        let refused = kv
            .admit(claim - 1, claim, unit)
            .expect_err("a byte short refuses");
        assert!(
            refused
                .to_string()
                .contains(&format!("backs in {claim} bytes")),
            "{refused}"
        );
        // Claimed as the chunks' bytes, the same pool came short.
        let (bodies, programs) = holds(kv.bytes(), live, MEASURING, programs).expect("held");
        growth > growth_room(free, unheld, bodies + programs, PAGE)
    }

    // `kv_device_tokens` sized from outside the fit missed the list's
    // chunk rounding and the chunks the other shapes take: 3128 pages where
    // 3248 were asked. The fit counts them itself.
    fn stated_tokens_are_exactly_their_pages_beside_the_other_shapes() {
        const CHUNK: u64 = 2 << 20;
        let shape = |id, tokens, per_chunk: u32| layout::SlotShape {
            id,
            tokens,
            slot_bytes: CHUNK / u64::from(per_chunk),
            per_chunk,
            grain: CHUNK / u64::from(per_chunk),
            window: None,
            rows: Vec::new(),
            pools: Vec::new(),
        };
        let layout = layout::Layout {
            chunk_bytes: CHUNK,
            segment_chunks: 1,
            // Seven pages a chunk: no page count of the test's is whole
            // chunks.
            list: Some(shape(layout::LIST, 16, 7)),
            windows: Vec::new(),
            fixed: None,
            ring: None,
            rs: None,
            instance: shape(layout::INSTANCE, 0, 512),
            channel: shape(layout::CHANNEL, 0, 2048),
            // Ten blocks of a streamed table, three a chunk: four chunks.
            rows: vec![layout::RowsShape {
                id: engine::chunk::ShapeId(layout::WEIGHTS),
                param: 0,
                slot_bytes: CHUNK / 3,
                per_chunk: 3,
                block_rows: 1,
                row_bytes: CHUNK / 3,
                rows: 10,
                blocks: 10,
                floor: 1,
            }],
            experts: Vec::new(),
        };
        let paging = Paging::of(16, 4096, 1, 1).expect("a paging");
        let kv = StatedKv::of(&layout, &paging, 3248 * 16);
        // 3248 pages, 464 list chunks (3248 / 7 rounded up) beside the null
        // chunk and a chunk of each guest shape, and the weights' four.
        assert_eq!(
            (kv.pages, kv.kv_chunks, kv.weight_chunks, kv.chunks()),
            (3248, 1 + 2 + 464, 4, 471)
        );
        assert_eq!(kv.bytes(), 471 * CHUNK);
        // The list's chunks hold the asked pages and no fewer: the fit's
        // page search over them stops at the asked count.
        assert!(chunks_at(&layout, &paging, 3248, 0) <= kv.kv_chunks);
        // A token past a whole page is a page more, and here a chunk more.
        let past = StatedKv::of(&layout, &paging, 3248 * 16 + 1);
        assert_eq!((past.pages, past.kv_chunks), (3249, 1 + 2 + 465));
        // The load is refused by the key's name a byte short of the room,
        // and admitted at it.
        assert!(kv.admit(kv.bytes(), kv.bytes(), CHUNK).is_ok());
        let refused = kv
            .admit(kv.bytes() - 1, kv.bytes(), CHUNK)
            .expect_err("a byte short refuses");
        assert!(
            refused.to_string().contains("kv_device_tokens = 51968")
                && refused.to_string().contains("3248 kv pages"),
            "{refused}"
        );
    }

    // A process that had bound a guest program could not be suspended: its
    // role slots' copy to the host tier named a shape the load "did not
    // state".
    fn a_slot_copy_names_the_guest_shapes() {
        assert_eq!(Pools::kind_of(layout::INSTANCE).ok(), Some(Kind::Instance));
        assert_eq!(Pools::kind_of(layout::CHANNEL).ok(), Some(Kind::Channel));
        // The id after the guests' is no shape of a load.
        let past = engine::chunk::ShapeId(layout::CHANNEL.0 + 1);
        assert!(Pools::kind_of(past).is_err());
    }

    // gemma-4 (no slab) with 256 slots reserved seated a lane at 312.
    fn a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse() {
        assert!(!slab_row(false, 256, 312).expect("no slab, no ceiling"));
        assert!(slab_row(true, 256, 255).expect("the last row"));
        assert!(matches!(
            slab_row(true, 256, 312),
            Err(Fault::Ceiling {
                what: "recurrent slots",
                need: 313,
                have: 256
            })
        ));
    }
}
