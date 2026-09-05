//! `Run`: one fire's dispatch state, and the resolution that turns a plan id into a device handle. Long-lived state is borrowed; fire-lived state is owned per fire.

use std::cell::Cell;

use kernels_cuda::attn::plan::{
    DecodePlan, Device, Live, MlaPlan, PrefillPlan, PrefillPlanSm90, Shape, Toggles, Workspace,
};
use kernels_cuda::linear::lora::Segments;
use kernels_cuda::linear::moe::{ExpertTable, GroupSeat};
use kernels_cuda::{Ctx, KvPool, Pad, RaggedTensor, RecurrentPool, Tensor};
use model_exec::fire::MaskSpan;
use model_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::dispatch::copy::CopyPlan;
use crate::record::Carve;
use crate::window::{Admit, At, Window, Windows};

/// One loader-resolved weight. Most rows are one dense handle; an mxfp4 bank is two device planes under one `Def::Weight` id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank — e2m1 codes beside e8m0 exponents, or MLX affine codes beside bf16 scales and zero points — resolved by [`Run::planes`], never as one tensor.
    Planes {
        codes: Tensor,
        scales: Tensor,
        /// The zero points, for an affine bank whose element is `code * scale + bias`; `None` for a scheme whose block centres itself (mxfp4).
        biases: Option<Tensor>,
        /// Which tier the group is on right now (streamed vs. resident); [`GroupSeat::RESIDENT`] for a group the store holds whole.
        seat: GroupSeat,
        /// Fragment order (`true`) vs. row-major (`false`) — read off the weight's declared dtype/place. Same bytes either way, so the wrong order answers nonsense rather than refusing.
        repacked: bool,
    },

    /// A streamed routed-expert bank: a device slab of fewer slots than the bank has experts, plus its indirection table and usage counters.
    Streamed {
        slab: Tensor,
        table: u64,
        counts: u64,
    },
}

/// Loader-resolved weights, one row per `Trace::params` entry — `Def::Weight(i)` resolves to row `i`. `None` marks a param the shell has not bound; resolving such a row is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

/// Arena slots at the compiler's offsets, `ValueId`-indexed. Op outputs and merges alike land here, so a φ resolves like any op output. Rows for ids that own no arena slot (inputs, weights, caches, structs) stay `None`.
#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

/// One resolved cache space — the storage pointer and nothing else; its geometry rides in [`FireBindings`] as declared inputs.
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space, and which geometry space's tables address it — restated here since `Run::pool` holds the pool, not the plan's cache table.
    Kv {
        /// The `CacheRow::Kv` space this row's geometry comes from.
        space: u32,
        pool: KvPool,
    },
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Trace::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The host half of one cache space's geometry, walked by `plan_decode`/`plan_prefill`. Bound only for cache spaces a plan op names.
#[derive(Clone, Debug)]
pub struct CachePlanning {
    /// Host copy of the space's `GeomKind::Indptr` contents, walked by `plan_decode`/`plan_prefill`.
    pub kv_indptr: Vec<i32>,

    /// Host copy of the space's `GeomKind::KvLen` contents — per-request kv lengths in tokens, walked by the sm90 and mla builders.
    pub kv_len: Vec<i32>,
}

/// What one attention schedule is carved for, and where it is staged. Keyed by the plan value, not the space: a family that carves two mints two plan values.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleSeat {
    /// The kv-side shape this schedule is carved at, at the fire's lanes; [`Run::planning`] narrows `num_requests` to the asking node's window. For a latent (mla) schedule, `head_dim` is the output head width.
    pub shape: Shape,

    /// The sliding window this schedule carved its kv spans for; the entries check each consumer's stated window against it.
    pub window: Option<u32>,

    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
}

/// One cache space's planning twin, cut to the window of the node asking. The slices borrow the fire-wide host twins; `shape` is rewritten to the window's own request count and origin.
#[derive(Clone, Copy, Debug)]
pub struct Planning<'a> {
    /// The window's slice of `GeomKind::Indptr`'s host contents.
    pub kv_indptr: &'a [i32],
    /// The window's slice of `GeomKind::KvLen`'s.
    pub kv_len: &'a [i32],
    /// The kv-side shape, at this window's request count and origin — or at the key's lane ceiling, on the one path that takes it.
    pub shape: Shape,
    /// The origin/extent half: always what the fire brought, even on the path that raises `shape`'s ceilings.
    pub live: Live,
    /// The row count the schedule is carved at: this window's own rows, or the sum of its classes' lattice rungs capped at the fire's bucket.
    pub rows: u32,
    /// The sliding window this schedule is carved for.
    pub window: Option<u32>,
    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
}

/// The geometry one cache space declared: the device seats the ops read, and the host planning twin beside them. Only what the plan names gets bound, so every seat is optional; resolving an unbound one panics.
#[derive(Clone, Debug, Default)]
pub struct CacheGeometry {
    /// `RuntimeInput::Geometry { kind: Indptr }`.
    pub indptr: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: Indices }`.
    pub indices: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: SeqLens }`.
    pub seq_lens: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: LastPageLen }`.
    pub last_page_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: KvLen }`.
    pub kv_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RowValid }`.
    pub row_valid: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RequestOfToken }`.
    pub request_of_token: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WritePage }`.
    pub write_page: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WriteOffset }`.
    pub write_offset: Option<Tensor>,

    /// `RuntimeInput::Mask`: this space's packed `u8` mask bits, one `rows x (held + rows)` rectangle per masked lane. `None` if none masked.
    pub mask: Option<Tensor>,

    /// The host twin the plan builders walk, bound for spaces a plan op names.
    pub planning: Option<CachePlanning>,
}

/// The dsv4 compressor state `attention.pool_gather` reads beside its cache. No IR seat: the engine binds the slabs it staged for the pooled space.
#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    /// The rolling kv window state.
    pub state_kv: Tensor,

    /// The rolling score state.
    pub state_score: Tensor,

    /// The absolute-position-embedding plane.
    pub ape: Tensor,
}

/// The engine-bound extras the cuda entries want beside the ops' named operands. No op names these; each seat is bound from fire state by the arm that uses it.
#[derive(Clone, Debug)]
pub struct FireTables {
    /// `i32`, `[lanes + 1]`: each fire lane's byte offset into the mask slab [`CacheGeometry::mask`] holds. `None` when this fire carries no mask; a masked consumer then gets a typed refusal, not a panic.
    pub mask_indptr: Option<Tensor>,

    /// The dsv4 compressor slabs, bound when a pooled space exists. One fire-wide seat: a plan carries at most one pooled space today.
    pub pool_state: Vec<(u32, PoolSlabs)>,
}

/// What one lane's recurrent state does with the buffer this fire. `None` is no move; addressing is `pages[token / page_tokens]`, not a contiguous range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsMove<'a> {
    /// Fold in the forward: nothing is copied.
    None,
    /// Scatter this lane's rows into its buffer: row `t` of the lane lands at buffer token `at + t`.
    Scatter {
        pages: &'a [u32],
        /// Which buffer token this fire's first row lands at.
        at: u32,
        /// How many of this fire's rows this lane also folds. `0` is a pure scatter; otherwise the fire also lands durable state on row `fold`, so speculation can restart without a second fire.
        fold: u32,
    },
    /// Gather this lane's buffer over its rows: buffer token `t` becomes row `t`, for as many rows as the lane has.
    Gather {
        /// The lane's buffer run, addressed exactly as `Scatter` wrote it.
        pages: &'a [u32],
        /// Which buffer token the replay starts at. A fold can only release whole covered pages, so a mid-page fold leaves survivors physically offset.
        at: u32,
    },
}

/// The buffered-activation plane, seated for one fire.
#[derive(Debug, Clone, Copy)]
pub struct RsSeat<'a> {
    /// The pool and the plan's reading of it.
    pub buffers: &'a crate::store::rs::Buffers,
    /// One verb per lane, in FIRE (seriated) order.
    pub lanes: &'a [RsMove<'a>],
}

impl RsSeat<'_> {
    /// Move one plane's rows for every lane of one window. `bounds` is the window-rebased row CSR; `lane_offset` turns a window request number into a fire lane.
    fn run(
        &self,
        stream: *mut core::ffi::c_void,
        plane: crate::store::rs::Plane,
        lane_offset: u32,
        bounds: &[i32],
        rows: Tensor,
    ) -> crate::error::Result<()> {
        let page_tokens = self.buffers.page_tokens();
        let elem = model_compiler::arena::elem_bytes(crate::store::rs::PLANE_DTYPE)
            .expect("the buffered planes are bf16, which has an element size");
        if u64::from(rows.width) != plane.width {
            return Err(crate::error::Fault::Unbound {
                what: format!(
                    "a buffered plane reserved at {} elements a token is bound {} wide this \
                     fire",
                    plane.width, rows.width
                ),
            });
        }
        for (at, pair) in bounds.windows(2).enumerate() {
            let Some(&verb) = self.lanes.get(lane_offset as usize + at) else {
                continue;
            };
            let (pages, from, count) = match verb {
                RsMove::None => continue,
                RsMove::Scatter { pages, at, .. } => (pages, at, (pair[1] - pair[0]) as u32),
                RsMove::Gather { pages, at } => (pages, at, (pair[1] - pair[0]) as u32),
            };
            if count == 0 {
                continue;
            }
            let capacity = pages.len() as u64 * u64::from(page_tokens);
            if u64::from(from) + u64::from(count) > capacity {
                return Err(crate::error::Fault::Ceiling {
                    what: "rs buffer tokens",
                    need: u64::from(from) + u64::from(count),
                    have: capacity,
                });
            }
            // One contiguous run per (page, plane): page-major layout keeps this loop memcpys rather than a strided kernel.
            let mut done = 0u32;
            while done < count {
                let token = from + done;
                let page = token / page_tokens;
                let in_page = token % page_tokens;
                let take = (page_tokens - in_page).min(count - done);
                // The run's page `page` is whatever slot the list names, not `first_page + page`.
                let page_slot = *pages.get(page as usize).ok_or(crate::error::Fault::Ceiling {
                    what: "rs buffer pages",
                    need: u64::from(page) + 1,
                    have: pages.len() as u64,
                })?;
                let slab = self.buffers.row(plane, page_slot, in_page)?;
                let rows_at = rows.ptr
                    + (u64::from(pair[0] as u32) + u64::from(done)) * plane.width * elem;
                let bytes = usize::try_from(u64::from(take) * plane.width * elem)
                    .unwrap_or(usize::MAX);
                let (dst, src) = match verb {
                    RsMove::Scatter { .. } => (slab, rows_at),
                    _ => (rows_at, slab),
                };
                crate::device::copy_d2d(stream, dst, src, bytes)?;
                done += take;
            }
        }
        Ok(())
    }
}

/// What the engine binds each fire, owned by the [`Run`] for its lifetime. The qo boundaries are not here — reached through [`Run::qo_indptr`] instead.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per token.
    pub positions: Tensor,

    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row, `-1` for a row whose lane registered none. `None` for a fire no lane carried an adapter into.
    pub adapter_routes: Option<Tensor>,

    /// The second row axis's three seats, all `None` for a fire no lane submitted an image into. `patches` is pre-unfolded patch rows; `patch_segments` its indptr; `patch_routes` the scatter target per patch row.
    pub patches: Option<Tensor>,
    /// The patch axis's indptr — see [`patches`](FireBindings::patches).
    pub patch_segments: Option<Tensor>,
    /// The embed merge's destination rows — see [`patches`](FireBindings::patches). Checked against this fire's token row count before the launch.
    pub patch_routes: Option<Tensor>,
    /// The tower's rotation stream: `[patch rows, 3]`, one `(t, h, w)` per patch row. Cut from the same submission as the three above.
    pub patch_positions: Option<Tensor>,
    /// The learned position table's gather indices: `[patch rows, taps]` — 1 tap on the native grid, 4 for bilinear, 16 for bicubic. `None` for a plan that declares no learned position table.
    pub patch_embed_rows: Option<Tensor>,
    /// How much of each tap: `[patch rows, taps]`. `None` for a native-grid plan (reads one table row per patch, unweighted).
    pub patch_embed_weights: Option<Tensor>,
    /// The trunk's rotation stream: `[token rows, 3]`. On the first row axis, not the second, since the trunk is one region over the whole token rectangle. `None` for a plan that does not declare it.
    pub mrope_positions: Option<Tensor>,
    /// `RuntimeInput::SelfCondRows`: `i32`, `[token rows, taps]` — the denoiser's self-conditioning taps, staged on every fire of a plan that declares them (zeros for lanes carrying none). `None` for a plan that does not.
    pub self_cond_rows: Option<Tensor>,
    /// `RuntimeInput::SelfCondWeights`: `f32`, `[token rows, taps]`, beside the rows.
    pub self_cond_weights: Option<Tensor>,

    /// Per cache space, aligned with `Trace::caches`.
    pub geometry: Vec<CacheGeometry>,

    /// Per (run, plan value), flat at `run * plan_values + value`: the reading and grant a launch stages into. `None` for non-plan values.
    pub schedules: Vec<Option<ScheduleSeat>>,
    /// How many plan values one run's slice of [`schedules`](FireBindings::schedules) holds.
    pub plan_values: usize,

    /// The seam extras the arms bind beside the ops' named operands.
    pub tables: FireTables,

    /// The observability seat, `None` for a load whose plan declares no `attn.scores` export and for every fire whose lanes all asked for nothing.
    pub scores: Option<crate::scores::ScoreSeat>,

    /// The device facts every builder takes, pre-probed by the shell (`Device::probe` once at boot); builders themselves never probe.
    pub device: Device,

    /// The operator toggles `plan_decode` takes, resolved once by the shell ([`Toggles::from_env`]) so no arm reads the environment per fire.
    pub toggles: Toggles,

    /// Whether this fire's capture phase will be captured as a CUDA graph. Builders carve graph-shaped, padded schedules under it; `PrefillPlan::graph_capturable` answers whether they managed.
    pub capture: bool,
}

/// One built plan payload. An enum over the four kinds this plane can be asked to build, not `Box<dyn Any>`: `StructKind` is closed, so a wrong kind is a named panic rather than a silent downcast failure.
#[derive(Clone, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::AttnPrefillPlanSm90` — built when a trace declares its prefill plan at this kind; the consumer entry (`attn::prefill_sm90`) still answers a typed refusal.
    PrefillSm90(PrefillPlanSm90),

    /// `StructKind::MlaPlan`.
    Mla(MlaPlan),
}

/// The facts a `Run` cannot recompute from a fire: the pad pair per row axis, this fire's `bodied` word, the load's `shifted` slice, the [`Admit`] table, and the key's carve.
#[derive(Clone, Copy, Default)]
pub struct Ceilings<'c> {
    /// Each rectangle's total rows and the lattice point above them, per row axis — the composition's counts, not a window's.
    pub pads: model_ir::PerAxis<Pad>,

    /// Is this fire a body's? `false` for every fire the eager path serves, and the short circuit of [`Ceilings::admit`].
    pub bodied: bool,

    /// Which regions address off the seat's start: `true` when every op in it is named by [`crate::shifted`] and computes over plane rows `[start, start + count)` given the plane's own base pointers.
    pub shifted: &'c [bool],

    /// Which regions this fire's body actually holds. Captured stretches run from the graph; the islands between are re-issued eagerly.
    pub admits: &'c [Admit],

    /// What this fire's body key says each class may be carved over — this fire's own class table. `None` for every fire off the bodies path, and then [`Standing`] takes no ceiling at all.
    pub carve: Option<Carve<'c>>,
}

impl<'c> Ceilings<'c> {
    /// This region's own axis's pad pair: the token pair for a trunk region, the patch pair for a tower one. Reads the axis from the window table so the launch side and the host side classify the same way.
    fn pad_on(&self, axis: model_ir::RowAxis) -> Pad {
        self.pads[axis]
    }

    /// This region's admission, or `None` for a fire no body holds. `bodied` is asked first, so an eager fire never indexes the table; an out-of-range region reads as absent.
    fn admit(&self, region: u32) -> Option<Admit> {
        self.bodied
            .then(|| self.admits.get(region as usize).copied())
            .flatten()
    }
}

/// What one region's launches are allowed, resolved once per `(region, run)`. [`pad`](Standing::pad) and [`whole`](Standing::whole) are read every fire; [`held`](Standing::held) is the bodies path's. `whole` and `plane` are not each other's negation.
#[derive(Clone, Copy)]
pub(crate) struct Standing {
    /// This region's own axis's pad pair, armed or not — the one field an eager fire still reads.
    pad: Pad,

    /// Is this run's window the whole fire, on that axis? Only meaningful against an armed pad, so callers only ask it where that's guaranteed.
    whole: bool,

    /// What a body grants this region, or [`Held::Eager`].
    held: Held,
}

/// May a graph hold this region's launches, and what does that buy?
#[derive(Clone, Copy)]
enum Held {
    /// No ceiling, no seat, no plane base: a tier-3 eager fire, or an island of a segmented body. Plans, grids and addresses at the live geometry the walk is standing in, same as the eager walk.
    Eager,

    /// A graph holds them, so every number here may be the key's rather than this fire's.
    Captured {
        /// Does this region get its plane's base instead of its window's? `whole` asks "is there nothing to shift"; this asks "may the shift be left to the device".
        plane: bool,

        /// How many rows stand in front of this span and how many it may be carved over. `None` for a fire with no ladder or a span the ladder cannot resolve (gathered or grouped).
        ceiling: Option<(u32, u32)>,

        /// The same ladder read as lanes. `None` off the token axis: the patch axis takes no lane ceiling.
        lanes: Option<(u32, u32)>,
    },
}

impl Standing {
    /// The one resolution, reached only through [`Run::standing_at`]. `captured` and `moves` are the [`Admit`] table and the `shifted` slice read at one region.
    fn of(
        window: &Window,
        axis: model_ir::RowAxis,
        pad: Pad,
        captured: bool,
        moves: bool,
        carve: Option<Carve<'_>>,
    ) -> Self {
        let held = if captured {
            let span = window.span();
            Held::Captured {
                plane: moves && window.is_interval(),
                ceiling: carve
                    .and_then(|carve| carve.on(axis))
                    .and_then(|carve| carve.ceiling(span)),
                // `lane_ceiling` answers `None` on the patch axis, so no axis check is needed here.
                lanes: carve
                    .and_then(|carve| carve.on(axis))
                    .and_then(|carve| carve.lanes(span)),
            }
        } else {
            Held::Eager
        };
        Self {
            pad,
            whole: window.is_whole(pad.rows),
            held,
        }
    }

    /// Does this region address off its plane's base?
    fn plane(&self) -> bool {
        matches!(self.held, Held::Captured { plane: true, .. })
    }

    /// Does this region own a retirement? True for a region a graph holds whose window is the whole fire, or whose ops read the seat's start; nothing else.
    fn armed(&self) -> bool {
        match self.held {
            Held::Eager => false,
            Held::Captured { plane, .. } => self.whole || plane,
        }
    }

    /// The row ceiling as `Run::planning` wants it: the raw `(before, own)` pair, gated on the region being one a graph holds.
    fn ceiling(&self) -> Option<(u32, u32)> {
        match self.held {
            Held::Eager => None,
            Held::Captured { ceiling, .. } => ceiling,
        }
    }

    /// The lane ceiling's raw pair, on the same terms — `Run::planning` clamps it by the staged kv vectors too, which is why this is the pair and not [`lanes`](Standing::lanes)'s number.
    fn lane_carve(&self) -> Option<(u32, u32)> {
        match self.held {
            Held::Eager => None,
            Held::Captured { lanes, .. } => lanes,
        }
    }

    /// How many rows a launch in this region is gridded over, or `None` for the window's own row count — a shifting region takes its classes' rungs capped at the fire's bucket.
    pub(crate) fn rows(&self, span: MaskSpan) -> Option<u32> {
        let Held::Captured {
            plane, ceiling, ..
        } = self.held
        else {
            return None;
        };
        assert!(
            self.pad.bucket >= self.pad.rows,
            "a bodied fire carries an armed pad, and an armed bucket holds the \
             fire's {} rows; this one spells {}",
            self.pad.rows,
            self.pad.bucket,
        );
        let rows = if self.whole {
            self.pad.bucket
        } else if plane {
            let (_, own) = ceiling?;
            own.min(self.pad.bucket)
        } else {
            return None;
        };
        (rows >= span.rows).then_some(rows)
    }

    /// How many requests a lane-gridded launch in this region is gridded over, or `None` for the window's own lane count; read off what was staged.
    pub(crate) fn lanes(&self, windows: &Windows, span: MaskSpan) -> Option<u32> {
        let Held::Captured {
            plane: true, lanes, ..
        } = self.held
        else {
            return None;
        };
        let (before, own) = lanes?;
        let staged = windows
            .qo_absolute()
            .map_or(0, |bounds| bounds.rows.saturating_sub(1));
        let lanes = own.min(staged.checked_sub(before)?);
        assert!(
            u64::from(lanes) + 1 <= windows.slots().stride(),
            "a ceiling grid of {lanes} requests wants {} boundary words, and a window \
             slot holds {}",
            lanes + 1,
            windows.slots().stride(),
        );
        (lanes >= span.lanes).then_some(lanes)
    }
}

/// One fire's dispatch state: the stream context, the resolution tables, the fire bindings, and the plan payloads. Built once per fire, prepare phase first.
pub struct Run<'c> {
    /// The stream and its companions (cuBLAS handle, communicator, jit cache behind it). Everything this crate does to the device goes through it, enqueue only.
    ctx: &'c Ctx,

    /// `Trace::values`, read by [`Run::tensor`] to send each id to its table.
    values: &'c [ValueDecl],

    /// `Trace::nodes`, for the one thing a resolution cannot do from a value id alone: read a whole region's operands at once, for `Fallback::Copy`.
    nodes: &'c [Node],

    /// `Def::Weight` rows, loader-resolved.
    weights: &'c WeightTable,

    /// `Def::Op` / `Def::Merge` rows, carved at the compiler's offsets.
    arena: &'c SlotTable,

    /// `Def::Cache` rows — pool pointers, resolved through [`Run::pool`] and [`Run::recurrent`], never through [`Run::tensor`].
    caches: &'c CacheTable,

    /// Plan payloads: filled by the plan-building arms in prepare, read by the consuming arms after. Keyed by `(run, value)`, flat at width [`Windows::max_runs`]; each slot also carries the [`Admit`] of the region that built it.
    structs: Vec<Option<(Admit, StructSlot)>>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    fire: FireBindings,

    /// The buffered-activation plane, for a fire that carries one — a per-lane RS verb and the pool it addresses.
    rs: Option<RsSeat<'c>>,

    /// Every region's window, resolved once per fire from the composition's class table.
    windows: &'c Windows,

    /// Which region the walk is inside and which run of its window, written by the cursor before each node is dispatched. Turns every resolution below from "the fire's rectangle" into "this node's window of it".
    place: &'c At,

    /// Side-stream contexts, in stream order: `side[0]` is stream 1. Empty is the eager mode, not a degradation: everything fires on the main stream, a legal serialization of the dependency DAG.
    side: &'c [&'c Ctx],

    /// Which stream the walk is on, written by the same cursor that writes [`place`](Run::place), at the same instant.
    stream: &'c Cell<u32>,

    /// The context a conditional body's launches land on, when opened. Not a member of [`side`](Run::side): written only over the conditional's own span.
    body: Option<&'c Ctx>,

    /// The `Fallback::Copy` currently in force: which rectangles the region's gather compacted, and where each landed in the scratch slab.
    copy: CopyPlan,

    /// The facts the fire path cannot recompute, in one piece ([`Ceilings`]). `Ceilings::default()` is the eager arm on every axis.
    ceilings: Ceilings<'c>,

    /// This region's resolution, resolved once — [`Standing`] for the `(region, run)` [`place`](Run::place) currently names.
    stood: Cell<Option<(u32, u32, Standing)>>,
}

impl<'c> Run<'c> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        ctx: &'c Ctx,
        values: &'c [ValueDecl],
        nodes: &'c [Node],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
    ) -> Self {
        Self {
            ctx,
            values,
            nodes,
            weights,
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            rs: None,
            windows,
            place,
            side: &[],
            stream: &place.region,
            body: None,
            copy: CopyPlan::default(),
            ceilings: Ceilings::default(),
            stood: Cell::new(None),
        }
    }

    /// The same `Run`, told the facts the fire path cannot recompute — see [`Ceilings`]. Additive: a `Run` never handed one pads and carves nothing, which is the whole of the eager path.
    #[must_use]
    pub fn ceilings(mut self, ceilings: Ceilings<'c>) -> Self {
        self.ceilings = ceilings;
        self
    }

    /// What the region the walk is standing in is allowed — [`Standing`] for this `(region, run)`, resolved once and held in [`stood`](Run::stood). Every ceiling below reads this and nothing else.
    fn standing(&self) -> Standing {
        let region = self.place.region.get();
        let run = self.place.run.get();
        if let Some((at_region, at_run, standing)) = self.stood.get()
            && at_region == region
            && at_run == run
        {
            return standing;
        }
        let standing = self.standing_at(region, run);
        self.stood.set(Some((region, run, standing)));
        standing
    }

    /// [`Standing`] for any `(region, run)` of this fire, off the same [`Ceilings`] the walk reads, so the ledger and the launch resolve one answer.
    pub(crate) fn standing_at(&self, region: u32, run: u32) -> Standing {
        self.standing_as(region, run, matches!(self.ceilings.admit(region), Some(Admit::Captured)))
    }

    /// `standing_at` with the captured clause stated by the caller: the grid ledger records every non-island region as captured.
    pub(crate) fn standing_as(&self, region: u32, run: u32, captured: bool) -> Standing {
        let axis = self.windows.axis_of(region);
        Standing::of(
            self.windows.at(region, run),
            axis,
            self.ceilings.pad_on(axis),
            captured,
            self.ceilings
                .shifted
                .get(region as usize)
                .copied()
                .unwrap_or(false),
            self.ceilings.carve,
        )
    }

    /// Every region's window, as this fire resolved it.
    pub(crate) fn windows(&self) -> &'c Windows {
        self.windows
    }

    /// The buffered-activation plane, for a fire that carries one. A builder rather than a [`FireBindings`] field: it is a per-lane verb plus a pool, read by the two dispatch arms that touch it.
    pub fn buffered(mut self, rs: RsSeat<'c>) -> Self {
        self.rs = Some(rs);
        self
    }

    /// Scatter or gather this operand's rows, if it is a buffered plane. A value no plan buffers, or a fire with no buffered lane, is a no-op. # Errors: a page slot past the pool, a buffer run too short, or the copy.
    pub(crate) fn rs_move(
        &self,
        op: &'static str,
        id: ValueId,
        rows: Tensor,
    ) -> Result<(), kernels_cuda::Error> {
        let Some(seat) = self.rs.as_ref() else {
            return Ok(());
        };
        let Some(plane) = seat.buffers.planes().of(id) else {
            return Ok(());
        };
        let span = self.window().span();
        let bounds = self.qo_indptr_host();
        // The rectangle is the window's, even where the op's is not: this is host arithmetic, so it takes the window back.
        let rows = self.windowed(rows);
        seat.run(self.ctx.stream(), plane, span.lane_offset, bounds, rows)
            .map_err(|fault| kernels_cuda::Error::Backend {
                op,
                detail: fault.to_string(),
            })
    }

    /// `Trace::values`, for the copy plan's own shape reading.
    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    /// `Trace::nodes`, for the same.
    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    /// Which region of the template the walk is inside.
    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }


    /// The FIRE-WIDE rectangle a value names, before any window is applied — what a copy's gather reads from and its scatter writes back to.
    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// Take the copy plan the region's gather just built.
    pub(crate) fn set_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    /// The same `Run`, told where P6's side streams are. Additive, chosen per pass: `record.rs` walks one `Run` twice (eager for the numbers, then capturing), and only the capturing walk is handed the streams.
    #[must_use]
    pub fn across(mut self, side: &'c [&'c Ctx], stream: &'c Cell<u32>) -> Self {
        self.side = side;
        self.stream = stream;
        self
    }

    /// The same `Run`, told which context a conditional body's launches go on. Also seats the stream cell, since the two must arrive together.
    #[must_use]
    pub fn conditional(mut self, body: &'c Ctx, stream: &'c Cell<u32>) -> Self {
        self.body = Some(body);
        self.stream = stream;
        self
    }

    /// The stream context the node being dispatched fires on. A lookup: which stream a region belongs on was decided at compile by `model_compiler::stream`.
    pub(crate) fn ctx(&self) -> &'c Ctx {
        let ctx = match self.body {
            // The conditional body is asked first: a conditional region is never forked, so the two readings never both have something to say.
            Some(body) if self.stream.get() == crate::window::BODY => body,
            _ if self.side.is_empty() => self.ctx,
            _ => match self.stream.get() {
                0 => self.ctx,
                n => self.side.get(n as usize - 1).copied().unwrap_or(self.ctx),
            },
        };
        // This is also the last instant the window is still in hand, so it decides whether this region may pad.
        ctx.arm(self.here());
        // And where this region's live row count is read from.
        ctx.arm_stage(self.live_at());
        // And which region's scratch this launch stages into: a staging plane belongs to the region that fills it and the region that reads it, which are one region because they are one dispatch.
        ctx.arm_region(self.place.region.get());
        ctx
    }

    /// The pad this region is allowed. Padded rows are only safe to scribble on when they are the fire's own tail, so a region pads only if its window is the whole fire.
    fn here(&self) -> Pad {
        let standing = self.standing();
        let pad = standing.pad;
        if pad.bucket <= pad.rows {
            return Pad::default();
        }
        if standing.whole { pad } else { Pad::default() }
    }

    /// Does this region get its plane's base instead of its window's? Requires the region to be graph-held, every op shifted, and the window an interval.
    fn plane_base(&self) -> bool {
        self.standing().plane()
    }

    /// Where this region's live row count is read from, or `0` — [`here`](Run::here)'s twin, bounding how far below its extent a launch may stop.
    fn live_at(&self) -> u64 {
        let at = self
            .windows
            .live_at(self.place.region.get(), self.place.run.get());
        if at == 0 || !self.standing().armed() {
            0
        } else {
            at
        }
    }

    /// How many rows a launch in this region is gridded over, or `None` for every launch gridded at its window's own live span. A captured body grids at the key's own ceiling so every fire of the key grids the same.
    fn carve_rows(&self) -> Option<u32> {
        self.standing().rows(self.window().span())
    }

    /// How many requests a lane-gridded launch in this region is gridded over, or `None` for the window's own lane count — [`carve_rows`](Run::carve_rows)'s twin on the lane axis, delivered via [`ragged_lanes`](Run::ragged_lanes).
    fn carve_lanes(&self) -> Option<u32> {
        self.standing().lanes(self.windows, self.window().span())
    }

    /// How many rows of a packed plane stand for one token row — the fan-out a `linear::mlp` entry scales the staged seat by. One for a dense MLP; a routed leg multiplies the token axis's carved rows by the fan.
    pub(crate) fn plane_fan(&self, plane_rows: u32) -> u32 {
        let tokens = self
            .carve_rows()
            .unwrap_or_else(|| self.window().span().rows);
        if tokens == 0 {
            return 1;
        }
        assert_eq!(
            plane_rows % tokens,
            0,
            "a {plane_rows}-row packed plane does not divide into {tokens} token rows, so              no fan-out scales the staged seat onto its row axis",
        );
        (plane_rows / tokens).max(1)
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// The window the node being dispatched runs over — this region's, cut at the run the walk is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// The rows of this window that are actually the node's, for a region answered `Fallback::Grouped`; `None` otherwise — an arm that takes this list must honour it.
    pub(crate) fn segments(&self) -> Option<Segments> {
        let window = self.window();
        let count = window.segs();
        if count == 0 {
            return None;
        }
        Some(Segments {
            list: window.segments,
            count,
            cap: window.segment_cap,
            max_rows: window.segment_rows(),
        })
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs). The run comes off the same cell the window does, so a schedule is stored and read at the same key by construction.
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's qo boundaries, staged — what a ragged view is cut by. Rebased; [`qo_indptr_absolute`](Run::qo_indptr_absolute) is a second, absolute reading of the same boundaries.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// Their host twin, for the prefill and mla builders that walk the contents. Rebased: entry 0 is 0, because the rectangle they bound is this window's, not the fire's.
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    /// The same boundaries, read absolutely — the fire's whole qo vector, `[fire lanes + 1]` entries with nothing subtracted — or `None` if none staged. Whole, not sliced: reached under [`plane_base`](Run::plane_base), where `lane_offset` isn't fixed by a `record::BodyKey`.
    pub(crate) fn qo_indptr_absolute(&self) -> Option<Tensor> {
        // The invariant the two readings owe each other: this window's lane slice of the fire's vector, minus its own first bound, equals the rebased vector beside it.
        debug_assert!(
            {
                let absolute = self.qo_indptr_absolute_host();
                let rebased = self.qo_indptr_host();
                rebased.is_empty()
                    || absolute.is_empty()
                    || (absolute.len() == rebased.len()
                        && absolute
                            .iter()
                            .zip(rebased)
                            .all(|(there, here)| there - absolute[0] == *here))
            },
            "a window's two readings of its qo boundaries disagree",
        );
        self.windows.qo_absolute()
    }

    /// Its host twin, and the one place a lane slice of the absolute reading is legal: this window's `[lanes + 1]` entries, un-subtracted.
    pub(crate) fn qo_indptr_absolute_host(&self) -> &'c [i32] {
        let span = self.window().span();
        let first = span.lane_offset as usize;
        let last = first + span.lanes as usize;
        self.windows
            .qo_absolute_host()
            .get(first..=last)
            .unwrap_or_default()
    }

    /// How many token rows this window carries — the `total_num_rows` the prefill builders take.
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span().rows
    }

    /// The mask span table this window's schedule should carry, or `None` if none masked. Sliced by lane, absolute in value.
    pub(crate) fn mask_indptr(&self) -> Option<Tensor> {
        // Whole under a plane base: slicing would send request `lane_offset` to lane zero's bits.
        if self.plane_base() {
            return self.fire.tables.mask_indptr;
        }
        let span = self.window().span();
        self.fire
            .tables
            .mask_indptr
            .map(|table| skip(table, span.lane_offset, span.lanes + 1))
    }

    /// Whether any lane of this window carries more than one token — the mla builder's `causal` word: multi-token lanes attend causally within themselves, single-token (decode) lanes have nothing to order.
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|span| span[1] - span[0] > 1)
    }

    /// One value's rectangle, cut to the window of the node asking for it. Every row-shaped table is indexed by absolute fire row, so a window is a slice; `GeomKind::Indices` and `RuntimeInput::Mask` are the two exceptions, cut absolute and lane-sliced instead.
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        // A gathered window's rows were compacted into a scratch slab, so a slice of the fire-wide column would be the wrong bytes.
        if self.window().gathered.is_some() {
            return self.compacted(id, handle);
        }
        if matches!(
            self.values[at].def,
            Def::Input(RuntimeInput::Mask { .. })
                | Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::Indices,
                    ..
                })
        ) {
            return handle;
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        let seated = self.window();
        let window = seated.span();
        // A fact about the region, not the value; both short-circuit on `Held::Eager` so asking once here costs an eager fire nothing.
        let plane = self.plane_base();
        let ceiling = self.carve_rows();
        let rows = ceiling.unwrap_or(window.rows);

        // Which rectangle this column is cut at is read off its own dim. A `Dim::Const` column belongs to no axis and is handed over whole, as is a value with no shape.
        let Some(&dim) = shape.first() else {
            return handle;
        };
        let Some(axis) = dim.axis() else {
            return handle;
        };
        let span = seated.on(axis);

        // A row column: one entry per row of this rectangle. The ceiling and plane base are the primary (token) axis's alone; a tower rectangle cuts at its own window, whole.
        let row = |times: u32| {
            // The extent is the launch's, the pointer is the plane's: a shifting region's ops address `start + r` off whatever base they're handed, so the base stays the plane's.
            let primary = axis == model_ir::RowAxis::PRIMARY;
            let extent = if primary { rows } else { span.rows };
            let offset = if primary && plane { 0 } else { span.row_offset };
            (offset * times, extent * times)
        };
        // A lane column: one entry per request of this rectangle, plus the `k` an indptr-shaped dim closes with.
        let lane = |plus: u32| (span.lane_offset, span.lanes + plus);

        let (skip, keep) = match dim {
            Dim::Const(_) => return handle,
            Dim::Tokens => row(1),
            Dim::TokensTimes(k) => row(k),
            Dim::Patches => row(1),
            Dim::Lanes => lane(0),
            Dim::LanesPlus(k) => lane(k),
            Dim::Images => lane(0),
            Dim::ImagesPlus(k) => lane(k),
        };
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
                panic!(
                    "value {at} is stored as {:?}, which has no element size and so no \
                     row to step by",
                    handle.dtype
                )
            });
        Tensor::new(
            handle.ptr + u64::from(skip) * stride,
            keep.min(handle.rows.saturating_sub(skip)),
            handle.width,
            handle.dtype,
        )
    }

    /// [`Run::cut`]'s other half: what a `Fallback::Copy` resolves to. A row-shaped value is the gather's slab rectangle; kv geometry vectors re-cut for the gathered lanes; everything else handed over whole.
    fn compacted(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        let gathered = self
            .window()
            .gathered
            .as_ref()
            .expect("`compacted` is reached only through a gathered window");
        if let Def::Input(RuntimeInput::Geometry { space, kind }) = &self.values[at].def {
            let Some(space) = gathered.spaces.get(*space as usize) else {
                return handle;
            };
            return match kind {
                GeomKind::Indptr => space.page_indptr,
                GeomKind::Indices => space.page_indices,
                GeomKind::LastPageLen => space.last_page_lens,
                GeomKind::KvLen => space.kv_len,
                // Unreachable in practice; answers the fire-wide vector rather than a wrong window if it ever is.
                _ => handle,
            };
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        match shape.first() {
            Some(Dim::Tokens | Dim::TokensTimes(_)) => {
                assert_eq!(
                    self.copy.region,
                    self.place.region.get(),
                    "value {at} is being resolved inside a copied region whose gather \
                     has not run; `model_exec::fire::walk` brackets a copied region's \
                     nodes and this is what says the bracket was lost",
                );
                self.copy.tight(handle.ptr).unwrap_or_else(|| {
                    panic!(
                        "value {at} is row-shaped and its column was not compacted; the \
                         copy plan is built from the same region's operands the walk is \
                         dispatching, so a miss is a plan and a template built apart"
                    )
                })
            }
            _ => handle,
        }
    }

    /// The crate's heart: one plan id in, one device handle out, routed on the id's `Def`, cut to the asking node's window ([`Run::cut`]). Cache ids and split-plane weights never resolve here — reaching that case is a dispatch-arm bug.
    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// The fire-wide rectangle, asked for by name — for a consumer whose own index vector is already absolute (the embed merge), so cutting at the window would double-count the offset.
    pub(crate) fn fire_wide(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// The same resolution, uncut — the fire-wide rectangle a value names.
    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            Def::Input(RuntimeInput::Mask { space }) => {
                let seat = self.geometry(at, *space);
                seat.mask.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the mask bits of cache space {space}, which \
                         this fire left unbound"
                    )
                })
            }
            // Bound only when a lane of this fire carried an adapter; a fire none did stages nothing and never dispatches a node that would reach this arm (the correction's window is empty).
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it carried"
                    )
                })
            }
            // The second row axis's runtime inputs, bound from what `enqueue` wrote; a fire whose lanes submitted no image binds none of them.
            Def::Input(RuntimeInput::Patches) => self.fire.patches.unwrap_or_else(|| {
                panic!(
                    "value {at} reads this fire's patch rows, which no lane of it submitted"
                )
            }),
            Def::Input(RuntimeInput::PatchSegments) => {
                self.fire.patch_segments.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's image boundaries, which no lane of it \
                         submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchRoutes) => {
                self.fire.patch_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads where this fire's tower rows land, and no lane of it \
                         submitted an image"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedRows) => {
                self.fire.patch_embed_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads which position-table rows this fire's patches gather, \
                         and no lane of it submitted an image"
                    )
                })
            }
            Def::Input(RuntimeInput::SelfCondRows) => {
                self.fire.self_cond_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's self-conditioning taps, which this load \
                         reserved no seat for"
                    )
                })
            }
            Def::Input(RuntimeInput::SelfCondWeights) => {
                self.fire.self_cond_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's self-conditioning weights, which this \
                         load reserved no seat for"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedWeights) => {
                self.fire.patch_embed_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's interpolation weights, which a native-grid \
                         plan declares none of"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchPositions) => {
                self.fire.patch_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads where this fire's patches sit in their grids, and no \
                         lane of it submitted an image"
                    )
                })
            }
            // The trunk's triple, on the token axis. Unlike the four above, not gated on an image lane: the trunk covers the whole rectangle.
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's (t, h, w) token positions, which this load \
                         reserved no stream for"
                    )
                })
            }
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                let seat = self.geometry(at, *space);
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                    GeomKind::KvLen => seat.kv_len,
                    GeomKind::RowValid => seat.row_valid,
                    GeomKind::RequestOfToken => seat.request_of_token,
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of cache space {space}, which this \
                         fire left unbound"
                    )
                })
            }
            Def::Weight(w) => {
                let row = *w as usize;
                match self.weights.0.get(row).copied().flatten() {
                    Some(WeightRow::Dense(handle) | WeightRow::Streamed { slab: handle, .. }) => {
                        handle
                    }
                    Some(WeightRow::Planes { .. }) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            // A φ resolves like the op output it merges: the compiler aliased every arm onto one arena slot, written at this id's row — so `Merge` is the same read as `Op`.
            Def::Op(_) | Def::Merge(_) => {
                self.arena.0.get(at).copied().flatten().unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                })
            }
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// A fire-aligned value viewed through this window's boundaries. The indptr is ambient: no op names it. Rebased to the window's own first row; the absolute reading is [`ragged_q`](Run::ragged_q) instead.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// The FA2 query axis's own reading of the same pairing — boundaries chosen by whether this region moves its own plane. FA2 has no seat offset, so its CSR must count from wherever `q` counts from. Goes over whole, never cut at `lane_offset`.
    pub(crate) fn ragged_q(&self, id: ValueId) -> RaggedTensor {
        let indptr = if self.plane_base() {
            self.qo_indptr_absolute().unwrap_or_else(|| self.qo_indptr())
        } else {
            self.qo_indptr()
        };
        RaggedTensor {
            data: self.tensor(id),
            indptr,
        }
    }

    /// The same pairing, with the boundary vector declared out to the key's lane ceiling — what every seated entry counting requests off its length takes; the kernel retires the padded tail against the live lane count.
    pub(crate) fn ragged_lanes(&self, id: ValueId) -> RaggedTensor {
        let indptr = self.qo_indptr();
        let indptr = match self.carve_lanes() {
            Some(lanes) if lanes + 1 > indptr.rows => {
                Tensor::new(indptr.ptr, lanes + 1, indptr.width, indptr.dtype)
            }
            _ => indptr,
        };
        RaggedTensor {
            data: self.tensor(id),
            indptr,
        }
    }

    /// The `(codes, scales)` planes of a split-plane bank and the seat that says where they are right now. `Some` for a bank the loader landed as planes, `None` for an ordinary handle.
    pub(crate) fn maybe_planes(
        &self,
        id: ValueId,
    ) -> Option<(Tensor, Tensor, Option<Tensor>, GroupSeat)> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        match self.weights.0.get(*w as usize).copied().flatten() {
            Some(WeightRow::Planes {
                codes,
                scales,
                biases,
                seat,
                repacked: false,
            }) => Some((codes, scales, biases, seat)),
            _ => None,
        }
    }

    /// The same triplet in fragment order — `Some` for a row relabelled at import, `None` otherwise. Disjoint on purpose: a repacked plane has the same dtype/rows/width as row-major, so the wrong order would answer nonsense.
    pub(crate) fn maybe_tiled_planes(
        &self,
        id: ValueId,
    ) -> Option<(Tensor, Tensor, Tensor, GroupSeat)> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        match self.weights.0.get(*w as usize).copied().flatten() {
            Some(WeightRow::Planes {
                codes,
                scales,
                biases: Some(biases),
                seat,
                repacked: true,
            }) => Some((codes, scales, biases, seat)),
            _ => None,
        }
    }

    /// A weight seated as one stored quantization block, re-badged as the byte rectangle a decode-in-dot entry reads — `Some` for a [`WeightRow::Dense`] whose declaration is a quant term. Returns `U8`: `rows` weight rows of `width` bytes.
    pub(crate) fn maybe_stored(&self, id: ValueId) -> Option<Tensor> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        let Some(WeightRow::Dense(handle)) = self.weights.0.get(*w as usize).copied().flatten()
        else {
            return None;
        };
        // A stored super-block carries its factors inside one plane. The affine families are not here — their factors ride companion planes and a different seat reads them.
        if !matches!(
            handle.dtype,
            model_ir::Dtype::U2g16k
                | model_ir::Dtype::I3g16k
                | model_ir::Dtype::U4g32k
                | model_ir::Dtype::U5g32k
                | model_ir::Dtype::I6g16k
        ) {
            return None;
        }
        // Through `cut` like every other resolution: a weight's leading dim is `Dim::Const`, so the window walk hands it back whole anyway.
        let seated = self.cut(id, handle);
        Some(Tensor::new(
            seated.ptr,
            seated.rows,
            seated.width,
            model_ir::Dtype::U8,
        ))
    }

    pub(crate) fn planes(&self, id: ValueId) -> (Tensor, Tensor, Option<Tensor>, GroupSeat) {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes {
                codes,
                scales,
                biases,
                seat,
                repacked: _,
            }) => (codes, scales, biases, seat),
            Some(WeightRow::Dense(_) | WeightRow::Streamed { .. }) => panic!(
                "value {at} is weight {row}, bound as one dense handle, and this op reads \
                 a split-plane bank"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// A routed expert bank, and where its experts are — the resolution `linear.moe_matmul_select` uses in place of [`Run::tensor`]; [`ExpertTable::RESIDENT`] for a fully-resident load.
    pub(crate) fn expert_bank(&self, id: ValueId) -> (Tensor, ExpertTable) {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and a routed bank is a weight row");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Dense(handle)) => (self.cut(id, handle), ExpertTable::RESIDENT),
            Some(WeightRow::Streamed {
                slab,
                table,
                counts,
            }) => (self.cut(id, slab), ExpertTable { table, hits: counts }),
            Some(WeightRow::Planes { .. }) => panic!(
                "value {at} is weight {row}, a split-plane bank; a dense routed select \
                 does not read one"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// The paged kv pool a cache id names, with its lane-indexed tables cut to the asking node's window. The storage is the whole pool; page bounds, last-page fills and the padding mask are sliced, the page-id list is not.
    pub(crate) fn pool(&self, id: ValueId) -> KvPool {
        match self.cache(id) {
            CachePool::Kv { space, pool } => {
                let window = self.window().span();
                // A gathered window's tables are re-cut, not sliced: the page-id list has other lanes' pages between gathered spans. `row_valid` needs no such treatment.
                if let Some(gathered) = &self.window().gathered {
                    let seat = gathered.spaces.get(*space as usize);
                    return KvPool {
                        page_indptr: seat.map_or(pool.page_indptr, |seat| seat.page_indptr),
                        page_indices: seat.map_or(pool.page_indices, |seat| seat.page_indices),
                        last_page_lens: seat.map_or(pool.last_page_lens, |seat| seat.last_page_lens),
                        row_valid: skip(pool.row_valid, 0, window.rows),
                        ..*pool
                    };
                }
                KvPool {
                    page_indptr: skip(pool.page_indptr, window.lane_offset, window.lanes + 1),
                    last_page_lens: skip(pool.last_page_lens, window.lane_offset, window.lanes),
                    row_valid: if self.plane_base() {
                        // The plane's own table, whole: the launch reaches its rows through `win[1]`.
                        pool.row_valid
                    } else {
                        skip(pool.row_valid, window.row_offset, window.rows)
                    },
                    ..*pool
                }
            }
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The same pool, with its per-lane tables read absolutely — what the five FA2 entries take in place of [`pool`](Run::pool). Under anything but a plane base this is [`pool`](Run::pool).
    pub(crate) fn pool_absolute(&self, id: ValueId) -> KvPool {
        let pool = self.pool(id);
        if !self.plane_base() {
            return pool;
        }
        // The page-id list and its bounds are already absolute; what moves here is which bound a request reads.
        match self.cache(id) {
            CachePool::Kv { pool: whole, .. } => KvPool {
                page_indptr: whole.page_indptr,
                last_page_lens: whole.last_page_lens,
                ..pool
            },
            CachePool::Recurrent(_) => pool,
        }
    }

    /// The observation's reading, and it is the window's — the `q` rectangle `attention.score_capture` fires over, launch-local.
    pub(crate) fn windowed(&self, handle: Tensor) -> Tensor {
        if !self.plane_base() {
            return handle;
        }
        let window = self.window().span();
        skip(handle, window.row_offset, window.rows)
    }

    /// The recurrent state pool a cache id names, with its slot map cut to the asking node's window; the slabs themselves are the model's state, whole.
    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        // The head segment, which for every fire but a splitting one is the whole row.
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, false)
        }
    }

    /// The same pool with its per-lane seats read absolutely — what the four chunked arms take in place of [`recurrent`](Run::recurrent); their tables go over whole and the kernel indexes at `r + win[3]`. Under anything but a plane base this is [`recurrent`](Run::recurrent).
    pub(crate) fn recurrent_absolute(&self, id: ValueId) -> RecurrentPool {
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, true)
        }
    }

    /// The tail segment of a row whose fold boundary is interior, or `None` for every fire that does not split one — the arm then fires twice, head `[0, n)` with the fold, then this tail reading what the head wrote.
    pub(crate) fn recurrent_tail_absolute(&self, id: ValueId) -> Option<RecurrentPool> {
        let cut = self.recurrent_cut(id, true);
        if cut.begin_at.is_absent() {
            return None;
        }
        Some(RecurrentPool {
            // The tail moves no state: the boundary is where the head left it, and a second writeback would carry it to the row's end.
            write_state: false,
            commit_len: Tensor::ABSENT,
            ..cut
        })
    }

    /// The pool with every per-request seat cut to the asking node's window. `absolute` picks the lane-axis reading: `false` slices at `lane_offset`, `true` hands the fire's vectors whole.
    fn recurrent_cut(&self, id: ValueId, absolute: bool) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => {
                let window = self.window().span();
                // A sliced address is legal under a plane base only where `lane_offset` is zero, since a body bakes the address.
                assert!(
                    absolute || !self.plane_base() || window.lane_offset == 0,
                    "value {} takes the window-local recurrent lane door under a \
                     plane base at lane offset {}; a body would bake that slice \
                     and replay it at another split (`crate::lane_shifted`)",
                    id.0,
                    window.lane_offset,
                );
                // One predicate for all four vectors: a launch reads them all at index `[r]`.
                let lanes = |table: Tensor| {
                    if absolute && self.plane_base() {
                        table
                    } else {
                        skip(table, window.lane_offset, window.lanes)
                    }
                };
                RecurrentPool {
                    slot_ids: lanes(pool.slot_ids),
                    write_state_mask: lanes(pool.write_state_mask),
                    commit_len: lanes(pool.commit_len),
                    begin_at: lanes(pool.begin_at),
                    ..*pool
                }
            }
            CachePool::Kv { .. } => panic!(
                "value {} is a paged kv space, and this op scans a recurrent state pool",
                id.0
            ),
        }
    }

    fn cache(&self, id: ValueId) -> &CachePool {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Cache(c) => {
                let row = *c as usize;
                self.caches.0.get(row).unwrap_or_else(|| {
                    panic!(
                        "value {at} is cache space {row}, and the shell binds {} pools",
                        self.caches.0.len()
                    )
                })
            }
            _ => panic!("value {at} is not a cache space; tensors resolve through `Run::tensor`"),
        }
    }

    fn geometry(&self, at: usize, space: u32) -> &CacheGeometry {
        let space = space as usize;
        self.fire.geometry.get(space).unwrap_or_else(|| {
            panic!(
                "value {at} names cache space {space}, and this fire binds {} geometry spaces",
                self.fire.geometry.len()
            )
        })
    }

    /// Everything a plan op's builder takes, in one place: the host geometry twins of the cache space its `kv_indptr` names, cut to the op's own window, beside the schedule's reading and grant. `kv_indptr` names the space, `plan` the struct value the op defines.
    pub(crate) fn planning(&self, geom: ValueId, plan: ValueId) -> Planning<'_> {
        let at = geom.0 as usize;
        let Def::Input(RuntimeInput::Geometry { space, .. }) = &self.values[at].def else {
            panic!(
                "value {at} is not declared cache geometry, and a plan op routes to its \
                 cache space through its geometry input"
            );
        };
        let seat = self.geometry(at, *space);
        let seat = seat.planning.as_ref().unwrap_or_else(|| {
            panic!(
                "cache space {space} carries no planning seat; the shell binds the host \
                 geometry twins before a plan op can fire"
            )
        });
        let run = self.place.run.get() as usize;
        let schedule = self
            .fire
            .schedules
            .get(run * self.fire.plan_values + plan.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "plan value {} carries no schedule seat for run {run} of its \
                     window; the shell reads every schedule's reading off the plan op \
                     that defines it and carves one grant per run of the region that \
                     builds it, so a plan op firing without one is a value \
                     `store::kv::probe` never walked",
                    plan.0
                )
            });
        let window = self.window();
        let span = window.span();
        // How wide this schedule may be carved: the key's ceilings for any plan the bodies path can serve, `None` otherwise — keeps the hash stable across batch sizes.
        let standing = self.standing();
        let carve = standing.ceiling();
        // The same arithmetic read as lanes. The lane reading is the token axis's alone; a patch region takes its window's own lane pair.
        let carve_lanes = standing.lane_carve();
        // How many lanes, and where they count from: `(origin, count)`, or `None` for this window's own pair — gated on graph-held, resolved span, and plane-base.
        let ceiling: Option<(u32, u32)> = standing
            .plane()
            .then_some(carve_lanes)
            .flatten()
            .and_then(|(before, own)| {
                let staged = self
                    .windows
                    .qo_absolute()
                    .map_or(0, |bounds| bounds.rows.saturating_sub(1))
                    .min((seat.kv_indptr.len() as u32).saturating_sub(1))
                    .min(seat.kv_len.len() as u32);
                let covered = staged.checked_sub(before)?;
                Some((before, own.min(covered)))
            })
            .filter(|(before, lanes)| {
                *lanes >= span.lanes && before + lanes >= span.lane_offset + span.lanes
            });
        let kind = self.declared(plan);
        // How many rows this schedule may be carved over: the key's rows for a row-reading schedule, `None` for decode — the prefill builders hash it into the kernel symbol and grid.
        let rows_ceiling: Option<u32> = (!matches!(kind, StructKind::AttnDecodePlan))
            .then_some(carve)
            .flatten()
            // Capped at this region's own axis's bucket.
            .map(|(_, own)| own.min(standing.pad.bucket))
            .filter(|rows| *rows >= span.rows);
        // A gathered window's twins are re-cut with its lanes: page bounds as a fresh prefix sum, kv lengths in gathered order.
        let (kv_indptr, kv_len) = match window.gathered.as_ref().and_then(|g| g.spaces.get(*space as usize)) {
            Some(gathered) => (
                gathered.page_indptr_host.as_slice(),
                gathered.kv_len_host.as_slice(),
            ),
            None => {
                let first = span.lane_offset as usize;
                // The slice is the carve's, not the fire's: the builders walk it to the lane count `shape` states, so a slice cut at the window's own lanes would read past the end.
                let lanes = ceiling.map_or(span.lanes, |(_, lanes)| lanes) as usize;
                (
                    seat.kv_indptr
                        .get(first..=first + lanes)
                        .unwrap_or(&seat.kv_indptr),
                    seat.kv_len
                        .get(first..first + lanes)
                        .unwrap_or(&seat.kv_len),
                )
            }
        };
        // Two channels built side by side out of one span: `shape` is what the schedule is carved at (rides the hashed plan payload); `live` is what this fire brought.
        let shape = Shape {
            // The carved count: the ladder's lane ceiling where one was taken, this window's lanes otherwise.
            num_requests: ceiling.map_or(span.lanes, |(_, lanes)| lanes),
            // Where this schedule's request numbers count from: sliced tables number from the window's zero, whole tables must name fire lanes since FA2 adds no offset.
            lane_offset: match ceiling {
                Some((first, _)) => first,
                None if standing.plane() => span.lane_offset,
                None => 0,
            },
            ..schedule.shape
        };
        let live = Live {
            requests: span.lanes,
            lane_offset: if standing.plane() { span.lane_offset } else { 0 },
            row_offset: if standing.plane() { span.row_offset } else { 0 },
            // What this fire brought; [`Planning::rows`] is the carved twin.
            rows: span.rows,
        };
        // The two channels may part in exactly one direction: a carve is wider than the fire or the same, never narrower.
        assert!(shape.num_requests >= live.requests, "a carve is never narrower than the fire");
        assert!(shape.lane_offset >= live.lane_offset, "a carve starts at or before the fire");
        assert!(
            shape.lane_offset + shape.num_requests >= live.lane_offset + live.requests
        );
        let rows = rows_ceiling.unwrap_or(span.rows);
        // The row axis's half of the same pin: a carve is wider than the fire or the same, never narrower.
        assert!(rows >= live.rows, "a row carve is never narrower than the fire");
        Planning {
            kv_indptr,
            kv_len,
            shape,
            live,
            rows,
            window: schedule.window,
            workspace: schedule.workspace,
        }
    }

    /// The `StructKind` a plan op's output value declares — how the prefill-building arm tells fa2 from sm90: the trace wrote the choice into `Trace::values`, the arm only follows it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The dsv4 compressor slabs, for `attention.pool_gather`'s seam.
    /// The compressor slabs staged for cache space `pages` (`Def::Cache`).
    pub(crate) fn slabs(&self, pages: ValueId) -> PoolSlabs {
        let at = pages.0 as usize;
        let Some(Def::Cache(space)) = self.values.get(at).map(|v| &v.def) else {
            panic!("value {at} is not a cache space; the pooled state is keyed by one")
        };
        self.fire
            .tables
            .pool_state
            .iter()
            .find(|(held, _)| held == space)
            .map(|(_, slabs)| *slabs)
            .unwrap_or_else(|| {
                panic!(
                    "this fire binds no dsv4 compressor slabs for cache space {space}, which \
                     `attention.pool_gather` reads beside the pool"
                )
            })
    }

    /// A hash of the shape of every plan payload this fire built — every number that can reach a kernel argument, not the workspace contents. A disagreement with the graph key demotes the body rather than silently reading stale numbers; the Debug image covers every field by construction.
    pub(crate) fn schedule_shape(&self) -> u64 {
        use core::fmt::Write;
        use std::hash::{DefaultHasher, Hasher};

        struct Sink(DefaultHasher);
        impl Write for Sink {
            fn write_str(&mut self, text: &str) -> core::fmt::Result {
                self.0.write(text.as_bytes());
                Ok(())
            }
        }

        let mut sink = Sink(DefaultHasher::new());
        for (at, held) in self.structs.iter().enumerate() {
            let Some((admit, slot)) = held else { continue };
            // An island region's plan is not in the hash: its numbers follow the fire and no capture holds a stale copy to catch.
            if *admit == Admit::Island {
                continue;
            }
            let _ = write!(sink, "{at}:");
            // `int_upload` is deliberately left out: it is supposed to differ every fire.
            let _ = match slot {
                StructSlot::Decode(p) => write!(
                    sink,
                    "d{:?}{:?}{:?}{:?}",
                    p.info, p.workspace, p.shape, p.window
                ),
                // `mask_indptr` is here because it's a pointer a capture bakes, unlike every other field, which is a number.
                StructSlot::Prefill(p) => write!(
                    sink,
                    "p{:?}{:?}{:?}{:?}{}{}{}{:?}",
                    p.info, p.workspace, p.shape, p.window, p.total_tokens, p.causal,
                    p.graph_capturable, p.mask_indptr
                ),
                StructSlot::PrefillSm90(p) => write!(
                    sink,
                    "s{:?}{:?}{:?}{}{}",
                    p.info, p.workspace, p.shape, p.total_tokens, p.causal
                ),
                // `causal` is here because it lives in `int_upload`, which this hash doesn't otherwise cover.
                StructSlot::Mla(p) => write!(
                    sink,
                    "m{:?}{:?}{}{}",
                    p.info, p.workspace, p.num_heads, p.causal
                ),
            };
        }
        sink.0.finish()
    }

    /// Did every schedule this fire built keep its graph-shaped padding? A schedule that didn't fit its workspace grant falls back to one that fits but isn't capturable.
    pub(crate) fn capturable(&self) -> bool {
        self.structs.iter().flatten().all(|(_, slot)| match slot {
            StructSlot::Prefill(plan) => plan.graph_capturable,
            _ => true,
        })
    }

    /// Store a plan payload a prepare-phase arm just built, with the admission of the region that built it — resolved here, at the same instant [`Standing`] reads it for [`planning`](Run::planning).
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        let admit = self
            .ceilings
            .admit(self.place.region.get())
            .unwrap_or(Admit::Captured);
        self.structs[at] = Some((admit, built));
    }

    /// One built slot, whichever kind it holds — for the arm that routes on the kind (prefill's fa2/sm90 fork); the typed accessors below are the single-kind reads.
    pub(crate) fn slot(&self, id: ValueId) -> &StructSlot {
        let at = self.struct_at(id);
        self.structs[at].as_ref().map(|(_, slot)| slot).unwrap_or_else(|| {
            panic!(
                "value {} holds no plan payload for run {} of its window; its plan \
                 op has not fired, and the prepare phase runs first",
                id.0,
                self.place.run.get(),
            )
        })
    }

    /// The decode plan a consuming arm names.
    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match self.slot(id) {
            StructSlot::Decode(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
        }
    }

    /// The fa2 prefill plan a consuming arm names.
    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match self.slot(id) {
            StructSlot::Prefill(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an fa2 prefill plan",
                id.0
            ),
        }
    }

    /// The mla plan a consuming arm names.
    pub(crate) fn mla_plan(&self, id: ValueId) -> &MlaPlan {
        match self.slot(id) {
            StructSlot::Mla(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an mla plan",
                id.0
            ),
        }
    }
}

/// A row-indexed handle, advanced past `skip` rows and cut to `keep` of them. The one arithmetic every windowed table shares.
fn skip(handle: Tensor, skip: u32, keep: u32) -> Tensor {
    if skip == 0 && keep >= handle.rows {
        return handle;
    }
    let stride = u64::from(handle.width)
        * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
            panic!(
                "a {:?} table has no element size and so no row to step by",
                handle.dtype
            )
        });
    Tensor::new(
        handle.ptr + u64::from(skip) * stride,
        keep.min(handle.rows.saturating_sub(skip)),
        handle.width,
        handle.dtype,
    )
}
