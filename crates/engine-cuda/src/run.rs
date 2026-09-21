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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),

    Planes {
        codes: Tensor,
        scales: Tensor,
        biases: Option<Tensor>,
        seat: GroupSeat,
        repacked: bool,
    },

    Streamed {
        slab: Tensor,
        table: u64,
        counts: u64,
    },
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    Kv { space: u32, pool: KvPool },
    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

#[derive(Clone, Debug)]
pub struct CachePlanning {
    pub kv_indptr: Vec<i32>,

    pub kv_len: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ScheduleSeat {
    pub shape: Shape,

    pub window: Option<u32>,

    pub workspace: Workspace,
}

#[derive(Clone, Copy, Debug)]
pub struct Planning<'a> {
    pub kv_indptr: &'a [i32],
    pub kv_len: &'a [i32],
    pub shape: Shape,
    pub live: Live,
    pub rows: u32,
    pub window: Option<u32>,
    pub workspace: Workspace,
}

#[derive(Clone, Debug, Default)]
pub struct CacheGeometry {
    pub indptr: Option<Tensor>,

    pub indices: Option<Tensor>,

    pub seq_lens: Option<Tensor>,

    pub last_page_len: Option<Tensor>,

    pub kv_len: Option<Tensor>,

    pub row_valid: Option<Tensor>,

    pub request_of_token: Option<Tensor>,

    pub write_page: Option<Tensor>,

    pub write_offset: Option<Tensor>,

    pub mask: Option<Tensor>,

    pub planning: Option<CachePlanning>,
}

#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    pub state_kv: Tensor,

    pub state_score: Tensor,

    pub ape: Tensor,
}

#[derive(Clone, Debug)]
pub struct FireTables {
    pub mask_indptr: Option<Tensor>,

    pub pool_state: Vec<(u32, PoolSlabs)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsMove<'a> {
    None,
    Scatter {
        pages: &'a [u32],
        at: u32,
        fold: u32,
        replay: u32,
    },
    Gather {
        pages: &'a [u32],
        at: u32,
    },
}

#[derive(Debug)]
pub struct RsScratch {
    ptr: u64,
    bytes: u64,
    cursor: Cell<u64>,
    ext: std::cell::RefCell<std::collections::HashMap<u32, Tensor>>,
}

impl RsScratch {
    #[must_use]
    pub fn new(ptr: u64, bytes: u64) -> Self {
        Self {
            ptr,
            bytes,
            cursor: Cell::new(0),
            ext: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    #[must_use]
    pub fn need(rows_ext: u32, ext_row_bytes: u64) -> u64 {
        u64::from(rows_ext.max(1)) * ext_row_bytes + 8 * 256 + 64 * 1024
    }

    fn take(&self, bytes: u64) -> crate::error::Result<u64> {
        let at = self.cursor.get();
        let end = at + bytes.max(4).next_multiple_of(256);
        if end > self.bytes {
            return Err(crate::error::Fault::Ceiling {
                what: "recurrent extended-run scratch bytes",
                need: end,
                have: self.bytes,
            });
        }
        self.cursor.set(end);
        Ok(self.ptr + at)
    }

    fn reset(&self) {
        self.cursor.set(0);
        self.ext.borrow_mut().clear();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RsSeat<'a> {
    pub buffers: &'a crate::store::rs::Buffers,
    pub lanes: &'a [RsMove<'a>],
    pub replays: &'a [u32],
    pub scratch: Option<&'a RsScratch>,
}

impl RsSeat<'_> {
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
            let mut done = 0u32;
            while done < count {
                let token = from + done;
                let page = token / page_tokens;
                let in_page = token % page_tokens;
                let take = (page_tokens - in_page).min(count - done);
                let page_slot = *pages
                    .get(page as usize)
                    .ok_or(crate::error::Fault::Ceiling {
                        what: "rs buffer pages",
                        need: u64::from(page) + 1,
                        have: pages.len() as u64,
                    })?;
                let slab = self.buffers.row(plane, page_slot, in_page)?;
                let rows_at =
                    rows.ptr + (u64::from(pair[0] as u32) + u64::from(done)) * plane.width * elem;
                let bytes =
                    usize::try_from(u64::from(take) * plane.width * elem).unwrap_or(usize::MAX);
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

    fn pages_to_rows(
        &self,
        stream: *mut core::ffi::c_void,
        plane: crate::store::rs::Plane,
        pages: &[u32],
        from: u32,
        count: u32,
        dst: u64,
    ) -> crate::error::Result<()> {
        let page_tokens = self.buffers.page_tokens();
        let elem = model_compiler::arena::elem_bytes(crate::store::rs::PLANE_DTYPE)
            .expect("the buffered planes are bf16, which has an element size");
        let row_bytes = plane.width * elem;
        let mut done = 0u32;
        while done < count {
            let token = from + done;
            let page = token / page_tokens;
            let in_page = token % page_tokens;
            let take = (page_tokens - in_page).min(count - done);
            let page_slot = *pages
                .get(page as usize)
                .ok_or(crate::error::Fault::Ceiling {
                    what: "rs buffer pages",
                    need: u64::from(page) + 1,
                    have: pages.len() as u64,
                })?;
            let slab = self.buffers.row(plane, page_slot, in_page)?;
            let bytes = usize::try_from(u64::from(take) * row_bytes).unwrap_or(usize::MAX);
            crate::device::copy_d2d(stream, dst + u64::from(done) * row_bytes, slab, bytes)?;
            done += take;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FireBindings {
    pub tokens: Tensor,

    pub positions: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub readout_rows: Tensor,

    pub patches: Option<Tensor>,
    pub patch_segments: Option<Tensor>,
    pub patch_routes: Option<Tensor>,
    pub patch_positions: Option<Tensor>,
    pub patch_embed_rows: Option<Tensor>,
    pub patch_embed_weights: Option<Tensor>,
    pub mrope_positions: Option<Tensor>,
    pub grid: Option<Tensor>,
    pub token_grid: Option<Tensor>,
    pub voxels: Option<Tensor>,
    pub clip_slots: Option<Tensor>,
    pub self_cond_rows: Option<Tensor>,
    pub self_cond_weights: Option<Tensor>,

    pub lane_of_row: Tensor,
    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<(model_ir::Selection, crate::inputs::PackingHandles)>,
    /// The fire's attention class table (`[count * count] u8`) when a lane
    /// stated classes: group-packed `attention.ragged` reads it.
    pub class_table: Option<(Tensor, u32)>,
    pub ports: Vec<PortBinding>,

    pub geometry: Vec<CacheGeometry>,

    pub schedules: Vec<Option<ScheduleSeat>>,
    pub plan_values: usize,

    pub tables: FireTables,

    pub scores: Option<crate::scores::ScoreSeat>,

    pub device: Device,

    pub toggles: Toggles,

    pub capture: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PortBinding {
    pub kind: engine::fire::PortKind,
    pub port: u8,
    pub tensor: Tensor,
}

#[derive(Clone, Debug)]
pub enum StructSlot {
    Decode(DecodePlan),

    Prefill(PrefillPlan),

    PrefillSm90(PrefillPlanSm90),

    Mla(MlaPlan),
}

#[derive(Clone, Copy, Default)]
pub struct Ceilings<'c> {
    pub pads: model_ir::PerAxis<Pad>,

    pub bodied: bool,

    pub shifted: &'c [bool],

    pub admits: &'c [Admit],

    pub readers: &'c [Option<u32>],

    pub carve: Option<Carve<'c>>,
}

impl<'c> Ceilings<'c> {
    fn pad_on(&self, axis: model_ir::RowAxis) -> Pad {
        self.pads[axis]
    }

    fn admit(&self, region: u32) -> Option<Admit> {
        self.bodied
            .then(|| self.admits.get(region as usize).copied())
            .flatten()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Standing {
    pad: Pad,

    whole: bool,

    held: Held,
}

#[derive(Clone, Copy)]
enum Held {
    Eager,

    Captured {
        plane: bool,

        ceiling: Option<(u32, u32)>,

        lanes: Option<(u32, u32)>,
    },
}

impl Standing {
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

    fn plane(&self) -> bool {
        matches!(self.held, Held::Captured { plane: true, .. })
    }

    fn absolute(&self) -> bool {
        match self.held {
            Held::Eager => false,
            Held::Captured { plane, .. } => plane || self.whole,
        }
    }

    fn armed(&self) -> bool {
        match self.held {
            Held::Eager => false,
            Held::Captured { plane, .. } => self.whole || plane,
        }
    }

    fn ceiling(&self) -> Option<(u32, u32)> {
        match self.held {
            Held::Eager => None,
            Held::Captured { ceiling, .. } => ceiling,
        }
    }

    fn lane_carve(&self) -> Option<(u32, u32)> {
        match self.held {
            Held::Eager => None,
            Held::Captured { lanes, .. } => lanes,
        }
    }

    pub(crate) fn rows(&self, span: MaskSpan) -> Option<u32> {
        let Held::Captured { plane, ceiling, .. } = self.held else {
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

pub struct Run<'c> {
    ctx: &'c Ctx,

    values: &'c [ValueDecl],

    nodes: &'c [Node],

    weights: &'c WeightTable,

    arena: &'c SlotTable,

    caches: &'c CacheTable,

    structs: Vec<Option<(Admit, StructSlot)>>,
    values_wide: usize,

    fire: FireBindings,

    rs: Option<RsSeat<'c>>,

    windows: &'c Windows,

    place: &'c At,

    side: &'c [&'c Ctx],

    stream: &'c Cell<u32>,

    body: Option<&'c Ctx>,

    copy: CopyPlan,

    ceilings: Ceilings<'c>,

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

    #[must_use]
    pub fn ceilings(mut self, ceilings: Ceilings<'c>) -> Self {
        self.ceilings = ceilings;
        self
    }

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

    fn reading_standing(&self, plan: ValueId) -> Standing {
        let run = self.place.run.get();
        let here = self.place.region.get();
        let eager = || self.standing_as(here, run, false);
        let Some(region) = self
            .ceilings
            .readers
            .get(plan.0 as usize)
            .copied()
            .flatten()
        else {
            return eager();
        };
        if run >= self.windows.runs(region) {
            return eager();
        }
        self.standing_at(region, run)
    }

    pub(crate) fn standing_at(&self, region: u32, run: u32) -> Standing {
        self.standing_as(
            region,
            run,
            matches!(self.ceilings.admit(region), Some(Admit::Captured)),
        )
    }

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

    pub(crate) fn windows(&self) -> &'c Windows {
        self.windows
    }

    pub fn buffered(mut self, rs: RsSeat<'c>) -> Self {
        self.rs = Some(rs);
        self
    }

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
        let rows = self.windowed(rows);
        seat.run(self.ctx.stream(), plane, span.lane_offset, bounds, rows)
            .map_err(|fault| kernels_cuda::Error::Backend {
                op,
                detail: fault.to_string(),
            })
    }

    pub(crate) fn rs_extended(&self) -> bool {
        let Some(seat) = self.rs.as_ref() else {
            return false;
        };
        if seat.scratch.is_none() {
            return false;
        }
        let lane0 = self.window().span().lane_offset as usize;
        let lanes = self.qo_indptr_host().len().saturating_sub(1);
        seat.replays
            .iter()
            .skip(lane0)
            .take(lanes)
            .any(|&replay| replay > 0)
    }

    fn rs_ext_layout(&self, seat: &RsSeat<'_>) -> (Vec<(u32, u32, u32)>, Vec<i32>) {
        let lane0 = self.window().span().lane_offset as usize;
        let bounds = self.qo_indptr_host();
        let mut lanes = Vec::with_capacity(bounds.len().saturating_sub(1));
        let mut csr = Vec::with_capacity(bounds.len());
        let mut begin = 0u32;
        csr.push(0i32);
        for (r, pair) in bounds.windows(2).enumerate() {
            let rows = (pair[1] - pair[0]).max(0) as u32;
            let replay = seat.replays.get(lane0 + r).copied().unwrap_or(0);
            lanes.push((begin, replay, rows));
            begin += replay + rows;
            csr.push(i32::try_from(begin).unwrap_or(i32::MAX));
        }
        (lanes, csr)
    }

    fn rs_fault(op: &'static str, fault: crate::error::Fault) -> kernels_cuda::Error {
        kernels_cuda::Error::Backend {
            op,
            detail: fault.to_string(),
        }
    }

    fn rs_seat_and_scratch(
        &self,
        op: &'static str,
    ) -> Result<(&RsSeat<'c>, &'c RsScratch), kernels_cuda::Error> {
        let seat = self
            .rs
            .as_ref()
            .ok_or_else(|| kernels_cuda::Error::Backend {
                op,
                detail: "the extended recurrent run of a fire with no buffered plane".to_string(),
            })?;
        let scratch = seat.scratch.ok_or_else(|| kernels_cuda::Error::Backend {
            op,
            detail: "the extended recurrent run of a fire the shell grew no scratch for"
                .to_string(),
        })?;
        Ok((seat, scratch))
    }

    pub(crate) fn rs_ext_csr(&self, op: &'static str) -> Result<Tensor, kernels_cuda::Error> {
        let (seat, scratch) = self.rs_seat_and_scratch(op)?;
        let (_, csr) = self.rs_ext_layout(seat);
        let bytes: Vec<u8> = csr.iter().flat_map(|n| n.to_le_bytes()).collect();
        let at = scratch
            .take(bytes.len() as u64)
            .map_err(|f| Self::rs_fault(op, f))?;
        crate::device::stage_raw(self.ctx.stream(), at, &bytes)
            .map_err(|f| Self::rs_fault(op, f))?;
        Ok(Tensor::new(at, csr.len() as u32, 1, model_ir::Dtype::I32))
    }

    pub(crate) fn rs_extend(
        &self,
        op: &'static str,
        id: ValueId,
        buffered: bool,
    ) -> Result<Tensor, kernels_cuda::Error> {
        let (seat, scratch) = self.rs_seat_and_scratch(op)?;
        let own = self.tensor(id);
        self.rs_move(op, id, own)?;
        let own = self.windowed(own);
        let plane = seat.buffers.planes().of(id);
        if buffered && plane.is_none() {
            return Err(kernels_cuda::Error::Backend {
                op,
                detail: format!(
                    "value {} is no plane this load buffers, so a lane's replay has nowhere to \
                     read its tokens from; the read path serves the chunked recurrence's class",
                    id.0
                ),
            });
        }
        let (lanes, csr) = self.rs_ext_layout(seat);
        let rows_ext = u32::try_from(csr.last().copied().unwrap_or(0)).unwrap_or(0);
        let elem = model_compiler::arena::elem_bytes(own.dtype).ok_or_else(|| {
            kernels_cuda::Error::Backend {
                op,
                detail: format!("{:?} has no element size", own.dtype),
            }
        })?;
        let row_bytes = u64::from(own.width) * elem;
        let at = scratch
            .take(u64::from(rows_ext) * row_bytes)
            .map_err(|f| Self::rs_fault(op, f))?;
        let ext = Tensor::new(at, rows_ext, own.width, own.dtype);
        let lane0 = self.window().span().lane_offset as usize;
        let bounds = self.qo_indptr_host();
        let stream = self.ctx.stream();
        for (r, &(begin, replay, rows)) in lanes.iter().enumerate() {
            if replay > 0
                && let Some(plane) = plane
                && let Some(RsMove::Scatter {
                    pages, at: token, ..
                }) = seat.lanes.get(lane0 + r)
            {
                if u64::from(plane.width) != u64::from(own.width) {
                    return Err(kernels_cuda::Error::Backend {
                        op,
                        detail: format!(
                            "a buffered plane reserved at {} elements a token is bound {} wide this fire",
                            plane.width, own.width
                        ),
                    });
                }
                let from = token.checked_sub(replay).ok_or_else(|| {
                    kernels_cuda::Error::Backend {
                        op,
                        detail: format!(
                            "lane replays {replay} buffered token(s) below buffer position {token}"
                        ),
                    }
                })?;
                seat.pages_to_rows(
                    stream,
                    plane,
                    pages,
                    from,
                    replay,
                    at + u64::from(begin) * row_bytes,
                )
                .map_err(|f| Self::rs_fault(op, f))?;
            }
            if rows > 0 {
                let src = own.ptr + u64::from(bounds[r].max(0) as u32) * row_bytes;
                let dst = at + u64::from(begin + replay) * row_bytes;
                crate::device::copy_d2d(
                    stream,
                    dst,
                    src,
                    usize::try_from(u64::from(rows) * row_bytes).unwrap_or(usize::MAX),
                )
                .map_err(|f| Self::rs_fault(op, f))?;
            }
        }
        Ok(ext)
    }

    pub(crate) fn rs_out(
        &self,
        op: &'static str,
        id: ValueId,
    ) -> Result<Tensor, kernels_cuda::Error> {
        let (seat, scratch) = self.rs_seat_and_scratch(op)?;
        let own = self.windowed(self.tensor(id));
        let (_, csr) = self.rs_ext_layout(seat);
        let rows_ext = u32::try_from(csr.last().copied().unwrap_or(0)).unwrap_or(0);
        let elem = model_compiler::arena::elem_bytes(own.dtype).ok_or_else(|| {
            kernels_cuda::Error::Backend {
                op,
                detail: format!("{:?} has no element size", own.dtype),
            }
        })?;
        let at = scratch
            .take(u64::from(rows_ext) * u64::from(own.width) * elem)
            .map_err(|f| Self::rs_fault(op, f))?;
        let ext = Tensor::new(at, rows_ext, own.width, own.dtype);
        scratch.ext.borrow_mut().insert(id.0, ext);
        Ok(ext)
    }

    pub(crate) fn rs_ext_of(
        &self,
        op: &'static str,
        id: ValueId,
    ) -> Result<Tensor, kernels_cuda::Error> {
        let (_, scratch) = self.rs_seat_and_scratch(op)?;
        scratch
            .ext
            .borrow()
            .get(&id.0)
            .copied()
            .ok_or_else(|| kernels_cuda::Error::Backend {
                op,
                detail: format!(
                    "value {} was not landed extended by an earlier recurrent op of this fire",
                    id.0
                ),
            })
    }

    pub(crate) fn rs_land(
        &self,
        op: &'static str,
        ext: Tensor,
        id: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let (seat, _) = self.rs_seat_and_scratch(op)?;
        let target = self.windowed(self.tensor(id));
        let elem = model_compiler::arena::elem_bytes(ext.dtype).ok_or_else(|| {
            kernels_cuda::Error::Backend {
                op,
                detail: format!("{:?} has no element size", ext.dtype),
            }
        })?;
        let row_bytes = u64::from(ext.width) * elem;
        let (lanes, _) = self.rs_ext_layout(seat);
        let bounds = self.qo_indptr_host();
        let stream = self.ctx.stream();
        for (r, &(begin, replay, rows)) in lanes.iter().enumerate() {
            if rows == 0 {
                continue;
            }
            let src = ext.ptr + u64::from(begin + replay) * row_bytes;
            let dst = target.ptr + u64::from(bounds[r].max(0) as u32) * row_bytes;
            crate::device::copy_d2d(
                stream,
                dst,
                src,
                usize::try_from(u64::from(rows) * row_bytes).unwrap_or(usize::MAX),
            )
            .map_err(|f| Self::rs_fault(op, f))?;
        }
        Ok(())
    }

    pub(crate) fn rs_layer_done(&self) {
        if let Some(scratch) = self.rs.as_ref().and_then(|seat| seat.scratch) {
            scratch.reset();
        }
    }

    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    pub(crate) fn set_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    #[must_use]
    pub fn across(mut self, side: &'c [&'c Ctx], stream: &'c Cell<u32>) -> Self {
        self.side = side;
        self.stream = stream;
        self
    }

    #[must_use]
    pub fn conditional(mut self, body: &'c Ctx, stream: &'c Cell<u32>) -> Self {
        self.body = Some(body);
        self.stream = stream;
        self
    }

    pub(crate) fn ctx(&self) -> &'c Ctx {
        let ctx = match self.body {
            Some(body) if self.stream.get() == crate::window::BODY => body,
            _ if self.side.is_empty() => self.ctx,
            _ => match self.stream.get() {
                0 => self.ctx,
                n => self.side.get(n as usize - 1).copied().unwrap_or(self.ctx),
            },
        };
        ctx.arm(self.here());
        ctx.arm_stage(self.live_at());
        let region = self.place.region.get();
        ctx.arm_region(region);
        ctx.arm_lane(self.place.lane.get());
        ctx
    }

    fn here(&self) -> Pad {
        let standing = self.standing();
        let pad = standing.pad;
        if pad.bucket <= pad.rows {
            return Pad::default();
        }
        if standing.whole { pad } else { Pad::default() }
    }

    fn plane_base(&self) -> bool {
        self.standing().plane()
    }

    fn absolute_base(&self) -> bool {
        self.standing().absolute()
    }

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

    fn carve_rows(&self) -> Option<u32> {
        self.standing().rows(self.window().span())
    }

    fn carve_lanes(&self) -> Option<u32> {
        self.standing().lanes(self.windows, self.window().span())
    }

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

    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    pub(crate) fn window(&self) -> &'c Window {
        self.windows
            .at(self.place.region.get(), self.place.run.get())
    }

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

    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    pub(crate) fn qo_indptr_absolute(&self) -> Option<Tensor> {
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

    pub(crate) fn qo_indptr_absolute_host(&self) -> &'c [i32] {
        let span = self.window().span();
        let first = span.lane_offset as usize;
        let last = first + span.lanes as usize;
        self.windows
            .qo_absolute_host()
            .get(first..=last)
            .unwrap_or_default()
    }

    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span().rows
    }

    pub(crate) fn mask_indptr(&self) -> Option<Tensor> {
        if self.plane_base() {
            return self.fire.tables.mask_indptr;
        }
        let span = self.window().span();
        self.fire
            .tables
            .mask_indptr
            .map(|table| skip(table, span.lane_offset, span.lanes + 1))
    }

    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|span| span[1] - span[0] > 1)
    }

    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
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
        let plane = self.plane_base();
        let ceiling = self.carve_rows();
        let rows = ceiling.unwrap_or(window.rows);

        let Some(&dim) = shape.first() else {
            return handle;
        };
        let Some(axis) = dim.axis() else {
            return handle;
        };
        let span = seated.on(axis);

        let row = |times: u32| {
            let primary = axis == model_ir::RowAxis::PRIMARY;
            let extent = if primary { rows } else { span.rows };
            let offset = if primary && plane { 0 } else { span.row_offset };
            (offset * times, extent * times)
        };
        let lane = |plus: u32| (span.lane_offset, span.lanes + plus);

        let (skip, keep) = match dim {
            Dim::Const(_) => return handle,
            Dim::Tokens => row(1),
            Dim::TokensTimes(k) => row(k),
            Dim::Patches => row(1),
            Dim::Readouts => return handle,
            Dim::Lanes => return handle,
            Dim::LanesPlus(k) => lane(k),
            Dim::Images => lane(0),
            Dim::ImagesPlus(k) => lane(k),
            Dim::Voxels => row(1),
            Dim::VoxelsTimes(k) => row(k),
            Dim::Clips => lane(0),
            Dim::ClipsPlus(k) => lane(k),
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

    pub(crate) fn packing(
        &self,
        at: usize,
        select: model_ir::Selection,
    ) -> crate::inputs::PackingHandles {
        self.fire
            .packings
            .iter()
            .find(|(have, _)| *have == select)
            .map(|(_, tables)| *tables)
            .unwrap_or_else(|| {
                panic!(
                    "value {at} reads the packing tables of selection {select:?}, which this \
                     fire staged none of; the load carves one per selection the plan reads"
                )
            })
    }

    fn port(&self, at: usize, kind: engine::fire::PortKind, port: u8) -> Tensor {
        self.fire
            .ports
            .iter()
            .find(|bound| bound.kind == kind && bound.port == port)
            .map(|bound| bound.tensor)
            .unwrap_or_else(|| {
                panic!(
                    "value {at} reads {kind:?} port {port}, which this load carved no \
                     rectangle for"
                )
            })
    }

    pub(crate) fn run_index(&self) -> u32 {
        self.place.run.get()
    }

    pub(crate) fn read_elsewhere(&self, normed: ValueId) -> bool {
        use model_ir::Operands as _;
        let mut inputs: Vec<ValueId> = Vec::new();
        self.nodes.iter().any(|node| {
            inputs.clear();
            node.op.inputs(&mut inputs);
            inputs.contains(&normed)
        })
    }

    pub(crate) fn lane_shaped(&self, id: ValueId) -> bool {
        matches!(
            &self.values[id.0 as usize].ty,
            Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Lanes))
        )
    }

    pub(crate) fn unseated<T>(&self, launch: impl FnOnce() -> T) -> T {
        let held = match self.ctx().stage() {
            kernels_cuda::ArgValue::Ptr(at) => at,
            _ => 0,
        };
        self.ctx().disarm_stage();
        let out = launch();
        if held != 0 {
            self.ctx().arm_stage(held);
        }
        out
    }

    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    pub(crate) fn resolvable(&self, id: ValueId) -> bool {
        let at = id.0 as usize;
        match self.values.get(at).map(|decl| &decl.def) {
            Some(Def::Op(_) | Def::Merge(_)) => self.arena.0.get(at).copied().flatten().is_some(),
            Some(Def::Weight(w)) => matches!(
                self.weights.0.get(*w as usize).copied().flatten(),
                Some(WeightRow::Dense(_) | WeightRow::Streamed { .. })
            ),
            _ => false,
        }
    }

    pub(crate) fn fire_wide(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    pub(crate) fn class_table(&self) -> Option<(Tensor, u32)> {
        self.fire.class_table
    }

    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::ReadoutRows) => self.fire.readout_rows,
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
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!("value {at} reads this fire's adapter ids, which no lane of it carried")
                })
            }
            Def::Input(RuntimeInput::Patches) => self.fire.patches.unwrap_or_else(|| {
                panic!("value {at} reads this fire's patch rows, which no lane of it submitted")
            }),
            Def::Input(RuntimeInput::PatchSegments) => {
                self.fire.patch_segments.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's image boundaries, which no lane of it \
                         submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchRoutes) => self.fire.patch_routes.unwrap_or_else(|| {
                panic!(
                    "value {at} reads where this fire's tower rows land, and no lane of it \
                         submitted an image"
                )
            }),
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
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's (t, h, w) token positions, which this load \
                         reserved no stream for"
                    )
                })
            }
            Def::Input(RuntimeInput::Grid) => self.fire.grid.unwrap_or_else(|| {
                panic!("value {at} reads this fire's clip grid, which no lane of it submitted")
            }),
            Def::Input(RuntimeInput::TokenGrid { .. }) => {
                self.fire.token_grid.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's token-side clip grid, which no lane of \
                         it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::Voxels { channels, .. }) => {
                let port = self.fire.voxels.unwrap_or_else(|| {
                    panic!("value {at} reads this fire's voxel port, which no lane of it fed")
                });
                assert_eq!(
                    port.width, *channels,
                    "value {at} reads a {channels}-wide voxel port and this fire fed a {}-wide \
                     one: the clips ran in another reading's class",
                    port.width
                );
                port
            }
            Def::Input(RuntimeInput::RowPermutation { select }) => {
                self.packing(at, *select).permutation
            }
            Def::Input(RuntimeInput::Latents { port, .. }) => {
                self.port(at, engine::fire::PortKind::Latents, *port)
            }
            Def::Input(RuntimeInput::LaneVector { port, .. }) => {
                self.port(at, engine::fire::PortKind::LaneVector, *port)
            }
            Def::Input(RuntimeInput::Context { port, .. }) => {
                self.port(at, engine::fire::PortKind::Context, *port)
            }
            Def::Input(RuntimeInput::AxisPositions { port, .. }) => {
                self.port(at, engine::fire::PortKind::AxisPositions, *port)
            }
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                match kind {
                    GeomKind::RequestOfToken => return self.fire.lane_of_row,
                    GeomKind::GroupOfLane => {
                        return self.fire.group_of_lane.unwrap_or_else(|| {
                            panic!(
                                "value {at} reads the group table, which this fire staged none of"
                            )
                        });
                    }
                    GeomKind::GroupIndptr { select } => {
                        return self.packing(at, *select).group_indptr;
                    }
                    GeomKind::LaneIndptr { select } => {
                        return self.packing(at, *select).lane_indptr;
                    }
                    GeomKind::ReferenceTag { select } => {
                        return self.packing(at, *select).reference_tag;
                    }
                    _ => {}
                }
                let seat = self.geometry(at, *space);
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                    GeomKind::KvLen => seat.kv_len,
                    GeomKind::RowValid => seat.row_valid,
                    GeomKind::RequestOfToken => unreachable!("the row-to-lane map returns early"),
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                    GeomKind::GroupOfLane
                    | GeomKind::GroupIndptr { .. }
                    | GeomKind::LaneIndptr { .. }
                    | GeomKind::ReferenceTag { .. } => unreachable!("the D2 tables return early"),
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

    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    pub(crate) fn ragged_q(&self, id: ValueId) -> RaggedTensor {
        let indptr = if self.absolute_base() {
            self.qo_indptr_absolute()
                .unwrap_or_else(|| self.qo_indptr())
        } else {
            self.qo_indptr()
        };
        RaggedTensor {
            data: self.tensor(id),
            indptr,
        }
    }

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

    pub(crate) fn maybe_stored(&self, id: ValueId) -> Option<Tensor> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        let Some(WeightRow::Dense(handle)) = self.weights.0.get(*w as usize).copied().flatten()
        else {
            return None;
        };
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
            }) => (
                self.cut(id, slab),
                ExpertTable {
                    table,
                    hits: counts,
                },
            ),
            Some(WeightRow::Planes { .. }) => panic!(
                "value {at} is weight {row}, a split-plane bank; a dense routed select \
                 does not read one"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    pub(crate) fn pool(&self, id: ValueId) -> KvPool {
        match self.cache(id) {
            CachePool::Kv { space, pool } => {
                let window = self.window().span();
                if let Some(gathered) = &self.window().gathered {
                    let seat = gathered.spaces.get(*space as usize);
                    return KvPool {
                        page_indptr: seat.map_or(pool.page_indptr, |seat| seat.page_indptr),
                        page_indices: seat.map_or(pool.page_indices, |seat| seat.page_indices),
                        last_page_lens: seat
                            .map_or(pool.last_page_lens, |seat| seat.last_page_lens),
                        row_valid: skip(pool.row_valid, 0, window.rows),
                        ..*pool
                    };
                }
                KvPool {
                    page_indptr: skip(pool.page_indptr, window.lane_offset, window.lanes + 1),
                    last_page_lens: skip(pool.last_page_lens, window.lane_offset, window.lanes),
                    row_valid: if self.plane_base() {
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

    pub(crate) fn pool_absolute(&self, id: ValueId) -> KvPool {
        let pool = self.pool(id);
        if !self.plane_base() {
            return pool;
        }
        match self.cache(id) {
            CachePool::Kv { pool: whole, .. } => KvPool {
                page_indptr: whole.page_indptr,
                last_page_lens: whole.last_page_lens,
                ..pool
            },
            CachePool::Recurrent(_) => pool,
        }
    }

    pub(crate) fn windowed(&self, handle: Tensor) -> Tensor {
        if !self.plane_base() {
            return handle;
        }
        let window = self.window().span();
        skip(handle, window.row_offset, window.rows)
    }

    pub(crate) fn clip_slots(&self) -> Option<Tensor> {
        self.fire.clip_slots
    }

    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, false)
        }
    }

    pub(crate) fn recurrent_absolute(&self, id: ValueId) -> RecurrentPool {
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, true)
        }
    }

    pub(crate) fn recurrent_tail_absolute(&self, id: ValueId) -> Option<RecurrentPool> {
        let cut = self.recurrent_cut(id, true);
        if cut.begin_at.is_absent() {
            return None;
        }
        Some(RecurrentPool {
            write_state: false,
            commit_len: Tensor::ABSENT,
            ..cut
        })
    }

    fn recurrent_cut(&self, id: ValueId, absolute: bool) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => {
                let window = self.window().span();
                assert!(
                    absolute || !self.plane_base() || window.lane_offset == 0,
                    "value {} takes the window-local recurrent lane door under a \
                     plane base at lane offset {}; a body would bake that slice \
                     and replay it at another split (`crate::lane_shifted`)",
                    id.0,
                    window.lane_offset,
                );
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
        let standing = self.reading_standing(plan);
        let carve = standing.ceiling();
        let carve_lanes = standing.lane_carve();
        let staged = self
            .windows
            .qo_absolute()
            .map_or(0, |bounds| bounds.rows.saturating_sub(1))
            .min((seat.kv_indptr.len() as u32).saturating_sub(1))
            .min(seat.kv_len.len() as u32);
        let ceiling: Option<(u32, u32)> = standing
            .absolute()
            .then_some(carve_lanes)
            .flatten()
            .and_then(|(before, own)| {
                let covered = staged.checked_sub(before)?;
                Some((before, own.min(covered)))
            })
            .filter(|(before, lanes)| {
                *lanes >= span.lanes && before + lanes >= span.lane_offset + span.lanes
            });
        let kind = self.declared(plan);
        let rows_ceiling: Option<u32> = (!matches!(kind, StructKind::AttnDecodePlan))
            .then_some(carve)
            .flatten()
            .map(|(_, own)| own.min(standing.pad.bucket))
            .filter(|rows| *rows >= span.rows);
        let (kv_indptr, kv_len) = match window
            .gathered
            .as_ref()
            .and_then(|g| g.spaces.get(*space as usize))
        {
            Some(gathered) => (
                gathered.page_indptr_host.as_slice(),
                gathered.kv_len_host.as_slice(),
            ),
            None => {
                let first = span.lane_offset as usize;
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
        let shape = Shape {
            num_requests: ceiling.map_or(span.lanes, |(_, lanes)| lanes),
            lane_offset: match ceiling {
                Some((_, lanes)) => staged.saturating_sub(lanes),
                None if standing.plane() => span.lane_offset,
                None => 0,
            },
            ..schedule.shape
        };
        let live = Live {
            requests: span.lanes,
            lane_offset: if standing.plane() {
                span.lane_offset
            } else {
                0
            },
            row_offset: if standing.plane() { span.row_offset } else { 0 },
            rows: span.rows,
        };
        assert!(
            shape.num_requests >= live.requests,
            "a carve is never narrower than the fire"
        );
        assert!(
            shape.lane_offset >= live.lane_offset,
            "a carve starts at or before the fire"
        );
        assert!(shape.lane_offset + shape.num_requests >= live.lane_offset + live.requests);
        let rows = rows_ceiling.unwrap_or(span.rows);
        assert!(
            rows >= live.rows,
            "a row carve is never narrower than the fire"
        );
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

    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

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
            if *admit == Admit::Island {
                continue;
            }
            let _ = write!(sink, "{at}:");
            let _ = match slot {
                StructSlot::Decode(p) => write!(
                    sink,
                    "d{:?}{:?}{:?}{:?}",
                    p.info, p.workspace, p.shape, p.window
                ),
                StructSlot::Prefill(p) => write!(
                    sink,
                    "p{:?}{:?}{:?}{:?}{}{}{}{:?}",
                    p.info,
                    p.workspace,
                    p.shape,
                    p.window,
                    p.total_tokens,
                    p.causal,
                    p.graph_capturable,
                    p.mask_indptr
                ),
                StructSlot::PrefillSm90(p) => write!(
                    sink,
                    "s{:?}{:?}{:?}{}{}",
                    p.info, p.workspace, p.shape, p.total_tokens, p.causal
                ),
                StructSlot::Mla(p) => write!(
                    sink,
                    "m{:?}{:?}{}{}",
                    p.info, p.workspace, p.num_heads, p.causal
                ),
            };
        }
        sink.0.finish()
    }

    pub(crate) fn capturable(&self) -> bool {
        self.structs.iter().flatten().all(|(_, slot)| match slot {
            StructSlot::Prefill(plan) => plan.graph_capturable,
            _ => true,
        })
    }

    pub(crate) fn uncapturable_why(&self) -> Vec<String> {
        self.structs
            .iter()
            .flatten()
            .filter_map(|(_, slot)| match slot {
                StructSlot::Prefill(plan) => plan.graph_refusal.clone(),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        let admit = self
            .ceilings
            .admit(self.place.region.get())
            .unwrap_or(Admit::Captured);
        self.structs[at] = Some((admit, built));
    }

    pub(crate) fn slot(&self, id: ValueId) -> &StructSlot {
        let at = self.struct_at(id);
        self.structs[at]
            .as_ref()
            .map(|(_, slot)| slot)
            .unwrap_or_else(|| {
                panic!(
                    "value {} holds no plan payload for run {} of its window; its plan \
                 op has not fired, and the prepare phase runs first",
                    id.0,
                    self.place.run.get(),
                )
            })
    }

    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match self.slot(id) {
            StructSlot::Decode(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
        }
    }

    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match self.slot(id) {
            StructSlot::Prefill(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an fa2 prefill plan",
                id.0
            ),
        }
    }

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
