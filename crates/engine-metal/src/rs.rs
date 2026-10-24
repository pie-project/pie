use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::rs::{LanePlan, Run};
use kernels_metal::Tensor;
use poem_ir::{Def, Dim, Dtype, Gather, Operation, Ssm, Trace, Ty, ValueId};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::store::kv::Paging;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    ConvX,
    Ba,
    Ids,
    KdaF,
    KdaB,
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub cache: u32,
    pub role: Role,
    pub width: u32,
    pub dtype: Dtype,
    pub row_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub width: u32,
    pub dtype: Dtype,
    pub row_bytes: u64,
}

#[derive(Debug, Default)]
pub struct Layout {
    pub planes: Vec<Plane>,
    pub regions: Vec<Region>,
    pub in_of: HashMap<u32, usize>,
    pub out_of: HashMap<u32, usize>,
    pub token_bytes: u64,
    pub work_elems: u64,
}

/// The snapshot rows one lane lands in a fire (`Lane::rs_snapshots`):
/// [`kernels_metal::ssm::RS_SNAPSHOTS`] where every op that reads a
/// state slot is a conv or gated-delta scan, whose prompt arms land them;
/// else none, and a pass is cut at each.
#[must_use]
pub fn snapshots(trace: &Trace) -> u32 {
    use poem_ir::Operands;
    let state = |id: &ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|value| &value.def),
            Some(Def::Cache(row))
                if matches!(trace.caches.get(*row as usize), Some(poem_ir::CacheRow::State { .. }))
        )
    };
    let mut reads = Vec::new();
    let mut any = false;
    let lands = trace.nodes.iter().all(|node| {
        reads.clear();
        node.op.inputs(&mut reads);
        if !reads.iter().any(state) {
            return true;
        }
        any = true;
        matches!(
            &node.op,
            Operation::Ssm(
                Ssm::CausalConv1d { .. }
                    | Ssm::CausalConv1dChunked { .. }
                    | Ssm::GatedDelta { .. }
                    | Ssm::GatedDeltaChunked { .. },
            )
        )
    });
    if lands && any {
        kernels_metal::ssm::RS_SNAPSHOTS
    } else {
        0
    }
}

fn row_of(trace: &Trace, id: ValueId) -> Result<(u32, Dtype, u64)> {
    let decl = trace
        .values
        .get(id.0 as usize)
        .ok_or_else(|| Fault::Unbound {
            what: format!("value {}, which the trace does not declare", id.0),
        })?;
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return Err(Fault::Unbound {
            what: format!(
                "value {}, a struct where the recurrence reads token rows",
                id.0
            ),
        });
    };
    let width: u64 = shape
        .iter()
        .skip(1)
        .map(|dim| match dim {
            Dim::Const(n) => *n,
            _ => 1,
        })
        .product();
    let element = poem_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
        what: format!("value {}, whose element {dtype:?} has no size", id.0),
    })?;
    Ok((
        u32::try_from(width).unwrap_or(u32::MAX),
        *dtype,
        width * element,
    ))
}

fn cache_of(trace: &Trace, state: ValueId) -> Result<u32> {
    match trace.values.get(state.0 as usize).map(|decl| &decl.def) {
        Some(Def::Cache(row)) => Ok(*row),
        _ => Err(Fault::Unbound {
            what: format!(
                "value {}, read as a recurrent state and declared as no cache",
                state.0
            ),
        }),
    }
}

impl Layout {
    pub fn read(trace: &Trace) -> Result<Option<Layout>> {
        let mut layout = Layout::default();
        let mut plane_keys: HashMap<(u32, Role), usize> = HashMap::new();
        let mut region_keys: HashMap<(u32, &'static str), usize> = HashMap::new();
        let mut gates_cache: HashMap<u32, u32> = HashMap::new();

        let mut plane = |layout: &mut Layout,
                         cache: u32,
                         role: Role,
                         value: ValueId|
         -> Result<()> {
            let (width, dtype, row_bytes) = row_of(trace, value)?;
            let at = match plane_keys.get(&(cache, role)) {
                Some(&at) => {
                    let have = layout.planes[at];
                    if have.width != width || have.dtype != dtype {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache {cache}'s {role:?} plane, read {width} wide as {dtype:?} by value {} \
                                 and {} wide as {:?} elsewhere",
                                value.0, have.width, have.dtype
                            ),
                        });
                    }
                    at
                }
                None => {
                    layout.planes.push(Plane {
                        cache,
                        role,
                        width,
                        dtype,
                        row_bytes,
                    });
                    plane_keys.insert((cache, role), layout.planes.len() - 1);
                    layout.planes.len() - 1
                }
            };
            layout.in_of.insert(value.0, at);
            Ok(())
        };
        let mut region =
            |layout: &mut Layout, cache: u32, what: &'static str, value: ValueId| -> Result<()> {
                let (width, dtype, row_bytes) = row_of(trace, value)?;
                let at = match region_keys.get(&(cache, what)) {
                    Some(&at) => at,
                    None => {
                        layout.regions.push(Region {
                            width,
                            dtype,
                            row_bytes,
                        });
                        region_keys.insert((cache, what), layout.regions.len() - 1);
                        layout.regions.len() - 1
                    }
                };
                layout.out_of.insert(value.0, at);
                Ok(())
            };

        for node in &trace.nodes {
            match &node.op {
                Operation::Ssm(
                    Ssm::CausalConv1d { x, state, y, .. }
                    | Ssm::CausalConv1dChunked { x, state, y, .. },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::ConvX, *x)?;
                    region(&mut layout, cache, "conv", *y)?;
                }
                Operation::Ssm(
                    Ssm::GatedDelta {
                        gates,
                        state,
                        k_heads,
                        v_heads,
                        k_dim,
                        v_dim,
                        y,
                        ..
                    }
                    | Ssm::GatedDeltaChunked {
                        gates,
                        state,
                        k_heads,
                        v_heads,
                        k_dim,
                        v_dim,
                        y,
                        ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    let _ = k_heads;
                    gates_cache.insert(gates.0, cache);
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*v_heads) * u64::from(*v_dim) * u64::from(*k_dim));
                }
                Operation::Ssm(
                    Ssm::Kda {
                        f,
                        b,
                        state,
                        heads,
                        head_dim,
                        y,
                        ..
                    }
                    | Ssm::KdaChunked {
                        f,
                        b,
                        state,
                        heads,
                        head_dim,
                        y,
                        ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::KdaF, *f)?;
                    plane(&mut layout, cache, Role::KdaB, *b)?;
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*heads) * u64::from(*head_dim) * u64::from(*head_dim));
                }
                Operation::Gather(
                    Gather::NgramIds {
                        ids,
                        state,
                        ngram_ids,
                        ..
                    }
                    | Gather::NgramIdsChunked {
                        ids,
                        state,
                        ngram_ids,
                        ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::Ids, *ids)?;
                    region(&mut layout, cache, "ngram", *ngram_ids)?;
                }
                _ => {}
            }
        }
        for node in &trace.nodes {
            let Operation::Ssm(Ssm::GdnGates { ba, gates, .. }) = &node.op else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `ssm.gdn_gates` lands, which no delta scan reads",
                    gates.0
                ),
            })?;
            plane(&mut layout, cache, Role::Ba, *ba)?;
            region(&mut layout, cache, "gates", *gates)?;
        }
        if layout.planes.is_empty() {
            return Ok(None);
        }
        layout.token_bytes = layout.planes.iter().map(|plane| plane.row_bytes).sum();
        Ok(Some(layout))
    }
}

#[derive(Debug)]
pub struct Buffers {
    slab: Buffer,
    page_tokens: u32,
    slots: u32,
    page_bytes: u64,
    plane_at: Vec<u64>,
}

impl Buffers {
    /// The bytes `reserve` takes for `layout` at `paging`: the load's memory
    /// check counts them before anything is allocated.
    #[must_use]
    pub fn demand(layout: &Layout, paging: Paging) -> u64 {
        Self::geometry(layout, paging).2.max(256)
    }

    /// Each plane's offset in a slot, one slot's bytes, and the slab's.
    fn geometry(layout: &Layout, paging: Paging) -> (Vec<u64>, u64, u64) {
        let page_tokens = paging.page_size.max(1);
        let mut plane_at = Vec::with_capacity(layout.planes.len());
        let mut at = 0u64;
        for plane in &layout.planes {
            plane_at.push(at);
            at += u64::from(page_tokens) * plane.row_bytes;
        }
        let page_bytes = at.next_multiple_of(256);
        let bytes = page_bytes.saturating_mul(u64::from(paging.slots.max(1)));
        (plane_at, page_bytes, bytes)
    }

    pub fn reserve(device: &Context, layout: &Layout, paging: Paging) -> Result<Buffers> {
        let page_tokens = paging.page_size.max(1);
        let (plane_at, page_bytes, bytes) = Self::geometry(layout, paging);
        let slots = paging.slots.max(1);
        Ok(Buffers {
            slab: Buffer::zeroed(device, bytes.max(256))?,
            page_tokens,
            slots,
            page_bytes,
            plane_at,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slab.bytes()
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    pub fn pages(&mut self) -> (&mut Buffer, u64) {
        (&mut self.slab, self.page_bytes)
    }

    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    #[must_use]
    pub fn page_count(&self) -> u32 {
        self.slots
    }

    fn handle(&self, handles: &Handles) -> Result<u32> {
        handles.bind(&self.slab, 0, self.slab.bytes())
    }
}

/// The ints of one lane's row in the page table a verify arm reads its
/// buffer through: the in-page token of the lane's first extended row, or
/// -1 for a lane that buffers nothing, then the slab slots of the pages its
/// extended rows touch.
pub const PAGE_INTS: u32 = kernels_metal::ssm::RS_PAGE_INTS;

/// The page table of `lanes`, one [`PAGE_INTS`] row a lane, where every
/// lane's replayed and own rows fit the row's pages; else `None`, and the
/// fire's recurrent ops gather and scatter by copies.
#[must_use]
pub fn page_table(lanes: &[LanePlan], page_tokens: u32, pool_slots: u32) -> Option<Vec<i32>> {
    let page_tokens = page_tokens.max(1);
    let mut table = Vec::with_capacity(lanes.len() * PAGE_INTS as usize);
    for plan in lanes {
        let mut row = [0i32; PAGE_INTS as usize];
        row[0] = -1;
        let (first, pages) = match (&plan.gather, &plan.scatter) {
            (Some(gather), Some(scatter)) => {
                if scatter.from != gather.from + gather.count || scatter.pages != gather.pages {
                    return None;
                }
                (gather.from, &gather.pages)
            }
            (Some(run), None) | (None, Some(run)) => (run.from, &run.pages),
            (None, None) => {
                table.extend(row);
                continue;
            }
        };
        let span = plan.gather.as_ref().map_or(0, |run| run.count)
            + plan.scatter.as_ref().map_or(0, |run| run.count);
        let page0 = (first / page_tokens) as usize;
        let in_page = first % page_tokens;
        let touched = (in_page + span).div_ceil(page_tokens) as usize;
        if touched >= PAGE_INTS as usize {
            return None;
        }
        row[0] = i32::try_from(in_page).ok()?;
        for (at, &slot) in pages.get(page0..page0 + touched)?.iter().enumerate() {
            if slot >= pool_slots {
                return None;
            }
            row[1 + at] = i32::try_from(slot).ok()?;
        }
        table.extend(row);
    }
    Some(table)
}

#[derive(Debug)]
pub struct Seat {
    pub lanes: Vec<LanePlan>,
    pub replay: Tensor,
    pub commit: Tensor,
    pub slots: Tensor,
    pub pool: u32,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    pub page_bytes: u64,
    pub pool_slots: u32,
    pub ext_in: Vec<Tensor>,
    pub ext_out: Vec<Tensor>,
    pub work: Tensor,
    pub rows_ext: u32,
    /// Each lane's [`page_table`] row, when every lane's rows fit one: a
    /// verify arm then reads the replayed rows from the buffer and lands the
    /// window's own there itself.
    pub pages: Option<Tensor>,
    pub ext: RefCell<HashMap<u32, Tensor>>,
    /// The gate preps whose gates the verify scan derives as it stages its
    /// rows, by the gates value: the `[b | a]` projection, `dt_bias` and
    /// `a_log`.
    pub prepped: RefCell<HashMap<u32, (ValueId, Tensor, Tensor)>>,
}

/// A buffered window's conv, as its op states it.
#[derive(Clone, Copy, Debug)]
pub struct Convolved {
    pub x: ValueId,
    pub weight: Tensor,
    pub state: ValueId,
    pub conv_width: u32,
    pub dilation: u32,
}

impl Seat {
    #[must_use]
    pub fn scratch_bytes(layout: &Layout, rows_ext: u32, lanes: u32) -> u64 {
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        for plane in &layout.planes {
            at += (rows * plane.row_bytes).max(4).next_multiple_of(256);
        }
        for region in &layout.regions {
            at += (rows * region.row_bytes).max(4).next_multiple_of(256);
        }
        at += (u64::from(lanes.max(1)) * layout.work_elems * 4)
            .max(4)
            .next_multiple_of(256);
        at.max(256)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mint(
        handles: &Handles,
        layout: &Arc<Layout>,
        buffers: &Buffers,
        scratch: &Buffer,
        lanes: Vec<LanePlan>,
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
        pages: Option<Tensor>,
    ) -> Result<Seat> {
        let lane_count = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
        let need = Seat::scratch_bytes(layout, rows_ext, lane_count);
        if scratch.bytes() < need {
            return Err(Fault::Ceiling {
                what: "recurrent buffer scratch bytes",
                need,
                have: scratch.bytes(),
            });
        }
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        let mut take = |row_bytes: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            let bytes = rows * row_bytes;
            let here = at;
            at += bytes.next_multiple_of(256);
            Ok(Tensor::new(
                handles.bind(scratch, here, bytes.max(4))?,
                rows_ext.max(1),
                width,
                dtype,
            ))
        };
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for plane in &layout.planes {
            ext_in.push(take(plane.row_bytes, plane.width, plane.dtype)?);
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for region in &layout.regions {
            ext_out.push(take(region.row_bytes, region.width, region.dtype)?);
        }
        let work_bytes = u64::from(lane_count.max(1)) * layout.work_elems * 4;
        let work = Tensor::new(
            handles.bind(scratch, at, work_bytes.max(4))?,
            lane_count.max(1),
            u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
            Dtype::F32,
        );
        Ok(Seat {
            lanes,
            replay,
            commit,
            slots,
            pool: buffers.handle(handles)?,
            layout: Arc::clone(layout),
            plane_at: buffers.plane_at.clone(),
            page_tokens: buffers.page_tokens,
            page_bytes: buffers.page_bytes,
            pool_slots: buffers.slots,
            ext_in,
            ext_out,
            work,
            rows_ext,
            pages,
            ext: RefCell::new(HashMap::new()),
            prepped: RefCell::new(HashMap::new()),
        })
    }

    /// The buffer of the plane `value` is read into, as a verify arm reads
    /// it through the page table; `None` without a page table or a plane.
    pub fn paged(&self, handles: &Handles, value: ValueId) -> Option<kernels_metal::ssm::Paged> {
        let pages = self.pages?;
        let plane = *self.layout.in_of.get(&value.0)?;
        let at = self.plane_at[plane];
        let slab = self.page_bytes.checked_mul(u64::from(self.pool_slots))?;
        let view = handles.cut(self.pool, at, slab.checked_sub(at)?).ok()?;
        Some(kernels_metal::ssm::Paged {
            slab: Tensor::new(view, 1, 1, Dtype::U8),
            pages,
            page_tokens: self.page_tokens,
            page_bytes: u32::try_from(self.page_bytes).ok()?,
            row_bytes: u32::try_from(self.layout.planes[plane].row_bytes).ok()?,
        })
    }

    pub fn locate(&self, run: &Run, plane: usize, token: u32) -> Result<(u64, u32)> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "recurrent buffer pages",
            need: u64::from(page) + 1,
            have: run.pages.len() as u64,
        })?;
        if slot >= self.pool_slots {
            return Err(Fault::Ceiling {
                what: "recurrent buffer page slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.pool_slots),
            });
        }
        let row_bytes = self.layout.planes[plane].row_bytes;
        Ok((
            u64::from(slot) * self.page_bytes
                + self.plane_at[plane]
                + u64::from(in_page) * row_bytes,
            self.page_tokens - in_page,
        ))
    }
}
