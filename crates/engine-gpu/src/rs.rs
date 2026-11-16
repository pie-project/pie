use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::rs::{LanePlan, Run};
use kernels_slang::{Segments, Tensor};
use poem_ir::{Def, Dim, Dtype, Gather, Operands as _, Operation, Ssm, Trace, Ty, ValueId};

use crate::backend::Backend;
use crate::error::{Fault, Result};
use crate::handles::Handles;

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

    /// Each plane's group, then each region's ([`Layout::group`]): the
    /// recurrent ops a group's extended rows live through end before the
    /// next group's begin, so every group lays its rows over the same
    /// scratch.
    pub groups: Vec<usize>,
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
        layout.groups = layout.group(trace);
        Ok(Some(layout))
    }

    /// The group of each plane, then each region: the extended rows an op
    /// reads or writes live from the first recurrent op that names their
    /// value to the last, and rows whose spans overlap are one group. A
    /// layer's conv, gates and scan are one; the next layer's begins after.
    fn group(&self, trace: &Trace) -> Vec<usize> {
        let items = self.planes.len() + self.regions.len();
        let mut spans: Vec<Option<(usize, usize)>> = vec![None; items];
        for (at, node) in trace.nodes.iter().enumerate() {
            if !matches!(
                node.op,
                Operation::Ssm(_)
                    | Operation::Gather(Gather::NgramIds { .. } | Gather::NgramIdsChunked { .. })
            ) {
                continue;
            }
            let mut values = Vec::new();
            node.op.inputs(&mut values);
            node.op.outputs(&mut values);
            for value in values {
                let item = self.in_of.get(&value.0).copied().or_else(|| {
                    self.out_of
                        .get(&value.0)
                        .map(|&region| self.planes.len() + region)
                });
                if let Some(item) = item {
                    let span = spans[item].get_or_insert((at, at));
                    span.0 = span.0.min(at);
                    span.1 = span.1.max(at);
                }
            }
        }
        let mut order: Vec<usize> = (0..items).collect();
        order.sort_by_key(|&item| spans[item].map_or((0, 0), |span| span));
        let mut groups = vec![0; items];
        let mut group = 0;
        let mut end: Option<usize> = None;
        for item in order {
            let (from, to) = spans[item].unwrap_or((0, 0));
            if end.is_some_and(|end| from > end) {
                group += 1;
                end = None;
            }
            end = Some(end.map_or(to, |end| end.max(to)));
            groups[item] = group;
        }
        groups
    }
}

/// The rs list in the chunk pool (CONTRACT-BC 18, 20): an rs page is an
/// rs-list slot, `page_tokens` unfolded rows of every plane, plane `p` at
/// `plane_at[p]` in it; a page id counts `stride` bytes of the pool.
#[derive(Debug)]
pub struct Buffers {
    page_tokens: u32,
    stride: u64,
    plane_at: Vec<u64>,
}

impl Buffers {
    #[must_use]
    pub fn in_pool(plane_at: Vec<u64>, page_tokens: u32, stride: u64) -> Buffers {
        Buffers {
            page_tokens: page_tokens.max(1),
            stride,
            plane_at,
        }
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }
}

#[derive(Debug)]
pub struct Seat {
    pub lanes: Vec<LanePlan>,

    pub replay: Tensor,

    pub commit: Tensor,

    pub slots: Tensor,

    /// The chunk pool the rs pages lie in.
    pub pool: Segments,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    /// The bytes an rs page id counts.
    pub stride: u64,

    pub ext_in: Vec<Tensor>,

    pub ext_out: Vec<Tensor>,

    pub work: Tensor,
    pub rows_ext: u32,

    pub ext: RefCell<HashMap<u32, Tensor>>,
}

impl Seat {
    /// The extended rows a fire at a rung of `tokens` carries: its own rows
    /// and the buffered rows it replays ahead of them, that many of each at
    /// most (`engine::fire::rung_rows`: a fire's lanes carry no more than
    /// its rung's tokens). Each rung lays its recurrent scratch for this
    /// many ([`Seat::scratch_bytes`], `arena::Rung`).
    #[must_use]
    pub fn widest_rows(tokens: u32) -> u32 {
        tokens.saturating_mul(2)
    }

    /// A fire whose recurrent scratch of `need` bytes is past the `laid`
    /// bytes its rung laid is refused by name: nothing grows it.
    pub fn fits(need: u64, laid: u64) -> Result<()> {
        crate::error::within_laid("serve::rs", "recurrent scratch", need, laid)
    }

    /// Where each plane's extended rows lie, then each region's, and the
    /// work: a group's rows one after another from the scratch's start,
    /// every group over the same bytes ([`Layout::groups`]), the work past
    /// the widest group.
    fn placed(layout: &Layout, rows_ext: u32) -> (Vec<u64>, u64) {
        let rows = u64::from(rows_ext.max(1));
        let widths = layout
            .planes
            .iter()
            .map(|plane| plane.row_bytes)
            .chain(layout.regions.iter().map(|region| region.row_bytes));
        let mut cursor: Vec<u64> = Vec::new();
        let mut at = Vec::with_capacity(layout.groups.len());
        for (item, row_bytes) in widths.enumerate() {
            let group = layout.groups.get(item).copied().unwrap_or(0);
            if cursor.len() <= group {
                cursor.resize(group + 1, 0);
            }
            at.push(cursor[group]);
            cursor[group] += (rows * row_bytes).max(4).next_multiple_of(256);
        }
        (at, cursor.into_iter().max().unwrap_or(0))
    }

    /// The bytes an extended row takes in its widest group.
    #[must_use]
    pub fn row_bytes(layout: &Layout) -> u64 {
        let mut groups: Vec<u64> = Vec::new();
        let widths = layout
            .planes
            .iter()
            .map(|plane| plane.row_bytes)
            .chain(layout.regions.iter().map(|region| region.row_bytes));
        for (item, row_bytes) in widths.enumerate() {
            let group = layout.groups.get(item).copied().unwrap_or(0);
            if groups.len() <= group {
                groups.resize(group + 1, 0);
            }
            groups[group] += row_bytes;
        }
        groups.into_iter().max().unwrap_or(0)
    }

    /// The scratch a fire of `rows_ext` extended rows and `lanes` lanes
    /// takes: its widest group's rows and the lanes' work.
    #[must_use]
    pub fn scratch_bytes(layout: &Layout, rows_ext: u32, lanes: u32) -> u64 {
        let (_, work_at) = Seat::placed(layout, rows_ext);
        (work_at
            + (u64::from(lanes.max(1)) * layout.work_elems * 4)
                .max(4)
                .next_multiple_of(256))
        .max(256)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mint<B: Backend>(
        handles: &Handles<B>,
        layout: &Arc<Layout>,
        buffers: &Buffers,
        pool: Segments,
        scratch: (&B::Buffer, u64, u64),
        lanes: Vec<LanePlan>,
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
    ) -> Result<Seat> {
        let lane_count = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
        Seat::fits(Seat::scratch_bytes(layout, rows_ext, lane_count), scratch.2)?;
        let rows = u64::from(rows_ext.max(1));
        let (placed, work_at) = Seat::placed(layout, rows_ext);
        let (scratch, base, _) = scratch;
        let take = |item: usize, row_bytes: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            Ok(Tensor::new(
                handles.bind(scratch, base + placed[item], (rows * row_bytes).max(4))?,
                rows_ext.max(1),
                width,
                dtype,
            ))
        };
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for (at, plane) in layout.planes.iter().enumerate() {
            ext_in.push(take(at, plane.row_bytes, plane.width, plane.dtype)?);
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for (at, region) in layout.regions.iter().enumerate() {
            ext_out.push(take(
                layout.planes.len() + at,
                region.row_bytes,
                region.width,
                region.dtype,
            )?);
        }
        let work_bytes = u64::from(lane_count.max(1)) * layout.work_elems * 4;
        let work = Tensor::new(
            handles.bind(scratch, base + work_at, work_bytes.max(4))?,
            lane_count.max(1),
            u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
            Dtype::F32,
        );
        Ok(Seat {
            lanes,
            replay,
            commit,
            slots,
            pool,
            layout: Arc::clone(layout),
            plane_at: buffers.plane_at.clone(),
            page_tokens: buffers.page_tokens,
            stride: buffers.stride,
            ext_in,
            ext_out,
            work,
            rows_ext,
            ext: RefCell::new(HashMap::new()),
        })
    }

    /// Where token `token` of `run`'s plane `plane` lies: the segment's
    /// handle, the byte there, and the rows left in its page.
    pub fn locate(&self, run: &Run, plane: usize, token: u32) -> Result<(u32, u64, u32)> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let id = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "recurrent buffer pages",
            need: u64::from(page) + 1,
            have: run.pages.len() as u64,
        })?;
        let row_bytes = self.layout.planes[plane].row_bytes;
        let at =
            u64::from(id) * self.stride + self.plane_at[plane] + u64::from(in_page) * row_bytes;
        let span = u64::from(self.pool.words) * 4;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let handle = self
            .pool
            .bufs
            .get(segment)
            .copied()
            .filter(|&handle| handle != kernels_slang::ABSENT)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: 0,
            })?;
        Ok((handle, at % span, self.page_tokens - in_page))
    }
}

/// The snapshot rows one lane lands in a fire (`Lane::rs_snapshots`):
/// [`kernels_slang::ssm::RS_SNAPSHOTS`] where the trace holds state rows and
/// every op that reads one is a causal conv, a gated delta or a KDA scan,
/// whose chunked and committed arms land them; else none, and the runtime
/// cuts a pass at each.
#[must_use]
pub fn snapshots(trace: &poem_ir::Trace) -> u32 {
    use poem_ir::{Def, Operands, Operation, Ssm};
    let state = |id: &poem_ir::ValueId| {
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
                    | Ssm::GatedDeltaChunked { .. }
                    | Ssm::Kda { .. }
                    | Ssm::KdaChunked { .. },
            )
        )
    });
    if lands && any {
        kernels_slang::ssm::RS_SNAPSHOTS
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{Layout, Plane, Region, Role, Seat};
    use poem_ir::Dtype;

    /// Two planes and a region of rows, and a lane's work.
    fn layout() -> Layout {
        let plane = |cache, width| Plane {
            cache,
            role: Role::ConvX,
            width,
            dtype: Dtype::Bf16,
            row_bytes: u64::from(width) * 2,
        };
        Layout {
            planes: vec![plane(0, 384), plane(1, 384)],
            regions: vec![Region {
                width: 128,
                dtype: Dtype::F32,
                row_bytes: 512,
            }],
            work_elems: 1024,
            ..Layout::default()
        }
    }

    /// The scratch laid at the widest rung's extended rows for the most
    /// lanes holds every fire of those, and a fire past it is refused by
    /// name with the bytes it wants and the bytes laid.
    #[test]
    fn a_fire_past_the_recurrent_scratch_laid_is_refused_by_name() {
        let layout = layout();
        let (top, lanes) = (256, 4);
        let widest = Seat::widest_rows(top);
        assert_eq!(
            widest,
            2 * top,
            "a rung's tokens of own rows and as many replayed"
        );
        let laid = Seat::scratch_bytes(&layout, widest, lanes);
        for (rows, lanes) in [(1, 1), (top, 1), (widest, lanes), (top + 7, lanes)] {
            Seat::fits(Seat::scratch_bytes(&layout, rows, lanes), laid)
                .unwrap_or_else(|why| panic!("{rows} rows of {lanes} lanes: {why}"));
        }
        let need = Seat::scratch_bytes(&layout, widest + 1, lanes);
        assert!(need > laid);
        let why = Seat::fits(need, laid)
            .expect_err("a fire past the scratch laid is refused")
            .to_string();
        assert!(
            why.contains(&format!(
                "a fire's recurrent scratch of {need} bytes is past the {laid} laid at load"
            )),
            "{why}"
        );
    }

    /// Two layers whose extended rows live through ops that do not overlap
    /// are two groups over the same bytes: the scratch is the wider layer's,
    /// not the two together.
    #[test]
    fn layers_lay_their_extended_rows_over_the_same_scratch() {
        let mut layout = layout();
        let (rows, lanes) = (512, 4);
        let together = Seat::scratch_bytes(&layout, rows, lanes);
        // Planes of caches 0 and 1, the region with cache 0's.
        layout.groups = vec![0, 1, 0];
        let apart = Seat::scratch_bytes(&layout, rows, lanes);
        let plane = (u64::from(rows) * 768).next_multiple_of(256);
        assert_eq!(together - apart, plane, "one layer's plane less");
        let (placed, work_at) = Seat::placed(&layout, rows);
        assert_eq!(placed, vec![0, 0, plane], "each group from the scratch's start");
        assert_eq!(work_at, plane + (u64::from(rows) * 512).next_multiple_of(256));
    }
}
