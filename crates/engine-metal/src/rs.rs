use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::rs::{LanePlan, Run};
use kernels_metal::Tensor;
use poem_ir::{Def, Dim, Dtype, Gather, Operation, Ssm, Trace, Ty, ValueId};

use crate::device::{Buffer, Handles};
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
    /// What the region holds (`conv`, `scan`, `gates`, `ngram`): regions
    /// of one kind may share a scratch slot.
    pub kind: &'static str,
    pub width: u32,
    pub dtype: Dtype,
    pub row_bytes: u64,
}

#[derive(Debug, Default)]
pub struct Layout {
    pub planes: Vec<Plane>,
    pub regions: Vec<Region>,
    /// The scratch slot each plane's extended rows lie in, and each
    /// region's ([`Layout::share`]): a plane's or region's own place in a
    /// fire's rs scratch, which planes of one role (regions of one kind)
    /// whose ops never overlap share, a recurrent layer after another.
    pub plane_slot: Vec<usize>,
    pub region_slot: Vec<usize>,
    /// Each scratch slot's bytes a row: the widest that lies in it.
    pub slots: Vec<u64>,
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
                            kind: what,
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

        // The convs the rs buffer covers, by their output: the chunk plan's
        // rule (`poem_compiler::rs::buffered_convs`), not a mirror of it. This
        // engine replays no Mamba-2 layer, so a Mamba-2 conv commits with it.
        let deferred: std::collections::HashSet<u32> =
            poem_compiler::rs::buffered_convs(trace, false)
                .map(|conv| conv.y.0)
                .collect();
        for node in &trace.nodes {
            match &node.op {
                Operation::Ssm(
                    Ssm::CausalConv1d { x, state, y, .. }
                    | Ssm::CausalConv1dChunked { x, state, y, .. },
                ) if deferred.contains(&y.0) => {
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
        layout.share(trace);
        Ok(Some(layout))
    }

    /// Lays the planes and regions in scratch slots: one of a role (a
    /// kind) whose live span (from the first node naming one of its values
    /// to the last that reads one) ends before another's starts shares its
    /// slot, as one recurrent layer's extended rows are done with before the
    /// next layer's are made. A fire's recurrent ops run in the forward's
    /// serial pass, a dispatch after another (`Shell::walk`; the only
    /// concurrent passes are a guest program's grouped steps, which run no
    /// recurrent op), so a slot is read and rewritten in node order. The
    /// scratch then holds the widest layer's planes, not every layer's.
    fn share(&mut self, trace: &Trace) {
        use poem_ir::Operands;
        let mut spans: Vec<Option<(usize, usize)>> =
            vec![None; self.planes.len() + self.regions.len()];
        let mut widen = |at: usize, node: usize| {
            let span = spans[at].get_or_insert((node, node));
            span.0 = span.0.min(node);
            span.1 = span.1.max(node);
        };
        let mut values = Vec::new();
        for (node, n) in trace.nodes.iter().enumerate() {
            values.clear();
            n.op.inputs(&mut values);
            n.op.outputs(&mut values);
            for value in &values {
                if let Some(&plane) = self.in_of.get(&value.0) {
                    widen(plane, node);
                }
                if let Some(&region) = self.out_of.get(&value.0) {
                    widen(self.planes.len() + region, node);
                }
            }
        }
        let members: Vec<(String, u64, Option<(usize, usize)>)> = self
            .planes
            .iter()
            .map(|plane| (format!("plane {:?}", plane.role), plane.row_bytes))
            .chain(
                self.regions
                    .iter()
                    .map(|region| (format!("region {}", region.kind), region.row_bytes)),
            )
            .zip(spans)
            .map(|((key, row_bytes), span)| (key, row_bytes, span))
            .collect();
        let (slot_of, slots) = share_slots(&members);
        self.plane_slot = slot_of[..self.planes.len()].to_vec();
        self.region_slot = slot_of[self.planes.len()..].to_vec();
        self.slots = slots;
    }
}

/// The scratch slot of each member (a key, its bytes a row, its live span
/// over the plan's nodes; none where no node names it), and each slot's
/// bytes a row: members of one key whose spans don't overlap share a slot,
/// first-fit in the order their spans start, each slot as wide as its
/// widest member. A member with no span takes a slot of its own.
fn share_slots(members: &[(String, u64, Option<(usize, usize)>)]) -> (Vec<usize>, Vec<u64>) {
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_by_key(|&at| members[at].2.map_or(0, |span| span.0));
    // Each slot: its key, its bytes a row, the last node of its latest member.
    let mut slots: Vec<(&str, u64, usize)> = Vec::new();
    let mut slot_of = vec![0; members.len()];
    for at in order {
        let (key, row_bytes, span) = (&members[at].0, members[at].1, members[at].2);
        let free = span.and_then(|(start, _)| {
            slots
                .iter()
                .position(|(k, _, last)| *k == key.as_str() && *last < start)
        });
        slot_of[at] = match free {
            Some(slot) => {
                slots[slot].1 = slots[slot].1.max(row_bytes);
                slots[slot].2 = span.map_or(usize::MAX, |span| span.1);
                slot
            }
            None => {
                slots.push((key, row_bytes, span.map_or(usize::MAX, |span| span.1)));
                slots.len() - 1
            }
        };
    }
    (slot_of, slots.into_iter().map(|(_, row_bytes, _)| row_bytes).collect())
}

#[cfg(test)]
mod page_tests {
    use super::page_table;
    use engine::rs::{LanePlan, Run};

    fn run(from: u32, count: u32) -> Run {
        Run {
            pages: vec![7, 8, 9],
            from,
            count,
        }
    }

    /// A lane replaying one run before its own rows reads them in place,
    /// through its pages; one replaying its ancestry among siblings' slots
    /// (two runs) reads no one stretch, so the fire gathers by copies,
    /// which take every run (`rs_extend`). Read as one stretch from its
    /// first run, a beam on a shared set replayed its siblings' rows.
    #[test]
    fn a_lane_of_two_runs_takes_no_page_row() {
        let one = LanePlan {
            replay: 2,
            gather: Some(run(3, 2)),
            gathers: vec![run(3, 2)],
            scatter: Some(run(5, 1)),
            ..LanePlan::default()
        };
        assert!(page_table(&[one], 16).is_some(), "one run reads in place");
        let two = LanePlan {
            replay: 2,
            gather: Some(run(3, 1)),
            gathers: vec![run(3, 1), run(5, 1)],
            scatter: Some(run(6, 1)),
            ..LanePlan::default()
        };
        assert!(page_table(&[two], 16).is_none(), "two runs gather by copies");
    }
}

#[cfg(test)]
mod share_tests {
    use super::share_slots;

    fn m(key: &str, row_bytes: u64, span: (usize, usize)) -> (String, u64, Option<(usize, usize)>) {
        (key.to_string(), row_bytes, Some(span))
    }

    /// Two members of one key whose spans overlap take two slots; one
    /// whose span starts after another's ends shares its slot, sized by the
    /// wider; one overlapping both takes its own; another key never shares.
    #[test]
    fn a_slot_is_shared_only_by_members_whose_spans_never_overlap() {
        let members = vec![
            m("plane ConvX", 100, (0, 5)),
            m("plane ConvX", 300, (3, 8)),
            m("plane ConvX", 200, (6, 9)),
            m("plane ConvX", 50, (2, 10)),
            m("region conv", 400, (6, 9)),
        ];
        let (slot_of, slots) = share_slots(&members);
        assert_ne!(slot_of[0], slot_of[1], "overlapping spans 0..5 and 3..8 share no slot");
        assert_eq!(slot_of[0], slot_of[2], "0..5 is done before 6..9 starts: one slot");
        assert_eq!(slots[slot_of[0]], 200, "the shared slot is as wide as its widest member");
        assert_ne!(slot_of[3], slot_of[0], "2..10 overlaps 0..5 and 6..9");
        assert_ne!(slot_of[3], slot_of[1], "and 3..8");
        assert_ne!(slot_of[4], slot_of[2], "a region shares no plane's slot, overlap or not");
        assert_eq!(slots.len(), 4);
    }
}

#[derive(Debug)]
pub struct Buffers {
    /// The pool's buffers, the first the one kernels bind (reaching the
    /// others from it, `kv_page.h`).
    parts: Vec<Buffer>,
    page_tokens: u32,
    /// An rs-list slot's bytes: what a page id steps by in a chunk.
    page_bytes: u64,
    /// The rs list's slots a chunk.
    per_chunk: u32,
    /// How a page id names its slot.
    ids: crate::store::pool::PageIds,
    plane_at: Vec<u64>,
}

impl Buffers {
    /// The rs buffer as the state pool's rs list over the pool's buffers
    /// `parts`: each plane's `page_tokens` rows at `plane_at` in an rs-list
    /// slot of `slots`, a lane's pages named as `ids` states.
    #[must_use]
    pub fn in_pool(
        plane_at: Vec<u64>,
        page_tokens: u32,
        parts: Vec<Buffer>,
        slots: crate::store::layout::Slots,
        ids: crate::store::pool::PageIds,
    ) -> Buffers {
        Buffers {
            parts,
            page_tokens: page_tokens.max(1),
            page_bytes: slots.slot_bytes,
            per_chunk: slots.per_chunk,
            ids,
            plane_at,
        }
    }

    /// The bytes per-page rs buffers would take for `layout` at `paging`:
    /// the load's estimate before any pool.
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

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    fn handles(&self, handles: &Handles) -> Result<Vec<u32>> {
        self.parts
            .iter()
            .map(|part| handles.bind(part, 0, part.bytes()))
            .collect()
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
pub fn page_table(lanes: &[LanePlan], page_tokens: u32) -> Option<Vec<i32>> {
    let page_tokens = page_tokens.max(1);
    let mut table = Vec::with_capacity(lanes.len() * PAGE_INTS as usize);
    for plan in lanes {
        // A lane replaying more than one run (its ancestry among siblings'
        // slots) reads no one stretch of pages: the fire gathers by copies.
        if plan.gathers.len() > 1 {
            return None;
        }
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
    /// Each of the pool's buffers, bound: the first the one kernels read
    /// every page through.
    pub pool: Vec<u32>,
    /// The first buffer's bytes.
    pub first_bytes: u64,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    pub page_bytes: u64,
    pub per_chunk: u32,
    pub ids: crate::store::pool::PageIds,
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
    /// A decode fire's step (one row a lane): its lanes' carried rows walk
    /// as the step walks its row.
    pub step: bool,
}

impl Seat {
    #[must_use]
    pub fn scratch_bytes(layout: &Layout, rows_ext: u32, lanes: u32) -> u64 {
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        for &slot in &layout.slots {
            at += (rows * slot).max(4).next_multiple_of(256);
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
        (scratch, (base, have)): (&Buffer, (u64, u64)),
        lanes: Vec<LanePlan>,
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
        pages: Option<Tensor>,
    ) -> Result<Seat> {
        let lane_count = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
        let need = Seat::scratch_bytes(layout, rows_ext, lane_count);
        if have < need {
            return Err(Fault::Ceiling {
                what: "recurrent buffer scratch bytes",
                need,
                have,
            });
        }
        let rows = u64::from(rows_ext.max(1));
        // Each scratch slot's place; a plane or region binds its own rows
        // at its slot's.
        let mut at = 0u64;
        let slot_at: Vec<u64> = layout
            .slots
            .iter()
            .map(|&slot| {
                let here = at;
                at += (rows * slot).max(4).next_multiple_of(256);
                here
            })
            .collect();
        let take = |slot: usize, row_bytes: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            let bytes = rows * row_bytes;
            Ok(Tensor::new(
                handles.bind(scratch, base + slot_at[slot], bytes.max(4))?,
                rows_ext.max(1),
                width,
                dtype,
            ))
        };
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for (plane, &slot) in layout.planes.iter().zip(&layout.plane_slot) {
            ext_in.push(take(slot, plane.row_bytes, plane.width, plane.dtype)?);
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for (region, &slot) in layout.regions.iter().zip(&layout.region_slot) {
            ext_out.push(take(slot, region.row_bytes, region.width, region.dtype)?);
        }
        let work_bytes = u64::from(lane_count.max(1)) * layout.work_elems * 4;
        let work = Tensor::new(
            handles.bind(scratch, base + at, work_bytes.max(4))?,
            lane_count.max(1),
            u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
            Dtype::F32,
        );
        Ok(Seat {
            lanes,
            replay,
            commit,
            slots,
            pool: buffers.handles(handles)?,
            first_bytes: buffers.parts.first().map_or(0, Buffer::bytes),
            layout: Arc::clone(layout),
            plane_at: buffers.plane_at.clone(),
            page_tokens: buffers.page_tokens,
            page_bytes: buffers.page_bytes,
            per_chunk: buffers.per_chunk,
            ids: buffers.ids,
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
        let view = handles
            .cut(*self.pool.first()?, at, self.first_bytes.checked_sub(at)?)
            .ok()?;
        Some(kernels_metal::ssm::Paged {
            slab: Tensor::new(view, 1, 1, Dtype::U8),
            pages,
            page_tokens: self.page_tokens,
            page_bytes: u32::try_from(self.page_bytes).ok()?,
            row_bytes: u32::try_from(self.layout.planes[plane].row_bytes).ok()?,
        })
    }

    /// Where `token` of `run` lies in plane `plane`: the bound buffer it
    /// lies in, the byte there, and the tokens left in its page.
    pub fn locate(&self, run: &Run, plane: usize, token: u32) -> Result<(u32, u64, u32)> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "recurrent buffer pages",
            need: u64::from(page) + 1,
            have: run.pages.len() as u64,
        })?;
        let (part, at) = self.ids.place(slot, self.per_chunk, self.page_bytes);
        let buffer = *self.pool.get(part).ok_or(Fault::Ceiling {
            what: "the pool's buffers a recurrent buffer page names",
            need: part as u64 + 1,
            have: self.pool.len() as u64,
        })?;
        let row_bytes = self.layout.planes[plane].row_bytes;
        Ok((
            buffer,
            at + self.plane_at[plane] + u64::from(in_page) * row_bytes,
            self.page_tokens - in_page,
        ))
    }
}
