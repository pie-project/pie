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
    /// A gated-delta row's keys (the conv output's key heads, bf16): what
    /// a decode step reads a buffered row's attention weight by.
    Keys,
    /// A gated-delta row's corrected value `u` (f32): the value its delta
    /// rule writes, which a decode step reads as a linear attention.
    U,
    /// A gated-delta row's per-head scalars (f32, `v_heads` log-gates then
    /// `v_heads` key-norm inverses, `poem_compiler::rs::gdn_scalars_bytes`):
    /// written by the step that lands the row, read by a later decode step
    /// in place of the row's `[b | a]` softplus and its key norm.
    Scalars,
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
    /// Whether every op that reads a state slot runs the paged forms (the
    /// verify conv and the gated-delta verify and carried scans): then a
    /// decode lane reads its carried rows from its buffer pages wherever it
    /// lies, and takes no extended rows ([`compact`], [`scratch_rows`]).
    pub compact: bool,
}

/// The extended rows a fire's rs scratch holds: its own rows and the rows
/// its lanes carry in, as each counts toward the rung
/// ([`engine::fire::lane_carried`] where the fire is compact: a decode
/// lane's none; else every carried row). What the load reserves for each
/// rung, at the rung's bound, and what each fire is checked against.
#[must_use]
pub fn scratch_rows(own: u32, carried: u32) -> u32 {
    own.saturating_add(carried)
}

/// Whether every op of `trace` that reads a state slot is a conv or a
/// gated-delta scan the paged forms run: a conv of a multiple of four
/// channels within the verify conv's taps and history, no bias; a scan of
/// the verify scan's heads.
#[must_use]
pub fn compact(trace: &Trace) -> bool {
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
    let paged = trace.nodes.iter().all(|node| {
        reads.clear();
        node.op.inputs(&mut reads);
        if !reads.iter().any(state) {
            return true;
        }
        any = true;
        match &node.op {
            Operation::Ssm(
                Ssm::CausalConv1d {
                    x,
                    bias: None,
                    conv_width,
                    dilation,
                    ..
                }
                | Ssm::CausalConv1dChunked {
                    x,
                    bias: None,
                    conv_width,
                    dilation,
                    ..
                },
            ) => row_of(trace, *x).is_ok_and(|(width, _, _)| {
                kernels_metal::ssm::causal_conv1d_verify_fits(width, *conv_width, *dilation)
            }),
            Operation::Ssm(
                Ssm::GatedDelta { k_dim, v_dim, .. } | Ssm::GatedDeltaChunked { k_dim, v_dim, .. },
            ) => kernels_metal::ssm::gated_delta_verify_fits(*k_dim, *v_dim),
            _ => false,
        }
    });
    paged && any
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
                         value: ValueId,
                         sized: Option<(u32, Dtype, u64)>|
         -> Result<()> {
            let (width, dtype, row_bytes) = match sized {
                Some(sized) => sized,
                None => row_of(trace, value)?,
            };
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
                    plane(&mut layout, cache, Role::ConvX, *x, None)?;
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
                    plane(&mut layout, cache, Role::KdaF, *f, None)?;
                    plane(&mut layout, cache, Role::KdaB, *b, None)?;
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
                    plane(&mut layout, cache, Role::Ids, *ids, None)?;
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
            plane(&mut layout, cache, Role::Ba, *ba, None)?;
            region(&mut layout, cache, "gates", *gates)?;
        }
        // The corrected row (`ChunkProfile::rs_corrected`, Metal's profile):
        // a buffered gated-delta row keeps its keys (the conv output's key
        // heads, bf16) and its corrected values `u` (f32), as the chunk plan
        // lays them (`poem_compiler::chunk::census::rs_rows`): a decode step
        // reads the buffered rows as a linear attention and re-steps none.
        if poem_compiler::chunk::ChunkProfile::ON_DEVICE_METAL.rs_corrected {
            let gated: std::collections::HashSet<u32> = trace
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    Operation::Ssm(Ssm::GdnGates { gates, .. }) => Some(gates.0),
                    _ => None,
                })
                .collect();
            for node in &trace.nodes {
                let Operation::Ssm(Ssm::GatedDeltaChunked {
                    qkv,
                    gates,
                    state,
                    y,
                    k_heads,
                    k_dim,
                    v_heads,
                    v_dim,
                    ..
                }) = &node.op
                else {
                    continue;
                };
                if !deferred.contains(&qkv.0) || !gated.contains(&gates.0) {
                    continue;
                }
                let cache = cache_of(trace, *state)?;
                let keys = k_heads * k_dim;
                let values = v_heads * v_dim;
                plane(
                    &mut layout,
                    cache,
                    Role::Keys,
                    *qkv,
                    Some((keys, Dtype::Bf16, u64::from(keys) * 2)),
                )?;
                plane(
                    &mut layout,
                    cache,
                    Role::U,
                    *y,
                    Some((values, Dtype::F32, u64::from(values) * 4)),
                )?;
                plane(
                    &mut layout,
                    cache,
                    Role::Scalars,
                    *gates,
                    Some((
                        2 * v_heads,
                        Dtype::F32,
                        poem_compiler::rs::gdn_scalars_bytes(*v_heads),
                    )),
                )?;
            }
            // The decode form (`GatedDelta`) of a layer reads the planes its
            // prompt form lays, by its own values: the same state's.
            let of = |layout: &Layout, cache: u32, role: Role| {
                layout
                    .planes
                    .iter()
                    .position(|plane| plane.cache == cache && plane.role == role)
            };
            for node in &trace.nodes {
                let Operation::Ssm(Ssm::GatedDelta {
                    qkv,
                    gates,
                    state,
                    y,
                    ..
                }) = &node.op
                else {
                    continue;
                };
                let cache = cache_of(trace, *state)?;
                if let (Some(keys), Some(u), Some(scalars)) = (
                    of(&layout, cache, Role::Keys),
                    of(&layout, cache, Role::U),
                    of(&layout, cache, Role::Scalars),
                ) {
                    layout.in_of.insert(qkv.0, keys);
                    layout.in_of.insert(y.0, u);
                    layout.in_of.insert(gates.0, scalars);
                }
            }
        }
        if layout.planes.is_empty() {
            return Ok(None);
        }
        layout.token_bytes = layout.planes.iter().map(|plane| plane.row_bytes).sum();
        layout.share(trace);
        layout.compact = compact(trace);
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

    /// The GPU address an rs page id's slot is reached from (`pie_page_at`
    /// over the pool's first buffer), as a beam's rs copies address it.
    #[must_use]
    pub fn slab_address(&self) -> Option<u64> {
        self.parts.first()?.address_at(0)
    }

    /// Each plane's rows' byte offset in an rs page.
    #[must_use]
    pub fn plane_offsets(&self) -> &[u64] {
        &self.plane_at
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
/// buffer through (the in-page token of the lane's first extended row, or
/// -1 for a lane that buffers nothing, then the slab slots of the pages its
/// extended rows touch) over an rs buffer of `page_tokens` rows a page, for
/// a load of `context` tokens: the pages of the most rows a lane touches in
/// one fire (its buffered rows and its own, all within its context: a
/// decode lane's carried rows count toward no rung, as
/// `engine::fire::lane_carried` states, and a beam in a parents pass
/// carries every row since its fold) from any in-page offset, and the
/// header. 0 where the load buffers no rs row.
#[must_use]
pub fn page_ints(context: u32, page_tokens: u32) -> u32 {
    if page_tokens == 0 {
        return 0;
    }
    1 + (page_tokens - 1).saturating_add(context).div_ceil(page_tokens)
}

/// The page table of `lanes`, one `ints` row a lane ([`page_ints`]); `None`
/// where a lane's rows lie in no one stretch of pages, and the fire's
/// recurrent ops gather and scatter by copies. A lane touching more pages
/// than the row holds is a count past the rung the width was derived from:
/// refused by name.
pub fn page_table(lanes: &[LanePlan], page_tokens: u32, ints: u32) -> Result<Option<Vec<i32>>> {
    let page_tokens = page_tokens.max(1);
    if ints < 2 {
        return Ok(None);
    }
    let mut table = Vec::with_capacity(lanes.len() * ints as usize);
    for plan in lanes {
        // A lane replaying more than one run (its ancestry among siblings'
        // slots) reads no one stretch of pages: the fire gathers by copies.
        if plan.gathers.len() > 1 {
            return Ok(None);
        }
        let mut row = vec![0i32; ints as usize];
        row[0] = -1;
        let (first, pages) = match (&plan.gather, &plan.scatter) {
            (Some(gather), Some(scatter)) => {
                if scatter.from != gather.from + gather.count || scatter.pages != gather.pages {
                    return Ok(None);
                }
                (gather.from, &gather.pages)
            }
            (Some(run), None) | (None, Some(run)) => (run.from, &run.pages),
            (None, None) => {
                table.extend(row.iter().copied());
                continue;
            }
        };
        let span = plan.gather.as_ref().map_or(0, |run| run.count)
            + plan.scatter.as_ref().map_or(0, |run| run.count);
        let page0 = (first / page_tokens) as usize;
        let in_page = first % page_tokens;
        let touched = (in_page + span).div_ceil(page_tokens) as usize;
        if touched >= ints as usize {
            return Err(Fault::program(
                "rs::page_table",
                format!(
                    "a lane's {span} recurrent rows from in-page row {in_page} touch {touched} \
                     pages of {page_tokens} rows, past the {} a page table row of this rung \
                     holds (rs::page_ints); the fire is refused and writes nothing",
                    ints - 1
                ),
            ));
        }
        let (Ok(at), Some(slots)) = (i32::try_from(in_page), pages.get(page0..page0 + touched))
        else {
            return Ok(None);
        };
        row[0] = at;
        for (at, &slot) in slots.iter().enumerate() {
            let Ok(slot) = i32::try_from(slot) else {
                return Ok(None);
            };
            row[1 + at] = slot;
        }
        table.extend(row.iter().copied());
    }
    Ok(Some(table))
}

#[derive(Debug)]
pub struct Seat {
    pub lanes: Vec<LanePlan>,
    /// Each lane's carried rows among the extended rows: its replay, or
    /// none for a decode lane of a compact fire ([`scratch_rows`]).
    pub ext_replay: Vec<u32>,
    pub ext_replay_table: Tensor,
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
    /// The ints of a lane's row of `pages` ([`page_ints`]).
    pub page_ints: u32,
    /// The fire's beam groups' folds before each recurrent layer's step,
    /// read through `pages`; `None` where no group
    /// folds this fire.
    pub folds: Option<kernels_metal::ssm::GroupFolds>,
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
        (ext_replay, ext_replay_table): (Vec<u32>, Tensor),
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
        pages: Option<Tensor>,
        page_ints: u32,
        folds: Option<(Tensor, u32)>,
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
            ext_replay,
            ext_replay_table,
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
            page_ints,
            folds: folds.map(|(table, count)| kernels_metal::ssm::GroupFolds { table, count }),
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
            page_ints: self.page_ints,
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
