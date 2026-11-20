use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::rs::{LanePlan, Run};
use kernels_slang::{Segments, Tensor};
use poem_ir::{Def, Dim, Dtype, Gather, Operation, Ssm, Trace, Ty, ValueId};

use crate::backend::{Backend, Buffer as _};
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

    /// The heads of the scan whose work is widest: a committed dispatch's
    /// workgroups are its lanes times these.
    pub work_heads: u32,

    /// Each plane's life over the trace's nodes, first and last: a plane is
    /// read only by the ops that extend its value.
    pub plane_life: Vec<(usize, usize)>,
    /// Each region's life: from the op that lands it extended to the last
    /// op that reads it so (`Run::rs_ext_of`).
    pub region_life: Vec<(usize, usize)>,
}

/// A load's recurrent planes and work plane: per-fire temporaries, laid
/// once at load in reservations of their own, not in the activation span.
/// The span is one binding's range, which a phone holds to as little as
/// 1 GiB; the planes (for the top rung's tokens of own rows and as many
/// carried, `engine::fire::ActivationRows`) and the work plane (a state a
/// lane, for the top rung's lanes) beside its carve would push the top rung
/// past it. Each is a binding of its own; one past a binding is refused by
/// name at load.
pub struct Reservation<B: Backend> {
    planes: B::Buffer,
    rows: u32,
    work: B::Buffer,
    lanes: u32,
}

impl<B: Backend> Reservation<B> {
    /// The planes for `rows` extended rows, packed by their lives, and the
    /// work plane for `lanes` lanes, each within `binding` bytes.
    pub fn lay(
        device: &B::Context,
        layout: &Layout,
        rows: u32,
        lanes: u32,
        binding: u64,
    ) -> Result<Reservation<B>> {
        let (_, _, planes) = layout.pack(rows);
        let work = (u64::from(lanes.max(1)) * layout.work_elems * 4)
            .max(4)
            .next_multiple_of(256);
        for (what, need) in [
            (
                "bytes of the recurrent planes for the top rung's rows",
                planes,
            ),
            (
                "bytes of the recurrent work plane for the top rung's lanes",
                work,
            ),
        ] {
            if need > binding {
                return Err(Fault::Ceiling {
                    what,
                    need,
                    have: binding,
                });
            }
        }
        tracing::info!(
            planes_bytes = planes,
            rows,
            work_bytes = work,
            work_lanes = lanes.max(1),
            "the recurrent planes and work plane, each a reservation of its own"
        );
        Ok(Reservation {
            planes: B::Buffer::zeroed(device, planes.max(256), crate::books::owners::RS_WORK)?,
            rows,
            work: B::Buffer::zeroed(device, work, crate::books::owners::RS_WORK)?,
            lanes: lanes.max(1),
        })
    }
}

/// `life` widened to hold `node`.
fn live(life: &mut (usize, usize), node: usize) {
    life.0 = life.0.min(node);
    life.1 = life.1.max(node);
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
                         node: usize|
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
                    layout.plane_life.push((node, node));
                    plane_keys.insert((cache, role), layout.planes.len() - 1);
                    layout.planes.len() - 1
                }
            };
            live(&mut layout.plane_life[at], node);
            layout.in_of.insert(value.0, at);
            Ok(())
        };
        let mut region = |layout: &mut Layout,
                          cache: u32,
                          what: &'static str,
                          value: ValueId,
                          node: usize|
         -> Result<()> {
            let (width, dtype, row_bytes) = row_of(trace, value)?;
            let at = match region_keys.get(&(cache, what)) {
                Some(&at) => at,
                None => {
                    layout.regions.push(Region {
                        width,
                        dtype,
                        row_bytes,
                    });
                    layout.region_life.push((node, node));
                    region_keys.insert((cache, what), layout.regions.len() - 1);
                    layout.regions.len() - 1
                }
            };
            live(&mut layout.region_life[at], node);
            layout.out_of.insert(value.0, at);
            Ok(())
        };

        // The convs the rs buffer covers: the chunk plan's rule
        // (`poem_compiler::rs::buffered_convs`), the one the recipe's rs rows
        // read, not a walk of this shell's own. A conv feeding a recurrence
        // and a conv no recurrence reads (Qwen4's PLE conv, `Feeds::Alone`)
        // alike buffer their input and replay it at a fold. This shell
        // replays no Mamba-2 layer, so a Mamba-2 conv commits with it and
        // keeps no plane.
        for conv in poem_compiler::rs::buffered_convs(trace, false) {
            let cache = cache_of(trace, conv.state)?;
            plane(&mut layout, cache, Role::ConvX, conv.x, conv.node)?;
            region(&mut layout, cache, "conv", conv.y, conv.node)?;
        }
        for (at, node) in trace.nodes.iter().enumerate() {
            match &node.op {
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
                    region(&mut layout, cache, "scan", *y, at)?;
                    let elems = u64::from(*v_heads) * u64::from(*v_dim) * u64::from(*k_dim);
                    if elems > layout.work_elems {
                        layout.work_elems = elems;
                        layout.work_heads = *v_heads;
                    }
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
                    plane(&mut layout, cache, Role::KdaF, *f, at)?;
                    plane(&mut layout, cache, Role::KdaB, *b, at)?;
                    region(&mut layout, cache, "scan", *y, at)?;
                    let elems = u64::from(*heads) * u64::from(*head_dim) * u64::from(*head_dim);
                    if elems > layout.work_elems {
                        layout.work_elems = elems;
                        layout.work_heads = *heads;
                    }
                }
                Operation::Gather(
                    Gather::NgramIds { ids, state, .. } | Gather::NgramIdsChunked { ids, state, .. },
                ) => {
                    // Its own rows' table ids land where the fire's land
                    // (`gather.ngram_ids_committed`): no extended region.
                    plane(&mut layout, cache_of(trace, *state)?, Role::Ids, *ids, at)?;
                }
                _ => {}
            }
        }
        for (at, node) in trace.nodes.iter().enumerate() {
            let Operation::Ssm(Ssm::GdnGates { ba, gates, .. }) = &node.op else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `ssm.gdn_gates` lands, which no delta scan reads",
                    gates.0
                ),
            })?;
            plane(&mut layout, cache, Role::Ba, *ba, at)?;
            region(&mut layout, cache, "gates", *gates, at)?;
        }
        // A region lives on to the last op that reads its value extended.
        for (at, node) in trace.nodes.iter().enumerate() {
            let read: &[ValueId] = match &node.op {
                Operation::Ssm(
                    Ssm::GatedDelta { qkv, gates, .. } | Ssm::GatedDeltaChunked { qkv, gates, .. },
                ) => &[*qkv, *gates],
                Operation::Ssm(Ssm::Kda { mixed, .. } | Ssm::KdaChunked { mixed, .. }) => &[*mixed],
                _ => &[],
            };
            for value in read {
                if let Some(&region) = layout.out_of.get(&value.0) {
                    live(&mut layout.region_life[region], at);
                }
            }
        }
        if layout.planes.is_empty() {
            return Ok(None);
        }
        layout.token_bytes = layout.planes.iter().map(|plane| plane.row_bytes).sum();
        Ok(Some(layout))
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

impl Layout {
    /// Where each plane and region lies for `rows` extended rows, and the
    /// bytes they take: first fit by their lives, so two whose lives do not
    /// overlap (one layer's and the next's) share bytes. Every plane and
    /// region starts at a multiple of 256.
    #[must_use]
    pub fn pack(&self, rows: u32) -> (Vec<u64>, Vec<u64>, u64) {
        let rows = u64::from(rows.max(1));
        let items: Vec<(u64, (usize, usize))> = self
            .planes
            .iter()
            .map(|plane| plane.row_bytes)
            .zip(self.plane_life.iter().copied())
            .chain(
                self.regions
                    .iter()
                    .map(|region| region.row_bytes)
                    .zip(self.region_life.iter().copied()),
            )
            .map(|(row_bytes, life)| ((rows * row_bytes).max(4).next_multiple_of(256), life))
            .collect();
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&at| (items[at].1.0, std::cmp::Reverse(items[at].0)));
        let mut placed: Vec<(u64, u64, (usize, usize))> = Vec::with_capacity(items.len());
        let mut offsets = vec![0u64; items.len()];
        let mut total = 0u64;
        for at in order {
            let (bytes, life) = items[at];
            let mut busy: Vec<(u64, u64)> = placed
                .iter()
                .filter(|(_, _, other)| other.0 <= life.1 && life.0 <= other.1)
                .map(|&(offset, size, _)| (offset, offset + size))
                .collect();
            busy.sort_unstable();
            let mut offset = 0u64;
            for (start, end) in busy {
                if offset + bytes <= start {
                    break;
                }
                offset = offset.max(end);
            }
            offsets[at] = offset;
            placed.push((offset, bytes, life));
            total = total.max(offset + bytes);
        }
        let regions = offsets.split_off(self.planes.len());
        (offsets, regions, total)
    }
}

impl Seat {
    #[allow(clippy::too_many_arguments)]
    pub fn mint<B: Backend>(
        handles: &Handles<B>,
        layout: &Arc<Layout>,
        buffers: &Buffers,
        pool: Segments,
        held: &Reservation<B>,
        lanes: Vec<LanePlan>,
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
    ) -> Result<Seat> {
        // The planes lie where the load's reservation lays them for the top
        // rung's rows, bound at their room: one view whatever the fire.
        let (buffer, room_rows) = (&held.planes, held.rows);
        if rows_ext > room_rows {
            return Err(Fault::Ceiling {
                what: "extended rows of the recurrent planes the load lays",
                need: u64::from(rows_ext),
                have: u64::from(room_rows),
            });
        }
        // Each plane and region where the room packs them by their lives.
        let rows = u64::from(room_rows.max(1));
        let (plane_at, region_at, _) = layout.pack(room_rows);
        let take = |offset: u64, row_bytes: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            Ok(Tensor::new(
                handles.bind(buffer, offset, (rows * row_bytes).max(4))?,
                rows_ext.max(1),
                width,
                dtype,
            ))
        };
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for (plane, &offset) in layout.planes.iter().zip(&plane_at) {
            ext_in.push(take(offset, plane.row_bytes, plane.width, plane.dtype)?);
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for (region, &offset) in layout.regions.iter().zip(&region_at) {
            ext_out.push(take(offset, region.row_bytes, region.width, region.dtype)?);
        }
        // The work plane holds a committed dispatch's lanes: a window of more
        // takes as many dispatches (`dispatch::ssm`).
        let work = Tensor::new(
            handles.bind(&held.work, 0, held.work.bytes())?,
            held.lanes,
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
mod pack_tests {
    use super::*;

    fn layout(lives: &[(u64, (usize, usize))]) -> Layout {
        let mut layout = Layout::default();
        for &(row_bytes, life) in lives {
            layout.planes.push(Plane {
                cache: 0,
                role: Role::ConvX,
                width: 1,
                dtype: Dtype::Bf16,
                row_bytes,
            });
            layout.plane_life.push(life);
        }
        layout
    }

    /// Planes whose lives overlap never share a byte; planes whose lives
    /// do not (one layer's and the next's) share; every plane starts at a
    /// multiple of 256, whatever the rows.
    #[test]
    fn planes_share_bytes_only_where_their_lives_do_not_overlap() {
        // Two layers of two planes each, and a plane living across both.
        let lives = [
            (1000, (0, 2)),
            (600, (1, 2)),
            (1000, (3, 5)),
            (600, (4, 5)),
            (100, (0, 5)),
        ];
        let layout = layout(&lives);
        for rows in [1u32, 3, 7, 64] {
            let (at, _, total) = layout.pack(rows);
            let size = |i: usize| (u64::from(rows) * lives[i].0).max(4).next_multiple_of(256);
            for i in 0..lives.len() {
                assert_eq!(at[i] % 256, 0, "plane {i} at {} for {rows} rows", at[i]);
                for j in i + 1..lives.len() {
                    let (a, b) = (lives[i].1, lives[j].1);
                    let overlap_life = a.0 <= b.1 && b.0 <= a.1;
                    let overlap_bytes = at[i] < at[j] + size(j) && at[j] < at[i] + size(i);
                    assert!(
                        !(overlap_life && overlap_bytes),
                        "planes {i} and {j} live together and share bytes for {rows} rows"
                    );
                }
            }
            let sum: u64 = (0..lives.len()).map(size).sum();
            assert!(
                total < sum,
                "{rows} rows: {total} bytes, no sharing of {sum}"
            );
        }
    }
}
