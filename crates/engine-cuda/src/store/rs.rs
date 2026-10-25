use std::collections::HashMap;

use poem_ir::{Dim, Dtype, Operands, Operation, Ssm, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::device::elastic::{self, Arena};
use crate::error::{Fault, Result};
use crate::run::RsScratch;
use crate::store::Pools;
use crate::store::kv::Paging;

pub const PLANE_DTYPE: Dtype = Dtype::Bf16;

const ELEMENT: u64 = 2;

const SCRATCH: &str = "rs extended-run scratch";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pub layer: u32,
    pub at: u64,
    pub width: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Planes(HashMap<u32, Plane>);

impl Planes {
    #[must_use]
    pub fn of(&self, id: ValueId) -> Option<Plane> {
        self.0.get(&id.0).copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Debug)]
pub struct Buffers {
    arena: Arena,
    committed_pages: u32,
    /// The extended run's scratch: a fire's rows and the buffered slots its
    /// lanes replay ahead of them, a row each at the widest recurrence's
    /// width. Backed beside the pool as fires reach further, as a kernel's
    /// slab is, and mapped under the base the fires in flight hold, so a
    /// growth waits for none of them.
    scratch: Arena,
    planes: Planes,
    page_tokens: u32,
    slots: u32,
    per_page: u64,
    layers: u32,
    ext_row_bytes: u64,
}

impl Buffers {
    /// Reserves the page slots `paging` names, and the scratch's range for
    /// fires of up to `max_rows` rows beside every page slot replayed once.
    /// A flat arena backs its whole range on its first commit, so its
    /// scratch is reserved when a fire first asks, for what it asks.
    pub fn reserve(
        trace: &Trace,
        paging: Paging,
        max_rows: u32,
        pools: &Pools,
    ) -> Result<Option<Buffers>> {
        let Some((planes, per_token, layers)) = read(trace, paging.page_size)? else {
            return Ok(None);
        };
        let per_page = u64::from(paging.page_size) * per_token;
        let ext_row_bytes = ext_row_bytes(trace)?;
        let bytes = per_page
            .saturating_mul(u64::from(paging.slots))
            .saturating_mul(u64::from(layers))
            .saturating_mul(ELEMENT);
        let reach = if elastic::mapped_on_demand() {
            let rows = max_rows.saturating_add(paging.slots.saturating_mul(paging.page_size));
            RsScratch::need(rows, ext_row_bytes)
        } else {
            0
        };
        Ok(Some(Buffers {
            arena: pools.reserve_arena(bytes, "rs buffered activations")?,
            committed_pages: 0,
            scratch: pools.reserve_arena(reach, SCRATCH)?,
            planes,
            page_tokens: paging.page_size,
            slots: paging.slots,
            per_page,
            layers,
            ext_row_bytes,
        }))
    }
}

fn ext_row_bytes(trace: &Trace) -> Result<u64> {
    let mut defined: HashMap<u32, usize> = HashMap::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        for out in outs {
            defined.insert(out.0, at);
        }
    }
    let row_bytes = |id: ValueId| -> Result<u64> {
        let decl = trace.values.get(id.0 as usize);
        let Some(Ty::Tensor { shape, dtype }) = decl.map(|d| &d.ty) else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let Some(Dim::Const(width)) = shape.last() else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let elem = poem_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
            what: format!("value {} has a dtype with no element size", id.0),
        })?;
        Ok(*width * elem)
    };
    let mut widest = 0u64;
    for node in &trace.nodes {
        let total = match &node.op {
            Operation::Ssm(Ssm::GatedDeltaChunked {
                qkv, z, gates, y, ..
            }) => {
                let mut total =
                    row_bytes(*qkv)? + row_bytes(*z)? + row_bytes(*gates)? + row_bytes(*y)?;
                if let Some(Operation::Ssm(Ssm::CausalConv1dChunked { x, .. })) =
                    defined.get(&qkv.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                if let Some(Operation::Ssm(Ssm::GdnGates { ba, .. })) =
                    defined.get(&gates.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*ba)?;
                }
                total
            }
            Operation::Ssm(Ssm::KdaChunked { mixed, f, b, y, .. }) => {
                let mut total =
                    row_bytes(*mixed)? + row_bytes(*f)? + row_bytes(*b)? + row_bytes(*y)?;
                if let Some(Operation::Ssm(Ssm::CausalConv1dChunked { x, .. })) =
                    defined.get(&mixed.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                total
            }
            _ => continue,
        };
        widest = widest.max(total);
    }
    Ok(widest)
}

/// The planes a recurrence reads besides the conv's rows: the gated-delta
/// pair's `[b | a]` stands behind an `SsmGdnPrep`, the KDA pair's forget and
/// beta projections are read by the scan itself.
#[derive(Clone, Copy)]
enum Reads {
    GatedDelta {
        qkv: ValueId,
        gates: ValueId,
    },
    Kda {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
    },
}

pub fn read(trace: &Trace, page_tokens: u32) -> Result<Option<(Planes, u64, u32)>> {
    {
        let paging = page_tokens;
        let mut defined: HashMap<u32, usize> = HashMap::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            node.op.outputs(&mut outs);
            for out in outs {
                defined.insert(out.0, at);
            }
        }

        let mut planes: HashMap<u32, Plane> = HashMap::new();
        let mut per_token: Option<u64> = None;
        let mut layers = 0u32;
        let mut decode_layers = 0u32;
        for (at, node) in trace.nodes.iter().enumerate() {
            // The fused decode step reads the planes its primitives would:
            // the conv's rows (the split it lands) and the gate prep's
            // `[b | a]`.
            if let Operation::Fused(poem_ir::Fused::GdnStep { qkv, ba, .. }) = &node.op {
                let page = u64::from(paging);
                let mut here = 0u64;
                for (what, id) in [
                    ("the conv's rows", *qkv),
                    ("the gate prep's `[b | a]`", *ba),
                ] {
                    let width = width_of(trace, id).ok_or_else(|| unsized_plane(what, id))?;
                    planes.insert(
                        id.0,
                        Plane {
                            layer: decode_layers,
                            at: page * here,
                            width,
                        },
                    );
                    here += width;
                }
                match per_token {
                    None => per_token = Some(here),
                    Some(first) if first == here => {}
                    Some(first) => {
                        return Err(Fault::Unbound {
                            what: format!(
                                "recurrent layer {decode_layers} buffers {here} elements a token \
                                 where layer 0 buffers {first} — this pool cuts one page-slot \
                                 stride for every layer"
                            ),
                        });
                    }
                }
                decode_layers += 1;
                continue;
            }
            let (reads, chunked) = match &node.op {
                Operation::Ssm(Ssm::GatedDeltaChunked { qkv, gates, .. }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                    },
                    true,
                ),
                Operation::Ssm(Ssm::GatedDelta { qkv, gates, .. }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                    },
                    false,
                ),
                Operation::Ssm(Ssm::KdaChunked { mixed, f, b, .. }) => (
                    Reads::Kda {
                        mixed: *mixed,
                        f: *f,
                        b: *b,
                    },
                    true,
                ),
                Operation::Ssm(Ssm::Kda { mixed, f, b, .. }) => (
                    Reads::Kda {
                        mixed: *mixed,
                        f: *f,
                        b: *b,
                    },
                    false,
                ),
                _ => continue,
            };
            let written = |id: ValueId| defined.get(&id.0).map(|n| (*n, &trace.nodes[*n].op));
            let conv = match reads {
                Reads::GatedDelta { qkv, .. } | Reads::Kda { mixed: qkv, .. } => written(qkv),
            }
            .and_then(|(n, op)| match op {
                Operation::Ssm(Ssm::CausalConv1dChunked { x, .. }) if chunked => Some((n, *x)),
                Operation::Ssm(Ssm::CausalConv1d { x, .. }) if !chunked => Some((n, *x)),
                _ => None,
            });
            // The planes read beside the conv's rows, each with the node that
            // writes it when one does: the gate prep's `[b | a]`, or the KDA
            // scan's forget and beta projections.
            let others: Option<Vec<(&str, ValueId, Option<usize>)>> = match reads {
                Reads::GatedDelta { gates, .. } => written(gates).and_then(|(n, op)| match op {
                    Operation::Ssm(Ssm::GdnGates { ba, .. }) => {
                        Some(vec![("the gate prep's `[b | a]`", *ba, Some(n))])
                    }
                    _ => None,
                }),
                Reads::Kda { f, b, .. } => Some(vec![
                    ("the forget projection's rows", f, None),
                    ("the beta projection's rows", b, None),
                ]),
            };
            let (Some((conv_at, x)), Some(others)) = (conv, others) else {
                if !chunked {
                    continue;
                }
                return Err(Fault::Unbound {
                    what: match reads {
                        Reads::GatedDelta { .. } => format!(
                            "the chunked recurrence at node {at}, whose `qkv` and `gates` are \
                             not written by a chunked conv and a gate prep — this shell buffers \
                             the two in-projection planes those ops read and knows no third \
                             shape"
                        ),
                        Reads::Kda { .. } => format!(
                            "the chunked KDA recurrence at node {at}, whose `mixed` is not \
                             written by a chunked conv — this shell buffers the conv's rows \
                             beside the forget and beta projections and knows no other shape"
                        ),
                    },
                });
            };
            if let Some(late) = core::iter::once(conv_at)
                .chain(others.iter().filter_map(|(_, _, wrote)| *wrote))
                .find(|wrote| *wrote >= at)
            {
                return Err(Fault::Unbound {
                    what: format!(
                        "the recurrence at node {at} reads a plane written at node {late}, \
                         which does not stand before it"
                    ),
                });
            }
            let page = u64::from(paging);
            let layer = if chunked { layers } else { decode_layers };
            let mut here = 0u64;
            for (what, id, _) in core::iter::once(("the conv's rows", x, None)).chain(others) {
                let width = width_of(trace, id).ok_or_else(|| unsized_plane(what, id))?;
                planes.insert(
                    id.0,
                    Plane {
                        layer,
                        at: page * here,
                        width,
                    },
                );
                here += width;
            }
            match per_token {
                None => per_token = Some(here),
                Some(first) if first == here => {}
                Some(first) => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "recurrent layer {layer} buffers {here} elements a token where \
                             layer 0 buffers {first} — this pool cuts one page-slot stride \
                             for every layer"
                        ),
                    });
                }
            }
            if chunked {
                layers += 1;
            } else {
                decode_layers += 1;
            }
        }
        if decode_layers != 0 && decode_layers != layers {
            return Err(Fault::Unbound {
                what: format!(
                    "the plan runs {decode_layers} step recurrence(s) and {layers} chunked \
                     one(s); the buffered planes pair the two chains layer by layer"
                ),
            });
        }
        let Some(per_token) = per_token else {
            return Ok(None);
        };
        Ok(Some((Planes(planes), per_token, layers)))
    }
}

impl Buffers {
    #[must_use]
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arena.committed_bytes()
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.slot_bytes().saturating_mul(u64::from(self.slots))
    }

    #[must_use]
    pub fn pages(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.per_page
            .saturating_mul(u64::from(self.layers))
            .saturating_mul(ELEMENT)
    }

    #[must_use]
    pub fn map_unit(&self) -> u64 {
        self.arena.map_unit()
    }

    pub fn ensure(&mut self, pools: &mut Pools, pages: u32) -> Result<()> {
        if pages <= self.committed_pages {
            return Ok(());
        }
        if pages > self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(pages),
                have: u64::from(self.slots),
            });
        }
        let bytes = u64::from(pages).saturating_mul(self.slot_bytes());
        pools.commit_arena(&mut self.arena, bytes)?;
        self.committed_pages = pages;
        Ok(())
    }

    /// Whether the scratch's range holds an extended run of `rows_ext`
    /// rows. Past it [`Buffers::ensure_scratch`] reserves the scratch anew,
    /// at a base the fires in flight do not hold, so they land first.
    #[must_use]
    pub fn scratch_reaches(&self, rows_ext: u32) -> bool {
        RsScratch::need(rows_ext, self.ext_row_bytes) <= self.scratch.max_bytes()
    }

    /// Backs the scratch for an extended run of `rows_ext` rows: in place
    /// within its range, anew past it (a flat arena's first fire, or the
    /// lanes of one group replaying one buffer, more slots between them
    /// than the pool holds).
    pub fn ensure_scratch(&mut self, pools: &Pools, rows_ext: u32) -> Result<()> {
        let bytes = RsScratch::need(rows_ext, self.ext_row_bytes);
        if bytes <= self.scratch.committed_bytes() {
            return Ok(());
        }
        if !self.scratch_reaches(rows_ext) {
            self.scratch = pools.reserve_arena(bytes, SCRATCH)?;
        }
        pools.commit_beside(&mut self.scratch, bytes)
    }

    /// The scratch's base and the bytes of it backed.
    #[must_use]
    pub fn scratch(&self) -> (u64, u64) {
        (self.scratch.base(), self.scratch.committed_bytes())
    }

    /// The page `slot` names across the layers, when it is backed.
    #[must_use]
    pub fn page_span(&self, slot: u32) -> Option<(u64, u64)> {
        (slot < self.committed_pages).then(|| {
            let bytes = self.slot_bytes();
            (self.arena.base() + u64::from(slot) * bytes, bytes)
        })
    }

    /// `plane` as the table-driven copies read it: every layer's planes of a
    /// page slot behind one another, `per_page` elements a layer.
    #[must_use]
    pub fn paged(&self, plane: Plane) -> kernels_cuda::gather::PagedPlane {
        kernels_cuda::gather::PagedPlane {
            base: self.arena.base(),
            per_slot: self.per_page * u64::from(self.layers),
            at: u64::from(plane.layer) * self.per_page + plane.at,
            width: u32::try_from(plane.width).unwrap_or(u32::MAX),
            page_tokens: self.page_tokens,
            dtype: PLANE_DTYPE,
        }
    }

    /// Buffered token `token` of a lane whose buffer is `pages`, as the
    /// table-driven copies name it: its page slot times the page's tokens,
    /// plus its token within the page. The slot is backed before the fire.
    pub fn token(&self, pages: &[u32], token: u32) -> Result<i32> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let slot = *pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "rs buffer pages",
            need: u64::from(page) + 1,
            have: pages.len() as u64,
        })?;
        self.backed(slot, in_page)?;
        let named = u64::from(slot) * u64::from(self.page_tokens) + u64::from(in_page);
        i32::try_from(named).map_err(|_| Fault::Ceiling {
            what: "rs buffer tokens an i32 names",
            need: named,
            have: i32::MAX as u64,
        })
    }

    /// The most rows an extended run takes: a fire's rows beside every page
    /// slot replayed once.
    #[must_use]
    pub fn ext_reach(&self, max_rows: u32) -> u32 {
        max_rows.saturating_add(self.slots.saturating_mul(self.page_tokens))
    }

    /// Whether token `row` of page slot `page_slot` is one this pool backs.
    fn backed(&self, page_slot: u32, row: u32) -> Result<()> {
        if page_slot >= self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.slots),
            });
        }
        if row >= self.page_tokens {
            return Err(Fault::Ceiling {
                what: "rs buffer page tokens",
                need: u64::from(row) + 1,
                have: u64::from(self.page_tokens),
            });
        }
        if page_slot >= self.committed_pages {
            return Err(Fault::Ceiling {
                what: "rs buffer pages committed before this fire",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.committed_pages),
            });
        }
        Ok(())
    }
}

fn unsized_plane(what: &str, id: ValueId) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what} at value {}, whose row is not a constant width of {PLANE_DTYPE:?} this \
             shell can reserve a buffered page for",
            id.0
        ),
    }
}

fn width_of(trace: &Trace, id: ValueId) -> Option<u64> {
    let decl = trace.values.get(id.0 as usize)?;
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return None;
    };
    if *dtype != PLANE_DTYPE {
        return None;
    }
    match shape.last()? {
        Dim::Const(width) => Some(*width),
        _ => None,
    }
}

/// The snapshot rows one lane lands in a fire (`Lane::rs_snapshots`), and
/// the chunks every fire's scans keep for them, so a body recorded with none
/// replays with up to this many.
pub const SNAPSHOTS: u32 = 16;

/// The snapshots this engine lands in a lane: [`SNAPSHOTS`] where every op
/// that reads a state slot is a recurrent scan or window whose chunked arm
/// lands them (a single-token arm never meets one), else none. On CUDA the
/// gated delta rule's prefill lands them on the two-launch scan, where
/// `gdn_lands` says this device takes it at a key width.
#[must_use]
pub fn snapshots(trace: &Trace, gdn_lands: impl Fn(u32) -> bool) -> u32 {
    let state = |id: &ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|value| &value.def),
            Some(poem_ir::Def::Cache(row))
                if matches!(trace.caches.get(*row as usize), Some(poem_ir::CacheRow::State { .. }))
        )
    };
    let mut reads = Vec::new();
    let lands = trace.nodes.iter().all(|node| {
        reads.clear();
        node.op.inputs(&mut reads);
        !reads.iter().any(state) || lands_snapshots(&node.op, &gdn_lands)
    });
    if lands { SNAPSHOTS } else { 0 }
}

#[cfg(feature = "rocm")]
fn lands_snapshots(op: &Operation, _gdn_lands: &impl Fn(u32) -> bool) -> bool {
    matches!(
        op,
        Operation::Ssm(
            Ssm::CausalConv1d { .. }
                | Ssm::CausalConv1dChunked { .. }
                | Ssm::ShortConv { .. }
                | Ssm::ShortConvChunked { .. }
                | Ssm::GatedDelta { .. }
                | Ssm::GatedDeltaChunked { .. }
                | Ssm::Kda { .. }
                | Ssm::KdaChunked { .. }
        ) | Operation::Gather(
            poem_ir::Gather::NgramIds { .. } | poem_ir::Gather::NgramIdsChunked { .. }
        ) | Operation::Fused(poem_ir::Fused::GdnStep { .. })
    )
}

/// CUDA's arms that land a snapshot: the causal conv's row-tiled commit
/// (a reach of at most 32 rows) and the gated delta rule's scans.
#[cfg(not(feature = "rocm"))]
fn lands_snapshots(op: &Operation, gdn_lands: &impl Fn(u32) -> bool) -> bool {
    match op {
        Operation::Ssm(Ssm::CausalConv1dChunked {
            conv_width,
            dilation,
            ..
        }) => conv_width.saturating_sub(1).saturating_mul(*dilation) <= 32,
        Operation::Ssm(Ssm::GatedDeltaChunked { k_dim, v_dim, .. }) => {
            *k_dim == 128 && *v_dim == 128 && gdn_lands(*k_dim)
        }
        Operation::Ssm(Ssm::CausalConv1d { .. } | Ssm::GatedDelta { .. })
        | Operation::Fused(poem_ir::Fused::GdnStep { .. }) => true,
        _ => false,
    }
}

/// A fire's snapshots in fire-lane order: lane `l`'s are entries
/// `indptr[l]..indptr[l + 1]` of `rows` and `slots`.
#[derive(Debug, Default, Clone)]
pub struct SnapTable {
    pub indptr: Vec<i32>,
    pub rows: Vec<i32>,
    pub slots: Vec<i32>,
}

#[derive(Debug)]
pub struct Predicate {
    region: Buffer,
    at_one: u64,
    at_zero: u64,
    at_commits: u64,
    at_indptr: u64,
    at_mask: u64,
    at_len: u64,
    at_snap: Option<[u64; 3]>,
    /// The lane bounds the last fire left nonzero, which the next fire that
    /// takes no snapshot zeroes; a run of fires taking none stages nothing.
    snap_dirty: usize,
    lanes: u32,
}

impl Predicate {
    pub fn reserve(max_lanes: u32, snapshots: bool) -> Result<Predicate> {
        let lanes = u64::from(max_lanes);
        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(256);
            here
        };
        let at_one = take(4);
        let at_zero = take(4);
        let at_commits = take(lanes * 8);
        let at_indptr = take((lanes + 1) * 4);
        let at_mask = take(lanes);
        let at_len = take(lanes * 4);
        let at_snap = snapshots.then(|| {
            let entries = lanes * u64::from(SNAPSHOTS);
            [take((lanes + 1) * 4), take(entries * 4), take(entries * 4)]
        });
        let mut region = Buffer::zeroed(usize::try_from(at).unwrap_or(usize::MAX))?;
        region.write(at_one, &1u32.to_le_bytes())?;
        let identity: Vec<u8> = (0..=max_lanes)
            .flat_map(|l| (l as i32).to_le_bytes())
            .collect();
        region.write(at_indptr, &identity)?;
        Ok(Predicate {
            region,
            at_one,
            at_zero,
            at_commits,
            at_indptr,
            at_mask,
            at_len,
            at_snap,
            snap_dirty: 0,
            lanes: max_lanes,
        })
    }

    #[must_use]
    pub fn always(&self) -> u64 {
        self.region.ptr() + self.at_one
    }

    #[must_use]
    pub fn never(&self) -> u64 {
        self.region.ptr() + self.at_zero
    }

    pub fn write(
        &mut self,
        stream: *mut core::ffi::c_void,
        commits: &[u64],
        lens: &[i32],
    ) -> Result<()> {
        if commits.len() as u64 > u64::from(self.lanes) {
            return Err(Fault::Ceiling {
                what: "rs fold predicates",
                need: commits.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let words: Vec<u8> = commits.iter().flat_map(|c| c.to_le_bytes()).collect();
        let at_commits = self.at_commits;
        let at_len = self.at_len;
        self.region.stage(stream, at_commits, &words)?;
        let lens: Vec<u8> = lens.iter().flat_map(|n| n.to_le_bytes()).collect();
        self.region.stage(stream, at_len, &lens)
    }

    #[must_use]
    pub fn commits(&self) -> u64 {
        self.region.ptr() + self.at_commits
    }

    #[must_use]
    pub fn indptr(&self) -> u64 {
        self.region.ptr() + self.at_indptr
    }

    #[must_use]
    pub fn mask(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_mask, lanes, 1, Dtype::U8)
    }

    #[must_use]
    pub fn commit_len(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_len, lanes, 1, Dtype::I32)
    }

    /// Whether this load lands snapshots: its table's planes are reserved.
    #[must_use]
    pub fn lands_snapshots(&self) -> bool {
        self.at_snap.is_some()
    }

    /// The fire's snapshot table, staged on every fire of an engine that
    /// lands them (its lanes' bounds, zero where none takes one), so every
    /// launch reads it where a body recorded it.
    pub fn snapshots(
        &mut self,
        stream: *mut core::ffi::c_void,
        table: &SnapTable,
    ) -> Result<kernels_cuda::Snapshots> {
        let Some(at_snap) = self.at_snap else {
            return Ok(kernels_cuda::Snapshots::NONE);
        };
        let entries = u64::from(self.lanes) * u64::from(SNAPSHOTS);
        if table.rows.len() as u64 > entries {
            return Err(Fault::Ceiling {
                what: "rs snapshots",
                need: table.rows.len() as u64,
                have: entries,
            });
        }
        if !table.rows.is_empty() {
            for (at, values) in at_snap
                .into_iter()
                .zip([&table.indptr, &table.rows, &table.slots])
            {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.region.stage(stream, at, &bytes)?;
            }
            self.snap_dirty = table.indptr.len();
        } else if self.snap_dirty > 0 {
            self.region
                .stage(stream, at_snap[0], &vec![0u8; self.snap_dirty * 4])?;
            self.snap_dirty = 0;
        }
        let planes =
            at_snap.map(|at| kernels_cuda::Tensor::new(self.region.ptr() + at, 1, 1, Dtype::I32));
        let [indptr, rows, slots] = planes;
        Ok(kernels_cuda::Snapshots {
            indptr,
            rows,
            slots,
            count: SNAPSHOTS.max(table.rows.len() as u32),
        })
    }

    /// The fold predicate's mask for `lanes`, read on `stream` behind the
    /// fire that wrote it.
    pub fn read_mask(&self, stream: *mut core::ffi::c_void, lanes: u32) -> Result<Vec<u8>> {
        let mut out = vec![0u8; lanes as usize];
        self.region.read_on(stream, self.at_mask, &mut out)?;
        Ok(out)
    }
}

/// A buffered fire's movement, as the tables its recurrent ops read in
/// place of host-looped copies; every entry an `i32`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RsStaged {
    /// The fire rows the per-row tables span: the rows the fire carves, so
    /// every fire of one body key lays its tables alike.
    pub rows: u32,
    /// The extended run's rows ([`crate::record::RsShape::ext`]); 0 when no
    /// lane replays.
    pub ext: u32,
    /// Each fire row's buffered token ([`Buffers::token`]), -1 for a row of
    /// a lane that folds.
    pub scatter: Vec<i32>,
    /// Each fire row's row of the extended run, -1 past the live rows;
    /// empty when no lane replays.
    pub own: Vec<i32>,
    /// Each extended row's buffered token, -1 for a lane's own row (or one
    /// past the last lane's).
    pub replay: Vec<i32>,
    /// Each window's lane boundaries in the extended run, at the word its
    /// own boundaries stand at among the windows' slots (`Slots::at`).
    pub csr: Vec<(u64, Vec<i32>)>,
    /// Each window seat's stage word over the extended run (its rows to its
    /// last lane's end, no row offset, its lanes and lane offset), laid as
    /// `Windows::live` lays the seats' words over the fire's rows.
    pub live: Vec<u32>,
}

/// Where a fire's [`RsTables`] stand on the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RsTableSeat {
    pub scatter: u64,
    pub own: u64,
    pub replay: u64,
    pub csr: u64,
    pub live: u64,
}

/// The device region every buffered fire stages its [`RsStaged`] into, in
/// one copy: the seats' stage words and the windows' boundaries at fixed
/// words, then the per-row tables at offsets set by the fire's carved rows
/// and extended rows alone, both of which its body key holds. Two fires of
/// one key therefore read their tables where the body recorded them.
#[derive(Debug)]
pub struct RsTables {
    region: Buffer,
    live_words: u64,
    csr_words: u64,
    max_rows: u64,
    max_ext: u64,
}

impl RsTables {
    /// Tables for fires of up to `max_rows` rows and `max_ext` extended
    /// rows, over windows whose slots take `csr_words` words and whose
    /// seats' stage words take `live_words`.
    pub fn reserve(
        max_rows: u32,
        max_ext: u32,
        csr_words: u64,
        live_words: u64,
    ) -> Result<RsTables> {
        let (max_rows, max_ext) = (u64::from(max_rows), u64::from(max_ext));
        let words = live_words + csr_words + 2 * max_rows + max_ext + 16;
        Ok(RsTables {
            region: Buffer::zeroed(usize::try_from(words * 4).unwrap_or(usize::MAX))?,
            live_words,
            csr_words,
            max_rows,
            max_ext,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.region.bytes() as u64
    }

    /// The word offsets of `fire`'s sections: live, csr, scatter, own, replay.
    fn layout(&self, rows: u64) -> [u64; 5] {
        let live = 0;
        let csr = self.live_words;
        let scatter = csr + self.csr_words;
        let own = scatter + rows.next_multiple_of(4);
        let replay = own + rows.next_multiple_of(4);
        [live, csr, scatter, own, replay]
    }

    /// Stages `fire`'s tables, one host-to-device copy behind the fires
    /// before it on `stream`.
    pub fn stage(
        &mut self,
        stream: *mut core::ffi::c_void,
        fire: &RsStaged,
    ) -> Result<RsTableSeat> {
        let rows = u64::from(fire.rows);
        let ext = u64::from(fire.ext);
        for (what, need, have) in [
            ("rs buffered fire rows", rows, self.max_rows),
            ("rs extended rows", ext, self.max_ext),
            (
                "rs window seat words",
                fire.live.len() as u64,
                self.live_words,
            ),
            (
                "rs fire rows the scatter names",
                fire.scatter.len() as u64,
                rows,
            ),
            (
                "rs fire rows the own map names",
                fire.own.len() as u64,
                rows,
            ),
            (
                "rs extended rows the replay names",
                fire.replay.len() as u64,
                ext,
            ),
        ] {
            if need > have {
                return Err(Fault::Ceiling { what, need, have });
            }
        }
        let [live, csr, scatter, own, replay] = self.layout(rows);
        let mut image = vec![0i32; usize::try_from(replay + ext).unwrap_or(usize::MAX)];
        for (at, word) in fire.live.iter().enumerate() {
            image[live as usize + at] = *word as i32;
        }
        for (at, bounds) in &fire.csr {
            if at + bounds.len() as u64 > self.csr_words {
                return Err(Fault::Ceiling {
                    what: "rs extended window boundary words",
                    need: at + bounds.len() as u64,
                    have: self.csr_words,
                });
            }
            let from = (csr + at) as usize;
            image[from..from + bounds.len()].copy_from_slice(bounds);
        }
        for (from, values) in [
            (scatter, &fire.scatter),
            (own, &fire.own),
            (replay, &fire.replay),
        ] {
            let from = from as usize;
            image[from..from + values.len()].copy_from_slice(values);
        }
        let bytes: Vec<u8> = image.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.region.stage(stream, 0, &bytes)?;
        let base = self.region.ptr();
        Ok(RsTableSeat {
            scatter: base + scatter * 4,
            own: base + own * 4,
            replay: base + replay * 4,
            csr: base + csr * 4,
            live: base + live * 4,
        })
    }
}

#[cfg(test)]
mod tests {
    use poem_ir::{Def, Guard, Node, Platform, ValueDecl};

    use super::*;

    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 32;
    const PAGE: u32 = 16;

    fn row(width: u64, dtype: Dtype) -> ValueDecl {
        ValueDecl {
            def: Def::Op(0),
            ty: Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(width)],
                dtype,
            },
        }
    }

    /// One KDA layer as `glm5_next` and `kimi_k3` state it: the conv's rows
    /// (0) through the chunked conv into `mixed` (3), which the scan reads
    /// beside its forget (4) and beta (5) projections, landing `y` (9).
    fn kda_values() -> Vec<ValueDecl> {
        let wide = HEADS * HEAD_DIM;
        vec![
            row(3 * wide, Dtype::Bf16),
            row(4, Dtype::Bf16),
            row(4, Dtype::Bf16),
            row(3 * wide, Dtype::Bf16),
            row(wide, Dtype::Bf16),
            row(HEADS, Dtype::Bf16),
            row(wide, Dtype::F32),
            row(HEADS, Dtype::F32),
            row(wide, Dtype::F32),
            row(wide, Dtype::F32),
        ]
    }

    fn kda_chunked(mixed: u32) -> Operation {
        Operation::Ssm(Ssm::KdaChunked {
            mixed: ValueId(mixed),
            f: ValueId(4),
            b: ValueId(5),
            dt_bias: ValueId(6),
            a_log: ValueId(7),
            state: ValueId(8),
            heads: HEADS as u32,
            head_dim: HEAD_DIM as u32,
            norm_eps: 1e-6,
            gate_floor: 0.0,
            y: ValueId(9),
        })
    }

    fn plan(values: Vec<ValueDecl>, nodes: Vec<Operation>) -> Trace {
        Trace {
            facts: Default::default(),
            name: "rs".to_string(),
            platform: Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values,
            nodes: nodes
                .into_iter()
                .map(|op| Node {
                    op,
                    guard: Guard::Always,
                    layer: None,
                })
                .collect(),
            seams: Vec::new(),
            drafter: None,
            inputs: Vec::new(),
        }
    }

    /// A KDA layer buffers three planes a token — the conv's rows, then the
    /// forget and beta projections — laid page by page behind one another,
    /// and its extended run stages those beside `mixed` and `y`.
    #[test]
    fn a_kda_recurrence_buffers_its_conv_rows_beside_its_two_projections() {
        let wide = HEADS * HEAD_DIM;
        let trace = plan(
            kda_values(),
            vec![
                Operation::Ssm(Ssm::CausalConv1dChunked {
                    x: ValueId(0),
                    weight: ValueId(1),
                    bias: None,
                    state: ValueId(2),
                    conv_width: 4,
                    dilation: 1,
                    y: ValueId(3),
                }),
                kda_chunked(3),
            ],
        );
        let (planes, per_token, layers) = read(&trace, PAGE)
            .expect("the KDA pair reads clean")
            .expect("the KDA pair reserves buffered planes");
        assert_eq!((layers, per_token), (1, 3 * wide + wide + HEADS));
        assert_eq!(planes.len(), 3);
        let page = u64::from(PAGE);
        let plane = |at: u64, width: u64| {
            Some(Plane {
                layer: 0,
                at,
                width,
            })
        };
        assert_eq!(planes.of(ValueId(0)), plane(0, 3 * wide));
        assert_eq!(planes.of(ValueId(4)), plane(page * 3 * wide, wide));
        assert_eq!(planes.of(ValueId(5)), plane(page * 4 * wide, HEADS));
        assert_eq!(
            ext_row_bytes(&trace).expect("the extended run's rows size"),
            2 * (3 * wide) + 2 * (3 * wide) + 2 * wide + 2 * HEADS + 4 * wide
        );
    }

    /// A chunked KDA scan whose `mixed` no chunked conv writes has no plane
    /// this shell knows to buffer, and the refusal names the scan.
    #[test]
    fn a_kda_recurrence_without_its_conv_is_refused_by_name() {
        let trace = plan(kda_values(), vec![kda_chunked(0)]);
        let why = read(&trace, PAGE).expect_err("no conv writes `mixed`");
        assert!(
            why.to_string().contains("chunked KDA recurrence at node 0"),
            "{why}"
        );
        assert!(
            read(&plan(Vec::new(), Vec::new()), PAGE)
                .expect("an empty plan reads clean")
                .is_none()
        );
    }
}
