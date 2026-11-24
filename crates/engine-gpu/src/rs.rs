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

    /// A gated-delta row's keys (the conv output's key heads, bf16), which
    /// a decode step reads its carried rows' attention weights by
    /// (`ChunkProfile::rs_corrected`).
    Keys,

    /// A gated-delta row's corrected value `u = beta (v - g S^T k)` (f32),
    /// the value its delta rule writes, landed once, when the row is a
    /// fire's own.
    U,

    /// A gated-delta row's per-head scalars (f32, `v_heads` log-gates then
    /// `v_heads` key-norm inverses, `poem_compiler::rs::gdn_scalars_bytes`),
    /// which the recipe's rs row lays after its corrected values. This
    /// shell recomputes them from the row's `[b | a]` and keys: it lays the
    /// plane (a beam's fork copies it with the rest) and neither writes nor
    /// reads it.
    Scalars,
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub cache: u32,
    pub role: Role,
    pub width: u32,
    pub dtype: Dtype,

    pub row_bytes: u64,

    /// Whether a fire's extended run holds rows of it: a recurrence input
    /// replayed through the activation path. A gated-delta row's keys and
    /// corrected values are read in the pool where they lie, never extended.
    pub extended: bool,
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

    /// Each plane by its recurrence's state cache row and its role: a
    /// decode step finds its planes by the state it reads, whichever form
    /// of the layer's ops named them.
    pub by_role: HashMap<(u32, Role), usize>,

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

        // A plane of `value`'s rows, or (`fixed`) of a row `width` wide of
        // `dtype` read off it in place, which no fire extends.
        let mut plane = |layout: &mut Layout,
                         cache: u32,
                         role: Role,
                         value: ValueId,
                         node: usize,
                         fixed: Option<(u32, Dtype)>|
         -> Result<()> {
            let (width, dtype, row_bytes) = match fixed {
                None => row_of(trace, value)?,
                Some((width, dtype)) => (
                    width,
                    dtype,
                    u64::from(width)
                        * poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| {
                            Fault::Unbound {
                                what: format!("a {dtype:?} plane, whose element has no size"),
                            }
                        })?,
                ),
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
                        extended: fixed.is_none(),
                    });
                    layout.plane_life.push((node, node));
                    plane_keys.insert((cache, role), layout.planes.len() - 1);
                    layout.by_role.insert((cache, role), layout.planes.len() - 1);
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
            plane(&mut layout, cache, Role::ConvX, conv.x, conv.node, None)?;
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
                    gates_cache.insert(gates.0, cache);
                    region(&mut layout, cache, "scan", *y, at)?;
                    // The prompt form's rows keep their keys and corrected
                    // values (the recipe's rs row lays them by its `q k v`
                    // and its output): a decode step reads its carried rows
                    // as a linear attention over them.
                    if let Operation::Ssm(Ssm::GatedDeltaChunked { qkv, .. }) = &node.op {
                        plane(
                            &mut layout,
                            cache,
                            Role::Keys,
                            *qkv,
                            at,
                            Some((k_heads * k_dim, Dtype::Bf16)),
                        )?;
                        plane(
                            &mut layout,
                            cache,
                            Role::U,
                            *y,
                            at,
                            Some((v_heads * v_dim, Dtype::F32)),
                        )?;
                        plane(
                            &mut layout,
                            cache,
                            Role::Scalars,
                            *gates,
                            at,
                            Some((2 * v_heads, Dtype::F32)),
                        )?;
                    }
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
                    plane(&mut layout, cache, Role::KdaF, *f, at, None)?;
                    plane(&mut layout, cache, Role::KdaB, *b, at, None)?;
                    region(&mut layout, cache, "scan", *y, at)?;
                    // The prompt form's rows keep their keys and corrected
                    // values too (the recipe lays them by its `mixed` and its
                    // output). This shell's KDA decode step has no carried
                    // form: it replays its rows extended and reads neither.
                    if let Operation::Ssm(Ssm::KdaChunked { mixed, .. }) = &node.op {
                        plane(&mut layout, cache, Role::Keys, *mixed, at, Some((heads * head_dim, Dtype::Bf16)))?;
                        plane(&mut layout, cache, Role::U, *y, at, Some((heads * head_dim, Dtype::F32)))?;
                    }
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
                    plane(&mut layout, cache_of(trace, *state)?, Role::Ids, *ids, at, None)?;
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
            plane(&mut layout, cache, Role::Ba, *ba, at, None)?;
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

    /// Each plane's first byte in an rs page.
    #[must_use]
    pub fn plane_at(&self) -> Vec<u64> {
        self.plane_at.clone()
    }

    /// The bytes an rs page id counts.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.stride
    }
}

pub struct Seat {
    pub lanes: Vec<LanePlan>,

    /// Whether each lane's decode steps carry its buffered rows in their own
    /// kernels, reading them through [`Self::carry`] where they lie: such a
    /// lane's plan moves nothing from the host.
    pub carried: Vec<bool>,

    pub replay: Tensor,

    pub commit: Tensor,

    /// The fire's staged rs table (`kernels_slang::ssm::Carry`).
    pub carry: Tensor,

    pub slots: Tensor,

    /// The chunk pool the rs pages lie in.
    pub pool: Segments,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    /// The bytes an rs page id counts.
    pub stride: u64,

    /// Each plane's extended rows, where a fire extends it.
    pub ext_in: Vec<Option<Tensor>>,

    pub ext_out: Vec<Tensor>,

    pub work: Tensor,
    pub rows_ext: u32,

    /// The dispatches every recurrent copy of a fire takes
    /// ([`kernels_slang::gather::rows::copy_runs`]): enough for the most
    /// runs a fire of the load can move (every lane's rows over the most
    /// pages the room's rows span, twice, and a run of its own each), the
    /// same for every fire.
    pub run_dispatches: usize,

    pub ext: RefCell<HashMap<u32, Tensor>>,

    /// The fire's beam groups' folds (`BEAM-PAGE-LISTS.md` §11.3): their
    /// staged table and records, a lane a fold, and how many. Each
    /// recurrent layer's folds run ahead of its decode step.
    pub folds: Option<Folds>,
    /// Each fold's group's first lane in the fire (its rs table header),
    /// ascending: a window's folds are those of its lanes.
    pub fold_lanes: Vec<u32>,
}

/// A fire's beam groups' folds as a carried step's fold twin reads them:
/// a rs table (`kernels_slang::ssm::Carry`) and slot table of their own,
/// a lane a group's set, and their count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Folds {
    pub table: Tensor,
    pub slots: Tensor,
    pub count: u32,
}

/// What a fire's recording is keyed by: everything its kernels bind and every
/// host move it makes. A lane's replay and commit lengths are the staged
/// tables' data, read on the device, and key nothing: only the plans of lanes
/// whose rows the host moves (an extended lane's gathers and scatter) do. A
/// carried lane moves none, so its decodes replay one recording whatever rows
/// they carry, and a folding lane beside them keys none either.
impl std::fmt::Debug for Seat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let moved: Vec<(usize, &LanePlan)> = self
            .lanes
            .iter()
            .enumerate()
            .filter(|(lane, plan)| {
                !self.carried.get(*lane).copied().unwrap_or(false)
                    && (!plan.gathers.is_empty() || plan.scatter.is_some())
            })
            .collect();
        f.debug_struct("Seat")
            .field("moved", &moved)
            .field("carried", &self.carried)
            .field("replay", &self.replay)
            .field("commit", &self.commit)
            .field("carry", &self.carry)
            .field("slots", &self.slots)
            .field("pool", &self.pool)
            .field("plane_at", &self.plane_at)
            .field("page_tokens", &self.page_tokens)
            .field("stride", &self.stride)
            .field("folds", &self.folds)
            .field("fold_lanes", &self.fold_lanes)
            .field("ext_in", &self.ext_in)
            .field("ext_out", &self.ext_out)
            .field("work", &self.work)
            .field("rows_ext", &self.rows_ext)
            .finish_non_exhaustive()
    }
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
            .map(|plane| plane.extended.then_some(plane.row_bytes))
            .zip(self.plane_life.iter().copied())
            .chain(
                self.regions
                    .iter()
                    .map(|region| Some(region.row_bytes))
                    .zip(self.region_life.iter().copied()),
            )
            // A plane no fire extends takes no bytes of the run.
            .map(|(row_bytes, life)| {
                (
                    row_bytes.map_or(0, |row_bytes| (rows * row_bytes).max(4).next_multiple_of(256)),
                    life,
                )
            })
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
        carried: Vec<bool>,
        replay: Tensor,
        commit: Tensor,
        carry: Tensor,
        slots: Tensor,
        rows_ext: u32,
        folds: Option<Folds>,
        fold_lanes: Vec<u32>,
        max_lanes: u32,
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
        // A fire whose lanes all carry their rows in their decode steps
        // extends none: it binds no view of the run.
        let extends = !carried.iter().all(|&carried| carried);
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for (plane, &offset) in layout.planes.iter().zip(&plane_at) {
            ext_in.push(if plane.extended && extends {
                Some(take(offset, plane.row_bytes, plane.width, plane.dtype)?)
            } else {
                None
            });
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for (region, &offset) in layout.regions.iter().zip(&region_at) {
            if extends {
                ext_out.push(take(offset, region.row_bytes, region.width, region.dtype)?);
            }
        }
        // The work plane holds a committed dispatch's lanes: a window of more
        // takes as many dispatches (`dispatch::ssm`).
        let work = Tensor::new(
            handles.bind(&held.work, 0, held.work.bytes())?,
            held.lanes,
            u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
            Dtype::F32,
        );
        // The most runs a fire's recurrent copy moves: a lane's gathered and
        // scattered rows each split at the pages they span (at most the
        // room's rows over a page, and one more a run), and its own rows.
        let pages = u64::from(room_rows.div_ceil(buffers.page_tokens.max(1)));
        let most_lanes = u64::from(max_lanes.max(1)).min(u64::from(room_rows.max(1)));
        let most = 2 * (pages + most_lanes) + most_lanes;
        let run_dispatches = usize::try_from(
            most.div_ceil(kernels_slang::gather::rows::COPY_RUNS as u64),
        )
        .unwrap_or(usize::MAX)
        .max(1);
        Ok(Seat {
            lanes,
            carried,
            replay,
            commit,
            carry,
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
            run_dispatches,
            ext: RefCell::new(HashMap::new()),
            folds,
            fold_lanes,
        })
    }

    /// Whether the window from fire lane `lane0` carries its lanes' buffered
    /// rows in its decode steps' own kernels.
    #[must_use]
    pub fn carries(&self, lane0: u32) -> bool {
        self.carried.get(lane0 as usize).copied().unwrap_or(false)
    }

    /// The staged rs table as the window from fire lane `lane0` reads it.
    pub fn carry(&self, lane0: u32) -> Result<kernels_slang::ssm::Carry> {
        let stride_words = u32::try_from(self.stride / 4)
            .ok()
            .filter(|_| self.stride.is_multiple_of(4))
            .ok_or(Fault::Ceiling {
                what: "words of an rs page a kernel addresses",
                need: self.stride,
                have: u64::from(u32::MAX),
            })?;
        Ok(kernels_slang::ssm::Carry {
            table: self.carry,
            slots: self.slots,
            lane0,
            page_tokens: self.page_tokens,
            stride_words,
        })
    }

    /// Where plane `role` of the recurrence keeping state cache row `cache`
    /// lies in an rs page, in words.
    /// The beam groups' folds' rs table from fold `lane0`, as their fold
    /// twins read it.
    pub fn fold_carry(&self, folds: &Folds, lane0: u32) -> Result<kernels_slang::ssm::Carry> {
        let stride_words = u32::try_from(self.stride / 4).map_err(|_| Fault::Ceiling {
            what: "words an rs page id counts",
            need: self.stride / 4,
            have: u64::from(u32::MAX),
        })?;
        Ok(kernels_slang::ssm::Carry {
            table: folds.table,
            slots: folds.slots,
            lane0,
            page_tokens: self.page_tokens,
            stride_words,
        })
    }

    pub fn carry_plane(&self, cache: u32, role: Role) -> Result<kernels_slang::ssm::CarryPlane> {
        let plane = *self.layout.by_role.get(&(cache, role)).ok_or_else(|| Fault::Unbound {
            what: format!(
                "the {role:?} plane of state cache row {cache}, which this load buffers no rows of"
            ),
        })?;
        let (at, row) = (self.plane_at[plane], self.layout.planes[plane].row_bytes);
        let words = |bytes: u64| {
            u32::try_from(bytes / 4)
                .ok()
                .filter(|_| bytes.is_multiple_of(4))
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "the {role:?} plane of state cache row {cache} at byte {at}, {row} bytes \
                         a token: a kernel reads its rows in whole words"
                    ),
                })
        };
        Ok(kernels_slang::ssm::CarryPlane {
            at: words(at)?,
            row: words(row)?,
        })
    }

    /// Where token `token` of `run`'s plane `plane` lies: the segment's
    /// handle, the byte there, and the rows left in its page.
    /// [`Seat::locate`] by the segment's place among the pool's: the segment,
    /// the byte in it, and the rows from there to the page's end.
    pub fn locate_segment(&self, run: &Run, plane: usize, token: u32) -> Result<(u32, u64, u32)> {
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
        let segment = u32::try_from(at / span).unwrap_or(u32::MAX);
        Ok((segment, at % span, self.page_tokens - in_page))
    }

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
                extended: true,
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
