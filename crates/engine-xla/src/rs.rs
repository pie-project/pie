//! Recurrent-state verbs: which rows of a hybrid model's recurrence
//! inputs are buffered per lane, replayed ahead of its own rows, and how
//! much of the extended run commits its state. The host half (the layout
//! the trace implies and each lane's plan) is engine-wgpu's; the device half
//! is `Seat` below and `dispatch/rs.rs`.

use std::collections::HashMap;

use engine::fire::{FoldLen, RsVerb};
use model_ir::{Attention, Def, Dim, Dtype, Operation, Trace, Ty, ValueId};

use crate::error::{Fault, Result};

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
    let element = model_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
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
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            match op {
                Attention::SsmCausalConv1d { x, state, y, .. }
                | Attention::SsmCausalConv1dChunked { x, state, y, .. } => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::ConvX, *x)?;
                    region(&mut layout, cache, "conv", *y)?;
                }
                Attention::SsmGatedDelta {
                    gates,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }
                | Attention::SsmGatedDeltaChunked {
                    gates,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                } => {
                    let cache = cache_of(trace, *state)?;
                    let _ = k_heads;
                    gates_cache.insert(gates.0, cache);
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*v_heads) * u64::from(*v_dim) * u64::from(*k_dim));
                }
                Attention::SsmKdaStep {
                    f,
                    b,
                    state,
                    heads,
                    head_dim,
                    y,
                    ..
                }
                | Attention::SsmKdaChunked {
                    f,
                    b,
                    state,
                    heads,
                    head_dim,
                    y,
                    ..
                } => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::KdaF, *f)?;
                    plane(&mut layout, cache, Role::KdaB, *b)?;
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*heads) * u64::from(*head_dim) * u64::from(*head_dim));
                }
                Attention::PleNgramIds {
                    ids,
                    state,
                    ngram_ids,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    ids,
                    state,
                    ngram_ids,
                    ..
                } => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::Ids, *ids)?;
                    region(&mut layout, cache, "ngram", *ngram_ids)?;
                }
                _ => {}
            }
        }
        for node in &trace.nodes {
            let Operation::Attention(Attention::SsmGdnPrep { ba, gates, .. }) = &node.op else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `attention.ssm_gdn_prep` lands, which no delta scan reads",
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub pages: Vec<u32>,
    pub from: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct LanePlan {
    pub replay: u32,

    pub commit: u32,

    pub rows: u32,

    pub gather: Option<Run>,

    pub scatter: Option<Run>,

    pub override_rows: bool,
}

impl LanePlan {
    pub fn of(verb: &RsVerb, rows: u32, lane: u32, device_fold: Option<u32>) -> Result<LanePlan> {
        let host = |len: FoldLen| -> Result<u32> {
            match len {
                FoldLen::Host(n) => Ok(n),
                FoldLen::Device(port) => device_fold.ok_or_else(|| {
                    program(
                        "serve::rs",
                        format!(
                            "lane {lane} states a device-resident fold length on port {}, and the \
                             program attached to it resolved no such port this fire",
                            port.name()
                        ),
                    )
                }),
            }
        };
        Ok(match verb {
            RsVerb::Fold => LanePlan {
                replay: 0,
                commit: rows,
                rows,
                ..LanePlan::default()
            },
            RsVerb::Buffer {
                pages,
                at,
                fold,
                replay,
            } => {
                let commit = host(*fold)?;
                if commit > replay.saturating_add(rows) {
                    return Err(program(
                        "serve::rs",
                        format!(
                            "lane {lane} folds {commit} of the {rows} rows it carries plus the \
                             {replay} it replays"
                        ),
                    ));
                }
                if *replay > *at {
                    return Err(program(
                        "serve::rs",
                        format!(
                            "lane {lane} replays {replay} buffered token(s) below buffer position \
                             {at}, which has only {at}"
                        ),
                    ));
                }
                LanePlan {
                    replay: *replay,
                    commit,
                    rows,
                    gather: (*replay > 0).then(|| Run {
                        pages: pages.clone(),
                        from: at - replay,
                        count: *replay,
                    }),
                    scatter: (rows > 0).then(|| Run {
                        pages: pages.clone(),
                        from: *at,
                        count: rows,
                    }),
                    override_rows: false,
                }
            }
            RsVerb::Window { read, write, fold } => {
                let fold = host(*fold)?;
                LanePlan {
                    replay: fold,
                    commit: fold,
                    rows,
                    gather: (fold > 0).then(|| Run {
                        pages: read.clone(),
                        from: 0,
                        count: fold,
                    }),
                    scatter: (rows > 0).then(|| Run {
                        pages: write.clone(),
                        from: 0,
                        count: rows,
                    }),
                    override_rows: false,
                }
            }
            RsVerb::FoldBuffered {
                pages,
                at,
                bound,
                len,
            } => {
                if *bound != rows {
                    return Err(program(
                        "serve::rs",
                        format!(
                            "lane {lane} replays a buffer bounded at {bound} tokens in a fire that \
                             gave it {rows} rows — the bound IS what sizes the launch, so the two \
                             are one number"
                        ),
                    ));
                }
                let commit = host(*len)?.min(*bound);
                LanePlan {
                    replay: 0,
                    commit,
                    rows,
                    gather: Some(Run {
                        pages: pages.clone(),
                        from: *at,
                        count: *bound,
                    }),
                    scatter: None,
                    override_rows: true,
                }
            }
        })
    }
}

fn program(at: &'static str, why: String) -> Fault {
    Fault::Program { at, why }
}

/// Where the recurrence inputs of one window sit in its extended rows:
/// for each extended row, whether it is replayed from the buffer (and from
/// which buffer row) or is one of the window's own rows (and which); for each
/// own row, the buffer row it is stashed into (or out of range) and the
/// extended row its output lands from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowMap {
    pub rows_ext: u32,
    pub from_buffer: Vec<i32>,
    pub buffer_row: Vec<i32>,
    pub own_row: Vec<i32>,
    pub stash: Vec<i32>,
    pub land: Vec<i32>,
}

/// Out of every buffer's range: a write there is dropped.
pub const DROP: i32 = i32::MAX;

impl WindowMap {
    /// The map of the window covering lanes `lane0..lane0 + lanes` whose own
    /// rows are cut by `indptr` (window-local), for buffers of
    /// `page_tokens`-row pages.
    pub fn of(
        plans: &[LanePlan],
        indptr: &[i32],
        lane0: u32,
        page_tokens: u32,
        pool_slots: u32,
    ) -> Result<WindowMap> {
        let mut map = WindowMap::default();
        let rows_own = indptr.last().copied().unwrap_or(0).max(0) as usize;
        map.stash = vec![DROP; rows_own];
        map.land = vec![0; rows_own];
        let locate = |run: &Run, token: u32| -> Result<i32> {
            let page = token / page_tokens;
            let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
                what: "recurrent buffer pages",
                need: u64::from(page) + 1,
                have: run.pages.len() as u64,
            })?;
            if slot >= pool_slots {
                return Err(Fault::Ceiling {
                    what: "recurrent buffer page slots",
                    need: u64::from(slot) + 1,
                    have: u64::from(pool_slots),
                });
            }
            Ok((slot * page_tokens + token % page_tokens) as i32)
        };
        for r in 0..indptr.len().saturating_sub(1) {
            let start = indptr[r].max(0) as u32;
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let Some(plan) = plans.get(lane0 as usize + r) else {
                continue;
            };
            let begin = map.from_buffer.len() as i32;
            if plan.override_rows {
                if let Some(run) = &plan.gather {
                    for t in 0..run.count {
                        map.from_buffer.push(1);
                        map.buffer_row.push(locate(run, run.from + t)?);
                        map.own_row.push(0);
                    }
                }
                for i in 0..rows {
                    map.land[(start + i) as usize] = begin + i as i32;
                }
                continue;
            }
            if let Some(run) = &plan.gather {
                for t in 0..run.count {
                    map.from_buffer.push(1);
                    map.buffer_row.push(locate(run, run.from + t)?);
                    map.own_row.push(0);
                }
            }
            let replay = map.from_buffer.len() as i32 - begin;
            for i in 0..rows {
                map.from_buffer.push(0);
                map.buffer_row.push(0);
                map.own_row.push((start + i) as i32);
                map.land[(start + i) as usize] = begin + replay + i as i32;
            }
            if let Some(run) = &plan.scatter {
                for t in 0..run.count.min(rows) {
                    map.stash[(start + t) as usize] = locate(run, run.from + t)?;
                }
            }
        }
        map.rows_ext = map.from_buffer.len() as u32;
        Ok(map)
    }
}

/// One window's maps, landed as fire inputs.
#[derive(Clone, Copy, Debug)]
pub struct WindowInputs {
    pub rows_ext: u32,
    pub from_buffer: kernels_xla::Tensor,
    pub buffer_row: kernels_xla::Tensor,
    pub own_row: kernels_xla::Tensor,
    pub stash: kernels_xla::Tensor,
    pub land: kernels_xla::Tensor,
}

/// A fire's recurrent seat: each lane's plan and tables, the buffer planes,
/// and the maps of every window the walk may run the recurrence over.
#[derive(Debug)]
pub struct Seat {
    pub plans: Vec<LanePlan>,
    pub replay: kernels_xla::Tensor,
    pub commit: kernels_xla::Tensor,
    pub slots: kernels_xla::Tensor,
    pub layout: std::sync::Arc<Layout>,
    /// The buffer plane of each layout plane.
    pub buffers: Vec<kernels_xla::Tensor>,
    /// By window `(lane_offset, lanes, row_offset, rows)`.
    pub maps: HashMap<(u32, u32, u32, u32), WindowInputs>,
    /// Extended outputs landed by an earlier op of the current window.
    pub ext: std::cell::RefCell<HashMap<u32, kernels_xla::Tensor>>,
}
