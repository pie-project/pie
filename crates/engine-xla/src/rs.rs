//! Recurrent-state verbs: which rows of a hybrid model's recurrence
//! inputs are buffered per lane, replayed ahead of its own rows, and how
//! much of the extended run commits its state. The host half is the layout
//! the trace implies and each lane's [`LanePlan`]; the device half is `Seat`
//! below and `dispatch/rs.rs`.

use std::collections::HashMap;

use engine::rs::{LanePlan, Run};
use poem_ir::{Def, Dim, Dtype, Gather, Operation, Ssm, Trace, Ty, ValueId};

use crate::error::{Fault, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    ConvX,

    Ba,

    Ids,

    KdaF,

    KdaB,

    M2Dt,

    M2Z,

    /// A gated-delta row's raw keys (bf16, `k_heads · k_dim`): what the
    /// state + buffer step reads a carried row's key by.
    GdnK,

    /// A gated-delta row's corrected value `u` (f32, `v_heads · v_dim`):
    /// what the step reads a carried row's update by and folds it with.
    GdnU,
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

    /// A gated-delta recurrence's corrected planes (its `GdnK`, `GdnU`), by
    /// the value it lands.
    pub corrected: HashMap<u32, (usize, usize)>,

    /// Where the conv landing a gated-delta layer's `q k v` lays its rows'
    /// keys: the `GdnK` plane and the channel of the conv output they start
    /// at (after the query heads), by the conv's output.
    pub keys_of: HashMap<u32, (usize, u32)>,

    /// A gate prep's `[b | a]`, `dt_bias` and `a_log`, by the gates it
    /// lands: what a decode step forms its carried rows' log decays from.
    pub gate_prep: HashMap<u32, (ValueId, ValueId, ValueId)>,

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
        // the planes no trace value carries whole (a gated-delta row's keys
        // and corrected values), each once a cache
        let mut laid: HashMap<(u32, Role), usize> = HashMap::new();

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
                        qkv,
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
                        qkv,
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
                    // the step reads a carried row as a linear attention:
                    // its raw keys and corrected values ride the buffer
                    let mut at = [0usize; 2];
                    for (i, (role, width, dtype, bytes)) in [
                        (Role::GdnK, k_heads * k_dim, Dtype::Bf16, 2u64),
                        (Role::GdnU, v_heads * v_dim, Dtype::F32, 4u64),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        at[i] = *laid.entry((cache, role)).or_insert_with(|| {
                            layout.planes.push(Plane {
                                cache,
                                role,
                                width,
                                dtype,
                                row_bytes: u64::from(width) * bytes,
                            });
                            layout.planes.len() - 1
                        });
                    }
                    layout.corrected.insert(y.0, (at[0], at[1]));
                    layout.keys_of.insert(qkv.0, (at[0], k_heads * k_dim));
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
                Operation::Ssm(
                    Ssm::Mamba2 {
                        dt, z, state, y, ..
                    }
                    | Ssm::Mamba2Chunked {
                        dt, z, state, y, ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::M2Dt, *dt)?;
                    plane(&mut layout, cache, Role::M2Z, *z)?;
                    region(&mut layout, cache, "scan", *y)?;
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
            let Operation::Ssm(Ssm::GdnGates {
                ba,
                gates,
                dt_bias,
                a_log,
            }) = &node.op
            else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `ssm.gdn_gates` lands, which no delta scan reads",
                    gates.0
                ),
            })?;
            plane(&mut layout, cache, Role::Ba, *ba)?;
            layout.gate_prep.insert(gates.0, (*ba, *dt_bias, *a_log));
            region(&mut layout, cache, "gates", *gates)?;
        }
        if layout.planes.is_empty() {
            return Ok(None);
        }
        layout.token_bytes = layout.planes.iter().map(|plane| plane.row_bytes).sum();
        Ok(Some(layout))
    }
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
    /// The tail: each lane's extended rows past its commit, which it leaves
    /// carried, in a `[lanes, tail_rows]` grid of extended rows (0 past its
    /// own), its count, and for each own row its cell of the grid (`-1` for
    /// a row its commit folds). `tail_rows` 0: no lane leaves any.
    pub tail_rows: u32,
    pub tail: Vec<i32>,
    pub tail_count: Vec<i32>,
    pub own_tail: Vec<i32>,
}

/// Out of every buffer's range: a write there is dropped.
pub const DROP: i32 = i32::MAX;

/// The carried rows a fire of a `rung`-token rung replays at most, summed
/// over its lanes (`engine::fire::rung_rows`): its rung's tokens. The one
/// place the bound is read: a fire past it is refused at prepare, and a
/// fire that replays lays each window's scratch for it.
#[must_use]
pub fn carried_room(rung: u32) -> u32 {
    rung
}

/// A decode fire's carried rows as the state + buffer step reads them
/// (`kernels_xla::ssm::carried`): each lane's buffered rows' rs list rows in
/// a `[lanes, rows]` grid (`DROP` past its own), how many it carries, and
/// its commit (the carried and own rows its fold takes, `LanePlan::commit`,
/// the step's `0..=count + 1`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CarriedMap {
    pub rows: u32,
    pub buffer_rows: Vec<i32>,
    pub count: Vec<i32>,
    pub commit: Vec<i32>,
}

impl CarriedMap {
    /// The map of the window over lanes `lane0..lane0 + lanes`, `rows` the
    /// carried rows a lane may hold (the step's grid); a lane carrying more
    /// is refused by name.
    pub fn of(
        plans: &[LanePlan],
        lanes: u32,
        lane0: u32,
        page_tokens: u32,
        rows: u32,
    ) -> Result<CarriedMap> {
        let mut map = CarriedMap {
            rows,
            buffer_rows: vec![DROP; (lanes * rows) as usize],
            count: vec![0; lanes as usize],
            commit: vec![0; lanes as usize],
        };
        for r in 0..lanes {
            let Some(plan) = plans.get((lane0 + r) as usize) else {
                continue;
            };
            let mut at = 0u32;
            for run in &plan.gathers {
                for t in 0..run.count {
                    if at >= rows {
                        return Err(Fault::Ceiling {
                            what: "carried rows a decode lane's step reads",
                            need: u64::from(plan.gathers.iter().map(|g| g.count).sum::<u32>()),
                            have: u64::from(rows),
                        });
                    }
                    let token = run.from + t;
                    let page = token / page_tokens;
                    let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
                        what: "recurrent buffer pages",
                        need: u64::from(page) + 1,
                        have: run.pages.len() as u64,
                    })?;
                    let row =
                        u64::from(slot) * u64::from(page_tokens) + u64::from(token % page_tokens);
                    map.buffer_rows[(r * rows + at) as usize] = i32::try_from(row)
                        .ok()
                        .filter(|&v| v != DROP)
                        .ok_or(Fault::Ceiling {
                            what: "rs list rows an i32 addresses",
                            need: row,
                            have: i32::MAX as u64 - 1,
                        })?;
                    at += 1;
                }
            }
            map.count[r as usize] = at as i32;
            map.commit[r as usize] = i32::try_from(plan.commit).unwrap_or(i32::MAX);
        }
        Ok(map)
    }
}

impl WindowMap {
    /// The map of the window covering lanes `lane0..lane0 + lanes` whose own
    /// rows are cut by `indptr` (window-local), for rs list slots of
    /// `page_tokens` rows. A buffer row is `q · page_tokens + token` of the
    /// slot's grain index `q` (each plane maps `q` to its own block before
    /// it reads or writes, `dispatch/rs.rs`).
    ///
    /// `rung`: the fire's rung where it replays carried rows: the window's
    /// scratch holds the rung's own rows and its carried room
    /// ([`carried_room`]), one size at every such fire of the rung. `None`:
    /// the fire replays nothing, the scratch holds the window's own rows.
    pub fn of(
        plans: &[LanePlan],
        indptr: &[i32],
        lane0: u32,
        page_tokens: u32,
        rung: Option<u32>,
        tail_width: u32,
    ) -> Result<WindowMap> {
        let mut map = WindowMap::default();
        let rows_own = indptr.last().copied().unwrap_or(0).max(0) as usize;
        map.stash = vec![DROP; rows_own];
        map.land = vec![0; rows_own];
        // each lane's (first extended row, commit, first own row, own rows)
        let mut tails: Vec<(i32, i32, u32, u32)> = Vec::new();
        let locate = |run: &Run, token: u32| -> Result<i32> {
            let page = token / page_tokens;
            let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
                what: "recurrent buffer pages",
                need: u64::from(page) + 1,
                have: run.pages.len() as u64,
            })?;
            let row = u64::from(slot) * u64::from(page_tokens) + u64::from(token % page_tokens);
            i32::try_from(row)
                .ok()
                .filter(|&r| r != DROP)
                .ok_or(Fault::Ceiling {
                    what: "rs list rows an i32 addresses",
                    need: row,
                    have: i32::MAX as u64 - 1,
                })
        };
        for r in 0..indptr.len().saturating_sub(1) {
            let start = indptr[r].max(0) as u32;
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let Some(plan) = plans.get(lane0 as usize + r) else {
                continue;
            };
            let begin = map.from_buffer.len() as i32;
            tails.push((begin, plan.commit as i32, start, rows));
            // a single-token lane's carried rows are its decode step's input
            // (`CarriedMap`), none replayed (`engine::fire::lane_carried`)
            let gathers: &[Run] = if rows == 1 { &[] } else { &plan.gathers };
            for run in gathers {
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
        let laid = map.from_buffer.len();
        let ext = match rung {
            Some(rung) => {
                let room = rung as usize + carried_room(rung) as usize;
                if laid > room {
                    return Err(Fault::Ceiling {
                        what: "carried rows a fire's lanes replay (the tokens of the rung it runs at)",
                        need: laid as u64,
                        have: room as u64,
                    });
                }
                room
            }
            None => laid,
        };
        let ext = ext.max(16).next_power_of_two();
        map.from_buffer.resize(ext, 0);
        map.buffer_row.resize(ext, 0);
        map.own_row.resize(ext, 0);
        map.rows_ext = ext as u32;
        // a lane's extended rows run from `begin` to the next lane's (or
        // the laid end): its tail is the part past its commit
        let ends: Vec<i32> = tails
            .iter()
            .skip(1)
            .map(|t| t.0)
            .chain([laid as i32])
            .collect();
        let counts: Vec<i32> = tails
            .iter()
            .zip(&ends)
            .map(|(&(begin, commit, _, _), &end)| (end - begin - commit).max(0))
            .collect();
        let widest = counts.iter().copied().max().unwrap_or(0) as u32;
        if widest > tail_width {
            return Err(Fault::Ceiling {
                what: "rows a lane leaves past its commit (the run-ahead window)",
                need: u64::from(widest),
                have: u64::from(tail_width),
            });
        }
        if tail_width > 0 {
            // the run-ahead window wide, whatever the lanes leave: one
            // program per rung
            let t = tail_width;
            map.tail_rows = t;
            map.tail = vec![0; tails.len() * t as usize];
            map.own_tail = vec![-1; rows_own];
            for (r, (&(begin, commit, start, rows), &n)) in tails.iter().zip(&counts).enumerate() {
                for j in 0..n {
                    map.tail[r * t as usize + j as usize] = begin + commit + j;
                }
                // own row i sits at extended row begin + (span - rows) + i
                let span = n + commit;
                for i in 0..rows as i32 {
                    let at = span - rows as i32 + i - commit;
                    if at >= 0 {
                        map.own_tail[(start as i32 + i) as usize] = r as i32 * t as i32 + at;
                    }
                }
            }
            map.tail_count = counts;
        }
        Ok(map)
    }
}

/// One window's maps, landed as fire inputs.
#[derive(Clone, Copy, Debug)]
pub struct WindowInputs {
    pub rows_ext: u32,
    /// The window's tail (`WindowMap::tail`), where any lane leaves one.
    pub tail: Option<TailInputs>,
    pub from_buffer: crate::handle::Tensor,
    pub buffer_row: crate::handle::Tensor,
    pub own_row: crate::handle::Tensor,
    pub stash: crate::handle::Tensor,
    pub land: crate::handle::Tensor,
}

/// A window's tail staged (`WindowMap::tail`): the `[lanes, rows]` grid of
/// extended rows, each lane's count, and each own row's cell (`-1` none).
#[derive(Clone, Copy, Debug)]
pub struct TailInputs {
    pub rows: u32,
    pub grid: crate::handle::Tensor,
    pub count: crate::handle::Tensor,
    pub own: crate::handle::Tensor,
}

/// A [`CarriedMap`] staged: each lane's carried rows' rs list rows
/// (`[lanes, rows]`, 0 past its own: the step reads none of them), count
/// and commit.
#[derive(Clone, Copy, Debug)]
pub struct CarriedInputs {
    pub rows: u32,
    pub buffer_rows: crate::handle::Tensor,
    pub count: crate::handle::Tensor,
    pub commit: crate::handle::Tensor,
}

/// A fire's recurrent seat: each lane's plan and tables, the buffer planes,
/// and the maps of every window the walk may run the recurrence over.
#[derive(Debug)]
pub struct Seat {
    pub plans: Vec<LanePlan>,
    pub replay: crate::handle::Tensor,
    pub commit: crate::handle::Tensor,
    /// Each lane's slot (grain index) of each record piece
    /// (`store::Pools::records`).
    pub slots: Vec<crate::handle::Tensor>,
    pub layout: std::sync::Arc<Layout>,
    /// The rs list plane of each layout plane, and its map.
    pub buffers: Vec<(crate::handle::Tensor, crate::store::Map)>,
    /// The rows a slot of the rs list holds.
    pub page_tokens: u32,
    /// By window `(lane_offset, lanes, row_offset, rows)`.
    pub maps: HashMap<(u32, u32, u32, u32), WindowInputs>,
    /// By window, as `maps`: a window of single-token lanes' carried rows,
    /// the state + buffer step's input.
    pub carried: HashMap<(u32, u32, u32, u32), CarriedInputs>,
    /// Extended outputs landed by an earlier op of the current window.
    pub ext: std::cell::RefCell<HashMap<u32, crate::handle::Tensor>>,
}

#[cfg(test)]
mod carried_map {
    use super::*;

    #[test]
    fn a_decode_lanes_carried_rows_land_in_its_row_of_the_grid() {
        let run = |pages: Vec<u32>, from: u32, count: u32| Run { pages, from, count };
        let plans = vec![
            LanePlan {
                commit: 0,
                ..LanePlan::default()
            },
            LanePlan {
                gathers: vec![run(vec![7, 9], 14, 3)],
                commit: 4,
                ..LanePlan::default()
            },
        ];
        let map = CarriedMap::of(&plans, 2, 0, 16, 4).unwrap();
        assert_eq!(map.count, vec![0, 3]);
        assert_eq!(map.commit, vec![0, 4]);
        // tokens 14, 15 on page 7, 16 on page 9
        assert_eq!(
            &map.buffer_rows[4..],
            &[7 * 16 + 14, 7 * 16 + 15, 9 * 16, DROP]
        );
        assert!(map.buffer_rows[..4].iter().all(|&r| r == DROP));
        // a lane carrying past the grid is refused
        assert!(CarriedMap::of(&plans, 2, 0, 16, 2).is_err());
    }
}

#[cfg(test)]
mod tail_map {
    use super::*;

    #[test]
    fn a_lanes_rows_past_its_commit_are_its_tail() {
        let run = |from: u32, count: u32| Run {
            pages: vec![0, 1],
            from,
            count,
        };
        // lane 0 replays 2 carried rows then 3 own rows and commits 3 of the
        // 5; lane 1 commits all of its 2 own rows
        let plans = vec![
            LanePlan {
                replay: 2,
                commit: 3,
                gathers: vec![run(4, 2)],
                scatter: Some(run(6, 3)),
                ..LanePlan::default()
            },
            LanePlan {
                commit: 2,
                ..LanePlan::default()
            },
        ];
        let map = WindowMap::of(&plans, &[0, 3, 5], 0, 16, None, 3).unwrap();
        assert_eq!(map.tail_count, vec![2, 0]);
        assert_eq!(map.tail_rows, 3);
        assert_eq!(map.tail, vec![3, 4, 0, 0, 0, 0]);
        assert_eq!(map.own_tail, vec![-1, 0, 1, -1, -1]);
        // past the window, refused
        assert!(WindowMap::of(&plans, &[0, 3, 5], 0, 16, None, 1).is_err());
    }
}
