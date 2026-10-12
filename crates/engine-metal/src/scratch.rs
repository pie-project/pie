use kernels_metal::Tensor;
use kernels_metal::attn::ple;
use kernels_metal::linear::moe::{self, RoutedScratch};
use kernels_metal::linear::quant;
use poem_compiler::{Budget, Budgets, CompiledModel, Fallback, FireRows};
use poem_exec::store::arena::rect;
use poem_ir::{Attention, Def, Dim, Dtype, Linear, Operation, Trace, Ty, ValueId};

use crate::store::kv::Paging;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;
use crate::run::{WeightRow, WeightTable};

const ALIGN: u64 = 256;

/// The rows a split-K matmul tiles at.
const SPLITK_ROWS: u64 = 8;

#[derive(Clone, Copy, Debug)]
struct Room {
    at: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

impl Room {
    fn lay(at: &mut u64, rows: u64, width: u64, dtype: Dtype) -> Room {
        let start = at.next_multiple_of(ALIGN);
        let room = Room {
            at: start,
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype,
        };
        *at = start + room.bytes();
        room
    }

    fn bytes(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.width) * self.dtype.bytes_ceil()
    }

    fn bind(&self, handles: &Handles, plane: &Buffer) -> Result<Tensor> {
        Ok(Tensor::new(
            handles.bind(plane, self.at, self.bytes())?,
            self.rows,
            self.width,
            self.dtype,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct CopyRoom {
    at: u64,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct PleHash {
    key: Box<[u64]>,
    room: Room,
}

#[derive(Clone, Copy, Debug)]
struct Routed {
    perm: Room,
    row_expert: Room,
    tile_expert: Room,
    inv: Room,
    x: Room,
    y: Room,
    sums: Room,
}

#[derive(Debug)]
pub struct Scratch {
    plane: Buffer,

    precast: Option<Room>,
    partials: Option<Room>,
    routed: Option<Routed>,
    index: Option<Room>,
    copy: Option<CopyRoom>,

    spatial: Option<Room>,

    ple: Vec<PleHash>,

    capacity: Vec<u32>,

    routers: Vec<u32>,
}

/// The partials of a split matmul: up to `slices` split-K slices of
/// [`SPLITK_ROWS`] rows, or for MPP four partitions of up to `mpp_rows`
/// padded rows.
fn split_partial_room(
    precast: Option<Room>,
    width: u64,
    mpp: bool,
    slices: u64,
    mpp_rows: u64,
) -> Room {
    let mut at = if mpp {
        precast.map_or(0, |room| room.at + room.bytes())
    } else {
        0
    };
    Room::lay(
        &mut at,
        if mpp {
            (4 * mpp_rows).max(SPLITK_ROWS * slices)
        } else {
            SPLITK_ROWS * slices
        },
        width,
        Dtype::F32,
    )
}

impl Scratch {
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        weights: &WeightTable,
        compiled: &CompiledModel,
        budgets: &Budgets,
        paging: Paging,
        tuning: &kernels_metal::Tuning,
    ) -> Result<Scratch> {
        let budget = &budgets.tokens;
        let map = &compiled.arena;
        let ceiling = FireRows::ceilings(budgets);
        let of = |id: ValueId| rect(map, id, ceiling);
        let banked = |id: ValueId| match trace.values.get(id.0 as usize).map(|v| &v.def) {
            Some(Def::Weight(w)) => matches!(
                weights.0.get(*w as usize).copied().flatten(),
                Some(WeightRow::Planes(_))
            ),
            _ => false,
        };

        let routers = routers(trace);

        let mut dense: Vec<(u32, u32)> = Vec::new();
        let (mut act_rows, mut act_k) = (0u64, 0u64);
        let (mut sorted, mut pairs, mut routed_k, mut routed_n) = (0u64, 0u64, 0u64, 0u64);
        for node in &trace.nodes {
            for (act, w, y) in crate::legs::matmuls(&node.op) {
                if !banked(w) {
                    continue;
                }
                let (Some(act), Some(y)) = (of(act), of(y)) else {
                    continue;
                };
                act_rows = act_rows.max(u64::from(act.rows));
                act_k = act_k.max(u64::from(act.width));
                let pair = (y.width, act.width);
                if !dense.contains(&pair) {
                    dense.push(pair);
                }
            }
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            match op {
                Linear::MoeMatmulSelectBias { x, routes, y, .. }
                | Linear::MoeMatmulSelectQuant { x, routes, y, .. } => {
                    let (Some(x), Some(y)) = (of(*x), of(*y)) else {
                        continue;
                    };
                    let experts = routers.get(routes.0 as usize).copied().unwrap_or(0);
                    if experts == 0 {
                        continue;
                    }
                    if moe::tile_rows(y.rows, experts, tuning) <= 1 {
                        continue;
                    }
                    let rows = moe::sorted_rows(y.rows, experts, tuning);
                    sorted = sorted.max(u64::from(rows));
                    pairs = pairs.max(u64::from(y.rows));
                    routed_k = routed_k.max(u64::from(x.width));
                    routed_n = routed_n.max(u64::from(y.width));
                }
                _ => {}
            }
        }

        let precast = (act_rows > 0 && act_k > 0).then(|| {
            let mut at = 0u64;
            Room::lay(&mut at, act_rows, act_k, Dtype::F16)
        });
        let partials = dense
            .iter()
            .filter(|&&(n, k)| {
                let (Ok(n), Ok(k)) = (i32::try_from(n), i32::try_from(k)) else {
                    return false;
                };
                let limit = tuning.qmm_splitk();
                quant::splitk(n, 8, 8, k, 32, 4, limit) > 1 || (tuning.qmm_mpp() && n <= 65536)
            })
            .map(|&(n, _)| u64::from(n))
            .max()
            .map(|n| {
                // MPP splits up to twice `split_rows` rows, padded to 32.
                let kernels_metal::tuning::QmmMpp { split_rows, .. } = tuning.qmm_mpp_tiles();
                let mpp_rows = u64::from(2 * split_rows).div_ceil(32) * 32;
                let slices = u64::from(tuning.qmm_splitk_max());
                split_partial_room(precast, n, tuning.qmm_mpp(), slices, mpp_rows)
            });

        let routed = (sorted > 0).then(|| {
            let mut at = 0u64;
            Routed {
                perm: Room::lay(&mut at, 1, sorted, Dtype::I32),
                row_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                tile_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                inv: Room::lay(&mut at, 1, pairs, Dtype::I32),
                x: Room::lay(&mut at, sorted, routed_k, Dtype::Bf16),
                y: Room::lay(&mut at, sorted, routed_n, Dtype::Bf16),
                sums: Room::lay(&mut at, sorted, routed_k.div_ceil(64), Dtype::F32),
            }
        });

        let index = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attention(Attention::IndexTopk { selection, .. }) => of(*selection),
                _ => None,
            })
            .map(|rect| u64::from(rect.rows))
            .max()
            .filter(|rows| *rows > 0)
            .and_then(|rows| {
                let keys = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
                (keys > 0).then(|| {
                    let mut at = 0u64;
                    Room::lay(&mut at, rows, keys, Dtype::F32)
                })
            });

        let spatial = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Spatial(poem_ir::Spatial::GroupNorm { groups, .. }) => Some(*groups),
                _ => None,
            })
            .max()
            .filter(|groups| *groups > 0)
            .map(|groups| {
                let clips = budgets.max_clips().max(1);
                let floats = kernels_metal::spatial::norm::scratch_floats(clips, groups);
                let mut at = 0u64;
                Room::lay(&mut at, floats.div_ceil(4).max(1), 4, Dtype::F32)
            });

        let union = precast
            .map_or(0, |r| r.at + r.bytes())
            .max(partials.map_or(0, |r| r.at + r.bytes()))
            .max(routed.map_or(0, |r| r.sums.at + r.sums.bytes()))
            .max(index.map_or(0, |r| r.at + r.bytes()))
            .max(spatial.map_or(0, |r| r.at + r.bytes()));
        let copy = copy_ceiling(trace, compiled, budget).map(|bytes| CopyRoom {
            at: union.next_multiple_of(ALIGN),
            bytes,
        });

        let mut at = copy.map_or(union, |room| room.at + room.bytes);

        let mut ple: Vec<PleHash> = Vec::new();
        for node in &trace.nodes {
            let Operation::Attention(
                Attention::PleNgramIds {
                    mults,
                    primes,
                    offsets,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    mults,
                    primes,
                    offsets,
                    ..
                },
            ) = &node.op
            else {
                continue;
            };
            let key: Box<[u64]> = ple::hash_constants(mults, primes, offsets).into();
            if key.is_empty() || ple.iter().any(|held| held.key == key) {
                continue;
            }
            let room = Room::lay(&mut at, 1, key.len() as u64, Dtype::U64);
            ple.push(PleHash { key, room });
        }

        let mut plane = Buffer::zeroed(device, at)?;
        for hashing in &ple {
            let bytes: Vec<u8> = hashing.key.iter().flat_map(|v| v.to_ne_bytes()).collect();
            plane.write(hashing.room.at, &bytes)?;
        }

        Ok(Scratch {
            plane,
            precast,
            partials,
            routed,
            index,
            copy,
            spatial,
            ple,
            capacity: crate::arena::capacities(map),
            routers,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.plane.bytes()
    }

    #[must_use]
    pub fn capacity(&self, id: ValueId) -> u32 {
        self.capacity.get(id.0 as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn experts(&self, routes: ValueId) -> u32 {
        self.routers.get(routes.0 as usize).copied().unwrap_or(0)
    }

    pub fn precast(
        &self,
        handles: &Handles,
        rows: u32,
        contraction: u32,
    ) -> Option<Result<Tensor>> {
        let room = self.precast?;
        if u64::from(rows) * u64::from(contraction) > room.bytes() / 2 {
            return None;
        }
        Some(
            Room {
                rows,
                width: contraction,
                ..room
            }
            .bind(handles, &self.plane),
        )
    }

    pub fn spatial_moments(
        &self,
        handles: &Handles,
        clips: u32,
        groups: u32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        let room = self.spatial?;
        let splits = kernels_metal::spatial::norm::SPLITS;
        let partial_floats = u64::from(clips) * u64::from(splits) * u64::from(groups) * 4;
        let stat_floats = u64::from(clips) * u64::from(groups) * 2;
        if partial_floats + stat_floats > u64::from(room.rows) * u64::from(room.width) {
            return None;
        }
        let partials = Room {
            at: room.at,
            rows: u32::try_from(u64::from(clips) * u64::from(splits)).unwrap_or(u32::MAX),
            width: groups * 4,
            dtype: Dtype::F32,
        };
        let stats = Room {
            at: room.at + partials.bytes(),
            rows: clips,
            width: groups * 2,
            dtype: Dtype::F32,
        };
        Some((|| {
            Ok((
                partials.bind(handles, &self.plane)?,
                stats.bind(handles, &self.plane)?,
            ))
        })())
    }

    pub fn partials(&self, handles: &Handles, rows: u32, width: u32) -> Option<Result<Tensor>> {
        let room = self.partials?;
        if u64::from(rows) * u64::from(width) > room.bytes() / 4 {
            return None;
        }
        Some(
            Room {
                rows,
                width,
                ..room
            }
            .bind(handles, &self.plane),
        )
    }

    pub fn copy(&self, handles: &Handles, offset: u64, bytes: u64) -> Option<Result<u32>> {
        let room = self.copy?;
        if offset.saturating_add(bytes) > room.bytes {
            return None;
        }
        Some(handles.bind(&self.plane, room.at + offset, bytes))
    }

    pub fn routed(&self, handles: &Handles) -> Option<Result<RoutedScratch>> {
        let r = self.routed?;
        let mint = || {
            Ok(RoutedScratch {
                perm: r.perm.bind(handles, &self.plane)?,
                row_expert: r.row_expert.bind(handles, &self.plane)?,
                tile_expert: r.tile_expert.bind(handles, &self.plane)?,
                inv: r.inv.bind(handles, &self.plane)?,
                x: r.x.bind(handles, &self.plane)?,
                y: r.y.bind(handles, &self.plane)?,
                sums: r.sums.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
    }

    pub fn index_scores(&self, handles: &Handles) -> Option<Result<Tensor>> {
        Some(self.index?.bind(handles, &self.plane))
    }

    pub fn ple_hash(
        &self,
        handles: &Handles,
        mults: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Option<Result<Tensor>> {
        let want = ple::hash_constants(mults, primes, offsets);
        let held = self.ple.iter().find(|h| *h.key == *want)?;
        Some(held.room.bind(handles, &self.plane))
    }
}

fn copy_ceiling(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> Option<u64> {
    let lanes = u64::from(budget.max_lanes);
    let mut most = 0u64;
    for region in compiled.template() {
        let rows = compiled
            .fallback
            .rows
            .iter()
            .filter(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
            .flat_map(|row| row.buckets.clone())
            .filter_map(|bucket| budget.buckets.get(bucket as usize).copied())
            .max()
            .or_else(|| {
                compiled
                    .fallback
                    .rows
                    .iter()
                    .any(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
                    .then_some(budget.max_tokens)
            });
        let Some(rows) = rows else { continue };
        if !crate::window::copyable_mask(trace, compiled, &region.mask) {
            continue;
        }
        let Some((ins, outs)) = crate::window::operands(&trace.nodes, region) else {
            continue;
        };
        let mut seen: Vec<u64> = Vec::new();
        let mut at = 0u64;
        for id in ins.iter().chain(outs.iter()) {
            let Some(decl) = trace.values.get(id.0 as usize) else {
                continue;
            };
            if !matches!(
                &decl.ty,
                Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Tokens))
            ) {
                continue;
            }
            let Some(rect) = rect(
                &compiled.arena,
                *id,
                FireRows::text_only(u64::from(rows), lanes),
            ) else {
                continue;
            };
            if seen.contains(&rect.offset) {
                continue;
            }
            seen.push(rect.offset);
            let row = u64::from(rect.width) * rect.dtype.bytes_ceil();
            at = at.next_multiple_of(COPY_ALIGN) + u64::from(rows) * row;
        }
        most = most.max(at);
    }
    (most > 0).then_some(most)
}

const COPY_ALIGN: u64 = 16;

fn routers(trace: &Trace) -> Vec<u32> {
    let mut out = vec![0u32; trace.values.len()];
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, experts) = match op {
            Linear::MoeTopkSoftmax {
                routes, experts, ..
            }
            | Linear::MoeTopkSoftmaxScaled {
                routes, experts, ..
            }
            | Linear::MoeTopkSigmoid {
                routes, experts, ..
            }
            | Linear::MoeTopkSqrtSoftplus {
                routes, experts, ..
            }
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if let Some(seat) = out.get_mut(routes.0 as usize) {
            *seat = (*seat).max(experts);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpp_partials_do_not_overwrite_live_input_sums() {
        for rows in [8, 32, 64, 2048] {
            for width in [512, 5120, 17408] {
                let precast = Room::lay(&mut 0, rows, width, Dtype::F16);
                let partial = split_partial_room(Some(precast), 34816, true, 8, 64);
                assert!(partial.at >= precast.at + precast.bytes());
                assert_eq!(partial.at % ALIGN, 0);
                assert!(partial.rows >= 64 * 4);
            }
        }
    }
}
