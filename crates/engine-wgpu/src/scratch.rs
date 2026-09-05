use kernels_wgpu::Tensor;
use kernels_wgpu::attn::ple;
use kernels_wgpu::linear::moe::{self, RoutedScratch};
use model_compiler::{Budget, Budgets, CompiledModel, Fallback, FireRows};
use model_exec::store::arena::rect;
use model_ir::{Attention, Def, Dim, Dtype, Linear, Operation, Trace, Ty, ValueId};

use crate::store::kv::Paging;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;
use crate::run::{PoolSlabs, WeightRow, WeightTable};

const ALIGN: u64 = 256;

#[derive(Clone, Copy, Debug)]
struct SdpaSplit {
    o: Room,
    lse: Room,

    keys: u32,
}

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

#[derive(Clone, Copy, Debug)]
struct Pool {
    space: u32,
    state_kv: Room,
    state_score: Room,
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
}

#[derive(Debug)]
pub struct Scratch {
    plane: Buffer,

    precast: Option<Room>,
    routed: Option<Routed>,

    index: Option<Room>,

    split: Option<SdpaSplit>,

    copy: Option<CopyRoom>,

    pool: Vec<Pool>,

    ple: Vec<PleHash>,

    capacity: Vec<u32>,

    routers: Vec<u32>,
}

impl Scratch {
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        weights: &WeightTable,
        compiled: &CompiledModel,
        budgets: &Budgets,
        paging: Paging,
    ) -> Result<Scratch> {
        let budget = &budgets.tokens;
        let map = &compiled.arena;
        let ceiling = FireRows {
            tokens: u64::from(budget.max_tokens),
            lanes: u64::from(budget.max_lanes),
            patches: u64::from(budgets.max_patches()),
            images: u64::from(budgets.max_images()),
        };
        let of = |id: ValueId| rect(map, id, ceiling);
        let banked = |id: ValueId| match trace.values.get(id.0 as usize).map(|v| &v.def) {
            Some(Def::Weight(w)) => matches!(
                weights.0.get(*w as usize).copied().flatten(),
                Some(WeightRow::Planes(_))
            ),
            _ => false,
        };

        let routers = routers(trace);
        let tuning = kernels_wgpu::tuning::current();

        let mut dense: Vec<(u32, u32)> = Vec::new();
        let (mut act_rows, mut act_k) = (0u64, 0u64);
        let (mut sorted, mut pairs, mut routed_k, mut routed_n) = (0u64, 0u64, 0u64, 0u64);
        for node in &trace.nodes {
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            match op {
                Linear::Matmul { act, w, y } | Linear::LmHead { act, w, y } if banked(*w) => {
                    let (Some(act), Some(y)) = (of(*act), of(*y)) else {
                        continue;
                    };
                    act_rows = act_rows.max(u64::from(act.rows));
                    act_k = act_k.max(u64::from(act.width));
                    let pair = (y.width, act.width);
                    if !dense.contains(&pair) {
                        dense.push(pair);
                    }
                }
                Linear::MoeMatmulSelectBias { x, routes, y, .. }
                | Linear::MoeMatmulSelectQuant { x, routes, y, .. } => {
                    let (Some(x), Some(y)) = (of(*x), of(*y)) else {
                        continue;
                    };
                    let experts = routers.get(routes.0 as usize).copied().unwrap_or(0);
                    if experts == 0 {
                        continue;
                    }

                    if moe::tile_rows(y.rows, experts, &tuning) <= 1 {
                        continue;
                    }
                    let rows = moe::sorted_rows(y.rows, experts, &tuning);
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
        let routed = (sorted > 0).then(|| {
            let mut at = 0u64;
            Routed {
                perm: Room::lay(&mut at, 1, sorted, Dtype::I32),
                row_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),

                tile_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                inv: Room::lay(&mut at, 1, pairs, Dtype::I32),
                x: Room::lay(&mut at, sorted, routed_k, Dtype::Bf16),
                y: Room::lay(&mut at, sorted, routed_n, Dtype::Bf16),
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

        let split_max = u64::from(tuning.sdpa_split_max.max(1));
        let split = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attention(
                    Attention::Decode { o, head_dim, .. }
                    | Attention::DecodeLse { o, head_dim, .. },
                ) => of(*o).map(|rect| (rect, *head_dim)),
                _ => None,
            })
            .filter(|(rect, head_dim)| *head_dim > 0 && rect.rows > 0 && rect.width > 0)
            .map(|(rect, head_dim)| (u64::from(rect.rows), u64::from(rect.width), head_dim))
            .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1), a.2.min(b.2)))
            .map(|(rows, width, head_dim)| {
                let mut at = 0u64;

                let heads = width / u64::from(head_dim).max(1);
                SdpaSplit {
                    o: Room::lay(&mut at, split_max * rows, width, Dtype::Bf16),
                    lse: Room::lay(&mut at, split_max * rows, heads.max(1), Dtype::F32),
                    keys: paging.context(),
                }
            });

        let union = precast
            .map_or(0, |r| r.at + r.bytes())
            .max(routed.map_or(0, |r| r.y.at + r.y.bytes()))
            .max(index.map_or(0, |r| r.at + r.bytes()))
            .max(split.map_or(0, |r| r.lse.at + r.lse.bytes()));

        let copy = copy_ceiling(trace, compiled, budget).map(|bytes| CopyRoom {
            at: union.next_multiple_of(ALIGN),
            bytes,
        });

        let mut at = copy.map_or(union, |room| room.at + room.bytes);

        let pool: Vec<Pool> = pool_state(trace, paging)
            .into_iter()
            .map(|(space, cells, width)| Pool {
                space,
                state_kv: Room::lay(&mut at, cells, width, Dtype::Bf16),
                state_score: Room::lay(&mut at, cells, width, Dtype::Bf16),
            })
            .collect();

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
            routed,
            index,
            split,
            copy,
            pool,
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
            })
        };
        Some(mint())
    }

    pub fn index_scores(&self, handles: &Handles) -> Option<Result<Tensor>> {
        Some(self.index?.bind(handles, &self.plane))
    }

    pub fn sdpa_split(&self, handles: &Handles) -> Option<Result<kernels_wgpu::attn::Split>> {
        let rooms = self.split?;
        let mint = || {
            Ok(kernels_wgpu::attn::Split {
                o: rooms.o.bind(handles, &self.plane)?,
                lse: rooms.lse.bind(handles, &self.plane)?,
                keys: rooms.keys,
            })
        };
        Some(mint())
    }

    pub fn pool_state(&self, handles: &Handles, space: u32) -> Option<Result<PoolSlabs>> {
        let rooms = *self.pool.iter().find(|held| held.space == space)?;
        let mint = || {
            Ok(PoolSlabs {
                state_kv: rooms.state_kv.bind(handles, &self.plane)?,
                state_score: rooms.state_score.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
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

const fn compressor_coff(ratio: u32) -> u64 {
    if ratio == 4 { 2 } else { 1 }
}

fn pool_state(trace: &Trace, paging: Paging) -> Vec<(u32, u64, u64)> {
    let cells = paging.pages() * u64::from(paging.page_size);
    if cells == 0 {
        return Vec::new();
    }
    let mut spaces: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let Operation::Attention(Attention::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(Def::Cache(space)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = compressor_coff(*ratio) * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match spaces.iter_mut().find(|(held, _)| *held == *space) {
            Some((_, held)) => *held = (*held).max(width),
            None => spaces.push((*space, width)),
        }
    }
    spaces
        .into_iter()
        .map(|(space, width)| (space, cells, width))
        .collect()
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
