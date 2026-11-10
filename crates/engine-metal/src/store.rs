#![allow(unsafe_code)]
pub mod accounting;
pub mod kv;
pub mod layout;
pub mod pool;
mod pools;

pub use pools::{Pools, Rows, RsRows, TablePaging};

use kernels_metal::{StateMap, Tensor};
use poem_ir::{Attn, Def, Dtype, Operation, Trace};

use crate::error::{Fault, Result};
use crate::store::kv::Paging;

impl From<poem_exec::store::Fault> for Fault {
    fn from(fault: poem_exec::store::Fault) -> Fault {
        match fault {
            poem_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            poem_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            poem_exec::store::Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            } => Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            },
        }
    }
}

const STATE_DTYPE: Dtype = Dtype::F32;

fn state_dtype(declared: Dtype) -> Dtype {
    match declared {
        Dtype::I32 => Dtype::I32,
        _ => STATE_DTYPE,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TableSeat {
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub row_valid: Tensor,
}

#[derive(Debug, Clone)]
pub struct Seats {
    pub lanes: u32,
    pub rows: u32,
    pub pages: u32,
    pub page_tables: Vec<TableSeat>,
    pub slot_ids: Tensor,
    pub slot_of_row: Tensor,
    /// Each band's lane and row tables (`Pools::bands`).
    pub lane_bands: Vec<Tensor>,
    pub row_bands: Vec<Tensor>,
}

/// Bytes of one state row's segment table: its head and one segment.
const STATE_MAP_BYTES: u64 = ((StateMap::HEAD + 1) * 8) as u64;

#[derive(Debug)]
struct Spill {
    file: engine::disk::SlotFile,
    stage: engine::disk::Aligned,
}

impl Spill {
    fn open(
        dir: &std::path::Path,
        name: &str,
        slot_bytes: u64,
        budget: u64,
    ) -> Result<Option<Spill>> {
        let file =
            engine::disk::SlotFile::open(dir, name, slot_bytes, budget).map_err(slot_file)?;
        Ok(file.map(|file| Spill {
            stage: engine::disk::Aligned::zeroed(file.stride()),
            file,
        }))
    }

    fn slots(spill: Option<&Spill>) -> u32 {
        spill.map_or(0, |spill| spill.file.slots())
    }

    fn room(spill: Option<&Spill>, what: &'static str, slots: &[u32]) -> Result<()> {
        let have = Spill::slots(spill);
        match slots.iter().find(|&&slot| slot >= have) {
            Some(&slot) => Err(Fault::Ceiling {
                what,
                need: u64::from(slot) + 1,
                have: u64::from(have),
            }),
            None => Ok(()),
        }
    }
}

/// What a load's held state takes, unpacked: a part per kv page (every kv
/// row's tokens and the compressor's), and a part per state slot the page
/// count leaves alone. An estimate before the pool is laid: the pool rounds
/// each run inside a slot, and whole chunks, up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolDemand {
    pub per_page: u64,
    pub fixed: u64,
}

impl PoolDemand {
    /// `trace`'s held state at `paging`'s page size and slots, beside
    /// `rs`'s recurrent-state buffers; `paging`'s page count is not read.
    pub fn of(trace: &Trace, paging: Paging, rs: Option<&crate::rs::Layout>) -> Result<PoolDemand> {
        let (token, state, side) = Rows::estimate(trace, paging.page_size)?;
        let buffers = rs.map_or(0, |layout| crate::rs::Buffers::demand(layout, paging));
        Ok(PoolDemand {
            per_page: token * u64::from(paging.page_size) + side,
            fixed: buffers + state * u64::from(paging.slots),
        })
    }

    #[must_use]
    pub fn bytes(self, pages: u64) -> u64 {
        self.fixed
            .saturating_add(self.per_page.saturating_mul(pages))
    }
}

/// Each kv row a pool gather compresses, with the widest state row its
/// gathers keep per cell (`ratio` 4 overlaps two windows, so twice the head).
fn compressed_rows(trace: &Trace) -> Vec<(u32, u64)> {
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let Operation::Attn(Attn::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(Def::Cache(row)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = if *ratio == 4 { 2 } else { 1 } * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match rows.iter_mut().find(|(held, _)| *held == *row) {
            Some((_, held)) => *held = (*held).max(width),
            None => rows.push((*row, width)),
        }
    }
    rows
}

/// The bytes `trace`'s kv and state rows wire at `paging`.
pub fn pool_demand(trace: &Trace, paging: Paging) -> Result<u64> {
    Ok(PoolDemand::of(trace, paging, None)?.bytes(paging.pages()))
}

/// The bytes a load's pool takes at `paging`: the kv and state rows and the
/// recurrent-state buffers beside them, what `[model] device_kv_budget`
/// states for it.
pub fn pool_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let rs = crate::rs::Layout::read(trace)?;
    Ok(PoolDemand::of(trace, paging, rs.as_ref())?.bytes(paging.pages()))
}

struct Widths {
    keys: u64,
    values: u64,
    shared: bool,
}

fn split(name: &str, planes: &[u64]) -> Result<Widths> {
    match planes {
        [shared] => Ok(Widths {
            keys: *shared,
            values: *shared,
            shared: true,
        }),
        [keys, values] => Ok(Widths {
            keys: *keys,
            values: *values,
            shared: false,
        }),
        other => Err(Fault::Unbound {
            what: format!(
                "cache `{name}`, which declares {} plane(s) — this shell cuts kv \
                 pages into one shared plane or into a key half and a value half, \
                 and knows no other form",
                other.len()
            ),
        }),
    }
}

fn slot_file(error: std::io::Error) -> Fault {
    Fault::Device {
        call: "slot file",
        why: error.to_string(),
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

/// Bytes one cache row occupies for `width` elements of `dtype`.
///
/// A scalar dtype is `width * elem_bytes`. A PACKED dtype has no per-element byte
/// size — its row is sized from the block layout.
///
/// The `KvU4` codec's block IS the model's `head_dim` (rotation-alignment) — pie
/// serves many head_dims, not just 256 — so its packed size is head_dim-driven,
/// NOT a baked 256-group: a row of `heads` whole heads packs to
/// `heads * (head_dim/2 + 2)` bytes (130 per head at head_dim 256, 66 at 128).
/// When `head_dim` is known (the allocation and migration paths have it), the
/// KvU4 row is sized from it. When it is not (the pre-facts demand estimate has
/// only the trace), the format's own `row_bytes` gives the head_dim-256 anchor
/// (`repr().row_bytes(256) == 130`), which is exact for the head_dim-256 flagship
/// — the only KvU4 geometry that ships today.
///
/// Any other packed dtype (block-quantized weight caches) has no head_dim and is
/// sized from its format's `row_bytes` as before. This is the ONE place packed
/// vs. scalar sizing forks — reserve, migration copy and demand accounting all
/// route through it so a packed plane is contiguous, whole-slot cells that a
/// byte-blit can move.
fn row_stride(name: &str, dtype: Dtype, width: u64, head_dim: Option<u32>) -> Result<u64> {
    if let Some(element) = poem_compiler::arena::elem_bytes(dtype) {
        return Ok(width * element);
    }
    // KvU4 sizes per head_dim-block when the head geometry is known.
    let kv_block = (dtype == Dtype::KvU4)
        .then(|| head_dim.filter(|&d| d > 0))
        .flatten();
    if let Some(block) = kv_block {
        let block = u64::from(block);
        if !width.is_multiple_of(block) {
            return Err(Fault::Unbound {
                what: format!(
                    "cache `{name}`, whose {width}-wide packed KV row is not a whole number of \
                     {block}-wide heads"
                ),
            });
        }
        let heads = width / block;
        return Ok(heads * (block / 2 + 2));
    }
    let k = u32::try_from(width).map_err(|_| Fault::Unbound {
        what: format!("cache `{name}`, whose {width}-wide row overflows a packed-plane width"),
    })?;
    dtype.row_bytes(k).ok_or_else(|| Fault::Unbound {
        what: format!(
            "cache `{name}`, stored as {dtype:?}, whose {width}-wide row is not a whole number of \
             the format's {}-element blocks",
            dtype.quantum().elems(k)
        ),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}
