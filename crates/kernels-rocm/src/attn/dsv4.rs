//! DeepSeek-V4's quantized cache rows and fused passes, which are CUDA's:
//! ROCm lays DeepSeek's caches bf16 and runs the unfused ops, so what reads
//! the rows refuses.

use dtype::Dtype;

use crate::error::Error;
use crate::jit::Ctx;
use crate::tensor::{KvPool, Tensor};

/// The quantized rows a DeepSeek pool may hold, each a u8 plane of its
/// own width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// V4's fp8 rows.
    Ds,
    /// V4.1's window: 512 e4m3 codes and 16 ue8m0 exponents.
    Mx8,
    /// V4.1's compressed rows: 256 bytes of e2m1 pairs and 32 e4m3 scales.
    Nv4,
}

impl Rows {
    /// The bytes of a position.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Ds => 584,
            Self::Mx8 => 528,
            Self::Nv4 => 288,
        }
    }
}

/// Whether `pool` holds DeepSeek's quantized rows of any format, a row a
/// position or compact.
#[must_use]
pub fn ds_rows(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::U8
        && pool.layout == 0
        && pool.seq_stride > 0
        && [1i64, 2, 4, 8, 16].into_iter().any(|div| {
            [Rows::Ds, Rows::Mx8, Rows::Nv4]
                .into_iter()
                .any(|r| pool.seq_stride * div == i64::from(r.bytes()))
        })
}

/// A row's compressed rows: the pool, its ratio, and the indexer's picks.
#[derive(Clone, Copy, Debug)]
pub struct Pooled<'a> {
    pub entries: &'a KvPool,
    pub ratio: u32,
    pub selection: Option<(Tensor, u32)>,
}

/// The rotation the output is turned back by.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    pub dim: u32,
    pub theta: f32,
    pub yarn: Option<crate::elemwise::rope::Yarn>,
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    _ctx: &Ctx,
    _q: Tensor,
    _positions: Tensor,
    _request_of_token: Tensor,
    _floors: Option<Tensor>,
    _window_pool: &KvPool,
    _window: u32,
    _pooled: Option<Pooled<'_>>,
    _sinks: Tensor,
    _heads: u32,
    _head_dim: u32,
    _sm_scale: f32,
    _turn: Option<Turn>,
    _o: &mut Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attention.window_sink",
    })
}

pub fn append(
    _ctx: &Ctx,
    _rows: Tensor,
    _pool: &KvPool,
    _write_page: Tensor,
    _write_offset: Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attention.kv_append_shared",
    })
}

pub fn store_entries(
    _ctx: &Ctx,
    _entries: Tensor,
    _boundary_pos: Tensor,
    _boundary_req: Tensor,
    _pool: &KvPool,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attention.pool_kv_append",
    })
}
