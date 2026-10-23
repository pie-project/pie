use crate::error::Error;
use dtype::Dtype;

use crate::jit::{refuse, stated};
use crate::tensor::{KvPool, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvScheme {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
}

impl KvScheme {
    #[must_use]
    pub const fn of_byte(byte: i32) -> Option<Self> {
        match byte {
            0 => Some(Self::Native),
            1 => Some(Self::Fp8PerTensor),
            2 => Some(Self::Int8PerTokenHead),
            3 => Some(Self::Fp8PerTokenHead),
            4 => Some(Self::Fp4Block),
            _ => None,
        }
    }
}

pub(crate) fn scheme_of(op: &'static str, pool: &KvPool) -> Result<KvScheme, Error> {
    KvScheme::of_byte(pool.scheme_byte).ok_or_else(|| {
        refuse(
            op,
            format!("no kv scheme is named by byte {}", pool.scheme_byte),
        )
    })
}

pub(crate) const FP8_E4M3: u32 = 0;

pub(crate) const fn fp4_block_size(block_size: i32) -> i32 {
    if block_size > 0 { block_size } else { 16 }
}

#[must_use]
pub fn native_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16
}

/// Whether the pool's pages are e4m3 under a unit scale, read as they are
/// (vLLM's `fp8` KV): no planes of scales, no bf16 copy.
#[must_use]
pub fn native_e4m3(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::E4m3
        && pool.values.dtype == Dtype::E4m3
        && KvScheme::of_byte(pool.scheme_byte) == Some(KvScheme::Native)
}

#[must_use]
pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
    if page_size <= 0 {
        return 0;
    }
    (total_tokens + page_size - 1) / page_size + num_requests
}

pub(crate) fn head_split(
    op: &'static str,
    pool: &KvPool,
    row_width: u32,
) -> Result<(i32, i32), Error> {
    let wide = if pool.layout != 0 {
        pool.seq_stride
    } else {
        pool.head_stride
    };
    let head_dim = i32::try_from(wide).ok().filter(|d| *d > 0).ok_or_else(|| {
        refuse(
            op,
            format!("the pool row's strides spell no head width ({wide})"),
        )
    })?;
    let row = stated(op, row_width)?;
    if row <= 0 || row % head_dim != 0 {
        return Err(refuse(
            op,
            format!(
                "the {row}-wide appended row does not divide by the pool's head width {head_dim}"
            ),
        ));
    }
    Ok((row / head_dim, head_dim))
}

pub(crate) fn lanes_of(op: &'static str, indptr: Tensor) -> Result<i32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's indptr is {:?}, and this entry walks an i32 boundary vector",
                indptr.dtype
            ),
        ));
    }
    let lanes = indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Err(refuse(op, "the fire's indptr spells no requests"));
    }
    stated(op, lanes)
}
