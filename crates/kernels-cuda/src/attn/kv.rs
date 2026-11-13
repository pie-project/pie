//! Host-only: a paged pool's layout as the attention units read it (its
//! quantization scheme, the heads a row splits into, the lanes an indptr
//! names).

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

/// Plane `plane`'s page stride (0 keys, 1 values) in its own elements, as
/// the kernels step a typed pointer from page to page.
pub(crate) fn page_elems(op: &'static str, pool: &KvPool, plane: usize) -> Result<i64, Error> {
    let (bytes, dtype) = match plane {
        0 => (pool.page_stride[0], pool.keys.dtype),
        _ => (pool.page_stride[1], pool.values.dtype),
    };
    let bits = dtype.bits();
    if bits % 8 != 0 || bytes % (bits / 8) != 0 {
        return Err(refuse(
            op,
            format!("a {bytes}-byte page stride is no whole number of {dtype:?} elements"),
        ));
    }
    i64::try_from(bytes / (bits / 8))
        .map_err(|_| refuse(op, format!("a {bytes}-byte page stride passes an i64")))
}

/// [`page_elems`] as the u32 a kernel argument carries.
pub(crate) fn page_elems_u32(op: &'static str, pool: &KvPool, plane: usize) -> Result<u32, Error> {
    let elems = page_elems(op, pool, plane)?;
    u32::try_from(elems).map_err(|_| {
        refuse(
            op,
            format!("a {elems}-element page stride passes this kernel's u32"),
        )
    })
}

/// The page stride the key and value planes share, in their elements, for
/// a kernel stepping both by one stride.
pub(crate) fn shared_page_elems(op: &'static str, pool: &KvPool) -> Result<u32, Error> {
    let keys = page_elems_u32(op, pool, 0)?;
    let values = page_elems_u32(op, pool, 1)?;
    if keys != values {
        return Err(refuse(
            op,
            format!(
                "this kernel steps keys and values by one page stride; the pool's are {keys} and \
                 {values} elements"
            ),
        ));
    }
    Ok(keys)
}

/// Refuses a pool whose planes do not lie page after page, for a path that
/// still reads planes or side planes (scales, envelopes, a bf16 shadow) at
/// their packed offsets.
pub(crate) fn packed_only(op: &'static str, pool: &KvPool, what: &str) -> Result<(), Error> {
    if pool.packed() {
        return Ok(());
    }
    Err(refuse(
        op,
        format!("{what} reads its pages packed, and this pool's page stride is not"),
    ))
}
