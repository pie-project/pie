use dtype::Dtype;

use crate::encode::{ABSENT, ArgValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub buf: u32,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Tensor {
    #[must_use]
    pub const fn new(buf: u32, rows: u32, width: u32, dtype: Dtype) -> Self {
        Self {
            buf,
            rows,
            width,
            dtype,
        }
    }

    #[must_use]
    pub const fn arg(self) -> ArgValue {
        ArgValue::Buffer(self.buf)
    }

    #[must_use]
    pub const fn arg_mut(self) -> ArgValue {
        ArgValue::BufferMut(self.buf)
    }
}

/// A quantized weight bank: codes, a scale per group, and a bias per group
/// when the bank is affine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bank {
    pub codes: Tensor,

    pub scales: Tensor,

    pub biases: Option<Tensor>,

    pub group: u32,

    pub bits: u32,
}

/// A ragged batch: `data`'s rows split by the i32 CSR `indptr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

/// The segments a chunk pool is bound in, at most this many
/// (`kernels/prim/pool.slang`).
pub const SEGMENTS: usize = 4;

/// A chunk pool as a kernel binds it: up to [`SEGMENTS`] buffers, each
/// `words` u32 words of one virtual address space (segment `s` from word
/// `s * words`), a chunk never straddling two. A segment the pool does not
/// have is [`crate::ABSENT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segments {
    pub bufs: [u32; SEGMENTS],

    pub words: u32,
}

impl Segments {
    /// One buffer of `words` words.
    #[must_use]
    pub const fn one(buf: u32, words: u32) -> Self {
        Self {
            bufs: [buf, ABSENT, ABSENT, ABSENT],
            words,
        }
    }

    /// The segments as a kernel's four consecutive bindings.
    #[must_use]
    pub fn args(self, mutable: bool) -> [ArgValue; SEGMENTS] {
        self.bufs.map(|buf| {
            if mutable {
                ArgValue::BufferMut(buf)
            } else {
                ArgValue::Buffer(buf)
            }
        })
    }
}

/// u32 words `t`'s rows take, rounded down.
#[must_use]
pub fn words_of(t: Tensor) -> u32 {
    let bytes = u64::from(t.rows) * u64::from(t.width) * t.dtype.bits() / 8;
    u32::try_from(bytes / 4).unwrap_or(u32::MAX)
}

/// A paged KV cache in a chunk pool: the pages each request reads
/// (`page_indices` split by `page_indptr`), the strides of a page row, and
/// where the pages lie: page `p`'s keys at word `origin[0] + p *
/// page_stride[0]` of `pool`, its values at `origin[1] + p * page_stride[1]`
/// (SPEC 10: `base + slot * stride + offset`). `keys` and `values` carry the
/// planes' rows, width and dtype; kernels bind `pool`, not their buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub page_size: i32,

    pub seq_stride: u64,

    pub head_stride: u64,

    pub pool: Segments,

    /// u32 words from one page to the next, keys then values.
    pub page_stride: [u32; 2],

    /// u32 words from the pool's origin to page 0's keys, then its values.
    pub origin: [u32; 2],
}

impl KvPool {
    /// A pool of two planes of their own: `keys` (segment 0) and `values`
    /// (segment 1, or the keys again where one plane is shared), each a run
    /// of `page_size`-row pages `seq_stride` elements a row.
    #[must_use]
    pub fn planes(
        keys: Tensor,
        values: Tensor,
        page_indices: Tensor,
        page_indptr: Tensor,
        page_size: i32,
        seq_stride: u64,
        head_stride: u64,
    ) -> Self {
        let shared = keys.buf == values.buf;
        let words = words_of(keys).max(words_of(values)).max(1);
        let stride = |plane: Tensor, width: u64| {
            let bits = u64::from(page_size.unsigned_abs()) * width * plane.dtype.bits();
            u32::try_from(bits / 32).unwrap_or(u32::MAX)
        };
        Self {
            keys,
            values,
            page_indices,
            page_indptr,
            page_size,
            seq_stride,
            head_stride,
            pool: if shared {
                Segments::one(keys.buf, words)
            } else {
                Segments {
                    bufs: [keys.buf, values.buf, ABSENT, ABSENT],
                    words,
                }
            },
            page_stride: [
                stride(keys, seq_stride),
                stride(values, u64::from(values.width)),
            ],
            origin: [0, if shared { 0 } else { words }],
        }
    }
}

/// A communicator as a device's collectives see it: this rank, the ranks,
/// and the band every rank reads, `slot_bytes` of it a rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Comm {
    pub rank: u32,

    pub world: u32,

    pub band: u32,

    pub slot_bytes: u32,
}
