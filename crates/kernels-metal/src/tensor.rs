use dtype::Dtype;

use crate::encode::ArgValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub page_size: i32,

    pub seq_stride: u64,

    pub head_stride: u64,

    /// Whether attention reads this pool through the 8-bit kernels: the
    /// load's `kernel.sdpa.kv_q8`.
    pub q8: bool,

    /// Bytes from one page to the next, keys then values
    /// (`kernels/prim/kv_page.h`). Each plane's tensor is bound at its rows
    /// of page 0 (a layer's offset into a page is the binding's offset).
    pub page_stride: [u64; 2],
}

/// The plane offset every paged kernel takes after the page strides: always
/// 0, since a pool binds each plane at its offset into a page
/// (`setBuffer:offset:`). A runtime offset applied inside the kernels cost
/// paged decode attention ~1.4% of a gemma-4-e4b decode token (M1 Max).
pub const PLANE_AT_BINDING: u64 = 0;

impl KvPool {
    /// The page stride of a plane that is its own run of `page_size`-row
    /// pages, each row `plane.width` wide (a packed 4-bit row is a block of
    /// `head_dim / 2 + 2` bytes a head).
    #[must_use]
    pub fn dense_page_stride(plane: Tensor, page_size: i32, head_dim: u64) -> u64 {
        let width = u64::from(plane.width);
        let row = match plane.dtype {
            Dtype::KvU4 if head_dim != 0 => width / head_dim * (head_dim / 2 + 2),
            dtype => width * dtype.bits() / 8,
        };
        row * u64::from(page_size.unsigned_abs())
    }

    /// This pool with each plane its own run of `page_size`-row pages from
    /// the plane's origin: the layout every pool had before page strides.
    #[must_use]
    pub fn paged_densely(self) -> Self {
        Self {
            page_stride: [
                Self::dense_page_stride(self.keys, self.page_size, self.head_stride),
                Self::dense_page_stride(self.values, self.page_size, self.head_stride),
            ],
            ..self
        }
    }

    /// The four arguments every paged kernel takes last: each plane's page
    /// stride, then each plane's offset ([`PLANE_AT_BINDING`]).
    #[must_use]
    pub fn page_args(&self) -> [ArgValue; 4] {
        [
            ArgValue::Usize(self.page_stride[0]),
            ArgValue::Usize(self.page_stride[1]),
            ArgValue::Usize(PLANE_AT_BINDING),
            ArgValue::Usize(PLANE_AT_BINDING),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    pub state: Tensor,

    /// A chunk id per piece of each lane's record: `[lanes, pieces]`.
    pub slots: Tensor,

    pub conv_state: Tensor,

    pub new_conv_state: Tensor,

    /// The bank's segment table, a [`StateMap`]'s [`StateMap::words`] in
    /// u64; every state kernel addresses the bank through it.
    pub segments: Tensor,
}

/// How a fixed state bank is laid out over its segments
/// (`kernels/prim/state_map.h`): a record's ids name a slot each, a slot
/// sits in a chunk, a chunk in a segment, and every count is in elements of
/// the bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateMap {
    /// Elements from one slot of a chunk to the next.
    pub slot_stride: u64,
    /// Slots one chunk holds.
    pub chunk_slots: u32,
    /// Elements from one chunk of a segment to the next.
    pub chunk_stride: u64,
    /// Chunks one segment holds.
    pub seg_chunks: u32,
    /// Atoms (heads, channels or frames) one piece of a record holds.
    pub piece_atoms: u32,
    /// Pieces one record is cut into.
    pub pieces: u32,
    /// Each segment's first element, from the bank's origin.
    pub bases: Vec<u64>,
}

/// The word a kernel divides by `d` with (`pie_div` in
/// `kernels/prim/state_map.h`): a multiply-high and two shifts in place of
/// a division (the round-up method; exact for every `u32` dividend). Low 32
/// bits the multiplier, then the two shifts a byte each.
#[must_use]
pub fn divisor_word(d: u32) -> u64 {
    let d = d.max(1);
    // ceil(log2(d)), 0 for 1.
    let l = 32 - (d - 1).leading_zeros();
    let m = ((((1u64 << l) - u64::from(d)) << 32) / u64::from(d)) as u32 as u64 + 1;
    let (sh1, sh2) = (l.min(1), l.saturating_sub(1));
    (m & 0xffff_ffff) | (u64::from(sh1) << 32) | (u64::from(sh2) << 40)
}

/// `n / d` as a kernel finds it from [`divisor_word`]`(d)`.
#[must_use]
pub fn divide_by_word(n: u32, word: u64) -> u32 {
    let m = word as u32;
    let sh1 = ((word >> 32) & 0xff) as u32;
    let sh2 = ((word >> 40) & 0xff) as u32;
    let t = ((u64::from(n) * u64::from(m)) >> 32) as u32;
    (t + ((n - t) >> sh1)) >> sh2
}

impl StateMap {
    /// Ulongs before the segment bases.
    pub const HEAD: usize = 9;

    /// The layout every bank had before segments: slot `id` at
    /// `id * stride`, one piece a record, one segment.
    #[must_use]
    pub fn whole(stride: u64) -> Self {
        Self {
            slot_stride: stride,
            chunk_slots: 1,
            chunk_stride: stride,
            seg_chunks: u32::MAX,
            piece_atoms: u32::MAX,
            pieces: 1,
            bases: vec![0],
        }
    }

    /// The table a kernel reads.
    #[must_use]
    pub fn words(&self) -> Vec<u64> {
        let mut words = vec![
            self.slot_stride,
            u64::from(self.chunk_slots),
            self.chunk_stride,
            u64::from(self.seg_chunks),
            u64::from(self.piece_atoms),
            u64::from(self.pieces),
            divisor_word(self.chunk_slots),
            divisor_word(self.seg_chunks),
            divisor_word(self.piece_atoms),
        ];
        words.extend_from_slice(&self.bases);
        words
    }

    /// The table's bytes, little-endian as the device reads them.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        self.words().iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The first element of the slot `id` names, as the kernels find it.
    #[must_use]
    pub fn slot(&self, id: u32) -> u64 {
        let chunk = id / self.chunk_slots;
        let seg = (chunk / self.seg_chunks) as usize;
        self.bases[seg]
            + u64::from(chunk % self.seg_chunks) * self.chunk_stride
            + u64::from(id % self.chunk_slots) * self.slot_stride
    }

    /// The first element of atom `atom` (atoms outermost, `atom_elems`
    /// each) of the record whose piece ids are `ids`.
    #[must_use]
    pub fn atom(&self, ids: &[u32], atom: u32, atom_elems: u64) -> u64 {
        self.slot(ids[(atom / self.piece_atoms) as usize])
            + u64::from(atom % self.piece_atoms) * atom_elems
    }

    /// Whether this is [`StateMap::whole`]'s layout, which needs no piece
    /// table of its own.
    #[must_use]
    pub fn is_whole(&self) -> bool {
        *self == Self::whole(self.slot_stride)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    pub codes: Tensor,

    pub mpp_codes: Option<Tensor>,

    pub scales: Tensor,

    pub biases: Option<Tensor>,

    pub group: u32,

    pub bits: u32,

    /// The row stride of `codes` in columns: `codes.width` for a whole bank,
    /// more for a view over the leading columns of a wider one. Scales and
    /// biases follow it at `ld / group` per row.
    pub ld: u32,
}

impl Bank {
    #[must_use]
    pub const fn affine(&self) -> bool {
        self.biases.is_some()
    }
}

#[cfg(test)]
mod divisor_tests {
    use super::{divide_by_word, divisor_word};

    #[test]
    fn a_divisor_word_divides_every_dividend_as_division_does() {
        let divisors = [
            1u32,
            2,
            3,
            5,
            7,
            8,
            13,
            63,
            64,
            65,
            100,
            127,
            128,
            255,
            256,
            1000,
            4097,
            65_535,
            65_536,
            1 << 20,
            (1 << 20) + 3,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            u32::MAX - 1,
            u32::MAX,
        ];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for &d in &divisors {
            let word = divisor_word(d);
            let mut check = |n: u32| assert_eq!(divide_by_word(n, word), n / d, "{n} / {d}");
            for n in [
                0,
                1,
                d.wrapping_sub(1),
                d,
                d.wrapping_add(1),
                u32::MAX - 1,
                u32::MAX,
            ] {
                check(n);
            }
            for k in 0..4096u32 {
                check(k);
                check(d.wrapping_mul(k));
                check(d.wrapping_mul(k).wrapping_sub(1));
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                check(x as u32);
            }
        }
    }
}
