//! A slot-layout state slab scattered into pieces across a two-segment
//! pool: each slot's row cut along its atom axis into equal pieces, every
//! piece in a chunk of its own at a shuffled chunk id, `OFF` elements into
//! the chunk, garbage around it. A kernel that reads and writes the pool
//! through `Segments` must answer what it answers over the slab, bit for
//! bit, and leave the garbage alone.

use super::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::{Segments, Tensor};

/// Garbage elements before a piece inside its chunk (a multiple of eight, so
/// vector arms stay vector arms) and after it.
pub const OFF: usize = 8;
const TAIL: usize = 8;

/// How a slot's row cuts: `pieces` pieces of `piece` elements, element `i`
/// of piece `p` being element `at(p, i)` of the row.
pub struct Cut<'a> {
    pub pieces: usize,
    pub piece: usize,
    pub at: &'a dyn Fn(usize, usize) -> usize,
}

impl Cut<'_> {
    /// Whole runs of a contiguous row, `piece` elements a piece.
    pub fn runs(pieces: usize, piece: usize) -> Cut<'static> {
        let at: &'static dyn Fn(usize, usize) -> usize =
            Box::leak(Box::new(move |p, i| p * piece + i));
        Cut { pieces, piece, at }
    }
}

/// The pool: `segs` the two segments' device bases, `chunk_of[slot][p]` the
/// chunk piece `p` of `slot` sits in.
pub struct Scattered {
    segs: [u64; 2],
    table: u64,
    seg_chunks: usize,
    chunk_of: Vec<Vec<i32>>,
    pieces: usize,
    piece: usize,
    seg_elems: usize,
    image: [Vec<u64>; 2],
    elem: usize,
}

/// One state element as raw bits: bf16 rides as `u16`, f32 and i32 as `u32`.
pub trait Bits: Copy + Default + PartialEq + std::fmt::Debug {
    const BYTES: usize;
    fn of(v: u64) -> Self;
    fn wide(self) -> u64;
}

impl Bits for u16 {
    const BYTES: usize = 2;
    fn of(v: u64) -> Self {
        v as u16
    }
    fn wide(self) -> u64 {
        u64::from(self)
    }
}

impl Bits for u32 {
    const BYTES: usize = 4;
    fn of(v: u64) -> Self {
        v as u32
    }
    fn wide(self) -> u64 {
        u64::from(self)
    }
}

impl Scattered {
    /// `slab`'s `slots` rows (`stride` apart) cut by `cut`, scattered.
    pub fn new<T: Bits>(
        gpu: &mut Gpu,
        seed: u64,
        slab: &[T],
        slots: usize,
        stride: usize,
        cut: &Cut,
    ) -> Self {
        let mut lcg = Lcg::seeded(seed);
        let total = slots * cut.pieces;
        // One spare chunk a segment, garbage throughout.
        let seg_chunks = total.div_ceil(2) + 1;
        let chunk = OFF + cut.piece + TAIL;
        let seg_elems = seg_chunks * chunk;
        let mut order: Vec<usize> = (0..2 * seg_chunks).collect();
        for i in (1..order.len()).rev() {
            let j = ((lcg.bits() >> 33) as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut image: [Vec<T>; 2] = [
            (0..seg_elems).map(|_| T::of(lcg.bits() >> 17)).collect(),
            (0..seg_elems).map(|_| T::of(lcg.bits() >> 17)).collect(),
        ];
        let mut chunk_of = vec![vec![0i32; cut.pieces]; slots];
        for slot in 0..slots {
            for p in 0..cut.pieces {
                let id = order[slot * cut.pieces + p];
                chunk_of[slot][p] = id as i32;
                let (s, c) = (id / seg_chunks, id % seg_chunks);
                for i in 0..cut.piece {
                    image[s][c * chunk + OFF + i] = slab[slot * stride + (cut.at)(p, i)];
                }
            }
        }
        let segs = [gpu.up(&image[0]), gpu.up(&image[1])];
        let table = gpu.up(&segs);
        Scattered {
            segs,
            table,
            seg_chunks,
            chunk_of,
            pieces: cut.pieces,
            piece: cut.piece,
            seg_elems,
            image: [
                image[0].iter().map(|v| v.wide()).collect(),
                image[1].iter().map(|v| v.wide()).collect(),
            ],
            elem: T::BYTES,
        }
    }

    /// A chunk's span in elements.
    pub fn chunk(&self) -> i64 {
        (OFF + self.piece + TAIL) as i64
    }

    /// The pool's segments, the piece offset moved by `skew` elements (0:
    /// where the pieces are).
    pub fn segments(&self, skew: i64) -> Segments {
        Segments {
            table: Tensor::new(self.table, 2, 1, Dtype::U64),
            seg_chunks: self.seg_chunks as u32,
            pieces: self.pieces as u32,
            piece_off: OFF as i64 + skew,
            piece_atoms: 0,
        }
    }

    /// A slab-shaped tensor the kernels read the pool through: its pointer
    /// one segment's base (unused under a table), its width a row's.
    pub fn slab(&self, row: u32, dtype: Dtype) -> Tensor {
        Tensor::new(self.segs[0], self.chunk_of.len() as u32, row, dtype)
    }

    /// A slot table (negative: no state) as the pool's chunk ids, `pieces`
    /// a slot.
    pub fn ids(&self, gpu: &mut Gpu, slots: &[i32]) -> Tensor {
        let ids: Vec<i32> = slots
            .iter()
            .flat_map(|&s| {
                if s < 0 {
                    vec![s; self.pieces]
                } else {
                    self.chunk_of[s as usize].clone()
                }
            })
            .collect();
        Tensor::new(gpu.up(&ids), ids.len() as u32, 1, Dtype::I32)
    }

    /// The pool gathered back into `slab`'s layout (elements no piece holds
    /// kept from `slab`), and whether every garbage element is as it was.
    pub fn gather<T: Bits>(
        &self,
        gpu: &Gpu,
        slab: &[T],
        stride: usize,
        cut: &Cut,
    ) -> (Vec<T>, bool) {
        let down = [
            gpu.down::<T>(self.segs[0], self.seg_elems),
            gpu.down::<T>(self.segs[1], self.seg_elems),
        ];
        assert_eq!(T::BYTES, self.elem);
        let chunk = self.chunk() as usize;
        let mut out = slab.to_vec();
        let mut held = [vec![false; self.seg_elems], vec![false; self.seg_elems]];
        for (slot, ids) in self.chunk_of.iter().enumerate() {
            for (p, &id) in ids.iter().enumerate() {
                let (s, c) = (id as usize / self.seg_chunks, id as usize % self.seg_chunks);
                for i in 0..self.piece {
                    out[slot * stride + (cut.at)(p, i)] = down[s][c * chunk + OFF + i];
                    held[s][c * chunk + OFF + i] = true;
                }
            }
        }
        let kept = (0..2).all(|s| {
            (0..self.seg_elems).all(|e| held[s][e] || down[s][e].wide() == self.image[s][e])
        });
        (out, kept)
    }
}
