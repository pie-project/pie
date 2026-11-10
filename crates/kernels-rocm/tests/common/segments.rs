//! A slot-layout state slab scattered into pieces across a two-segment
//! pool: each slot's row cut along its atom axis into equal pieces, every
//! piece in a chunk of its own at a shuffled chunk id, `OFF` elements into
//! the chunk, garbage around it. A kernel that reads and writes the pool
//! through `Segments` must answer what it answers over the slab, bit for
//! bit, and leave the garbage alone (`agree`).

#![allow(dead_code)]

use crate::common::{Gpu, STATE, bf16};
use dtype::Dtype;
use kernels_rocm::{RecurrentPool, Segments, Snapshots, Tensor};

/// Garbage elements before a piece inside its chunk (a multiple of eight, so
/// vector arms stay vector arms) and after it.
pub const OFF: usize = 8;
const TAIL: usize = 8;

/// A 64-bit LCG: shuffles, garbage and test values.
pub struct Lcg(u64);

impl Lcg {
    pub fn seeded(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// The generator's next raw state.
    pub fn bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.bits() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    /// `n` bf16 values in [-scale, scale), as raw bits.
    pub fn bf16s(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16(scale * self.unit())).collect()
    }

    /// `n` f32 values in [-scale, scale), as raw bits.
    pub fn f32s(&mut self, n: usize, scale: f32) -> Vec<u32> {
        (0..n).map(|_| (scale * self.unit()).to_bits()).collect()
    }
}

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

/// One state element as raw bits: bf16 rides as `u16`, f32 and i32 as `u32`.
pub trait Bits: Copy + Default + PartialEq + std::fmt::Debug {
    /// A dtype of the element's width, for moving raw images.
    const RAW: Dtype;
    fn of(v: u64) -> Self;
    fn wide(self) -> u64;
}

impl Bits for u16 {
    const RAW: Dtype = Dtype::Bf16;
    fn of(v: u64) -> Self {
        v as u16
    }
    fn wide(self) -> u64 {
        u64::from(self)
    }
}

impl Bits for u32 {
    const RAW: Dtype = Dtype::F32;
    fn of(v: u64) -> Self {
        v as u32
    }
    fn wide(self) -> u64 {
        u64::from(self)
    }
}

/// The pool: `segs` the two segments, `chunk_of[slot][p]` the chunk piece
/// `p` of `slot` sits in.
pub struct Scattered {
    segs: [Tensor; 2],
    table: Tensor,
    seg_chunks: usize,
    chunk_of: Vec<Vec<i32>>,
    pieces: usize,
    piece: usize,
    seg_elems: usize,
    image: [Vec<u64>; 2],
}

impl Scattered {
    /// `slab`'s `slots` rows (`stride` apart) cut by `cut`, scattered.
    pub fn new<T: Bits>(
        gpu: &Gpu,
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
        for (slot, ids) in chunk_of.iter_mut().enumerate() {
            for (p, id) in ids.iter_mut().enumerate() {
                let c = order[slot * cut.pieces + p];
                *id = c as i32;
                let (s, c) = (c / seg_chunks, c % seg_chunks);
                for i in 0..cut.piece {
                    image[s][c * chunk + OFF + i] = slab[slot * stride + (cut.at)(p, i)];
                }
            }
        }
        let segs = [
            gpu.upload(&image[0], seg_elems as u32, 1, T::RAW),
            gpu.upload(&image[1], seg_elems as u32, 1, T::RAW),
        ];
        let table = gpu.upload(&[segs[0].ptr, segs[1].ptr], 2, 1, Dtype::U64);
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
            table: self.table,
            seg_chunks: self.seg_chunks as u32,
            pieces: self.pieces as u32,
            piece_off: OFF as i64 + skew,
            piece_atoms: 0,
        }
    }

    /// A slab-shaped tensor the kernels read the pool through: its pointer
    /// one segment's base (unused under a table), its width a row's.
    pub fn slab(&self, row: u32, dtype: Dtype) -> Tensor {
        Tensor::new(self.segs[0].ptr, self.chunk_of.len() as u32, row, dtype)
    }

    /// A slot table (negative: no state) as the pool's chunk ids, `pieces`
    /// a slot.
    pub fn ids(&self, gpu: &Gpu, slots: &[i32]) -> Tensor {
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
        gpu.column(&ids)
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
            gpu.read_as::<T>(&self.segs[0]),
            gpu.read_as::<T>(&self.segs[1]),
        ];
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

/// A fire over `pool`: its output rows, as bits.
pub type Fire<'a> = dyn FnMut(&Gpu, RecurrentPool) -> Vec<u64> + 'a;

/// The state a family keeps: `slots` rows `stride` elements apart, cut by
/// `cut`.
pub struct Slab<'a, T> {
    pub seed: Vec<T>,
    pub slots: usize,
    pub stride: usize,
    pub dtype: Dtype,
    pub cut: Cut<'a>,
}

fn pool(
    slab: Tensor,
    ids: Tensor,
    chunk: i64,
    segments: Segments,
    snapshots: Snapshots,
) -> RecurrentPool {
    RecurrentPool {
        slab,
        slot_ids: ids,
        slot_stride_elems: chunk,
        conv_slab: slab,
        conv_stride: chunk,
        snapshots,
        segments,
        ..STATE
    }
}

/// `fire` over the slot layout and over the scattered pool (lanes in slots
/// `lanes`, snapshots landing in slots `snap_slots` where `snaps` asks):
/// the same rows and state, bit for bit, and the garbage kept; with the
/// piece offset moved by `skew` elements, not.
#[allow(clippy::too_many_arguments)]
pub fn agree<T: Bits>(
    gpu: &Gpu,
    what: &str,
    slab: &Slab<T>,
    lanes: &[i32],
    snaps: Snapshots,
    snap_slots: &[i32],
    fire: &mut Fire,
    skew: i64,
) {
    let row = slab.stride as u32;
    let with_slots = |slots: Tensor| {
        if snap_slots.is_empty() {
            snaps
        } else {
            Snapshots { slots, ..snaps }
        }
    };

    let flat = gpu.upload(&slab.seed, slab.slots as u32, row, slab.dtype);
    let snap_ids = with_slots(if snap_slots.is_empty() {
        Tensor::ABSENT
    } else {
        gpu.column(snap_slots)
    });
    let want_rows = fire(
        gpu,
        pool(
            flat,
            gpu.column(lanes),
            slab.stride as i64,
            Segments::ONE,
            snap_ids,
        ),
    );
    gpu.sync();
    let want_slab: Vec<T> = gpu.read_as::<T>(&flat);
    assert_eq!(want_slab.len(), slab.seed.len());

    let scatter = |skew: i64, fire: &mut Fire| {
        let sc = Scattered::new(gpu, 0x5e65, &slab.seed, slab.slots, slab.stride, &slab.cut);
        let ids = sc.ids(gpu, lanes);
        let snap_ids = with_slots(if snap_slots.is_empty() {
            Tensor::ABSENT
        } else {
            sc.ids(gpu, snap_slots)
        });
        let p = pool(
            sc.slab(row, slab.dtype),
            ids,
            sc.chunk(),
            sc.segments(skew),
            snap_ids,
        );
        let rows = fire(gpu, p);
        gpu.sync();
        let (state, kept) = sc.gather(gpu, &slab.seed, slab.stride, &slab.cut);
        (rows, state, kept)
    };

    let (rows, state, kept) = scatter(0, fire);
    assert!(
        rows == want_rows,
        "{what}: the scattered pool's rows are not the slot layout's"
    );
    assert!(
        state == want_slab,
        "{what}: the scattered pool's state is not the slot layout's"
    );
    assert!(
        kept,
        "{what}: a fire over the scattered pool wrote between its pieces"
    );

    let (rows, state, kept) = scatter(skew, fire);
    assert!(
        rows != want_rows || state != want_slab || !kept,
        "{what}: a piece offset {skew} elements off still answers the slot layout"
    );
}

/// The raw bits of `t`'s elements, widened.
pub fn bits<T: Bits>(gpu: &Gpu, t: &Tensor) -> Vec<u64> {
    gpu.read_as::<T>(t).into_iter().map(Bits::wide).collect()
}

/// A request CSR over `lens`, i32.
pub fn indptr(gpu: &Gpu, lens: &[usize]) -> Tensor {
    let mut at = vec![0i32];
    for &len in lens {
        at.push(at.last().unwrap() + len as i32);
    }
    gpu.column(&at)
}
