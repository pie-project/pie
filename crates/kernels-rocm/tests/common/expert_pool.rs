//! A routed bank's experts scattered across a pool, for the expert-table
//! tests: each expert's planes copied into slots of a garbage-filled pool in
//! a shuffled order with garbage between them, and the tables that name
//! them. A kernel that reads its experts through the table lands the bits
//! it lands off the contiguous bank; one that reads the bank past the table
//! reads garbage once the bank is garbled.
#![allow(unsafe_code)]

use crate::common::Gpu;
use dtype::Dtype;
use kernels_rocm::hip::{self, Direction};
use kernels_rocm::linear::moe::contiguous_entries;

/// One plane of a contiguous bank: expert `e`'s `bytes` at `base + e *
/// stride` (`base` 0: a plane the bank lacks, its entries null).
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub base: u64,
    pub stride: u64,
    pub bytes: u64,
}

impl Plane {
    #[must_use]
    pub const fn of(base: u64, stride: u64) -> Plane {
        Plane {
            base,
            stride,
            bytes: stride,
        }
    }
}

/// A scattered bank: the pool and the table's entries (host; upload with
/// [`table`]).
pub struct Scattered {
    pub pool: u64,
    pub pool_bytes: u64,
    pub entries: Vec<u64>,
}

impl Scattered {
    /// The `(base, bytes)` window every entry falls in.
    #[must_use]
    pub const fn window(&self) -> (u64, u64) {
        (self.pool, self.pool_bytes)
    }
}

/// Every pool copy starts on 256 bytes (the widest a kernel's loads want).
const ALIGN: u64 = 256;

fn copy(g: &Gpu, dst: u64, src: *const core::ffi::c_void, bytes: u64, kind: Direction) {
    // SAFETY: the caller names `bytes` of live memory at both ends; the
    // stream is synchronized before either goes.
    unsafe {
        hip::memcpy_async(
            dst as *mut core::ffi::c_void,
            src,
            bytes as usize,
            kind,
            g.stream.raw(),
        )
    }
    .expect("hipMemcpyAsync");
    g.sync();
}

/// A deterministic shuffle of `0..n` under `seed`.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        order.swap(i, (s % (i as u64 + 1)) as usize);
    }
    order
}

/// Bytes of garbage: no NaN-free promise, deliberately (a kernel that reads
/// one lands it).
#[must_use]
pub fn garbage(bytes: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..bytes)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 56) as u8
        })
        .collect()
}

/// `entries` uploaded as a device table.
pub fn table(g: &Gpu, entries: &[u64]) -> u64 {
    g.upload(entries, entries.len() as u32, 2, Dtype::U32).ptr
}

/// The contiguous bank's own entries: expert `e`'s entry `p` at `base_p + e
/// * stride_p`.
#[must_use]
pub fn degenerate(experts: usize, planes: &[Plane]) -> Vec<u64> {
    let pairs: Vec<(u64, u64)> = planes
        .iter()
        .map(|p| {
            if p.base == 0 {
                (0, 0)
            } else {
                (p.base, p.stride)
            }
        })
        .collect();
    contiguous_entries(experts as u32, &pairs)
}

/// `experts` experts of `planes` copied into a fresh pool in a shuffled
/// order (an expert's planes apart, not side by side), each copy aligned
/// with one to three alignments of garbage before it.
pub fn scatter(g: &Gpu, experts: usize, planes: &[Plane], seed: u64) -> Scattered {
    let count = planes.len();
    let order = shuffled(experts * count, seed);
    let mut at = 0u64;
    let mut offsets = vec![0u64; experts * count];
    for (i, &slot) in order.iter().enumerate() {
        let plane = planes[slot % count];
        at = at.next_multiple_of(ALIGN) + ALIGN * (1 + (i as u64 * 7 + seed) % 3);
        offsets[slot] = at;
        at += plane.bytes;
    }
    let pool_bytes = at.next_multiple_of(4096) + 4096;
    let pool = g
        .upload(
            &garbage(pool_bytes as usize, seed ^ 0x9e37),
            1,
            pool_bytes as u32,
            Dtype::U8,
        )
        .ptr;
    let entries: Vec<u64> = (0..experts * count)
        .map(|slot| {
            let (e, p) = (slot / count, planes[slot % count]);
            if p.base == 0 {
                return 0;
            }
            let dst = pool + offsets[slot];
            let src = (p.base + e as u64 * p.stride) as *const core::ffi::c_void;
            copy(g, dst, src, p.bytes, Direction::DeviceToDevice);
            dst
        })
        .collect();
    Scattered {
        pool,
        pool_bytes,
        entries,
    }
}

/// A mixed table: even experts where the contiguous bank has them, odd ones
/// in the pool.
#[must_use]
pub fn mixed(experts: usize, planes: &[Plane], pool: &Scattered) -> Vec<u64> {
    let own = degenerate(experts, planes);
    let count = planes.len();
    (0..experts * count)
        .map(|slot| {
            if (slot / count).is_multiple_of(2) {
                own[slot]
            } else {
                pool.entries[slot]
            }
        })
        .collect()
}

/// `entries` with experts `a` and `b` swapped: a wrong table.
#[must_use]
pub fn swapped(entries: &[u64], planes: usize, a: usize, b: usize) -> Vec<u64> {
    let mut wrong = entries.to_vec();
    for p in 0..planes {
        wrong.swap(a * planes + p, b * planes + p);
    }
    wrong
}

/// Overwrites the contiguous bank's planes with garbage: a kernel reading
/// them past its table lands it.
pub fn garble(g: &Gpu, experts: usize, planes: &[Plane]) {
    for (i, p) in planes.iter().enumerate() {
        if p.base == 0 {
            continue;
        }
        let bytes = (experts as u64 - 1) * p.stride + p.bytes;
        let junk = garbage(bytes as usize, 0x5a ^ i as u64);
        copy(
            g,
            p.base,
            junk.as_ptr().cast(),
            bytes,
            Direction::HostToDevice,
        );
    }
}

/// `bytes` of `byte` at `at`: a result plane set to a sentinel.
pub fn fill(g: &Gpu, at: u64, byte: u8, bytes: usize) {
    let host = vec![byte; bytes];
    copy(
        g,
        at,
        host.as_ptr().cast(),
        bytes as u64,
        Direction::HostToDevice,
    );
}
