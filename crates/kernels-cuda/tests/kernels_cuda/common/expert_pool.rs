//! A routed bank's experts scattered across a pool, for the location-table
//! tests: each expert's planes copied into slots of a garbage-filled pool in
//! a shuffled order with garbage between them, and the table that names
//! them. A kernel that reads its experts through the table lands the bits
//! it lands off the contiguous bank; one that reads the bank past the table
//! reads garbage.
#![allow(unsafe_code)]

use super::Gpu;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::linear::moe::contiguous_entries;

/// One plane of a contiguous bank: expert `e`'s `bytes` at `base + e *
/// stride`, its copy in the pool starting on a multiple of `align` (a TMA
/// reader's row bytes, or at least 16 for vector loads).
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub base: u64,
    pub stride: u64,
    pub bytes: u64,
    pub align: u64,
}

impl Plane {
    #[must_use]
    pub const fn of(base: u64, stride: u64) -> Plane {
        Plane {
            base,
            stride,
            bytes: stride,
            align: 256,
        }
    }

    /// The same plane, its pool copies aligned to `align` bytes.
    #[must_use]
    pub const fn aligned(self, align: u64) -> Plane {
        Plane { align, ..self }
    }
}

/// A scattered bank: the pool, its table (device), and the table's entries.
pub struct Scattered {
    pub pool: u64,
    pub pool_bytes: u64,
    pub table: u64,
    pub entries: Vec<u64>,
}

impl Scattered {
    /// The `(base, bytes)` window every entry falls in.
    #[must_use]
    pub const fn window(&self) -> (u64, u64) {
        (self.pool, self.pool_bytes)
    }
}

fn copy_d2d(dst: u64, src: u64, bytes: u64) {
    let _legacy = super::legacy();
    // SAFETY: the caller names `bytes` of live device memory at both ends.
    let code = unsafe {
        rt::cudaMemcpy(
            dst as *mut core::ffi::c_void,
            src as *const core::ffi::c_void,
            bytes as usize,
            rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        )
    };
    assert_eq!(code, rt::cudaError::cudaSuccess, "`cudaMemcpy D2D`");
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
fn garbage(bytes: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..bytes)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 56) as u8
        })
        .collect()
}

/// The contiguous bank's own table, uploaded: expert `e`'s entry `p` at
/// `base_p + e * stride_p`.
pub fn degenerate(g: &mut Gpu, experts: usize, planes: &[Plane]) -> u64 {
    let pairs: Vec<(u64, u64)> = planes.iter().map(|p| (p.base, p.stride)).collect();
    g.up(&contiguous_entries(experts as u32, &pairs))
}

/// `experts` experts of `planes` copied into a fresh pool in a shuffled
/// order (an expert's planes apart, not side by side), each copy on its
/// plane's alignment with one to three alignments of garbage before it.
pub fn scatter(g: &mut Gpu, experts: usize, planes: &[Plane], seed: u64) -> Scattered {
    let count = planes.len();
    let order = shuffled(experts * count, seed);
    let mut at = 0u64;
    let mut offsets = vec![0u64; experts * count];
    for (i, &slot) in order.iter().enumerate() {
        let plane = planes[slot % count];
        let align = plane.align.max(16);
        at = at.next_multiple_of(align) + align * (1 + (i as u64 * 7 + seed) % 3);
        offsets[slot] = at;
        at += plane.bytes;
    }
    let pool_bytes = at.next_multiple_of(4096) + 4096;
    let pool = g.up(&garbage(pool_bytes as usize, seed ^ 0x9e37));
    // An absent plane (a bank without bias) keeps its null entry.
    let entries: Vec<u64> = (0..experts * count)
        .map(|slot| {
            let (e, p) = (slot / count, planes[slot % count]);
            if p.base == 0 {
                return 0;
            }
            let dst = pool + offsets[slot];
            copy_d2d(dst, p.base + e as u64 * p.stride, p.bytes);
            dst
        })
        .collect();
    let table = g.up(&entries);
    Scattered {
        pool,
        pool_bytes,
        table,
        entries,
    }
}

/// `entries` with experts `a` and `b` swapped, uploaded: a wrong table.
pub fn swapped(g: &mut Gpu, entries: &[u64], planes: usize, a: usize, b: usize) -> u64 {
    let mut wrong = entries.to_vec();
    for p in 0..planes {
        wrong.swap(a * planes + p, b * planes + p);
    }
    g.up(&wrong)
}

/// Overwrites the contiguous bank's planes with garbage: a kernel reading
/// them past its table lands it.
pub fn garble(g: &Gpu, experts: usize, planes: &[Plane]) {
    for (i, p) in planes.iter().enumerate() {
        if p.base == 0 {
            continue;
        }
        let bytes = (experts as u64 - 1) * p.stride + p.bytes;
        g.fill(p.base, 0x5a ^ i as u8, bytes as usize);
    }
}

/// `entries` with every other expert's planes (from expert 0) copied into
/// mapped host memory and named there, uploaded: a streamed table whose
/// experts lie half on the host copy, half in the pool. The host copy is
/// held for the process's life.
pub fn half_on_host(g: &mut Gpu, entries: &[u64], planes: &[Plane]) -> u64 {
    let count = planes.len();
    let mut mixed = entries.to_vec();
    for (slot, entry) in mixed.iter_mut().enumerate() {
        let (e, p) = (slot / count, planes[slot % count]);
        if e % 2 != 0 || *entry == 0 {
            continue;
        }
        let _legacy = super::legacy();
        let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a live out-parameter; the allocation is never freed (a
        // test's mapped host copy, held for the process).
        let code = unsafe { rt::cudaHostAlloc(&raw mut host, p.bytes as usize, 2) };
        assert_eq!(code, rt::cudaError::cudaSuccess, "`cudaHostAlloc` mapped");
        // SAFETY: `host` is `p.bytes` of pinned memory just allocated, and
        // `entry` names `p.bytes` of live device memory.
        let code = unsafe {
            rt::cudaMemcpy(
                host,
                *entry as *const core::ffi::c_void,
                p.bytes as usize,
                rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(code, rt::cudaError::cudaSuccess, "`cudaMemcpy D2H`");
        let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: `host` is a mapped allocation; the call writes `device`.
        let code = unsafe { rt::cudaHostGetDevicePointer(&raw mut device, host, 0) };
        assert_eq!(
            code,
            rt::cudaError::cudaSuccess,
            "`cudaHostGetDevicePointer`"
        );
        *entry = device as u64;
    }
    g.up(&mixed)
}
