//! The weight cache's staging gather: experts (or row blocks) read
//! zero-copy from pinned host memory into staging slots, each pointing its
//! table entry at its slot, on a stream and replayed from a graph captured
//! once over whatever list the host writes before each replay.
#![allow(unsafe_code)]

use crate::common::{Gpu, captured, forget, legacy, replay};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::gather::{StageCopy, stage_copies};

/// `bytes` of mapped pinned host memory: its host pointer and the device's
/// address of it. Freed on drop.
struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
}

impl Pinned {
    fn new(bytes: usize) -> Pinned {
        let _legacy = legacy();
        let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: local out-pointers; the allocation is freed once in drop.
        unsafe {
            let code = rt::cudaHostAlloc(&raw mut host, bytes, rt::cudaHostAllocMapped);
            assert_eq!(code, rt::cudaError::cudaSuccess, "`cudaHostAlloc`");
            let code = rt::cudaHostGetDevicePointer(&raw mut device, host, 0);
            assert_eq!(
                code,
                rt::cudaError::cudaSuccess,
                "`cudaHostGetDevicePointer`"
            );
        }
        Pinned {
            host: host.cast(),
            device: device as u64,
            bytes,
        }
    }

    fn fill(&mut self, seed: u64) {
        let mut s = seed | 1;
        // SAFETY: `host` holds `bytes` bytes this struct owns.
        let all = unsafe { core::slice::from_raw_parts_mut(self.host, self.bytes) };
        for b in all {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 56) as u8;
        }
    }

    fn at(&self, offset: usize, bytes: usize) -> &[u8] {
        // SAFETY: the caller's range lies within the allocation.
        unsafe { core::slice::from_raw_parts(self.host.add(offset), bytes) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by `cudaHostAlloc` in `new`, freed once.
        unsafe {
            rt::cudaFreeHost(self.host.cast());
        }
    }
}

/// Copies of `(host offset, slot, bytes)`: each into slot `slot` of
/// `slot_bytes` at `pool`, entry `slot` of the table at `table`.
fn copies(
    host: &Pinned,
    pool: u64,
    slot_bytes: u64,
    table: u64,
    of: &[(u64, u64, u64)],
) -> Vec<StageCopy> {
    of.iter()
        .map(|&(src, slot, bytes)| StageCopy {
            src: host.device + src,
            dst: pool + slot * slot_bytes,
            bytes,
            entry: table + slot * 8,
        })
        .collect()
}

/// Every copy landed: its slot holds the host's bytes, its entry the slot.
fn landed(g: &Gpu, host: &Pinned, list: &[StageCopy], table: u64, what: &str) {
    g.sync();
    let entries = g.down::<u64>(table, 16);
    for c in list {
        let got = g.down::<u8>(c.dst, c.bytes as usize);
        let src = (c.src - host.device) as usize;
        assert!(
            got == host.at(src, c.bytes as usize),
            "{what}: the slot at {:#x} does not hold the host's bytes",
            c.dst
        );
        let slot = ((c.entry - table) / 8) as usize;
        assert_eq!(entries[slot], c.dst, "{what}: entry {slot} names its slot");
    }
}

#[test]
fn the_staging_gather_copies_pinned_host_experts_into_their_slots() {
    let mut g = Gpu::open();
    let mut host = Pinned::new(1 << 20);
    host.fill(0x57a6e);
    let slot_bytes = 64u64 << 10;
    let pool = g.zeros(16 * slot_bytes as usize);
    let table = g.zeros(16 * 8);
    // Whole 16-byte runs, a long one split across blocks, and a ragged one
    // (odd offset and length) byte by byte.
    let list = copies(
        &host,
        pool,
        slot_bytes,
        table,
        &[
            (0, 3, 4096),
            (8192, 0, slot_bytes),
            (12_345, 7, 3001),
            (65_536, 1, 16),
        ],
    );
    let (at, count) = (g.up(&list), g.up(&[list.len() as u32]));
    let ctx = g.ctx();
    stage_copies(&ctx, at, count, list.len() as u32, slot_bytes).expect("the gather fires");
    landed(&g, &host, &list, table, "on the stream");
    // A slot past the count is left alone.
    g.fill(pool + 9 * slot_bytes, 0xee, 64);
    let past = copies(&host, pool, slot_bytes, table, &[(0, 2, 64), (0, 9, 64)]);
    let (at, count) = (g.up(&past), g.up(&[1u32]));
    stage_copies(&ctx, at, count, 2, 64).expect("the gather fires");
    landed(&g, &host, &past[..1], table, "the first of two");
    g.sync();
    assert!(
        g.down::<u8>(pool + 9 * slot_bytes, 64)
            .iter()
            .all(|&b| b == 0xee),
        "a copy past the count landed"
    );
}

#[test]
fn a_captured_staging_gather_replays_whatever_list_it_is_given() {
    let mut g = Gpu::open();
    let mut host = Pinned::new(512 << 10);
    host.fill(0xca97);
    let slot_bytes = 32u64 << 10;
    let (max, span) = (8u32, slot_bytes);
    let pool = g.zeros(max as usize * slot_bytes as usize);
    let table = g.zeros(16 * 8);
    let list_at = g.zeros(max as usize * core::mem::size_of::<StageCopy>());
    let count_at = g.zeros(4);
    let ctx = g.ctx();
    let exec = captured(&ctx, || {
        stage_copies(&ctx, list_at, count_at, max, span).expect("the gather captures");
    });
    let write = |list: &[StageCopy]| {
        crate::common::host::upload_to(list_at, list);
        crate::common::host::upload_to(count_at, &[list.len() as u32]);
    };
    for (round, of) in [
        vec![(0u64, 2u64, 1024u64), (4096, 5, slot_bytes)],
        vec![
            (100_000, 0, 777),
            (40_960, 7, 8192),
            (200_000, 1, 16),
            (300_000, 6, 32_000),
        ],
    ]
    .into_iter()
    .enumerate()
    {
        let list = copies(&host, pool, slot_bytes, table, &of);
        g.sync();
        write(&list);
        replay(&ctx, exec);
        landed(&g, &host, &list, table, &format!("replay {round}"));
    }
    forget(exec);
}
