#![cfg(feature = "device")]
#![allow(unsafe_code)]

use engine_cuda::store::{Pools, kv};
use poem::{Dtype, Platform};
use poem_exec::store::kv::Paging;

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SLOTS: u32 = 4;

fn copy(dst: u64, src: u64, bytes: usize, kind: engine_cuda::device::rt::cudaMemcpyKind) {
    use engine_cuda::device::rt;
    // SAFETY: both spans are live: one a host buffer of `bytes`, the other a
    // page of a plane the export says is mapped.
    let said = unsafe {
        rt::cudaMemcpy(
            dst as *mut core::ffi::c_void,
            src as *const core::ffi::c_void,
            bytes,
            kind,
        )
    };
    assert_eq!(said, rt::cudaError::cudaSuccess, "the page is mapped");
}

#[test]
fn an_exported_kv_stays_mapped() {
    use engine_cuda::device::rt::cudaMemcpyKind::{cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice};

    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem_compiler::catalog::trace_of(
        "qwen3-a3b-micro",
        Dtype::Bf16,
        Dtype::Bf16,
        Platform::Cuda,
    );
    let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(SLOTS, 64));
    let facts = kv::probe(&trace).expect("the micro trace's cache rows probe");
    let paging = Paging::of(PAGE, CONTEXT, SLOTS, u64::from(SLOTS * CONTEXT / PAGE))
        .expect("a pool of four full sequences");
    let mut pools = Pools::reserve(
        0,
        0.5,
        &trace,
        paging,
        None,
        &facts,
        &budgets,
        0,
        None,
        false,
        &[],
        true,
    )
    .expect("the pool reserves");
    let exported = pools
        .export_kv(0)
        .expect("a full-attention trace exports its kv");

    let pages = paging.pages();
    let planes = trace
        .caches
        .iter()
        .filter(|row| matches!(row, poem_ir::CacheRow::Kv { .. }))
        .count();
    assert_eq!(
        exported.regions.len(),
        2 * planes,
        "a region per key and per value plane"
    );
    // A plane's region is the state pool from the plane's place in a page's
    // slot on: a page is read a stride past the one before it, by its slot
    // ref, and holds the plane's own bytes there.
    for region in &exported.regions {
        assert!(
            region.len >= pages * region.page_stride,
            "every declared page"
        );
        assert_eq!(region.domain, engine::MemoryDomain::CudaDevice(0));
    }
    let offsets: std::collections::BTreeSet<u64> =
        exported.regions.iter().map(|region| region.base).collect();
    assert_eq!(offsets.len(), exported.regions.len(), "each plane apart");
    const PROBE: usize = 64;

    // The last slot of every plane holds what is put there: every chunk of
    // the pool is backed.
    let last = |region: &engine::KvRegion| {
        region.base + (region.len / region.page_stride - 1) * region.page_stride
    };
    for (at, region) in exported.regions.iter().enumerate() {
        let pattern = [at as u8 + 1; PROBE];
        copy(
            last(region),
            pattern.as_ptr() as u64,
            pattern.len(),
            cudaMemcpyHostToDevice,
        );
    }
    for (at, region) in exported.regions.iter().enumerate() {
        let mut read = [0u8; PROBE];
        copy(
            read.as_mut_ptr() as u64,
            last(region),
            read.len(),
            cudaMemcpyDeviceToHost,
        );
        assert!(
            read.iter().all(|&byte| byte == at as u8 + 1),
            "plane {at}'s last page holds what was put there"
        );
    }
}
