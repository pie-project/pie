#![cfg(feature = "device")]

use engine_cuda::store::{Pools, kv};
use poem::{Dtype, Platform};
use poem_exec::store::kv::Paging;

const PAGE: u32 = 16;
const CONTEXT: u32 = 1024;
const SLOTS: u32 = 2;

// CONTRACT-BC 19: a windowed load's ring pages (its record's pieces, read
// as window-buffer pages) and its own-row pages (at the activation's top)
// are named by page ids of the buffer's grain, which the kernels read at
// `base + id * grain`: each page's id reaches the address its fold and its
// fill copy at.
#[test]
fn a_ring_page_id_reaches_its_address() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem_compiler::catalog::trace_of(
        "gemma4-e4b-mini-l6",
        Dtype::Bf16,
        Dtype::Bf16,
        Platform::Cuda,
    );
    let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(SLOTS, 256));
    let facts = kv::probe(&trace).expect("the mini gemma's cache rows probe");
    let paging = Paging::of(PAGE, CONTEXT, SLOTS, u64::from(SLOTS * CONTEXT / PAGE))
        .expect("a pool of two full sequences")
        .windowed(
            engine_cuda::store::window_of(&trace).expect("the trace's window"),
            256,
        );
    let (free, total) = engine_cuda::store::device_memory().expect("the card reports its memory");
    let mut pools = Pools::reserve(
        0,
        engine_cuda::device::elastic::ceiling_of(
            total,
            total - free,
            engine::Allowance {
                budget: Some(4 << 30),
            },
        ),
        &trace,
        paging,
        None,
        &facts,
        &budgets,
        1 << 20,
        None,
        &[],
        true,
        256,
    )
    .expect("the pool reserves");
    let grain = pools.window_grain();
    // The base is read once a record is seated: a flat pool (ROCm's)
    // reserves its range at the first seat, so before it the base is 0.
    let mut seated = Vec::new();
    for key in 0..SLOTS {
        seated.push(pools.ring_pages(key).expect("the record's ring pages"));
    }
    let base = pools.pool_base();
    assert_ne!(base, 0, "a seated pool has a base");
    let reaches = |id: i32, at: u64| u64::try_from(id).expect("a page id") * grain + base == at;
    for (key, pages) in seated.into_iter().enumerate() {
        assert!(!pages.is_empty(), "a windowed load's record holds a ring");
        for (page, &(id, at)) in pages.iter().enumerate() {
            assert!(reaches(id, at), "record {key}'s ring page {page}: id {id} at {at:#x}");
        }
    }
    for page in 0..4 {
        let (id, at) = (
            pools.own_page_id(page).expect("an own-row page"),
            pools.own_page_at(page).expect("an own-row page"),
        );
        assert!(reaches(id, at), "own-row page {page}: id {id} at {at:#x}");
    }
}
