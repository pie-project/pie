//! The elastic VMM arena, against a real driver.
//!
//! This is the part of the crate that unit tests cannot reach. [`PoolBudget`]
//! is a pure state machine and is tested exhaustively in `src/cuda/vmm.rs`,
//! but everything it is *accounting for* -- `cuMemCreate`, `cuMemMap`,
//! `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease` -- only means anything when
//! a driver is on the other end. A budget that balances perfectly while the
//! mappings underneath it are wrong is exactly the failure this file exists to
//! rule out.
//!
//! Skipped when no device is present, or when the device has no VMM support.

mod common;
use driver_cuda::gpu::device::{Allocator, Arena, Device, OwnedStream, PhysicalPool, pages_for_bytes};
use driver_cuda::gpu::device::LOGICAL_PAGE_BYTES;
use common::{device_or_skip, gpu_guard};

/// A pool sized well under the L40S's 45 GiB so the tests never compete with
/// whatever else is on the card.
const POOL_BYTES: usize = 512 * 1024 * 1024;

fn pool_or_skip(what: &str) -> Option<(Device, PhysicalPool)> {
    let dev = device_or_skip(what)?;
    match dev.supports_vmm() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("skipping {what}: device has no VMM support");
            return None;
        }
        Err(e) => {
            eprintln!("skipping {what}: {e}");
            return None;
        }
    }
    match PhysicalPool::new(dev.ordinal(), POOL_BYTES, PhysicalPool::DEFAULT_HANDLE_BYTES) {
        Ok(p) => Some((dev, p)),
        Err(e) => {
            eprintln!("skipping {what}: pool: {e}");
            None
        }
    }
}

/// Write a recognisable pattern through the arena's mapped range.
fn fill(arena: &Arena<'_>, bytes: usize, seed: u8, stream: &OwnedStream) {
    let pattern: Vec<u8> = (0..bytes).map(|i| seed.wrapping_add((i % 251) as u8)).collect();
    // SAFETY: `base()` is mapped for at least `bytes` -- every caller here has
    // just committed that much -- and the copy is synchronised before the
    // borrow ends.
    let code = unsafe {
        cuda_memcpy_h2d(arena.base(), pattern.as_ptr(), bytes)
    };
    assert!(code, "h2d into the arena failed");
    stream.as_ref().synchronize().expect("sync");
}

fn read_back(arena: &Arena<'_>, bytes: usize, stream: &OwnedStream) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    // SAFETY: same range that `fill` wrote.
    let code = unsafe { cuda_memcpy_d2h(out.as_mut_ptr(), arena.base(), bytes) };
    assert!(code, "d2h out of the arena failed");
    stream.as_ref().synchronize().expect("sync");
    out
}

fn expected(bytes: usize, seed: u8) -> Vec<u8> {
    (0..bytes).map(|i| seed.wrapping_add((i % 251) as u8)).collect()
}

// The crate does not expose a raw-address copy: `DeviceBuffer` owns its
// pointer, and an arena's storage is deliberately not a `DeviceBuffer`. Going
// straight to the runtime is the stronger test anyway -- it proves an arena's
// mapping is ordinary device memory that any CUDA call can address.
unsafe fn cuda_memcpy_h2d(dst: u64, src: *const u8, bytes: usize) -> bool {
    use driver_cuda::cudarc::runtime::sys as rt;
    // SAFETY: `dst` is mapped for `bytes`; `src` is valid for `bytes` reads.
    unsafe {
        rt::cudaMemcpy(
            dst as *mut std::ffi::c_void,
            src.cast::<std::ffi::c_void>(),
            bytes,
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ) == rt::cudaError::cudaSuccess
    }
}

unsafe fn cuda_memcpy_d2h(dst: *mut u8, src: u64, bytes: usize) -> bool {
    use driver_cuda::cudarc::runtime::sys as rt;
    // SAFETY: `src` is mapped for `bytes`; `dst` is valid for `bytes` writes.
    unsafe {
        rt::cudaMemcpy(
            dst.cast::<std::ffi::c_void>(),
            src as *const std::ffi::c_void,
            bytes,
            rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ) == rt::cudaError::cudaSuccess
    }
}

#[test]
fn an_arena_maps_memory_that_can_actually_be_written_and_read() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("arena round trip") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut arena = Arena::new(&pool, 64 << 20, "roundtrip").expect("arena");

    assert_eq!(arena.committed_bytes(), 0, "a fresh arena backs nothing");
    arena.ensure_committed(1 << 20).expect("commit 1 MiB");
    assert!(arena.committed_bytes() >= (1 << 20));

    fill(&arena, 1 << 20, 0x40, &stream);
    assert_eq!(read_back(&arena, 1 << 20, &stream), expected(1 << 20, 0x40));
}

/// The property the whole elastic design rests on: growing an arena must be
/// invisible to data already in it. If growth remapped, moved, or re-created
/// the existing range, every device pointer handed out before the growth would
/// silently become wrong -- and the KV cache hands out a great many.
#[test]
fn growth_preserves_everything_already_written() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("growth preserves data") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut arena = Arena::new(&pool, 128 << 20, "growth").expect("arena");

    let first = 4 << 20;
    arena.ensure_committed(first).expect("initial commit");
    let base_before = arena.base();
    fill(&arena, first, 0x11, &stream);

    // Grow several times, checking the original bytes after each step.
    for step in 1..=4usize {
        let target = first + step * (8 << 20);
        arena.ensure_committed(target).expect("grow");
        assert_eq!(arena.base(), base_before, "the base address moved on growth");
        assert_eq!(
            read_back(&arena, first, &stream),
            expected(first, 0x11),
            "data written before growth step {step} did not survive it"
        );
    }

    // And the newly grown region is usable too.
    let total = first + 4 * (8 << 20);
    fill(&arena, total, 0x22, &stream);
    assert_eq!(read_back(&arena, total, &stream), expected(total, 0x22));
}

#[test]
fn a_trim_releases_backing_and_a_regrow_reuses_the_cached_handle() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("trim and regrow") else { return };
    let mut arena = Arena::new(&pool, 256 << 20, "trim").expect("arena");
    let unit = arena.map_unit_bytes();

    arena.ensure_committed(unit * 4).expect("commit 4 units");
    assert_eq!(arena.committed_bytes(), unit * 4);
    let charged_at_peak = pool.budget().charged_pages();

    // Trim to two units, not one: the cache exists only for an arena that is
    // still holding at least two handles. An arena trimmed to one is on its
    // way out, and holding physical memory for it is the leak this cap exists
    // to prevent.
    arena.trim_committed(unit * 2).expect("trim to 2 units");
    assert_eq!(arena.committed_bytes(), unit * 2);
    assert_eq!(arena.cached_handle_count(), 1, "one handle should be cached against a regrow");
    let charged_after_trim = pool.budget().charged_pages();
    assert!(
        charged_after_trim < charged_at_peak,
        "a trim that releases handles must give budget back: {charged_after_trim} vs {charged_at_peak}"
    );

    // The cached handle is already charged, so regrowing by exactly one unit
    // must not charge again.
    let before = pool.budget().charged_pages();
    assert_eq!(arena.physical_growth_pages(unit * 3).expect("growth pages"), 0);
    arena.ensure_committed(unit * 3).expect("regrow");
    assert_eq!(pool.budget().charged_pages(), before, "regrow double-charged the pool");
    assert_eq!(arena.cached_handle_count(), 0, "the cached handle was not reused");
}

#[test]
fn trimming_to_nothing_keeps_no_cache_at_all() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("trim to zero") else { return };
    let mut arena = Arena::new(&pool, 256 << 20, "empty").expect("arena");
    let unit = arena.map_unit_bytes();

    arena.ensure_committed(unit * 3).expect("commit");
    arena.trim_committed(0).expect("trim to zero");

    assert_eq!(arena.committed_bytes(), 0);
    assert_eq!(
        arena.cached_handle_count(),
        0,
        "an arena being emptied must not hold physical memory hostage"
    );
    assert_eq!(pool.budget().charged_pages(), 0, "emptying an arena must return the whole charge");
}

#[test]
fn the_budget_actually_refuses_an_over_commit() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("budget refusal") else { return };
    // The arena's capacity is larger than the pool, so the refusal has to come
    // from the budget rather than from `target_committed_bytes`.
    let mut arena = Arena::new(&pool, POOL_BYTES * 4, "greedy").expect("arena");

    let err = arena.ensure_committed(POOL_BYTES * 2).expect_err("must refuse");
    assert!(format!("{err}").contains("budget"), "unexpected error: {err}");
    assert_eq!(arena.committed_bytes(), 0, "a refused commit must leave nothing mapped");
    assert_eq!(pool.budget().charged_pages(), 0, "a refused commit must leave nothing charged");

    // And the pool is still usable afterwards -- the failure path returned the
    // reservation rather than leaking it.
    arena.ensure_committed(4 << 20).expect("pool still usable after a refusal");
    assert!(arena.committed_bytes() >= (4 << 20));
}

#[test]
fn several_arenas_share_one_budget() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("shared budget") else { return };
    let mut a = Arena::new(&pool, POOL_BYTES, "a").expect("arena a");
    let mut b = Arena::new(&pool, POOL_BYTES, "b").expect("arena b");

    a.ensure_committed(POOL_BYTES / 2).expect("a takes half");
    let after_a = pool.budget().charged_pages();
    assert!(after_a > 0);

    // b can have the rest...
    b.ensure_committed(POOL_BYTES / 4).expect("b takes a quarter");
    assert!(pool.budget().charged_pages() > after_a, "b's commit was not charged");

    // ...but not more than the rest.
    b.ensure_committed(POOL_BYTES).expect_err("b must not exceed the shared budget");

    // a giving memory back lets b grow.
    a.trim_committed(0).expect("a releases");
    b.ensure_committed(POOL_BYTES / 2).expect("b grows into a's freed budget");
}

#[test]
fn dropping_an_arena_returns_its_charge_to_the_pool() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("drop returns budget") else { return };
    {
        let mut arena = Arena::new(&pool, 128 << 20, "temporary").expect("arena");
        arena.ensure_committed(64 << 20).expect("commit");
        assert!(pool.budget().charged_pages() > 0);
    }
    assert_eq!(
        pool.budget().charged_pages(),
        0,
        "a dropped arena leaked its charge; the pool is now permanently smaller"
    );

    // Prove it is really reusable, not merely accounted as such.
    let mut again = Arena::new(&pool, 128 << 20, "reuse").expect("arena");
    again.ensure_committed(64 << 20).expect("the freed memory is genuinely available again");
}

/// Repeated grow/trim cycles must not ratchet the charge upward. This is the
/// shape of the bug found while porting: caching unmapped handles without a
/// cap, while also reporting them as uncommitted, leaks budget on every
/// oscillation and only shows up after many cycles.
#[test]
fn oscillating_does_not_ratchet_the_budget() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("oscillation") else { return };
    let mut arena = Arena::new(&pool, 256 << 20, "oscillate").expect("arena");
    let unit = arena.map_unit_bytes();

    arena.ensure_committed(unit * 4).expect("initial");
    let steady = pool.budget().charged_pages();

    for cycle in 0..24 {
        arena.trim_committed(unit).expect("trim");
        arena.ensure_committed(unit * 4).expect("regrow");
        assert_eq!(
            pool.budget().charged_pages(),
            steady,
            "charge drifted after {cycle} grow/trim cycles"
        );
    }
}

#[test]
fn recalibrate_never_drops_below_what_is_already_charged() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("recalibrate") else { return };
    let mut arena = Arena::new(&pool, POOL_BYTES, "recal").expect("arena");
    arena.ensure_committed(POOL_BYTES / 2).expect("commit half");
    let charged = pool.budget().charged_pages();

    // Ask for a budget far below what is outstanding.
    pool.recalibrate(LOGICAL_PAGE_BYTES, 0, true);
    let budget = pool.budget();
    assert!(
        budget.budget_pages() >= charged,
        "recalibrate cut the budget below the {charged} pages already handed out \
         (now {})",
        budget.budget_pages()
    );
    // The mapping is untouched and still readable.
    assert_eq!(arena.committed_bytes(), POOL_BYTES / 2);
}

/// The arena's virtual reservation is deliberately larger than its capacity,
/// and the commit ceiling is enforced against capacity rather than against the
/// reservation.
#[test]
fn capacity_is_the_ceiling_not_the_virtual_reservation() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("capacity ceiling") else { return };
    let mut arena = Arena::new(&pool, 16 << 20, "capped").expect("arena");
    let err = arena.ensure_committed((16 << 20) + 1).expect_err("must refuse");
    assert!(format!("{err}").contains("exceeds capacity"), "unexpected error: {err}");
}

#[test]
fn pool_granularity_and_pages_agree_with_the_driver() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("granularity") else { return };
    let g = pool.allocation_granularity();
    assert!(g > 0 && g.is_power_of_two(), "implausible granularity {g}");
    assert_eq!(
        LOGICAL_PAGE_BYTES % g,
        0,
        "the 2 MiB accounting page must be a whole number of driver granules"
    );
    assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES * 3, LOGICAL_PAGE_BYTES), 3);
}

/// An arena and the ordinary caching allocator have to coexist: the store uses
/// the arena, everything else uses `cudaMalloc`, and they draw on the same
/// physical device.
#[test]
fn an_arena_coexists_with_ordinary_allocations() {
    let _gpu = gpu_guard();
    let Some((_dev, pool)) = pool_or_skip("coexistence") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let mut arena = Arena::new(&pool, 64 << 20, "coexist").expect("arena");

    arena.ensure_committed(8 << 20).expect("commit");
    fill(&arena, 8 << 20, 0x55, &stream);

    let mut buf = alloc.alloc(8 << 20).expect("ordinary alloc");
    buf.memset(0x99, stream.as_ref()).expect("memset");
    stream.as_ref().synchronize().expect("sync");

    assert_eq!(
        read_back(&arena, 8 << 20, &stream),
        expected(8 << 20, 0x55),
        "an ordinary allocation disturbed the arena's mapping"
    );
}
