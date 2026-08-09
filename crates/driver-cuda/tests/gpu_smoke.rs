//! Smoke test: does this crate actually talk to a GPU?
//!
//! Skipped when no device is present: the crate is deliberately buildable and
//! testable on machines with no CUDA at all -- that is what the `cuda-12` /
//! `cuda-13` feature pair plus `fallback-dynamic-loading` buys, and a suite
//! that failed without hardware would throw it away.
//!
//! Run with `--nocapture` to see what was found.

use driver_cuda::device::{Allocator, Event, OwnedStream};

mod common;
use common::{device_or_skip, gpu_guard};

#[test]
fn a_device_can_be_bound_and_described() {
    let Some(dev) = device_or_skip("device query") else { return };
    let (major, minor) = dev.compute_capability().expect("compute capability");
    let sms = dev.sm_count().expect("sm count");
    let (free, total) = dev.memory_info().expect("memory info");
    let vmm = dev.supports_vmm().expect("vmm support");
    eprintln!(
        "device {}: sm_{major}{minor}, {sms} SMs, {} MiB free / {} MiB total, vmm={vmm}",
        dev.ordinal(),
        free / (1 << 20),
        total / (1 << 20)
    );
    assert!(major >= 5, "this crate targets Maxwell and later");
    assert!(sms > 0);
    assert!(total > 0 && free <= total);
}

#[test]
fn a_round_trip_through_device_memory_returns_what_went_in() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("memcpy round trip") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let mut buf = alloc.alloc(src.len()).expect("alloc");
    buf.copy_from_host(&src, stream.as_ref()).expect("h2d");

    let mut back = vec![0u8; src.len()];
    buf.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    assert_eq!(back, src);
}

#[test]
fn memset_reaches_the_device() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("memset") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let mut buf = alloc.alloc(1024).expect("alloc");
    buf.memset(0xab, stream.as_ref()).expect("memset");
    let mut back = vec![0u8; 1024];
    buf.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    assert!(back.iter().all(|&b| b == 0xab), "memset did not take");
}

#[test]
fn an_event_orders_work_across_two_streams() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("cross-stream event") else { return };
    let a = OwnedStream::new(0).expect("stream a");
    let b = OwnedStream::new(0).expect("stream b");
    let alloc = Allocator::new();

    let payload = vec![7u8; 1 << 20];
    let mut buf = alloc.alloc(payload.len()).expect("alloc");

    buf.copy_from_host(&payload, a.as_ref()).expect("h2d on a");
    let done = Event::new().expect("event");
    a.as_ref().record(&done).expect("record");
    b.as_ref().wait_event(&done).expect("wait");

    let mut back = vec![0u8; payload.len()];
    buf.copy_to_host(&mut back, b.as_ref()).expect("d2h on b");
    b.as_ref().synchronize().expect("sync b");

    assert_eq!(back, payload, "stream b read before stream a's write landed");
}

#[test]
fn a_timing_event_pair_measures_something_nonnegative() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("event timing") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let start = Event::with_timing().expect("start");
    let end = Event::with_timing().expect("end");

    let mut buf = alloc.alloc(1 << 22).expect("alloc");
    stream.as_ref().record(&start).expect("record start");
    buf.memset(1, stream.as_ref()).expect("memset");
    stream.as_ref().record(&end).expect("record end");
    end.synchronize().expect("sync");

    let ms = start.elapsed_ms(&end).expect("elapsed");
    assert!((0.0..10_000.0).contains(&ms), "implausible elapsed time {ms}ms");
}

/// The live `DeviceMemory` behind the sideband arena (retirement plan
/// phase B, first seam): real `cudaMalloc` under the arena's oracle-proven
/// growth discipline. The HOST semantics were pinned by the sideband oracle;
/// what this adds is that the live ops uphold the two claims the recorders
/// could only record — the address is stable while capacity suffices, and a
/// growth returns fresh usable memory.
#[test]
fn the_live_arena_carves_grows_and_frees_on_the_device() {
    use driver_cuda::fire::sideband_arena::{
        LiveDeviceMemory, Region, SidebandArena,
    };

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live sideband arena") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut mem = LiveDeviceMemory::new(stream.as_ref());
    let mut arena = SidebandArena::new();

    let base = arena.acquire(&mut mem, Region::Mask, 1024).expect("first acquire");
    assert!(!base.is_null());
    assert_eq!(arena.generation(), 1, "capacity 0 -> the first acquire grows");

    arena.release(Region::Mask);
    let again = arena.acquire(&mut mem, Region::Mask, 512).expect("re-acquire");
    assert_eq!(again, base, "while capacity suffices, the address is stable");

    arena.release(Region::Mask);
    let grown = arena
        .acquire(&mut mem, Region::Mask, 256 * 1024)
        .expect("growth acquire");
    assert!(!grown.is_null());
    assert_eq!(arena.generation(), 2, "a growth bumps the generation");

    arena.release(Region::Mask);
    arena.destroy(&mut mem);
}

/// The live `CublasOps` behind the ported handle (retirement plan phase B,
/// second seam): create binds the stream and tensor-op math the way the C++
/// constructor does, `get_stream` answers with the stream that was bound,
/// and a rebind moves it. The HOST discipline (ordering, the reproduced
/// constructor leak) is the cublas oracle's; this is the library saying yes.
#[test]
fn the_live_cublas_handle_binds_and_rebinds_its_stream() {
    use driver_cuda::device::cublas::{CublasHandle, LiveCublas};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live cublas handle") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw = stream.as_ref().as_raw().cast::<std::ffi::c_void>();

    let mut ops = LiveCublas;
    let mut h = CublasHandle::create(&mut ops, raw).expect("create");
    assert_eq!(h.stream(&mut ops), raw, "the bound stream reads back");

    h.set_stream(&mut ops, std::ptr::null_mut()).expect("rebind");
    assert!(h.stream(&mut ops).is_null(), "a null rebind lands");

    h.release(&mut ops);
}

/// The live `StagingOps` behind the attention workspace (retirement plan
/// phase B, third seam): real device scratch, a real pinned slot, and a
/// real upload fence. The rotation/fence DISCIPLINE is the attn_ws
/// oracle's; what this proves live is that a recorded event actually
/// retires and the slot survives a full begin -> write pin -> record ->
/// begin (sync) cycle on the device.
#[test]
fn the_live_workspace_pins_stages_and_fences_on_the_device() {
    use driver_cuda::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live attention workspace") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw = stream.as_ref().as_raw().cast::<std::ffi::c_void>();

    let mut ops = LiveStagingOps;
    let mut ws =
        AttentionWorkspace::allocate(&mut ops, 4096, 1024, 2).expect("allocate");
    let view = ws.view();
    assert!(!view.float_buffer.is_null() && !view.int_buffer.is_null());
    assert_eq!(view.float_bytes, 4096);

    // Two full staging cycles: the second begin lands on the second slot,
    // the third wraps to slot 0 and must SYNC its pending fence first —
    // the sync-before-reuse path, now against a real event.
    for _ in 0..3 {
        ws.begin_plan_update(&mut ops).expect("begin");
        let pin = ws.view().page_locked_int;
        assert!(!pin.is_null(), "begin pinned the active slot");
        unsafe { pin.cast::<u8>().write_bytes(0x5a, 16) };
        ws.end_plan_update(&mut ops, raw);
    }

    ws.release(&mut ops);
}
