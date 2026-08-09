//! Stream capture, graph instantiation, and conditional nodes, on hardware.
//!
//! The conditional-node half is here because of a claim made while choosing
//! the CUDA binding: that cudarc's raw `sys` layer exposes conditional graph
//! nodes fully, and that the only thing genuinely missing is
//! `cudaGraphSetConditional`, which is a `__device__` function and therefore
//! nvcc's job forever. Grepping the bindings supports the first half. Only a
//! GPU can settle it.
//!
//! Skipped when no device is present.

mod common;
use driver_cuda::cuda::{Allocator, DeviceBuffer, OwnedStream, StreamRef};
use common::{device_or_skip, gpu_guard};

const N: usize = 1 << 20;

fn read(buf: &DeviceBuffer, stream: StreamRef<'_>) -> Vec<u8> {
    let mut out = vec![0u8; N];
    buf.copy_to_host(&mut out, stream).expect("d2h");
    stream.synchronize().expect("sync");
    out
}

#[test]
fn a_captured_graph_replays_the_work_that_was_captured() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("graph capture") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();
    let mut buf = alloc.alloc(N).expect("alloc");

    buf.memset(0, stream.as_ref()).expect("clear");
    stream.as_ref().synchronize().expect("sync");

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    buf.memset(0x5a, scope.stream()).expect("memset under capture");
    let graph = scope.end().expect("end capture");

    // Capture records work; it does not perform it.
    assert!(
        read(&buf, stream.as_ref()).iter().all(|&b| b == 0),
        "capture executed the work instead of recording it"
    );

    let exec = graph.instantiate().expect("instantiate");
    exec.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");

    assert!(read(&buf, stream.as_ref()).iter().all(|&b| b == 0x5a), "the graph did not run");
}

#[test]
fn an_instantiated_graph_can_be_relaunched() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("graph relaunch") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();
    let mut buf = alloc.alloc(N).expect("alloc");

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    buf.memset(0x77, scope.stream()).expect("memset");
    let exec = scope.end().expect("end capture").instantiate().expect("instantiate");

    for round in 0..8 {
        buf.memset(0, stream.as_ref()).expect("clear");
        exec.launch(stream.as_ref()).expect("launch");
        stream.as_ref().synchronize().expect("sync");
        assert!(
            read(&buf, stream.as_ref()).iter().all(|&b| b == 0x77),
            "relaunch {round} did not take effect"
        );
    }
}

/// The whole point of the `Option<bool>` on `add_conditional_if`: with a
/// default assigned, the body's execution is decided by the host at build
/// time and needs no device-side set at all.
///
/// This is also the test that would have caught the flag bug. The earlier
/// version of the API passed the default value with `flags = 0`, so CUDA
/// ignored it -- `Some(true)` and `Some(false)` behaved identically, and
/// nothing but hardware could say so.
#[test]
fn a_conditional_body_runs_exactly_when_its_default_says_so() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("conditional default") else { return };

    for run_it in [false, true] {
        let stream = OwnedStream::new(0).expect("stream");
        let mut alloc = Allocator::new();
        let mut buf = alloc.alloc(N).expect("alloc");
        buf.memset(0, stream.as_ref()).expect("clear");
        stream.as_ref().synchronize().expect("sync");

        // Capture an empty graph, then attach the conditional node to it.
        let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
        let mut graph = scope.end().expect("end capture");

        let branch = graph.add_conditional_if(&[], Some(run_it)).expect("conditional node");
        // Populate the body by capturing into it.
        fill_body_with_memset(branch.body(), &buf, 0xc3);

        let exec = graph.instantiate().expect("instantiate");
        exec.launch(stream.as_ref()).expect("launch");
        stream.as_ref().synchronize().expect("sync");

        let out = read(&buf, stream.as_ref());
        if run_it {
            assert!(out.iter().all(|&b| b == 0xc3), "default=true did not run the body");
        } else {
            assert!(out.iter().all(|&b| b == 0), "default=false ran the body anyway");
        }
    }
}

/// Populate a conditional node's body graph with a single memset.
///
/// The body is a `cudaGraph_t` owned by the parent, so it is filled with the
/// explicit node API rather than by capture -- there is no stream to capture
/// into.
fn fill_body_with_memset(body: driver_cuda::cudarc::runtime::sys::cudaGraph_t, buf: &DeviceBuffer, v: u8) {
    use driver_cuda::cudarc::runtime::sys as rt;
    let mut params: rt::cudaMemsetParams = unsafe { std::mem::zeroed() };
    params.dst = buf.as_ptr();
    params.pitch = 0;
    params.value = u32::from(v);
    params.elementSize = 1;
    params.width = N;
    params.height = 1;

    let mut node: rt::cudaGraphNode_t = std::ptr::null_mut();
    // SAFETY: `body` is a live graph owned by the parent, and `params`
    // describes a region inside `buf`, which outlives the graph's use here.
    let code =
        unsafe { rt::cudaGraphAddMemsetNode(&raw mut node, body, std::ptr::null(), 0, &raw const params) };
    assert_eq!(code, rt::cudaError::cudaSuccess, "cudaGraphAddMemsetNode: {code:?}");
}

/// A conditional node with no default is the `supergraph.cu` shape: the host
/// contributes nothing and the predicate comes from the device. Without a
/// kernel to set it there is nothing to assert about *which* way it goes, so
/// this only pins that building, instantiating, and launching such a graph is
/// accepted by the driver.
#[test]
fn a_conditional_node_without_a_default_still_builds_and_launches() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("conditional without default") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();
    let buf = alloc.alloc(N).expect("alloc");

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    let mut graph = scope.end().expect("end capture");

    let branch = graph.add_conditional_if(&[], None).expect("conditional node");
    assert_ne!(branch.handle(), 0, "the handle is what a device kernel writes to");
    assert!(!branch.body().is_null());
    fill_body_with_memset(branch.body(), &buf, 0x01);

    let exec = graph.instantiate().expect("instantiate");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");
}

/// Two conditional nodes in one graph get distinct handles -- the supergraph
/// builds one per layer, and handle collisions would make them fire together.
#[test]
fn conditional_handles_are_distinct_per_node() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("distinct handles") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    let mut graph = scope.end().expect("end capture");

    let a = graph.add_conditional_if(&[], Some(true)).expect("first");
    let ha = a.handle();
    let b = graph.add_conditional_if(&[], Some(false)).expect("second");
    assert_ne!(ha, b.handle(), "two conditional nodes share one handle");
}

#[test]
fn allocations_made_before_a_capture_are_usable_inside_it() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("capture with prior allocation") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let mut a = alloc.alloc(N).expect("a");
    let mut b = alloc.alloc(N).expect("b");
    a.memset(0x11, stream.as_ref()).expect("seed a");
    stream.as_ref().synchronize().expect("sync");

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    b.memset(0x22, scope.stream()).expect("memset b");
    let exec = scope.end().expect("end").instantiate().expect("instantiate");

    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");

    assert!(read(&a, stream.as_ref()).iter().all(|&x| x == 0x11));
    assert!(read(&b, stream.as_ref()).iter().all(|&x| x == 0x22));
}

/// Freeing during capture is handled at runtime rather than forbidden by the
/// type system, because `Drop` can run anywhere. The buffer must survive to
/// the end of the capture and only then be released.
#[test]
fn a_buffer_dropped_during_capture_is_freed_after_it_ends() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("deferred free") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let doomed = alloc.alloc(N).expect("alloc");
    assert_eq!(alloc.deferred_free_count(), 0);

    let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
    drop(doomed);
    let graph = scope.end().expect("end capture");
    drop(graph);

    assert_eq!(
        alloc.deferred_free_count(),
        0,
        "the deferred free was not drained when the capture ended"
    );
    assert!(!alloc.is_capturing());

    // The allocator is still healthy afterwards.
    let _fresh = alloc.alloc(N).expect("allocator usable after a deferred free");
}
