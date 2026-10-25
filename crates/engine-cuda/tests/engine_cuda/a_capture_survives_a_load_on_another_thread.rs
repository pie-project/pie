//! What a load and a teardown do on one thread leaves a graph another
//! thread is capturing whole. The legacy stream waits on every blocking
//! stream of the context, and a capture on one is invalidated by anything
//! that reaches it (`cudaErrorStreamCaptureInvalidated`, 901), as it is by
//! a call that synchronizes the device: a buffer's synchronous copies and
//! sets once ran there, a graph's upload while weighing its instantiation
//! did, and a context's teardown destroyed its cuBLAS handles, which
//! synchronizes the device. A buffer kept for reuse is fenced on every
//! compute stream of its device, and an event recorded on one another
//! thread is capturing joins that capture, which the fence's query then
//! invalidates. Two models loaded at once, one arming its graphs while the
//! other uploaded, freed or dropped, met each.
#![cfg(feature = "cuda")]

use std::sync::{Arc, Barrier};

use engine_cuda::device::{Buffer, Context, Graph};

/// Runs `beside` on this thread while another holds a graph's capture
/// open on its context's compute stream, as a load arming its bodies does;
/// whether the capture ends whole.
fn capture_beside(beside: impl FnOnce()) -> bool {
    let open = Arc::new(Barrier::new(2));
    let done = Arc::new(Barrier::new(2));
    let (opened, finished) = (Arc::clone(&open), Arc::clone(&done));
    let capturing = std::thread::spawn(move || {
        let context = Context::bind(0, None).expect("device 0 binds");
        let mut cells = Buffer::zeroed(1 << 20).expect("a buffer");
        let stream = context.stream();
        let graph = Graph::capture(stream, || {
            cells.zero_span_on(stream, 0, 1 << 20)?;
            opened.wait();
            finished.wait();
            cells.zero_span_on(stream, 0, 1 << 20)
        });
        graph.is_ok()
    });
    open.wait();
    beside();
    done.wait();
    capturing.join().expect("the capturing thread ends")
}

#[test]
fn a_capture_survives_a_load_on_another_thread() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let cases: [(&str, fn()); 6] = [
        ("a buffer's copies and sets", || {
            let mut held = Buffer::zeroed(4 << 20).expect("a buffer");
            held.write(0, &[7u8; 1 << 20]).expect("the write");
            held.zero_span(0, 4096).expect("the set");
            let mut back = [0u8; 4096];
            held.read(1 << 12, &mut back).expect("the read");
            assert!(
                back.iter().all(|&byte| byte == 7),
                "the read lands the write"
            );
            engine_cuda::device::copy_d2h(held.ptr(), &mut back).expect("the copy");
            assert!(back.iter().all(|&byte| byte == 0), "the copy lands the set");
        }),
        ("a buffer dropped and its size asked for again", || {
            drop(Buffer::zeroed(96 << 20).expect("a buffer"));
            drop(Buffer::zeroed(96 << 20).expect("the kept one"));
        }),
        ("a heap cell dropped and its class asked for again", || {
            drop(Buffer::zeroed(8192).expect("a cell"));
            drop(Buffer::zeroed(8192).expect("the kept one"));
        }),
        ("a context bound and dropped", || {
            drop(Context::bind(0, None).expect("device 0 binds"));
        }),
        ("a context's side stream opened and dropped", || {
            let mut context = Context::bind(0, None).expect("device 0 binds");
            context.open_lanes(1, 1).expect("a side stream opens");
            drop(context);
        }),
        ("a graph's instantiation weighed", || {
            let context = Context::bind(0, None).expect("device 0 binds");
            let mut held = Buffer::zeroed(4096).expect("a buffer");
            let graph = engine_cuda::device::Graph::capture(context.stream(), || {
                held.zero_span_on(context.stream(), 0, 4096)
            })
            .expect("a graph");
            assert!(graph.instantiated_bytes(2).is_ok(), "the weighing answers");
        }),
    ];
    for (what, beside) in cases {
        assert!(
            capture_beside(beside),
            "{what} on one thread invalidated another's capture"
        );
    }
}
