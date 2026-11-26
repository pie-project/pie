#![cfg(feature = "device")]
//! Memory a closed context gave back serves the next context of the
//! process: the first context's buffers go back to the kept stores behind
//! fences on its stream, the context closes, and a second context's
//! allocations take them. The close waited its stream out and let go of
//! every mark of it, so no take queries an event of a destroyed stream (on
//! ROCm that query blocked on a lock no thread held, and the take hung).

use std::collections::HashSet;
use std::sync::mpsc;
use std::time::Duration;

use engine_cuda::device::{Buffer, Context};

/// Sizes across the heap's classes and the recycler's.
const SIZES: [usize; 4] = [4 << 10, 64 << 10, 1_986_560, 4 << 20];

#[test]
fn a_closed_contexts_kept_memory_serves_the_next() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let first = Context::bind(0, None).expect("a device context");
        let given: HashSet<u64> = SIZES
            .iter()
            .flat_map(|&bytes| (0..4).map(move |_| bytes))
            .map(|bytes| Buffer::zeroed(bytes).expect("a buffer"))
            .collect::<Vec<Buffer>>()
            .into_iter()
            .map(|buffer| buffer.ptr())
            .collect();
        // The buffers above dropped behind fences on the first context's
        // stream; it closes with them kept.
        drop(first);
        let dead = engine_cuda::device::held::dead_marks();
        let second = Context::bind(0, None).expect("a second device context");
        let taken: Vec<Buffer> = SIZES
            .iter()
            .flat_map(|&bytes| (0..4).map(move |_| bytes))
            .map(|bytes| Buffer::zeroed(bytes).expect("a buffer"))
            .collect();
        let reused = taken
            .iter()
            .filter(|buffer| given.contains(&buffer.ptr()))
            .count();
        second
            .synchronize()
            .expect("the second context's stream drains");
        drop(taken);
        drop(second);
        done.send((reused, dead)).expect("the test waits");
    });
    let (reused, dead) = finished
        .recv_timeout(Duration::from_secs(120))
        .expect("the second context's allocations finish: none waits on the first's stream");
    assert_eq!(
        dead, 0,
        "the first context closed with marks of its stream still in kept fences"
    );
    assert!(
        reused > 0,
        "the second context took none of the memory the first gave back"
    );
}
