#![cfg(feature = "device")]
//! A load lays its own buffers from its first, whatever its thread served
//! before: a worker loading a second model on the thread that served the
//! first measures its rates and binds its context unrefused, while a
//! buffer asked by serving stays refused
//! (`a_buffer_asked_after_ready_is_refused_by_name`).

use crate::common_encoder::{Rig, Weights, trace};
use engine_cuda::Graphs;
use engine_cuda::device::alloc;

#[test]
fn a_second_load_on_a_serving_thread_lays_its_own() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 13);
    let first = Rig::load(&weights, 64, vec![16, 32, 64], Graphs::On);
    // The thread serves the first load, as a worker's does.
    alloc::serving_from_here();
    drop(first);
    alloc::serving_from_here();
    let _second = Rig::load(&weights, 64, vec![16, 32, 64], Graphs::On);
}
