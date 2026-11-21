#![cfg(feature = "device")]
//! The rates a first boot measures (`rates::measure`, before the load reads
//! the card's room) leave the card as the step found it: none of its
//! buffers is a cell of the device heap, whose chunks the process keeps, and
//! none is left in the idle store. An L40S's first boot at a 40960 context
//! fit beside 7.5 GiB the step no longer held (2.5 GiB of heap chunks, the
//! rest idle) and found no rung with room for one sequence; the next boot,
//! reading the saved rates, measured nothing and served.

use engine_cuda::device::{Context, held};

#[test]
fn the_rates_step_gives_the_card_back() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    // The load's own context, bound before it measures (`serve::load`).
    let load = Context::bind(0, None).expect("a device context");
    let heap_before = held::heap_bytes();
    let measured = engine_cuda::rates::measure(0, &engine_cuda::chip::profile(""), None)
        .expect("the rates measure");
    assert!(
        measured.rates.launch_ns.is_some(),
        "the step measured the launch"
    );
    assert_eq!(
        held::heap_bytes(),
        heap_before,
        "the step carved its buffers from the device heap, whose chunks stay with the process"
    );
    assert_eq!(
        held::kept_bytes(),
        0,
        "the step's dropped buffers stayed in the idle store past it"
    );
    drop(load);
}
