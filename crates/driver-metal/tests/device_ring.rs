//! The device-backed channel ring, against a real device.
//!
//! What is worth proving: that a fresh ring is empty and its buffers really
//! are device-visible, that the cell arithmetic tiles the ring modulo its
//! physical slot count, that a kernel advancing the words is observed by the
//! host exactly the way the shipped commit kernel does it, and that dropping
//! the ring releases its buffers rather than leaking them resident — the
//! failure `release_standalone_buffer` existed to invite.

#![allow(clippy::print_stdout)]

use driver_metal::pipeline::{Effect, Readiness, Reason, Ticket, check_words};
use driver_metal::{Compiler, Context, Error, Region, Ring, Stepper, Tables};
use objc2::Message;
use objc2::rc::{Weak, autoreleasepool};
use tensor_ir::DType;

/// The shipped commit kernel's shape of a word write: a plain device store.
const ADVANCE: &str = r"
kernel void advance(device ulong* words [[buffer(0)]],
                    device uint* cells [[buffer(1)]]) {
    cells[2] = 0xC0FFEEu;
    words[1] = words[1] + 1;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

#[test]
fn a_new_ring_is_empty_zeroed_and_addressable_by_both_sides() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let ring = Ring::new(&context, DType::U32, 4, 2).expect("ring");

    assert_eq!((ring.head(), ring.tail()), (0, 0));
    assert_eq!((ring.poison(), ring.closed()), (0, 0));
    assert_eq!(ring.capacity(), 2);
    assert_eq!(ring.cell_bytes(), 16, "four u32 lanes on the wire");
    // capacity + 1 slots: the pending slot a put writes before commit.
    assert_eq!(ring.cells().len(), 3 * 16);
    assert_ne!(ring.cells().gpu_address(), 0);
    assert_ne!(ring.words().gpu_address(), 0);

    // SAFETY: no GPU work names this fresh ring.
    let cells = unsafe {
        std::slice::from_raw_parts(
            ring.cells().contents().as_ptr().cast::<u8>(),
            ring.cells().len() as usize,
        )
    };
    assert!(
        cells.iter().all(|&b| b == 0),
        "a ring born with garbage cells serves garbage takes"
    );
}

#[test]
fn cells_tile_the_ring_modulo_the_physical_slot_count() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // capacity 2 -> cap1 = 3 physical slots.
    let ring = Ring::new(&context, DType::F32, 1, 2).expect("ring");
    let base = ring.cells().gpu_address();
    let cell = ring.cell_bytes() as u64;

    assert_eq!(ring.committed_cell(0).expect("slot").gpu_address(), base);
    assert_eq!(
        ring.committed_cell(1).expect("slot").gpu_address(),
        base + cell
    );
    assert_eq!(
        ring.committed_cell(2).expect("slot").gpu_address(),
        base + 2 * cell
    );
    // Sequence 3 wraps to slot 0: the modulo is cap1, not capacity.
    assert_eq!(ring.committed_cell(3).expect("slot").gpu_address(), base);
    assert_eq!(
        ring.pending_cell(5).expect("slot").gpu_address(),
        base + 2 * cell,
        "a pending cell is the same arithmetic at the tail sequence"
    );
}

#[test]
fn a_kernel_advancing_the_words_is_observed_by_the_host() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, ADVANCE, "advance")
        .expect("advance");
    let ring = Ring::new(&context, DType::U32, 1, 2).expect("ring");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, ring.words().gpu_address())
        .expect("bind words");
    tables
        .bind_address(&context, 0, 1, ring.cells().gpu_address())
        .expect("bind cells");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([1, 1, 1], [1, 1, 1])
        })
        .expect("the step ran");

    assert_eq!(ring.tail(), 1, "the kernel's put did not publish");
    assert_eq!(ring.head(), 0);
    // The pending cell the kernel wrote is slot 2 = sequence 2 of cap1 = 3.
    let cell = ring.pending_cell(2).expect("slot");
    // SAFETY: the step's fence signalled; one u32 wide.
    let value = unsafe { cell.contents().cast::<u32>().as_ptr().read() };
    assert_eq!(value, 0xC0FFEE);
}

#[test]
fn a_snapshot_feeds_the_same_check_the_interpreter_uses() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let ring = Ring::new(&context, DType::I32, 1, 1).expect("ring");
    let takes = Effect {
        requires_full: true,
        take: true,
        capacity: 1,
        ..Effect::default()
    };

    // Empty ring, a taker: early, not broken.
    assert_eq!(
        check_words(&[ring.snapshot()], &[takes], &[Ticket::default()]),
        Readiness::Retry {
            channel: 0,
            reason: Reason::Empty
        }
    );
}

#[test]
fn dropping_the_ring_releases_its_buffers_rather_than_leaking_them() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // Inside an autorelease pool, so a buffer parked in the pool by its own
    // creation cannot masquerade as a leak — the same discipline as
    // `device_pso.rs`.
    let (cells, words) = autoreleasepool(|_| {
        let ring = Ring::new(&context, DType::F32, 8, 4).expect("ring");
        let cells = Weak::from_retained(&ring.cells().buffer().retain());
        let words = Weak::from_retained(&ring.words().buffer().retain());
        assert!(cells.load().is_some() && words.load().is_some());
        drop(ring);
        (cells, words)
    });

    assert!(
        cells.load().is_none(),
        "the cell buffer outlived its ring: retained after removal from residency"
    );
    assert!(words.load().is_none(), "the words buffer outlived its ring");
}
