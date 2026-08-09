//! The supergraph claim, on hardware.
//!
//! Every other test of this machinery is structural: the slot vocabulary
//! matches the trace's, the union lowering nests, the walk is a stack
//! diff. None of them settles the question the whole design rests on —
//! **can ONE instantiated graph run two different programs, chosen from
//! device memory, with no host round-trip and no recapture?**
//!
//! That is what this file asks. It captures a conditional whose two arms
//! write different bytes, instantiates ONCE, and then launches the same
//! exec twice with only the predicate word changed between them. If the
//! two launches produce different memory, the mechanism is real; if they
//! produce the same memory, everything above it is scaffolding over
//! nothing.
//!
//! Deliberately no kernel table: the arms are `cudaMemsetAsync`, which is
//! capturable and needs neither the bridge's launchers nor a model. What
//! is under test is the CONDITIONAL, not the work inside it.
//!
//! Skipped when no device is present.

#![cfg(all(feature = "_cuda", feature = "bridge"))]

mod common;

use common::{device_or_skip, gpu_guard};
use driver_cuda::cuda::{
    Allocator, DeviceBuffer, OwnedStream, PredicateWord, StreamRef, SupergraphBuilder,
    SLOT_HAS_LORA,
};

const N: usize = 256;
const IF_BYTE: u8 = 0xAA;
const ELSE_BYTE: u8 = 0xBB;
const NEITHER: u8 = 0x11;

fn read(buf: &DeviceBuffer, stream: StreamRef<'_>) -> Vec<u8> {
    let mut out = vec![0u8; N];
    buf.copy_to_host(&mut out, stream).expect("d2h");
    stream.synchronize().expect("sync");
    out
}

#[test]
fn one_exec_runs_two_programs_chosen_from_device_memory() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("supergraph conditional") else { return };

    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let mut preds = PredicateWord::new(&alloc).expect("predicate word");
    let mut out = alloc.alloc(N).expect("out");

    // Both arms must be REACHABLE from the capture, so the graph has to
    // be built once with both bodies populated. Which one runs is decided
    // later, per launch, by the word.
    let exec = {
        let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
        let mut b = SupergraphBuilder::new(scope.stream(), &preds);

        let cond = b.open_cond(SLOT_HAS_LORA, true).expect("open_cond");

        b.begin_body(cond.if_body()).expect("begin if");
        let body = b.stream();
        out.memset(IF_BYTE, body).expect("if arm");
        b.end_body().expect("end if");

        b.begin_body(cond.else_body().expect("else body")).expect("begin else");
        let body = b.stream();
        out.memset(ELSE_BYTE, body).expect("else arm");
        b.end_body().expect("end else");

        b.close_cond(&cond).expect("close_cond");
        drop(b);

        let graph = scope.end().expect("end capture");
        graph.instantiate().expect("instantiate")
    };

    // ── the predicate holds ────────────────────────────────────────────
    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(SLOT_HAS_LORA, true).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch true");
    stream.as_ref().synchronize().expect("sync");
    let took_if = read(&out, stream.as_ref());

    // ── the same exec, the predicate cleared ───────────────────────────
    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(SLOT_HAS_LORA, false).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch false");
    stream.as_ref().synchronize().expect("sync");
    let took_else = read(&out, stream.as_ref());

    assert!(
        took_if.iter().all(|&b| b == IF_BYTE),
        "the predicate held, so the if arm should have run; got {:#04x}",
        took_if[0]
    );
    assert!(
        took_else.iter().all(|&b| b == ELSE_BYTE),
        "the predicate was clear, so the else arm should have run; got {:#04x}",
        took_else[0]
    );
    assert_ne!(
        took_if, took_else,
        "ONE exec produced the same memory twice — the conditional is not \
         reading the device word, and every union above this is scaffolding \
         over nothing"
    );
}

#[test]
fn an_arm_that_is_not_taken_writes_nothing() {
    // The complement of the test above, and the one that catches a
    // conditional that always runs BOTH bodies -- which would still pass
    // an "the arms differ" check if the second write happened to land
    // last.
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("supergraph conditional") else { return };

    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let mut preds = PredicateWord::new(&alloc).expect("predicate word");
    let mut out = alloc.alloc(N).expect("out");

    // Only the IF arm writes; the else body is left empty. With the
    // predicate clear the buffer must keep whatever was primed into it.
    let exec = {
        let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
        let mut b = SupergraphBuilder::new(scope.stream(), &preds);
        let cond = b.open_cond(SLOT_HAS_LORA, true).expect("open_cond");
        b.begin_body(cond.if_body()).expect("begin if");
        let body = b.stream();
        out.memset(IF_BYTE, body).expect("if arm");
        b.end_body().expect("end if");
        b.close_cond(&cond).expect("close_cond");
        drop(b);
        scope.end().expect("end capture").instantiate().expect("instantiate")
    };

    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(SLOT_HAS_LORA, false).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");

    let got = read(&out, stream.as_ref());
    assert!(
        got.iter().all(|&b| b == NEITHER),
        "the predicate was clear, so nothing should have written; got {:#04x}",
        got[0]
    );
}

#[test]
fn nesting_holds_to_the_depth_a_guard_chain_needs() {
    // A guard CHAIN lowers to nested conditionals -- arm k runs when
    // predicates 0..k did not hold and k did -- so the builder's stream
    // pool has to work at depth, not just at the root. Two levels is
    // enough to prove the pool indexes by depth rather than reusing one
    // stream.
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("supergraph nesting") else { return };

    let stream = OwnedStream::new(0).expect("stream");
    let mut alloc = Allocator::new();

    let mut preds = PredicateWord::new(&alloc).expect("predicate word");
    let mut out = alloc.alloc(N).expect("out");

    let inner_slot = driver_cuda::cuda::SLOT_HAS_CUSTOM_MASK;

    let exec = {
        let scope = alloc.begin_capture(stream.as_ref()).expect("begin capture");
        let mut b = SupergraphBuilder::new(scope.stream(), &preds);
        assert_eq!(b.depth(), 0);

        let outer = b.open_cond(SLOT_HAS_LORA, true).expect("outer");
        b.begin_body(outer.if_body()).expect("begin outer if");
        assert_eq!(b.depth(), 1);

        let inner = b.open_cond(inner_slot, true).expect("inner");
        b.begin_body(inner.if_body()).expect("begin inner if");
        assert_eq!(b.depth(), 2);
        let body = b.stream();
        out.memset(IF_BYTE, body).expect("inner if arm");
        b.end_body().expect("end inner if");

        b.begin_body(inner.else_body().expect("inner else")).expect("begin inner else");
        let body = b.stream();
        out.memset(ELSE_BYTE, body).expect("inner else arm");
        b.end_body().expect("end inner else");

        b.close_cond(&inner).expect("close inner");
        b.end_body().expect("end outer if");
        assert_eq!(b.depth(), 0);
        b.close_cond(&outer).expect("close outer");
        drop(b);

        scope.end().expect("end capture").instantiate().expect("instantiate")
    };

    // outer on, inner on  -> IF_BYTE
    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(SLOT_HAS_LORA, true).expect("slot");
    preds.set(inner_slot, true).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");
    assert!(read(&out, stream.as_ref()).iter().all(|&b| b == IF_BYTE));

    // outer on, inner off -> ELSE_BYTE
    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(inner_slot, false).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");
    assert!(read(&out, stream.as_ref()).iter().all(|&b| b == ELSE_BYTE));

    // outer OFF -> the whole nest is skipped, whatever the inner says
    out.memset(NEITHER, stream.as_ref()).expect("prime");
    preds.set(SLOT_HAS_LORA, false).expect("slot");
    preds.set(inner_slot, true).expect("slot");
    preds.upload(stream.as_ref()).expect("upload");
    exec.launch(stream.as_ref()).expect("launch");
    stream.as_ref().synchronize().expect("sync");
    assert!(
        read(&out, stream.as_ref()).iter().all(|&b| b == NEITHER),
        "an outer arm that did not run must not let its inner arm run"
    );
}
