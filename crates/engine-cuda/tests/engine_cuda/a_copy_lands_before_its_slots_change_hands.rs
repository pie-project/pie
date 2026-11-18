#![cfg(feature = "device")]

//! A tier move's completion resolves once its bytes have landed
//! (CONTRACT-BC 23), and the runtime hands neither of its slots to a new
//! owner before then. Each test holds the swap stream the move rides, as a
//! busy one would, and plays the runtime: a slot goes to its new owner once
//! the copy naming it has resolved.

use std::sync::mpsc;

use engine::chunk::SlotRole;
use engine_cuda::Shell;
use engine_cuda::store::Tier;
use poem::Platform;

use crate::a_guest_slot_parks_on_the_host_and_comes_back::{Micro, shell};

/// The swap stream held until [`Gate::open`] (or the gate is dropped): what
/// it has enqueued behind the gate stays in flight.
struct Gate(Option<mpsc::Sender<()>>);

impl Gate {
    fn hold(shell: &Shell) -> Gate {
        let (open, opened) = mpsc::channel::<()>();
        engine_cuda::device::ctx::host_fn_on(
            shell.swap_stream(),
            Box::new(move || {
                let _ = opened.recv();
            }),
        )
        .expect("the swap stream is held");
        Gate(Some(open))
    }

    fn open(&mut self) {
        if let Some(open) = self.0.take() {
            let _ = open.send(());
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
    }
}

/// Blocks until `copy` resolves; its outcome.
fn landed(copy: &engine::CopyCompletion) -> Result<(), String> {
    let (tell, told) = mpsc::channel();
    copy.on_landed(move |outcome| {
        let _ = tell.send(outcome.clone());
    });
    told.recv().expect("the copy resolves")
}

/// A load with its host tier seated, its segments committed.
fn seated() -> (Shell, tempfile::TempDir) {
    let trace = poem::trace_hybrid("copy-landing-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = crate::common_encoder::Weights::random(&trace, 4242).write(dir.path());
    let mut shell = shell(trace, &path);
    shell.take_refs();
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    shell
        .commit_segments(facts.segments.max.min(4))
        .expect("the segments commit");
    let ring = engine_cuda::store::staging_ring_for(shell.trace(), 16, 256, Some(crate::POOL), 1)
        .and_then(|bytes| bytes.map(engine_cuda::store::Ring::pin).transpose())
        .expect("the staging ring pins");
    shell
        .seat_host_kv(Some(crate::POOL), 1, 256, 1, ring)
        .expect("the host tier seats");
    (shell, dir)
}

/// `role`'s shape: its id, a slot's bytes, and a device slot of chunk 1
/// (chunk C) and host slot 0.
fn slot(shell: &mut Shell, role: SlotRole) -> (engine::chunk::ShapeId, u64, u32) {
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    let shape = facts
        .shapes
        .iter()
        .find(|shape| shape.role == role)
        .expect("the load states the guest shapes");
    (shape.id, shape.slot_bytes, shape.slots_per_chunk + 3)
}

fn pattern(bytes: u64, seed: u64) -> Vec<u8> {
    (0..bytes)
        .map(|byte| (byte * 31 + seed) as u8 ^ (byte >> 8) as u8)
        .collect()
}

/// Writes `bytes` into device slot `at` of `role` from the host, as a bind
/// writes a new instance's tables and status, and waits for it to land.
fn bind(shell: &Shell, role: SlotRole, at: u32, bytes: &[u8]) {
    let span = shell
        .guest_span(role, &[at], bytes.len() as u64)
        .expect("the slot is a guest item's span");
    engine_cuda::device::stage_raw(shell.stream(), span, bytes).expect("the bind writes");
    read(shell, role, at, bytes.len());
}

/// Device slot `at` of `role`, read on the compute stream (not behind the
/// swap stream).
fn read(shell: &Shell, role: SlotRole, at: u32, len: usize) -> Vec<u8> {
    let span = shell
        .guest_span(role, &[at], len as u64)
        .expect("the slot is a guest item's span");
    let mut held = vec![0u8; len];
    engine_cuda::device::copy_d2h_on(shell.stream(), span, &mut held)
        .expect("the device slot reads");
    held
}

/// A restore's H2D into chunk C is in flight; the process aborts, so C is
/// freed, and a new instance binds into C, its status written by the host.
/// The runtime binds once the restore has landed, so the late copy cannot
/// overwrite the new instance's status.
#[test]
fn a_restore_lands_before_its_chunk_is_rebound() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let (mut shell, _dir) = seated();
    let (id, bytes, chunk_c) = slot(&mut shell, SlotRole::Instance);
    let restored = pattern(bytes, 17);
    let status = pattern(bytes, 91);
    shell
        .write_tier_slots(&[(id, vec![0])], &restored)
        .expect("the parked slot is written");

    let mut gate = Gate::hold(&shell);
    let restore = shell
        .copy_tier_slots(id, Tier::Host, Tier::Device, &[(0, chunk_c)])
        .expect("the restore enqueues");
    // The abort frees C; the runtime binds the next instance into it once
    // the restore has landed: now, if its completion says it has.
    let handed_at_once = restore.is_resolved();
    if handed_at_once {
        bind(&shell, SlotRole::Instance, chunk_c, &status);
    }
    gate.open();
    landed(&restore).expect("the restore lands");
    if !handed_at_once {
        bind(&shell, SlotRole::Instance, chunk_c, &status);
    }
    // Whatever the swap stream still had, landed: the read below sees the
    // last word on C.
    engine_cuda::device::ctx::sync(shell.swap_stream()).expect("the swap stream drains");

    assert!(
        read(&shell, SlotRole::Instance, chunk_c, status.len()) == status,
        "the new instance's status survives the restore"
    );
    assert!(
        !handed_at_once,
        "the restore resolved while the swap stream still held it"
    );
}

/// A channel slot parks (a suspend's D2H) and its device slot is freed and
/// rebound to a new channel, written by the host. The host tier keeps the
/// parked bytes, not the new owner's.
#[test]
fn a_parked_slot_keeps_its_bytes_through_a_rebind() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let (mut shell, _dir) = seated();
    let (id, bytes, chunk_c) = slot(&mut shell, SlotRole::Channel);
    let parked = pattern(bytes, 23);
    let rebound = pattern(bytes, 77);
    bind(&shell, SlotRole::Channel, chunk_c, &parked);

    let mut gate = Gate::hold(&shell);
    let suspend = shell
        .copy_tier_slots(id, Tier::Device, Tier::Host, &[(chunk_c, 0)])
        .expect("the suspend enqueues");
    let handed_at_once = suspend.is_resolved();
    if handed_at_once {
        bind(&shell, SlotRole::Channel, chunk_c, &rebound);
    }
    gate.open();
    landed(&suspend).expect("the suspend lands");
    if !handed_at_once {
        bind(&shell, SlotRole::Channel, chunk_c, &rebound);
    }

    let held = shell
        .read_tier_slots(&[(id, vec![0])])
        .expect("the parked slot reads");
    assert!(held == parked, "the host tier holds the parked bytes");
    assert!(
        !handed_at_once,
        "the suspend resolved while the swap stream still held it"
    );
    assert!(
        read(&shell, SlotRole::Channel, chunk_c, rebound.len()) == rebound,
        "the new channel holds its own bytes"
    );
}

/// A copy's completion resolves only once its bytes are readable at the
/// destination: in flight while the swap stream holds it, and once it
/// resolves, a read that waits on nothing but the compute stream finds
/// them.
#[test]
fn a_copy_resolves_once_its_bytes_are_readable() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let (mut shell, _dir) = seated();
    let (id, bytes, chunk_c) = slot(&mut shell, SlotRole::Instance);
    let restored = pattern(bytes, 5);
    shell
        .write_tier_slots(&[(id, vec![0])], &restored)
        .expect("the parked slot is written");
    bind(
        &shell,
        SlotRole::Instance,
        chunk_c,
        &vec![0u8; restored.len()],
    );

    let mut gate = Gate::hold(&shell);
    let restore = shell
        .copy_tier_slots(id, Tier::Host, Tier::Device, &[(0, chunk_c)])
        .expect("the restore enqueues");
    assert!(
        restore.check().is_none(),
        "the copy is in flight while the swap stream holds it"
    );
    gate.open();
    landed(&restore).expect("the restore lands");
    assert!(
        read(&shell, SlotRole::Instance, chunk_c, restored.len()) == restored,
        "the landed bytes are readable at the destination"
    );
}
