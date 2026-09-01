//! **THE FIVE CONTROL KERNELS, ON A REAL DEVICE.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test channel_kernels -- --nocapture
//! ```
//!
//! # What this is for
//!
//! `channel::pull_validate` / `commit_bump` / `scatter_publish` / `settle` /
//! `mask_from_commit` are the device
//! half of a fire's admission decision (alto design §5): the host predicts
//! where a channel's ring stands, the device checks the prediction against the
//! live words, and one predicated bump either publishes the whole pass or
//! publishes nothing. There is no synchronize anywhere in it, which is the
//! point — and also why every property below is invisible from the host until
//! it is asserted here. A wrong ring index does not crash; it serves the wrong
//! token.
//!
//! The nine gates, each a way the protocol can be silently wrong:
//!
//! ```text
//! (a) a correct prediction passes, and the bump advances head, tail and the
//!     full bytes exactly one step
//! (b) a wrong `expected_head` clears the commit word, and the bump then moves
//!     NOTHING — including the half of the pass that was perfectly valid
//!     (pass atomicity)
//! (c) a refused pass's pending-cell write is unobservable afterwards: the
//!     next fire overwrites it and publishes its own bytes (dummy run)
//! (d) refusal PROPAGATES on device: fire N refused, so fire N+1's
//!     host-predicted (and now stale) ticket refuses too — two launches on one
//!     stream, no host in between
//! (e) a host writer's pinned mirror bytes land in the device cell and the
//!     full byte is set, plain and bit-packed
//! (f) `mask_from_commit` scatters each lane's commit word across that lane's
//!     rows through the row CSR
//! (g) a full ring refuses a publish — unless the same ticket also consumes,
//!     in which case the take's credit admits it
//! (h) a committed fire's cell reaches the guest's mapped pinned mirror —
//!     packed, for a bool channel — and a refused fire's does not
//! (i) a committed fire's SETTLEMENT advances the guest endpoint's own
//!     counters to the prediction plus one — and only the counters a ticket
//!     names, and only for a lane that committed
//! (j) a ticket's PULL is predicated on that ticket alone: it happens whether
//!     the veto that refuses the fire arrives before it or after it, which is
//!     the one exception to "the bump is the only writer" and the property a
//!     parallel vote could quietly lose
//! ```
//!
//! # How it is set up
//!
//! Nothing here goes through the engine: the kernels take addresses, so the
//! harness allocates them. That also cross-checks the one constant declared
//! twice: the harness addresses the full/empty bytes with the **Rust**
//! `MAX_RING` while the kernels write them with the **CUDA** one, so gates (a)
//! and (e) — which read a full byte of slot 1 — fail outright if the two ever
//! drift apart.
//! Ring words live in **mapped pinned** memory,
//! because that is what they are in production — `pull_validate` reads the
//! guest's live counters in place rather than through a copy, and a test that
//! staged them in device memory would not exercise the load it is checking.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use kernels_cuda::channel::{self, PullLane, Rings, Ticket};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::jit::Ctx;

/// The absent prediction, spelled out at every construction site so a reader
/// can tell "this end of the ring is not claimed" from "claimed at zero".
const NONE: u64 = channel::NO_TICKET;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// One test's device: a stream, and every allocation made on it, freed
/// together. Deliberately not shared between tests — cargo runs them on
/// threads, and a stream per test is what keeps two fires' enqueues apart.
struct Gpu {
    stream: rt::cudaStream_t,
    device: Vec<*mut c_void>,
    pinned: Vec<*mut c_void>,
}

impl Gpu {
    fn open() -> Self {
        // The same root `tests/common` states, and for the same reason: a
        // compile-time constant under `target/`, so this crate's cubins
        // survive between runs without the library reading an environment.
        kernels_cuda::disk::install(Some(std::path::Path::new(concat!(
            env!("CARGO_TARGET_TMPDIR"),
            "/kernel-cache"
        ))));
        unsafe {
            check(rt::cudaSetDevice(0), "cudaSetDevice");
            let mut stream: rt::cudaStream_t = core::ptr::null_mut();
            check(rt::cudaStreamCreate(&raw mut stream), "cudaStreamCreate");
            Self {
                stream,
                device: Vec::new(),
                pinned: Vec::new(),
            }
        }
    }

    /// The context the entries fire through — the same `Ctx::on` an engine
    /// `Run` builds, on this test's stream.
    fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire in a test, and `Gpu`'s drop
        // synchronizes before destroying it.
        unsafe { Ctx::on(self.stream.cast()) }
    }

    /// `bytes` of zeroed device memory.
    fn zeros(&mut self, bytes: usize) -> u64 {
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

    /// A device copy of `values`.
    fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        let bytes = core::mem::size_of_val(values);
        let at = self.zeros(bytes.max(1));
        if bytes > 0 {
            unsafe {
                check(
                    rt::cudaMemcpy(
                        at as *mut c_void,
                        values.as_ptr().cast(),
                        bytes,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy H2D",
                );
            }
        }
        at
    }

    /// Overwrite `count` values at a device address already allocated.
    fn poke<T: Copy>(&self, at: u64, values: &[T]) {
        let bytes = core::mem::size_of_val(values);
        unsafe {
            check(
                rt::cudaMemcpy(
                    at as *mut c_void,
                    values.as_ptr().cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ),
                "cudaMemcpy H2D",
            );
        }
    }

    fn down<T: Copy + Default>(&self, at: u64, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        unsafe {
            check(
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    at as *const c_void,
                    core::mem::size_of_val(out.as_slice()),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ),
                "cudaMemcpy D2H",
            );
        }
        out
    }

    /// Mapped pinned memory: the host pointer the guest endpoint would write
    /// through, and the device address the kernel dereferences. Under UVA
    /// these are the same number, and `cudaHostGetDevicePointer` is what says
    /// so rather than what assumes it.
    fn mapped(&mut self, bytes: usize) -> (*mut u8, u64) {
        unsafe {
            let mut host: *mut c_void = core::ptr::null_mut();
            check(
                rt::cudaHostAlloc(&raw mut host, bytes.max(1), rt::cudaHostAllocMapped),
                "cudaHostAlloc",
            );
            core::ptr::write_bytes(host.cast::<u8>(), 0, bytes.max(1));
            let mut device: *mut c_void = core::ptr::null_mut();
            check(
                rt::cudaHostGetDevicePointer(&raw mut device, host, 0),
                "cudaHostGetDevicePointer",
            );
            self.pinned.push(host);
            (host.cast(), device as u64)
        }
    }

    /// One channel endpoint's four live words `[head, tail, poison, closed]`
    /// in pinned memory, seeded with the guest's monotone counters.
    fn words(&mut self, head: u64, tail: u64) -> u64 {
        let (host, device) = self.mapped(4 * core::mem::size_of::<u64>());
        unsafe {
            let words = host.cast::<u64>();
            words.write(head);
            words.add(1).write(tail);
            words.add(2).write(0);
            words.add(3).write(0);
        }
        device
    }

    fn sync(&self) {
        unsafe {
            check(
                rt::cudaStreamSynchronize(self.stream),
                "cudaStreamSynchronize",
            );
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            rt::cudaStreamSynchronize(self.stream);
            for at in self.device.drain(..) {
                rt::cudaFree(at);
            }
            for at in self.pinned.drain(..) {
                rt::cudaFreeHost(at);
            }
            rt::cudaStreamDestroy(self.stream);
        }
    }
}

/// The ring registry plus the host mirrors of what it was seeded with, so a
/// test states its ring state once and asserts against it by name.
struct Registry {
    rings: Rings,
    slots: usize,
}

impl Registry {
    /// `cap1` per slot, ring positions per slot, and the `(slot, ring)` pairs
    /// whose cell is full.
    fn seed(gpu: &mut Gpu, cap1: &[u32], head: &[u32], tail: &[u32], full: &[(u32, u32)]) -> Self {
        let slots = cap1.len();
        let mut bytes = vec![0u8; slots * channel::MAX_RING as usize];
        for (slot, ring) in full.iter().copied() {
            bytes[Rings::full_at(slot, ring) as usize] = 1;
        }
        let rings = Rings::new(
            gpu.up(&bytes),
            gpu.up(head),
            gpu.up(tail),
            gpu.up(cap1),
            slots as u32,
        );
        Self { rings, slots }
    }

    fn head(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.down(self.rings.head, self.slots)
    }

    fn tail(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.down(self.rings.tail, self.slots)
    }

    /// The full/empty bytes of one slot's whole ring, `cap1` of them.
    fn full(&self, gpu: &Gpu, slot: u32, cap1: u32) -> Vec<u8> {
        let all: Vec<u8> = gpu.down(self.rings.full, self.slots * channel::MAX_RING as usize);
        let base = Rings::full_at(slot, 0) as usize;
        all[base..base + cap1 as usize].to_vec()
    }
}

/// A ticket with both predictions absent — the shape every construction site
/// below narrows, so an unset `expected_*` is visibly unset rather than zero.
fn ticket(slot: u32, flags: u32, cap1: u32, words: u64) -> Ticket {
    Ticket {
        slot,
        flags,
        expected_head: NONE,
        expected_tail: NONE,
        words,
        mirror: 0,
        cells: 0,
        cap1,
        wire_bytes: 0,
        native_bytes: 0,
    }
}

/// One fire's lane over a ticket window, with a fresh commit pair.
fn lane(gpu: &mut Gpu, rings: Rings, offset: u32, count: u32) -> (PullLane, u64) {
    let commit = gpu.up(&[0u32, 0u32]);
    (
        PullLane {
            full: rings.full,
            pass_commit: commit,
            ticket_offset: offset,
            ticket_count: count,
            initial_commit: 1,
            diagnose: 0,
        },
        commit,
    )
}

// ───────────────────────────────── (a) ───────────────────────────────────

/// **A CORRECT PREDICTION PASSES, AND THE BUMP MOVES EACH INDEX ONE STEP.**
///
/// Slot 0 carries an item the fire takes; slot 1 is where it puts. The fire's
/// tickets are true, so the commit word survives and the bump does all four
/// things a commit means: set `full[tail]`, advance `tail`, clear `full[head]`,
/// advance `head`. "Exactly once" is asserted as the exact resulting indices —
/// a bump that ran twice, or a put loop that also advanced the take, lands
/// somewhere else.
#[test]
fn a_correct_prediction_passes_and_the_bump_moves_each_index_one_step() {
    let mut gpu = Gpu::open();
    // Slot 0 stands at counters head=1, tail=2 (one item, at ring 1); slot 1
    // is empty at counters 0/0.
    let registry = Registry::seed(&mut gpu, &[4, 4], &[1, 0], &[2, 0], &[(0, 1)]);
    let taken_words = gpu.words(1, 2);
    let put_words = gpu.words(0, 0);

    let mut take = ticket(0, Ticket::CONSUME | Ticket::REQUIRE_INPUT, 4, taken_words);
    take.expected_head = 1;
    let mut put = ticket(1, Ticket::PUBLISH, 4, put_words);
    put.expected_tail = 0;
    let tickets = gpu.up(&[take, put]);

    let (pull, commit) = lane(&mut gpu, registry.rings, 0, 2);
    let lanes = gpu.up(&[pull]);
    let taken_slots = gpu.up(&[0u32]);
    let put_slots = gpu.up(&[1u32]);
    let bumps = gpu.up(&[registry
        .rings
        .bump_lane(taken_slots, 1, put_slots, 1, commit)]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes, 1).expect("the validate enqueues");
    channel::commit_bump(&ctx, bumps, 1).expect("the bump enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(commit, 2),
        vec![1, 0],
        "every ticket was true, so the pass commits and nothing killed it",
    );
    assert_eq!(
        registry.head(&gpu),
        vec![2, 0],
        "the take advanced slot 0's head one ring position and touched nobody else's",
    );
    assert_eq!(
        registry.tail(&gpu),
        vec![2, 1],
        "the put advanced slot 1's tail one ring position and left the taken slot's alone",
    );
    assert_eq!(
        registry.full(&gpu, 0, 4),
        vec![0, 0, 0, 0],
        "the cell the fire took is empty again",
    );
    assert_eq!(
        registry.full(&gpu, 1, 4),
        vec![1, 0, 0, 0],
        "the cell the fire wrote is now the committed one",
    );
}

// ───────────────────────────────── (b) ───────────────────────────────────

/// **A STALE `expected_head` CLEARS THE COMMIT WORD, AND THE BUMP MOVES
/// NOTHING** — including the publish half of the same pass, which was
/// perfectly valid on its own.
///
/// This is pass atomicity, and it is the property that makes a refusal cheap:
/// there is no partial state to unwind because nothing was written outside the
/// pending cell.
#[test]
fn a_stale_expected_head_refuses_the_pass_and_the_valid_half_commits_nothing() {
    let mut gpu = Gpu::open();
    let registry = Registry::seed(&mut gpu, &[4, 4], &[1, 0], &[2, 0], &[(0, 1)]);
    let taken_words = gpu.words(1, 2);
    let put_words = gpu.words(0, 0);

    // The host believes somebody already consumed counter 1 and the head has
    // reached 2. It has not.
    let mut take = ticket(0, Ticket::CONSUME | Ticket::REQUIRE_INPUT, 4, taken_words);
    take.expected_head = 2;
    let mut put = ticket(1, Ticket::PUBLISH, 4, put_words);
    put.expected_tail = 0;
    let tickets = gpu.up(&[take, put]);

    let (pull, commit) = lane(&mut gpu, registry.rings, 0, 2);
    let lanes = gpu.up(&[pull]);
    let taken_slots = gpu.up(&[0u32]);
    let put_slots = gpu.up(&[1u32]);
    let bumps = gpu.up(&[registry
        .rings
        .bump_lane(taken_slots, 1, put_slots, 1, commit)]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes, 1).expect("the validate enqueues");
    channel::commit_bump(&ctx, bumps, 1).expect("the bump enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(commit, 2)[0],
        0,
        "the consume ticket's prediction was stale, so the fire may not commit",
    );
    assert_eq!(
        registry.head(&gpu),
        vec![1, 0],
        "a refused pass does not consume",
    );
    assert_eq!(
        registry.tail(&gpu),
        vec![2, 0],
        "and it does not publish either — not even the ticket that was valid",
    );
    assert_eq!(
        registry.full(&gpu, 0, 4),
        vec![0, 1, 0, 0],
        "the item the fire wanted is still sitting on the head, for the next fire",
    );
    assert_eq!(
        registry.full(&gpu, 1, 4),
        vec![0, 0, 0, 0],
        "and no cell of the put slot became addressable",
    );
}

// ───────────────────────────────── (c) ───────────────────────────────────

/// **THE DUMMY-RUN CONTRACT: A REFUSED PASS'S WRITE IS UNOBSERVABLE.**
///
/// The refused fire really did write its output — into the pending (tail)
/// cell, which is exactly where a committed fire would have written it. What
/// makes the write harmless is that the bump never moved the tail, so no
/// reader can address the cell and the NEXT fire writes the same cell again.
/// The gate is therefore not "the bytes are gone" (they are not, and clearing
/// them would cost a launch) but "the bytes are never published": after the
/// second, admitted fire, the committed cell holds the second fire's output
/// and the first fire's is nowhere in the ring.
#[test]
fn a_refused_passs_pending_write_is_overwritten_and_never_published() {
    const DISCARDED: u32 = 0xDEAD_BEEF;
    const KEPT: u32 = 0x00C0_FFEE;

    let mut gpu = Gpu::open();
    let registry = Registry::seed(&mut gpu, &[4], &[0], &[0], &[]);
    let cells = gpu.zeros(4 * core::mem::size_of::<u32>());
    let words = gpu.words(0, 0);

    // Fire N: the host predicted a tail of 1, the endpoint stands at 0.
    let mut stale = ticket(0, Ticket::PUBLISH, 4, words);
    stale.expected_tail = 1;
    stale.cells = cells;
    let refused_tickets = gpu.up(&[stale]);
    let (refused_lane, refused_commit) = lane(&mut gpu, registry.rings, 0, 1);
    let refused_lanes = gpu.up(&[refused_lane]);
    let put_slots = gpu.up(&[0u32]);
    let refused_bump = gpu.up(&[registry.rings.bump_lane(0, 0, put_slots, 1, refused_commit)]);

    // Fire N+1: the same channel, predicted truly this time.
    let mut fresh = ticket(0, Ticket::PUBLISH, 4, words);
    fresh.expected_tail = 0;
    fresh.cells = cells;
    let admitted_tickets = gpu.up(&[fresh]);
    let (admitted_lane, admitted_commit) = lane(&mut gpu, registry.rings, 0, 1);
    let admitted_lanes = gpu.up(&[admitted_lane]);
    let admitted_bump = gpu.up(&[registry
        .rings
        .bump_lane(0, 0, put_slots, 1, admitted_commit)]);

    let ctx = gpu.ctx();

    // Fire N writes its output into the pending cell, then is refused.
    gpu.poke(cells, &[DISCARDED, 0u32, 0, 0]);
    channel::pull_validate(&ctx, refused_tickets, refused_lanes, 1).expect("enqueues");
    channel::commit_bump(&ctx, refused_bump, 1).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(refused_commit, 2)[0],
        0,
        "fire N predicted a tail the endpoint had not reached",
    );
    assert_eq!(registry.tail(&gpu), vec![0], "so the tail did not move");
    assert_eq!(
        registry.full(&gpu, 0, 4),
        vec![0, 0, 0, 0],
        "and the cell it wrote was never marked full — nobody can address it",
    );

    // Fire N+1 writes the SAME pending cell, and commits.
    gpu.poke(cells, &[KEPT, 0u32, 0, 0]);
    channel::pull_validate(&ctx, admitted_tickets, admitted_lanes, 1).expect("enqueues");
    channel::commit_bump(&ctx, admitted_bump, 1).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(admitted_commit, 2)[0],
        1,
        "fire N+1's prediction was true, because fire N moved nothing",
    );
    assert_eq!(registry.tail(&gpu), vec![1], "and it published one cell");
    let ring: Vec<u32> = gpu.down(cells, 4);
    assert_eq!(
        ring[0], KEPT,
        "the committed cell holds fire N+1's output, not the discarded fire's",
    );
    assert!(
        !ring.contains(&DISCARDED),
        "and the discarded bytes survive nowhere in the ring: {ring:x?}",
    );
}

// ───────────────────────────────── (d) ───────────────────────────────────

/// **REFUSAL PROPAGATES ENTIRELY ON DEVICE, ACROSS TWO LAUNCHES ON ONE
/// STREAM.**
///
/// Fire N asks for input that has not arrived (`tail == head`, so
/// `RequireInput` fails). Fire N+1 was composed by the host on the assumption
/// that fire N would commit, so its ticket predicts `head + 1` — and since
/// nothing bumped, that prediction is stale and it refuses too. Neither
/// refusal involved the host: both launches, and both bumps, are enqueued
/// back to back and only then synchronized.
///
/// The control lane in fire N+1 is what makes the gate say something. It
/// publishes on a different slot with a true prediction, so a launch that
/// refused everything wholesale — a broken seed, a wrong grid — would fail
/// here instead of passing by accident.
#[test]
fn a_refusal_propagates_to_the_next_fires_stale_ticket_without_the_host() {
    let mut gpu = Gpu::open();
    // Slot 0: an input channel standing empty at counters 1/1. Slot 1: the
    // control lane's output, empty at 0/0.
    let registry = Registry::seed(&mut gpu, &[4, 4], &[1, 0], &[1, 0], &[]);
    let input_words = gpu.words(1, 1);
    let control_words = gpu.words(0, 0);

    // Fire N's ticket: head is right, but there is nothing to take.
    let mut first = ticket(0, Ticket::CONSUME | Ticket::REQUIRE_INPUT, 4, input_words);
    first.expected_head = 1;
    // Fire N+1's ticket: composed assuming N consumed counter 1.
    let mut second = ticket(0, Ticket::CONSUME | Ticket::REQUIRE_INPUT, 4, input_words);
    second.expected_head = 2;
    // Fire N+1's control ticket, which is simply true.
    let mut control = ticket(1, Ticket::PUBLISH, 4, control_words);
    control.expected_tail = 0;
    let tickets = gpu.up(&[first, second, control]);

    let (lane_n, commit_n) = lane(&mut gpu, registry.rings, 0, 1);
    let (lane_stale, commit_stale) = lane(&mut gpu, registry.rings, 1, 1);
    let (lane_control, commit_control) = lane(&mut gpu, registry.rings, 2, 1);
    let lanes_n = gpu.up(&[lane_n]);
    let lanes_next = gpu.up(&[lane_stale, lane_control]);

    let taken_slots = gpu.up(&[0u32]);
    let put_slots = gpu.up(&[1u32]);
    let bump_n = gpu.up(&[registry.rings.bump_lane(taken_slots, 1, 0, 0, commit_n)]);
    let bump_next = gpu.up(&[
        registry.rings.bump_lane(taken_slots, 1, 0, 0, commit_stale),
        registry.rings.bump_lane(0, 0, put_slots, 1, commit_control),
    ]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes_n, 1).expect("enqueues");
    channel::commit_bump(&ctx, bump_n, 1).expect("enqueues");
    channel::pull_validate(&ctx, tickets, lanes_next, 2).expect("enqueues");
    channel::commit_bump(&ctx, bump_next, 2).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(commit_n, 2)[0],
        0,
        "fire N required an input the channel does not hold",
    );
    assert_eq!(
        gpu.down::<u32>(commit_stale, 2)[0],
        0,
        "fire N+1 predicted the head fire N would have left, and fire N left none",
    );
    assert_eq!(
        gpu.down::<u32>(commit_control, 2)[0],
        1,
        "while a lane whose prediction was true in the same launch still commits",
    );
    assert_eq!(
        registry.head(&gpu),
        vec![1, 0],
        "two refused fires consumed nothing between them",
    );
    assert_eq!(
        registry.tail(&gpu),
        vec![1, 1],
        "and the only tail that moved is the control lane's",
    );
}

// ───────────────────────────────── (e) ───────────────────────────────────

/// **A HOST WRITER'S MIRROR BYTES LAND IN THE DEVICE CELL, AND THE FULL BYTE
/// IS SET.**
///
/// The pull is the one thing `pull_validate` does besides voting: for a ticket
/// flagged `HostWriter|Consume`, the cell the fire is about to take is copied
/// out of the guest's pinned staging ring into the device slab, block-strided,
/// and marked full so the rest of the fire addresses it like any other cell.
/// Both wire shapes are checked — plain bytes, and the bit-packed form a bool
/// channel arrives in, which the pull widens to one byte per element.
///
/// The ring index is `expected_head % cap1`, not zero and not `head`: slot 0
/// stands at counter 5 in a ring of 4, so the bytes must land in cell 1.
#[test]
fn a_host_writers_mirror_lands_in_the_cell_it_names_and_marks_it_full() {
    let mut gpu = Gpu::open();
    let registry = Registry::seed(&mut gpu, &[4, 4], &[1, 0], &[2, 0], &[]);

    // Plain channel: 8 bytes a cell, standing at counters 5/6 — ring 1.
    const PAYLOAD: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let plain_words = gpu.words(5, 6);
    let (plain_host, plain_mirror) = gpu.mapped(4 * 8);
    unsafe {
        core::ptr::copy_nonoverlapping(PAYLOAD.as_ptr(), plain_host.add(8), PAYLOAD.len());
    }
    let plain_cells = gpu.zeros(4 * 8);
    let mut plain = ticket(
        0,
        Ticket::CONSUME | Ticket::REQUIRE_INPUT | Ticket::HOST_WRITER,
        4,
        plain_words,
    );
    plain.expected_head = 5;
    plain.mirror = plain_mirror;
    plain.cells = plain_cells;
    plain.wire_bytes = 8;
    plain.native_bytes = 8;

    // Bool channel: 12 elements a cell, bit-packed into 2 wire bytes,
    // standing at counters 0/1 — ring 0.
    const BITS: [u8; 2] = [0b1010_0101, 0b0000_1100];
    const WIDENED: [u8; 12] = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1];
    let packed_words = gpu.words(0, 1);
    let (packed_host, packed_mirror) = gpu.mapped(4 * 2);
    unsafe {
        core::ptr::copy_nonoverlapping(BITS.as_ptr(), packed_host, BITS.len());
    }
    let packed_cells = gpu.zeros(4 * 12);
    let mut packed = ticket(
        1,
        Ticket::CONSUME | Ticket::REQUIRE_INPUT | Ticket::HOST_WRITER | Ticket::PACKED_BOOL,
        4,
        packed_words,
    );
    packed.expected_head = 0;
    packed.mirror = packed_mirror;
    packed.cells = packed_cells;
    packed.wire_bytes = 2;
    packed.native_bytes = 12;

    let tickets = gpu.up(&[plain, packed]);
    let (pull, commit) = lane(&mut gpu, registry.rings, 0, 2);
    let lanes = gpu.up(&[pull]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes, 1).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(commit, 2)[0],
        1,
        "both tickets predicted truly, so the fire commits",
    );

    let cells: Vec<u8> = gpu.down(plain_cells, 4 * 8);
    assert_eq!(
        &cells[8..16],
        &PAYLOAD,
        "the host's bytes landed in cell `expected_head % cap1` = 1",
    );
    assert!(
        cells[..8].iter().all(|b| *b == 0) && cells[16..].iter().all(|b| *b == 0),
        "and the pull wrote one cell, not the slab: {cells:x?}",
    );
    assert_eq!(
        registry.full(&gpu, 0, 4),
        vec![0, 1, 0, 0],
        "the pulled cell is now full, so the fire's readers address it",
    );

    let widened: Vec<u8> = gpu.down(packed_cells, 4 * 12);
    assert_eq!(
        &widened[..12],
        &WIDENED,
        "a bool channel arrives bit-packed and is widened one byte per element",
    );
    assert_eq!(
        registry.full(&gpu, 1, 4),
        vec![1, 0, 0, 0],
        "and its cell is full too",
    );
}

// ───────────────────────────────── (f) ───────────────────────────────────

/// **`mask_from_commit` SCATTERS EACH LANE'S WORD ACROSS THAT LANE'S ROWS.**
///
/// The recurrent-state scan folds a row only where its mask byte is non-zero
/// (`attn/ssm.cuh`'s `row_persists`), and the rows allowed to fold are the
/// rows of the lanes whose fire committed — so the fold predicate reaches the
/// scan as device data and nobody reads a commit word on the host.
///
/// Four things are checked at once, because each is a way the scatter can be
/// wrong: both polarities land, a lane with an EMPTY row span writes nothing
/// (and does not steal its neighbour's rows), a null commit pointer reads as
/// "did not commit", and every row in the CSR is written — the mask is seeded
/// with a poison byte that must not survive anywhere.
#[test]
fn mask_from_commit_scatters_each_lanes_word_across_its_rows() {
    const POISON: u8 = 0xAA;

    let mut gpu = Gpu::open();
    let yes = gpu.up(&[1u32, 0]);
    let no = gpu.up(&[0u32, 0]);
    let empty_span = gpu.up(&[1u32, 0]);
    let also_yes = gpu.up(&[1u32, 0]);

    // Five lanes over seven rows: lane 2 owns none of them, lane 4 has no
    // commit word at all.
    let commits = gpu.up(&[yes, no, empty_span, also_yes, 0u64]);
    let indptr = gpu.up(&[0i32, 2, 4, 4, 6, 7]);
    let mask = gpu.up(&[POISON; 7]);

    let ctx = gpu.ctx();
    channel::mask_from_commit(&ctx, commits, indptr, mask, 5).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u8>(mask, 7),
        vec![1, 1, 0, 0, 1, 1, 0],
        "each lane's rows carry its own commit word, and a null word does not fold",
    );
}

// ───────────────────────────────── (g) ───────────────────────────────────

/// **A FULL RING REFUSES A PUBLISH — UNLESS THE SAME TICKET ALSO CONSUMES.**
///
/// The room test is `tail - head < (cap1 - 1) + credit`, where `credit` is 1
/// for a ticket that consumes in the same fire: the take frees the cell the
/// put needs, in the same pass, so a ring that is full to a pure producer has
/// room for a ping-pong. `cap1 - 1` rather than `cap1` is the spare-cell
/// convention — a capacity-2 channel holds two unconsumed items, never three.
///
/// One channel, two lanes, one launch: the same full ring refuses the first
/// and admits the second, which is the sharpest way to state that the credit
/// is what makes the difference and not the ring state.
#[test]
fn a_full_ring_refuses_a_publish_but_admits_one_that_consumes_in_the_same_fire() {
    let mut gpu = Gpu::open();
    // cap1 = 3 is a capacity-2 channel. At counters head=4, tail=6 it holds
    // its two items, at rings 1 and 2.
    let registry = Registry::seed(&mut gpu, &[3], &[1], &[0], &[(0, 1), (0, 2)]);
    let words = gpu.words(4, 6);

    let mut producer = ticket(0, Ticket::PUBLISH, 3, words);
    producer.expected_tail = 6;
    let mut pingpong = ticket(0, Ticket::PUBLISH | Ticket::CONSUME, 3, words);
    pingpong.expected_head = 4;
    pingpong.expected_tail = 6;
    let tickets = gpu.up(&[producer, pingpong]);

    let (lane_producer, commit_producer) = lane(&mut gpu, registry.rings, 0, 1);
    let (lane_pingpong, commit_pingpong) = lane(&mut gpu, registry.rings, 1, 1);
    let lanes = gpu.up(&[lane_producer, lane_pingpong]);

    let slots = gpu.up(&[0u32]);
    let bumps = gpu.up(&[
        registry.rings.bump_lane(0, 0, slots, 1, commit_producer),
        registry
            .rings
            .bump_lane(slots, 1, slots, 1, commit_pingpong),
    ]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes, 2).expect("enqueues");
    channel::commit_bump(&ctx, bumps, 2).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(commit_producer, 2)[0],
        0,
        "a pure producer has nowhere to put: the ring holds its capacity already",
    );
    assert_eq!(
        gpu.down::<u32>(commit_pingpong, 2)[0],
        1,
        "the same ring admits a ticket that frees a cell in the same pass",
    );
    assert_eq!(
        registry.tail(&gpu),
        vec![1],
        "one publish happened — the ping-pong's — advancing the tail off ring 0",
    );
    assert_eq!(
        registry.head(&gpu),
        vec![2],
        "and its take advanced the head off ring 1",
    );
    assert_eq!(
        registry.full(&gpu, 0, 3),
        vec![1, 0, 1],
        "leaving the ring still holding two items, now at rings 2 and 0",
    );
}

// ───────────────────────────────── (h) ───────────────────────────────────

/// **A COMMITTED FIRE'S CELL REACHES THE GUEST'S PINNED MIRROR, AND A REFUSED
/// FIRE'S DOES NOT.**
///
/// `scatter_publish` is the outward half of gate (e): (e) proves the host's
/// bytes reach the device cell without a `cudaMemcpy`, this proves the pass's
/// bytes reach the host's mirror the same way. Three things are checked at
/// once, because each is a way the crossing can be silently wrong:
///
/// ```text
/// * a committed lane copies the cell at `expected_tail % cap1` — the cell
///   its own ticket named — and copies only that cell
/// * a bool channel is PACKED on the way out, one bit per lane, which is the
///   wire form the guest's ring holds (the inverse of (e)'s widening)
/// * a REFUSED lane scatters nothing at all, so a guest whose tail never
///   advanced can never read a dummy run's bytes
/// ```
///
/// The refused lane shares the launch with the committed ones rather than
/// running on its own: a predicate read once per grid instead of once per
/// lane would pass a test that ran them apart.
#[test]
fn a_committed_fires_cell_reaches_the_pinned_mirror_and_a_refused_fires_does_not() {
    const POISON: u8 = 0xEE;
    const PAYLOAD: [u8; 8] = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04];
    /// One byte per lane on the device, as a scan writes them.
    const WIDE: [u8; 12] = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1];
    /// The same twelve lanes as the guest's ring holds them.
    const BITS: [u8; 2] = [0b1010_0101, 0b0000_1100];

    let mut gpu = Gpu::open();

    // Slot 0, plain: 8-byte cells, cap1 4, this fire publishes at counter 6 —
    // ring 2.
    let plain_words = gpu.words(5, 6);
    let (plain_host, plain_mirror) = gpu.mapped(4 * 8);
    let plain_cells = gpu.zeros(4 * 8);
    gpu.poke(plain_cells + 2 * 8, &PAYLOAD);
    let mut plain = ticket(0, Ticket::PUBLISH | Ticket::HOST_READER, 4, plain_words);
    plain.expected_tail = 6;
    plain.mirror = plain_mirror;
    plain.cells = plain_cells;
    plain.wire_bytes = 8;
    plain.native_bytes = 8;

    // Slot 1, bool: 12 lanes a cell, 2 wire bytes, publishing at counter 0.
    let packed_words = gpu.words(0, 0);
    let (packed_host, packed_mirror) = gpu.mapped(4 * 2);
    let packed_cells = gpu.zeros(4 * 12);
    gpu.poke(packed_cells, &WIDE);
    let mut packed = ticket(
        1,
        Ticket::PUBLISH | Ticket::HOST_READER | Ticket::PACKED_BOOL,
        4,
        packed_words,
    );
    packed.expected_tail = 0;
    packed.mirror = packed_mirror;
    packed.cells = packed_cells;
    packed.wire_bytes = 2;
    packed.native_bytes = 12;

    // Slot 2, plain, on the lane that did NOT commit.
    let refused_words = gpu.words(0, 0);
    let (refused_host, refused_mirror) = gpu.mapped(4 * 8);
    let refused_cells = gpu.zeros(4 * 8);
    gpu.poke(refused_cells, &PAYLOAD);
    let mut refused = ticket(2, Ticket::PUBLISH | Ticket::HOST_READER, 4, refused_words);
    refused.expected_tail = 0;
    refused.mirror = refused_mirror;
    refused.cells = refused_cells;
    refused.wire_bytes = 8;
    refused.native_bytes = 8;

    // The mirrors start poisoned, so "nothing was written" is asserted as a
    // byte nobody could have produced rather than as a zero.
    for (host, bytes) in [
        (plain_host, 4 * 8),
        (packed_host, 4 * 2),
        (refused_host, 4 * 8),
    ] {
        // SAFETY: `mapped` allocated `bytes` at `host` and nothing else holds it.
        unsafe { core::ptr::write_bytes(host, POISON, bytes) };
    }

    let tickets = gpu.up(&[plain, packed, refused]);
    let committed = gpu.up(&[1u32, 0]);
    let declined = gpu.up(&[0u32, 0]);
    let lanes = gpu.up(&[
        channel::PublishLane {
            commit: committed,
            ticket_offset: 0,
            ticket_count: 2,
        },
        channel::PublishLane {
            commit: declined,
            ticket_offset: 2,
            ticket_count: 1,
        },
    ]);

    let ctx = gpu.ctx();
    channel::scatter_publish(&ctx, tickets, lanes, 2).expect("enqueues");
    gpu.sync();

    // SAFETY: the three mirrors are alive until `gpu` drops, and the launch
    // that wrote them has been synchronized.
    let mirror = |host: *mut u8, bytes: usize| unsafe {
        core::slice::from_raw_parts(host.cast_const(), bytes).to_vec()
    };

    let plain_out = mirror(plain_host, 4 * 8);
    assert_eq!(
        &plain_out[16..24],
        &PAYLOAD,
        "the pass's cell landed at `expected_tail % cap1` = 2",
    );
    assert!(
        plain_out[..16].iter().all(|b| *b == POISON) && plain_out[24..].iter().all(|b| *b == POISON),
        "and one cell crossed, not the slab: {plain_out:x?}",
    );

    assert_eq!(
        &mirror(packed_host, 4 * 2)[..2],
        &BITS,
        "a bool channel is packed one bit per lane on the way out",
    );

    assert_eq!(
        mirror(refused_host, 4 * 8),
        vec![POISON; 32],
        "a refused lane publishes nothing, so a dummy run's bytes never reach a guest",
    );
}

// ───────────────────────────────── (i) ───────────────────────────────────

/// **THE SETTLEMENT ADVANCES THE GUEST'S OWN COUNTERS, AND ONLY THE ONES A
/// TICKET NAMES.**
///
/// The kernel that replaced a `cudaStreamSynchronize` per frame boundary. Its
/// whole job is four words wide, and every way it can be wrong is silent: a
/// counter left where it was makes the next fire's prediction stale and
/// `pull_validate` refuses forever; a counter advanced on a lane that did not
/// commit hands the guest a cell nothing published; a counter advanced for a
/// ticket that only PEEKED skips a cell nobody read.
///
/// Four endpoints, one launch:
///
/// ```text
/// slot 0  ADVANCE_HEAD | ADVANCE_TAIL   both counters move, committed lane
/// slot 1  CONSUME, no ADVANCE_HEAD      a peek: nothing moves, and this is
///                                       the flag divergence from dev, where
///                                       kTicketConsume alone would advance
/// slot 2  ADVANCE_TAIL only             the head is the GUEST's counter on a
///                                       channel the host reads, so a store
///                                       here would be two writers on one word
/// slot 3  ADVANCE_HEAD | ADVANCE_TAIL   on the REFUSED lane: nothing moves
/// ```
#[test]
fn a_committed_fires_settlement_advances_exactly_the_counters_its_tickets_name() {
    let mut gpu = Gpu::open();

    // One endpoint's `[head, tail, poison, closed]` in mapped pinned memory,
    // with the host pointer kept so the words can be read back.
    let endpoint = |gpu: &mut Gpu, head: u64, tail: u64| {
        let (host, device) = gpu.mapped(4 * core::mem::size_of::<u64>());
        // SAFETY: `mapped` allocated four u64s at `host` and nothing else
        // holds it; the allocation is 8-aligned (`cudaHostAlloc` is page
        // aligned).
        unsafe {
            let words = host.cast::<u64>();
            words.write(head);
            words.add(1).write(tail);
            // Non-zero sentinels: the settlement writes neither, and dev's
            // `words+2 = 0` is deliberately NOT ported (nothing in this engine
            // owns the poison word yet, and a kernel storing it would race the
            // host callback that will).
            words.add(2).write(0xAA);
            words.add(3).write(0xBB);
        }
        (host.cast::<u64>(), device)
    };

    let (both_host, both_words) = endpoint(&mut gpu, 41, 43);
    let (peek_host, peek_words) = endpoint(&mut gpu, 7, 9);
    let (tail_host, tail_words) = endpoint(&mut gpu, 100, 200);
    let (refused_host, refused_words) = endpoint(&mut gpu, 3, 4);

    let mut both = ticket(
        0,
        Ticket::CONSUME | Ticket::PUBLISH | Ticket::ADVANCE_HEAD | Ticket::ADVANCE_TAIL,
        4,
        both_words,
    );
    both.expected_head = 41;
    both.expected_tail = 43;

    // CONSUME without ADVANCE_HEAD: a `read` that addresses the committed cell
    // without taking it.
    let mut peek = ticket(1, Ticket::CONSUME, 4, peek_words);
    peek.expected_head = 7;

    let mut tail_only = ticket(2, Ticket::PUBLISH | Ticket::ADVANCE_TAIL, 4, tail_words);
    tail_only.expected_head = 100;
    tail_only.expected_tail = 200;

    let mut refused = ticket(
        3,
        Ticket::CONSUME | Ticket::PUBLISH | Ticket::ADVANCE_HEAD | Ticket::ADVANCE_TAIL,
        4,
        refused_words,
    );
    refused.expected_head = 3;
    refused.expected_tail = 4;

    let tickets = gpu.up(&[both, peek, tail_only, refused]);
    let committed = gpu.up(&[1u32, 0]);
    let declined = gpu.up(&[0u32, 0]);
    let lanes = gpu.up(&[
        channel::SettleLane {
            commit: committed,
            ticket_offset: 0,
            ticket_count: 3,
        },
        channel::SettleLane {
            commit: declined,
            ticket_offset: 3,
            ticket_count: 1,
        },
    ]);

    let ctx = gpu.ctx();
    channel::settle(&ctx, tickets, lanes, 2).expect("enqueues");
    gpu.sync();

    // SAFETY: every endpoint is alive until `gpu` drops, and the launch that
    // wrote them has been synchronized.
    let read = |host: *mut u64| unsafe { core::slice::from_raw_parts(host.cast_const(), 4).to_vec() };

    assert_eq!(
        read(both_host),
        vec![42, 44, 0xAA, 0xBB],
        "a committed fire leaves each named counter at the PREDICTION plus one, \
         and the poison and closed words alone",
    );
    assert_eq!(
        read(peek_host),
        vec![7, 9, 0xAA, 0xBB],
        "a ticket that addresses the head without taking it moves nothing: \
         CONSUME is not ADVANCE_HEAD",
    );
    assert_eq!(
        read(tail_host),
        vec![100, 201, 0xAA, 0xBB],
        "the head of a channel the host reads is the GUEST's counter and no \
         kernel may store it",
    );
    assert_eq!(
        read(refused_host),
        vec![3, 4, 0xAA, 0xBB],
        "a refused lane settles nothing, so the guest's endpoint stands exactly \
         where it stood and the next fire predicts the same numbers",
    );
}


// ───────────────────────────────── (j) ───────────────────────────────────

/// **A PULL IS PREDICATED ON ITS OWN TICKET, AND ON NOTHING ELSE IN THE FIRE.**
///
/// The module header's one documented exception to "`commit_bump` is the only
/// writer": a ticket that passes copies its mirror cell in and sets the full
/// byte, and a LATER ticket in the same lane can still veto the fire — leaving
/// that byte set on a pass that did not commit. It is safe because the byte
/// records something the GUEST published, the head does not move, and the next
/// fire re-pulls the same cell and sets the same byte.
///
/// What makes it worth a gate of its own is that the exception is stated in
/// TICKET ORDER, and the kernel no longer walks its tickets in order: the vote
/// is taken one thread per ticket, all the loads in flight together, and the
/// pulls follow one warp per ticket. So this asks the property in both
/// directions at once — one lane whose veto arrives AFTER the pulling ticket,
/// one whose veto arrives BEFORE it — and demands the same answer from both.
/// A kernel that gated the pull on the lane's running commit word would pass
/// the first lane and fail the second; one that gated it on the whole fire
/// would fail both.
#[test]
fn a_pulled_cell_lands_whether_the_veto_comes_before_it_or_after_it() {
    let mut gpu = Gpu::open();
    // Four slots, one per ticket: two lanes of (host writer, stale peek), in
    // opposite orders. Every ring stands at 0/1 in the registry.
    let registry = Registry::seed(&mut gpu, &[4, 4, 4, 4], &[0, 0, 0, 0], &[1, 1, 1, 1], &[]);

    const PAYLOAD: [u8; 16] = [
        0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE,
        0xAF,
    ];

    // The two pulling tickets — slot 0 for the lane vetoed after it, slot 2
    // for the lane vetoed before it. Both predict truly.
    let mut cells = Vec::new();
    let mut pullers = Vec::new();
    for slot in [0u32, 2] {
        let words = gpu.words(0, 1);
        let (host, mirror) = gpu.mapped(4 * PAYLOAD.len());
        // SAFETY: `mapped` just allocated `4 * PAYLOAD.len()` bytes at `host`.
        unsafe {
            core::ptr::copy_nonoverlapping(PAYLOAD.as_ptr(), host, PAYLOAD.len());
        }
        let cell = gpu.zeros(4 * PAYLOAD.len());
        cells.push(cell);
        let mut puller = ticket(
            slot,
            Ticket::CONSUME | Ticket::REQUIRE_INPUT | Ticket::HOST_WRITER,
            4,
            words,
        );
        puller.expected_head = 0;
        puller.mirror = mirror;
        puller.cells = cell;
        puller.wire_bytes = PAYLOAD.len() as u32;
        puller.native_bytes = PAYLOAD.len() as u32;
        pullers.push(puller);
    }

    // The two vetoing tickets: the endpoint stands at 9 and the fire believes 3.
    let mut vetoes = Vec::new();
    for slot in [1u32, 3] {
        let words = gpu.words(9, 10);
        let mut veto = ticket(slot, Ticket::CONSUME, 4, words);
        veto.expected_head = 3;
        vetoes.push(veto);
    }

    // Lane 0: pull, then veto. Lane 1: veto, then pull.
    let tickets = gpu.up(&[pullers[0], vetoes[0], vetoes[1], pullers[1]]);
    let (after, after_commit) = lane(&mut gpu, registry.rings, 0, 2);
    let (before, before_commit) = lane(&mut gpu, registry.rings, 2, 2);
    let lanes = gpu.up(&[after, before]);

    let ctx = gpu.ctx();
    channel::pull_validate(&ctx, tickets, lanes, 2).expect("enqueues");
    gpu.sync();

    assert_eq!(
        gpu.down::<u32>(after_commit, 2)[0],
        0,
        "the stale ticket refuses the fire whatever order it sits in",
    );
    assert_eq!(
        gpu.down::<u32>(before_commit, 2)[0],
        0,
        "and so does the one that sits first",
    );

    for (which, (cell, slot)) in [(cells[0], 0u32), (cells[1], 2)].into_iter().enumerate() {
        let order = if which == 0 { "after" } else { "before" };
        let bytes: Vec<u8> = gpu.down(cell, 4 * PAYLOAD.len());
        assert_eq!(
            &bytes[..PAYLOAD.len()],
            &PAYLOAD,
            "the veto arriving {order} the pull suppressed it; a pull is predicated on its own \
             ticket, not on the fire's commit word",
        );
        assert_eq!(
            registry.full(&gpu, slot, 4),
            vec![1, 0, 0, 0],
            "and the full byte the pull sets is set on a pass that did not commit — the header's \
             documented exception, veto {order}",
        );
    }
}
