//! **WHAT `pull_validate` COSTS PER FIRE, AND THE SHAPE THAT COST HAS.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test channel_pull_cost -- --nocapture
//! ```
//!
//! `channel_kernels.rs` asks whether the five control kernels are RIGHT. This
//! file asks the one question a correctness gate cannot: what the admission
//! decision costs the fire it sits in front of, and — the part that matters —
//! **what that cost is a function of.**
//!
//! The kernel reads words another agent writes: `ticket.words[0]` and `[1]`
//! live in the guest's MAPPED PINNED memory, so every load is a PCIe read from
//! the device, ~1–2 µs of latency that no amount of arithmetic hides. A fire
//! carries one ticket per host-visible endpoint it addresses. So the whole
//! question is whether those reads happen ONCE, overlapped, or one after
//! another:
//!
//! ```text
//!   serialized:  cost ≈ tickets_per_lane × PCIe_latency
//!   overlapped:  cost ≈ PCIe_latency
//! ```
//!
//! The kernel used to be the first: one block per lane, and inside it a
//! `for` loop over the lane's tickets where **thread 0 alone** did both
//! acquire loads and every other thread waited on a `__syncthreads()`. Four
//! endpoints meant four dependent round trips deep in the fire's prologue,
//! and eight meant eight. The tickets do not depend on one another — each is
//! an independent claim about a different ring — so the loads can all be in
//! flight at once, and the vote is an `and` over the answers, which is
//! order-free.
//!
//! # The two axes, and why both are here
//!
//! * **TICKETS** (fixed small cell): the latency axis. A kernel whose ticket
//!   loop is serialized grows linearly here; one that issues the loads in
//!   parallel is flat. This is the axis the gate at the bottom asserts on,
//!   because it is the one that is a STRUCTURAL property of the kernel rather
//!   than a number about this card.
//! * **CELL BYTES** (fixed one ticket): the bandwidth axis. The pull copies a
//!   host-writer's mirror cell across the same PCIe aperture, and the copy is
//!   strided over the block — so this measures how wide each thread's load is,
//!   not how many threads there are.
//!
//! Printed rather than bounded, except for the one structural assertion:
//! absolute microseconds are a fact about an L40S on a particular link, and a
//! gate that pinned them would fail on the next card for no reason.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;
use std::time::Instant;

use kernels_cuda::channel::{self, PullLane, Rings, Ticket};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::jit::Ctx;

/// Lanes in the measured wave — c=64 is the batch shape every profile in
/// `.wiki/contention` was taken at, and the shape `program::wave` exists for.
const LANES: u32 = 64;

/// Fires timed per point, after the warm-up. Enough that the launch overhead
/// averages out and small enough that the whole file stays under a second.
const REPEATS: u32 = 200;

/// The ring depth every slot in this file is cut for. The pull indexes
/// `expected_head % cap1`, so anything ≥ 2 exercises the same arithmetic.
const CAP1: u32 = 4;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(code, rt::cudaError::cudaSuccess, "`{call}` answered {code:?}");
}

/// One measurement's device: a stream and every allocation on it, freed
/// together. Deliberately its own — the file times launches, and a shared
/// stream would time somebody else's queue.
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

    fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire here, and `Drop` synchronizes
        // before destroying it.
        unsafe { Ctx::on(self.stream.cast()) }
    }

    fn zeros(&mut self, bytes: usize) -> u64 {
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

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

    /// **ONE mapped-pinned block for the whole wave**, sliced by the caller.
    ///
    /// Production allocates a guest endpoint's words and its staging ring
    /// separately; a thousand `cudaHostAlloc`s here would measure the
    /// allocator. What the kernel dereferences is a device-visible address
    /// either way, and `cudaHostGetDevicePointer` is what says the two are the
    /// same number rather than what assumes it.
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

    fn sync(&self) {
        check(
            unsafe { rt::cudaStreamSynchronize(self.stream) },
            "cudaStreamSynchronize",
        );
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

/// The staged wave: `LANES` lanes of `per_lane` host-writer tickets each, every
/// one of them predicting truly so that the kernel takes the expensive path —
/// validate, then pull.
struct Wave {
    tickets: u64,
    lanes: u64,
}

fn stage(gpu: &mut Gpu, per_lane: u32, cell_bytes: u32) -> Wave {
    let slots = (LANES * per_lane) as usize;
    let cells = gpu.zeros(slots * CAP1 as usize * cell_bytes as usize);

    // Every endpoint's four live words, contiguous: `[head, tail, poison,
    // closed]`, seeded so that head == expected_head and tail > head.
    let (words_host, words_device) = gpu.mapped(slots * 4 * size_of::<u64>());
    for slot in 0..slots {
        unsafe {
            let at = words_host.cast::<u64>().add(slot * 4);
            at.write(0);
            at.add(1).write(1);
            at.add(2).write(0);
            at.add(3).write(0);
        }
    }
    let (_, mirror) = gpu.mapped(slots * CAP1 as usize * cell_bytes as usize);

    let mut tickets = Vec::with_capacity(slots);
    for slot in 0..slots {
        let stride = (CAP1 as usize * cell_bytes as usize * slot) as u64;
        tickets.push(Ticket {
            slot: u32::try_from(slot).unwrap(),
            flags: Ticket::CONSUME | Ticket::REQUIRE_INPUT | Ticket::HOST_WRITER,
            expected_head: 0,
            expected_tail: channel::NO_TICKET,
            words: words_device + (slot * 4 * size_of::<u64>()) as u64,
            mirror: mirror + stride,
            cells: cells + stride,
            cap1: CAP1,
            wire_bytes: cell_bytes,
            native_bytes: cell_bytes,
        });
    }

    let full = gpu.zeros(slots * channel::MAX_RING as usize);
    let head = gpu.up(&vec![0u32; slots]);
    let tail = gpu.up(&vec![1u32; slots]);
    let cap1 = gpu.up(&vec![CAP1; slots]);
    let rings = Rings::new(full, head, tail, cap1, u32::try_from(slots).unwrap());
    let commit = gpu.up(&vec![0u32; 2 * LANES as usize]);

    let lanes: Vec<PullLane> = (0..LANES)
        .map(|lane| PullLane {
            full: rings.full,
            pass_commit: commit + u64::from(lane) * 2 * size_of::<u32>() as u64,
            ticket_offset: lane * per_lane,
            ticket_count: per_lane,
            initial_commit: 1,
            diagnose: 0,
        })
        .collect();

    Wave {
        tickets: gpu.up(&tickets),
        lanes: gpu.up(&lanes),
    }
}

/// Microseconds a single `pull_validate` over the whole wave costs, wall
/// clock, synchronized — which is what the fire pays, since the prologue is on
/// the critical path of everything behind it.
fn microseconds(gpu: &Gpu, wave: &Wave) -> f64 {
    let ctx = gpu.ctx();
    for _ in 0..20 {
        channel::pull_validate(&ctx, wave.tickets, wave.lanes, LANES).expect("enqueues");
    }
    gpu.sync();
    let start = Instant::now();
    for _ in 0..REPEATS {
        channel::pull_validate(&ctx, wave.tickets, wave.lanes, LANES).expect("enqueues");
    }
    gpu.sync();
    start.elapsed().as_secs_f64() * 1e6 / f64::from(REPEATS)
}

/// **THE LATENCY AXIS: A FIRE'S TICKETS ARE INDEPENDENT, SO ITS PCIe READS
/// MUST OVERLAP.**
///
/// One point per ticket count at a cell small enough that the copy is noise.
/// A serialized ticket loop grows linearly here; the parallel vote is flat,
/// because every ticket's two acquire loads are in flight at once and the
/// block waits out one round trip rather than `n`.
///
/// The bound is deliberately loose (2× over eight tickets, where serialization
/// costs 8×): what is being defended is the SHAPE, and a bound tight enough to
/// pin the shape on this card would be a bound that fails on the next one.
#[test]
fn the_admission_decision_does_not_grow_with_a_fires_ticket_count() {
    let mut gpu = Gpu::open();
    const CELL: u32 = 64;

    let mut costs = Vec::new();
    for per_lane in [1u32, 2, 4, 8] {
        let wave = stage(&mut gpu, per_lane, CELL);
        let us = microseconds(&gpu, &wave);
        println!("  {LANES} lanes x {per_lane} tickets x {CELL} B: {us:8.2} us");
        costs.push((per_lane, us));
    }

    let one = costs[0].1;
    let eight = costs[3].1;
    println!("  eight tickets cost {:.2}x one", eight / one);
    assert!(
        eight < 2.0 * one,
        "eight tickets a lane cost {eight:.2} us against {one:.2} us for one \
         ({:.1}x): the ticket loop is serializing its mapped-pinned loads \
         instead of issuing them together, so a fire pays one PCIe round trip \
         per endpoint it addresses",
        eight / one,
    );
}

/// **THE BANDWIDTH AXIS: WHAT A HOST WRITER'S CELL COSTS TO DRAG ACROSS.**
///
/// Reported, not bounded. The pull is a real transfer over the same aperture
/// the words are read through, and the number it produces is the argument for
/// `.wiki/alto/adapter.md` §6.1 — a channel is not a weight transport, and a
/// cell wide enough to matter belongs on the blob path instead of being
/// re-dragged every fire.
#[test]
fn a_host_writers_cell_costs_what_the_aperture_costs() {
    let mut gpu = Gpu::open();
    for cell in [64u32, 1024, 16 * 1024] {
        let wave = stage(&mut gpu, 1, cell);
        let us = microseconds(&gpu, &wave);
        let bytes = f64::from(LANES) * f64::from(cell);
        println!(
            "  {LANES} lanes x 1 ticket x {cell:6} B: {us:8.2} us  \
             ({:.2} GB/s over {bytes:.0} B)",
            bytes / us / 1e3,
        );
    }
}
