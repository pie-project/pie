//! **A ROUTE NOTHING ROUTED — the router that mints it, and the GEMV that
//! reads it.**
//!
//! `topk_idx` holds -1 by design. `moe/topk_softmax.cuh` is where: both of its
//! block-argmax writers store `out_idx[k] = best_i` with `best_i` still -1 when
//! no expert is left to pick, which is every `k >= num_experts`, and the second
//! pairs it with `weight = 0.f` — this tree's convention that padding computes
//! something the combine then multiplies by zero.
//!
//! That convention needs the something to be FINITE, and `moe_decode_gemv_body`
//! gave it neither half: it indexed `weight_base + expert * expert_stride` at
//! -1, one whole expert's stride BEFORE the bank, and never wrote the row at
//! all. An in-arena read of somebody else's slot does not fault, and
//! `batched_weighted_sum` then multiplies it by zero — `0 * NaN` is `NaN`, for
//! the whole token's hidden state.
//!
//! Three siblings in the same file already guarded (`moe_decode_wmma` returns,
//! `build_moe_ptrs_aligned` clamps to expert 0, `moe_add_bias` returns) and
//! nothing had ever asked this one. `driver-metal/tests/device_routing.rs` has
//! the matching test one stage upstream — `moe.topk_sigmoid at k > n` — and
//! metal's router parks the spare slots on expert 0 at weight 0 rather than
//! emitting -1 at all, so the two planes disagree about the sentinel and only
//! cuda's choice reaches this kernel.
//!
//! # Two poisons, and the first draft of this test had only one
//!
//! **THE OUTPUT IS PRE-FILLED WITH NaN.** `y` is a carved arena slot in the
//! fire that matters and nothing zeroes it per fire, so "the kernel skipped the
//! row" and "the kernel wrote zero" are the same observation over zeroed memory
//! and different ones over this.
//!
//! **AND SO IS THE ELEMENT BEFORE THE BANK**, which is the half that was
//! missing. Measured: with only the first poison, this test PASSED against the
//! unguarded kernel — expert -1 read `weight_base - expert_stride`, whatever
//! `Allocator` left one stride below happened to be zeros, and the dot product
//! came out `0.0`, which is the right answer by accident. A regression test
//! that passes either way is not one.
//!
//! So the bank is the SECOND HALF of one allocation whose first half is NaN,
//! which puts a known poison exactly where a negative expert index lands. Both
//! poisons are load-bearing and each one alone lets the defect through.

use driver_cuda::device::{Allocator, OwnedStream};
use kernels::plane::{Const, In, Out};
use kernels::points::Moe as _;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;

mod common;
use common::{device_or_skip, gpu_guard};

/// One expert exists, and the fanout asks for two.
const EXPERTS: usize = 1;
const TOP_K: usize = 2;
/// The activation width, in whole `float4` loads of eight.
const K: usize = 8;
/// The bank's output width.
const N: usize = 4;

const ONE: u16 = 0x3F80;
/// A quiet bf16 NaN — the top half of `f32::NAN`.
const NAN: u16 = 0x7FC0;

#[test]
fn an_unrouted_route_is_written_zero_and_not_left_as_it_was() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("moe.matmul_select over an unrouted route") else {
        return;
    };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let host = |words: &[u16]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };

    // **THE BANK IS THE SECOND HALF OF THIS.** The first half is one expert
    // stride of NaN, sitting exactly where `weight_base + expert * stride`
    // lands when the expert is -1. Handing the point a pointer into the middle
    // of an allocation is what puts a known value under the out-of-bounds read
    // instead of whatever the allocator happened to leave there.
    let mut bank = alloc.alloc((EXPERTS + 1) * N * K * 2).expect("bank");
    let mut words = vec![NAN; N * K];
    words.extend(std::iter::repeat_n(ONE, EXPERTS * N * K));
    bank.copy_from_host(&host(&words), stream.as_ref())
        .expect("the bank uploads");
    // One token's activation row, and `matmul_select` reads it by TOKEN when
    // the rows times the fanout are the routes.
    let mut act = alloc.alloc(K * 2).expect("act");
    act.copy_from_host(&host(&vec![ONE; K]), stream.as_ref())
        .expect("the activation uploads");
    // **ROUTE 1 IS THE ONE NOTHING ROUTED.** Two slots over one expert.
    let mut routes = alloc.alloc(TOP_K * 4).expect("routes");
    routes
        .copy_from_host(
            &[0i32, -1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            stream.as_ref(),
        )
        .expect("the fanout uploads");
    let mut y = alloc.alloc(TOP_K * N * 2).expect("y");
    y.copy_from_host(&host(&vec![NAN; TOP_K * N]), stream.as_ref())
        .expect("the destination is poisoned");
    stream.as_ref().synchronize().expect("the uploads land");

    let ctx = unsafe { Ctx::on(stream.as_ref().as_raw().cast()) };
    let ptr = |b: &driver_cuda::device::DeviceBuffer, len: usize| {
        b.ptr_at(0, len).expect("the whole buffer")
    };
    ctx.matmul_select::<bf16>(
        In {
            ptr: ptr(&act, K * 2).cast::<bf16>().cast_const(),
            rows: 1,
            width: K as i32,
        },
        Const {
            v: bank
                .ptr_at(N * K * 2, EXPERTS * N * K * 2)
                .expect("the bank, past the poison")
                .cast::<bf16>()
                .cast_const(),
        },
        In {
            ptr: ptr(&routes, TOP_K * 4).cast::<i32>().cast_const(),
            rows: 1,
            width: TOP_K as i32,
        },
        Out {
            ptr: ptr(&y, TOP_K * N * 2).cast::<bf16>(),
            rows: (TOP_K * 1) as i32,
            width: N as i32,
        },
    )
    .expect("`moe.matmul_select` fires");
    stream.as_ref().synchronize().expect("the fire lands");

    let mut back = vec![0u8; TOP_K * N * 2];
    y.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("the read-out lands");
    let got: Vec<f32> = back
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();
    eprintln!("routed {:?} | unrouted {:?}", &got[..N], &got[N..]);

    // Route 0 is real: eight ones against eight ones.
    assert!(
        got[..N].iter().all(|v| (*v - 8.0).abs() < 1e-3),
        "the routed half is {:?} and should be eight everywhere; the guard \
         must not have moved what a real route computes",
        &got[..N],
    );
    // Route 1 is not, and every one of its channels must be a written ZERO —
    // not the NaN that was there, which is what a `return` would have left.
    for (i, v) in got[N..].iter().enumerate() {
        assert!(
            v.is_finite(),
            "channel {i} of the unrouted route is {v}, which is the poison this \
             fire was supposed to overwrite; the combine multiplies this by a \
             zero weight and `0 * NaN` is `NaN`",
        );
        assert_eq!(*v, 0.0, "channel {i} of the unrouted route is {v}");
    }
}

/// **THE ROUTER THAT MINTS THE -1, AND THE WEIGHT IT GAVE IT.**
///
/// `moe.topk_softmax` fires the BLOCK form, whose argmax-with-exclusion seeds
/// `best_value` at the floor and wins only strictly above it — so once every
/// expert is taken it answers `(-1.f, -1)`, and `-1.f` was written into
/// `topk_w` and added to the normaliser.
///
/// **AT `K == num_experts + 1` THAT IS EXACTLY FATAL.** The real weights are
/// normalised probabilities summing to 1, so the normaliser is `1 - 1 == 0`,
/// its reciprocal is `+inf`, and every weight in the row — the REAL ones
/// included — comes out infinite. One more spare slot and they come out
/// negated. Nothing faults either way, and the combine folds the result into
/// the token's hidden state.
///
/// The other two routers in the same file already answered zero: the warp
/// rungs by construction (`expf(-flt_max() - row_max)` is `0`) and
/// `topk_sqrt_softplus_body` in as many words. This one was the odd one out,
/// and `k > n` is a case `driver-metal/tests/device_routing.rs` has had a
/// device test for on the other plane the whole time.
#[test]
fn a_router_asked_for_more_slots_than_it_has_experts_weighs_the_spares_zero() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("moe.topk_softmax at k > n") else {
        return;
    };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // Two experts, three slots. `1 - 1 == 0` is the normaliser that made the
    // old form divide by zero, so this is the shape that fails loudest.
    const NARROW: usize = 2;
    const WIDE: usize = 3;

    let mut logits = alloc.alloc(NARROW * 2).expect("logits");
    logits
        .copy_from_host(
            &[ONE, 0x0000u16]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            stream.as_ref(),
        )
        .expect("the logits upload");
    let routes = alloc.alloc(WIDE * 4).expect("routes");
    let weights = alloc.alloc(WIDE * 4).expect("weights");
    stream.as_ref().synchronize().expect("the upload lands");

    let ctx = unsafe { Ctx::on(stream.as_ref().as_raw().cast()) };
    ctx.topk_softmax::<bf16>(
        In {
            ptr: logits
                .ptr_at(0, NARROW * 2)
                .expect("the logits")
                .cast::<bf16>()
                .cast_const(),
            rows: 1,
            width: NARROW as i32,
        },
        NARROW as u32,
        WIDE as u32,
        Out {
            ptr: routes.ptr_at(0, WIDE * 4).expect("routes").cast::<i32>(),
            rows: 1,
            width: WIDE as i32,
        },
        Out {
            ptr: weights.ptr_at(0, WIDE * 4).expect("weights").cast::<f32>(),
            rows: 1,
            width: WIDE as i32,
        },
    )
    .expect("`moe.topk_softmax` fires");
    stream.as_ref().synchronize().expect("the fire lands");

    let mut raw = vec![0u8; WIDE * 4];
    routes.copy_to_host(&mut raw, stream.as_ref()).expect("d2h");
    let ids: Vec<i32> = raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    weights
        .copy_to_host(&mut raw, stream.as_ref())
        .expect("d2h");
    stream.as_ref().synchronize().expect("the read-out lands");
    let w: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    eprintln!("ids {ids:?} weights {w:?}");

    assert!(
        w.iter().all(|v| v.is_finite()),
        "the row weighs {w:?}; a spare slot took the argmax FLOOR into the \
         normaliser and every weight in the row went with it",
    );
    assert_eq!(
        &ids[..NARROW],
        &[0, 1],
        "the two real slots take the two experts, largest logit first",
    );
    assert_eq!(ids[NARROW], -1, "the spare slot has no expert");
    assert_eq!(w[NARROW], 0.0, "and a spare slot weighs nothing");
    // The real weights still sum to one, which is what says the normaliser was
    // the sum of the real ones and nothing else.
    let sum: f32 = w[..NARROW].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "the real weights sum to {sum} and a softmax router's must sum to one",
    );
}
