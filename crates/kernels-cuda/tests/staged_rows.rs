//! **THE STAGED-GEOMETRY SEAT, ON A REAL DEVICE** (bodies design, waves 1
//! and 2a).
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test staged_rows -- --nocapture
//! ```
//!
//! # What this is for
//!
//! A body is a graph captured once, at a bucket's ceiling, and replayed for
//! every fire that rounds up to it — so the rows a fire does NOT have must be
//! retired at run time, from a word the fire staged, not from a parameter the
//! recording baked. The seat is that word's address, threaded
//! `Ctx::arm_stage` → entry → device guard
//! (`if (win != nullptr && r >= win[0]) return;`), and the exemplar entry is
//! `elemwise.res_blend`.
//!
//! Wave 2a made the seat point at a PAIR rather than a lone word: `win[0]` is
//! still the live-row COUNT the guard reads, and `win[1]` is the row those
//! live rows START at. Armed, a block's row is `win[1] + blockIdx.x` and the
//! entry is handed the plane's BASE pointers; unarmed, the pointers arrive
//! pre-shifted and the block index is the row, which is what every gate below
//! but (d) exercises.
//!
//! **AND THE SEAT IS FOUR WORDS NOW** — `[rows, row_offset, lanes,
//! lane_offset]`, the chunked-arm wave's widening — so every staging below
//! allocates four even though the exemplar reads TWO. `elemwise.res_blend` is
//! a row-gridded entry: it reads `win[0]` and `win[1]` and nothing else, and
//! it is the lane pair's absence from its text, not from the buffer, that
//! makes that true. The buffers are four words wide anyway, because the
//! engine's seat is and because a kernel that grew a `win[2]` read against a
//! two-word staging would walk off the end of it without a symptom. The two
//! words the gates write are the two the exemplar reads; the lane pair stays
//! zero and is never consulted.
//!
//! The six gates, each a way the seat can be silently wrong:
//!
//! ```text
//! (a) the null seat is today's plane: an entry fired with no stage armed
//!     writes every row it was launched over, bit-for-bit the old behavior
//! (b) an armed stage guards an EAGER launch too: rows at and past `win[0]`
//!     keep their bytes — the guard is the device text's, not the graph's
//! (c) one capture serves three row counts: a graph recorded at 8 rows with
//!     the seat armed replays at 5, at 8, and at 3, retiring exactly the
//!     rows each staging says — no exec update, no re-capture, no host
//!     mutation of the graph between replays
//! (d) the pair ADDRESSES and does not only retire: one capture handed the
//!     plane's base replays at `(count 4, start 3)` and writes rows 3..7 and
//!     nothing else, then at `(8, 0)` and writes the whole plane — a start
//!     the recording never saw, off the same two words
//! (e) a ROUTER addresses off the pair: `linear.moe_topk_softmax` armed at
//!     `(3, 2)` over a six-row plane lands the routes and weights an unarmed
//!     run over the pre-sliced logits lands, and leaves rows 0, 1 and 5 as
//!     the arena left them
//! (f) the routed SELECT converts the pair: `linear.moe_matmul_select`'s grid
//!     counts ROUTES, `top_k` of them per token row, so armed at `(2, 1)` at
//!     fan-out two it computes result rows 2..6 — the window's routes and no
//!     others — against the same reference cut by hand
//! ```
//!
//! Gate (c) is wave 1's whole claim and gate (d) is wave 2a's. The staged
//! words are rewritten with a stream-ordered copy between replays, which is
//! precisely how an engine serving bodies will stage geometry: the write and
//! the launch ride one stream, so each replay reads the pair that was staged
//! for it.
//!
//! Gates (e) and (f) are the MoE wave's, and they are stated as EQUIVALENCE
//! rather than as a written-row mask: a router's answer is data and not a
//! sentinel, so the only honest verdict is that the armed window computes
//! what a launch handed pre-sliced planes computes. (f) is the one gate whose
//! kernel does not read the pair in the units it was written in — the seat's
//! words are token rows and its grid axis is routes — so it is the arithmetic
//! `win[0] * top_k`, `win[1] * top_k` that is on trial here.

mod common;

use core::ffi::c_void;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::res_blend;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select, topk_softmax};
use kernels_cuda::tensor::Tensor;

const ROWS: u32 = 8;
const HIDDEN: u32 = 32;
const EPS: f32 = 1.0e-6;

/// The sentinel a guarded-off row must keep: a bf16 bit pattern `res_blend`
/// never produces from all-ones inputs.
const SENTINEL: u16 = 0xAAAA;

/// bf16 `1.0`.
const ONE: u16 = 0x3F80;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// Fire the exemplar once: prefix of ones, no candidate blocks, unit weights.
/// Every row the guard admits comes out non-sentinel; every row it retires
/// keeps [`SENTINEL`].
fn fire(gpu: &mut Gpu, ctx: &Ctx, prefix_at: u64, w_at: u64, y_at: u64) {
    let _ = gpu;
    let mut y = Tensor::new(y_at, ROWS, HIDDEN, Dtype::Bf16);
    res_blend(
        ctx,
        Tensor::new(prefix_at, ROWS, HIDDEN, Dtype::Bf16),
        &[],
        Tensor::new(w_at, 1, HIDDEN, Dtype::Bf16),
        EPS,
        Tensor::new(w_at, 1, HIDDEN, Dtype::Bf16),
        &mut y,
    )
    .expect("the exemplar enqueues");
}

/// How many leading rows hold non-sentinel bytes, asserting the tail is
/// untouched — the shape every gate's verdict reads.
fn live_rows(gpu: &Gpu, y_at: u64) -> usize {
    let y: Vec<u16> = gpu.down(y_at, (ROWS * HIDDEN) as usize);
    let mut live = 0usize;
    for r in 0..ROWS as usize {
        let row = &y[r * HIDDEN as usize..(r + 1) * HIDDEN as usize];
        if row.iter().all(|&v| v == SENTINEL) {
            for later in r..ROWS as usize {
                let row = &y[later * HIDDEN as usize..(later + 1) * HIDDEN as usize];
                assert!(
                    row.iter().all(|&v| v == SENTINEL),
                    "row {later} written past a retired row {r}"
                );
            }
            return live;
        }
        assert!(
            row.iter().all(|&v| v != SENTINEL),
            "row {r} half-written"
        );
        live += 1;
    }
    live
}

/// Which rows hold non-sentinel bytes, as a mask — the verdict gate (d) needs
/// and [`live_rows`] cannot give, because an addressed window writes an
/// INTERVAL and a retired row before it is not the end of the answer.
fn written_rows(gpu: &Gpu, y_at: u64) -> Vec<bool> {
    let y: Vec<u16> = gpu.down(y_at, (ROWS * HIDDEN) as usize);
    (0..ROWS as usize)
        .map(|r| {
            let row = &y[r * HIDDEN as usize..(r + 1) * HIDDEN as usize];
            let written = row.iter().all(|&v| v != SENTINEL);
            assert!(
                written || row.iter().all(|&v| v == SENTINEL),
                "row {r} half-written"
            );
            written
        })
        .collect()
}

fn sentinel_fill(_gpu: &Gpu, y_at: u64) {
    let fill = vec![SENTINEL; (ROWS * HIDDEN) as usize];
    unsafe {
        check(
            rt::cudaMemcpy(
                y_at as *mut c_void,
                fill.as_ptr().cast(),
                fill.len() * 2,
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy sentinel",
        );
    }
}

#[test]
fn the_staged_seat_holds_its_six_gates() {
    let mut gpu = Gpu::open();
    let prefix_at = gpu.up(&vec![ONE; (ROWS * HIDDEN) as usize]);
    let w_at = gpu.up(&vec![ONE; HIDDEN as usize]);
    let y_at = gpu.zeros((ROWS * HIDDEN) as usize * 2);
    // The seat is FOUR words: the live-row count, the row it starts at, and
    // the lane pair this row-gridded exemplar never reads (see the header).
    let staged_at = gpu.up(&[5u32, 0, 0, 0]);
    let ctx = gpu.ctx();

    // (a) no stage armed: all eight rows written.
    sentinel_fill(&gpu, y_at);
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    gpu.sync();
    assert_eq!(live_rows(&gpu, y_at), 8, "the null seat retired rows");

    // (b) armed at 5: three rows keep their bytes, eagerly.
    ctx.arm_stage(staged_at);
    sentinel_fill(&gpu, y_at);
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    ctx.disarm_stage();
    gpu.sync();
    assert_eq!(live_rows(&gpu, y_at), 5, "the armed seat did not guard");
    drop(gpu);

    // (c) runs after (a)/(b) IN THIS ONE TEST FUNCTION, deliberately: a
    // stream capture is process-visible (a legacy-stream memcpy on another
    // test thread answers `cudaErrorStreamCaptureImplicit`), so the capture
    // gate shares nobody's wall clock.
    one_capture_serves_three_row_counts_off_the_staged_word();

    // (d) runs after (c) for the same reason (c) runs after (a)/(b): it
    // captures a stream, and a capture is process-visible.
    one_capture_serves_a_start_the_recording_never_saw();

    // (e) and (f) capture nothing — an armed EAGER fire is all the seat's
    // arithmetic needs to be made to speak — but they run last anyway, on
    // their own devices, because that is this file's one order.
    a_router_lands_the_window_the_pair_names();
    a_routed_select_converts_the_pair_to_its_own_axis();
}

fn one_capture_serves_three_row_counts_off_the_staged_word() {
    let mut gpu = Gpu::open();
    let prefix_at = gpu.up(&vec![ONE; (ROWS * HIDDEN) as usize]);
    let w_at = gpu.up(&vec![ONE; HIDDEN as usize]);
    let y_at = gpu.zeros((ROWS * HIDDEN) as usize * 2);
    let staged_at = gpu.up(&[ROWS, 0, 0, 0]);
    let ctx = gpu.ctx();

    // Warm: the module must be resident before the capture — a JIT inside
    // `cudaStreamBeginCapture` is host work the capture mode refuses.
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    gpu.sync();

    // Record once, at the ceiling, seat armed.
    let stream: rt::cudaStream_t = ctx.stream().cast();
    let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
    unsafe {
        check(
            rt::cudaStreamBeginCapture(
                stream,
                rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        );
    }
    ctx.arm_stage(staged_at);
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    ctx.disarm_stage();
    unsafe {
        check(
            rt::cudaStreamEndCapture(stream, &raw mut graph),
            "cudaStreamEndCapture",
        );
    }
    let mut exec: rt::cudaGraphExec_t = core::ptr::null_mut();
    unsafe {
        check(
            rt::cudaGraphInstantiate(&raw mut exec, graph, 0),
            "cudaGraphInstantiate",
        );
    }

    // Replay at 5, at 8, at 3: the staged COUNT is the only thing that moves —
    // the start stays 0, which is gate (d)'s to vary.
    for rows in [5u32, 8, 3] {
        sentinel_fill(&gpu, y_at);
        unsafe {
            check(
                rt::cudaMemcpyAsync(
                    staged_at as *mut c_void,
                    (&raw const rows).cast(),
                    4,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync staged rows",
            );
            check(rt::cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
        }
        gpu.sync();
        assert_eq!(
            live_rows(&gpu, y_at),
            rows as usize,
            "the replay did not serve the staged count"
        );
    }

    unsafe {
        rt::cudaGraphExecDestroy(exec);
        rt::cudaGraphDestroy(graph);
    }
}

/// **GATE (d): THE PAIR ADDRESSES.**
///
/// Everything above stages a start of 0, where `win[1]` cannot be told from
/// its own absence — the engine arms only whole-fire windows today, so the
/// addressing half of the seat is behaviorally invisible in production and
/// this is the only place it is made to speak.
///
/// The entry is handed the PLANE's base pointers, which is what the earlier
/// gates already do (their window is the whole plane) and what an engine
/// serving a windowed body must do: a pre-shifted pointer plus a start would
/// count the offset twice. One capture at the ceiling then replays at
/// `(count 4, start 3)` — eight blocks launch, four survive the guard, and
/// they write plane rows 3, 4, 5 and 6, leaving 0..3 and 7 as the arena left
/// them — and at `(8, 0)`, which is the whole plane again.
fn one_capture_serves_a_start_the_recording_never_saw() {
    let mut gpu = Gpu::open();
    let prefix_at = gpu.up(&vec![ONE; (ROWS * HIDDEN) as usize]);
    let w_at = gpu.up(&vec![ONE; HIDDEN as usize]);
    let y_at = gpu.zeros((ROWS * HIDDEN) as usize * 2);
    let staged_at = gpu.up(&[ROWS, 0, 0, 0]);
    let ctx = gpu.ctx();

    // Warm, then record once at the ceiling — the same order gate (c) keeps.
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    gpu.sync();

    let stream: rt::cudaStream_t = ctx.stream().cast();
    let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
    unsafe {
        check(
            rt::cudaStreamBeginCapture(
                stream,
                rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        );
    }
    ctx.arm_stage(staged_at);
    fire(&mut gpu, &ctx, prefix_at, w_at, y_at);
    ctx.disarm_stage();
    unsafe {
        check(
            rt::cudaStreamEndCapture(stream, &raw mut graph),
            "cudaStreamEndCapture",
        );
    }
    let mut exec: rt::cudaGraphExec_t = core::ptr::null_mut();
    unsafe {
        check(
            rt::cudaGraphInstantiate(&raw mut exec, graph, 0),
            "cudaGraphInstantiate",
        );
    }

    // `(count, start)` in, the mask of written plane rows out.
    for (pair, want) in [
        ([4u32, 3], [false, false, false, true, true, true, true, false]),
        ([ROWS, 0], [true; ROWS as usize]),
    ] {
        sentinel_fill(&gpu, y_at);
        unsafe {
            check(
                rt::cudaMemcpyAsync(
                    staged_at as *mut c_void,
                    pair.as_ptr().cast(),
                    8,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync staged pair",
            );
            check(rt::cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
        }
        gpu.sync();
        assert_eq!(
            written_rows(&gpu, y_at),
            want.to_vec(),
            "the replay did not serve the staged interval {pair:?}"
        );
    }

    unsafe {
        rt::cudaGraphExecDestroy(exec);
        rt::cudaGraphDestroy(graph);
    }
}

/// **GATE (e): A ROUTER ADDRESSES OFF THE PAIR.**
///
/// The routers are the first MoE entries on the seat, and what they have to
/// prove is not a mask of written rows — a router lands ids and weights, both
/// of them data no sentinel can be told apart from by looking — but an
/// EQUIVALENCE: armed at `(count 3, start 2)` over a six-row plane handed at
/// its own base, `linear.moe_topk_softmax` lands on plane rows 2, 3 and 4
/// exactly what a second, unarmed fire lands when it is handed the same three
/// rows of logits pre-sliced. The rows outside the window are the arena's, and
/// they keep the fill this test put there.
fn a_router_lands_the_window_the_pair_names() {
    const TOKENS: u32 = 6;
    const EXPERTS: u32 = 4;
    const TOP_K: u32 = 2;
    const START: u32 = 2;
    const LIVE: u32 = 3;

    /// Ids and weights a top-2 softmax over four experts never lands.
    const ROUTE_FILL: i32 = -12_345;
    const WEIGHT_FILL: f32 = -12_345.0;

    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x5eed_0e);
    let (logits, _) = lcg.row((TOKENS * EXPERTS) as usize);
    let logits_at = gpu.up(&logits);
    let routes_at = gpu.up(&vec![ROUTE_FILL; (TOKENS * TOP_K) as usize]);
    let weights_at = gpu.up(&vec![WEIGHT_FILL; (TOKENS * TOP_K) as usize]);
    let staged_at = gpu.up(&[LIVE, START, 0, 0]);
    let ctx = gpu.ctx();

    // Armed: six blocks launch over the whole plane, three survive the guard,
    // and the three that do own plane rows 2, 3 and 4.
    let mut routes = Tensor::new(routes_at, TOKENS, TOP_K, Dtype::I32);
    let mut weights = Tensor::new(weights_at, TOKENS, TOP_K, Dtype::F32);
    ctx.arm_stage(staged_at);
    topk_softmax(
        &ctx,
        Tensor::new(logits_at, TOKENS, EXPERTS, Dtype::Bf16),
        EXPERTS,
        TOP_K,
        &mut routes,
        &mut weights,
    )
    .expect("the armed router enqueues");
    ctx.disarm_stage();

    // The reference: the same three rows, cut by hand, no seat at all.
    let want_routes_at = gpu.up(&vec![ROUTE_FILL; (LIVE * TOP_K) as usize]);
    let want_weights_at = gpu.up(&vec![WEIGHT_FILL; (LIVE * TOP_K) as usize]);
    let mut want_routes = Tensor::new(want_routes_at, LIVE, TOP_K, Dtype::I32);
    let mut want_weights = Tensor::new(want_weights_at, LIVE, TOP_K, Dtype::F32);
    topk_softmax(
        &ctx,
        Tensor::new(
            logits_at + u64::from(START * EXPERTS) * 2,
            LIVE,
            EXPERTS,
            Dtype::Bf16,
        ),
        EXPERTS,
        TOP_K,
        &mut want_routes,
        &mut want_weights,
    )
    .expect("the unarmed reference enqueues");
    gpu.sync();

    let got_routes: Vec<i32> = gpu.down(routes_at, (TOKENS * TOP_K) as usize);
    let got_weights: Vec<f32> = gpu.down(weights_at, (TOKENS * TOP_K) as usize);
    let want_routes: Vec<i32> = gpu.down(want_routes_at, (LIVE * TOP_K) as usize);
    let want_weights: Vec<f32> = gpu.down(want_weights_at, (LIVE * TOP_K) as usize);
    for row in 0..TOKENS as usize {
        let inside = row >= START as usize && row < (START + LIVE) as usize;
        for k in 0..TOP_K as usize {
            let at = row * TOP_K as usize + k;
            if inside {
                let want = (row - START as usize) * TOP_K as usize + k;
                assert_eq!(
                    got_routes[at], want_routes[want],
                    "row {row} slot {k}: the armed router chose another row's expert"
                );
                assert!(
                    close(got_weights[at], want_weights[want]),
                    "row {row} slot {k}: {} against the pre-sliced {}",
                    got_weights[at],
                    want_weights[want]
                );
            } else {
                assert_eq!(
                    got_routes[at], ROUTE_FILL,
                    "row {row} slot {k} is outside the window and was written"
                );
                assert_eq!(
                    got_weights[at].to_bits(),
                    WEIGHT_FILL.to_bits(),
                    "row {row} slot {k} is outside the window and was written"
                );
            }
        }
    }
}

/// **GATE (f): THE ROUTED SELECT CONVERTS THE PAIR TO ITS OWN AXIS.**
///
/// `linear.moe_matmul_select` is the first name on `engine_cuda::SHIFTED`
/// whose grid does not count the rows the seat's words count. The seat is the
/// REGION's, and a region's rows are TOKEN rows; this grid's y axis is one
/// ROUTE per token row per slot. So the kernel multiplies: `win[0] * top_k`
/// routes are live and they begin at route `win[1] * top_k`, and the fan-out
/// it multiplies by is the one the routes plane's width states.
///
/// Armed at `(count 2, start 1)` at fan-out two, the four live routes are
/// result rows 2, 3, 4 and 5 — read against an unarmed fire handed the same
/// two token rows of activation and of routes, pre-sliced. The window holds a
/// negative route id on purpose, so the zero arm is made to take the shifted
/// ordinal too.
fn a_routed_select_converts_the_pair_to_its_own_axis() {
    const TOKENS: u32 = 4;
    const TOP_K: u32 = 2;
    const EXPERTS: u32 = 3;
    /// K in whole float4 loads, which is the only K this GEMV accepts.
    const K: u32 = 8;
    const N: u32 = 4;
    const START: u32 = 1;
    const LIVE: u32 = 2;

    let routes_wide = TOKENS * TOP_K;

    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x5eed_1f);
    let (x, _) = lcg.row((TOKENS * K) as usize);
    let (bank, _) = lcg.row((EXPERTS * N * K) as usize);
    let x_at = gpu.up(&x);
    let bank_at = gpu.up(&bank);
    // Flat, route-major, the layout the kernel reads: token row `t`, slot `k`
    // at `t * top_k + k`. Routes 2..6 are the window's, and one of them is the
    // negative id that lands a zero row.
    let routes_at = gpu.up(&[0_i32, 1, 2, -1, 1, 0, 2, 1]);
    let y_at = gpu.up(&vec![SENTINEL; (routes_wide * N) as usize]);
    let staged_at = gpu.up(&[LIVE, START, 0, 0]);
    let ctx = gpu.ctx();

    let mut y = Tensor::new(y_at, routes_wide, N, Dtype::Bf16);
    ctx.arm_stage(staged_at);
    matmul_select(
        &ctx,
        Tensor::new(x_at, TOKENS, K, Dtype::Bf16),
        Tensor::new(bank_at, EXPERTS * N, K, Dtype::Bf16),
        Tensor::new(routes_at, TOKENS, TOP_K, Dtype::I32),
        &mut y,
        ExpertTable::RESIDENT,
    )
    .expect("the armed select enqueues");
    ctx.disarm_stage();

    // The reference: two token rows of activation and of routes, cut by hand,
    // and a result that is only their four routes.
    let want_at = gpu.up(&vec![SENTINEL; (LIVE * TOP_K * N) as usize]);
    let mut want = Tensor::new(want_at, LIVE * TOP_K, N, Dtype::Bf16);
    matmul_select(
        &ctx,
        Tensor::new(x_at + u64::from(START * K) * 2, LIVE, K, Dtype::Bf16),
        Tensor::new(bank_at, EXPERTS * N, K, Dtype::Bf16),
        Tensor::new(
            routes_at + u64::from(START * TOP_K) * 4,
            LIVE,
            TOP_K,
            Dtype::I32,
        ),
        &mut want,
        ExpertTable::RESIDENT,
    )
    .expect("the unarmed reference enqueues");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, (routes_wide * N) as usize);
    let want: Vec<u16> = gpu.down(want_at, (LIVE * TOP_K * N) as usize);
    let first = (START * TOP_K) as usize;
    let past = ((START + LIVE) * TOP_K) as usize;
    for route in 0..routes_wide as usize {
        for col in 0..N as usize {
            let at = route * N as usize + col;
            if route >= first && route < past {
                let mirror = (route - first) * N as usize + col;
                assert!(
                    close(from_bf16(got[at]), from_bf16(want[mirror])),
                    "route {route} col {col}: {} against the pre-sliced {}",
                    from_bf16(got[at]),
                    from_bf16(want[mirror])
                );
            } else {
                assert_eq!(
                    got[at], SENTINEL,
                    "route {route} col {col} is outside the window and was written"
                );
            }
        }
    }
}
