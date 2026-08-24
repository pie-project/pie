//! The two dense `Moe` bodies R4a claimed: `moe.matmul_select` and
//! `moe.sigmoid_gate_add`.
//!
//! # `moe.matmul_select`
//!
//! THE REFERENCE IS THE ARITHMETIC, in fp64, one product at a time. The
//! point is `y[r] = x[r] @ bank[routes[r]]` over a dense `[E, N, K]` stack,
//! and unlike its biased MXFP4 sibling there is no decoding to get wrong —
//! so the only things a reference can disagree with are WHICH row of the
//! bank each route reads and in WHAT ORDER the dot accumulates. The kernel
//! walks `K` strided by lane in `float4`s and folds with a warp shuffle
//! tree; the reference walks the row. Both accumulate in fp32-or-wider, so
//! the difference that survives the bf16 store is reassociation, far below
//! bf16's eight mantissa bits.
//!
//! The canon row this body replaces was `moe::moe_grouped_gemm`, whose own
//! `supported` gate refused every `K` above 512 — so it never fired for any
//! SKU that states this point, and there is no second implementation in the
//! tree to check against. The spec is the bar for the same reason it is for
//! `matmul_select_bias`.
//!
//! ## The mutation
//!
//! The ROUTES rotated. Every row still names a legal expert, so nothing
//! faults and nothing reads out of bounds; what changes is only WHICH expert
//! each route picks. And the accounting is PER ROUTE, which is the stronger
//! claim: where a rotation happens to leave a route on the expert it already
//! had, that route's whole block must be BIT-IDENTICAL, and every other
//! block must move. A body that indexed the bank by the route's own ordinal,
//! or that read the route run at an offset, fails one half or the other.
//!
//! ## And `act_div`, which is measured and not stated
//!
//! Both shapes are fired: the gate/up leg's `x` is PER TOKEN and the down
//! leg's is already fanned out to one row per route. The body reads the
//! ratio off `y.rows / x.rows` and picks between the kernel's two
//! instantiations, so a body that guessed would be wrong on one of the two.
//!
//! # `moe.sigmoid_gate_add`
//!
//! `y = routed + shared * sigmoid(gate)`, with one gate value per row
//! broadcast across it. The reference is that expression in fp64.
//!
//! ## The mutations
//!
//! * the GATE dropped — replaced by a zero column, which is `sigmoid(0) =
//!   0.5` and not "no shared expert": a body that ignored the column would
//!   agree with the unmutated run instead;
//! * the gate column ROTATED between rows — every value is still a legal
//!   gate, so only the row-to-row assignment changes, and a body that read
//!   `scalar_gate[0]` for every row (or walked the column with the wrong
//!   pitch) passes a self-consistent test and fails this one.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Moe;
use kernels::routine::{Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time. `matmul_select_bias.rs`'s lock, verbatim and for its
/// reason.
static FIRE: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── the device, or a skip ────────────────────────────────────────────────

fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

fn device_or_skip(what: &str) -> bool {
    let Some(count) = quietly(|| {
        let mut n: i32 = 0;
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        (code == rt::cudaError::cudaSuccess).then_some(n)
    }) else {
        eprintln!("skipping {what}: no CUDA runtime library on this machine");
        return false;
    };
    match count {
        Some(n) if n > 0 => {}
        _ => {
            eprintln!("skipping {what}: no CUDA device this build can drive");
            return false;
        }
    }
    assert_eq!(
        unsafe { rt::cudaSetDevice(0) },
        rt::cudaError::cudaSuccess,
        "a device is present but cudaSetDevice(0) failed"
    );
    assert_eq!(
        unsafe { rt::cudaFree(core::ptr::null_mut()) },
        rt::cudaError::cudaSuccess,
        "a device is present but the primary context would not come up"
    );
    true
}

// ── device memory, freed when the run ends ───────────────────────────────

struct Slab {
    ptr: *mut c_void,
}

impl Slab {
    fn of(bytes: &[u8]) -> Slab {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes.len().max(1)) },
            rt::cudaError::cudaSuccess,
            "cudaMalloc({})",
            bytes.len()
        );
        let slab = Slab { ptr };
        if !bytes.is_empty() {
            assert_eq!(
                unsafe {
                    rt::cudaMemcpy(
                        slab.ptr,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )
                },
                rt::cudaError::cudaSuccess,
                "host to device"
            );
        }
        slab
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        let mut bytes = vec![0u8; elems * 2];
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.ptr,
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { rt::cudaFree(self.ptr) };
        }
    }
}

// ── elements ─────────────────────────────────────────────────────────────

fn wide(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `__float2bfloat16`: round to nearest, ties to even.
fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// One bf16 ulp at `x`: the gap between neighbouring bf16 values there.
fn ulp(x: f32) -> f32 {
    let e = x.abs().to_bits() & 0x7F80_0000;
    f32::from_bits(e.max(1 << 23)) / 128.0
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.bits() >> 40) as f32) / 8_388_608.0 - 1.0
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.bits() >> 32) as u32 % n
    }
}

/// The NaN poison every result slab starts as: no reference value below is
/// anything but finite, so a survivor is a slot the kernel never wrote.
const POISON: u16 = 0x7FC0;

// ── moe.matmul_select ────────────────────────────────────────────────────

/// What the point is handed, at one shape.
#[derive(Clone, Copy)]
struct Select {
    what: &'static str,
    /// Whether the activation is already per ROUTE (the down leg) or per
    /// TOKEN (the gate/up leg) — the `act_div` the body measures rather than
    /// is told.
    per_route: bool,
    experts: usize,
    tokens: usize,
    top_k: usize,
    n: usize,
    k: usize,
}

impl Select {
    fn routes(self) -> usize {
        self.tokens * self.top_k
    }

    fn act_rows(self) -> usize {
        if self.per_route {
            self.routes()
        } else {
            self.tokens
        }
    }
}

struct Ran {
    got: Vec<u16>,
    want: Vec<u16>,
    exact: Vec<f32>,
}

fn run_select(ctx: &Ctx<'_>, c: Select, bank: &[u16], act: &[u16], routes: &[i32]) -> Ran {
    let routes_n = c.routes();
    let out_n = routes_n * c.n;

    let d_bank = Slab::of(&bytes_of_u16(bank));
    let d_act = Slab::of(&bytes_of_u16(act));
    let d_routes = Slab::of(&bytes_of_i32(routes));
    let d_out = Slab::of(&bytes_of_u16(&vec![POISON; out_n]));

    Moe::matmul_select::<bf16>(
        ctx,
        In {
            ptr: d_act.ptr.cast(),
            rows: c.act_rows() as i32,
            width: c.k as i32,
        },
        Const::new(d_bank.ptr.cast_const().cast()),
        In {
            ptr: d_routes.ptr.cast(),
            rows: c.tokens as i32,
            width: c.top_k as i32,
        },
        Out {
            ptr: d_out.ptr.cast(),
            rows: routes_n as i32,
            width: c.n as i32,
        },
    )
    .expect("the claimed `moe.matmul_select` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the routed GEMV did not complete"
    );

    let mut want = Vec::with_capacity(out_n);
    let mut exact = Vec::with_capacity(out_n);
    for r in 0..routes_n {
        let e = routes[r] as usize;
        let at = if c.per_route { r } else { r / c.top_k };
        let x = &act[at * c.k..(at + 1) * c.k];
        for row in 0..c.n {
            let w = &bank[(e * c.n + row) * c.k..(e * c.n + row + 1) * c.k];
            let v: f64 = (0..c.k)
                .map(|j| f64::from(wide(w[j])) * f64::from(wide(x[j])))
                .sum();
            exact.push(v as f32);
            want.push(narrow(v as f32));
        }
    }
    Ran {
        got: d_out.read_u16(out_n),
        want,
        exact,
    }
}

/// a3b's two routed statements, at a3b's own contraction depths.
///
/// `k = 2048` is the real hidden size of qwen3.5-a3b (the gate/up leg's
/// contraction) and `k = 768` its real per-expert intermediate (the down
/// leg's) — both far past the `K <= 512` the retired `moe_grouped_gemm`
/// would have accepted, which is the measurement that says this body is the
/// launcher and that one never was. `n` is cut down from `2 * 768` and
/// `2048`: the reference walks one product at a time in fp64, so the product
/// `routes * n * k` is what a test can afford, and `n` is the axis the
/// kernel parallelises over rather than the one it accumulates along.
const A3B: &[Select] = &[
    Select {
        what: "gate/up (per-token activation, k = 2048)",
        per_route: false,
        experts: 6,
        tokens: 3,
        top_k: 4,
        n: 128,
        k: 2048,
    },
    Select {
        what: "down (per-route activation, k = 768)",
        per_route: true,
        experts: 6,
        tokens: 3,
        top_k: 4,
        n: 96,
        k: 768,
    },
    // `n` is not a whole number of the four rows a warp writes, so the last
    // warp overhangs. A body that let the overhang store would corrupt row
    // `n - 1`; one that let it LOAD would read past the bank.
    Select {
        what: "an n that is not a whole warp slab",
        per_route: false,
        experts: 5,
        tokens: 2,
        top_k: 3,
        n: 99,
        k: 128,
    },
];

fn sample_select(c: Select, seed: u64) -> (Vec<u16>, Vec<u16>, Vec<i32>) {
    let mut rng = Rng(seed);
    // Scaled so a `k`-deep random-walk dot stays of order one; a bank whose
    // every dot saturated would be comparing two saturations.
    let scale = (c.k as f32).sqrt().recip();
    let bank: Vec<u16> = (0..c.experts * c.n * c.k)
        .map(|_| narrow(rng.unit() * scale))
        .collect();
    let act: Vec<u16> = (0..c.act_rows() * c.k)
        .map(|_| narrow(rng.unit()))
        .collect();
    let routes: Vec<i32> = (0..c.routes())
        .map(|_| rng.below(c.experts as u32) as i32)
        .collect();
    (bank, act, routes)
}

#[test]
fn the_routed_gemv_is_the_dot_the_route_names() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.matmul_select") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    for (i, c) in A3B.iter().enumerate() {
        let (bank, act, routes) = sample_select(*c, 0x9e37_79b9_7f4a_7c15 ^ (i as u64 + 1));
        let r = run_select(&ctx, *c, &bank, &act, &routes);
        let n = r.want.len();

        let identical = (0..n).filter(|i| r.got[*i] == r.want[*i]).count();
        let (mut worst, mut worst_at) = (0.0f32, 0usize);
        for i in 0..n {
            let miss = (wide(r.got[i]) - r.exact[i]).abs() / ulp(r.exact[i]);
            if miss > worst {
                worst = miss;
                worst_at = i;
            }
        }
        eprintln!(
            "moe.matmul_select [{}]: {identical}/{n} bit-identical to the fp64 reference, \
             worst miss {worst:.3} bf16 ulp at {worst_at} (got {:+.6}, want {:+.6})",
            c.what,
            wide(r.got[worst_at]),
            r.exact[worst_at],
        );
        assert!(
            r.got.iter().all(|b| wide(*b).is_finite()),
            "[{}]: the kernel left a slot unwritten (the NaN poison survived)",
            c.what
        );
        assert!(
            worst <= 1.0,
            "[{}]: {worst:.3} bf16 ulp at element {worst_at}",
            c.what
        );
        assert!(
            identical * 100 >= n * 95,
            "[{}]: only {identical}/{n} elements are bit-identical",
            c.what
        );
    }
}

#[test]
fn the_routes_pick_the_expert() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.matmul_select") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    let c = A3B[0];
    let (bank, act, routes) = sample_select(c, 0x2545_f491_4f6c_dd1d);
    let base = run_select(&ctx, c, &bank, &act, &routes);

    let mut rotated = routes.clone();
    rotated.rotate_left(1);
    let permuted = run_select(&ctx, c, &bank, &act, &rotated);

    let (mut moved, mut held) = (0usize, 0usize);
    for r in 0..c.routes() {
        let block = r * c.n..(r + 1) * c.n;
        let differ = block
            .clone()
            .filter(|i| permuted.got[*i] != base.got[*i])
            .count();
        if rotated[r] == routes[r] {
            assert_eq!(
                differ, 0,
                "route {r} kept expert {} and {differ}/{} of its row moved",
                routes[r], c.n
            );
            held += c.n;
        } else {
            assert!(
                differ * 100 >= c.n * 99,
                "route {r} moved from expert {} to {} and only {differ}/{} of its row changed",
                routes[r],
                rotated[r],
                c.n
            );
            moved += differ;
        }
    }
    assert!(
        moved > 0,
        "the rotation happened to be the identity: nothing was measured"
    );
    eprintln!(
        "moe.matmul_select [routes rotated]: {moved} elements moved, {held} held on the \
         routes the rotation left alone"
    );
}

// ── moe.sigmoid_gate_add ─────────────────────────────────────────────────

/// Fire the combine once and read both sides. `rows`/`width` are the shared
/// expert's landing; `gate_width` is the column's own rectangle, which is
/// the pitch between two rows' gate values.
fn run_gate(
    ctx: &Ctx<'_>,
    rows: usize,
    width: usize,
    routed: &[u16],
    shared: &[u16],
    gate: &[u16],
    gate_width: usize,
) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    let n = rows * width;
    let d_routed = Slab::of(&bytes_of_u16(routed));
    let d_shared = Slab::of(&bytes_of_u16(shared));
    let d_gate = Slab::of(&bytes_of_u16(gate));
    let d_out = Slab::of(&bytes_of_u16(&vec![POISON; n]));

    Moe::sigmoid_gate_add::<bf16>(
        ctx,
        In {
            ptr: d_routed.ptr.cast(),
            rows: rows as i32,
            width: width as i32,
        },
        In {
            ptr: d_shared.ptr.cast(),
            rows: rows as i32,
            width: width as i32,
        },
        In {
            ptr: d_gate.ptr.cast(),
            rows: rows as i32,
            width: gate_width as i32,
        },
        Out {
            ptr: d_out.ptr.cast(),
            rows: rows as i32,
            width: width as i32,
        },
    )
    .expect("the claimed `moe.sigmoid_gate_add` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the combine did not complete"
    );

    let mut want = Vec::with_capacity(n);
    let mut exact = Vec::with_capacity(n);
    for r in 0..rows {
        let g = f64::from(wide(gate[r * gate_width]));
        let s = 1.0 / (1.0 + (-g).exp());
        for h in 0..width {
            let i = r * width + h;
            let v = f64::from(wide(routed[i])) + f64::from(wide(shared[i])) * s;
            exact.push(v as f32);
            want.push(narrow(v as f32));
        }
    }
    (d_out.read_u16(n), want, exact)
}

fn sample_gate(rows: usize, width: usize, seed: u64) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut rng = Rng(seed);
    let routed: Vec<u16> = (0..rows * width).map(|_| narrow(rng.unit())).collect();
    let shared: Vec<u16> = (0..rows * width).map(|_| narrow(rng.unit())).collect();
    // Gates spread over the sigmoid's whole interesting range, so no two
    // rows share a factor and a mis-assigned column is visible.
    let gate: Vec<u16> = (0..rows).map(|_| narrow(rng.unit() * 6.0)).collect();
    (routed, shared, gate)
}

#[test]
fn the_combine_is_the_gated_sum() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.sigmoid_gate_add") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    // a3b's hidden, at a decode row and at a prefill window.
    for (rows, width) in [(1usize, 2048usize), (23, 2048), (7, 129)] {
        let (routed, shared, gate) = sample_gate(rows, width, 0xdead_beef ^ rows as u64);
        let (got, want, exact) = run_gate(&ctx, rows, width, &routed, &shared, &gate, 1);
        let n = want.len();
        let identical = (0..n).filter(|i| got[*i] == want[*i]).count();
        let (mut worst, mut worst_at) = (0.0f32, 0usize);
        for i in 0..n {
            let miss = (wide(got[i]) - exact[i]).abs() / ulp(exact[i]);
            if miss > worst {
                worst = miss;
                worst_at = i;
            }
        }
        eprintln!(
            "moe.sigmoid_gate_add [{rows}x{width}]: {identical}/{n} bit-identical, worst \
             miss {worst:.3} bf16 ulp at {worst_at}"
        );
        assert!(
            got.iter().all(|b| wide(*b).is_finite()),
            "[{rows}x{width}]: the kernel left a slot unwritten (the NaN poison survived)"
        );
        assert!(worst <= 1.0, "[{rows}x{width}]: {worst:.3} bf16 ulp");
        assert!(
            identical * 100 >= n * 99,
            "[{rows}x{width}]: only {identical}/{n} elements are bit-identical"
        );
    }
}

#[test]
fn the_gate_column_is_read_per_row() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.sigmoid_gate_add") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    let (rows, width) = (16usize, 512usize);
    let (routed, shared, gate) = sample_gate(rows, width, 0x0f1e_2d3c_4b5a_6978);
    let (base, _, _) = run_gate(&ctx, rows, width, &routed, &shared, &gate, 1);

    // ── the gate dropped ──
    //
    // A ZERO column, not an absent one: `sigmoid(0) = 0.5`, so the shared
    // expert still lands at half weight and a body that ignored the column
    // entirely would agree with `base` rather than with this.
    let (zeroed, _, _) = run_gate(&ctx, rows, width, &routed, &shared, &vec![0u16; rows], 1);
    let gate_moved = (0..base.len()).filter(|i| zeroed[*i] != base[*i]).count();

    // ── the gate rotated between rows ──
    //
    // Every value is still a legal gate; only which row gets which changes.
    // A body that read `scalar_gate[0]` for every row, or walked the column
    // with the wrong pitch, is bit-identical to `base` here.
    let mut rotated = gate.clone();
    rotated.rotate_left(1);
    let (moved_gate, _, _) = run_gate(&ctx, rows, width, &routed, &shared, &rotated, 1);
    let mut per_row_moved = 0usize;
    for r in 0..rows {
        let block = r * width..(r + 1) * width;
        let differ = block.filter(|i| moved_gate[*i] != base[*i]).count();
        assert!(
            differ * 100 >= width * 95,
            "row {r}'s gate went from {:+.4} to {:+.4} and only {differ}/{width} of the \
             row changed",
            wide(gate[r]),
            wide(rotated[r])
        );
        per_row_moved += differ;
    }

    eprintln!(
        "moe.sigmoid_gate_add [mutations]: gate zeroed moves {gate_moved}/{}, gate rotated \
         moves {per_row_moved}/{}",
        base.len(),
        base.len()
    );
    assert!(
        gate_moved * 100 >= base.len() * 95,
        "the gate is barely read: only {gate_moved}/{} elements move when it is zeroed",
        base.len()
    );

    // ── the column as one column of a WIDER rectangle ──
    //
    // `stride` in the device text is the pitch between two rows' gate
    // values, and it is the column's own rectangle that says what it is.
    // The same gates, padded to a three-wide row with junk beside them, must
    // give the same answer.
    const PAD: usize = 3;
    let mut wide_gate = vec![narrow(9.0); rows * PAD];
    for r in 0..rows {
        wide_gate[r * PAD] = gate[r];
    }
    let (strided, _, _) = run_gate(&ctx, rows, width, &routed, &shared, &wide_gate, PAD);
    assert_eq!(
        strided, base,
        "the same gates one column of a {PAD}-wide row did not give the same answer"
    );
}
