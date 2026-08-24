//! `moe.matmul_select_bias`'s claimed body against a host reading of the
//! MXFP4 bank it was handed.
//!
//! THE REFERENCE IS NOT ANOTHER KERNEL, and it cannot be. `swiglu_clamp_alpha`
//! next door checks a transcription — one activation, two operand shapes — so
//! its bar is bit equality against the kernel it was copied from. This point
//! is a body with NO sibling: the two MXFP4 GEMVs in `crate::quant` fuse an
//! activation into their epilogue and one of them hard-wires its operand
//! indexing, so neither computes `y[r] = x[r] @ bank[routes[r]] + bias` on its
//! own. The third party that decides is therefore the SPEC — the OCP MX FP4
//! decoding, written out here in fp64 — and the bar is bf16 rounding.
//!
//! # What the reference is allowed to know
//!
//! The E2M1 codepoint table and the E8M0 exponent rule, both spelled from the
//! standard rather than lifted from `dequant_fp4.cuh`; the row-major byte
//! layout the checkpoint ships. Nothing else. It walks the bank the slow way,
//! one code at a time in fp64, so a disagreement is the kernel's.
//!
//! # The tolerance, and why it is tight
//!
//! Both sides sum in floating point and neither sums in the other's ORDER —
//! the kernel walks 32-code groups strided by lane and folds with a warp
//! shuffle tree, the reference walks the row. So exact equality is not the
//! bar. But the kernel accumulates in fp32 (the header of
//! `mxfp4_matmul_select_bias` argues why, against the fp16 the shipped decode
//! GEMVs use), so the only difference that survives the bf16 store is fp32
//! reassociation, which is far below bf16's own eight mantissa bits. The
//! assertion is therefore that nearly every element is BIT-IDENTICAL and that
//! no element misses by more than one bf16 ulp.
//!
//! # The mutations
//!
//! A numeric gate that only ever compares two agreeing implementations proves
//! nothing about what it would catch, so three faults are injected and
//! measured:
//!
//! * the BIAS dropped — gpt-oss's own gate/up bias is `|max| 3.7` against a
//!   dot of order one, so this is not a subtle difference, and it is exactly
//!   the fault the legacy MXFP4 leg shipped: `mxfp4_moe_gate_up_decode_bf16`
//!   passes `ctx.absent()` for both bias pointer arrays, so the shipped
//!   gpt-oss path adds no gate/up bias at all;
//! * the ROUTES permuted — every row still reads a legal expert, so a body
//!   that ignored `routes[r]` and used route `r`'s ordinal, or that indexed
//!   the bank at the wrong stride, passes a self-consistent test and fails
//!   this one;
//! * the SCALES flattened to a constant exponent — a body that read the codes
//!   and never applied the block scale would otherwise agree to within a
//!   global factor on a bank whose exponents happened to be uniform.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::{Moe, Mxfp4};
use kernels::routine::{Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::{Planes, bf16};

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time. `swiglu_clamp_alpha.rs`'s lock, verbatim and for its
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

    fn byte(&mut self) -> u8 {
        (self.bits() >> 33) as u8
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.bits() >> 32) as u32 % n
    }
}

// ── the MXFP4 spec, as the reference reads it ────────────────────────────

/// OCP MX FP4's E2M1 codepoints: sign in bit 3, a two-bit exponent biased at
/// 1 in bits 2..1, one mantissa bit in bit 0. Written out rather than
/// derived, because the two subnormal codes (0 and 0.5) are the ones a
/// derivation gets wrong.
const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E8M0: the byte IS the exponent, biased at 127. `0` means `2^-127` and not
/// zero, which is the one place a bit-pattern shortcut diverges.
fn e8m0(b: u8) -> f64 {
    (f64::from(b) - 127.0).exp2()
}

/// One bank's two planes: `codes[e][n][k/2]` and `scales[e][n][k/32]`, which
/// is the `[E, N, K/32, 16]` / `[E, N, K/32]` pair a checkpoint ships,
/// flattened. The kernel indexes bytes; so does this.
struct Bank {
    codes: Vec<u8>,
    scales: Vec<u8>,
    n: usize,
    k: usize,
}

impl Bank {
    /// A random bank whose exponents sit in a window that keeps a `k`-deep dot
    /// of order one. A bank scaled so every dot saturated would be comparing
    /// two saturations.
    fn random(rng: &mut Rng, experts: usize, n: usize, k: usize) -> Bank {
        let groups = k / 32;
        Bank {
            codes: (0..experts * n * k / 2).map(|_| rng.byte()).collect(),
            // 2^-9 .. 2^-5, against `E2M1`'s mean magnitude near 2 and a
            // random-walk depth of `sqrt(k)`.
            scales: (0..experts * n * groups)
                .map(|_| 118 + rng.byte() % 5)
                .collect(),
            n,
            k,
        }
    }

    /// `bank[e][row] · x`, in fp64, one code at a time.
    fn dot(&self, e: usize, row: usize, x: &[u16]) -> f64 {
        let groups = self.k / 32;
        let row_codes = (e * self.n + row) * (self.k / 2);
        let row_scales = (e * self.n + row) * groups;
        let mut acc = 0.0f64;
        for g in 0..groups {
            // The block scale is constant over its 32 codes and folded in
            // once, which is also where the kernel folds it.
            let scale = e8m0(self.scales[row_scales + g]);
            let mut part = 0.0f64;
            for j in 0..32 {
                let at = g * 32 + j;
                let byte = self.codes[row_codes + at / 2];
                // Two codes to a byte, LOW nibble first — the order the
                // checkpoint packs and `dequant_mxfp4` unpacks.
                let nibble = if at % 2 == 0 { byte & 0xF } else { byte >> 4 };
                part += E2M1[nibble as usize] * f64::from(wide(x[at]));
            }
            acc += part * scale;
        }
        acc
    }
}

// ── one case ─────────────────────────────────────────────────────────────

/// What the point is handed, at one shape.
#[derive(Clone, Copy)]
struct Case {
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

impl Case {
    fn routes(self) -> usize {
        self.tokens * self.top_k
    }

    fn act_rows(self) -> usize {
        if self.per_route { self.routes() } else { self.tokens }
    }
}

/// What one fire of the point wrote, and what the spec says it should have.
struct Ran {
    got: Vec<u16>,
    want: Vec<u16>,
    /// The reference in fp32, for the ulp report.
    exact: Vec<f32>,
}

/// Fire the point once and read both sides.
///
/// `bias` and `routes` are handed in rather than generated so a mutation can
/// change one of them and nothing else — a mutated run and its baseline share
/// the bank, the activation and the shape to the byte.
fn run(ctx: &Ctx<'_>, c: Case, bank: &Bank, act: &[u16], bias: &[u16], routes: &[i32]) -> Ran {
    let routes_n = c.routes();
    let out_n = routes_n * c.n;

    let d_codes = Slab::of(&bank.codes);
    let d_scales = Slab::of(&bank.scales);
    let d_act = Slab::of(&bytes_of_u16(act));
    let d_bias = Slab::of(&bytes_of_u16(bias));
    let d_routes = Slab::of(&bytes_of_i32(routes));
    // A poison no legitimate result can be: every reference value below is
    // finite and of order one, so a NaN survivor is a slot the kernel never
    // wrote. Comparing bit patterns then says "it wrote" as well as "it
    // agrees".
    let d_out = Slab::of(&bytes_of_u16(&vec![0x7FC0u16; out_n]));

    Moe::matmul_select_bias::<bf16, Mxfp4>(
        ctx,
        In {
            ptr: d_act.ptr.cast(),
            rows: c.act_rows() as i32,
            width: c.k as i32,
        },
        Const::new(Planes {
            codes: d_codes.ptr.cast_const().cast(),
            scales: d_scales.ptr.cast_const().cast(),
        }),
        Const::new(d_bias.ptr.cast_const().cast()),
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
    .expect("the claimed `moe.matmul_select_bias` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the routed GEMM did not complete"
    );

    let mut want = Vec::with_capacity(out_n);
    let mut exact = Vec::with_capacity(out_n);
    for r in 0..routes_n {
        let e = routes[r] as usize;
        let at = if c.per_route { r } else { r / c.top_k };
        let x = &act[at * c.k..(at + 1) * c.k];
        for row in 0..c.n {
            let v = bank.dot(e, row, x) + f64::from(wide(bias[e * c.n + row]));
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

/// One bf16 ulp at `x`: the gap between neighbouring bf16 values there.
fn ulp(x: f32) -> f32 {
    let e = x.abs().to_bits() & 0x7F80_0000;
    // bf16 keeps 8 significand bits (7 stored), so one ulp is 2^-7 of the
    // binade. A subnormal-or-zero binade answers with the smallest normal's,
    // which is far below anything this test produces.
    f32::from_bits(e.max(1 << 23)) / 128.0
}

// ── the shapes ───────────────────────────────────────────────────────────

/// gpt-oss's two routed statements, at gpt-oss's own contraction.
///
/// `k = 2880` is the real hidden size and the real per-expert intermediate —
/// both legs contract over 2880 — so the 90-group walk, the tail-free
/// `k % 32`, and the fp32 accumulation depth are the shipped ones. `n` is cut
/// down from 5760 and 2880: the reference is fp64 and one code at a time, so
/// the product `routes * n * k` is what a test can afford, and `n` is the axis
/// the kernel parallelises over rather than the one it accumulates along.
const GPTOSS: &[Case] = &[
    // The gate/up leg: a PER-TOKEN activation, `top_k` routes reading each
    // row. `n` is a whole number of 16-row slabs (4 rows a warp, 4 warps a
    // block), which is the shipped 5760's case.
    Case {
        what: "gate/up (per-token activation, whole slabs)",
        per_route: false,
        experts: 6,
        tokens: 3,
        top_k: 4,
        n: 192,
        k: 2880,
    },
    // The down leg: an ALREADY ROUTED activation, one row per route.
    Case {
        what: "down (per-route activation)",
        per_route: true,
        experts: 6,
        tokens: 3,
        top_k: 4,
        n: 128,
        k: 2880,
    },
    // A TAIL: `n` is not a multiple of the 16-row slab, so the last warp
    // overhangs and clamps its rows onto the last real one. A body that let
    // the overhang store would corrupt row `n - 1`; one that let it LOAD
    // would read past the bank.
    Case {
        what: "an n that is not a whole slab",
        per_route: false,
        experts: 5,
        tokens: 2,
        top_k: 3,
        n: 100,
        k: 128,
    },
];

fn sample(c: Case, seed: u64) -> (Bank, Vec<u16>, Vec<u16>, Vec<i32>) {
    let mut rng = Rng(seed);
    let bank = Bank::random(&mut rng, c.experts, c.n, c.k);
    let act: Vec<u16> = (0..c.act_rows() * c.k).map(|_| narrow(rng.unit())).collect();
    // gpt-oss's own gate/up bias reaches |3.7| against a dot of order one, so
    // a bias this size is the shipped ratio and not a stacked deck.
    let bias: Vec<u16> = (0..c.experts * c.n)
        .map(|_| narrow(rng.unit() * 3.0))
        .collect();
    let routes: Vec<i32> = (0..c.routes())
        .map(|_| rng.below(c.experts as u32) as i32)
        .collect();
    (bank, act, bias, routes)
}

#[test]
fn the_routed_gemm_is_the_mxfp4_spec() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.matmul_select_bias") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    for (i, c) in GPTOSS.iter().enumerate() {
        let (bank, act, bias, routes) = sample(*c, 0x9e37_79b9_7f4a_7c15 ^ (i as u64 + 1));
        let r = run(&ctx, *c, &bank, &act, &bias, &routes);
        let n = r.want.len();

        let identical = (0..n).filter(|i| r.got[*i] == r.want[*i]).count();
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..n {
            let miss = (wide(r.got[i]) - r.exact[i]).abs() / ulp(r.exact[i]);
            if miss > worst {
                worst = miss;
                worst_at = i;
            }
        }
        eprintln!(
            "moe.matmul_select_bias [{}]: {identical}/{n} bit-identical to the fp64 \
             reference, worst miss {worst:.3} bf16 ulp at {worst_at} \
             (got {:+.6}, want {:+.6})",
            c.what,
            wide(r.got[worst_at]),
            r.exact[worst_at],
        );
        assert!(
            r.got.iter().all(|b| wide(*b).is_finite()),
            "[{}]: the kernel left a slot unwritten (the NaN poison survived)",
            c.what
        );
        // Two bars, and the second is what the first cannot say. A run where
        // every element missed by half an ulp would be a systematic error the
        // narrowing happens to hide; a run where one element missed by ten
        // would be a real fault the mean would hide.
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
fn the_bias_and_the_routes_are_read() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("moe.matmul_select_bias") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    let c = GPTOSS[0];
    let (bank, act, bias, routes) = sample(c, 0x2545_f491_4f6c_dd1d);
    let base = run(&ctx, c, &bank, &act, &bias, &routes);
    let n = base.want.len();
    let moved = |other: &Ran| (0..n).filter(|i| other.got[*i] != base.got[*i]).count();

    // ── the bias dropped ──
    //
    // THE FAULT THE SHIPPED LEG HAS. `mxfp4_moe_gate_up_decode_bf16` binds
    // `ctx.absent()` — a null pointer — to both of its bias pointer arrays, so
    // the legacy gpt-oss path never adds the gate/up bias the checkpoint
    // carries. This measures what that costs.
    let zeroed = vec![0u16; bias.len()];
    let no_bias = run(&ctx, c, &bank, &act, &zeroed, &routes);
    let bias_moved = moved(&no_bias);

    // ── the routes permuted ──
    //
    // Every row still names a legal expert, so nothing faults and nothing
    // reads out of bounds; what changes is only WHICH expert each route picks.
    //
    // AND THE ACCOUNTING IS PER ROUTE, which is the stronger claim. A
    // rotation leaves route `r` reading `routes[r + 1]`, and where those two
    // happened to name the same expert route `r`'s whole block must be
    // BIT-IDENTICAL while every other block must move. A body that indexed
    // the bank by the route's own ordinal, or that read the route run at an
    // offset, fails one half or the other.
    let mut rotated = routes.clone();
    rotated.rotate_left(1);
    let permuted = run(&ctx, c, &bank, &act, &bias, &rotated);
    let mut route_moved = 0usize;
    let mut route_held = 0usize;
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
            route_held += c.n;
        } else {
            assert!(
                differ * 100 >= c.n * 99,
                "route {r} moved from expert {} to {} and only {differ}/{} of its \
                 row changed",
                routes[r],
                rotated[r],
                c.n
            );
            route_moved += differ;
        }
    }
    assert!(
        route_moved > 0,
        "the rotation happened to be the identity: nothing was measured"
    );

    // ── the block scales flattened ──
    let flat = Bank {
        codes: bank.codes.clone(),
        scales: vec![127u8; bank.scales.len()],
        n: bank.n,
        k: bank.k,
    };
    let unscaled = run(&ctx, c, &flat, &act, &bias, &routes);
    let scale_moved = moved(&unscaled);

    eprintln!(
        "moe.matmul_select_bias mutations over {n} elements: \
         bias dropped {bias_moved}, routes rotated {route_moved} moved with \
         {route_held} held by routes the rotation did not relabel, \
         scales flattened {scale_moved}"
    );
    assert!(
        bias_moved * 100 >= n * 99,
        "dropping the bias moved only {bias_moved}/{n} elements"
    );
    assert!(
        scale_moved * 100 >= n * 99,
        "flattening the block scales moved only {scale_moved}/{n} elements"
    );
    // AND EACH MUTATION STILL AGREES WITH ITS OWN REFERENCE, which is what
    // separates "the kernel reads this operand" from "the kernel broke".
    for (what, r) in [
        ("bias dropped", &no_bias),
        ("routes rotated", &permuted),
        ("scales flattened", &unscaled),
    ] {
        let bad = (0..n)
            .filter(|i| (wide(r.got[*i]) - r.exact[*i]).abs() > ulp(r.exact[*i]))
            .count();
        assert_eq!(bad, 0, "{what}: {bad}/{n} elements past one bf16 ulp");
    }
}

