//! FA4's forward runs on a device and produces attention's numbers.
//!
//! Every other check the crate makes about `attn/fa4.cuh` is a claim about
//! TEXT. `source.rs` says the file is carried, `every_carried_file_is_reachable`
//! says a root reaches it, `every_instantiation_compiles` says NVRTC lowers the
//! four template-ids, and `attn::fa4`'s unit tests say the host arithmetic —
//! the grid, the shared-memory figure, the instantiation table — is what it
//! claims. None of that would have caught the bug that actually happened.
//!
//! While this kernel was being written it spent a while producing, for every
//! shape, a relative error around 1.0: `ldmatrix` hands its four registers out
//! by lane octet and the MMA's A and B fragments disagree about which octet
//! holds which quadrant, so keys were being read through the A map. The
//! failure is silent in the worst way — every output is still a real dot
//! product, just of the wrong pair of vectors. It compiles, it launches, it
//! writes plausible finite numbers into every element of `o`, and the ONLY
//! thing that distinguishes it from a correct kernel is comparing against an
//! independent softmax. That comparison is this file.
//!
//! # Why a reference in Rust and not a stored expectation
//!
//! A golden buffer would have to be produced by something, and the only
//! candidates are this kernel (which would make the test a tautology) or a
//! Python session on a machine with a GPU (which would make it unreproducible
//! and would rot at the first tile retune). [`reference`] is thirty lines of
//! f32 arithmetic that reads like the definition of attention, is obviously
//! not sharing an implementation with the thing under test, and re-derives
//! itself for whatever shapes the table below happens to hold.
//!
//! # Tolerance
//!
//! The kernel's inputs are bf16 and its two GEMMs accumulate in f32, so it
//! cannot agree with an f32 reference exactly and does not try. Measured max
//! relative error across fifteen shapes is 2.2e-3 to 3.0e-3 — the same figure
//! FA4's own CuteDSL kernel posts against the same reference, because it is
//! bf16's error and not the kernel's. [`TOLERANCE`] is 1e-2, which leaves
//! headroom over that without being loose enough to matter: the layout bug
//! this file exists to catch missed by a factor of a hundred.
//!
//! Without a device the target still COMPILES and every test skips with a
//! stated reason, on `fire.rs`'s argument — a test that silently passes on a
//! laptop is how a broken launch path ships.

#![cfg(feature = "_cuda")]

use cudarc::driver::sys::{
    CUresult, cuCtxSynchronize, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
};
use kernels_cuda::attn::fa4::{self, Fa4, split_scratch_elems};
use kernels_cuda::jit::{Ctx, abi::bf16, cache};

/// The largest tolerated `|o - ref| / max|ref|`. See the module header.
const TOLERANCE: f32 = 1e-2;

/// The largest tolerated absolute error in the log-sum-exp.
///
/// Looser than [`TOLERANCE`] in absolute terms and much tighter in relative
/// ones: `lse` is a logarithm, so its scale is units rather than the tensor's,
/// and a wrong-by-a-factor softmax denominator would move it by `ln` of that
/// factor. Measured error is around 1e-6.
const LSE_TOLERANCE: f32 = 5e-2;

/// `sm_XY` for the current device, or a stated reason there is none.
///
/// `fire.rs` carries the identical function and they are deliberately not
/// shared: an integration test is its own binary, and a `mod common` would buy
/// nine lines at the price of a file that neither test's reader can see.
fn arch_or_skip(what: &str) -> Option<&'static str> {
    match quietly(cache::arch).flatten() {
        Some(arch) => match cache::bind_context() {
            Ok(()) => Some(arch),
            Err(why) => {
                eprintln!("SKIP {what}: no usable context ({why})");
                None
            }
        },
        None => {
            eprintln!("SKIP {what}: no CUDA device is current");
            None
        }
    }
}

/// Run `f`, turning a panic into `None` and printing nothing.
///
/// The `Option` above reads as "ask whether there is a device", and it answers
/// only half the question. `cudarc` is built `fallback-dynamic-loading`, so
/// every CUDA symbol is `dlopen`'d on first use and a MISSING library is a
/// PANIC -- `panic_no_lib_found` -- rather than an `Err`. So the skip covered
/// "driver present, no device" and the case it was actually written for, an
/// ordinary runner with no CUDA installed at all, unwound straight past it and
/// FAILED. Measured with an `LD_PRELOAD` shim that refuses to `dlopen` the
/// CUDA libraries: this file went from 3 skips to 2 failures.
///
/// Only the FIRST call needs wrapping. Once one answers, the library is
/// loaded and everything after it is an ordinary call that can return `Err`.
fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

/// A device allocation that frees itself.
///
/// Sized in BYTES rather than in elements because one of these holds bf16 and
/// another holds f32, and a length that does not say which it counts is how a
/// download reads half a buffer.
struct Dev {
    ptr: u64,
    bytes: usize,
}

impl Dev {
    /// `bytes` of device memory, uninitialised.
    fn new(bytes: usize) -> Self {
        let mut ptr = 0u64;
        // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
        let code = unsafe { cuMemAlloc_v2(&raw mut ptr, bytes) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
        Self { ptr, bytes }
    }

    /// `data` uploaded into a fresh allocation of exactly its size.
    fn upload<T: Copy>(data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let dev = Self::new(bytes);
        // SAFETY: the allocation is `bytes` long and `data` is exactly that.
        let code = unsafe { cuMemcpyHtoD_v2(dev.ptr, data.as_ptr().cast(), bytes) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "upload");
        dev
    }

    /// The allocation read back as `len` elements of `T`, after a synchronise.
    fn download<T: Copy + Default>(&self, len: usize) -> Vec<T> {
        // SAFETY: no outstanding work beyond the launch under test.
        let code = unsafe { cuCtxSynchronize() };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "synchronise");

        let mut host = vec![T::default(); len];
        let bytes = std::mem::size_of_val(host.as_slice());
        assert!(bytes <= self.bytes, "reading {bytes} out of a {}-byte buffer", self.bytes);
        // SAFETY: the read is `bytes` long and the assert above bounds it by
        // the allocation.
        let code = unsafe { cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), self.ptr, bytes) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "download");
        host
    }
}

impl Drop for Dev {
    fn drop(&mut self) {
        // SAFETY: the allocation is still live and is freed exactly once.
        unsafe { cuMemFree_v2(self.ptr) };
    }
}

/// f32 to bf16, round-to-nearest-even — the rounding a GPU cast does.
///
/// Truncation would be one line shorter and would bias every input low, which
/// on a sum of `head_dim` products is a systematic error the tolerance would
/// then have to absorb. The inputs are generated here, so this is also the
/// only place the reference and the kernel can disagree about what the input
/// WAS: both read the same 16 bits after this.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// bf16 to f32 — the exact widening, since bf16 is an f32's top half.
fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Deterministic inputs, so a failure reproduces exactly.
///
/// A 64-bit LCG rather than `rand`: the crate does not depend on it, and a
/// test whose inputs change per run reports a different max error every time
/// and cannot be bisected.
struct Lcg(u64);

impl Lcg {
    /// The next value, roughly uniform on `[-1, 1)` and rounded to bf16.
    ///
    /// Rounded HERE rather than at upload, because the reference must see the
    /// value the kernel sees and not the f32 it came from.
    fn next(&mut self) -> u16 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32);
        to_bf16(unit * 2.0 - 1.0)
    }

    /// `n` such values.
    fn fill(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// One shape to check.
#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    seqlen_q: usize,
    seqlen_k: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    causal: bool,
}

impl Shape {
    /// The shape, as a caller would say it, for an assertion message.
    fn tag(&self) -> String {
        format!(
            "b{} sq{} sk{} hq{} hk{} d{} {}",
            self.batch,
            self.seqlen_q,
            self.seqlen_k,
            self.heads_q,
            self.heads_kv,
            self.head_dim,
            if self.causal { "causal" } else { "full" }
        )
    }
}

/// Attention in f32, written as its definition.
///
/// Returns `(o, lse)` in the kernel's layouts: `o` is `bshd` like `q`, and
/// `lse` is a packed `[batch, heads_q, seqlen_q]`.
///
/// Two conventions here are the kernel's and are the whole reason the
/// comparison is worth anything — get either wrong and this file would be
/// asserting its own misunderstanding:
///
/// 1. **Causal alignment is bottom-right**, not top-left: the mask is
///    `j <= i + (seqlen_k - seqlen_q)`, so a one-row decode query attends to
///    the whole cache rather than to its first key. Top-left alignment would
///    make every `seqlen_q != seqlen_k` case disagree and every square one
///    agree, which is the failure that looks like a ragged-shape bug.
/// 2. **A fully masked row is defined, not NaN.** With bottom-right alignment
///    and `seqlen_q > seqlen_k` the leading queries attend to nothing, and a
///    naive `exp(s - max) / sum` over an all-`-inf` row is `0/0`. The kernel
///    writes exactly zero with `lse = -inf`, matching FA4 bit for bit, and
///    this does the same — a reference that returned NaN there would report
///    three failures the kernel does not have.
fn reference(shape: &Shape, q: &[u16], k: &[u16], v: &[u16]) -> (Vec<f32>, Vec<f32>) {
    let (sq, sk, d) = (shape.seqlen_q, shape.seqlen_k, shape.head_dim);
    let (hq, hk) = (shape.heads_q, shape.heads_kv);
    let group = hq / hk;
    let scale = (d as f32).powf(-0.5);
    let causal_offset = sk as i64 - sq as i64;

    let mut o = vec![0f32; shape.batch * sq * hq * d];
    let mut lse = vec![0f32; shape.batch * hq * sq];

    for b in 0..shape.batch {
        for h in 0..hq {
            let hkv = h / group;
            for i in 0..sq {
                let limit = if shape.causal {
                    ((i as i64 + causal_offset + 1).max(0) as usize).min(sk)
                } else {
                    sk
                };

                let q_row = ((b * sq + i) * hq + h) * d;
                let mut scores = vec![0f32; limit];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k_row = ((b * sk + j) * hk + hkv) * d;
                    let dot: f32 =
                        (0..d).map(|e| from_bf16(q[q_row + e]) * from_bf16(k[k_row + e])).sum();
                    *score = dot * scale;
                }

                let o_row = ((b * sq + i) * hq + h) * d;
                let lse_at = (b * hq + h) * sq + i;
                if limit == 0 {
                    lse[lse_at] = f32::NEG_INFINITY;
                    continue;
                }

                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                lse[lse_at] = max + sum.ln();
                for (j, score) in scores.iter().enumerate() {
                    let p = (score - max).exp() / sum;
                    let v_row = ((b * sk + j) * hk + hkv) * d;
                    for e in 0..d {
                        o[o_row + e] += p * from_bf16(v[v_row + e]);
                    }
                }
            }
        }
    }

    (o, lse)
}

/// Fire one shape and return `(output error, lse error)`, both normalised.
///
/// The output error is `max|o - ref| / max|ref|` rather than a per-element
/// relative error, because attention's outputs pass through zero and a
/// per-element ratio there reports a large number for a small absolute miss.
fn fire(ctx: &Ctx, shape: &Shape) -> (f32, f32) {
    let (sq, sk, d) = (shape.seqlen_q, shape.seqlen_k, shape.head_dim);
    let (hq, hk) = (shape.heads_q, shape.heads_kv);

    let mut rng = Lcg(0x5eed_1234_9abc_def0);
    let q = rng.fill(shape.batch * sq * hq * d);
    let k = rng.fill(shape.batch * sk * hk * d);
    let v = rng.fill(shape.batch * sk * hk * d);

    let d_q = Dev::upload(&q);
    let d_k = Dev::upload(&k);
    let d_v = Dev::upload(&v);
    let d_o = Dev::new(q.len() * 2);
    let d_lse = Dev::new(shape.batch * hq * sq * 4);

    // Split-KV scratch, sized by the launcher's own plan for this shape. Zero
    // means it will not split, and then the pair of nulls is what says so.
    let (o_part, lse_part) = split_scratch_elems(
        ctx,
        sq as u32,
        sk as u32,
        shape.batch as u32,
        hq as u32,
        hk as u32,
        d as u32,
    );
    let d_op = (o_part > 0).then(|| Dev::new(o_part * 4));
    let d_lp = (lse_part > 0).then(|| Dev::new(lse_part * 4));

    let job = Fa4 {
        q: d_q.ptr as *const bf16,
        k: d_k.ptr as *const bf16,
        v: d_v.ptr as *const bf16,
        o: d_o.ptr as *mut bf16,
        lse: d_lse.ptr as *mut f32,
        o_partial: d_op.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr as *mut f32),
        lse_partial: d_lp.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr as *mut f32),

        // `bshd`, contiguous: the sequence stride is a whole row of heads and
        // the head stride is one head's `head_dim`.
        q_stride_b: (sq * hq * d) as i32,
        q_stride_s: (hq * d) as i32,
        q_stride_h: d as i32,
        k_stride_b: (sk * hk * d) as i32,
        k_stride_s: (hk * d) as i32,
        k_stride_h: d as i32,
        v_stride_b: (sk * hk * d) as i32,
        v_stride_s: (hk * d) as i32,
        v_stride_h: d as i32,
        o_stride_b: (sq * hq * d) as i32,
        o_stride_s: (hq * d) as i32,
        o_stride_h: d as i32,
        lse_stride_b: (hq * sq) as i32,
        lse_stride_h: sq as i32,

        batch: shape.batch as u32,
        heads_q: hq as u32,
        heads_kv: hk as u32,
        head_dim: d as u32,
        seqlen_q: sq as u32,
        seqlen_k: sk as u32,
        causal: shape.causal,
        // The kernel's exponentials are base 2, so the host folds `log2(e)`
        // into the scale and the kernel never scales again.
        scale_log2: (d as f32).powf(-0.5) * std::f32::consts::LOG2_E,
    };

    // SAFETY: every pointer bound above is a live device allocation of the
    // extent the strides describe, and all five outlive the synchronise below.
    unsafe { fa4::forward(ctx, job) }.unwrap_or_else(|why| panic!("{}: {why:?}", shape.tag()));

    let got: Vec<u16> = d_o.download(q.len());
    let got_lse: Vec<f32> = d_lse.download(shape.batch * hq * sq);
    let (want, want_lse) = reference(shape, &q, &k, &v);

    let denom = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    let err = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (&g, &w)| m.max((from_bf16(g) - w).abs()))
        / denom;

    // A row with no keys is compared as an identity, not as a difference:
    // `-inf` minus `-inf` is NaN and would poison the max.
    let lse_err = got_lse.iter().zip(&want_lse).fold(0f32, |m, (&g, &w)| {
        if w == f32::NEG_INFINITY {
            assert_eq!(g, f32::NEG_INFINITY, "{}: a masked row has a finite lse", shape.tag());
            m
        } else {
            m.max((g - w).abs())
        }
    });

    (err, lse_err)
}

/// The kernel computes attention, over the shapes that reach its edges.
///
/// The table is chosen so that each row can fail alone:
///
/// - both head dims, since they are different instantiations with different
///   warp counts (4 at 64, 8 at 128) and different shared-memory footprints;
/// - both mask modes, since `causal` is a template parameter and the two
///   compile to genuinely different loops;
/// - GQA, since `group_size` is the only thing that makes a query head read a
///   key head other than its own;
/// - lengths that are not multiples of `TILE_M`, since the row-tile boundary
///   is where an out-of-range query row would be read;
/// - `seqlen_q != seqlen_k` in both directions, which is where bottom-right
///   causal alignment differs from top-left, and where — with `sq > sk` — the
///   fully masked leading rows appear;
/// - a single query row against a long cache, which is the decode shape and
///   the one where the M tile is almost entirely padding;
/// - batched decode, which is the only case `attn::fa4::wants_packing` answers
///   yes for, and so the only way the four packed instantiations run at all.
///   Packing changes what a block means — the M row becomes a
///   `(position, head)` pair and the grid walks KV heads — so it is a second
///   indexing scheme over the same arithmetic, and the reference below does
///   not know which one ran. That is the point: both must produce attention.
///
/// Three of the plan's axes are therefore covered by the table rather than
/// asserted, and each has been checked by breaking it and watching which rows
/// fail — which is the only way to know a shape reaches the code it is here
/// for, since a plan that quietly stopped choosing a variant would leave every
/// row green:
///
/// - collapsing every packed row to head 0 fails exactly the batched-decode
///   rows and no others;
/// - pinning the combine weight at 1.0 fails exactly the eight rows the plan
///   splits, and leaves `lse` correct, since `lse` is not what it reweights;
/// - scaling the 64-row tile's output by 1.05 fails exactly the twelve rows
///   with a small enough `seqlen_q * pack`, and again leaves `lse` correct.
#[test]
fn the_kernel_computes_attention() {
    let Some(_) = arch_or_skip("the_kernel_computes_attention") else { return };

    const SHAPES: &[Shape] = &[
        Shape { batch: 1, seqlen_q: 128, seqlen_k: 128, heads_q: 1, heads_kv: 1, head_dim: 64, causal: false },
        Shape { batch: 1, seqlen_q: 128, seqlen_k: 128, heads_q: 1, heads_kv: 1, head_dim: 64, causal: true },
        Shape { batch: 1, seqlen_q: 128, seqlen_k: 128, heads_q: 1, heads_kv: 1, head_dim: 128, causal: false },
        Shape { batch: 1, seqlen_q: 128, seqlen_k: 128, heads_q: 1, heads_kv: 1, head_dim: 128, causal: true },
        Shape { batch: 2, seqlen_q: 320, seqlen_k: 320, heads_q: 4, heads_kv: 2, head_dim: 64, causal: true },
        Shape { batch: 2, seqlen_q: 320, seqlen_k: 320, heads_q: 4, heads_kv: 2, head_dim: 128, causal: false },
        Shape { batch: 1, seqlen_q: 100, seqlen_k: 377, heads_q: 2, heads_kv: 1, head_dim: 128, causal: true },
        Shape { batch: 3, seqlen_q: 333, seqlen_k: 177, heads_q: 6, heads_kv: 2, head_dim: 64, causal: true },
        Shape { batch: 1, seqlen_q: 1, seqlen_k: 512, heads_q: 4, heads_kv: 1, head_dim: 128, causal: true },
        Shape { batch: 1, seqlen_q: 129, seqlen_k: 1, heads_q: 1, heads_kv: 1, head_dim: 64, causal: true },
        // Batched decode, which is where `wants_packing` says yes: enough
        // blocks that folding the group saves whole waves. These are the only
        // rows that reach the packed instantiations at all, so without them
        // half the kernel table would be compiled and never run.
        Shape { batch: 16, seqlen_q: 1, seqlen_k: 600, heads_q: 32, heads_kv: 8, head_dim: 128, causal: true },
        Shape { batch: 16, seqlen_q: 1, seqlen_k: 600, heads_q: 32, heads_kv: 8, head_dim: 64, causal: true },
        Shape { batch: 24, seqlen_q: 3, seqlen_k: 257, heads_q: 16, heads_kv: 2, head_dim: 64, causal: true },
        Shape { batch: 24, seqlen_q: 3, seqlen_k: 257, heads_q: 16, heads_kv: 2, head_dim: 128, causal: false },
        // Group sizes that do NOT divide `TILE_M`. A packed tile then straddles
        // a query position: its first and last rows belong to different
        // positions AND different heads, and the count of positions it covers
        // is not even an integer. Nothing in `PackedRow` may assume otherwise —
        // it maps a row index through `pos = pm / group` and
        // `head = slot * group + pm % group`, which is a bijection over the
        // packed row space for any group, aligned or not. These shapes are the
        // evidence, on both paths: the first two decline packing (a tie on
        // waves) and exercise the unpacked `kv_head = slot / group` division,
        // the last two are batched decode wide enough to take it.
        Shape { batch: 1, seqlen_q: 96, seqlen_k: 512, heads_q: 12, heads_kv: 4, head_dim: 64, causal: true },
        Shape { batch: 1, seqlen_q: 40, seqlen_k: 300, heads_q: 15, heads_kv: 3, head_dim: 128, causal: true },
        Shape { batch: 32, seqlen_q: 1, seqlen_k: 600, heads_q: 24, heads_kv: 8, head_dim: 128, causal: true },
        Shape { batch: 32, seqlen_q: 1, seqlen_k: 600, heads_q: 20, heads_kv: 4, head_dim: 64, causal: false },
        // Shapes the plan splits the key range for, which is the only way the
        // combine pass runs. A split block holds an attention over its own
        // range, normalised by its own denominator; nothing but the combine
        // knows the real one, so an error there is an error in every one of
        // these and in none of the rest.
        Shape { batch: 1, seqlen_q: 1, seqlen_k: 4096, heads_q: 32, heads_kv: 8, head_dim: 128, causal: true },
        Shape { batch: 1, seqlen_q: 1, seqlen_k: 4096, heads_q: 32, heads_kv: 8, head_dim: 64, causal: true },
        Shape { batch: 2, seqlen_q: 2, seqlen_k: 2000, heads_q: 16, heads_kv: 4, head_dim: 128, causal: false },
        Shape { batch: 1, seqlen_q: 5, seqlen_k: 999, heads_q: 8, heads_kv: 2, head_dim: 64, causal: true },
    ];

    // SAFETY: the null stream is always live.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };

    let mut failures = Vec::new();
    for shape in SHAPES {
        let (err, lse_err) = fire(&ctx, shape);
        if err > TOLERANCE || lse_err > LSE_TOLERANCE || err.is_nan() || lse_err.is_nan() {
            failures.push(format!("{}: err {err:.2e}, lse err {lse_err:.2e}", shape.tag()));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} shapes disagree with the reference:\n  {}",
        failures.len(),
        SHAPES.len(),
        failures.join("\n  ")
    );
}

/// A null `lse` skips the log-sum-exp and does not change the output.
///
/// `lse` is the one nullable operand, and the kernel honours it branch by
/// branch rather than writing to a scratch buffer. A caller that does not want
/// the statistic — every non-split-KV forward — passes null, so the path that
/// actually ships is the one with the null in it, and it is worth a check that
/// the branch does not also skip something else.
#[test]
fn a_null_lse_is_honoured() {
    let Some(_) = arch_or_skip("a_null_lse_is_honoured") else { return };

    let shape = Shape {
        batch: 1,
        seqlen_q: 256,
        seqlen_k: 256,
        heads_q: 2,
        heads_kv: 1,
        head_dim: 64,
        causal: true,
    };
    let (sq, sk, d) = (shape.seqlen_q, shape.seqlen_k, shape.head_dim);
    let (hq, hk) = (shape.heads_q, shape.heads_kv);

    let mut rng = Lcg(0x5eed_1234_9abc_def0);
    let q = rng.fill(shape.batch * sq * hq * d);
    let k = rng.fill(shape.batch * sk * hk * d);
    let v = rng.fill(shape.batch * sk * hk * d);

    let d_q = Dev::upload(&q);
    let d_k = Dev::upload(&k);
    let d_v = Dev::upload(&v);
    let d_o = Dev::new(q.len() * 2);

    let job = Fa4 {
        q: d_q.ptr as *const bf16,
        k: d_k.ptr as *const bf16,
        v: d_v.ptr as *const bf16,
        o: d_o.ptr as *mut bf16,
        lse: std::ptr::null_mut(),
        o_partial: std::ptr::null_mut(),
        lse_partial: std::ptr::null_mut(),

        q_stride_b: (sq * hq * d) as i32,
        q_stride_s: (hq * d) as i32,
        q_stride_h: d as i32,
        k_stride_b: (sk * hk * d) as i32,
        k_stride_s: (hk * d) as i32,
        k_stride_h: d as i32,
        v_stride_b: (sk * hk * d) as i32,
        v_stride_s: (hk * d) as i32,
        v_stride_h: d as i32,
        o_stride_b: (sq * hq * d) as i32,
        o_stride_s: (hq * d) as i32,
        o_stride_h: d as i32,
        // Ignored while `lse` is null, and deliberately left as values that
        // would fault if they were not.
        lse_stride_b: 0,
        lse_stride_h: 0,

        batch: shape.batch as u32,
        heads_q: hq as u32,
        heads_kv: hk as u32,
        head_dim: d as u32,
        seqlen_q: sq as u32,
        seqlen_k: sk as u32,
        causal: shape.causal,
        scale_log2: (d as f32).powf(-0.5) * std::f32::consts::LOG2_E,
    };

    // SAFETY: q, k, v and o are live allocations of the extents the strides
    // describe; `lse` is null, which the kernel tests before dereferencing.
    unsafe { fa4::forward(&ctx_null(), job) }.expect("fires with a null lse");

    let got: Vec<u16> = d_o.download(q.len());
    let (want, _) = reference(&shape, &q, &k, &v);
    let denom = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    let err = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (&g, &w)| m.max((from_bf16(g) - w).abs()))
        / denom;

    assert!(err <= TOLERANCE, "a null lse changed the output: err {err:.2e}");
}

/// A `Ctx` on the null stream.
///
/// A `fn` rather than a `let` at each site because `Ctx::on` is `unsafe` and
/// the reason it is sound — the null stream is always live — is the same one
/// every time and is better written once.
fn ctx_null() -> Ctx<'static> {
    // SAFETY: the null stream is always live.
    unsafe { Ctx::on(std::ptr::null_mut()) }
}

/// An unsupported head dimension is refused, not launched.
///
/// The instantiation table has four points and nothing behind them: a head dim
/// of 96 has no kernel, and the honest answer is a refusal rather than a launch
/// with the 64 or 128 geometry, which would read past the end of every row.
/// No device needed — the refusal is decided before anything touches CUDA.
#[test]
fn an_unsupported_head_dim_is_refused() {
    let mut host = [0u16; 8];
    let job = Fa4 {
        q: host.as_mut_ptr() as *const bf16,
        k: host.as_mut_ptr() as *const bf16,
        v: host.as_mut_ptr() as *const bf16,
        o: host.as_mut_ptr() as *mut bf16,
        lse: std::ptr::null_mut(),
        o_partial: std::ptr::null_mut(),
        lse_partial: std::ptr::null_mut(),
        q_stride_b: 0,
        q_stride_s: 0,
        q_stride_h: 0,
        k_stride_b: 0,
        k_stride_s: 0,
        k_stride_h: 0,
        v_stride_b: 0,
        v_stride_s: 0,
        v_stride_h: 0,
        o_stride_b: 0,
        o_stride_s: 0,
        o_stride_h: 0,
        lse_stride_b: 0,
        lse_stride_h: 0,
        batch: 1,
        heads_q: 1,
        heads_kv: 1,
        head_dim: 96,
        seqlen_q: 1,
        seqlen_k: 1,
        causal: false,
        scale_log2: 1.0,
    };

    // SAFETY: the call returns on the head-dim arm before reading a pointer.
    let refusal = unsafe { fa4::forward(&ctx_null(), job) };
    assert!(refusal.is_err(), "head dim 96 launched something");
}
