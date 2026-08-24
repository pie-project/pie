//! The two points that READ a log-sum-exp, against the base the floor now
//! states — `attention.sink` and `attention.merge_lse`.
//!
//! WHY THESE TWO ARE ONE FILE. An lse has a base, and until this commit the
//! floor did not say which: flashinfer's `_lse` readings publish
//! `m + log2(d)` (`flashinfer/attention/state.cuh:45`, with
//! `params.sm_scale *= math::log2e` in front of it), dsv4's compressed
//! reading published `logf(z) + row_max`, and a text papered over the
//! difference with an `attention.lse_ln` statement it had to know to write.
//! `attention.decode_lse` states base two now — the base every attention
//! kernel already has, because `exp2` is the instruction — and the two
//! consumers below are the whole of what that costs: a merge that folds in
//! base two, and a sink that rebases ONCE, where a checkpoint's natural-log
//! logit meets the normaliser.
//!
//! # What the sink test can see
//!
//! `sigmoid(lse·ln2 − sink)` and `sigmoid(lse − sink)` are the same shape of
//! curve, so a shape check would pass either. What separates them is the
//! NUMBER, and at gpt-oss's real sinks (`[2.51, 0.55, 1.71, …]`, BF16 `[64]`
//! in the shipped checkpoint) against real lse magnitudes the two differ by
//! tens of percent per element — `the_rebase_is_the_correction` measures
//! that separation rather than asserting it, so the margin is on record.
//!
//! The OTHER half of the seam cannot be tested here at all, and that is the
//! point of the redesign: the sink slot rides the point's element now, so
//! `ctx.sink::<bf16>(.., Const<Tensor<f32>>, ..)` is a type error and no
//! test can reach the reinterpretation that used to fire. What IS reachable
//! is the sibling launcher that still reads f32 — `norm::attn_sink_
//! correction`, the legacy dsv4 text's — and handing it the same bf16 bytes
//! is exactly the fire that refused gpt-oss at step 11. The last test does
//! that, and measures how far off the answer would have been had the bytes
//! gone through.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Attention;
use kernels::routine::{Const, In, InOut, Out};
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::jit::Ctx;

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time. `gdn_chunk_prefill.rs`'s lock, verbatim and for its
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

    fn read(&self, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
        out
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        self.read(elems * 2)
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    fn read_f32(&self, elems: usize) -> Vec<f32> {
        self.read(elems * 4)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

// ── gpt-oss's shape, and its real sinks ──────────────────────────────────

/// Not a round number of anything: the sink launch is one block per
/// `(token, head)` with the block striding `head_dim`, so a token count that
/// divides nothing catches a grid read off the wrong axis.
const ROWS: i32 = 7;

/// gpt-oss-20b's head width.
const HEAD_DIM: i32 = 64;

/// Six of gpt-oss-20b's own attention sinks, read out of
/// `model.layers.0.self_attn.sinks` (BF16 `[64]`) in the shipped checkpoint.
/// THE MAGNITUDES ARE THE TEST: a sink near zero halves the output whatever
/// base the lse is in, and one at 4.06 does almost nothing — the spread is
/// what makes a rebase visible.
const SINKS: [f32; 6] = [2.515_625, 0.558_593_75, 1.718_75, -2.453_125, 4.0625, 0.193_359_38];

const HEADS: i32 = SINKS.len() as i32;

/// One `[ROWS, HEADS * HEAD_DIM]` output and one `[ROWS, HEADS]` lse.
///
/// THE LSE RANGE IS THE REGIME THE CORRECTION LIVES IN, measured rather
/// than assumed. A sink competes with the whole key set, so the factor
/// `sigmoid(lse − sink)` is only interesting while `lse` is within a few
/// units of the sink: at a base-two lse of 12 every head's factor is
/// 0.997 or better and the sink is doing nothing, while at 3 the same
/// heads sit at 0.39, 0.82, 0.59. `[-3, 6]` is the span a decode actually
/// walks — one key at position 0 puts the lse at the single score, a few
/// hundred keys put it near the top of this range — and it is where a
/// dropped rebase is a different answer rather than a different ulp.
///
/// One row is `-inf`: the causally-masked-out row every one of these
/// kernels carries a guard for.
fn operands() -> (Vec<u16>, Vec<f32>) {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let n = (ROWS * HEADS * HEAD_DIM) as usize;
    let o: Vec<u16> = (0..n).map(|_| narrow(rng.next() * 3.0)).collect();
    let lse: Vec<f32> = (0..(ROWS * HEADS) as usize)
        .map(|i| {
            if i == 9 {
                f32::NEG_INFINITY
            } else {
                1.5 + rng.next() * 4.5
            }
        })
        .collect();
    (o, lse)
}

/// `o *= sigmoid(lse·ln2 − sink)`, on the host, in fp32.
///
/// `rebase` is the multiplier the kernel is supposed to apply to the lse:
/// `ln 2` for a base-two lse against a natural-log sink. Passing `1.0` is
/// the mutation — the arithmetic that shipped before the ln2 was found, and
/// what `attn_sink_correction` beside it still does.
fn host_sink(o: &[u16], lse: &[f32], sinks: &[f32], rebase: f32) -> Vec<f32> {
    let mut out = vec![0f32; o.len()];
    for t in 0..ROWS as usize {
        for h in 0..HEADS as usize {
            let l = lse[t * HEADS as usize + h];
            let r = if l.is_finite() {
                1.0 / (1.0 + (-(l * rebase - sinks[h])).exp())
            } else {
                1.0
            };
            for d in 0..HEAD_DIM as usize {
                let at = (t * HEADS as usize + h) * HEAD_DIM as usize + d;
                out[at] = wide(o[at]) * r;
            }
        }
    }
    out
}

/// The largest relative gap between a device row and a host row, over the
/// elements big enough for a relative gap to mean anything.
fn worst(device: &[u16], host: &[f32]) -> f32 {
    let mut worst = 0f32;
    for (d, h) in device.iter().zip(host) {
        let d = wide(*d);
        let scale = h.abs().max(1e-3);
        worst = worst.max((d - h).abs() / scale);
    }
    worst
}

/// How many elements of two host readings differ by more than a bf16 can
/// hide. `1/256` is one step of an 8-bit mantissa.
fn separated(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (*x - *y).abs() > x.abs().max(1e-3) / 256.0)
        .count()
}

// ── the tests ────────────────────────────────────────────────────────────

/// The point's claim IS the virtual-key softmax, at the stated base.
#[test]
fn the_sink_is_a_virtual_key_at_base_two() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.sink") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let (o, lse) = operands();
    let sinks: Vec<u16> = SINKS.iter().map(|s| narrow(*s)).collect();

    let d_o = Slab::of(&bytes_of_u16(&o));
    let d_lse = Slab::of(&bytes_of_f32(&lse));
    let d_sink = Slab::of(&bytes_of_u16(&sinks));

    Attention::sink::<bf16>(
        &ctx,
        InOut {
            ptr: d_o.ptr.cast(),
            rows: ROWS,
            width: HEADS * HEAD_DIM,
        },
        In {
            ptr: d_lse.ptr.cast(),
            rows: ROWS,
            width: HEADS,
        },
        Const {
            v: d_sink.ptr.cast_const().cast(),
        },
        HEAD_DIM as u32,
    )
    .expect("`attention.sink` at bf16");

    let got = d_o.read_u16(o.len());
    let want = host_sink(&o, &lse, &SINKS, core::f32::consts::LN_2);
    let worst = worst(&got, &want);
    assert!(
        worst < 1.0 / 128.0,
        "the sink correction is off the host reading by {worst} relative"
    );

    // The `-inf` row is the one the kernel must LEAVE ALONE, and it is a
    // separate assertion because a factor of 1 is invisible to the sweep
    // above (it agrees with a reference that also passes it through).
    let masked = (9 * HEAD_DIM) as usize;
    assert_eq!(
        got[masked..masked + HEAD_DIM as usize],
        o[masked..masked + HEAD_DIM as usize],
        "a causally-masked-out row was rescaled by something other than 1"
    );
}

/// THE MUTATION: drop the rebase and the answer moves, at gpt-oss's own
/// sinks and at lse magnitudes a real decode leaves.
///
/// This is the bug `attn_sink.cuh`'s header records having been found once
/// already — "matched HF's top-1 on most prompts by accident and then
/// drifted" — and the number below is why it drifted.
#[test]
fn the_rebase_is_the_correction() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.sink") {
        return;
    }
    let (o, lse) = operands();
    let with = host_sink(&o, &lse, &SINKS, core::f32::consts::LN_2);
    let without = host_sink(&o, &lse, &SINKS, 1.0);

    let moved = separated(&with, &without);
    // Every finite row is a candidate: `ROWS * HEADS` rows less the one
    // `-inf` row, times the head width. NOT ALL OF THEM MOVE, and the ones
    // that do not are on record: a head whose sink is far below the lse
    // (head 3 at −2.45) is already saturated at 0.999 in both bases, so its
    // 64 elements agree to within a bf16 step. 2560 of 2624 at this draw.
    let finite = ((ROWS * HEADS - 1) * HEAD_DIM) as usize;
    assert!(
        moved * 100 >= finite * 95,
        "dropping the ln2 moved {moved} of {finite} finite elements — under \
         95%, so the operands have drifted out of the regime the correction \
         lives in"
    );

    // And not by an ulp: at these lse magnitudes the correction FACTOR is a
    // different number, not a differently-rounded one. `sink = 2.515` (head
    // 0) at a base-two lse of 3 is 0.393 rebased and 0.619 not — the middle
    // of the distribution, which is what decides an argmax.
    let mut worst = 0f32;
    for (a, b) in with.iter().zip(&without) {
        worst = worst.max((a - b).abs() / a.abs().max(1e-3));
    }
    assert!(
        worst > 0.25,
        "the ln2 rebase only moves the answer by {worst} relative — the \
         operands are not exercising the correction"
    );
}

/// THE OTHER MUTATION, and the one that refused gpt-oss by name: the same
/// `[heads]` of BF16 sink bytes, read at an f32 stride.
///
/// `attention.sink` cannot be asked for this any more — its sink slot rides
/// the point's element, so the call does not typecheck — but the legacy
/// sibling `norm::attn_sink_correction` still reads `const float*`, and
/// firing it at the checkpoint's own bytes is what the old declaration
/// would have compiled to. The assertion is that the answer is WRONG, and
/// wrong by more than a tolerance: two bf16 sinks under one f32 read is a
/// number with no relation to either.
#[test]
fn a_bf16_sink_read_at_f32_is_a_different_number() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.sink") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let (o, lse) = operands();
    let sinks: Vec<u16> = SINKS.iter().map(|s| narrow(*s)).collect();

    let d_o = Slab::of(&bytes_of_u16(&o));
    let d_lse = Slab::of(&bytes_of_f32(&lse));
    // The bytes the checkpoint ships, and nothing else: `[64]` at BF16 is
    // 128 bytes, and an f32 read of it walks off the end at head 3. The
    // slab is padded so the fire stays in bounds and the numbers stay
    // nonsense, which is the honest shape of the defect — a reinterpretation
    // reads SOMETHING and the something is not a sink.
    let mut padded = bytes_of_u16(&sinks);
    padded.extend(std::iter::repeat_n(0u8, padded.len()));
    let d_sink = Slab::of(&padded);

    kernels_cuda::norm::attn_sink_correction::<bf16>(
        &ctx,
        InOut {
            ptr: d_o.ptr.cast(),
            rows: ROWS,
            width: HEADS * HEAD_DIM,
        },
        In {
            ptr: d_lse.ptr.cast(),
            rows: ROWS,
            width: HEADS,
        },
        Const {
            v: d_sink.ptr.cast_const().cast(),
        },
        Const::new(HEAD_DIM),
    )
    .expect("the f32-reading sibling fires");

    let got = d_o.read_u16(o.len());
    let want = host_sink(&o, &lse, &SINKS, core::f32::consts::LN_2);
    let moved = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| (wide(**g) - **w).abs() > w.abs().max(1e-3) / 256.0)
        .count();
    let finite = ((ROWS * HEADS - 1) * HEAD_DIM) as usize;
    assert!(
        moved > finite / 2,
        "reading the checkpoint's bf16 sinks at an f32 stride produced the \
         RIGHT answer at {} of {finite} elements — too plausible for this \
         test to be measuring the reinterpretation",
        finite - moved
    );
}

/// `attention.merge_lse` folds two halves in base two, and leaves an lse
/// the sink beside it can read.
///
/// THE WEIGHTS ARE BASE-FREE AND THE RESULT IS NOT. `b^(l1−m)` against
/// `b^(l2−m)` is the same ratio in any base, so a merge that used the wrong
/// exponential still produces a plausible output rectangle — what it gets
/// wrong is `lse_out`, and the only thing that reads `lse_out` is the next
/// merge or the sink. So this checks the folded output AND the folded lse,
/// and the second assertion is the one with teeth.
#[test]
fn the_merge_folds_in_base_two() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.merge_lse") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let n = (ROWS * HEADS * HEAD_DIM) as usize;
    let rowsheads = (ROWS * HEADS) as usize;
    let o1: Vec<u16> = (0..n).map(|_| narrow(rng.next() * 2.0)).collect();
    let o2: Vec<u16> = (0..n).map(|_| narrow(rng.next() * 2.0)).collect();
    // Two halves whose lses OVERLAP: a fold only folds while the two are
    // within a few units of each other — further apart than that and both
    // bases agree that the larger half wins outright, which is a merge
    // neither base can get wrong.
    let lse1: Vec<f32> = (0..rowsheads).map(|_| 4.0 + rng.next() * 2.0).collect();
    let lse2: Vec<f32> = (0..rowsheads).map(|_| 4.0 + rng.next() * 2.0).collect();

    let d_o1 = Slab::of(&bytes_of_u16(&o1));
    let d_o2 = Slab::of(&bytes_of_u16(&o2));
    let d_l1 = Slab::of(&bytes_of_f32(&lse1));
    let d_l2 = Slab::of(&bytes_of_f32(&lse2));
    let d_o = Slab::of(&bytes_of_u16(&vec![narrow(-99.0); n]));
    let d_l = Slab::of(&bytes_of_f32(&vec![-99.0; rowsheads]));

    Attention::merge_lse::<bf16>(
        &ctx,
        In {
            ptr: d_o1.ptr.cast(),
            rows: ROWS,
            width: HEADS * HEAD_DIM,
        },
        In {
            ptr: d_l1.ptr.cast(),
            rows: ROWS,
            width: HEADS,
        },
        In {
            ptr: d_o2.ptr.cast(),
            rows: ROWS,
            width: HEADS * HEAD_DIM,
        },
        In {
            ptr: d_l2.ptr.cast(),
            rows: ROWS,
            width: HEADS,
        },
        HEADS as u32,
        HEAD_DIM as u32,
        Out {
            ptr: d_o.ptr.cast(),
            rows: ROWS,
            width: HEADS * HEAD_DIM,
        },
        Out {
            ptr: d_l.ptr.cast(),
            rows: ROWS,
            width: HEADS,
        },
    )
    .expect("`attention.merge_lse` at bf16");

    let got_o = d_o.read_u16(n);
    let got_l = d_l.read_f32(rowsheads);

    let mut want_o = vec![0f32; n];
    let mut want_l = vec![0f32; rowsheads];
    for i in 0..rowsheads {
        let (l1, l2) = (lse1[i], lse2[i]);
        let m = l1.max(l2);
        let (w1, w2) = ((l1 - m).exp2(), (l2 - m).exp2());
        want_l[i] = m + (w1 + w2).log2();
        for d in 0..HEAD_DIM as usize {
            let at = i * HEAD_DIM as usize + d;
            want_o[at] = (wide(o1[at]) * w1 + wide(o2[at]) * w2) / (w1 + w2);
        }
    }

    let worst = worst(&got_o, &want_o);
    assert!(worst < 1.0 / 128.0, "the merged output is off by {worst}");

    // The lse, to fp32 and not to bf16: it is an f32 slot and the sink
    // downstream reads every bit of it.
    for (i, (g, w)) in got_l.iter().zip(&want_l).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "merged lse[{i}] is {g}, base two says {w}"
        );
    }

    // THE MUTATION: the same fold in natural log. It agrees with base two
    // on the OUTPUT to a few parts in a thousand — which is why this test
    // asserts on the lse — and disagrees on every lse it leaves.
    let mut ln_l = vec![0f32; rowsheads];
    for i in 0..rowsheads {
        let (l1, l2) = (lse1[i], lse2[i]);
        let m = l1.max(l2);
        ln_l[i] = m + ((l1 - m).exp() + (l2 - m).exp()).ln();
    }
    assert_eq!(
        separated(&want_l, &ln_l),
        rowsheads,
        "an ln fold and a base-two fold leave the same lse somewhere, so \
         this file cannot tell the two apart"
    );
}
