//! `mlp.swiglu_clamp_alpha`'s claimed body against `gpt_oss_glu`, the split
//! form it was always the packed reading of.
//!
//! THE REFERENCE IS THE OTHER KERNEL, and that is the whole design of this
//! check. The point was never a missing activation — `gpt_oss_glu` has
//! computed gpt-oss's clamp-and-QuickGELU since gpt-oss landed — it was a
//! missing OPERAND SHAPE: that kernel takes gate and up as two planes and
//! the text states one `[gate | up]` row. So `chunked_gpt_oss_glu` is the
//! same arithmetic with the packed indexing around it, and the honest bar is
//! not a tolerance against a host float model but BIT EQUALITY against the
//! kernel it was transcribed from. Both widen to fp32, compute in fp32 and
//! narrow once; if the two ever disagree by an ulp, one of them was tidied.
//!
//! WHAT THE BAR CAN AND CANNOT SEE. bf16 keeps eight mantissa bits, so a
//! difference smaller than that narrowing is invisible here: swapping `expf`
//! for `__expf` in either kernel passes this test, measured. What it catches
//! is everything that survives the round — a symmetric clamp where gpt-oss's
//! gate clamp is one-sided (304 of 1500 elements move), a dropped `alpha`
//! (1140 move), a swapped half. The `.cuh` keeps the spelling identical
//! anyway, which is what leaves those as the only ways the two can drift.
//!
//! # What the packed half adds
//!
//! One thing the split form cannot get wrong: WHICH HALF IS THE GATE. The
//! packed row is `[gate | up]`, `I` wide each, and a kernel that read them
//! the other way round would still produce a clamped GLU — of the wrong
//! operands. The two halves below are drawn from different distributions and
//! the up half is pushed past `limit` on purpose, so a swap changes the
//! answer at nearly every element rather than at a few.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Mlp;
use kernels::routine::{Const, In, Out};
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

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
    }

    fn bf16(&mut self, scale: f32) -> u16 {
        narrow(self.next() * scale)
    }
}

// ── the shape ────────────────────────────────────────────────────────────

/// gpt-oss's numbers, at a toy width.
///
/// `ROWS` is not a multiple of anything and `I` is not a multiple of the
/// 256-thread block, so the packed launch's `i >= I` guard runs and the
/// second block of every row is partly idle — which is where an off-by-one
/// in the half offsets would show.
const ROWS: i32 = 5;
const I: i32 = 300;
/// gpt-oss's `swiglu_limit`.
const LIMIT: f32 = 7.0;
/// gpt-oss's `alpha`, the one `mlp::GPT_OSS_GLU_ALPHA` names.
const ALPHA: f32 = 1.702;

/// The two halves, drawn wide enough that both clamps bite.
///
/// The gate is scaled past `+limit` (an ASYMMETRIC clamp: `fminf(g, limit)`
/// only, so the negative tail is untouched) and the up half past `±limit`
/// (symmetric). A run where neither clamp fires would compare two plain
/// QuickGELUs and say nothing about the clamp at all.
fn halves() -> (Vec<u16>, Vec<u16>) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let n = (ROWS * I) as usize;
    let gate: Vec<u16> = (0..n).map(|_| rng.bf16(12.0)).collect();
    let up: Vec<u16> = (0..n).map(|_| rng.bf16(9.0)).collect();
    (gate, up)
}

#[test]
fn the_packed_row_is_the_split_pair() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mlp.swiglu_clamp_alpha") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let (gate, up) = halves();
    let n = (ROWS * I) as usize;

    // ── the reference: `gpt_oss_glu` over the two halves as two planes ──
    let d_gate = Slab::of(&bytes_of_u16(&gate));
    let d_up = Slab::of(&bytes_of_u16(&up));
    // TWO DIFFERENT POISONS, which is what makes the bit-equality below say
    // "both kernels wrote" as well as "they agree". A single poison in both
    // buffers would let two kernels that wrote NOTHING compare equal; with
    // two, a slot neither touched holds two different values and fails.
    // (Counting poison survivors instead does not work: this activation's
    // range covers any bf16 a poison could be, so a legitimate result may
    // land on it.)
    let d_split = Slab::of(&bytes_of_u16(&vec![narrow(-11.0); n]));
    kernels_cuda::mlp::gpt_oss_glu::<bf16>(
        &ctx,
        In {
            ptr: d_gate.ptr.cast(),
            rows: ROWS,
            width: I,
        },
        In {
            ptr: d_up.ptr.cast(),
            rows: ROWS,
            width: I,
        },
        Out {
            ptr: d_split.ptr.cast(),
            rows: ROWS,
            width: I,
        },
        Const::new(LIMIT),
        Const::new(ALPHA),
    )
    .expect("the split-operand `gpt_oss_glu`");

    // ── the point: one `[gate | up]` row, `I` wide each ──
    let mut packed = Vec::with_capacity(2 * n);
    for r in 0..ROWS as usize {
        packed.extend_from_slice(&gate[r * I as usize..(r + 1) * I as usize]);
        packed.extend_from_slice(&up[r * I as usize..(r + 1) * I as usize]);
    }
    let d_packed = Slab::of(&bytes_of_u16(&packed));
    let d_out = Slab::of(&bytes_of_u16(&vec![narrow(37.5); n]));
    Mlp::swiglu_clamp_alpha::<bf16>(
        &ctx,
        In {
            ptr: d_packed.ptr.cast(),
            rows: ROWS,
            width: 2 * I,
        },
        I.unsigned_abs(),
        LIMIT,
        ALPHA,
        Out {
            ptr: d_out.ptr.cast(),
            rows: ROWS,
            width: I,
        },
    )
    .expect("the claimed `mlp.swiglu_clamp_alpha` body");

    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the two activations did not complete"
    );

    let want = d_split.read_u16(n);
    let got = d_out.read_u16(n);
    let bad = (0..n).filter(|i| got[*i] != want[*i]).count();
    let clamped_gate = gate.iter().filter(|g| wide(**g) > LIMIT).count();
    let clamped_up = up.iter().filter(|u| wide(**u).abs() > LIMIT).count();
    eprintln!(
        "mlp.swiglu_clamp_alpha: {}/{n} bit-identical to `gpt_oss_glu`; \
         {clamped_gate} gate and {clamped_up} up element(s) hit the clamp",
        n - bad
    );
    assert!(
        clamped_gate > n / 8 && clamped_up > n / 8,
        "the sample never exercised the clamps: {clamped_gate} gate, \
         {clamped_up} up out of {n}"
    );
    // BIT EQUALITY, not a tolerance: this is one activation with two operand
    // shapes, so an ulp of disagreement means the transcription drifted.
    assert_eq!(
        bad, 0,
        "{bad} element(s) of the packed reading disagree with the split one"
    );
}

/// The gate half is the FIRST half, and the test above would notice if it
/// were not.
///
/// A separate assertion because it is a separate claim: swapping the halves
/// leaves a legal clamped GLU, so "the outputs match" only means what it
/// should if a swap would break it. This fires the point over the swapped
/// row and requires that the answer CHANGES.
#[test]
fn the_gate_half_comes_first() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mlp.swiglu_clamp_alpha") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let (gate, up) = halves();
    let n = (ROWS * I) as usize;

    let fire = |first: &[u16], second: &[u16]| -> Vec<u16> {
        let mut packed = Vec::with_capacity(2 * n);
        for r in 0..ROWS as usize {
            packed.extend_from_slice(&first[r * I as usize..(r + 1) * I as usize]);
            packed.extend_from_slice(&second[r * I as usize..(r + 1) * I as usize]);
        }
        let d_packed = Slab::of(&bytes_of_u16(&packed));
        let d_out = Slab::of(&bytes_of_u16(&vec![narrow(-11.0); n]));
        Mlp::swiglu_clamp_alpha::<bf16>(
            &ctx,
            In {
                ptr: d_packed.ptr.cast(),
                rows: ROWS,
                width: 2 * I,
            },
            I.unsigned_abs(),
            LIMIT,
            ALPHA,
            Out {
                ptr: d_out.ptr.cast(),
                rows: ROWS,
                width: I,
            },
        )
        .expect("the claimed `mlp.swiglu_clamp_alpha` body");
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "the activation did not complete"
        );
        d_out.read_u16(n)
    };

    let straight = fire(&gate, &up);
    let swapped = fire(&up, &gate);
    let differing = (0..n).filter(|i| straight[*i] != swapped[*i]).count();
    eprintln!("mlp.swiglu_clamp_alpha: {differing}/{n} element(s) move when the halves swap");
    assert!(
        differing > n / 2,
        "swapping `[gate | up]` for `[up | gate]` moved only {differing} of \
         {n} elements — this shape cannot tell the halves apart, so the \
         agreement test above proves less than it looks like"
    );
}
