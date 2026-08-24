//! Where Metal's transcendentals stop being the functions they name.
//!
//! This file exists because the same arithmetic, written three times for
//! three backends, was wrong on one of them in a way no reasoning about the
//! formula would find.
//!
//! WGSL's `tanh` is `(exp(2x) - 1) / (exp(2x) + 1)` on the wgpu backend.
//! `exp(2x)` overflows f32 at `2x > ln(f32::MAX) = 88.7`, and past there the
//! expression is `inf / inf` -- **NaN**, for arguments whose correct answer is
//! the most obvious one the function has, exactly 1. Two shaders were sitting
//! on it:
//!
//!   * `logit_softcap` returned NaN for any logit past `44.36 * cap`, which
//!     at gemma's `cap = 12.5` is 555.
//!   * `gelu_tanh`'s inner term is `0.798 * (g + 0.0447 g^3)`, which crosses
//!     44.36 at a **gate of 10.5** -- an ordinary FFN activation.
//!
//! Metal spells the same two functions with `precise::tanh`, which is the
//! library implementation and is supposed to saturate. "Supposed to" is the
//! word this crate does not accept, and the wgpu defect is exactly what a
//! reasonable person would have assumed away: `tanh` saturating is the least
//! surprising property any transcendental has.
//!
//! So this measures it. Not to fix Metal -- as of this writing Metal is right
//! -- but so that "Metal is right" is a fact this tree checks on every device
//! it runs on, rather than a thing somebody once believed. The two shaders
//! whose Metal text these tests fire are the SAME TWO the wgpu fix touched.
//!
//! # What a failure here would mean
//!
//! That this device's `precise::tanh` has the wgpu implementation's shape,
//! and that gemma's FFN and gemma's logit cap both produce NaN on it. The
//! remedy would be the one the wgpu shaders now carry: clamp the argument to
//! ±16, which changes nothing that was right, because `tanh(x)` is exactly
//! 1.0 in f32 for `x >= 9.02`.

#![cfg(target_vendor = "apple")]

use driver_metal::baker::dispatch::{Dispatch, ParamSlot, Touches};
use driver_metal::baker::{BoundRegion as BoundArg, Slice};
use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;

/// The overflow boundary an `exp(2x)` tanh has, and the arguments around it.
///
/// `44.36` is `ln(f32::MAX) / 2`. Everything at or past it is where a naive
/// implementation returns NaN and a correct one returns 1.
use driver_metal::skip::skipped;

const GELU_GATES: [f32; 8] = [1.5, -1.5, 10.0, 10.5, 11.0, 64.0, 1024.0, -1024.0];

/// gemma's cap, and logits either side of `44.36 * cap = 554.5`.
const SOFTCAP: f32 = 12.5;
const SOFTCAP_LOGITS: [f32; 8] = [3.0, -3.0, 512.0, 552.0, 560.0, -560.0, 4096.0, 3.0e38];

/// `gelu_tanh(g) * u`, in f32, as the shader states it.
fn gelu_tanh_reference(g: f32, u: f32) -> f32 {
    let k = 0.797_884_6_f32;
    let inner = k * (g + 0.044_715 * g * g * g);
    0.5 * g * (1.0 + inner.tanh()) * u
}

/// `precise::tanh` saturates, so gemma's GeGLU is finite for any gate.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gelu_tanh_activation_saturates_rather_than_overflowing() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let n = GELU_GATES.len();
    let ups: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.5 } else { -1.5 })
        .collect();
    let gate = alloc_bf16(&context, &GELU_GATES, "gate");
    let up = alloc_bf16(&context, &ups, "up");
    let out = alloc_bf16(&context, &vec![-99.0; n], "out");

    fire(
        &context,
        &compiler,
        "geglu_tanh_bfloat16",
        "mlp/gated.metal",
        &[gate.gpu_address(), up.gpu_address(), out.gpu_address()],
        4,
        3,
        &[0],
        n as u32,
    );

    let got = read_bf16(&out, n);
    let mut worst = 0.0f32;
    for (i, (&g, u)) in GELU_GATES.iter().zip(&ups).enumerate() {
        let seen_g = from_bf16(to_bf16(g));
        let seen_u = from_bf16(to_bf16(*u));
        let want = from_bf16(to_bf16(gelu_tanh_reference(seen_g, seen_u)));
        assert!(
            got[i].is_finite(),
            "gate {seen_g} gave {} -- `precise::tanh` on this device has the \
             overflow shape wgpu's did, and every gemma FFN activation past \
             10.5 is a NaN. Clamp the inner term to ±16 in \
             `mlp/gated.metal`, which changes nothing that was right",
            got[i]
        );
        let bound = (want.abs() / 128.0).max(1.0 / 64.0);
        let took = (got[i] - want).abs() / bound;
        worst = worst.max(took);
        assert!(
            took <= 1.0,
            "gate {seen_g}, up {seen_u}: got {} want {want}",
            got[i]
        );
    }
    tolerance_holds(worst, "the gelu activation");

    // The saturation, named rather than left to the comparison: at 1024 the
    // inner term is 3.8e7, and `0.5 * g * (1 + tanh)` is `g * u` exactly.
    let far = from_bf16(to_bf16(1024.0)) * from_bf16(to_bf16(ups[6]));
    assert_eq!(
        got[6], far,
        "a gate of 1024 must pass straight through -- `tanh` of 3.8e7 is one, \
         so gelu is the identity there"
    );
    assert_eq!(
        got[7], 0.0,
        "a gate of -1024 must be a finite ZERO: `tanh` is -1, `1 + tanh` is \
         0, and the product is 0 and not `-inf * 0`"
    );
}

/// `cap * tanh(x / cap)` is the cap for any logit, not a NaN.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_logit_softcap_saturates_rather_than_overflowing() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let n = SOFTCAP_LOGITS.len();
    let logits = alloc_bf16(&context, &SOFTCAP_LOGITS, "logits");
    let out = alloc_bf16(&context, &vec![-99.0; n], "out");

    fire(
        &context,
        &compiler,
        "logit_softcap_bfloat16",
        "attn/logit_softcap.metal",
        &[logits.gpu_address(), out.gpu_address()],
        3,
        2,
        &[SOFTCAP.to_bits()],
        n as u32,
    );

    let got = read_bf16(&out, n);
    let mut worst = 0.0f32;
    for (i, &x) in SOFTCAP_LOGITS.iter().enumerate() {
        let seen = from_bf16(to_bf16(x));
        let want = from_bf16(to_bf16(SOFTCAP * (seen / SOFTCAP).tanh()));
        assert!(
            got[i].is_finite(),
            "logit {seen} (x/cap = {}) gave {} -- `precise::tanh` on this \
             device overflows the way wgpu's did, and every logit past \
             {} is a NaN",
            seen / SOFTCAP,
            got[i],
            44.36 * SOFTCAP
        );
        let bound = (want.abs() / 128.0).max(1.0 / 64.0);
        let took = (got[i] - want).abs() / bound;
        worst = worst.max(took);
        assert!(took <= 1.0, "logit {seen}: got {} want {want}", got[i]);
    }
    tolerance_holds(worst, "the capped logits");

    for (i, &x) in SOFTCAP_LOGITS.iter().enumerate() {
        if x.abs() > 554.5 {
            assert_eq!(
                got[i].abs(),
                SOFTCAP,
                "logit {x} is past `44.36 * cap`, where the answer is the cap \
                 exactly and where an `exp(2x)` tanh gives NaN"
            );
        }
    }
}

/// The band the measured comparisons must land in.
///
/// Above 1 the assertions did not hold; below 1/8 the bound is more than
/// eight times the arithmetic the device delivers, which is a bound that
/// would pass a broken kernel. `worst == 0.0` is exact agreement -- the
/// absence of anything to bound, not a loose bound -- and is let through so a
/// device more accurate than this one cannot turn a better answer red.
fn tolerance_holds(worst: f32, what: &str) {
    if worst == 0.0 {
        return;
    }
    assert!(
        (0.125..=1.0).contains(&worst),
        "{what}: the widest comparison took {worst} of its bound. Under 1/8 \
         the bound is not measuring this device; over 1 the assertions above \
         did not hold"
    );
}

/// One elementwise dispatch: buffers at 0.., a scalar struct, 1D grid.
#[allow(clippy::too_many_arguments)]
fn fire(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    symbol: &'static str,
    file: &'static str,
    buffers: &[u64],
    slots: usize,
    params_slot: usize,
    params: &[u32],
    n: u32,
) {
    let mut args = vec![
        BoundArg {
            slice: Slice {
                address: buffers[0],
                bytes: 1 << 16,
            },
            width: 0,
        };
        slots
    ];
    for (slot, address) in buffers.iter().enumerate() {
        args[slot] = BoundArg {
            slice: Slice {
                address: *address,
                bytes: 1 << 16,
            },
            width: 0,
        };
    }
    let dispatch = Dispatch {
        symbol,
        file,
        stamp: "",
        grid: [n, 1, 1],
        threadgroup: [n.min(32), 1, 1],
        // Conservative, as a positional row's would be: this fixture drives
        // one dispatch at a time, so what it says about hazards is moot.
        touches: Touches::everything(&args),
        args,
        params: params.to_vec(),
        // ONE SLOT AS WIDE AS THE RUN, which is how today's `ParamSlot`
        // spells what `packed: true` used to. The shader takes its scalars as
        // a `constant Params&` — the address of the run's first word — so the
        // slot binds at offset zero and reads every word of it. A slot fixed
        // at four bytes would stage `params[0]` and leave the rest of the
        // struct reading whatever the region held.
        param_slots: vec![ParamSlot {
            slot: params_slot,
            at: 0,
            bytes: u32::try_from(size_of_val(params)).expect("a small run"),
            value: 0,
        }],
        layers: 0..1,
        op: 0,
    };
    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, std::slice::from_ref(&dispatch))
        .expect("the kernel builds");
    let staged =
        Params::stage(context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table = ArgumentTable::new(context, slots).expect("a table");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .expect("the kernel fires");
}

fn kernels_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn from_bf16(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

fn alloc_bf16(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let narrow: Vec<u16> = values.iter().copied().map(to_bf16).collect();
    let bytes = std::mem::size_of_val(narrow.as_slice()) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        let raw = core::slice::from_raw_parts(
            narrow.as_ptr().cast::<u8>(),
            std::mem::size_of_val(narrow.as_slice()),
        );
        a.write(0, raw).expect("the halves fit");
    }
    a
}

fn read_bf16(a: &Allocation, n: usize) -> Vec<f32> {
    let words = unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<u16>(), n) };
    words.iter().copied().map(from_bf16).collect()
}
