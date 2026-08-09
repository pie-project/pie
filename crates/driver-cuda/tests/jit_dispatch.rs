//! The cutover's proof: a row fires through **this driver's dispatcher** and
//! the kernel that ran was compiled by `kernels-cuda-new` at run time.
//!
//! `tier_a_pilot::a_row_with_no_launcher_still_fires` already showed that a
//! shim-less row fires — but it fired it by hand, calling `KernelModule::fire`
//! with a `Dims` it built itself. `kernels-cuda-new`'s own `tests/fire.rs`
//! does the same on the other side of the seam. Neither goes through
//! [`driver_cuda::bind::dispatch`], and the whole cutover is the claim that
//! the *dispatcher* now reaches the JIT crate — so neither test can fail if
//! the routing is wrong. Both would stay green with the arm deleted.
//!
//! This one cannot. It builds a [`BoundLaunch`] the way the executor does,
//! hands it to the same `dispatch` a real fire calls, and reads the bytes
//! back:
//!
//! ```text
//! BoundLaunch + LaunchSpec + DispatchCtx
//!   -> driver_cuda::bind::dispatch          the executor's entry
//!   -> dispatch_generated                   the emitted match
//!   -> jit_dims                             nine axes from ctx + operands
//!   -> crate::bind::jit::fire               the seam
//!   -> kernels_cuda_new::fire               table, cache, NVRTC, cuLaunchKernel
//! ```
//!
//! Every link is the production one. `norm::scalar_mul_bf16` has no
//! `pie_k_norm_scalar_mul_bf16` to fall back to — `emit_c_shim` skips the rows
//! on `JIT_DISPATCHED` — so if the arm did not route, this binary would not
//! link, and if it routed to a symbol the JIT table lacks the fire would
//! refuse and the bytes would be unchanged. The assertion on the bytes is what
//! separates "the call happened" from "the kernel ran".
//!
//! Needs `bridge`: `dispatch` is the bridge executor's entry point and is
//! gated on it. Skipped without a device, like every `gpu_*` binary here.

use std::collections::BTreeMap;
use std::ffi::c_void;

use driver_cuda::bind::{
    BoundArg, BoundLaunch, DispatchCtx, Frame, LaunchSpec, Resolver, dispatch,
};
use driver_cuda::device::{Allocator, OwnedStream};
use model_compiler::trace::ValueId;

mod common;
use common::{device_or_skip, gpu_guard};

/// Wide enough that the elementwise rule spans several blocks, small enough
/// to check every element.
const N: usize = 4096;
const SCALE: f32 = 2.5;

/// `f32 -> bf16`, round-to-nearest-even, as `__float2bfloat16` converts.
///
/// Written out rather than pulled from a crate for `tier_a_pilot`'s reason:
/// the reference has to round the way the hardware does, or the comparison
/// measures the rounding instead of the kernel.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return (bits >> 16) as u16 | 0x0040;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// `bf16 -> f32`, exact — the low sixteen bits are zero.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s_of_bytes(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Values spread across exponents so a wrong grid shows up as a wrong element
/// rather than as a plausible one.
fn sample(i: usize) -> f32 {
    ((i % 97) as f32).mul_add(0.125, -6.0)
}

/// A resolver that answers nothing, because the launch under test names
/// nothing.
///
/// `norm::scalar_mul_bf16` reads one arena operand and a wire scalar: no
/// weight, no seam value, no `outs`, no `aux`. `dispatch` still takes a
/// resolver because its signature serves every arm, and a `None` from here
/// would be a DRIFT diagnosis — which is exactly what should happen if a
/// future edit gives this row a name to resolve.
struct Nothing;
impl Resolver for Nothing {
    fn weight(&mut self, _name: &str) -> Option<*const c_void> {
        None
    }
    fn named(&mut self, _value: ValueId) -> Option<*mut c_void> {
        None
    }
}

/// The routed set, as the driver's own build read it.
///
/// Not a GPU test — it is the statement the rest of this file depends on, and
/// it fails on a host with no driver too. `kernels_cuda_new::hosts` answers
/// off the table with no CUDA call, which is what lets this run anywhere; the
/// build script asserts the same thing at generation time, and this is the
/// binary's copy of that claim. Both exist because they fail differently: the
/// build script stops a bad `JIT_DISPATCHED` from ever producing an arm, and
/// this catches a binary whose linked table disagrees with the one its
/// dispatcher was generated from.
#[test]
fn every_routed_symbol_is_hosted_by_the_jit_crate() {
    for symbol in kernels_cuda_new::device::JIT_DISPATCHED {
        assert!(
            kernels_cuda_new::hosts(symbol),
            "{symbol} is routed to the JIT crate and no unit there hosts it -- \
             the dispatcher would call a symbol that resolves to nothing and \
             there is no shim left to catch it"
        );
    }
}

/// The deliverable: `dispatch` fires a shim-less row and the bytes are right.
#[test]
fn a_routed_row_fires_through_the_drivers_own_dispatcher() {
    let _gpu = gpu_guard();
    let Some(_device) = device_or_skip("dispatcher -> kernels-cuda-new") else {
        return;
    };

    // The premise, stated rather than assumed: this symbol is on the list the
    // shim emitter skips, so the arm the dispatcher takes below is the JIT one
    // or there is no arm at all.
    assert!(
        kernels_cuda_new::device::JIT_DISPATCHED.contains(&"norm::scalar_mul_bf16"),
        "shorten JIT_DISPATCHED and this test is measuring the AOT path"
    );

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();
    let raw_stream = s.as_raw().cast::<c_void>();

    let host: Vec<u16> = (0..N).map(|i| to_bf16(sample(i))).collect();
    let mut x = alloc.alloc(N * 2).expect("x");
    x.copy_from_host(&bytes_of_u16(&host), s).expect("upload");
    let x_ptr = x.ptr_at(0, N * 2).expect("x is live");

    // The launch as the executor builds it. `n_in = 0` is the in-place shape
    // the lowering states for a scalar multiply: one operand, which is the
    // output, so the arm's `stage_d2d` copy is skipped and the kernel reads
    // and writes the same bytes.
    let bound = BoundLaunch {
        kernel: "norm::scalar_mul_bf16",
        rows: 0..1,
        layers: 0..1,
        args: vec![BoundArg {
            ptr: x_ptr,
            width: u32::try_from(N).expect("N fits u32"),
        }],
    };
    let spec = LaunchSpec {
        n_in: 0,
        n_out: 1,
        // The scale rides the wire as BITS, which is why the arm reads it
        // through `f32::from_bits` and not as an integer.
        params: vec![SCALE.to_bits()],
        ..LaunchSpec::default()
    };

    // The arena is the operand's own allocation: nothing here is resolved by
    // offset, and `dispatch` only bounds-checks what it resolves.
    let frame = Frame {
        arena: x_ptr,
        arena_bytes: N * 2,
    };

    // Geometry an elementwise row does not read, spelled anyway, because
    // `jit_dims` reads all of it and a zero here would be indistinguishable
    // from the two axes that are zero on purpose.
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: std::ptr::null_mut(),
        eps: 1e-6,
        rope_theta: 1e4,
        rope_theta_by_layer: Vec::new(),
        rotary_by_layer: Vec::new(),
        head_dim: 128,
        num_q_heads: 16,
        num_kv_heads: 8,
        vocab: 0,
        gate_second: false,
        rope_interleaved: false,
        token_ids: std::ptr::null_mut(),
        positions: std::ptr::null_mut(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: BTreeMap::new(),
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        // ZERO IS ABSENCE, and this fixture is a dense fire. `RoutedQmv`,
        // `RoutedQmvTransposed` and `RoutedQmvQuad` read it and refuse a
        // zero, which is what a fire with no routed statement should get.
        experts_per_token: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        lora: None,
        peel_window: std::ptr::null(),
        rows_total: 0,
        sampling_indices: std::ptr::null(),
        sampled_rows: 0,
    };

    let mut resolver = Nothing;
    dispatch(&bound, &spec, frame, &mut resolver, &ctx, None, None)
        .expect("the dispatcher has an arm for a routed row");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; N * 2];
    x.copy_to_host(&mut got, s).expect("download");
    let got = u16s_of_bytes(&got);

    // The kernel multiplies in bf16, so the reference has to round the scale
    // to bf16 first or the last bit disagrees on about half the elements.
    let rounded = from_bf16(to_bf16(SCALE));
    let mut changed = 0usize;
    for i in 0..N {
        let want = to_bf16(from_bf16(host[i]) * rounded);
        assert_eq!(
            got[i], want,
            "scalar_mul[{i}]: {:?} -> {:?}",
            host[i], got[i]
        );
        changed += usize::from(got[i] != host[i]);
    }
    // A refused fire leaves the buffer alone, and `sample` crosses zero — so
    // "every element matches" is satisfiable by doing nothing on exactly one
    // of them. This is the check that the kernel ran.
    assert!(
        changed > N / 2,
        "only {changed} of {N} elements moved: did the fire happen?"
    );
}
