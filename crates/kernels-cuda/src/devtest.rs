//! The two kernels W3 added, fired on a real device against a host
//! reference — `norm.scale` and `layout.select`.
//!
//! A UNIT MODULE AND NOT `tests/`, because an integration test cannot reach
//! `cudarc`. This crate's device dependency is `optional = true` behind
//! `_cuda`, and Cargo does not allow an optional dev-dependency, so a
//! `tests/` file would either have to pull `cudarc` in unconditionally —
//! which breaks the plain `cargo check` this crate's Cargo.toml goes to some
//! length to keep GPU-free — or reach `jit::device`, which is
//! `pub(crate)`. A unit module has both, at the cost of one `#[cfg]`.
//!
//! WHY THESE TWO AND NOT THE FAMILY. Every other body in `norm.rs` and
//! `layout.rs` delegates to a routine the legacy driver has been firing
//! against real checkpoints for months; these two are the only ones whose
//! device text is NEW, so they are the only ones whose arithmetic has never
//! been run. A reference here is a first execution, not a regression net.
//!
//! The comparisons are BIT-EXACT, which they can be because both kernels are
//! exactly reproducible on the host: `select` copies, and `scale` widens to
//! fp32, multiplies, and narrows round-to-nearest-even. `to_bf16` below is
//! `prelude/device.cuh`'s `f32_to_bf16` transcribed, so a rounding mode that
//! drifts on either side fails here rather than showing up as a tolerance
//! nobody can source.

use core::ffi::c_void;

use cudarc::runtime::sys as rt;

use crate::jit::Ctx;
use crate::jit::abi::{Tensor, bf16};
use kernels::points::{Layout, Norm};
use kernels::routine::{Const, In, InOut, Out};

/// `prelude/device.cuh`'s `f32_to_bf16`, on the host. Round-to-nearest-even,
/// with no NaN arm: nothing below feeds one.
fn to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b + rounding) >> 16) as u16
}

fn of_bf16(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// A stream and the slabs a test allocates on it, freed together.
struct Device {
    stream: *mut c_void,
    slabs: Vec<*mut c_void>,
}

impl Device {
    /// The device, or `None` on a machine with no usable one — which is a
    /// SKIP and not a failure, for `driver-cuda/tests/common`'s reason: a
    /// host sweep has to be able to run this crate's suite.
    fn open() -> Option<Device> {
        if unsafe { rt::cudaSetDevice(0) } != rt::cudaError::cudaSuccess {
            return None;
        }
        // The primary context, up: `cudaSetDevice` only records a
        // thread-local ordinal, and the driver-API calls the JIT makes need
        // the context to exist (`baker-smoke/src/dev.rs` says the same).
        if unsafe { rt::cudaFree(core::ptr::null_mut()) } != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut raw: rt::cudaStream_t = core::ptr::null_mut();
        let code = unsafe {
            rt::cudaStreamCreateWithPriority(&raw mut raw, rt::cudaStreamNonBlocking, 0)
        };
        (code == rt::cudaError::cudaSuccess).then(|| Device {
            stream: raw.cast(),
            slabs: Vec::new(),
        })
    }

    fn up(&mut self, host: &[u16]) -> *mut bf16 {
        let bytes = std::mem::size_of_val(host);
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes) },
            rt::cudaError::cudaSuccess
        );
        self.slabs.push(ptr);
        assert_eq!(
            unsafe {
                rt::cudaMemcpyAsync(
                    ptr,
                    host.as_ptr().cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream.cast(),
                )
            },
            rt::cudaError::cudaSuccess
        );
        ptr.cast()
    }

    fn down(&self, src: *const bf16, len: usize) -> Vec<u16> {
        let mut host = vec![0u16; len];
        assert_eq!(
            unsafe {
                rt::cudaMemcpyAsync(
                    host.as_mut_ptr().cast(),
                    src.cast(),
                    len * 2,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    self.stream.cast(),
                )
            },
            rt::cudaError::cudaSuccess
        );
        assert_eq!(
            unsafe { rt::cudaStreamSynchronize(self.stream.cast()) },
            rt::cudaError::cudaSuccess,
            "the fire itself failed, not the copy"
        );
        host
    }

    /// A `Ctx` on this stream.
    ///
    /// # Safety
    ///
    /// The stream outlives every fire, because `Device` owns it and nothing
    /// here hands the `Ctx` past the borrow.
    fn ctx(&self) -> Ctx<'_> {
        unsafe { Ctx::on(self.stream) }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        for slab in self.slabs.drain(..) {
            let _ = unsafe { rt::cudaFree(slab) };
        }
        let _ = unsafe { rt::cudaStreamDestroy(self.stream.cast()) };
    }
}

/// `x *= s[0]` over a rectangle whose element count is NOT a whole number of
/// blocks — 3 x 37 is 111, and the launch covers 256. The tail threads are
/// the only guard this kernel has, and a rectangle that filled its blocks
/// exactly would never test it.
///
/// The scalar is `0.1` rather than a power of two on purpose: a dyadic
/// factor multiplies bf16 exactly and would agree with any rounding mode at
/// all, which is not a measurement of the one this kernel states.
#[test]
fn scale_multiplies_by_a_device_scalar() {
    let Some(mut dev) = Device::open() else {
        eprintln!("no cuda device: skipping");
        return;
    };
    const ROWS: i32 = 3;
    const WIDTH: i32 = 37;
    let n = (ROWS * WIDTH) as usize;

    let host: Vec<u16> = (0..n)
        .map(|i| to_bf16((i as f32) * 0.5 - 7.0))
        .collect();
    let s = to_bf16(0.1);

    let x = dev.up(&host);
    let scalar = dev.up(&[s]);

    dev.ctx()
        .scale::<bf16>(
            Const::<Tensor<bf16>>::new(scalar.cast_const()),
            InOut::<Tensor<bf16>> {
                ptr: x,
                rows: ROWS,
                width: WIDTH,
            },
        )
        .expect("the point is claimed and the rectangle is well formed");

    let got = dev.down(x.cast_const(), n);
    let want: Vec<u16> = host
        .iter()
        .map(|&v| to_bf16(of_bf16(v) * of_bf16(s)))
        .collect();
    assert_eq!(got, want, "s = {}", of_bf16(s));
}

/// One layer's slice out of a `[rows, layers * width]` relay.
///
/// Every element carries its own address — `row * 1000 + column`, all of
/// which are integers bf16 represents exactly — so a slice taken at the
/// wrong column or off the wrong row fails with the offset it actually
/// read rather than with a norm.
///
/// The middle layer, not the first or the last: layer 0 would pass with the
/// offset dropped entirely, and the last would pass with the row pitch
/// mistaken for the slice width.
#[test]
fn select_takes_one_layers_slice() {
    let Some(mut dev) = Device::open() else {
        eprintln!("no cuda device: skipping");
        return;
    };
    const ROWS: i32 = 4;
    const LAYERS: u32 = 5;
    const WIDTH: u32 = 7;
    const LAYER: u32 = 3;
    let stride = (LAYERS * WIDTH) as i32;

    let table: Vec<u16> = (0..ROWS)
        .flat_map(|r| (0..stride).map(move |c| to_bf16((r * 1000 + c) as f32)))
        .collect();
    let src = dev.up(&table);
    let dst = dev.up(&vec![to_bf16(-1.0); (ROWS * WIDTH as i32) as usize]);

    dev.ctx()
        .select::<bf16>(
            In::<Tensor<bf16>> {
                ptr: src.cast_const(),
                rows: ROWS,
                width: stride,
            },
            LAYER,
            WIDTH,
            Out::<Tensor<bf16>> {
                ptr: dst,
                rows: ROWS,
                width: WIDTH as i32,
            },
        )
        .expect("the point is claimed and the slice is inside the row");

    let got = dev.down(dst.cast_const(), (ROWS * WIDTH as i32) as usize);
    let want: Vec<u16> = (0..ROWS)
        .flat_map(|r| {
            (0..WIDTH as i32).map(move |i| to_bf16((r * 1000 + (LAYER * WIDTH) as i32 + i) as f32))
        })
        .collect();
    assert_eq!(got, want);
}

/// A slice that runs off the end of the relay is REFUSED, not clamped and
/// not read.
///
/// No device needed and none taken: the check stands ahead of the fire, so
/// the null stream is never touched. That is the property being asserted as
/// much as the refusal is — a bound checked after the launch is not a bound.
#[test]
fn select_refuses_a_layer_the_relay_does_not_reach() {
    // SAFETY: every path below returns before `fire`, so the stream is
    // never dereferenced.
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let table = In::<Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: 2,
        width: 20,
    };
    let out = |width: i32| Out::<Tensor<bf16>> {
        ptr: core::ptr::null_mut(),
        rows: 2,
        width,
    };

    // Layer 4 of a 20-wide relay at width 7 starts at column 28 — well past
    // the end, the case a missing check would fault on.
    let past = ctx.select::<bf16>(table, 4, 7, out(7));
    assert!(
        matches!(past, Err(kernels::Refusal::Narrow { .. })),
        "{past:?}"
    );

    // Layer 2 spans columns 14..21, ONE column short, which is the boundary
    // an off-by-one check gets wrong in the direction that still reads.
    let edge = ctx.select::<bf16>(table, 2, 7, out(7));
    assert!(
        matches!(edge, Err(kernels::Refusal::Narrow { .. })),
        "{edge:?}"
    );

    // The stated width and the result's own rectangle disagree, which is the
    // walk and the fire disagreeing about what was allocated.
    let mismatch = ctx.select::<bf16>(table, 1, 7, out(9));
    assert!(
        matches!(mismatch, Err(kernels::Refusal::Narrow { .. })),
        "{mismatch:?}"
    );
}
