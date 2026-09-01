//! **THE DEVICE A KERNEL GOLDEN NEEDS AND NOTHING MORE**: a stream, the
//! allocations made on it, and the two bf16 conversions the host half of a
//! comparison does.
//!
//! Shared by the tower goldens (`tower_*.rs`) the way `tests/common` is
//! shared everywhere — a directory under `tests/` is not a target, so this
//! file is compiled once per test that says `mod common;` and never on its
//! own. The `Gpu` here is `channel_kernels.rs`'s, trimmed to what a kernel
//! golden uses: nothing pinned, nothing mapped, no rings.

#![allow(dead_code)]

use core::ffi::c_void;

use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::jit::Ctx;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// One test's device: a stream, and every allocation made on it, freed
/// together. Deliberately not shared between tests — cargo runs them on
/// threads, and a stream per test is what keeps two fires' enqueues apart.
pub struct Gpu {
    stream: rt::cudaStream_t,
    device: Vec<*mut c_void>,
}

/// **THE CACHE ROOT A TEST RUN STATES**, so that nineteen test binaries do not
/// each pay NVRTC for the same instantiations, run after run.
///
/// The library reads no environment and this does not change that:
/// `CARGO_TARGET_TMPDIR` is a COMPILE-TIME macro cargo defines for integration
/// tests, so what the harness installs is a constant baked into this binary.
/// It lands under `target/`, which means `cargo clean` reclaims it, it never
/// escapes the workspace, and it cannot appear in a shipped binary — the macro
/// is not defined for a library build at all.
///
/// Shared by every test that opens a device, and idempotent by
/// [`install`](kernels_cuda::disk::install)'s own contract, so no test has to
/// know whether it ran first.
pub fn arm_cache() {
    kernels_cuda::disk::install(Some(std::path::Path::new(concat!(
        env!("CARGO_TARGET_TMPDIR"),
        "/kernel-cache"
    ))));
}

impl Gpu {
    pub fn open() -> Self {
        arm_cache();
        unsafe {
            check(rt::cudaSetDevice(0), "cudaSetDevice");
            let mut stream: rt::cudaStream_t = core::ptr::null_mut();
            check(rt::cudaStreamCreate(&raw mut stream), "cudaStreamCreate");
            Self {
                stream,
                device: Vec::new(),
            }
        }
    }

    /// The context the entries fire through — the same `Ctx::on` an engine
    /// `Run` builds, on this test's stream.
    pub fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire in a test, and `Gpu`'s drop
        // synchronizes before destroying it.
        unsafe { Ctx::on(self.stream.cast()) }
    }

    /// `bytes` of zeroed device memory.
    pub fn zeros(&mut self, bytes: usize) -> u64 {
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

    /// A device copy of `values`.
    pub fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        let bytes = core::mem::size_of_val(values);
        let at = self.zeros(bytes.max(1));
        if bytes > 0 {
            unsafe {
                check(
                    rt::cudaMemcpy(
                        at as *mut c_void,
                        values.as_ptr().cast(),
                        bytes,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy H2D",
                );
            }
        }
        at
    }

    pub fn down<T: Copy + Default>(&self, at: u64, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        unsafe {
            check(
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    at as *const c_void,
                    core::mem::size_of_val(out.as_slice()),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ),
                "cudaMemcpy D2H",
            );
        }
        out
    }

    pub fn sync(&self) {
        unsafe {
            check(
                rt::cudaStreamSynchronize(self.stream),
                "cudaStreamSynchronize",
            );
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            rt::cudaStreamSynchronize(self.stream);
            for at in self.device.drain(..) {
                rt::cudaFree(at);
            }
            rt::cudaStreamDestroy(self.stream);
        }
    }
}

/// **THE DEVICE'S OWN ROUNDING, TRANSCRIBED** — `prelude/device.cuh`'s
/// `f32_to_bf16`, tie-to-even and NaN-quieting included. A golden that
/// rounded differently from the kernel would be measuring the rounding.
#[must_use]
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}

#[must_use]
pub fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// A deterministic filler. Golden inputs are the same on every machine and
/// every run — a kernel test that drew from a clock could not be bisected.
pub struct Lcg(u64);

impl Lcg {
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// The next value in `[-1, 1)`, already rounded through bf16 so the host
    /// reference and the device read the same numbers.
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        #[allow(clippy::cast_precision_loss)]
        let raw = (bits as f32 / 8_388_608.0) - 1.0;
        from_bf16(to_bf16(raw))
    }

    /// `count` bf16 values, and the f32 the device will read them back as.
    pub fn row(&mut self, count: usize) -> (Vec<u16>, Vec<f32>) {
        let mut raw = Vec::with_capacity(count);
        let mut exact = Vec::with_capacity(count);
        for _ in 0..count {
            let value = self.unit();
            raw.push(to_bf16(value));
            exact.push(value);
        }
        (raw, exact)
    }
}

/// How far two numbers may sit apart before a golden calls it a difference.
/// bf16 carries eight mantissa bits, so one rounding at `|x| ~ 1` is already
/// `2^-8`; the towers' rows are `O(1)` and the kernels round twice (input and
/// output) over a `__expf` whose own error is `2^-21`.
pub const TOLERANCE: f32 = 3.0e-2;

pub fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= TOLERANCE * want.abs().max(1.0)
}
