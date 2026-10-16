#![allow(dead_code)]

use core::ffi::c_void;

use kernels_cuda::cudarc::cublas::sys as blas;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::jit::{Ctx, Slabs};

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

pub struct Gpu {
    stream: rt::cudaStream_t,
    slabs: Slabs,
    device: Vec<*mut c_void>,
    cublas: blas::cublasHandle_t,
}

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
            let mut cublas: blas::cublasHandle_t = core::ptr::null_mut();
            assert_eq!(
                blas::cublasCreate_v2(&raw mut cublas),
                blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "`cublasCreate_v2` did not answer a handle"
            );
            assert_eq!(
                blas::cublasSetStream_v2(cublas, stream.cast()),
                blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "`cublasSetStream_v2` did not bind the test's stream"
            );
            let slabs = Slabs::open();
            slabs.attach(stream.cast());
            Self {
                stream,
                slabs,
                device: Vec::new(),
                cublas,
            }
        }
    }

    pub fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire in a test, and `Gpu`'s drop
        // synchronizes before destroying it.
        unsafe {
            Ctx::on(self.stream.cast())
                .with_cublas(self.cublas.cast())
                .with_slabs(self.slabs)
        }
    }

    pub fn zeros(&mut self, bytes: usize) -> u64 {
        let _legacy = legacy();
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

    pub fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        let bytes = core::mem::size_of_val(values);
        let at = self.zeros(bytes.max(1));
        if bytes > 0 {
            let _legacy = legacy();
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
        let _legacy = legacy();
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
        let _legacy = legacy();
        unsafe {
            rt::cudaStreamSynchronize(self.stream);
            for at in self.device.drain(..) {
                rt::cudaFree(at);
            }
            blas::cublasDestroy_v2(self.cublas);
            rt::cudaStreamDestroy(self.stream);
        }
    }
}

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

pub struct Lcg(u64);

impl Lcg {
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

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

pub const TOLERANCE: f32 = 3.0e-2;

/// The registry's tuning with the tcgen05 selects and GLU epilogues on, as
/// B200's profile has them.
pub const TC5: kernels_cuda::tuning::Tuning = kernels_cuda::tuning::Tuning {
    moe_tc5: true,
    gemm_tc5_glu_from: 64,
    ..kernels_cuda::tuning::Tuning::DEFAULT
};

/// The routed-MoE decisions B200's profile (`engine-cuda`'s `sm100.toml`)
/// sets, as a fire there runs them: a select's arm turns on these, and a
/// case run only at [`Tuning::DEFAULT`] or [`TC5`] misses the arm the card
/// takes. `engine-cuda`'s `the_b200_moe_decisions_are_what_the_kernel_tests_run`
/// holds the profile to them.
pub const SM100: kernels_cuda::tuning::Tuning = kernels_cuda::tuning::Tuning {
    moe_tc5: true,
    moe_tc5_from: 16,
    gemm_tc5_glu_from: 65_536,
    mxfp4_mma_from: 28,
    mxfp4_mma_until: 1024,
    mxfp4_cvt: true,
    gemm_nvfp4_k128: true,
    moe_w4a4_fused_from: 1,
    moe_w4a8_fused_from: 512,
    nvfp4_mma_from: 32,
    nvfp4_mma_until: 8192,
    ..kernels_cuda::tuning::Tuning::DEFAULT
};

/// Tests run on threads of one process, and a copy, set or free on the
/// legacy stream syncs with every blocking stream, invalidating whatever
/// capture another thread has open on one. Such calls hold this shared, a
/// capture holds it alone.
static CAPTURE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Held around a call on the legacy stream; see [`CAPTURE`].
pub fn legacy() -> std::sync::RwLockReadGuard<'static, ()> {
    CAPTURE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Held around a stream capture; see [`CAPTURE`].
pub fn capturing() -> std::sync::RwLockWriteGuard<'static, ()> {
    CAPTURE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The graph `body` fires on `ctx`'s stream, captured and instantiated.
pub fn captured(
    ctx: &Ctx,
    body: impl FnOnce(),
) -> kernels_cuda::cudarc::runtime::sys::cudaGraphExec_t {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let ok = |code: rt::cudaError, call: &str| {
        assert_eq!(
            code,
            rt::cudaError::cudaSuccess,
            "`{call}` answered {code:?}"
        );
    };
    let stream = ctx.stream();
    let _alone = capturing();
    unsafe {
        ok(
            rt::cudaStreamBeginCapture(
                stream.cast(),
                rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        );
        body();
        let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
        ok(
            rt::cudaStreamEndCapture(stream.cast(), &raw mut graph),
            "cudaStreamEndCapture",
        );
        let mut exec: rt::cudaGraphExec_t = core::ptr::null_mut();
        ok(
            rt::cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0),
            "cudaGraphInstantiateWithFlags",
        );
        rt::cudaGraphDestroy(graph);
        exec
    }
}

/// Launches a [`captured`] graph on `ctx`'s stream.
pub fn replay(ctx: &Ctx, exec: kernels_cuda::cudarc::runtime::sys::cudaGraphExec_t) {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let code = unsafe { rt::cudaGraphLaunch(exec, ctx.stream().cast()) };
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`cudaGraphLaunch` answered {code:?}"
    );
}

/// Destroys a [`captured`] graph.
pub fn forget(exec: kernels_cuda::cudarc::runtime::sys::cudaGraphExec_t) {
    unsafe {
        kernels_cuda::cudarc::runtime::sys::cudaGraphExecDestroy(exec);
    }
}

pub fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= TOLERANCE * want.abs().max(1.0)
}

pub mod spatial;
pub mod tc5;
