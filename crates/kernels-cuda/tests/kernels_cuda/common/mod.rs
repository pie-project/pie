#![allow(dead_code)]
#![allow(unsafe_code)]

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
    ordinal: i32,
    stream: rt::cudaStream_t,
    slabs: Slabs,
    device: Vec<*mut c_void>,
    edges: Vec<Edge>,
    cublas: blas::cublasHandle_t,
    /// Dropped after `Drop` has freed everything above.
    _shared: Shared,
}

/// A mapping this `Gpu` made: `mapped` bytes at the front of a `reserved`
/// reservation. An edge maps one granule of two, so a read past the mapped
/// end faults instead of landing on a neighbour's bytes; a shared mapping
/// maps all it reserves.
struct Edge {
    base: u64,
    mapped: usize,
    reserved: usize,
    handle: u64,
}

/// The devices this process sees.
pub fn device_count() -> i32 {
    let mut count = 0;
    // SAFETY: `count` is a local out-pointer.
    check(
        unsafe { rt::cudaGetDeviceCount(&raw mut count) },
        "cudaGetDeviceCount",
    );
    count
}

pub fn arm_cache() {
    kernels_cuda::disk::install(Some(std::path::Path::new(concat!(
        env!("CARGO_TARGET_TMPDIR"),
        "/kernel-cache"
    ))));
}

impl Gpu {
    pub fn open() -> Self {
        Self::open_on(0)
    }

    /// A `Gpu` on device `ordinal`. Every call on it binds this thread to
    /// that device first, so `Gpu`s on several devices share a thread; a
    /// fire on its [`Gpu::ctx`] goes to the device last bound.
    pub fn open_on(ordinal: i32) -> Self {
        arm_cache();
        let shared = share();
        // SAFETY: every out-pointer is a local; the handle is bound to the
        // stream just created, and both, with the slabs attached to it, live
        // in the `Gpu` until its `Drop` synchronizes and frees them.
        unsafe {
            check(rt::cudaSetDevice(ordinal), "cudaSetDevice");
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
                ordinal,
                stream,
                slabs,
                device: Vec::new(),
                edges: Vec::new(),
                cublas,
                _shared: shared,
            }
        }
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Binds this thread to the `Gpu`'s device.
    pub fn bind(&self) {
        // SAFETY: an ordinal `open_on` already bound once.
        check(unsafe { rt::cudaSetDevice(self.ordinal) }, "cudaSetDevice");
    }

    pub fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire in a test, and `Gpu`'s drop
        // synchronizes before destroying it.
        let ctx = unsafe {
            Ctx::on(self.stream.cast())
                .with_cublas(self.cublas.cast())
                .with_slabs(self.slabs)
        };
        // `PIE_TEST_PDL=1` runs every test on this context with sm100's
        // launch decisions on (programmatic dependents, split tickets).
        if std::env::var("PIE_TEST_PDL").is_ok_and(|v| v == "1") {
            let tuning = ctx.tuning();
            ctx.with_tuning(kernels_cuda::tuning::Tuning {
                pdl: true,
                gemm_tc5_split_tickets: 65_536,
                gemm_rows_m: 8,
                ssm_kda_chunked: true,
                ..tuning
            })
        } else {
            ctx
        }
    }

    pub fn zeros(&mut self, bytes: usize) -> u64 {
        self.bind();
        let _legacy = legacy();
        // SAFETY: `at` is a local out-pointer, and the memset covers exactly
        // the `bytes.max(1)` just allocated; `Drop` frees it.
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

    /// `bytes` of device memory at `at` (this `Gpu`'s, or a scratch slab's)
    /// set to `byte`, once the stream's work before is done.
    pub fn fill(&self, at: u64, byte: u8, bytes: usize) {
        self.sync();
        let _legacy = legacy();
        // SAFETY: the caller names `bytes` of live device memory at `at`.
        unsafe {
            check(
                rt::cudaMemset(at as *mut c_void, i32::from(byte), bytes),
                "cudaMemset",
            );
        }
    }

    pub fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        let bytes = core::mem::size_of_val(values);
        let at = self.zeros(bytes.max(1));
        if bytes > 0 {
            let _legacy = legacy();
            // SAFETY: `at` was just allocated with `bytes` bytes and
            // `values` is borrowed across the synchronous copy.
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

    /// `values` uploaded so their last byte is the last mapped byte of the
    /// device's memory there: a kernel that reads even one element past
    /// them faults (an illegal address) rather than reading whatever an
    /// allocator happened to place next.
    pub fn up_at_edge<T: Copy>(&mut self, values: &[T]) -> u64 {
        use kernels_cuda::cudarc::driver::sys as dr;
        let bytes = core::mem::size_of_val(values);
        let drv = |code: dr::CUresult, call: &str| {
            assert_eq!(
                code,
                dr::CUresult::CUDA_SUCCESS,
                "`{call}` answered {code:?}"
            );
        };
        // SAFETY: both driver structs are plain C data, valid zeroed. Every
        // out-pointer is a local; the mapping covers the first of the two
        // reserved granules, which the memset fills, and `at` puts the
        // `bytes <= granule` copy at that granule's end. `Drop` unmaps and
        // frees each piece.
        unsafe {
            let mut prop: dr::CUmemAllocationProp = core::mem::zeroed();
            prop.type_ = dr::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = self.ordinal;
            let mut granule: usize = 0;
            drv(
                dr::cuMemGetAllocationGranularity(
                    &raw mut granule,
                    &raw const prop,
                    dr::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "cuMemGetAllocationGranularity",
            );
            assert!(
                bytes <= granule,
                "{bytes} bytes do not fit one {granule}-byte granule"
            );
            let mut handle: dr::CUmemGenericAllocationHandle = 0;
            drv(
                dr::cuMemCreate(&raw mut handle, granule, &raw const prop, 0),
                "cuMemCreate",
            );
            let mut base: dr::CUdeviceptr = 0;
            drv(
                dr::cuMemAddressReserve(&raw mut base, 2 * granule, granule, 0, 0),
                "cuMemAddressReserve",
            );
            drv(dr::cuMemMap(base, granule, 0, handle, 0), "cuMemMap");
            let mut desc: dr::CUmemAccessDesc = core::mem::zeroed();
            desc.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            desc.location.id = self.ordinal;
            desc.flags = dr::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            drv(
                dr::cuMemSetAccess(base, granule, &raw const desc, 1),
                "cuMemSetAccess",
            );
            self.edges.push(Edge {
                base,
                mapped: granule,
                reserved: 2 * granule,
                handle,
            });
            let at = base + (granule - bytes) as u64;
            self.bind();
            let _legacy = legacy();
            check(
                rt::cudaMemset(base as *mut c_void, 0, granule),
                "cudaMemset",
            );
            if bytes > 0 {
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
            at
        }
    }

    pub fn down<T: Copy + Default>(&self, at: u64, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        self.bind();
        let _legacy = legacy();
        // SAFETY: `out` holds `count` elements and the copy is synchronous;
        // `at` is an allocation of this `Gpu` at least that long, which
        // every test passes.
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

    /// `bytes` of zeros on this device that every device in `readers` loads
    /// and stores, as the engine's EP buffers are (`elastic::shared`): one
    /// mapping whose access is granted to each reader, without turning on
    /// peer access for the whole device.
    pub fn shared(&mut self, bytes: usize, readers: &[i32]) -> u64 {
        use kernels_cuda::cudarc::driver::sys as dr;
        let drv = |code: dr::CUresult, call: &str| {
            assert_eq!(
                code,
                dr::CUresult::CUDA_SUCCESS,
                "`{call}` answered {code:?}"
            );
        };
        let mut readers = readers.to_vec();
        readers.push(self.ordinal);
        readers.sort_unstable();
        readers.dedup();
        // SAFETY: the driver structs are plain C data, valid zeroed. Every
        // out-pointer is a local; the mapping covers the whole reservation,
        // which the memset fills. `Drop` unmaps and frees each piece.
        unsafe {
            let mut prop: dr::CUmemAllocationProp = core::mem::zeroed();
            prop.type_ = dr::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = self.ordinal;
            let mut granule: usize = 0;
            drv(
                dr::cuMemGetAllocationGranularity(
                    &raw mut granule,
                    &raw const prop,
                    dr::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "cuMemGetAllocationGranularity",
            );
            let size = bytes.max(1).next_multiple_of(granule);
            let mut handle: dr::CUmemGenericAllocationHandle = 0;
            drv(
                dr::cuMemCreate(&raw mut handle, size, &raw const prop, 0),
                "cuMemCreate",
            );
            let mut base: dr::CUdeviceptr = 0;
            drv(
                dr::cuMemAddressReserve(&raw mut base, size, granule, 0, 0),
                "cuMemAddressReserve",
            );
            drv(dr::cuMemMap(base, size, 0, handle, 0), "cuMemMap");
            let descs: Vec<dr::CUmemAccessDesc> = readers
                .iter()
                .map(|&d| {
                    let mut desc: dr::CUmemAccessDesc = core::mem::zeroed();
                    desc.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                    desc.location.id = d;
                    desc.flags = dr::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    desc
                })
                .collect();
            drv(
                dr::cuMemSetAccess(base, size, descs.as_ptr(), descs.len()),
                "cuMemSetAccess",
            );
            self.edges.push(Edge {
                base,
                mapped: size,
                reserved: size,
                handle,
            });
            self.bind();
            let _legacy = legacy();
            check(rt::cudaMemset(base as *mut c_void, 0, size), "cudaMemset");
            base
        }
    }

    /// `values` in a [`Gpu::shared`] mapping `readers` reach.
    pub fn up_shared<T: Copy>(&mut self, values: &[T], readers: &[i32]) -> u64 {
        let at = self.shared(core::mem::size_of_val(values), readers);
        host::upload_to(at, values);
        at
    }

    pub fn sync(&self) {
        // SAFETY: the stream `open` created, live until `Drop`.
        unsafe {
            check(
                rt::cudaStreamSynchronize(self.stream),
                "cudaStreamSynchronize",
            );
        }
    }

    /// The device's microseconds a `fire` on this stream takes, the best of
    /// 25 runs of `reps` (events either side of each run).
    pub fn time(&self, reps: u32, mut fire: impl FnMut()) -> f64 {
        let mut best = f64::INFINITY;
        // SAFETY: the events are locals created and destroyed here, recorded
        // on the stream `open` created.
        unsafe {
            let (mut a, mut b): (rt::cudaEvent_t, rt::cudaEvent_t) =
                (core::ptr::null_mut(), core::ptr::null_mut());
            check(rt::cudaEventCreate(&raw mut a), "cudaEventCreate");
            check(rt::cudaEventCreate(&raw mut b), "cudaEventCreate");
            for _ in 0..25 {
                check(rt::cudaEventRecord(a, self.stream), "cudaEventRecord");
                for _ in 0..reps {
                    fire();
                }
                check(rt::cudaEventRecord(b, self.stream), "cudaEventRecord");
                check(rt::cudaEventSynchronize(b), "cudaEventSynchronize");
                let mut ms = 0f32;
                check(
                    rt::cudaEventElapsedTime(&raw mut ms, a, b),
                    "cudaEventElapsedTime",
                );
                best = best.min(f64::from(ms) * 1e3 / f64::from(reps));
            }
            check(rt::cudaEventDestroy(a), "cudaEventDestroy");
            check(rt::cudaEventDestroy(b), "cudaEventDestroy");
        }
        best
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.bind();
        let _legacy = legacy();
        // SAFETY: the stream is drained first, so nothing still reads the
        // memory; each allocation, mapping, the handle and the stream were
        // made by this `Gpu` and are freed once, here.
        unsafe {
            rt::cudaStreamSynchronize(self.stream);
            for at in self.device.drain(..) {
                rt::cudaFree(at);
            }
            for edge in self.edges.drain(..) {
                use kernels_cuda::cudarc::driver::sys as dr;
                let _ = dr::cuMemUnmap(edge.base, edge.mapped);
                let _ = dr::cuMemRelease(edge.handle);
                let _ = dr::cuMemAddressFree(edge.base, edge.reserved);
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

/// The registry's tuning with the tcgen05 selects on, as
/// B200's profile has them.
pub const TC5: kernels_cuda::tuning::Tuning = kernels_cuda::tuning::Tuning {
    moe_tc5: true,
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
    mxfp4_mma_from: 28,
    mxfp4_mma_until: 1024,
    mxfp4_cvt: true,
    mxfp4_a8_from: 1,
    mxfp4_cvt_ksplit_until: 6,
    mxfp4_a8_until: 48,
    moe_w4a4_fused_from: 1,
    moe_w4a4_fused_layout: true,
    moe_even_waves_from: 3,
    moe_routes_sized: true,
    moe_gather_ahead_until: 128,
    moe_w4a8_fused_from: 512,
    moe_mega_until: 2048,
    moe_mega_bf16_from: 64,
    attn_res_fused_norm: true,
    mla_expanded_from: 256,
    gemm_fp8_gemv_until: 4,
    nvfp4_mma_from: 32,
    nvfp4_mma_until: 8192,
    gate_tc5_until: 256,
    ..kernels_cuda::tuning::Tuning::DEFAULT
};

/// Tests run on threads of one process, and a collective's ranks spin on
/// the device for one another: anything that waits on the device meanwhile
/// (a module load, an allocation or free) from a test beside them waits on
/// that spin while it holds back the next rank's launch, until the spin
/// traps and the process's context is lost. A collective test holds the
/// devices [`alone`]; every [`Gpu`] holds them shared for its life, as
/// often over as the thread opens them.
struct Gate {
    users: usize,
    alone: bool,
    waiting: usize,
}

static GATE: std::sync::Mutex<Gate> = std::sync::Mutex::new(Gate {
    users: 0,
    alone: false,
    waiting: 0,
});
static OPENED: std::sync::Condvar = std::sync::Condvar::new();

thread_local! {
    /// The holds this thread has on the gate: its `Gpu`s, and its `Alone`.
    static HOLDS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// A hold on the devices beside other tests; not `Send`, as the count is
/// the thread's.
pub struct Shared(core::marker::PhantomData<*const ()>);

/// The devices with no other test's `Gpu` open; see [`Gate`].
pub struct Alone(core::marker::PhantomData<*const ()>);

fn gate() -> std::sync::MutexGuard<'static, Gate> {
    GATE.lock().expect("gate poisoned")
}

fn share() -> Shared {
    if HOLDS.get() == 0 {
        let mut g = gate();
        while g.alone || g.waiting > 0 {
            g = OPENED.wait(g).expect("gate poisoned");
        }
        g.users += 1;
    }
    HOLDS.set(HOLDS.get() + 1);
    Shared(core::marker::PhantomData)
}

impl Drop for Shared {
    fn drop(&mut self) {
        HOLDS.set(HOLDS.get() - 1);
        if HOLDS.get() == 0 {
            gate().users -= 1;
            OPENED.notify_all();
        }
    }
}

/// Waits until no other test holds a `Gpu`, then keeps every other test
/// from opening one until dropped. Taken before the test opens its own.
pub fn alone() -> Alone {
    assert_eq!(
        HOLDS.get(),
        0,
        "a test takes the devices alone before it opens a Gpu"
    );
    let mut g = gate();
    g.waiting += 1;
    while g.alone || g.users > 0 {
        g = OPENED.wait(g).expect("gate poisoned");
    }
    g.waiting -= 1;
    g.alone = true;
    HOLDS.set(1);
    Alone(core::marker::PhantomData)
}

impl Drop for Alone {
    fn drop(&mut self) {
        HOLDS.set(HOLDS.get() - 1);
        let mut g = gate();
        g.alone = false;
        // A `Gpu` of this test that outlives the guard holds on, shared.
        if HOLDS.get() > 0 {
            g.users += 1;
        }
        drop(g);
        OPENED.notify_all();
    }
}

/// Tests run on threads of one process, and a copy, set or free on the
/// legacy stream syncs with every blocking stream, invalidating whatever
/// capture another thread has open on one. Such calls hold this shared, a
/// capture holds it alone.
static CAPTURE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Held around a call on the legacy stream; see [`CAPTURE`].
pub fn legacy() -> std::sync::RwLockReadGuard<'static, ()> {
    CAPTURE.read().expect("capture lock poisoned")
}

/// Held around a stream capture; see [`CAPTURE`].
pub fn capturing() -> std::sync::RwLockWriteGuard<'static, ()> {
    CAPTURE.write().expect("capture lock poisoned")
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
    // SAFETY: `ctx`'s stream is live by its contract and no other thread
    // touches the legacy stream while `_alone` is held. `graph` and `exec`
    // are local out-pointers; the graph is freed once its exec is made.
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
    // SAFETY: `exec` is a graph [`captured`] made and [`forget`] has not
    // yet freed, as every test passes; the stream is `ctx`'s.
    let code = unsafe { rt::cudaGraphLaunch(exec, ctx.stream().cast()) };
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`cudaGraphLaunch` answered {code:?}"
    );
}

/// Destroys a [`captured`] graph.
pub fn forget(exec: kernels_cuda::cudarc::runtime::sys::cudaGraphExec_t) {
    // SAFETY: `exec` is a graph [`captured`] made, which each test forgets
    // once, after its last replay.
    unsafe {
        kernels_cuda::cudarc::runtime::sys::cudaGraphExecDestroy(exec);
    }
}

pub fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= TOLERANCE * want.abs().max(1.0)
}

pub mod attn_ref;
pub mod host;
pub mod linear_ref;
pub mod spatial;
pub mod tc5;
