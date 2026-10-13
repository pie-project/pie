#![cfg(feature = "cuda")]

//! The tcgen05 GEMMs' rate beside cuBLASLt's at the shapes a decoder
//! layer fires, L2 flushed before every timed launch. Ignored: it is a
//! measurement, not a check. Run alone on an idle GPU:
//!
//!     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//!         the_tcgen05_gemm_reports -- --ignored --nocapture

use core::ffi::c_void;

use crate::common::Gpu;
use kernels_cuda::cudarc::cublaslt::sys as lt;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::linear::tc5::{Elem, Operand, atom_bytes, gemm, supported};

const FLUSH: usize = 512 << 20;
const RUNS: usize = 15;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Bytes whose every reading is finite: bf16 pairs, fp8 or fp4 codes,
/// and scale bytes, all of modest magnitude.
fn noise(bytes: usize, mask: u8, base: u8, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..bytes)
        .map(|_| (splitmix(&mut state) as u8 & mask) | base)
        .collect()
}

struct Timer {
    flush: u64,
    stream: rt::cudaStream_t,
}

impl Timer {
    /// The median microseconds of `RUNS` launches, each after an L2 flush.
    fn median(&self, mut launch: impl FnMut()) -> f64 {
        launch();
        let mut times = Vec::with_capacity(RUNS);
        unsafe {
            let (mut start, mut stop): (rt::cudaEvent_t, rt::cudaEvent_t) =
                (core::ptr::null_mut(), core::ptr::null_mut());
            rt::cudaEventCreate(&raw mut start);
            rt::cudaEventCreate(&raw mut stop);
            for _ in 0..RUNS {
                rt::cudaMemsetAsync(self.flush as *mut c_void, 0, FLUSH, self.stream);
                rt::cudaEventRecord(start, self.stream);
                launch();
                rt::cudaEventRecord(stop, self.stream);
                rt::cudaEventSynchronize(stop);
                let mut ms = 0f32;
                rt::cudaEventElapsedTime(&raw mut ms, start, stop);
                times.push(f64::from(ms) * 1e3);
            }
            rt::cudaEventDestroy(start);
            rt::cudaEventDestroy(stop);
        }
        times.sort_by(f64::total_cmp);
        times[RUNS / 2]
    }
}

/// One cuBLASLt matmul, `D[m][n] = act . w^T` in bf16, planned once.
struct Lt {
    handle: lt::cublasLtHandle_t,
    desc: lt::cublasLtMatmulDesc_t,
    layouts: [lt::cublasLtMatrixLayout_t; 3],
    algo: lt::cublasLtMatmulAlgo_t,
    ws: u64,
    ws_bytes: usize,
    alpha: f32,
}

#[derive(Clone, Copy)]
enum LtKind {
    Bf16,
    /// fp8 e4m3 with ue8m0 a 32 (MXFP8).
    Mxfp8,
    /// e2m1 with ue4m3 a 16.
    Nvfp4,
}

fn ok(status: lt::cublasStatus_t, what: &str) -> Result<(), String> {
    if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} answered {status:?}"))
    }
}

impl Lt {
    #[allow(clippy::too_many_arguments)]
    unsafe fn plan(
        kind: LtKind,
        [m, n, k]: [usize; 3],
        w_scales: u64,
        act_scales: u64,
        ws: u64,
        ws_bytes: usize,
    ) -> Result<Self, String> {
        use lt::cublasLtMatmulDescAttributes_t as A;
        use lt::cudaDataType_t as D;
        unsafe {
            let mut handle = core::ptr::null_mut();
            ok(lt::cublasLtCreate(&raw mut handle), "cublasLtCreate")?;
            let mut desc = core::ptr::null_mut();
            ok(
                lt::cublasLtMatmulDescCreate(
                    &raw mut desc,
                    lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    D::CUDA_R_32F,
                ),
                "cublasLtMatmulDescCreate",
            )?;
            let set = |attr: A, value: *const c_void, size: usize| {
                ok(
                    lt::cublasLtMatmulDescSetAttribute(desc, attr, value, size),
                    "cublasLtMatmulDescSetAttribute",
                )
            };
            let t = kernels_cuda::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
            let nn = kernels_cuda::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
            set(A::CUBLASLT_MATMUL_DESC_TRANSA, (&raw const t).cast(), 4)?;
            set(A::CUBLASLT_MATMUL_DESC_TRANSB, (&raw const nn).cast(), 4)?;
            let (ty, mode) = match kind {
                LtKind::Bf16 => (D::CUDA_R_16BF, None),
                LtKind::Mxfp8 => (
                    D::CUDA_R_8F_E4M3,
                    Some(lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0),
                ),
                LtKind::Nvfp4 => (
                    D::CUDA_R_4F_E2M1,
                    Some(lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3),
                ),
            };
            if let Some(mode) = mode {
                let mode = mode as i32;
                set(
                    A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    (&raw const mode).cast(),
                    4,
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    (&raw const mode).cast(),
                    4,
                )?;
                let (a, b) = (w_scales as *const c_void, act_scales as *const c_void);
                set(
                    A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    (&raw const a).cast(),
                    8,
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    (&raw const b).cast(),
                    8,
                )?;
            }
            let layout = |ty: D, rows: usize, cols: usize| -> Result<_, String> {
                let mut l = core::ptr::null_mut();
                ok(
                    lt::cublasLtMatrixLayoutCreate(
                        &raw mut l,
                        ty,
                        rows as u64,
                        cols as u64,
                        rows as i64,
                    ),
                    "cublasLtMatrixLayoutCreate",
                )?;
                Ok(l)
            };
            let layouts = [
                layout(ty, k, n)?,
                layout(ty, k, m)?,
                layout(D::CUDA_R_16BF, n, m)?,
            ];
            let mut pref = core::ptr::null_mut();
            ok(
                lt::cublasLtMatmulPreferenceCreate(&raw mut pref),
                "preference",
            )?;
            let bytes = ws_bytes as u64;
            ok(
                lt::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&raw const bytes).cast(),
                    8,
                ),
                "workspace preference",
            )?;
            let mut result: lt::cublasLtMatmulHeuristicResult_t = core::mem::zeroed();
            let mut found = 0i32;
            ok(
                lt::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    layouts[0],
                    layouts[1],
                    layouts[2],
                    layouts[2],
                    pref,
                    1,
                    &raw mut result,
                    &raw mut found,
                ),
                "cublasLtMatmulAlgoGetHeuristic",
            )?;
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if found == 0 {
                return Err("no algorithm".into());
            }
            Ok(Self {
                handle,
                desc,
                layouts,
                algo: result.algo,
                ws,
                ws_bytes,
                alpha: 1.0,
            })
        }
    }

    fn run(&self, w: u64, act: u64, y: u64, stream: rt::cudaStream_t) {
        let beta = 0f32;
        let status = unsafe {
            lt::cublasLtMatmul(
                self.handle,
                self.desc,
                (&raw const self.alpha).cast(),
                w as *const c_void,
                self.layouts[0],
                act as *const c_void,
                self.layouts[1],
                (&raw const beta).cast(),
                y as *const c_void,
                self.layouts[2],
                y as *mut c_void,
                self.layouts[2],
                &raw const self.algo,
                self.ws as *mut c_void,
                self.ws_bytes,
                stream.cast(),
            )
        };
        assert_eq!(
            status,
            lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cublasLtMatmul"
        );
    }
}

impl Drop for Lt {
    fn drop(&mut self) {
        unsafe {
            for l in self.layouts {
                lt::cublasLtMatrixLayoutDestroy(l);
            }
            lt::cublasLtMatmulDescDestroy(self.desc);
            lt::cublasLtDestroy(self.handle);
        }
    }
}

fn tflops(m: usize, n: usize, k: usize, us: f64) -> f64 {
    2.0 * (m * n * k) as f64 / us * 1e-6
}

#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_tcgen05_gemm_reports_its_rate_beside_cublaslt() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let stream: rt::cudaStream_t = ctx.stream().cast();
    let timer = Timer {
        flush: gpu.zeros(FLUSH),
        stream,
    };
    let ws_bytes = 64 << 20;
    let ws = gpu.zeros(ws_bytes);

    println!(
        "| M | N | K | cuBLASLt bf16 | tc5 bf16 | cuBLASLt mxfp8 | tc5 fp8 per-tensor | tc5 fp8 1x128 ue8m0 | tc5 fp8 mx32 | cuBLASLt nvfp4 | tc5 nvfp4 | tc5 w4a8 |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (n, k) in [(4096, 4096), (7168, 2048), (2048, 7168), (14336, 4096)] {
        let w16 = gpu.up(&noise(n * k * 2, 0xbf, 0x00, 1));
        let w8 = gpu.up(&noise(n * k, 0xb7, 0x00, 2));
        let w4 = gpu.up(&noise(n * k / 2, 0xff, 0x00, 3));
        let sw128 = gpu.up(&noise(atom_bytes(n, k, 128), 0x03, 0x7c, 4));
        let sw32 = gpu.up(&noise(atom_bytes(n, k, 32), 0x03, 0x7c, 5));
        let sw16 = gpu.up(&noise(atom_bytes(n, k, 16), 0x07, 0x30, 6));
        for m in [1usize, 16, 128, 1024, 8192] {
            let a16 = gpu.up(&noise(m * k * 2, 0xbf, 0x00, 7));
            let a8 = gpu.up(&noise(m * k, 0xb7, 0x00, 8));
            let a4 = gpu.up(&noise(m * k / 2, 0xff, 0x00, 9));
            let sa128 = gpu.up(&noise(atom_bytes(m, k, 128), 0x03, 0x7c, 10));
            let sa32 = gpu.up(&noise(atom_bytes(m, k, 32), 0x03, 0x7c, 11));
            let sa16 = gpu.up(&noise(atom_bytes(m, k, 16), 0x07, 0x30, 12));
            let y = gpu.zeros(m * n * 2);
            let shape = [m as u32, n as u32, k as u32];

            let lt_rate = |kind: LtKind, w: u64, a: u64, sw: u64, sa: u64| -> String {
                match unsafe { Lt::plan(kind, [m, n, k], sw, sa, ws, ws_bytes) } {
                    Ok(plan) => {
                        let us = timer.median(|| plan.run(w, a, y, stream));
                        format!("{:.0} ({us:.1}us)", tflops(m, n, k, us))
                    }
                    Err(why) => format!("n/a: {why}"),
                }
            };
            let tc_rate = |a: Operand, w: Operand, alpha: f32| -> String {
                match gemm(&ctx, a, w, y, shape, alpha) {
                    Ok(()) => {
                        let us = timer.median(|| {
                            gemm(&ctx, a, w, y, shape, alpha).expect("fires again");
                        });
                        format!("{:.0} ({us:.1}us)", tflops(m, n, k, us))
                    }
                    Err(why) => format!("n/a: {why}"),
                }
            };
            let row = [
                lt_rate(LtKind::Bf16, w16, a16, 0, 0),
                tc_rate(
                    Operand::plain(a16, Elem::Bf16),
                    Operand::plain(w16, Elem::Bf16),
                    1.0,
                ),
                lt_rate(LtKind::Mxfp8, w8, a8, sw32, sa32),
                tc_rate(
                    Operand::plain(a8, Elem::E4m3),
                    Operand::plain(w8, Elem::E4m3),
                    1.0,
                ),
                tc_rate(
                    Operand::scaled(a8, Elem::E4m3, sa128, 128),
                    Operand::scaled(w8, Elem::E4m3, sw128, 128),
                    1.0,
                ),
                tc_rate(
                    Operand::scaled(a8, Elem::E4m3, sa32, 32),
                    Operand::scaled(w8, Elem::E4m3, sw32, 32),
                    1.0,
                ),
                lt_rate(LtKind::Nvfp4, w4, a4, sw16, sa16),
                tc_rate(
                    Operand::scaled(a4, Elem::E2m1, sa16, 16),
                    Operand::scaled(w4, Elem::E2m1, sw16, 16),
                    1.0,
                ),
                tc_rate(
                    Operand::scaled(a8, Elem::E4m3, sa128, 128),
                    Operand::scaled(w4, Elem::E2m1, sw32, 32),
                    1.0,
                ),
            ];
            println!("| {m} | {n} | {k} | {} |", row.join(" | "));
        }
    }
}
