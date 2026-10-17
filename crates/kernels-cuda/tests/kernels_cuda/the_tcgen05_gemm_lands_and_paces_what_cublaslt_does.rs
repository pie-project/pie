#![cfg(feature = "cuda")]

//! The tcgen05 GEMMs beside cuBLASLt: the block-scaled kinds land its
//! products from the same scale bytes.

use core::ffi::c_void;

use crate::common::Gpu;
use kernels_cuda::cudarc::cublaslt::sys as lt;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::linear::tc5::{Elem, Operand, atom_bytes, gemm, supported};

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

/// The atom layout is cuBLASLt's own: the same scale bytes land the same
/// NVFP4 and MXFP8 products through its block-scaled matmul.
#[test]
fn the_block_scaled_kinds_land_what_cublaslt_lands_from_the_same_scales() {
    use crate::common::from_bf16;
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let stream: rt::cudaStream_t = ctx.stream().cast();
    let ws_bytes = 32 << 20;
    let ws = gpu.zeros(ws_bytes);
    // A decode's few rows, and a prefill's off the tile.
    for (m, n, k) in [(40, 256, 2048), (1000, 768, 512)] {
        for kind in [LtKind::Nvfp4, LtKind::Mxfp8] {
            let (elem, gran, per_byte, mask, sf) = match kind {
                LtKind::Nvfp4 => (Elem::E2m1, 16, 2, 0xff, (0x07, 0x30)),
                _ => (Elem::E4m3, 32, 1, 0xb7, (0x03, 0x7c)),
            };
            let w = gpu.up(&noise(n * k / per_byte, mask, 0, 21));
            let a = gpu.up(&noise(m * k / per_byte, mask, 0, 22));
            let sw = gpu.up(&noise(atom_bytes(n, k, gran), sf.0, sf.1, 23));
            let sa = gpu.up(&noise(atom_bytes(m, k, gran), sf.0, sf.1, 24));
            let (ours, theirs) = (gpu.zeros(m * n * 2), gpu.zeros(m * n * 2));
            let lt = unsafe { Lt::plan(kind, [m, n, k], sw, sa, ws, ws_bytes) }
                .unwrap_or_else(|why| panic!("cuBLASLt plans no {m}x{n}x{k}: {why}"));
            lt.run(w, a, theirs, stream);
            gemm(
                &ctx,
                Operand::scaled(a, elem, sa, gran as u32),
                Operand::scaled(w, elem, sw, gran as u32),
                ours,
                [m as u32, n as u32, k as u32],
                1.0,
            )
            .expect("tc5 fires");
            gpu.sync();
            let ours: Vec<u16> = gpu.down(ours, m * n);
            let theirs: Vec<u16> = gpu.down(theirs, m * n);
            for (i, (&o, &t)) in ours.iter().zip(&theirs).enumerate() {
                let (o, t) = (from_bf16(o), from_bf16(t));
                assert!(
                    (o - t).abs() <= t.abs() / 64.0 + 1e-3,
                    "{m}x{n}x{k}: y[{}][{}] = {o}, cuBLASLt {t}",
                    i / n,
                    i % n
                );
            }
        }
    }
}
