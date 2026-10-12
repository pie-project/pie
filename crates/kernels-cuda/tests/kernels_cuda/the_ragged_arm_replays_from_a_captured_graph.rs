#![cfg(feature = "cuda")]

use core::ffi::c_void;

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

fn upload(at: u64, values: &[u16]) {
    unsafe {
        check(
            rt::cudaMemcpy(
                at as *mut c_void,
                values.as_ptr().cast(),
                core::mem::size_of_val(values),
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy H2D",
        );
    }
}

fn upload_u32(at: u64, values: &[u32]) {
    unsafe {
        check(
            rt::cudaMemcpy(
                at as *mut c_void,
                values.as_ptr().cast(),
                core::mem::size_of_val(values),
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy H2D",
        );
    }
}

fn upload_i32(at: u64, values: &[i32]) {
    unsafe {
        check(
            rt::cudaMemcpy(
                at as *mut c_void,
                values.as_ptr().cast(),
                core::mem::size_of_val(values),
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy H2D",
        );
    }
}

#[test]
fn the_ragged_arm_replays_from_a_captured_graph() {
    let hd = 128u32;
    let (q_heads, kv_heads) = (4u32, 2u32);
    let sizes = [200u32, 130, 300];
    let table: Vec<i32> = {
        let mut out = vec![0i32];
        for &n in &sizes {
            out.push(out.last().unwrap() + n as i32);
        }
        out
    };
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let sm_scale = 1.0 / (hd as f32).sqrt();

    let mut gpu = Gpu::open();
    let q_at = gpu.zeros(rows * qw * 2);
    let k_at = gpu.zeros(rows * kw * 2);
    let v_at = gpu.zeros(rows * kw * 2);
    let table_at = gpu.up(&table);
    let o_at = gpu.zeros(rows * qw * 2);
    let o_eager = gpu.zeros(rows * qw * 2);
    let win_at = gpu.up(&[rows as u32, 0u32, sizes.len() as u32, 0u32]);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let fire = |ctx: &kernels_cuda::jit::Ctx, o_ptr: u64| {
        let mut o = Tensor::new(o_ptr, rows as u32, qw as u32, Dtype::Bf16);
        attn_ragged::ragged(
            ctx,
            q,
            k,
            v,
            groups,
            groups,
            hd,
            sm_scale,
            RaggedMask::None,
            &mut o,
        )
    };

    let mut lcg = Lcg::seeded(0x9a);
    let (q0, _) = lcg.row(rows * qw);
    let (k0, _) = lcg.row(rows * kw);
    let (v0, _) = lcg.row(rows * kw);
    upload(q_at, &q0);
    upload(k_at, &k0);
    upload(v_at, &v0);
    ctx.arm_stage(win_at);
    fire(&ctx, o_at).expect("the warm fire");
    gpu.sync();

    let stream = ctx.stream();
    let exec = unsafe {
        check(
            rt::cudaStreamBeginCapture(
                stream.cast(),
                rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        );
        let fired = fire(&ctx, o_at);
        let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
        check(
            rt::cudaStreamEndCapture(stream.cast(), &raw mut graph),
            "cudaStreamEndCapture",
        );
        fired.expect("the entry fires under capture (the slab was warmed)");
        let mut exec: rt::cudaGraphExec_t = core::ptr::null_mut();
        check(
            rt::cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0),
            "cudaGraphInstantiateWithFlags",
        );
        exec
    };

    let (q1, _) = lcg.row(rows * qw);
    let (k1, _) = lcg.row(rows * kw);
    let (v1, _) = lcg.row(rows * kw);
    upload(q_at, &q1);
    upload(k_at, &k1);
    upload(v_at, &v1);
    ctx.disarm_stage();
    fire(&ctx, o_eager).expect("the eager twin");
    unsafe {
        check(rt::cudaGraphLaunch(exec, stream.cast()), "cudaGraphLaunch");
    }
    gpu.sync();
    let want: Vec<u16> = gpu.down(o_eager, rows * qw);
    let got: Vec<u16> = gpu.down(o_at, rows * qw);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let (g, w) = (from_bf16(g), from_bf16(w));
        assert!(
            (g - w).abs() <= TOLERANCE * w.abs().max(1.0),
            "replay one, element {i}: {g} against {w}"
        );
    }

    let (q2, _) = lcg.row(rows * qw);
    let (k2, _) = lcg.row(rows * kw);
    let (v2, _) = lcg.row(rows * kw);
    upload(q_at, &q2);
    upload(k_at, &k2);
    upload(v_at, &v2);
    upload_u32(win_at, &[rows as u32, 0, 2, 1]);
    let mut emptied = table.clone();
    emptied[0] = emptied[1];
    upload_i32(table_at, &emptied);
    fire(&ctx, o_eager).expect("the eager twin over the rewritten table");
    unsafe {
        check(rt::cudaGraphLaunch(exec, stream.cast()), "cudaGraphLaunch");
    }
    gpu.sync();
    let want: Vec<u16> = gpu.down(o_eager, rows * qw);
    let got2: Vec<u16> = gpu.down(o_at, rows * qw);
    let split = table[1] as usize * qw;
    assert_eq!(
        got2[..split],
        got[..split],
        "group 0 sits outside the seat's window and must keep replay one's bytes"
    );
    for (i, (&g, &w)) in got2[split..].iter().zip(&want[split..]).enumerate() {
        let (g, w) = (from_bf16(g), from_bf16(w));
        assert!(
            (g - w).abs() <= TOLERANCE * w.abs().max(1.0),
            "replay two, element {}: {g} against {w}",
            split + i
        );
    }
    unsafe {
        rt::cudaGraphExecDestroy(exec);
    }
}
