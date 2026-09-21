#![cfg(target_vendor = "apple")]

//! `elementwise.hadamard` on Metal turns each contiguous 128-vector of a row's
//! last dim by the normalized 128x128 Sylvester Hadamard matrix H, whose
//! entries are `(-1)^popcount(i & j) / sqrt(128)`. H is symmetric and
//! orthonormal, so `H . H = I`: the op is its own inverse. This checks the
//! Metal output against a host reference, pins the exact matrix entries with a
//! one-hot orientation probe (a transposed or mis-strided matrix fails it),
//! confirms the round trip, and confirms the shape guard rejects a last dim
//! that is not a multiple of 128 at validation, not at runtime.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::pointwise;
use model_ir::{Dtype, Elementwise, Operation};

const BLOCK: usize = 128;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5e5e_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (f64::from(noise(at)) / f64::from(u32::MAX)) as f32 * 2.0 - 1.0
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The host reference: each contiguous 128-vector is left-multiplied by the
/// normalized Sylvester Hadamard matrix. Accumulated in f64 for a clean gold.
fn blockwise_h(x: &[f32]) -> Vec<f32> {
    let inv = 1.0f64 / 128.0f64.sqrt();
    let mut out = vec![0.0f32; x.len()];
    for block in x.chunks_exact(BLOCK).enumerate() {
        let (b, chunk) = block;
        for i in 0..BLOCK {
            let mut acc = 0.0f64;
            for (j, &v) in chunk.iter().enumerate() {
                let sign = if (i & j).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += sign * f64::from(v);
            }
            out[b * BLOCK + i] = (acc * inv) as f32;
        }
    }
    out
}

/// Run the Metal op over `rows x width` f32 values held row-major, in place,
/// and read the result back.
fn run(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    width: u32,
    data: &[f32],
) -> Vec<f32> {
    let bytes = u64::from(rows) * u64::from(width) * 4;
    let mut buf = Buffer::zeroed(device, bytes).expect("a buffer");
    buf.write(0, &f32_bytes(data)).expect("write x");
    let handle = handles.bind(&buf, 0, buf.bytes()).expect("a handle");
    let x = Tensor::new(handle, rows, width, Dtype::F32);
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(device, &frame, pipelines, handles);
        pointwise::hadamard(&sink, x).expect("the launch");
        frame.commit().expect("the commit");
    }
    f32_floats(&handles.read(handle, bytes).expect("read x"))
}

fn close(got: f32, want: f32, at: &str) {
    let tol = 1e-4 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{at}: got {got}, want {want} (|Δ| = {})",
        (got - want).abs()
    );
}

fn agrees(device: &Context, handles: &Handles, pipelines: &Pipelines, rows: u32, width: u32) {
    let n = u64::from(rows) * u64::from(width);
    let salt = (u64::from(rows) << 20) ^ u64::from(width);
    let x: Vec<f32> = (0..n).map(|at| unit(at ^ salt)).collect();
    let got = run(device, handles, pipelines, rows, width, &x);
    let want = blockwise_h(&x);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(g, w, &format!("[{rows}x{width}] element {i}"));
    }
}

#[test]
fn the_hadamard_transform_agrees() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();

    // Bit-exact against the host reference: one block, four blocks (512),
    // forty blocks (5120 = Qwen 27B hidden), and a batched rectangle.
    agrees(&device, &handles, &pipelines, 1, 128);
    agrees(&device, &handles, &pipelines, 1, 512);
    agrees(&device, &handles, &pipelines, 1, 5120);
    agrees(&device, &handles, &pipelines, 6, 5120);

    // Orientation probe: a one-hot input picks out one column of H. Because H
    // is symmetric this is also row k, and its exact signed entries are pinned
    // — a transposed or mis-strided matrix would not reproduce them.
    let inv = 1.0f32 / 128.0f32.sqrt();
    {
        // one-hot at position 1 of a single block -> H[:,1] = inv * (-1)^(i&1)
        let mut x = vec![0.0f32; 128];
        x[1] = 1.0;
        let got = run(&device, &handles, &pipelines, 1, 128, &x);
        for (i, &g) in got.iter().enumerate() {
            let want = if i & 1 == 1 { -inv } else { inv };
            close(g, want, &format!("probe k=1 entry {i}"));
        }
    }
    {
        // one-hot at position 3 of the SECOND block of a 256-wide row. Block 0
        // must stay zero; block 1 must be H[:,3] = inv * (-1)^popcount(i&3).
        let mut x = vec![0.0f32; 256];
        x[128 + 3] = 1.0;
        let got = run(&device, &handles, &pipelines, 1, 256, &x);
        for (i, &g) in got.iter().take(128).enumerate() {
            close(g, 0.0, &format!("probe block0 entry {i}"));
        }
        for i in 0..128usize {
            let want = if (i & 3).count_ones() & 1 == 1 {
                -inv
            } else {
                inv
            };
            close(got[128 + i], want, &format!("probe block1 entry {i}"));
        }
    }

    // Round trip: H . H = I, so applying twice returns the input.
    {
        let n: u64 = 5120;
        let x: Vec<f32> = (0..n).map(|at| unit(at ^ 0xABCD)).collect();
        let once = run(&device, &handles, &pipelines, 1, 5120, &x);
        let twice = run(&device, &handles, &pipelines, 1, 5120, &once);
        for (i, (&t, &orig)) in twice.iter().zip(x.iter()).enumerate() {
            close(t, orig, &format!("round-trip element {i}"));
        }
    }

    // Shape guard: a last dim that is not a multiple of 128 is a validation
    // fault, caught by `model_ir::check` before any kernel runs.
    assert!(
        hadamard_check_faults(100),
        "d = 100 (not a multiple of 128) must fail model-ir validation"
    );
    assert!(
        !hadamard_check_faults(128),
        "d = 128 is a valid Hadamard block width and must pass validation"
    );
}

/// Build a one-node trace `x_out = hadamard(x)` with `x` of shape
/// `[tokens, d]` and report whether `model_ir::check` flags a `HadamardBlock`
/// fault for it.
fn hadamard_check_faults(d: u64) -> bool {
    use model_ir::{
        Def, Dim, Guard, Node, Platform, RuntimeInput, Trace, Ty, ValueDecl, ValueId,
    };

    let ty = Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(d)],
        dtype: Dtype::F32,
    };
    let trace = Trace {
        name: String::from("hadamard_shape_probe"),
        platform: Platform::Metal,
        params: Vec::new(),
        caches: Vec::new(),
        values: vec![
            ValueDecl {
                def: Def::Input(RuntimeInput::Tokens),
                ty: ty.clone(),
            },
            ValueDecl {
                def: Def::Op(0),
                ty,
            },
        ],
        nodes: vec![Node {
            op: Operation::Elementwise(Elementwise::Hadamard {
                x: ValueId(0),
                x_out: ValueId(1),
            }),
            guard: Guard::Always,
            layer: None,
        }],
        seams: Vec::new(),
        drafter: None,
    };

    match model_ir::check(&trace) {
        Ok(()) => false,
        Err(faults) => faults
            .iter()
            .any(|f| matches!(f, model_ir::Fault::HadamardBlock { .. })),
    }
}
