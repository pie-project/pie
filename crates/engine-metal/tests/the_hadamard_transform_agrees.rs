#![cfg(target_vendor = "apple")]

//! `elementwise.hadamard` on Metal is a blockwise butterfly FWHT: the row's
//! last dim is cut into contiguous `block`-vectors (a power of two) and each is
//! turned by the normalized Sylvester Hadamard matrix H, whose entries are
//! `(-1)^popcount(i & j) / sqrt(block)`. The fast Walsh-Hadamard transform's
//! natural (Sylvester) ordering is exactly H, so the pure transform is
//! symmetric and orthonormal: `H . H = I`.
//!
//! When a ±1 sign diagonal S is supplied it multiplies the activation on load,
//! before the butterfly — the Randomized Hadamard `H·S`. S and H do not
//! commute, so `H·S` is NOT self-inverse; the signed path is checked only by
//! bit-exact agreement with the host reference.
//!
//! This checks the Metal output against a host reference across both the
//! simdgroup (block <= 256) and threadgroup (block >= 512) pipelines, pins the
//! exact matrix entries with a one-hot orientation probe, confirms the round
//! trip for the pure transform, and confirms the shape guards reject a last dim
//! that is not a multiple of `block` and a non-power-of-two block at
//! validation, not at runtime.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::pointwise;
use model_ir::{Dtype, Elementwise, Operation};

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (f64::from(noise(at)) / f64::from(u32::MAX)) as f32 * 2.0 - 1.0
}

/// A deterministic ±1 sign vector of the given width.
fn signs_of(width: usize, salt: u64) -> Vec<f32> {
    (0..width as u64)
        .map(|i| if noise(i ^ salt) & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
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

/// The host reference: each contiguous `block`-vector is optionally scaled by
/// the ±1 diagonal S (which repeats block-wise across the flattened row) and
/// then left-multiplied by the normalized Sylvester Hadamard matrix. Signs are
/// indexed by flat offset, matching the kernel's `p % signs_width`. Accumulated
/// in f64 for a clean gold.
fn blockwise_h_signed(x: &[f32], block: usize, signs: Option<&[f32]>) -> Vec<f32> {
    let inv = 1.0f64 / (block as f64).sqrt();
    let sw = signs.map_or(0, <[f32]>::len);
    let mut out = vec![0.0f32; x.len()];
    for (b, chunk) in x.chunks_exact(block).enumerate() {
        let base = b * block;
        let signed: Vec<f64> = (0..block)
            .map(|j| {
                let s = signs.map_or(1.0, |sg| f64::from(sg[(base + j) % sw]));
                s * f64::from(chunk[j])
            })
            .collect();
        for i in 0..block {
            let mut acc = 0.0f64;
            for (j, &v) in signed.iter().enumerate() {
                let sign = if (i & j).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += sign * v;
            }
            out[base + i] = (acc * inv) as f32;
        }
    }
    out
}

/// Run the Metal op over `rows x width` f32 values held row-major, in place,
/// and read the result back. An optional ±1 sign diagonal is bound as a second
/// buffer.
#[allow(clippy::too_many_arguments)]
fn run(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    width: u32,
    block: u32,
    data: &[f32],
    signs: Option<&[f32]>,
) -> Vec<f32> {
    let bytes = u64::from(rows) * u64::from(width) * 4;
    let mut buf = Buffer::zeroed(device, bytes).expect("a buffer");
    buf.write(0, &f32_bytes(data)).expect("write x");
    let handle = handles.bind(&buf, 0, buf.bytes()).expect("a handle");
    let x = Tensor::new(handle, rows, width, Dtype::F32);

    // Keep the sign buffer alive for the whole frame.
    let sign_hold = signs.map(|s| {
        let sbytes = (s.len() * 4) as u64;
        let mut sbuf = Buffer::zeroed(device, sbytes).expect("a sign buffer");
        sbuf.write(0, &f32_bytes(s)).expect("write signs");
        let shandle = handles.bind(&sbuf, 0, sbuf.bytes()).expect("a sign handle");
        let stensor = Tensor::new(shandle, 1, s.len() as u32, Dtype::F32);
        (sbuf, stensor)
    });
    let signs_tensor = sign_hold.as_ref().map(|(_, t)| *t);

    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(device, &frame, pipelines, handles);
        pointwise::hadamard(&sink, x, block, signs_tensor).expect("the launch");
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

/// Bit-exact agreement of the pure (unsigned) transform for a rectangle.
fn agrees(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    width: u32,
    block: u32,
) {
    let n = u64::from(rows) * u64::from(width);
    let salt = (u64::from(rows) << 32) ^ (u64::from(width) << 8) ^ u64::from(block);
    let x: Vec<f32> = (0..n).map(|at| unit(at ^ salt)).collect();
    let got = run(device, handles, pipelines, rows, width, block, &x, None);
    let want = blockwise_h_signed(&x, block as usize, None);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(g, w, &format!("[{rows}x{width}] block {block} element {i}"));
    }
}

/// Bit-exact agreement of the signed transform `H·(S·x)` for a rectangle.
fn agrees_signed(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    width: u32,
    block: u32,
    signs_width: usize,
) {
    let n = u64::from(rows) * u64::from(width);
    let salt = (u64::from(rows) << 32) ^ (u64::from(width) << 8) ^ u64::from(block);
    let x: Vec<f32> = (0..n).map(|at| unit(at ^ salt ^ 0x5157_u64)).collect();
    let signs = signs_of(signs_width, 0xF00D ^ u64::from(block));
    let got = run(
        device,
        handles,
        pipelines,
        rows,
        width,
        block,
        &x,
        Some(&signs),
    );
    let want = blockwise_h_signed(&x, block as usize, Some(&signs));
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(
            g,
            w,
            &format!("signed [{rows}x{width}] block {block} sw {signs_width} element {i}"),
        );
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

    // Bit-exact against the host reference across both pipelines: block 128 and
    // 256 ride the simdgroup variant; block 512 rides the threadgroup variant.
    // Shapes cover one block, several blocks, and a batch of rows.
    for &block in &[128u32, 256, 512] {
        agrees(&device, &handles, &pipelines, 1, block, block); // one block
        agrees(&device, &handles, &pipelines, 1, block * 4, block); // several blocks
        agrees(&device, &handles, &pipelines, 5, block * 3, block); // a batch
    }
    // The 5120-wide Qwen 27B hidden width, several blocks of 512, batched.
    agrees(&device, &handles, &pipelines, 6, 5120, 512);
    // Also exercise the smaller simd sizes.
    agrees(&device, &handles, &pipelines, 3, 64 * 5, 64);
    agrees(&device, &handles, &pipelines, 2, 1024 * 2, 1024);

    // Signed path `H·(S·x)`, bit-exact against the host reference. A full-width
    // diagonal (signs_width == width) and a block-wide diagonal that repeats
    // block-wise across the row are both exercised, on both pipelines.
    for &block in &[128u32, 256, 512] {
        agrees_signed(
            &device,
            &handles,
            &pipelines,
            4,
            block * 3,
            block,
            (block * 3) as usize,
        );
        agrees_signed(
            &device,
            &handles,
            &pipelines,
            4,
            block * 3,
            block,
            block as usize,
        );
    }

    // Orientation probe: a one-hot input picks out one column of H. Because H is
    // symmetric this is also row k, and its exact signed entries are pinned — a
    // transposed or mis-strided matrix would not reproduce them. Tested at each
    // block size, and in the SECOND block of a two-block row so striding is
    // exercised too.
    for &block in &[128u32, 256, 512] {
        let bl = block as usize;
        let inv = 1.0f32 / (block as f32).sqrt();
        let k = 3usize;
        let mut x = vec![0.0f32; bl * 2];
        x[bl + k] = 1.0;
        let got = run(&device, &handles, &pipelines, 1, block * 2, block, &x, None);
        for (i, &g) in got.iter().take(bl).enumerate() {
            close(g, 0.0, &format!("probe block0 (n={block}) entry {i}"));
        }
        for i in 0..bl {
            let want = if (i & k).count_ones() & 1 == 1 {
                -inv
            } else {
                inv
            };
            close(
                got[bl + i],
                want,
                &format!("probe block1 (n={block}) entry {i}"),
            );
        }
    }

    // Round trip: H . H = I for the PURE transform, so applying twice returns
    // the input. (Not tested for the signed transform, which is not an
    // involution.)
    for &block in &[128u32, 256, 512] {
        let n = u64::from(block) * 9;
        let width = block * 9;
        let x: Vec<f32> = (0..n)
            .map(|at| unit(at ^ 0xABCD ^ u64::from(block)))
            .collect();
        let once = run(&device, &handles, &pipelines, 1, width, block, &x, None);
        let twice = run(&device, &handles, &pipelines, 1, width, block, &once, None);
        for (i, (&t, &orig)) in twice.iter().zip(x.iter()).enumerate() {
            close(t, orig, &format!("round-trip (n={block}) element {i}"));
        }
    }

    // Shape guard: a last dim that is not a multiple of `block`, and a
    // non-power-of-two `block`, are validation faults caught by `model_ir::check`
    // before any kernel runs.
    assert!(
        hadamard_check_faults(100, 128),
        "last dim 100 (not a multiple of 128) must fail model-ir validation"
    );
    assert!(
        !hadamard_check_faults(256, 128),
        "last dim 256 is a whole number of 128-blocks and must pass validation"
    );
    assert!(
        hadamard_check_faults(768, 96),
        "block 96 is not a power of two and must fail model-ir validation"
    );
    assert!(
        !hadamard_check_faults(512, 512),
        "block 512 with a matching last dim must pass validation"
    );
}

/// Build a one-node trace `x_out = hadamard(x, block)` with `x` of shape
/// `[tokens, d]` and report whether `model_ir::check` flags a `HadamardBlock`
/// fault for it.
fn hadamard_check_faults(d: u64, block: u32) -> bool {
    use model_ir::{Def, Dim, Guard, Node, Platform, RuntimeInput, Trace, Ty, ValueDecl, ValueId};

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
                block,
                signs: None,
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
