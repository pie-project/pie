#![cfg(target_vendor = "apple")]

//! A concurrent frame runs launches side by side and puts a barrier only
//! ahead of one that touches bytes a launch since the last barrier wrote.
//! Two activations here are disjoint halves of one buffer, and a third is
//! the first rows of the first. A Hadamard turns its rows in place and
//! `H·H = I`, so the data comes back as it went in only if every pair over
//! the same bytes ran in order; the frame's census says how many barriers
//! that took.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::pointwise;
use poem_ir::Dtype;

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

#[test]
fn a_concurrent_frame_orders_only_what_shares_bytes() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();

    let (rows, width, block) = (4u32, 256u32, 256u32);
    let half = u64::from(rows) * u64::from(width) * 4;
    let data: Vec<f32> = (0..2 * rows * width)
        .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let mut buf = Buffer::zeroed(&device, 2 * half).expect("a buffer");
    buf.write(0, &f32_bytes(&data)).expect("write the data");
    let x = Tensor::new(
        handles.bind(&buf, 0, half).expect("x"),
        rows,
        width,
        Dtype::F32,
    );
    let y = Tensor::new(
        handles.bind(&buf, half, half).expect("y"),
        rows,
        width,
        Dtype::F32,
    );
    let head = Tensor::new(
        handles.bind(&buf, 0, half / 2).expect("the head of x"),
        rows / 2,
        width,
        Dtype::F32,
    );

    let frame = device.concurrent_frame().expect("a frame");
    if !frame.concurrent() {
        eprintln!("not asked: the `serial` diagnostic hands out serial frames");
        return;
    }
    let census = {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        // x and y are other bytes: only the second turn of each waits.
        pointwise::hadamard(&sink, x, block, None).expect("x");
        pointwise::hadamard(&sink, y, block, None).expect("y");
        pointwise::hadamard(&sink, x, block, None).expect("x again");
        pointwise::hadamard(&sink, y, block, None).expect("y again");
        let census = frame.census();
        // The head is inside x: each turn over it, and x's after, waits.
        pointwise::hadamard(&sink, x, block, None).expect("x");
        pointwise::hadamard(&sink, head, block, None).expect("the head");
        pointwise::hadamard(&sink, head, block, None).expect("the head again");
        pointwise::hadamard(&sink, x, block, None).expect("x again");
        (census, frame.census())
    };
    frame.commit().expect("the commit");

    assert_eq!(
        census.0,
        (4, 1),
        "the second turn of x waits on the first, and y's turns on nothing \
         across x's barrier"
    );
    assert_eq!(
        census.1,
        (8, 5),
        "x waits on its own turn before it, each turn over the head waits, and so does \
         x's after them"
    );
    let mut back = vec![0u8; usize::try_from(2 * half).expect("a small buffer")];
    buf.read(0, &mut back).expect("read back");
    let got = f32_floats(&back);
    for (i, (&g, &w)) in got.iter().zip(&data).enumerate() {
        assert!((g - w).abs() <= 1e-4, "element {i}: got {g}, want {w}");
    }
}
