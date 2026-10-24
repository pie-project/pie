#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! Two ranks on one device stand in for a pair of peers, as the peer
//! collectives' test does: the wire's landings and returns are the mapped
//! buffers peers would reach. A token dispatched out and combined back,
//! with each expert answering its row as it landed, comes back as it left.

use engine_cuda::device::{Buffer, Context};
use engine_cuda::ep::{Widest, wire};
use kernels_cuda::collective::{Placement, Tokens, combine, dispatch};
use kernels_cuda::{Ctx, Tensor};
use poem_ir::Dtype;

fn down<T: Copy + Default>(at: u64, count: usize) -> Vec<T> {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let mut out = vec![T::default(); count];
    // SAFETY: `out` holds `count` elements; `at` is a device range of at
    // least that many the wire carved; the copy is synchronous.
    let code = unsafe {
        rt::cudaMemcpy(
            out.as_mut_ptr().cast(),
            at as *const core::ffi::c_void,
            core::mem::size_of_val(out.as_slice()),
            rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
    };
    assert_eq!(code, rt::cudaError::cudaSuccess, "cudaMemcpy D2H");
    out
}

fn up<T: Copy>(values: &[T]) -> Buffer {
    let bytes = core::mem::size_of_val(values);
    let mut buf = Buffer::zeroed(bytes.max(1)).unwrap();
    // SAFETY: plain-old-data values viewed as their bytes for the copy.
    let raw = unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) };
    buf.write(0, raw).unwrap();
    buf
}

#[test]
fn an_expert_parallel_wire_carries_tokens_out_and_back() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let (world, tokens, width, top_k, experts) = (2u32, 24u32, 128u32, 4u32, 8u32);
    let row_bytes = u64::from(width) * 2;
    let widest = Widest {
        tokens: 32,
        row_bytes,
        out_bytes: row_bytes,
        top_k,
    };
    let wires = wire(&[0, 0], widest).expect("one device reaches itself");
    let devices: Vec<Context> = (0..world)
        .map(|_| Context::bind(0, None).unwrap())
        .collect();
    // SAFETY: each context's stream is live for the test.
    let ctxs: Vec<Ctx> = devices
        .iter()
        .map(|d| unsafe { Ctx::on(d.stream()) })
        .collect();

    let slots = experts / world;
    let table: Vec<i32> = (0..experts as i32).flat_map(|e| [1, e]).collect();
    let table = up(&table);
    let place = Placement {
        table: table.ptr(),
        stride: 2,
        experts,
        slots,
        loads: 0,
    };
    let xs: Vec<Vec<u16>> = (0..world)
        .map(|r| {
            (0..tokens * width)
                .map(|i| ((i * 3 + r * 1009) % 0x7fff) as u16)
                .collect()
        })
        .collect();
    let routes: Vec<Vec<i32>> = (0..world)
        .map(|r| {
            (0..tokens * top_k)
                .map(|i| ((i / top_k * 3 + i % top_k + r) % experts) as i32)
                .collect()
        })
        .collect();
    let held: Vec<(Buffer, Buffer, Buffer)> = (0..world as usize)
        .map(|r| {
            let counts = Buffer::zeroed(slots as usize * 4).unwrap();
            (up(&xs[r]), up(&routes[r]), counts)
        })
        .collect();
    let landing: Vec<_> = wires
        .iter()
        .map(|w| w.landing(tokens, top_k, width, Dtype::Bf16, false))
        .collect();
    for (r, ctx) in ctxs.iter().enumerate() {
        let (x, rt, counts) = &held[r];
        let mut counts = Tensor::new(counts.ptr(), 1, slots, Dtype::I32);
        let toks = Tokens {
            x: Tensor::new(x.ptr(), tokens, width, Dtype::Bf16),
            routes: Tensor::new(rt.ptr(), tokens, top_k, Dtype::I32),
            weights: None,
            step: tokens,
        };
        dispatch(ctx, &wires[r].ep, place, toks, &landing, &mut counts).unwrap();
    }
    devices.iter().for_each(|d| d.synchronize().unwrap());

    // Each expert answers the row it was handed.
    let cap = (world * tokens) as usize;
    let ys: Vec<Buffer> = landing
        .iter()
        .map(|l| {
            let rows: Vec<u16> = down(l.rows.ptr, cap * width as usize);
            let local: Vec<i32> = down(l.local.ptr, cap * top_k as usize);
            let y: Vec<u16> = (0..cap * top_k as usize)
                .flat_map(|at| {
                    let row = &rows[at / top_k as usize * width as usize..][..width as usize];
                    let live = local[at] >= 0;
                    row.iter().map(move |&v| if live { v } else { 0 })
                })
                .collect();
            up(&y)
        })
        .collect();
    let routed: Vec<_> = wires
        .iter()
        .map(|w| w.routed(tokens, top_k, width, Dtype::Bf16))
        .collect();
    for (r, ctx) in ctxs.iter().enumerate() {
        let y = Tensor::new(ys[r].ptr(), (cap as u32) * top_k, width, Dtype::Bf16);
        combine(ctx, &wires[r].ep, &landing[r], y, &routed).unwrap();
    }
    devices.iter().for_each(|d| d.synchronize().unwrap());

    for r in 0..world as usize {
        let got: Vec<u16> = down(routed[r].ptr, (tokens * top_k * width) as usize);
        for t in 0..tokens as usize {
            for j in 0..top_k as usize {
                let back = &got[(t * top_k as usize + j) * width as usize..][..width as usize];
                let left = &xs[r][t * width as usize..][..width as usize];
                assert_eq!(
                    back, left,
                    "rank {r} token {t} route {j} came back as it left"
                );
            }
        }
    }
}
