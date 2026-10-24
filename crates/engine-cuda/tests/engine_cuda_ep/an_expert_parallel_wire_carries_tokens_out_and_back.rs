#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! Rank `r` on device `r % count`, as the peer collectives' test places
//! them: the wire's landings and returns are the buffers mapped into every
//! peer. Each landed row's routes name its experts' slots on the rank, with
//! their weights, and each slot counts its rows; a token dispatched out and
//! combined back, with each expert answering its row as it landed, comes
//! back as it left. Several rounds over the same buffers, as layers and
//! steps reuse them.

use engine_cuda::device::{Buffer, Context};
use engine_cuda::ep::{Widest, wire};
use kernels_cuda::collective::{Placement, Tokens, combine, dispatch};
use kernels_cuda::{Ctx, Tensor};
use poem_ir::Dtype;

use crate::micro::{ordinals, ranks};

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
    let world = ranks();
    let ordinals = ordinals(world);
    // A small step, and the micro's (32 experts, top 4, 512 wide) at fewer
    // live tokens than its widest bucket.
    check(world, &ordinals, [24, 128, 4, 8, 32]);
    check(world, &ordinals, [82, 512, 4, 32, 256]);
}

/// `tokens` live tokens of `width` a rank, `top_k` routes each over
/// `experts`, through a wire sized for `widest` tokens.
fn check(world: u32, ordinals: &[i32], [tokens, width, top_k, experts, widest]: [u32; 5]) {
    let row_bytes = u64::from(width) * 2;
    let widest = Widest {
        tokens: widest,
        row_bytes,
        out_bytes: row_bytes,
        top_k,
    };
    let wires = wire(ordinals, widest).expect("every rank reaches every other");
    let devices: Vec<Context> = (ordinals.iter())
        .map(|&ordinal| Context::bind(ordinal, None).unwrap())
        .collect();
    // SAFETY: each context's stream is live for the test.
    let ctxs: Vec<Ctx> = devices
        .iter()
        .map(|d| unsafe { Ctx::on(d.stream()) })
        .collect();

    let slots = experts / world;
    let table: Vec<i32> = (0..experts as i32).flat_map(|e| [1, e]).collect();
    let tables: Vec<Buffer> = (devices.iter())
        .map(|device| {
            device.bind_thread().unwrap();
            up(&table)
        })
        .collect();
    let place = |rank: usize| Placement {
        table: tables[rank].ptr(),
        stride: 2,
        experts,
        slots,
        loads: 0,
    };
    // Rounds over the same buffers, as a trace's layers and steps reuse them.
    for round in 0..4u32 {
        let xs: Vec<Vec<u16>> = (0..world)
            .map(|r| {
                (0..tokens * width)
                    .map(|i| ((i * 3 + (r + 7 * round) * 1009) % 0x7fff) as u16)
                    .collect()
            })
            .collect();
        let routes: Vec<Vec<i32>> = (0..world)
            .map(|r| {
                (0..tokens * top_k)
                    .map(|i| ((i / top_k * 3 + i % top_k + r + 5 * round) % experts) as i32)
                    .collect()
            })
            .collect();
        let weights: Vec<Vec<f32>> = (0..world)
            .map(|r| {
                (0..tokens * top_k)
                    .map(|i| (r * 100_000 + round * 10_000 + i) as f32)
                    .collect()
            })
            .collect();
        let held: Vec<(Buffer, Buffer, Buffer, Buffer)> = (0..world as usize)
            .map(|r| {
                devices[r].bind_thread().unwrap();
                let counts = Buffer::zeroed(slots as usize * 4).unwrap();
                (up(&xs[r]), up(&routes[r]), up(&weights[r]), counts)
            })
            .collect();
        let landing: Vec<_> = wires
            .iter()
            .map(|w| w.landing(tokens, top_k, width, Dtype::Bf16, true))
            .collect();
        for (r, ctx) in ctxs.iter().enumerate() {
            devices[r].bind_thread().unwrap();
            let (x, rt, wt, counts) = &held[r];
            let mut counts = Tensor::new(counts.ptr(), 1, slots, Dtype::I32);
            let toks = Tokens {
                x: Tensor::new(x.ptr(), tokens, width, Dtype::Bf16),
                routes: Tensor::new(rt.ptr(), tokens, top_k, Dtype::I32),
                weights: Some(Tensor::new(wt.ptr(), tokens, top_k, Dtype::F32)),
                step: tokens,
            };
            dispatch(ctx, &wires[r].ep, place(r), toks, &landing, &mut counts).unwrap();
        }
        devices.iter().for_each(|d| {
            d.bind_thread().unwrap();
            d.synchronize().unwrap();
        });

        // Each expert answers the row it was handed.
        let cap = (world * tokens) as usize;
        let ys: Vec<Buffer> = (landing.iter().zip(&devices).enumerate())
            .map(|(d, (l, device))| {
                device.bind_thread().unwrap();
                let rows: Vec<u16> = down(l.rows.ptr, cap * width as usize);
                let local: Vec<i32> = down(l.local.ptr, cap * top_k as usize);
                // Each landed row's routes name the slots of their experts here.
                let origin: Vec<i32> = down(l.origin.ptr, cap);
                let row_weights: Vec<f32> =
                    down(l.row_weights.expect("weighed").ptr, cap * top_k as usize);
                let counts: Vec<i32> = down(held[d].3.ptr(), slots as usize);
                let mut want_counts = vec![0i32; slots as usize];
                for (row, &from) in origin.iter().enumerate().filter(|(_, o)| **o >= 0) {
                    let (src, t) = (from as u32 / tokens, from as u32 % tokens);
                    for j in 0..top_k {
                        let at = (t * top_k + j) as usize;
                        let e = routes[src as usize][at];
                        let here = e as u32 / slots == d as u32;
                        let got = row * top_k as usize + j as usize;
                        assert_eq!(
                            local[got],
                            if here { e % slots as i32 } else { -1 },
                            "rank {d} row {row} (rank {src} token {t}) route {j}"
                        );
                        if here {
                            want_counts[(e % slots as i32) as usize] += 1;
                            assert_eq!(
                                row_weights[got], weights[src as usize][at],
                                "rank {d} row {row} route {j}'s weight"
                            );
                        }
                    }
                }
                assert_eq!(counts, want_counts, "rank {d}'s rows a slot");
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
            devices[r].bind_thread().unwrap();
            let y = Tensor::new(ys[r].ptr(), (cap as u32) * top_k, width, Dtype::Bf16);
            combine(ctx, &wires[r].ep, &landing[r], y, &routed).unwrap();
        }
        devices.iter().for_each(|d| {
            d.bind_thread().unwrap();
            d.synchronize().unwrap();
        });

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
}
