//! The port cast on the first wgpu device against an exact host reference:
//! f32 cells cast into bf16 seats of one store, seats starting and ending
//! mid-word and sharing words with their neighbours, all in one frame, the
//! store's other elements left as they were.

use poem_ir::Dtype;

use crate::gpu::{Gpu, bf16, noise, widen};

const STORE: usize = 64;

/// (first element, values): odd and even starts and ends, a seat of one,
/// neighbours sharing a word (10 and 11), a seat ending the store.
const SEATS: [(usize, usize); 6] = [(0, 6), (3, 7), (10, 1), (11, 4), (20, 9), (50, 14)];

#[test]
fn the_port_cast_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    // Seats that overlap ((0, 6) and (3, 7)) run in two frames, in order.
    for frames in [&SEATS[..1], &SEATS[1..]] {
        let (store, mut want) = gpu.up(&noise(1, STORE));
        let cells: Vec<(u32, usize, Vec<f32>)> = frames
            .iter()
            .map(|&(at, n)| {
                let v = noise(at as u64 + 9, n);
                (gpu.up_f32(&v), at, v)
            })
            .collect();
        for (_, at, v) in &cells {
            for (i, x) in v.iter().enumerate() {
                want[at + i] = widen(bf16(*x));
            }
        }
        gpu.run(|c| {
            for (cell, at, v) in &cells {
                kernels_slang::runtime::port::cast(
                    c,
                    kernels_slang::Tensor::new(*cell, 1, v.len() as u32, Dtype::F32),
                    kernels_slang::Tensor::new(store, 1, STORE as u32, Dtype::Bf16),
                    *at as u32,
                )?;
            }
            Ok(())
        });
        let got = gpu.down(store, STORE);
        assert_eq!(got, want, "port cast {frames:?}");
        eprintln!("port cast {frames:?}: exact");
    }
}
