//! The QKV split, on the GPU, through the generic executor.
//!
//! Not a hand-written dispatch: the text states the launch and its two widths,
//! `dispatch::plan` turns the rectangle into a grid, and `encode` binds the
//! operands and the scalars. If the numbers come out right, the scalar channel
//! works end to end — which is what every `_strided` kernel needs too.


use std::path::PathBuf;

use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::program::Compiler;
use driver_metal::lowering::dispatch::Dispatch;
use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::layout::region::Region as _;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[test]
fn the_split_puts_every_channel_where_its_width_says() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");

    // Deliberately unequal, so a kernel that guessed the boundary from any
    // operand shape would land somewhere visible.
    const ROWS: u32 = 3;
    const Q_W: u32 = 8;
    const KV_W: u32 = 2;
    const PACKED: u32 = Q_W + 2 * KV_W;

    let packed = Allocation::new(&context, u64::from(ROWS * PACKED) * 2, "packed").expect("packed");
    let q = Allocation::new(&context, u64::from(ROWS * Q_W) * 2, "q").expect("q");
    let k = Allocation::new(&context, u64::from(ROWS * KV_W) * 2, "k").expect("k");
    let v = Allocation::new(&context, u64::from(ROWS * KV_W) * 2, "v").expect("v");

    // Each element is its own flat index, so a misplaced channel names itself.
    let src: Vec<u16> = (0..ROWS * PACKED).map(|i| bf16(i as f32)).collect();
    unsafe {
        packed
            .write(0, bytemuck_cast(&src))
            .expect("the source fits");
    }

    let params = [Q_W, KV_W];
    let dispatch = Dispatch {
        symbol: "split_qkv_bf16",
        file: "attn/split_qkv.metal",
        grid: [PACKED, ROWS, 1],
        threadgroup: [256, 1, 1],
        args: vec![
            bound(packed.gpu_address()),
            bound(q.gpu_address()),
            bound(k.gpu_address()),
            bound(v.gpu_address()),
        ],
        params: params.to_vec(),
        // The row places its params at buffer 4, after the four operands, and
        // as one packed struct — two `u32`s, so four bytes at offset zero.
        param_slots: vec![driver_metal::lowering::dispatch::ParamSlot {
            slot: 4,
            at: 0,
            bytes: 4,
            packed: true,
            value: Some(0),
        }],
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(&context, &compiler, std::slice::from_ref(&dispatch))
        .expect("the split compiles");
    let staged = Params::stage(&context, std::slice::from_ref(&dispatch)).expect("scalars stage");
    let table = ArgumentTable::new(&context, 8).expect("a table");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    stepper
        .run(|encoder| encode(encoder, &table, &pipelines, &staged, std::slice::from_ref(&dispatch)))
        .expect("the fire runs");

    let read = |region: &driver_metal::device::Handle, n: u32| -> Vec<f32> {
        let mut out = vec![0u16; n as usize];
        let bytes = unsafe {
            core::slice::from_raw_parts(region.contents().as_ptr().cast::<u16>(), n as usize)
        };
        out.copy_from_slice(bytes);
        out.into_iter().map(from_bf16).collect()
    };

    let got_q = read(&q, ROWS * Q_W);
    let got_k = read(&k, ROWS * KV_W);
    let got_v = read(&v, ROWS * KV_W);
    for row in 0..ROWS {
        let base = (row * PACKED) as f32;
        for c in 0..Q_W {
            assert_eq!(
                got_q[(row * Q_W + c) as usize],
                base + c as f32,
                "q row {row} channel {c}"
            );
        }
        for c in 0..KV_W {
            assert_eq!(
                got_k[(row * KV_W + c) as usize],
                base + (Q_W + c) as f32,
                "k row {row} channel {c}"
            );
            assert_eq!(
                got_v[(row * KV_W + c) as usize],
                base + (Q_W + KV_W + c) as f32,
                "v row {row} channel {c}"
            );
        }
    }
}

fn bound(address: u64) -> driver_metal::lowering::executor::BoundArg {
    driver_metal::lowering::executor::BoundArg {
        slice: driver_metal::lowering::executor::Slice {
            address,
            bytes: 1 << 20,
        },
        width: 0,
    }
}

fn bytemuck_cast(v: &[u16]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
