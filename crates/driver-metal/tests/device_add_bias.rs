//! The Qwen-2 projection bias, on the GPU.
//!
//! `norm/add_bias.metal` was written for all three backends so that a Qwen-2
//! would stop being served without its q/k/v biases, and on Metal it sat
//! unreachable long enough for its own capability flag to say so in prose:
//! "No Mac has run this." This is the Mac running it.
//!
//! Two things are under test and they fail differently. The SHADER's index
//! arithmetic — `out[y*width + x] += bias[x]` — is wrong if the bias is
//! broadcast down instead of across, which a square fixture would hide. The
//! BINDER's `Source::OutWidth` arm is what puts `width` at buffer 2 at all;
//! without it the shader reads whatever the encoder last left there, so the
//! test builds its scalar slot the way `param_layout` now does rather than
//! asserting a shape it wishes for.

use std::path::PathBuf;

use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;
use driver_metal::lowering::dispatch::{Dispatch, ParamSlot, Touches};

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
fn the_bias_lands_on_every_row_and_on_the_right_column() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    // NOT square, and not a multiple of the threadgroup: a kernel that swapped
    // the two axes would read past the bias vector, and one that rounded the
    // grid up would write past the last row.
    const ROWS: u32 = 5;
    const WIDTH: u32 = 3;

    let out = Allocation::new(&context, u64::from(ROWS * WIDTH) * 2, "out").expect("out");
    let bias = Allocation::new(&context, u64::from(WIDTH) * 2, "bias").expect("bias");

    // The value carries its own coordinates: row r, column c holds
    // `10r + c` and the bias holds `100 + c`, so every wrong answer names the
    // row or the column it actually read.
    //
    // Every value and every sum is a whole number below 256, which is where
    // bfloat16 stops being exact — eight bits of significand. The first
    // fixture here used `100r + c` and `1000 + c`, and the kernel was RIGHT:
    // 1002 is not a bfloat16, and both sides rounded it to 1000. A numeric
    // test whose oracle the format cannot hold tests the format.
    let src: Vec<u16> = (0..ROWS)
        .flat_map(|r| (0..WIDTH).map(move |c| bf16((10 * r + c) as f32)))
        .collect();
    let b: Vec<u16> = (0..WIDTH).map(|c| bf16((100 + c) as f32)).collect();
    unsafe {
        out.write(0, cast(&src)).expect("the source fits");
        bias.write(0, cast(&b)).expect("the bias fits");
    }

    let dispatch = Dispatch {
        symbol: "add_bias_bfloat16",
        file: "norm/add_bias.metal",
        // `LaunchRule::RouteRows`: the column is `tid.x` because the bias is
        // broadcast and the column has to be recoverable from the invocation.
        grid: [WIDTH, ROWS, 1],
        threadgroup: [256, 1, 1],
        touches: Touches::everything(&[bound(out.gpu_address())]),
        args: vec![bound(out.gpu_address()), bound(bias.gpu_address())],
        params: vec![WIDTH],
        // What `param_layout`'s `derived` arm emits: a lone four-byte scalar
        // at the buffer the row names, unpacked, because the row places it.
        param_slots: vec![ParamSlot {
            slot: 2,
            at: 0,
            bytes: 4,
            packed: false,
            value: Some(0),
        }],
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(&context, &compiler, std::slice::from_ref(&dispatch))
        .expect("the bias add compiles on this device");
    let staged = Params::stage(&context, std::slice::from_ref(&dispatch)).expect("scalars stage");
    let table = ArgumentTable::new(&context, 8).expect("a table");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .expect("the fire runs");

    let mut got = vec![0u16; (ROWS * WIDTH) as usize];
    got.copy_from_slice(unsafe {
        core::slice::from_raw_parts(
            out.contents().as_ptr().cast::<u16>(),
            (ROWS * WIDTH) as usize,
        )
    });
    for r in 0..ROWS {
        for c in 0..WIDTH {
            assert_eq!(
                from_bf16(got[(r * WIDTH + c) as usize]),
                (10 * r + c + 100 + c) as f32,
                "row {r} column {c}"
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

fn cast(v: &[u16]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
