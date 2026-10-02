//! Runs the SDK tutorial `gemv-03-memcpy` (y = A x + b on one PE) through the
//! pure-Rust SDK binding on the fabric simulator.
//!
//! ```text
//! cd $CEREBRAS_SDK_ROOT/examples/tutorials/gemv-03-memcpy
//! cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 -o out --memcpy --channels 1
//! cargo run -p engine-cerebras --example gemv -- out [wse2|wse3]
//! ```

use engine_cerebras::sdk::{
    DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target, f32_words, words_f32,
};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "out".into());
    let target = match args.next().as_deref() {
        Some("wse2") => Target::Wse2,
        _ => Target::Wse3,
    };

    let sdk = Sdk::open().expect("open Cerebras SDK");
    let mut rt = Runtime::new(
        sdk,
        Path::new(&out),
        Platform::Simulator(Simulator::new(target)),
        "WARNING",
    )
    .expect("runtime");
    println!(
        "rpc symbols: {:?}",
        rt.rpc().symbols().keys().collect::<Vec<_>>()
    );
    rt.load();
    rt.run();

    let (m, n) = (4usize, 6usize);
    let a: Vec<f32> = (0..(m * n)).map(|i| i as f32).collect();
    let x = vec![1.0f32; n];
    let b = vec![2.0f32; m];
    let expected: Vec<f32> = (0..m)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f32>() + b[i])
        .collect();

    let pe = Rect::single();
    rt.memcpy_h2d(
        "A",
        &f32_words(&a),
        pe,
        m * n,
        Order::RowMajor,
        DataType::Bits32,
    )
    .unwrap();
    rt.memcpy_h2d(
        "x",
        &f32_words(&x),
        pe,
        n,
        Order::RowMajor,
        DataType::Bits32,
    )
    .unwrap();
    rt.memcpy_h2d(
        "b",
        &f32_words(&b),
        pe,
        m,
        Order::RowMajor,
        DataType::Bits32,
    )
    .unwrap();
    rt.launch("init_and_compute", &[]).unwrap();
    let mut y = vec![0u32; m];
    rt.memcpy_d2h("y", &mut y, pe, m, Order::RowMajor, DataType::Bits32)
        .unwrap();
    rt.stop();

    let y = words_f32(&y);
    println!("y        = {y:?}");
    println!("expected = {expected:?}");
    if y.iter().zip(&expected).all(|(p, q)| (p - q).abs() < 1e-2) {
        println!("SUCCESS!");
    } else {
        println!("MISMATCH");
        std::process::exit(1);
    }
}
