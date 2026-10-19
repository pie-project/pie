//! The pure-Rust SDK binding drives a compiled program on the fabric
//! simulator in this process: host to device copies, an RPC launch and a
//! device to host copy agree with the host reference, and the host-side
//! checks refuse before touching the device.
//!
//! Needs the SDK on this box, a compiled `gemv-03-memcpy` output directory in
//! `PIE_CEREBRAS_GEMV_OUT` (see `examples/gemv.rs`), and `--test-threads=1`
//! (the SDK ties runtimes to one thread); skips otherwise.

use engine_cerebras::sdk::{
    DataType, Error, Order, Platform, Rect, Runtime, Sdk, Simulator, Target, f32_words, words_f32,
};
use std::path::PathBuf;

fn setup() -> Option<(std::sync::Arc<Sdk>, PathBuf)> {
    let out = std::env::var_os("PIE_CEREBRAS_GEMV_OUT").map(PathBuf::from)?;
    let out = std::fs::canonicalize(out).ok()?;
    // The simulator writes logs and scratch directories into the cwd.
    let scratch = std::env::temp_dir().join(format!("pie-cerebras-gemv-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).ok()?;
    std::env::set_current_dir(&scratch).ok()?;
    let sdk = match Sdk::open() {
        Ok(sdk) => sdk,
        Err(e) => {
            eprintln!("skipping: {e}");
            return None;
        }
    };
    Some((sdk, out))
}

#[test]
fn a_gemv_round_trips_and_host_checks_refuse_first() {
    let Some((sdk, out)) = setup() else { return };
    let mut rt = match Runtime::new(
        sdk.clone(),
        &out,
        Platform::Simulator(Simulator::new(Target::Wse3)),
        "WARNING",
    ) {
        Ok(rt) => rt,
        Err(Error::MainThread) => {
            eprintln!("skipping: run with --test-threads=1");
            return;
        }
        Err(e) => panic!("{e}"),
    };

    // Host-side checks refuse before touching the device.
    let pe = Rect::single();
    assert!(matches!(
        rt.memcpy_h2d("nope", &[0; 4], pe, 4, Order::RowMajor, DataType::Bits32),
        Err(Error::UnknownSymbol(_))
    ));
    assert!(matches!(
        rt.memcpy_h2d("A", &[0; 3], pe, 24, Order::RowMajor, DataType::Bits32),
        Err(Error::Size { .. })
    ));
    assert!(matches!(rt.launch("A", &[]), Err(Error::WrongKind(..))));
    assert!(matches!(
        rt.memcpy_h2d(
            "init_and_compute",
            &[0],
            pe,
            1,
            Order::RowMajor,
            DataType::Bits32
        ),
        Err(Error::WrongKind(..))
    ));

    rt.load();
    rt.run();
    let (m, n) = (4usize, 6usize);
    let a: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.5).collect();
    let x: Vec<f32> = (0..n).map(|j| j as f32 - 2.0).collect();
    let b: Vec<f32> = (0..m).map(|i| 10.0 * i as f32).collect();
    let expected: Vec<f32> = (0..m)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f32>() + b[i])
        .collect();
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
    let y = words_f32(&y);
    for (got, want) in y.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-3,
            "y = {y:?}, expected {expected:?}"
        );
    }
}
