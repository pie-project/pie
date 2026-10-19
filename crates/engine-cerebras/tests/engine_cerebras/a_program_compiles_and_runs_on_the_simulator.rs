//! End to end through this crate alone: CSL sources written from Rust are
//! compiled with `cslc`, loaded on the fabric simulator and driven through
//! the pure-Rust SDK binding. Skips when the SDK toolchain is not on this box.

use engine_cerebras::sdk::{
    Arch, Compile, Cslc, DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target,
    f32_words, words_f32,
};

const LAYOUT: &str = r#"
param M: i16;
param N: i16;
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "pe.csl", .{ .memcpy_params = memcpy.get_params(0), .M = M, .N = N });
  @export_name("A", [*]f32, true);
  @export_name("x", [*]f32, true);
  @export_name("y", [*]f32, false);
  @export_name("gemv", fn(u32)void);
}
"#;

const PE: &str = r#"
param memcpy_params;
param M: i16;
param N: i16;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var A: [M*N]f32;
var x: [N]f32;
var y = @zeros([M]f32);

var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i*N] });
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M });

var A_ptr: [*]f32 = &A;
var x_ptr: [*]f32 = &x;
const y_ptr: [*]f32 = &y;

// y = scale * (A x), scale arriving as a u32-encoded f32 launch argument.
fn gemv(scale_bits: u32) void {
  const scale: f32 = @bitcast(f32, scale_bits);
  for (@range(i16, N)) |i| {
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
    A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
  }
  @fmuls(y_dsd, y_dsd, scale);
  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(A_ptr, "A");
  @export_symbol(x_ptr, "x");
  @export_symbol(y_ptr, "y");
  @export_symbol(gemv);
}
"#;

#[test]
fn a_parameterised_gemv_round_trips() {
    if !Cslc::available() {
        eprintln!("skipping: cslc not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // The SDK notes the cwd when its libraries load and the simulator writes
    // logs and scratch directories there: move first, then open.
    std::env::set_current_dir(dir.path()).unwrap();
    let sdk = match Sdk::open() {
        Ok(sdk) => sdk,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    std::fs::write(dir.path().join("layout.csl"), LAYOUT).unwrap();
    std::fs::write(dir.path().join("pe.csl"), PE).unwrap();

    let (m, n) = (3usize, 5usize);
    let out = dir.path().join("out");
    let compile = Compile::memcpy(Arch::Wse3, dir.path().join("layout.csl"), 1, 1, &out)
        .param("M", m)
        .param("N", n);
    Cslc::find().unwrap().compile(&compile).unwrap();

    let mut rt = match Runtime::new(
        sdk,
        &out,
        Platform::Simulator(Simulator::new(Target::Wse3)),
        "WARNING",
    ) {
        Ok(rt) => rt,
        Err(engine_cerebras::sdk::Error::MainThread) => {
            eprintln!("skipping: run with --test-threads=1");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    rt.load();
    rt.run();

    let a: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let x: Vec<f32> = (0..n).map(|j| (j as f32) + 0.5).collect();
    let scale = 2.0f32;
    let expected: Vec<f32> = (0..m)
        .map(|i| scale * (0..n).map(|j| a[i * n + j] * x[j]).sum::<f32>())
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
    rt.launch("gemv", &[scale.to_bits()]).unwrap();
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
