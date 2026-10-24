//! A fused region (`eta_compiler::region::Region`): a function of one row
//! the tier-0 interpreter runs, and whose body the stream core writes in the
//! row frame's hooks, the row kept in registers across its reductions.

use std::process::Command;

use eta_compiler::eval::interp::Value;
use eta_compiler::plan::{Dimension, SymbolicExtent, SymbolicType};
use eta_compiler::region::{Region, RegionError};
use eta_ir::op::Op;
use eta_ir::types::{Dtype, Literal, Shape};

const WIDTH: u32 = 64;
const EPS: f32 = 1e-6;

fn row() -> SymbolicType {
    SymbolicType {
        dtype: Dtype::F32,
        dims: vec![Dimension::Static(WIDTH)],
    }
}

/// `h = y + x` (stored in place, result 1), then `rmsnorm(h) * w` (result
/// 0): parameters 0 `x`, 1 `y`, 2 `w`.
fn residual_add_rmsnorm() -> Region {
    Region {
        params: vec![row(), row(), row()],
        ops: vec![
            Op::Add(1, 0),
            Op::Mul(3, 3),
            Op::ReduceSum(4),
            Op::Const(Literal::F32(WIDTH as f32)),
            Op::Div(5, 6),
            Op::Const(Literal::F32(EPS)),
            Op::Add(7, 8),
            Op::Rsqrt(9),
            Op::Broadcast {
                value: 10,
                shape: Shape::vector(WIDTH),
            },
            Op::Mul(3, 11),
            Op::Mul(12, 2),
        ],
        results: vec![13, 3],
    }
}

/// `(x - mean) / std * w + b` over the row: parameters 0 `x`, 1 `w`, 2 `b`.
fn layernorm() -> Region {
    let n = 3;
    Region {
        params: vec![row(), row(), row()],
        ops: vec![
            Op::ReduceSum(0),                      // 3
            Op::Const(Literal::F32(WIDTH as f32)), // 4
            Op::Div(n, n + 1),                     // 5 mean
            Op::Broadcast {
                value: 5,
                shape: Shape::vector(WIDTH),
            }, // 6
            Op::Sub(0, 6),                         // 7 d
            Op::Mul(7, 7),                         // 8
            Op::ReduceSum(8),                      // 9
            Op::Div(9, 4),                         // 10 var
            Op::Const(Literal::F32(EPS)),          // 11
            Op::Add(10, 11),                       // 12
            Op::Rsqrt(12),                         // 13
            Op::Broadcast {
                value: 13,
                shape: Shape::vector(WIDTH),
            }, // 14
            Op::Mul(7, 14),                        // 15
            Op::Mul(15, 1),                        // 16
            Op::Add(16, 2),                        // 17
        ],
        results: vec![17],
    }
}

/// A row of bf16-representable values, as a plane would hold them.
fn plane(seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9).wrapping_add(1);
    (0..WIDTH)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            bf16(unit * 4.0 - 2.0)
        })
        .collect()
}

fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

fn floats(value: &Value) -> &[f32] {
    match value {
        Value::F32(x) => x,
        other => panic!("an f32 result, not {other:?}"),
    }
}

#[test]
fn a_region_types_its_results_after_its_parameters() {
    let region = residual_add_rmsnorm();
    let types = region.types().expect("the region types");
    assert_eq!(types.len(), 3 + region.ops.len());
    assert_eq!(types[5].dims, [], "a row's sum is one value a row");
    assert_eq!(types[13], row());
    assert_eq!(region.width(), Ok(Some(WIDTH)));

    let mut unsized_ = residual_add_rmsnorm();
    unsized_.params[0].dims = vec![Dimension::Symbolic(SymbolicExtent::RowCount)];
    assert_eq!(unsized_.types(), Err(RegionError::Symbolic { param: 0 }));

    let mut impure = residual_add_rmsnorm();
    impure.ops.push(Op::ChanRead(0));
    assert_eq!(impure.types(), Err(RegionError::Impure { op: 11 }));

    let mut matrix = residual_add_rmsnorm();
    matrix.params[2].dims = vec![Dimension::Static(2), Dimension::Static(WIDTH)];
    assert!(matches!(matrix.types(), Err(RegionError::Untyped(_))));
}

#[test]
fn the_interpreter_runs_a_region_one_row_at_a_time() {
    let (x, y, w) = (plane(1), plane(2), plane(3));
    let out = residual_add_rmsnorm()
        .eval(&[
            Value::F32(x.clone()),
            Value::F32(y.clone()),
            Value::F32(w.clone()),
        ])
        .expect("the region runs");
    let h: Vec<f32> = y.iter().zip(&x).map(|(y, x)| y + x).collect();
    assert_eq!(
        floats(&out[1]),
        h,
        "result 1 is the stream plus the residual"
    );
    let ss: f32 = h.iter().map(|v| v * v).sum();
    let r = 1.0 / (ss / WIDTH as f32 + EPS).sqrt();
    for (e, got) in floats(&out[0]).iter().enumerate() {
        let want = h[e] * r * w[e];
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{e}: {got} vs {want}"
        );
    }
    assert!(matches!(
        residual_add_rmsnorm().eval(&[Value::F32(x)]),
        Err(RegionError::Arguments { .. })
    ));
}

#[test]
fn a_residual_norm_body_keeps_its_row_across_the_reduction() {
    let body = residual_add_rmsnorm()
        .body()
        .expect("the stream core covers it");
    let s = &body.source;
    assert_eq!(body.passes, 2, "{s}");
    assert!(s.contains("struct Body {"), "{s}");
    assert!(s.contains("void operator()(F& f) const"), "{s}");
    assert!(
        s.contains("v3[e] = f.st(1, e, (f.ld(1, e) + f.ld(0, e)));"),
        "{s}"
    );
    assert!(
        s.contains("float v3[F::N];"),
        "the stored stream is kept: {s}"
    );
    assert!(s.contains("if (f.valid(e)) a5 = a5 + v4;"), "{s}");
    let flushed = s.find("f.template flush<1>();").expect("result 1 flushes");
    let summed = s.find("f.reduce_sum(a5)").expect("the squares are summed");
    assert!(flushed < summed, "the stream leaves before the sum: {s}");
    assert!(s.contains("f.st(0, e, (v12 * f.ld(2, e)))"), "{s}");
    assert_eq!(
        s.matches("for (int e = 0; e < F::N; ++e)").count(),
        2,
        "{s}"
    );
    assert_eq!(body.hash, eta_ir::fnv1a64(s.as_bytes()));
}

#[test]
fn a_layernorm_body_passes_over_its_row_once_a_reduction() {
    let body = layernorm().body().expect("the stream core covers it");
    assert_eq!(body.passes, 3, "{}", body.source);
    assert!(body.source.contains("float v7[F::N];"), "{}", body.source);
}

#[test]
fn an_op_no_hook_serves_leaves_the_region_without_a_body() {
    let mut argmax = residual_add_rmsnorm();
    argmax.ops.push(Op::ReduceArgmax(3));
    assert!(argmax.body().is_err());

    let scalar = Region {
        params: vec![
            row(),
            SymbolicType {
                dtype: Dtype::F32,
                dims: vec![],
            },
        ],
        ops: vec![
            Op::Broadcast {
                value: 1,
                shape: Shape::vector(WIDTH),
            },
            Op::Mul(0, 2),
        ],
        results: vec![3],
    };
    assert!(
        scalar
            .eval(&[Value::F32(plane(1)), Value::F32(vec![2.0])])
            .is_ok()
    );
    let refused = scalar.body().expect_err("a frame's planes are rows");
    assert_eq!(refused.op, None);

    let sum_out = Region {
        params: vec![row()],
        ops: vec![Op::ReduceSum(0)],
        results: vec![1],
    };
    assert!(sum_out.body().is_err(), "a scalar result is no plane");
}

/// The host stand-in for a row frame: one thread holds the row, padded past
/// it with NaNs, and `st` rounds to bf16 as a plane does.
const HOST_FRAME: &str = r#"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define __device__
#define __forceinline__ inline
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static float bf16(float v) {
  uint32_t u; memcpy(&u, &v, 4);
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  float r; memcpy(&r, &u, 4); return r;
}
template <int N_, int IN, int OUT> struct HostFrame {
  static constexpr int N = N_;
  int width;
  float in[IN][N_];
  float out[OUT][N_];
  float ld(int k, int e) const { return in[k][e]; }
  float st(int k, int e, float v) { out[k][e] = bf16(v); return out[k][e]; }
  bool valid(int e) const { return e < width; }
  float reduce_sum(float v) { return v; }
  float reduce_max(float v) { return v; }
  template <int K> void flush() {}
};
"#;

/// `region`'s body compiled for the host and run over `args`, each result's
/// row as the frame stores it; `None` where no C++ compiler answers.
fn run_on_host(region: &Region, args: &[Vec<f32>], name: &str) -> Option<Vec<Vec<f32>>> {
    let body = region.body().expect("the stream core covers it");
    let (ins, outs, padded) = (args.len(), region.results.len(), WIDTH + 8);
    let mut src = String::from(HOST_FRAME);
    src.push_str(&body.source);
    src.push_str(&format!(
        "int main() {{\n  static HostFrame<{padded}, {ins}, {outs}> f;\n  f.width = {WIDTH};\n"
    ));
    for (k, row) in args.iter().enumerate() {
        for e in 0..padded as usize {
            let v = row.get(e).map_or("NAN".to_string(), |v| format!("{v:?}f"));
            src.push_str(&format!("  f.in[{k}][{e}] = {v};\n"));
        }
    }
    src.push_str("  Body{}(f);\n");
    src.push_str(&format!(
        "  for (int k = 0; k < {outs}; ++k) for (int e = 0; e < {WIDTH}; ++e) printf(\"%a\\n\", f.out[k][e]);\n  return 0;\n}}\n"
    ));
    let dir = std::env::temp_dir().join(format!("eta-region-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch dir");
    let (cpp, exe) = (dir.join("body.cpp"), dir.join("body"));
    std::fs::write(&cpp, &src).expect("the harness");
    let built = Command::new("c++")
        .args(["-std=c++17", "-O1", "-ffp-contract=off", "-o"])
        .arg(&exe)
        .arg(&cpp)
        .output();
    let built = match built {
        Ok(built) => built,
        Err(_) => {
            eprintln!("not asked: no c++ answers");
            return None;
        }
    };
    assert!(
        built.status.success(),
        "the body compiles: {}\n{src}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().expect("the harness runs");
    let _ = std::fs::remove_dir_all(&dir);
    let printed = String::from_utf8(ran.stdout).expect("utf-8");
    let all: Vec<f32> = printed.lines().map(hex_float).collect();
    Some(all.chunks(WIDTH as usize).map(<[f32]>::to_vec).collect())
}

/// A `%a` float back to its value.
fn hex_float(s: &str) -> f32 {
    let (sign, s) = s.strip_prefix('-').map_or((1.0, s), |rest| (-1.0, rest));
    let s = s.strip_prefix("0x").expect("a hex float");
    let (mantissa, exp) = s.split_once('p').expect("an exponent");
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut m = f64::from(u32::from_str_radix(whole, 16).expect("hex"));
    for (i, c) in frac.chars().enumerate() {
        let digit = f64::from(c.to_digit(16).expect("hex"));
        m += digit / 16f64.powi(i as i32 + 1);
    }
    (sign * m * 2f64.powi(exp.parse::<i32>().expect("exp"))) as f32
}

#[test]
fn the_body_lands_what_the_interpreter_lands() {
    for (name, region, args) in [
        (
            "rms",
            residual_add_rmsnorm(),
            vec![plane(1), plane(2), plane(3)],
        ),
        ("ln", layernorm(), vec![plane(4), plane(5), plane(6)]),
    ] {
        let Some(stored) = run_on_host(&region, &args, name) else {
            return;
        };
        let want = region
            .eval(&args.iter().cloned().map(Value::F32).collect::<Vec<_>>())
            .expect("the region runs");
        for (k, (got, want)) in stored.iter().zip(&want).enumerate() {
            for (e, (got, want)) in got.iter().zip(floats(want)).enumerate() {
                // A stored bf16 against the interpreter's f32, the
                // reduction's order apart: a bf16 ulp and a little.
                let tolerance = 1.0 / 64.0 * want.abs().max(0.25);
                assert!(
                    (got - want).abs() <= tolerance,
                    "{name} result {k} element {e}: {got} vs {want}"
                );
            }
        }
    }
}
