//! Probes of CSL control-flow and integer semantics the kernels rely on,
//! checked on the simulator against plain Rust.
//!
//! Known hazard (SDK 2.10): in the paged attention kernel's key loop (a
//! `while` two loops deep in a large body) a scalar read `mask[r * w + kp]`
//! resolved to the wrong row, while the same expression in the small bodies
//! below is correct. Kernels copy a row out with a DSD move and index the
//! copy by the innermost variable alone; the probes here pin the forms that
//! are known to work.

use engine_cerebras::bench::{Bench, assert_close};

#[test]
fn while_continue_division_and_computed_indexing_behave() {
    let rows = 4u32;
    let width = 6u32;
    let mask: Vec<u8> = (0..rows * width)
        .map(|i| ((i * 7 + 3) % 5 != 0) as u8)
        .collect();
    let hi = [3i32, 6, 5, 6];
    let mut b = Bench::new();
    let m = b.u8(rows, width, &mask);
    let h = b.i32(rows, 1, &hi);
    // Per row: count of admitted kp, sum of kp / 2, sum of kp % 4, sum of admitted kp.
    let out = b.zeros(dtype::Dtype::F32, rows, 4);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let mb = cx.read(m)?;
                let hb = cx.read(h)?;
                let ob = cx.write(out)?;
                cx.for_range("r", rows, |blk| {
                    blk.line(format!("var hi: i32 = {}[r];", hb.name));
                    blk.line("var count: f32 = 0.0;");
                    blk.line("var divs: f32 = 0.0;");
                    blk.line("var mods: f32 = 0.0;");
                    blk.line("var sum: f32 = 0.0;");
                    blk.line("var kp: i32 = 0;");
                    blk.nest("while (kp < hi) : (kp += 1) {", "}", |blk| {
                        blk.line("if (kp < 0) { continue; }");
                        blk.line(format!(
                            "if (hi != 0) {{ if (kp >= {width}) {{ continue; }} if ({}[@as(i32, r) * {width} + kp] == 0) {{ continue; }} }}",
                            mb.name
                        ));
                        blk.line("count = count + 1.0;");
                        blk.line("divs = divs + @as(f32, kp / 2);");
                        blk.line("mods = mods + @as(f32, kp % 4);");
                        blk.line("sum = sum + @as(f32, kp);");
                    });
                    blk.line(format!("{}[@as(i32, r) * 4 + 0] = count;", ob.name));
                    blk.line(format!("{}[@as(i32, r) * 4 + 1] = divs;", ob.name));
                    blk.line(format!("{}[@as(i32, r) * 4 + 2] = mods;", ob.name));
                    blk.line(format!("{}[@as(i32, r) * 4 + 3] = sum;", ob.name));
                });
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want = Vec::new();
    for r in 0..rows as usize {
        let (mut c, mut d, mut mo, mut s) = (0.0, 0.0, 0.0, 0.0);
        for kp in 0..hi[r] {
            if kp >= width as i32 || mask[r * width as usize + kp as usize] == 0 {
                continue;
            }
            c += 1.0;
            d += (kp / 2) as f32;
            mo += (kp % 4) as f32;
            s += kp as f32;
        }
        want.extend([c, d, mo, s]);
    }
    let got = b.read_f32(out);
    eprintln!("got  {got:?}\nwant {want:?}");
    assert_close(&got, &want, 0.0, 0.0);
}

#[test]
fn a_hoisted_row_offset_survives_three_loop_levels() {
    let (rows, heads, width) = (4u32, 2u32, 6u32);
    let mask: Vec<u8> = (0..rows * width)
        .map(|i| ((i * 7 + 3) % 5 != 0) as u8)
        .collect();
    let mut b = Bench::new();
    let m = b.u8(rows, width, &mask);
    let out = b.zeros(dtype::Dtype::F32, rows, heads);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let mb = cx.read(m)?;
                let ob = cx.write(out)?;
                cx.for_range("r", rows, |blk| {
                    blk.line(format!("var mrow: i32 = @as(i32, r) * {width};"));
                    blk.line(format!("var obase: i32 = @as(i32, r) * {heads};"));
                    blk.for_range("h", heads, |blk| {
                        blk.line("var count: f32 = 0.0;");
                        blk.line("var kp: i32 = 0;");
                        blk.nest(
                            &format!("while (kp < {width}) : (kp += 1) {{"),
                            "}",
                            |blk| {
                                blk.line(format!(
                                    "if ({}[mrow + kp] == 0) {{ continue; }}",
                                    mb.name
                                ));
                                blk.line("count = count + 1.0 + @as(f32, h);");
                            },
                        );
                        blk.line(format!("{}[obase + @as(i32, h)] = count;", ob.name));
                    });
                });
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            let ones = mask[r * width as usize..(r + 1) * width as usize]
                .iter()
                .filter(|m| **m != 0)
                .count() as f32;
            (0..heads).map(move |h| ones * (1.0 + h as f32))
        })
        .collect();
    assert_close(&b.read_f32(out), &want, 0.0, 0.0);
}

/// How a DSD reduction into a scalar behaves: sums of a vector (and of
/// products just written to scratch) by `@fadds(&s, s, dsd)` against the
/// plain sum, with the accumulator local or global, once or repeated.
#[test]
fn dsd_reductions_into_a_scalar() {
    let n = 300u32;
    let xs: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 + 1.0).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.25 + 1.0).collect();
    let mut b = Bench::new();
    let x = b.f32(1, n, &xs);
    let y = b.f32(1, n, &ys);
    let out = b.zeros(dtype::Dtype::F32, 1, 8);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let xb = cx.read(x)?;
                let yb = cx.read(y)?;
                let ob = cx.write(out)?;
                let tmp = cx.scratch("tmp", u64::from(n));
                cx.program().global("var g_acc: f32 = 0.0;".to_string());
                cx.line(format!(
                    "var dx = @set_dsd_base_addr(k_base_f32, {});",
                    xb.ptr()
                ));
                cx.line(format!("dx = @set_dsd_length(dx, {n});"));
                cx.line(format!(
                    "var dy = @set_dsd_base_addr(k_base_f32, {});",
                    yb.ptr()
                ));
                cx.line(format!("dy = @set_dsd_length(dy, {n});"));
                cx.line(format!(
                    "var dt = @set_dsd_base_addr(k_base_f32, @ptrcast([*]f32, &{tmp}));"
                ));
                cx.line(format!("dt = @set_dsd_length(dt, {n});"));
                // 0: sum of x, local accumulator.
                cx.line("var s0: f32 = 0.0;");
                cx.line("@fadds(&s0, s0, dx);");
                cx.line(format!("{}[0] = s0;", ob.name));
                // 1: sum of x, global accumulator.
                cx.line("g_acc = 0.0;");
                cx.line("@fadds(&g_acc, g_acc, dx);");
                cx.line(format!("{}[1] = g_acc;", ob.name));
                // 2: sum of x, local, called three times.
                cx.line("var s2: f32 = 0.0;");
                cx.line("@fadds(&s2, s2, dx);");
                cx.line("@fadds(&s2, s2, dx);");
                cx.line("@fadds(&s2, s2, dx);");
                cx.line(format!("{}[2] = s2;", ob.name));
                // 3: products into scratch, then the sum, local.
                cx.line("@fmuls(dt, dx, dy);");
                cx.line("var s3: f32 = 0.0;");
                cx.line("@fadds(&s3, s3, dt);");
                cx.line(format!("{}[3] = s3;", ob.name));
                // 4: the same, global.
                cx.line("g_acc = 0.0;");
                cx.line("@fadds(&g_acc, g_acc, dt);");
                cx.line(format!("{}[4] = g_acc;", ob.name));
                // 5: plain loop over the products.
                cx.line("var s5: f32 = 0.0;");
                cx.line(format!(
                    "var i: i32 = 0; while (i < {n}) : (i += 1) {{ s5 = s5 + {tmp}[i]; }}"
                ));
                cx.line(format!("{}[5] = s5;", ob.name));
                // 6: local accumulator initialised from a global read (not a constant).
                cx.line("var s6: f32 = g_acc * 0.0;");
                cx.line("@fadds(&s6, s6, dx);");
                cx.line(format!("{}[6] = s6;", ob.name));
                // 7: sum of x with a 1.0 start.
                cx.line("var s7: f32 = 1.0;");
                cx.line("@fadds(&s7, s7, dx);");
                cx.line(format!("{}[7] = s7;", ob.name));
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let sum_x: f32 = xs.iter().sum();
    let dot: f32 = xs.iter().zip(&ys).map(|(a, b)| a * b).sum();
    let got = b.read_f32(out);
    eprintln!(
        "sum x {sum_x}: local {} global {} thrice {} | dot {dot}: local {} global {} loop {} | from-global-init {} start-1 {}",
        got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7]
    );
    assert_close(&got[5..6], &[dot], 1e-2, 1e-4);
}
