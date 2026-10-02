//! Probes of CSL integer semantics the eta guest lowering relies on, checked
//! on the simulator against plain Rust: u32 wrap-around, 64-bit integers,
//! shifts, i32 division, float/int conversions.

use engine_cerebras::bench::Bench;

#[test]
fn integer_arithmetic_wraps_shifts_and_converts_like_rust() {
    let mut b = Bench::new();
    let xs: [u32; 4] = [0xFFFF_FFFF, 0x8000_0000, 7, 0x1234_5678];
    let x = b.i32(4, 1, &xs.map(|v| v as i32));
    let fs: [f32; 6] = [3.7, -3.7, 3.0e9, -3.0e9, f32::NAN, 2.5e9];
    let f = b.f32(6, 1, &fs);
    let out = b.zeros(dtype::Dtype::I32, 1, 40);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let xb = cx.read(x)?;
                let fb = cx.read(f)?;
                let ob = cx.write(out)?;
                let o = ob.name.clone();
                let xn = xb.name.clone();
                let fn_ = fb.name.clone();
                // u32 wrap-around on add and mul.
                cx.line(format!("var a: u32 = @bitcast(u32, {xn}[0]); var bb: u32 = @bitcast(u32, {xn}[3]);"));
                cx.line(format!("{o}[0] = @bitcast(i32, a + 2);"));
                cx.line(format!("{o}[1] = @bitcast(i32, bb * bb);"));
                cx.line(format!("{o}[2] = @bitcast(i32, a << 4);"));
                cx.line(format!("{o}[3] = @bitcast(i32, a >> 4);"));
                cx.line(format!("{o}[4] = @bitcast(i32, (bb ^ a) & 0x0F0F0F0F);"));
                // i32 shifts and compares.
                cx.line(format!("var s: i32 = {xn}[1];"));
                cx.line(format!("{o}[5] = s >> 4;"));
                cx.line(format!("{o}[6] = @bitcast(i32, @bitcast(u32, s) >> 4);"));
                cx.line(format!("if (s < 0) {{ {o}[7] = 1; }} else {{ {o}[7] = 0; }}"));
                cx.line(format!("if (@bitcast(u32, s) > 5) {{ {o}[8] = 1; }} else {{ {o}[8] = 0; }}"));
                // i32 division and remainder (negative operands).
                cx.line(format!("var seven: i32 = {xn}[2];"));
                cx.line(format!("{o}[9] = (-seven) / 2;"));
                cx.line(format!("{o}[10] = (-seven) % 2;"));
                cx.line(format!("{o}[11] = seven / (-2);"));
                cx.line(format!("{o}[12] = seven % (-2);"));
                cx.line(format!("{o}[13] = @bitcast(i32, a / 3);"));
                cx.line(format!("{o}[14] = @bitcast(i32, a % 3);"));
                // u64 arithmetic.
                cx.line("var w: u64 = @as(u64, bb) * @as(u64, a);");
                cx.line(format!("{o}[15] = @bitcast(i32, @as(u32, w & 0xFFFFFFFF));"));
                cx.line(format!("{o}[16] = @bitcast(i32, @as(u32, w >> 32));"));
                cx.line("var m: u64 = 0x9E3779B97F4A7C15;");
                cx.line("var z: u64 = (w ^ (w >> 27)) * m;");
                cx.line(format!("{o}[17] = @bitcast(i32, @as(u32, z & 0xFFFFFFFF));"));
                cx.line(format!("{o}[18] = @bitcast(i32, @as(u32, z >> 32));"));
                cx.line("var q: u64 = @as(u64, a) + @as(u64, a) * 0xFFFFFFFF;");
                cx.line(format!("{o}[19] = @bitcast(i32, @as(u32, q >> 40));"));
                // Conversions.
                cx.line(format!("{o}[20] = @as(i32, {fn_}[0]);"));
                cx.line(format!("{o}[21] = @as(i32, {fn_}[1]);"));
                cx.line(format!("{o}[22] = @bitcast(i32, @as(u32, {fn_}[0]));"));
                cx.line(format!("{o}[23] = @bitcast(i32, @bitcast(u32, @as(f32, a)));"));
                cx.line(format!("{o}[24] = @bitcast(i32, @bitcast(u32, @as(f32, s)));"));
                cx.line(format!("{o}[25] = @bitcast(i32, @bitcast(u32, @as(f32, @as(u64, a) * 16)));"));
                cx.line(format!("var big: f32 = {fn_}[5];"));
                cx.line("var half: f32 = 2147483648.0; var g26: u32 = 0;");
                cx.line("if (big >= half) { g26 = @as(u32, @as(i32, big - half)) + 0x80000000; } else { g26 = @as(u32, big); }");
                cx.line(format!("{o}[26] = @bitcast(i32, g26);"));
                cx.line(format!("{o}[27] = @bitcast(i32, @bitcast(u32, @as(f32, g26)));"));
                // Float remainder and sign-aware ops.
                cx.line(format!("{o}[28] = @bitcast(i32, @bitcast(u32, math.abs_f32({fn_}[1])));"));
                cx.line(format!("{o}[29] = @bitcast(i32, @bitcast(u32, math.floor_f32({fn_}[1])));"));
                cx.line(format!("if (math.isNaN_f32({fn_}[4])) {{ {o}[30] = 1; }} else {{ {o}[30] = 0; }}"));
                cx.line(format!("if ({fn_}[4] == {fn_}[4]) {{ {o}[31] = 1; }} else {{ {o}[31] = 0; }}"));
                cx.line(format!("if ({fn_}[4] > 0.0 or {fn_}[4] <= 0.0) {{ {o}[32] = 1; }} else {{ {o}[32] = 0; }}"));
                cx.line(format!("var x15: f32 = 1.5; {o}[33] = @bitcast(i32, @bitcast(u32, math.exp_f32(x15)));"));
                cx.line(format!("{o}[34] = @bitcast(i32, @bitcast(u32, math.log_f32(x15)));"));
                cx.line(format!("{o}[35] = @bitcast(i32, @bitcast(u32, math.sin_f32(x15)));"));
                cx.line(format!("{o}[36] = @bitcast(i32, @bitcast(u32, math.cos_f32(x15)));"));
                cx.line(format!("{o}[37] = @bitcast(i32, @bitcast(u32, math.sqrt_f32(x15)));"));
                cx.line(format!("var one: f32 = 1.0; var three: f32 = 3.0; {o}[38] = @bitcast(i32, @bitcast(u32, one / three));"));
                cx.line(format!("var big_i: i32 = 16777217; {o}[39] = @bitcast(i32, @bitcast(u32, @as(f32, big_i)));"));
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let got = b.read_i32(out);
    let a = xs[0];
    let bb = xs[3];
    let s = xs[1] as i32;
    let w = u64::from(bb) * u64::from(a);
    let m = 0x9E37_79B9_7F4A_7C15u64;
    let z = (w ^ (w >> 27)).wrapping_mul(m);
    let q = u64::from(a) + u64::from(a) * 0xFFFF_FFFF;
    let want: Vec<i32> = vec![
        a.wrapping_add(2) as i32,
        bb.wrapping_mul(bb) as i32,
        (a << 4) as i32,
        (a >> 4) as i32,
        ((bb ^ a) & 0x0F0F_0F0F) as i32,
        s >> 4,
        ((s as u32) >> 4) as i32,
        i32::from(s < 0),
        i32::from((s as u32) > 5),
        -7 / 2,
        -7 % 2,
        7 / -2,
        7 % -2,
        (a / 3) as i32,
        (a % 3) as i32,
        (w & 0xFFFF_FFFF) as u32 as i32,
        (w >> 32) as u32 as i32,
        (z & 0xFFFF_FFFF) as u32 as i32,
        (z >> 32) as u32 as i32,
        (q >> 40) as u32 as i32,
        3.7f32 as i32,
        -3.7f32 as i32,
        3.7f32 as u32 as i32,
        (a as f32).to_bits() as i32,
        (s as f32).to_bits() as i32,
        ((u64::from(a) * 16) as f32).to_bits() as i32,
        2.5e9f32 as u32 as i32,
        (2.5e9f32 as u32 as f32).to_bits() as i32,
        3.7f32.to_bits() as i32,
        (-4.0f32).to_bits() as i32,
        1,
        0,
        0,
        1.5f32.exp().to_bits() as i32,
        1.5f32.ln().to_bits() as i32,
        1.5f32.sin().to_bits() as i32,
        1.5f32.cos().to_bits() as i32,
        1.5f32.sqrt().to_bits() as i32,
        (1.0f32 / 3.0).to_bits() as i32,
        (16_777_217i32 as f32).to_bits() as i32,
    ];
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        eprintln!(
            "[{i:2}] got {g:#010x} want {w:#010x}{}",
            if g == w { "" } else { "  <-- differs" }
        );
    }
    // The transcendental slots may differ in the last bits; everything else
    // is exact.
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        // Float conversions above 2^31 saturate in `@as(u32, f32)`; the
        // guarded form in slot 26 is what the lowering emits.
        if (33..=36).contains(&i) {
            let (gf, wf) = (f32::from_bits(*g as u32), f32::from_bits(*w as u32));
            assert!(
                (gf - wf).abs() <= 1e-6 * wf.abs().max(1.0),
                "slot {i}: {gf} vs {wf}"
            );
        } else {
            assert_eq!(g, w, "slot {i}");
        }
    }
}
