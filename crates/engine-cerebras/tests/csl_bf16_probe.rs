//! Probes of CSL's native `bf16` (the runtime half type under
//! `--fp16-format=bf16`; `f16` is then comptime-only): storing f32 values
//! as bf16 and reading them back, and what a DSD move from a bf16 array
//! into f32 does (it copies raw halves). The device-side half storage of
//! weights depends on these.

use engine_cerebras::bench::Bench;

#[test]
fn bf16_arrays_round_trip_through_f32() {
    let mut b = Bench::new();
    let fs: [f32; 4] = [1.0, -2.5, 3.140_7, 65504.0];
    let f = b.f32(4, 1, &fs);
    let out = b.zeros(dtype::Dtype::F32, 1, 12);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let fb = cx.read(f)?;
                let ob = cx.write(out)?;
                let (o, fn_) = (ob.name.clone(), fb.name.clone());
                cx.program().global("var hb: [4]bf16;");
                cx.program().global("var hw: [4]bf16;");
                cx.line("var i: i32 = 0;");
                cx.line(format!("while (i < 4) : (i += 1) {{ hb[i] = @as(bf16, {fn_}[i]); hw[i] = @as(bf16, 2.0); }}"));
                cx.line("i = 0;");
                cx.line(format!("while (i < 4) : (i += 1) {{ {o}[i] = @as(f32, hb[i]); }}"));
                // bf16 arithmetic widened to f32.
                cx.line(format!("{o}[4] = @as(f32, hb[0]) * @as(f32, hw[0]) + @as(f32, hb[1]);"));
                // A DSD over a bf16 array moved into f32 copies the raw
                // halves, two a word: a DSD does not convert (found here),
                // so a half array is widened element by element.
                cx.line("const dsb = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{4} -> hb[j] });");
                cx.line(format!("const dso = @get_dsd(mem1d_dsd, .{{ .tensor_access = |j|{{4}} -> {o}[j + 5] }});"));
                cx.line("@fmovs(dso, dsb);");
                cx.line(format!("{o}[9] = @as(f32, @bitcast(u16, hb[2]));"));
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let got = b.read_f32(out);
    let round = |v: f32| f32::from_bits((v.to_bits() + 0x7FFF + ((v.to_bits() >> 16) & 1)) & 0xFFFF_0000);
    for (i, v) in fs.iter().enumerate() {
        assert_eq!(got[i], round(*v), "slot {i}");
    }
    assert_eq!(got[4], round(1.0) * 2.0 + round(-2.5));
    // The raw move: word 0 holds halves 0 and 1 (little end first).
    let half = |v: f32| round(v).to_bits() >> 16;
    assert_eq!(got[5].to_bits(), half(fs[0]) | (half(fs[1]) << 16), "fmovs packs two halves a word");
    assert_eq!(got[9], half(fs[2]) as f32);
}
