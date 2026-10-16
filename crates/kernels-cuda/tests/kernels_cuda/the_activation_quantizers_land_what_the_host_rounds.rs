#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{
    AlphaFrom, Bank, Format, Source, Weight, held_planes, matmul, matmul_held, quantize, read_back,
    scales_to_atoms, select,
};
use kernels_cuda::linear::tc5::{Elem, atom_bytes, atom_offset, supported};
use kernels_cuda::tensor::Tensor;

const OP: &str = "linear.act_quant";

fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((code >> 3) & 0xf);
    let man = f32::from(code & 7);
    sign * if exp == 0 {
        man / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// The nearest of `codes` positive codes under `value`, ties to the even
/// code, saturating at the largest; the sign in `sign`, kept on a zero as
/// the hardware's conversions keep it.
fn nearest(v: f32, codes: u8, value: impl Fn(u8) -> f32, sign: u8) -> u8 {
    let a = v.abs();
    let mut best = 0u8;
    for c in 1..codes {
        let (d, db) = ((value(c) - a).abs(), (value(best) - a).abs());
        if d < db || (d == db && c % 2 == 0) {
            best = c;
        }
    }
    if v < 0.0 { best | sign } else { best }
}

fn to_e4m3(v: f32) -> u8 {
    nearest(v, 0x7f, e4m3, 0x80)
}

fn to_e2m1(v: f32) -> u8 {
    nearest(v, 8, e2m1, 8)
}

/// `rows x k` bf16 activations, each row at its own magnitude.
fn activations(lcg: &mut Lcg, rows: usize, k: usize) -> (Vec<u16>, Vec<f32>) {
    let mut raw = Vec::with_capacity(rows * k);
    let mut exact = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let gain = 2f32.powi(r as i32 % 9 - 4);
        for _ in 0..k {
            let v = from_bf16(to_bf16(lcg.unit() * gain));
            raw.push(to_bf16(v));
            exact.push(v);
        }
    }
    (raw, exact)
}

/// DeepSeek's `act_quant`: the codes and the ue8m0 scale of one block.
fn host_e4m3(block: &[f32]) -> (Vec<u8>, u8) {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
    let bits = (amax * (1.0 / 448.0)).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    let s = 2f32.powi(e);
    let codes = block
        .iter()
        .map(|v| to_e4m3((v / s).clamp(-448.0, 448.0)))
        .collect();
    (codes, (e + 127) as u8)
}

/// The NVFP4 recipe, exact reciprocals where the device's are approximate.
fn host_nvfp4(block: &[f32], gs: f32) -> (Vec<u8>, u8) {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sf = to_e4m3(gs * (amax * (1.0 / 6.0)));
    let sv = e4m3(sf);
    let out = if sv != 0.0 {
        1.0 / (sv * (1.0 / gs))
    } else {
        0.0
    };
    (block.iter().map(|v| to_e2m1(v * out)).collect(), sf)
}

fn e4m3_codes_and_atoms(gran: usize) {
    let (rows, k) = (37, 512);
    let mut lcg = Lcg::seeded(0xa8 + gran as u64);
    let (raw, x) = activations(&mut lcg, rows, k);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let codes_at = gpu.zeros(rows * k);
    let atoms_at = gpu.zeros(atom_bytes(rows, k, gran));
    let values_at = gpu.zeros(rows * k * 2);
    let ctx = gpu.ctx();
    let fmt = Format::E4m3 { gran: gran as u32 };
    let src = Source::rows(Tensor::new(x_at, rows as u32, k as u32, Dtype::Bf16), None);
    quantize(
        &ctx,
        OP,
        fmt,
        src,
        codes_at,
        atoms_at,
        atom_bytes(rows, k, gran),
    )
    .expect("the quantizer fires");
    read_back(&ctx, OP, fmt, src, values_at).expect("the read-back fires");
    gpu.sync();
    let codes: Vec<u8> = gpu.down(codes_at, rows * k);
    let atoms: Vec<u8> = gpu.down(atoms_at, atom_bytes(rows, k, gran));
    let values: Vec<u16> = gpu.down(values_at, rows * k);
    let cols = k / gran;
    for r in 0..rows {
        for b in 0..cols {
            let at = r * k + b * gran;
            let (want, scale) = host_e4m3(&x[at..at + gran]);
            assert_eq!(
                atoms[atom_offset(r, b, cols)],
                scale,
                "row {r} block {b}'s scale"
            );
            assert_eq!(
                &codes[at..at + gran],
                &want[..],
                "row {r} block {b}'s codes"
            );
            for i in 0..gran {
                let v = e4m3(want[i]) * 2f32.powi(i32::from(scale) - 127);
                assert_eq!(
                    from_bf16(values[at + i]),
                    v,
                    "row {r} element {}",
                    b * gran + i
                );
            }
        }
    }
}

fn nvfp4_codes_and_atoms() {
    let (rows, k) = (45, 256);
    let mut lcg = Lcg::seeded(0x4a4);
    let (raw, x) = activations(&mut lcg, rows, k);
    let input_scale = [0.0123f32];
    let gs = 1.0 / input_scale[0];
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let is_at = gpu.up(&input_scale);
    let codes_at = gpu.zeros(rows * k / 2);
    let atoms_at = gpu.zeros(atom_bytes(rows, k, 16));
    let values_at = gpu.zeros(rows * k * 2);
    let ctx = gpu.ctx();
    let src = Source::rows(
        Tensor::new(x_at, rows as u32, k as u32, Dtype::Bf16),
        Some(Tensor::new(is_at, 1, 1, Dtype::F32)),
    );
    quantize(
        &ctx,
        OP,
        Format::Nvfp4,
        src,
        codes_at,
        atoms_at,
        atom_bytes(rows, k, 16),
    )
    .expect("the quantizer fires");
    read_back(&ctx, OP, Format::Nvfp4, src, values_at).expect("the read-back fires");
    gpu.sync();
    let codes: Vec<u8> = gpu.down(codes_at, rows * k / 2);
    let atoms: Vec<u8> = gpu.down(atoms_at, atom_bytes(rows, k, 16));
    let values: Vec<u16> = gpu.down(values_at, rows * k);
    let cols = k / 16;
    let (mut off_scales, mut off_codes) = (0, 0);
    for r in 0..rows {
        for b in 0..cols {
            let at = r * k + b * 16;
            let (want, sf) = host_nvfp4(&x[at..at + 16], gs);
            let got_sf = atoms[atom_offset(r, b, cols)];
            off_scales += usize::from(got_sf != sf);
            for i in 0..16 {
                let byte = codes[(at + i) / 2];
                let code = if i % 2 == 0 { byte & 0xf } else { byte >> 4 };
                off_codes += usize::from(code != want[i] && got_sf == sf);
                // The read-back is the codes under their scale, exactly.
                assert_eq!(
                    from_bf16(values[at + i]),
                    e2m1(code) * e4m3(got_sf),
                    "row {r} element {}",
                    b * 16 + i
                );
            }
        }
    }
    // The device's approximate reciprocals may tip a tie the exact ones
    // round the other way.
    assert!(
        off_scales * 200 <= rows * cols,
        "{off_scales} of {} scales differ from the host's",
        rows * cols
    );
    assert!(
        off_codes * 200 <= rows * k,
        "{off_codes} of {} codes differ from the host's",
        rows * k
    );
    eprintln!("nvfp4: {off_scales} scales and {off_codes} codes off the exact recipe");
}

/// A route's activations are its token's, under its expert's global; a
/// dropped route lands zeros.
fn routed_nvfp4_reads_each_expert_global() {
    let (tokens, top_k, experts, k) = (5usize, 3usize, 4usize, 128usize);
    let mut lcg = Lcg::seeded(0x7e4);
    let (raw, _) = activations(&mut lcg, tokens, k);
    let input_scales = [0.01f32, 0.02, 0.005, 0.0371];
    let mut routes: Vec<i32> = (0..tokens * top_k)
        .map(|r| ((r * 7 + 1) % experts) as i32)
        .collect();
    routes[4] = -1;
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let is_at = gpu.up(&input_scales);
    let routes_at = gpu.up(&routes);
    let count = tokens * top_k;
    let routed_at = gpu.zeros(count * k * 2);
    let one_at = gpu.zeros(tokens * k * 2 * experts);
    let ctx = gpu.ctx();
    let x = Tensor::new(x_at, tokens as u32, k as u32, Dtype::Bf16);
    read_back(
        &ctx,
        OP,
        Format::Nvfp4,
        Source {
            x,
            rows: count as u32,
            div: top_k as u32,
            routes: Some(Tensor::new(
                routes_at,
                tokens as u32,
                top_k as u32,
                Dtype::I32,
            )),
            input_scales: Some(Tensor::new(is_at, experts as u32, 1, Dtype::F32)),
            pos: 0,
            m_max: 0,
        },
        routed_at,
    )
    .expect("the routed read-back fires");
    for e in 0..experts {
        let global = Tensor::new(is_at + 4 * e as u64, 1, 1, Dtype::F32);
        read_back(
            &ctx,
            OP,
            Format::Nvfp4,
            Source::rows(x, Some(global)),
            one_at + (e * tokens * k * 2) as u64,
        )
        .expect("the read-back fires");
    }
    gpu.sync();
    let routed: Vec<u16> = gpu.down(routed_at, count * k);
    let one: Vec<u16> = gpu.down(one_at, experts * tokens * k);
    for r in 0..count {
        let row = &routed[r * k..(r + 1) * k];
        if routes[r] < 0 {
            assert!(
                row.iter().all(|v| from_bf16(*v) == 0.0),
                "dropped route {r}"
            );
            continue;
        }
        let e = routes[r] as usize;
        let at = (e * tokens + r / top_k) * k;
        assert_eq!(row, &one[at..at + k], "route {r} (expert {e})");
    }
}

fn scales_lay_into_atoms() {
    let (rows, cols, div) = (300usize, 11usize, 128usize);
    let plane: Vec<u8> = (0..rows.div_ceil(div) * cols)
        .map(|i| (i * 37 % 251) as u8)
        .collect();
    let mut gpu = Gpu::open();
    let plane_at = gpu.up(&plane);
    let bytes = rows.div_ceil(128) * cols.div_ceil(4) * 512;
    let atoms_at = gpu.up(&vec![0xffu8; bytes]);
    let ctx = gpu.ctx();
    scales_to_atoms(
        &ctx,
        OP,
        plane_at,
        atoms_at,
        rows as u32,
        cols as u32,
        div as u32,
    )
    .expect("the layout fires");
    gpu.sync();
    let atoms: Vec<u8> = gpu.down(atoms_at, bytes);
    let mut want = vec![0u8; bytes];
    for r in 0..rows {
        for c in 0..cols {
            want[atom_offset(r, c, cols)] = plane[r / div * cols + c];
        }
    }
    assert_eq!(
        atoms, want,
        "every scale at its atom byte, the padding zero"
    );
}

/// e4m3 activations in `gran`s against MXFP4 codes on the tcgen05 GEMM: the
/// dot of the read-back activations and the dequantized weight.
fn w4a8_matmul(m: usize, n: usize, k: usize, gran: usize) {
    let mut lcg = Lcg::seeded(0x48 ^ (m * n + k + gran) as u64);
    let (raw, _) = activations(&mut lcg, m, k);
    let mut codes = vec![0u8; n * k / 2];
    let mut w = vec![0f32; n * k];
    let scales: Vec<u8> = (0..n * k / 32).map(|i| 121 + (i * 7 % 12) as u8).collect();
    for i in 0..n * k {
        let code = ((lcg.unit() + 1.0) * 8.0) as u8 & 0xf;
        codes[i / 2] |= code << (4 * (i % 2));
        w[i] = e2m1(code) * 2f32.powi(i32::from(scales[i / 32]) - 127);
    }
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let y_at = gpu.zeros(m * n * 2);
    let values_at = gpu.zeros(m * k * 2);
    let ctx = gpu.ctx();
    let fmt = Format::E4m3 { gran: gran as u32 };
    let src = Source::rows(Tensor::new(x_at, m as u32, k as u32, Dtype::Bf16), None);
    let mut y = Tensor::new(y_at, m as u32, n as u32, Dtype::Bf16);
    matmul(
        &ctx,
        OP,
        fmt,
        src,
        Weight {
            codes: codes_at,
            scales: scales_at,
            n: n as u32,
            k: k as u32,
            elem: Elem::E2m1,
            gran: 32,
            laid: false,
        },
        &mut y,
        1.0,
    )
    .expect("the W4A8 product fires");
    read_back(&ctx, OP, fmt, src, values_at).expect("the read-back fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, m * n);
    let a: Vec<f32> = gpu
        .down::<u16>(values_at, m * k)
        .into_iter()
        .map(from_bf16)
        .collect();
    let mut bad = 0;
    for r in 0..m {
        for c in 0..n {
            let want: f64 = (0..k)
                .map(|i| f64::from(a[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            let g = from_bf16(got[r * n + c]);
            if !close(g, want as f32) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("({r}, {c}): got {g} want {want}");
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} outputs off the read-back dot", m * n);
}

/// DeepSeek's W8A8: e4m3 activations in 1x128 blocks against e4m3 codes
/// under one ue8m0 exponent a 128x128 block, the block's exponent spread to
/// each of its rows' atoms.
fn w8a8_block_matmul(m: usize, n: usize, k: usize) {
    let mut lcg = Lcg::seeded(0x88 ^ (m * n + k) as u64);
    let (raw, _) = activations(&mut lcg, m, k);
    let (bn, bk) = (n.div_ceil(128), k.div_ceil(128));
    let blocks: Vec<u8> = (0..bn * bk).map(|i| 118 + (i * 5 % 9) as u8).collect();
    let mut codes = vec![0u8; n * k];
    let mut w = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let u = ((lcg.unit() + 1.0) * 128.0) as u32;
            let code = ((u & 0x80) | ((3 + (u >> 2) % 8) << 3) | (u & 7)) as u8;
            codes[r * k + c] = code;
            let s = blocks[(r / 128) * bk + c / 128];
            w[r * k + c] = e4m3(code) * 2f32.powi(i32::from(s) - 127);
        }
    }
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let codes_at = gpu.up(&codes);
    let blocks_at = gpu.up(&blocks);
    let atoms_at = gpu.zeros(atom_bytes(n, k, 128));
    let y_at = gpu.zeros(m * n * 2);
    let values_at = gpu.zeros(m * k * 2);
    let ctx = gpu.ctx();
    scales_to_atoms(&ctx, OP, blocks_at, atoms_at, n as u32, bk as u32, 128)
        .expect("the exponents spread");
    let fmt = Format::E4m3 { gran: 128 };
    let src = Source::rows(Tensor::new(x_at, m as u32, k as u32, Dtype::Bf16), None);
    let mut y = Tensor::new(y_at, m as u32, n as u32, Dtype::Bf16);
    matmul(
        &ctx,
        OP,
        fmt,
        src,
        Weight {
            codes: codes_at,
            scales: atoms_at,
            n: n as u32,
            k: k as u32,
            elem: Elem::E4m3,
            gran: 128,
            laid: true,
        },
        &mut y,
        1.0,
    )
    .expect("the W8A8 product fires");
    read_back(&ctx, OP, fmt, src, values_at).expect("the read-back fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, m * n);
    let a: Vec<f32> = gpu
        .down::<u16>(values_at, m * k)
        .into_iter()
        .map(from_bf16)
        .collect();
    let mut bad = 0;
    for r in 0..m {
        for c in 0..n {
            let want: f64 = (0..k)
                .map(|i| f64::from(a[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            let g = from_bf16(got[r * n + c]);
            if !close(g, want as f32) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("W8A8 ({r}, {c}): got {g} want {want}");
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad} of {} W8A8 outputs off the read-back dot",
        m * n
    );
    // A producer's codes in the held planes (here the quantizer's bytes,
    // which `e4m3_block_store` lands too) read by the GEMM with no
    // quantize launch: the same product, bit for bit.
    let planes = held_planes(&ctx, OP, m as u32, k as u32).expect("the held planes");
    let atom_bytes = atom_bytes(m, k, 128);
    quantize(&ctx, OP, fmt, src, planes[0], planes[1], atom_bytes).expect("the codes land");
    let held_at = gpu.zeros(m * n * 2);
    let mut held = Tensor::new(held_at, m as u32, n as u32, Dtype::Bf16);
    matmul_held(
        &ctx,
        OP,
        planes,
        m as u32,
        Weight {
            codes: codes_at,
            scales: atoms_at,
            n: n as u32,
            k: k as u32,
            elem: Elem::E4m3,
            gran: 128,
            laid: true,
        },
        &mut held,
        1.0,
    )
    .expect("the held product fires");
    gpu.sync();
    assert_eq!(
        gpu.down::<u16>(held_at, m * n),
        got,
        "a held W8A8 product lands the quantizing one's bits"
    );
}

#[test]
fn the_activation_quantizers_land_what_the_host_rounds_every_case() {
    e4m3_codes_and_atoms(128);
    e4m3_codes_and_atoms(32);
    nvfp4_codes_and_atoms();
    routed_nvfp4_reads_each_expert_global();
    scales_lay_into_atoms();
    if !supported() {
        eprintln!("skipped the tcgen05 products: no sm_100a or NVRTC 12.9 here");
        return;
    }
    w4a8_matmul(7, 256, 512, 128);
    w4a8_matmul(200, 384, 1024, 128);
    w4a8_matmul(64, 128, 256, 32);
    w8a8_block_matmul(5, 384, 640);
    w8a8_block_matmul(300, 256, 1024);
    for tokens in [3, 40, 150] {
        routed_select(Format::Nvfp4, tokens, true);
        routed_select(Format::E4m3 { gran: 128 }, tokens, true);
    }
    routed_select(Format::Nvfp4, 70, false);
    routed_select(Format::E4m3 { gran: 128 }, 5, false);
    // A captured fire's live window: the routes past it were never written
    // by this fire, so their ids are no expert's.
    for (tokens, live) in [(4, 3), (40, 17)] {
        routed_select_live(Format::Nvfp4, tokens, live);
        routed_select_live(Format::E4m3 { gran: 128 }, tokens, live);
    }
    // DeepSeek-V4-Flash's routed legs at one token.
    routed_select_at(
        Format::E4m3 { gran: 128 },
        1,
        true,
        [16, 6, 2048, 4096],
        None,
    );
    routed_select_at(
        Format::E4m3 { gran: 128 },
        1,
        false,
        [16, 6, 4096, 2048],
        None,
    );
    // The same products read back to bf16 on the e2m1-convert
    // register-dequant kernel, as a decode-wide W4A8 select runs.
    for (tokens, by_token, live) in [(8, true, None), (5, false, None), (40, true, Some(17))] {
        routed_select_cvt(tokens, by_token, [8, 3, 256, 512], live);
    }
    routed_select_cvt(8, true, [16, 6, 2048, 4096], None);
    routed_select_cvt(8, false, [16, 6, 4096, 2048], None);
}

thread_local! {
    static CVT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`routed_select_at`] (e4m3 in 128s) with the activations read back and
/// the bank's atoms read by `moe::select_mxfp4_atoms`.
fn routed_select_cvt(tokens: usize, by_token: bool, dims: [usize; 4], live: Option<usize>) {
    CVT.with(|c| c.set(true));
    routed_select_at(Format::E4m3 { gran: 128 }, tokens, by_token, dims, live);
    CVT.with(|c| c.set(false));
}

/// A routed bank on the grouped tcgen05 GEMM, masked (few tokens) or
/// contiguous: each route the dot of its read-back activations (under its
/// expert's global) and its expert's dequantized weight, times the expert's
/// product scale.
fn routed_select(fmt: Format, tokens: usize, by_token: bool) {
    routed_select_at(fmt, tokens, by_token, [8, 3, 256, 512], None);
}

/// The select under a live window of `live` tokens from the first: the
/// routes past it hold ids no expert has, and are neither read nor written.
fn routed_select_live(fmt: Format, tokens: usize, live: usize) {
    routed_select_at(fmt, tokens, true, [8, 3, 256, 512], Some(live));
}

fn routed_select_at(
    fmt: Format,
    tokens: usize,
    by_token: bool,
    dims: [usize; 4],
    live: Option<usize>,
) {
    let [experts, top_k, n, k] = dims;
    let nvfp4 = fmt == Format::Nvfp4;
    let gran = if nvfp4 { 16 } else { 32 };
    let mut lcg = Lcg::seeded(0x5e1 ^ (tokens * 7 + usize::from(by_token)) as u64);
    let count = tokens * top_k;
    let act_rows = if by_token { tokens } else { count };
    let (raw, _) = activations(&mut lcg, act_rows, k);
    let mut routes: Vec<i32> = (0..count)
        .map(|r| ((r / top_k * 5 + r % top_k * 3 + r / 97) % experts) as i32)
        .collect();
    routes[1] = -1;
    let live_routes = live.map_or(count, |l| l * top_k);
    for r in &mut routes[live_routes..] {
        *r = 0x3fff_ffff;
    }
    let input_scales: Vec<f32> = (0..experts).map(|e| 0.004 + 0.003 * e as f32).collect();
    let tensor: Vec<f32> = (0..experts * n)
        .map(|r| 0.5 + (r / n) as f32 * 0.125)
        .collect();
    let mut codes = vec![0u8; experts * n * k / 2];
    let mut w = vec![0f32; experts * n * k];
    let scales: Vec<u8> = (0..experts * n * k / gran)
        .map(|i| {
            if nvfp4 {
                0x30 + (i * 5 % 16) as u8
            } else {
                121 + (i * 7 % 12) as u8
            }
        })
        .collect();
    for i in 0..experts * n * k {
        let code = ((lcg.unit() + 1.0) * 8.0) as u8 & 0xf;
        codes[i / 2] |= code << (4 * (i % 2));
        let s = scales[i / gran];
        w[i] = e2m1(code)
            * if nvfp4 {
                e4m3(s)
            } else {
                2f32.powi(i32::from(s) - 127)
            };
    }
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let routes_at = gpu.up(&routes);
    let is_at = gpu.up(&input_scales);
    let ts_at = gpu.up(&tensor);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let atoms_at = gpu.zeros(atom_bytes(experts * n, k, gran));
    let alpha_at = gpu.zeros(experts * 4);
    const UNTOUCHED: u16 = 0x7f7f;
    let y_at = gpu.up(&vec![UNTOUCHED; count * n]);
    let values_at = gpu.zeros(count * k * 2);
    let ctx = gpu.ctx();
    let x = Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16);
    let route_t = Tensor::new(routes_at, tokens as u32, top_k as u32, Dtype::I32);
    let globals = nvfp4.then(|| Tensor::new(is_at, experts as u32, 1, Dtype::F32));
    scales_to_atoms(
        &ctx,
        OP,
        scales_at,
        atoms_at,
        (experts * n) as u32,
        (k / gran) as u32,
        1,
    )
    .expect("the bank's atoms land");
    let mut y = Tensor::new(y_at, count as u32, n as u32, Dtype::Bf16);
    if let Some(l) = live {
        let window = gpu.up(&[l as u32, 0, 0, 0]);
        ctx.arm_stage(window);
    }
    let back_at = gpu.zeros(act_rows * k * 2);
    let cvt_ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        mxfp4_mma_from: 1,
        mxfp4_mma_until: 65_536,
        mxfp4_cvt: true,
        mxfp4_a8_until: 65_536,
        ..kernels_cuda::tuning::Tuning::DEFAULT
    });
    if let Some(l) = live {
        let window = gpu.up(&[l as u32, 0, 0, 0]);
        cvt_ctx.arm_stage(window);
    }
    if CVT.with(std::cell::Cell::get) {
        let back = read_back(&cvt_ctx, OP, fmt, Source::rows(x, None), back_at)
            .expect("the read-back fires");
        let taken = kernels_cuda::linear::moe::select_mxfp4_atoms(
            &cvt_ctx,
            back,
            Tensor::new(codes_at, experts as u32, (n * k / 2) as u32, Dtype::U8),
            atoms_at,
            route_t,
            &mut y,
        )
        .expect("the e2m1-convert select fires");
        assert!(taken, "the e2m1-convert select takes {count} routes");
    } else {
        select(
            &ctx,
            OP,
            fmt,
            x,
            route_t,
            globals,
            Bank {
                codes: codes_at,
                atoms: atoms_at,
                experts: experts as u32,
                n: n as u32,
                k: k as u32,
                elem: Elem::E2m1,
                gran: gran as u32,
                alpha_e: if nvfp4 { alpha_at } else { 0 },
                // The select lays the product scales with the routes.
                alpha_from: nvfp4.then_some(AlphaFrom {
                    input_scales: is_at,
                    tensor_scales: ts_at,
                    rows: n as u32,
                }),
            },
            &mut y,
        )
        .expect("the routed product fires");
    }
    ctx.disarm_stage();
    cvt_ctx.disarm_stage();
    let div = if by_token { top_k } else { 1 };
    let live_t = Tensor::new(
        routes_at,
        (live_routes / top_k) as u32,
        top_k as u32,
        Dtype::I32,
    );
    read_back(
        &ctx,
        OP,
        fmt,
        Source {
            x,
            rows: live_routes as u32,
            div: div as u32,
            routes: nvfp4.then_some(live_t),
            input_scales: globals,
            pos: 0,
            m_max: 0,
        },
        values_at,
    )
    .expect("the read-back fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, count * n);
    let a: Vec<f32> = gpu
        .down::<u16>(values_at, count * k)
        .into_iter()
        .map(from_bf16)
        .collect();
    let mut bad = 0;
    for r in 0..count {
        if r >= live_routes {
            assert!(
                got[r * n..(r + 1) * n].iter().all(|&v| v == UNTOUCHED),
                "{fmt:?}: route {r} is past the live window and was written"
            );
            continue;
        }
        for c in 0..n {
            let g = from_bf16(got[r * n + c]);
            if routes[r] < 0 {
                assert_eq!(g, 0.0, "dropped route {r}");
                continue;
            }
            let e = routes[r] as usize;
            let alpha = if nvfp4 {
                input_scales[e] * tensor[e * n]
            } else {
                1.0
            };
            let want: f64 = (0..k)
                .map(|i| f64::from(a[r * k + i]) * f64::from(w[(e * n + c) * k + i]))
                .sum::<f64>()
                * f64::from(alpha);
            if !close(g, want as f32) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("{fmt:?} route {r} (expert {e}) column {c}: got {g} want {want}");
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{fmt:?} {tokens} tokens: {bad} of {} outputs off",
        count * n
    );
}
