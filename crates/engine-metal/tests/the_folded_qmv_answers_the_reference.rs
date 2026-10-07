#![cfg(target_vendor = "apple")]

//! Pins the affine `qmv_rows` kernel to an absolute, hand-built reference.
//!
//! The companion test `a_quantized_matmul_is_priced_by_its_rows` proves the
//! folded kernel answers the *one-row* kernel identically. That is a relative
//! check: if both kernels shared a bug it would say nothing. This test builds
//! affine-quantized weights and a bf16 activation in-process, dequantizes the
//! weights back to f32 and computes the matmul in **f64** as ground truth, then
//! drives the Metal `qmv_rows` kernel over the full dispatch grid and asserts
//! every output row lands on that f64 truth. Because the reference never runs a
//! kernel, no compiler miscompile can hide inside it, and because the grid
//! walks every row-count remainder against the fold width, no fix can pass by
//! special-casing a shape.
//!
//! Grid walked here (every point the dispatcher stamps):
//!   row_count m  in {1..=17, 31, 32, 33}   -- every remainder vs the fold width
//!   fold width   in the stamped rungs for (packs, bits)
//!   packs        in {1, 2}
//!   group size   in {32, 64, 128}
//!   bit width    in {2, 4, 8}

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use poem_ir::Dtype;

const K: u32 = 2048; // contraction; a multiple of every group size and pack width
const N: u32 = 256; // output columns
const ROWS_MAX: u32 = 33; // the widest row_count we launch

/// Splitmix-style byte noise, deterministic per index.
fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

/// Store an f32 as bf16 exactly the way the kernel's inputs are laid down:
/// keep the high 16 bits (truncation). Little-endian pair.
fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

/// The f64 value the kernel actually sees after a bf16 input store: the same
/// top-16-bit truncation `bf16` applies, promoted back. Using this in the
/// reference makes input rounding cancel, so the only reference-vs-kernel gap
/// is the kernel's f32 accumulation and its bf16 *output* store.
fn as_seen(v: f32) -> f64 {
    f64::from(f32::from_bits(v.to_bits() & 0xFFFF_0000))
}

fn to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u32::from(hi) << 24) | (u32::from(lo) << 16))
}

/// Pack one row's integer codes LSB-first into a contiguous bit stream, the
/// layout the affine kernels read (codes of `bits` bits, no code straddles a
/// 16-bit word because 2, 4 and 8 all divide 16). `out.len() == qrow.len() *
/// bits / 8`.
fn pack_row(qrow: &[u8], bits: u32, out: &mut [u8]) {
    for b in out.iter_mut() {
        *b = 0;
    }
    let mask = (1u32 << bits) - 1;
    let mut bitpos = 0usize;
    for &q in qrow {
        let v = u32::from(q) & mask;
        for bi in 0..bits {
            if (v >> bi) & 1 == 1 {
                out[bitpos / 8] |= 1u8 << (bitpos % 8);
            }
            bitpos += 1;
        }
    }
}

/// The row-fold widths the dispatcher stamps for a (packs, bits) pair. Mirrors
/// `qmv_rungs_at` in the kernel crate (which is private).
fn rungs(packs: i32, bits: i32) -> &'static [i32] {
    match (packs, bits) {
        (2, 2) => &[2, 4, 8],
        (2, _) => &[2, 3, 4, 6, 7, 8],
        (_, 2) => &[2, 3],
        _ => &[2, 3, 4, 5, 6, 7, 8],
    }
}

#[test]
fn the_folded_qmv_answers_the_reference() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    let (k, n) = (K, N);
    let row_counts: Vec<u32> = (1..=17u32).chain([31, 32, 33]).collect();

    // Buffers sized for the worst case (bits=8 codes, gs=32 factors), bound once.
    let codes_cap = u64::from(n) * u64::from(k); // bits/8 <= 1 byte per code
    let factors_cap = u64::from(n) * u64::from(k / 32); // gs=32 => most groups
    let mut codes_b = Buffer::zeroed(&device, codes_cap).expect("codes");
    let mut scales_b = Buffer::zeroed(&device, factors_cap * 2).expect("scales");
    let mut biases_b = Buffer::zeroed(&device, factors_cap * 2).expect("biases");
    let mut act_b = Buffer::zeroed(&device, u64::from(ROWS_MAX) * u64::from(k) * 2).expect("act");
    let out_b = Buffer::zeroed(&device, u64::from(ROWS_MAX) * u64::from(n) * 2).expect("out");

    // One shared activation [ROWS_MAX][k], bf16 in the buffer, f64-as-seen here.
    let mut act_seen = vec![0.0f64; (ROWS_MAX * k) as usize];
    {
        let mut act = vec![0u8; (u64::from(ROWS_MAX) * u64::from(k) * 2) as usize];
        for (idx, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = 0.02 * (f32::from(noise(idx as u64) % 16) - 8.0);
            *pair = bf16(v);
            act_seen[idx] = as_seen(v);
        }
        act_b.write(0, &act).expect("write act");
    }

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, hs, hb, ha, ho) = (
        bind(&codes_b),
        bind(&scales_b),
        bind(&biases_b),
        bind(&act_b),
        bind(&out_b),
    );
    let read = |rows: u32| -> Vec<f32> {
        handles
            .read(ho, u64::from(rows) * u64::from(n) * 2)
            .expect("read")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| to_f32(p[0], p[1]))
            .collect()
    };

    // Tolerance, per output element: allowed = BF16_ULP*max(|ref|,|got|) + ACCUM*mass,
    // where mass = sum_k |act_k * W_k| is accumulated in f64 beside the reference.
    //
    //   BF16_ULP * max(|ref|,|got|)  -- the result is stored as bf16 (8-bit
    //                                   mantissa), so one output ulp is 2^-8
    //                                   relative (round-to-nearest leaves a
    //                                   2^-9 half-ulp; 2^-8 doubles it for
    //                                   headroom). This term governs well-
    //                                   conditioned rows.
    //   ACCUM * mass                 -- the dominant term for cancelling rows,
    //                                   where |result| << mass. The reference
    //                                   contracts in f64; the device does not
    //                                   contract in IEEE f32 either. Measured
    //                                   over the whole grid (476k elements) the
    //                                   worst device error is ~2^-10.7 * mass,
    //                                   uniform across 2/4/8-bit (so it is the
    //                                   relaxed-precision float MAD/reduction
    //                                   chain, not the low-bit decode): a plain
    //                                   f32 contraction of the same terms on CPU
    //                                   sits at ~2^-21..2^-26 * mass, ~1000x
    //                                   tighter. 2^-9 is the honest device floor
    //                                   with ~3x margin over the observed worst.
    //                                   It still pins truth hard: the miscompile
    //                                   this test guards corrupts a whole
    //                                   accumulator (order-1 relative error),
    //                                   dwarfing 2^-9*mass.
    // Both inputs are fed to the reference bf16-truncated (see `as_seen`), so
    // input rounding cancels and is deliberately absent from this budget.
    const BF16_ULP: f64 = 1.0 / 256.0; // 2^-8
    const ACCUM: f64 = 1.0 / 512.0; // 2^-9: the measured relaxed-float device floor + margin

    let mut points = 0usize;
    let mut launches = 0usize;
    let mut worst_rel = 0.0f64;
    let mut worst_ratio = 0.0f64; // |err| / allowed; must stay < 1
    let mut failures: Vec<String> = Vec::new();
    // Per-bit-width worst device error as a fraction of |term| mass: the
    // relaxed-float accumulation floor the ACCUM budget is justified against.
    let mut em = std::collections::BTreeMap::<i32, f64>::new();

    for &bits in &[2i32, 4, 8] {
        let ubits = bits.unsigned_abs();
        // Integer codes q[col][k], packed into the codes buffer for this width.
        let mut q = vec![0u8; (n * k) as usize];
        let lvls = 1u16 << ubits;
        for (idx, cell) in q.iter_mut().enumerate() {
            *cell = (u16::from(noise(idx as u64 ^ 0x5151)) % lvls) as u8;
        }
        {
            let row_bytes = (k * ubits / 8) as usize;
            let mut packed = vec![0u8; (n as usize) * row_bytes];
            for col in 0..n as usize {
                let qrow = &q[col * k as usize..(col + 1) * k as usize];
                pack_row(
                    qrow,
                    ubits,
                    &mut packed[col * row_bytes..(col + 1) * row_bytes],
                );
            }
            codes_b.write(0, &packed).expect("write codes");
        }

        for &gs in &[32u32, 64, 128] {
            let groups = k / gs;
            // Scales and biases [col][group], bf16 in the buffers, f64-as-seen
            // for the reference.
            let mut scale_seen = vec![0.0f64; (n * groups) as usize];
            let mut bias_seen = vec![0.0f64; (n * groups) as usize];
            {
                let mut sc = vec![0u8; (u64::from(n) * u64::from(groups) * 2) as usize];
                let mut bi = vec![0u8; (u64::from(n) * u64::from(groups) * 2) as usize];
                for (idx, (sp, bp)) in sc
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(bi.as_chunks_mut::<2>().0.iter_mut())
                    .enumerate()
                {
                    let s = 0.01 + 0.001 * f32::from(noise(idx as u64 ^ 0xAA) % 8);
                    let b = -0.05 + 0.01 * f32::from(noise(idx as u64 ^ 0x55) % 8);
                    *sp = bf16(s);
                    *bp = bf16(b);
                    scale_seen[idx] = as_seen(s);
                    bias_seen[idx] = as_seen(b);
                }
                scales_b.write(0, &sc).expect("write scales");
                biases_b.write(0, &bi).expect("write biases");
            }

            // f64 ground truth: reference[r][col] and its |term| mass, over the
            // full k, using the dequantized weight W = scale*q + bias.
            let nn = n as usize;
            let kk = k as usize;
            let gg = groups as usize;
            let mut reference = vec![0.0f64; (ROWS_MAX as usize) * nn];
            let mut mass = vec![0.0f64; (ROWS_MAX as usize) * nn];
            for col in 0..nn {
                for ck in 0..kk {
                    let w = scale_seen[col * gg + ck / gs as usize] * f64::from(q[col * kk + ck])
                        + bias_seen[col * gg + ck / gs as usize];
                    for r in 0..ROWS_MAX as usize {
                        let t = act_seen[r * kk + ck] * w;
                        reference[r * nn + col] += t;
                        mass[r * nn + col] += t.abs();
                    }
                }
            }

            for &packs in &[1i32, 2] {
                for &rung in rungs(packs, bits) {
                    let Ok(point) = quant::qmv_rows_point("ref-grid", gs as i32, bits, rung, packs)
                    else {
                        continue;
                    };
                    points += 1;
                    let (ki, ni) = (k as i32, n as i32);
                    for &m in &row_counts {
                        let mi = m as i32;
                        let frame = device.frame().expect("a frame");
                        let sink = Sink::new(&device, &frame, &pipelines, &handles);
                        let args = [
                            Tensor::new(hc, n, k, Dtype::U4g64).arg(),
                            Tensor::new(hs, n, groups, Dtype::Bf16).arg(),
                            Tensor::new(hb, n, groups, Dtype::Bf16).arg(),
                            Tensor::new(ha, m, k, Dtype::Bf16).arg(),
                            Tensor::new(ho, m, n, Dtype::Bf16).arg_mut(),
                            ki.arg(),
                            ni.arg(),
                            mi.arg(),
                        ];
                        sink.fire(
                            Fire::at("linear/quant_qmv_rows.metal", point.entry)
                                .stamp(point.stamp)
                                .apply(Grid::of(
                                    quant::qmv_rows_grid("ref-grid", mi, rung, ni).expect("grid"),
                                    [32, 2, 1],
                                )),
                            &args,
                        )
                        .expect("the fold");
                        frame.commit().expect("commit");
                        launches += 1;

                        let got = read(m);
                        for r in 0..m as usize {
                            for col in 0..nn {
                                let want = reference[r * nn + col];
                                let have = f64::from(got[r * nn + col]);
                                let err = (want - have).abs();
                                let allowed = BF16_ULP * want.abs().max(have.abs())
                                    + ACCUM * mass[r * nn + col];
                                let rel = err / want.abs().max(have.abs()).max(1e-6);
                                worst_rel = worst_rel.max(rel);
                                worst_ratio = worst_ratio.max(err / allowed.max(f64::MIN_POSITIVE));
                                let m_here = mass[r * nn + col].max(f64::MIN_POSITIVE);
                                let e = em.entry(bits).or_insert(0.0);
                                *e = e.max(err / m_here);
                                if err > allowed && failures.len() < 12 {
                                    failures.push(format!(
                                        "{} m={m} row={r} col={col}: want {want:.6} got {have:.6} \
                                         (err {err:.3e} > allowed {allowed:.3e})",
                                        point.entry
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!(
        "swept {points} stamped points x {} row counts = {launches} launches against f64 truth",
        row_counts.len()
    );
    eprintln!(
        "worst relative error {worst_rel:.3e}; worst |err|/allowed {worst_ratio:.3} (must be < 1)"
    );
    for (b, v) in &em {
        eprintln!(
            "  bits={b}: worst device err/mass = {v:.3e}  (log2 {:.2}; budget 2^-9)",
            v.log2()
        );
    }
    assert!(
        failures.is_empty(),
        "{} grid points miss the f64 reference:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
