//! Per-kernel device time at a decode fire of Qwen3.5-0.8B's shapes, for
//! finding what scales with the batch. Asked for with `PIE_XLA_BENCH=1`
//! (`PIE_XLA_BENCH_WIDTH` sets the lanes, default 64).

use dtype::Dtype;
use engine_xla::bench::Bench;
use kernels_xla::attn::{self, ssm};
use kernels_xla::{DecodePlan, KvPool, RecurrentPool};

fn width() -> Option<u32> {
    std::env::var("PIE_XLA_BENCH").ok()?;
    Some(
        std::env::var("PIE_XLA_BENCH_WIDTH")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(64),
    )
}

fn report(what: &str, t: Option<f64>) {
    if let Some(t) = t {
        eprintln!("{what:<28} {:>8.3} ms", t * 1e3);
    }
}

#[test]
fn decode_kernels_at_width() {
    let Some(n) = width() else {
        eprintln!("not asked: set PIE_XLA_BENCH");
        return;
    };
    let (qh, kvh, d, ps, pages_per) = (8u32, 2u32, 256u32, 16u32, 4u32);
    let cells = (n * pages_per + 1) * ps;
    let mut b = Bench::new();
    let q = b.bf16(n, qh * d, &vec![0.01; (n * qh * d) as usize]);
    let o = b.zeros(Dtype::Bf16, n, qh * d);
    let keys = b.bf16(cells, kvh * d, &vec![0.02; (cells * kvh * d) as usize]);
    let values = b.bf16(cells, kvh * d, &vec![0.03; (cells * kvh * d) as usize]);
    let indices: Vec<i32> = (0..n * pages_per).map(|i| i as i32).collect();
    let indptr: Vec<i32> = (0..=n).map(|l| (l * pages_per) as i32).collect();
    let page_indices = b.i32(indices.len() as u32, 1, &indices);
    let page_indptr = b.i32(n + 1, 1, &indptr);
    let pool = KvPool {
        keys,
        values,
        page_indices,
        page_indptr,
        page_size: ps as i32,
        max_pages: pages_per,
        seq_stride: u64::from(kvh * d),
        head_stride: u64::from(d),
    };
    let positions = b.i32(n, 1, &vec![40; n as usize]);
    let request = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let mask = b.u8(n, 1, &vec![0; n as usize]);
    let mask_enabled = b.u8(n, 1, &vec![0; n as usize]);
    let plan = DecodePlan {
        positions,
        request_of_token: request,
        mask,
        mask_enabled,
        mask_stride: 0,
    };
    report(
        "attn::decode",
        b.time(20, |ctx| attn::decode(ctx, q, &plan, &pool, None, d, 0.0625, o))
            .unwrap(),
    );

    // Gated delta: k/v heads 16 of 128.
    let (hk, hv, dk, dv) = (16u32, 16u32, 128u32, 128u32);
    let qkv = b.bf16(n, 2 * hk * dk + hv * dv, &vec![0.01; (n * (2 * hk * dk + hv * dv)) as usize]);
    let z = b.zeros(Dtype::Bf16, n, hv * dv);
    let gates = b.f32(n, 2 * hv, &vec![0.1; (n * 2 * hv) as usize]);
    let stride = hv * dv * dk;
    let bank = b.f32(n + 1, stride, &vec![0.0; ((n + 1) * stride) as usize]);
    let slots = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let y = b.zeros(Dtype::F32, n, hv * dv);
    let state = RecurrentPool {
        state: bank,
        slots,
        conv_state: bank,
        new_conv_state: bank,
    };
    report(
        "ssm::gated_delta",
        b.time(20, |ctx| ssm::gated_delta(ctx, qkv, z, gates, &state, hk, hv, dk, dv, y))
            .unwrap(),
    );

    // Causal conv over 6144 channels, width 4.
    let c = 6144u32;
    let x = b.bf16(n, c, &vec![0.01; (n * c) as usize]);
    let w = b.bf16(c, 4, &vec![0.1; (c * 4) as usize]);
    let cbank = b.f32(n + 1, 4 * c, &vec![0.0; ((n + 1) * 4 * c) as usize]);
    let cy = b.zeros(Dtype::Bf16, n, c);
    let cstate = RecurrentPool {
        state: cbank,
        slots,
        conv_state: cbank,
        new_conv_state: cbank,
    };
    report(
        "ssm::causal_conv1d",
        b.time(20, |ctx| ssm::causal_conv1d(ctx, x, w, &cstate, 4, 1, cy))
            .unwrap(),
    );

    // The logits: [n, 1024] x [248320, 1024].
    let act = b.bf16(n, 1024, &vec![0.01; (n * 1024) as usize]);
    let table = b.bf16(248_320, 1024, &vec![0.001; 248_320 * 1024]);
    let logits = b.zeros(Dtype::Bf16, n, 248_320);
    report(
        "gemm::lm_head",
        b.time(20, |ctx| {
            kernels_xla::linear::gemm::lm_head(ctx, act, table, logits)
        })
        .unwrap(),
    );
}

#[test]
fn state_row_gather_and_scatter() {
    let Some(n) = width() else {
        return;
    };
    let stride = 16 * 128 * 128u32;
    let mut b = Bench::new();
    let bank = b.f32(n + 1, stride, &vec![0.0; ((n + 1) * stride) as usize]);
    let slots = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let out = b.zeros(Dtype::F32, n, stride);
    report(
        "take_rows",
        b.time(20, |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let i = cx.read(slots)?;
                let i = cx.reshape(i, &[i64::from(n)])?;
                let r = cx.take_rows(s, i)?;
                cx.write(out, r)
            })
        })
        .unwrap(),
    );
    let rows = b.f32(n, stride, &vec![1.0; (n * stride) as usize]);
    report(
        "put_rows",
        b.time(20, |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let i = cx.read(slots)?;
                let i = cx.reshape(i, &[i64::from(n)])?;
                let r = cx.read(rows)?;
                let s = cx.put_rows(s, i, r, kernels_xla::hlo::Combine::Set)?;
                cx.write(bank, s)
            })
        })
        .unwrap(),
    );
}

#[test]
fn state_row_scatter_hinted_in_place() {
    let Some(n) = width() else {
        return;
    };
    let stride = 16 * 128 * 128u32;
    let mut b = Bench::new();
    let bank = b.f32(n + 1, stride, &vec![0.0; ((n + 1) * stride) as usize]);
    let slots = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let rows = b.f32(n, stride, &vec![1.0; (n * stride) as usize]);
    for unique in [false, true] {
        report(
            if unique { "put_rows unique" } else { "put_rows" },
            b.time(20, |ctx| {
                ctx.emit(&mut |cx| {
                    let s = cx.read(bank)?;
                    let i = cx.read(slots)?;
                    let i = cx.reshape(i, &[i64::from(n)])?;
                    let r = cx.read(rows)?;
                    let s = cx.put_rows_hinted(s, i, r, kernels_xla::hlo::Combine::Set, unique)?;
                    cx.write(bank, s)
                })
            })
            .unwrap(),
        );
    }
}

#[test]
fn state_row_update_by_loop() {
    let Some(n) = width() else {
        return;
    };
    let stride = 16 * 128 * 128u32;
    let mut b = Bench::new();
    let bank = b.f32(n + 1, stride, &vec![0.0; ((n + 1) * stride) as usize]);
    let slots = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let rows = b.f32(n, stride, &vec![1.0; (n * stride) as usize]);
    report(
        "dus loop",
        b.time(20, |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let i = cx.read(slots)?;
                let i = cx.reshape(i, &[i64::from(n)])?;
                let r = cx.read(rows)?;
                let out = cx.for_loop(i64::from(n), &[s], |f, at, carried| {
                    let zero = f.const_i(kernels_xla::hlo::Elem::I32, 0, &[]);
                    let slot = f.dynamic_slice(i, &[at], &[1])?;
                    let slot = f.reshape(slot, &[])?;
                    let row = f.dynamic_slice(r, &[at, zero], &[1, i64::from(stride)])?;
                    Ok(vec![f.dynamic_update_slice(carried[0], row, &[slot, zero])?])
                })?;
                cx.write(bank, out[0])
            })
        })
        .unwrap(),
    );
}

#[test]
fn state_row_scatter_tiled() {
    let Some(n) = width() else {
        return;
    };
    use kernels_xla::hlo::{Elem, ScatterDims, GatherDims, Ty};
    let mut b = Bench::new();
    // A [slots, 2048 * 128] bank, read as tiles of [2048, 128] per slot.
    let bank = b.f32(n + 1, 2048 * 128, &vec![0.0; ((n + 1) * 2048 * 128) as usize]);
    let slots = b.i32(n, 1, &(0..n as i32).collect::<Vec<_>>());
    let rows = b.f32(n, 2048 * 128, &vec![1.0; (n * 2048 * 128) as usize]);
    let _ = Ty::new(Elem::F32, &[]);
    report(
        "scatter 3d",
        b.time(20, |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let s = cx.reshape(s, &[i64::from(n) + 1, 2048, 128])?;
                let i = cx.read(slots)?;
                let r = cx.read(rows)?;
                let r = cx.reshape(r, &[i64::from(n), 2048, 128])?;
                let s = cx.scatter(
                    s,
                    i,
                    r,
                    &ScatterDims {
                        update_window_dims: vec![1, 2],
                        inserted_window_dims: vec![0],
                        scatter_dims_to_operand_dims: vec![0],
                        index_vector_dim: 1,
                        ..ScatterDims::default()
                    },
                    kernels_xla::hlo::Combine::Set,
                )?;
                let s = cx.reshape(s, &[i64::from(n) + 1, 2048 * 128])?;
                cx.write(bank, s)
            })
        })
        .unwrap(),
    );
    let out = b.zeros(Dtype::F32, n, 2048 * 128);
    report(
        "gather 3d",
        b.time(20, |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let s = cx.reshape(s, &[i64::from(n) + 1, 2048, 128])?;
                let i = cx.read(slots)?;
                let g = cx.gather(
                    s,
                    i,
                    &GatherDims {
                        offset_dims: vec![1, 2],
                        collapsed_slice_dims: vec![0],
                        start_index_map: vec![0],
                        index_vector_dim: 1,
                        ..GatherDims::default()
                    },
                    &[1, 2048, 128],
                )?;
                cx.write(out, g)
            })
        })
        .unwrap(),
    );
}
