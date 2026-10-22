#![cfg(target_vendor = "apple")]
//! A routed MLP's fused launches land their unfused launches' rows, bit for
//! bit: the routed gate-up with its swiglu.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::{mlp, moe};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn plane(n: u64, salt: u64, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|at| bf16((f32::from(noise(at ^ salt)) - 127.0) * scale))
        .collect()
}

#[test]
fn the_routed_gate_up_with_its_swiglu_lands_both_launches_rows() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len().max(16) as u64).expect("a plane");
        b.write(0, bytes).expect("write");
        b
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let read = |b: &Buffer| handles.read(bind(b), b.bytes()).expect("read");
    // (experts, tokens, fan-out, K, intermediate): the 35B's verify and decode, and a
    // narrow bank.
    for (experts, tokens, top_k, k, inter) in [
        (16u32, 8u32, 8u32, 2048u32, 512u32),
        (16, 1, 8, 2048, 512),
        (6, 5, 2, 256, 68),
    ] {
        let (n, groups) = (2 * inter, k / 64);
        let columns = u64::from(experts * n);
        let codes: Vec<u8> = (0..columns * u64::from(k) / 2)
            .map(|at| noise(at ^ 0x77))
            .collect();
        let codes = upload(&codes);
        let scales = upload(&plane(columns * u64::from(groups), 0x11, 0.0002));
        let biases = upload(&plane(columns * u64::from(groups), 0x22, 0.003));
        let x = upload(&plane(u64::from(tokens * k), 0x33, 0.02));
        let routes: Vec<u8> = (0..tokens)
            .flat_map(|t| {
                (0..top_k).flat_map(move |s| ((t * 5 + s * 3 + t / 3) % experts).to_le_bytes())
            })
            .collect();
        let routes = upload(&routes);
        let pairs = tokens * top_k;
        let out = |width: u32| upload(&vec![0x5a; (pairs * width * 2) as usize]);
        let (packed_own, hidden_own, packed_one, hidden_one) =
            (out(n), out(inter), out(n), out(inter));
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&codes), experts * n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&scales), experts * n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&biases), experts * n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        let t = |b: &Buffer, rows: u32, width: u32| Tensor::new(bind(b), rows, width, Dtype::Bf16);
        let x_t = t(&x, tokens, k);
        let routes_t = Tensor::new(bind(&routes), tokens, top_k, Dtype::I32);
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        moe::matmul_select_quant(&sink, x_t, bank, routes_t, t(&packed_own, pairs, n))
            .expect("the gate-up");
        mlp::swiglu(
            &sink,
            t(&packed_own, pairs, n),
            inter,
            t(&hidden_own, pairs, inter),
        )
        .expect("the swiglu");
        let fused = moe::matmul_select_swiglu(
            &sink,
            x_t,
            bank,
            routes_t,
            t(&packed_one, pairs, n),
            t(&hidden_one, pairs, inter),
        )
        .expect("the fused launch");
        assert!(fused, "a 4-bit, 64-group gate-up of whole fours fuses");
        frame.commit().expect("the commit");
        let (own, one) = (read(&packed_own), read(&packed_one));
        assert!(own.chunks(2).any(|v| v != [0x5a, 0x5a]), "the gate-up ran");
        assert!(
            own == one,
            "experts={experts} tokens={tokens}: the fused launch lands other gate-up rows"
        );
        assert!(
            read(&hidden_own) == read(&hidden_one),
            "experts={experts} tokens={tokens}: the fused launch lands other swiglu rows"
        );
    }
}
