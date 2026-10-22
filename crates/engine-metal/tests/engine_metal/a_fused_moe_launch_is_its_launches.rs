#![cfg(target_vendor = "apple")]
//! A routed MLP's fused launches land their unfused launches' rows, bit for
//! bit: the routed gate-up with its swiglu, and the routes' fold with the
//! gated shared expert's add.

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

#[test]
fn the_fold_with_the_shared_experts_add_lands_both_launches_rows() {
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
    for (tokens, top_k, width) in [(8u32, 8u32, 2048u32), (1, 8, 2048), (5, 2, 300)] {
        let pairs = tokens * top_k;
        let routed = upload(&plane(u64::from(pairs * width), 0x44, 0.05));
        let shared = upload(&plane(u64::from(tokens * width), 0x55, 0.05));
        let gate = upload(&plane(u64::from(tokens), 0x66, 0.03));
        let weights: Vec<u8> = (0..pairs)
            .flat_map(|at| (f32::from(noise(u64::from(at) ^ 0x88)) / 997.0).to_le_bytes())
            .collect();
        let weights = upload(&weights);
        let out = || upload(&vec![0x5a; (tokens * width * 2) as usize]);
        let (folded, own, folded_one, one) = (out(), out(), out(), out());
        let t = |b: &Buffer, rows: u32| Tensor::new(bind(b), rows, width, Dtype::Bf16);
        let w = Tensor::new(bind(&weights), tokens, top_k, Dtype::F32);
        let g = Tensor::new(bind(&gate), tokens, 1, Dtype::Bf16);
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        moe::weighted_sum(&sink, t(&routed, pairs), w, t(&folded, tokens)).expect("the fold");
        moe::sigmoid_gate_add(
            &sink,
            t(&folded, tokens),
            t(&shared, tokens),
            g,
            t(&own, tokens),
        )
        .expect("the gate add");
        moe::weighted_sum_gate_add(
            &sink,
            t(&routed, pairs),
            w,
            t(&folded_one, tokens),
            t(&shared, tokens),
            g,
            t(&one, tokens),
        )
        .expect("the fused launch");
        frame.commit().expect("the commit");
        let own = read(&own);
        assert!(own.chunks(2).any(|v| v != [0x5a, 0x5a]), "the add ran");
        assert!(
            own == read(&one),
            "tokens={tokens} width={width}: the fused launch lands other rows"
        );
        assert!(
            read(&folded) == read(&folded_one),
            "tokens={tokens} width={width}: the fused launch lands other fold rows"
        );
        // Over the routed rows' own bytes, as the planner may lay the add's.
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        moe::weighted_sum_gate_add(
            &sink,
            t(&routed, pairs),
            w,
            t(&folded_one, tokens),
            t(&shared, tokens),
            g,
            t(&routed, tokens),
        )
        .expect("the fused launch in place");
        frame.commit().expect("the commit");
        let over = read(&routed);
        assert!(
            over[..own.len()] == own[..],
            "tokens={tokens} width={width}: the fused launch over the routed rows lands other rows"
        );
    }
}

/// Microbench (not a test): the 35B's verify (8 rows) and decode (1 row) MoE launches, fused and
/// not, over distinct 256-expert banks so the weights stream from memory as a fire's do.
/// Run: cargo test --release -p engine-metal --test engine_metal \
///        a_fused_moe_launch_prices -- --ignored --nocapture
#[test]
#[ignore = "a microbenchmark"]
fn a_fused_moe_launch_prices() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let keep = std::cell::RefCell::new(Vec::new());
    let bind = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len().max(16) as u64).expect("a plane");
        b.write(0, bytes).expect("write");
        let h = handles.bind(&b, 0, b.bytes()).expect("a handle");
        keep.borrow_mut().push(b);
        h
    };
    let (experts, top_k, k, inter, banks) = (256u32, 8u32, 2048u32, 512u32, 4u32);
    let (n, groups) = (2 * inter, k / 64);
    let columns = u64::from(experts * n);
    let bank: Vec<Bank> = (0..banks)
        .map(|at| Bank {
            mpp_codes: None,
            codes: Tensor::new(
                bind(
                    &(0..columns * u64::from(k) / 2)
                        .map(|i| noise(i ^ u64::from(at)))
                        .collect::<Vec<_>>(),
                ),
                experts * n,
                k,
                Dtype::U4g64,
            ),
            scales: Tensor::new(
                bind(&plane(columns * u64::from(groups), 0x11, 0.0002)),
                experts * n,
                groups,
                Dtype::Bf16,
            ),
            biases: Some(Tensor::new(
                bind(&plane(columns * u64::from(groups), 0x22, 0.003)),
                experts * n,
                groups,
                Dtype::Bf16,
            )),
            group: 64,
            bits: 4,
            ld: k,
        })
        .collect();
    for tokens in [8u32, 1] {
        let pairs = tokens * top_k;
        let x = Tensor::new(
            bind(&plane(u64::from(tokens * k), 0x33, 0.02)),
            tokens,
            k,
            Dtype::Bf16,
        );
        let routes: Vec<u8> = (0..tokens)
            .flat_map(|t| (0..top_k).flat_map(move |s| ((t * 37 + s * 29) % experts).to_le_bytes()))
            .collect();
        let routes = Tensor::new(bind(&routes), tokens, top_k, Dtype::I32);
        let packed = Tensor::new(
            bind(&vec![0; (pairs * n * 2) as usize]),
            pairs,
            n,
            Dtype::Bf16,
        );
        let hidden = Tensor::new(
            bind(&vec![0; (pairs * inter * 2) as usize]),
            pairs,
            inter,
            Dtype::Bf16,
        );
        let routed = Tensor::new(
            bind(&plane(u64::from(pairs * k), 0x44, 0.05)),
            pairs,
            k,
            Dtype::Bf16,
        );
        let weights: Vec<u8> = (0..pairs)
            .flat_map(|at| (at as f32 / 997.0).to_le_bytes())
            .collect();
        let w = Tensor::new(bind(&weights), tokens, top_k, Dtype::F32);
        let row = |salt| {
            Tensor::new(
                bind(&plane(u64::from(tokens * k), salt, 0.05)),
                tokens,
                k,
                Dtype::Bf16,
            )
        };
        let (folded, shared, out) = (row(0x55), row(0x66), row(0x77));
        let gate = Tensor::new(
            bind(&plane(u64::from(tokens), 0x88, 0.03)),
            tokens,
            1,
            Dtype::Bf16,
        );
        // Each fused launch lands its unfused launches' bytes here too, the add over
        // the routed rows' own as the planner lays it.
        let routed_copy = Tensor::new(
            bind(&plane(u64::from(pairs * k), 0x44, 0.05)),
            pairs,
            k,
            Dtype::Bf16,
        );
        let (packed_one, hidden_one) = (
            Tensor::new(
                bind(&vec![0; (pairs * n * 2) as usize]),
                pairs,
                n,
                Dtype::Bf16,
            ),
            Tensor::new(
                bind(&vec![0; (pairs * inter * 2) as usize]),
                pairs,
                inter,
                Dtype::Bf16,
            ),
        );
        let (folded_one, out_one) = (row(0x99), row(0xaa));
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        moe::matmul_select_quant(&sink, x, bank[0], routes, packed).expect("gate-up");
        mlp::swiglu(&sink, packed, inter, hidden).expect("swiglu");
        moe::matmul_select_swiglu(&sink, x, bank[0], routes, packed_one, hidden_one)
            .expect("fused");
        moe::weighted_sum(&sink, routed, w, folded).expect("fold");
        moe::sigmoid_gate_add(&sink, folded, shared, gate, out).expect("add");
        moe::weighted_sum_gate_add(&sink, routed, w, folded_one, shared, gate, out_one)
            .expect("fused");
        let over = Tensor::new(routed_copy.buf, tokens, k, Dtype::Bf16);
        moe::weighted_sum_gate_add(&sink, routed_copy, w, folded_one, shared, gate, over)
            .expect("fused in place");
        frame.commit().expect("commit");
        let read = |t: Tensor| {
            handles
                .read(t.buf, u64::from(t.rows) * u64::from(t.width) * 2)
                .expect("read")
        };
        assert!(
            read(packed) == read(packed_one),
            "{tokens} row(s): other gate-up rows"
        );
        assert!(
            read(hidden) == read(hidden_one),
            "{tokens} row(s): other swiglu rows"
        );
        assert!(
            read(folded) == read(folded_one),
            "{tokens} row(s): other fold rows"
        );
        assert!(
            read(out) == read(out_one),
            "{tokens} row(s): other add rows"
        );
        assert!(
            read(out) == read(over),
            "{tokens} row(s): other add rows in place"
        );
        let time = |body: &dyn Fn(&Sink<'_>)| {
            let mut v = Vec::new();
            for i in 0..=20 {
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..10 {
                    body(&sink);
                }
                let s = frame.commit_timed().expect("commit");
                if i > 0 {
                    v.push(s * 1e6 / 10.0);
                }
            }
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let gate_up = time(&|sink| {
            for &b in &bank {
                moe::matmul_select_quant(sink, x, b, routes, packed).expect("gate-up");
                mlp::swiglu(sink, packed, inter, hidden).expect("swiglu");
            }
        }) / f64::from(banks);
        let gate_up_fused = time(&|sink| {
            for &b in &bank {
                moe::matmul_select_swiglu(sink, x, b, routes, packed, hidden).expect("fused");
            }
        }) / f64::from(banks);
        let fold = time(&|sink| {
            moe::weighted_sum(sink, routed, w, folded).expect("fold");
            moe::sigmoid_gate_add(sink, folded, shared, gate, out).expect("add");
        });
        let fold_fused = time(&|sink| {
            moe::weighted_sum_gate_add(sink, routed, w, folded, shared, gate, out).expect("fused");
        });
        eprintln!(
            "{tokens} row(s): gate-up + swiglu {gate_up:.1} us, fused {gate_up_fused:.1} us; \
             fold + add {fold:.1} us, fused {fold_fused:.1} us"
        );
    }
}
