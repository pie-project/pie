//! bf16 guest values: the DSL types them by eta-ir's rules, and the host
//! interpreter computes them in f32 and rounds each store to nearest even,
//! a reduction and a scan accumulating in f32, against an f64 reference.

use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_ir::types::{bf16_bits, bf16_to_f32, round_bf16};

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

fn idx(traced: &Traced, ch: &Channel) -> u32 {
    traced
        .channel_order()
        .iter()
        .position(|&gid| gid == ch.gid())
        .expect("channel in the trace") as u32
}

/// What one program lands, each from the same `[V]` logits.
struct Landed {
    /// `cast(logits, bf16)`.
    narrow: Vec<u16>,
    /// `narrow * 0.5 + 1.0` (an f32 constant mixed in, unrounded).
    scaled: Vec<u16>,
    softmax: Vec<u16>,
    log_softmax: Vec<u16>,
    /// `reduce_sum(narrow)`, `reduce_max(narrow)`: f32.
    sum: f32,
    max: f32,
    cumsum: Vec<u16>,
    top: Vec<u16>,
    top_ids: Vec<u32>,
    /// `cast(narrow, f32)`.
    wide: Vec<f32>,
    argmax: i32,
}

const TOP: u32 = 3;

fn run(logits: &[f32]) -> Landed {
    let vocab = logits.len() as u32;
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    let klen: &'static Channel = leak(Channel::from([1u32]));
    let positions: &'static Channel = leak(Channel::from([0u32]));
    tok.put([1i32]);
    let bf = |n: u32| -> &'static Channel { leak(Channel::new([n], dtype::bf16)) };
    let narrow = bf(vocab);
    let scaled = bf(vocab);
    let softmax_c = bf(vocab);
    let log_softmax_c = bf(vocab);
    let cumsum_c = bf(vocab);
    let top_c = bf(TOP);
    let top_ids: &'static Channel = leak(Channel::new([TOP], dtype::u32));
    let sums: &'static Channel = leak(Channel::new([2], dtype::f32));
    let wide: &'static Channel = leak(Channel::new([vocab], dtype::f32));
    let argmax: &'static Channel = leak(Channel::new([1], dtype::i32));
    let readers = [
        narrow,
        scaled,
        softmax_c,
        log_softmax_c,
        cumsum_c,
        top_c,
        top_ids,
        sums,
        wide,
        argmax,
    ];

    let mut b = Builder::new(vocab, 4);
    b.outputs(crate::outputs::logits(vocab));
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::Positions, positions);
    b.stage(Stage::Epilogue, move || {
        let raw = reshape(intrinsics::output(0, "logits"), [vocab]);
        let x = cast(&raw, dtype::bf16);
        assert_eq!(x.dtype(), dtype::bf16);
        let y = &x * 0.5f32 + 1.0f32;
        assert_eq!(y.dtype(), dtype::bf16, "an f32 constant mixes into bf16");
        let sm = softmax(&x);
        let lsm = log_softmax(&x);
        assert_eq!((sm.dtype(), lsm.dtype()), (dtype::bf16, dtype::bf16));
        let s = reduce_sum(&x);
        let m = reduce_max(&x);
        assert_eq!(
            (s.dtype(), m.dtype()),
            (dtype::f32, dtype::f32),
            "a reduction is f32"
        );
        let cs = cumsum(&x);
        assert_eq!(cs.dtype(), dtype::bf16, "a scan keeps bf16");
        let (vals, ids) = top_k(&x, TOP);
        assert_eq!(vals.dtype(), dtype::bf16);
        let both = select(
            eq(iota(2), broadcast(0u32, [2])),
            broadcast(reshape(&s, []), [2]),
            broadcast(reshape(&m, []), [2]),
        );
        let t = reduce_argmax(&x);
        tok.put(&reshape(&t, [1]));
        positions.put(Tensor::constant([0u32]));
        narrow.put(&x);
        scaled.put(&y);
        softmax_c.put(&sm);
        log_softmax_c.put(&lsm);
        cumsum_c.put(&cs);
        top_c.put(&vals);
        top_ids.put(&ids);
        sums.put(&both);
        wide.put(&cast(&x, dtype::f32));
        argmax.put(&reshape(&t, [1]));
    });
    for reader in readers {
        reader.note_host_take();
    }
    let traced = b.build().expect("the bf16 program builds");
    let bytes = traced.encode();
    assert_eq!(
        eta_ir::container::decode(&bytes).unwrap(),
        *traced.container(),
        "bf16 round-trips the wire"
    );
    let bound = eta_ir::validate::bind(
        traced.container().clone(),
        crate::outputs::logits(vocab),
        eta_ir::registry::EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the bf16 program binds");
    let mut inst = Instance::new(
        &bound,
        &[
            (idx(&traced, tok), Value::I32(vec![1])),
            (idx(&traced, indptr), Value::U32(vec![0, 1])),
            (idx(&traced, klen), Value::U32(vec![1])),
            (idx(&traced, positions), Value::U32(vec![0])),
        ],
    )
    .unwrap();
    let inputs = PassInputs {
        outputs: vec![("logits".into(), 0, Value::F32(logits.to_vec()))],
    };
    let report = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(report.committed, "{:?}", report.missed);
    let mut take = |ch: &Channel| inst.host_take(&bound, idx(&traced, ch)).unwrap();
    let h = |v: Value| match v {
        Value::Bf16(x) => x,
        other => panic!("not bf16: {other:?}"),
    };
    let f = |v: Value| match v {
        Value::F32(x) => x,
        other => panic!("not f32: {other:?}"),
    };
    let sums = f(take(sums));
    Landed {
        narrow: h(take(narrow)),
        scaled: h(take(scaled)),
        softmax: h(take(softmax_c)),
        log_softmax: h(take(log_softmax_c)),
        sum: sums[0],
        max: sums[1],
        cumsum: h(take(cumsum_c)),
        top: h(take(top_c)),
        top_ids: match take(top_ids) {
            Value::U32(x) => x,
            other => panic!("{other:?}"),
        },
        wide: f(take(wide)),
        argmax: match take(argmax) {
            Value::I32(x) => x[0],
            other => panic!("{other:?}"),
        },
    }
}

fn logits(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 20_000) as f32 / 1000.0 - 10.0
        })
        .collect()
}

/// Every store rounds the f32 result once, to nearest even; a reduction and
/// a scan accumulate in f32 (4096 ones count past 256, where a bf16
/// accumulator stops); the rest is within bf16's half-ulp of an f64
/// reference.
#[test]
fn bf16_matches_an_f64_reference() {
    // An odd vocabulary: a value whose last word holds one element.
    let rows = [
        logits(1001, 7),
        logits(64, 3),
        logits(5, 11),
        vec![1.0f32; 4096],
    ];
    for x in rows {
        let n = x.len();
        let got = run(&x);
        let narrow: Vec<f32> = x.iter().map(|&a| round_bf16(a)).collect();
        assert_eq!(
            got.narrow,
            x.iter().map(|&a| bf16_bits(a)).collect::<Vec<_>>(),
            "the cast rounds to nearest even"
        );
        assert_eq!(got.wide, narrow, "the cast back is exact");
        let scaled: Vec<u16> = narrow.iter().map(|&a| bf16_bits(a * 0.5 + 1.0)).collect();
        assert_eq!(
            got.scaled, scaled,
            "each op computes in f32, rounded once on store"
        );

        // The f64 references.
        let max = narrow
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |m, a| m.max(a as f64));
        let sum: f64 = narrow.iter().map(|&a| a as f64).sum();
        let lse = max
            + narrow
                .iter()
                .map(|&a| (a as f64 - max).exp())
                .sum::<f64>()
                .ln();
        assert_eq!(got.max as f64, max);
        assert!(
            (got.sum as f64 - sum).abs() <= 1e-5 * sum.abs().max(1.0),
            "{} {sum}",
            got.sum
        );
        let half_ulp = |r: f64| r.abs() * 2f64.powi(-8) + 1e-30;
        for i in 0..n {
            let p = (narrow[i] as f64 - lse).exp();
            let sm = bf16_to_f32(got.softmax[i]) as f64;
            // The shifted row `c`, its exponentials and the quotient are
            // each bf16: `c`'s half-ulp moves `exp(c)` by `|c|` half-ulps,
            // then three roundings more.
            let c = (narrow[i] as f64 - max).abs();
            let bound = (c + 3.0) * half_ulp(p) + 1e-9;
            assert!((sm - p).abs() <= bound, "softmax[{i}] {sm} vs {p}");
            let lp = narrow[i] as f64 - lse;
            let lsm = bf16_to_f32(got.log_softmax[i]) as f64;
            assert!(
                (lsm - lp).abs() <= half_ulp(lp) + 1e-5,
                "log_softmax[{i}] {lsm} vs {lp}"
            );
        }
        let mut prefix = 0f64;
        for i in 0..n {
            prefix += narrow[i] as f64;
            let cs = bf16_to_f32(got.cumsum[i]) as f64;
            assert!(
                (cs - prefix).abs() <= half_ulp(prefix) + 1e-4,
                "cumsum[{i}] {cs} vs {prefix}"
            );
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| narrow[b].total_cmp(&narrow[a]).then(a.cmp(&b)));
        let want = (TOP as usize).min(n);
        assert_eq!(
            got.top_ids[..want],
            order[..want].iter().map(|&i| i as u32).collect::<Vec<_>>()[..]
        );
        for k in 0..want {
            assert_eq!(
                got.top[k],
                bf16_bits(narrow[order[k]]),
                "a top-k value is exact"
            );
        }
        assert_eq!(got.argmax as usize, order[0]);
    }
}
