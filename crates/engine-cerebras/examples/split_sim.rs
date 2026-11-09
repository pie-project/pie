//! The role-split model (`SplitModel`) on a compiled program, against the
//! prep's golden: greedy sequences through the wafer token loop (two from
//! position 0 on one load: the state resets), a rollback (replayed from 0),
//! and a logits row (`lgt`; on a Qwen3.5 prep also against a host reference).
//! Qwen3.5 and Gemma 4 preps; gpt-oss has `split_sim_gptoss`.
//!
//! `split_sim <program out dir> <prep dir> <prefix> <overrides> <steps a launch> [cmaddr]`
//! (no cmaddr: the simulator).

use engine_cerebras::sdk::{Platform, Runtime, Sdk, Simulator, Target};
use engine_cerebras::spatial::split::{SplitModel, SplitSpec, argmax};

fn f16(h: u16) -> f32 {
    let s = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

fn run(m: &mut SplitModel, prompt: &[u32], n: usize, steps: usize) -> Vec<u32> {
    let mut got = Vec::new();
    let mut feed = prompt.to_vec();
    let mut start = 0;
    while got.len() < n {
        let extra = (steps - 1).min(n - got.len() - 1);
        let a = m.greedy(&feed, start, extra).unwrap();
        let new = &a[feed.len() - 1..];
        start += feed.len() + new.len() - 1;
        got.extend_from_slice(new);
        feed = vec![*new.last().unwrap()];
    }
    got.truncate(n);
    got
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let spec = SplitSpec::parse(&format!("{}:{}:{}", a[2], a[3], a[4])).unwrap();
    let steps: usize = a[5].parse().unwrap();
    let platform = match a.get(6) {
        Some(c) => Platform::System { cmaddr: c.clone() },
        None => Platform::Simulator(Simulator {
            num_threads: 8,
            ..Simulator::new(Target::Wse3)
        }),
    };
    let gold: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(spec.file("golden.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        gold[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let (prompt, want) = (ids("prompt_ids"), ids("tokens"));
    let n: usize = std::env::var("SPLIT_NTOK")
        .map_or(want.len(), |v| v.parse().unwrap())
        .min(want.len());
    let vocab = (spec.int("HR").unwrap() * spec.int("VR").unwrap()) as usize;
    let sdk = Sdk::open().unwrap();
    let mut rt = Runtime::new(sdk, std::path::Path::new(&a[1]), platform, "WARNING").unwrap();
    rt.load();
    rt.run();
    let mut m = SplitModel::open(Box::new(rt), &spec, vocab).unwrap();
    let mut ok = true;
    for rep in 0..2 {
        let t0 = std::time::Instant::now();
        let got = run(&mut m, &prompt, n, steps);
        let mis = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        ok &= mis == 0;
        println!(
            "SEQ {rep}: {mis} mismatches of {n}, {:.3} ms a token, launches {} steps {} got {:?}",
            t0.elapsed().as_secs_f64() * 1e3 / (prompt.len() + n) as f64,
            m.launches,
            m.steps,
            &got[..n.min(12)]
        );
    }
    // A rollback: the wafer holds the whole sequence; a run at NP + 2 replays from 0.
    if n > 4 {
        let at = prompt.len() + 2;
        let tok = want[1];
        let a2 = m.greedy(&[tok], prompt.len() + 1, 1).unwrap();
        let good = a2 == vec![want[2], want[3]];
        ok &= good;
        println!(
            "ROLLBACK to {at}: {:?} want {:?} {}",
            a2,
            &want[2..4],
            if good { "ok" } else { "BAD" }
        );
    }
    // A logits row: the prompt's last row's lgt argmax is the first token,
    // and the scaled row is the head's matvec over the final norm of the
    // last unit's x (a host reference from the prep's tiles).
    let (_, l) = m.forward(&prompt, 0, true).unwrap();
    let l = l.unwrap();
    let best = argmax(&l).unwrap();
    ok &= best == want[0];
    println!("LOGITS argmax {best} want {}", want[0]);
    if spec.meta.get("geo").is_some() {
        // Gemma's tied head has its own tile layout: the argmax check above is the test
        println!("LOGITS host reference: Qwen3.5 preps only");
    } else {
        let g = spec.geo().unwrap();
        let (lx, ly) = g.unit_origin(g.units() - 1);
        let hc = spec.int("Hc").unwrap() as usize;
        let py = spec.int("PY").unwrap();
        let mut x = Vec::new();
        for (r0, r1) in [(0, g.yc), (g.yc, py)] {
            let y0 = ly + r0 + u32::from(r0 >= g.yc);
            let r = engine_cerebras::sdk::Rect {
                x: (lx + g.xc) as i32,
                y: y0 as i32,
                w: 1,
                h: (r1 - r0) as i32,
            };
            x.extend(
                m.dev
                    .get("x", r, hc)
                    .unwrap()
                    .into_iter()
                    .map(f32::from_bits),
            );
        }
        let hidden = spec.int("H").unwrap() as f64;
        let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hidden + 1e-6).sqrt();
        let hw = spec.npy("hw").unwrap();
        let hg = spec.npy("hg").unwrap();
        let (hr, hcn, per) = (hw.shape[0], hw.shape[1], hw.shape[2]);
        let vr = spec.int("VR").unwrap() as usize;
        let hdc = per / vr;
        let mut w = Vec::new();
        hw.words(0, hr * hcn * per, &mut w).unwrap();
        let mut gv = Vec::new();
        hg.words(0, hcn * hdc, &mut gv).unwrap();
        let mut worst = 0f64;
        let mut big = 0f64;
        for (v, &lv) in l.iter().enumerate() {
            let (rr, j) = (v / vr, v % vr);
            let mut acc = 0f64;
            for c in 0..hcn {
                for d in 0..hdc {
                    let h = f16(w[(rr * hcn + c) * per + d * vr + j] as u16) as f64;
                    acc += h * f32::from_bits(gv[c * hdc + d]) as f64 * x[c * hdc + d] as f64;
                }
            }
            let rf = acc / rms;
            worst = worst.max((rf - lv as f64).abs());
            big = big.max(rf.abs());
        }
        let rel = worst / big.max(1e-30);
        ok &= rel < 2e-2;
        println!(
            "LOGITS scaled vs host reference: max |diff| {worst:.5} of max |logit| {big:.4} (rel {rel:.2e}), rms {rms:.4}"
        );
    }
    println!(
        "{}",
        if ok {
            "SPLIT_SIM OK"
        } else {
            "SPLIT_SIM FAILED"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
