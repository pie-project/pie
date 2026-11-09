//! gpt-oss's role-split model (`SplitModel` on a `prep_gptoss.py` prep) on a compiled program, against the
//! prep's golden: greedy sequences through the wafer token loop (two from position 0 on one load), a
//! rollback (replayed from 0) and a logits row (`lgt`: its argmax is the first token).
//!
//! `split_sim_gptoss <program out dir> <prep dir> <prefix> <overrides> <steps a launch> [cmaddr]`
//! (no cmaddr: the simulator; SPLIT_NTOK caps the tokens).

use engine_cerebras::sdk::{Platform, Runtime, Sdk, Simulator, Target};
use engine_cerebras::spatial::split::{SplitModel, SplitSpec, argmax};

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
    let vocab = spec.int("V").unwrap() as usize;
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
            &got[..n.min(16)]
        );
    }
    if n > 4 {
        let at = prompt.len() + 2;
        let a2 = m.greedy(&[want[1]], prompt.len() + 1, 1).unwrap();
        let good = a2 == vec![want[2], want[3]];
        ok &= good;
        println!(
            "ROLLBACK to {at}: {:?} want {:?} {}",
            a2,
            &want[2..4],
            if good { "ok" } else { "BAD" }
        );
    }
    let (_, l) = m.forward(&prompt, 0, true).unwrap();
    let l = l.unwrap();
    let best = argmax(&l).unwrap();
    ok &= best == want[0];
    let mut s: Vec<f32> = l.clone();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!(
        "LOGITS argmax {best} want {} (top {:.4}, margin {:.4}, rows {})",
        want[0],
        s[0],
        s[0] - s[1],
        l.len()
    );
    println!("SPLIT_SIM {}", if ok { "ok" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
