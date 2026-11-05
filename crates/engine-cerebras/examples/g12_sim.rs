//! Gemma 4 dense on serve8b_region (`G12 1`) through the engine's wafer path
//! against a reference: `g12_sim <program dir> <ckpt dir> <ref.json> <chunk>
//! [cmaddr]` (no cmaddr: the simulator). `ref.json` (scratch/g12/ref/tiny_ref.py):
//! `sequence` (a prompt and its greedy continuation), every position's raw
//! `logits` and `argmax`, and its `prompt_ids`/`tokens`. One lane, teacher-forced:
//! launches of `chunk` passes over the sequence, each pass's argmax against the
//! reference's and each launch's last logits row against its position's
//! reference row (max |diff| over max |ref|); then the greedy run from the
//! prompt, its tokens against the reference's. Prints one JSON line.

mod common;

use std::path::Path;

use engine_cerebras::spatial::serve2d::BatchLane;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let golden = common::read_json(&args[3])?;
    let chunk: usize = args[4].parse().map_err(|e| format!("{e}"))?;
    let platform = match args.get(5) {
        Some(cmaddr) => engine_cerebras::sdk::Platform::System {
            cmaddr: cmaddr.clone(),
        },
        None => common::simulator(8),
    };
    let mut m = engine_cerebras::spatial::wafer::WaferModel::open(
        Path::new(&args[1]),
        Path::new(&args[2]),
        platform,
    )?;
    let (b_n, shape) = common::batched(&m)?;
    if b_n != 1 {
        return Err(format!("g12_sim drives one lane; the build launches {b_n}"));
    }
    let ids = |k: &str| -> Vec<u32> {
        golden[k]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
            .unwrap_or_default()
    };
    let (seq, want, prompt, tokens) = (
        ids("sequence"),
        ids("argmax"),
        ids("prompt_ids"),
        ids("tokens"),
    );
    let refl: Vec<Vec<f32>> = golden["logits"]
        .as_array()
        .ok_or("ref logits")?
        .iter()
        .map(|r| {
            r.as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default()
        })
        .collect();
    let margins: Vec<f64> = golden["margins_tf"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
        .unwrap_or_default();
    let n = seq.len();
    let npg = (n + 2).div_ceil(shape.page_size) as u32;
    if npg as usize > shape.pages() {
        return Err(format!(
            "{n} positions want {npg} pages; the pool has {}",
            shape.pages()
        ));
    }
    let pages: Vec<u32> = (0..npg).collect();
    let lane = |pos: usize, forced: Vec<u32>| BatchLane {
        pos: pos as u32,
        slot: pos as u32,
        pages: pages.clone(),
        visible: None,
        forced,
        state: None,
    };
    // Teacher-forced.
    let (mut mism, mut rows) = (Vec::new(), Vec::new());
    let mut worst: f64 = 0.0;
    let mut a = 0;
    while a < n {
        let b = n.min(a + chunk);
        let (out, logits) = m.fire_batch(&[lane(a, seq[a..b].to_vec())], b - a, true)?;
        for (i, &t) in out[0].iter().enumerate() {
            let p = a + i;
            if p < n && t != want[p] {
                mism.push(serde_json::json!({"pos": p, "got": t, "want": want[p], "margin": margins.get(p)}));
            }
        }
        if let (Some(row), Some(r)) = (logits.first(), refl.get(b - 1)) {
            let scale = r.iter().fold(0f64, |s, &x| s.max(f64::from(x).abs()));
            let diff = row.iter().zip(r).fold(0f64, |s, (&x, &y)| {
                s.max((f64::from(x) - f64::from(y)).abs())
            });
            let rel = diff / scale.max(1e-30);
            worst = worst.max(rel);
            rows.push(
                serde_json::json!({"pos": b - 1, "max_abs": diff, "scale": scale, "rel": rel,
                "argmax": common::argmax(row), "want": want[b - 1]}),
            );
            eprintln!(
                "launch {a}..{b}: logits rel {rel:.3e} (max |d| {diff:.3e}, scale {scale:.3})"
            );
        }
        a = b;
    }
    // Greedy from the prompt, on fresh slots past the sequence's pages.
    let g_new = tokens.len();
    let (gout, _) = m.fire_batch(&[lane(0, prompt.clone())], prompt.len() + g_new - 1, false)?;
    let got: Vec<u32> = gout[0][prompt.len() - 1..].to_vec();
    let agree = got.iter().zip(&tokens).take_while(|(a, b)| a == b).count();
    println!(
        "{}",
        serde_json::json!({
            "positions": n, "chunk": chunk, "tf_mismatches": mism, "logit_rows": rows,
            "worst_rel": worst, "greedy": got, "greedy_ref": tokens, "greedy_agree": agree,
            "pass": mism.is_empty() && worst < 1e-3 && agree == tokens.len().min(got.len()),
        })
    );
    Ok(())
}
