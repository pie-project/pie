//! `cpu_golden`: a decode deployment's golden from Pie's own CPU path (`engine_cpu::Forward`
//! over the checkpoint as a load lands it): the prompt's tokens and greedy steps after them, a
//! token a step, each step's top five logits; the JSON `wafer-bench --decode` holds a wafer run
//! to (an HF golden's shape). With `--against <golden.json>`, the run is teacher-forced through
//! that golden's tokens instead and each step's argmax and top logit compared with it (a check
//! of the CPU path against HF's).
//!
//! ```text
//! cargo run --release -p engine-cerebras --example cpu_golden -- --model <deployment>
//!     --checkpoint <dir> (--prompt 1,2,3 --steps N | --against <golden.json>) --out <golden.json>
//! ```

use std::collections::HashMap;

use serde_json::{Value, json};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get: HashMap<String, String> = args
        .chunks(2)
        .filter_map(|p| Some((p[0].strip_prefix("--")?.to_string(), p.get(1)?.clone())))
        .collect();
    let need = |k: &str| get.get(k).cloned().ok_or(format!("--{k}"));
    let model = need("model")?;
    let trace = poem_compiler::catalog::repository()
        .parse(&model)
        .ok_or_else(|| format!("{model}: no such deployment"))?
        .trace(poem_ir::Platform::Cerebras);
    let ckpt =
        std::fs::canonicalize(need("checkpoint")?).map_err(|e| format!("--checkpoint: {e}"))?;
    let contract = poem_compiler::catalog::repository().contract_for(&trace, &ckpt)?;
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt)?;
    let of = |name: &str, count: usize| engine_cerebras::prepare::plane(&landed, name, count);
    let mut fwd = engine_cpu::Forward::new(trace.clone(), None, &of)?;
    let word = poem_compiler::spatial::layer::stream::decode_classes(&trace, 1)[0].word;
    let against: Option<Value> = get
        .get("against")
        .map(|p| {
            serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)
                .map_err(|e| format!("{p}: {e}"))
        })
        .transpose()?;
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|t| t as u32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let (prompt, forced, steps) = match &against {
        Some(g) => {
            let toks = ids(&g["tokens"]);
            (ids(&g["prompt"]), Some(toks.clone()), toks.len() - 1)
        }
        None => {
            let p: Vec<u32> = need("prompt")?
                .split(',')
                .map(|t| t.parse().map_err(|e| format!("--prompt: {e}")))
                .collect::<Result<_, _>>()?;
            let n: usize = need("steps")?
                .parse()
                .map_err(|e| format!("--steps: {e}"))?;
            let len = p.len();
            (p, None, len + n - 1)
        }
    };
    let mut seq = prompt.clone();
    let mut top5 = Vec::new();
    let mut agree = 0;
    for s in 0..steps {
        let tok = match &forced {
            Some(f) => f[s],
            None => seq[s],
        };
        let outs = fwd.step(&[tok], s as u32, word)?;
        let logits = outs
            .iter()
            .find(|o| o.name == "logits")
            .ok_or("no logits")?;
        let engine_cpu::forward::Values::F32(l) = &logits.values else {
            return Err("logits not f32".into());
        };
        let mut order: Vec<usize> = (0..l.len()).collect();
        order.sort_by(|a, b| l[*b].total_cmp(&l[*a]));
        let top: Vec<(usize, f32)> = order.iter().take(5).map(|&i| (i, l[i])).collect();
        if s + 1 >= prompt.len() {
            top5.push(json!(
                top.iter().map(|(i, v)| json!([i, v])).collect::<Vec<_>>()
            ));
            if let Some(g) = &against {
                let k = s + 1 - prompt.len();
                let want = g["top5"][k][0][0].as_u64().unwrap_or(0) as usize;
                let wl = g["top5"][k][0][1].as_f64().unwrap_or(0.0);
                agree += usize::from(want == top[0].0);
                println!(
                    "CPU_STEP {s}: argmax {} ({:.4}), golden {want} ({wl:.4})",
                    top[0].0, top[0].1
                );
            }
            if forced.is_none() {
                seq.push(top[0].0 as u32);
            }
        }
    }
    if let Some(g) = &against {
        println!(
            "CPU_AGAINST {agree} of {} argmaxes the golden's",
            g["top5"].as_array().map_or(0, Vec::len)
        );
    }
    let out =
        json!({ "model": model, "prompt": prompt, "tokens": forced.unwrap_or(seq), "top5": top5 });
    let path = need("out")?;
    std::fs::write(
        &path,
        serde_json::to_string(&out).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{path}: {e}"))?;
    println!("CPU_GOLDEN {path}");
    Ok(())
}
