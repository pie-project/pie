//! Teacher-forced prefill, every row read out, for `scripts/diffusiongemma_parity_ref.py`.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    ids: String,
}

#[derive(Serialize, Default)]
struct Rows {
    argmax: Vec<i32>,
    entropy: Vec<f32>,
    top8_ids: Vec<u32>,
    top8_probs: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    prefill_rows: Rows,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let ids: Vec<u32> =
        inferlet::serde_json::from_str(&input.ids).map_err(|why| format!("ids json: {why}"))?;
    let n = u32::try_from(ids.len()).map_err(|_| "prompt is too long")?;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let prompt_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    let mut rows = Rows::default();
    let max = model::lane_of(&[])?
        .outputs
        .iter()
        .find(|output| output.name == "logits")
        .map_or(1, |output| output.max_rows);
    for &(base, end) in readout_spans(n, max).iter() {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]);
        let embed_indptr = Channel::from([0u32, len]);
        let readout = Channel::from_iter(0..len);
        let arg_out = Channel::new([len], dtype::i32);
        let ent_out = Channel::new([len], dtype::f32);
        let ids_out = Channel::new([len, 8], dtype::u32);
        let probs_out = Channel::new([len, 8], dtype::f32);

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::output(0, "logits");
            let probs = softmax(&logits);
            let (top_p, top_i) = top_k(&probs, 8);
            arg_out.put(reduce_argmax(&logits));
            ent_out.put(entropy(&probs));
            ids_out.put(top_i);
            probs_out.put(top_p);
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        rows.argmax.extend(arg_out.take_host::<Vec<i32>>().await?);
        rows.entropy.extend(ent_out.take_host::<Vec<f32>>().await?);
        rows.top8_ids.extend(ids_out.take_host::<Vec<u32>>().await?);
        rows.top8_probs
            .extend(probs_out.take_host::<Vec<f32>>().await?);
    }
    pipe.close();
    Ok(Output { prefill_rows: rows })
}
