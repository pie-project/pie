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
    if model::pass_kind() == model::ForwardKind::Diffusion {
        return Err("use diffusion-parity --only_prefill for a diffusion model".into());
    }
    let ids: Vec<u32> =
        inferlet::serde_json::from_str(&input.ids).map_err(|why| format!("ids json: {why}"))?;
    let n = u32::try_from(ids.len()).map_err(|_| "prompt is too long")?;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let prompt_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    let mut rows = Rows::default();
    for (at, &(base, end)) in readout_spans(n, model::ReadoutKind::Logits)
        .iter()
        .enumerate()
    {
        let len = end - base;
        let toks =
            Channel::from(&prompt_i32[base as usize..end as usize]).named(&format!("toks_{at}"));
        let embed_indptr = Channel::from([0u32, len]).named(&format!("embed_indptr_{at}"));
        let readout = Channel::from_iter(0..len).named(&format!("readout_{at}"));
        let arg_out = Channel::new([len], dtype::i32).named(&format!("arg_{at}"));
        let ent_out = Channel::new([len], dtype::f32).named(&format!("ent_{at}"));
        let ids_out = Channel::new([len, 8], dtype::u32).named(&format!("ids_{at}"));
        let probs_out = Channel::new([len, 8], dtype::f32).named(&format!("probs_{at}"));

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::logits();
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
