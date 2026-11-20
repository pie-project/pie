
use inferlet::eta::adapter::Peft;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_top")]
    top: u32,
}

fn default_max_tokens() -> usize {
    8
}

fn default_top() -> u32 {
    20
}

#[derive(Serialize)]
struct Output {
    tokens: Vec<u32>,
    top: Vec<(u32, f32)>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let peft = match (
        inferlet::session::file("adapter_config.json"),
        inferlet::session::file("adapter_model.safetensors"),
    ) {
        (Some(config), Some(weights)) => Some(Peft::load(&config, &weights)?),
        _ => None,
    };
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let toks = Channel::from(&prompt[..]);
    let indptr = Channel::from([0u32, prompt.len() as u32]);
    let first = Channel::new([1], dtype::i32);
    let (values, ids) = (
        Channel::new([input.top], dtype::f32),
        Channel::new([input.top], dtype::i32),
    );
    let top = input.top;
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    if let Some(peft) = &peft {
        fwd.peft(peft)?;
    }
    fwd.epilogue(move || {
        let logits = reshape(
            intrinsics::output(0, "logits"),
            [model::output_vocab_size()],
        );
        first.put(reshape(reduce_argmax(&logits), [1]));
        let (v, i) = top_k(&logits, top);
        values.put(&v);
        ids.put(cast(&i, dtype::i32));
    });
    fwd.submit(&pipe)?;
    let mut tokens = vec![first.take_host::<i32>().await? as u32];
    let values = values.take_host::<Vec<f32>>().await?;
    let ids = ids.take_host::<Vec<i32>>().await?;

    if input.max_tokens > 1 {
        let tok_in = Channel::from([tokens[0] as i32]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let indptr = Channel::from([0u32, 1]);
        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        if let Some(peft) = &peft {
            fwd.peft(peft)?;
        }
        fwd.epilogue(move || {
            let token = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
            tok_in.put(&token);
            tok_out.put(&token);
        });
        run_ahead(&pipe, &fwd, input.max_tokens - 1, async || {
            tokens.push(tok_out.take_host::<i32>().await? as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();
    Ok(Output {
        tokens,
        top: ids.iter().map(|&i| i as u32).zip(values).collect(),
    })
}
