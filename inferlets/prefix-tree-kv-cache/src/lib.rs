//! Builds a two-level prompt tree with copy-on-write KV-cache sharing.
//!
//! The common prompt is prefilled once. Two first-level branches fork that
//! working set, append distinct text, and are each forked again into two leaves.
//! Generation then continues independently from all four shared-prefix leaves.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_num_tokens")]
    num_tokens: usize,
}

fn default_num_tokens() -> usize {
    32
}

fn bx<T>(value: T) -> &'static T {
    Box::leak(Box::new(value))
}

fn append_tokens(
    ws: &'static WorkingSet,
    pipeline: &Pipeline,
    start: u32,
    tokens: &[u32],
) -> Result<i32> {
    if tokens.is_empty() {
        return Err("cannot append an empty token sequence".into());
    }
    let n = tokens.len() as u32;
    let total = start + n;
    let token_input = bx(Channel::from(
        tokens.iter().map(|&token| token as i32).collect::<Vec<_>>(),
    ));
    let positions = bx(Channel::from((start..total).collect::<Vec<_>>()));
    let klen = bx(Channel::from(vec![total]));
    let next_token = bx(Channel::new([1], dtype::i32).named("next_token"));

    let fwd: ForwardPass<'static> = ForwardPass::new();
    fwd.embed(token_input, Tensor::constant(vec![0u32, n]));
    fwd.positions(positions);
    fwd.attn_working_set(ws, klen);
    fwd.epilogue(move || {
        next_token.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    fwd.submit(pipeline)
        .map_err(|e| format!("append shared prefix: {e}"))?;
    Ok(next_token
        .take()
        .get::<i32>()
        .map_err(|e| format!("read branch token: {e}"))?[0])
}

fn generate(
    ws: &'static WorkingSet,
    seq_len: u32,
    first_token: i32,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }

    let stop_tokens = chat::stop_tokens();
    let mut generated = Vec::with_capacity(max_tokens);
    if !stop_tokens.contains(&(first_token as u32)) {
        generated.push(first_token as u32);
    }
    if generated.len() >= max_tokens || stop_tokens.contains(&(first_token as u32)) {
        return Ok(generated);
    }

    let token_in = bx(Channel::from(vec![first_token]).named("token_in"));
    let position = bx(Channel::from(vec![seq_len]).named("position"));
    let klen = bx(Channel::from(vec![seq_len + 1]).named("klen"));
    let fill = bx(Channel::from(vec![seq_len + 1]).named("fill"));
    let token_out = bx(Channel::new([1], dtype::i32).named("token_out"));

    let fwd: ForwardPass<'static> = ForwardPass::new();
    fwd.embed(token_in, Tensor::constant(vec![0u32, 1]));
    fwd.positions(position);
    fwd.attn_working_set(ws, klen);
    fwd.epilogue(move || {
        let base = fill.take().tensor();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next = add(&base, 1u32);

        token_in.put(&token);
        token_out.put(&token);
        position.put(&base);
        klen.put(&next);
        fill.put(&next);
    });

    let pipeline = Pipeline::new();
    while generated.len() < max_tokens {
        fwd.submit(&pipeline)
            .map_err(|e| format!("generate leaf: {e}"))?;
        let token = token_out
            .take()
            .get::<i32>()
            .map_err(|e| format!("read leaf token: {e}"))?[0] as u32;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
    }
    pipeline.close();
    Ok(generated)
}

struct Branch {
    label: String,
    ws: &'static WorkingSet,
    seq_len: u32,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let root: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, root.page_size(), 1);

    let root_tokens = wit_model::encode("Write a short scene set");
    if root_tokens.is_empty() {
        return Err("tokenizer produced an empty root prompt".into());
    }

    let tree_pipeline = Pipeline::new();
    append_tokens(root, &tree_pipeline, 0, &root_tokens)?;
    let root_len = root_tokens.len() as u32;

    let mut first_level = Vec::new();
    for suffix in [" in a city", " in a forest"] {
        let child: &'static WorkingSet = bx(root.fork(&tree_pipeline)?);
        let tokens = wit_model::encode(suffix);
        append_tokens(child, &tree_pipeline, root_len, &tokens)?;
        first_level.push(Branch {
            label: suffix.trim().into(),
            ws: child,
            seq_len: root_len + tokens.len() as u32,
        });
    }

    let mut leaves = Vec::new();
    for parent in first_level {
        for suffix in [" at dawn", " at night"] {
            let leaf: &'static WorkingSet = bx(parent.ws.fork(&tree_pipeline)?);
            let tokens = wit_model::encode(suffix);
            let first = append_tokens(leaf, &tree_pipeline, parent.seq_len, &tokens)?;
            leaves.push((
                format!("{} {}", parent.label, suffix.trim()),
                leaf,
                parent.seq_len + tokens.len() as u32,
                first,
            ));
        }
    }
    tree_pipeline.close();

    let mut outputs = Vec::with_capacity(leaves.len());
    for (label, ws, seq_len, first) in leaves {
        let generated = generate(ws, seq_len, first, input.num_tokens)?;
        outputs.push(format!("{label}: {}", wit_model::decode(&generated)?));
    }
    Ok(outputs.join("\n"))
}
