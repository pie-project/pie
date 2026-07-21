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

async fn append_tokens(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    start: u32,
    tokens: &[u32],
    _last: bool,
) -> Result<i32> {
    if tokens.is_empty() {
        return Err("cannot append an empty token sequence".into());
    }
    let n = tokens.len() as u32;
    let total = start + n;
    // The generated geometry spans `max_pages`; extend the (purely logical)
    // lease so it covers the appended extent by fire time.
    let max_pages = total.div_ceil(ws.page_size()).max(1);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have)
            .map_err(|e| format!("reserve append KV: {e}"))?;
    }
    let token_input = Channel::from(tokens.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let embed_indptr = Channel::from(vec![0u32, n]).named("embed_indptr");
    let positions = Channel::from((start..total).collect::<Vec<_>>()).named("positions");
    let pages = Channel::from((0..ws.page_len()).collect::<Vec<_>>()).named("pages");
    let page_indptr =
        Channel::from(vec![0u32, total.div_ceil(ws.page_size())]).named("page_indptr");
    let w_slot = Channel::from(
        (start..total)
            .map(|p| p / ws.page_size())
            .collect::<Vec<_>>(),
    )
    .named("w_slot");
    let w_off = Channel::from(
        (start..total)
            .map(|p| p % ws.page_size())
            .collect::<Vec<_>>(),
    )
    .named("w_off");
    let next_token = Channel::new([1], dtype::i32).named("next_token");

    let fwd = ForwardPass::new();
    fwd.embed(&token_input, &embed_indptr)?;
    let kv_len = Channel::from(vec![total]).named("kv_len");
    fwd.attention(
        ws,
        ..,
        (start / ws.page_size())..,
        &kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )?;
    fwd.epilogue(move || {
        next_token.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    fwd.submit(pipeline)
        .map_err(|e| format!("append shared prefix: {e}"))?;
    Ok(next_token
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read branch token: {e}"))?[0])
}

async fn generate(
    ws: &WorkingSet,
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

    // The generated geometry spans `max_pages`; extend the (purely logical)
    // lease so it covers the whole decode by fire time.
    let max_pages = (seq_len + max_tokens as u32 + 1)
        .div_ceil(ws.page_size())
        .max(1);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have)
            .map_err(|e| format!("reserve leaf KV: {e}"))?;
    }
    let token_in = Channel::from(vec![first_token]).named("token_in");
    let page_size = ws.page_size();
    let embed_indptr = Channel::from(vec![0u32, 1]).named("embed_indptr");
    let positions = Channel::from(vec![seq_len]).named("positions");
    let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
    let page_indptr =
        Channel::from(vec![0u32, (seq_len + 1).div_ceil(page_size)]).named("page_indptr");
    let w_slot = Channel::from(vec![seq_len / page_size]).named("w_slot");
    let w_off = Channel::from(vec![seq_len % page_size]).named("w_off");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("token_out");

    let fwd = ForwardPass::new();
    fwd.embed(&token_in, &embed_indptr)?;
    let kv_len = Channel::from(vec![seq_len + 1]).named("kv_len");
    fwd.attention(
        ws,
        ..,
        (seq_len / page_size)..,
        &kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )?;
    fwd.epilogue(move || {
        let length = kv_len.take().tensor();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next_length = add(&length, 1u32);
        let page_count = div(add(&next_length, page_size - 1), page_size);

        token_in.put(&token);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(div(&length, page_size));
        w_off.put(rem(&length, page_size));
        page_indptr.take();
        page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
        token_out.put(&token);
    });

    let pipeline = Pipeline::new();
    let budget = max_tokens.saturating_sub(generated.len());
    let mut submitted = 0usize;
    let mut in_flight = 0usize;
    while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
        fwd.submit(&pipeline)
            .map_err(|e| format!("generate leaf: {e}"))?;
        submitted += 1;
        in_flight += 1;
    }
    while in_flight > 0 {
        let token = token_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("read leaf token: {e}"))?[0] as u32;
        in_flight -= 1;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        if submitted < budget {
            fwd.submit(&pipeline)
                .map_err(|e| format!("generate leaf: {e}"))?;
            submitted += 1;
            in_flight += 1;
        }
    }
    while in_flight > 0 {
        token_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("drain leaf run-ahead token: {e}"))?;
        in_flight -= 1;
    }
    pipeline.close();
    Ok(generated)
}

struct Branch {
    label: String,
    ws: WorkingSet,
    seq_len: u32,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let root = WorkingSet::new();

    let root_tokens = wit_model::encode("Write a short scene set");
    if root_tokens.is_empty() {
        return Err("tokenizer produced an empty root prompt".into());
    }

    let tree_pipeline = Pipeline::new();
    append_tokens(&root, &tree_pipeline, 0, &root_tokens, false).await?;
    let root_len = root_tokens.len() as u32;

    let mut first_level = Vec::new();
    for suffix in [" in a city", " in a forest"] {
        let child = root.fork(&tree_pipeline)?;
        let tokens = wit_model::encode(suffix);
        append_tokens(&child, &tree_pipeline, root_len, &tokens, false).await?;
        first_level.push(Branch {
            label: suffix.trim().into(),
            ws: child,
            seq_len: root_len + tokens.len() as u32,
        });
    }

    let mut leaves = Vec::new();
    let num_parents = first_level.len();
    let leaf_suffixes = [" at dawn", " at night"];
    for (pi, parent) in first_level.into_iter().enumerate() {
        for (si, suffix) in leaf_suffixes.into_iter().enumerate() {
            let leaf = parent.ws.fork(&tree_pipeline)?;
            let tokens = wit_model::encode(suffix);
            // The last leaf's append is the build stream's final submission.
            let last = pi + 1 == num_parents && si + 1 == leaf_suffixes.len();
            let first = append_tokens(&leaf, &tree_pipeline, parent.seq_len, &tokens, last).await?;
            leaves.push((
                format!("{} {}", parent.label, suffix.trim()),
                leaf,
                parent.seq_len + tokens.len() as u32,
                first,
            ));
        }
    }
    // Every append's token was awaited above, so the build stream is fully
    // drained — this cancels nothing (R4-1).
    tree_pipeline.close();

    let mut outputs = Vec::with_capacity(leaves.len());
    for (label, ws, seq_len, first) in leaves {
        let generated = generate(&ws, seq_len, first, input.num_tokens).await?;
        outputs.push(format!("{label}: {}", wit_model::decode(&generated)?));
    }
    Ok(outputs.join("\n"))
}
