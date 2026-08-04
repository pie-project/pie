//! Builds a two-level prompt tree with copy-on-write KV-cache sharing.
//!
//! The common prompt is prefilled once. Two first-level branches fork that
//! working set, append distinct text, and are each forked again into two leaves.
//! Generation then continues independently from all four shared-prefix leaves.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
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
    let max_pages = total.div_ceil(kv_page_size()).max(1);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have).context("reserve append KV")?;
    }
    let token_input = Channel::from_iter(tokens.iter().map(|&token| token as i32));
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr");
    let positions = Channel::from_iter(start..total).named("positions");
    let pages = Channel::from_iter(0..ws.page_len()).named("pages");
    let page_indptr = Channel::from([0u32, total.div_ceil(kv_page_size())]).named("page_indptr");
    let w_slot = Channel::from_iter((start..total).map(|p| p / kv_page_size())).named("w_slot");
    let w_off = Channel::from_iter((start..total).map(|p| p % kv_page_size())).named("w_off");
    let next_token = Channel::new([1], dtype::i32).named("next_token");

    let fwd = ForwardPass::new();
    fwd.embed(&token_input, &embed_indptr)?;
    let kv_len = Channel::from([total]).named("kv_len");
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (start / kv_page_size())..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        next_token.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    fwd.submit(pipeline).context("append shared prefix")?;
    Ok(next_token.take_host::<i32>().await?)
}

async fn generate(
    ws: &WorkingSet,
    pipeline: &Pipeline,
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
        .div_ceil(kv_page_size())
        .max(1);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have).context("reserve leaf KV")?;
    }
    let token_in = Channel::from([first_token]).named("token_in");
    let page_size = kv_page_size();
    let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
    let positions = Channel::from([seq_len]).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, (seq_len + 1).div_ceil(page_size)]).named("page_indptr");
    let w_slot = Channel::from([seq_len / page_size]).named("w_slot");
    let w_off = Channel::from([seq_len % page_size]).named("w_off");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("token_out");

    let fwd = ForwardPass::new();
    fwd.embed(&token_in, &embed_indptr)?;
    let kv_len = Channel::from([seq_len + 1]).named("kv_len");
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (seq_len / page_size)..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        let length = kv_len.take();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(page_size);

        token_in.put(&token);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(&length / page_size);
        w_off.put(&length % page_size);
        page_indptr.put(indptr(1, &page_count));
        token_out.put(&token);
    });

    let budget = max_tokens.saturating_sub(generated.len());
    run_ahead(&pipeline, &fwd, budget as usize, async || {
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
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

    let root_tokens = model::encode("Write a short scene set");
    if root_tokens.is_empty() {
        return Err("tokenizer produced an empty root prompt".into());
    }

    let tree_pipeline = Pipeline::new();
    append_tokens(&root, &tree_pipeline, 0, &root_tokens, false).await?;
    let root_len = root_tokens.len() as u32;

    let mut first_level = Vec::new();
    for suffix in [" in a city", " in a forest"] {
        let child = root.fork(&tree_pipeline)?;
        let tokens = model::encode(suffix);
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
            let tokens = model::encode(suffix);
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
    // A KV working set is scoped to the FIRST pipeline that fires it, and every
    // leaf was built on `tree_pipeline` — so generation must stay on that same
    // stream. It closes once, after the last leaf is drained.
    let mut outputs = Vec::with_capacity(leaves.len());
    for (label, ws, seq_len, first) in leaves {
        let generated = generate(&ws, &tree_pipeline, seq_len, first, input.num_tokens).await?;
        outputs.push(format!("{label}: {}", model::decode(&generated)?));
    }
    tree_pipeline.close();
    Ok(outputs.join("\n"))
}
