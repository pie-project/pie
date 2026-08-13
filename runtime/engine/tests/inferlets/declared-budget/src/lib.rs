//! **Admission-demand probe guest.** The exact shape of the tts-bench decoder
//! inferlet's `Forward` (`ttb-inferlet-session/src/decode.rs`): the pages
//! channel spans the pool implied by the DECLARED output budget
//! (`prompt + declared + 1` tokens), while the logical reservation
//! (`reserve`) and the decode loop advance one token at a time and stop
//! after `actual` tokens.
//!
//! Input: `"<declared>,<actual>"` — declared output-token budget and the
//! number of tokens really generated. A correct, incremental engine admits
//! this guest on a physical pool that holds `prompt + actual` (plus one
//! headroom page), regardless of how large `declared` is; an engine that
//! prices the declaration at admission rejects it.

use inferlet::ptir::attention::prelude::*;
use inferlet::{Result, model as wit_model};

const ECHO_TOKEN: i32 = 42;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let (declared, actual) = input
        .trim()
        .split_once(',')
        .ok_or_else(|| format!("input must be \"declared,actual\", got {input:?}"))?;
    let declared: u32 = declared.trim().parse().map_err(|e| format!("declared: {e}"))?;
    let actual: usize = actual.trim().parse().map_err(|e| format!("actual: {e}"))?;
    if actual == 0 {
        return Err("actual must be >= 1 (the decode loop must run)".to_string());
    }

    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    // The decoder inferlet's page ceiling: committed + prompt + DECLARED
    // budget + 1, in pages (decode.rs `Forward::new`).
    let max_pages = (n + declared + 1).div_ceil(page_size);
    let reserve_to_tokens = |tokens: u32| -> std::result::Result<(), String> {
        let target = tokens.div_ceil(page_size).saturating_add(1).min(max_pages);
        let current = ws.page_len();
        if current < target {
            ws.reserve(target - current)?;
        }
        Ok(())
    };
    reserve_to_tokens(n.max(1)).context("ws.reserve prompt")?;

    // Prefill fire: pages channel spans the DECLARED pool; page_indptr
    // selects the live prefix (the decoder's `prefill` shape).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from_iter((0..n).map(|position| position / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|position| position % page_size)).named("w_off_p");
    let echo_p = Channel::from([ECHO_TOKEN]).named("echo_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len_p,
            pages: &pages_p,
            page_indptr: &page_indptr_p,
            w_slot: &w_slot_p,
            w_off: &w_off_p,
            positions: &positions_p,
            mask: None,
        },
    )?;
    fwd_p.epilogue(move || {
        let t = echo_p.take();
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;
    let g0 = g0_ch.take_host::<i32>().await?;

    let mut generated: Vec<u32> = vec![g0 as u32];

    // Decode loop, device loop-carried, EXACTLY the decoder's
    // `build_decode_loop`: the pages channel spans the declared pool.
    if generated.len() < actual {
        let tok_in = Channel::from([g0]).named("tok_in");
        let echo = Channel::from([ECHO_TOKEN]).named("echo");
        let out = Channel::new([1], dtype::i32).named("out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        let kv_len = Channel::from([n + 1]).named("kv_len");
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
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
            let t = echo.take();
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);
            let next_page_indptr = indptr(1, &page_count);
            tok_in.put(&t);
            echo.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(&next_page_indptr);
            out.put(&t);
        });

        for step in 1..actual {
            reserve_to_tokens(n + step as u32)
                .with_context(|| format!("reserve decode @{step}"))?;
            fwd.submit(&pipe)
                .with_context(|| format!("decode submit @{step}"))?;
            let t = out
                .take_host::<Vec<i32>>()
                .await
                .with_context(|| format!("@{step}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            generated.push(t0 as u32);
        }
    }
    pipe.close();

    let result = format!(
        "declared {declared} generated {} tokens: {:?}",
        generated.len(),
        generated
    );
    eprintln!("[DECLARED-BUDGET] {result}");
    Ok(result)
}
