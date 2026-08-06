//! **Greedy decode on a GDN/hybrid model, submitted as FULL k-slot frames.**
//!
//! `generate-gdn` submits its recurrent-state decode pass one slot at a time,
//! because until the RS publication seam moved to prepare time a pass binding
//! recurrent state could not share a frame at all: it serialized behind every
//! predecessor fire's settlement, and a frame's fires cannot settle until all
//! of them are submitted, so the frame sat one fire short of sealing forever.
//! The engine rejected that shape rather than hanging on it.
//!
//! RS mappings now publish at prepare, in slot order, under the store lock —
//! so slot i+1 classifies against slot i's decision without waiting for it,
//! and the advanced state's contents are ordered by the stream like any other
//! intra-frame dependency. This inferlet is the end-to-end proof: it fills
//! EVERY slot of the frame with the same recurrent-state decode pass, so a
//! k = 3 deployment runs three RS fires per frame.
//!
//! Otherwise identical to `generate-gdn`: same greedy-argmax epilogue, same
//! KvLen-root dense geometry, same single bound decode pass resubmitted for
//! the whole run. Input: an optional token budget (default 6), e.g. `"9"`.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

const DEFAULT_MAX_TOKENS: usize = 6;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);
    let k = frame_size();

    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let rs = RsWorkingSet::new();
    let has_rs = rs.state_size() > 0;
    eprintln!(
        "[GDN_FRAME] k={k} rs_state_size={} has_rs={has_rs}",
        rs.state_size()
    );

    if max_tokens == 0 {
        return Ok("generated 0 tokens: []".to_string());
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages).context("ws.reserve")?;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from_iter((0..n).map(|position| position / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|position| position % page_size)).named("w_off_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    fwd_p.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
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
        }),
        std::slice::from_ref(&rs),
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;
    let g0 = g0_ch.take_host::<i32>().await?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    generated.push(g0 as u32);

    // ─────────────────── 2. DECODE, FULL k-SLOT RS FRAMES ───────────────────
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        // One committed cell per wave of the deepest frame in flight, so a
        // full frame's outputs never overflow the ring before the host drains.
        let out = Channel::new([1], dtype::i32)
            .capacity((k * 2) as u32)
            .named("out");
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
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
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
            }),
            std::slice::from_ref(&rs),
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        // Every descriptor is device-advanced, so slot i+1 of the SAME frame
        // reads what slot i wrote — no host round trip inside a frame.
        fwd.epilogue(move || {
            let length = kv_len.take();
            let t = reduce_argmax(intrinsics::logits());
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);
            tok_in.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            out.put(&t);
        });

        let mut remaining = max_tokens - generated.len();
        while remaining > 0 {
            // FULL frame: every slot binds the recurrent-state pass.
            let live = remaining.min(k);
            let slots: Vec<Option<&ForwardPass>> = (0..live).map(|_| Some(&fwd)).collect();
            submit_frame(&pipe, &slots)
                .with_context(|| format!("rs frame submit (live={live}, k={k})"))?;
            for wave in 0..live {
                let t = out
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@wave {wave}"))?;
                let Some(&t0) = t.first() else {
                    return Err(format!("out.take @wave {wave}: empty tensor"));
                };
                generated.push(t0 as u32);
            }
            remaining -= live;
        }
    }
    pipe.close();

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GDN_FRAME] {result}");
    Ok(result)
}
