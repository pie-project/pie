//! **Prices the RS buffer.**
//!
//! The question this answers is narrow on purpose: on a linear model, what
//! does it cost to BUFFER a decoded token instead of FOLDING it?
//!
//! Both arms run the SAME device-carried decode loop — one bound
//! `ForwardPass` resubmitted per step, `tok_in` looped on the device, no host
//! round trip — so the only independent variable is the RS geometry:
//!
//!   * `fold`   — `fold-len = u32::MAX`. Every fire folds its own token into
//!                the recurrent state. This is what `ForwardPass::recurrent`
//!                does and what every linear decode does today.
//!   * `buffer` — `fold-len = 0` with a buffer grant. Every fire appends its
//!                token to the RS buffer and the folded boundary never moves.
//!                This is the speculative shape: tokens that are never folded
//!                cost nothing to abandon.
//!
//! Neither arm REPLAYS, so this measures the buffer's write and bookkeeping
//! cost alone, not the cost of a commit. That separation is the point: the
//! replay is a per-window cost a speculator pays knowingly, while the write
//! cost is paid by every single token and is the one that decides whether
//! buffering is affordable at all.
//!
//! Input: `"<steps>:<mode>"`, e.g. `"256:buffer"` (default `"256:fold"`).

use inferlet::ptir::hybrid::prelude::*;
use inferlet::ptir::hybrid::submit_frame;
use inferlet::{Result, model as wit_model};

const DEFAULT_STEPS: usize = 256;
const PROMPT: &str = "The history of computing is a history of abstractions, each one \
    hiding a machine that the previous generation had to program directly.";

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let (steps, mode, coframe) = parse(&input);

    let rs = RsWorkingSet::new();
    if rs.state_size() == 0 {
        return Err("rs-buffer-bench needs a model that folds a recurrent state".into());
    }
    // `buffer` never folds, so the buffer must hold the whole run. Asking for
    // it up front keeps allocation out of the timed region in BOTH arms.
    let _buffer_pages = if mode == Mode::Buffer {
        let page = rs.buffer_page_size().max(1);
        let pages = (steps as u32 + 1).div_ceil(page) + 1;
        rs.alloc_buffer(pages)
            .with_context(|| format!("alloc_buffer({pages})"))?;
        pages
    } else {
        0
    };

    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let prompt = wit_model::encode(PROMPT);
    let n = prompt.len() as u32;
    let max_pages = (n + steps as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages).context("ws.reserve")?;

    // ── Prefill. Folds, in BOTH arms: a buffer that starts out holding the
    // whole prompt would price the prompt, not the decode.
    let toks_p = Channel::from_iter(prompt.iter().map(|&t| t as i32));
    let embed_indptr_p = Channel::from([0u32, n]);
    let positions_p = Channel::from_iter(0..n);
    let pages_p = Channel::from_iter(0..max_pages);
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]);
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size));
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size));
    let g0_ch = Channel::new([1], dtype::i32).named("g0");
    // Created BEFORE the prefill so the prefill epilogue can seed it on the
    // DEVICE. That is what lets a decode fire ride in the same frame as the
    // prefill (`coframe`) without waiting for a sampled token.
    let tok_in = Channel::new([1], dtype::i32).named("tok_in");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from([n]);
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
    let tok_in_p = tok_in.clone();
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        tok_in_p.put(&t);
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();

    // ── Decode. One bound pass, resubmitted; identical in both arms except
    // for the RS geometry bound below.
    // The out ring must not be undersized: a publish that finds no free cell
    // POISONS the channel rather than throttling. Size it to the engine's own
    // run-ahead capacity and keep exactly that many fires in flight below.
    let capacity = channel_capacity();
    let window = capacity.saturating_sub(1).max(1);
    let out = Channel::new([1], dtype::i32)
        .capacity(capacity as u32)
        .named("out");
    let lane1 = Channel::from([0u32, 1u32]);
    let positions = Channel::from([n]);
    let pages = Channel::from_iter(0..max_pages);
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]);
    let w_slot = Channel::from([n / page_size]);
    let w_off = Channel::from([n % page_size]);

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    let kv_len = Channel::from([n + 1]);
    let kv = || {
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
        })
    };
    // THE ONLY DIFFERENCE BETWEEN THE ARMS: the RS geometry. The KV half and
    // the working sets are identical, so only `RsGeometry` is written twice —
    // its `buffer` type differs per arm, so the two cannot share a binding.
    let fold_len = Channel::from([0u32]).named("fold_len");
    let rs_ws = std::slice::from_ref(&rs);
    match mode {
        Mode::Fold => fwd.attention(
            kv(),
            rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?,
        Mode::Buffer => fwd
            .attention(
                kv(),
                rs_ws,
                RsGeometry {
                    fold_len: Some(&fold_len),
                    buffer: ..,
                },
            )
            .context("buffered recurrent binding")?,
    }
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

    // Prefill goes out here, not above, because `coframe` needs the decode
    // pass to exist so it can ride in the SAME frame — the shape every real
    // run-ahead scheduler uses and the one that fails on a linear model.
    let mut submitted = 0usize;
    if coframe {
        submit_frame(&pipe, &[Some(&fwd_p), Some(&fwd)]).context("first frame submit")?;
        submitted = 1;
    } else {
        fwd_p.submit(&pipe).context("prefill")?;
    }
    let g0 = g0_ch.take_host::<i32>().await?;
    let mut generated: Vec<u32> = vec![g0 as u32];

    // Timed region: every decode fire, and nothing else. Allocation and
    // binding are above so neither arm pays for them here.
    let fires = steps - 1;
    let clock = std::time::Instant::now();
    while submitted < fires.min(window) {
        fwd.submit(&pipe).context("decode")?;
        submitted += 1;
    }
    for _ in 0..fires {
        let t = out.take_host::<i32>().await?;
        generated.push(t as u32);
        if submitted < fires {
            fwd.submit(&pipe).context("decode")?;
            submitted += 1;
        }
    }
    let elapsed_us = clock.elapsed().as_micros() as u64;

    Ok(report(mode, steps, elapsed_us, &generated))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fold,
    Buffer,
}

fn parse(input: &str) -> (usize, Mode, bool) {
    let mut parts = input.trim().splitn(2, ':');
    let steps = parts
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(DEFAULT_STEPS)
        .max(1);
    let raw = parts.next().map(str::trim).unwrap_or("fold");
    let coframe = raw.starts_with("coframe");
    let mode = if raw.ends_with("buffer") {
        Mode::Buffer
    } else {
        Mode::Fold
    };
    (steps, mode, coframe)
}

fn report(mode: Mode, steps: usize, elapsed_us: u64, generated: &[u32]) -> String {
    let name = match mode {
        Mode::Fold => "fold",
        Mode::Buffer => "buffer",
    };
    let fires = generated.len().saturating_sub(1).max(1) as u64;
    let per_fire_us = elapsed_us as f64 / fires as f64;
    // The token stream is reported so the two arms can be checked for
    // agreement: buffering must not change what the model says, only when the
    // boundary moves.
    let head: Vec<u32> = generated.iter().take(8).copied().collect();
    let result = format!(
        "rs-buffer-bench: mode={name} steps={steps} fires={fires} \
         elapsed_us={elapsed_us} per_fire_us={per_fire_us:.2} \
         tok_per_s={:.1} head={head:?}",
        if elapsed_us == 0 {
            0.0
        } else {
            fires as f64 * 1e6 / elapsed_us as f64
        }
    );
    eprintln!("[RS_BUFFER_BENCH] {result}");
    result
}
