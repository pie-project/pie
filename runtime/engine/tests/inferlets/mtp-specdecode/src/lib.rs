//! **MTP Stage-2 — spec-decode, drafts-channel swap of `mtp-native-verify`.**
//!
//! The ORIGINAL design read this window's drafts DEVICE-RESIDENT via a
//! `Binding::MtpDrafts` intrinsic plus a driver-side `carrier::next_inputs_drafts`
//! retain/inject command, so the `[k]` drafts never touched the host. Both the
//! author-facing eDSL wrapper and the retain/inject WIT command were REMOVED in
//! the ptir refactor (there is no `intrinsics::mtp_drafts()` in `pie-dsl`, and no
//! `pipeline_source_kind`/retain surface in the current `forward*`/`pipeline` WIT
//! interfaces). That is a genuine capability gap, not a stale rename.
//!
//! So the A-vs-B comparison in `cuda_mtp_specdecode_ab.rs` is, for now, a
//! comparison of two host-round-trip decoders, and its perf delta is noise. What
//! this inferlet still earns its keep for is COVERAGE: it is a second, independent
//! `pie:inferlet/forward-hybrid` client driving the same fold/discard boundary
//! machinery, so a boundary regression has to break two call sites to go unnoticed.
//!
//! **This is a LINEAR model, so the window is one fire, not a staircase.** It used
//! to build a `k+1`-REQUEST staircase through `pie:inferlet/forward`. That is
//! inexpressible here on two counts: the generic `forward` interface refuses a
//! model with a folded recurrent state, and a linear model carries one recurrent
//! state per REQUEST, so `k+1` rows would demand `k+1` divergent copies of a single
//! sequence's state. The shape that has a single state to advance is ONE request
//! row of `k+1` causal tokens.
//!
//! Each window is a single fire because the fold looks BACKWARD:
//!
//!   * `fold-len` = the PREVIOUS window's accepted prefix, which is known by now
//!     and still sits in the buffer behind this fire's new tokens;
//!   * this fire does NOT fold its own `k` drafts — how many are real is decided
//!     by its own logits, and a fold is irreversible;
//!   * the rejected tail is dropped with `discard-buffered`, which moves the live
//!     end left without emptying the buffer, leaving the accepted prefix parked
//!     for the NEXT fire to fold.
//!
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;
const PAGE_T: u32 = 16;

/// Committed length of a sentinel `[k+1]` tail = the count before the first
/// `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

fn bind_single_sequence<B>(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    token_count: u32,
    pool_pages: u32,
    readout: &[u32],
    rs: &[RsWorkingSet],
    rs_geom: RsGeometry<'_, B>,
) -> Result<()>
where
    B: std::ops::RangeBounds<u32>,
{
    let embed_indptr = Channel::from([0u32, token_count]).named("embed_indptr");
    let positions = Channel::from_iter(0..token_count).named("positions");
    let pages = Channel::from_iter(0..pool_pages).named("pages");
    let page_indptr = Channel::from([0u32, token_count.div_ceil(PAGE_T)]).named("page_indptr");
    let w_slot = Channel::from_iter((0..token_count).map(|p| p / PAGE_T)).named("w_slot");
    let w_off = Channel::from_iter((0..token_count).map(|p| p % PAGE_T)).named("w_off");
    let readout = Channel::from(readout).named("readout");
    pass.embed(toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        rs,
        rs_geom,
    )
}

/// One request row of `count` tokens starting at absolute position `first_pos`,
/// reading out the rows named by `readout` (row indices within the window).
fn bind_window<B>(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    first_pos: u32,
    count: u32,
    pool_pages: u32,
    readout: &[u32],
    rs: &[RsWorkingSet],
    rs_geom: RsGeometry<'_, B>,
) -> Result<()>
where
    B: std::ops::RangeBounds<u32>,
{
    let embed_indptr = Channel::from([0u32, count]).named("w_embed_indptr");
    let positions = Channel::from_iter(first_pos..first_pos + count).named("w_positions");
    let pages = Channel::from_iter(0..pool_pages).named("w_pages");
    let page_indptr = Channel::from([0u32, (first_pos + count).div_ceil(PAGE_T).min(pool_pages)])
        .named("w_page_indptr");
    let w_slot =
        Channel::from_iter((first_pos..first_pos + count).map(|p| p / PAGE_T)).named("w_w_slot");
    let w_off =
        Channel::from_iter((first_pos..first_pos + count).map(|p| p % PAGE_T)).named("w_w_off");
    let readout = Channel::from(readout).named("w_readout");
    pass.readout(&readout)?;
    pass.embed(toks, &embed_indptr)?;
    pass.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        rs,
        rs_geom,
    )
}

/// Bootstrap fire over `prompt + (k-1)` fillers: yields the seed (row-0 target
/// argmax at the prompt's REAL last position) + the first REAL `[k]` drafts
/// (native MTP argmax) for window 1.
async fn bootstrap(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipeline: &Pipeline,
    prompt: &[u32],
    k: u32,
    max_pages: u32,
) -> Result<(i32, Vec<i32>)> {
    let l = prompt.len() as u32;
    let mut window: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    window.extend(std::iter::repeat(0i32).take((k - 1) as usize));
    let n = l + k - 1;

    let toks = Channel::from(window).named("b_toks");
    let seed_out = Channel::new([1], dtype::i32).named("b_seed");
    let drafts_out = Channel::new([k], dtype::i32).named("b_drafts");

    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from([n]).named("b_kv_len");
    // The prompt is final by construction, so the bootstrap folds all of it.
    bind_single_sequence(
        &fwd,
        ws,
        &toks,
        &kv_len,
        n,
        max_pages,
        &readout,
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits());
        let seed = gather(&picked, Tensor::constant(vec![0u32]));
        let mtp = intrinsics::mtp_logits(k);
        let drafts = reduce_argmax(mtp);
        seed_out.put(&seed);
        drafts_out.put(&drafts);
    });

    fwd.submit(pipeline).context("bootstrap submit")?;
    let seed = seed_out
        .take_host::<Vec<i32>>()
        .await?
        .first()
        .copied()
        .ok_or_else(|| "bootstrap: empty seed".to_string())?;
    let drafts = drafts_out.take_host::<Vec<i32>>().await?;
    Ok((seed, drafts))
}

/// One `[k+1]`-wide verify window, in ONE fire: embed `[seed, draft]` at
/// positions derived from the pre-envelope `seq_len` cursor, fold the PREVIOUS
/// window's accepted prefix (`fold_len`, already buffered behind these tokens),
/// verify `draft` (device-alias peeked off the SAME embedded tokens) against the
/// target's per-row argmax, and draft the NEXT window natively off `mtp_logits`.
/// Returns `(commit [k+1], next_drafts [k])`.
#[allow(clippy::too_many_arguments)]
async fn verify_window(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipeline: &Pipeline,
    k: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
    max_pages: u32,
    fold_len: u32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = Channel::from(window).named("v_toks");
    let commit_out = Channel::new([kp1], dtype::i32).named("v_commit");
    let drafts_out = Channel::new([k], dtype::i32).named("v_drafts");
    let commit_out_h = commit_out.clone();
    let drafts_out_h = drafts_out.clone();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from([seq_len + kp1]).named("v_kv_len");
    let readout: Vec<u32> = (0..kp1).collect();
    // Fold BEHIND, never ahead: `fold_len` is the previous window's accepted
    // prefix, whose finality is settled. This fire's own k drafts land in the
    // buffer with the boundary held still, so a rejected tail stays abandonable.
    let fold_len = Channel::from([fold_len]).named("v_fold_len");
    bind_window(
        &fwd,
        ws,
        &toks,
        &kv_len,
        seq_len,
        kp1,
        max_pages,
        &readout,
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: Some(&fold_len),
            buffer: ..,
        },
    )
    .context("verify binding")?;
    fwd.epilogue(move || {
        let win = toks.read(); // [k+1] i32 device-alias peek
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>()));
        let picked = reduce_argmax(intrinsics::logits()); // [k+1]
        let head = gather(&picked, iota(k));
        let hit = eq(&head, &draft_v);
        let ones = broadcast(1.0f32, [k]);
        let zeros = broadcast(0.0f32, [k]);
        let run = cumprod(select(&hit, &ones, &zeros));
        let n_acc = cast(reduce_sum(run), dtype::u32);
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1));
        let neg1 = broadcast(-1i32, [kp1]);
        let commit = select(&keep, &picked, &neg1);

        let mtp = intrinsics::mtp_logits(k);
        let next_drafts = reduce_argmax(mtp);

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    fwd.submit(pipeline).context("verify submit")?;
    let commit = commit_out_h.take_host::<Vec<i32>>().await?;
    let drafts = drafts_out_h.take_host::<Vec<i32>>().await?;
    Ok((commit, drafts))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    if wit_model::pass_kind() == wit_model::ForwardKind::Attention {
        // Same guard as `mtp-native-verify`: the buffer this decoder's accept
        // step depends on only exists on a linear model.
        return Ok("skipped: mtp-specdecode needs a linear model".to_string());
    }
    let ws = WorkingSet::new();
    let rs = RsWorkingSet::new();
    let rs_page = rs.buffer_page_size();
    if rs_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let max_pages = (prompt.len() as u32 + MAX_TOKENS + k + 1).div_ceil(PAGE_T);
    ws.reserve(max_pages).context("ws.reserve")?;

    // ONE pipeline for the whole stream (R4-4): the bootstrap and every
    // verify window continue the same sequential decode, so all their fires
    // submit here. The loop is acceptance-driven (the last submit is not
    // knowable at submit time), so the stream ends with a close after the
    // final drain instead of a final-submit marker.
    let pipeline = Pipeline::new();

    let (seed0, draft0) = bootstrap(&ws, &rs, &pipeline, &prompt, k, max_pages).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 1;

    let kp1 = k + 1;
    let window_slabs = kp1.div_ceil(rs_page);
    // The previous window's accepted prefix, still buffered, waiting for the
    // next fire to fold it. Zero on the first pass: nothing precedes it.
    let mut pending_fold: u32 = 0;
    while generated < MAX_TOKENS {
        // The buffer never empties: it carries the previous window's accepted
        // prefix into this fire, which folds it while appending its own tokens
        // past it. `advance_fold` releases the head pages the fold covers, so
        // this reservation stays bounded instead of growing once per window.
        let live = pending_fold + kp1 + rs_page;
        while rs.buffer_size() * rs_page < live {
            rs.alloc_buffer(window_slabs.max(1))
                .context("rs.alloc_buffer")?;
        }

        let (commit, drafts) = verify_window(
            &ws,
            &rs,
            &pipeline,
            k,
            seed,
            &draft,
            seq_len,
            max_pages,
            pending_fold,
        )
        .await?;
        let clen = committed_len(&commit);
        let n_acc = clen.saturating_sub(1);
        accepted_lengths.push(n_acc);
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();

        // The rejected tail never touched the recurrent state, so it is enough
        // to say it never happened. The accepted prefix stays buffered --
        // unfolded -- and the NEXT window's fire folds it while writing its own
        // tokens over the slots just released.
        let rejected = kp1 - clen as u32;
        if rejected > 0 {
            rs.discard_buffered(rejected).context("discard_buffered")?;
        }
        pending_fold = clen as u32;

        seq_len += clen as u32;
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        draft = drafts;
        seed = *committed.last().unwrap_or(&0) as i32;
    }
    // Every window's takes have drained: this close cancels nothing.
    pipeline.close();

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 {
        total_acc as f64 / steps as f64
    } else {
        0.0
    };
    let result = format!(
        "mtp-specdecode: k={k} steps={steps} fires={steps} \
         accepted_lengths={accepted_lengths:?} mean_accept={mean_acc:.2} committed={} \
         (forward-hybrid, one fire per window: fold-behind + discard-buffered; host \
         round-trip drafts — the device-resident MtpDrafts/carrier-retain path is \
         unavailable on the current inferlet::ptir surface, see the module docs)",
        committed.len()
    );
    eprintln!("{result}");
    eprintln!(
        "[mtp-specdecode] committed[{}]={committed:?}",
        committed.len()
    );
    Ok(result)
}
