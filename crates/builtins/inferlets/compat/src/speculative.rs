use inferlet::eta::hybrid::prelude::*;
use std::ops::ControlFlow;

#[allow(clippy::too_many_arguments)]
async fn pass(
    pipe: &Pipeline,
    ws: &WorkingSet,
    rs: &[RsWorkingSet],
    pool: &[u32],
    tokens: &[i32],
    base: u32,
    fold: Option<u32>,
    buffer_pages: u32,
    drafting: bool,
    last_only: bool,
) -> Result<Vec<i32>> {
    let count = tokens.len() as u32;
    let page = kv_page_size();
    let end = base + count;
    let toks = Channel::from(tokens).named("spec_tokens");
    let indptr = Channel::from([0u32, count]).named("spec_indptr");
    let positions = Channel::from_iter(base..end).named("spec_positions");
    let pages = Channel::from(pool.to_vec()).named("spec_pages");
    let pidx = Channel::from([0u32, end.div_ceil(page)]).named("spec_page_indptr");
    let w_slot =
        Channel::from_iter((base..end).map(|p| pool[(p / page) as usize])).named("spec_w_slot");
    let w_off = Channel::from_iter((base..end).map(|p| p % page)).named("spec_w_off");
    let kv_len = Channel::from([end]).named("spec_kv_len");
    let readouts = if last_only {
        vec![count - 1]
    } else {
        (0..count).collect()
    };
    let shown = readouts.len() as u32;
    let readout = Channel::from(readouts).named("spec_readout");
    let fold = fold.map(|n| Channel::from([n]).named("spec_fold"));
    let out = Channel::new([shown], dtype::i32).named("spec_output");
    let fwd = ForwardPass::new();
    if drafting {
        fwd.set_drafting_block(true).context("draft block")?;
    }
    fwd.embed(&toks, &indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &pidx,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        rs,
        RsGeometry {
            fold_len: fold.as_ref(),
            buffer: 0..buffer_pages,
        },
    )?;
    fwd.epilogue(move || {
        let next = if drafting {
            intrinsics::mtp_drafts(shown)
        } else {
            reduce_argmax(intrinsics::logits())
        };
        out.put(reshape(next, [shown]));
    });
    fwd.submit(pipe).context("speculative submit")?;
    out.take_host::<Vec<i32>>()
        .await
        .context("speculative readback")
}

pub async fn generate(
    prompt: &[u32],
    max_tokens: usize,
    draft: model::BlockDrafter,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Result<usize> {
    let block = draft.rows;
    let rs: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        _ => return Err("block speculation requires attention or hybrid state".into()),
    };
    let page = kv_page_size();
    let ws = WorkingSet::new();
    let prompt_len = u32::try_from(prompt.len()).map_err(|_| "prompt length overflow")?;
    let limit = u32::try_from(max_tokens).map_err(|_| "generation length overflow")?;
    let extent = prompt_len
        .checked_add(limit)
        .and_then(|n| block.checked_mul(2).and_then(|extra| n.checked_add(extra)))
        .ok_or("speculative extent overflow")?;
    let reservation = ws
        .reserve(extent.div_ceil(page))
        .context("speculative KV reservation")?;
    let pool = reservation.ids().to_vec();
    let pipe = Pipeline::new();
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut anchor = 0;
    for &(base, end) in &prefill_chunks(prompt.len() as u32, None) {
        anchor = pass(
            &pipe,
            &ws,
            &rs,
            &pool,
            &prompt[base as usize..end as usize],
            base,
            None,
            0,
            false,
            true,
        )
        .await?[0];
    }
    let mut generated = 1;
    if on_token(anchor as u32).is_break() || generated == max_tokens {
        pipe.close();
        return Ok(generated);
    }
    let mut held = prompt.len() as u32;
    let rs_page = model::rs_buffer_page_size().max(1);
    let mut survivors = 0;
    while generated < max_tokens {
        let buffer_pages = (rs_page - 1 + survivors + block).div_ceil(rs_page).max(1) + 1;
        if let Some(state) = rs.first() {
            let have = state.buffer_size();
            if have < buffer_pages {
                state
                    .alloc_buffer(buffer_pages - have)
                    .context("draft RS buffer")?;
            }
        }
        let mut input = vec![draft.mask_token as i32; block as usize];
        input[0] = anchor;
        let picks = pass(
            &pipe,
            &ws,
            &rs,
            &pool,
            &input,
            held,
            Some(0),
            buffer_pages,
            true,
            false,
        )
        .await?;
        if let Some(state) = rs.first() {
            state
                .discard_buffered(block)
                .context("discard draft state")?;
        }
        let proposals = &picks[1..block as usize];
        let mut verify = vec![anchor];
        verify.extend_from_slice(proposals);
        let truth = pass(
            &pipe,
            &ws,
            &rs,
            &pool,
            &verify,
            held,
            Some(survivors),
            buffer_pages,
            false,
            false,
        )
        .await?;
        let kept = proposals
            .iter()
            .zip(&truth)
            .take_while(|(p, t)| p == t)
            .count();
        let rejected = block - 1 - kept as u32;
        if rejected > 0
            && let Some(state) = rs.first()
        {
            state
                .discard_buffered(rejected)
                .context("discard rejected state")?;
        }
        survivors = kept as u32 + 1;
        anchor = truth[kept];
        held += survivors;
        for &token in proposals[..kept].iter().chain(std::iter::once(&anchor)) {
            generated += 1;
            if on_token(token as u32).is_break() || generated == max_tokens {
                pipe.close();
                return Ok(generated);
            }
        }
    }
    pipe.close();
    Ok(generated)
}
