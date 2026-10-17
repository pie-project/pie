use inferlet::eta::prelude::*;
use std::ops::ControlFlow;

enum Round<'a> {
    Prompt(&'a [u32]),
    Draft,
    Verify(u32),
}

async fn pass(
    pipe: &Pipeline,
    ws: &WorkingSet,
    tokens: &[i32],
    base: u32,
    round: Round<'_>,
) -> Result<Vec<i32>> {
    let count = tokens.len() as u32;
    let end = base + count;
    let drafting = matches!(round, Round::Draft);
    let toks = Channel::from(tokens).named("spec_tokens");
    let indptr = Channel::from([0u32, count]).named("spec_indptr");
    let slots = Channel::from_iter(base..end).named("spec_slots");
    let readouts = match round {
        Round::Prompt(_) => vec![count - 1],
        Round::Draft | Round::Verify(_) => (0..count).collect(),
    };
    let shown = readouts.len() as u32;
    let readout = Channel::from(readouts).named("spec_readout");
    let fold = match round {
        Round::Prompt(_) => None,
        Round::Draft => Some(0),
        Round::Verify(survivors) => Some(survivors),
    };
    let fold = fold.map(|n| Channel::from([n]).named("spec_fold"));
    let fold = match &fold {
        Some(count) => Fold::Count(count),
        None => Fold::All,
    };
    let out = Channel::new([shown], dtype::i32).named("spec_output");
    let fwd = ForwardPass::new();
    if drafting {
        fwd.set_drafting_block(true).context("draft block")?;
    }
    fwd.embed(&toks, &indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold,
            ..StateGeometry::default()
        },
    )?;
    if let Round::Prompt(ends) = round {
        fwd.checkpoints(ends)?;
    }
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
    ends: &[u32],
    max_tokens: usize,
    draft: model::BlockDrafter,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Result<(usize, usize)> {
    let block = draft.rows;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut anchor = pass(&pipe, &ws, &prompt, 0, Round::Prompt(ends)).await?[0];
    let cached = ws.cached_tokens() as usize;
    let mut generated = 1;
    if on_token(anchor as u32).is_break() || generated == max_tokens {
        pipe.close();
        return Ok((generated, cached));
    }
    let mut held = prompt.len() as u32;
    let mut survivors = 0;
    while generated < max_tokens {
        // The draft and its verify write the same slots: the verify replaces
        // the draft, and the next round's slots replace what it rejected.
        let mut input = vec![draft.mask_token as i32; block as usize];
        input[0] = anchor;
        let picks = pass(&pipe, &ws, &input, held, Round::Draft).await?;
        let proposals = &picks[1..block as usize];
        let mut verify = vec![anchor];
        verify.extend_from_slice(proposals);
        let truth = pass(&pipe, &ws, &verify, held, Round::Verify(survivors)).await?;
        let kept = proposals
            .iter()
            .zip(&truth)
            .take_while(|(p, t)| p == t)
            .count();
        survivors = kept as u32 + 1;
        anchor = truth[kept];
        held += survivors;
        for &token in proposals[..kept].iter().chain(std::iter::once(&anchor)) {
            generated += 1;
            if on_token(token as u32).is_break() || generated == max_tokens {
                pipe.close();
                return Ok((generated, cached));
            }
        }
    }
    pipe.close();
    Ok((generated, cached))
}
