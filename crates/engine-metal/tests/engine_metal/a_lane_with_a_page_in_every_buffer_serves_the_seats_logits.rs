#![cfg(target_vendor = "apple")]

//! A pool spans as many buffers as its range takes, none fixed: under a
//! one-segment buffer cap and a kv budget of a few hundred MiB, more than
//! four. A lane whose kv pages put one page in each of them (a page id's
//! buffer in as many bits as count the buffers, reached through the delta
//! table) serves the logits of the same lane on the seats' runs, bit for bit.

use engine_metal::serve::Seated;
use engine_metal::store::layout::LIST;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_of, same, word};

const CONTEXT: u32 = 2048;
const SEATS: u32 = 4;
const STEPS: u32 = 3;
/// The most prompt rows one fire carries here.
const PROMPT_FIRE: usize = 192;

fn load(fixture: &Fixture, kv_budget: Option<u64>) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(SEATS, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: SEATS,
        kv_budget,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

/// A prompt of `rows` tokens, then `STEPS` greedy decodes: every logit row.
/// `pages` are the lane's kv slot refs, or none for seat 0's run.
fn serve(shell: &mut Shell, fixture: &Fixture, rows: u32, pages: Option<&[u32]>) -> Vec<Vec<f32>> {
    let prompt: Vec<u32> = (0..rows).map(|at| (at * 37 + 11) % 256).collect();
    let mut out = Vec::new();
    let mut held = 0u32;
    let fire = |shell: &mut Shell, tokens: &[u32], held: u32| -> Vec<Vec<f32>> {
        shell
            .fire_seated(&[Seated {
                held: Some(held),
                refs: pages.is_some(),
                pages: pages.unwrap_or(&[]),
                ..Seated::of(Lane {
                    slot: 0,
                    word: word(&fixture.trace, tokens.len() as u32),
                    tokens,
                })
            }])
            .expect("the fire lands")
    };
    // The prompt in fires of at most `PROMPT_FIRE` rows, then greedy steps.
    for part in prompt.chunks(PROMPT_FIRE) {
        out.extend(fire(shell, part, held));
        held += part.len() as u32;
    }
    for _ in 0..STEPS {
        let fed = [argmax(out.last().expect("a row"))];
        out.extend(fire(shell, &fed, held));
        held += 1;
    }
    out
}

#[test]
fn a_lane_with_a_page_in_every_buffer_serves_the_seats_logits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);

    // Buffers of two segments each: the last holds a segment of state
    // under the span's.
    let probe = load(&fixture, Some(1280 << 20)).pool_chunk_facts();
    let part_bytes = 2 * u64::from(probe.segment_chunks) * probe.chunk_bytes;
    engine_metal::store::pool::cap_parts(part_bytes);
    let mut shell = load(&fixture, Some(1280 << 20));
    let (ids, parts) = shell.page_ids();
    let facts = shell.pool_chunk_facts();
    let layout = shell.pool_layout();
    let segment = 2 * facts.segment_chunks;
    let list = layout.list.expect("a kv list");
    let page = shell.paging().page_size.max(1);
    // A page in each buffer's lower segment, past the seats' runs.
    let chunks: Vec<u32> = (0..parts as u32)
        .map(|b| {
            let first = b * segment;
            let from = first.max(layout.free_chunk());
            from + (first + segment / 2 - from) / 2
        })
        .collect();
    assert!(
        parts > 4,
        "the range spans more than four buffers ({parts}); a page id's buffer in {} bits",
        31 - ids.part_shift
    );
    shell
        .commit_segments(2 * parts as u32 - 1)
        .expect("every segment under the span's commits");
    let per = facts.shape(LIST).expect("a kv list").slots_per_chunk;
    let pages: Vec<u32> = chunks.iter().map(|&chunk| chunk * per + per / 2).collect();
    // The last buffer's page: its id names that buffer over the part bits
    // and its slot in the buffer under them, and decodes on the host to
    // where the slot lies there.
    let last = parts - 1;
    let id = shell.kv_page_ids(&pages[last..]).expect("the id")[0];
    let local_chunk = chunks[last] - last as u32 * segment;
    assert_eq!(
        id,
        ((last as u32) << ids.part_shift) | (local_chunk * per + per / 2),
        "the last buffer's page id"
    );
    assert_eq!(
        ids.place(id, per, list.slot_bytes),
        (
            last,
            u64::from(local_chunk) * layout.chunk_bytes + u64::from(per / 2) * list.slot_bytes
        ),
        "the last buffer's page id decodes on the host to its slot"
    );
    // The prompt reaches the last page; the decodes stay in it.
    let rows = (parts as u32 - 1) * page + 1;
    assert!(rows + STEPS <= parts as u32 * page && rows + STEPS <= CONTEXT);
    eprintln!(
        "qwen3-micro-text: {parts} buffers of {segment} chunks, a page id's buffer in {} bits; \
         one page of {page} tokens in each, chunks {chunks:?}",
        31 - ids.part_shift
    );
    let got = serve(&mut shell, &fixture, rows, Some(&pages));
    drop(shell);
    engine_metal::store::pool::cap_parts(0);

    let mut seats = load(&fixture, None);
    let want = serve(&mut seats, &fixture, rows, None);
    same(
        &format!("a page in each of {parts} buffers against the seats' run"),
        &got,
        &want,
    );
}
