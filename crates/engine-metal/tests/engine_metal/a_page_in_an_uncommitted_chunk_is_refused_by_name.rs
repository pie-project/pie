#![cfg(target_vendor = "apple")]

//! Every page a fire reads is checked backed, ref by ref. Under two-ended
//! growth the pool backs a bottom run of segments and the span's segment at
//! the top for state, not the segments between: a lane whose highest page
//! lies in the backed top and whose first lies in an unbacked chunk under
//! it is refused by name before anything is encoded. (The check by the
//! highest page alone passed it, and the kernels read an unmapped tile.)
//! The same lane with its first page moved into the bottom run fires.

use engine_metal::serve::Seated;
use engine_metal::store::layout::LIST;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, word};

const CONTEXT: u32 = 256;

/// The pool at a stated size, not the device's free memory: a pool sized
/// to fill what the device has free counts as many chunks as that holds,
/// and where its chunks less the activation's fell on a segment boundary
/// (one load in some, after other tests held memory) the span's segment
/// held no state under the activation and the test could not lay its page
/// there.
const KV_BYTES: u64 = 1 << 30;

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: 4,
        kv_budget: Some(KV_BYTES),
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

fn fire(shell: &mut Shell, fixture: &Fixture, pages: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    let page = shell.paging().page_size;
    let tokens: Vec<u32> = (0..page + 1).map(|at| (at * 37 + 11) % 256).collect();
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            refs: true,
            pages,
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens: &tokens,
            })
        }])
        .map_err(|why| why.to_string())
}

#[test]
fn a_page_in_an_uncommitted_chunk_is_refused_by_name() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load(&fixture);
    let facts = shell.pool_chunk_facts();
    let layout = shell.pool_layout();
    let segment = facts.segment_chunks;
    let per = facts.shape(LIST).expect("a kv list").slots_per_chunk;
    // The bottom run: the seats' segments and one more.
    let bottom = layout.free_chunk().div_ceil(segment) + 1;
    shell
        .commit_segments(bottom)
        .expect("the bottom run commits");
    // The span's segment, backed for state under the activation.
    let total = facts.total_chunks;
    let act = layout.act_span();
    let top_first = (total - act) / segment * segment;
    assert!(
        top_first < total - act,
        "the span's segment holds state under the activation ({total} chunks, the \
         activation's {act}, a segment {segment}): choose another KV_BYTES"
    );
    shell
        .commit_span(1)
        .expect("the span's segment backs state");
    let top = top_first * per;
    let unbacked = (bottom * segment + 1) * per;
    assert!(unbacked < top, "an unbacked chunk lies between");
    let why = fire(&mut shell, &fixture, &[unbacked, top])
        .expect_err("a page in an unbacked chunk is refused");
    assert!(
        why.contains("committed chunks of the state pool"),
        "the refusal names the unbacked chunk: {why}"
    );
    eprintln!("refused: {why}");
    let backed = (bottom * segment - 1) * per;
    fire(&mut shell, &fixture, &[backed, top]).expect("every page backed: the fire lands");
}
