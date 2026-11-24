#![cfg(target_vendor = "apple")]

//! A windowed lane whose fold lags (CONTRACT-BC 19, 21: rows past `folded`
//! held in window-buffer slots for some fires, as a run-ahead program's are)
//! answers each step what the same tokens folded every fire answer, through
//! the ring and past its wrap. Gemma 4 E4B served through its rings garbled
//! from its first decode steps on north-star: each fire refilled a buffer
//! slot whole from its ring page, over the unfolded rows the slot alone
//! held.

use engine::fire::{FoldLen, Lineage, WindowFold};
use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{argmax, close, fixture_of, load_with, word};

/// Rows a lane's fold trails its written end by (the M1's run-ahead
/// window less one: the bench's carried rows).
const LAG: u32 = 7;
const PROMPT: u32 = 40;

/// One lane's ring and buffer, by slot ref.
struct Rings {
    /// The ring's pieces, `[run, cow]` chunk refs.
    ring: Vec<[u32; 2]>,
    /// The chunk the lane's buffer slots lie in.
    buffer_chunk: u32,
    per_chunk: u32,
}

impl Rings {
    /// The buffer slot logical page `n` takes (a slot of its own for every
    /// page a buffer spans at once).
    fn slot(&self, n: u32) -> u32 {
        self.buffer_chunk * self.per_chunk + n % self.per_chunk
    }

    /// The buffer over pages `[base / page, ceil(end / page))`.
    fn buffer(&self, base: u32, end: u32, page: u32) -> Vec<u32> {
        (base / page..end.div_ceil(page)).map(|n| self.slot(n)).collect()
    }
}

#[test]
fn a_lagged_window_fold_serves_the_folded_logits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("gemma4-e4b-mini-l1", None);
    let mut shell: Shell = load_with(&fixture, 1024);
    let Some(ring) = shell.pool_ring().cloned() else {
        panic!("gemma4-e4b-mini-l1 lays rings");
    };
    let layout = shell.pool_layout();
    let window = layout.window(0).expect("one window list").clone();
    let page = window.tokens;
    let per_chunk = shell.pool_chunk_facts().shape(engine_metal::store::layout::WINDOWS[0]).expect("the window slots").slots_per_chunk;
    let ring_rows = window.rows.div_ceil(page) * page + page;
    let pieces = ring.pieces;
    eprintln!("window {} rows, {page}-row pages, ring {ring_rows} rows in {pieces} piece(s)", window.rows);
    // Past the ring's wrap, within the load's context.
    let steps = (ring_rows + 2 * page).saturating_sub(PROMPT).min(1024 - PROMPT - 1);
    let low = layout.free_chunk();
    let facts = shell.pool_chunk_facts();
    let held = 2 * (pieces + 1);
    shell
        .commit_segments((low + held).div_ceil(facts.segment_chunks))
        .expect("the segments commit");
    let lanes: Vec<Rings> = (0..2)
        .map(|lane| {
            let first = low + lane * (pieces + 1);
            Rings {
                ring: (0..pieces).map(|p| [first + p, first + p]).collect(),
                buffer_chunk: first + pieces,
                per_chunk,
            }
        })
        .collect();

    let prompt: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut fed = prompt.clone();
    let mut have = 0u32;
    // Each lane's folded row.
    let mut folded = [0u32; 2];
    let mut logits: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
    for step in 0..=steps {
        let rows = fed.len() as u32;
        let end = have + rows;
        // Lane 0 folds every row each fire; lane 1 folds the prompt, then
        // trails its end by LAG rows.
        let advance = [end - folded[0], if step == 0 { rows } else { end.saturating_sub(LAG).saturating_sub(folded[1]) }];
        let plan: Vec<(WindowFold, Vec<u32>)> = (0..2)
            .map(|lane| {
                let after = folded[lane] + advance[lane];
                // A lane carrying buffered rows bases at its folded row
                // before the fire; one carrying none, after it.
                let carries = folded[lane] < have;
                let base = if carries { folded[lane] } else { after } / page * page;
                let buffer = if carries || after < end { lanes[lane].buffer(base, end, page) } else { Vec::new() };
                (
                    WindowFold {
                        base,
                        fold: FoldLen::Host(advance[lane]),
                        lineage: Lineage::Set,
                    },
                    buffer,
                )
            })
            .collect();
        let tables: Vec<Vec<Vec<u32>>> = plan.iter().map(|(_, buffer)| vec![buffer.clone()]).collect();
        let seated: Vec<Seated<'_>> = (0..2)
            .map(|lane| Seated {
                replay_stated: None,
                held: Some(have),
                folded: folded[lane],
                refs: true,
                ring: Some(lanes[lane].ring.as_slice()),
                window: Some(plan[lane].0),
                windows: &tables[lane],
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(&fixture.trace, rows),
                    tokens: &fed,
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the fire lands");
        for lane in 0..2 {
            folded[lane] += advance[lane];
            logits[lane].push(got[lane].clone());
        }
        have = end;
        // Both lanes take lane 0's token: the same rows, step by step.
        fed = vec![argmax(&got[0])];
    }
    assert!(folded[1] + LAG >= have && folded[1] < have, "lane 1 still trails");
    close("a lagged window fold against folding every fire", &logits[1], &logits[0]);
}
