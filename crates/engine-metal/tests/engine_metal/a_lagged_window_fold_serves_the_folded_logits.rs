#![cfg(target_vendor = "apple")]

//! A windowed lane whose fold lags (CONTRACT-BC 19, 21: rows past `folded`
//! held in window-buffer slots for some fires, as a run-ahead program's are)
//! answers each step what the same tokens folded every fire answer, through
//! its ring and past the ring's wrap. Gemma 4 E4B served through its rings
//! garbled from its first decode steps on north-star: each fire refilled a
//! buffer slot whole from its ring page, over the unfolded rows the slot
//! alone held.
//!
//! Every windowed load on Metal lays rings (`chunks::MetalRing`), so the
//! same holds where windowed layers sit beside a kv list (a sliding-window
//! decoder of 4-row slots past its 2048-row window, a latent model whose
//! compressor reads its windowed rows), and where layers of two windows
//! share the widest one's ring.

use std::path::Path;

use engine::fire::{FoldLen, Lineage, WindowFold};
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST, WINDOWS};
use engine_metal::{Lane, Shell};
use poem::{Dtype, Platform};

use super::a_scattered_pool_serves_the_same_logits::{
    argmax, close, fixture_from, fixture_of, load_with, word, Fixture,
};

/// Rows a lane's fold trails its written end by (the M1's run-ahead
/// window less one: the bench's carried rows).
const LAG: u32 = 7;

#[test]
fn a_lagged_window_fold_serves_the_folded_logits() {
    lagged("gemma4-e4b-mini-l1", &fixture_of("gemma4-e4b-mini-l1", None), 1024, 40);
}

/// Windowed layers of 4-row buffer slots beside a kv list, past a 2048-row
/// window from the prompt on.
#[test]
fn a_lagged_window_fold_of_small_slots_serves_the_folded_logits() {
    lagged(
        "muse-glimmer-30b-mini-l8",
        &fixture_of("muse-glimmer-30b-mini-l8", Some(4)),
        2560,
        2200,
    );
}

/// A latent model's 128-row window beside its compressed kv list: the
/// compressor reads the windowed rows where they lie.
#[test]
fn a_lagged_window_fold_beside_a_compressed_kv_list_serves_the_folded_logits() {
    lagged("dsv4-flash-mini", &fixture_of("dsv4-flash-mini", None), 1024, 700);
}

/// Two windows (a synthetic miniature, muse-glimmer-30b-mini-l8 with every
/// other windowed layer's window halved, 1024 beside 2048, its catalog
/// copied and patched for this test alone): one ring at the widest, each
/// layer's reads bound by its own window.
#[test]
fn a_lagged_window_fold_over_two_windows_serves_the_folded_logits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let model = "muse-glimmer-30b-mini-l8";
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let copy = std::env::temp_dir().join(format!("pie-bm-pool-{}-models", std::process::id()));
    let _ = std::fs::remove_dir_all(&copy);
    std::fs::create_dir_all(&copy).expect("the catalog copy");
    for entry in std::fs::read_dir(&root).expect("the catalog") {
        let entry = entry.expect("a catalog entry");
        let into = copy.join(entry.file_name());
        if entry.file_type().expect("its type").is_dir() {
            let status = std::process::Command::new("cp")
                .arg("-R")
                .arg(entry.path())
                .arg(&into)
                .status()
                .expect("cp runs");
            assert!(status.success());
        } else {
            std::fs::copy(entry.path(), &into).expect("the file copies");
        }
    }
    let poem = copy.join("muse_glimmer.poem");
    let text = std::fs::read_to_string(&poem).expect("the package reads");
    let patched = text.replace(
        "        win = windows[reading]\n",
        "        win = windows[reading]\n        if win and l % 2 == 1:\n            win = m.window // 2\n",
    );
    assert_ne!(patched, text, "the patch applies");
    std::fs::write(&poem, patched).expect("the package writes");
    let catalog = poem_compiler::catalog::Catalog::from_dir(&copy).expect("the copy reads");
    let package = catalog.package_of(model).expect("the model is listed");
    let trace = package
        .trace(
            model,
            &poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16),
            model,
            Platform::Metal,
        )
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"));
    std::fs::remove_dir_all(&copy).expect("the copy goes");
    let fixture = fixture_from(trace, &format!("{model}-two-windows"));
    lagged(model, &fixture, 2560, 2200);
}

/// One lane's slots, by ref, each in chunks of its own: its kv pages, its
/// record, its ring and its buffer.
struct Placed {
    pages: Vec<u32>,
    record: Vec<u32>,
    ring: Vec<[u32; 2]>,
    buffer_chunk: u32,
    per_window: u32,
}

impl Placed {
    /// The buffer over pages `[base / page, ceil(end / page))`, a slot of
    /// its own for every page a buffer spans at once.
    fn buffer(&self, base: u32, end: u32, page: u32) -> Vec<u32> {
        (base / page..end.div_ceil(page))
            .map(|n| self.buffer_chunk * self.per_window + n % self.per_window)
            .collect()
    }
}

/// Two lanes on one load fed the same tokens, each on slots of its own:
/// lane 0 folds every row each fire, lane 1 folds its prompt, then trails
/// its end by [`LAG`] rows through buffer slots. Prefills `prompt` rows,
/// then decodes past the ring's wrap; every step's logits agree within two
/// bf16 units and the argmax.
fn lagged(model: &str, fixture: &Fixture, context: u32, prompt: u32) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let mut shell: Shell = load_with(fixture, context);
    let Some(ring) = shell.pool_ring().cloned() else {
        panic!("{model}: a windowed load lays rings");
    };
    let layout = shell.pool_layout();
    let facts = shell.pool_chunk_facts();
    let window = layout.window(0).expect("a ring load's window buffer").clone();
    let page = window.tokens;
    let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk.max(1));
    let per_window = per(WINDOWS[0]);
    let ring_rows = window.rows.div_ceil(page) * page + page;
    let kv_page = shell.paging().page_size.max(1);
    let kv_pages = context.div_ceil(kv_page);
    let list_chunks = facts.shape(LIST).map_or(0, |s| kv_pages.div_ceil(s.slots_per_chunk.max(1)));
    let pieces = layout.pieces;
    let lane_chunks = list_chunks + pieces + ring.pieces + 1;
    eprintln!(
        "{model}: window {} rows, {page}-row buffer slots, ring {ring_rows} rows in {} piece(s), \
         {list_chunks} kv chunk(s), {pieces} record piece(s)",
        window.rows, ring.pieces
    );
    let low = layout.free_chunk();
    shell
        .commit_segments((low + 2 * lane_chunks).div_ceil(facts.segment_chunks))
        .expect("the segments commit");
    let lanes: Vec<Placed> = (0..2)
        .map(|lane| {
            let mut at = low + lane * lane_chunks;
            let list = facts.shape(LIST).map(|s| s.slots_per_chunk.max(1));
            let pages = list.map_or_else(Vec::new, |per| (0..kv_pages).map(|p| at * per + p).collect());
            at += list_chunks;
            let record = (0..pieces).map(|p| (at + p) * per(FIXED)).collect();
            at += pieces;
            let rings = (0..ring.pieces).map(|p| [at + p, at + p]).collect();
            at += ring.pieces;
            Placed {
                pages,
                record,
                ring: rings,
                buffer_chunk: at,
                per_window,
            }
        })
        .collect();

    // Past the ring's wrap, within the load's context.
    let past = ring_rows + 2 * page;
    let steps = past.saturating_sub(prompt).max(24).min(context - prompt - 1);
    // The prompt in pieces a fire of both lanes takes, each lane folding
    // every piece; then the decodes, lane 1 trailing.
    let piece = (shell.budgets().tokens.max_tokens / 2).max(1);
    let prompt_tokens: Vec<u32> = (0..prompt).map(|at| (at * 37 + 11) % 256).collect();
    let mut pieces = prompt_tokens.chunks(piece as usize).map(<[u32]>::to_vec).collect::<Vec<_>>().into_iter();
    let mut fed: Vec<u32> = pieces.next().expect("a prompt");
    let mut have = 0u32;
    let mut folded = [0u32; 2];
    let mut logits: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
    let mut decodes = 0u32;
    loop {
        let rows = fed.len() as u32;
        let end = have + rows;
        let prefilling = end <= prompt;
        let advance = [
            end - folded[0],
            if prefilling { end - folded[1] } else { end.saturating_sub(LAG).saturating_sub(folded[1]) },
        ];
        let plan: Vec<(WindowFold, Vec<Vec<u32>>)> = (0..2)
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
                    vec![buffer],
                )
            })
            .collect();
        let seated: Vec<Seated<'_>> = (0..2)
            .map(|lane| Seated {
                replay_stated: None,
                held: Some(have),
                folded: folded[lane],
                refs: true,
                pages: &lanes[lane].pages,
                record: (!lanes[lane].record.is_empty()).then_some(lanes[lane].record.as_slice()),
                ring: Some(lanes[lane].ring.as_slice()),
                window: Some(plan[lane].0),
                windows: &plan[lane].1,
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
        if let Some(next) = pieces.next() {
            fed = next;
            continue;
        }
        if decodes == steps {
            break;
        }
        decodes += 1;
        // Both lanes take lane 0's token: the same rows, step by step.
        fed = vec![argmax(&got[0])];
    }
    assert!(folded[1] < have && folded[1] + LAG >= have, "{model}: lane 1 still trails");
    assert!(have > ring_rows, "{model}: past the ring's wrap");
    close(
        &format!("{model}: a lagged window fold against folding every fire"),
        &logits[1],
        &logits[0],
    );
}
