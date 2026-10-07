//! Sliding-window kv rows kept in the windowed pool (a window per sequence,
//! page 0 the null page) answer what the same rows kept in full pages do,
//! over a prompt longer than the window. One lane is seated by slot (the
//! pool's ring), one is handed its pages and windowed ids as the runtime
//! hands them: ids behind the window released to 0 and reused.
//! Asked for with `PIE_XLA_ARTIFACT` (or `PIE_XLA_SNAPSHOT` + `PIE_XLA_SKU`)
//! naming a model with windowed kv rows (Gemma4).

mod common;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem_compiler::Budget;
use poem_dsl::{Platform, Request};

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const CHUNK: usize = 128;
const DECODE: usize = 24;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The runtime's side of a lane handed windowed ids: a free list over the
/// pool's ids past `first_id`, ids behind the window returned to it.
struct Handed {
    pages: Vec<u32>,
    window: Vec<u32>,
    free: Vec<u32>,
    reused: usize,
}

impl Handed {
    fn new(base: u32, first_id: u32, ids: u32) -> Handed {
        Handed {
            pages: (0..CONTEXT / PAGE).map(|p| base + p).collect(),
            window: Vec::new(),
            free: (first_id..first_id + ids).rev().collect(),
            reused: 0,
        }
    }

    /// The tables for a fire landing `rows` rows after `have`.
    fn seat(&mut self, have: u32, rows: u32, window_tokens: u32) {
        let needed = (have + rows).div_ceil(PAGE).max(1) as usize;
        let first = (have.saturating_sub(window_tokens) / PAGE) as usize;
        self.window.resize(needed, 0);
        for page in 0..needed {
            if page < first {
                if self.window[page] != 0 {
                    self.free.insert(0, self.window[page]);
                    self.window[page] = 0;
                    self.reused += 1;
                }
            } else if self.window[page] == 0 {
                self.window[page] = self.free.pop().expect("a windowed id is free");
            }
        }
    }
}

struct Answer {
    ring: Vec<u32>,
    handed: Vec<u32>,
    pool_bytes: u64,
    weight_bytes: u64,
    window_pages: u64,
    reused: usize,
}

fn answer(m: &common::Model, prompt: &[u32], full_windows: bool) -> Answer {
    // SAFETY: the test sets the knob before the shell reads it, on one thread.
    unsafe {
        if full_windows {
            std::env::set_var("PIE_XLA_FULL_WINDOWS", "1");
        } else {
            std::env::remove_var("PIE_XLA_FULL_WINDOWS");
        }
    }
    let sku = m.sku;
    let facts = sku.trace(models::Platform::Xla).facts;
    let word = |rows: u32| facts.word(&Request::new(rows, false));
    let mut shell = Shell::load(Boot {
        trace: sku.trace(Platform::Xla),
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(4, 2 * CHUNK as u32),
        page_size: PAGE,
        context: CONTEXT,
        slots: 2,
        pages: 2 * CONTEXT / PAGE,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    let paging = shell.paging();
    let window_tokens = paging.window.map_or(u32::MAX, |w| w.tokens);
    let ring = paging.window_ring();
    eprintln!(
        "full_windows={full_windows}: {} kv pages, window {:?}, {} windowed pages, ring {ring}",
        paging.pages(),
        paging.window,
        paging.window_pages()
    );
    // Slot 0 cycles through the ring's ids `1..=ring`; the handed lane takes
    // the ids past it. Its full pages are slot 1's.
    let mut handed = Handed::new(
        paging.base(1) as u32,
        1 + ring,
        (paging.window_pages() as u32).saturating_sub(1 + ring),
    );
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");

    let mut have = 0u32;
    let mut last: Vec<Vec<f32>> = Vec::new();
    let mut ring_out = Vec::new();
    let mut handed_out = Vec::new();
    let feed = |shell: &mut Shell,
                have: u32,
                ring_rows: &[u32],
                handed_rows: &[u32],
                handed: &mut Handed|
     -> Vec<Vec<f32>> {
        let rows = ring_rows.len() as u32;
        if paging.window.is_some() {
            handed.seat(have, rows, window_tokens);
        }
        let needed = (have + rows).div_ceil(PAGE) as usize;
        let mut ring_lane = Seated::of(Lane {
            slot: 0,
            word: word(rows),
            tokens: ring_rows,
        });
        ring_lane.held = None;
        let mut handed_lane = Seated::of(Lane {
            slot: 1,
            word: word(rows),
            tokens: handed_rows,
        });
        handed_lane.pages = &handed.pages[..needed];
        handed_lane.held = Some(have);
        handed_lane.window = &handed.window;
        shell
            .fire_seated(&[ring_lane, handed_lane])
            .expect("the fire runs")
    };
    for chunk in prompt.chunks(CHUNK) {
        last = feed(&mut shell, have, chunk, chunk, &mut handed);
        have += chunk.len() as u32;
    }
    for _ in 0..DECODE {
        let a = argmax(&last[0]);
        let b = argmax(&last[1]);
        ring_out.push(a);
        handed_out.push(b);
        last = feed(&mut shell, have, &[a], &[b], &mut handed);
        have += 1;
    }
    assert_eq!(shell.held(0), have, "the ring lane's slot holds every row");
    let (weight_bytes, pool_bytes) = shell.footprint();
    Answer {
        ring: ring_out,
        handed: handed_out,
        pool_bytes,
        weight_bytes,
        window_pages: paging.window_pages(),
        reused: handed.reused,
    }
}

#[test]
fn a_windowed_pool_answers_what_full_pages_do() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_ARTIFACT (a model with windowed kv rows)");
        return;
    };
    let trace = m.sku.trace(Platform::Xla);
    let window = trace
        .caches
        .iter()
        .filter_map(|row| match row {
            poem_ir::CacheRow::Kv { window, .. } => *window,
            poem_ir::CacheRow::State { .. } => None,
        })
        .max();
    let Some(window) = window else {
        eprintln!(
            "{} declares no windowed kv rows; nothing to compare",
            m.sku.name
        );
        return;
    };
    let tokenizer = common::tokenizer(&m);
    let text = "The history of the lighthouse keeper's island began with a storm. \
                Every evening the keeper climbed the spiral stairs, trimmed the wick, \
                wrote the weather into the log, and counted the ships that passed. ";
    let mut prompt = Vec::new();
    while prompt.len() < window as usize + 200 {
        prompt.extend(tokenizer.encode(text));
    }
    prompt.extend(tokenizer.encode("Question: what did the keeper count each evening? Answer:"));
    eprintln!(
        "prompt of {} tokens against a window of {window}",
        prompt.len()
    );

    let _device = engine_xla::bench::lock_device();
    let windowed = answer(&m, &prompt, false);
    let full = answer(&m, &prompt, true);
    eprintln!(
        "windowed: ring {:?}\n          handed {:?}\n  => {:?}",
        windowed.ring,
        windowed.handed,
        tokenizer.decode(&windowed.ring, true)
    );
    eprintln!(
        "full:     ring {:?}\n          handed {:?}",
        full.ring, full.handed
    );
    eprintln!(
        "pools: windowed {} MiB ({} windowed pages, {} ids reused), full {} MiB; weights {} MiB",
        windowed.pool_bytes >> 20,
        windowed.window_pages,
        windowed.reused,
        full.pool_bytes >> 20,
        windowed.weight_bytes >> 20
    );
    assert!(
        windowed.reused > 0,
        "the handed lane released ids behind its window"
    );
    assert!(
        windowed.pool_bytes < full.pool_bytes,
        "the windowed pool is smaller"
    );
    assert_eq!(
        windowed.ring, full.ring,
        "the slot-seated lane decodes the same tokens"
    );
    assert_eq!(
        windowed.handed, full.handed,
        "the handed lane decodes the same tokens"
    );
}
