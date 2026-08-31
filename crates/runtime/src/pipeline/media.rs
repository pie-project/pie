//! **The run scan: tokens are the ledger, and this is the audit**
//! (`.wiki/alto/media-door.md` §3).
//!
//! A media span enters a sequence as the token run `image.tokens()` answered —
//! prefix + the model's reserved pad repeated `token_count` times + suffix —
//! and as nothing else. No anchor list crosses the boundary, no offset, no
//! length; the guest states the correspondence by *ordering* alone, attaching
//! its spans through `forward-pass.media` in the order their runs appear.
//!
//! **SO THE HOST FINDS THE RUNS RATHER THAN BEING TOLD WHERE THEY ARE.** The
//! pad is a reserved special: a tokenizer never emits it from text, so a
//! maximal run of it in a submitted token list came from `tokens()` and from
//! nowhere else. Scanning for it is what turns "the guest promised" into "the
//! host checked", and it is why the door needed no second bookkeeping
//! structure to be safe.
//!
//! **EVERY DISAGREEMENT IS A REFUSAL BY NAME, BEFORE ANYTHING LAUNCHES.** The
//! alternative is not a slower failure, it is a WRONG ANSWER: a run one row
//! short of its span still fires, still scatters, and still decodes into
//! fluent text about an image the model half-saw. There is no test downstream
//! of here that catches that, so the check is here and it is total.

use ::model::media::EncodedSpan;
use std::sync::Arc;

/// **A NAMED REFUSAL, WITH THE DOOR'S OWN SENTENCE.**
///
/// Each variant is one of media-door §3's bullets. They are an enum rather
/// than four `format!`s at the call site so a test can name the disagreement
/// it provoked instead of matching on prose that will be reworded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The submission's runs and its attached spans do not agree in number.
    RunCount { runs: usize, spans: usize },
    /// Run `index` is not as long as the span attached at that position. This
    /// is the one that catches a guest slicing a run in half, or splicing two
    /// spans' runs into one.
    RunLength {
        index: usize,
        run_rows: u32,
        span_rows: u32,
    },
    /// A span was attached but the tokens carry no media run at all.
    OrphanSpans { spans: usize },
    /// The tokens carry media runs but no span was attached.
    OrphanRuns { runs: usize },
}

// **`EngineDoor` IS GONE, AND THAT IS THE WAVE** (media-door §6, MD-C). A
// fifth variant stood here: the submission's spans matched, and the seam that
// carried them to the engine was not cut. It refused rather than dropped,
// because a pass that quietly forgot its images would answer fluently about
// none of them. The seam is cut — `engine::fire::StepMedia` — so the refusal
// has nothing left to refuse, and a stop kept past the thing it was stopping
// for is a stop nobody can reach.

impl Refusal {
    /// The refusal's own name.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the point of a named refusal is that a test can name the \
                  disagreement it provoked instead of matching on prose that \
                  will be reworded; every variant is asserted by name in this \
                  module's tests, and MD-C's wiring reads it at the seam"
    )]
    pub const fn name(&self) -> &'static str {
        match self {
            Refusal::RunCount { .. } => "MediaRunCount",
            Refusal::RunLength { .. } => "MediaRunLength",
            Refusal::OrphanSpans { .. } => "MediaOrphanSpans",
            Refusal::OrphanRuns { .. } => "MediaOrphanRuns",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::RunCount { runs, spans } => write!(
                f,
                "MediaRunCount: the tokens carry {runs} media runs and the pass \
                 attached {spans} spans"
            ),
            Refusal::RunLength {
                index,
                run_rows,
                span_rows,
            } => write!(
                f,
                "MediaRunLength: run {index} is {run_rows} rows and span {index} \
                 occupies {span_rows}"
            ),
            Refusal::OrphanSpans { spans } => write!(
                f,
                "MediaOrphanSpans: the pass attached {spans} spans and the tokens \
                 carry no media run — a span enters the sequence as the run \
                 `tokens()` answers, and nothing else puts it there"
            ),
            Refusal::OrphanRuns { runs } => write!(
                f,
                "MediaOrphanRuns: the tokens carry {runs} media runs and the pass \
                 attached no spans — the run is the ledger entry and the handle \
                 is the payload; neither stands alone"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

/// **ONE RUN, MATCHED TO ITS SPAN** — and the anchor the contract wants.
#[derive(Clone, Debug)]
pub struct MatchedRun {
    /// Which lane of the submission this run fell in.
    pub lane: u32,
    /// Where the run starts, as an offset into THAT LANE's token rows.
    ///
    /// **LANE-RELATIVE, AND DELIBERATELY SO.** The submission cannot know the
    /// seriated fire it will land in — the batcher decides that after this has
    /// been written — so an anchor says "my seventh token row" and the shell
    /// rebases it by the lane's own row offset. It is `serve::Media::routes`'
    /// convention, kept here so MD-C's marshaling is a copy and not an
    /// arithmetic.
    pub anchor: u32,
    /// How many token rows the run occupies. Equal to the span's
    /// `token_count`, which is exactly what [`scan`] refuses otherwise.
    pub rows: u32,
    /// The span whose payload this run stands for.
    pub span: Arc<EncodedSpan>,
}

/// **THE CONTRACT'S OWN MEDIA ROW IS WHAT THIS BUILDS** (media-door §6).
///
/// MD-A's hand-off named a `LaneMedia` here — `engine_cuda::serve::Media` field
/// for field, owned rather than borrowed — and said MD-C would restate it on
/// the contract. It is restated ([`engine::fire::StepMedia`]), so the local
/// copy is gone rather than converted: two structs with the same eight fields
/// and a `From` between them is two spellings of one record, and article 8
/// forbids the second. What survives is the DERIVATION, which is this module's
/// and only this module's — an anchor is where a run landed, a route is which
/// token row a payload row scatters to, and neither exists until the scan has
/// matched.
pub use engine::fire::StepMedia as LaneMedia;

/// The sentinel a fold-space tail row routes to. `serve::Media`'s own
/// `PATCH_ROUTE_DROP`, restated here because the runtime may not name an
/// engine crate's private constant.
const PATCH_ROUTE_DROP: i32 = -1;

/// **FIND THIS LANE'S MAXIMAL PLACEHOLDER RUNS.**
///
/// Maximal, so a span's run is one run and not `token_count` of them, and so a
/// guest that spliced two spans' runs adjacently produces ONE over-long run —
/// which [`scan`] then refuses on length rather than silently accepting as a
/// pair. That is the whole reason the scan is on maximal runs: it makes the
/// half-sliced and the double-spliced cases the same refusal.
fn runs_of(tokens: &[u32], pad: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != pad {
            i += 1;
            continue;
        }
        let start = i;
        while i < tokens.len() && tokens[i] == pad {
            i += 1;
        }
        out.push((start as u32, (i - start) as u32));
    }
    out
}

/// **THE SCAN** (media-door §3): the submission's runs, matched to its spans in
/// order, or the first disagreement, by name.
///
/// `lanes` is the submission's token lists in lane order — the ledger. `spans`
/// is what the pass attached. Both are read once, here, at the one instant
/// both are final.
///
/// **THE PAD COMES FROM THE SPANS AND NOT FROM THE MODEL.** A span was spelled
/// at `from_bytes` by the bound checkpoint's own tokenizer, so it already
/// carries the id its run is written with — asking the model a second time
/// would be a second reading of the same fact, and the two could disagree. A
/// submission with no spans has no pad to look for and therefore no runs,
/// which is exactly the right answer: a text-only pass scans nothing.
///
/// # Errors
///
/// [`Refusal`], the first disagreement found.
pub fn scan(lanes: &[&[u32]], spans: &[Arc<EncodedSpan>]) -> Result<Vec<MatchedRun>, Refusal> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let mut found: Vec<MatchedRun> = Vec::new();
    // Every span of one submission is spelled by one tokenizer, but two
    // MODALITIES spell their runs with two different pads, so the scan looks
    // for each pad the attached spans name.
    let mut pads: Vec<u32> = spans.iter().map(|s| s.placeholder).collect();
    pads.sort_unstable();
    pads.dedup();
    for (lane, tokens) in lanes.iter().enumerate() {
        let mut here: Vec<(u32, u32)> = Vec::new();
        for &pad in &pads {
            here.extend(runs_of(tokens, pad));
        }
        // Positional order within the lane, whichever pad found them.
        here.sort_unstable();
        for (anchor, rows) in here {
            found.push(MatchedRun {
                lane: lane as u32,
                anchor,
                rows,
                // Provisional; replaced by the ordered match below.
                span: Arc::clone(&spans[0]),
            });
        }
    }

    if found.is_empty() {
        return Err(Refusal::OrphanSpans {
            spans: spans.len(),
        });
    }
    if found.len() != spans.len() {
        return Err(Refusal::RunCount {
            runs: found.len(),
            spans: spans.len(),
        });
    }
    for (index, (run, span)) in found.iter_mut().zip(spans).enumerate() {
        if run.rows != span.token_count {
            return Err(Refusal::RunLength {
                index,
                run_rows: run.rows,
                span_rows: span.token_count,
            });
        }
        run.span = Arc::clone(span);
    }
    Ok(found)
}

/// **THE OTHER DIRECTION**: runs with no spans behind them.
///
/// Called only when the pass attached nothing, because that is the only case
/// [`scan`] cannot see — with no spans it has no pad to look for. The model's
/// own placeholder is the id to scan for here, and it is the caller's to
/// supply because a pass with no spans has none to read it off.
///
/// # Errors
///
/// [`Refusal::OrphanRuns`] when the tokens carry runs of `pad` and no span was
/// attached.
pub fn refuse_orphan_runs(lanes: &[&[u32]], pad: Option<u32>) -> Result<(), Refusal> {
    let Some(pad) = pad else { return Ok(()) };
    let runs: usize = lanes.iter().map(|tokens| runs_of(tokens, pad).len()).sum();
    if runs == 0 {
        Ok(())
    } else {
        Err(Refusal::OrphanRuns { runs })
    }
}

/// **DERIVE WHAT THE CONTRACT WANTS, PER LANE** — media-door §3's "from the
/// matched runs the RUNTIME derives everything".
///
/// The guest sees none of this and the front-end computed none of it: an
/// anchor is where a run landed, a route is which token row a payload row
/// scatters to, and both are facts about this submission that only exist once
/// the scan has matched.
#[must_use]
pub fn lane_media(matched: &[MatchedRun], lane_rows: &[u32]) -> Vec<LaneMedia> {
    let mut out: Vec<LaneMedia> = Vec::new();
    for run in matched {
        let slot = match out.iter().position(|m| m.lane == run.lane) {
            Some(i) => i,
            None => {
                out.push(LaneMedia {
                    lane: run.lane,
                    ..LaneMedia::default()
                });
                out.len() - 1
            }
        };
        let m = &mut out[slot];
        let span = &run.span;
        m.rows.push(span.rows);
        m.patches.extend_from_slice(&span.payload);
        m.embed_rows.extend_from_slice(&span.embed_rows);
        m.embed_weights.extend_from_slice(&span.embed_weights);
        // **THE ROUTES: PAYLOAD ROW k OF THIS SPAN LANDS AT ITS RUN'S k-TH
        // TOKEN ROW.** A span whose payload is longer than its run — a merge
        // that folds four patches into one token — spends the surplus on the
        // fold-space tail, which the contract spells `-1`.
        for k in 0..span.rows {
            m.routes.push(if k < run.rows {
                (run.anchor + k) as i32
            } else {
                PATCH_ROUTE_DROP
            });
        }
        // **THE TOWER'S ROTATION STREAM IS READ OFF THE SPAN, NOT DERIVED
        // FROM ITS GRID** (wave MD-C, the first of MD-B's two pinned seams).
        //
        // What stood here walked `0..span.rows` and answered `(k / gw, k % gw)`
        // — RASTER order — and the payload rows are not in raster order. Both
        // front-ends lay their rows out FOLD-BLOCK-MAJOR by statute: qwen's
        // `patchify` emits block row, block column, then row and column inside
        // the 2 x 2 merge block, and gemma's the same sentence at k = 3, pool-
        // block-major, precisely so `layout.merge_rows` / `layout.pool_rows`
        // read no geometry. So the raster derivation named a DIFFERENT patch
        // for every row but the block diagonal, and the tower would have
        // rotated a correct payload by the wrong coordinates — a pass that
        // fires, scatters, and answers fluently about a scrambled image.
        //
        // `EncodedSpan::positions` is the front-end's own answer, two entries
        // per payload row, `(y, x)`, in `payload`'s order. Widening it to
        // `(t, y, x)` is the whole conversion: a still image has no temporal
        // axis and the tower states `sections[0] == 0`, so `t` is zero and
        // nothing reads it. A video's frames arrive as separate spans, each
        // with its own run, so no span here spans two `t`.
        let owed = 2 * span.rows as usize;
        if span.positions.len() == owed {
            for yx in span.positions.chunks_exact(2) {
                m.positions.push(0);
                m.positions.push(i32::try_from(yx[0]).unwrap_or(i32::MAX));
                m.positions.push(i32::try_from(yx[1]).unwrap_or(i32::MAX));
            }
        } else {
            // A front-end whose tower reads no position stream answers none
            // ([`EncodedSpan::positions`]'s own doc). The contract still owes
            // the engine three numbers per payload row — `Fault::PatchPayload`
            // is exact, not empty-or-exact — so the origin is written and the
            // plan that declares no `PatchPositions` reads nothing from it.
            for _ in 0..span.rows {
                m.positions.extend_from_slice(&[0, 0, 0]);
            }
        }
    }
    // **THE TRUNK'S STREAM IS PER LANE AND ONLY WHEN M-ROPE ASKS FOR IT.**
    // Empty means scalar `(p, p, p)`, which is the right and cheap answer for
    // every 1-D-RoPE model; a lane under M-RoPE owes one triple per token row.
    for m in &mut out {
        let needs = matched
            .iter()
            .any(|r| r.lane == m.lane && r.span.uses_mrope);
        if !needs {
            continue;
        }
        let rows = lane_rows.get(m.lane as usize).copied().unwrap_or(0);
        m.token_positions = Vec::with_capacity(3 * rows as usize);
        for p in 0..rows {
            // A text row takes `(p, p, p)`; a row inside a run takes its
            // patch's merged-grid coordinate. MD-B's front-ends carry the
            // merged grid, so the second case is filled from the span rather
            // than guessed here.
            let inside = matched.iter().find(|r| {
                r.lane == m.lane && p >= r.anchor && p < r.anchor + r.rows
            });
            match inside {
                Some(run) => {
                    let k = p - run.anchor;
                    let g = run.span.grid;
                    let hw = g.h.max(1) * g.w.max(1);
                    let t = k / hw.max(1);
                    let rem = k % hw.max(1);
                    m.token_positions.push(i32::try_from(t).unwrap_or(i32::MAX));
                    m.token_positions
                        .push(i32::try_from(rem / g.w.max(1)).unwrap_or(i32::MAX));
                    m.token_positions
                        .push(i32::try_from(rem % g.w.max(1)).unwrap_or(i32::MAX));
                }
                None => {
                    let p = i32::try_from(p).unwrap_or(i32::MAX);
                    m.token_positions.extend_from_slice(&[p, p, p]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::model::media::Grid;

    const PAD: u32 = 151_655;

    fn span(rows: u32) -> Arc<EncodedSpan> {
        Arc::new(EncodedSpan {
            token_count: rows,
            position_span: rows,
            grid: Grid::still(1, rows),
            patch_grid: Grid::still(1, rows),
            uses_mrope: false,
            payload: vec![0.5; rows as usize],
            rows,
            positions: Vec::new(),
            embed_rows: Vec::new(),
            embed_weights: Vec::new(),
            prefix: vec![1],
            placeholder: PAD,
            suffix: vec![2],
        })
    }

    /// The happy path is one `extend`: text, then the span's own spelling,
    /// then text. The scan finds it without being told where it is.
    #[test]
    fn a_matching_submission_scans_to_its_runs() {
        let s = span(3);
        let toks: Vec<u32> = [&[10, 11][..], &s.tokens(), &[12][..]].concat();
        assert_eq!(toks, vec![10, 11, 1, PAD, PAD, PAD, 2, 12]);
        let matched = scan(&[&toks], &[Arc::clone(&s)]).expect("matches");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].lane, 0);
        assert_eq!(matched[0].anchor, 3, "the run starts after the delimiter");
        assert_eq!(matched[0].rows, 3);
    }

    #[test]
    fn two_spans_match_two_runs_in_order() {
        let a = span(2);
        let b = span(4);
        let toks: Vec<u32> = [&a.tokens()[..], &[9][..], &b.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&b)]).expect("matches");
        assert_eq!(matched.len(), 2);
        assert_eq!((matched[0].anchor, matched[0].rows), (1, 2));
        assert_eq!((matched[1].anchor, matched[1].rows), (6, 4));
    }

    /// A text-only submission scans nothing and is unchanged.
    #[test]
    fn a_text_only_submission_matches_nothing() {
        let matched = scan(&[&[1, 2, 3]], &[]).expect("text is fine");
        assert!(matched.is_empty());
    }

    #[test]
    fn more_runs_than_spans_is_refused_by_name() {
        let a = span(2);
        let toks: Vec<u32> = [&a.tokens()[..], &[9][..], &a.tokens()[..]].concat();
        let err = scan(&[&toks], &[Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunCount");
        assert_eq!(err, Refusal::RunCount { runs: 2, spans: 1 });
        assert!(err.to_string().contains("2 media runs"), "{err}");
        assert!(err.to_string().contains("1 spans"), "{err}");
    }

    #[test]
    fn more_spans_than_runs_is_refused_by_name() {
        let a = span(2);
        let toks = a.tokens();
        let err = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunCount");
        assert_eq!(err, Refusal::RunCount { runs: 1, spans: 2 });
    }

    /// The one that matters: a guest that sliced a run in half fires a pass
    /// whose count is right and whose geometry is wrong.
    #[test]
    fn a_half_sliced_run_is_refused_on_length() {
        let a = span(4);
        // The guest kept the delimiters and dropped two pads.
        let toks = vec![1, PAD, PAD, 2];
        let err = scan(&[&toks], &[Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunLength");
        assert_eq!(
            err,
            Refusal::RunLength {
                index: 0,
                run_rows: 2,
                span_rows: 4
            }
        );
        assert!(err.to_string().contains("run 0 is 2 rows"), "{err}");
        assert!(err.to_string().contains("occupies 4"), "{err}");
    }

    /// Two runs spliced adjacently are ONE maximal run of the wrong length,
    /// which is why the scan is over maximal runs.
    #[test]
    fn two_runs_spliced_adjacently_are_one_over_long_run() {
        let a = span(2);
        let toks = vec![1, PAD, PAD, PAD, PAD, 2];
        let err = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunCount");
        assert_eq!(err, Refusal::RunCount { runs: 1, spans: 2 });
    }

    #[test]
    fn a_span_with_no_run_anywhere_is_an_orphan_by_name() {
        let a = span(2);
        let err = scan(&[&[10, 11, 12]], &[Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaOrphanSpans");
        assert!(err.to_string().contains("no media run"), "{err}");
    }

    #[test]
    fn a_run_with_no_span_is_an_orphan_by_name() {
        let toks = vec![1, PAD, PAD, 2];
        let err = refuse_orphan_runs(&[&toks], Some(PAD)).unwrap_err();
        assert_eq!(err.name(), "MediaOrphanRuns");
        assert_eq!(err, Refusal::OrphanRuns { runs: 1 });
        assert!(err.to_string().contains("attached no spans"), "{err}");
    }

    #[test]
    fn text_only_tokens_carry_no_orphan_run() {
        assert!(refuse_orphan_runs(&[&[1, 2, 3]], Some(PAD)).is_ok());
        assert!(refuse_orphan_runs(&[&[1, 2, 3]], None).is_ok());
    }

    /// The scan is over the submission, so a second lane's run is found too.
    #[test]
    fn a_run_in_the_second_lane_is_found_and_stays_lane_relative() {
        let a = span(2);
        let lane0 = vec![7, 8, 9];
        let lane1 = a.tokens();
        let matched = scan(&[&lane0, &lane1], &[Arc::clone(&a)]).expect("matches");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].lane, 1);
        assert_eq!(matched[0].anchor, 1, "an anchor is its own lane's offset");
    }

    /// The MD-C hand-off: a matched run derives its lane's contract rows, and
    /// route k of the payload is token row `anchor + k`.
    #[test]
    fn the_hand_off_derives_routes_from_the_anchor() {
        let a = span(3);
        let toks: Vec<u32> = [&[5, 5][..], &a.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a)]).expect("matches");
        let media = lane_media(&matched, &[toks.len() as u32]);
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].lane, 0);
        assert_eq!(media[0].rows, vec![3]);
        assert_eq!(media[0].routes, vec![3, 4, 5]);
        assert_eq!(media[0].patches.len(), 3);
        assert_eq!(media[0].positions.len(), 9, "three per payload row");
        assert!(
            media[0].token_positions.is_empty(),
            "1-D RoPE owes no trunk stream — empty means (p, p, p)"
        );
    }

    /// **THE DERIVED ROWS ARE THE CONTRACT'S ROWS** (media-door §6, MD-C).
    ///
    /// The hand-off's whole claim is that MD-C's marshaling is a move: what
    /// the scan derives IS `engine::fire::StepMedia`, so the step carries it
    /// without a second struct and without a conversion that could disagree.
    /// A `Vec<StepMedia>` binding that compiles is that claim, checked.
    #[test]
    fn what_the_scan_derives_is_the_contract_s_own_media_row() {
        let s = span(2);
        let toks: Vec<u32> = [&[9][..], &s.tokens()].concat();
        let matched = scan(&[&toks], &[Arc::clone(&s)]).expect("one run, one span");
        let rows: Vec<engine::fire::StepMedia> =
            lane_media(&matched, &[toks.len() as u32]);
        assert_eq!(rows.len(), 1);
        rows[0]
            .validate(toks.len() as u32)
            .expect("the contract admits what the scan derived");
    }
}
