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

use models::media::EncodedSpan;
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
        // **THE ROUTES ARE THE LANE'S FOLD-SPACE PREFIX, AND THE `-1` TAIL IS
        // ONE TAIL AT THE END OF THE LANE** (multimodal §17; `place_routes`'
        // own doc in both engines).
        //
        // A route is read at the FOLD's OUTPUT row, not at a patch row: both
        // shells take `patches / fold` entries starting at `patch_offset /
        // fold`, where `patches` is the LANE's total. So output row `j` of the
        // lane is `routes[j]`, and the lane's spans contribute their soft
        // tokens BACK TO BACK — span 0's `token_count` entries, then span 1's,
        // with nothing between them.
        //
        // **THIS IS THE TWO-IMAGES-ONE-LANE BUG, AND IT WAS SILENT.** What
        // stood here wrote `span.rows` entries per span — `run.rows` addresses
        // then that SPAN's own `-1` padding — which is the same vector as this
        // one for exactly one span, because then the padding is already at the
        // end. With two spans the reader's live prefix walked span 0's
        // addresses, ran into span 0's padding, and stopped before span 1's
        // addresses were ever reached: image 1's soft tokens were all dropped
        // and the pass answered fluently about the image it half-saw.
        //
        // The vector still owes ONE ENTRY PER PAYLOAD ROW — `StepMedia::
        // validate` and both shells' `Fault::PatchPayload` check that length —
        // so the surplus a compacting fold spends is padded on at the end of
        // the lane, below, once every span has laid down its prefix.
        for k in 0..run.rows {
            m.routes.push((run.anchor + k) as i32);
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
    // **AND THE FOLD-SPACE TAIL, ONCE PER LANE** (see the routes push above).
    // Every span of the lane has laid its addresses down back to back; the
    // rest of the `[Dim::Patches]` rectangle is rows the fold spends and the
    // contract spells them `-1`. Written as a `while` and not a `resize` so a
    // lane whose spans somehow claim more token rows than payload rows keeps
    // its over-long vector and is refused by length downstream rather than
    // silently truncated here.
    for m in &mut out {
        let owed = m.rows.iter().copied().fold(0usize, |a, r| a + r as usize);
        while m.routes.len() < owed {
            m.routes.push(PATCH_ROUTE_DROP);
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
        // This lane's runs in token order. `scan` already emits them that way
        // — it sorts each lane's runs by anchor — and the walk below reads
        // them once, in step with the cursor, so the order is stated here
        // rather than assumed.
        let mut runs: Vec<&MatchedRun> =
            matched.iter().filter(|r| r.lane == m.lane).collect();
        runs.sort_by_key(|r| r.anchor);

        // **THE CURSOR IS NOT THE TOKEN ROW, AND THAT IS THE WHOLE POINT**
        // (`get_rope_index`, qwen3.5's own; transcribed in the pinned test).
        //
        // Upstream walks a sequence in modality groups carrying one
        // `current_pos`:
        //
        // * a text group of length `L` takes `current_pos + 0..L` on all three
        //   axes and advances the cursor by `L`;
        // * an image takes `get_vision_position_ids(current_pos, grid)`, which
        //   is `(current_pos + t, current_pos + h, current_pos + w)` over the
        //   MERGED grid in `ij` meshgrid order — the RUN-START OFFSET on every
        //   axis, `t` included — and advances the cursor by
        //   `max(llm_grid_h, llm_grid_w)`, which is `EncodedSpan::position_span`.
        //
        // **THE OFFSET USED TO BE MISSING AND THE BUG WAS LATENT.** The raw
        // merged-grid triple starts at `(0, 0, 0)` however deep into the
        // context its run lands, so an image's positions ran BACKWARD past
        // everything before it. A caption prompt is short enough that the
        // rotation barely notices; a long context rotates the image as though
        // it sat at the start of the sequence.
        //
        // **AND THE TEXT AFTER AN IMAGE DOES NOT RESUME AT ITS TOKEN ROW.** An
        // image spends `t·h·w` token rows and only `max(h, w)` positions, so
        // the cursor and the row index part company at the first image and
        // never meet again — which is exactly what upstream's
        // `mrope_position_deltas` records.
        m.token_positions = Vec::with_capacity(3 * rows as usize);
        let mut cursor: u32 = 0;
        let mut p: u32 = 0;
        let mut next = 0usize;
        while p < rows {
            match runs.get(next).filter(|r| r.anchor == p) {
                Some(run) => {
                    let start = cursor;
                    let g = run.span.grid;
                    let (gh, gw) = (g.h.max(1), g.w.max(1));
                    let hw = gh * gw;
                    for k in 0..run.rows.min(rows - p) {
                        let t = k / hw;
                        let rem = k % hw;
                        for axis in [t, rem / gw, rem % gw] {
                            m.token_positions.push(
                                i32::try_from(start.saturating_add(axis))
                                    .unwrap_or(i32::MAX),
                            );
                        }
                    }
                    cursor = cursor.saturating_add(run.span.position_span);
                    p = p.saturating_add(run.rows);
                    next += 1;
                }
                None => {
                    let at = i32::try_from(cursor).unwrap_or(i32::MAX);
                    m.token_positions.extend_from_slice(&[at, at, at]);
                    cursor = cursor.saturating_add(1);
                    p += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::media::Grid;

    const PAD: u32 = 151_655;

    /// **THE SHELLS' OWN READING OF `routes`, TRANSCRIBED** — `place_routes`
    /// in `engine_metal::serve` and the identical loop in `engine_cuda::serve`,
    /// written out here so this crate's gate reads the vector the way the two
    /// consumers do rather than the way the producer wrote it.
    ///
    /// A lane owns `patches / fold` slots of the fire's route vector, starting
    /// at `patch_offset / fold`, and reads the leading `patches / fold`
    /// entries of the submission's own vector into them — `patches` being the
    /// LANE's total across every span it carries. This answers that prefix,
    /// rebased by `row_offset` the way a shell rebases it, with the sentinel
    /// left alone.
    fn as_a_shell_reads_it(routes: &[i32], patch_rows: u32, fold: u32, row_offset: i32) -> Vec<i32> {
        let live = (patch_rows / fold.max(1)) as usize;
        routes
            .iter()
            .take(live)
            .map(|&route| if route < 0 { route } else { route + row_offset })
            .collect()
    }

    /// A span that folds `fold` payload rows into one soft token — qwen's
    /// 2 x 2 spatial merge is `fold == 4`, and the fold is the whole reason
    /// `routes` is longer than the addresses in it.
    fn folded_span(soft: u32, fold: u32) -> Arc<EncodedSpan> {
        let rows = soft * fold;
        Arc::new(EncodedSpan {
            token_count: soft,
            position_span: soft,
            grid: Grid::still(1, soft),
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

    /// An M-RoPE image span: a merged grid of `h x w`, so `h · w` soft tokens
    /// over `4 · h · w` payload rows, and a cursor advance of `max(h, w)` —
    /// `Qwen35Vision`'s own `position_span`.
    fn image_span(h: u32, w: u32) -> Arc<EncodedSpan> {
        let soft = h * w;
        let rows = soft * 4;
        Arc::new(EncodedSpan {
            token_count: soft,
            position_span: h.max(w),
            grid: Grid::still(h, w),
            patch_grid: Grid::still(2 * h, 2 * w),
            uses_mrope: true,
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

    /// **ONE IMAGE IS THE CASE THE TWO READINGS AGREE ON**, which is why the
    /// bug below stayed latent: with a single span, "each span padded to its
    /// own payload rows" and "the lane's addresses then one tail" are the same
    /// vector, because the only padding is already at the end.
    #[test]
    fn one_folded_image_is_its_addresses_then_the_tail() {
        let a = folded_span(3, 4);
        let toks: Vec<u32> = [&[7][..], &a.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a)]).expect("one run, one span");
        let media = lane_media(&matched, &[toks.len() as u32]);
        assert_eq!(media[0].routes.len(), 12, "one entry per payload row");
        assert_eq!(media[0].routes[..3], [2, 3, 4], "the run starts at row 2");
        assert!(media[0].routes[3..].iter().all(|&r| r == PATCH_ROUTE_DROP));
        media[0]
            .validate(toks.len() as u32)
            .expect("the contract's own length rule is payload rows");
    }

    /// **TWO IMAGES IN ONE LANE, AND THE SECOND ONE'S SOFT TOKENS SURVIVE.**
    ///
    /// The bug this gate was written for: `routes` was built per span — each
    /// span's addresses followed by that SPAN's `-1` padding — while both
    /// shells read it as ONE fold-space prefix `patches / fold` long over the
    /// LANE. With one span the two readings coincide; with two, the prefix
    /// walked image 0's addresses, ran into image 0's padding, and stopped
    /// before image 1's addresses were reached. Image 1's soft tokens were all
    /// dropped, every length still agreed, and the pass answered fluently
    /// about an image it never saw.
    ///
    /// So the gate is not on the vector alone — it reads it the way the
    /// consumers do ([`as_a_shell_reads_it`]) and asks that every soft token
    /// row of BOTH runs is named.
    #[test]
    fn two_images_in_one_lane_both_reach_their_soft_tokens() {
        const FOLD: u32 = 4;
        let a = folded_span(2, FOLD);
        let b = folded_span(3, FOLD);
        let toks: Vec<u32> =
            [&[7][..], &a.tokens()[..], &[8][..], &b.tokens()[..]].concat();
        // 7, <1 PAD PAD 2>, 8, <1 PAD PAD PAD 2> — image 0 at token row 2,
        // image 1 at token row 7.
        let matched = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&b)])
            .expect("two runs, two spans");
        assert_eq!((matched[0].anchor, matched[1].anchor), (2, 7));

        let media = lane_media(&matched, &[toks.len() as u32]);
        assert_eq!(media.len(), 1, "one lane, two spans, one media row");
        assert_eq!(media[0].rows, vec![8, 12]);
        let patch_rows: u32 = media[0].rows.iter().sum();
        assert_eq!(
            media[0].routes.len(),
            patch_rows as usize,
            "the vector is over the FULL patch rectangle, however the live \
             prefix is laid out inside it",
        );

        let read = as_a_shell_reads_it(&media[0].routes, patch_rows, FOLD, 0);
        assert_eq!(
            read,
            vec![2, 3, 7, 8, 9],
            "the lane's fold output rows are one interval, so its spans' \
             addresses are back to back and image 1's are inside the prefix",
        );
        for run in &matched {
            for k in 0..run.rows {
                assert!(
                    read.contains(&((run.anchor + k) as i32)),
                    "token row {} of the run at {} is named by no route the \
                     shell reads: {read:?}",
                    k,
                    run.anchor,
                );
            }
        }
        assert!(
            media[0].routes[read.len()..]
                .iter()
                .all(|&r| r == PATCH_ROUTE_DROP),
            "everything past the live prefix is the fold's own tail",
        );
        media[0]
            .validate(toks.len() as u32)
            .expect("the contract admits the lane's routes");
    }

    /// The shell's rebase does not disturb the prefix: a lane that lands at
    /// fire row 20 moves every address by 20 and leaves the sentinel alone.
    #[test]
    fn the_rebase_moves_the_addresses_and_not_the_tail() {
        const FOLD: u32 = 4;
        let a = folded_span(1, FOLD);
        let b = folded_span(1, FOLD);
        let toks: Vec<u32> = [&a.tokens()[..], &b.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&b)]).expect("two runs");
        let media = lane_media(&matched, &[toks.len() as u32]);
        let patch_rows: u32 = media[0].rows.iter().sum();
        assert_eq!(
            as_a_shell_reads_it(&media[0].routes, patch_rows, FOLD, 20),
            vec![21, 24],
            "row 1 and row 4 of the lane, at fire rows 21 and 24",
        );
        assert_eq!(
            media[0].routes[2..]
                .iter()
                .copied()
                .filter(|&r| r == PATCH_ROUTE_DROP)
                .count(),
            6,
            "the tail is the fold's surplus, six rows of the eight",
        );
    }

    /// **THE M-ROPE TRIPLES CARRY THE RUN'S START POSITION** — `get_rope_index`,
    /// `transformers` v5.15.1 `models/qwen3_5/modeling_qwen3_5.py`, transcribed:
    ///
    /// ```text
    /// current_pos = 0
    /// for each modality group:
    ///     text of length L:  positions = current_pos + arange(L) on all three
    ///                        axes;  current_pos += L
    ///     image:             positions = get_vision_position_ids(current_pos, grid)
    ///                                  = (current_pos + t, current_pos + h,
    ///                                     current_pos + w) over the MERGED grid
    ///                                    in `meshgrid(..., indexing="ij")` order
    ///                        current_pos += max(llm_grid_h, llm_grid_w)
    /// ```
    ///
    /// Two facts fall out and both are asserted below:
    ///
    /// * **the offset is on every axis, `t` included** — `get_vision_position_ids`
    ///   adds `start_position` to the height and width ranges and then does
    ///   `vision_position_ids[0] += start_position` for the temporal one. What
    ///   stood here emitted the RAW merged-grid triple `(0, 0..h, 0..w)`, so an
    ///   image's positions ran backward past everything before it — invisible
    ///   in a caption prompt, wrong in a long context;
    /// * **the text after an image does not resume at its token row** — the
    ///   cursor advances by `max(h, w)` (which is `EncodedSpan::position_span`,
    ///   pinned against the reference processor in `model`'s own
    ///   `qwen3_5_media_is_the_pinned_arithmetic`) while the run spent `h · w`
    ///   token rows, so the two part company at the first image. That is
    ///   exactly what upstream's `mrope_position_deltas` records.
    #[test]
    fn an_images_triples_start_at_the_position_its_run_begins_at() {
        let img = image_span(2, 3);
        // 10, 11, <1 PAD x 6 2>, 12, 13 — twelve token rows, the run at 3.
        let toks: Vec<u32> =
            [&[10, 11][..], &img.tokens()[..], &[12, 13][..]].concat();
        assert_eq!(toks.len(), 12);
        let matched = scan(&[&toks], &[Arc::clone(&img)]).expect("one run");
        assert_eq!(matched[0].anchor, 3, "the pad run starts after the prefix");
        let media = lane_media(&matched, &[toks.len() as u32]);

        let want: Vec<i32> = vec![
            // Two text tokens and the vision-start delimiter: the cursor is
            // the token row while nothing has spent a position yet.
            0, 0, 0, //
            1, 1, 1, //
            2, 2, 2, //
            // The run begins at position 3, so every triple is 3 + the merged
            // grid's own `(t, h, w)` over a 2 x 3 grid in `ij` order.
            3, 3, 3, //
            3, 3, 4, //
            3, 3, 5, //
            3, 4, 3, //
            3, 4, 4, //
            3, 4, 5, //
            // `max(2, 3) == 3`, so the cursor is 3 + 3 = 6 and the suffix
            // takes it — one past the largest component the image carried (5),
            // which is upstream's `llm_positions.max() + 1`.
            6, 6, 6, //
            7, 7, 7, //
            8, 8, 8, //
        ];
        assert_eq!(media[0].token_positions, want);
        assert_eq!(media[0].token_positions.len(), 3 * toks.len());
        media[0]
            .validate(toks.len() as u32)
            .expect("one triple per token row of the lane");
    }

    /// **AND THE OFFSET COMPOUNDS**: the second image starts where the first
    /// one's cursor left off, not at its own token row and not at zero. This
    /// is the long-context shape in miniature — the further into the sequence
    /// an image sits, the larger the divergence the missing offset caused.
    #[test]
    fn a_second_image_starts_where_the_firsts_cursor_left_off() {
        let a = image_span(1, 2);
        let b = image_span(2, 1);
        let toks: Vec<u32> = [&a.tokens()[..], &[9][..], &b.tokens()[..]].concat();
        // <1 PAD PAD 2> 9 <1 PAD PAD 2> — nine token rows, runs at 1 and 6.
        assert_eq!(toks.len(), 9);
        let matched = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&b)]).expect("two runs");
        assert_eq!((matched[0].anchor, matched[1].anchor), (1, 6));
        let media = lane_media(&matched, &[toks.len() as u32]);
        assert_eq!(
            media[0].token_positions,
            vec![
                0, 0, 0, // the first vision-start delimiter
                1, 1, 1, // a's run begins at 1: (0, 0, 0) + 1
                1, 1, 2, //                      (0, 0, 1) + 1
                3, 3, 3, // a spends max(1, 2) = 2, so the cursor is 3
                4, 4, 4, // the text between the images
                5, 5, 5, // the second vision-start delimiter
                6, 6, 6, // b's run begins at 6: (0, 0, 0) + 6
                6, 7, 6, //                      (0, 1, 0) + 6
                8, 8, 8, // b spends max(2, 1) = 2, so the cursor is 8
            ],
        );
    }

    /// A lane whose spans rotate scalar owes no trunk stream at all, and the
    /// cursor walk above must not invent one — empty IS the answer, and it is
    /// what every 1-D-RoPE model submits.
    #[test]
    fn a_scalar_lane_still_owes_no_trunk_stream() {
        let a = folded_span(2, 4);
        let toks: Vec<u32> = [&[5][..], &a.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a)]).expect("one run");
        let media = lane_media(&matched, &[toks.len() as u32]);
        assert!(media[0].token_positions.is_empty());
    }
}
