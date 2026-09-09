use models::media::EncodedSpan;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    RunCount {
        runs: usize,
        spans: usize,
    },
    RunLength {
        index: usize,
        run_rows: u32,
        span_rows: u32,
    },
    OrphanSpans {
        spans: usize,
    },
    OrphanRuns {
        runs: usize,
    },
}

impl Refusal {
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

#[derive(Clone, Debug)]
pub struct MatchedRun {
    pub lane: u32,
    pub anchor: u32,
    pub rows: u32,
    pub span: Arc<EncodedSpan>,
}

pub use engine::fire::StepMedia as LaneMedia;

const PATCH_ROUTE_DROP: i32 = -1;

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

pub fn scan(lanes: &[&[u32]], spans: &[Arc<EncodedSpan>]) -> Result<Vec<MatchedRun>, Refusal> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let mut found: Vec<MatchedRun> = Vec::new();
    let mut pads: Vec<u32> = spans.iter().map(|s| s.placeholder).collect();
    pads.sort_unstable();
    pads.dedup();
    for (lane, tokens) in lanes.iter().enumerate() {
        let mut here: Vec<(u32, u32)> = Vec::new();
        for &pad in &pads {
            here.extend(runs_of(tokens, pad));
        }
        here.sort_unstable();
        for (anchor, rows) in here {
            found.push(MatchedRun {
                lane: lane as u32,
                anchor,
                rows,
                span: Arc::clone(&spans[0]),
            });
        }
    }

    if found.is_empty() {
        return Err(Refusal::OrphanSpans { spans: spans.len() });
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

pub fn refuse_orphan_runs(lanes: &[&[u32]], pad: Option<u32>) -> Result<(), Refusal> {
    let Some(pad) = pad else { return Ok(()) };
    let runs: usize = lanes.iter().map(|tokens| runs_of(tokens, pad).len()).sum();
    if runs == 0 {
        Ok(())
    } else {
        Err(Refusal::OrphanRuns { runs })
    }
}

#[must_use]
pub fn lane_media(matched: &[MatchedRun], lane_rows: &[u32], lane_base: &[u32]) -> Vec<LaneMedia> {
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
        for k in 0..run.rows {
            m.routes.push((run.anchor + k) as i32);
        }
        let owed = 2 * span.rows as usize;
        if span.positions.len() == owed {
            for yx in span.positions.as_chunks::<2>().0 {
                m.positions.push(0);
                m.positions.push(i32::try_from(yx[0]).unwrap_or(i32::MAX));
                m.positions.push(i32::try_from(yx[1]).unwrap_or(i32::MAX));
            }
        } else {
            for _ in 0..span.rows {
                m.positions.extend_from_slice(&[0, 0, 0]);
            }
        }
    }
    for m in &mut out {
        let owed = m.rows.iter().copied().fold(0usize, |a, r| a + r as usize);
        while m.routes.len() < owed {
            m.routes.push(PATCH_ROUTE_DROP);
        }
    }

    for m in &mut out {
        let needs = matched
            .iter()
            .any(|r| r.lane == m.lane && r.span.uses_mrope);
        if !needs {
            continue;
        }
        let rows = lane_rows.get(m.lane as usize).copied().unwrap_or(0);
        let mut runs: Vec<&MatchedRun> = matched.iter().filter(|r| r.lane == m.lane).collect();
        runs.sort_by_key(|r| r.anchor);

        m.token_positions = Vec::with_capacity(3 * rows as usize);
        let mut cursor: u32 = lane_base.get(m.lane as usize).copied().unwrap_or(0);
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
                                i32::try_from(start.saturating_add(axis)).unwrap_or(i32::MAX),
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

    #[test]
    fn media_every_case() {
        a_matching_submission_scans_to_its_runs();
        two_spans_match_two_runs_in_order();
        more_runs_than_spans_is_refused_by_name();
        more_spans_than_runs_is_refused_by_name();
        a_half_sliced_run_is_refused_on_length();
        a_span_with_no_run_anywhere_is_an_orphan_by_name();
        a_run_with_no_span_is_an_orphan_by_name();
        a_run_in_the_second_lane_is_found_and_stays_lane_relative();
    }

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

    fn two_spans_match_two_runs_in_order() {
        let a = span(2);
        let b = span(4);
        let toks: Vec<u32> = [&a.tokens()[..], &[9][..], &b.tokens()[..]].concat();
        let matched = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&b)]).expect("matches");
        assert_eq!(matched.len(), 2);
        assert_eq!((matched[0].anchor, matched[0].rows), (1, 2));
        assert_eq!((matched[1].anchor, matched[1].rows), (6, 4));
    }

    fn more_runs_than_spans_is_refused_by_name() {
        let a = span(2);
        let toks: Vec<u32> = [&a.tokens()[..], &[9][..], &a.tokens()[..]].concat();
        let err = scan(&[&toks], &[Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunCount");
        assert_eq!(err, Refusal::RunCount { runs: 2, spans: 1 });
        assert!(err.to_string().contains("2 media runs"), "{err}");
        assert!(err.to_string().contains("1 spans"), "{err}");
    }

    fn more_spans_than_runs_is_refused_by_name() {
        let a = span(2);
        let toks = a.tokens();
        let err = scan(&[&toks], &[Arc::clone(&a), Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaRunCount");
        assert_eq!(err, Refusal::RunCount { runs: 1, spans: 2 });
    }

    fn a_half_sliced_run_is_refused_on_length() {
        let a = span(4);
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

    fn a_span_with_no_run_anywhere_is_an_orphan_by_name() {
        let a = span(2);
        let err = scan(&[&[10, 11, 12]], &[Arc::clone(&a)]).unwrap_err();
        assert_eq!(err.name(), "MediaOrphanSpans");
        assert!(err.to_string().contains("no media run"), "{err}");
    }

    fn a_run_with_no_span_is_an_orphan_by_name() {
        let toks = vec![1, PAD, PAD, 2];
        let err = refuse_orphan_runs(&[&toks], Some(PAD)).unwrap_err();
        assert_eq!(err.name(), "MediaOrphanRuns");
        assert_eq!(err, Refusal::OrphanRuns { runs: 1 });
        assert!(err.to_string().contains("attached no spans"), "{err}");
    }

    fn a_run_in_the_second_lane_is_found_and_stays_lane_relative() {
        let a = span(2);
        let lane0 = vec![7, 8, 9];
        let lane1 = a.tokens();
        let matched = scan(&[&lane0, &lane1], &[Arc::clone(&a)]).expect("matches");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].lane, 1);
        assert_eq!(matched[0].anchor, 1, "an anchor is its own lane's offset");
    }
}
