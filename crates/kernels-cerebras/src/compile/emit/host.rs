//! What the host driver needs to find on the wafer: the PEs it reads and writes, the shapes of
//! what it moves and the host's share of the model's math (the logit softcap). Everything the
//! engine's `ArtifactModel` does that depends on the program is read from here (through the
//! [`super::super::artifact::Artifact`]): the host names no layout.

use serde::{Deserialize, Serialize};

use crate::compile::floor::{Coord, CtlLine, Rect, Unit, unskip};
use crate::compile::program::Program;

/// The host's map of the program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostLayout {
    pub w: u32,
    pub h: u32,
    /// Unit 0's CX column (x in, the embedding / forced tokens) and the last unit's (the final x).
    pub first_cx: Rect,
    pub last_cx: Rect,
    /// The head's weight rectangle, its control line and the token loop's root (ring, force,
    /// nforce, lgt).
    pub head: Rect,
    pub head_ctl: Rect,
    pub root: Coord,
    /// x's dims a CX row, by row run (Gemma 12B's attention / plain rows; one run elsewhere):
    /// (first row, last row + 1, dims, the run's wafer-row offset from the unit's top).
    pub x_runs: Vec<(u32, u32, u32, u32)>,
    /// Vocabulary rows a head PE, the head's embedding words a PE.
    pub vocab_rows: u32,
    pub hemb_words: u32,
    /// Positions the program serves (the plan's contract).
    pub positions: u32,
    pub cap: u32,
    pub token_loop: bool,
    /// Posted loop steps a launch, by default (the plan's contract).
    pub loop_len: u32,
    /// Weight rows a unit (the hidden dim's split) and the CY row's local index among them (the
    /// x runs split there: the CX column skips the CY row).
    pub rows: u32,
    pub cy_row: u32,
    /// x's dims a CX row where every row has the same (the last run's elsewhere).
    pub row_dims: u32,
    /// The head's weight rows and columns (its control line not counted): vocabulary rows by
    /// dims columns (Qwen, Gemma), dims rows by vocabulary columns (gpt-oss's band).
    pub head_rows: u32,
    pub head_cols: u32,
    /// A head whose control line is a row (gpt-oss): the logits are read along it,
    /// (x0, y, count).
    pub band: Option<(u32, u32, u32)>,
    /// Where `lgt` is read (the head's control line: a column, or the band's row).
    pub logits: Rect,
    /// Where a launch's first token id goes (`tin`) when per-layer-input storage PEs look it up
    /// (the plan's PLE): the head control line.
    pub token_in: Option<Rect>,
    /// The final logit softcap the host applies (monotone: the wafer's argmax stands).
    pub softcap: Option<f32>,
    /// A ring of seats ([`crate::compile::block::seats`]); `None`: one sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seats: Option<SeatsHost>,
}

/// Where a ring's host words go: its seats, and every unit's CY row (`spos`) as rectangles (the
/// CX corner left out). Each seat's first x goes to unit 0's CX column (`xs`, `[seat][dims]` a
/// PE), its forced laps to the root (`nforce`, a seat's laps; `force` by pass).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeatsHost {
    pub m: u32,
    pub cy: Vec<Rect>,
    /// The positions the seats' shared KV pool holds (its entries a row by the rows); each seat's
    /// window starts at a position the host states (`sbase`), a multiple of the rows. 0: a
    /// program of fixed windows (`NE / m` entries a row each), no `sbase`.
    #[serde(default)]
    pub pool: u32,
    /// The pool's KV rows (a window's first position their multiple); 0: the unit's rows.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rows: u32,
    /// The per-layer-input storage columns: a launch's first lap's words go to their units' CY
    /// rows from the host (`pls`), never down the bus.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ple: Vec<PleHost>,
}

/// A storage column's words for a ring's first lap: token `t`'s are storage row `t / tokens`'s,
/// staged in `pt<store>_<row / per>` (`[row % per][column][tokens][words]`, f16), at weight column
/// `c`; each of its units' CY row PEs (a rectangle and its first weight column) takes a seat's
/// `words` values (f32) and a word past them, a seat after another.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PleHost {
    pub store: u32,
    pub per: u32,
    pub tokens: u32,
    pub words: u32,
    pub cols: u32,
    pub cy: Vec<(Rect, u32)>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// A unit's CX column.
fn cx_rect(u: &Unit) -> Rect {
    Rect {
        x: u.cx_x(),
        y: u.rect.y,
        w: 1,
        h: u.rect.h,
    }
}

/// A unit's CY row, either side of its CX column.
fn cy_rects(u: &Unit) -> Vec<Rect> {
    let (x0, x1, cx) = (u.rect.x, u.rect.x + u.rect.w, u.cx_x());
    [(x0, cx), (cx + 1, x1)]
        .into_iter()
        .filter(|(a, b)| b > a)
        .map(|(a, b)| Rect {
            x: a,
            y: u.cy_y(),
            w: b - a,
            h: 1,
        })
        .collect()
}

impl HostLayout {
    pub fn of(prog: &Program) -> Result<HostLayout, String> {
        let (f, p, c) = (&prog.floor, &prog.plan.partition, &prog.plan.contract);
        let first = f.units.first().ok_or("program: no units")?;
        let last = f.units.last().ok_or("program: no units")?;
        let cy_row = first.cy;
        // x runs: the plan's row runs, each cut at the CY row, which the CX column skips
        let runs: Vec<(u32, u32, u32)> = p
            .row_dims
            .iter()
            .map(|r| (r.rows.0, r.rows.1, r.dims))
            .collect();
        let mut x_runs = Vec::new();
        for (r0, r1, d) in runs {
            for (a, b) in [(r0, r1.min(cy_row)), (r0.max(cy_row), r1)] {
                if b > a {
                    x_runs.push((a, b, d, first.row_y(a) - first.rect.y));
                }
            }
        }
        let row_dims = x_runs.last().map_or(0, |r| r.2);
        let hb = &f.head;
        let (head_ctl, head_rows, head_cols, band) = match hb.ctl {
            CtlLine::Column(x) => (
                Rect {
                    x,
                    y: hb.rect.y,
                    w: 1,
                    h: hb.rect.h,
                },
                hb.rect.h,
                hb.rect.w.saturating_sub(1),
                None,
            ),
            CtlLine::Row(y) => (
                Rect {
                    x: hb.rect.x,
                    y,
                    w: hb.rect.w,
                    h: 1,
                },
                hb.rect.h.saturating_sub(1),
                hb.rect.w,
                Some((hb.rect.x, y, hb.rect.w)),
            ),
        };
        // a head PE's embedding words: the hidden over the head's dims split (its columns, or
        // the band's rows)
        let dims_split = if band.is_some() { head_rows } else { head_cols };
        let hemb_words = p.hidden_placed().checked_div(dims_split).unwrap_or(0);
        let logits = match band {
            Some((x, y, n)) => Rect { x, y, w: n, h: 1 },
            None => Rect {
                x: head_ctl.x,
                y: 0,
                w: 1,
                h: head_rows,
            },
        };
        Ok(HostLayout {
            w: f.w,
            h: f.h,
            first_cx: cx_rect(first),
            last_cx: cx_rect(last),
            head: hb.rect,
            head_ctl,
            root: hb.root,
            x_runs,
            vocab_rows: p.head.vocab_rows,
            hemb_words,
            positions: c.positions,
            cap: c.cap,
            token_loop: c.token_loop,
            loop_len: c.loop_len,
            rows: p.rows,
            cy_row,
            row_dims,
            head_rows,
            head_cols,
            band,
            logits,
            token_in: prog.plan.shape.ple.map(|_| logits),
            softcap: prog.plan.shape.head.final_softcap,
            seats: (prog.plan.seats > 1).then(|| SeatsHost {
                m: prog.plan.seats,
                cy: f.units.iter().flat_map(cy_rects).collect(),
                pool: p.kv_entries.per_row * p.kv_rows,
                rows: if p.kv_rows == p.rows { 0 } else { p.kv_rows },
                ple: f
                    .stores
                    .iter()
                    .enumerate()
                    .map(|(i, st)| {
                        let lv = crate::compile::block::ple::levels(&prog.plan, st.units.len());
                        let cy = st
                            .units
                            .iter()
                            .filter_map(|&u| f.units.iter().find(|v| v.index == u))
                            .flat_map(|u| {
                                cy_rects(u)
                                    .into_iter()
                                    .map(|r| (r, unskip(r.x - u.rect.x, u.cx)))
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                        PleHost {
                            store: i as u32,
                            per: crate::compile::block::ple::rows_a_file(
                                p.cols,
                                st.tokens_per_pe,
                                lv,
                            ),
                            tokens: st.tokens_per_pe,
                            words: lv,
                            cols: p.cols,
                            cy,
                        }
                    })
                    .collect(),
            }),
        })
    }

    /// The rows a ring's windows start at a multiple of: its pool's KV rows (at least 1).
    #[must_use]
    pub fn pool_rows(&self) -> u32 {
        match self.seats.as_ref().map_or(0, |s| s.rows) {
            0 => self.rows.max(1),
            r => r,
        }
    }

    /// Every CX row's x offset in the hidden vector, `rows + 1` entries (the runs' dims summed).
    pub fn x_offsets(&self) -> Vec<usize> {
        let mut dims = vec![0u32; self.rows as usize];
        for &(r0, r1, d, _) in &self.x_runs {
            for r in r0..r1 {
                dims[r as usize] = d;
            }
        }
        let mut off = vec![0usize];
        for d in dims {
            off.push(off.last().unwrap() + d as usize);
        }
        off
    }
}
