//! What the host driver needs to find on the wafer: the PEs it reads and writes, the shapes of
//! what it moves and the host's share of the model's math (the logit softcap). Everything the
//! engine's `SplitModel` does that depends on the program is read from here (through the
//! [`super::super::artifact::Artifact`]): the host names no family.

use serde::{Deserialize, Serialize};

use crate::spatial::floor::{Coord, CtlLine, Rect, Unit};
use crate::spatial::program::Program;

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

impl HostLayout {
    pub fn of(prog: &Program) -> Result<HostLayout, String> {
        let (f, p, c) = (&prog.floor, &prog.plan.partition, &prog.plan.contract);
        let first = f.units.first().ok_or("split program: no units")?;
        let last = f.units.last().ok_or("split program: no units")?;
        let hidden = prog.plan.shape.hidden_placed;
        let cy_row = first.cy;
        // x runs: the plan's row runs (one covering every row when it states none), each cut at
        // the CY row, which the CX column skips
        let runs: Vec<(u32, u32, u32)> = if p.row_dims.is_empty() {
            if p.rows == 0 || !hidden.is_multiple_of(p.rows) {
                return Err(format!(
                    "split program: {} rows do not divide the hidden {hidden}",
                    p.rows
                ));
            }
            vec![(0, p.rows, hidden / p.rows)]
        } else {
            p.row_dims
                .iter()
                .map(|r| (r.rows.0, r.rows.1, r.dims))
                .collect()
        };
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
        let hemb_words = hidden.checked_div(dims_split).unwrap_or(0);
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
            rows: p.rows,
            cy_row,
            row_dims,
            head_rows,
            head_cols,
            band,
            logits,
            token_in: prog.plan.shape.ple.map(|_| logits),
            softcap: prog.plan.shape.head.final_softcap,
        })
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
