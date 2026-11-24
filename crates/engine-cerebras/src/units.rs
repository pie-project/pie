//! A decode program of stream units (`stream::compile_decode`'s). Its statics are the
//! checkpoint's, written once as it opens; each launch runs B rows (a lane slot each), each a
//! token at its position over its lane's pages ([`Row`]), the argmax (or the logits) a row out.
//! The column's last page is the scratch an idle row stores into.

use engine_cpu::Forward;
use engine_cpu::forward::Values;
use poem_ir::Trace;

use crate::reading::{Fabric, Reading};

type R<T> = Result<T, String>;

pub struct UnitModel {
    rt: Box<dyn Fabric>,
    reading: Reading,
    trace: Trace,
    host: Forward,
    /// Tokens the wafer has taken, from position 0.
    pub fed: Vec<u32>,
    /// The positions its K / V columns hold.
    pub positions: usize,
    pub vocab: usize,
    pub launches: u64,
    pub steps: u64,
    /// The lane slots' page tables as last written (B x pages).
    table: Vec<u32>,
}

/// A row of a launch: `token` at `pos` over its lane's `pages` and its `state` slot, its logits
/// read back with `logits` (else its argmax alone).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub token: u32,
    pub pos: u32,
    pub pages: Vec<u32>,
    /// Its lane's recurrent state slot (a program over a recurrence's).
    pub state: u32,
    pub logits: bool,
}

impl UnitModel {
    /// The program `reading` on `rt` (loaded and running), its statics made from `plane` and
    /// written.
    pub fn open(
        mut rt: Box<dyn Fabric>,
        mut reading: Reading,
        trace: Trace,
        plane: &dyn Fn(&str, usize) -> Vec<f32>,
        positions: usize,
    ) -> R<UnitModel> {
        let mut weights = Forward::new(trace.clone(), None, plane)?;
        let statics = reading.statics(&mut weights, &reading.step(&trace, None, &[]))?;
        drop(weights);
        let host = Forward::host_for(trace.clone(), plane, &reading.entry_values())?;
        reading.write_statics(rt.as_mut(), &statics)?;
        let vocab = reading.outs.iter().map(|o| o.2).max().unwrap_or(0);
        // a served step reads no stamps (the per-PE cycles: 3 words a PE)
        reading.stamped = false;
        Ok(UnitModel {
            rt,
            reading,
            trace,
            host,
            fed: Vec::new(),
            positions,
            vocab,
            launches: 0,
            steps: 0,
            table: Vec::new(),
        })
    }

    /// The program's rows a launch (its lane slots).
    #[must_use]
    pub fn lanes(&self) -> usize {
        (self.reading.classes.iter())
            .map(|c| c.rows as usize)
            .sum::<usize>()
            .max(1)
    }

    /// The pages a lane may hold (the column's but the scratch) and their positions each.
    #[must_use]
    pub fn pages(&self) -> Option<(u32, u32)> {
        self.reading.paging.map(|g| (g.pages - 1, g.size))
    }

    /// `rows` run, B a launch (idle rows at the scratch page): each row's argmax, and its logits
    /// where it asks.
    pub fn rows(&mut self, rows: &[Row]) -> R<Vec<(u32, Option<Vec<f32>>)>> {
        let b = self.lanes();
        let mut out = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(b) {
            if let Some(g) = self.reading.paging {
                // the lane slots' tables: a row's pages, an idle row's the scratch
                let maxp = g.pages as usize;
                let mut table = vec![g.pages - 1; b * maxp];
                for (t, r) in chunk.iter().enumerate() {
                    if r.pages.len() > maxp - 1 || r.pos >= r.pages.len() as u32 * g.size {
                        return Err(format!(
                            "units: a row at {} over {} pages of {} positions",
                            r.pos,
                            r.pages.len(),
                            g.size
                        ));
                    }
                    table[t * maxp..t * maxp + r.pages.len()].copy_from_slice(&r.pages);
                }
                if table != self.table {
                    self.reading.write_pages(self.rt.as_mut(), &table)?;
                    self.table = table;
                }
            }
            if let Some(r) = chunk.iter().find(|r| r.state as usize >= b) {
                return Err(format!("units: state slot {} of {b}", r.state));
            }
            let toks: Vec<u32> = (0..b)
                .map(|t| chunk.get(t).map_or(chunk[0].token, |r| r.token))
                .collect();
            let pos: Vec<u32> = (0..b).map(|t| chunk.get(t).map_or(0, |r| r.pos)).collect();
            let (step, entries) =
                self.reading
                    .decode_step(&mut self.host, &self.trace, &toks, &pos)?;
            self.reading.write_pos(self.rt.as_mut(), pos[0])?;
            let slots: Vec<(u32, u32, u32)> = (0..b)
                .map(|t| (t as u32, chunk.get(t).map_or(0, |r| r.state), pos[t]))
                .collect();
            self.reading.write_rows(self.rt.as_mut(), b, &slots)?;
            self.launches += 1;
            self.steps += chunk.len() as u64;
            if !chunk.iter().any(|r| r.logits) && !self.reading.argmax.is_empty() {
                // the argmax off the wafer: two words a head root a row, not the vocabulary
                self.reading.write_entries(self.rt.as_mut(), &entries)?;
                self.rt.launch(&self.reading)?;
                self.reading.write_ahead(
                    self.rt.as_mut(),
                    &mut self.host,
                    &self.trace,
                    &toks,
                    &pos,
                )?;
                if let Some(a) = self.rt.argmax(&self.reading)? {
                    out.extend(a.into_iter().take(chunk.len()).map(|k| (k, None)));
                    continue;
                }
            }
            let (got, _) = self.reading.launch_on(self.rt.as_mut(), &step, &entries)?;
            self.reading
                .write_ahead(self.rt.as_mut(), &mut self.host, &self.trace, &toks, &pos)?;
            let back = self.reading.rows_back(&got)?;
            let logits = back
                .into_iter()
                .max_by_key(|c| c.width)
                .ok_or("units: no logits read back")?;
            let Values::F32(v) = logits.values else {
                return Err("units: the logits are not f32".into());
            };
            let w = logits.width;
            for (t, r) in chunk.iter().enumerate() {
                let l = &v[t * w..(t + 1) * w];
                // (NaN logits: a step whose attention refused a query; refused here too)
                if l.iter().any(|x| x.is_nan()) {
                    return Err(format!(
                        "units: row {t}'s logits are NaN (a refused attention query?)"
                    ));
                }
                out.push((argmax_of(l), r.logits.then(|| l.to_vec())));
            }
        }
        Ok(out)
    }

    /// One token at position `at`: the argmax after it and, with `logits` (or where the program
    /// reduces none on the wafer), the logits.
    fn step(&mut self, token: u32, at: usize, logits: bool) -> R<(u32, Option<Vec<f32>>)> {
        if at >= self.positions {
            return Err(format!(
                "units: position {at} past the program's {}",
                self.positions
            ));
        }
        let at = u32::try_from(at).map_err(|e| e.to_string())?;
        // one sequence: lane slot 0 over the pages in order, state slot 0
        let pages = self.pages().map_or(Vec::new(), |(pages, size)| {
            (0..(at / size + 1).min(pages)).collect()
        });
        let row = Row {
            token,
            pos: at,
            pages,
            state: 0,
            logits,
        };
        let mut got = self.rows(&[row])?;
        got.pop().ok_or_else(|| "units: no row back".to_string())
    }

    /// The cache brought to `start` tokens: a rewind replays the kept tokens from 0.
    fn seat(&mut self, start: usize) -> R<()> {
        if start == self.fed.len() {
            return Ok(());
        }
        if start > self.fed.len() {
            return Err(format!(
                "units: a run at {start} past the wafer's {} tokens",
                self.fed.len()
            ));
        }
        let keep = self.fed[..start].to_vec();
        tracing::info!(
            rows = start,
            held = self.fed.len(),
            "units: sequence replayed from 0"
        );
        self.fed.clear();
        for (i, &t) in keep.iter().enumerate() {
            self.step(t, i, false)?;
            self.fed.push(t);
        }
        Ok(())
    }

    /// `tokens` at `start..`: the argmax after each row, and with `logits` the last row's.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        logits: bool,
    ) -> R<(Vec<u32>, Option<Vec<f32>>)> {
        self.seat(start)?;
        let (mut argmax, mut last) = (Vec::with_capacity(tokens.len()), None);
        for (i, &t) in tokens.iter().enumerate() {
            let read = logits && i + 1 == tokens.len();
            let (a, l) = self.step(t, start + i, read)?;
            self.fed.push(t);
            argmax.push(a);
            last = l;
        }
        Ok((argmax, if logits { last } else { None }))
    }

    /// The token loop: `tokens` at `start..`, then `extra` steps each fed the argmax before; the
    /// argmax after each row.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let (mut argmax, _) = self.forward(tokens, start, false)?;
        for _ in 0..extra {
            let t = *argmax.last().ok_or("units: an empty token loop")?;
            let (a, _) = self.forward(&[t], self.fed.len(), false)?;
            argmax.extend(a);
        }
        Ok(argmax)
    }
}

/// The first index of the largest logit (torch.argmax's tie-break).
fn argmax_of(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}
