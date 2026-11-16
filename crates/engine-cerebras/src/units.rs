//! A decode program of stream units (GENERIC.md G1: `stream::compile_decode`'s), served as the
//! template's [`ArtifactModel`](crate::artifact::ArtifactModel) is: one sequence a cache, a
//! token a launch. Its statics are the checkpoint's (engine-cpu's [`Forward`]), written once as
//! it opens; each step's entries are the host forward's, the K / V and state PEs told the step's
//! position, and the head's rows (the logits) read back.

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
}

impl UnitModel {
    /// The program `reading` on `rt` (loaded and running), its statics made from `plane` (the
    /// landed checkpoint's) and written.
    pub fn open(
        mut rt: Box<dyn Fabric>,
        reading: Reading,
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
        })
    }

    /// One token at position `at`: the logits after it.
    fn step(&mut self, token: u32, at: usize) -> R<Vec<f32>> {
        if at >= self.positions {
            return Err(format!(
                "units: position {at} past the program's {}",
                self.positions
            ));
        }
        let at = u32::try_from(at).map_err(|e| e.to_string())?;
        let (step, entries) =
            self.reading
                .decode_step(&mut self.host, &self.trace, &[token], at)?;
        self.reading.write_pos(self.rt.as_mut(), at)?;
        let (got, _) = self.reading.launch_on(self.rt.as_mut(), &step, &entries)?;
        self.launches += 1;
        self.steps += 1;
        let back = self.reading.rows_back(&got)?;
        let logits = back
            .into_iter()
            .max_by_key(|c| c.width)
            .ok_or("units: no logits read back")?;
        match logits.values {
            Values::F32(v) => Ok(v),
            _ => Err("units: the logits are not f32".into()),
        }
    }

    /// The cache brought to `start` tokens: a rewind replays the kept tokens from 0 (a
    /// recurrent state holds no earlier position).
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
            self.step(t, i)?;
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
            let l = self.step(t, start + i)?;
            self.fed.push(t);
            argmax.push(argmax_of(&l));
            last = Some(l);
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
