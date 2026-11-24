//! The shell's harvest: what a settled fire's lanes read back, reaped and
//! forgotten.

use super::*;

impl Shell {

    /// Lets step `landed`'s rows go, landed or not: a frame no one will ask
    /// the rows of.
    pub fn forget(&mut self, landed: &Landed) {
        if self.landed.remove(&landed.seq).is_none()
            && self.inflight.iter().any(|flight| flight.seq == landed.seq)
        {
            self.forgotten.insert(landed.seq);
        }
    }

    pub fn reap(&mut self) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.pending.landed())
        {
            self.harvest_one()?;
        }
        Ok(())
    }

    pub fn harvest_through(&mut self, seq: u64) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.seq <= seq)
        {
            self.harvest_one()?;
        }
        Ok(())
    }

    pub(super) fn harvest_one(&mut self) -> Result<()> {
        let Some(flight) = self.inflight.pop_front() else {
            return Ok(());
        };
        let waited = flight.pending.wait();
        // Each attached instance's epilogue, settled once the buffer
        // completed: one that did not commit faults its own lanes alone,
        // and the rest of the step commits.
        let mut faulted = Vec::new();
        if waited.is_ok() {
            for (instance, lanes) in &flight.attached {
                let outcome = self
                    .programs
                    .settle_launched(*instance)
                    .and_then(|fired| match fired {
                        crate::Fired::Committed => Ok(()),
                        other => Err(refused(&other, *instance)),
                    });
                if let Err(fault) = outcome {
                    if flight.told.is_none() {
                        // The handler told this step from the epilogue's
                        // status words; the full report is here.
                        tracing::warn!(instance = %instance, %fault, "an attached epilogue did not commit");
                    }
                    faulted.push((lanes.clone(), *instance, fault.to_string()));
                }
            }
        }
        if waited.is_ok()
            && let Some(check) = flight.beam_check.as_ref()
        {
            faulted.extend(beam_faulted(check));
        }
        // The sink hears the fire here, the buffer complete and its
        // epilogues settled, before the harvest returns.
        if let Some(told) = flight.told.as_ref() {
            told(waited.as_ref().err().map(ToString::to_string), &faulted);
        }
        if let Some(keepalive) = &self.keepalive {
            keepalive.end();
        }
        fire_trace(|| {
            let (start, end) = flight.pending.gpu_span_us();
            format!(
                "device-done seq={} rows={} gpu_us={}",
                flight.seq,
                flight.lanes,
                end.saturating_sub(start)
            )
        });
        if let Err(fault) = waited {
            self.arms.give(flight.arm);
            // Nothing the buffer ran committed: each attached instance's
            // pass is let go, so the instance stands where it stood before
            // the fire and its next fire stages afresh (an airborne pass
            // left set refused every later staging and fire of it).
            for (instance, _) in &flight.attached {
                self.programs.abandon_launched(*instance);
            }
            self.forgotten.remove(&flight.seq);
            if flight.told.is_none() {
                // The completion handler told the sink the step was lost:
                // whoever harvests next (another submit's staging, a close's
                // fence) is not the frame that failed and is not refused
                // for it.
                tracing::warn!(seq = flight.seq, %fault, "a frame's command buffer failed; its step was told lost");
                return Ok(());
            }
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }

        if let Some(plane) = self.nan_flags.as_ref() {
            let words = self.trace.values.len() + 1;
            let mut raw = vec![0u8; words * 4];
            if plane.read(0, &mut raw).is_ok() {
                let mut said = 0usize;
                for (value, word) in raw.as_chunks::<4>().0.iter().enumerate() {
                    let at = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                    if at == 0 {
                        continue;
                    }
                    said += 1;
                    if said <= 12 {
                        let node = self.trace.nodes.iter().position(|node| {
                            use poem_ir::Operands as _;
                            let mut outs = Vec::new();
                            node.op.outputs(&mut outs);
                            outs.iter().any(|out| out.0 as usize == value)
                        });
                        let named = node
                            .and_then(|at| {
                                use poem_ir::Operands as _;
                                self.trace.nodes.get(at).map(|n| (at, n.op.name()))
                            })
                            .map_or_else(
                                || "?".to_string(),
                                |(at, name)| format!("node {at} {name}"),
                            );
                        eprintln!(
                            "nan-check: seq {} value {value} ({named}) carries a non-finite \
                             element at or before {}",
                            flight.seq,
                            at - 1
                        );
                    }
                }
                if said > 0 {
                    eprintln!("nan-check: seq {} — {said} value(s)", flight.seq);
                }
            }
        }
        let mut rows = vec![(String::new(), Vec::new()); flight.lanes];
        for (lane, read) in flight.readbacks.iter().enumerate() {
            let Some((name, width, element)) = read else {
                continue;
            };
            let mut raw = vec![0u8; *width as usize * *element as usize];
            self.readout[self.arena.current()][flight.arm]
                .read(lane as u64 * self.readout_row_bytes, &mut raw)?;
            rows[lane] = (name.clone(), widen(&raw, *element as usize));
        }
        fire_trace(|| format!("readout-done seq={} rows={}", flight.seq, flight.lanes));
        self.arms.give(flight.arm);
        if !self.forgotten.remove(&flight.seq) {
            self.landed.insert(flight.seq, rows);
        }
        Ok(())
    }
}
