//! What a load measures: the decode products' kernels timed by their
//! bodies, the refit a measuring boot makes, and the fit it settles at.

#[allow(clippy::wildcard_imports)]
use super::*;

impl Shell {
    /// A measuring load's second fit: the bodies armed over the least range
    /// The decode product shapes the armed bodies met with no measured time
    /// (`kernels_cuda::linear::routes`): each chosen by its decode body's
    /// time under each kernel (`Shell::routes_by_body`), saved in the
    /// device's user profile, and the lattice left armed under the choices.
    /// A load whose profile holds every shape measures nothing.
    pub(super) fn routes_measured(&mut self, profiles: Option<&std::path::Path>) -> Result<()> {
        let Some(routes) = self.device.ctx().routes().cloned() else {
            return Ok(());
        };
        let wanted = routes.wanted();
        if wanted.is_empty() {
            tracing::info!(
                "decode products' kernels from the profile: every shape the bodies \
                 hold reused, 0 measured"
            );
            return Ok(());
        }
        // A load that records no bodies serves every fire as an eager walk:
        // there is no body to time a product's kernel by, so it keeps the
        // routes it has, saves no choice to the profile, and says so.
        if !self.records_bodies() {
            tracing::info!(
                "card: routes unmeasured, no recorded bodies: {} decode product \
                 shape(s) keep their routes as they stand, and none is saved",
                wanted.len()
            );
            return Ok(());
        }
        // Under `[engine] measure = false` nothing is timed: each shape no
        // layer states keeps its stated kernel, by name, and none is saved.
        if self.routes_measuring == engine::rates::Measuring::Skip {
            tracing::info!(
                "rates: skipped: `[rates] product_ns` for {} decode product \
                 shape(s): nothing timed (`[engine] measure = false`); each keeps its stated \
                 kernel",
                wanted.len()
            );
            return Ok(());
        }
        let started = std::time::Instant::now();
        let measured = self.routes_by_body(routes, &wanted)?;
        if measured.is_empty() {
            // Every shape a decode body times is the profile's: nothing is
            // measured or saved; the rest (rows no decode body carries) keep
            // their stated kernel.
            tracing::info!(
                "decode products' kernels from the profile: 0 measured; {} shape(s) \
                 at rows no decode body carries keep their stated kernel",
                wanted.len()
            );
            return Ok(());
        }
        self.measured
            .push(format!("product_ns ({} decode product times)", measured.len()));
        let saved = crate::rates::save_product_times(
            profiles,
            self.device.name(),
            &self.keys,
            &self.products,
            &measured,
        )
        .map_err(Fault::Card)?;
        // The table this load's products are chosen from from now on.
        self.products = crate::rates::joined_product_times(&self.products, &measured);
        let chosen = self.device.ctx().routes().cloned();
        let said: Vec<String> = wanted
            .iter()
            .map(|(s, noted)| {
                let us = |kernel| {
                    measured
                        .iter()
                        .find(|(shape, k, ..)| shape == s && *k == kernel)
                        .map_or(f64::NAN, |&(.., ns, _)| ns / 1e3)
                };
                let pick = chosen
                    .as_ref()
                    .map_or(noted.candidates.first().copied().unwrap_or(0), |c| {
                        c.pick(*s, 0, &noted.candidates)
                    });
                let times: Vec<String> = noted
                    .candidates
                    .iter()
                    .map(|&kernel| format!("{kernel:08x} {:.1} us", us(kernel)))
                    .collect();
                format!(
                    "{:?} {}x{} m{}: step {} -> {pick:08x}",
                    s.kind,
                    s.rows,
                    s.k,
                    s.m,
                    times.join(" / "),
                )
            })
            .collect();
        tracing::info!(
            "chose the kernel of {} decode product shapes by their bodies' times in \
             {:.1} s ({}); {}",
            said.len(),
            started.elapsed().as_secs_f64(),
            saved.map_or_else(|| "not saved: no profile directory".to_string(), |p| format!("saved to {p}")),
            said.join("; "),
        );
        Ok(())
    }

    /// took `bodies_took`; they are forgotten, the range is fitted to what
    /// the card has past that hold, and the lattice armed again over it.
    /// Its record holds the figure, so a later boot fits once.
    pub(super) fn measured_refit(&mut self, device_total: u64, free_before: u64) -> Result<()> {
        let started = std::time::Instant::now();
        let mut hold = self.bodies_took;
        let weighed = self.cache.body_stats().census.bytes as u64;
        self.cache.forget_bodies();
        // A flat range was allocated at the least range the bodies armed
        // over; where the card backs more, it is let go (the weights
        // preseated in it with it) so the refit reserves it anew at what the
        // card backs, and the weights are seated again in it below.
        let preseated = self.pools.preseated();
        self.device.synchronize()?;
        let refilled = self.pools.free_for_refit()?;
        let resident = footprint(self, device_total, free_before);
        let fitted = match self
            .pools
            .fit_the_card(hold, self.held_for_fires, false, &resident)
        {
            // What the captures took (the device's free bytes they cost)
            // past one working set no longer fits: an arming that stopped for
            // room holds what its bodies weigh, the rest stated eager.
            Err(Fault::Residency(_)) if weighed < hold => {
                hold = weighed;
                self.pools
                    .fit_the_card(hold, self.held_for_fires, false, &resident)?
            }
            fitted => fitted?,
        };
        if refilled && let (true, Some((filed, copied))) = (preseated, self.preseated_as) {
            let began = std::time::Instant::now();
            self.preseat_weights(filed, copied)?;
            let chunks = self
                .pools
                .preseat_plan()
                .map_or(0, |plan| self.pools.weight_chunks(&plan).len());
            tracing::info!(
                "a measuring boot's flat range held the least range the bodies \
                 armed over; it was reserved anew at {} chunks, and the weights the pool holds \
                 ({chunks} chunks) were seated in it again from their copy in {:.1} s",
                fitted.chunks,
                began.elapsed().as_secs_f64()
            );
        }
        self.lay_workspace()?;
        self.bodies_mem = usize::try_from(hold).unwrap_or(usize::MAX);
        self.rearming = true;
        self.arming_at_load = true;
        let armed = self.arm_bodies();
        self.arming_at_load = false;
        self.rearming = false;
        armed?;
        tracing::info!(
            "a measuring boot: with no record of what this deployment's bodies \
             take, they armed over the least range that serves and took {} MiB; the range was \
             fitted again to the rest ({} chunks) and the lattice armed over it in {:.1} s. \
             Later boots hold the measured figure and fit once",
            hold >> 20,
            fitted.chunks,
            started.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    /// The decode products' times this load chooses their kernels from: the
    /// profile's under a matching key, joined by what it measured. For a
    /// test that serves a later load with the same choices.
    #[doc(hidden)]
    #[must_use]
    pub fn product_times(&self) -> &[profile::rates::ProductTime] {
        &self.products
    }

    /// Two load facts of this shell
    /// ([`engine::chunk::ChunkFacts::held_bytes`] and `outside_bytes`):
    /// what the fit holds back of the chunks it stated, and what the driver
    /// holds outside the engine's span, as the ledger enters it (read by
    /// the policy, taken out of no range).
    /// The second is the settled figure where the load settled one
    /// ([`Self::settle_outside`]): what the driver counts this worker
    /// holding past the ledger's own entries; else the ledger's own
    /// outside entries.
    #[must_use]
    pub fn held_and_outside(&self) -> (u64, u64) {
        (
            self.pools.kept_back_bytes(),
            self.outside.map_or_else(
                || self.ledger().bytes(engine::memory::Class::Outside),
                |(outside, ledgered)| outside.beyond(ledgered),
            ),
        )
    }

    /// Settles what this worker holds of the device beyond the ledger's
    /// own entries, before the load's last fit: once arming has fired
    /// every composition, each step kind (a decode step, then a prompt) is
    /// fired once more on a fresh slot, the driver's count of this worker
    /// read before and after each ([`crate::device::own::used_now`]). Two
    /// readings a fire apart within one of the device's allocation units
    /// settle it; else the largest stands, said so
    /// ([`engine::memory::Outside::settle`]). The pool holds what of it lies
    /// beside its committed pages from here on (the larger of it and its
    /// reading at sizing), and the last fit sizes the pool by that and the
    /// pool's own count under the budget: the driver is not asked again.
    /// Where the driver does not say, before the fires or between them, or
    /// the load seats no slot to fire on, it is said by name and nothing is
    /// settled: the reading at sizing stands.
    pub(in super::super) fn settle_outside(&mut self) -> Result<()> {
        use engine::memory::Class;
        let first = match crate::device::own::used_now() {
            Ok(bytes) => bytes,
            Err(why) => {
                tracing::info!(
                    "outside: unmeasured: {why}; budget_device bounds the card's free figure \
                     and the reading at sizing, if any"
                );
                return Ok(());
            }
        };
        if self.held.is_empty() || self.pools.by_refs() {
            tracing::info!("outside: unmeasured: this load seats no slot of its own to fire on");
            return Ok(());
        }
        let token = |i: u32| 100 + (i * 7919) % 5000;
        let prompt: Vec<u32> = (0..32).map(token).collect();
        let decode = [token(1)];
        let mut readings = vec![first];
        for tokens in [&decode[..], &prompt[..]] {
            self.open(0)?;
            let word = self
                .trace()
                .facts
                .word(&poem::Request::new(tokens.len() as u32, false));
            self.fire(&[crate::Lane {
                slot: 0,
                word,
                tokens,
            }])?;
            self.device.synchronize()?;
            match crate::device::own::used_now() {
                Ok(bytes) => readings.push(bytes),
                Err(why) => {
                    self.open(0)?;
                    tracing::info!(
                        "outside: unsettled: the reading after fire {} failed: {why}; nothing is \
                         settled, the reading at sizing stands (readings so far {:?} MiB)",
                        readings.len(),
                        readings.iter().map(|&b| b >> 20).collect::<Vec<_>>(),
                    );
                    return Ok(());
                }
            }
        }
        self.open(0)?;
        let unit = self.pools.map_unit_bytes().max(1);
        let Some(outside) = engine::memory::Outside::settle(&readings, unit) else {
            return Ok(());
        };
        let held = self.pools.settle_outside(outside.used);
        let ledger = self.ledger();
        let ledgered = [Class::Model, Class::State, Class::Fire]
            .iter()
            .map(|class| ledger.bytes(*class))
            .sum::<u64>();
        let mib = |bytes: u64| bytes >> 20;
        tracing::info!(
            "outside: {} MiB this worker holds of the device past the ledger's own {} MiB, \
             {} over {} fires (each step kind once more after arming; readings {:?} MiB, a \
             unit {} MiB); {} MiB of it beside the pool's committed pages, held from here on{}",
            mib(outside.beyond(ledgered)),
            mib(ledgered),
            if outside.settled { "settled" } else { "unsettled, the largest reading taken" },
            outside.fires,
            readings.iter().map(|&b| mib(b)).collect::<Vec<_>>(),
            mib(unit),
            mib(held),
            crate::device::card::budget().map_or_else(String::new, |budget| format!(
                "; the pool is sized under budget_device {} MiB less all this worker holds \
                 and the {} MiB floor",
                mib(budget),
                mib(engine::SAFETY_FLOOR_BYTES)
            )),
        );
        self.outside = Some((outside, ledgered));
        Ok(())
    }

    /// Settles the fit of a load that armed at its starting weight
    /// residency. Its last fit ran with nothing fired: it kept the bodies'
    /// allowance and the activation's whole span out of the cache rows
    /// ([`Staged::finish`]). The lattice has fired now, so the activation
    /// is at its high water and the bodies hold what they take: the rest is
    /// the cache rows', as on a load that armed before its last fit.
    ///
    /// The range stays where it lies (the bodies name it) and the chunks
    /// stated to the runtime stand: what moves is how many of them the fit
    /// holds back ([`Self::held_and_outside`]), which the residency answers
    /// and the runtime lifts its pool's hold-back by. What that opens is
    /// backed when the runtime comes to take it (`Pools::back_whole`),
    /// before the load returns.
    pub(in super::super) fn settle_fit(&mut self) -> Result<()> {
        self.pools.pin();
        // A pool stated as the chunks it backs has nothing held back to
        // lift: its load held the bodies' allowance out of it.
        if !self.pools.stated_whole() {
            tracing::info!(
                "outside: unsettled: this load states its pool as the chunks it backs, so no \
                 last fit follows to size it by a settled reading; the reading at sizing stands"
            );
            return Ok(());
        }
        self.settle_outside()?;
        let resident = footprint(self, self.card.0, self.card.1);
        let before = (self.pools.declared_bytes(), self.pools.kept_back_bytes());
        let fitted = self
            .pools
            .fit_the_card(0, self.held_for_fires, true, &resident)?;
        if super::super::diag::on().arm_trace {
            eprintln!(
                "[arm-trace] settled at the starting weight residency: cache rows {} -> {} MiB \
                 ({} sequences at the declared context) of the {} MiB the card hands out for \
                 them ({resident}); {} -> {} MiB of the stated chunks held back",
                before.0 >> 20,
                self.pools.declared_bytes() >> 20,
                fitted.fit / u64::from(self.pools.paging().pages_per_slot),
                fitted.room >> 20,
                before.1 >> 20,
                self.pools.kept_back_bytes() >> 20,
            );
        }
        Ok(())
    }
}

/// The decode product banks a load's picks read a figure stored under a
/// stale key of (`Routes::behind`, noted by its arming), said: they serve by
/// those figures, a decode body's times, until `pie tune` or a re-export
/// times them again; nothing times them before ready or behind serving.
pub(super) fn say_stale_banks(
    noted: &std::collections::BTreeMap<
        kernels_cuda::linear::routes::Shape,
        kernels_cuda::linear::routes::Wanted,
    >,
) {
    if noted.is_empty() {
        return;
    }
    let banks: Vec<String> = noted
        .keys()
        .map(|s| format!("{:?} {}x{} m{}", s.kind, s.rows, s.k, s.m))
        .collect();
    eprintln!(
        "engine-cuda: rates: `[rates] product_ns`: {} decode product bank(s) served by their \
         figures stored under a stale key (`pie tune` or a re-export refreshes them): {}",
        banks.len(),
        banks.join(", ")
    );
}
