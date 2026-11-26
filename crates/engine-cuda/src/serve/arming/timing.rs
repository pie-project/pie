//! A decode product's kernel timed in the body the lattice armed at its
//! bucket.

#[allow(clippy::wildcard_imports)]
use super::*;

impl Shell {
    /// The decode products `wanted` names, each timed in its real
    /// neighbourhood: for each shape (by rows, ascending), its bucket's
    /// bodies armed with that bank on each of its candidate kernels in turn
    /// (the other buckets' kept armed)
    /// (every bank decided before it as decided), and the decode body of its
    /// rows (that many one-row lanes) replayed [`BODY_ROUNDS`] rounds of
    /// [`BODY_FIRES`] fires under each. The fastest past the rounds'
    /// interquartile spread is kept for the shapes after it; equal within
    /// it, the stated candidate (`profile::routes`). The table the bodies end armed under is laid
    /// on the context. What it measured: each shape under each kernel, the
    /// step's median ns and spread.
    pub(in super::super) fn routes_by_body(
        &mut self,
        base: std::sync::Arc<kernels_cuda::linear::routes::Routes>,
        wanted: &std::collections::BTreeMap<
            kernels_cuda::linear::routes::Shape,
            kernels_cuda::linear::routes::Wanted,
        >,
    ) -> Result<crate::rates::ProductTimes> {
        // A load that decodes nothing chooses no decode product's kernel.
        if self.decoding.iter().next().is_none() {
            return Ok(Vec::new());
        }
        // A shape is timed by the decode body of its rows: one at rows no
        // decode body carries (a prefill's or a mixed fire's) keeps its
        // stated kernel, unmeasured.
        let targets = self
            .lattice_targets()
            .map_or_else(Vec::new, |lattice| lattice.targets);
        let timable = |m: u32| {
            timing_target(
                targets
                    .iter()
                    .map(|(bucket, target, treatment)| (*bucket, target, treatment.rs)),
                m,
            )
            .is_some()
        };
        let mut shapes: Vec<_> = wanted.keys().copied().filter(|s| timable(s.m)).collect();
        shapes.sort_by_key(|s| (s.m, s.rows, s.k));
        let mut decided = base;
        let mut measured = Vec::new();
        for shape in shapes {
            let mut times = Vec::new();
            for &kernel in &wanted[&shape].candidates {
                self.device
                    .set_routes(std::sync::Arc::new(decided.forcing(shape, kernel)));
                // A product's rows in a body are its bucket's: only that
                // bucket's bodies change with its kernel.
                self.rearm_bucket(shape.m)?;
                let (ns, spread) = self.time_decode_body(shape.m)?;
                times.push((kernel, ns, spread));
            }
            let entries: Vec<_> = times
                .iter()
                .map(|&(kernel, ns, spread)| (shape, kernel, ns, spread))
                .collect();
            decided = std::sync::Arc::new(decided.with(&entries));
            measured.extend(entries);
        }
        self.device.set_routes(decided);
        let mut buckets: Vec<u32> = wanted.keys().map(|s| s.m).filter(|&m| timable(m)).collect();
        buckets.dedup();
        for bucket in buckets {
            self.rearm_bucket(bucket)?;
        }
        Ok(measured)
    }

    /// The body the lattice armed at bucket `m` that carries `m` rows
    /// ([`timing_target`]: the decode step's `m` one-row lanes where it
    /// armed one), replayed: its median ns a fire over [`BODY_ROUNDS`]
    /// rounds and their interquartile spread. Every timed fire must replay
    /// the body (none walks eagerly).
    pub(super) fn time_decode_body(&mut self, m: u32) -> Result<(f64, f64)> {
        let targets = self
            .lattice_targets()
            .map_or_else(Vec::new, |lattice| lattice.targets);
        let (target, treatment) = timing_target(
            targets
                .iter()
                .map(|(bucket, target, treatment)| (*bucket, target, treatment.rs)),
            m,
        )
        .ok_or_else(|| {
            Fault::Card(format!(
                "timing a product of {m} rows: no body the lattice arms at bucket {m} \
                 carries them"
            ))
        })?;
        self.arm_rs = treatment;
        let (lanes, media) = target.lanes();
        let owned = self.synthetic_lanes_with(&lanes, &media);
        for _ in 0..BODY_FIRES {
            self.fire_synthetic(&owned)?;
            self.device.synchronize()?;
        }
        let before = self.cache.body_stats().tally.hits;
        let mut rounds = Vec::with_capacity(BODY_ROUNDS);
        for _ in 0..BODY_ROUNDS {
            let began = std::time::Instant::now();
            for _ in 0..BODY_FIRES {
                self.fire_synthetic(&owned)?;
                self.device.synchronize()?;
            }
            rounds.push(began.elapsed().as_secs_f64() * 1e9 / BODY_FIRES as f64);
        }
        let replayed = self.cache.body_stats().tally.hits - before;
        if replayed < (BODY_ROUNDS * BODY_FIRES) as u64 {
            return Err(Fault::Card(format!(
                "timing the decode body of {m} lanes: {replayed} of {} fires replayed it; a \
                 product's kernel is chosen by its body's time, never an eager walk's",
                BODY_ROUNDS * BODY_FIRES
            )));
        }
        rounds.sort_by(f64::total_cmp);
        Ok((
            rounds[BODY_ROUNDS / 2],
            rounds[3 * BODY_ROUNDS / 4] - rounds[BODY_ROUNDS / 4],
        ))
    }
}

/// The composition a product of `m` rows is timed in
/// (`Shell::routes_by_body`), with what rides it (its recurrent rows'
/// treatment): a body the lattice armed at bucket `m` that carries `m`
/// rows, the decode step's (`m` one-row lanes) where the lattice armed one,
/// else the first that does (where a fire carries fewer lanes than `m`, an
/// artifact baked for fewer, its rows are a prompt's). `None` where no
/// armed body carries them.
pub(super) fn timing_target<'a, R: Copy>(
    targets: impl IntoIterator<Item = (u32, &'a BodySynth, R)>,
    m: u32,
) -> Option<(&'a BodySynth, R)> {
    let rows = |target: &BodySynth| target.lanes().0.iter().map(|&(_, rows)| rows).sum::<u32>();
    let at: Vec<(&BodySynth, R)> = targets
        .into_iter()
        .filter(|&(bucket, target, _)| bucket == m && rows(target) == m)
        .map(|(_, target, ride)| (target, ride))
        .collect();
    at.iter()
        .copied()
        .find(|(target, _)| matches!(target, BodySynth::Decode { .. }))
        .or_else(|| at.first().copied())
}

/// Rounds a decode body is timed in (`Shell::routes_by_body`): the median
/// kept, the interquartile range the spread a choice must clear. A rates
/// step's on CUDA; ROCm's own.
pub(super) const BODY_ROUNDS: usize = if cfg!(feature = "rocm") {
    9
} else {
    engine::rates::ROUNDS
};

/// Fires a round of a decode body's timing: each replayed and waited on, as
/// a serving step is.
pub(super) const BODY_FIRES: usize = if cfg!(feature = "rocm") { 20 } else { 10 };
