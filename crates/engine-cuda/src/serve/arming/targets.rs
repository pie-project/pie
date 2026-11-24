//! What the arm walk arms: the load's deployment, the compositions each
//! stated step's buckets cross with the engine's classes, and the order they
//! are captured in.

#[allow(clippy::wildcard_imports)]
use super::*;

pub(super) struct Deployment {
    /// The most lanes a decode step at any bucket carries.
    pub(super) ceiling: u32,
    /// The widest bucket a step of decoding lanes alone composes at: its
    /// lanes' rows padded.
    pub(super) decode_top: u32,
    pub(super) patch_points: Vec<u32>,
    pub(super) decoders: Vec<usize>,
    pub(super) prefilling: Vec<usize>,
    /// The classes keyed only for a speculative decoder's fires (plain but
    /// for reading the drafts): armed alone, never beside another class.
    pub(super) drafting: Vec<usize>,
    /// The classes past the plain tiers a served program fires (a switch):
    /// keyed by the bucket alone, beside any class.
    pub(super) switched: Vec<usize>,
    pub(super) media: Vec<usize>,
    pub(super) wide: Vec<usize>,
    pub(super) decoding: poem_ir::ClassSet,
    pub(super) seats: u32,
    pub(super) context: u32,
    pub(super) max_lanes: u32,
    /// `engine.prefill_lanes`: the most lanes a fire of prompts alone pads to.
    pub(super) prefill_lanes: u32,
    pub(super) patch_fold: u32,
    /// Whether the load is an expert-parallel rank's, which fires a step
    /// with no lanes of its own at every bucket its group fires.
    pub(super) idle: bool,
}

#[derive(Default)]
pub(super) struct Targets {
    pub(super) targets: Vec<(u32, BodySynth)>,
    pub(super) unfireable: Vec<String>,
}

pub(super) fn unfireable_line(deployment: &Deployment, what: &str, at: &str, note: Option<&str>) -> String {
    let (note, comma) = match note {
        Some(note) => (note, ", "),
        None => ("", ""),
    };
    format!(
        "{what} at {at} ({note}{comma}{} seat(s) x {} context, {} lane(s))",
        deployment.seats, deployment.context, deployment.max_lanes,
    )
}

pub(super) fn decode_keys(deployment: &Deployment, bucket: u32, into: &mut Targets) {
    if bucket > deployment.decode_top {
        return;
    }
    let lanes = bucket.min(deployment.ceiling);
    for class in deployment.decoders.iter().copied() {
        into.targets.push((bucket, BodySynth::Decode { lanes, class }));
    }
}

pub(super) fn prefill_keys(deployment: &Deployment, point: u32, into: &mut Targets) {
    for class in deployment.prefilling.iter().copied() {
        // A drafting class fires its rows on one lane (a block draft, a
        // verify): its body is recorded for that schedule.
        let spread = if deployment.drafting.contains(&class) {
            (point <= deployment.context).then(|| vec![point])
        } else {
            Shell::spread(
                point,
                deployment.seats.min(match deployment.prefill_lanes {
                    0 => deployment.max_lanes,
                    cap => deployment.max_lanes.min(cap),
                }),
                deployment.context,
            )
        };
        match spread {
            Some(rows) => into
                .targets
                .push((point, BodySynth::Prefill { class, rows })),
            None => into.unfireable.push(unfireable_line(
                deployment,
                &format!("prefill c{class}"),
                &format!("bucket {point}"),
                None,
            )),
        }
    }
}

pub(super) fn mixed_keys(deployment: &Deployment, point: u32, into: &mut Targets) {
    let alone = |class: &usize| !deployment.drafting.contains(class);
    for decode in deployment.decoders.iter().copied().filter(alone) {
        for class in deployment.prefilling.iter().copied().filter(alone) {
            let rows = (point >= 2 && deployment.seats >= 2)
                .then(|| {
                    Shell::spread(
                        point - 1,
                        (deployment.seats - 1).min(deployment.max_lanes.saturating_sub(1)),
                        deployment.context,
                    )
                })
                .flatten();
            match rows {
                Some(rows) => into.targets.push((
                    point,
                    BodySynth::Mixed {
                        decode,
                        class,
                        rows,
                    },
                )),
                None => into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("mixed c{decode}+c{class}"),
                    &format!("bucket {point}"),
                    None,
                )),
            }
        }
    }
}

pub(super) fn tower_keys(deployment: &Deployment, point: u32, into: &mut Targets) {
    for patches in deployment.patch_points.iter().copied() {
        for class in deployment.media.iter().copied() {
            let fold = deployment.patch_fold.max(1);
            let owed = patches.div_ceil(fold).max(1);
            if owed > point || point > deployment.context || deployment.seats == 0 {
                into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("tower c{class}"),
                    &format!("bucket {point} + patch rung {patches}"),
                    Some(&format!("{owed} placeholder row(s) owed")),
                ));
                continue;
            }
            into.targets.push((
                point,
                BodySynth::Tower {
                    class,
                    rows: point,
                    images: 1,
                    patches,
                },
            ));
        }
    }
}

pub(super) fn wide_keys(deployment: &Deployment, point: u32, into: &mut Targets) {
    let plain: Vec<usize> = deployment
        .wide
        .iter()
        .copied()
        .filter(|class| !deployment.drafting.contains(class))
        .collect();
    match wide_lanes(deployment, point, &plain) {
        Some(lanes) => into.targets.push((point, BodySynth::Wide { lanes })),
        None => into.unfireable.push(unfireable_line(
            deployment,
            "wide",
            &format!("bucket {point}"),
            None,
        )),
    }
    // Every class a fire may run at once, a switched class among them: a
    // fire with a class past the plain tiers keys by its bucket alone
    // (`record::Ladder::of`), whichever classes it mixes, so one such
    // composition a bucket records it.
    if !deployment.switched.is_empty() {
        let every: Vec<usize> = plain
            .iter()
            .copied()
            .chain(deployment.switched.iter().copied())
            .collect();
        match wide_lanes(deployment, point, &every) {
            Some(lanes) => into.targets.push((point, BodySynth::Wide { lanes })),
            None => into.unfireable.push(unfireable_line(
                deployment,
                "wide with a switched class",
                &format!("bucket {point}"),
                None,
            )),
        }
    }
}

/// A wide composition of `classes` at bucket `point`, as many lanes as the
/// bucket seats (a body's launch grids hold the lanes it recorded, no more):
/// a lane of each class, the rest of the seats each prompting class's in
/// turn (each decoding class's where none prompts); a decoding lane takes a
/// row and the bucket's other rows spread over the prompting lanes. `None`
/// where fewer than two lanes result.
pub(super) fn wide_lanes(deployment: &Deployment, point: u32, classes: &[usize]) -> Option<Vec<LaneRows>> {
    let seats = deployment.seats.min(deployment.max_lanes).min(point) as usize;
    let decodes = |class: &usize| deployment.decoding.contains(*class);
    let mut lanes: Vec<usize> = classes.iter().copied().take(seats).collect();
    let prompting: Vec<usize> = lanes.iter().copied().filter(|c| !decodes(c)).collect();
    let fill = if prompting.is_empty() { lanes.clone() } else { prompting };
    while lanes.len() < seats && !fill.is_empty() {
        lanes.push(fill[(lanes.len() - classes.len().min(seats)) % fill.len()]);
    }
    if lanes.len() < 2 {
        return None;
    }
    let decoders = lanes.iter().filter(|class| decodes(class)).count() as u32;
    let prompts = lanes.len() as u32 - decoders;
    let mut spread = if prompts == 0 {
        Vec::new()
    } else {
        Shell::spread(point.checked_sub(decoders)?, prompts, deployment.context)?
    }
    .into_iter();
    lanes
        .into_iter()
        .map(|class| {
            Some(if decodes(&class) {
                (class, 1)
            } else {
                (class, spread.next()?)
            })
        })
        .collect()
}

/// An expert-parallel rank idles through its peers' steps at every bucket
/// (`Shell::fire_idle`): armed like any composition, an idle step replays
/// what a busy one's peers replay, and never walks eagerly beside them.
pub(super) fn idle_keys(deployment: &Deployment, point: u32, into: &mut Targets) {
    if deployment.idle {
        into.targets.push((point, BodySynth::Idle));
    }
}

/// The compositions a step at a bucket is recorded by: the engine's classes
/// crossed with it, whatever the stated step's kind. A body's key is its
/// bucket, the classes its lanes are of and their treatment
/// (`record::BodyKey`), never how many rows a lane takes, so a step of any
/// kind keys one of these: each decoding class's lanes alone, each
/// prompting class's (a one-row piece of a prompt among them; an image's
/// tower too), a decoding class's beside a prompting class's, and every
/// class at once (a wide composition). An expert-parallel rank's idle step
/// rides each bucket. Crossed with every kind, not the stated step's: a key
/// names the classes present (`record::Ladder`), which a `StepKind` does not,
/// so a stated step of one kind forms keys of another's composition (a
/// prefill step's lanes of several classes keys as the wide one).
pub(super) const ARMS: [fn(&Deployment, u32, &mut Targets); Kind::COUNT] = [
    decode_keys,
    prefill_keys,
    mixed_keys,
    tower_keys,
    wide_keys,
    idle_keys,
];

impl core::fmt::Display for BodySynth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodySynth::Decode { lanes, class } => write!(f, "decode c{class} x{lanes}"),
            BodySynth::Prefill { class, rows } => {
                write!(f, "prefill c{class} {rows:?}")
            }
            BodySynth::Mixed {
                decode,
                class,
                rows,
            } => {
                write!(f, "mixed c{decode}+c{class} {rows:?}")
            }
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => write!(f, "tower c{class} {rows}r x{images} img {patches}p"),
            BodySynth::Wide { lanes } => {
                write!(f, "wide ")?;
                for (at, (class, rows)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class}:{rows}")?;
                }
                Ok(())
            }
            BodySynth::Idle => f.write_str("idle"),
        }
    }
}

impl Shell {
    /// Every composition the lattice arms, in arming's order (the top
    /// bucket's first, then by kind and bucket), each with its recurrent
    /// rows' treatment and the stated step it records; `None` where there
    /// is nothing to arm.
    ///
    /// The compositions are the load's stated step set's (`Shell::steps`:
    /// every step the runtime may stamp), walked in its order: each shape at each bucket its
    /// rows pad to, by the engine's classes crossed with it ([`ARMS`]),
    /// recorded for its treatment ([`Shell::rs_bodies`]). A bucket an
    /// earlier shape of the same treatment walked is not walked again.
    pub(super) fn lattice_targets(&self) -> Option<LatticeTargets> {
        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return None;
        }
        let seats = self.held.len() as u32;
        let context = if self.page_tables == 0 {
            self.budget.max_tokens
        } else {
            self.pools.paging().context()
        };
        let max_lanes = self.fire_lanes();
        let classes = self.compiled.classes.classes.len();
        let textual = |class: usize| !self.media.contains(class);
        // A class past the plain tiers whose fires run a kernel arm of their
        // own (a switch: dflash's block draft) but no adapter, no score
        // capture and no media: a served program fires it, so the lattice
        // captures it at load, keyed as its fire keys (by its bucket alone).
        let switched = |class: usize| {
            !self.tiers.plain.contains(class)
                && !self.landing[class].is_empty()
                && self.landing[class].iter().any(|request| {
                    !request.has_adapter() && !request.captures_scores() && !request.has_media()
                })
        };
        let staged = self
            .towered
            .then_some(self.patch_seat)
            .flatten()
            .is_some_and(|seat| seat.row_bytes > 0);
        let patch_points: Vec<u32> = match (staged, self.budgets.patches.as_ref()) {
            (true, Some(ladder)) => ladder.buckets.clone(),
            _ => Vec::new(),
        };
        let deployment = Deployment {
            ceiling,
            decode_top: self.bucket_for(ceiling).unwrap_or(ceiling),
            patch_points,
            decoders: self
                .decoding
                .iter()
                .filter(|class| {
                    textual(*class) && (self.tiers.plain.contains(*class) || switched(*class))
                })
                .collect(),
            prefilling: (0..classes)
                .filter(|class| {
                    !self.decoding.contains(*class)
                        && textual(*class)
                        && (self.tiers.plain.contains(*class) || switched(*class))
                })
                .collect(),
            media: self
                .media
                .iter()
                .filter(|class| self.tiers.plain.contains(*class))
                .collect(),
            wide: self
                .decoding
                .iter()
                .chain((0..classes).filter(|class| !self.decoding.contains(*class)))
                .filter(|class| {
                    textual(*class)
                        && !self.landing[*class].is_empty()
                        && self.tiers.plain.contains(*class)
                })
                .collect(),
            switched: (0..classes)
                .filter(|class| textual(*class) && switched(*class))
                .collect(),
            drafting: (0..classes)
                .filter(|class| {
                    !self.landing[*class]
                        .iter()
                        .any(crate::exports::plain_or_adapted)
                })
                .collect(),
            decoding: self.decoding.clone(),
            seats,
            context,
            max_lanes,
            prefill_lanes: self.prefill_lanes,
            patch_fold: self.patch_fold,
            idle: self.ep.is_some(),
        };
        let mut walked = std::collections::HashSet::new();
        let mut targets: Vec<(u32, BodySynth, Treatment)> = Vec::new();
        let mut unfireable = Vec::new();
        for (at, shape) in self.steps.shapes.iter().enumerate() {
            for bucket in self.buckets_of(shape) {
                if !walked.insert((shape.treatment, bucket)) {
                    continue;
                }
                let mut found = Targets::default();
                for enumerate in ARMS {
                    enumerate(&deployment, bucket, &mut found);
                }
                unfireable.extend(found.unfireable);
                for (bucket, target) in found.targets {
                    for rs in self.rs_bodies(&target, shape.treatment) {
                        targets.push((bucket, target.clone(), Treatment { rs, shape: at }));
                    }
                }
            }
        }
        unfireable.sort();
        unfireable.dedup();
        let top = targets.iter().map(|(bucket, ..)| *bucket).max();
        // Captured in this order (the top bucket's compositions first, then
        // by kind and bucket; a bucket's folding body before those that
        // buffer, as the set orders them): the order bodies are captured in
        // leaves process-wide state behind that every later graph's
        // kernel-to-kernel transitions pay for (an L40S decode at one lane
        // ran 3.4% slower after a capture across kinds by rows), so it is
        // kept apart from which compositions the allowance keeps: that is
        // the set's order (`Shell::rule_rank`), applied by giving back the
        // worst-ranked capture whenever the bodies pass their allowance
        // (`arm_target`).
        let rank = |kind: Kind| match kind {
            Kind::Decode => 0u8,
            Kind::Mixed => 1,
            Kind::Prefill => 2,
            Kind::Wide => 3,
            Kind::Tower => 4,
            Kind::Idle => 5,
        };
        targets.sort_by_key(|(bucket, target, _)| {
            (Some(*bucket) != top, rank(target.kind()), *bucket)
        });
        if targets.is_empty() {
            return None;
        }
        Some(LatticeTargets {
            targets,
            top,
            unfireable,
        })
    }

    /// The buckets a stated shape's steps compose at: each its rows pad to
    /// (`Shell::bucket_for`), its rows themselves on a load with no buckets;
    /// a buffering step with a prompt's rows at its rung's alone
    /// ([`Shell::buffered_rung`]).
    pub(super) fn buckets_of(&self, shape: &engine::StepShape) -> Vec<u32> {
        if shape.treatment == engine::Treatment::Buffers && shape.kind != engine::StepKind::Decode
        {
            return self.bucket_for(shape.rung).into_iter().collect();
        }
        let buckets = &self.budget.buckets;
        if buckets.is_empty() {
            return (shape.rows.lo..=shape.rows.hi).collect();
        }
        let top = buckets.last().copied().unwrap_or(shape.rows.hi);
        let padded = |rows: u32| self.bucket_for(rows).unwrap_or(top);
        let (from, to) = (padded(shape.rows.lo), padded(shape.rows.hi));
        buckets
            .iter()
            .copied()
            .filter(|&bucket| from <= bucket && bucket <= to)
            .collect()
    }

    /// The bodies a composition is recorded as for a step of `treatment`
    /// (`record::RsBody`, the key a serving fire's lanes pick): a folding
    /// step's one; a buffering step's, one where its lanes buffer rows, one
    /// more where a decoding lane's carry may run past a chunk (a beam
    /// lane's: split over the device's spare blocks), and one more where a
    /// prompting lane may replay buffered rows ahead of its own (a decoding
    /// lane replays its carried rows in its decode step's own kernels,
    /// `dispatch::ssm`). A tower's and an idle step's buffer none.
    pub(super) fn rs_bodies(&self, target: &BodySynth, treatment: engine::Treatment) -> Vec<record::RsBody> {
        if treatment == engine::Treatment::Folds {
            return vec![record::RsBody::Folds];
        }
        if matches!(target, BodySynth::Tower { .. } | BodySynth::Idle) {
            return Vec::new();
        }
        let (lanes, _) = target.lanes();
        let decodes = lanes
            .iter()
            .any(|&(class, _)| self.decoding.contains(class));
        let prompts = lanes
            .iter()
            .any(|&(class, _)| !self.decoding.contains(class));
        let splits = self
            .buffers
            .as_ref()
            .is_some_and(|buffers| buffers.split_room() != 0);
        let mut bodies = vec![record::RsBody::Buffers];
        if decodes && splits {
            bodies.push(record::RsBody::Splits);
        }
        if prompts {
            bodies.push(record::RsBody::Replays);
        }
        bodies
    }

    /// The step compositions this load's builder forms,
    /// over `rules` and the load's facts as they stand before arming: what
    /// the arm walk takes its targets from. The rungs are the arena's
    /// (every one a fire can carve, up to one fire's tokens).
    pub(crate) fn walk_steps(&mut self, rules: &engine::StepRules) {
        let max_tokens = self.budget.max_tokens;
        let mut rungs: Vec<u32> = self
            .arena
            .rungs(self.compiled.arena_forked.as_ref())
            .into_iter()
            .map(|(tokens, _)| tokens)
            .filter(|&tokens| tokens < max_tokens)
            .collect();
        rungs.push(max_tokens);
        let state_rows = self
            .trace
            .caches
            .iter()
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
        let facts = engine::StepFacts {
            rungs,
            max_lanes: self.fire_lanes(),
            max_tokens,
            max_readouts: crate::api::readout_facts().fire_rows(
                u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budget))
                    .unwrap_or(u32::MAX),
            ),
            buffers: self.buffers.is_some(),
            attachments: engine::Attachments::served(
                true,
                !self.inputs().is_empty(),
                state_rows && self.lands_snapshots(),
            ),
            // (CUDA lays every mixed step within its limits)
            mixed: engine::MixedLanes::Any,
        };
        self.steps = engine::StepShapes::of(rules, &facts);
    }

    /// The bucket a buffering step with a prompt's rows (a mixed or prompt
    /// step whose lanes buffer or replay recurrent rows) of `rows` rows
    /// composes at: the smallest buffered rung in the walked set holding
    /// them (`engine::StepFacts::buffered_rungs`, coarser than the folded
    /// lattice), which the runtime asks the step's activation rung for
    /// (`FireRequest::pads_to`). Its bodies are recorded there alone, not
    /// at every finer bucket: they are many (each class set, each of
    /// buffering, splitting and replaying), and a prompt's are the widest.
    /// A decode step's buffering lanes keep their own bucket.
    pub(in super::super) fn buffered_rung(&self, rows: u32) -> Option<u32> {
        self.steps
            .shapes
            .iter()
            .filter(|shape| shape.treatment == engine::Treatment::Buffers)
            .map(|shape| shape.rung)
            .find(|&rung| rung >= rows)
            .and_then(|rung| self.bucket_for(rung))
    }

    /// The bucket a step stamped `shape` composes at where its shape names
    /// one: a buffering mixed or prompt step's rung ([`Shell::buffered_rung`]);
    /// `None` for any other (it composes at its rows' own bucket).
    pub(in super::super) fn stamped_bucket(&self, shape: engine::FireShape) -> Option<u32> {
        let stated = self.steps.shapes.get(shape.id as usize)?;
        (stated.treatment == engine::Treatment::Buffers && stated.kind != engine::StepKind::Decode)
            .then(|| self.bucket_for(stated.rung))
            .flatten()
    }

    /// The set the arm walk takes its targets from ([`Shell::walk_steps`]).
    pub(crate) fn steps(&self) -> &engine::StepShapes {
        &self.steps
    }

    /// The set the load states (`shapes`, from its stated facts) with
    /// those of it its arm walk left eager: each whose steps compose at a
    /// bucket where the last whole arming recorded no body of its treatment
    /// for some composition it walked, or walked none. Each is named.
    pub(crate) fn stated_steps(&self, shapes: engine::StepShapes) -> engine::StatedSteps {
        let mut eager = Vec::new();
        if self.arm_deferred {
            tracing::info!(
                "the arm walk runs at the starting weight residency: what it \
                 leaves eager is named then"
            );
            return engine::StatedSteps { shapes, eager };
        }
        for (at, shape) in shapes.shapes.iter().enumerate() {
            let missing: Vec<u32> = self
                .buckets_of(shape)
                .into_iter()
                .filter(|&bucket| {
                    self.walked.get(&(shape.treatment, bucket)) != Some(&true)
                })
                .collect();
            if !missing.is_empty() {
                tracing::info!(
                    "step shape {at} [{shape}] runs eager: no body recorded at \
                     bucket(s) {missing:?}"
                );
                eager.push(u32::try_from(at).unwrap_or(u32::MAX));
            }
        }
        engine::StatedSteps { shapes, eager }
    }
}

/// What [`Shell::lattice_targets`] enumerates: each composition with its
/// bucket and recurrent rows' treatment, the top bucket, and the keys no
/// synthetic fire reaches.
pub(super) struct LatticeTargets {
    pub(super) targets: Vec<(u32, BodySynth, Treatment)>,
    pub(super) top: Option<u32>,
    pub(super) unfireable: Vec<String>,
}
