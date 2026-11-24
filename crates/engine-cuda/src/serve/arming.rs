mod targets;
use self::targets::*;
mod synthetic;
mod golden;
mod timing;
use engine::fire::{Mask, Masking, RsReset, RsVerb};

use crate::error::{Fault, Result};
use crate::record;

use super::{Lane, MROPE_COORDS, Media, PATCH_ROUTE_DROP, Seated, Shell};

struct Ballot {
    slip: Option<crate::device::Buffer>,
}

impl Ballot {
    fn open(device: &crate::device::Context) -> Ballot {
        let grouped = device.ctx().comm("arming ballot").is_ok();
        Ballot {
            slip: grouped
                .then(|| crate::device::Buffer::zeroed(4).ok())
                .flatten(),
        }
    }

    fn any(&mut self, device: &crate::device::Context, mine: bool) -> bool {
        let Some(slip) = self.slip.as_mut() else {
            return mine;
        };
        let vote: f32 = if mine { 1.0 } else { 0.0 };
        let sum = (|| -> Result<f32> {
            slip.write(0, &vote.to_le_bytes())?;
            let mut wire = kernels_cuda::Tensor::new(slip.ptr(), 1, 1, poem_ir::Dtype::F32);
            kernels_cuda::collective::all_reduce(device.ctx(), &mut wire)
                .map_err(crate::error::kernel)?;
            device.synchronize()?;
            let mut back = [0u8; 4];
            slip.read(0, &mut back)?;
            Ok(f32::from_le_bytes(back))
        })();
        match sum {
            Ok(total) => total >= 0.5,
            Err(_) => {
                self.slip = None;
                mine
            }
        }
    }
}

#[derive(Clone)]
struct Synthetic {
    word: u64,
    tokens: Vec<u32>,
    mask: Option<Masking>,
    adapter: Option<u32>,
    slot: u32,
    pages: Vec<u32>,
    window: Vec<u32>,
    held: Option<u32>,
    media: Option<SyntheticMedia>,
    voxels: Option<SyntheticVoxels>,
}

/// A synthetic lane's voxel input, where its class reads one: a clip of one
/// token patch and a zero payload at the input's width, so a composition
/// over a voxel-reading class fires (and records) before ready like any
/// other. What the clips fill is never read back.
#[derive(Clone)]
struct SyntheticVoxels {
    clips: Vec<[u32; 3]>,
    payload: Vec<u8>,
}

#[derive(Clone)]
struct SyntheticMedia {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    grids: Vec<i32>,
}

/// A synthetic lane's class and row count.
type LaneRows = (usize, u32);
/// What a body is armed for beyond its composition: how its lanes treat
/// their recurrent rows (a lane's attention mask, a group's too, is data
/// one body reads: no treatment), and the stated step shape it records (its
/// index in `Shell::steps`, the set's order: what the allowance keeps by).
#[derive(Debug, Clone, Copy)]
struct Treatment {
    rs: record::RsBody,
    shape: usize,
}

/// Where the allowance ranks a composition (`Shell::rule_rank`), the least
/// kept first.
type Rank = (usize, u32, core::cmp::Reverse<Option<u64>>, core::cmp::Reverse<usize>, u8);

/// A composition captured during arming and held: its rank by the user's
/// rule, its key, where it came from, and what it weighed.
struct Held {
    rank: Rank,
    key: record::BodyKey,
    bucket: u32,
    target: BodySynth,
    treatment: Treatment,
    weight: u64,
    at: usize,
}

/// A lattice being armed: the targets left, and what arming has counted.
pub(super) struct Arming {
    targets: std::collections::VecDeque<(u32, BodySynth, Treatment)>,
    /// Every target the walk began with, for what it recorded of each
    /// stated step (`Shell::walked`).
    planned: Vec<(u32, BodySynth, Treatment)>,
    top: Option<u32>,
    unfireable: Vec<String>,
    verify: bool,
    group: bool,
    records: bool,
    dropped: std::collections::BTreeSet<u32>,
    weighed: [(usize, usize); Kind::COUNT],
    armed: usize,
    wanted: usize,
    tally: [(usize, usize); Kind::COUNT],
    refused: Option<String>,
    never: usize,
    warmed_only: usize,
    never_from: u32,
    belted: bool,
    starved: Option<u64>,
    /// Each composition captured and still held, by the user's rule's rank
    /// (`Shell::rule_rank`): the worst is given back first when the bodies
    /// pass their allowance.
    held: Vec<Held>,
    /// The compositions given back for the allowance: stated eager after
    /// arming, by name.
    given_back: Vec<(u32, BodySynth, Treatment)>,
    /// The bytes of the captures given back: room the pool's spare does not
    /// show again (the driver keeps what it backed), which the next capture
    /// takes.
    freed: u64,
    peer_stopped: bool,
    unit_margin: u64,
    /// The bodies' allowance is a record's measurement
    /// ([`super::measured`]): the lattice arms as recorded, unpriced, and
    /// no headroom estimate stops it.
    from_record: bool,
    /// The allowance caps arming. Not at load on a record's figure: the
    /// pool is not backed yet, so the figure is the fit's reservation, and
    /// the finish fit accounts what the bodies took; only a device failure
    /// stops arming there.
    capped: bool,
    /// What the device's free memory fell by across each composition's
    /// capturing fire, less what the pool backed in that window, summed:
    /// what the captures took, and nothing the pool, serving, the runtime
    /// or a first launch's process-wide loads took around them. None where
    /// free memory cannot be read.
    took: Option<u64>,
    /// The buckets of the profile's lattice, before arming coarsened any
    /// away.
    profiled: usize,
    ballot: Ballot,
    /// When arming began.
    began: std::time::Instant,
}

impl Arming {
    fn narrower(&self, bucket: u32, target: &BodySynth) -> bool {
        Some(bucket) != self.top && target.kind() != Kind::Decode
    }
}

/// A synthetic lane's image and patch counts.
type LaneMedia = (u32, u32);

#[derive(Debug, Clone)]
enum BodySynth {
    Decode {
        lanes: u32,
        class: usize,
    },
    Prefill {
        class: usize,
        rows: Vec<u32>,
    },
    Mixed {
        decode: usize,
        class: usize,
        rows: Vec<u32>,
    },
    Tower {
        class: usize,
        rows: u32,
        images: u32,
        patches: u32,
    },
    Wide {
        lanes: Vec<LaneRows>,
    },
    /// An expert-parallel rank's step with no lanes of its own: only the
    /// dispatches, the experts over what its peers send, and the combines.
    Idle,
}

impl BodySynth {
    fn kind(&self) -> Kind {
        match self {
            BodySynth::Decode { .. } => Kind::Decode,
            BodySynth::Prefill { .. } => Kind::Prefill,
            BodySynth::Mixed { .. } => Kind::Mixed,
            BodySynth::Tower { .. } => Kind::Tower,
            BodySynth::Wide { .. } => Kind::Wide,
            BodySynth::Idle => Kind::Idle,
        }
    }

    fn lanes(&self) -> (Vec<LaneRows>, Vec<LaneMedia>) {
        match self {
            BodySynth::Decode { lanes, class } => {
                (vec![(*class, 1u32); *lanes as usize], Vec::new())
            }
            BodySynth::Prefill { class, rows } => (
                rows.iter().map(|rows| (*class, *rows)).collect(),
                Vec::new(),
            ),
            BodySynth::Mixed {
                decode,
                class,
                rows,
            } => (
                core::iter::once((*decode, 1u32))
                    .chain(rows.iter().map(|rows| (*class, *rows)))
                    .collect(),
                Vec::new(),
            ),
            BodySynth::Wide { lanes } => (lanes.clone(), Vec::new()),
            BodySynth::Idle => (Vec::new(), Vec::new()),
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => (vec![(*class, *rows)], vec![(*images, *patches)]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Decode,
    Prefill,
    Mixed,
    Tower,
    Wide,
    Idle,
}

impl Kind {
    pub const COUNT: usize = 6;

    pub const ALL: [Kind; Kind::COUNT] = [
        Kind::Decode,
        Kind::Prefill,
        Kind::Mixed,
        Kind::Tower,
        Kind::Wide,
        Kind::Idle,
    ];

    pub fn at(self) -> usize {
        self as usize
    }
}

impl core::fmt::Display for Kind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Kind::Decode => "decode",
            Kind::Prefill => "prefill",
            Kind::Mixed => "mixed",
            Kind::Tower => "tower",
            Kind::Wide => "wide",
            Kind::Idle => "idle",
        })
    }
}

impl Shell {
    /// Forgets the recorded bodies and arms the lattice again within the
    /// same load, allowed what its last arming took: the pool is backed by
    /// now and those bodies' memory is what the forgetting frees, whatever
    /// the record said. Capped by that figure.
    pub(super) fn rearm_bodies(&mut self) -> Result<()> {
        self.cache.forget_bodies();
        self.bodies_mem = self
            .bodies_mem
            .max(usize::try_from(self.bodies_took).unwrap_or(usize::MAX));
        // A measured allowance, as a record's is: the estimates a load that
        // knows nothing prices by are not consulted (on a flat range backed
        // whole they read no spare at all).
        self.rearming = true;
        let armed = self.arm_bodies();
        self.rearming = false;
        armed
    }

    /// Bucket `bucket`'s bodies forgotten and armed again, the other
    /// buckets' kept: a re-arm allowed what the load's own arming took.
    fn rearm_bucket(&mut self, bucket: u32) -> Result<()> {
        self.cache.forget_bodies_at(bucket);
        self.arm_only = Some(bucket);
        self.rearming = true;
        let armed = self.arm_bodies();
        self.rearming = false;
        self.arm_only = None;
        armed
    }

    pub(super) fn arm_bodies(&mut self) -> Result<()> {
        // A beam group's folds launch ahead of a fire's body, eagerly, and
        // no arming fire carries one: their kernels are made here.
        if self.buffers.is_some() {
            for fold in crate::dispatch::ssm::group_folds(&self.trace) {
                fold.load().map_err(|error| Fault::from(crate::error::kernel(error)))?;
            }
        }
        let armed = self.arm_lattice(false);
        self.pools.forget_weight_uses(self.device.stream())?;
        armed
    }

    /// Arms the lattice, an expert-parallel rank with its collectives
    /// fenced (`EpRank::fence`): every rank fires the same synthetic steps,
    /// and whatever they first use (a module, a tune, a pinned slot, a
    /// graph) is made with no collective of the group spinning on the
    /// device, so none of it waits on a peer that waits on it. What the
    /// steps warm is warm for the fires that serve after, unfenced.
    fn arm_lattice(&mut self, verify: bool) -> Result<()> {
        if let Some(ep) = self.ep.as_ref() {
            ep.fenced(true);
        }
        let armed = self.arm_lattice_fenced(verify);
        if let Some(ep) = self.ep.as_ref() {
            ep.fenced(false);
            // Sealed however arming went (nothing to arm, nothing armed):
            // past arming the group's ranks fire different steps at once,
            // and a capture's instantiation takes the context beside a
            // peer's spinning collective. What is not armed walks eagerly.
            if !verify {
                self.cache.seal_bodies();
            }
        }
        armed
    }

    /// The scratch slabs this load's fires take: each composition of the
    /// lattice fired once, size-only (`crate::sizing`), in arming's order.
    /// Nothing launches, copies or commits; the slabs are laid at what the
    /// walk asked before any fire runs, and none is allocated later.
    pub(super) fn size_arming(&mut self) -> Result<Option<crate::sizing::Footprint>> {
        let targets = self
            .lattice_targets()
            .map_or_else(Vec::new, |lattice| lattice.targets);
        if targets.is_empty() && self.voxels.is_none() {
            return Ok(None);
        }
        crate::sizing::begin();
        let mut walked = Ok(());
        // Where the load seats adapters, each composition is walked once
        // more with every lane it can adapt adapted: the lattice names the
        // plain classes, and a lane with an adapter runs kernels they do
        // not (the correction's waist, a slab a lane), which a fire must
        // find laid like any other.
        let adapts = self.adapters().slots().seats() > 0;
        'targets: for (bucket, target, treatment) in &targets {
            self.arm_rs = treatment.rs;
            let (lanes, media) = target.lanes();
            let adapted: Vec<LaneRows> = lanes
                .iter()
                .map(|&(class, rows)| (self.adapted_twin(class).unwrap_or(class), rows))
                .collect();
            let walks = if adapts && adapted != lanes {
                vec![lanes, adapted]
            } else {
                vec![lanes]
            };
            let launched = kernels_cuda::sizing::launches();
            for lanes in walks {
                let owned = self.synthetic_lanes_with(&lanes, &media);
                if matches!(target, BodySynth::Idle) {
                    self.at_bucket(*bucket);
                }
                // Landed as arming lands each fire: its staging slots come
                // back.
                if let Err(why) = self
                    .fire_synthetic(&owned)
                    .and_then(|()| self.device.synchronize())
                {
                    walked = Err(Fault::Unbound {
                        what: format!("the size-only walk of bucket {bucket}, {target}: {why}"),
                    });
                    break 'targets;
                }
            }
            let launches = kernels_cuda::sizing::launches() - launched;
            let entry = self.walked_launches.entry((*bucket, target.to_string())).or_insert(0);
            *entry = (*entry).max(launches);
        }
        // The voxel walk is a plain fire: no recurrent treatment.
        self.arm_rs = record::RsBody::Folds;
        if walked.is_ok() {
            walked = self.size_voxels();
        }
        let footprint = crate::sizing::end();
        walked.map(|()| Some(footprint))
    }

    /// The size-only walk of a voxel fire at the ladder's top: the lattice
    /// names token compositions alone, and a fire over voxels runs kernels
    /// none of them does (a conv decoder's group norm). Each class is walked
    /// at the most voxel rows whose widest value the seat holds, in the
    /// seat's every clip, a row wide: the most rows and the most clips a
    /// fire states, at the boxes that pad most.
    fn size_voxels(&mut self) -> Result<()> {
        let Some(seat) = self.voxels.as_ref().map(crate::voxels::Store::seat) else {
            return Ok(());
        };
        let row_bytes = seat.payload_bytes() / seat.rows.max(1);
        for class in 0..self.landing.len() {
            if self.landing[class].is_empty() {
                continue;
            }
            let Some((boxes, rows)) = self
                .seat_patches(class)
                .and_then(|(patches, clips)| self.clips_of(patches, clips))
            else {
                continue;
            };
            let payload = vec![0u8; (u64::from(rows) * row_bytes) as usize];
            let owned = self.synthetic_lanes_with(&[(class, 1)], &[]);
            let seated = Self::seated(&owned);
            let shot = [crate::voxels::Clips {
                lane: 0,
                clips: &boxes,
                payload: &payload,
            }];
            self.arming = true;
            let fired = self
                .fire_voxels(&seated, &shot)
                .and_then(|_| self.device.synchronize());
            self.arming = false;
            fired.map_err(|why| Fault::Unbound {
                what: format!(
                    "the size-only walk of a voxel fire, class {class} over {} clip(s) of \
                     {rows} row(s): {why}",
                    boxes.len()
                ),
            })?;
        }
        Ok(())
    }

    /// What arming the lattice starts from, in arming's order; `None`
    /// where this shell arms nothing.
    fn arm_begin(&mut self, verify: bool) -> Option<Arming> {
        // An expert-parallel rank fires the lattice whether or not it
        // records bodies: past arming its group serves every composition it
        // does not replay eagerly and unfenced, beside peers spinning in
        // their collectives, so what those walks first use (a module, a
        // tune, a pinned slot) must already have been made here, fenced.
        let group = self.ep.is_some();
        let records = self.records_bodies() && Self::keyable_units(&self.compiled);
        if !records && !group {
            return None;
        }
        if !records && !verify {
            self.cache.seal_bodies();
        }

        if self.lane_ceiling() == 0 {
            return None;
        }
        if super::diag::on().arm_trace {
            for (class, requests) in self.landing.iter().enumerate() {
                eprintln!("[arm-trace] class c{class}: {requests:?}");
            }
        }
        let LatticeTargets {
            mut targets,
            top,
            unfireable,
        } = self.lattice_targets()?;
        if let Some(bucket) = self.arm_only {
            targets.retain(|(at, ..)| *at == bucket);
        }
        // By rows, every kind's fewest first (`lattice_targets`): the
        // allowance running out leaves the largest rungs, stated eager.
        Some(Arming {
            planned: targets.clone(),
            targets: targets.into(),
            top,
            unfireable,
            verify,
            group,
            records,
            dropped: std::collections::BTreeSet::new(),
            weighed: [(0usize, 0usize); Kind::COUNT],
            armed: 0,
            wanted: 0,
            tally: [(0usize, 0usize); Kind::COUNT],
            refused: None,
            never: 0,
            warmed_only: 0,
            never_from: 0,
            belted: false,
            starved: None,
            held: Vec::new(),
            given_back: Vec::new(),
            freed: 0,
            peer_stopped: false,
            unit_margin: self.pools.arming_margin_bytes(),
            from_record: self.bodies_from_record || self.rearming,
            capped: !(self.bodies_from_record && self.arming_at_load),
            took: Some(0),
            began: std::time::Instant::now(),
            profiled: self.budget.buckets.len(),
            ballot: Ballot::open(&self.device),
        })
    }

    /// Selects before anything captures: a composition's fire backs the
    /// activation span to its rung's carve (its synthetic lanes' pages lie in
    /// the state pool's chunks, backed whole before arming), so priced by
    /// that growth from the load's own rungs, the compositions whose growth
    /// the pool's spare holds are the ones captured, in base order, and the
    /// rest are stated eager untouched. The rung grows with the bucket the
    /// user's rule ranks by, so what is selected is its best-ranked prefix,
    /// and one left out is never warmed, so it never backs a span the
    /// better-ranked ones then find no room beside. Where the spare holds
    /// every carve (or the allowance is a measurement), all are selected and
    /// base order is unchanged; the bodies themselves are weighed as they
    /// capture ([`Shell::arm_target`]).
    fn select_within_headroom(&mut self, st: &mut Arming) {
        if st.verify || st.group || self.ep.is_some() || st.from_record {
            return;
        }
        let mut spare = self
            .pools
            .spare_bytes_for_bodies(self.bodies_mem as u64)
            .unwrap_or(u64::MAX);
        let committed = self.arena.bytes();
        let forked = self.compiled.arena_forked.as_ref();
        let targets: Vec<(u32, BodySynth, Treatment)> = st.targets.drain(..).collect();
        let growth = |target: &BodySynth| {
            let (lanes, _) = target.lanes();
            let tokens: u64 = lanes.iter().map(|(_, rows)| u64::from(*rows)).sum();
            self.arena
                .carve_of(forked, tokens, lanes.len() as u64)
                .saturating_sub(committed)
        };
        // By the user's rule, best first: each priced by what it backs past
        // the compositions already selected.
        let mut order: Vec<usize> = (0..targets.len()).collect();
        order.sort_by_key(|&at| self.rule_rank(targets[at].0, &targets[at].1, targets[at].2));
        let mut selected = vec![false; targets.len()];
        let mut backed = 0u64;
        for at in order {
            let carve = growth(&targets[at].1);
            let grows = carve.saturating_sub(backed);
            if grows <= spare {
                selected[at] = true;
                spare -= grows;
                backed = backed.max(carve);
            }
        }
        let total = targets.len();
        let mut kept = 0usize;
        for (at, target) in targets.into_iter().enumerate() {
            if selected[at] {
                kept += 1;
                st.targets.push_back(target);
            } else {
                st.given_back.push(target);
            }
        }
        if kept < total {
            tracing::info!(
                "arming selects {kept} of {total} compositions by the user's rule: \
                 the rest would back more of the activation span than the pool's spare holds"
            );
        }
    }

    fn arm_lattice_fenced(&mut self, verify: bool) -> Result<()> {
        let Some(mut st) = self.arm_begin(verify) else {
            return Ok(());
        };
        self.select_within_headroom(&mut st);
        while let Some((bucket, target, treatment)) = st.targets.pop_front() {
            self.arm_target(&mut st, bucket, target, treatment)?;
        }
        // The compositions given back for the allowance, stated eager by
        // name: each fired once, sealed, so nothing captures.
        if !st.given_back.is_empty() {
            self.cache.seal_bodies();
            self.cache.state_eager(true);
            for (bucket, target, treatment) in std::mem::take(&mut st.given_back) {
                if st.never == 0 {
                    st.never_from = bucket;
                }
                st.never += 1;
                self.arm_rs = treatment.rs;
                if matches!(target, BodySynth::Idle) {
                    self.at_bucket(bucket);
                }
                let (lanes, media) = target.lanes();
                let owned = self.synthetic_lanes_with(&lanes, &media);
                if let Err(why) = self.fire_synthetic(&owned).and(self.device.synchronize()) {
                    st.refused = Some(format!("bucket {bucket}, {target}: {why}"));
                }
            }
            self.cache.state_eager(false);
        }
        self.arm_end(st)
    }

    /// Where the user's rule ranks a composition for the bodies' allowance,
    /// the least kept first: by the stated step it records, in the set's
    /// order (by rung across every kind, the smallest first, so what goes
    /// eager is always the tail of the set); within one, by rows; within one
    /// row count the one that launches more kernels first (more launches,
    /// more an eager fire of it costs), as the size-only walk counted them
    /// (`walked_launches`), else by its classes; then by kind.
    fn rule_rank(&self, bucket: u32, target: &BodySynth, treatment: Treatment) -> Rank {
        let (lanes, _) = target.lanes();
        let mut classes: Vec<usize> = lanes.iter().map(|(class, _)| *class).collect();
        classes.sort_unstable();
        classes.dedup();
        let kind = match target.kind() {
            Kind::Decode => 0u8,
            Kind::Mixed => 1,
            Kind::Prefill => 2,
            Kind::Wide => 3,
            Kind::Tower => 4,
            Kind::Idle => 5,
        };
        (
            treatment.shape,
            bucket,
            core::cmp::Reverse(self.walked_launches.get(&(bucket, target.to_string())).copied()),
            core::cmp::Reverse(classes.len()),
            kind,
        )
    }

    fn arm_target(
        &mut self,
        st: &mut Arming,
        bucket: u32,
        target: BodySynth,
        treatment: Treatment,
    ) -> Result<()> {
        let weighed_before = self.cache.body_stats().census;
        if st.dropped.contains(&bucket) {
            return Ok(());
        }
        self.arm_rs = treatment.rs;
        // Past its kind's widest bucket a target captures cold: the top
        // bucket's fires made what its warm fire would.
        self.arm_cold = st.narrower(bucket, &target) && !st.verify;
        // A single rank that is short of room (its allowance spent, or the
        // pool's headroom under a unit's margin) gives back the held capture
        // the user's rule ranks worst while that ranks below this one; where
        // none does, this composition is the one left eager (stated after
        // the loop) and arming goes on: a composition later in capture order
        // may rank above what is held.
        if !st.verify && !st.group && self.ep.is_none() {
            let rank = self.rule_rank(bucket, &target, treatment);
            loop {
                // The room a capture has is the allowance net of what is
                // held: a body given back frees its bytes for the next one.
                // The pool's spare under its ceiling does not rise when one
                // is given back (the driver keeps what it backed), so judged
                // by it every better-ranked capture gave back every worse one
                // and still found none, and even a one-lane decode walked
                // eager at a tight budget.
                // A body lies outside the pool, in the device's free memory:
                // its room is what the device holds free, and a body given
                // back frees its bytes for the next capture though the device
                // does not show them free again (the driver keeps what it
                // backed), so they count beside it. Short where that has no
                // room for a body at the price the held ones set, or where
                // the allowance net of what is held is spent.
                let spent = self.cache.body_stats();
                let free = crate::device::free_bytes()
                    .unwrap_or(u64::MAX)
                    .saturating_add(st.freed);
                let price = (2 * spent.census.bytes / spent.census.bodies.max(1)) as u64;
                let short = (!st.from_record && free < price)
                    || (st.capped && allowance_spent(spent.census.bytes, self.bodies_mem));
                if !short || spent.census.bodies >= record::MAX_BODIES {
                    break;
                }
                let worst = st
                    .held
                    .iter()
                    .enumerate()
                    .filter(|(_, held)| held.rank > rank)
                    .max_by(|(_, a), (_, b)| a.rank.cmp(&b.rank))
                    .map(|(index, _)| index);
                let Some(worst) = worst else {
                    st.given_back.push((bucket, target, treatment));
                    return Ok(());
                };
                let given = st.held.remove(worst);
                self.cache.forget_body(&given.key);
                st.freed = st.freed.saturating_add(given.weight);
                st.took = st.took.map(|took| took.saturating_sub(given.weight));
                st.armed -= 1;
                st.tally[given.at].0 -= 1;
                st.weighed[given.at].0 =
                    st.weighed[given.at].0.saturating_sub(given.weight as usize);
                st.weighed[given.at].1 = st.weighed[given.at].1.saturating_sub(1);
                st.given_back.push((given.bucket, given.target, given.treatment));
            }
        }
        let spent = self.cache.body_stats();
        let room = (self.bodies_mem as u64).saturating_sub(spent.census.bytes as u64);
        let spare = self.pools.spare_bytes_for_bodies(room).unwrap_or(u64::MAX);
        let price = (2 * spent.census.bytes / spent.census.bodies.max(1)) as u64;
        // A map unit's margin never past a quarter of the allowance: an
        // allowance the fit held under the margin would otherwise stop
        // arming before its first body.
        let margin = price.max(st.unit_margin.min(self.bodies_mem as u64 / 4));
        // Estimated where nothing is known; a record's allowance is a
        // measurement, and the headroom estimate is not consulted.
        let headroom = !st.from_record && spare < margin;
        // A single rank gave back what it could above: only the bodies'
        // count stops it here. An expert-parallel group stops every rank's
        // captures where one runs out (the allowance or the headroom).
        let single = !st.group && self.ep.is_none();
        let mine = !st.verify
            && (spent.census.bodies >= record::MAX_BODIES
                || (!single
                    && ((st.capped && allowance_spent(spent.census.bytes, self.bodies_mem))
                        || headroom)));
        let stop = match self.ep.as_ref() {
            // The group's ranks fire one sequence of steps: the allowance
            // one rank runs out of stops every rank's captures.
            Some(ep) => !st.verify && ep.wire().quiet.any(mine)?,
            None => !st.verify && st.ballot.any(&self.device, mine),
        };
        if stop {
            if st.never == 0 {
                st.never_from = bucket;
                st.belted = spent.census.bodies >= record::MAX_BODIES;
                if headroom && !st.belted && spent.census.bytes < self.bodies_mem {
                    st.starved = Some(spare);
                }
                st.peer_stopped = !mine;
            }
            st.never += 1;
            if !st.group {
                // Fired once, sealed, so nothing captures: its key is stated
                // eager by name at load, and its fires walk eagerly by that
                // statement (`BodyTally::stated_eager`), not as a body
                // missing (`sealed_declines`).
                self.cache.seal_bodies();
                self.cache.state_eager(true);
                let (lanes, media) = target.lanes();
                let owned = self.synthetic_lanes_with(&lanes, &media);
                let fired = self.fire_synthetic(&owned).and(self.device.synchronize());
                self.cache.state_eager(false);
                if let Err(why) = fired {
                    st.refused = Some(format!("bucket {bucket}, {target}: {why}"));
                }
                return Ok(());
            }
            // An expert-parallel rank fires it anyway, sealed so nothing
            // captures: its served eager walks find what they use made.
            self.cache.seal_bodies();
            st.warmed_only += 1;
        }
        let began = std::time::Instant::now();
        let at = target.kind().at();
        let (lanes, media) = target.lanes();
        if !stop {
            st.wanted += 1;
            st.tally[at].1 += 1;
        }
        let owned = self.synthetic_lanes_with(&lanes, &media);
        self.armed_body = None;
        // What the composition's capture took: the free bytes the device lost
        // across the fire that captured it (the last one run), less what the
        // pool backed then. The warm fires before it are walks: a kernel's
        // first launch there loads what the process keeps for every load
        // after (code objects, a library's kernels, the runtime's queues),
        // which no load's bodies hold, so a load's figure counts none of it
        // and does not depend on which load of the process ran first.
        let (mut free_before, mut backed_before) = (None, 0);
        let idle = matches!(target, BodySynth::Idle);
        for _ in 0..record::WARM_FIRES {
            if idle {
                self.at_bucket(bucket);
            }
            free_before = crate::device::free_bytes();
            backed_before = self.pools.backed_bytes();
            let fired = self.fire_synthetic(&owned);
            let landed = self.device.synchronize();
            if let Err(why) = fired.and(landed) {
                if super::diag::on().arm_trace {
                    eprintln!("[arm-trace] refused bucket {bucket}, {target}: {why}");
                }
                st.refused = Some(format!("bucket {bucket}, {target}: {why}"));
                break;
            }
            // Armed on its first fire (a cold capture): a second would
            // only replay it.
            if self
                .armed_body
                .as_ref()
                .is_some_and(|key| self.cache.holds_body(key))
            {
                break;
            }
        }
        self.arm_cold = false;
        // What the pool backed across the capture's fire (a mapped range
        // backs the activation's holds and segments as the fires first touch
        // them) is the pool's, not the capture's; a flat range, allocated,
        // backs none.
        let pool_grew = self.pools.backed_bytes().saturating_sub(backed_before);
        st.took = st
            .took
            .zip(free_before)
            .zip(crate::device::free_bytes())
            .map(|((took, before), after)| {
                took + before.saturating_sub(after).saturating_sub(pool_grew)
            });
        let key = self.armed_body.take();
        let warmed_ms = began.elapsed().as_secs_f64() * 1000.0;
        let mut golden_ms = (0f64, 0f64);
        if st.verify
            && !idle
            && let Some(key) = key.as_ref()
            && self.cache.holds_body(key)
        {
            let clock = std::time::Instant::now();
            self.golden(key, &owned)?;
            golden_ms.0 = clock.elapsed().as_secs_f64() * 1000.0;
            let clock = std::time::Instant::now();
            self.golden_real(key, &owned)?;
            golden_ms.1 = clock.elapsed().as_secs_f64() * 1000.0;
        }
        if key.as_ref().is_some_and(|key| self.cache.body_armed(key)) {
            let now = self.cache.body_stats().census;
            // The stop above weighs what the bodies took before this
            // capture, not what it took: the capture that crossed the
            // allowance is given back, and its composition is the first
            // stated eager (pushed back for the stop to state), so the
            // bodies end within what the fit measured for them.
            let weight = now.bytes.saturating_sub(weighed_before.bytes) as u64;
            st.armed += 1;
            st.tally[at].0 += 1;
            if let Some(key) = key.as_ref() {
                st.held.push(Held {
                    rank: self.rule_rank(bucket, &target, treatment),
                    key: key.clone(),
                    bucket,
                    target: target.clone(),
                    treatment,
                    weight,
                    at,
                });
            }
            // A capture takes the room given-back ones freed first.
            st.freed = st.freed.saturating_sub(weight);
            // Past the allowance (a capped single-rank arming): the capture
            // the user's rule ranks last among those held (this one or an
            // earlier) is given back, its weight off what the bodies took,
            // until they fit; each is stated eager after the loop.
            if st.capped && !st.group && !st.verify {
                while allowance_spent(self.cache.body_stats().census.bytes, self.bodies_mem) {
                    let Some(worst) = st
                        .held
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.rank.cmp(&b.rank))
                        .map(|(index, _)| index)
                    else {
                        break;
                    };
                    let given = st.held.remove(worst);
                    self.cache.forget_body(&given.key);
                    st.took = st.took.map(|took| took.saturating_sub(given.weight));
                    st.armed -= 1;
                    st.tally[given.at].0 -= 1;
                    st.weighed[given.at].0 =
                        st.weighed[given.at].0.saturating_sub(given.weight as usize);
                    st.weighed[given.at].1 = st.weighed[given.at].1.saturating_sub(1);
                    st.given_back.push((given.bucket, given.target, given.treatment));
                }
                if !st.held.iter().any(|held| Some(&held.key) == key.as_ref()) {
                    return Ok(());
                }
            }
            if now.bodies > weighed_before.bodies {
                st.weighed[at].0 += now.bytes.saturating_sub(weighed_before.bytes);
                st.weighed[at].1 += now.bodies - weighed_before.bodies;
            }
            if let Some(wanted) = super::diag::on().arm_bench.as_deref()
                && let Some(key) = key.as_ref()
                && key.to_string().contains(wanted)
            {
                let mut shapes: Vec<(&str, Vec<(usize, u32)>)> = vec![("lattice", lanes.clone())];
                let classes: Vec<usize> =
                    lanes
                        .iter()
                        .map(|(class, _)| *class)
                        .fold(Vec::new(), |mut seen, class| {
                            if !seen.contains(&class) {
                                seen.push(class);
                            }
                            seen
                        });
                let decoders: Vec<usize> = classes
                    .iter()
                    .copied()
                    .filter(|class| self.decoding.contains(*class))
                    .collect();
                if matches!(target, BodySynth::Wide { .. }) && decoders.len() > 1 {
                    let seats = (bucket as usize).min(lanes.len()).max(2);
                    for &class in &decoders[1..] {
                        shapes.push(("alone", vec![(class, 1u32); seats]));
                        let mut pair = vec![(decoders[0], 1u32); seats / 2];
                        pair.extend(vec![(class, 1u32); seats - seats / 2]);
                        shapes.push(("pair", pair));
                    }
                } else if classes.len() > 1 && lanes.iter().all(|(_, rows)| *rows == 1) {
                    let mut skewed = vec![(classes[0], 1u32); lanes.len() + 1 - classes.len()];
                    skewed.extend(classes[1..].iter().map(|class| (*class, 1u32)));
                    shapes.push(("skewed", skewed));
                }
                for (what, shape) in shapes {
                    let owned = self.synthetic_lanes(&shape);
                    let mut timed = Ok(());
                    let mut rows_back = 0usize;
                    for _ in 0..10 {
                        timed = timed
                            .and(
                                self.fire_synthetic_as(&owned, crate::serve::Golden::Off)
                                    .map(|out| rows_back = out.len()),
                            )
                            .and(self.device.synchronize());
                    }
                    let before = self.cache.body_stats().tally;
                    let began = std::time::Instant::now();
                    const FIRES: u32 = 50;
                    for _ in 0..FIRES {
                        timed = timed
                            .and(self.fire_synthetic(&owned))
                            .and(self.device.synchronize());
                    }
                    let us = began.elapsed().as_secs_f64() * 1e6 / f64::from(FIRES);
                    let after = self.cache.body_stats().tally;
                    let replayed = after.hits - before.hits;
                    let lanes_of = |class: usize| shape.iter().filter(|(c, _)| *c == class).count();
                    let layout: Vec<String> = shape
                        .iter()
                        .map(|(class, _)| *class)
                        .fold(Vec::new(), |mut seen, class| {
                            if !seen.contains(&class) {
                                seen.push(class);
                            }
                            seen
                        })
                        .iter()
                        .map(|c| format!("c{c}x{}", lanes_of(*c)))
                        .collect();
                    match timed {
                        Ok(()) => eprintln!(
                            "[arm-bench] {key} {what} [{}] {us:.0} us/fire, replayed {replayed}/{FIRES}, rows back {rows_back}, misses +{} declines +{} eager +{}",
                            layout.join(" "),
                            after.misses - before.misses,
                            after.sealed_declines - before.sealed_declines,
                            (after.eager_buffered + after.eager_copy_world)
                                - (before.eager_buffered + before.eager_copy_world)
                        ),
                        Err(why) => eprintln!("[arm-bench] {key} {what} refused: {why}"),
                    }
                }
            }
            if super::diag::on().arm_trace
                && let Some(key) = key.as_ref()
            {
                let last = self.cache.body_stats().last_capture;
                let islands: Vec<String> = self
                    .cache
                    .body_script(key)
                    .into_iter()
                    .filter(|(island, _, _)| *island)
                    .map(|(_, from, upto)| {
                        let ops: Vec<&str> = (from..upto)
                            .map(|at| {
                                self.compiled
                                    .template()
                                    .get(at as usize)
                                    .and_then(|region| {
                                        self.trace.nodes.get(region.nodes.start as usize)
                                    })
                                    .map_or("?", |node| poem_ir::Operands::name(&node.op))
                            })
                            .collect();
                        format!("{from}..{upto}:{}", ops.join("+"))
                    })
                    .collect();
                eprintln!(
                    "[arm-trace] armed {key} nodes={} edges={} islands={} ms={:.0} (warm+capture {warmed_ms:.0}, golden {:.0}, real {:.0}){}",
                    last.nodes,
                    last.edges,
                    last.islands,
                    began.elapsed().as_secs_f64() * 1000.0,
                    golden_ms.0,
                    golden_ms.1,
                    if islands.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", islands.join(" "))
                    }
                );
            }
        }
        Ok(())
    }

    fn arm_end(&mut self, mut st: Arming) -> Result<()> {
        self.arm_rs = record::RsBody::Folds;
        // What the walk recorded of each stated step, by treatment and
        // bucket: every composition it walked there held.
        if !st.verify && self.arm_only.is_none() {
            let held: std::collections::HashSet<(u32, String, record::RsBody)> = st
                .held
                .iter()
                .map(|held| (held.bucket, held.target.to_string(), held.treatment.rs))
                .collect();
            self.walked.clear();
            for (bucket, target, treatment) in &st.planned {
                let shape = &self.steps.shapes[treatment.shape];
                let recorded = held.contains(&(*bucket, target.to_string(), treatment.rs));
                *self
                    .walked
                    .entry((shape.treatment, *bucket))
                    .or_insert(true) &= recorded;
            }
        }
        let stats = self.cache.body_stats();
        // An expert-parallel rank never captures past arming, whatever armed:
        // its peers fire other steps beside its, and an instantiation (a
        // capture's) takes the context while their collectives spin.
        let seal = if st.armed == 0 && self.ep.is_some() {
            Seal::Empty
        } else if st.armed == 0 {
            Seal::Open
        } else if st.never == 0 && st.refused.is_none() {
            Seal::Complete
        } else {
            Seal::Partial { never: st.never }
        };
        let pool_line = format!(
            "pool declared {} MiB, high water {} MiB, committed {} MiB, spare under the ceiling {} MiB",
            self.pools.declared_bytes() >> 20,
            self.pools.high_water_bytes() >> 20,
            self.pools.committed_bytes() >> 20,
            self.pools.spare_bytes().map_or(0, |bytes| bytes >> 20),
        );
        let report = Armed {
            pool_line,
            wanted: st.wanted,
            armed: st.armed,
            kinds: st.tally,
            warmed_only: st.warmed_only,
            segmented: stats.census.segmented,
            bytes: stats.census.bytes,
            unweighed: stats.census.unweighed,
            bodies_mem: self.bodies_mem,
            bodies: stats.census.bodies,
            last_refusal: st.refused.clone(),
            unfireable: std::mem::take(&mut st.unfireable),
            never: st.never,
            never_from: st.never_from,
            belted: st.belted,
            starved: st.starved,
            peer_stopped: st.peer_stopped,
            declines: stats.tally.declines,
            refusals: stats.tally.refusals,
            seal,
        };
        tracing::info!(
            "{report}; armed in {:.1} s",
            st.began.elapsed().as_secs_f64()
        );
        // Every composition arming stopped before (the report above says
        // why), by name: they walk eagerly for the life of the load by this
        // statement, their fires tallied as such (`BodyTally::stated_eager`).
        let stated_now = self.cache.stated_eager();
        if !stated_now.is_empty() && self.arm_only.is_none() {
            tracing::info!(
                "{} composition key(s) walk eagerly by statement, arming stopped \
                 before them from bucket {}: {}",
                stated_now.len(),
                st.never_from,
                stated_now.join(", ")
            );
        }
        // The lattice the bodies armed over against the profile's: fewer
        // buckets is the allowance binding, each fire padded further.
        tracing::info!(
            "lattice armed over {} of the profile's {} buckets",
            self.budget.buckets.len(),
            st.profiled
        );
        // The record: written where every rung armed, at what the bodies
        // took and what their captures took; dropped, and said, where a record's
        // allowance armed short of whole.
        let took = stats.census.bytes as u64;
        let captured = st.took.unwrap_or(took);
        if !st.verify {
            // What this arming took, the allowance a re-arm within the same
            // load starts from ([`Shell::rearm_bodies`]) and a measuring
            // load's refit holds: a load with no record to keep (no cache
            // directory) measures all the same.
            self.bodies_took = captured.max(took);
        }
        // A re-arm of one bucket (a product's kernel choice) is no
        // arming of the deployment's: it concludes no record.
        let stated = self.cache.stated_eager();
        if let Some(record) = self.bodies_record.as_deref()
            && !st.verify
            && self.arm_only.is_none()
        {
            let measured = super::measured::Measured {
                bytes: took,
                bodies: stats.census.bodies as u64,
                captured,
            };
            if super::diag::on().arm_trace {
                eprintln!(
                    "[arm-trace] arming's captures took {} MiB; {} bodies weigh {} MiB",
                    captured >> 20,
                    measured.bodies,
                    took >> 20,
                );
            }
            match super::measured::conclude(
                record,
                self.bodies_from_record,
                // Whole only where nothing walks eagerly by statement.
                matches!(seal, Seal::Complete) && stated.is_empty(),
                measured,
                st.armed,
                st.wanted + st.never,
            ) {
                Ok(Some(line)) => eprintln!("{line}"),
                Ok(None) => {}
                Err(why) => tracing::warn!(
                    "the bodies' record at {} is not kept: {why}",
                    record.display()
                ),
            }
        }
        // A load that stopped arming for want of room serves the rungs past
        // the stop (the largest: arming runs by rows) eagerly, each stated
        // by name above, its fires counted apart (`BodyTally::stated_eager`),
        // even where that is every rung. One whose recordings failed while
        // room remained armed none for a fault, not for room: refused. A load
        // that stages host-resident experts reads each fire's routes back,
        // so it records none by design (`Run::capturable`; stated at load).
        let attempted = st.wanted + st.never;
        if st.records
            && self.expert_staged.is_none()
            && !st.verify
            && !st.group
            && attempted > 0
            && st.armed == 0
            && st.never == 0
            && st.refused.is_some()
        {
            return Err(Fault::Residency(format!(
                "this deployment's graph bodies would not record: {} of {attempted} \
                 compositions armed{}, with room left. Lower `[engine] max_context`, raise \
                 `[engine.memory] budget_device`, or free what else holds the device; on a \
                 device another process shares, state `[engine.memory] budget_device` as this \
                 engine's part of it, so the fit leaves the other process its room. \
                 ({report})",
                st.armed,
                match st.starved {
                    Some(spare) => format!(
                        ", arming stopped with {} MiB spare under the ceiling",
                        spare >> 20
                    ),
                    None => String::new(),
                },
            )));
        }
        if super::diag::on().arm_trace {
            let free = crate::device::free_bytes().unwrap_or(0);
            eprintln!(
                "[arm-trace] device memory: free {} MiB; arena committed {} MiB, inputs reserve {} MiB, buffered activations {} MiB, readout rows {} MiB, rs scratch {} MiB, pool committed {} MiB",
                free >> 20,
                self.arena.bytes() >> 20,
                self.inputs.bytes() >> 20,
                self.buffer_bytes() >> 20,
                self.readout_rows.bytes() >> 20,
                self.buffers
                    .as_ref()
                    .map_or(0, |buffers| buffers.scratch_bytes())
                    >> 20,
                self.pools.committed_bytes() >> 20,
            );
        }
        if !st.verify && !matches!(seal, Seal::Open) {
            self.cache.seal_bodies();
        }
        self.armed = Some(report);
        Ok(())
    }

    fn spread(rows: u32, lanes: u32, context: u32) -> Option<Vec<u32>> {
        let lanes = lanes.min(rows);
        if lanes == 0 || u64::from(context) * u64::from(lanes) < u64::from(rows) {
            return None;
        }
        let base = rows / lanes;
        let over = rows % lanes;
        Some((0..lanes).map(|at| base + u32::from(at < over)).collect())
    }
}

#[derive(Debug, Clone)]
pub struct Armed {
    pub pool_line: String,
    pub wanted: usize,
    pub armed: usize,
    pub kinds: [(usize, usize); Kind::COUNT],
    /// Compositions an expert-parallel rank fired past its allowance (or
    /// recording none), eagerly, to make what serving them first uses.
    pub warmed_only: usize,
    pub segmented: usize,
    pub bytes: usize,
    pub unweighed: usize,
    pub bodies_mem: usize,
    pub bodies: usize,
    pub last_refusal: Option<String>,
    pub unfireable: Vec<String>,
    pub never: usize,
    pub never_from: u32,
    pub belted: bool,
    pub starved: Option<u64>,
    pub peer_stopped: bool,
    pub declines: u64,
    pub refusals: u64,
    pub seal: Seal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seal {
    Complete,
    Partial {
        never: usize,
    },
    /// Nothing armed, and the map sealed anyway: an expert-parallel rank's,
    /// whose every composition walks eagerly.
    Empty,
    Open,
}

impl core::fmt::Display for Armed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let columns = Kind::ALL
            .iter()
            .filter(|kind| **kind != Kind::Idle || self.kinds[kind.at()].1 > 0)
            .map(|kind| {
                format!(
                    "{kind} {}/{}",
                    self.kinds[kind.at()].0,
                    self.kinds[kind.at()].1
                )
            })
            .collect::<Vec<String>>()
            .join(", ");
        write!(
            f,
            "bodies armed {} of {} compositions ({columns}; {} segmented; {} MiB of {} \
             MiB, {} of {} seats)",
            self.armed,
            self.wanted,
            self.segmented,
            self.bytes >> 20,
            self.bodies_mem >> 20,
            self.bodies,
            record::MAX_BODIES,
        )?;
        if self.warmed_only > 0 {
            write!(
                f,
                " ({} more fired eagerly, uncaptured, to warm what serving them uses)",
                self.warmed_only
            )?;
        }
        if self.unweighed > 0 {
            write!(f, " ({} unweighed)", self.unweighed)?;
        }
        write!(f, " ({})", self.pool_line)?;
        if let Some(why) = &self.last_refusal {
            write!(f, " (last refusal: {why})")?;
        }
        if !self.unfireable.is_empty() {
            write!(
                f,
                " [{} key(s) never fired — this deployment cannot synthesize them, or their \
                 present set was already refused admission; e.g. {}]",
                self.unfireable.len(),
                self.unfireable[0],
            )?;
        }
        match self.seal {
            Seal::Complete => write!(f, " [sealed: every key attempted]")?,
            Seal::Partial { never } if never > 0 => write!(
                f,
                " [sealed partial: {never} key(s) never attempted, {} at bucket {}, and they \
                 walk eagerly for the life of this load]",
                match self.starved {
                    _ if self.peer_stopped =>
                        "a peer rank's bound (the stop is agreed across the group)".to_string(),
                    Some(spare) => format!(
                        "the ceiling's spare ran out ({} MiB left beyond the elastic pool's growth)",
                        spare >> 20
                    ),
                    None if self.belted => "record::MAX_BODIES".to_string(),
                    None => "the bodies' hold".to_string(),
                },
                self.never_from,
            )?,
            Seal::Partial { .. } => write!(f, " [sealed partial: a key refused]")?,
            Seal::Empty => write!(
                f,
                " [sealed empty: nothing armed, and an expert-parallel rank captures nothing \
                 past arming, so every composition walks eagerly]"
            )?,
            Seal::Open => write!(f, " [not sealed: nothing armed]")?,
        }
        if self.declines != 0 || self.refusals != 0 {
            write!(
                f,
                " [{} declined a workspace grant, {} inadmissible]",
                self.declines, self.refusals
            )?;
        }
        Ok(())
    }
}

/// Whether bodies weighing `bytes` have spent an allowance of `allowance`:
/// past it, not at it. A re-arm is allowed exactly what the arming before it
/// took (`Shell::measured_refit`), and its compositions past the last new
/// body re-capture bodies it already holds (a wide composition's key a
/// mixed one's): at the allowance they cost nothing, and stopping there
/// would leave them walking eagerly for the life of the load.
fn allowance_spent(bytes: usize, allowance: usize) -> bool {
    bytes > allowance
}

#[cfg(test)]
mod allowance_tests {
    use super::allowance_spent;

    #[test]
    fn a_re_arm_at_exactly_what_the_arming_before_took_arms_on() {
        // The measuring boot's refit: 53 bodies weighing what the first
        // arming's 53 weighed, ten compositions still to re-capture.
        let took = 150 << 20;
        assert!(!allowance_spent(took, took));
        assert!(allowance_spent(took + 1, took));
        assert!(!allowance_spent(took - 1, took));
    }
}

#[cfg(test)]
mod timing_tests {
    use super::BodySynth;
    use super::timing::timing_target;

    #[test]
    fn a_product_is_timed_in_a_body_the_lattice_armed() {
        let decode = BodySynth::Decode { lanes: 4, class: 0 };
        let prompt = BodySynth::Prefill {
            class: 0,
            rows: vec![4],
        };
        let decoded = |found: Option<(&BodySynth, ())>| found.map(|(target, ())| target.lanes().0);
        // A lattice that arms the decode step at bucket 4: its body.
        let both = [(4, &prompt, ()), (4, &decode, ()), (8, &decode, ())];
        assert_eq!(decoded(timing_target(both, 4)), Some(vec![(0, 1); 4]));
        // An artifact baked for one lane arms no four-lane decode step
        // (golden_offload's micro model on ROCm): the prompt's body of 4.
        let one = BodySynth::Decode { lanes: 1, class: 0 };
        let one_lane = [(4, &prompt, ()), (4, &one, ())];
        assert_eq!(decoded(timing_target(one_lane, 4)), Some(vec![(0, 4)]));
        // Nothing armed at the bucket carries its rows.
        assert_eq!(decoded(timing_target([(2, &prompt, ())], 4)), None);
    }
}
