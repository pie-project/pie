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
/// A lattice being armed: the targets left, and what arming has counted.
pub(super) struct Arming {
    targets: std::collections::VecDeque<(u32, BodySynth, record::RsBody)>,
    /// The rung whose targets are arming.
    rung: u32,
    top: Option<u32>,
    /// Whether a narrower target may capture cold: the top bucket's fires
    /// have made what its warm fire would.
    topped: bool,
    unfireable: Vec<String>,
    verify: bool,
    group: bool,
    records: bool,
    priced: bool,
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
    peer_stopped: bool,
    unit_margin: u64,
    /// The buckets of the profile's lattice, before arming coarsened any
    /// away.
    profiled: usize,
    ballot: Ballot,
}

impl Arming {
    /// Whether every composition the lattice planned holds a body: those
    /// it fired for one (`wanted`) and those the allowance stopped before
    /// their fire (`never`) alike. A bucket the lattice coarsened away is
    /// not planned: its fires pad to a wider bucket's body.
    fn whole(&self) -> Result<()> {
        let planned = self.wanted + self.never;
        if self.armed < planned {
            return Err(Fault::Residency(format!(
                "arming ended with {} of {planned} compositions holding a body ({} the \
                 allowance stopped{}): the rungs past the last armed stay unarmed",
                self.armed,
                self.never,
                self.refused
                    .as_ref()
                    .map_or_else(String::new, |why| format!("; last refusal: {why}")),
            )));
        }
        Ok(())
    }

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticePoint {
    pub bucket: u32,
    pub lanes: u32,
}

struct Deployment {
    points: Vec<LatticePoint>,
    buckets: Vec<u32>,
    patch_points: Vec<u32>,
    decoders: Vec<usize>,
    prefilling: Vec<usize>,
    /// The classes keyed only for a speculative decoder's fires (plain but
    /// for reading the drafts): armed alone, never beside another class.
    drafting: Vec<usize>,
    /// The buckets a drafter's class is armed at (`Shell::rs_rungs`).
    rs_rungs: Vec<u32>,
    media: Vec<usize>,
    wide: Vec<usize>,
    decoding: poem_ir::ClassSet,
    seats: u32,
    context: u32,
    max_lanes: u32,
    /// `engine.prefill_lanes`: the most lanes a fire of prompts alone pads to.
    prefill_lanes: u32,
    patch_fold: u32,
    /// Whether the load is an expert-parallel rank's, which fires a step
    /// with no lanes of its own at every bucket its group fires.
    idle: bool,
}

#[derive(Default)]
struct Targets {
    targets: Vec<(u32, BodySynth)>,
    unfireable: Vec<String>,
}

fn unfireable_line(deployment: &Deployment, what: &str, at: &str, note: Option<&str>) -> String {
    let (note, comma) = match note {
        Some(note) => (note, ", "),
        None => ("", ""),
    };
    format!(
        "{what} at {at} ({note}{comma}{} seat(s) x {} context, {} lane(s))",
        deployment.seats, deployment.context, deployment.max_lanes,
    )
}

fn decode_keys(deployment: &Deployment, into: &mut Targets) {
    for LatticePoint { bucket, lanes } in deployment.points.iter().copied() {
        for class in deployment.decoders.iter().copied() {
            into.targets
                .push((bucket, BodySynth::Decode { lanes, class }));
        }
    }
}

fn prefill_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for class in deployment.prefilling.iter().copied() {
            if deployment.drafting.contains(&class) && !deployment.rs_rungs.contains(&point) {
                continue;
            }
            match Shell::spread(
                point,
                deployment.seats.min(match deployment.prefill_lanes {
                    0 => deployment.max_lanes,
                    cap => deployment.max_lanes.min(cap),
                }),
                deployment.context,
            ) {
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
}

fn mixed_keys(deployment: &Deployment, into: &mut Targets) {
    let alone = |class: &usize| !deployment.drafting.contains(class);
    for point in deployment.buckets.iter().copied() {
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
}

fn tower_keys(deployment: &Deployment, into: &mut Targets) {
    for patches in deployment.patch_points.iter().copied() {
        for class in deployment.media.iter().copied() {
            for point in deployment.buckets.iter().copied() {
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
}

fn wide_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        let seats = deployment.seats.min(deployment.max_lanes).min(point) as usize;
        let decoders: Vec<usize> = deployment
            .wide
            .iter()
            .copied()
            .filter(|class| !deployment.drafting.contains(class))
            .filter(|class| deployment.decoding.contains(*class))
            .take(seats)
            .collect();
        let prefilling: Vec<usize> = deployment
            .wide
            .iter()
            .copied()
            .filter(|class| !deployment.drafting.contains(class))
            .filter(|class| !deployment.decoding.contains(*class))
            .take(seats.saturating_sub(decoders.len()))
            .collect();
        let rows_left = point - decoders.len() as u32;
        let spread = if prefilling.is_empty() {
            Some(Vec::new())
        } else {
            Shell::spread(rows_left, prefilling.len() as u32, deployment.context)
        };
        match spread {
            Some(rows) if decoders.len() + prefilling.len() > 1 => {
                let mut lanes: Vec<(usize, u32)> =
                    decoders.iter().map(|class| (*class, 1u32)).collect();
                lanes.extend(
                    prefilling
                        .iter()
                        .zip(rows)
                        .map(|(class, rows)| (*class, rows)),
                );
                into.targets.push((point, BodySynth::Wide { lanes }));
            }
            _ => into.unfireable.push(unfireable_line(
                deployment,
                "wide",
                &format!("bucket {point}"),
                None,
            )),
        }
    }
}

/// An expert-parallel rank idles through its peers' steps at every bucket
/// (`Shell::fire_idle`): armed like any composition, an idle step replays
/// what a busy one's peers replay, and never walks eagerly beside them.
fn idle_keys(deployment: &Deployment, into: &mut Targets) {
    if deployment.idle {
        for point in deployment.buckets.iter().copied() {
            into.targets.push((point, BodySynth::Idle));
        }
    }
}

const ARMS: [fn(&Deployment, &mut Targets); Kind::COUNT] = [
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
    fn synthetic_lanes(&self, lanes: &[LaneRows]) -> Vec<Synthetic> {
        self.synthetic_lanes_with(lanes, &[])
    }

    /// The records, kv pages and windowed pages the lattice's synthetic
    /// fires name ([`Pools::synthetic_room`]): the widest fire's.
    fn synthetic_room(&self) -> (u32, u64, u64) {
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let lanes_most = self.fire_lanes().max(1);
        self.pools.synthetic_room(
            (self.held.len().max(1) as u32).min(lanes_most),
            u64::from(self.budget.max_tokens).div_ceil(page_size) + u64::from(lanes_most),
        )
    }

    /// What the runtime reads of this shell while it serves.
    #[must_use]
    pub fn live(&self) -> engine::live::Live {
        self.live.clone()
    }

    /// The slots a step of the lattice left arming writes
    /// ([`Shell::arm_step`]); none where nothing is left.
    #[must_use]
    pub fn arming_room(&self) -> engine::live::ArmingRoom {
        if self.stepping.is_none() {
            return engine::live::ArmingRoom::default();
        }
        let (_, pages, _) = self.synthetic_room();
        engine::live::ArmingRoom {
            pages: u32::try_from(pages).unwrap_or(u32::MAX),
            // One record, every piece of it: a synthetic lane's rows are
            // never read back, so a step's lanes share it.
            records: self.pools.record_pieces(),
        }
    }

    fn synthetic_lanes_with(&self, lanes: &[LaneRows], media: &[LaneMedia]) -> Vec<Synthetic> {
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        // What the lanes name, inside the chunks the fit left the pool: a
        // stated kv budget may hold fewer records and pages than the
        // lattice's widest fire names, and a synthetic lane's rows are never
        // read back, so its lanes share them there. The pool seats what a
        // fire names for good, so every fire of the lattice folds into one
        // room: the widest fire's.
        let (slots, room, windowed) = self.synthetic_room();
        // A step arming while the shell serves names the slots it was lent,
        // by ref.
        let lent = self.lent.as_ref();
        let slot_of = |at: u32| match lent {
            Some(lent) => lent.records.get(at as usize).copied().unwrap_or(0),
            None => at,
        };
        let page_of = |at: u64| match lent {
            Some(lent) => lent.pages.get(at as usize).copied().unwrap_or(u32::MAX),
            None => u32::try_from(at).unwrap_or(u32::MAX),
        };
        let (slots, room) = match lent {
            Some(lent) => (
                (lent.records.len() as u32).clamp(1, slots.max(1)),
                (lent.pages.len() as u64).clamp(1, room.max(1)),
            ),
            None => (slots, room),
        };
        let mut next_page = 0u64;
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes) as usize;
        let fold = (self.patch_fold as usize).max(1);
        lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let wants_media = media.get(at).is_some_and(|(images, _)| *images > 0);
                let request = self.representative(class, rows, wants_media);
                Synthetic {
                    word: self.trace.facts.word(&request),
                    tokens: vec![0u32; rows as usize],
                    mask: request
                        .has_custom_mask()
                        .then(|| Masking::Extent(Mask::new(vec![0, rows], u64::from(rows)))),
                    adapter: request.has_adapter().then_some(0),
                    slot: slot_of((at as u32) % slots),
                    window: Vec::new(),
                    pages: {
                        let pages = u64::from(rows).div_ceil(page_size).max(1);
                        let table: Vec<u32> = (next_page..next_page + pages)
                            .map(|page| page_of(page % room))
                            .collect();
                        next_page += pages;
                        table
                    },
                    held: Some(0),
                    media: media
                        .get(at)
                        .copied()
                        .filter(|(images, patches)| *images > 0 && *patches > 0)
                        .map(|(images, patches)| {
                            let patches = patches as usize;
                            let per = patches / images as usize;
                            let mut per_image: Vec<u32> = vec![per as u32; images as usize];
                            per_image[0] += (patches - per * images as usize) as u32;
                            let live = patches / fold;
                            let mut routes = vec![
                                if self.drops_patch_rows {
                                    PATCH_ROUTE_DROP
                                } else {
                                    0
                                };
                                patches
                            ];
                            for (j, route) in routes.iter_mut().take(live).enumerate() {
                                *route = (j % rows.max(1) as usize) as i32;
                            }
                            let grids = per_image
                                .iter()
                                .flat_map(|&rows| [1, rows as i32, 1])
                                .collect();
                            SyntheticMedia {
                                rows: per_image,
                                patches: vec![0u8; patches * row_bytes],
                                routes,
                                positions: vec![0i32; patches * MROPE_COORDS],
                                grids,
                            }
                        }),
                }
            })
            .map(|mut lane| {
                // A synthetic lane's content is never read back, so its
                // pages fold into the windowed pool past the null page.
                let windowed = self
                    .pools
                    .paging()
                    .window_pages()
                    .min(windowed)
                    .saturating_sub(1)
                    .max(1);
                if self.pools.has_windowed() {
                    lane.window = lane
                        .pages
                        .iter()
                        .map(|&page| 1 + u32::try_from(u64::from(page) % windowed).unwrap_or(0))
                        .collect();
                }
                lane
            })
            .collect()
    }

    /// The class a lane of `class` lands in once it carries an adapter: the
    /// one whose requests depart from `class`'s as it does and by the
    /// adapter besides; `None` where the plan lands no such request.
    fn adapted_twin(&self, class: usize) -> Option<usize> {
        let plain = self.landing.get(class)?.first()?;
        if plain.has_adapter() {
            return None;
        }
        self.landing.iter().position(|requests| {
            requests.iter().any(|request| {
                request.has_adapter()
                    && request.has_media() == plain.has_media()
                    && request.drafts() == plain.drafts()
                    && (request.query_len() == 1) == (plain.query_len() == 1)
                    && request.departures() == plain.departures() + 1
            })
        })
    }

    fn representative(&self, class: usize, rows: u32, wants_media: bool) -> poem_ir::Request {
        let landing = &self.landing[class];
        landing
            .iter()
            .find(|request| {
                request.has_media() == wants_media && (request.query_len() == 1) == (rows == 1)
            })
            .or_else(|| {
                landing
                    .iter()
                    .find(|request| request.has_media() == wants_media)
            })
            .or_else(|| landing.first())
            .cloned()
            .unwrap_or_else(|| {
                panic!("the arming pass enumerated class {class}, which no request lands in")
            })
    }

    fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        self.fire_synthetic_as(owned, crate::serve::Golden::Off)
            .map(|_| ())
    }

    fn fire_synthetic_as(
        &mut self,
        owned: &[Synthetic],
        arm: crate::serve::Golden,
    ) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = owned
            .iter()
            .map(|lane| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: lane.word,
                    tokens: &lane.tokens,
                },
                positions: &[],
                pages: &lane.pages,
                held: lane.held,
                kv_less: false,
                translation: &[],
                window: &lane.window,
                more_windows: &[],
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                readout: None,
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
                rs_slot: None,
                rs_read: None,
                rs_snapshots: &[],
                group: None,
                inputs: &[],
                reports: 0,
                tap: None,
            })
            .collect();

        let media: Vec<Media<'_>> = owned
            .iter()
            .enumerate()
            .filter_map(|(at, lane)| {
                lane.media.as_ref().map(|shot| Media {
                    lane: at as u32,
                    rows: &shot.rows,
                    patches: &shot.patches,
                    routes: &shot.routes,
                    positions: &shot.positions,
                    token_positions: &[],
                    grids: &shot.grids,
                })
            })
            .collect();

        self.arming = true;
        self.golden_arm = arm;
        let armed = self.fire_media(&seated, &[], &media);
        self.golden_arm = crate::serve::Golden::Off;
        self.arming = false;
        armed
    }

    fn golden(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "lanes=[{}] {why}",
                owned
                    .iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            let out = shell.fire_synthetic_as(owned, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        if crate::record::ptr_traced(key) {
            crate::record::PTR_TAG.store(2, std::sync::atomic::Ordering::Relaxed);
        }
        let walked = fire(self, Golden::Eager);
        crate::record::PTR_TAG.store(0, std::sync::atomic::Ordering::Relaxed);
        let walked =
            walked.map_err(|fault| refused(format!("the control arm would not fire: {fault}")))?;
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => {
                if super::diag::on().golden_probe {
                    let walked_again = fire(self, Golden::Eager);
                    let replayed_again = fire(self, Golden::Body);
                    let same = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| {
                        evidence(a, b).map_or("identical".to_string(), |why| {
                            why.split("  lanes=").next().unwrap_or(&why).to_string()
                        })
                    };
                    if let Ok(w2) = &walked_again {
                        eprintln!("[golden-probe] {key} walk vs walk: {}", same(&walked, w2));
                    }
                    if let Ok(b2) = &replayed_again {
                        eprintln!("[golden-probe] {key} body vs body: {}", same(&replayed, b2));
                    }
                    let script = self.cache.body_script(key);
                    let execs = script.iter().filter(|(island, ..)| !island).count();
                    let mut culprit = None;
                    for k in 1..=execs {
                        crate::record::REPLAY_UPTO.store(k, std::sync::atomic::Ordering::Relaxed);
                        let partial = fire(self, Golden::Body);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        match partial {
                            Ok(out) => {
                                let agrees = evidence(&walked, &out).is_none();
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): {}",
                                    if agrees {
                                        "agrees with the walk"
                                    } else {
                                        "DIFFERS"
                                    }
                                );
                                if !agrees {
                                    culprit = Some(k);
                                    break;
                                }
                            }
                            Err(fault) => {
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): would not fire: {fault}"
                                );
                                break;
                            }
                        }
                    }
                    if let Some(k) = culprit {
                        let mut seen = 0usize;
                        for (at_step, (island, from, upto)) in script.iter().enumerate() {
                            if *island {
                                continue;
                            }
                            seen += 1;
                            if seen == k {
                                eprintln!(
                                    "[golden-probe] {key} first disagreeing exec is step {at_step}: regions {from}..{upto}"
                                );
                                for region in *from..*upto {
                                    if let Some(template) =
                                        self.compiled.template().get(region as usize)
                                    {
                                        eprintln!(
                                            "[golden-probe]   region {region}: nodes {:?} phase {:?} stream {} lowering {:?}",
                                            template.nodes,
                                            template.phase,
                                            template.stream,
                                            template.lowering
                                        );
                                        for node in template.nodes.clone() {
                                            if let Some(held) = self.trace.nodes.get(node as usize)
                                            {
                                                let op = format!("{:?}", held.op);
                                                let head: String = op.chars().take(90).collect();
                                                eprintln!(
                                                    "[golden-probe]     node {node} layer {:?}: {head}",
                                                    held.layer
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut nth = 0usize;
                    for (at_step, (island, from, upto)) in script.iter().enumerate() {
                        if *island {
                            continue;
                        }
                        let j = nth;
                        nth += 1;
                        crate::record::REPLAY_FROM.store(j, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(j + 1, std::sync::atomic::Ordering::Relaxed);
                        let alone = fire(self, Golden::Body);
                        crate::record::REPLAY_FROM.store(0, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        let verdict = match &alone {
                            Ok(out) => match evidence(&walked, out) {
                                None => "agrees".to_string(),
                                Some(why) => format!(
                                    "DIFFERS ({})",
                                    why.chars().take(120).collect::<String>()
                                ),
                            },
                            Err(fault) => format!("would not fire: {fault}"),
                        };
                        let nodes = self
                            .compiled
                            .template()
                            .get(*from as usize)
                            .map(|t| t.nodes.start)
                            .unwrap_or(0);
                        let nodes_end = self
                            .compiled
                            .template()
                            .get((*upto as usize).saturating_sub(1))
                            .map(|t| t.nodes.end)
                            .unwrap_or(0);
                        let layers: Vec<Option<u32>> = {
                            let mut seen = Vec::new();
                            for n in nodes..nodes_end {
                                if let Some(h) = self.trace.nodes.get(n as usize)
                                    && seen.last() != Some(&h.layer)
                                {
                                    seen.push(h.layer);
                                }
                            }
                            seen
                        };
                        eprintln!(
                            "[golden-probe] {key} exec {j} alone (step {at_step}, regions {from}..{upto}, nodes {nodes}..{nodes_end}, layers {layers:?}): {verdict}"
                        );
                    }
                    for (i, body_lane) in replayed.iter().enumerate() {
                        let best = walked
                            .iter()
                            .enumerate()
                            .map(|(j, walk_lane)| {
                                let agree = body_lane
                                    .iter()
                                    .zip(walk_lane)
                                    .filter(|(x, y)| x.to_bits() == y.to_bits())
                                    .count();
                                (agree, j)
                            })
                            .max()
                            .unwrap_or((0, 0));
                        eprintln!(
                            "[golden-probe] {key} body lane {i} best matches walk lane {} ({} of {} cells bit-exact); head body={:?} walk[{}]={:?} walk[{i}]={:?}",
                            best.1,
                            best.0,
                            body_lane.len(),
                            &body_lane[..body_lane.len().min(4)],
                            best.1,
                            &walked[best.1][..walked[best.1].len().min(4)],
                            &walked[i][..walked[i].len().min(4)],
                        );
                        if i >= 7 {
                            break;
                        }
                    }
                }
                Err(refused(why))
            }
        }
    }

    fn golden_real(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let context = self.pools.paging().context();
        if owned.is_empty()
            || owned.len() > self.held.len()
            || owned
                .iter()
                .any(|lane| lane.tokens.is_empty() || 2 * lane.tokens.len() > context as usize)
        {
            return Ok(());
        }
        let row = u32::from(self.adapters().slots().seats() >= 2);
        let real: Vec<Synthetic> = owned
            .iter()
            .map(|lane| {
                let rows = lane.tokens.len() as u32;
                let have = rows.max(1);
                let extent = have + rows;
                Synthetic {
                    tokens: (0..rows).map(|at| 1 + at % 16).collect(),
                    mask: lane.mask.as_ref().map(|_| {
                        Masking::Rows(
                            (0..rows)
                                .map(|at| {
                                    let allowed = have + at + 1;
                                    let mut runs = vec![0, allowed];
                                    if extent > allowed {
                                        runs.push(extent - allowed);
                                    }
                                    Mask::new(runs, u64::from(extent))
                                })
                                .collect(),
                        )
                    }),
                    adapter: lane.adapter.map(|_| row),
                    held: Some(have),
                    ..lane.clone()
                }
            })
            .collect();
        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "realistic pair: lanes=[{}] {why}",
                real.iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            shell.fire_synthetic_as(owned, Golden::Eager)?;
            let out = shell.fire_synthetic_as(&real, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        let Ok(walked) = fire(self, Golden::Eager) else {
            return Ok(());
        };
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => Err(refused(why)),
        }
    }

    /// Arms the lattice over synthetic lanes. Their reads of the streamed
    /// weights are the shell's, not a fire the runtime asked for: the use
    /// record forgets them, so the runtime's first window orders the cache
    /// by served tokens alone (a load arming at its starting residency arms
    /// after the runtime last read the record).
    /// The lattice's buckets a doubling past `engine.prefill_lattice_from`,
    /// halved until the bodies of `rest` (the narrower prefill, mixed and
    /// wide targets' buckets, still to arm) fit what the allowance has left
    /// each at what the bodies of its kind armed so far weigh (`weighed`,
    /// bytes and bodies a kind; the top bucket arms one of each first, and a
    /// kind's bodies hold about as many nodes at any bucket, where kinds
    /// differ: Qwen3.6-35B-A3B's mixed bodies 8.9 MiB, its lattice's 5.5 MiB
    /// on average): a load whose
    /// allowance arms every key keeps the profile's lattice, and one that
    /// would leave keys to walk eagerly pads its fires to fewer buckets
    /// instead. A coarser default lattice's buckets are the finer one's
    /// every other past the fine rows, so nothing compiled changes. Returns
    /// the buckets dropped; says so, and loudly where even the coarsest
    /// lattice does not fit.
    fn fit_lattice(
        &mut self,
        rest: &[(u32, Kind)],
        kept: usize,
        weighed: &[(usize, usize); Kind::COUNT],
    ) -> std::collections::BTreeSet<u32> {
        let Some((steps, from, mut fine)) = self.lattice else {
            return std::collections::BTreeSet::new();
        };
        let spent = self.cache.body_stats().census;
        if spent.bodies == 0 {
            return std::collections::BTreeSet::new();
        }
        let average = spent.bytes / spent.bodies;
        let price = |kind: Kind| match weighed[kind.at()] {
            (bytes, bodies) if bodies > 0 => bytes / bodies,
            _ => average,
        };
        let was = self.budget.buckets.clone();
        let mut buckets = was.clone();
        let need = |buckets: &[u32]| {
            rest.iter()
                .filter(|(b, _)| buckets.contains(b))
                .map(|&(_, kind)| price(kind))
                .sum::<usize>()
                + kept * average
        };
        // What the allowance has left, or the card where it has less: the
        // ceiling's spare stops arming as the allowance does.
        let allowance = self.bodies_mem.saturating_sub(spent.bytes);
        let spare = self
            .pools
            .spare_bytes_for_bodies(allowance as u64)
            .map_or(allowance, |spare| {
                usize::try_from(spare).unwrap_or(usize::MAX)
            });
        let room = allowance.min(spare);
        while need(&buckets) > room {
            if fine <= steps.max(1) {
                eprintln!(
                    "engine-cuda: WARNING: [engine] bodies_mem {} MiB (or the card) arms no \
                     lattice's every key: at {fine} bucket(s) a doubling past {from} rows the {} \
                     narrower bodies still to arm weigh some {} MiB of the {} MiB left. Raise \
                     bodies_mem or free the card.",
                    self.bodies_mem >> 20,
                    rest.iter().filter(|(b, _)| buckets.contains(b)).count(),
                    need(&buckets) >> 20,
                    room >> 20,
                );
                break;
            }
            fine /= 2;
            buckets = crate::api::default_lattice(self.budget.max_tokens, steps, from, fine);
            buckets.retain(|bucket| was.contains(bucket));
        }
        if buckets.len() < was.len() {
            eprintln!(
                "engine-cuda: [engine] bodies_mem {} MiB (or the card) arms every key at {fine} \
                 bucket(s) a doubling past {from} rows, not the profile's {}: {} bucket(s), not \
                 {} ({} bodies armed at {} KiB on average, {} still to arm at the profile's \
                 lattice, {} MiB left).",
                self.bodies_mem >> 20,
                self.lattice.map_or(0, |(_, _, fine)| fine),
                buckets.len(),
                was.len(),
                spent.bodies,
                average >> 10,
                rest.iter().filter(|(b, _)| was.contains(b)).count(),
                room >> 20,
            );
        }
        let dropped = was
            .iter()
            .filter(|b| !buckets.contains(b))
            .copied()
            .collect();
        self.budget.buckets = buckets;
        dropped
    }

    pub(super) fn arm_bodies(&mut self) -> Result<()> {
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

    /// Every composition the lattice arms, in arming's order (the top
    /// bucket's first, then by kind and bucket), each with its recurrent
    /// rows' treatment; `None` where there is nothing to arm.
    fn lattice_targets(&self) -> Option<LatticeTargets> {
        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return None;
        }
        let points: Vec<LatticePoint> = if self.budget.buckets.is_empty() {
            (1..=ceiling)
                .map(|n| LatticePoint {
                    bucket: n,
                    lanes: n,
                })
                .collect()
        } else {
            let mut points = Vec::new();
            for point in self.budget.buckets.iter().copied() {
                if point <= ceiling {
                    points.push(LatticePoint {
                        bucket: point,
                        lanes: point,
                    });
                } else {
                    if ceiling > points.last().map_or(0, |point| point.bucket) {
                        points.push(LatticePoint {
                            bucket: point,
                            lanes: ceiling,
                        });
                    }
                    break;
                }
            }
            points
        };
        let seats = self.held.len() as u32;
        let context = if self.page_tables == 0 {
            self.budget.max_tokens
        } else {
            self.pools.paging().context()
        };
        let max_lanes = self.fire_lanes();
        let classes = self.compiled.classes.classes.len();
        let textual = |class: usize| !self.media.contains(class);
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
            points,
            buckets: self.budget.buckets.clone(),
            patch_points,
            decoders: self
                .decoding
                .iter()
                .filter(|class| textual(*class) && self.tiers.plain.contains(*class))
                .collect(),
            prefilling: (0..classes)
                .filter(|class| {
                    !self.decoding.contains(*class)
                        && textual(*class)
                        && self.tiers.plain.contains(*class)
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
            drafting: (0..classes)
                .filter(|class| !self.landing[*class].iter().any(poem_ir::Request::plain))
                .collect(),
            rs_rungs: self.rs_rungs(),
            decoding: self.decoding.clone(),
            seats,
            context,
            max_lanes,
            prefill_lanes: self.prefill_lanes,
            patch_fold: self.patch_fold,
            idle: self.ep.is_some(),
        };
        let mut found = Targets::default();
        for enumerate in ARMS {
            enumerate(&deployment, &mut found);
        }
        let Targets {
            mut targets,
            unfireable,
        } = found;
        let top = targets.iter().map(|(bucket, _)| *bucket).max();
        let rank = |kind: Kind| match kind {
            Kind::Decode => 0u8,
            Kind::Mixed => 1,
            Kind::Prefill => 2,
            Kind::Wide => 3,
            Kind::Tower => 4,
            Kind::Idle => 5,
        };
        targets
            .sort_by_key(|(bucket, target)| (Some(*bucket) != top, rank(target.kind()), *bucket));
        if targets.is_empty() {
            return None;
        }
        // A load that buffers recurrent rows serves a spec-decoding fire
        // from a body too: each one-class composition (a verify's lanes, the
        // prompts beside them) is recorded once more for lanes that buffer
        // rows, and once more for lanes that replay them (`record::RsBody`),
        // right after its folding body.
        // Only a load that buffers recurrent rows and serves a drafter's
        // fires (classes that read the drafts) takes them, for the drafter's
        // classes; `rs-bodies` takes them for every class.
        let buffering = self.buffers.is_some();
        let every_class = super::diag::on().rs_bodies;
        let drafting = deployment.drafting.clone();
        let rungs = self.rs_rungs();
        let targets: Vec<(u32, BodySynth, record::RsBody)> = targets
            .into_iter()
            .flat_map(|(bucket, target)| {
                let class = match &target {
                    BodySynth::Decode { class, .. } | BodySynth::Prefill { class, .. } => {
                        Some(*class)
                    }
                    _ => None,
                };
                let buffers = buffering
                    && rungs.contains(&bucket)
                    && class.is_some_and(|class| every_class || drafting.contains(&class));
                let treatments: &[record::RsBody] = if buffers {
                    &[
                        record::RsBody::Folds,
                        record::RsBody::Buffers,
                        record::RsBody::Replays,
                    ]
                } else {
                    &[record::RsBody::Folds]
                };
                treatments
                    .iter()
                    .map(move |rs| (bucket, target.clone(), *rs))
                    .collect::<Vec<_>>()
            })
            .collect();
        Some(LatticeTargets {
            targets,
            top,
            unfireable,
        })
    }

    /// The scratch slabs this load's fires take: each composition of the
    /// lattice fired once, size-only (`crate::sizing`), in arming's order.
    /// Nothing launches, copies or commits; the slabs are laid at what the
    /// walk asked before any fire runs, and none is allocated later.
    pub(super) fn size_arming(&mut self) -> Result<Option<crate::sizing::Footprint>> {
        let Some(LatticeTargets { targets, .. }) = self.lattice_targets() else {
            return Ok(None);
        };
        crate::sizing::begin();
        let mut walked = Ok(());
        // Where the load seats adapters, each composition is walked once
        // more with every lane it can adapt adapted: the lattice names the
        // plain classes, and a lane with an adapter runs kernels they do
        // not (the correction's waist, a slab a lane), which a fire must
        // find laid like any other.
        let adapts = self.adapters().slots().seats() > 0;
        'targets: for (bucket, target, rs) in &targets {
            self.arm_rs = *rs;
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
        }
        let footprint = crate::sizing::end();
        self.arm_rs = record::RsBody::Folds;
        walked.map(|()| Some(footprint))
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
        // The top bucket's bodies and every decode body arm first; the
        // narrower prefill, mixed and wide ones after them are what the
        // lattice coarsens to fit the allowance, each priced at what its
        // kind's bodies armed so far weigh. A shell that arms while it
        // serves arms its rungs smallest first instead, each rung's targets
        // in that order.
        let stepped = self.arm_stepped && !verify && !group;
        if stepped {
            targets.sort_by_key(|(bucket, _, _)| self.rung_of(*bucket));
        }
        Some(Arming {
            rung: targets
                .first()
                .map_or(u32::MAX, |(bucket, _, _)| self.rung_of(*bucket)),
            targets: targets.into(),
            top,
            topped: top.is_none() || !stepped,
            unfireable,
            verify,
            group,
            records,
            priced: verify || group || self.lattice.is_none(),
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
            peer_stopped: false,
            unit_margin: self.pools.map_unit_bytes().saturating_mul(16),
            profiled: self.budget.buckets.len(),
            ballot: Ballot::open(&self.device),
        })
    }

    /// The activation rung a fire of `bucket` rows carves: the smallest
    /// that holds it, `u32::MAX` past every rung stated (the whole arena).
    fn rung_of(&self, bucket: u32) -> u32 {
        self.arena
            .rungs(None)
            .into_iter()
            .map(|(tokens, _)| tokens)
            .find(|tokens| bucket <= *tokens)
            .unwrap_or(u32::MAX)
    }

    fn arm_lattice_fenced(&mut self, verify: bool) -> Result<()> {
        let Some(mut st) = self.arm_begin(verify) else {
            return Ok(());
        };
        // At load: everything, or the rungs through the widest decode
        // fire's (a row a lane) where the rest arm while the shell serves
        // ([`Shell::arm_step`]): a load that is ready decodes at full
        // width, and only a prefill wider than that waits on a step.
        let first = self.rung_of(self.lane_ceiling());
        while let Some(&(bucket, _, _)) = st.targets.front() {
            if self.arm_stepped && self.rung_of(bucket) > first {
                break;
            }
            self.arm_next(&mut st)?;
        }
        if st.targets.is_empty() {
            return self.arm_end(st);
        }
        eprintln!(
            "engine-cuda: bodies armed through rung {first} ({} of {} compositions); {} arm \
             while it serves",
            st.armed,
            st.wanted,
            st.targets.len()
        );
        self.live = engine::live::Live::arming(first);
        self.stepping = Some(st);
        Ok(())
    }

    /// Arms one more composition of a lattice left arming at load, and says
    /// whether more are left. A rung is stated armed ([`engine::live`]) once
    /// its last composition is.
    pub fn arm_step(&mut self, lent: &engine::live::ArmingLease) -> Result<bool> {
        let room = self.arming_room();
        let Some(mut st) = self.stepping.take() else {
            return Ok(false);
        };
        // The records the step's lanes name, each by the slots lent for its
        // pieces: by ref once lanes name refs, else by the state slot those
        // slots are.
        let pieces = self.pools.record_pieces() as usize;
        if lent.records.len() < pieces || lent.pages.len() < room.pages as usize {
            return Err(Fault::program(
                "serve::arming",
                format!(
                    "a step lent {} kv pages and {} record slots, and it writes {} and {}",
                    lent.pages.len(),
                    lent.records.len(),
                    room.pages,
                    room.records
                ),
            ));
        }
        let mut named = Vec::new();
        for record in lent.records.chunks_exact(pieces.max(1)) {
            named.push(if self.pools.by_refs() {
                self.pools.name_record(record)?
            } else {
                record[0] / pieces as u32
            });
        }
        self.lent = Some(engine::live::ArmingLease {
            pages: lent.pages.clone(),
            records: named.clone(),
        });
        let rung = st
            .targets
            .front()
            .map_or(st.rung, |(bucket, _, _)| self.rung_of(*bucket));
        let stepped = self.arm_next(&mut st);
        self.device.synchronize()?;
        self.lent = None;
        if self.pools.by_refs() {
            for key in named {
                self.pools.forget_record(key);
            }
        }
        self.pools.forget_weight_uses(self.device.stream())?;
        stepped?;
        self.arm_stepped_on(st, rung)
    }

    fn arm_stepped_on(&mut self, st: Arming, rung: u32) -> Result<bool> {
        match st
            .targets
            .front()
            .map(|(bucket, _, _)| self.rung_of(*bucket))
        {
            Some(next) => {
                if next != rung {
                    self.live.arm(rung);
                }
                self.stepping = Some(st);
                Ok(true)
            }
            None => {
                // Every rung is stated armed only where every composition
                // the lattice planned holds a body: a fire past that would
                // walk eagerly.
                let whole = st.whole();
                self.arm_end(st)?;
                whole?;
                self.settle()?;
                self.live.arm(u32::MAX);
                Ok(false)
            }
        }
    }

    /// Arms the next target; a rung's targets are priced anew as the rung
    /// starts, at what the bodies armed by then weigh.
    fn arm_next(&mut self, st: &mut Arming) -> Result<()> {
        let Some((bucket, target, rs)) = st.targets.pop_front() else {
            return Ok(());
        };
        let rung = self.rung_of(bucket);
        if self.arm_stepped && rung != st.rung {
            st.rung = rung;
            st.priced = st.verify || st.group || self.lattice.is_none();
        }
        self.arm_target(st, bucket, target, rs)
    }

    fn arm_target(
        &mut self,
        st: &mut Arming,
        bucket: u32,
        target: BodySynth,
        rs: record::RsBody,
    ) -> Result<()> {
        if !st.priced && st.narrower(bucket, &target) {
            st.priced = true;
            // What is still to arm: the narrower targets a coarser lattice
            // may drop, and the rest (the top bucket's, the decodes'), which
            // arm whatever the lattice.
            let rest: Vec<(u32, Kind)> = std::iter::once((bucket, target.kind()))
                .chain(
                    st.targets
                        .iter()
                        .filter(|(bucket, target, _)| st.narrower(*bucket, target))
                        .map(|(bucket, target, _)| (*bucket, target.kind())),
                )
                .collect();
            let kept = st.targets.len() + 1 - rest.len();
            let dropped = self.fit_lattice(&rest, kept, &st.weighed);
            st.dropped.extend(dropped);
        }
        let weighed_before = self.cache.body_stats().census;
        if st.dropped.contains(&bucket) {
            return Ok(());
        }
        self.arm_rs = rs;
        // Past its kind's widest bucket a target captures cold: the top
        // bucket's fires made what its warm fire would.
        self.arm_cold = st.topped && st.narrower(bucket, &target) && !st.verify;
        st.topped |= Some(bucket) == st.top;
        let spent = self.cache.body_stats();
        let room = (self.bodies_mem as u64).saturating_sub(spent.census.bytes as u64);
        let spare = self.pools.spare_bytes_for_bodies(room).unwrap_or(u64::MAX);
        let price = (2 * spent.census.bytes / spent.census.bodies.max(1)) as u64;
        // A map unit's margin never past a quarter of the allowance: an
        // allowance the fit held under the margin would otherwise stop
        // arming before its first body.
        let margin = price.max(st.unit_margin.min(self.bodies_mem as u64 / 4));
        let headroom = spare < margin;
        let mine = !st.verify
            && (spent.census.bodies >= record::MAX_BODIES
                || spent.census.bytes >= self.bodies_mem
                || headroom);
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
        let idle = matches!(target, BodySynth::Idle);
        for _ in 0..record::WARM_FIRES {
            if idle {
                self.at_bucket(bucket);
            }
            let fired = self.fire_synthetic(&owned);
            let landed = self.device.synchronize();
            if let Err(why) = fired.and(landed) {
                if super::diag::on().arm_trace {
                    eprintln!("[arm-trace] refused bucket {bucket}, {target}: {why}");
                }
                // A lattice arming while the shell serves leaves no key to
                // walk eagerly: a refusal there is the step's fault.
                if self.lent.is_some() {
                    return Err(Fault::Unbound {
                        what: format!("arming bucket {bucket}, {target}: {why}"),
                    });
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
            st.armed += 1;
            st.tally[at].0 += 1;
            let now = self.cache.body_stats().census;
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
        eprintln!("engine-cuda: {report}");
        // The lattice the bodies armed over against the profile's: fewer
        // buckets is the allowance binding, each fire padded further.
        eprintln!(
            "engine-cuda: lattice armed over {} of the profile's {} buckets",
            self.budget.buckets.len(),
            st.profiled
        );
        // A load that records bodies serves every composition it fires from
        // one: one that armed none, or stopped arming for want of room, would
        // serve eagerly, which is never what a deployment asked for.
        let attempted = st.wanted + st.never;
        if st.records
            && !st.verify
            && !st.group
            && attempted > 0
            && (st.armed == 0 || st.starved.is_some())
        {
            return Err(Fault::Residency(format!(
                "the card does not hold this deployment's graph bodies: {} of {attempted} \
                 compositions armed{}. A load that records bodies serves every composition \
                 from one rather than eagerly. Lower `[engine] max_context`, raise \
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

fn evidence(walked: &[Vec<f32>], replayed: &[Vec<f32>]) -> Option<String> {
    fn ordered(bits: u32) -> u32 {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }

    if walked.len() != replayed.len() {
        return Some(format!(
            "the two arms answered {} and {} lanes  class=structural",
            walked.len(),
            replayed.len(),
        ));
    }
    let mut first: Option<(usize, f32, f32)> = None;
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut worst = 0u32;
    let mut worst_bf16 = 0u32;
    let mut worst_bf16_at: Option<(usize, usize, f32, f32)> = None;
    let mut beyond_one_bf16 = 0usize;
    let mut beyond_eight_bf16 = 0usize;
    let mut non_finite = (0usize, 0usize);
    let mut per_lane: Vec<usize> = Vec::with_capacity(walked.len());
    for (lane, (a, b)) in walked.iter().zip(replayed).enumerate() {
        per_lane.push(0);
        if a.len() != b.len() {
            return Some(format!(
                "lane {lane} answered {} and {} elements  class=structural",
                a.len(),
                b.len(),
            ));
        }
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            total += 1;
            if x.to_bits() == y.to_bits() {
                continue;
            }
            differing += 1;
            per_lane[lane] += 1;
            non_finite.0 += usize::from(!x.is_finite());
            non_finite.1 += usize::from(!y.is_finite());
            worst = worst.max(ordered(x.to_bits()).abs_diff(ordered(y.to_bits())));
            let steps = ordered(x.to_bits() & 0xffff_0000)
                .abs_diff(ordered(y.to_bits() & 0xffff_0000))
                >> 16;
            beyond_one_bf16 += usize::from(steps > 1);
            beyond_eight_bf16 += usize::from(steps > 8);
            if steps > worst_bf16 {
                worst_bf16 = steps;
                worst_bf16_at = Some((lane, at, *x, *y));
            }
            if first.is_none() {
                first = Some((at, *x, *y));
            }
        }
    }
    let (at, x, y) = first?;
    let class = if worst <= 2 { "numeric" } else { "structural" };
    let lane_runs = {
        let mut runs: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < per_lane.len() {
            let differs = per_lane[start] > 0;
            let mut end = start;
            while end < per_lane.len() && (per_lane[end] > 0) == differs {
                end += 1;
            }
            let width = walked[start].len().max(1);
            let mean = per_lane[start..end].iter().sum::<usize>() as f64
                / ((end - start) as f64 * width as f64);
            runs.push(if differs {
                format!("{start}..{end} differ ({:.0}% of cells)", mean * 100.0)
            } else {
                format!("{start}..{end} agree")
            });
            start = end;
        }
        runs.join(", ")
    };
    let worst_bf16_line = worst_bf16_at.map_or(String::new(), |(lane, at, x, y)| {
        format!("  worst_bf16=lane {lane} #{at} (walk={x}, body={y}, {worst_bf16} bf16 step(s))")
    });
    Some(format!(
        "differs at #{at} (walk={x} {:#010x}, body={y} {:#010x})  n_diff={differing}/{total}  \
         max_ulp={worst}  class={class}  beyond_1_bf16={beyond_one_bf16}  \
         beyond_8_bf16={beyond_eight_bf16}  non_finite(walk,body)={non_finite:?}  \
         lanes=[{lane_runs}]{worst_bf16_line}",
        x.to_bits(),
        y.to_bits(),
    ))
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
                    None => "[engine] bodies_mem".to_string(),
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

/// What [`Shell::lattice_targets`] enumerates: each composition with its
/// bucket and recurrent rows' treatment, the top bucket, and the keys no
/// synthetic fire reaches.
struct LatticeTargets {
    targets: Vec<(u32, BodySynth, record::RsBody)>,
    top: Option<u32>,
    unfireable: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice(armed: usize, wanted: usize, never: usize) -> Arming {
        Arming {
            targets: std::collections::VecDeque::new(),
            rung: u32::MAX,
            top: None,
            topped: true,
            unfireable: Vec::new(),
            verify: false,
            group: false,
            records: true,
            priced: true,
            dropped: std::collections::BTreeSet::new(),
            weighed: [(0, 0); Kind::COUNT],
            armed,
            wanted,
            tally: [(0, 0); Kind::COUNT],
            refused: None,
            never,
            warmed_only: 0,
            never_from: 0,
            belted: false,
            starved: None,
            peer_stopped: false,
            unit_margin: 0,
            profiled: 0,
            ballot: Ballot { slip: None },
        }
    }

    #[test]
    fn a_lattice_is_whole_only_where_every_planned_composition_holds_a_body() {
        assert!(lattice(178, 178, 0).whole().is_ok());
        // A composition fired and refused its body.
        assert!(lattice(177, 178, 0).whole().is_err());
        // Compositions the allowance stopped before their fire are planned
        // too: a fire at one would walk eagerly.
        let short = lattice(41, 41, 137).whole().unwrap_err().to_string();
        assert!(short.contains("41 of 178"), "{short}");
    }
}
