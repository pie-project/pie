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
/// What a body is armed for beyond its composition: how its lanes treat
/// their recurrent rows. (A lane's attention mask, a group's too, is data
/// one body reads: no treatment.)
#[derive(Debug, Clone, Copy)]
struct Treatment {
    rs: record::RsBody,
}

/// A composition captured during arming and held: its rank by the user's
/// rule, its key, where it came from, and what it weighed.
struct Held {
    rank: (u32, core::cmp::Reverse<Option<u64>>, core::cmp::Reverse<usize>, u8),
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

    fn synthetic_lanes_with(&self, lanes: &[LaneRows], media: &[LaneMedia]) -> Vec<Synthetic> {
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        // What the lanes name, inside the chunks the fit left the pool: a
        // stated kv budget may hold fewer records and pages than the
        // lattice's widest fire names, and a synthetic lane's rows are never
        // read back, so its lanes share them there. The pool seats what a
        // fire names for good, so every fire of the lattice folds into one
        // room: the widest fire's.
        let (slots, room, windowed) = self.synthetic_room();
        let slot_of = |at: u32| at;
        let page_of = |at: u64| u32::try_from(at).unwrap_or(u32::MAX);
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

    fn seated(owned: &[Synthetic]) -> Vec<Seated<'_>> {
        owned
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
                folded: 0,
                readable: 0,
                set: 0,
                kv_less: false,
                translation: &[],
                write_end: 0,
                window: &lane.window,
                window_fold: None,
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
            .collect()
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
        let seated = Self::seated(owned);

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
    pub(super) fn routes_by_body(
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
            let entries: Vec<_> = times.iter().map(|&(kernel, ns, spread)| (shape, kernel, ns, spread)).collect();
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
    fn time_decode_body(&mut self, m: u32) -> Result<(f64, f64)> {
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
        // Captured in this order (the top bucket's compositions first, then
        // by kind and bucket): the order bodies are captured in leaves
        // process-wide state behind that every later graph's kernel-to-kernel
        // transitions pay for (an L40S decode at one lane ran 3.4% slower
        // after a capture across kinds by rows), so it is kept apart from
        // which compositions the allowance keeps: that is the user's rule
        // (`Shell::rule_rank`), applied by giving back the worst-ranked
        // capture whenever the bodies pass their allowance (`arm_target`).
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
        // prompts beside them) at the rs rungs is recorded once more for
        // lanes that buffer rows, and once more for lanes that replay them
        // (`record::RsBody`), right after its folding body. Every class takes
        // them, not only a drafter's: a guest that drafts for itself (MTP,
        // with no drafts output the load can see) verifies on the plain
        // classes, and it registers after the load arms.
        let buffering = self.buffers.is_some();
        let rungs = self.rs_rungs();
        let targets: Vec<(u32, BodySynth, Treatment)> = targets
            .into_iter()
            .flat_map(|(bucket, target)| {
                let class = match &target {
                    BodySynth::Decode { class, .. } | BodySynth::Prefill { class, .. } => {
                        Some(*class)
                    }
                    _ => None,
                };
                let buffers = buffering && rungs.contains(&bucket) && class.is_some();
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
                    .map(move |&rs| (bucket, target.clone(), Treatment { rs }))
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
        let growth = poem_compiler::arena::voxel_growth(&self.trace, &self.compiled.classes);
        let clips = u32::try_from(seat.clips.max(1)).unwrap_or(u32::MAX);
        let row_bytes = seat.payload_bytes() / seat.rows.max(1);
        for (class, (grown, read)) in growth.into_iter().enumerate() {
            if self.landing[class].is_empty() {
                continue;
            }
            let rows = seat.rows * u64::from(read) / u64::from(grown.max(1));
            let rows = u32::try_from(rows).unwrap_or(u32::MAX);
            // A clip's box divides by the plan's token patch, as a fire's
            // clips must: each clip is whole patches (`[pt, ph, pw]`, a
            // single voxel without one) laid along t, the first taking what
            // the even split leaves, at most the seat's clips of them.
            let [pt, ph, pw] = seat.token_patch.unwrap_or([1, 1, 1]).map(|p| p.max(1));
            let patch = pt.saturating_mul(ph).saturating_mul(pw);
            let patches = rows / patch;
            let clips = clips.min(patches);
            if clips == 0 {
                continue;
            }
            let each = patches / clips;
            let boxes: Vec<[u32; 3]> = (0..clips)
                .map(|at| {
                    let units = each + if at == 0 { patches % clips } else { 0 };
                    [units * pt, ph, pw]
                })
                .collect();
            let rows = patches * patch;
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
                    "the size-only walk of a voxel fire, class {class} over {clips} clip(s) of \
                     {rows} row(s): {why}"
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
        order.sort_by_key(|&at| self.rule_rank(targets[at].0, &targets[at].1));
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
            eprintln!(
                "engine-cuda: arming selects {kept} of {total} compositions by the user's rule: \
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
    /// the least kept first: by rows across every kind, the fewest first
    /// (what goes eager is always the largest rungs); within one row count
    /// the one that launches more kernels first (more launches, more an
    /// eager fire of it costs), as the size-only walk counted them
    /// (`walked_launches`), else by its classes; then by kind.
    fn rule_rank(
        &self,
        bucket: u32,
        target: &BodySynth,
    ) -> (u32, core::cmp::Reverse<Option<u64>>, core::cmp::Reverse<usize>, u8) {
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
            let rank = self.rule_rank(bucket, &target);
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
                    rank: self.rule_rank(bucket, &target),
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
        eprintln!(
            "engine-cuda: {report}; armed in {:.1} s",
            st.began.elapsed().as_secs_f64()
        );
        // Every composition arming stopped before (the report above says
        // why), by name: they walk eagerly for the life of the load by this
        // statement, their fires tallied as such (`BodyTally::stated_eager`).
        let stated_now = self.cache.stated_eager();
        if !stated_now.is_empty() && self.arm_only.is_none() {
            eprintln!(
                "engine-cuda: {} composition key(s) walk eagerly by statement, arming stopped \
                 before them from bucket {}: {}",
                stated_now.len(),
                st.never_from,
                stated_now.join(", ")
            );
        }
        // The lattice the bodies armed over against the profile's: fewer
        // buckets is the allowance binding, each fire padded further.
        eprintln!(
            "engine-cuda: lattice armed over {} of the profile's {} buckets",
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
                Err(why) => eprintln!(
                    "engine-cuda: WARNING: the bodies' record at {} is not kept: {why}",
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

/// What [`Shell::lattice_targets`] enumerates: each composition with its
/// bucket and recurrent rows' treatment, the top bucket, and the keys no
/// synthetic fire reaches.
struct LatticeTargets {
    targets: Vec<(u32, BodySynth, Treatment)>,
    top: Option<u32>,
    unfireable: Vec<String>,
}

/// The composition a product of `m` rows is timed in
/// (`Shell::routes_by_body`), with what rides it (its recurrent rows'
/// treatment): a body the lattice armed at bucket `m` that carries `m`
/// rows, the decode step's (`m` one-row lanes) where the lattice armed one,
/// else the first that does (where a fire carries fewer lanes than `m`, an
/// artifact baked for fewer, its rows are a prompt's). `None` where no
/// armed body carries them.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
fn timing_target<'a, R: Copy>(
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
/// kept, the interquartile range the spread a choice must clear.
const BODY_ROUNDS: usize = if cfg!(feature = "rocm") { 9 } else { 5 };

/// Fires a round of a decode body's timing: each replayed and waited on, as
/// a serving step is.
const BODY_FIRES: usize = if cfg!(feature = "rocm") { 20 } else { 10 };

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
    use super::{BodySynth, timing_target};

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
