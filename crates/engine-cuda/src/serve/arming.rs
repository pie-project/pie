//! The bodies arming pass: enumerates the key space this deployment's traffic
//! can realize, fires one synthetic composition per key, then seals the body
//! map. Runs once, from the tail of [`Shell::load`], on the load thread,
//! through the ordinary fire path with `Shell::arming` set so that nobody's
//! numbers are read and no expert is promoted. Nothing it does can fail the
//! load except [`Shell::golden`].
//!
//! [`Kind`] names the six composition kinds; [`ARMS`] is one enumerator per
//! kind, and the tally and the boot line iterate [`Kind::ALL`]. A seventh
//! kind costs a `BodySynth` variant, a `Kind`, an enumerator and one entry.

use engine::fire::{Mask, Masking, RsReset, RsVerb};
use model_compiler::CompiledModel;

use crate::error::{Fault, Result};
use crate::record;

use super::{Lane, MROPE_COORDS, Media, PATCH_ROUTE_DROP, Seated, Shell};

/// One lane of a synthetic composition — the owned side of a [`Seated`] an
/// arming pass borrows. Its launches are real even though it computes and
/// reads back nothing.
#[derive(Clone)]
struct Synthetic {
    /// The class's representative word (`Class::word`).
    word: u64,
    tokens: Vec<u32>,
    /// An all-allowed mask, always `Masking::Extent`.
    mask: Option<Masking>,
    /// Adapter row 0, for a class inside the correction's window.
    adapter: Option<u32>,
    drafts: bool,
    captures: bool,
    /// Which real slot lends its page arithmetic.
    slot: u32,
    /// [`Seated::held`]. `Some(0)` for every enumerated lane; only
    /// [`Shell::golden_real`] writes anything else.
    held: Option<u32>,
    /// `(images, patches)` for a lane carrying an image; the payload is
    /// zero bytes, since only the geometry has to be plausible.
    media: Option<SyntheticMedia>,
}

/// The six vectors one synthetic image submission owns. Zeroed except
/// where a zero would not be a plausible address: `routes` stay in bounds
/// (modulo the lane's rows), and `embed_weights` is nearest-neighbour.
#[derive(Clone)]
struct SyntheticMedia {
    /// Patch rows per image, summing to the fire's patch total.
    rows: Vec<u32>,
    /// `patch_rows x row_bytes` of zeros.
    patches: Vec<u8>,
    /// One token row per tower output row, then the fold's dead tail.
    routes: Vec<i32>,
    /// `(t, h, w)` per patch row — the grid's origin.
    positions: Vec<i32>,
    /// Row 0, `embed_taps` times per patch row.
    embed_rows: Vec<i32>,
    /// Nearest-neighbour: `1.0` then zeros.
    embed_weights: Vec<f32>,
}

/// One key the bodies arming means to climb, as a geometry: the six
/// present-set shapes `Shell::arm_bodies` enumerates, each carrying the
/// lanes it will synthesize.
#[derive(Debug, Clone)]
enum BodySynth {
    /// One decode class, `lanes` lanes of one row each.
    Decode { lanes: u32, class: usize },
    /// One non-decode class, the bucket's rows spread over its lanes.
    Prefill { class: usize, rows: Vec<u32> },
    /// One decode lane beside one non-decode class's lanes.
    Mixed {
        decode: usize,
        class: usize,
        rows: Vec<u32>,
    },
    /// A present set that puts a foreign class's rows inside some region's
    /// window — the composition a segmented body exists for. One lane per
    /// class, ascending; a decode class takes exactly one row.
    Fragmented { lanes: Vec<(usize, u32)> },
    /// A fire that carries an image. One lane, one media class, one rung
    /// of the patch lattice. `images` is always one.
    Tower {
        class: usize,
        rows: u32,
        images: u32,
        patches: u32,
    },
    /// Every decode word this load has, present at once — what a hybrid
    /// SKU's decode traffic actually brings, and [`BodySynth::Decode`]'s
    /// singleton cannot name. One row per lane, split across the words.
    Ensemble { lanes: Vec<(usize, u32)> },
}

impl BodySynth {
    /// The present set: which classes this synthetic puts rows in,
    /// ascending and deduplicated.
    fn present(&self) -> Vec<usize> {
        let mut classes = match self {
            BodySynth::Decode { class, .. } | BodySynth::Prefill { class, .. } => vec![*class],
            BodySynth::Mixed { decode, class, .. } => vec![*decode, *class],
            BodySynth::Fragmented { lanes } | BodySynth::Ensemble { lanes } => {
                lanes.iter().map(|(class, _)| *class).collect()
            }
            BodySynth::Tower { class, .. } => vec![*class],
        };
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    fn kind(&self) -> Kind {
        match self {
            BodySynth::Decode { .. } => Kind::Decode,
            BodySynth::Prefill { .. } => Kind::Prefill,
            BodySynth::Mixed { .. } => Kind::Mixed,
            BodySynth::Fragmented { .. } => Kind::Fragmented,
            BodySynth::Tower { .. } => Kind::Tower,
            BodySynth::Ensemble { .. } => Kind::Ensemble,
        }
    }

    /// Does a verdict about this target speak for its present set? `false`
    /// for a tower target, whose second (patch) rectangle a text key's
    /// verdict says nothing about.
    fn skips_on_present_set(&self) -> bool {
        !matches!(self, BodySynth::Tower { .. })
    }

    /// Token lanes as `(class, rows)` pairs, plus an `(images, patches)`
    /// pair per lane for the one arm that carries an image.
    fn lanes(&self) -> (Vec<(usize, u32)>, Vec<(u32, u32)>) {
        match self {
            BodySynth::Decode { lanes, class } => {
                (vec![(*class, 1u32); *lanes as usize], Vec::new())
            }
            BodySynth::Prefill { class, rows } => (
                rows.iter().map(|rows| (*class, *rows)).collect(),
                Vec::new(),
            ),
            // Decode lane first: submission order does not affect the ladder.
            BodySynth::Mixed { decode, class, rows } => (
                core::iter::once((*decode, 1u32))
                    .chain(rows.iter().map(|rows| (*class, *rows)))
                    .collect(),
                Vec::new(),
            ),
            BodySynth::Fragmented { lanes } => (lanes.clone(), Vec::new()),
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => (vec![(*class, *rows)], vec![(*images, *patches)]),
            // Each decode word's lane count expanded to that many one-row lanes.
            BodySynth::Ensemble { lanes } => (
                lanes
                    .iter()
                    .flat_map(|(class, count)| vec![(*class, 1u32); *count as usize])
                    .collect(),
                Vec::new(),
            ),
        }
    }
}

/// Which of the six composition kinds a target is, and the word the boot
/// line spells that column with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// [`BodySynth::Decode`].
    Decode,
    /// [`BodySynth::Prefill`].
    Prefill,
    /// [`BodySynth::Mixed`].
    Mixed,
    /// [`BodySynth::Fragmented`].
    Fragmented,
    /// [`BodySynth::Tower`].
    Tower,
    /// Its own column rather than folded into `decode`: a short `decode`
    /// count means seats under the lattice, a short `ensemble` count means
    /// hybrid decode traffic has no key at all.
    Ensemble,
}

impl Kind {
    /// The tally's width, and [`ARMS`]'s.
    pub const COUNT: usize = 6;

    /// In the order the boot line and [`ARMS`] use; [`Kind::at`] indexes
    /// the tally by discriminant.
    pub const ALL: [Kind; Kind::COUNT] = [
        Kind::Decode,
        Kind::Prefill,
        Kind::Mixed,
        Kind::Fragmented,
        Kind::Tower,
        Kind::Ensemble,
    ];

    /// Its slot in a per-kind tally.
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
            Kind::Fragmented => "fragmented",
            Kind::Tower => "tower",
            Kind::Ensemble => "ensemble",
        })
    }
}

/// One decode lattice point: a token bucket, and the lane count a real
/// decode fire of that bucket can bring on this load's seats.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticePoint {
    pub bucket: u32,
    pub lanes: u32,
}

/// Every number the six enumerations read, taken once at the top of
/// [`Shell::arm_bodies`] — a struct rather than a borrowed `self` so an
/// enumerator can only read facts a deployment states.
struct Deployment {
    points: Vec<LatticePoint>,
    /// `Budget::buckets`, the token lattice, ascending.
    buckets: Vec<u32>,
    /// Empty for a deployment that cannot stage a patch row at all.
    patch_points: Vec<u32>,
    /// The decode classes a text synthetic may land in.
    decoders: Vec<usize>,
    /// The non-decode classes a text synthetic may land in.
    prefilling: Vec<usize>,
    /// The classes whose window runs the embed merge.
    media: Vec<usize>,
    /// The minimal present sets that break a window ([`Shell::fragmenting`]).
    fragmenting: Vec<Vec<usize>>,
    /// Same classes as `decoders`, as a `ClassSet` for membership tests.
    decoding: model_ir::ClassSet,
    seats: u32,
    /// KV tokens one seat holds.
    context: u32,
    max_lanes: u32,
    /// Patch rows the tower folds into one soft token.
    patch_fold: u32,
}

/// What the enumeration produces: the keys to attempt, and the ones this
/// deployment cannot synthesize at all, named.
#[derive(Default)]
struct Targets {
    targets: Vec<(u32, BodySynth)>,
    /// One sentence per key that was named rather than fired.
    unfireable: Vec<String>,
}

/// One "this deployment cannot fire that key" sentence, shared by every
/// enumerating arm.
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

/// One decode key per lattice point per decode class, at the lane count a
/// real decode fire of that rung can bring.
fn decode_keys(deployment: &Deployment, into: &mut Targets) {
    for LatticePoint { bucket, lanes } in deployment.points.iter().copied() {
        for class in deployment.decoders.iter().copied() {
            into.targets
                .push((bucket, BodySynth::Decode { lanes, class }));
        }
    }
}

/// One prefill key per lattice point per non-decode class — the bucket's
/// own row total spread over `min(bucket, seats, max_lanes)` lanes.
fn prefill_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for class in deployment.prefilling.iter().copied() {
            match Shell::spread(
                point,
                deployment.seats.min(deployment.max_lanes),
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

/// One mixed key per (decode class x non-decode class) pair per lattice
/// point — one decode lane of one row, and the remaining `bucket - 1` rows
/// spread over prefill lanes.
fn mixed_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for decode in deployment.decoders.iter().copied() {
            for class in deployment.prefilling.iter().copied() {
                // A mixed fire needs a seat for the decode lane and at least
                // one for the prefill class, and a bucket with a row for
                // each: two of everything, and `spread` refuses the rest.
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
                    Some(rows) => into.targets.push((point, BodySynth::Mixed {
                        decode,
                        class,
                        rows,
                    })),
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

/// The compositions a segmented body exists for ([`BodySynth::Fragmented`]).
fn fragmented_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for present in &deployment.fragmenting {
            match fragment_rows(deployment, present, point) {
                Some(lanes) => {
                    into.targets.push((point, BodySynth::Fragmented { lanes }));
                }
                None => into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("fragmented {present:?}"),
                    &format!("bucket {point}"),
                    None,
                )),
            }
        }
    }
}

/// The second capture unit's own lattice ([`BodySynth::Tower`]): one key
/// per (media class, patch rung, token bucket). Only one media class per
/// fire is armed. Enumerates nothing on a text-only SKU.
fn tower_keys(deployment: &Deployment, into: &mut Targets) {
    for patches in deployment.patch_points.iter().copied() {
        for class in deployment.media.iter().copied() {
            for point in deployment.buckets.iter().copied() {
                // Placeholder rows are owed: the tower folds `patch_fold`
                // patch rows into one soft token.
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
                into.targets.push((point, BodySynth::Tower {
                    class,
                    rows: point,
                    images: 1,
                    patches,
                }));
            }
        }
    }
}

/// One key per lattice point for the whole decode set
/// ([`BodySynth::Ensemble`]). Nothing on a bake with fewer than two decode
/// words: the singleton is the full set there ([`decode_keys`]).
fn ensemble_keys(deployment: &Deployment, into: &mut Targets) {
    let words = deployment.decoders.len() as u32;
    if words < 2 {
        return;
    }
    for LatticePoint { bucket, lanes } in deployment.points.iter().copied() {
        match ensemble_lanes(deployment, lanes) {
            Some(lanes) => {
                into.targets.push((bucket, BodySynth::Ensemble { lanes }));
            }
            None => into.unfireable.push(unfireable_line(
                deployment,
                &format!("ensemble {:?}", deployment.decoders),
                &format!("bucket {bucket}"),
                Some(&format!("{words} decode word(s) in {lanes} lane(s)")),
            )),
        }
    }
}

/// `lanes` lanes split evenly over every decode word, one row each, or
/// `None` for a rung that cannot seat a lane per word. Every word gets at
/// least one lane, or it would arm the singleton key under this name instead.
fn ensemble_lanes(deployment: &Deployment, lanes: u32) -> Option<Vec<(usize, u32)>> {
    let words = deployment.decoders.len() as u32;
    if words < 2 || lanes < words {
        return None;
    }
    let base = lanes / words;
    let over = lanes % words;
    Some(
        deployment
            .decoders
            .iter()
            .copied()
            .enumerate()
            .map(|(at, class)| (class, base + u32::from((at as u32) < over)))
            .collect(),
    )
}

/// The whole key space, one function per kind, in [`Kind::ALL`]'s order. A
/// seventh composition kind is a [`BodySynth`] variant, a [`Kind`], an
/// enumerator and one entry here — nothing else.
const ARMS: [fn(&Deployment, &mut Targets); Kind::COUNT] = [
    decode_keys,
    prefill_keys,
    mixed_keys,
    fragmented_keys,
    tower_keys,
    ensemble_keys,
];

impl core::fmt::Display for BodySynth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodySynth::Decode { lanes, class } => write!(f, "decode c{class} x{lanes}"),
            BodySynth::Prefill { class, rows } => {
                write!(f, "prefill c{class} {rows:?}")
            }
            BodySynth::Mixed { decode, class, rows } => {
                write!(f, "mixed c{decode}+c{class} {rows:?}")
            }
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => write!(f, "tower c{class} {rows}r x{images} img {patches}p"),
            BodySynth::Fragmented { lanes } => {
                write!(f, "fragmented ")?;
                for (at, (class, rows)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class}:{rows}")?;
                }
                Ok(())
            }
            BodySynth::Ensemble { lanes } => {
                write!(f, "ensemble ")?;
                for (at, (class, count)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class} x{count}")?;
                }
                Ok(())
            }
        }
    }
}

impl Shell {
    /// One synthetic composition's lanes, from a list of `(class, rows)`
    /// pairs. A pair may repeat its class — `n` lanes of one class is
    /// exactly a decode batch.
    fn synthetic_lanes(&self, lanes: &[(usize, u32)]) -> Vec<Synthetic> {
        self.synthetic_lanes_with(lanes, &[])
    }

    /// The same lanes, with images on some of them — the tower arm's door.
    /// `media[i]` is `(images, patch rows)` for lane `i`, `(0, 0)` or absent
    /// for a lane carrying none. Patch rows are spread evenly over the
    /// images, remainder on the first: an even split is always plausible,
    /// where an empty second image is a submission `Fault::PatchGeometry`
    /// refuses.
    fn synthetic_lanes_with(
        &self,
        lanes: &[(usize, u32)],
        media: &[(u32, u32)],
    ) -> Vec<Synthetic> {
        let slots = self.held.len().max(1) as u32;
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes) as usize;
        let taps = self.patch_seat.map_or(0, |seat| seat.embed_taps) as usize;
        let weight_taps = self
            .patch_seat
            .map_or(0, |seat| if seat.embed_weights { seat.embed_taps } else { 0 })
            as usize;
        let fold = (self.patch_fold as usize).max(1);
        lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let wants_media = media.get(at).is_some_and(|(images, _)| *images > 0);
                let request = self.representative(class, rows, wants_media);
                Synthetic {
                word: (self.classify)(&request),
                tokens: vec![0u32; rows as usize],
                mask: request.has_custom_mask().then(|| {
                    Masking::Extent(Mask::new(vec![0, rows], u64::from(rows)))
                }),
                adapter: request.has_adapter().then_some(0),
                drafts: request.drafts(),
                captures: request.captures_scores(),
                // Real slots, round-robin: the page arithmetic needs a slot
                // that exists.
                slot: (at as u32) % slots,
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
                        // The tower's live output; rows past it are the dead tail.
                        let live = patches / fold;
                        let mut routes = vec![
                            if self.drops_patch_rows { PATCH_ROUTE_DROP } else { 0 };
                            patches
                        ];
                        for (j, route) in routes.iter_mut().take(live).enumerate() {
                            *route = (j % rows.max(1) as usize) as i32;
                        }
                        let mut embed_weights = vec![0f32; patches * weight_taps];
                        for row in embed_weights.chunks_mut(weight_taps.max(1)) {
                            if let Some(first) = row.first_mut() {
                                *first = 1.0;
                            }
                        }
                        SyntheticMedia {
                            rows: per_image,
                            patches: vec![0u8; patches * row_bytes],
                            routes,
                            positions: vec![0i32; patches * MROPE_COORDS],
                            embed_rows: vec![0i32; patches * taps],
                            embed_weights,
                        }
                    }),
                }
            })
            .collect()
    }

    /// The fewest-flag request that lands in `class`, matching the lane's
    /// row kind and whether it carries images.
    fn representative(&self, class: usize, rows: u32, wants_media: bool) -> model_ir::Request {
        let landing = &self.landing[class];
        landing
            .iter()
            .find(|request| {
                request.has_media() == wants_media && (request.query_len() == 1) == (rows == 1)
            })
            .or_else(|| landing.iter().find(|request| request.has_media() == wants_media))
            .or_else(|| landing.first())
            .copied()
            .unwrap_or_else(|| panic!("the arming pass enumerated class {class}, which no request lands in"))
    }

    /// Fire one synthetic composition, with [`Shell::arming`] set, landing
    /// in `record::Bodies::fire_body` exactly as a caller's fire would.
    ///
    /// # Errors
    ///
    /// Whatever the synthetic fire refused — staging, a planner, the
    /// capture, the instantiate. The caller tallies the sentence; nothing is
    /// retried.
    fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        self.fire_synthetic_as(owned, crate::serve::Golden::Off).map(|_| ())
    }

    /// [`fire_synthetic`](Shell::fire_synthetic), told which half of the
    /// golden pass it is, answering the readout the two halves are compared
    /// on. `Golden::Off` answers an empty `Vec`, since a synthetic plans no
    /// readback.
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
                pages: &[],
                // A synthetic owns every row it reads: `Some(0)` for every
                // enumerated lane, begins its slot's sequence.
                // [`Shell::golden_real`] is the one caller that states
                // anything else.
                held: lane.held,
                // The arming pass resolves no port, so it crosses no space.
                translation: &[],
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                drafts: lane.drafts,
                captures_scores: lane.captures,
                // The arming pass computes nobody's numbers, so there is no
                // row list to read back.
                readout: None,
                // The plain fold is the one RS shape that graph-replays, so
                // it is the only one a body can be armed for.
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
            })
            .collect();

        // One `Media` per lane that carries images, keyed by lane exactly as
        // a caller's submission is. Empty for every text-only load.
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
                    // Scalar rope: a synthetic has no grid to state.
                    token_positions: &[],
                    embed_rows: &shot.embed_rows,
                    embed_weights: &shot.embed_weights,
                })
            })
            .collect();

        self.arming = true;
        self.golden_arm = arm;
        let armed = self.fire_media(&seated, &[], &media, &mut Vec::new());
        self.golden_arm = crate::serve::Golden::Off;
        self.arming = false;
        armed
    }

    /// Does this key's body answer what the walk answers? Fires the key's
    /// synthetic twice from one reset state — walking once, replaying once
    /// — and diffs the readout bit for bit; the only verdict in this pass
    /// that fails the load. Blind to geometry on purpose; not blind to a
    /// composition this pass should never have synthesized.
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
        let walked = fire(self, Golden::Eager)
            .map_err(|fault| refused(format!("the control arm would not fire: {fault}")))?;
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        // Bit for bit, not `close`: no accumulation order to differ by.
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => Err(refused(why)),
        }
    }

    /// [`Shell::golden`]'s second pair, over a composition a caller could
    /// actually have brought: each slot is first fired as a sequence
    /// beginning (leaving real kv behind), and the measured fire claims
    /// that history instead of the golden pair's frozen constants. A key
    /// it cannot compose, or whose control arm refuses, is skipped in
    /// silence.
    fn golden_real(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let context = self.pools.paging().context();
        // A seat per lane, and a seat that holds the history beside the rows.
        if owned.is_empty()
            || owned.len() > self.held.len()
            || owned
                .iter()
                .any(|lane| lane.tokens.is_empty() || 2 * lane.tokens.len() > context as usize)
        {
            return Ok(());
        }
        // Row 1 where the bank has one: an unbound row is zeros, same as row 0.
        let row = u32::from(self.adapters().slots().seats() >= 2);
        let real: Vec<Synthetic> = owned
            .iter()
            .map(|lane| {
                let rows = lane.tokens.len() as u32;
                let have = rows.max(1);
                let extent = have + rows;
                Synthetic {
                    // Non-zero and small: this shell has no vocabulary to
                    // bound an id against.
                    tokens: (0..rows).map(|at| 1 + at % 16).collect(),
                    // Causal, per row — the mask a prefill actually brings.
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
            // Writes the history: a sequence beginning leaves `have` rows of
            // real kv in each slot for the measured fire below to read.
            shell.fire_synthetic_as(owned, Golden::Eager)?;
            let out = shell.fire_synthetic_as(&real, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        // A control arm that will not fire is a composition this pass could
        // not compose, and is skipped.
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

    /// Arm this load's whole body lattice before any caller has fired
    /// anything, then close the map: enumerates every key this deployment
    /// can realize, fires each `record::WARM_FIRES` times to warm and
    /// capture it, then seals the map if anything armed. Called once, from
    /// the tail of [`Shell::load`]; fails the load only via the golden
    /// check ([`Shell::golden`]) — a refused key just keeps walking eagerly.
    pub(super) fn arm_bodies(&mut self) -> Result<()> {
        // `bodies` off, non-recording, too many capture units, a rotating
        // load, or `[engine] pad off` all produce only eager walks.
        if !self.records_bodies() || !Self::keyable_units(&self.compiled) {
            return Ok(());
        }

        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return Ok(());
        }
        // No lattice: one key per row count. A lattice: each point at the
        // lane count a real fire of that rung can bring.
        let points: Vec<LatticePoint> = if self.budget.buckets.is_empty() {
            (1..=ceiling)
                .map(|n| LatticePoint { bucket: n, lanes: n })
                .collect()
        } else {
            let mut points = Vec::new();
            for point in self.budget.buckets.iter().copied() {
                if point <= ceiling {
                    points.push(LatticePoint { bucket: point, lanes: point });
                } else {
                    // First rung past the seats, only if there is one.
                    if ceiling > points.last().map_or(0, |point| point.bucket) {
                        points.push(LatticePoint { bucket: point, lanes: ceiling });
                    }
                    break;
                }
            }
            points
        };
        // Budget is device memory (`[engine] bodies_mem`, with
        // `record::MAX_BODIES` a count belt), asked per key.
        let seats = self.held.len() as u32;
        let context = self.pools.paging().context();
        let max_lanes = self.budget.max_lanes;
        let classes = self.compiled.classes.classes.len();
        // A text synthetic may not land in a media class: a media-fact
        // word with no image is a composition the embed merge panics on.
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
                .filter(|class| textual(*class))
                .collect(),
            prefilling: (0..classes)
                .filter(|class| {
                    !self.decoding.contains(*class)
                        && textual(*class)
                        && !self.landing[*class].is_empty()
                })
                .collect(),
            media: self.media.iter().collect(),
            fragmenting: self.fragmenting(),
            decoding: self.decoding.clone(),
            seats,
            context,
            max_lanes,
            patch_fold: self.patch_fold,
        };
        let mut found = Targets::default();
        for enumerate in ARMS {
            enumerate(&deployment, &mut found);
        }
        let Targets {
            mut targets,
            mut unfireable,
        } = found;
        // Ascending bucket, so the budget is spent on the smallest first.
        targets.sort_by_key(|(bucket, _)| *bucket);
        // Top rung first, then ascending: a scratch slab grow retires the
        // old block, so firing the largest rung first forces that growth
        // before anything is recorded.
        if let Some(top) = targets.last().map(|(bucket, _)| *bucket) {
            targets.sort_by_key(|(bucket, _)| (*bucket != top, *bucket));
        }
        if targets.is_empty() {
            return Ok(());
        }

        // Asked of the map, not a pre-truncated list.
        let mut armed = 0usize;
        let mut wanted = 0usize;
        let mut tally = [(0usize, 0usize); Kind::COUNT];
        let mut refused: Option<String> = None;
        let mut unadmitted: Vec<Vec<usize>> = Vec::new();

        let mut never = 0usize;
        let mut never_from = 0u32;
        // Which of the two bounds stopped the pass.
        let mut belted = false;
        for (bucket, target) in targets {
            // Bytes, not seats: the live census reads device memory spent.
            let spent = self.cache.body_stats();
            if spent.census.bodies >= record::MAX_BODIES || spent.census.bytes >= self.bodies_mem {
                if never == 0 {
                    never_from = bucket;
                    belted = spent.census.bodies >= record::MAX_BODIES;
                }
                never += 1;
                continue;
            }
            let present = target.present();
            // Skip list is the token arms' alone: a tower target has a
            // different second (patch) rectangle.
            if target.skips_on_present_set() && unadmitted.contains(&present) {
                // Named, not fired: already known inadmissible.
                unfireable.push(format!("bucket {bucket}, {target}: inadmissible present set"));
                continue;
            }
            wanted += 1;
            let at = target.kind().at();
            let (lanes, media) = target.lanes();
            tally[at].1 += 1;
            // The media half is empty for the token kinds, so the door is
            // picked off the submission and not off the kind.
            let owned = if media.is_empty() {
                self.synthetic_lanes(&lanes)
            } else {
                self.synthetic_lanes_with(&lanes, &media)
            };
            self.armed_body = None;
            // A `Fault` is this geometry's, not necessarily this present
            // set's, so a faulted key teaches the skip list nothing.
            let mut faulted = false;
            for _ in 0..record::WARM_FIRES {
                let fired = self.fire_synthetic(&owned);
                // Each key's fires settle before the next: eviction needs a
                // quiet device to read a refusal as final.
                let landed = self.device.synchronize();
                if let Err(why) = fired.and(landed) {
                    refused = Some(format!("bucket {bucket}, {target}: {why}"));
                    faulted = true;
                    break;
                }
            }
            // The key the synthetic actually built (`Shell::armed_body`,
            // written by `prepare`), not reconstructed here. `None` means
            // the admissibility rule turned this present set away.
            let admitted = self.armed_body.is_some();
            let key = self.armed_body.take();
            // Asked of the cache, not the return value: `Ok` may still have
            // declined to seat. The golden runs before the key is counted
            // armed: a body that disagrees with its own walk fails the load.
            if self.golden
                && let Some(key) = key.as_ref()
                && self.cache.holds_body(key)
            {
                self.golden(key, &owned)?;
                // And the same verdict over a composition a caller could
                // have brought, which the synthetic cannot state.
                self.golden_real(key, &owned)?;
            }
            if key.is_some_and(|key| self.cache.body_armed(&key)) {
                armed += 1;
                tally[at].0 += 1;
            } else if !admitted && !faulted && target.skips_on_present_set() {
                // A tower target writes nothing here: its verdict is about a
                // second rectangle token arms do not have.
                unadmitted.push(present);
            }
        }

        // The boot line, per composition kind. `wanted` is what was
        // attempted, not enumerated.
        let stats = self.cache.body_stats();
        // What the seal proved, not merely that the map is closed.
        let seal = if armed == 0 {
            Seal::Open
        } else if never == 0 && refused.is_none() {
            Seal::Complete
        } else {
            Seal::Partial { never }
        };
        let report = Armed {
            wanted,
            armed,
            kinds: tally,
            segmented: stats.census.segmented,
            bytes: stats.census.bytes,
            unweighed: stats.census.unweighed,
            bodies_mem: self.bodies_mem,
            bodies: stats.census.bodies,
            last_refusal: refused,
            unfireable,
            never,
            never_from,
            belted,
            declines: stats.tally.declines,
            refusals: stats.tally.refusals,
            seal,
        };
        eprintln!("engine-cuda: {report}");
        if !matches!(seal, Seal::Open) {
            self.cache.seal_bodies();
        }
        self.armed = Some(report);
        Ok(())
    }

    /// The minimal present sets that break a window — the only enumeration
    /// that seeks a segmented body. A mask over one or two present classes
    /// is always one interval, so it takes a third class between two of a
    /// mask's own to split it. Enumerates the minimal witnesses only (three
    /// classes each), since the full present-set space is exponential.
    fn fragmenting(&self) -> Vec<Vec<usize>> {
        let classes = self.compiled.classes.classes.len();
        let mut seen: Vec<&model_ir::ClassSet> = Vec::new();
        let mut found: Vec<Vec<usize>> = Vec::new();
        for region in self.compiled.template() {
            if region.mask.len() < 2 || seen.contains(&&region.mask) {
                continue;
            }
            seen.push(&region.mask);
            for separator in 0..classes {
                if region.mask.contains(separator) {
                    continue;
                }
                let Some(present) = Self::witness(&self.compiled, &region.mask, separator)
                else {
                    continue;
                };
                // No witness may name a media class: every lane this arm
                // synthesizes is text-only, and a media-class lane with no
                // image is a composition the embed merge panics on.
                if present.iter().any(|class| self.media.contains(*class)) {
                    continue;
                }
                if Self::breaks(&self.compiled, &region.mask, &present)
                    && !found.contains(&present)
                {
                    found.push(present);
                }
            }
        }
        found
    }

    /// The three classes that witness one separator breaking one mask, or
    /// `None` when the separator does not stand between two of the mask's
    /// classes at all (it sits in front of, or behind, all of them — which
    /// the decode, prefill and mixed arms already arm).
    fn witness(
        compiled: &CompiledModel,
        mask: &model_ir::ClassSet,
        separator: usize,
    ) -> Option<Vec<usize>> {
        let mut whole: Vec<usize> = mask.iter().collect();
        whole.push(separator);
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(whole.iter().copied()));
        let mut before: Option<usize> = None;
        let mut after: Option<usize> = None;
        let mut passed = false;
        for class in order {
            let class = class as usize;
            if class == separator {
                passed = true;
                continue;
            }
            if !mask.contains(class) {
                continue;
            }
            if passed {
                after = Some(class);
                break;
            }
            before = Some(class);
        }
        let mut present = vec![before?, separator, after?];
        present.sort_unstable();
        Some(present)
    }

    /// Does `mask` cover more than one interval of the order `present`
    /// seriates to (`ClassOrder::class_order`, the same order `fire::compose`
    /// builds a real fire's from)?
    fn breaks(compiled: &CompiledModel, mask: &model_ir::ClassSet, present: &[usize]) -> bool {
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(present.iter().copied()));
        let mut runs = 0usize;
        let mut inside = false;
        for class in order {
            if mask.contains(class as usize) {
                runs += usize::from(!inside);
                inside = true;
            } else {
                inside = false;
            }
        }
        runs > 1
    }

    /// Spread `rows` over at most `lanes` lanes of at most `context` rows
    /// each, or `None` for a total this deployment cannot present. Even
    /// split, chosen only because it minimises the tallest lane.
    fn spread(rows: u32, lanes: u32, context: u32) -> Option<Vec<u32>> {
        let lanes = lanes.min(rows);
        if lanes == 0 || u64::from(context) * u64::from(lanes) < u64::from(rows) {
            return None;
        }
        let base = rows / lanes;
        let over = rows % lanes;
        Some(
            (0..lanes)
                .map(|at| base + u32::from(at < over))
                .collect(),
        )
    }
}

/// One lane per class of a fragmenting present set, at row counts that
/// land on `bucket`, or `None` for a set this deployment cannot fire here.
/// A decode class takes exactly one row; the rest spread evenly.
fn fragment_rows(
    deployment: &Deployment,
    present: &[usize],
    bucket: u32,
) -> Option<Vec<(usize, u32)>> {
    let width = present.len() as u32;
    if width < 2
        || deployment.seats < width
        || deployment.max_lanes < width
        || bucket < width
    {
        return None;
    }
    let decodes = present
        .iter()
        .filter(|class| deployment.decoding.contains(**class))
        .count() as u32;
    let prefilling: Vec<usize> = present
        .iter()
        .copied()
        .filter(|class| !deployment.decoding.contains(*class))
        .collect();
    if prefilling.is_empty() {
        return None;
    }
    let rows = Shell::spread(
        bucket - decodes,
        prefilling.len() as u32,
        deployment.context,
    )?;
    if rows.len() != prefilling.len() {
        return None;
    }
    let mut taken = rows.into_iter();
    Some(
        present
            .iter()
            .map(|class| {
                if deployment.decoding.contains(*class) {
                    (*class, 1u32)
                } else {
                    (*class, taken.next().unwrap_or(1))
                }
            })
            .collect(),
    )
}

/// Why the golden refused this key, in one line — `None` when the two
/// readouts agree. Classifies a disagreement `numeric` (within 2 ulp) or
/// `structural` (past 2 ulp — work is missing or wrong).
fn evidence(walked: &[Vec<f32>], replayed: &[Vec<f32>]) -> Option<String> {
    /// f32 bits in an order where adjacent representable numbers are
    /// adjacent integers.
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
    for (lane, (a, b)) in walked.iter().zip(replayed).enumerate() {
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
            worst = worst.max(ordered(x.to_bits()).abs_diff(ordered(y.to_bits())));
            if first.is_none() {
                first = Some((at, *x, *y));
            }
        }
    }
    let (at, x, y) = first?;
    // Two ulp: the width of one bf16 rounding either way.
    let class = if worst <= 2 { "numeric" } else { "structural" };
    Some(format!(
        "differs at #{at} (walk={x} {:#010x}, body={y} {:#010x})  n_diff={differing}/{total}  \
         max_ulp={worst}  class={class}",
        x.to_bits(),
        y.to_bits(),
    ))
}

/// The hybrid decode arm, asked without a device: [`ensemble_keys`] is a
/// pure function of a [`Deployment`], checkable by a host-only sweep.
#[cfg(test)]
mod tests {
    use super::{BodySynth, Deployment, LatticePoint, Targets, ensemble_keys};

    /// A deployment with `decoders` decode classes and one rung; every
    /// other field is set so the other five arms enumerate nothing.
    fn deployment(decoders: usize, point: LatticePoint) -> Deployment {
        Deployment {
            points: vec![point],
            buckets: Vec::new(),
            patch_points: Vec::new(),
            decoders: (0..decoders).collect(),
            prefilling: Vec::new(),
            media: Vec::new(),
            fragmenting: Vec::new(),
            decoding: model_ir::ClassSet::of(0..decoders),
            seats: point.lanes,
            context: 512,
            max_lanes: point.lanes,
            patch_fold: 1,
        }
    }

    /// The lane list a target carries, as `(class, lane count)`.
    fn lanes(target: &BodySynth) -> Vec<(usize, u32)> {
        match target {
            BodySynth::Ensemble { lanes } => lanes.clone(),
            other => panic!("the ensemble arm produced {other}"),
        }
    }

    /// Two decode words arm the pair, and the composition rounds to the
    /// rung: both words present at the rung's own lane count, one row per
    /// lane.
    #[test]
    fn two_decode_words_arm_the_pair_at_the_rungs_lane_count() {
        let mut found = Targets::default();
        ensemble_keys(
            &deployment(2, LatticePoint { bucket: 256, lanes: 256 }),
            &mut found,
        );
        assert_eq!(found.targets.len(), 1, "one key per rung, and there is one rung");
        let (bucket, target) = &found.targets[0];
        assert_eq!(*bucket, 256);
        assert_eq!(
            lanes(target),
            vec![(0, 128), (1, 128)],
            "the rung's lanes are split across the two decode words"
        );
        // Nothing else in `ARMS` produces a multi-decode present set.
        assert_eq!(target.present(), vec![0, 1]);
        let (rows, media) = target.lanes();
        assert!(media.is_empty(), "an ensemble lane submits no image");
        assert_eq!(rows.len(), 256, "one lane per row");
        assert!(rows.iter().all(|(_, rows)| *rows == 1), "a decode lane is one row");
        assert_eq!(rows.iter().map(|(_, rows)| *rows).sum::<u32>(), 256);
    }

}

/// What the arming pass did, as the boot line says it.
#[derive(Debug, Clone)]
pub struct Armed {
    pub wanted: usize,
    pub armed: usize,
    /// Per [`Kind`], `(armed, wanted)`.
    pub kinds: [(usize, usize); Kind::COUNT],
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
    pub declines: u64,
    pub refusals: u64,
    /// What the seal stands for. See [`Seal`].
    pub seal: Seal,
}

/// What the seal proved — not the same question as whether the map was
/// closed. Keys that never armed walk eagerly for the load's life, and
/// that has to be readable off the boot line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seal {
    /// Every key the loop reached was attempted, none refused.
    Complete,
    /// Closed over what armed; `never` key(s) were never attempted or one
    /// refused. Those compositions walk eagerly for the load's life.
    Partial {
        /// [`Armed::never_from`] is the bucket it stopped at;
        /// [`Armed::belted`] says which bound stopped it.
        never: usize,
    },
    /// Nothing armed, so the map is left open; traffic may still mint
    /// bodies of its own.
    Open,
}

impl core::fmt::Display for Armed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let columns = Kind::ALL
            .iter()
            .map(|kind| format!("{kind} {}/{}", self.kinds[kind.at()].0, self.kinds[kind.at()].1))
            .collect::<Vec<String>>()
            .join(", ");
        write!(
            f,
            "bodies armed {} of {} compositions at load ({columns}; {} segmented; {} MiB of {} \
             MiB, {} of {} seats)",
            self.armed,
            self.wanted,
            self.segmented,
            self.bytes >> 20,
            self.bodies_mem >> 20,
            self.bodies,
            record::MAX_BODIES,
        )?;
        if self.unweighed > 0 {
            write!(f, " ({} unweighed)", self.unweighed)?;
        }
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
        // The seal state, once — `never` is folded into it rather than
        // reported separately.
        match self.seal {
            Seal::Complete => write!(f, " [sealed: every key attempted]")?,
            Seal::Partial { never } if never > 0 => write!(
                f,
                " [sealed partial: {never} key(s) never attempted, {} at bucket {}, and they \
                 walk eagerly for the life of this load]",
                if self.belted { "record::MAX_BODIES" } else { "[engine] bodies_mem" },
                self.never_from,
            )?,
            Seal::Partial { .. } => write!(f, " [sealed partial: a key refused]")?,
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
