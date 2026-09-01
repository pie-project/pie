//! **WHAT A LOAD FIRES BEFORE ANY CALLER DOES** — the bodies arming pass: the
//! key space this deployment can realize, one synthetic fire per key, and the
//! seal.
//!
//! A child module of [`serve`](super) because it is the one pass that is
//! neither the load's call order nor a fire's. It runs ONCE, from the tail of
//! [`Shell::load`], on the load thread, and everything it does it does
//! through the ordinary fire path — `Shell::fire_media`, the same three
//! phases, the same router — with `Shell::arming` set so that nobody's
//! numbers are read and no expert is promoted. Nothing it does can fail the
//! load.
//!
//! ```text
//! BodySynth       one key the pass means to climb, as a geometry
//! Kind            which of the six it is, which is the boot line's column
//! Deployment      every number the six enumerators read, taken once
//! Shell::arm_bodies   enumerate, attempt in ascending bucket order, seal
//! Synthetic       one lane of a synthetic composition, owned
//! ```
//!
//! # The one file where a seventh composition kind is a seventh entry
//!
//! The enumeration used to be five loop nests inline in `arm_bodies`, three
//! index literals (`kinds[0]`..`kinds[4]`) hand-agreeing with an array width
//! and with the boot line's column order two hundred lines away, and four
//! `format!` siblings saying the same sentence four ways — one of which had
//! already drifted. It is a table now: [`Kind`] names the six, [`ARMS`] is
//! one enumerator per kind, and the tally and the boot line are both
//! iterations of [`Kind::ALL`]. What a seventh kind costs is a `BodySynth`
//! variant, a `Kind`, an enumerator and its entry — and nothing at all in the
//! attempt loop, which reads every per-kind fact through
//! [`BodySynth::kind`], [`BodySynth::lanes`] and
//! [`BodySynth::skips_on_present_set`].
//!
//! **AND THE SIXTH IS THE ONE THE TABLE WAS BUILT FOR** ([`Kind::Ensemble`]).
//! A SKU with two decode words — a hybrid attention bake, where the sliding
//! arm and the full arm are two classes and both run
//! `Attention::Decode` — has decode traffic that presents BOTH at once: the
//! scheduler batches whatever is decoding, and what is decoding is a mix of
//! the two words. Every arm above enumerated decode as a SINGLETON, so the
//! composition every decode fire of such a load actually brings had no key,
//! fell past the seal, and walked eagerly for the life of the load
//! (`b256[c0:256 c1:256]`, named in the boot line as the shape the sealed map
//! holds no body for). It cost one enumerator and one entry, which is what
//! the table exists to make true.

use engine::fire::{Mask, Masking, RsReset, RsVerb};
use model_compiler::CompiledModel;

use crate::error::Result;
use crate::record;

use super::{Lane, MROPE_COORDS, Media, PATCH_ROUTE_DROP, Seated, Shell};

/// One lane of a SYNTHETIC composition — the owned side of a [`Seated`] an
/// arming pass borrows. A private carrier, not a submission type: only
/// [`Shell::synthetic_lanes`] builds one, and only [`Shell::fire_synthetic`]
/// fires it.
///
/// **AND ITS LAUNCHES ARE REAL.** An arming pass walks EAGERLY first, exactly
/// as any miss does, which is the whole point of arming: the eager pass is
/// what warms the JIT, grows the scratch slabs and gives the dense tuner its
/// second sighting. Its numbers are still nobody's — no readback, no epilogue,
/// no `held` advance — but the kernels run.
struct Synthetic {
    /// The class's representative word (`Class::word`) — the one part of a
    /// submission decision #18 says the shell must not invent, invented here
    /// anyway and honestly: the sweep's own table is where the word comes
    /// from, so it names exactly the class it must.
    word: u64,
    /// Placeholder ids, one per row.
    tokens: Vec<u32>,
    /// An all-allowed mask, for a class whose window runs the masked arm.
    ///
    /// `Masking::Extent`, and it stays that way: the arming pass plans the
    /// SHAPES a class's launches take, and both mask forms expand to one
    /// `rows x kv` rectangle, so the per-row form has no plan of its own to
    /// arm (`crate::mask`'s own argument).
    mask: Option<Masking>,
    /// Adapter row 0, for a class inside the correction's window.
    adapter: Option<u32>,
    /// The word's draft bit, mirrored (`Fault::DraftWord` is checked per
    /// lane, synthetic or not).
    drafts: bool,
    /// The word's capture bit, mirrored.
    captures: bool,
    /// Which real slot lends its page arithmetic.
    slot: u32,
    /// **THE MEDIA-SHAPED HALF OF A SYNTHETIC LANE**, or `None` for a lane
    /// that submits no image — which is every lane of the five token kinds
    /// and every lane a text-only load ever synthesizes (the multi-unit
    /// bodies wave).
    ///
    /// `(images, patches)`, and the payload behind it is ZERO BYTES.
    /// [`Shell::fire_synthetic`] builds the `Media` from these two numbers and
    /// a zeroed patch buffer, which is the same discipline the mask synthetic
    /// is under: what an arming pass has to produce is a PLAUSIBLE GEOMETRY —
    /// a shape the planners take, the launches run and the composition can
    /// name — and not plausible NUMBERS, because the pass computes nobody's.
    /// An all-zero patch rectangle plans exactly as a real image's does; a
    /// route table that named no row would not.
    media: Option<SyntheticMedia>,
}

/// **THE SIX VECTORS ONE SYNTHETIC IMAGE SUBMISSION IS**, owned so that
/// [`Shell::fire_synthetic`] can lend a [`Media`] out of them (the multi-unit
/// bodies wave).
///
/// **PLAUSIBLE GEOMETRY, NOT PLAUSIBLE NUMBERS** — the mask synthetic's
/// discipline, one axis over and stated again because this is where it is
/// easiest to get wrong. What an arming pass has to produce is a submission
/// the planners take, the launches run and the composition can NAME: the
/// right lengths, entries inside the bounds the refusals check, and a
/// rectangle the tower's GEMMs can be gridded over. What it does not have to
/// produce is an image. Nobody reads this fire's numbers — no readback, no
/// epilogue, no `held` advance — so the patch payload is zeros, and zeros
/// plan exactly as pixels do.
///
/// The two places a zero would NOT have been plausible are written out
/// rather than defaulted, because both are read as addresses:
///
/// * **the routes** name a TOKEN row each, and `prepare` refuses one past the
///   lane's rows by name. They walk the lane's rows modulo its row count, so
///   the vector is in bounds for any pairing of patch rows and token rows this
///   pass can synthesize — which is what keeps the tower arm from needing a
///   `spread`-shaped feasibility question of its own;
/// * **the interpolation weights** are a bilinear partition of unity, and
///   all-zero would be a gather that contributes nothing. The first tap of
///   each row takes 1.0 and the rest 0.0 — nearest-neighbour, which is a real
///   answer of the same kernel rather than a degenerate one.
struct SyntheticMedia {
    /// Patch rows per image, summing to the fire's patch total.
    rows: Vec<u32>,
    /// `patch_rows x row_bytes` of zeros.
    patches: Vec<u8>,
    /// One token row per tower output row, then the fold's dead tail.
    routes: Vec<i32>,
    /// `(t, h, w)` per patch row — the grid's origin, which is a coordinate
    /// every tower rotation accepts.
    positions: Vec<i32>,
    /// The position table's gather rows: row 0, `embed_taps` times per patch
    /// row.
    embed_rows: Vec<i32>,
    /// Their weights: nearest-neighbour, `1.0` then zeros.
    embed_weights: Vec<f32>,
}

/// **ONE KEY THE BODIES ARMING MEANS TO CLIMB, AS A GEOMETRY** — the three
/// present-set shapes `Shell::arm_bodies` enumerates, each carrying the lanes
/// it will synthesize and nothing else.
///
/// **A KIND AND NOT A LANE LIST, BECAUSE THE BOOT LINE COUNTS BY KIND.** A
/// short decode tally, a short prefill tally and a short mixed tally are three
/// different sentences about a deployment (its seats, its context, both), and
/// an operator reading one number could act on none of them. The `Display`
/// below is what a refusal names, for the same reason.
///
/// The rows a prefill or mixed arm carries are `Shell::spread`'s answer, taken
/// BEFORE any fire, so a bucket this deployment cannot hold is refused by name
/// instead of by a planner's `Fault`.
#[derive(Debug, Clone)]
enum BodySynth {
    /// One decode class, `lanes` lanes of one row each — the composition that
    /// makes a fire a decode, at the lane count this rung admits.
    Decode { lanes: u32, class: usize },
    /// One non-decode class, the bucket's rows spread over its lanes.
    Prefill { class: usize, rows: Vec<u32> },
    /// One decode lane beside one non-decode class's lanes.
    Mixed {
        decode: usize,
        class: usize,
        rows: Vec<u32>,
    },
    /// **A PRESENT SET THAT PUTS A FOREIGN CLASS'S ROWS INSIDE SOME REGION'S
    /// WINDOW** — the composition a SEGMENTED body exists for (the tier-2
    /// campaign).
    ///
    /// The three kinds above top out at TWO present classes, and two classes
    /// can never break a window: a fire orders its classes by the shipped
    /// order with the absent ones dropped, and dropping a class can only CLOSE
    /// a gap (`model_exec::fire::fallback::bound` argues it), so a mask over a
    /// subset of two present classes is always one interval. It takes a THIRD
    /// class standing between two of a mask's own to put foreign rows inside
    /// that mask's span — and that is exactly the composition P4 answers with
    /// a `Fallback`, which the shell serves as a split, a gathered rectangle
    /// or a grouped segment list. The last two are ISLANDS, so without this
    /// arm no load would ever arm a segmented body at all and the tier-2 path
    /// would exist without a key to exercise it.
    ///
    /// **AND THE WITNESS IS THREE CLASSES AND NOT THE WHOLE MASK**
    /// ([`Shell::witness`]): the separator and its two nearest neighbours in
    /// the mask, which is the minimal set that breaks it. A witness carrying
    /// the mask's other classes needs a seat for each of them, and a
    /// deployment that cannot seat the wide one arms nothing where the narrow
    /// one — the composition its traffic actually brings — would have armed.
    ///
    /// One lane per class, in ascending class order, with the row counts
    /// `Shell::arm_bodies` spread — a decode class takes exactly one row,
    /// because one row per lane is what makes a fire a decode.
    Fragmented { lanes: Vec<(usize, u32)> },
    /// **A FIRE THAT CARRIES AN IMAGE** — the composition a two-unit
    /// artifact's SECOND capture unit exists for (the multi-unit bodies
    /// wave).
    ///
    /// The four kinds above all present a text lane and nothing else, as does
    /// [`BodySynth::Ensemble`] below them, so every one of their compositions
    /// has zero patch rows and a patch bucket of zero: they arm the tower
    /// SKU's text keys, which are real keys and worth arming, and no tower key
    /// at all. A `record::BodyKey` carries a
    /// `record::AxisKey` per unit now, so the patch bucket is a coordinate of
    /// the key and each rung of the patch lattice is a key of its own.
    ///
    /// **ONE LANE, ONE MEDIA CLASS, ONE RUNG OF THE PATCH LATTICE.** The
    /// class is one `Shell::media` names — a class whose window runs the embed
    /// merge, which is where a lane that submitted an image lands — and the
    /// token rows beside it are what a real caller brings: enough placeholder
    /// rows for the tower's soft tokens. `patches` is the rung itself, for
    /// [`BodySynth::Prefill`]'s reason: `patch_bucket_of` is idempotent on a
    /// lattice point, so the composition lands on THIS key by construction
    /// where a rung-plus-one would need this pass to re-derive the patch
    /// lattice's own ordering.
    ///
    /// `images` is how many images those patch rows are cut into — one, and
    /// the enumeration says why: the patch axis's LANE count is a coordinate
    /// of nothing. The key names the patch bucket and the patch ladder, and
    /// neither reads an image count, so two tower keys differing only in how
    /// many images their patch rows arrived as are the same key. Arming both
    /// would pay a capture to look the second one up under the first one's
    /// name.
    Tower {
        class: usize,
        rows: u32,
        images: u32,
        patches: u32,
    },
    /// **EVERY DECODE WORD THIS LOAD HAS, PRESENT AT ONCE** — the composition
    /// a HYBRID SKU's decode traffic actually brings, and the one the five
    /// above cannot name.
    ///
    /// [`BodySynth::Decode`] enumerates decode as a SINGLETON: one class,
    /// `lanes` lanes of one row. That is the whole decode key space of a bake
    /// with one decode word and none of it on a bake with two. A hybrid
    /// attention model ships two classes that both run `Attention::Decode`
    /// (`Shell::decoding` reads them off the template), a scheduler batches
    /// whatever sequences are decoding, and what is decoding at any instant
    /// is a mix of the two words — so the fire presents `{c0, c1}`, both
    /// decode, and keys to a ladder no singleton and no `Mixed` pair ever
    /// spells. On a two-decode-word load that is not a corner of the traffic:
    /// it is the traffic.
    ///
    /// **ONE ROW PER LANE, AND THE LANES SPLIT ACROSS THE WORDS** — which is
    /// what keeps this a DECODE composition rather than a small prefill. The
    /// rung's own lane count is the total, so the composition rounds to the
    /// same bucket [`BodySynth::Decode`]'s does at that rung, and each class's
    /// canon rung is `min(lane_ceiling, bucket)` — `record::Ladder::rung`'s
    /// answer for a decode class, which is what a REAL two-word fire's ladder
    /// gets from `record::Ladder::of`. The pass does not compute it: this
    /// arm's key is taken back out of the one instant that composed it
    /// (`Shell::armed_body`), exactly as the prefill, mixed and fragmented
    /// arms take theirs.
    ///
    /// **THE FULL SET AND THE SINGLETONS, AND NOTHING BETWEEN THEM.** With
    /// three or more decode words the subsets are a power set, and this arms
    /// two of its layers: the bottom (one word — [`BodySynth::Decode`]) and
    /// the top (all of them). The argument is what a fire IS. A decode-only
    /// fire is either one word's traffic — a load serving one kind of request,
    /// or an instant when only one word has work — or the scheduler's whole
    /// mix, which is every word it has. A fire of exactly two words out of
    /// three is a fire the scheduler assembled while the third word's queue
    /// happened to be empty: real, rarer than either end, and the same
    /// never-attempted long tail every other arm already accepts
    /// (`Shell::fragmenting` accepts it for masks, `tower_keys` for two media
    /// classes at once). It costs one unarmed key, which is what every unarmed
    /// key costs, and it is counted where they all are
    /// (`record::BodyTally::sealed_declines`) and named on the line above it.
    ///
    /// `lanes` is `(class, lane count)` per decode word, in
    /// `Deployment::decoders` order, every count at least one.
    Ensemble { lanes: Vec<(usize, u32)> },
}

impl BodySynth {
    /// **WHICH CLASSES THIS SYNTHETIC PUTS ROWS IN** — the PRESENT SET, which
    /// is half of a [`record::BodyKey`] and what `Windows::admits` reads
    /// alongside the bucket.
    ///
    /// Ascending and deduplicated, so two targets of one present set at two
    /// lattice points answer the same vector — which is what
    /// `Shell::arm_bodies`' skip list is keyed on (and what its own note
    /// argues is a budget rule rather than a theorem, now that the remaining
    /// decline can move with the bucket).
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

    /// **WHICH KIND THIS TARGET IS** — the tally's slot and the boot line's
    /// column, read off the variant instead of written beside it.
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

    /// **DOES A VERDICT ABOUT THIS TARGET SPEAK FOR ITS PRESENT SET?** — the
    /// skip list's one question, asked once here rather than matched against
    /// one variant at the two sites that read it.
    ///
    /// The list is keyed on the present set, on the argument that a set an
    /// earlier bucket found inadmissible is inadmissible at every bucket —
    /// and that argument is about the TOKEN rectangle's windows, which is all
    /// the five token kinds have. A tower target of the same class presents
    /// the same token classes and a completely different second rectangle:
    /// its regions on the patch axis have windows the text fire's composition
    /// does not resolve at all, so a text key's verdict says nothing about it
    /// and its own says nothing about a text key. `false` keeps the two apart
    /// in both directions, which is what one line of `matches!` at two sites
    /// had to be trusted to keep doing.
    fn skips_on_present_set(&self) -> bool {
        !matches!(self, BodySynth::Tower { .. })
    }

    /// **THE SUBMISSION THIS TARGET SYNTHESIZES** — the token lanes as
    /// `(class, rows)` pairs, and the `(images, patches)` pair per lane for
    /// the one arm that carries an image.
    ///
    /// Both halves in one answer because they are one decision — what
    /// [`Shell::synthetic_lanes_with`] is handed — and because two matches on
    /// the same variant two dozen lines apart is exactly the coupling a sixth
    /// arm would have had to find by reading. The second half is empty for
    /// the five token kinds, which is what makes the arming pass's call the
    /// same call it was before there was a second axis to submit on.
    fn lanes(&self) -> (Vec<(usize, u32)>, Vec<(u32, u32)>) {
        match self {
            // One-row lanes of one class: a decode fire that lands in this
            // lattice point — at the rung's own lane count when the seats
            // hold it, at the seat ceiling when the rung is the first one
            // past them (the composition still rounds up to `bucket`).
            BodySynth::Decode { lanes, class } => {
                (vec![(*class, 1u32); *lanes as usize], Vec::new())
            }
            BodySynth::Prefill { class, rows } => (
                rows.iter().map(|rows| (*class, *rows)).collect(),
                Vec::new(),
            ),
            // The decode lane FIRST, which is a statement about nothing:
            // `fire::compose` seriates by the baked class order and not by
            // submission order, so the ladder this key gets is the one a
            // real mixed fire gets whichever way round these are listed.
            BodySynth::Mixed { decode, class, rows } => (
                core::iter::once((*decode, 1u32))
                    .chain(rows.iter().map(|rows| (*class, *rows)))
                    .collect(),
                Vec::new(),
            ),
            // One lane per class, already paired with its rows by
            // [`fragment_rows`] — this arm has nothing to compose, because
            // the shape it needs is the one that shape function had to reason
            // about.
            BodySynth::Fragmented { lanes } => (lanes.clone(), Vec::new()),
            // One media lane of `rows` placeholder rows, and the images
            // beside it.
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => (vec![(*class, *rows)], vec![(*images, *patches)]),
            // Each decode word's lane count expanded into that many ONE-ROW
            // lanes, which is the same expansion `BodySynth::Decode` writes
            // one class at a time and for the same reason: one row per lane
            // is what makes a fire a decode.
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

/// **WHICH OF THE FIVE COMPOSITION KINDS A TARGET IS**, and the word the boot
/// line spells that column with.
///
/// **A TYPE AND NOT AN INDEX, BECAUSE THREE THINGS HAD TO AGREE AND NOTHING
/// MADE THEM.** The tally was five pairs in an array, the loop that counted
/// into it wrote a literal `0`..`4` beside each variant, and the boot line
/// spelled the columns' names two hundred lines further down — three
/// hand-kept agreements, none of them checkable, and a sixth kind meant
/// finding all three by reading. [`BodySynth::kind`] is the one reading now:
/// the tally is indexed by it, the boot line is an iteration of [`Kind::ALL`],
/// and a column's name is this `Display` rather than a token inside a format
/// string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
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
    /// [`BodySynth::Ensemble`].
    ///
    /// **AND IT IS ITS OWN COLUMN RATHER THAN MORE `decode`**, which is the
    /// decision this defect was diagnosed by. `decode 8/12` on a hybrid load
    /// is a sentence about the SINGLETON rungs and it read `8/12` while every
    /// real decode fire of that load walked — one number cannot be short for
    /// two different reasons. A short `decode` count is a deployment whose
    /// seats sit under its lattice; a short `ensemble` count is a load whose
    /// hybrid decode traffic has no key, which is a strictly worse thing and
    /// an operator's next move is different. `0/0` on every single-decode-word
    /// SKU, because the arm enumerates nothing there — the same `0/0` a
    /// text-only load reads under `tower`.
    Ensemble,
}

impl Kind {
    /// How many there are — the tally's width, and [`ARMS`]'s.
    const COUNT: usize = 6;

    /// All of them, in the order the boot line prints their columns and the
    /// order [`ARMS`] enumerates them in. The two are one order on purpose:
    /// [`Kind::at`] indexes the tally by discriminant, so a variant added
    /// anywhere but the end would move a column, and a reader who can see
    /// that here does not have to discover it from a wrong number.
    const ALL: [Kind; Kind::COUNT] = [
        Kind::Decode,
        Kind::Prefill,
        Kind::Mixed,
        Kind::Fragmented,
        Kind::Tower,
        Kind::Ensemble,
    ];

    /// Its slot in a per-kind tally.
    fn at(self) -> usize {
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

/// **EVERY NUMBER THE FIVE ENUMERATIONS READ**, taken once at the top of
/// [`Shell::arm_bodies`] and handed to each of them.
///
/// **A STRUCT AND NOT A BORROWED `self`, BECAUSE AN ENUMERATOR HAS TO BE ABLE
/// TO STATE WHAT IT DEPENDS ON.** Every field is a load constant — the seats,
/// the context, the lane ceiling's lattice, and which classes are decode,
/// prefill or media once the media partition has been applied — and not one
/// of them moves between the first key and the last. An arm handed the shell
/// could read anything; an arm handed this can read exactly the facts a
/// deployment states, and a sixth arm that needs a sixth fact adds a field
/// here where every other arm can see it.
struct Deployment {
    /// `(bucket, lanes)` per decode lattice point: the rung, and the lane
    /// count a real decode fire of it can bring on these seats.
    rungs: Vec<(u32, u32)>,
    /// `Budget::buckets`, the token lattice, ascending.
    buckets: Vec<u32>,
    /// The patch lattice, or empty for a deployment that cannot stage a patch
    /// row at all.
    patch_rungs: Vec<u32>,
    /// The decode classes a TEXT synthetic may land in.
    decoders: Vec<usize>,
    /// The non-decode classes a text synthetic may land in.
    prefilling: Vec<usize>,
    /// The classes whose window runs the embed merge — where a lane that
    /// submitted an image lands.
    media: Vec<usize>,
    /// The minimal present sets that break a window ([`Shell::fragmenting`]).
    fragmenting: Vec<Vec<usize>>,
    /// Which classes run a decode arm, as the class set itself: the ROWS a
    /// fragmenting witness gives each of its lanes turn on membership rather
    /// than on the filtered list above.
    decoding: model_ir::ClassSet,
    /// How many sequence seats this load holds.
    seats: u32,
    /// How many kv tokens one of them holds.
    context: u32,
    /// `Budget::max_lanes`.
    max_lanes: u32,
    /// How many patch rows the tower folds into one soft token.
    patch_fold: u32,
}

/// What the enumeration produces: the keys to attempt, and the ones this
/// deployment cannot synthesize at all, named.
#[derive(Default)]
struct Targets {
    /// `(bucket, target)` in enumeration order, until [`Shell::arm_bodies`]
    /// sorts it by bucket.
    targets: Vec<(u32, BodySynth)>,
    /// One sentence per key that was named rather than fired.
    unfireable: Vec<String>,
}

/// **ONE "THIS DEPLOYMENT CANNOT FIRE THAT KEY" SENTENCE**, in one place.
///
/// It was four `format!` siblings — one per enumerating arm — saying the same
/// thing four times, and the fourth had already drifted: the tower arm's line
/// stated the seats and the context and forgot the lane ceiling, so an
/// operator reading a short tower count saw two of the three numbers that
/// decide it. A sixth arm gets the sentence for free.
///
/// `what` names the composition (`prefill c3`, `tower c7`), `at` names the
/// lattice point it was asked at, and `note` is the one arm-specific clause
/// that earns its place — the tower's owed placeholder rows, which is the
/// number that says WHY that key is unfireable.
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

/// **ONE DECODE KEY PER LATTICE POINT PER DECODE CLASS**, at the lane count a
/// real decode fire of that rung can bring — the rung itself when the seats
/// allow it and the seat ceiling when they do not ([`Shell::arm_bodies`]
/// derives the pair).
///
/// The only arm that needs no lattice: a decode fire's rows are bounded by
/// the seats whether or not the deployment declared one, so `rungs` is never
/// empty where the others enumerate nothing.
fn decode_keys(deployment: &Deployment, into: &mut Targets) {
    for (bucket, lanes) in deployment.rungs.iter().copied() {
        for class in deployment.decoders.iter().copied() {
            into.targets
                .push((bucket, BodySynth::Decode { lanes, class }));
        }
    }
}

/// **ONE PREFILL KEY PER LATTICE POINT PER NON-DECODE CLASS** — the bucket's
/// own row total spread over `min(bucket, seats, max_lanes)` lanes.
///
/// The bucket itself and not "the previous rung plus one", because
/// `bucket_of` is idempotent on a lattice point: the total lands on this key
/// by construction, where a rung-plus-one would need this arm to re-derive
/// the lattice's own ordering and would name a DIFFERENT key on any
/// deployment whose buckets are not the ones it assumed.
///
/// This arm and the two below it need a LATTICE to enumerate over: a
/// deployment that declared none has `Composition::bucket == rows`, so its
/// key space is one key per row count and there is nothing finite to walk.
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

/// **ONE MIXED KEY PER (DECODE CLASS x NON-DECODE CLASS) PAIR PER LATTICE
/// POINT** — one decode lane of one row, and the remaining `bucket - 1` rows
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

/// **AND THE COMPOSITIONS A SEGMENTED BODY EXISTS FOR** (the tier-2 campaign,
/// [`BodySynth::Fragmented`]). Three present classes, which is both the
/// fewest that can put a foreign class's rows inside a mask's span — what
/// makes a region gathered, grouped or windowed-without-the-seat, an ISLAND —
/// and the most this arm asks for: a witness wider than the break needs a
/// seat per class, and a deployment that cannot seat it arms nothing where
/// the three-class fire it actually serves would have armed. Without this arm
/// the enumeration tops out at two classes and no load arms a segmented body
/// at all.
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

/// **AND THE SECOND CAPTURE UNIT'S OWN LATTICE** (the multi-unit bodies wave,
/// [`BodySynth::Tower`]) — one key per (media class, patch rung, token
/// bucket).
///
/// **THE PATCH PRESENT-SET IS SMALL, AND THAT IS WHY THIS ARM IS A PRODUCT
/// AND NOT AN ENUMERATION.** A `record::AxisKey` is a patch bucket and a
/// patch ladder, and the ladder is which classes have IMAGES with every rung
/// equal to that bucket — so the patch axis's free coordinates are the
/// present set and the rung, exactly as the token axis's are. What makes the
/// set small is that a class has images only if a lane of that class
/// submitted one, and only the classes `Shell::media` names can: one media
/// class per fire is the composition a caller brings and the one this arms. A
/// fire whose lanes land in TWO media classes at once keys to a two-class
/// patch ladder this pass does not arm and, past the seal, walks — which is
/// what every unarmed key does and is counted where they all are
/// (`record::BodyTally::sealed_declines`). Widening past one is a lattice
/// question, not a capture one, and is named here rather than guessed at.
///
/// **AND THE TOKEN BUCKET RIDES ALONG BECAUSE IT IS STILL IN THE KEY.** A
/// tower fire has token rows too — the placeholder rows the soft tokens
/// scatter onto — so its key names a token lattice point as well, and the
/// pair is what the product below walks. "6 + 6, not 6 x 6" is a statement
/// about the KEY's shape (two named coordinates rather than one fused one);
/// the number of realizable keys is still a product, and it is a small one
/// because each axis's ladder is short.
///
/// `patch_rungs` is empty for a deployment that cannot STAGE a patch row —
/// [`Shell::arm_bodies`] carries that gate's three clauses — so this arm
/// enumerates nothing on every text-only SKU.
fn tower_keys(deployment: &Deployment, into: &mut Targets) {
    for rung in deployment.patch_rungs.iter().copied() {
        for class in deployment.media.iter().copied() {
            for point in deployment.buckets.iter().copied() {
                // **THE PLACEHOLDER ROWS ARE OWED, NOT CHOSEN.** The tower
                // folds `patch_fold` patch rows into one soft token and the
                // merge scatters those onto token rows of the same lane, so a
                // fire of `rung` patch rows needs at least `rung / fold`
                // token rows to land them on. A bucket that cannot hold them
                // is a bucket no caller can bring this image in either, and
                // it is named rather than fired.
                let fold = deployment.patch_fold.max(1);
                let owed = rung.div_ceil(fold).max(1);
                if owed > point || point > deployment.context || deployment.seats == 0 {
                    into.unfireable.push(unfireable_line(
                        deployment,
                        &format!("tower c{class}"),
                        &format!("bucket {point} + patch rung {rung}"),
                        Some(&format!("{owed} placeholder row(s) owed")),
                    ));
                    continue;
                }
                into.targets.push((point, BodySynth::Tower {
                    class,
                    rows: point,
                    images: 1,
                    patches: rung,
                }));
            }
        }
    }
}

/// **AND ONE KEY PER LATTICE POINT FOR THE WHOLE DECODE SET** (the hybrid
/// decode wave, [`BodySynth::Ensemble`]) — every decode word present at once,
/// one row per lane, the rung's lanes split across the words.
///
/// **THIS IS THE ARM A HYBRID SKU'S DECODE TRAFFIC LANDS IN, AND WITHOUT IT
/// THAT TRAFFIC HAS NO KEY AT ALL.** [`decode_keys`] enumerates one class at
/// a time, [`mixed_keys`] pairs a decode class with a NON-decode one, and
/// [`fragmented_keys`] arms only the present sets that break a window. So on
/// a bake with two decode words — a hybrid attention model, where the sliding
/// arm and the full arm are two classes and `Shell::decoding` holds both —
/// nothing enumerated `{c0, c1}`, and `{c0, c1}` is what a batched decode
/// fire of that load presents every single time: the scheduler takes whatever
/// is decoding and what is decoding is a mix of the two words. Past the seal
/// every one of those fires walked eagerly and was counted in
/// `record::BodyTally::sealed_declines`, which is one whole catalog family's
/// steady state and not a long tail.
///
/// **THE RUNGS, NOT THE BUCKETS**, because this is a decode composition and
/// `Shell::arm_bodies` has already answered what a decode fire at each
/// lattice point can bring: `Deployment::rungs` is `(bucket, lanes)`, the
/// lanes being the rung itself where the seats hold it and the seat ceiling
/// where they do not. One row per lane makes the total `lanes`, which rounds
/// up to `bucket` exactly as [`decode_keys`]' singleton at the same rung does
/// — the two arms land on the same lattice point by construction, differing
/// only in how many classes have rows.
///
/// **AND THE SEAT BOUND IS ALREADY IN THAT LANE COUNT**, which is why this
/// arm asks for only one thing the others do not. `Shell::lane_ceiling` is
/// `min(slots, max_lanes, max_tokens)`, so a rung's lanes never exceed the
/// seats and never exceed `Budget::max_lanes`; what a rung can still fail to
/// hold is ONE LANE PER DECODE WORD, and a rung with fewer lanes than the
/// load has decode words is named rather than fired. That is the same
/// "a lane needs a seat, a fire needs a lane per present class" arithmetic
/// [`Shell::spread`] answers for the row arms, on the axis this arm counts in.
///
/// Nothing at all on a bake with fewer than two decode words: the singleton
/// IS the full set there, [`decode_keys`] already armed it, and arming it
/// twice would pay a capture to look one key up under its own name.
fn ensemble_keys(deployment: &Deployment, into: &mut Targets) {
    let words = deployment.decoders.len() as u32;
    if words < 2 {
        return;
    }
    for (bucket, lanes) in deployment.rungs.iter().copied() {
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

/// **`lanes` LANES SPLIT OVER EVERY DECODE WORD, ONE ROW EACH**, or `None`
/// for a rung that cannot seat a lane per word — [`BodySynth::Ensemble`]'s
/// geometry, and [`fragment_rows`]'s twin on the lane axis.
///
/// **EVENLY, AND THE SPLIT DOES NOT REACH THE KEY.** `record::Ladder::of`
/// reads each class's rows once, as a predicate — has this class any rows —
/// and the rung beside it is `record::Ladder::rung` of the bucket and the
/// lane ceiling. So a hundred and twenty-eight lanes of `c0` beside a hundred
/// and twenty-eight of `c1` and a two-hundred-and-fifty-five-to-one split are
/// the SAME key, and the capture taken over either serves both — the
/// grid-at-ceiling argument `Shell::spread` states for the row arms, one axis
/// over. Even is chosen for the reason it is chosen there: it is the split
/// that asks least of any one word.
///
/// **AND EVERY WORD GETS AT LEAST ONE LANE**, because a word with no lane is
/// a word with no rows and a class with no rows is not in the ladder at all
/// (`record::Ladder::of` filters on `window.rows > 0`) — which would arm the
/// SINGLETON key under the ensemble's name and spend a capture to find a body
/// [`decode_keys`] had already seated.
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

/// **THE WHOLE KEY SPACE, ONE FUNCTION PER KIND**, in [`Kind::ALL`]'s order.
///
/// This is the table the wave exists for. A seventh composition kind is a
/// [`BodySynth`] variant, a [`Kind`], an enumerator and one entry here —
/// nothing in the attempt loop, which reads every per-kind fact through
/// [`BodySynth::kind`], [`BodySynth::lanes`] and
/// [`BodySynth::skips_on_present_set`], and nothing in the boot line, which
/// iterates `Kind::ALL`.
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
    /// **ONE SYNTHETIC COMPOSITION'S LANES**, from a list of `(class, rows)`
    /// pairs — the geometry half of the arming pass, and
    /// [`Shell::fire_synthetic`]'s twin.
    ///
    /// The caller states the LIST and this states everything below it: the
    /// word, the placeholder ids, the mask form, the adapter row, the draft
    /// and capture bits, and which real slot lends its page arithmetic. A pair
    /// may repeat its class — `n` lanes of one class is a fire whose class
    /// table has one window of `n` rows and `n` lanes, which is exactly a
    /// decode batch.
    fn synthetic_lanes(&self, lanes: &[(usize, u32)]) -> Vec<Synthetic> {
        self.synthetic_lanes_with(lanes, &[])
    }

    /// **THE SAME LANES, WITH IMAGES ON SOME OF THEM** — the tower arm's door
    /// (the multi-unit bodies wave), and [`synthetic_lanes`](Shell::synthetic_lanes)
    /// is this with an empty media list.
    ///
    /// `media[i]` is `(images, patch rows)` for lane `i`, `(0, 0)` for a lane
    /// carrying none, and a short slice reads as none the rest of the way —
    /// which is what makes the text door above a call rather than a copy.
    ///
    /// **THE PATCH ROWS ARE SPREAD OVER THE IMAGES AS EVENLY AS THEY DIVIDE**,
    /// with the remainder on the first, because a lane's images are one
    /// concatenation and what the indptr has to be is `images + 1` ascending
    /// bounds. That is the only property anything downstream reads
    /// (`serve::prepare` builds the segment vector out of it, the tower's
    /// attention walks it), and an even split is the plausible one — a fire
    /// whose second image had zero rows would be a submission `Fault::PatchGeometry`
    /// exists to refuse.
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
            .map(|(at, &(class, rows))| Synthetic {
                word: self.compiled.classes.classes[class].word(),
                // Token id 0 in every cell: the ids only have to be
                // stageable, because the pass executes over a composition
                // whose numbers nobody reads.
                tokens: vec![0u32; rows as usize],
                // An all-allowed mask over the post-append extent, for a
                // class whose window runs the masked arm — the word and the
                // payload have to agree (`Fault::MaskWord`), and "attend
                // everything" is the plausible geometry that plans like any
                // real mask.
                mask: self.masked.contains(class).then(|| {
                    Masking::Extent(Mask::new(vec![0, rows + 1], u64::from(rows) + 1))
                }),
                adapter: self.corrected.contains(class).then_some(0),
                drafts: self
                    .exports
                    .mtp
                    .as_ref()
                    .is_some_and(|mtp| mtp.classes.contains(class)),
                captures: self.exports.capturing.contains(class),
                // Real slots, round-robin: the page arithmetic needs a slot
                // that exists, and `held: Some(1)` in `fire_synthetic` keeps
                // the borrow from touching the slot's own counting or
                // clearing its banks.
                slot: (at as u32) % slots,
                // **AND THE SECOND AXIS'S SUBMISSION, WHEN THIS LANE CARRIES
                // ONE** ([`SyntheticMedia`] argues every entry of it).
                media: media
                    .get(at)
                    .copied()
                    .filter(|(images, patches)| *images > 0 && *patches > 0)
                    .map(|(images, patches)| {
                        let patches = patches as usize;
                        let per = patches / images as usize;
                        let mut per_image: Vec<u32> = vec![per as u32; images as usize];
                        per_image[0] += (patches - per * images as usize) as u32;
                        // The tower's live output — what the fold compacts to
                        // and what `layout.scatter_live_rows` pairs with a
                        // route. The rows past it are the dead tail.
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
            })
            .collect()
    }

    /// **FIRE ONE SYNTHETIC COMPOSITION**, with [`Shell::arming`] set — the
    /// firing half of an arming pass, and [`Shell::synthetic_lanes`]'s twin.
    ///
    /// The borrow, the `held: Some(1)`, the absent readout, the plain RS verb
    /// and the flag restoration on both the success and the failure path: the
    /// walk lands in `record::Graphs::fire_body` exactly as a caller's fire
    /// would, which is the whole reason a load-armed body and a traffic-armed
    /// one are the same body.
    ///
    /// # Errors
    ///
    /// Whatever the synthetic fire refused — staging, a planner on synthetic
    /// geometry (kill factor 5), the capture, the instantiate. The caller
    /// tallies the sentence; nothing is retried.
    fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        let seated: Vec<Seated<'_>> = owned
            .iter()
            .map(|lane| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: lane.word,
                    tokens: &lane.tokens,
                },
                pages: &[],
                held: Some(1),
                // The arming pass resolves no port, so it crosses no space.
                translation: &[],
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                drafts: lane.drafts,
                captures_scores: lane.captures,
                // The arming pass computes nobody's numbers and plans no
                // readback, so there is no row list to carry and nothing that
                // would read one.
                readout: None,
                // The arming pass is SYNTHETIC: the plain fold is the one RS
                // shape that graph-replays (design §6), so it is also the
                // only one a body can be armed for.
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
            })
            .collect();

        // **AND THE SECOND AXIS'S SUBMISSION BESIDE IT** (the multi-unit
        // bodies wave), built the same way and out of the same owned vectors:
        // one `Media` per lane that carries images, keyed by lane exactly as a
        // caller's submission is. Empty for every text lane and every
        // text-only load, and then this `Vec` is allocated and never read —
        // which is the same nothing `attachments` costs a fire with none.
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
                    // The scalar rope, which is what an empty token stream
                    // means and what a synthetic has no grid to state.
                    token_positions: &[],
                    embed_rows: &shot.embed_rows,
                    embed_weights: &shot.embed_weights,
                })
            })
            .collect();

        self.arming = true;
        let armed = self.fire_media(&seated, &[], &media, &mut Vec::new());
        self.arming = false;
        armed.map(|_| ())
    }

    /// **ARM THIS LOAD'S WHOLE BODY LATTICE BEFORE A CALLER HAS FIRED
    /// ANYTHING, THEN CLOSE THE MAP** (the bodies design's chunk C, finished
    /// by the tier-1 key-collapse wave's chunk B), so that every fire this
    /// deployment can assemble replays from its first one and the serving path
    /// captures NOTHING.
    ///
    /// Called once, from the tail of [`Shell::load`], on the load thread,
    /// before any real fire and therefore before any real staging. Nothing it
    /// does can fail the load.
    ///
    /// # What it fires, and why the key space is a list rather than a guess
    ///
    /// A `record::BodyKey` is a lattice point and a class LADDER — which
    /// classes have rows, and the ceiling each one is carved to — and since
    /// the key collapse EVERY NUMBER in it comes from load constants: the
    /// present set from the class table, the bucket from `Budget::buckets`,
    /// and each rung from `record::Ladder::rung` of the two. So the keys this
    /// deployment can realize are enumerable, and this pass enumerates them:
    ///
    /// * **the present sets** are the DECODE classes ([`Shell::decoding`] —
    ///   which classes run an `attention.decode` arm, read off the template
    ///   the way [`Shell::masked`] is), the non-decode ones, and the pairs of
    ///   one of each. A shell cannot compute a lane's fact word, so "the
    ///   decode class" is asked as a question about ops rather than about
    ///   bits, and `Class::word` names a word that resolves back to it;
    /// * **the buckets** are `Budget::buckets`, filtered by what the
    ///   deployment can actually present — a decode key at a rung above the
    ///   seats, a prefill key whose rows will not fit `seats x context`, a
    ///   mixed key on a one-seat load are all named and skipped;
    /// * **and the CEILINGS ARE NOT READ OFF THE SYNTHETIC FIRE AT ALL** —
    ///   they are `record::Ladder::rung` of the bucket and
    ///   [`Shell::lane_ceiling`], the same call a real fire's ladder makes,
    ///   because a number this pass computed its own way is a key the traffic
    ///   cannot find. It used to be `rung_of` over the synthetic lane count,
    ///   and on a load whose seats sit under the lattice floor that armed
    ///   `c:8` while every fire of the bucket asked for `c:4`.
    ///
    /// **AND THE SYNTHETIC'S GEOMETRY INSIDE THE KEY DOES NOT MATTER**, which
    /// is what makes prefill and mixed arming possible where it once was not.
    /// A body's launches are gridded at the key's own ceilings
    /// (`Run::carve_rows`), so a capture taken over ANY split of the bucket
    /// stands for every split of it. What the synthetic has to be is fireable,
    /// not representative.
    ///
    /// The COPY POLICY used to be a fact of this key too, and is not any more:
    /// `record::BodyKey`'s own header argues why no fire the two policies
    /// could distinguish ever reaches a body.
    ///
    /// # And then the seal, which is the other half of "upfront"
    ///
    /// Having walked the key space, this pass closes the map
    /// (`record::Graphs::seal_bodies`) — but only if it armed something. Past
    /// that line a fire whose key holds no body keeps its eager numbers and is
    /// counted (`record::BodyTally::sealed_declines`) instead of warming
    /// toward a capture nobody asked for. The bodies path's whole claim is
    /// that its keys are known in advance; the seal is that claim enforced
    /// rather than hoped for.
    ///
    /// # The warm ladder, which is not optional and is not new
    ///
    /// At load nothing has been JIT-ed, no scratch slab has grown and the
    /// dense autotuner has seen no shape twice — the three reasons
    /// `crate::record`'s header gives for walking a miss eagerly BEFORE
    /// capturing it. Capture a body cold and its cuBLAS ladder is the untuned
    /// one, frozen for the life of the load, and every replay afterwards
    /// disagrees arithmetically with the eager walk it stands for.
    ///
    /// So each key is fired [`record::WARM_FIRES`] times through the ORDINARY
    /// bodied path, and the ordinary warm bookkeeping in
    /// `record::Graphs::fire_body` does the rest: the first fire walks
    /// eagerly and records nothing, and the `WARM_FIRES`-th walks eagerly
    /// again — the tuner's second sighting — and captures off that walk. That
    /// is the same ladder a real fire climbs; nothing here counts differently,
    /// and load-armed and traffic-armed bodies are the same bodies.
    ///
    /// # The one load it does not arm at all
    ///
    /// A ROTATING one. `Shell::enqueue_on` refuses to record any fire whose
    /// weights rotate — permanently, for the life of the load — so every rung
    /// this loop climbed would execute its warm fires against the eager walk
    /// and reach the end of the ladder with nothing captured. The pass exists
    /// to move a warm cost off the first caller and onto the boot; under a
    /// rotor there is nowhere to move it to, and paying it anyway is load-time
    /// device seconds spent on a cache that cannot exist. The gate's first
    /// lines refuse the whole pass, and the rotor's own boot line in
    /// [`Shell::load`] is where an operator reads that it happened.
    ///
    /// # What a refusal costs
    ///
    /// Nothing. A rung whose composition the admissibility rule turns away is
    /// refused into `bodies_refused` by `prepare`, exactly as a real fire's
    /// would be; a schedule that declines to be graph-shaped is
    /// `BodyTally::declines`; a synthetic geometry a planner will not take is
    /// a `Fault` this swallows. In every case the composition is left where
    /// it already was — walking eagerly, counted — and the loop moves to the
    /// next rung, because an armed SUBSET is a win and a load that refused to
    /// boot over it would be trading a whole deployment for a warm cache.
    pub(super) fn arm_bodies(&mut self) {
        // The FIVE outer clauses of the router's own gate, restated at the
        // one instant that can act on them. `bodies` off is the diagnostic
        // eager arm and arms nothing at all; a mode that records nothing has
        // no cache to arm; and an artifact with more capture units than the
        // key can name is refused from the body path
        // ([`Shell::keyable_units`]), so arming it would pay captures for
        // execs no fire will ever reach. That clause used to read
        // `CompiledModel::fold_refused` — the FOLD's refusal, borrowed — and
        // the multi-unit bodies wave gave it back: a `record::BodyKey` names a
        // lattice point per unit now, so a TOWER artifact arms here like any
        // other and the tower keys are enumerated beside the four token
        // kinds.
        //
        // **AND A ROTATING LOAD IS THE FOURTH, WHICH IS NOT A CAUTION BUT AN
        // ARITHMETIC.** `Shell::enqueue_on`'s `records` line refuses to
        // record ANY fire whose weights rotate — a rotation's backpressure is
        // a host cursor the walk advances and a replayed graph has no walk,
        // which `crate::rotate` argues in full — and that refusal is
        // permanent for the life of the load, not conditional on a fire.
        // So every rung this loop would climb lands in the router's eager
        // `else`: `record::WARM_FIRES` real executed walks per rung, real
        // device seconds at boot, and not one exec captured at the end of
        // them. The whole of this pass is moving a warm cost off the first
        // caller and onto the load, and under a rotor there is nothing to
        // move it TOWARD — the first caller pays the eager walk either way,
        // and so does the ten-thousandth. Work that can only produce eager
        // walks is refused where it is asked for. The boot line above says
        // the same thing in words, once, for the operator; this is the line
        // that stops the load paying for it.
        //
        // **AND THE PAD IS THE FIFTH, ON THE SAME ARITHMETIC.** `prepare`'s
        // gate will not record a body without an armed lattice point, so every
        // synthetic this pass fired under `[engine] pad off` would compose,
        // stage, be refused the body arm and return having armed nothing. The
        // clause is here for the reason the rotor's is: work that can only
        // produce nothing is refused where it is asked for.
        if !self.bodies
            || !self.pad
            || !self.graphs.records()
            || !Self::keyable_units(&self.compiled)
            || self.weights.rotating()
        {
            return;
        }

        // **THE CEILING IS A DEPLOYMENT FACT AND NOT A TUNING**, and it is
        // read from the one place that states it ([`Shell::lane_ceiling`]),
        // because the same number is in the key this pass is about to name. A
        // decode fire is one row per lane, so its lane count is its row count
        // and the seats bound both.
        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return;
        }
        // A deployment that declared no lattice has no rungs and every row
        // count is its own bucket (`Composition::bucket` is the row count
        // itself), so the rungs ARE the admissible lane counts. One that
        // declared a lattice arms its points — synthesized at the LANE count
        // a real fire of that rung can actually bring, which is the rung
        // itself when the seats allow it and the seat ceiling when they do
        // not. The second case is not a corner: `bucket_of` rounds a fire's
        // rows UP, so a deployment whose seats sit below the lattice floor
        // still serves every decode fire out of the FIRST rung — a rung this
        // loop would otherwise skip entirely, and did, until a four-seat
        // deployment armed nothing.
        let rungs: Vec<(u32, u32)> = if self.budget.buckets.is_empty() {
            (1..=ceiling).map(|n| (n, n)).collect()
        } else {
            let mut rungs = Vec::new();
            for point in self.budget.buckets.iter().copied() {
                if point <= ceiling {
                    rungs.push((point, point));
                } else {
                    // The first rung past the seats: every admissible lane
                    // count above the previous rung rounds up to it.
                    //
                    // **AND ONLY IF THERE IS SUCH A LANE COUNT.** When the
                    // seats land exactly ON the previous rung, the rung below
                    // already arms every decode fire this load can bring and
                    // this point names a bucket the synthesis cannot reach:
                    // `ceiling` lanes are `ceiling` rows, and `ceiling` rows
                    // round DOWN to the rung that holds them. Pushing it
                    // anyway spends a synthetic fire to seat a body under one
                    // key and then look for it under another, which arms
                    // nothing and — since the arming loop now cross-checks
                    // the key it named against the key `prepare` composed —
                    // trips that check besides.
                    if ceiling > rungs.last().map_or(0, |(point, _)| *point) {
                        rungs.push((point, ceiling));
                    }
                    break;
                }
            }
            rungs
        };
        // **AND IT IS THE WHOLE REALIZABLE LATTICE NOW, NOT DECODE ONLY** —
        // the tier-1 key-collapse wave's chunk B, and the paragraph that
        // stood here is retired rather than amended, because both of its
        // reasons died.
        //
        // What it argued was that only a DECODE composition is worth arming,
        // on two grounds. The first was `Body::grids`: "a body may serve a
        // fire only when the capture's per-launch `(rows, lanes)` dominate the
        // fire's, a decode composition at a rung has exactly ONE maximal
        // geometry and the key states it, and a prefill or mixed key has no
        // such corner, its rows and lanes being free of each other inside one
        // key". That was true and is not: the grids are issued at the KEY's
        // ceiling now (`Run::carve_rows`, `Run::carve_lanes`, and
        // `record::launch_grid` is the ledger's twin), so ANY in-key geometry
        // captures the key's maximum and the corner nobody could synthesize
        // stopped being needed.
        //
        // The second was the BUDGET, and its constant is retired with it.
        // `MAX_ARMED_BODIES` was eight — a quarter of `record::MAX_BODIES` —
        // on the argument that "the map has to have room left for traffic",
        // and the decode rungs of a doubling lattice filled it exactly. That
        // reservation stopped having a population to protect when the map was
        // SEALED at the end of this pass: traffic mints no bodies now, so
        // every body under this pass's budget belongs to this enumeration.
        //
        // **AND THE BUDGET IS DEVICE MEMORY, WHICH IS THE ONE BOUND NOBODY HAS
        // TO ARGUE FOR** (the capacity wave). It was a COUNT for two waves —
        // `record::MAX_BODIES`, sixty-four, then thirty-two before it — and
        // every number it held was picked and justified backwards, while what
        // a body actually takes from a deployment is a `cudaGraphExec_t`'s
        // device-side parameters off the card the KV pool pages into. So
        // `[engine] bodies_mem` states the bytes and each capture is weighed
        // as it is instantiated (`record::Body::bytes`): a card with room arms
        // the whole realizable lattice, one without arms the small buckets and
        // walks the rest, and the boot line reports which. The count survives
        // as a BELT for the two costs memory does not price — a load that
        // cannot weigh anything, and boot SECONDS, since what the old
        // constant's other argument said is still true: each key costs
        // `record::WARM_FIRES` EXECUTED walks at load.
        //
        // Either bound is asked of the MAP, per key, so that only a key which
        // actually seats a body spends anything. A refused present set costs
        // one synthetic fire and no budget at all, which is what keeps a baked
        // class table's phantom fact combinations from crowding out the shapes
        // traffic brings.
        //
        // **SO WHAT IS ENUMERATED IS THE KEY SPACE, WHICH IS FINALLY A THING
        // A LOAD CAN WALK.** A `record::BodyKey` is a PRESENT SET and a
        // BUCKET, and both are drawn from load constants — the class table and
        // `Budget::buckets` — with every number in the ladder a function of
        // the pair (`record::Ladder::rung`). SIX kinds of present set are
        // enumerated, one FUNCTION each, and `ARMS` is the list of them:
        //
        // * `decode_keys`, one key per lattice point per decode class, at the
        //   lane count a real decode fire of that rung can bring;
        // * `prefill_keys`, one key per lattice point per NON-decode class;
        // * `mixed_keys`, one key per (decode class x non-decode class) pair
        //   per lattice point;
        // * `fragmented_keys`, one key per (fragmentable region MASK,
        //   separator) per lattice point — THREE classes: the class that
        //   stands between two of the mask's own and the nearest of those two
        //   on either side of it, one lane each (the tier-2 campaign,
        //   `Shell::fragmenting`). **THIS IS THE ONLY ARM THAT SEEKS A
        //   SEGMENTED BODY**, and the reason is arithmetic rather than taste:
        //   the three above present one class or two, a mask over a subset of
        //   two present classes is always one interval, and a window that is
        //   one interval is never gathered, grouped or split. So without it
        //   the tier-2 path would exist with no key to exercise it, and every
        //   composition P4 wrote a `Fallback` row for would walk eagerly past
        //   the seal. What it does NOT do is cover the class-set space, which
        //   is exponential and is not a thing a boot walks — one witness per
        //   mask is this wave's reach and `Shell::fragmenting` states it;
        // * `tower_keys`, one key per (media class, patch rung, token
        //   bucket) — the second capture unit's own lattice;
        // * `ensemble_keys`, one key per lattice point for the WHOLE decode
        //   set, on a bake that has more than one decode word (the hybrid
        //   decode wave). **THIS IS THE ARM A HYBRID SKU'S DECODE TRAFFIC
        //   LANDS IN**, and the reason is the same arithmetic the fragmenting
        //   arm's is: `decode_keys` enumerates decode as a SINGLETON, so a
        //   bake whose template runs two `Attention::Decode` arms — two
        //   classes, both in `Shell::decoding` — had every one of its batched
        //   decode fires present `{c0, c1}` and find no key. Not a corner of
        //   that load's traffic: its steady state, walking eagerly past the
        //   seal for the life of the load.
        //
        // Each states its own argument where it lives; what stays here is the
        // shape of the space and the sentences that are true of all six.
        //
        // **AND THE SYNTHETIC'S OWN SPLIT DOES NOT MATTER, WHICH IS THE
        // SENTENCE THAT MAKES MIXED ARMING POSSIBLE AT ALL.** Every fire of
        // one key grids at the same ceiling, so the capture this pass takes
        // stands for every split of the bucket the key admits — nine prefill
        // rows beside three decode ones, or three beside nine. The synthetic
        // only has to BE a fire of the key: present the right classes, land in
        // the right bucket, and be something the deployment can actually
        // stage, plan and run.
        //
        // **A KEY THE DEPLOYMENT CANNOT FIRE IS REFUSED BY NAME AND NEVER
        // ARMED.** A lane needs a seat, a seat holds `Paging::context` tokens,
        // and a fire needs at least one lane per present class — so a bucket
        // whose rows cannot be spread over the seats this load has is a bucket
        // no caller can bring either. Arming it would spend a synthetic fire
        // to hear a refusal; skipping it silently would leave an operator
        // reading a short armed count with no sentence to explain it. So it is
        // named in the boot line and left out of `wanted`.
        //
        // Ascending buckets, because that is the order the budget is spent
        // in: a lattice wider than `[engine] bodies_mem` can hold never
        // reaches its LARGEST buckets, which are the fires a deployment
        // assembles least often and the ones whose captures cost the most —
        // in device bytes as well as in boot seconds, since a bigger bucket is
        // a bigger graph.
        let seats = self.held.len() as u32;
        let context = self.pools.paging().context();
        let max_lanes = self.budget.max_lanes;
        let classes = self.compiled.classes.classes.len();
        // **AND A TEXT SYNTHETIC MAY NOT LAND IN A MEDIA CLASS, WHICH IS THE
        // MULTI-UNIT WAVE'S OWN CLAUSE HERE AND IS NOT A REFINEMENT** (the
        // multi-unit bodies wave).
        //
        // The five text arms below synthesize a lane by taking its class's
        // REPRESENTATIVE WORD (`Class::word`, `Shell::synthetic_lanes`), which
        // is the one part of a submission this pass is allowed to invent
        // because the sweep's own table is where it comes from. On a two-unit
        // artifact some of those words carry the MEDIA fact — and a lane whose
        // word says media while its submission carries no image is a
        // composition **no caller can present**: the model's `Classify::of`
        // sets that bit for a request that has media and for no other, so a
        // real lane in one of these classes always submits an image.
        //
        // **AND IT IS NOT MERELY WASTED — IT PANICS, AND THAT IS THE SHAPE OF
        // THE BUG THIS CLAUSE EXISTS TO REMOVE.** The embed merge
        // (`layout.scatter_live_rows`) is a TRUNK-unit node guarded on the
        // media fact, so its window is decided by the token class table and is
        // FULL for such a lane; but the fire carries no patch row, so
        // `Shell::enqueue_on` stages no patch seats at all and the merge
        // resolves `RuntimeInput::PatchRoutes` against an unbound one
        // (`Run::whole`'s panic: "value N reads where this fire's tower rows
        // land, and no lane of it submitted an image"). The tower's own
        // regions are safe — they are on the patch axis, their windows have
        // zero rows, and the walk skips a zero-row region — which is exactly
        // why the merge is the node that reaches it: it is the one node that
        // CROSSES the two units.
        //
        // Until this wave `arm_bodies` returned on `CompiledModel::fold_refused`
        // before it enumerated anything, so no such synthetic was ever built.
        // The partition is the same shape as the decode/prefill one beside it:
        // a text lane lands in a non-media class, an image lane lands in a
        // media class, and [`BodySynth::Tower`] is the arm that presents the
        // second half — with an image, which is what makes its merge resolve.
        let textual = |class: usize| !self.media.contains(class);
        // **AND THE ARM IS GATED ON THE SEAT, NOT ON THE LADDER** — three
        // clauses, because a synthetic that composes patch rows it cannot
        // STAGE is the same unbound-seat panic the `textual` clause above
        // removes, arriving from the other side. `Shell::towered` says the
        // PLAN states the axis (a deployment may hand a `PatchLadder` to a
        // text that never asks for one, which is the G4 invariant);
        // `Shell::patch_seat` says the trace declares the patch INPUT, which
        // is where the row width comes from — and a synthetic built at a width
        // of zero has an empty payload, so `enqueue_on` stages nothing while
        // the composition claims rows and the tower's first launch resolves an
        // unbound seat. `Shell::media` being non-empty is the third and is the
        // one that makes a text-only artifact enumerate nothing here.
        let staged = self
            .towered
            .then_some(self.patch_seat)
            .flatten()
            .is_some_and(|seat| seat.row_bytes > 0);
        let patch_rungs: Vec<u32> = match (staged, self.budgets.patches.as_ref()) {
            (true, Some(ladder)) => ladder.buckets.clone(),
            _ => Vec::new(),
        };
        // **AND THE ENUMERATIONS ARE A TABLE, NOT A LOOP NEST EACH**
        // (the quality wave). What each arm reads is stated once, here, as a
        // [`Deployment`]; what each arm DOES is one function; and [`ARMS`] is
        // the list of them in [`Kind::ALL`]'s order, which is the order the
        // boot line's columns are printed in and the order the tally is
        // indexed by. The loop below is unchanged in what it attempts and in
        // what order: each arm pushes its buckets ascending and its keys
        // within a bucket in the order it always did, and the sort beneath is
        // STABLE, so the sequence of attempts is the one this pass has always
        // made.
        let deployment = Deployment {
            rungs,
            buckets: self.budget.buckets.clone(),
            patch_rungs,
            decoders: self
                .decoding
                .iter()
                .filter(|class| textual(*class))
                .collect(),
            prefilling: (0..classes)
                .filter(|class| !self.decoding.contains(*class) && textual(*class))
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
        // Ascending bucket, stable inside it, so the loop below spends the
        // map's seats on the smallest buckets first and whatever it runs out
        // of budget for is the largest.
        targets.sort_by_key(|(bucket, _)| *bucket);
        if targets.is_empty() {
            return;
        }

        // **THE BUDGET IS THE MAP, AND A KEY THAT SEATS NOTHING MUST NOT SPEND
        // IT** (the tier-1 key-collapse wave). The enumeration used to be
        // TRUNCATED to `record::MAX_BODIES` up front, which decided which keys
        // this load would arm before it knew which of them it CAN arm — and a
        // baked class table holds every fact combination the compiler can
        // distinguish, most of which no deployment's traffic ever presents. On
        // a two-decode-class, ten-prefill-class bake the first lattice point
        // alone enumerates thirty-two keys, so a load whose phantom pairs are
        // half of them spent half its map on compositions that refuse and
        // dropped every bucket above the floor to make room.
        //
        // So the loop attempts keys in ascending bucket order and asks the MAP
        // rather than the list: a key that arms spends the bytes its execs
        // took, a key that refuses costs one synthetic fire and nothing else,
        // and the pass stops when `[engine] bodies_mem` is spent (or when the
        // `record::MAX_BODIES` belt is reached, whichever comes first). What
        // is left unvisited is named in the boot line by the bucket it stopped
        // at AND by which of the two stopped it, which is the sentence an
        // operator can act on ("this lattice wants more memory than this
        // deployment allowed it").
        //
        // **AND A PRESENT SET THAT WAS REFUSED IS NOT ASKED AGAIN AT THE NEXT
        // BUCKET — WHICH IS A BUDGET RULE NOW AND NO LONGER A THEOREM** (the
        // tier-2 campaign). It used to be one: the refusal was
        // `Windows::covers_fire_shifted`, which reads the SHAPES of a
        // composition's windows, every one of those is a function of which
        // classes have rows and of the artifact (`window::seat`'s note: two
        // masks resolve to the same span exactly when their present classes
        // are the same set), and the bucket moves no window's shape — so one
        // refusal per present set was the whole of what there was to learn.
        //
        // A window's shape no longer refuses anything: `Windows::admits` makes
        // an ISLAND of it and `record::cuts` cuts the body around it. What is
        // left to learn here is the WIDENING's verdict (`record::Uncut::Eager`
        // — every region an island once the islands have grown to their legal
        // boundaries), and that one CAN move with the bucket, because
        // `fallback::copies` is bucket-keyed: a region that splits above the
        // crossover and gathers below it crosses the admissibility line with
        // the lattice point, and an island the widening spreads over a fork
        // group at one bucket is one region at the other. So this list is kept as what it
        // has to be: a bound on the synthetic fires a wide bake spends, not a
        // proof about them. A set skipped here that a larger bucket would have
        // armed costs exactly one unarmed key, which is what every other
        // unarmed key costs and is counted in the same place
        // (`record::BodyTally::sealed_declines`).
        //
        // This is still the only "which classes can fire" answer this shell
        // can derive: the class table says which fact combinations EXIST, and
        // nothing in the artifact says which of them a deployment's callers
        // will present.
        let mut armed = 0usize;
        let mut wanted = 0usize;
        let mut tally = [(0usize, 0usize); Kind::COUNT];
        let mut refused: Option<String> = None;
        let mut unadmitted: Vec<Vec<usize>> = Vec::new();
        let mut never = 0usize;
        let mut never_from = 0u32;
        // WHICH of the two bounds stopped the pass, taken at the instant it
        // first did — the byte budget or the count belt. An operator's next
        // move is different for each (`[engine] bodies_mem` against a bake
        // whose key space is a phantom class table), so the warning names it.
        let mut belted = false;
        for (bucket, target) in targets {
            // **THE BUDGET, ASKED OF THE MAP, AND IT IS BYTES** (the capacity
            // wave). `insert_body` bounds what is RESIDENT and the arming
            // pass's bodies are pinned, so the live census is the only honest
            // reading of what has been spent — the same argument the seat
            // count was asked under, on the quantity that is actually being
            // spent.
            //
            // **WHY BYTES AND NOT SEATS.** What arming costs a deployment is
            // device memory: a body is a `cudaGraphExec_t`'s device-side node
            // parameters, and the card it comes off is the card the KV pool
            // pages into. `record::MAX_BODIES` used to be the whole bound and
            // it was a count nobody could derive — the smoke SKU's
            // enumeration wants two hundred and forty-eight keys, sixty-four
            // fit, and the warning below had to tell an operator that a
            // hundred and eighty-four compositions walk eagerly with no
            // sentence about why that number. `[engine] bodies_mem` is the
            // number now, spent against what each capture was MEASURED to take
            // (`record::Body::bytes`).
            //
            // **AND THE COUNT SURVIVES AS A BELT, TESTED FIRST BECAUSE IT IS
            // THE ONE THAT HOLDS WHEN THE OTHER CANNOT.** A load with no
            // runtime to weigh with charges zero for every body, and boot
            // SECONDS are a cost memory does not price — `record::MAX_BODIES`
            // argues both. Whichever stops the pass, what is left unattempted
            // is the same and is named in the same warning.
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
            // **AND THE SKIP LIST IS THE TOKEN ARMS' ALONE** (the multi-unit
            // bodies wave). It is keyed on the PRESENT SET, on the argument
            // that a present set an earlier bucket found inadmissible will be
            // inadmissible at every bucket — and that argument is about the
            // TOKEN rectangle's windows, which is all the five token kinds
            // have. A tower target of the same class presents the same token
            // classes and a completely different second rectangle: its regions
            // on the patch axis have windows the text fire's composition does
            // not resolve at all, so a text key's verdict says nothing about
            // it. Conflating them would let one inadmissible prefill key
            // silently unarm every tower key of that class, which is the
            // whole of what this wave exists to enable.
            if target.skips_on_present_set() && unadmitted.contains(&present) {
                // Named, not fired: this present set has already told this
                // load that no composition of it is admissible.
                unfireable.push(format!("bucket {bucket}, {target}: inadmissible present set"));
                continue;
            }
            wanted += 1;
            // **THE GEOMETRY AND THE SECOND AXIS'S SUBMISSION, IN ONE
            // ANSWER** ([`BodySynth::lanes`]) — the lanes as `(class, rows)`
            // pairs, and an `(images, patches)` pair per lane on the one arm
            // that carries an image. Both used to be matches on the variant,
            // two dozen lines apart, agreeing by hand.
            let at = target.kind().at();
            let (lanes, media) = target.lanes();
            tally[at].1 += 1;
            // **AND THE CALL IS THE SAME CALL IT WAS FOR A TEXT FIRE.** The
            // media half is empty for the five token kinds, so the door is
            // picked off the SUBMISSION and not off the kind — which is what
            // keeps a sixth token arm from having to be named here.
            let owned = if media.is_empty() {
                self.synthetic_lanes(&lanes)
            } else {
                self.synthetic_lanes_with(&lanes, &media)
            };
            self.armed_body = None;
            // A `Fault` is this GEOMETRY's, not necessarily this present set's
            // — a planner may take a bucket and refuse the one above it — so a
            // key that faulted teaches the skip list nothing and is not
            // allowed to speak for its set's other lattice points.
            let mut faulted = false;
            for _ in 0..record::WARM_FIRES {
                let fired = self.fire_synthetic(&owned);
                // **AND EACH KEY'S FIRES ARE SETTLED BEFORE THE NEXT ONE**,
                // which is the one thing this loop does that a real fire path
                // must never do — and it is control plane, at load, with no
                // caller waiting. Two reasons, both structural rather than
                // cautious:
                //
                // * the settlement pool holds one event per IN-FLIGHT step
                //   (`Settlement::claim` answers `Fault::Ceiling`, not a
                //   wait), and this loop issues far more steps than a
                //   run-ahead depth without a `read_out` to bound them. A
                //   fire path is bounded by the caller's frames; this is
                //   bounded by nothing but the sync;
                // * and `insert_body`'s replacement and eviction paths both
                //   ask `Airborne::settled_past`. A key whose predecessors
                //   are all settled asks that question against a quiet
                //   device, so an arming refusal means what it says instead
                //   of meaning "the last key had not landed yet".
                let landed = self.device.synchronize();
                if let Err(why) = fired.and(landed) {
                    // A synthetic geometry this load will not stage, plan or
                    // land. The key is lost and the next one is not.
                    refused = Some(format!("bucket {bucket}, {target}: {why}"));
                    faulted = true;
                    break;
                }
            }
            // **THE KEY THE SYNTHETIC ACTUALLY BUILT, TAKEN FROM THE ONE
            // INSTANT THAT COMPOSED IT** (`Shell::armed_body`, written by
            // `prepare`). A prefill or mixed ladder has an ORDER — ascending
            // row offset, which `fire::compose` decides from the artifact's
            // baked class order — and this loop knows the classes it asked
            // for, not the order they were seriated into. Reconstructing it
            // here would be a second answer waiting to disagree with the one
            // the cache is keyed on, which is precisely the bug
            // `record::Ladder::rung`'s own note describes on the rung axis.
            //
            // The decode arm keeps building its key by hand, because that is
            // the arm the `Ladder::single` constructor exists for and its
            // single-class ladder has no order to lose — and the `debug_assert`
            // is what says the two readings agree.
            //
            // **AND THE ENSEMBLE ARM DOES NOT, WHICH IS THE WHOLE OF ITS
            // BIT-IDENTITY ARGUMENT** (the hybrid decode wave). Its ladder is
            // multi-class: it has an ORDER, seriated from the artifact's baked
            // class order by `fire::compose`, and the enumerator knows only
            // which decode words it asked for. So it takes the key `prepare`
            // composed, which IS `record::BodyKey::of_axes` over this fire's
            // own window table — the same call, on the same axis, that a real
            // two-word decode fire at this bucket makes. Same present set,
            // same seriation, and each class's rung
            // `record::Ladder::rung(class, bucket, decoding, lane_ceiling)` =
            // `min(lane_ceiling, bucket)` for a decode class. The armed key
            // and the traffic key are one key, bit for bit, because they are
            // one function of one pair.
            //
            // **AND WHETHER THE COMPOSITION WAS ADMITTED AT ALL IS READ FIRST,
            // OFF THE SAME WORD.** `prepare` writes `armed_body` on exactly
            // the arming fires its gate admitted, so `None` here is the
            // admissibility rule turning this PRESENT SET away — the same
            // answer it gives at every other bucket, which is why the set goes
            // on the skip list rather than being asked again once per lattice
            // point. It is read before the match because the decode arm builds
            // its key by hand and would otherwise report `Some` for a fire
            // nothing admitted.
            let admitted = self.armed_body.is_some();
            let key = match &target {
                BodySynth::Decode { class, .. } => {
                    let rung = record::Ladder::rung(*class, bucket, &self.decoding, ceiling);
                    let built = record::BodyKey {
                        bucket,
                        classes: record::Ladder::single(*class, rung),
                        // **A DECODE SYNTHETIC CARRIES NO IMAGE**, so on a
                        // tower artifact its key is the AXIS-EMPTY one — a
                        // `Some` whose bucket is zero and whose ladder is
                        // empty, which is the rung that launches no tower
                        // exec (`record::AxisKey`). `None` on a text-only
                        // one, which is what makes this literal the literal
                        // it always was there.
                        patch: self.towered.then(|| record::AxisKey {
                            bucket: 0,
                            classes: record::Ladder::default(),
                        }),
                    };
                    debug_assert!(
                        self.armed_body.as_ref().is_none_or(|armed| *armed == built),
                        "the decode arming built {built} and the synthetic fire composed \
                         {:?}",
                        self.armed_body,
                    );
                    Some(built)
                }
                _ => self.armed_body.take(),
            };
            // **ASKED OF THE CACHE AND NOT OF THE RETURN VALUE.** A fire that
            // came back `Ok` may still have declined to seat — the schedule
            // was not graph-shaped, or the map had no droppable body — and
            // both are tallied inside `fire_body` already. What "armed" means
            // is that the key holds an exec now, so that is what is asked.
            //
            // **AND THE ANSWER IS ALSO WHERE THE PIN IS WRITTEN**
            // (`record::Body::pinned`): this call is the one instant in the
            // engine that can tell a body the LOAD armed from one traffic
            // minted, because the capture itself went down `fire_body`
            // indistinguishable from a warm key's. It seats the exemption
            // from the bodies map's LRU at the same line it counts the key,
            // so the two can never disagree.
            if key.is_some_and(|key| self.cache.body_armed(&key)) {
                armed += 1;
                tally[at].0 += 1;
            } else if !admitted && !faulted && target.skips_on_present_set() {
                // **AND A TOWER TARGET WRITES NOTHING INTO IT**, which is the
                // other half of the clause above: its verdict is about a
                // second rectangle the token arms do not have, so recording
                // it against a token present set would turn one unarmed tower
                // key into every unarmed text key of that class.
                unadmitted.push(present);
            }
        }

        // **THE BOOT LINE, BECAUSE A PARTIAL ARM MUST NOT BE SILENT.** An
        // operator who states `[engine] bodies` is buying "every fire
        // replays"; a load that armed nine of thirteen keys has bought it for
        // nine shapes and, since the SEAL below, has bought the other four an
        // eager walk for the life of the load. That is the one fact this pass
        // produces and this is the only place it exists.
        //
        // **PER COMPOSITION KIND, BECAUSE THAT IS THE AXIS AN OPERATOR CAN
        // ACT ON.** A short decode count is a deployment whose seats sit under
        // its lattice; a short prefill count is a context or a lane ceiling; a
        // short mixed count is usually both; and a short ENSEMBLE count on a
        // hybrid bake is the one that says this load's ordinary decode traffic
        // has no body at all (the hybrid decode wave). One total would say
        // "something was lost" and nothing else.
        //
        // **AND `wanted` IS WHAT WAS ATTEMPTED, NOT WHAT WAS ENUMERATED**,
        // which is the arithmetic the budget change forced. A key the loop
        // never fired is not a key this load "wanted and missed": it is either
        // a present set already known inadmissible or a key past the map's
        // last seat, and both have their own clause below. Counting them in
        // the denominator would report `22 of 182` on a load that armed
        // everything its map can hold.
        //
        // **AND THE FOUR WAYS A KEY CAN BE LOST WITHOUT FAILING ANYTHING**,
        // stated when they happened and absent when they did not. A `Fault`
        // from the synthetic fire is the `refused` sentence; a bucket this
        // deployment cannot synthesize at all — or a present set an earlier
        // bucket already proved inadmissible — is `unfireable`, named before a
        // fire was spent on it; a lattice wider than this deployment's budget
        // is the `never` warning, which names the bucket the pass ran out at
        // and which of the two bounds ran out; and
        // two are quiet by construction because `record::Graphs::fire_body`
        // counts them rather than returning them — a composition the
        // admissibility rule turned away is `refusals`, and a schedule that
        // would not fit its workspace grant is `declines`, which under the
        // bucket ceiling is a property of the KEY and so is permanent for it.
        //
        // **AND HOW MANY OF THEM ARE SEGMENTED** (the tier-2 campaign,
        // `record::BodyCensus::segmented`). A body with an island replays
        // through an EAGER stretch every fire — the stretch's launches are
        // re-issued on the host, one at a time, between two
        // `cudaGraphLaunch`es — so a load whose armed bodies are mostly
        // segmented is a load whose replay is buying less than the whole of
        // what a graph can buy. It is not a warning: the alternative for those
        // compositions is the eager walk end to end, which is strictly worse.
        // It is the number that says which SKUs are worth a `crate::SHIFTED`
        // look, which is the seat-first half of this campaign's discipline.
        let stats = self.cache.body_stats();
        // **AND THE COLUMNS ARE AN ITERATION OF [`Kind::ALL`]**, so that the
        // one place a kind's name is spelled is its `Display` and the one
        // place a kind's counts live is the tally slot [`Kind::at`] names.
        // The text is what it always was — `decode a/b, prefill a/b, ...` in
        // this order, separated by `, ` — and the columns that existed before
        // each wave print byte for byte what they printed.
        //
        // **AND HOW MANY OF THE SECOND UNIT'S KEYS ARMED** (the multi-unit
        // bodies wave). `0/0` on every text-only SKU, because `Shell::media`
        // is empty there and the arm enumerates nothing — which is the same
        // `0/0` a plan with no decode arm reads in the first column, and is
        // the G4 invariant on this line: a text load's boot sentence gains one
        // field that says "no tower", and no field it had says anything
        // different. A short tower count on a vision SKU is a deployment whose
        // token buckets cannot hold the placeholder rows its patch rungs owe,
        // which is the thing the `never fired` list names.
        let columns = Kind::ALL
            .iter()
            .map(|kind| format!("{kind} {}/{}", tally[kind.at()].0, tally[kind.at()].1))
            .collect::<Vec<String>>()
            .join(", ");
        eprintln!(
            "engine-cuda: bodies armed {armed} of {wanted} compositions at load \
             ({columns}; {} segmented; {} MiB of {} MiB, {} of {} seats){}{}{}{}",
            stats.census.segmented,
            // **AND WHAT IT COST, AGAINST WHAT IT WAS ALLOWED** (the capacity
            // wave). Both bounds are printed on every load and not only when
            // one of them bit, because this is the line an operator sizes
            // `[engine] bodies_mem` FROM: a load reading `40 MiB of 1024 MiB`
            // has a lattice its card could hold four times over, and one
            // reading `1021 MiB of 1024 MiB` beside a truncation warning has
            // the knob to turn. Megabytes because that is the unit the knob is
            // stated in, floored rather than rounded so the number never reads
            // larger than what was taken.
            //
            // **A ZERO ON A LOAD THAT ARMED SOMETHING IS NOT A BUG.** It means
            // no body could be weighed — no runtime to ask, or a driver that
            // refused the query — and the seats beside it are then the only
            // bound that was in force (`record::Body::bytes`).
            stats.census.bytes >> 20,
            self.bodies_mem >> 20,
            stats.census.bodies,
            record::MAX_BODIES,
            match &refused {
                Some(why) => format!(" (last refusal: {why})"),
                None => String::new(),
            },
            if unfireable.is_empty() {
                String::new()
            } else {
                format!(
                    " [{} key(s) never fired — this deployment cannot synthesize \
                     them, or their present set was already refused admission; \
                     e.g. {}]",
                    unfireable.len(),
                    unfireable[0],
                )
            },
            if never == 0 {
                String::new()
            } else {
                format!(
                    " [WARNING: {never} key(s) never attempted — {} at bucket \
                     {never_from}, so every key from there up walks eagerly for the \
                     life of this load; {}]",
                    if belted {
                        "the count belt record::MAX_BODIES was reached"
                    } else {
                        "[engine] bodies_mem was spent"
                    },
                    if belted {
                        "a lattice this wide is usually a bake's phantom class \
                         table rather than a deployment's shapes"
                    } else {
                        "raise it if this card has the memory"
                    },
                )
            },
            match (stats.tally.declines, stats.tally.refusals) {
                (0, 0) => String::new(),
                (declines, refusals) => format!(
                    " [{declines} declined a workspace grant, {refusals} inadmissible]"
                ),
            },
        );

        // **AND NOW THE MAP IS CLOSED** (`record::Graphs::seal_bodies`). The
        // enumeration above walked every key this deployment can realize, so
        // what is left unarmed is not a key that is behind the traffic — it is
        // one this pass could not fire or chose to drop, and a serving fire
        // that minted it would be paying `record::WARM_FIRES` eager walks, a
        // capture and an instantiation on somebody's critical path to reach a
        // decision the boot already made. Past this line the bodies path
        // mints nothing: a key with no body walks and is counted
        // (`record::BodyTally::sealed_declines`).
        //
        // **ONLY IF SOMETHING WAS ACTUALLY ARMED.** A pass that armed zero has
        // proved nothing about this deployment — every key refused, or the
        // enumeration held none — and sealing on it would turn a load that used
        // to warm its bodies from traffic into a load with no bodies at all.
        // That is strictly worse than the behaviour this wave replaced, and it
        // is the one direction a seal must not be wrong in.
        if armed > 0 {
            self.cache.seal_bodies();
        }
    }

    /// **THE MINIMAL PRESENT SETS THAT BREAK A WINDOW** — the arming pass's
    /// fourth enumeration, and the only one that SEEKS a SEGMENTED body
    /// (the tier-2 campaign).
    ///
    /// # Why the singleton and pair arms cannot
    ///
    /// A fire orders its classes by the artifact's shipped order with the
    /// absent ones dropped, and dropping a class can only CLOSE a gap
    /// (`model_exec::fire::fallback::bound` carries the argument). So a mask
    /// over a subset of ONE or TWO present classes is always one interval, and
    /// the decode-only, prefill-only and mixed enumerations — which present
    /// one class and two — can never produce a fire P4 answers a `Fallback`
    /// for. Every window of every one of their compositions is a single span,
    /// which is either whole-fire or windowed-and-shifting, which is
    /// `Admit::Captured`. They arm bodies with no islands, always, and that is
    /// not a property anybody chose.
    ///
    /// It takes a THIRD class standing between two of a mask's own to put
    /// foreign rows inside that mask's span. Then the shell resolves the
    /// region as a split (`r` windows), a gathered rectangle (`Fallback::Copy`
    /// and `[engine] copies`) or a grouped segment list — and the last two are
    /// islands, which is what tier 2 was built for.
    ///
    /// **AND [`BodySynth::Ensemble`] MAY REACH ONE BY ACCIDENT, WHICH CHANGES
    /// NOTHING HERE** (the hybrid decode wave). A bake with three or more
    /// decode words presents three or more classes in one ensemble key, and
    /// three classes is exactly the width that can break a mask — so such a
    /// key MAY come back segmented, and `record::cuts` will have cut it. That
    /// is a fact about the composition that arm arms for its own reason (it is
    /// the traffic a hybrid load brings), not a second witness enumeration:
    /// it walks the DECODE set and nothing else, so it demonstrates whichever
    /// breaks that set happens to contain and none of the ones it does not.
    /// The witnesses below are still the only systematic reach at the
    /// segmented path.
    ///
    /// # The MINIMAL sets only, and MINIMAL means THREE CLASSES
    ///
    /// The set of present sets is exponential in the class count and no boot
    /// pass can walk it. What a boot CAN walk is the minimal witnesses: for
    /// each distinct region mask and each class that stands between two of
    /// that mask's own, the THREE classes that witness it — the separator,
    /// the nearest mask class in front of it and the nearest one behind it.
    /// Three, because two of a mask's classes with a foreign one between them
    /// is the whole of what makes a window two intervals, and the mask's other
    /// classes are rows on one side or the other of a break that has already
    /// happened.
    ///
    /// **AND THAT IS THE FIX, NOT A REFINEMENT.** This enumerated the mask's
    /// WHOLE class set plus a separator, which is a present set of `|mask| + 1`
    /// classes — and a fire needs a lane per present class and a seat per
    /// lane, so on a four-seat deployment a four-class mask armed nothing at
    /// all: every one of its keys was named in the boot line's `never fired`
    /// list and the composition traffic actually brings — three classes, one
    /// lane each — was never armed, fell past the seal and walked for the life
    /// of the load. The count is unchanged (one witness per mask per
    /// separator, deduplicated); what changed is that the sets are the sets a
    /// caller can present.
    ///
    /// A load whose traffic presents a LARGER superset — the witness plus a
    /// fourth class — keys to a body this does not arm and, past the seal,
    /// walks (`record::BodyTally::sealed_declines` is where that shows).
    /// Widening past minimal is a lattice question — how many class sets a
    /// deployment's callers can realize — and not a capture one, so it is
    /// named here rather than guessed at.
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
                // **AND NO WITNESS MAY NAME A MEDIA CLASS** (the multi-unit
                // bodies wave). Every lane this arm synthesizes is a TEXT
                // lane — one class, some rows, no image — and a lane whose
                // word carries the media fact without an image is a
                // composition no caller can present and one the embed merge
                // panics on (`Shell::arm_bodies`' `textual` clause carries the
                // whole derivation). A witness that needs one is a break this
                // pass cannot demonstrate with text alone; it is left unarmed
                // and its fires walk past the seal, counted like every other
                // unarmed key.
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

    /// **THE THREE CLASSES THAT WITNESS ONE SEPARATOR BREAKING ONE MASK**, or
    /// `None` when this separator does not stand BETWEEN two of the mask's
    /// classes at all — [`Shell::fragmenting`]'s minimal set.
    ///
    /// The neighbours are read off the order the mask's own classes and the
    /// separator seriate to (`ClassOrder::class_order`), and they are still
    /// the neighbours in the TRIPLE's order because dropping classes only
    /// closes gaps: a `Seriated` order filters one fixed frontier and an
    /// `Identity` one is ascending, so no subset ever reorders what it keeps.
    /// That is the same property `model_exec::fire::fallback::bound` rests on
    /// and the reason a three-class witness is enough.
    ///
    /// `None` for a separator that sits in front of every one of the mask's
    /// classes or behind all of them: it closes no gap and opens none, and a
    /// present set built around it would be a key with a whole window in it —
    /// which the decode, prefill and mixed arms already arm.
    fn witness(
        compiled: &CompiledModel,
        mask: &model_ir::ClassSet,
        separator: usize,
    ) -> Option<Vec<usize>> {
        let mut whole: Vec<usize> = mask.iter().collect();
        whole.push(separator);
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(whole.iter().copied()), None);
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

    /// **DOES `mask` COVER MORE THAN ONE INTERVAL OF THE ORDER `present`
    /// SERIATES TO?** — [`Shell::fragmenting`]'s one predicate.
    ///
    /// Asked of `ClassOrder::class_order` and not of the fallback table,
    /// because the question is about a HYPOTHETICAL fire's row order and the
    /// table answers about the shipped one. `model_exec::fire::compose` builds
    /// a fire's order the same way, from the same call, so a present set this
    /// says breaks a mask is one whose fire will find that window in pieces.
    fn breaks(compiled: &CompiledModel, mask: &model_ir::ClassSet, present: &[usize]) -> bool {
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(present.iter().copied()), None);
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

    /// **SPREAD `rows` OVER AT MOST `lanes` LANES OF AT MOST `context` ROWS
    /// EACH**, or `None` for a total this deployment cannot present — the
    /// geometry half of a prefill or mixed arming key.
    ///
    /// `lanes` is the caller's own reading of how many the class may take — a
    /// prefill-only key may use every seat, a mixed one has already spent a
    /// seat on its decode lane — and it is narrowed to `rows` here, because a
    /// lane with no row is `fire::Fault::EmptyLane`.
    ///
    /// **THE ONLY THING THE SPLIT HAS TO BE IS FIREABLE.** Which lane carries
    /// which row does not reach the key — a `record::BodyKey` holds the
    /// present set and the bucket, and the ceilings beside them are functions
    /// of that pair — and it does not reach the CAPTURE either, because the
    /// launches are gridded at those ceilings and not at this fire's split
    /// (`Run::carve_rows`). So the even spread below is chosen for being the
    /// one that fits soonest: it is the split that minimises the tallest lane,
    /// which is the number a slot's context bounds.
    ///
    /// `None` says the deployment cannot hold this total at all — no lane to
    /// put a row in, or more rows than `lanes x context` cells — and the
    /// caller names it in the boot line rather than spending a synthetic fire
    /// to be told the same thing by a planner.
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

/// **ONE LANE PER CLASS OF A FRAGMENTING PRESENT SET, AT ROW COUNTS THAT
/// LAND ON `bucket`** — [`BodySynth::Fragmented`]'s geometry, or `None`
/// for a set this deployment cannot fire at this lattice point.
///
/// A free function over the [`Deployment`] rather than a method on the
/// shell, because it is where [`fragmented_keys`] puts its whole geometry
/// and the six arms read their facts from one place.
///
/// **A DECODE CLASS TAKES EXACTLY ONE ROW**, because one row per lane is
/// what makes a fire a decode and a lane whose word says so with three
/// tokens in it is a composition no caller can bring. The rows left over
/// go to the non-decode classes, spread evenly by [`Shell::spread`] for
/// the reason it always spreads evenly: the split does not reach the key,
/// so the only thing it has to be is fireable.
///
/// `None` when the deployment cannot seat one lane per class, when the
/// lattice point has fewer rows than the set has classes, or when the set
/// is ALL decode classes — the last because their total is the class count
/// itself, which lands on whichever bucket holds it and not on the one
/// being enumerated.
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

/// **THE HYBRID DECODE ARM, ASKED WITHOUT A DEVICE** — [`ensemble_keys`] is a
/// pure function of a [`Deployment`], so the claim that a two-decode-word load
/// enumerates the composition its traffic brings is a HOST claim and belongs
/// where a plain workspace sweep runs it.
///
/// **AND THAT IS WHERE IT HAS TO LIVE, BECAUSE THE HARDWARE GATE CANNOT CARRY
/// IT ALONE.** The defect was found on a `gemma4-e4b` load — two decode words
/// out of a hybrid ATTENTION bake — and that SKU has no CI gate on this tree.
/// The smoke SKU reaches two decode words the other way (the adapter fact
/// crossing `qo_one`, classes `1` and `3`), which
/// `bodies_gate::a_two_word_decode_fire_replays_from_the_load` fires on real
/// hardware; what neither of them can do is state the enumeration's shape at
/// three words and at one. This does.
#[cfg(test)]
mod tests {
    use super::{BodySynth, Deployment, Targets, ensemble_keys};

    /// A deployment with `decoders` decode classes and one rung — every other
    /// field at the value that makes the other five arms enumerate nothing,
    /// because this module is about one arm.
    fn deployment(decoders: usize, rung: (u32, u32)) -> Deployment {
        Deployment {
            rungs: vec![rung],
            buckets: Vec::new(),
            patch_rungs: Vec::new(),
            decoders: (0..decoders).collect(),
            prefilling: Vec::new(),
            media: Vec::new(),
            fragmenting: Vec::new(),
            decoding: model_ir::ClassSet::of(0..decoders),
            seats: rung.1,
            context: 512,
            max_lanes: rung.1,
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

    /// **ONE DECODE WORD ENUMERATES NOTHING HERE**, because the singleton IS
    /// the full set and `decode_keys` already armed it. A key armed twice is a
    /// capture spent to look one body up under its own name.
    #[test]
    fn a_single_decode_word_arms_no_ensemble() {
        let mut found = Targets::default();
        ensemble_keys(&deployment(1, (256, 256)), &mut found);
        assert!(
            found.targets.is_empty() && found.unfireable.is_empty(),
            "a one-word bake enumerated an ensemble key, which is the decode \
             singleton under a second name"
        );
    }

    /// **TWO DECODE WORDS ARM THE PAIR, AND THE COMPOSITION ROUNDS TO THE
    /// RUNG.** This is the shape the `gemma4-e4b` boot line said no body held:
    /// a present set of both decode words at the rung's own lane count, one
    /// row per lane, so the total is the lane count and the lane count is what
    /// `Shell::arm_bodies` derived as reachable at this lattice point.
    #[test]
    fn two_decode_words_arm_the_pair_at_the_rungs_lane_count() {
        let mut found = Targets::default();
        ensemble_keys(&deployment(2, (256, 256)), &mut found);
        assert_eq!(found.targets.len(), 1, "one key per rung, and there is one rung");
        let (bucket, target) = &found.targets[0];
        assert_eq!(*bucket, 256);
        assert_eq!(
            lanes(target),
            vec![(0, 128), (1, 128)],
            "the rung's lanes are split across the two decode words"
        );
        // The present set is BOTH words — which is the whole defect: nothing
        // else in `ARMS` produces a multi-decode one.
        assert_eq!(target.present(), vec![0, 1]);
        // One row per lane, and the rows total the rung's lane count, so
        // `Composition::bucket` lands on this key's bucket by construction —
        // the same arithmetic `BodySynth::Decode` rides at the same rung.
        let (rows, media) = target.lanes();
        assert!(media.is_empty(), "an ensemble lane submits no image");
        assert_eq!(rows.len(), 256, "one lane per row");
        assert!(rows.iter().all(|(_, rows)| *rows == 1), "a decode lane is one row");
        assert_eq!(rows.iter().map(|(_, rows)| *rows).sum::<u32>(), 256);
    }

    /// **THREE WORDS ARM THE FULL SET AND NO SUBSET**, which is this arm's
    /// stated reach: the singletons are `decode_keys`', the top is this, and
    /// the layers between are the never-attempted long tail every other arm
    /// accepts. An odd lane count still gives every word a lane.
    #[test]
    fn three_decode_words_arm_the_whole_set_and_nothing_between() {
        let mut found = Targets::default();
        ensemble_keys(&deployment(3, (8, 8)), &mut found);
        assert_eq!(found.targets.len(), 1, "the full set, and no pair out of three");
        assert_eq!(lanes(&found.targets[0].1), vec![(0, 3), (1, 3), (2, 2)]);
        assert_eq!(found.targets[0].1.present(), vec![0, 1, 2]);
    }

    /// **A RUNG WITH FEWER LANES THAN THE LOAD HAS DECODE WORDS IS NAMED,
    /// NOT FIRED** — the seat arithmetic every arm is under, on the axis this
    /// one counts in: a fire needs a lane per present class and a lane needs a
    /// seat, so a two-seat load has no three-word decode fire to arm.
    #[test]
    fn a_rung_that_cannot_seat_a_lane_per_word_is_named() {
        let mut found = Targets::default();
        ensemble_keys(&deployment(3, (8, 2)), &mut found);
        assert!(found.targets.is_empty(), "a two-lane rung armed a three-word key");
        assert_eq!(found.unfireable.len(), 1, "and it was named rather than silent");
        assert!(
            found.unfireable[0].contains("ensemble"),
            "the sentence an operator reads does not say which arm lost the key: {}",
            found.unfireable[0],
        );
    }
}
