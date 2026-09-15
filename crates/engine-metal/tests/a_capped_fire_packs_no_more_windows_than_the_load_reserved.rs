//! The window region is carved at load, before any run cap or pass count is
//! known, and `Inputs::write` refuses a fire whose packed boundaries overrun
//! it. Every capped case here first asserts that the fire outgrows what the
//! load carved BEFORE the cut terms existed, so the bound is read against a
//! fire that would really have overrun. Device-free: the class table and the
//! bound arithmetic only.

use engine_metal::Fault;
use engine_metal::inputs::window_ints;
use engine_metal::store::kv::Paging;
use engine_metal::window::{Copies, Windows, gathers};
use model_compiler::{Budget, CompiledModel, DeviceProfile, compile};
use model_exec::fire::{ClassWindow, WindowTable, fallback, max_runs};
use model_ir::{ClassSet, Platform, RowAxis, Trace};

const SERVED: &str = "qwen35-d0.8b-u4g64-kv-bf16";

/// Every rung at or above `model_compiler`'s copy/split crossover — 512 rows
/// scaled by this profile's 132 SMs over the 82 they were measured on, so
/// 825. Below it the bake writes `Fallback::Copy` rows, and one gathered mask
/// buys the reservation `3 * rows + ...` ints of slack no cut here could
/// outgrow.
const LATTICE: [u32; 2] = [1024, 4096];

/// The row ceiling every fire in this file stands at.
const ROWS: u32 = 4096;

/// One kv space, which is what the reservation is told to carve for.
const SPACES: usize = 1;

/// A deployment of `lanes` lanes at the row ceiling. One lane is the shape
/// that overran in production: the base term pays `lanes + 1` per window, so
/// a wide fire hides a cut behind its own slack.
fn budget(lanes: u32) -> Budget {
    Budget {
        max_lanes: lanes,
        max_tokens: ROWS,
        buckets: LATTICE.to_vec(),
        max_adapters: 8,
    }
}

fn paging() -> Paging {
    Paging::of(16, 256, 64, 1_024).expect("a page size of 16 is a paging")
}

fn profile() -> DeviceProfile {
    DeviceProfile {
        side_streams: 0,
        ..DeviceProfile::default()
    }
}

/// The served text, baked for this plane at this deployment's ceilings.
fn baked(budget: &Budget) -> (Trace, CompiledModel) {
    // The whole SKU name, not the text it serves: a prefix match also takes
    // the `-eagle` and `-vision` rows, whose names start with this text.
    let trace = models::sku(SERVED)
        .unwrap_or_else(|| panic!("the catalog no longer ships {SERVED}"))
        .trace;
    let trace = trace(Platform::Metal);
    let compiled = compile(&trace, budget, &profile()).expect("the served text bakes");
    assert_eq!(
        gathers(&trace, &compiled),
        0,
        "the bake wrote a `Fallback::Copy` row at a lattice of {LATTICE:?}, \
         which is meant to stand entirely above the crossover; a gathered \
         mask makes the reservation so much larger than the cut that every \
         claim in this file would hold against the unfixed expression too"
    );
    (trace, compiled)
}

/// One fire's tables: which rows and lanes each class owns, and the qo
/// boundaries beside them.
struct Fixture {
    classes: WindowTable,
    indptr: Vec<i32>,
}

/// Every row and the one lane in a single class. A mask either covers that
/// class or it does not, so no region is ever fragmented and the cut's pieces
/// are almost the whole packed blob.
fn one_lane(compiled: &CompiledModel) -> Fixture {
    let mut table = vec![ClassWindow::default(); compiled.classes.classes.len()];
    table[busiest(compiled)] = ClassWindow {
        row_offset: 0,
        rows: ROWS,
        lane_offset: 0,
        lanes: 1,
    };
    Fixture {
        classes: WindowTable::new(table),
        indptr: vec![0, ROWS as i32],
    }
}

/// Two classes with a lane each, consecutive and unevenly split, so a mask
/// over both is one span of every row while a mask over either alone is a
/// narrower one. One lane cannot state two widths, which is why the two
/// cases that need them cost a second lane.
fn two_lanes(compiled: &CompiledModel, first: u8, second: u8) -> Fixture {
    let mut table = vec![ClassWindow::default(); compiled.classes.classes.len()];
    table[first as usize] = ClassWindow {
        row_offset: 0,
        rows: ROWS / 4,
        lane_offset: 0,
        lanes: 1,
    };
    table[second as usize] = ClassWindow {
        row_offset: ROWS / 4,
        rows: ROWS - ROWS / 4,
        lane_offset: 1,
        lanes: 1,
    };
    Fixture {
        classes: WindowTable::new(table),
        indptr: vec![0, (ROWS / 4) as i32, ROWS as i32],
    }
}

/// One row and one lane per class in shipped class order, with a gap between
/// every pair. `fallback::bound` counts a mask's runs in that order, and no
/// two of these classes are adjacent, so a mask the order keeps together
/// still falls into one span per class — more than its bound.
fn every_class_apart(compiled: &CompiledModel) -> Fixture {
    let classes = compiled.classes.classes.len();
    let order = compiled.order.class_order(&ClassSet::of(0..classes));
    let mut table = vec![ClassWindow::default(); classes];
    for (at, &class) in order.iter().enumerate() {
        table[class as usize] = ClassWindow {
            row_offset: 2 * at as u32,
            rows: 1,
            lane_offset: at as u32,
            lanes: 1,
        };
    }
    Fixture {
        classes: WindowTable::new(table),
        indptr: (0..=classes as i32).collect(),
    }
}

/// The windows this fire cuts under `caps`/`passes`. Copies are off, so
/// nothing is gathered whatever the bake decided.
fn fire(
    trace: &Trace,
    compiled: &CompiledModel,
    fixture: &Fixture,
    caps: &[u32],
    passes: &[u32],
) -> Windows {
    let no_patches = WindowTable::new(vec![ClassWindow::default(); compiled.classes.classes.len()]);
    Windows::of(
        trace,
        compiled,
        &fixture.classes,
        &no_patches,
        &no_patches,
        &fixture.indptr,
        Copies::off(),
        caps,
        passes,
    )
    .expect("a fire whose classes are consecutive is one the artifact promised")
}

/// The window ints this load carves for these caps — the number
/// `Inputs::write` holds a fire's packed boundaries against.
fn reserved(
    trace: &Trace,
    compiled: &CompiledModel,
    budget: &Budget,
    caps: &[u32],
    passes: &[u32],
) -> u64 {
    window_ints(
        budget,
        paging(),
        SPACES,
        compiled.classes.classes.len(),
        max_runs(compiled),
        gathers(trace, compiled),
        caps,
        passes,
    )
}

/// One gathered window's payload: the row tables and the per-space pool
/// tables `Windows::packed` writes AFTER the window's own boundaries.
fn per_gathered(budget: &Budget) -> u64 {
    let rows = u64::from(budget.max_tokens);
    let lanes = u64::from(budget.max_lanes);
    let pages = u64::from(budget.max_lanes) * u64::from(paging().pages_per_slot);
    3 * rows + SPACES as u64 * (2 * lanes + (lanes + 1) + pages)
}

/// What `Inputs::reserve` carved before the cut and the passes were counted.
fn before(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> u64 {
    let classes = compiled.classes.classes.len();
    let lanes = u64::from(budget.max_lanes);
    (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1)
        + gathers(trace, compiled) as u64 * per_gathered(budget)
}

/// Without this the `<=` that follows would hold against the old expression
/// too, and the case would pin nothing.
fn outgrows_the_old_reservation(packed: u64, before: u64) {
    assert!(
        packed > before,
        "this fire packs {packed} ints and the pre-fix reservation carved \
         {before}; the case no longer reproduces the overrun it exists to \
         pin, so the `<=` below would hold against the unfixed expression too"
    );
}

/// Class intervals: what the bound pays a window group for.
fn intervals(compiled: &CompiledModel) -> u64 {
    let classes = compiled.classes.classes.len() as u64;
    classes * (classes + 1) / 2
}

/// A cap that cuts `rows` into at least `pieces` pieces. Floor division, so
/// `rows / cap >= pieces` however it rounds.
fn cap_for(rows: u32, pieces: u64) -> u32 {
    let pieces = u32::try_from(pieces).unwrap_or(u32::MAX);
    (rows / pieces.max(1)).max(1)
}

/// Enough pieces that the cut alone outgrows the whole pre-fix reservation,
/// with room for the floor division above and for a second, coarser group.
fn pieces_wanted(compiled: &CompiledModel) -> u64 {
    4 * (intervals(compiled) + 2)
}

/// The class the most template regions cover: a one-lane fire puts every row
/// in one class, and this is the class that leaves the most regions with
/// something to cut.
fn busiest(compiled: &CompiledModel) -> usize {
    (0..compiled.classes.classes.len())
        .max_by_key(|&class| {
            compiled
                .template()
                .iter()
                .filter(|region| region.mask.contains(class))
                .count()
        })
        .expect("the bake states at least one class")
}

/// Token-axis regions, widest interval first. A patch region's window states
/// no boundaries at all, and a region whose mask covers no rows is never cut.
fn token_regions(compiled: &CompiledModel, plain: &Windows) -> Vec<usize> {
    let mut out: Vec<usize> = (0..compiled.template().len())
        .filter(|&at| matches!(compiled.axis_of(at), RowAxis::Tokens))
        .filter(|&at| widest(plain, at) > 0)
        .collect();
    out.sort_unstable_by_key(|&at| std::cmp::Reverse(widest(plain, at)));
    out
}

/// The widest interval a region's mask covers on an uncapped fire — what
/// `pass_spans` sizes that region's pass count from.
fn widest(fire: &Windows, at: usize) -> u32 {
    (0..fire.runs(at as u32))
        .map(|run| fire.at(at as u32, run).span.rows)
        .max()
        .unwrap_or(0)
}

fn zeros(compiled: &CompiledModel) -> Vec<u32> {
    vec![0u32; compiled.template().len()]
}

/// A two-lane fire and two of its regions the class table gives different
/// widths, which is what makes `pass_spans` hand them different pass counts.
/// Only neighbouring classes merge into one span, so only adjacent pairs are
/// tried.
fn two_widths(trace: &Trace, compiled: &CompiledModel) -> Option<(Fixture, usize, usize)> {
    let classes = compiled.classes.classes.len();
    let order = compiled.order.class_order(&ClassSet::of(0..classes));
    for pair in order.windows(2) {
        let fixture = two_lanes(compiled, pair[0], pair[1]);
        let plain = fire(trace, compiled, &fixture, &[], &[]);
        let regions = token_regions(compiled, &plain);
        for (i, &first) in regions.iter().enumerate() {
            let other = regions[i + 1..]
                .iter()
                .find(|&&other| widest(&plain, other) != widest(&plain, first));
            if let Some(&other) = other {
                return Some((fixture, first, other));
            }
        }
    }
    None
}

#[test]
fn two_regions_under_different_caps_both_fit_the_reservation() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);
    let fixture = one_lane(&compiled);
    let plain = fire(&trace, &compiled, &fixture, &[], &[]);
    let regions = token_regions(&compiled, &plain);
    assert!(
        regions.len() >= 2,
        "the {SERVED} template no longer holds two token-axis regions over \
         the class this fire's lane is in"
    );

    let fine = cap_for(ROWS, pieces_wanted(&compiled));
    let coarse = fine * 2;
    let mut caps = zeros(&compiled);
    caps[regions[0]] = fine;
    caps[regions[1]] = coarse;
    let passes = zeros(&compiled);

    let packed = fire(&trace, &compiled, &fixture, &caps, &passes)
        .packed()
        .len() as u64;
    outgrows_the_old_reservation(packed, before(&trace, &compiled, &budget));
    let reserved = reserved(&trace, &compiled, &budget, &caps, &passes);
    assert!(
        packed <= reserved,
        "a fire capping two regions at {fine} and {coarse} packed {packed} \
         ints into a reservation of {reserved}; windows cut at different caps \
         do not deduplicate against each other, so the reservation must ADD \
         the two cut groups and not take the wider of them"
    );

    // The same claim read off the fire: capping the second region on top of
    // the first adds windows rather than landing on the first's.
    let mut only_fine = zeros(&compiled);
    only_fine[regions[0]] = fine;
    let mut only_coarse = zeros(&compiled);
    only_coarse[regions[1]] = coarse;
    let first = fire(&trace, &compiled, &fixture, &only_fine, &passes)
        .packed()
        .len() as u64;
    let other = fire(&trace, &compiled, &fixture, &only_coarse, &passes)
        .packed()
        .len() as u64;
    assert!(
        packed > first && packed > other,
        "capping both regions packed {packed} ints, no more than capping one \
         of them alone ({first} and {other}); the two cut groups deduplicated \
         against each other, and a reservation that took the wider group \
         instead of adding them would have been right after all"
    );
}

/// The only case that reaches the chunk arm's grouping with more than one
/// region in a group: two token-axis regions, one cap, no pass ceiling.
#[test]
fn two_regions_under_one_cap_take_one_chunk_group_and_fit() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);
    let fixture = one_lane(&compiled);
    let plain = fire(&trace, &compiled, &fixture, &[], &[]);
    let regions = token_regions(&compiled, &plain);
    assert!(
        regions.len() >= 2,
        "the {SERVED} template no longer holds two token-axis regions over \
         the class this fire's lane is in"
    );

    let cap = cap_for(ROWS, pieces_wanted(&compiled));
    let mut caps = zeros(&compiled);
    caps[regions[0]] = cap;
    caps[regions[1]] = cap;
    let passes = zeros(&compiled);

    let packed = fire(&trace, &compiled, &fixture, &caps, &passes)
        .packed()
        .len() as u64;
    outgrows_the_old_reservation(packed, before(&trace, &compiled, &budget));
    let reserved = reserved(&trace, &compiled, &budget, &caps, &passes);
    assert!(
        packed <= reserved,
        "two regions capped at {cap} packed {packed} ints into a reservation \
         of {reserved}; one cut group covers both, and the group's term must \
         bound the base spans of every region carrying the cap"
    );

    // A group's term is `min(intervals, regions * max_runs)`, so the region
    // count shows only where the run bound is the smaller factor. That needs
    // a class table narrow enough for the clamp to bite, not the baked one.
    let narrow = 2usize;
    let base = window_ints(&budget, paging(), SPACES, narrow, 2, 0, &[], &[]);
    let alone = window_ints(&budget, paging(), SPACES, narrow, 2, 0, &[8], &[0]);
    let shared = window_ints(&budget, paging(), SPACES, narrow, 2, 0, &[8, 8], &[0, 0]);
    assert!(
        shared > alone,
        "a second region under the same cap left the reservation at {shared}; \
         the group holds both regions' base spans and its bound must grow \
         with the count"
    );
    assert!(
        shared - base < 2 * (alone - base),
        "two regions under one cap reserve {} ints of cut where two separate \
         groups would reserve {}; they were charged a group each and the \
         grouping arm counted nothing",
        shared - base,
        2 * (alone - base)
    );
}

#[test]
fn a_pass_replicated_region_fits_the_reservation() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);
    let fixture = one_lane(&compiled);
    let plain = fire(&trace, &compiled, &fixture, &[], &[]);
    let regions = token_regions(&compiled, &plain);
    assert!(
        !regions.is_empty(),
        "the {SERVED} template no longer holds a token-axis region over the \
         class this fire's lane is in"
    );

    let mut caps = zeros(&compiled);
    let mut passes = zeros(&compiled);
    caps[regions[0]] = cap_for(ROWS, pieces_wanted(&compiled));
    // Half what the cap would have cut, so `pass_spans` clamps here and the
    // fire walks exactly the ceiling the reservation was handed.
    let ceiling = u32::try_from(pieces_wanted(&compiled)).unwrap_or(u32::MAX);
    passes[regions[0]] = ceiling;

    let fire = fire(&trace, &compiled, &fixture, &caps, &passes);
    assert!(
        (0..fire.runs(regions[0] as u32)).any(|run| fire.at(regions[0] as u32, run).passes > 1),
        "the capped region came back with one pass; `pass_spans` declined to \
         replicate and this test is about a fire that did not happen"
    );

    let packed = fire.packed().len() as u64;
    outgrows_the_old_reservation(packed, before(&trace, &compiled, &budget));
    let reserved = reserved(&trace, &compiled, &budget, &caps, &passes);
    assert!(
        packed <= reserved,
        "a region walked up to {ceiling} times packed {packed} ints into a \
         reservation of {reserved}; a pass is its own window even over the \
         same rows, so the reservation owes one group per pass"
    );
}

#[test]
fn a_multi_lane_window_states_a_boundary_per_lane_and_still_fits() {
    let budget = budget(2);
    let (trace, compiled) = baked(&budget);
    let (fixture, wide, narrow) = two_widths(&trace, &compiled).expect(
        "no two classes of the baked template give two regions different \
         widths; the two-lane fixture this case needs cannot be built",
    );

    // The narrow region is the one cut, so the wide region's window is left
    // to state a boundary per lane.
    let plain = fire(&trace, &compiled, &fixture, &[], &[]);
    let mut caps = zeros(&compiled);
    caps[narrow] = cap_for(widest(&plain, narrow), pieces_wanted(&compiled));
    let passes = zeros(&compiled);

    let fire = fire(&trace, &compiled, &fixture, &caps, &passes);
    assert!(
        (0..compiled.template().len() as u32)
            .flat_map(|at| (0..fire.runs(at)).map(move |run| (at, run)))
            .any(|(at, run)| fire.at(at, run).indptr_host.len() > 2),
        "no window of a two-lane fire states more than two boundaries, so the \
         `lanes + 1` the reservation pays per window is untested; region \
         {wide} was meant to cover both lanes"
    );

    let packed = fire.packed().len() as u64;
    outgrows_the_old_reservation(packed, before(&trace, &compiled, &budget));
    let reserved = reserved(&trace, &compiled, &budget, &caps, &passes);
    assert!(
        packed <= reserved,
        "a two-lane fire packed {packed} ints into a reservation of \
         {reserved}; a window costs one boundary per lane it stands over, \
         plus the terminator"
    );
}

/// The `<=` leg here does not discriminate: the ceiling is well above either
/// region's count, so a reservation that paid one pass group per cap would
/// satisfy it too. The discriminating claim is the assertion below that the
/// two regions came out with different pass counts.
#[test]
fn two_regions_under_one_cap_and_one_pass_ceiling_both_fit() {
    let budget = budget(2);
    let (trace, compiled) = baked(&budget);
    let (fixture, wide, narrow) = two_widths(&trace, &compiled).expect(
        "no two classes of the baked template give two regions different \
         widths; two regions under one cap can no longer come out with \
         different pass counts, which is the case this test exists for",
    );

    let plain = fire(&trace, &compiled, &fixture, &[], &[]);
    // `pass_spans` sizes a region's count off the widest interval its OWN
    // mask covers, so one cap over two widths is two counts. The ceiling is
    // well above either, or both would clamp to it and come out equal.
    let cap = cap_for(widest(&plain, wide), pieces_wanted(&compiled));
    let ceiling = u32::try_from(4 * pieces_wanted(&compiled)).unwrap_or(u32::MAX);
    let mut caps = zeros(&compiled);
    let mut passes = zeros(&compiled);
    for at in [wide, narrow] {
        caps[at] = cap;
        passes[at] = ceiling;
    }

    let fire = fire(&trace, &compiled, &fixture, &caps, &passes);
    assert!(
        fire.at(wide as u32, 0).passes != fire.at(narrow as u32, 0).passes,
        "regions {wide} and {narrow} came out with the same pass count after \
         all: `pass_spans` clamped both to {ceiling}, and the per-region sum \
         this test is about is untested"
    );

    let packed = fire.packed().len() as u64;
    outgrows_the_old_reservation(packed, before(&trace, &compiled, &budget));
    let reserved = reserved(&trace, &compiled, &budget, &caps, &passes);
    assert!(
        packed <= reserved,
        "regions {wide} and {narrow} share a cap and a pass ceiling and \
         packed {packed} ints into a reservation of {reserved}; their pass \
         counts are their own, windows key on the count, so the reservation \
         must sum the pass groups PER REGION and not once per cap"
    );
}

#[test]
fn an_uncapped_load_reserves_exactly_what_it_always_did() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);

    // `baked` holds this fixture to no gathered mask, so the correction the
    // next leg pins is zero here and the two expressions coincide. On a load
    // that gathers, the new expression is deliberately the larger.
    assert_eq!(
        reserved(&trace, &compiled, &budget, &[], &[]),
        before(&trace, &compiled, &budget),
        "a load that caps no run and gathers no mask must reserve the number \
         it reserved before the cut terms existed; anything else is a silent \
         change to every deployment's store size"
    );

    // What no fire in this file can show: `packed` writes a gathered window's
    // own boundaries before its payload, so one gathered mask costs the
    // payload AND a boundary vector. Pure arithmetic on the public expression.
    let classes = compiled.classes.classes.len();
    let lanes = u64::from(budget.max_lanes);
    let none = window_ints(&budget, paging(), SPACES, classes, 1, 0, &[], &[]);
    let one = window_ints(&budget, paging(), SPACES, classes, 1, 1, &[], &[]);
    assert_eq!(
        one - none - per_gathered(&budget),
        lanes + 1,
        "one gathered mask moved the reservation by {} ints over its {} of \
         payload; the pre-fix expression paid the payload alone and left the \
         gathered window's own `indptr_host` unreserved",
        one - none,
        per_gathered(&budget)
    );

    let fixture = one_lane(&compiled);
    let packed = fire(&trace, &compiled, &fixture, &[], &[]).packed().len() as u64;
    let reserved = reserved(&trace, &compiled, &budget, &[], &[]);
    assert!(
        packed <= reserved,
        "an uncapped fire packed {packed} ints into a reservation of {reserved}"
    );
}

/// The refusal the tightened reservation rests on. A cut group's base spans
/// are bounded by `max_runs` only because a region that falls into more spans
/// than its own bound is refused BEFORE the cut runs — so a capped fire that
/// fragments past the bound must come back as a fault with no window packed.
#[test]
fn a_region_past_its_run_bound_is_refused_before_any_window_is_packed() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);
    let fixture = every_class_apart(&compiled);
    let no_patches = WindowTable::new(vec![ClassWindow::default(); compiled.classes.classes.len()]);

    let cap = cap_for(ROWS, pieces_wanted(&compiled));
    let mut caps = zeros(&compiled);
    for (at, slot) in caps.iter_mut().enumerate() {
        if matches!(compiled.axis_of(at), RowAxis::Tokens) {
            *slot = cap;
        }
    }

    let refused = Windows::of(
        &trace,
        &compiled,
        &fixture.classes,
        &no_patches,
        &no_patches,
        &fixture.indptr,
        Copies::off(),
        &caps,
        &zeros(&compiled),
    );
    match refused {
        Err(Fault::Fragmented { region, runs, .. }) => {
            let at = region as usize;
            let bound = fallback::bound(
                &compiled,
                compiled.axis_of(at),
                &compiled.template()[at].mask,
            );
            assert!(
                runs > bound as usize,
                "region {at} fell into {runs} runs against a bound of {bound} \
                 and was refused for P4's promise instead; the reservation \
                 bounds a cut group by that bound, and this fixture no longer \
                 fragments any region past one"
            );
        }
        Ok(built) => panic!(
            "a fire that scatters every class packed {} windows under a cap; \
             the fragmentation check has moved after the cut, and the \
             reservation's `max_runs` factor now bounds nothing",
            built.len()
        ),
        Err(other) => panic!("the fire was refused, but not as fragmented: {other:?}"),
    }
}

/// The figure that forced the tightening. Class intervals bound how many base
/// spans a cut group holds, but so does the artifact's own run bound, and at a
/// wide class table the second is smaller by three orders of magnitude. Pure
/// arithmetic on the public expression: no bake, no fire, no device.
#[test]
fn a_wide_class_table_reserves_by_the_run_bound_and_not_by_interval_count() {
    let classes = 64usize;
    let budget = budget(64);
    let lanes = u64::from(budget.max_lanes);
    let intervals = (classes * (classes + 1) / 2) as u64;
    let base = (intervals + 1) * (lanes + 1) * 4;
    assert_eq!(intervals, 2_080);

    // One region cut at 8 rows a piece, against an artifact whose every
    // region the layout seated whole: the group holds one base span, not 2080.
    let pieces = u64::from(ROWS).div_ceil(8);
    assert_eq!(pieces, 512);
    let chunk_loose = intervals * pieces * (lanes + 1) * 4;
    let chunk_tight = window_ints(&budget, paging(), SPACES, classes, 1, 0, &[8], &[0]) * 4;
    assert_eq!(chunk_loose, 276_889_600);
    assert!(
        chunk_tight < 2 * base,
        "one region capped at 8 rows reserves {chunk_tight} bytes, where the \
         interval bound alone would have reserved {chunk_loose} for the cut; \
         a chunk group's spans are bounded by how many regions carry the cap \
         times the run bound, and a seated artifact owes one span here"
    );

    // The pass arm carries the same factor per region, and 60 capped regions
    // at 16 passes make it the larger of the two holes.
    let caps = vec![8u32; 60];
    let passes = vec![16u32; 60];
    let pass_loose = intervals * 60 * 16 * (lanes + 1) * 4;
    let pass_tight = window_ints(&budget, paging(), SPACES, classes, 1, 0, &caps, &passes) * 4;
    assert_eq!(pass_loose, 519_168_000);
    assert!(
        pass_tight < 2 * base,
        "60 regions walked 16 times reserve {pass_tight} bytes, where the \
         interval bound alone would have reserved {pass_loose} for the \
         passes; one region yields at most the run bound in base spans, \
         whatever the class table's width"
    );
}
