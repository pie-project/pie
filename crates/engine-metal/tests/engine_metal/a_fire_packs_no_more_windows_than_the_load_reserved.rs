//! The window region is carved at load, and `Inputs::write` refuses a fire
//! whose packed boundaries overrun it: a fire packs no more than the load
//! reserved, and a region past its run bound is refused before any window
//! is packed. Device-free: the class table and the bound arithmetic only.

use engine_metal::Fault;
use engine_metal::inputs::window_ints;
use engine_metal::store::kv::Paging;
use engine_metal::window::{Copies, Windows};
use poem_compiler::{Budget, CompiledModel, DeviceProfile, compile};
use poem_exec::fire::fallback::gathers;
use poem_exec::fire::{ClassWindow, WindowTable, fallback};
use poem_ir::{ClassSet, Platform, Trace};

const SERVED: &str = "qwen35-d0.8b-u4g64-kv-bf16";

/// Every rung at or above `poem_compiler`'s copy/split crossover — 512 rows
/// scaled by this profile's 132 SMs over the 82 they were measured on, so
/// 825. Below it the bake writes `Fallback::Copy` rows, and one gathered mask
/// buys the reservation `3 * rows + ...` ints of slack no cut here could
/// outgrow.
const LATTICE: [u32; 2] = [1024, 4096];

/// The row ceiling every fire in this file stands at.
const ROWS: u32 = 4096;

/// One page table, which is what the reservation is told to carve for.
const PAGE_TABLES: usize = 1;

/// A deployment of `lanes` lanes at the row ceiling. One lane is the shape
/// that overran in production: the base term pays `lanes + 1` per window, so
/// a wide fire hides a cut behind its own slack.
fn budget(lanes: u32) -> Budget {
    Budget {
        max_lanes: lanes,
        max_tokens: ROWS,
        buckets: LATTICE.to_vec(),
        max_adapters: 8,
        readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
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
    // The whole deployment name, not the text it serves: a prefix match also takes
    // the `-eagle` and `-vision` rows, whose names start with this text.
    let trace = poem_compiler::catalog::deployment(SERVED)
        .unwrap_or_else(|| panic!("the catalog no longer ships {SERVED}"))
        .trace(Platform::Metal);
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

/// The class the most regions' masks hold.
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

/// The windows this fire cuts. Copies are off, so nothing is gathered
/// whatever the bake decided.
fn fire(trace: &Trace, compiled: &CompiledModel, fixture: &Fixture) -> Windows {
    let no_patches = WindowTable::new(vec![ClassWindow::default(); compiled.classes.classes.len()]);
    Windows::of(
        trace,
        compiled,
        &fixture.classes,
        &no_patches,
        &no_patches,
        &fixture.indptr,
        Copies::off(),
    )
    .expect("a fire whose classes are consecutive is one the artifact promised")
}

/// The window ints this load carves: the number `Inputs::write` holds a
/// fire's packed boundaries against.
fn reserved(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> u64 {
    window_ints(
        budget,
        paging(),
        PAGE_TABLES,
        compiled.classes.classes.len(),
        gathers(trace, compiled),
    )
}

/// One gathered window's payload: the row tables and the per-table pool
/// tables `Windows::packed` writes AFTER the window's own boundaries.
fn per_gathered(budget: &Budget) -> u64 {
    let rows = u64::from(budget.max_tokens);
    let lanes = u64::from(budget.max_lanes);
    let pages = u64::from(budget.max_lanes) * u64::from(paging().pages_per_slot);
    3 * rows + PAGE_TABLES as u64 * (2 * lanes + (lanes + 1) + pages)
}

/// What `Inputs::reserve` carved before the cut and the passes were counted.
fn before(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> u64 {
    let classes = compiled.classes.classes.len();
    let lanes = u64::from(budget.max_lanes);
    (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1)
        + gathers(trace, compiled) as u64 * per_gathered(budget)
}

#[test]
fn a_fire_packs_no_more_than_the_load_reserved() {
    let budget = budget(1);
    let (trace, compiled) = baked(&budget);

    // `baked` holds this fixture to no gathered mask, so the correction the
    // next leg pins is zero here and the two expressions coincide. On a load
    // that gathers, the new expression is deliberately the larger.
    assert_eq!(
        reserved(&trace, &compiled, &budget),
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
    let none = window_ints(&budget, paging(), PAGE_TABLES, classes, 0);
    let one = window_ints(&budget, paging(), PAGE_TABLES, classes, 1);
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
    let packed = fire(&trace, &compiled, &fixture).packed().len() as u64;
    let reserved = reserved(&trace, &compiled, &budget);
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

    let refused = Windows::of(
        &trace,
        &compiled,
        &fixture.classes,
        &no_patches,
        &no_patches,
        &fixture.indptr,
        Copies::off(),
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
