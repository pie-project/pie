//! **THE PLANE-BASE FLIP, ON AN ARTIFACT THAT CAN ACTUALLY TAKE IT** (the
//! bodies design's chunk 2b-ii) — with no device in the room.
//!
//! # Why the subject is hand-built
//!
//! The bodies path keys on the COMPOSITION and replays one exec for every row
//! count a bucket admits. Chunk 2b-i built the wide admissibility reading
//! (`Windows::covers_fire_shifted`) and wired nothing to it; 2b-ii flips the
//! gate and moves the launch plane under it — `Run::plane_base` hands a
//! shifting region the PLANE's base pointers, `Run::live_at` arms the staged
//! `(count, start)` seat for it, and the kernel reaches its own rows through
//! `win[1]` instead of through a pointer the host advanced.
//!
//! **AND SINCE THE TIER-2 CAMPAIGN THE ARITHMETIC IS ANSWERED PER REGION.**
//! The clauses did not move an inch — `Windows::admits` asks exactly what
//! `covers_fire_shifted` asked — but it answers a table of
//! `Admit::{Captured, Island}` instead of one `bool`, and the collapse to a
//! `bool` is now one reading of that table rather than the arithmetic itself.
//! What changed for `Shell::prepare` is what it SPENDS the answer on: a
//! composition with one unrecordable region used to be a composition no body
//! served, and it is now a body captured in the stretches that ARE
//! recordable, with the islands re-issued eagerly between the execs
//! (`record::Cut`, `record::cuts`). So this file's subject is the same
//! subject and its claim is one clause wider: the shift is what admits a
//! region INTO a body, and taking it away costs that region and not the
//! composition. Both readings are asserted below, against each other.
//! None of that could be demonstrated on a catalog SKU when this file was
//! written, and the reason was a fact about the catalog rather than about
//! this shell. A real mixed fire is mixed precisely at its ATTENTION: the
//! regions that are windowed are the ones guarded on prefill-vs-decode, and
//! `attention.prefill`, `attention.decode` and the dense GEMMs were all OFF
//! [`SHIFTED`] for reasons `engine_cuda`'s own list states one by one.
//!
//! Chunk 2c-b moved the five FA2 names onto the list, so a mixed fire is no
//! longer refused for THAT reason and `bodies_gate.rs` fires one — the
//! plan-op regions are what stands in front of it now, and that gate says so
//! where it meets it. This file keeps its synthetic subject anyway, and the
//! reason is the one it always had: what it asserts is the shell's own
//! ARITHMETIC over a window table, one region at a time, with no model
//! between the claim and the check. The dense GEMMs are still off the list,
//! so a catalog SKU cannot be all-shifting whatever attention does.
//!
//! What this file does instead is build a trace whose every op IS on the list
//! — three `elementwise.layernorm_no_scale` chains and a two-way class split —
//! and ask the shell's own arithmetic what it says about two fires of it.
//!
//! ```text
//! (a) the premise: every node of the subject is named by `SHIFTED`, so the
//!     per-region reading `exports::regions_shifting` takes is all-true, and
//!     the fire really is windowed — some region does not begin at row zero
//! (b) the flip: the same window table the NARROW gate refuses
//!     (`Windows::covers_fire`) is one the WIDE gate admits
//!     (`covers_fire_shifted`), which is the line chunk 2b-ii changed in
//!     `Shell::prepare` — and, since tier 2, the same clauses read per region
//!     (`Windows::admits`): cripple one region's shift and exactly ONE entry
//!     becomes an `Admit::Island` while the rest stay `Captured`, which is
//!     what says the two readings are one arithmetic
//! (c) the staleness hazard, stated as arithmetic: two fires of ONE
//!     `record::BodyKey` — same classes, same bucket, and therefore the same
//!     per-class CEILING, since a rung is a function of those two and of
//!     nothing a fire measured — whose per-LAUNCH row counts disagree, with
//!     one launch GROWING. That is the
//!     fire the old `Body::rows: u32` would have replayed: its total did not
//!     move, so a resident body would have looked current, and a launch
//!     recorded over three rows would have been asked for five. It is why
//!     `Body::grids` is a per-launch vector now.
//! ```
//!
//! # What this file does NOT prove, said plainly
//!
//! It does not fire, and it does not cut. `record::cuts` — which turns the
//! table part (b) checks into the segment script a body is captured as, and
//! which is where a fork group or a conditional bracket refuses a boundary —
//! has its own unit tests in `record.rs`; this file stops at the table.
//!
//! And the device half of the claim — that a body captured at
//! one row split and REPLAYED at another produces the bytes an eager fire of
//! the second split produces — needs a loaded shell, and a loaded shell needs
//! a checkpoint and a contract for this hand-built text. That fixture is real
//! work (a written `ztensor` container, a `ModelContract` over the trace's own
//! params, and a readout for a plan with no attention in it) and it is not
//! here. `bodies_gate.rs` is the device gate for the SINGLE-CLASS half of the
//! path; the mixed-fire replay gate is owed, and this file is the host-side
//! argument that the shell now admits the fire that gate would need.
//!
//! ```text
//! cargo test -p engine-cuda --test a_shifted_body_admits_the_split_that_moved
//! ```

use engine_cuda::record::BodyKey;
use engine_cuda::window::{Copies, Windows};
use model_compiler::{Budget, CompiledModel, DeviceProfile, compile};
use model_exec::fire::{Composition, Lane, compose};
use model_ir::ops::Elementwise;

/// A slot table generously above every hand-built fire below — the tests ask
/// about window semantics, not the carve, so the ceiling only has to hold.
/// The last three are what one GATHERED payload is bounded by (rows, kv
/// spaces, pages), which `Slots` owns since the tail acquired a stride.
fn test_slots() -> engine_cuda::window::Slots {
    engine_cuda::window::Slots::new(8, 512, 8, 1, 4096, 4, 4096)
}
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Operands, Platform, RuntimeInput, Seam, Trace, Ty,
    ValueDecl, ValueId,
};

/// The activation width. Nothing depends on it; a row has to be some number
/// of elements wide.
const WIDTH: u64 = 8;

/// The ceilings. `Budget::new` states no lattice, so a fire's bucket is its
/// own row count — which is what makes the two splits below share a bucket by
/// sharing a total.
fn budget() -> Budget {
    Budget::new(4, 64)
}

/// **NO CLASS OF THIS TEXT IS A DECODE CLASS**, which is the whole of what
/// the key's rung arithmetic needs to know about it.
///
/// A `record::BodyKey` carries a rung PER CLASS (the ceiling design's Option
/// B), and since the tier-1 key collapse a rung is a CEILING the key's own
/// coordinates spell rather than anything a fire measured: the bucket for a
/// prefill class, the load's lane ceiling for a decode one
/// (`record::Ladder::rung`). This trace runs no `attention.decode` arm at
/// all, so `Shell::decoding` over it is empty and both classes below are
/// carved to the bucket — which is why the two splits share a key without the
/// test having to say anything about rows at all.
fn no_decode_class() -> model_ir::ClassSet {
    model_ir::ClassSet::default()
}

/// The lane ceiling a load of this budget would state (`Shell::lane_ceiling`
/// is `min(slots, max_lanes, max_tokens)`, and `max_lanes` is four here, which
/// binds). Nothing below is a decode class, so nothing below reads it; it is
/// passed because the key's arithmetic takes it.
const LANES: u32 = 4;

fn act() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(WIDTH)],
        dtype: Dtype::Bf16,
    }
}

struct Build {
    trace: Trace,
}

impl Build {
    fn new() -> Build {
        Build {
            trace: Trace {
                name: "hand-built shifted split".to_string(),
                platform: Platform::Cuda,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
            },
        }
    }

    fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    /// One row-plane op, guarded.
    ///
    /// **`elementwise.layernorm_no_scale` AND NOT ITS RMS SIBLING**, which is
    /// the whole reason the subject is built out of this op: the layernorm arm
    /// is named by [`SHIFTED`] and the rmsnorm one is not (its per-head
    /// launches flatten `rows x heads` into `blockIdx.x`, so the launcher
    /// seats `ABSENT` under some settings of a field the NAME admits). It also
    /// takes no weight, which is what lets this trace declare no params and
    /// therefore need no checkpoint to be compiled.
    fn op(&mut self, x: ValueId, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), act());
        self.trace.nodes.push(Node {
            op: Elementwise::LayernormNoScale { x, eps: 1e-6, y }.into(),
            guard,
            layer: None,
        });
        y
    }
}

/// **THE SUBJECT**: a shared head, a two-way split on one fact, a merge, and
/// a shared tail — design §0's window-split at its smallest, with every op on
/// the list.
///
/// The two guarded arms are what make a fire WINDOWED. `Guard::Fact(0)`'s arm
/// runs for the class whose word has bit 0 and nothing else, so its region's
/// rows are that class's slice of the fire and its window begins wherever the
/// class order puts it — which for one of the two arms is never row zero.
fn subject() -> Trace {
    let mut b = Build::new();
    let tokens = b.value(Def::Input(RuntimeInput::Tokens), act());
    let head = b.op(tokens, Guard::Always);
    let hot = b.op(head, Guard::Fact(0));
    let cold = b.op(head, Guard::not(Guard::Fact(0)));
    let merged = b.value(
        Def::Merge(vec![
            (hot, Guard::Fact(0)),
            (cold, Guard::not(Guard::Fact(0))),
        ]),
        act(),
    );
    let y = b.op(merged, Guard::Always);
    b.trace.seams.push(Seam {
        seam: "out".to_string(),
        values: vec![y],
        layer: None,
    });
    b.trace
}

fn baked() -> (Trace, CompiledModel) {
    let trace = subject();
    let compiled =
        compile(&trace, &budget(), &DeviceProfile::default()).expect("the subject bakes");
    (trace, compiled)
}

/// **THE TEST'S OWN READING OF [`SHIFTED`]**, one `bool` per region — the same
/// walk `exports::regions_shifting` takes, spelled here because that function
/// is crate-private and this is an integration test.
///
/// It is the same rule and it is stated the same way: ALL rather than ANY,
/// because one guard-only op in a region addresses the wrong row for the whole
/// region's launch, and a node index the trace does not hold reads as NOT
/// shifting.
fn shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace
                    .nodes
                    .get(node as usize)
                    .is_some_and(|node| engine_cuda::SHIFTED.contains(&Operands::name(&node.op)))
            })
        })
        .collect()
}

/// The fire-wide qo boundaries, in the composed row order the windows are cut
/// against.
/// **AND THE LANE AXIS'S ANSWER FOR THIS SUBJECT**, which is `true`
/// everywhere and is a statement rather than a waiver
/// (`exports::regions_lane_shifting`, `engine_cuda::LANE_SHIFTED`).
///
/// That reading admits a region when every op in it either finds its own lane
/// absolutely or NAMES NOTHING LANE-INDEXED — no cache space, no lane-shaped
/// rectangle. This subject is `elementwise.layernorm_no_scale` end to end over
/// row-shaped values and declares no cache at all, so no op in it can be
/// handed a `lane_offset`-advanced pointer and the second clause answers for
/// every region. Which is the point: this file varies where a window's ROWS
/// begin, and the lane clause is not what it is testing.
fn lane_shifting(compiled: &CompiledModel) -> Vec<bool> {
    vec![true; compiled.template().len()]
}

fn boundaries(fire: &Composition) -> Vec<i32> {
    let mut lanes: Vec<(u32, u32)> = fire
        .lanes()
        .iter()
        .map(|lane| (lane.row_offset, lane.rows))
        .collect();
    lanes.sort_unstable();
    let mut out = vec![0i32];
    for (_, rows) in lanes {
        out.push(out[out.len() - 1] + rows as i32);
    }
    out
}

/// One composition's window table.
fn windows(trace: &Trace, compiled: &CompiledModel, fire: &Composition) -> Windows {
    Windows::of(
        trace,
        compiled,
        model_ir::PerAxis::new([fire.classes(), fire.patch_classes()]),
        &boundaries(fire),
        Copies::off(),
        test_slots(),
    )
    .expect("every region seats a window")
}

/// **THE PER-LAUNCH ROW COUNTS**, in the order the walk makes its launches —
/// the row half of `record::launch_grids`' own layout, spelled here for the
/// same reason `shifting` is.
fn launch_rows(compiled: &CompiledModel, table: &Windows) -> Vec<u32> {
    let mut rows = Vec::new();
    for region in 0..compiled.template().len() as u32 {
        for run in 0..table.runs(region) {
            rows.push(table.at(region, run).span().rows);
        }
    }
    rows
}

/// Two lanes, one in each class, at the stated row counts.
fn split(compiled: &CompiledModel, hot: u32, cold: u32) -> Composition {
    compose(compiled, &budget(), &[Lane::new(1, hot), Lane::new(0, cold)])
        .expect("the two-class fire composes")
}

/// **(a) THE PREMISE.** Every node is on the list, so every region moves its
/// own base — and the fire really is windowed, which is what makes the
/// question worth asking.
#[test]
fn every_region_of_the_subject_addresses_off_the_seat() {
    let (trace, compiled) = baked();
    let shifted = shifting(&trace, &compiled);
    assert!(
        shifted.iter().all(|&moves| moves),
        "the subject was built out of `SHIFTED` names and the reading says \
         otherwise: {shifted:?}",
    );
    assert!(
        !compiled.template().is_empty(),
        "a template of no regions would make every claim below vacuous",
    );

    let fire = split(&compiled, 5, 3);
    let table = windows(&trace, &compiled, &fire);
    let offset = (0..compiled.template().len() as u32).any(|region| {
        (0..table.runs(region)).any(|run| {
            let window = table.at(region, run);
            window.span().rows > 0 && window.span().row_offset > 0
        })
    });
    assert!(
        offset,
        "no region of this fire begins above row zero, so it is not the \
         windowed fire this file is about",
    );
}

/// **(b) THE FLIP, AND THE PER-REGION READING OF IT.** The narrow gate
/// refuses this table and the wide one admits it — what chunk 2b-ii bought at
/// the gate — and `Windows::admits` says the same thing one region at a time,
/// which is what `Shell::prepare` spends now (the tier-2 campaign).
#[test]
fn the_gate_the_narrow_reading_refuses_is_one_the_wide_reading_admits() {
    let (trace, compiled) = baked();
    let shifted = shifting(&trace, &compiled);
    let fire = split(&compiled, 5, 3);
    let table = windows(&trace, &compiled, &fire);

    assert!(
        !table.covers_fire(fire.rows()),
        "the narrow reading admitted a two-class fire, so this subject is not \
         windowed and proves nothing",
    );
    assert!(
        table.covers_fire_shifted(fire.rows(), &shifted, &lane_shifting(&compiled)),
        "every region of this fire is either the whole fire or one whose ops \
         all read the seat's start, and the wide gate refused it anyway",
    );

    // AND THE WIDE READING IS NOT A BLANKET YES: take the shift away from one
    // region and the refusal comes back, which is what says the `shifted`
    // slice is what is doing the admitting.
    let mut crippled = shifted.clone();
    let windowed = (0..compiled.template().len())
        .find(|&region| {
            (0..table.runs(region as u32)).any(|run| {
                let window = table.at(region as u32, run);
                window.span().rows > 0 && window.span().row_offset > 0
            })
        })
        .expect("(a) found a windowed region");
    crippled[windowed] = false;
    assert!(
        !table.covers_fire_shifted(fire.rows(), &crippled, &lane_shifting(&compiled)),
        "a windowed region that does NOT move its own base was admitted; the \
         launch plane would hand it pre-shifted pointers under a disarmed seat",
    );

    // **AND WHAT THAT REFUSAL COSTS IS ONE REGION AND NOT THE COMPOSITION**
    // (the tier-2 campaign). `covers_fire_shifted` is the collapsed reading of
    // `Windows::admits` — "every region is capturable" — and the collapse is
    // what a caller asks when it wants to know whether the whole fire fits in
    // one graph. The per-region table is what a body is CUT with: the crippled
    // region below is an `Admit::Island`, the stretches around it are captured,
    // and the island is re-issued eagerly between the execs. So exactly one
    // entry moves, and it is the one whose shift was taken away.
    let table_admits = table.admits(fire.rows(), &crippled, &lane_shifting(&compiled));
    assert_eq!(
        table_admits[windowed],
        engine_cuda::window::Admit::Island,
        "the region whose shift was taken away is the one that cannot be in a \
         graph, and the table has to say so at ITS index",
    );
    assert_eq!(
        table_admits
            .iter()
            .filter(|admit| **admit == engine_cuda::window::Admit::Island)
            .count(),
        1,
        "crippling one region made more than one island, so the table is not \
         answering per region: {table_admits:?}",
    );
    assert!(
        table
            .admits(fire.rows(), &shifted, &lane_shifting(&compiled))
            .iter()
            .all(|admit| *admit == engine_cuda::window::Admit::Captured),
        "the uncrippled slice left an island behind, so the collapsed reading \
         and the table disagree",
    );
}

/// **(c) THE STALENESS HAZARD, AS ARITHMETIC.** Two fires of ONE body key
/// whose totals agree and whose LAUNCHES do not.
///
/// This is the fire `Body::rows: u32` could not have seen. Same classes, same
/// bucket, same ceiling for each class, same eight rows — so the key matches
/// (which is exactly why the ceiling design's Option B does not close this
/// hazard, and the tier-1 collapse widens it: a rung is a ceiling, not a row,
/// so every split of one bucket is now one body) and the old scalar check
/// ("did the fire's total grow?") answers no — while the region guarded on
/// `Fact(0)` goes from five rows to three and its sibling from three to five.
/// A body captured on the first and replayed on the second would run a launch
/// recorded with a three-row grid over five rows of live data: the right key,
/// the right bucket, the right total, and two rows that never get computed.
#[test]
fn two_splits_of_one_key_move_a_launch_the_total_does_not() {
    let (trace, compiled) = baked();
    let first = split(&compiled, 5, 3);
    let second = split(&compiled, 3, 5);

    assert_eq!(first.rows(), second.rows(), "the two splits share a total");
    assert_eq!(
        first.bucket(),
        second.bucket(),
        "and therefore a lattice point",
    );
    assert_eq!(
        BodyKey::of_axes(first.classes(), first.bucket(), &no_decode_class(), LANES, None),
        BodyKey::of_axes(second.classes(), second.bucket(), &no_decode_class(), LANES, None),
        "the two fires must reach the SAME body, or there is no hazard here",
    );

    let a = launch_rows(&compiled, &windows(&trace, &compiled, &first));
    let b = launch_rows(&compiled, &windows(&trace, &compiled, &second));
    assert_eq!(
        a.len(),
        b.len(),
        "one key, one launch count — the comparison in `record` walks these \
         pairwise and a length that moved would be a key that lied",
    );
    assert!(
        a.iter().zip(&b).any(|(&was, &now)| now > was),
        "no launch of the second split asks for more rows than the first \
         recorded ({a:?} then {b:?}), so this pair is not the hazard the \
         per-launch check exists for",
    );
    assert!(
        a.iter().sum::<u32>() == b.iter().sum::<u32>(),
        "the launches' rows sum the same both ways ({a:?}, {b:?}) — which is \
         exactly why summing them, as `Body::rows: u32` effectively did, \
         cannot see the move",
    );
}
