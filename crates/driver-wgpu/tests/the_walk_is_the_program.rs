//! The walk IS the program: one step per step, in order, with the columns the
//! plan states.
//!
//! # Why this can run with no adapter
//!
//! Because `kernels_wgpu::routine::Ctx` is `dyn Encode` and nothing else. A
//! claim body states an entrypoint, an invocation count and an argument list;
//! the driver is what turns those into a dispatch. So the executor
//! (`driver_wgpu::baker`) can be handed an `Encode` that RECORDS instead of
//! planning, and every question this file asks — did the walk visit the steps
//! the program states, in the program's order; did each step reach the point
//! the plan names; did each operand arrive at the region the slot says — is
//! answered with no adapter, no driver and no shader compiler in the process.
//!
//! What it does NOT answer is whether the shader computes the right numbers, or
//! whether the entrypoint it named exists in the `.wgsl` tree. Those need a
//! device, and `tests/device_fire.rs` is where one of them is asked for real.
//!
//! # Why the PROGRAM is built by hand, and not only the plan
//!
//! This is the one place this file departs from `driver-metal`'s twin, and the
//! reason is a measurement rather than a convenience.
//!
//! Metal builds its plan by hand and then runs the REAL
//! `model_compiler::program::bound` over it, because metal claims
//! `layout.embed` through a `CANON` row and so a synthetic tower can be seeded.
//! **On this plane no tower can be seeded at all.** A statement can only be
//! SIZED if the width rule for its results does not have to read an operand's
//! rectangle, and exactly two points in the whole floor qualify —
//! `layout.embed`, whose width is an embedding table's axis, and the three
//! `gemm.*`, whose width is a weight's. `kernels-wgpu` claims none of them, and
//! it states no `CANON` table for one to be reached by symbol either.
//!
//! Every other point rides its input. So a plan that opens on a runtime plane
//! refuses with `Why::Unsized` at its first statement, whichever claimed point
//! that is — which [`the_points_that_can_seed_a_tower_are_named`] measures
//! across every point this plane claims, so that the day one can, this file's
//! premise changes visibly. IT DID: the claim table went 21 to 50 and three
//! of the new points size their own rectangle. See that test.
//!
//! What that costs is precise and worth naming: the real width walk and the
//! real arena carve are NOT exercised here, because nothing on this plane can
//! reach them. Everything else still is — the real `Slot` list is what the
//! hand-built `Program` states, and the real `BoundOp`, the real generated
//! dispatch and the real claim bodies are what the walk drives. The hand-built
//! part is exactly the part `bound` would have filled in, and
//! [`the_real_bound_refuses_this_plan_and_says_why`] pins the refusal that
//! makes it necessary.
//!
//! # Mutation-checked
//!
//! A transcript comparison is only worth what it can tell apart, so three
//! deliberate corruptions are applied and each is asserted to change the
//! transcript: a DROPPED STEP, a WRONG POINT, and a SWAPPED OPERAND (two
//! operands of one statement exchanged). The third is the one a weaker test
//! would miss — every entrypoint, every invocation count and every argument
//! COUNT is unchanged by it, and only the regions move.

use std::cell::RefCell;
use std::collections::BTreeMap;

use driver_wgpu::baker::Bank;
use driver_wgpu::baker::marks::{Bindings, BufferId, NOTHING, Slice};
use driver_wgpu::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_wgpu::baker::walk::{Extent, Fire};
use kernels::plane::Refusal;
use kernels_wgpu::plane::{ArgValue, Encode, Fire as Launch};
use model_compiler::program::{Call, Dt, Program, Rows, Slot, Step};
use model_ir::plan::{Cond, Op, Param, Plan, Shard, ValueDef};

// ── the mock: an `Encode` that writes down what it was asked to fire ────

/// One launch, as the recorder saw it.
#[derive(Clone, Debug, PartialEq)]
struct Fired {
    /// The entrypoint the claim body named.
    entrypoint: &'static str,
    /// The shader it named it in.
    file: &'static str,
    /// TOTAL INVOCATIONS. There is no group beside it: a `kernels-wgpu` body
    /// states lanes only and the workgroup size is the module's, which is the
    /// divergence `baker::dispatch` argues at length. A recorder that wrote
    /// down `fire.group` would be recording three zeros for every launch.
    lanes: [u32; 3],
    /// Every argument, resolved through the fire's binding list: a buffer
    /// becomes the region it addresses, a scalar its own bits.
    args: Vec<Arg>,
}

/// One argument, in terms that survive the crossing.
///
/// A HANDLE IS NOT WHAT IS COMPARED, and that distinction is the test's. A
/// handle is a number the executor minted in the order it happened to mint
/// them, so two runs that bind the same regions in the same order agree on it
/// trivially — and a run that binds a DIFFERENT region at the same slot agrees
/// on it too. What must hold is that slot `n` of dispatch `k` addresses the
/// bytes the plan's slot said, so the recorder resolves every handle through
/// the fire's own binding list before writing it down.
///
/// THE BUFFER IS PART OF THE ANSWER on this plane, where metal's recorder
/// writes an address and a length. Two wgpu allocations have no ordering
/// between them, so "which buffer" is a fact the region carries rather than one
/// an address implies — see `baker::marks`.
///
/// There is no `mutable` flag either, and its absence is the same measurement
/// `baker::encode` records: `kernels_wgpu::routine::ArgValue` has no
/// `BufferMut` variant, because on WebGPU the direction is the SHADER's
/// (`var<storage, read>` against `read_write`) and not the binding's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arg {
    /// A bound region: which allocation, where in it, and how long.
    Buffer {
        buffer: u32,
        at: u64,
        bytes: u64,
    },
    /// A handle the fire never minted — a body reaching past its statement.
    Unbound(u32),
    I32(i32),
    U32(u32),
    F32(u32),
    Usize(u64),
}

struct Recorder<'b> {
    /// The fire's binding list, so a handle can be resolved to its region.
    bindings: &'b RefCell<Bindings>,
    fired: RefCell<Vec<Fired>>,
}

impl<'b> Recorder<'b> {
    fn over(bindings: &'b RefCell<Bindings>) -> Self {
        Self {
            bindings,
            fired: RefCell::new(Vec::new()),
        }
    }

    fn transcript(&self) -> Vec<Fired> {
        self.fired.borrow().clone()
    }
}

impl Encode for Recorder<'_> {
    /// The one thing a wgpu claim body asks the fire for: a binding the point
    /// does not carry, which the entrypoint still declares.
    fn absent(&self) -> Result<ArgValue, Refusal> {
        Ok(ArgValue::Buffer(self.bindings.borrow_mut().take(NOTHING)))
    }

    fn fire(&self, fire: Launch, args: &[ArgValue]) -> Result<(), Refusal> {
        let bindings = self.bindings.borrow();
        let resolve = |handle: u32| {
            bindings
                .at(handle)
                .map_or(Arg::Unbound(handle), |b| Arg::Buffer {
                    buffer: b.slice.buffer.0,
                    at: b.slice.at,
                    bytes: b.slice.bytes,
                })
        };
        self.fired.borrow_mut().push(Fired {
            entrypoint: fire.entrypoint,
            file: fire.file,
            lanes: fire.lanes,
            args: args
                .iter()
                .map(|a| match *a {
                    ArgValue::Buffer(h) => resolve(h),
                    ArgValue::I32(v) => Arg::I32(v),
                    ArgValue::U32(v) => Arg::U32(v),
                    ArgValue::F32(v) => Arg::F32(v.to_bits()),
                    ArgValue::Usize(v) => Arg::Usize(v),
                })
                .collect(),
        });
        Ok(())
    }
}

// ── the staging: pools and planes at recognisable regions ──────────────

/// Buffer ids this fixture hands out. Distinct numbers so a transcript says
/// which allocation a region came out of.
const ARENA_BUF: BufferId = BufferId(0);
const WEIGHTS_BUF: BufferId = BufferId(1);
const TABLES_BUF: BufferId = BufferId(2);
const POOLS_BUF: BufferId = BufferId(3);

/// The fire's staged planes and pools, at addresses a reader can recognise.
struct Staging {
    tables: BTreeMap<FireTable, Slice>,
}

impl Staging {
    fn new() -> Self {
        let every = [
            FireTable::TokenIds,
            FireTable::Positions,
            FireTable::RequestOfToken,
            FireTable::QoIndptr,
            FireTable::RowValid,
            FireTable::SamplingIndices,
            FireTable::KvPageIndices,
            FireTable::KvPageIndptr,
            FireTable::KvWritePage,
            FireTable::KvWriteOffset,
            FireTable::RecurrentSlots,
            FireTable::AttentionMask,
            FireTable::AttentionMaskEnabled,
            FireTable::AttnPartials,
        ];
        Self {
            tables: every
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    (
                        t,
                        Slice {
                            buffer: TABLES_BUF,
                            at: (i as u64) * 0x1000,
                            bytes: 0x1000,
                        },
                    )
                })
                .collect(),
        }
    }
}

impl Pools for Staging {
    fn kv(&self, layer: u32, values: bool) -> Option<Slice> {
        Some(Slice {
            buffer: POOLS_BUF,
            at: u64::from(layer) * 0x2000 + u64::from(values) * 0x1000,
            bytes: 0x1000,
        })
    }

    fn slab(&self, _layer: u32, _which: Slab) -> Option<Slice> {
        // This fixture states no recurrent tower, and `None` is what a driver
        // holding no slab must answer: a scan handed a null carry answers
        // fluently and wrongly.
        None
    }

    // ONE SHAPE FOR EVERY LAYER. This fixture stages a single pool, so the
    // layer changes nothing; the argument exists for a tower that attends at
    // two widths, which `driver-wgpu/tests/banked_argmax.rs` names.
    fn kv_geometry(&self, _layer: u32) -> KvGeometry {
        KvGeometry {
            page_size: 16,
            seq_stride: 64,
            head_stride: 1024,
            kv_heads: 2,
        }
    }

    fn table(&self, which: FireTable) -> Option<Slice> {
        self.tables.get(&which).copied()
    }
}

// ── the plan, and the program this plane cannot bind for it ────────────

const HIDDEN: u64 = 64;
const HEAD_DIM: u64 = 16;
const ARENA: u64 = 0x100_0000;
const ROWS: i32 = 2;

/// A weight this plane can bind: bf16, dense, in the weight arena.
fn bank(at: u64, shape: Vec<u64>) -> Bank {
    Bank {
        slice: Slice {
            buffer: WEIGHTS_BUF,
            at: at * 0x1000,
            bytes: 0x1000,
        },
        shape,
        dtype: model::produce::Dtype::Bf16,
        repr: "dense".to_string(),
    }
}

/// A four-statement tower of points this plane DOES claim.
///
/// Value 0 is a rectangle the walk can address and the width walk cannot size —
/// which on this plane is every rectangle, and is why the program below is
/// stated rather than bound. Two statements are `InOut`
/// (`norm.residual_add`'s hidden slot, `attention.logit_softcap`'s x) so the
/// copy path is exercised, and one (`norm.rmsnorm_per_head`) states a scalar so
/// the uniform block is too.
fn plan() -> Plan {
    let stmt = |kernel: &str,
                inputs: Vec<u32>,
                outputs: Vec<u32>,
                weights: Vec<&str>,
                params: Vec<u64>| Op {
        kernel: kernel.to_string(),
        inputs,
        outputs,
        weights: weights.into_iter().map(str::to_string).collect(),
        params,
        cache: None,
        layer: Some(0),
        cond: Cond::Always,
    };
    Plan {
        name: "a-walk-fixture".into(),
        plane: model_ir::kernels::Backend::Wgpu,
        facts: vec!["qo_one".into()],
        params: vec![Param {
            name: "norm.weight".into(),
            shape: vec![HIDDEN],
            shard: Shard::Replicated,
            repr: "dense".into(),
        }],
        caches: Vec::new(),
        // 0: the tower's input. Declared `Runtime` so the plan is well formed;
        // what the walk actually reads for it is the program's slot, which the
        // fixture states as an arena rectangle.
        values: vec![
            ValueDef::Runtime("token_ids".into()),
            ValueDef::Stmt(0),
            ValueDef::Stmt(1),
            ValueDef::Stmt(2),
            ValueDef::Stmt(3),
        ],
        ops: vec![
            stmt(
                "norm.rmsnorm_no_scale",
                vec![0],
                vec![1],
                vec![],
                vec![HEAD_DIM, f32::to_bits(1e-6).into()],
            ),
            stmt(
                "norm.rmsnorm_per_head",
                vec![1],
                vec![2],
                vec!["norm.weight"],
                vec![HEAD_DIM, f32::to_bits(1e-6).into()],
            ),
            stmt("norm.residual_add", vec![2, 1], vec![3], vec![], vec![]),
            stmt(
                "attention.logit_softcap",
                vec![3],
                vec![4],
                vec![],
                vec![f32::to_bits(30.0).into()],
            ),
        ],
        seams: vec![model_ir::plan::Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: vec![4],
            layer: None,
        }],
    }
}

/// The `Program` this plane's `bound` would have produced, stated by hand.
///
/// Five bf16 rectangles of `HIDDEN` elements, laid out value-major at one row
/// pitch — which is the layout `carve` produces when nothing's life overlaps
/// enough to share. Every offset is a multiple of the row's bytes, so the
/// arithmetic `Fire::rect` does (`offset * fire_rows`) lands where a reader
/// expects.
///
/// It is NOT a claim about what `carve` would choose. `carve` reuses offsets by
/// liveness and would very likely pick a tighter set; what this fixture needs
/// is a `Slot` list that is internally consistent and distinguishable, so the
/// walk can be checked against it. What `carve` decides is
/// `model-compiler`'s own tests' business.
fn program() -> Program {
    let row = HIDDEN * 2;
    let arena = |i: u64| Slot::Arena {
        offset: i * row,
        rows: Rows::Fire,
        width: HIDDEN,
        dtype: Dt::Bf16,
    };
    Program {
        words: vec![0, 1],
        steps: (0..4)
            .map(|op| Step {
                op,
                call: Call::Point(plan().ops[op as usize].kernel.clone()),
            })
            .collect(),
        slots: (0..5).map(arena).collect(),
        row_pitch: row * 5,
    }
}

/// Walk `program` over `plan` and record what was fired, with the copies the
/// `InOut` points scheduled.
fn walk(plan: &Plan, program: &Program) -> (Vec<Fired>, usize) {
    let banks: BTreeMap<String, Bank> = [("norm.weight".to_string(), bank(0, vec![HIDDEN]))]
        .into_iter()
        .collect();
    let pools = Staging::new();
    let fire = Fire::over(
        plan,
        program,
        Extent {
            arena: Slice {
                buffer: ARENA_BUF,
                at: 0,
                bytes: ARENA,
            },
            rows: ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let recorder = Recorder::over(&fire.bindings);
    fire.walk(&recorder)
        .unwrap_or_else(|why| panic!("the fixture's walk refused: {why}"));
    let blits = fire.blits.borrow().len();
    (recorder.transcript(), blits)
}

// ── the premise, measured ──────────────────────────────────────────────

/// NO POINT THIS PLANE CLAIMS CAN START A TOWER, and this is the measurement
/// that makes the hand-built `Program` above necessary.
///
/// A result can only be SIZED if its width rule does not read an operand's
/// rectangle. Across the whole floor exactly four points qualify —
/// `layout.embed` (an embedding table's axis) and the three `gemm.*` (a
/// weight's) — and `kernels-wgpu` claims none of them. So the first statement
/// of any plan that opens on a runtime plane refuses, and this walks all
/// twenty-one claims to say so rather than asserting it of one.
///
/// The twenty-first is skipped and the skip is the interesting row:
/// `attention.kv_append` states no result at all, so `bound` answers `Ok` for
/// it. That is not a seed — it writes into the pool and leaves no rectangle
/// behind — and the loop says so where it skips.
///
/// THE DAY THIS FAILS IS THE DAY THIS FILE GETS SIMPLER: a claimed point that
/// The claimed points that can START a tower on this plane, as measured
/// against the synthetic plan below.
///
/// A point can seed only if its shape sizes a rectangle WITHOUT reading an
/// operand: every other rule rides its input, and a runtime plane has no
/// rectangle for one to ride.
///
/// THE THREE ROUTERS, AND NOT `layout.embed`, WHICH IS THE SURPRISE. The
/// obvious candidate is the embed — `[fire, table.axis(1)]` is why every
/// shipping text opens with one — and it is claimed here now. It does not
/// appear because THIS fixture hands one weight, `norm.weight`, and the
/// embed's second axis is the TABLE's; sized against the wrong bank it is not
/// sized at all. What does appear is `moe.topk_*`, whose output is
/// `[fire, top_k]` and whose `top_k` is a STATED scalar, so the rectangle
/// comes off the params run and no operand is read.
///
/// So this list measures what the fixture can prove rather than what the
/// plane can do, and the assertion below says the same thing in its own
/// words: `bound` can build a real Program now, so the fixture should stop
/// stating one by hand. `driver-vulkan`'s twin already does.
const SEEDS: &[&str] = &[
    "moe.topk_sigmoid",
    "moe.topk_softmax",
    "moe.topk_sqrt_softplus",
];

/// can seed means `bound` can build a real `Program` for a synthetic tower, and
/// [`program`] should be deleted in favour of calling it.
#[test]
fn the_points_that_can_seed_a_tower_are_named() {
    let mut seeds = Vec::new();
    for (point, _, _) in kernels_wgpu::points_dispatch::CLAIMED {
        // THE RESULT COUNT IS THE DECLARATION'S, read off the floor rather
        // than assumed to be one. `attention.kv_append` declares NO result --
        // it writes into the pool -- and `program::bind` ASSERTS that a
        // statement's output count matches the width rule's rather than
        // refusing, so a probe that handed it a spare output would panic
        // inside the compiler instead of measuring anything here.
        let results = kernels::points::point_of(point).map_or(1, |p| p.outs.len());
        // A POINT THAT STATES NO RESULT CANNOT SEED A TOWER, and it is worth
        // saying why rather than just skipping it. `attention.kv_append` binds
        // here — `bound` returns `Ok` — but it binds VACUOUSLY: it declares no
        // result, so there is no width rule for the walk to fail at and no
        // rectangle for a next statement to ride. Seeding means producing a
        // sized rectangle out of a fire that has none, which is exactly what a
        // point with nothing in its `outs` column cannot do.
        if results == 0 {
            continue;
        }
        let outputs: Vec<u32> = (0..results).map(|i| 1 + i as u32).collect();
        let mut p = plan();
        p.values = core::iter::once(ValueDef::Runtime("token_ids".into()))
            .chain((0..results).map(|_| ValueDef::Stmt(0)))
            .collect();
        p.ops = vec![Op {
            kernel: (*point).to_string(),
            inputs: vec![0],
            outputs: outputs.clone(),
            weights: vec!["norm.weight".to_string()],
            params: vec![HIDDEN, HEAD_DIM, f32::to_bits(1e-6).into()],
            cache: None,
            layer: Some(0),
            cond: Cond::Always,
        }];
        p.seams = vec![model_ir::plan::Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: outputs.first().copied().into_iter().collect(),
            layer: None,
        }];
        if model_compiler::program::bound(&p)
            .into_iter()
            .any(|l| l.is_ok())
        {
            seeds.push(*point);
        }
    }
    seeds.sort_unstable();
    assert_eq!(
        seeds, SEEDS,
        "the claimed points that can seed a tower have moved. A point GAINED \
         belongs in `SEEDS`; a point LOST means this plane stopped being able \
         to start a tower with it, which is a regression.\n\nAnd the note this \
         assertion used to carry still stands: `bound` can build a real \
         Program for a synthetic plan now, so this file SHOULD stop stating \
         one by hand. `driver-vulkan`'s twin already does, because \
         `layout.embed` was claimed there first.",
    );
}

/// The real `bound` refuses this fixture's plan, and the refusal names the
/// first statement rather than something further down.
#[test]
fn the_real_bound_refuses_this_plan_and_says_why() {
    let plan = plan();
    let lanes = model_compiler::program::bound(&plan);
    assert_eq!(lanes.len(), 1, "nothing here is conditional, so one lane");
    let refusal = lanes[0]
        .as_ref()
        .expect_err("no lane binds on this plane; see this file's header");
    let first = refusal.gaps.first().expect("a refusal states its gaps");
    assert_eq!(
        first.point, "norm.rmsnorm_no_scale",
        "the refusal should name the seed statement, not a consumer of it",
    );
    assert!(
        matches!(first.why, model_compiler::program::Why::Unsized),
        "the seed is UNSIZED (it rides a runtime plane), not unclaimed: {:?}",
        first.why,
    );
}

// ── the walk ───────────────────────────────────────────────────────────

/// One step per step, in the program's order, and nothing else.
#[test]
fn the_walk_visits_every_step_once_in_the_programs_order() {
    let plan = plan();
    let (fired, _) = walk(&plan, &program());
    assert_eq!(
        fired.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        vec![
            "vnorm_single_row_bfloat16",
            "rms_single_row_bfloat16",
            "residual_add_bfloat16",
            "logit_softcap_bfloat16",
        ],
        "the transcript is the program's steps, in the program's order",
    );
    for f in &fired {
        assert!(
            f.lanes.iter().all(|&n| n > 0),
            "`{}` asked for a zero invocation count on some axis: {:?}",
            f.entrypoint,
            f.lanes,
        );
        assert!(!f.file.is_empty(), "every fire names the shader it is in");
    }
}

/// A point this plane does not claim refuses with the statement named, before
/// anything fires.
#[test]
fn a_point_this_plane_does_not_claim_refuses_with_the_statement_named() {
    let mut plan = plan();
    plan.ops[2].kernel = "gemm.matmul".to_string();
    let program = Program {
        steps: vec![Step {
            op: 2,
            call: Call::Point("gemm.matmul".into()),
        }],
        ..program()
    };
    let banks: BTreeMap<String, Bank> = [("norm.weight".to_string(), bank(0, vec![HIDDEN]))]
        .into_iter()
        .collect();
    let pools = Staging::new();
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice {
                buffer: ARENA_BUF,
                at: 0,
                bytes: ARENA,
            },
            rows: ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let recorder = Recorder::over(&fire.bindings);
    let why = fire
        .walk(&recorder)
        .expect_err("gemm.matmul is unclaimed here");
    assert_eq!(why.op, 2);
    assert_eq!(why.kernel, "gemm.matmul");
    assert!(
        recorder.transcript().is_empty(),
        "nothing should fire before the refusal",
    );
}

/// Every operand addresses the region its slot names, in the buffer it names.
#[test]
fn every_operand_addresses_the_region_its_slot_names() {
    let plan = plan();
    let program = program();
    let (fired, _) = walk(&plan, &program);
    let row_bytes = u64::from(ROWS.unsigned_abs()) * HIDDEN * 2;

    // The first statement reads value 0 and writes value 1; the second reads
    // value 1. So the second's first operand is the first's result — which is
    // the join a slot list exists to make.
    let first_result = fired[0]
        .args
        .iter()
        .find_map(|a| match a {
            Arg::Buffer { buffer, at, bytes } if *buffer == ARENA_BUF.0 && *at == row_bytes => {
                Some((*at, *bytes))
            }
            _ => None,
        })
        .expect("statement 0 writes value 1, which is one row-run into the arena");
    assert_eq!(first_result.1, row_bytes, "a value is `rows * width * dt`");

    let second_operand = fired[1]
        .args
        .iter()
        .find_map(|a| match a {
            Arg::Buffer { buffer, at, bytes } if *buffer == ARENA_BUF.0 => Some((*at, *bytes)),
            _ => None,
        })
        .expect("statement 1 reads an arena rectangle");
    assert_eq!(
        second_operand, first_result,
        "statement 1's operand is statement 0's result",
    );

    // Every arena region stays inside the fire's arena, and the weight binds
    // out of the WEIGHTS buffer rather than the arena.
    for f in &fired {
        for a in &f.args {
            if let Arg::Buffer { buffer, at, bytes } = a
                && *buffer == ARENA_BUF.0
            {
                assert!(
                    at + bytes <= ARENA,
                    "`{}` bound {at}+{bytes}, past a {ARENA}-byte arena",
                    f.entrypoint,
                );
            }
        }
    }
    assert!(
        fired[1]
            .args
            .iter()
            .any(|a| matches!(a, Arg::Buffer { buffer, .. } if *buffer == WEIGHTS_BUF.0)),
        "the `Const` slot binds out of the weight arena, not the activation one",
    );
}

/// A stated scalar reaches the claim body, and arrives as itself.
#[test]
fn a_stated_scalar_reaches_the_claim_body() {
    let plan = plan();
    let (fired, _) = walk(&plan, &program());
    assert!(
        fired[1].args.contains(&Arg::F32(1e-6f32.to_bits())),
        "statement 1 states an epsilon: {:?}",
        fired[1].args,
    );
    assert!(
        fired[1].args.contains(&Arg::I32(HEAD_DIM as i32)),
        "statement 1 states a head width: {:?}",
        fired[1].args,
    );
    assert!(
        fired[3].args.contains(&Arg::F32(30.0f32.to_bits())),
        "statement 3 states a softcap: {:?}",
        fired[3].args,
    );
}

/// An in-place point schedules a copy of its operand into its result, and the
/// two are disjoint.
#[test]
fn an_in_place_point_schedules_a_disjoint_copy() {
    let plan = plan();
    let (_, blits) = walk(&plan, &program());
    assert_eq!(
        blits, 2,
        "`norm.residual_add` and `attention.logit_softcap` each state an InOut",
    );
}

/// A canon symbol refuses by name, and on this plane the refusal says there is
/// no canon table at all.
#[test]
fn the_walk_refuses_a_canon_symbol_by_name() {
    let plan = plan();
    let program = Program {
        steps: vec![Step {
            op: 0,
            call: Call::Symbol("norm::some_symbol"),
        }],
        ..program()
    };
    let banks: BTreeMap<String, Bank> = BTreeMap::new();
    let pools = Staging::new();
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice {
                buffer: ARENA_BUF,
                at: 0,
                bytes: ARENA,
            },
            rows: ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let recorder = Recorder::over(&fire.bindings);
    let why = fire.walk(&recorder).expect_err("a symbol has no shim here");
    assert_eq!(why.op, 0);
    assert!(
        format!("{why}").contains("canon"),
        "the refusal should name what it could not answer for: {why}",
    );
    assert!(
        recorder.transcript().is_empty(),
        "nothing should fire before the refusal",
    );
}

// ── mutation checks ────────────────────────────────────────────────────

/// MUTATION 1: dropping a step changes what is fired.
#[test]
fn dropping_a_step_changes_what_is_fired() {
    let plan = plan();
    let (before, _) = walk(&plan, &program());
    let mut shorter = program();
    shorter.steps.pop();
    let (after, _) = walk(&plan, &shorter);
    assert_eq!(after.len(), before.len() - 1);
    assert_eq!(
        after[..],
        before[..before.len() - 1],
        "dropping the last step should leave the others untouched",
    );
    assert_ne!(before, after);
}

/// MUTATION 2: firing the wrong point changes what is fired — and this pair is
/// deliberately the hardest of its kind.
///
/// `norm.rmsnorm_per_head` and `norm.rmsnorm_per_head_plus_one` fire the SAME
/// entrypoint with the SAME slot list and the SAME argument count. One word of
/// the uniform block differs: `plus_one`, the flag that decides whether the
/// weight is read as `w` or `1 + w`. A transcript that compared entrypoints and
/// arities would call the two equal.
///
/// It is not hypothetical. W3 found an import whose `plus_one` fold made a
/// model agree with itself and disagree with its checkpoint.
#[test]
fn firing_the_wrong_point_changes_what_is_fired() {
    let plan = plan();
    let (before, _) = walk(&plan, &program());

    let mut other = plan.clone();
    other.ops[1].kernel = "norm.rmsnorm_per_head_plus_one".to_string();
    let mut program = program();
    program.steps[1].call = Call::Point("norm.rmsnorm_per_head_plus_one".into());
    let (after, _) = walk(&other, &program);

    assert_eq!(
        before[1].entrypoint, after[1].entrypoint,
        "the two points share an entrypoint, which is what makes this hard",
    );
    assert_eq!(
        before[1].args.len(),
        after[1].args.len(),
        "and they share an argument count",
    );
    assert_ne!(
        before[1].args, after[1].args,
        "but one word of the uniform block must differ",
    );
}

/// MUTATION 3: swapping two operands changes what is BOUND, and nothing else.
///
/// This is the one a handle-number transcript is blind to. `norm.residual_add`
/// takes two rectangles of one shape; exchanging them leaves every entrypoint,
/// every invocation count and every argument count identical, and moves only
/// the regions.
#[test]
fn swapping_two_operands_changes_what_is_bound() {
    let plan = plan();
    let (before, _) = walk(&plan, &program());

    let mut swapped = plan.clone();
    swapped.ops[2].inputs.swap(0, 1);
    let (after, _) = walk(&swapped, &program());

    assert_eq!(
        before.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        after.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        "swapping operands must not change which entrypoints run",
    );
    assert_eq!(before[2].lanes, after[2].lanes, "nor the grid");
    assert_eq!(
        before[2].args.len(),
        after[2].args.len(),
        "nor the argument count",
    );
    assert_ne!(
        before[2].args, after[2].args,
        "only the regions move — and a transcript of handle NUMBERS would miss it",
    );
}

/// A handle the fire never minted refuses, rather than binding nothing.
///
/// The real defect this catches is on record: `mxfp4_qmv_routed_bias` read an
/// additive bias off a null pointer for every expert logit, and nothing in the
/// path said a word.
#[test]
fn a_handle_the_fire_never_minted_refuses() {
    use driver_wgpu::baker::encode::Encoder;
    use std::cell::Cell;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(driver_wgpu::baker::walk::Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor);
    let why = encoder
        .fire(
            Launch::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply([256u32, 1, 1]),
            &[ArgValue::Buffer(7)],
        )
        .expect_err("handle 7 was never minted");
    assert!(
        matches!(why, Refusal::Absent { .. }),
        "an unminted handle is Absent, not an empty binding: {why:?}",
    );
}

/// A body that computed a zero invocation count refuses rather than dispatching
/// nothing and reporting success.
#[test]
fn a_zero_invocation_count_refuses() {
    use driver_wgpu::baker::encode::Encoder;
    use std::cell::Cell;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(driver_wgpu::baker::walk::Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor);
    let why = encoder
        .fire(
            Launch::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply([0u32, 1, 1]),
            &[],
        )
        .expect_err("a zero grid runs nothing and would report success");
    assert!(matches!(why, Refusal::Grid { .. }), "{why:?}");
}

/// A ZERO WORKGROUP IS NOT REFUSED, because on this plane it is what every body
/// states.
///
/// The mirror of the test above, and it is here because porting
/// `driver-metal`'s second guard would have refused all twenty-one claims. A
/// `kernels-wgpu` body passes a bare `[u32; 3]` to `Fire::apply`, which sets
/// `lanes` and leaves `group` at zero; the workgroup size is the MODULE's and
/// `src/encode.rs` reads it off the reflected pipeline.
#[test]
fn a_zero_workgroup_is_not_refused_because_the_module_states_it() {
    use driver_wgpu::baker::encode::Encoder;
    use std::cell::Cell;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(driver_wgpu::baker::walk::Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor);
    let fire = Launch::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply([256u32, 1, 1]);
    assert_eq!(
        fire.group,
        [0, 0, 0],
        "a bare `[u32; 3]` geometry states lanes and leaves the group alone",
    );
    assert!(
        encoder.fire(fire, &[]).is_ok(),
        "a zero group must not refuse: it is what every claim body on this plane states",
    );
}

/// The resolve pass reports the whole backlog and dedupes it.
#[test]
fn the_resolve_pass_reports_the_backlog_and_dedupes_it() {
    let plan = plan();
    assert!(
        driver_wgpu::baker::resolve::check(&plan, &program()).is_empty(),
        "every point this fixture states is claimed here",
    );

    let mut unclaimed = plan.clone();
    let mut program = program();
    for (i, step) in program.steps.iter_mut().enumerate() {
        // `mla.absorb_q`, and it USED TO BE `mlp.swiglu` — which this plane
        // claims now. A test whose subject is "a point this plane does not
        // claim" has to be re-pointed the day the plane claims it, or it goes
        // on passing for a reason that is no longer the one written here. The
        // whole `mla` family is still unclaimed; so are `index`, `pool`, `hc`
        // and the three lse points.
        unclaimed.ops[i].kernel = "mla.absorb_q".to_string();
        step.call = Call::Point("mla.absorb_q".into());
    }
    let out = driver_wgpu::baker::resolve::check(&unclaimed, &program);
    assert_eq!(out.len(), 1, "four statements of one point are one row");
    assert_eq!(out[0].op, 0, "and it names the first that asked");
    assert!(
        out[0].why.contains("no point of that name"),
        "{}",
        out[0].why
    );
}
