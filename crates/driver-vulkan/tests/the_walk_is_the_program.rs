//! The walk IS the program: one step per step, in order, with the columns the
//! plan states.
//!
//! # Why this can run with no adapter
//!
//! Because `kernels_vulkan::plane::Ctx` is `dyn Encode` and nothing else. A
//! claim body states an entrypoint, an invocation count and an argument list;
//! the driver is what turns those into a dispatch. So the executor
//! (`driver_vulkan::baker`) can be handed an `Encode` that RECORDS instead of
//! planning, and every question this file asks — did the walk visit the steps
//! the program states, in the program's order; did each step reach the point
//! the plan names; did each operand arrive at the region the slot says — is
//! answered with no instance, no queue, no descriptor pool and no `slangc` in
//! the process.
//!
//! What it does NOT answer is whether the shader computes the right numbers, or
//! whether the entrypoint it named is one the SPIR-V tree stamps. Those need a
//! device and a build, and `tests/device.rs` is where the second is asked.
//!
//! # WHY THE PROGRAM IS STATED BY HAND, AND WHY THAT IS A NARROWER CLAIM HERE
//!
//! `driver-wgpu`'s twin states its `Program` by hand because on that plane
//! nothing can be bound at all: a statement can only be SIZED if the width rule
//! for its results does not read an operand's rectangle, exactly four points in
//! the whole floor qualify (`layout.embed` and the three `gemm.*`), and
//! `kernels-wgpu` claims none of them.
//!
//! **This plane claims `layout.embed`.** So the real
//! `model_compiler::program::bound` CAN seed a tower here, and
//! [`the_real_bound_seeds_a_tower_on_this_plane`] measures that it does — which
//! is the sharpest single difference between this driver and its nearest
//! sibling.
//!
//! What stops that being the fixture is the OTHER gap:
//! `kernels_vulkan::layout::embed` reads its table through
//! `kernels_vulkan::points::Staged::bank`, and that trait's blanket impl on
//! `dyn Encode` refuses every one of its five methods by name. So a seeded
//! tower binds and then refuses at its first statement, which
//! [`the_seeded_tower_refuses_at_the_fire_because_the_staged_door_is_shut`]
//! pins. The transcript below is therefore stated over four points that reach
//! no `Staged` method — `norm.rmsnorm_no_scale`, `norm.rmsnorm_per_head`,
//! `norm.residual_add` and `attention.logit_softcap` — and the day the door
//! opens, that second test fails and this file should bind its fixture instead.
//!
//! What the hand-stated half costs is precise and worth naming: the real width
//! walk and the real arena carve are not exercised by the transcript tests.
//! Everything else still is — the real `Slot` list is what the hand-built
//! `Program` states, and the real `BoundOp`, the real generated dispatch and
//! the real claim bodies are what the walk drives.
//!
//! # Mutation-checked
//!
//! A transcript comparison is only worth what it can tell apart, so three
//! deliberate corruptions are applied and each is asserted to change the
//! transcript: a DROPPED STEP, a WRONG POINT, and a SWAPPED OPERAND (two
//! operands of one statement exchanged). The third is the one a weaker test
//! would miss — every entrypoint, every invocation count and every argument
//! COUNT is unchanged by it, and only the regions move.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use driver_vulkan::baker::Bank;
use driver_vulkan::baker::marks::{Bindings, BufferId, Slice};
use driver_vulkan::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_vulkan::baker::walk::{Cursor, Extent, Fire};
use kernels::plane::Refusal;
use kernels_vulkan::Capability;
use kernels_vulkan::plane::{ArgValue, Encode, Fire as Launch};
use model_compiler::program::{Call, Dt, Program, Rows, Slot, Step};
use model_ir::plan::{Cond, Op, Param, Plan, Seam, Shard, ValueDef};

// ── the mock: an `Encode` that writes down what it was asked to fire ────

/// One launch, as the recorder saw it.
#[derive(Clone, Debug, PartialEq)]
struct Fired {
    /// The entrypoint the claim body named.
    entrypoint: &'static str,
    /// The ARTIFACT it named it in, which the body composed out of the
    /// entrypoint and the tier [`Recorder::best`] advertises.
    file: &'static str,
    /// TOTAL INVOCATIONS. There is no group beside it: a `kernels-vulkan` body
    /// states lanes only and the workgroup size is declared by `[numthreads]`
    /// in the Slang, which is the divergence `baker::dispatch` argues at
    /// length. A recorder that wrote down `fire.group` would be recording three
    /// zeros for every launch.
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
/// THE DIRECTION IS PART OF THE ANSWER on this plane, where wgpu's recorder has
/// no such field. `kernels_vulkan::plane::ArgValue::Buffer` carries `writes`,
/// because a Vulkan descriptor's direction is the BINDING's and the driver is
/// what places the barriers — see `baker::dispatch::Touches`. A recorder that
/// dropped it could not tell `residual_add`'s two bindings of one rectangle
/// apart, and those two are exactly the pair the hazard set is built from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arg {
    /// A bound region: which allocation, where in it, how long, and whether the
    /// body marked it writable.
    Buffer {
        buffer: u32,
        at: u64,
        bytes: u64,
        writes: bool,
    },
    /// A handle the fire never minted — a body reaching past its statement.
    Unbound(u32),
    I32(i32),
    U32(u32),
    F32(u32),
    Usize(u64),
    Raised,
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
    /// BASELINE, AND THAT IS WHAT MAKES THE ARTIFACT NAME PREDICTABLE.
    ///
    /// A body composes its module name with `module::path(entrypoint,
    /// self.best())`, which walks `Capability::PREFERENCE` down from the tier
    /// it is handed to the first one this build actually compiled. Handing it
    /// the bottom of that list means the walk has one element and the answer is
    /// always `"{entrypoint}.spv"`, whether or not `slangc` ran — which is what
    /// lets this file assert the file name rather than only its non-emptiness.
    fn best(&self) -> Capability {
        Capability::Baseline
    }

    fn fire(&self, fire: Launch, args: &[ArgValue]) -> Result<(), Refusal> {
        let bindings = self.bindings.borrow();
        let resolve = |handle: u32, writes: bool| {
            bindings
                .at(handle)
                .map_or(Arg::Unbound(handle), |b| Arg::Buffer {
                    buffer: b.slice.buffer.0,
                    at: b.slice.at,
                    bytes: b.slice.bytes,
                    writes,
                })
        };
        self.fired.borrow_mut().push(Fired {
            entrypoint: fire.entrypoint,
            file: fire.file,
            lanes: fire.lanes,
            args: args
                .iter()
                .map(|a| match *a {
                    ArgValue::Buffer { handle, writes, .. } => resolve(handle, writes),
                    ArgValue::I32(v) => Arg::I32(v),
                    ArgValue::U32(v) => Arg::U32(v),
                    ArgValue::F32(v) => Arg::F32(v.to_bits()),
                    ArgValue::Usize(v) => Arg::Usize(v),
                    ArgValue::Raised(_) => Arg::Raised,
                })
                .collect(),
        });
        Ok(())
    }
    /// The by-name door, refused here as it is refused by the real encoder:
    /// this walk hands a body every operand its point declares.
    fn staged(&self, name: &'static str) -> Result<u32, kernels::plane::Refusal> {
        let _ = name;
        Err(kernels::plane::Refusal::Unstated {
            what: "a runtime plane asked for BY NAME, in a recorder that \
                   stages none",
        })
    }

    /// A window, recorded as a fresh handle so a transcript shows the cut.
    fn windowed(&self, of: u32, at: u64) -> Result<u32, kernels::plane::Refusal> {
        let _ = (of, at);
        Err(kernels::plane::Refusal::Unstated {
            what: "a window, in a recorder with no bindings behind it",
        })
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
        // holding no slab must answer — which on this driver is every driver:
        // nothing here allocates one. A scan handed a null carry answers
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
            // Two KV heads of 32 elements: `seq_stride` is one token inside a
            // head and `head_stride` steps a head's pages, so neither of them
            // is this number and a driver that knows it states it.
            kv_heads: 2,
            head_dim: 32,
        }
    }

    fn table(&self, which: FireTable) -> Option<Slice> {
        self.tables.get(&which).copied()
    }
}

// ── the plan, and the program this fixture states for it ───────────────

const HIDDEN: u64 = 64;
const HEAD_DIM: u64 = 16;
const VOCAB: u64 = 128;
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

/// One statement, in the columns a `Plan` states them.
fn stmt(
    kernel: &str,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    weights: Vec<&str>,
    params: Vec<u64>,
) -> Op {
    Op {
        kernel: kernel.to_string(),
        inputs,
        outputs,
        weights: weights.into_iter().map(str::to_string).collect(),
        params,
        cache: None,
        layer: Some(0),
        cond: Cond::Always,
    }
}

/// A four-statement tower of points this plane claims AND can fire.
///
/// Value 0 is a rectangle the walk can address and the width walk cannot size —
/// on this plane that is every rectangle whose statement is not a
/// `layout.embed` — which is why the program below is stated rather than bound.
/// Two statements are `InOut` (`norm.residual_add`'s stream slot,
/// `attention.logit_softcap`'s x) so the copy path is exercised, and two state
/// scalars so the parameter run is too.
fn plan() -> Plan {
    Plan {
        name: "a-walk-fixture".into(),
        plane: model_ir::kernels::Backend::Vulkan,
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
        seams: vec![Seam {
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
/// walk can be checked against it.
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

/// This fixture's banks: one dense bf16 weight, at a recognisable region.
fn banks() -> BTreeMap<String, Bank> {
    [("norm.weight".to_string(), bank(0, vec![HIDDEN]))]
        .into_iter()
        .collect()
}

/// The fire this fixture runs, over whatever plan and program it is given.
fn over<'a>(
    plan: &'a Plan,
    program: &'a Program,
    banks: &'a BTreeMap<String, Bank>,
    pools: &Staging,
) -> Fire<'a> {
    Fire::over(
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
        banks,
        pools,
    )
}

/// Walk `program` over `plan` and record what was fired, with the copies the
/// `InOut` points scheduled.
fn walk(plan: &Plan, program: &Program) -> (Vec<Fired>, usize) {
    let banks = banks();
    let pools = Staging::new();
    let fire = over(plan, program, &banks, &pools);
    let recorder = Recorder::over(&fire.bindings);
    fire.walk(&recorder)
        .unwrap_or_else(|why| panic!("the fixture's walk refused: {why}"));
    let blits = fire.blits.borrow().len();
    (recorder.transcript(), blits)
}

// ── the premise, measured ──────────────────────────────────────────────

/// A tower seeded on `layout.embed`: the plan the real compiler CAN size.
///
/// `#[shape(y = [fire, table.axis(1)])]` is the embed's width rule, so the
/// result's width comes off the PARAMETER TABLE's second axis rather than off
/// an operand's rectangle — which is the whole of why it can start a tower.
fn seeded() -> Plan {
    let mut plan = plan();
    plan.params.push(Param {
        name: "embed.table".into(),
        shape: vec![VOCAB, HIDDEN],
        shard: Shard::Replicated,
        repr: "dense".into(),
    });
    plan.values = vec![
        ValueDef::Runtime("token_ids".into()),
        ValueDef::Stmt(0),
        ValueDef::Stmt(1),
        ValueDef::Stmt(2),
        ValueDef::Stmt(3),
        ValueDef::Stmt(4),
    ];
    plan.ops = vec![
        stmt(
            "layout.embed",
            vec![0],
            vec![1],
            vec!["embed.table"],
            vec![VOCAB],
        ),
        stmt(
            "norm.rmsnorm_no_scale",
            vec![1],
            vec![2],
            vec![],
            vec![HEAD_DIM, f32::to_bits(1e-6).into()],
        ),
        stmt(
            "norm.rmsnorm_per_head",
            vec![2],
            vec![3],
            vec!["norm.weight"],
            vec![HEAD_DIM, f32::to_bits(1e-6).into()],
        ),
        stmt("norm.residual_add", vec![3, 2], vec![4], vec![], vec![]),
        stmt(
            "attention.logit_softcap",
            vec![4],
            vec![5],
            vec![],
            vec![f32::to_bits(30.0).into()],
        ),
    ];
    plan.seams = vec![Seam {
        seam: model_ir::seam::OUT.name.to_string(),
        values: vec![5],
        layer: None,
    }];
    plan
}

/// THE REAL COMPILER SIZES A TOWER ON THIS PLANE, which is where this file
/// parts from `driver-wgpu`'s twin.
///
/// A result can only be SIZED at load if its width rule does not read an
/// operand's rectangle. Across the whole floor exactly four points qualify —
/// `layout.embed` (an embedding table's axis) and the three `gemm.*` (a
/// weight's) — and `kernels-wgpu` claims none of the four, so that driver's
/// version of this file has to state every `Slot` by hand and says so.
///
/// `kernels-vulkan` claims `layout.embed`, so `bound` runs the real width walk
/// and the real arena carve over the five statements of [`seeded`]. THE DAY
/// THIS FAILS, the claim went.
#[test]
fn the_real_bound_seeds_a_tower_on_this_plane() {
    let plan = seeded();
    let lanes = model_compiler::program::bound(&plan);
    assert_eq!(lanes.len(), 1, "nothing here is conditional, so one lane");
    let program = lanes[0].as_ref().unwrap_or_else(|why| {
        panic!("`layout.embed` should seed this tower, and the compiler refused: {why}")
    });
    assert_eq!(
        program.steps.len(),
        5,
        "one step per statement, in the plan's order",
    );
    assert_eq!(
        program.steps.iter().map(|s| s.op).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4],
    );
    // The embed's result is `[fire, table.axis(1)]` — a bf16 rectangle of
    // HIDDEN elements — read off the parameter table and not off value 0,
    // which is the runtime plane nothing can size.
    assert!(
        matches!(
            program.slots.get(1),
            Some(Slot::Arena {
                width, dtype: Dt::Bf16, ..
            }) if *width == HIDDEN
        ),
        "the seed's rectangle should be the embedding table's row: {:?}",
        program.slots.get(1),
    );
}

/// AND IT FIRES — which it did not, and the paragraph that stood here said what
/// to do on the day it did.
///
/// `kernels_vulkan::layout::embed` asked `Staged::bank(table)` for the code,
/// scale and bias planes of a QUANTISED embedding table, unconditionally, and
/// `Staged::bank` is an unconditional refusal. So this point passed the
/// load-time pass and refused mid-fire — exactly the shape
/// `crate::walk::resolve` exists to prevent and cannot, because the door was on
/// the floor.
///
/// The door did not open. The ARM beside it was missing: every row in the
/// catalog states a `bf16` embedding and this plane had no dense gather at all,
/// so `layout.embed` could fire for NOTHING. `layout/embed.slang` is that
/// gather, and this test is the one that has to change shape when a refusal
/// becomes an answer — which is what it was written to do.
///
/// It now asserts the walk SUCCEEDS and that the transcript holds the one
/// dispatch a gather is. A test that only asserted `is_ok` would pass for a
/// body that fired nothing.
#[test]
fn the_seeded_tower_fires_its_embed_now_that_the_dense_arm_exists() {
    let plan = seeded();
    let lanes = model_compiler::program::bound(&plan);
    let program = lanes[0].as_ref().expect("the seeded tower binds");

    // The load-time pass is happy: every point is claimed at the element its
    // witness rides.
    assert!(
        driver_vulkan::baker::resolve::check(&plan, program).is_empty(),
        "every point of the seeded tower is claimed here",
    );

    let banks: BTreeMap<String, Bank> = [
        ("norm.weight".to_string(), bank(0, vec![HIDDEN])),
        ("embed.table".to_string(), bank(1, vec![VOCAB, HIDDEN])),
    ]
    .into_iter()
    .collect();
    let pools = Staging::new();
    let fire = over(&plan, program, &banks, &pools);
    let recorder = Recorder::over(&fire.bindings);
    fire.walk(&recorder)
        .unwrap_or_else(|why| panic!("the seeded tower refused at {}: {why}", why.kernel));
    // ONE DISPATCH PER STATEMENT, and the count comes from the PLAN rather than
    // being written down: a body may state more than one launch, and if one of
    // these ever does, that is the day this number should be derived
    // differently rather than raised.
    assert_eq!(
        recorder.transcript().len(),
        plan.ops.len(),
        "the seeded tower's {} statements should each have fired once",
        plan.ops.len(),
    );
}

/// THE GAP `crate::walk::resolve` NAMES, CLOSED FOR THE POINTS THIS FILE FIRES.
///
/// That module's header states what its load-time pass cannot answer: *"whether
/// the ENTRY POINT a claim body will name is one the shader tree actually
/// stamps... the name is built INSIDE the body, out of the operands it was
/// handed."* It then names the seam — each plane enumerates every entry point
/// its tree can reach — and says the two lists can be joined when a generator
/// learns to emit a `warm(point, axes)` beside each arm.
///
/// **They can be joined for a transcript without waiting for that**, and this
/// is the join: the walk produced four artifact names by running the real claim
/// bodies, and `kernels_vulkan::module` is asked whether each one is a module
/// this build stamped. It is a narrower claim than a `warm` list — four points
/// rather than thirty-two — and it is a REAL one, because the names came out of
/// the bodies rather than out of this file.
///
/// SKIPPED WHEN THE BUILD HAS NO MODULES, which is the portable half: `slangc`
/// runs behind `kernels-vulkan/native` and `module::embedded()` is how that
/// crate says whether it did. A skip here is not a pass — it is this test
/// declining to assert about a tree that was not built.
#[test]
fn every_artifact_the_walk_named_is_one_the_spirv_tree_stamps() {
    if !kernels_vulkan::module::embedded() {
        // No `slangc` in this build: `MODULES` is empty and every lookup would
        // answer `None` for a reason that is about the build and not the walk.
        return;
    }
    let plan = plan();
    let (fired, _) = walk(&plan, &program());
    let mut missing = Vec::new();
    for f in &fired {
        if kernels_vulkan::module::at(f.file).is_none() {
            missing.push(f.file);
        }
    }
    assert!(
        missing.is_empty(),
        "the walk named {} artifact(s) this build does not stamp: {missing:?} — the \
         claim body composed a name the shader tree has no module for, which is \
         exactly the failure the resolve pass cannot see",
        missing.len(),
    );
    assert_eq!(fired.len(), 4, "four statements, four artifacts");
}

/// AND THE LANES THE BODIES ASKED FOR DIVIDE BY THE WORKGROUP THE SPIR-V
/// DECLARES.
///
/// The one number `baker::dispatch` says is not this half's — *"the divisor is
/// read off the reflected module at encode time"* — read off the reflected
/// module HERE, against the real bytes `slangc` produced, on a walk that ran
/// the real claim bodies. Nothing in the chain is stated by this file: the
/// artifact name is the body's, the lane count is the body's, the local size is
/// the shader's, and `crate::spirv` is the driver's own reader.
///
/// What it would catch is the defect `crate::spirv` opens by naming: a grid
/// planned on the wrong axis. *"This crate's first `Rule::Rms` put the row
/// count on y, while `norm/rms.slang` reads its row from `gl_WorkGroupID.x`.
/// Every row but the first was left holding the zeros its buffer was born with,
/// four dispatches returned success, and the lane-coverage sweeps all passed."*
/// So the axes the module says it is INDEXED BY are asked for a non-zero
/// workgroup count, which is the half a lane count alone cannot answer.
///
/// It also joins the descriptor run: a body's dense argument list must fit
/// inside the binding range the module decorates. `Declared::bindings` is one
/// PAST the highest number, not a count, because 79 modules in this tree
/// decorate a non-contiguous set — so this is `<=` and it is still a real
/// bound.
#[test]
fn the_lanes_a_body_asked_for_divide_by_the_workgroup_its_module_declares() {
    if !kernels_vulkan::module::embedded() {
        return;
    }
    let plan = plan();
    let (fired, _) = walk(&plan, &program());
    assert_eq!(fired.len(), 4, "four statements, four artifacts");
    for f in &fired {
        let code = kernels_vulkan::module::at(f.file)
            .unwrap_or_else(|| panic!("`{}` is a module this build stamped", f.file));
        let words = driver_vulkan::spirv::words(code)
            .unwrap_or_else(|why| panic!("`{}` is not readable SPIR-V: {why:?}", f.file));
        let declared = driver_vulkan::spirv::declared(&words)
            .unwrap_or_else(|why| panic!("`{}` declares nothing readable: {why:?}", f.file));

        assert!(
            !declared.local.contains(&0),
            "`{}` declares a zero workgroup extent {:?}, which is not a divisor",
            f.file,
            declared.local,
        );
        for axis in 0..3 {
            if !declared.grid_axes[axis] {
                continue;
            }
            let groups = f.lanes[axis].div_ceil(declared.local[axis]);
            assert!(
                groups > 0,
                "`{}` is indexed by axis {axis} and the body's {} lane(s) there \
                 divide by {} to nothing",
                f.file,
                f.lanes[axis],
                declared.local[axis],
            );
        }

        let buffers = f
            .args
            .iter()
            .filter(|a| matches!(a, Arg::Buffer { .. } | Arg::Unbound(_)))
            .count();
        assert!(
            buffers <= declared.bindings as usize,
            "`{}` decorates bindings up to {} and the body bound {buffers} buffer(s)",
            f.file,
            declared.bindings,
        );
    }
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
        // THE ARTIFACT IS THE BODY'S OWN ANSWER, composed from the entrypoint
        // and the tier the recorder advertises. At `Capability::Baseline` that
        // is the untiered module, whatever this build compiled.
        assert_eq!(
            f.file,
            format!("{}.spv", f.entrypoint),
            "the body should name the baseline artifact for its entrypoint",
        );
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
    let banks = banks();
    let pools = Staging::new();
    let fire = over(&plan, &program, &banks, &pools);
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

/// Every operand addresses the region its slot names, in the allocation it
/// names.
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
            Arg::Buffer {
                buffer, at, bytes, ..
            } if *buffer == ARENA_BUF.0 && *at == row_bytes => Some((*at, *bytes)),
            _ => None,
        })
        .expect("statement 0 writes value 1, which is one row-run into the arena");
    assert_eq!(first_result.1, row_bytes, "a value is `rows * width * dt`");

    let second_operand = fired[1]
        .args
        .iter()
        .find_map(|a| match a {
            Arg::Buffer {
                buffer, at, bytes, ..
            } if *buffer == ARENA_BUF.0 => Some((*at, *bytes)),
            _ => None,
        })
        .expect("statement 1 reads an arena rectangle");
    assert_eq!(
        second_operand, first_result,
        "statement 1's operand is statement 0's result",
    );

    // Every arena region stays inside the fire's arena, and the weight binds
    // out of the WEIGHTS allocation rather than the arena.
    for f in &fired {
        for a in &f.args {
            if let Arg::Buffer {
                buffer, at, bytes, ..
            } = a
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
    let fire = over(&plan, &program, &banks, &pools);
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
/// the scalar run differs: `plus_one`, the flag that decides whether the weight
/// is read as `w` or `1 + w`. A transcript that compared entrypoints and
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
        "but one word of the scalar run must differ",
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

// ── the encoder, which is what a device half actually gets ─────────────

/// A handle the fire never minted refuses, rather than binding nothing.
///
/// The real defect this catches is on record: `mxfp4_qmv_routed_bias` read an
/// additive bias off a null pointer for every expert logit, and nothing in the
/// path said a word.
#[test]
fn a_handle_the_fire_never_minted_refuses() {
    use driver_vulkan::baker::encode::Encoder;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor, Capability::Baseline);
    let why = encoder
        .fire(
            Launch::at("rms_single_row_bfloat16.spv", "rms_single_row_bfloat16")
                .apply([256u32, 1, 1]),
            &[ArgValue::Buffer {
                handle: 7,
                writes: false,
                rows: 1,
                width: 1,
            }],
        )
        .expect_err("handle 7 was never minted");
    assert!(
        matches!(why, Refusal::Absent { .. }),
        "an unminted handle is Absent, not an empty binding: {why:?}",
    );
}

/// A body that computed a zero invocation count refuses rather than dispatching
/// nothing and reporting success.
///
/// This backend has paid for it once: `vkCmdDispatch(0, 1, 1)` is legal Vulkan
/// that runs nothing over a buffer that kept its zeros, and a shared expert's
/// gate came back untouched with every routed token combined under
/// `sigmoid(0)`.
#[test]
fn a_zero_invocation_count_refuses() {
    use driver_vulkan::baker::encode::Encoder;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor, Capability::Baseline);
    let why = encoder
        .fire(
            Launch::at("rms_single_row_bfloat16.spv", "rms_single_row_bfloat16")
                .apply([0u32, 1, 1]),
            &[],
        )
        .expect_err("a zero grid runs nothing and would report success");
    assert!(matches!(why, Refusal::Grid { .. }), "{why:?}");
}

/// A ZERO WORKGROUP IS NOT REFUSED, because on this plane it is what every body
/// states.
///
/// The mirror of the test above, and it is here because porting
/// `driver-metal`'s second guard would have refused all thirty-two claims. A
/// `kernels-vulkan` body passes a bare `[u32; 3]` to `Fire::apply`, which sets
/// `lanes` and leaves `group` at zero; the workgroup size is declared by
/// `[numthreads]` in the Slang and recovered from `OpExecutionMode LocalSize`.
#[test]
fn a_zero_workgroup_is_not_refused_because_the_module_states_it() {
    use driver_vulkan::baker::encode::Encoder;

    let bindings = RefCell::new(Bindings::new());
    let cursor = Cell::new(Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor, Capability::Baseline);
    let fire =
        Launch::at("rms_single_row_bfloat16.spv", "rms_single_row_bfloat16").apply([256u32, 1, 1]);
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

/// THE HAZARD SET IS THE DRIVER'S HERE, which is where this plane parts from
/// wgpu.
///
/// `wgpu-core` emits a barrier before every dispatch and will not be told not
/// to, so `driver-wgpu`'s `Dispatch` carries no such column. `vkCmdDispatch`
/// runs concurrently with its neighbours until a `vkCmdPipelineBarrier` says
/// otherwise, so this driver decides — and what it decides from is the `writes`
/// bit `kernels_vulkan::plane::ArgValue::Buffer` carries.
///
/// Two dispatches that only READ one region need nothing between them; a write
/// into a region a later dispatch reads does.
#[test]
fn the_encoder_reads_a_hazard_set_off_the_bodys_own_direction() {
    use driver_vulkan::baker::dispatch::Touches;
    use driver_vulkan::baker::encode::Encoder;
    use driver_vulkan::baker::marks::{Rect, rin, rout};

    let bindings = RefCell::new(Bindings::new());
    let rect = |at: u64| Rect {
        slice: Slice {
            buffer: ARENA_BUF,
            at,
            bytes: 0x100,
        },
        rows: 1,
        width: 8,
        dt: Dt::Bf16,
    };
    let (read, written) = {
        let mut b = bindings.borrow_mut();
        (
            rin::<kernels_vulkan::points::bf16>(&mut b, rect(0)),
            rout::<kernels_vulkan::points::bf16>(&mut b, rect(0)),
        )
    };
    let cursor = Cell::new(Cursor::default());
    let encoder = Encoder::over(&bindings, &cursor, Capability::Baseline);

    let reader = ArgValue::Buffer {
        handle: read.ptr.handle,
        writes: false,
        rows: 1,
        width: 8,
    };
    let writer = ArgValue::Buffer {
        handle: written.ptr.handle,
        writes: true,
        rows: 1,
        width: 8,
    };
    let at = Launch::at("x.spv", "x").apply([1u32, 1, 1]);
    encoder.fire(at, &[reader]).expect("a read binds");
    encoder.fire(at, &[writer]).expect("a write binds");
    encoder.fire(at, &[reader]).expect("a read binds");
    let plan = encoder.finish();

    assert!(
        !plan[0].touches.hazards_after(&Touches::default()),
        "nothing precedes the first dispatch",
    );
    assert!(
        plan[1].touches.hazards_after(&plan[0].touches),
        "a write after a read of the same region needs a barrier",
    );
    assert!(
        plan[2].touches.hazards_after(&plan[1].touches),
        "a read after a write of the same region needs a barrier",
    );
    // Two readers of one region are the pair this exists to answer `false` for:
    // a fire binds one weight arena in every statement of a layer, and a rule
    // that ordered every pair sharing a byte would order the whole fire.
    assert!(
        !plan[2].touches.hazards_after(&plan[0].touches),
        "two reads of one region are concurrent",
    );
}

/// The resolve pass reports the whole backlog and dedupes it.
///
/// THE STAND-IN POINT USED TO BE `gemm.matmul`, and it had to move: this plane
/// claims all three `gemm.*` now, so the four statements would resolve and the
/// backlog would be empty. `dist.all_reduce` replaces it — `Dist` is one of
/// the six families `kernels-vulkan` writes no `#[claims]` block for at all,
/// which is the property this test needs and the only property it needs.
///
/// A point that is merely unclaimed TODAY is the wrong choice here. The five
/// `ssm.*` are a family in flight, and picking one would put this test back on
/// the same fuse the gemm spelling just burned through.
#[test]
fn the_resolve_pass_reports_the_backlog_and_dedupes_it() {
    let plan = plan();
    assert!(
        driver_vulkan::baker::resolve::check(&plan, &program()).is_empty(),
        "every point this fixture states is claimed here",
    );

    let mut unclaimed = plan.clone();
    let mut program = program();
    for (i, step) in program.steps.iter_mut().enumerate() {
        unclaimed.ops[i].kernel = "dist.all_reduce".to_string();
        step.call = Call::Point("dist.all_reduce".into());
    }
    let out = driver_vulkan::baker::resolve::check(&unclaimed, &program);
    assert_eq!(out.len(), 1, "four statements of one point are one row");
    assert_eq!(out[0].op, 0, "and it names the first that asked");
    assert!(
        out[0].why.contains("no point of that name"),
        "{}",
        out[0].why
    );
}
