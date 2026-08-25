//! The walk IS the program: one step per step, in order, with the columns
//! the plan states.
//!
//! # Why this can run on a Linux box
//!
//! Because `kernels_metal::routine::Ctx` is `dyn Encode` and nothing else. A
//! claim body states an entrypoint, a grid and an argument list; the driver
//! is what turns those into a dispatch. So the executor
//! (`driver_metal::baker`) can be handed an `Encode` that RECORDS instead of
//! encoding, and every question this file asks — did the walk visit the steps
//! the program states, in the program's order; did each step reach the point
//! the plan names; did each operand arrive at the region the slot says — is
//! answered without a Metal device, a macOS toolchain, or a shader compiler
//! anywhere in the process.
//!
//! What it does NOT answer is whether the shader computes the right numbers,
//! or whether the entrypoint it named exists in the `.metal` tree. Those need
//! a device and are stated as unproven rather than approximated here.
//!
//! # Why the plan is built by hand
//!
//! `kernels-metal` claims 22 points and every catalog row states more than
//! those, so **no shipping SKU has a metal lane that BINDS today** —
//! `model_compiler::program::bound` answers `Err(Refusal)` for all 35 of
//! them, with `gemm.matmul` and the quantised banks at the head of every gap
//! list. A walk test written against a catalog row would therefore have
//! nothing to walk, and would go on having nothing to walk for as long as it
//! took the gemm family to land — which is exactly when a walk test stops
//! being written.
//!
//! So the plan below is a four-statement tower of points this plane DOES
//! claim, run through the real `model_compiler::program::bound`. Everything
//! downstream of that call is production code: the real width walk, the real
//! arena carve, the real `Slot` list, the real `BoundOp`, the real generated
//! dispatch, the real claim bodies. The only thing this file supplies that a
//! catalog row would have supplied is the statements.
//!
//! [`every_catalog_row_traces_for_this_plane`] is the other half: it asks the
//! real sixteen rows for their metal lanes and pins what they answer, so the
//! day a lane binds this file's premise changes visibly.
//!
//! # Mutation-checked
//!
//! A transcript comparison is only worth what it can tell apart, so three
//! deliberate corruptions are applied to the program and each is asserted to
//! change the transcript: a DROPPED STEP, a WRONG POINT, and a SWAPPED
//! HANDLE (two operands of one statement exchanged). The third is the one a
//! weaker test would miss — every entrypoint, every grid and every argument
//! COUNT is unchanged by it, and only the regions move.

use std::cell::RefCell;
use std::collections::BTreeMap;

use driver_metal::baker::marks::{Bindings, Slice};
use driver_metal::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_metal::baker::walk::{Extent, Fire};
use driver_metal::baker::{Baked, Bank, Metal};
use kernels::plane::Refusal;
use kernels_metal::plane::{ArgValue, Encode, Fire as Launch};
use model_compiler::program::{Call, Program};
use model_ir::plan::{Cond, Op, Param, Plan, Shard, ValueDef};

// ── the mock: an `Encode` that writes down what it was asked to fire ────

/// One launch, as the recorder saw it.
#[derive(Clone, Debug, PartialEq)]
struct Fired {
    /// The entrypoint the claim body named.
    entrypoint: &'static str,
    /// The shader it named it in.
    file: &'static str,
    /// Total threads, then threads per group.
    grid: [u32; 3],
    group: [u32; 3],
    /// Every argument, resolved through the fire's binding list: a buffer
    /// becomes the region it addresses, a scalar its own bits.
    args: Vec<Arg>,
}

/// One argument, in terms that survive the crossing.
///
/// A HANDLE IS NOT WHAT IS COMPARED, and that distinction is the test's. A
/// handle is a number the executor minted in the order it happened to mint
/// them, so two runs that bind the same regions in the same order agree on it
/// trivially and a run that binds a DIFFERENT region at the same slot agrees
/// on it too. What must hold is that slot `n` of dispatch `k` addresses the
/// bytes the plan's slot said, so the recorder resolves every handle through
/// the fire's own binding list before writing it down.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arg {
    /// A bound region: where it starts and how long it is.
    Buffer {
        address: u64,
        bytes: u64,
        mutable: bool,
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
    /// The one thing a metal claim body asks the fire for: a slot the point
    /// does not carry, which the shader still declares.
    fn absent(&self) -> Result<ArgValue, Refusal> {
        Ok(ArgValue::Buffer(
            self.bindings
                .borrow_mut()
                .take(driver_metal::baker::marks::NOTHING),
        ))
    }

    fn fire(&self, fire: Launch, args: &[ArgValue]) -> Result<(), Refusal> {
        let bindings = self.bindings.borrow();
        let resolve = |handle: u32, mutable: bool| {
            bindings
                .at(handle)
                .map_or(Arg::Unbound(handle), |b| Arg::Buffer {
                    address: b.slice.address,
                    bytes: b.slice.bytes,
                    mutable,
                })
        };
        self.fired.borrow_mut().push(Fired {
            entrypoint: fire.entrypoint,
            file: fire.file,
            grid: fire.lanes,
            group: fire.group,
            args: args
                .iter()
                .map(|a| match *a {
                    ArgValue::Buffer(h) => resolve(h, false),
                    ArgValue::BufferMut(h) => resolve(h, true),
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

// ── the fire's staging, as plain numbers ───────────────────────────────

/// Regions with recognisable addresses, so a transcript reads.
struct Staging {
    tables: BTreeMap<FireTable, Slice>,
}

impl Staging {
    fn new() -> Self {
        let mut tables = BTreeMap::new();
        for (at, which) in [
            FireTable::TokenIds,
            FireTable::Positions,
            FireTable::RequestOfToken,
            FireTable::QoIndptr,
            FireTable::RowValid,
        ]
        .into_iter()
        .enumerate()
        {
            tables.insert(
                which,
                Slice {
                    address: 0x1_0000 + (at as u64) * 0x1000,
                    bytes: 0x1000,
                },
            );
        }
        Self { tables }
    }
}

impl Pools for Staging {
    fn kv(&self, layer: u32, values: bool) -> Option<Slice> {
        Some(Slice {
            address: 0x10_0000 + u64::from(layer) * 0x2000 + u64::from(values) * 0x1000,
            bytes: 0x1000,
        })
    }

    fn slab(&self, _layer: u32, _which: Slab) -> Option<Slice> {
        None
    }

    fn kv_geometry(&self) -> KvGeometry {
        KvGeometry {
            page_size: 16,
            seq_stride: 64,
            head_stride: 1024,
        }
    }

    fn table(&self, which: FireTable) -> Option<Slice> {
        self.tables.get(&which).copied()
    }
}

// ── the plan ───────────────────────────────────────────────────────────

const HIDDEN: u64 = 64;
const VOCAB: u64 = 128;
const HEAD_DIM: u64 = 16;
const ARENA: u64 = 0x100_0000;

/// A weight this plane can bind: bf16, dense, at a recognisable address.
fn bank(at: u64, shape: Vec<u64>) -> Bank {
    Bank {
        slice: Slice {
            address: 0x100_0000 + at * 0x1000,
            bytes: 0x1000,
        },
        shape,
        dtype: model::produce::Dtype::Bf16,
        repr: "dense".to_string(),
    }
}

/// A five-statement tower: embed, normalise, normalise in another
/// convention, add the residual back, squash the logits in place.
///
/// `layout.embed` IS THE SEED AND IT IS DELIBERATELY UNCLAIMED. Its shape is
/// `[fire, table.axis(1)]` — a row per token, as wide as the embedding table
/// — which is the only rule in the whole point table that sizes a rectangle
/// WITHOUT reading an operand, and therefore the only thing that can start a
/// tower. Every other rule rides its input, and a runtime plane has no
/// rectangle for one to ride. That is not a fixture accident: it is why every
/// shipping text opens with an embed.
///
/// On this plane it is a CANON row (`kernels_metal::CANON` answers it with
/// `layout::embed_gather_mb_4bit`), so the lane BINDS — resolution finds the
/// symbol — and the walk REFUSES it, because the statement's operands are not
/// that routine's. [`the_walk_refuses_a_canon_symbol_by_name`] is that
/// assertion; the transcript tests walk from the first statement this plane
/// can actually fire, with the rectangle the embed would have written already
/// sized by the same walk.
///
/// Every other point here is on this plane's claim table. Two of them are
/// `InOut` (`norm.residual_add`'s hidden slot, `attention.logit_softcap`'s x)
/// so the blit path is exercised, and one (`norm.rmsnorm_per_head`) states a
/// scalar so the params run is too.
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
        plane: model_ir::kernels::Backend::Metal,
        facts: vec!["qo_one".into()],
        params: vec![
            Param {
                name: "embed.table".into(),
                shape: vec![VOCAB, HIDDEN],
                shard: Shard::Replicated,
                repr: "dense".into(),
            },
            Param {
                name: "norm.weight".into(),
                shape: vec![HIDDEN],
                shard: Shard::Replicated,
                repr: "dense".into(),
            },
        ],
        caches: Vec::new(),
        // 0: the fire's token ids (a runtime plane, so the walk reads it off
        // `Pools`); 1..: one result per statement.
        values: vec![
            ValueDef::Runtime("token_ids".into()),
            ValueDef::Stmt(0),
            ValueDef::Stmt(1),
            ValueDef::Stmt(2),
            ValueDef::Stmt(3),
            ValueDef::Stmt(4),
        ],
        ops: vec![
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
        ],
        seams: vec![model_ir::plan::Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: vec![5],
            layer: None,
        }],
    }
}

fn program(plan: &Plan) -> Program {
    let mut lanes = model_compiler::program::bound(plan);
    // ONE LANE, and the reason is `sweep::lanes`': a lane is a distinct set
    // of statements, not a fact word. Nothing in this fixture is conditional,
    // so both readings of `qo_one` run the same four statements and the sweep
    // collapses them into one lane serving both words. A text whose attention
    // arm branched on the fact would give two.
    assert_eq!(lanes.len(), 1, "nothing here is conditional, so one lane");
    lanes.remove(0).unwrap_or_else(|r| {
        panic!(
            "the fixture states only points this plane answers, and the lane refused: {:?}",
            r.gaps
        )
    })
}

/// The whole program, every step of which this plane can now fire.
///
/// THIS FUNCTION USED TO DROP A STEP. `layout.embed` was the seed and was
/// deliberately unclaimed, so it resolved to a `CANON` symbol this executor
/// refuses, and every transcript test below started at step 1. The metal
/// plane claims the point now — with four other families beside it — so the
/// fixture fires end to end and the drop would be a lie about what the plane
/// answers. The refusal path it used to exercise is
/// [`the_walk_refuses_a_symbol_call_by_name`], which builds its symbol
/// rather than borrowing one from a table that no longer has any.
fn fireable(plan: &Plan) -> Program {
    program(plan)
}

fn banks() -> BTreeMap<String, Bank> {
    let mut out = BTreeMap::new();
    out.insert("embed.table".to_string(), bank(0, vec![VOCAB, HIDDEN]));
    out.insert("norm.weight".to_string(), bank(1, vec![HIDDEN]));
    out
}

/// Walk `program` against a recorder and hand back what it fired.
fn walk(plan: &Plan, program: &Program) -> (Vec<Fired>, usize) {
    let pools = Staging::new();
    let banks = banks();
    let fire = Fire::over(
        plan,
        program,
        Extent {
            arena: Slice {
                address: 0x1000_0000,
                bytes: ARENA,
            },
            rows: 2,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let recorder = Recorder::over(&fire.bindings);
    fire.walk(&recorder)
        .unwrap_or_else(|why| panic!("the fixture's lane refused: {why}"));
    let blits = fire.blits.borrow().len();
    (recorder.transcript(), blits)
}

// ── the questions ──────────────────────────────────────────────────────

/// THE WALK ORDER IS THE PROGRAM'S ORDER, and every statement is visited
/// exactly once.
///
/// `model_compiler::program` puts the steps in a total order the arena's
/// liveness was carved against, so a walk that resequenced them would be
/// reading rectangles outside their spans — silently, because every address
/// would still be inside the arena.
#[test]
fn the_walk_visits_every_step_once_in_the_programs_order() {
    let plan = plan();
    let program = fireable(&plan);
    let (fired, _) = walk(&plan, &program);

    // Five fireable statements, and this plane answers each with exactly one
    // launch: the embed, three `norm/rms.metal` arms and one softcap. The
    // embed is FIRST and it is new — this fixture ran four steps for as long
    // as `layout.embed` was answered by a `CANON` symbol this executor
    // refuses.
    assert_eq!(program.steps.len(), 5, "five fireable statements");
    assert_eq!(
        fired.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        vec![
            "embed_bfloat16",
            // `rmsnorm_no_scale` is `norm/rms.metal`'s UNWEIGHTED arm and a
            // different entrypoint, which is the plane's own decomposition
            // and not the declaration's: four points share `rms_single_row`
            // because they differ only in an axis and a bank convention, and
            // this one has no bank at all.
            "vnorm_single_row_bfloat16",
            "rms_single_row_bfloat16",
            "residual_add_bfloat16",
            "logit_softcap_bfloat16",
        ],
        "one launch per statement, in the program's order",
    );
    assert!(
        fired.iter().all(|f| f.grid.iter().all(|&n| n > 0)),
        "a zero grid runs nothing and reports success: {fired:?}",
    );
}

/// EVERY STEP REACHES THE POINT THE PLAN NAMES, and a point this plane does
/// not claim refuses BY NAME rather than firing something near it.
#[test]
fn a_point_this_plane_does_not_claim_refuses_with_the_statement_named() {
    let mut plan = plan();
    // `mla.absorb_q` is declared on the floor, stated by the latent-attention
    // texts, and claimed by no metal family.
    //
    // IT USED TO BE `gemm.matmul`, which was the gap at the head of all 35
    // lane refusals until the gemm family landed. A test whose subject is "a
    // point this plane does not claim" has to be re-pointed the day the plane
    // claims it, or it goes on passing for a reason that is no longer the one
    // written here — the refusal would then be about the statement's operands
    // rather than about the point being absent, and nothing would say so.
    // Thirty points are still unclaimed on this plane; the whole `mla` family
    // is ten of them.
    plan.ops[1].kernel = "mla.absorb_q".to_string();
    let lanes = model_compiler::program::bound(&plan);
    let refusal = lanes[0]
        .as_ref()
        .expect_err("a lane stating an unclaimed point cannot bind");
    assert!(
        refusal.gaps.iter().any(|g| g.point == "mla.absorb_q"),
        "the refusal names the point: {:?}",
        refusal.gaps,
    );
}

/// EVERY OPERAND ARRIVES AT THE REGION ITS SLOT NAMES.
///
/// The transcript resolves each handle through the fire's binding list, so
/// this is a statement about BYTES and not about the numbering: statement 1
/// reads what statement 0 wrote, statement 2 reads both, and the weight slot
/// carries the bank the load put on the device.
#[test]
fn every_operand_addresses_the_region_its_slot_names() {
    let plan = plan();
    let program = fireable(&plan);
    let (fired, _) = walk(&plan, &program);

    let of = |f: &Fired, at: usize| f.args[at];

    // THE SECOND STATEMENT READS THE ARENA, and the rectangle it reads is
    // the one the EMBED wrote — sized by the same walk, carved into the same
    // arena. Two rows of 64 bf16 elements is 256 bytes, which is the embed's
    // `[fire, table.axis(1)]` at this fixture's dimensions. The embed itself
    // reads `token_ids` off `Pools` and a bank off the load, so it is the one
    // statement here whose inputs are not the walk's own.
    let Arg::Buffer { address, bytes, .. } = of(&fired[1], 0) else {
        panic!("the operand is a buffer: {:?}", fired[1].args);
    };
    assert!(address >= 0x1000_0000, "it is in the arena: {address:#x}");
    assert_eq!(bytes, 2 * HIDDEN * 2, "two rows of {HIDDEN} bf16 elements");

    // The weight is the bank the load registered, at its own address and its
    // own extent — not a rectangle out of the arena.
    assert_eq!(
        of(&fired[2], 1),
        Arg::Buffer {
            address: 0x100_0000 + 0x1000,
            bytes: 0x1000,
            mutable: false
        },
        "the `Const` slot binds the bank the load put on the device",
    );

    // Statement 2's operand is statement 1's result: the same region.
    let (Arg::Buffer { address: wrote, .. }, Arg::Buffer { address: read, .. }) =
        (of(&fired[1], 1), of(&fired[2], 0))
    else {
        panic!("both are buffers: {fired:?}");
    };
    assert_eq!(read, wrote, "statement 2 reads what statement 1 wrote");

    // And the embed's own output is what statement 1 reads, which is the
    // link this fixture could not assert while the seed was refused.
    let (
        Arg::Buffer {
            address: seeded, ..
        },
        Arg::Buffer { address: first, .. },
    ) = (of(&fired[0], 2), of(&fired[1], 0))
    else {
        panic!("both are buffers: {fired:?}");
    };
    assert_eq!(first, seeded, "the tower reads what the embed wrote");

    // Every arena region is inside the arena.
    for f in &fired {
        for a in &f.args {
            if let Arg::Buffer { address, bytes, .. } = *a
                && address >= 0x1000_0000
            {
                assert!(
                    address + bytes <= 0x1000_0000 + ARENA,
                    "a rectangle leaves the arena: {a:?}",
                );
            }
        }
    }
}

/// THE SCALARS THE STATEMENT STATES REACH THE BODY, in the params run's own
/// order.
#[test]
fn a_stated_scalar_reaches_the_claim_body() {
    let plan = plan();
    let program = fireable(&plan);
    let (fired, _) = walk(&plan, &program);

    // `norm.rmsnorm_per_head` states `head_dim` then `eps`; the body forwards
    // eps and the axis it derived from the head width.
    assert!(
        fired[2].args.contains(&Arg::F32(f32::to_bits(1e-6))),
        "the stated eps reaches the launch: {:?}",
        fired[2].args,
    );
    assert!(
        fired[2].args.contains(&Arg::I32(16)),
        "the stated head width becomes the norm's axis: {:?}",
        fired[2].args,
    );
    assert!(
        fired[4].args.contains(&Arg::F32(f32::to_bits(30.0))),
        "the stated softcap reaches the launch: {:?}",
        fired[4].args,
    );
}

/// AN `InOut` POINT SCHEDULES THE COPY THAT MAKES IT HONEST.
///
/// The walk mints a FRESH rectangle for every result, so a kernel that writes
/// through an in-place handle would leave the result's column holding
/// whatever the arena held. Two of the fixture's statements are `InOut`
/// (`norm.residual_add`'s hidden slot and `attention.logit_softcap`'s x), so
/// two blits are scheduled — and their `from`/`to` are disjoint, which is the
/// property the arena's inclusive spans guarantee and the copy depends on.
#[test]
fn an_in_place_point_schedules_a_disjoint_copy() {
    let plan = plan();
    let program = fireable(&plan);
    let (_, blits) = walk(&plan, &program);
    assert_eq!(blits, 2, "two `InOut` statements, two staged copies");
}

/// A SYMBOL CALL REFUSES WITH THE STATEMENT NAMED.
///
/// The staging between `embed(ids, table, vocab, y)` and a signature-table
/// routine like `layout::embed_gather_mb_4bit` is the affine bank's three
/// operands, which the statement does not carry. So the walk refuses, and
/// the refusal names the STATEMENT rather than faulting inside a shader.
///
/// THE SYMBOL IS BUILT HERE AND NOT READ OFF `CANON`, because
/// `kernels_metal::CANON` is empty: both of its rows — `layout.embed` and
/// `moe.weighted_sum` — are claimed now, and a claim wins at resolution, so
/// a row that survived would be one nothing ever reads (which is what
/// `points-dispatch`'s `every_canon_row_is_an_unclaimed_point` refuses).
/// What is under test here is the WALK's answer to a `Call::Symbol`, and
/// that is unchanged by where the variant came from. The resolver's half —
/// that a `CANON` row becomes a `Symbol` — is `sweep::resolve`'s to hold.
#[test]
fn the_walk_refuses_a_symbol_call_by_name() {
    let plan = plan();
    let mut whole = program(&plan);
    whole.steps[0].call = Call::Symbol("layout::embed_gather_mb_4bit");
    let pools = Staging::new();
    let banks = banks();
    let fire = Fire::over(
        &plan,
        &whole,
        Extent {
            arena: Slice {
                address: 0x1000_0000,
                bytes: ARENA,
            },
            rows: 2,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let recorder = Recorder::over(&fire.bindings);
    let refused = fire.walk(&recorder).expect_err("the canon seed refuses");
    assert_eq!(refused.op, 0, "the FIRST statement, not one deeper");
    assert_eq!(refused.kernel, "layout.embed");
    assert!(
        refused.to_string().contains("staging shim"),
        "the refusal says what is missing: {refused}",
    );
    assert!(
        recorder.transcript().is_empty(),
        "nothing fires before the refusal",
    );
}

/// A DROPPED STEP CHANGES THE TRANSCRIPT.
///
/// The first of three mutations. It is the cheapest thing a walk can get
/// wrong — an early `break`, a filter, a `continue` on a call variant — and
/// the cheapest to miss, because everything that DID run still ran correctly.
#[test]
fn dropping_a_step_changes_what_is_fired() {
    let plan = plan();
    let whole = fireable(&plan);
    let (before, _) = walk(&plan, &whole);

    let mut cut = fireable(&plan);
    cut.steps.pop().expect("four fireable steps");
    let (after, _) = walk(&plan, &cut);

    assert_ne!(before, after, "a dropped step has to be visible");
    assert_eq!(after.len(), before.len() - 1);
    assert_eq!(after[..], before[..before.len() - 1], "the rest is unmoved");
}

/// A WRONG POINT CHANGES THE TRANSCRIPT.
///
/// The second mutation, and it is chosen to be the HARDEST one of its kind:
/// `norm.rmsnorm_per_head` and `norm.rmsnorm_per_head_plus_one` declare the
/// same five slots in the same order and this plane answers both with the
/// same entrypoint. What separates them is ONE WORD of the scalar run — the
/// bank convention, `plus_one` — so a walk that dispatched the wrong one
/// would fire the right shader over the right rectangles with the right grid
/// and quietly compute `1 + w` where the checkpoint stores `w`.
///
/// That is not hypothetical: it is `.wiki/baker-todo.md`'s W3 finding, where
/// an import's `plus_one` fold made a model agree with itself and disagree
/// with its checkpoint.
#[test]
fn firing_the_wrong_point_changes_what_is_fired() {
    let plan = plan();
    let whole = fireable(&plan);
    let (before, _) = walk(&plan, &whole);

    let mut swapped = fireable(&plan);
    swapped.steps[2].call = Call::Point("norm.rmsnorm_per_head_plus_one".to_string());
    let (after, _) = walk(&plan, &swapped);

    assert_ne!(before, after, "a different point has to be visible");
    assert_eq!(
        after[2].entrypoint, before[2].entrypoint,
        "both are the same shader — which is why the arguments are what tells them apart",
    );
    assert_eq!(
        after[2].args.len(),
        before[2].args.len(),
        "the same argument count — one word of the scalar run is the whole difference",
    );
    assert_ne!(after[2].args, before[2].args, "the bank convention moved");
}

/// A SWAPPED HANDLE CHANGES THE TRANSCRIPT.
///
/// The third mutation and the one a weaker test misses. Exchanging two
/// operands of `norm.residual_add` leaves every entrypoint, every grid and
/// every argument COUNT identical; the only thing that moves is which region
/// arrives at which slot. A transcript that recorded handle NUMBERS rather
/// than the regions they resolve to would be blind to it, which is why
/// [`Arg`] resolves.
#[test]
fn swapping_two_operands_changes_what_is_bound() {
    let plan = plan();
    let whole = fireable(&plan);
    let (before, _) = walk(&plan, &whole);

    let mut moved = plan.clone();
    moved.ops[3].inputs.swap(0, 1);
    let moved_program = fireable(&moved);
    let (after, _) = walk(&moved, &moved_program);

    assert_ne!(before, after, "a swapped operand has to be visible");
    assert_eq!(
        after.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        before.iter().map(|f| f.entrypoint).collect::<Vec<_>>(),
        "the same shaders fire — only the regions moved",
    );
    assert_eq!(
        after[3].args.len(),
        before[3].args.len(),
        "the same argument count — only the regions moved",
    );
    assert_ne!(
        after[3].args, before[3].args,
        "the swapped statement is `norm.residual_add`, which fires third \
         behind the embed and the two norms",
    );
}

/// A HANDLE THE FIRE NEVER MINTED IS NOT ANSWERED WITH A ZERO ADDRESS.
///
/// The production encoder refuses it (`baker::encode::lay_out`), and the
/// refusal matters: on the legacy table path the answer was a zero address,
/// and `mxfp4_qmv_routed_bias` read an additive bias off a null pointer for
/// every expert logit with nothing in the path saying a word.
#[test]
fn a_handle_the_fire_never_minted_refuses() {
    let bindings = RefCell::new(Bindings::new());
    let cursor = std::cell::Cell::new(driver_metal::baker::walk::Cursor::default());
    let encoder = driver_metal::baker::encode::Encoder::over(&bindings, &cursor);
    let why = encoder
        .fire(
            Launch::at("norm/rms.metal", "rms_single_row_bfloat16")
                .apply(kernels::Grid::of([32, 1, 1], [32, 1, 1])),
            &[ArgValue::Buffer(7)],
        )
        .expect_err("handle 7 was never minted");
    assert!(
        matches!(why, Refusal::Absent { .. }),
        "an unminted handle is absent, not a zero address: {why:?}",
    );
}

// ── the catalog, as it actually stands ─────────────────────────────────

/// EVERY CATALOG ROW TRACES FOR THIS PLANE, and this is where the state of
/// the migration is written down rather than assumed.
///
/// Naming `Backend::Metal` is the whole of what selects a plane, so every row
/// in the catalog produces a metal `Plan` and every plan's lanes go through
/// `bound`.
///
/// THE DAY CAME. This test used to assert that NO row could bind, on the
/// grounds that every shipping text states `gemm.matmul` and no metal family
/// claimed it, and it said the assertion would fail the day one did. It did:
/// the plane went from 22 claimed points to 51 in one wave — the gemm family,
/// the three attention scores, the packed activations, the routed points, the
/// whole ssm family and the norm/rope tails — and [`BOUND`] is what binds
/// now. The assertion is inverted rather than deleted, so a regression that
/// UN-binds a row is as loud as the landing was.
///
/// Binding is not serving. A lane that binds has an answer for every point it
/// states; whether the shaders compute the right numbers is a question only an
/// Apple device can answer, and none of this MSL has ever been compiled.
#[test]
fn every_catalog_row_traces_for_this_plane() {
    let mut bound = Vec::new();
    let mut traced = 0usize;
    for row in model::serve::ROWS {
        let Ok(baked) = Baked::of::<Metal>(row.id) else {
            // `Deployment::of` refuses gemma/glm/kimi/dsv4 by name, at the
            // pool and not at the plane. Those rows still TRACE; what they do
            // not do is project a deployment this driver has a pool for.
            continue;
        };
        traced += 1;
        assert!(
            !baked.plan.ops.is_empty(),
            "`{}` traced an empty plan for this plane",
            row.id,
        );
        assert_eq!(baked.plan.plane, model_ir::kernels::Backend::Metal);
        if baked.lanes.iter().any(Result::is_ok) {
            bound.push(row.id);
        }
    }
    assert!(traced > 0, "no catalog row traced for this plane at all");
    bound.sort_unstable();
    let mut want = BOUND.to_vec();
    want.sort_unstable();
    assert_eq!(
        bound, want,
        "the rows whose lanes bind on this plane have moved. If a row was \
         GAINED, add it to `BOUND` — that edit is the record of the point \
         that landed. If a row was LOST, a claim this plane used to answer \
         stopped answering, and that is a regression rather than a list to \
         edit.",
    );
}

/// The catalog rows with at least one lane that binds on metal.
///
/// One wave of claims moved this from empty. It is a list and not a count
/// because which rows bind says which families landed: the two gemma rows and
/// the three qwen rows state the dense tower — embed, gemm, norms, rope,
/// attention scores, packed activations — and the routed qwen row adds the
/// moe family on top.
const BOUND: &[&str] = &[
    "gemma4-e4b-bf16-kv-bf16",
    "gemma4-31b-bf16-kv-bf16",
    "qwen35-a3b-bf16-kv-bf16",
    "qwen35-d3b-bf16-kv-bf16",
    "qwen35-d0.8b-bf16-kv-bf16",
    // THE TWO GPT-OSS ROWS ARRIVED LAST, and what they were waiting on says
    // what a lane is: three points, `attention.{decode_lse, prefill_lse,
    // sink}`, none of them a kernel this plane lacked. The sdpa arms could
    // already compute a sink-bearing softmax — they folded it into the
    // denominator — but `decode_lse` DECLARES an `Out` lse plane, and a
    // point is a contract about what is written. So the plane published the
    // number it had been keeping to itself, and the rescale that reads it
    // is a decomposition rather than a rewrite: fired end to end, the two
    // roads agree to within a bf16 store.
    "gptoss-20b-bf16-mxfp4-kv-bf16",
    "gptoss-120b-bf16-mxfp4-kv-bf16",
];

/// The eager resolve pass reports the WHOLE backlog, not the first gap.
///
/// A lane missing four points should report four, so one load says what the
/// whole backlog is; and it dedupes, so a 24-layer stack reports a missing
/// point once with the first statement that asked.
#[test]
fn the_resolve_pass_reports_the_backlog_and_dedupes_it() {
    let plan = plan();
    let program = fireable(&plan);
    assert!(
        driver_metal::baker::resolve::check(&plan, &program).is_empty(),
        "every point the fixture states is claimed at bf16",
    );

    // The same program, with one step pointed at a claimed point AT AN
    // ELEMENT this plane's arms do not instantiate for it.
    let mut absent = program.clone();
    for step in &mut absent.steps {
        step.call = Call::Point("norm.res_blend".to_string());
    }
    let gaps = driver_metal::baker::resolve::check(&plan, &absent);
    assert_eq!(
        gaps.len(),
        1,
        "five statements, one point, one row: {gaps:?}"
    );
    assert_eq!(gaps[0].op, 0, "the FIRST statement that asked");
    assert!(gaps[0].why.contains("no point of that name"), "{}", gaps[0]);
}
