//! The guest-program plane, host half against device half, byte for byte.
//!
//! **THIS IS THE POINT OF STEP 7.** Every other test in this crate can pass
//! while a PTIR fire computes the wrong thing: the launch succeeded, the
//! commit slot survived, a channel holds *a* value. What cannot be faked is
//! agreement with `engine::program` — a complete interpreter of the same
//! launch package, written against the same op table, that never touches a
//! device. So both halves are handed the same programs, the same seeds and the
//! same per-fire inputs, and after every fire the test compares:
//!
//! * the outcome, including WHICH channel a blocked fire blocked on;
//! * every ring, slot for slot, in wire bytes — not just what was drained,
//!   because a program that writes the right value into the wrong slot is
//!   invisible from the drain;
//! * both cursors of every channel;
//! * what a reader takes out, cell for cell.
//!
//! Bit-for-bit reproducibility is the channel plane's first-class contract,
//! which is why the device compiles with `--fmad=false --prec-div=true
//! --prec-sqrt=true` and why "close enough" is not an option anywhere below.
//!
//! **THE SUBJECTS ARE REAL GOLDEN TRACES**, not fixtures written here: the
//! `tensor-compiler` corpus under `tests/golden/`, decoded from the `container:`
//! line, bound against the profile each was authored for, planned by
//! `compile_bound`, and emitted for CUDA by `emit_program`. That is exactly the
//! path a registration takes in production, so what this test exercises is the
//! path and not a mock of it. The five chosen are the corpus's programs whose
//! two halves must agree EXACTLY rather than nearly: every op integral or
//! boolean, no RNG to seed, and where a `logits` row is read it is a buffer
//! this test resident-ises itself and points BOTH halves at — which is also
//! the only exercise the runtime's attachment seam gets until the runtime
//! builds it.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-cuda --features cuda-13 --test program_parity -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine::engine_api::program::{
    LaunchChannel, LaunchPackage, ProgramRegistration, ValueSource,
};
use engine::tensor_ir::container::HostRole;
use tensor_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::Stage;
use tensor_ir::types::{Literal, Shape};
use engine::tensor_ir::DType;
use engine::tensor_ir::op::IntrinsicId;
use engine::{
    Boundaries, ExecPlan, Extents, HostOp, InterpInstance, PassInputs, StepOutcome, Value,
    adopt_launch_package_with, concrete_dtype, encode_wire, host_put, host_take,
    make_host_instance, step, wire_cell_bytes,
};
use engine_cuda::device::{Buffer, Context, present};
use engine_cuda::program::{Disk, Fired, Plane};
use tensor_ir::registry::{GeometryClass, KernelInfo, ModelProfile};

/// One shell per process — `kernels-cuda`'s scratch slabs are process-global
/// and this crate's other GPU suites say the same. The guest-program plane
/// allocates its own bytes and does not touch them, but it does bind a context
/// on the calling thread, and two contexts racing for device 0 is a flake
/// nobody can read.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One program under test.
struct Subject {
    /// Its name in the corpus.
    name: &'static str,
    /// How many readout rows its `logits` intrinsic reads, or zero for a
    /// program that reads no intrinsic.
    ///
    /// **NOT GUESSABLE, SO IT IS STATED.** A plan's value shapes are symbolic
    /// in the seven extent roles; the launch package's are concrete, because
    /// the trace was bound at a profile. `SampledRows` is the one role these
    /// programs use, and a fire that resolves it to the wrong number sizes
    /// every downstream value wrong — a put whose value is one row where the
    /// channel cell is two zero-fills the tail and diverges silently. In
    /// production this number is the model fire's; here it is the corpus's,
    /// and it is checked against the package's own concrete shape below.
    rows: u32,
    /// How many rows its `mtp_logits` intrinsic reads, or zero for a program
    /// that reads no draft column (palo C3b).
    ///
    /// **A SECOND NUMBER, BECAUSE THERE IS A SECOND COLUMN.** Until this wave
    /// a draft was rows `mtp_draft_row ..` of the ONE readout, because there
    /// was one rectangle for a shell to bind. The `mtp` export is its own
    /// rectangle now — `model_compiler::arena` holds it open past the last
    /// node and `Shell::fire_captured` points `IntrinsicId::MtpLogits` at its
    /// base — so a subject that reads drafts is handed two buffers and the
    /// host interpreter is handed the same two.
    draft_rows: u32,
    /// Why this program is in the list.
    why: &'static str,
}

/// The programs both halves run.
///
/// Every one is integral where it matters: the two halves have to agree on the
/// bit, and an `exp` in the middle would turn a real disagreement into an
/// argument about the last mantissa bit of somebody's libm. The three that
/// read `logits` are handed the same f32 buffer on both sides — exact binary
/// fractions, all distinct, so an argmax has no tie to break differently.
const SUBJECTS: &[Subject] = &[
    // A loop-carried counter: one channel taken AND put in the same fire,
    // which is the shape every decode loop has and the one where a commit
    // that consumed before it published leaves the wrong value behind.
    Subject {
        name: "counter_pingpong",
        rows: 0,
        draft_rows: 0,
        why: "take, add, put back into the same ring",
    },
    // Reads that peek without consuming beside a take that does, plus a
    // gather, a select and an argmax over a bound `logits` row — the case
    // where conflating `chan_read` with `chan_take` drops a cell per fire.
    Subject {
        name: "dfa_ingraph",
        rows: 1,
        draft_rows: 0,
        why: "two peeked channels, one taken, two published",
    },
    // A bool matrix in, an i32 selection out: the bool cell is the one dtype
    // whose device and wire spellings differ, so this is the packing door.
    Subject {
        name: "matrix_select_mask",
        rows: 4,
        draft_rows: 0,
        why: "a packed bool matrix selects rows of the logits",
    },
    // Seven channels, four of them bool masks — causal, sliding-window,
    // sink-window and a packed apply. The widest single region in the corpus
    // that reads no intrinsic at all.
    Subject {
        name: "structured_masks",
        rows: 0,
        draft_rows: 0,
        why: "three u32 inputs, four bool mask outputs",
    },
    // No channel inputs at all, one output at capacity one: the second fire
    // has nowhere to publish, so this is the blocked-fire case, and both
    // halves must block on the same channel.
    Subject {
        name: "matrix_mask_apply_packed",
        rows: 2,
        draft_rows: 0,
        why: "no channel inputs; blocks on its own full output",
    },
    // **THE DRAFT COLUMN'S DEVICE GATE** (palo C3b). Two intrinsic columns in
    // one epilogue — `Logits [K+1, V]` and `MtpLogits [K, V]` — each argmaxed
    // and published on its own channel. A shell that pointed the draft
    // intrinsic at the trunk's base, or at the right base with the trunk's row
    // offset, publishes a different token vector rather than a nearby number,
    // and the two halves diff it byte for byte.
    //
    // **NOT `mtp_verify_tail`, AND THE REASON WAS THE EMITTER — IT NO LONGER
    // IS.** The corpus does hold a two-column program, the full match-verify
    // DAG, and its accept-prefix logic is a `cumprod`. This backend used to
    // decline any region carrying a scan boundary (`region contains a
    // non-generated boundary (scan)`), which is why the subject was built here
    // from the two ops the gate is actually about. That gap is closed —
    // `tensor_compiler::codegen::cuda::scan` emits the region, and the
    // `scan_prefix` subject below is what proves the fold — so adopting the
    // corpus program is now a choice about this axis rather than a codegen
    // blocker. It stays synthetic because the argument for it never was the
    // gap: this is the SMALLEST program that can tell the two columns apart,
    // and the match-verify DAG around it would only add ops five other
    // subjects already cover.
    //
    // **THIS IS AS FAR AS AN L40S CAN TAKE THE MTP AXIS.** The one shipping
    // SKU whose checkpoint publishes a draft head is `qwen36-27b`, whose bf16
    // weights are ~52 GiB against 46 (`export_axes`'s gate (f) states the
    // refusal). What this subject proves is the PLUMBING — two rectangles, two
    // side-table slots, two host columns, one bit-for-bit diff — on a
    // synthetic readout, exactly as the other five prove theirs. The head's own
    // arithmetic waits for a checkpoint that fits.
    Subject {
        name: MTP_SUBJECT,
        rows: 4,
        draft_rows: 3,
        why: "two intrinsic columns in one epilogue, argmaxed apart",
    },
    // **THE `Order` LIBRARY REGIONS** — `top_k` and `sort_desc`, neither a
    // fused body nor a second-party name but a generated kernel of their own
    // (`tensor_compiler::codegen::cuda::order`, one emitter for both because
    // `sort_desc` IS `top_k` at `k = n`). Eight of them in one stage over two
    // rows of a thousand, so what is compared is the ORDER: descending by
    // value, ties to the lower index, values and indices agreeing element for
    // element with `engine::program`'s own `sort_desc_order`.
    //
    // The two rows are chosen to fail differently. The taken channel holds
    // sixteen distinct eighths over a thousand lanes — sixty-odd ties per
    // value, so every pick of every `k` is a tie broken, and a sort that is not
    // STABLE lands on the wrong index while landing on the right value. The
    // derived row, `taken - iota`, is mostly distinct and mostly NEGATIVE, so
    // it is the ordering itself under test: a descending key that mishandles
    // the sign bit sorts it exactly backwards.
    //
    // The `sort_desc` pair is what pins the FULL order, thousand entries deep,
    // where the `top_k`s pin only their first `k` — and it is the one shape
    // that reaches the emitter's runtime-width path, because a sorted row's
    // width can be symbolic in the plan where a `k` never is.
    Subject {
        name: ORDER_SUBJECT,
        rows: 0,
        draft_rows: 0,
        why: "top_k at k in {1,3,8} and sort_desc, over ties and negatives",
    },
    // **THE SCAN LIBRARY REGION** — `cumsum`/`cumprod`
    // (`tensor_compiler::codegen::cuda::scan`), and the one whose kernel is
    // deliberately NOT parallel within a row. The reference is a left-to-right
    // f32 fold and this suite compares bytes, so any reassociation is a
    // failure; what the kernel parallelises is rows, and the subject carries
    // both shapes to prove it: a single thousand-long row (one thread, the
    // dependent-FADD chain that a left-to-right fold IS) and a `[8, 125]`
    // reshape of the same data (eight threads, eight independent folds).
    //
    // THE ROW IS DIVIDED BY THREE FIRST, and that is the whole point. `cell`
    // hands out multiples of an eighth; a thousand of those sum to well under
    // 2^24, so every partial sum is EXACT and the accumulation order is
    // unobservable — a chunked scan would pass. One IEEE division
    // (`--prec-div=true` on the device, correctly rounded on both sides) makes
    // the mantissas full and the rounding order-dependent, which is what turns
    // this subject into a test of the fold and not just of the arithmetic.
    Subject {
        name: SCAN_SUBJECT,
        rows: 0,
        draft_rows: 0,
        why: "cumsum and cumprod, one long row and eight short ones",
    },
];

/// The one subject built here rather than read from the corpus.
const MTP_SUBJECT: &str = "mtp_two_columns";

/// The other two.
const ORDER_SUBJECT: &str = "order_ranks";
const SCAN_SUBJECT: &str = "scan_prefix";

/// How wide the row the two hand-built subjects rank and scan is.
///
/// A thousand, not a vocabulary, and the reason is the OTHER half: the host
/// interpreter sorts and folds this row once per region per fire, so a
/// 151,936-wide row costs the reference far more than it costs either kernel.
/// Neither branches on the width — the order kernel loops to
/// `input_desc.last`, the scan kernel to the same — so what a wider row adds
/// is runtime, not coverage.
///
/// It has been run at 151,936 BY HAND, both subjects at once and both
/// agreeing: a full vocabulary-deep `sort_desc` order, `top_k` at each `k`,
/// a 151,936-long single-row `cumsum` (one thread, that many dependent adds)
/// and an `8 x 18,992` `cumsum`/`cumprod` pair, over seven fires, in 3.8s of
/// a release build. That is the evidence for the scan kernel's cost argument
/// as much as for its correctness, and the width was then put back.
const ORDER_ROW: u32 = 1000;

/// The `k`s that subject asks for, each on both rows.
///
/// Small on purpose, because the whole-row case is covered by the `sort_desc`
/// pair beside them rather than by a `k = n` here: both go through the same
/// kernel, and the sorted form is the one that also exercises the runtime
/// width. What these three pin is the truncation — that the kernel writes `k`
/// entries of an order it computed in full, and the right `k`.
const ORDER_KS: [u32; 3] = [1, 3, 8];

/// How the scan subject reshapes its thousand-long row: `ROWS x COLUMNS`.
///
/// Eight rows and not one, because the kernel's only parallelism is across
/// rows — one thread per row — and a subject with a single row would never run
/// two folds at once, which is where a shared accumulator or a row-stride
/// error lives.
const SCAN_ROWS: u32 = 8;
const SCAN_COLUMNS: u32 = ORDER_ROW / SCAN_ROWS;

/// What the scan subject divides by to make its row non-dyadic, and what it
/// then adds to keep a `cumprod` of a hundred-odd terms inside f32's range.
///
/// Three is not special; being a non-power-of-two is. `0.7` puts the
/// multiplicands either side of one, so the running product wanders instead of
/// collapsing to zero or running off to infinity within the first few terms —
/// which would make every later element agree trivially.
const SCAN_DIVISOR: f32 = 3.0;
const SCAN_OFFSET: f32 = 0.7;

/// `mtp_two_columns`, whole: one epilogue, two intrinsic columns, two argmaxes,
/// two publishes.
///
/// Deliberately the SMALLEST program that can tell the two columns apart. Every
/// op in it is one the other five subjects already exercise; what is new is
/// that two `IntrinsicVal`s of different heights stand in one stage, which is
/// the shape a draft-reading guest has and the shape the side tables are
/// indexed for.
fn mtp_two_columns() -> TraceContainer {
    let (k, v) = (3u32, 8u32);
    let out = |len: u32| ChannelDecl {
        shape: Shape::vector(len),
        dtype: ChanDType::Concrete(DType::I32),
        capacity: 1,
        host_role: HostRole::Reader,
        seeded: false,
    };
    TraceContainer {
        names: vec![],
        channels: vec![out(k + 1), out(k)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // 0: the trunk's readout, `[K+1, V]`.
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(k + 1, v),
                    dtype: DType::F32,
                },
                // 1: the draft head's, `[K, V]` — a different height, so a
                // binding that confused the two cannot even be shape-correct.
                Op::IntrinsicVal {
                    intr: IntrinsicId::MtpLogits,
                    shape: Shape::matrix(k, v),
                    dtype: DType::F32,
                },
                Op::ReduceArgmax(0),
                Op::ReduceArgmax(1),
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanPut { chan: 1, value: 3 },
            ],
        }],
        externs: Vec::new(),
    }
}

/// A reader channel of `shape`, at capacity one so an undrained round fills it.
fn reader(shape: Shape, dtype: DType) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

/// The thousand-long f32 row both hand-built ranking subjects take, refilled
/// every round by the harness.
fn taken_row() -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(ORDER_ROW),
        dtype: ChanDType::Concrete(DType::F32),
        capacity: 1,
        host_role: HostRole::Writer,
        seeded: true,
    }
}

/// `order_ranks`, whole: one taken row, one derived from it, a `top_k` of each
/// `k` in [`ORDER_KS`] over both, and a `sort_desc` of each.
///
/// Every op here is exact on both halves. `cast` of an `iota` is an integer
/// under 2^24 in an f32, and the `sub` that follows is a subtraction of two
/// exactly-representable dyadic values — so the comparison downstream is byte
/// for byte, as it is for every other subject, and a disagreement is a wrong
/// answer rather than a rounding argument.
fn order_ranks() -> TraceContainer {
    let mut channels = vec![taken_row()];
    for k in ORDER_KS {
        for _ in 0..2 {
            channels.push(reader(Shape::vector(k), DType::F32));
            channels.push(reader(Shape::vector(k), DType::U32));
        }
    }
    // The `sort_desc` pair, one per row: the whole order, full width.
    for _ in 0..2 {
        channels.push(reader(Shape::vector(ORDER_ROW), DType::F32));
        channels.push(reader(Shape::vector(ORDER_ROW), DType::U32));
    }

    // 0: the taken row. 1..3: `iota` cast to f32 and subtracted from it.
    let mut ops = vec![
        Op::ChanTake(0),
        Op::Iota { len: ORDER_ROW },
        Op::Cast {
            value: 1,
            dtype: DType::F32,
        },
        Op::Sub(0, 2),
    ];
    // Each `Order` op defines two values, so the pair for the `i`th one starts
    // at `4 + 2 * i` — the same arithmetic `result_bases` does, written out
    // here because a trace names values, not ops.
    let mut chan = 1u32;
    let mut value = 4u32;
    let publish = |ops: &mut Vec<Op>, chan: &mut u32, value: &mut u32| {
        ops.push(Op::ChanPut {
            chan: *chan,
            value: *value,
        });
        ops.push(Op::ChanPut {
            chan: *chan + 1,
            value: *value + 1,
        });
        *chan += 2;
        *value += 2;
    };
    for k in ORDER_KS {
        for row in [0u32, 3u32] {
            ops.push(Op::TopK { input: row, k });
            publish(&mut ops, &mut chan, &mut value);
        }
    }
    for row in [0u32, 3u32] {
        ops.push(Op::SortDesc(row));
        publish(&mut ops, &mut chan, &mut value);
    }

    TraceContainer {
        names: vec![],
        channels,
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

/// `scan_prefix`, whole: one taken row made non-dyadic, then an inclusive
/// prefix of it at both shapes and both folds.
///
/// The three published values are the three things a scan kernel can get
/// wrong: the long single-row `cumsum` (the fold's ORDER, on one thread), the
/// reshaped `cumsum` (the row STRIDE, on eight), and the reshaped `cumprod`
/// (the IDENTITY — one, not zero — and the other combiner).
fn scan_prefix() -> TraceContainer {
    let matrix = || Shape::matrix(SCAN_ROWS, SCAN_COLUMNS);
    let channels = vec![
        taken_row(),
        reader(Shape::vector(ORDER_ROW), DType::F32),
        reader(matrix(), DType::F32),
        reader(matrix(), DType::F32),
    ];
    let ops = vec![
        // 0: the taken row, multiples of an eighth.
        Op::ChanTake(0),
        // 1..3: divided by a non-power-of-two, so the partial sums round.
        Op::Const(Literal::F32(SCAN_DIVISOR)),
        Op::Broadcast {
            value: 1,
            shape: Shape::vector(ORDER_ROW),
        },
        Op::Div(0, 2),
        // 4: the long single-row inclusive sum — one thread, `ORDER_ROW`
        // dependent adds.
        Op::CumSum(3),
        // 5: the same data as eight rows — eight threads, eight folds.
        Op::Reshape {
            value: 3,
            shape: Shape::matrix(SCAN_ROWS, SCAN_COLUMNS),
        },
        // 6: the inclusive sum of those.
        Op::CumSum(5),
        // 7..9: shifted either side of one so the running product wanders.
        Op::Const(Literal::F32(SCAN_OFFSET)),
        Op::Broadcast {
            value: 7,
            shape: Shape::matrix(SCAN_ROWS, SCAN_COLUMNS),
        },
        Op::Add(5, 8),
        // 10: the inclusive product.
        Op::CumProd(9),
        Op::ChanPut { chan: 1, value: 4 },
        Op::ChanPut { chan: 2, value: 6 },
        Op::ChanPut { chan: 3, value: 10 },
    ];

    TraceContainer {
        names: vec![],
        channels,
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

/// The vocabulary every corpus program that reads `logits` was bound at.
///
/// Read off [`golden_profile`] rather than restated: a program bound at a
/// different vocabulary would resolve a different logits row, and the shape
/// check in [`parity`] is what would catch it.
const VOCAB: u32 = 8;

/// How many fires each program runs. Enough that a loop-carried channel wraps
/// its ring several times: `counter_pingpong`'s capacity is one, so the ring
/// is two, and eight fires wrap it four times.
const FIRES: usize = 8;

/// The one round whose outputs are deliberately left in the ring, so that the
/// round after it blocks. See the drain loop in [`parity`].
const HOLD_THE_DRAIN: usize = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Getting a registration out of the corpus
// ─────────────────────────────────────────────────────────────────────────────

fn golden_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tensor-compiler/tests/golden"
    ))
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex"))
        .collect()
}

/// The bind-time profile each golden was authored against, transcribed from
/// `tensor-compiler`'s own corpus helper. The goldens do not carry it, and
/// binding one at the wrong vocabulary refuses rather than misbehaves — which
/// is why this is a copy and not a guess.
fn golden_profile(name: &str) -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    match name {
        "counter_pingpong" | "lora_prologue" | "section3_masked_gumbel" | "structured_masks" => {}
        "beam_epilogue" => {
            profile.vocab = 8;
            profile.page_size = 4;
        }
        "pentathlon_iter" => {
            profile.vocab = 8;
            profile.kernels.push(KernelInfo {
                name: "envelope_dot".into(),
                sink_scope: None,
                replayable: true,
            });
        }
        _ => profile.vocab = 8,
    }
    profile
}

/// One subject's trace: the corpus's, or — for [`MTP_SUBJECT`] — this file's.
fn container_of(name: &str) -> TraceContainer {
    if name == MTP_SUBJECT {
        return mtp_two_columns();
    }
    if name == ORDER_SUBJECT {
        return order_ranks();
    }
    if name == SCAN_SUBJECT {
        return scan_prefix();
    }
    let path = golden_dir().join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path:?} is missing"));
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("container: "))
        .unwrap_or_else(|| panic!("{name} has no container line"));
    tensor_ir::container::decode(&unhex(line))
        .unwrap_or_else(|why| panic!("{name} does not decode: {why:?}"))
}

/// One trace, all the way to what `register_program` takes.
fn registration(name: &str) -> ProgramRegistration {
    let container = container_of(name);
    let bound = tensor_ir::validate::bind(container, golden_profile(name))
        .unwrap_or_else(|why| panic!("{name} does not bind: {why:?}"));
    let stages = tensor_compiler::plan::compile_bound(&bound);
    let launch = tensor_compiler::codegen::launch::build(&bound, &stages);
    let backend = tensor_compiler::codegen::program::Backend::Cuda;
    let emitted = tensor_compiler::codegen::program::emit_program(backend, &stages, &bound);

    ProgramRegistration {
        // The bound trace's own hash, which is what the host uses: two
        // registrations of one program have to be recognised as one.
        program_hash: bound.hash,
        emitted_kernels: emitted
            .into_iter()
            .map(|kernel| engine::engine_api::program::EmittedKernel {
                kind: kernel.kind,
                stage_index: kernel.stage_index,
                region_index: kernel.region_index,
                entry_name: kernel.entry_name,
                source: kernel.source,
                error: kernel.error,
            })
            .collect(),
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The same bytes to both halves
// ─────────────────────────────────────────────────────────────────────────────

fn numel(declared: &LaunchChannel) -> usize {
    declared
        .shape
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1)
}

fn seeded(declared: &LaunchChannel) -> bool {
    declared.seeded
}

fn host_writes(declared: &LaunchChannel) -> bool {
    declared.host_role == HostRole::Writer
}

fn host_reads(declared: &LaunchChannel) -> bool {
    declared.host_role == HostRole::Reader
}

/// A deterministic cell for `(channel, round)`.
///
/// **DETERMINISTIC AND NOT RANDOM**, because the two halves have to be handed
/// the SAME bytes and a failure has to be reproducible from the test's name
/// alone. The values are small so that a u32 sum over a decode loop stays
/// nowhere near a wrap, and non-constant across rounds so a program that
/// ignores its input cannot pass by accident.
fn cell(dtype: DType, lanes: usize, channel: u32, round: usize) -> Value {
    let mut state = 0x9e37_79b9_u32
        .wrapping_mul(channel.wrapping_add(1))
        .wrapping_add(round as u32 * 0x0851_1e19);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    match dtype {
        DType::U32 => Value::U32((0..lanes).map(|_| next() % 7).collect()),
        DType::I32 => Value::I32((0..lanes).map(|_| (next() % 7) as i32).collect()),
        DType::Bool => Value::Bool((0..lanes).map(|_| u8::from(next() % 2 == 1)).collect()),
        // Small exact binary fractions: representable to the bit, so an f32
        // channel would still be comparable byte for byte.
        DType::F32 => Value::F32((0..lanes).map(|_| (next() % 16) as f32 / 8.0).collect()),
    }
}

fn wire(value: &Value, dtype: DType, lanes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; wire_cell_bytes(dtype, lanes)];
    encode_wire(value, &mut bytes);
    bytes
}

/// The seeds both halves start from: the host half's as decoded values, the
/// device half's as the wire bytes of the same values.
struct Seeds {
    /// What `make_host_instance` takes.
    host: BTreeMap<u32, Value>,
    /// What `Plane::bind` takes.
    device: Vec<(u32, Vec<u8>)>,
}

/// One cell for every channel the program declares a seed for.
fn seeds(package: &LaunchPackage) -> Seeds {
    let mut values = BTreeMap::new();
    let mut bytes = Vec::new();
    for (index, declared) in package.channels.iter().enumerate() {
        if !seeded(declared) {
            continue;
        }
        let channel = index as u32;
        let dtype = concrete_dtype(declared.dtype);
        let lanes = numel(declared);
        let value = cell(dtype, lanes, channel, 0);
        bytes.push((channel, wire(&value, dtype, lanes)));
        values.insert(channel, value);
    }
    Seeds {
        host: values,
        device: bytes,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The parity run
// ─────────────────────────────────────────────────────────────────────────────

/// Both halves' verdicts, in one vocabulary, so a mismatch names both.
fn same_outcome(host: &StepOutcome, device: &Fired) -> bool {
    match (host, device) {
        (StepOutcome::Committed, Fired::Committed) => true,
        (StepOutcome::Blocked(a), Fired::Blocked(b)) => a == b,
        (StepOutcome::Faulted(_), Fired::Faulted(_) | Fired::Declined) => true,
        _ => false,
    }
}

/// The readout both halves read, when a program reads one.
///
/// `rows * VOCAB` distinct, exactly-representable f32 values. Distinct because
/// an argmax over a tie is a decision, and the two halves are entitled to make
/// it differently; exactly representable because everything downstream of it
/// is compared byte for byte.
fn logits(rows: u32, name: &str) -> Vec<f32> {
    let salt = name.bytes().fold(1u32, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(u32::from(b))
    });
    (0..rows * VOCAB)
        .map(|index| ((index.wrapping_mul(5).wrapping_add(salt % 11) % 64) as f32 - 32.0) / 4.0)
        .collect()
}

/// Run `subject` on both halves for [`FIRES`] fires and assert they never
/// differ.
fn parity(context: &Context, plane: &mut Plane, subject: &Subject) {
    let Subject {
        name,
        rows,
        draft_rows,
        why,
    } = *subject;
    let registration = registration(name);
    let package = registration.launch.clone();
    let plan: ExecPlan = adopt_launch_package_with(package.clone(), Boundaries::CUDA)
        .unwrap_or_else(|error| panic!("{name} does not adopt: {error}"));
    assert_eq!(
        plan.needs_logits,
        rows != 0,
        "{name}: the subject table and the package disagree about whether this \
         program reads the readout"
    );
    assert_eq!(
        plan.needs_mtp_logits,
        draft_rows != 0,
        "{name}: the subject table and the package disagree about whether this \
         program reads the draft column"
    );

    // The stated row count, checked against the package's own concrete shape:
    // a plan's shapes are symbolic and a package's are not, and the number
    // that reconciles them cannot be guessed.
    // EACH COLUMN AGAINST ITS OWN NUMBER. `mtp_verify_tail` reads two
    // intrinsics with different heights — `[K+1, V]` of trunk logits and
    // `[K, V]` of drafts — so a single check against `rows` would have to be
    // wrong about one of them.
    for value in &package.values {
        if value.source != ValueSource::Intrinsic {
            continue;
        }
        let (want, which) = match value.intrinsic {
            Some(IntrinsicId::MtpLogits) => (draft_rows, "draft column"),
            _ => (rows, "readout"),
        };
        assert_eq!(
            value.shape,
            vec![want, VOCAB],
            "{name}: the {which} this program declares is not {want}x{VOCAB}"
        );
    }

    let seeds = seeds(&package);
    let mut interp: InterpInstance = make_host_instance(&plan, &BTreeMap::new(), &seeds.host);

    let program = plane
        .register(context, &registration)
        .unwrap_or_else(|error| panic!("{name} does not compile: {error}"));
    let extents = Extents {
        sampled_rows: rows.max(1),
        ..Extents::default()
    };
    let instance = plane
        .bind(program, &seeds.device, extents, GeometryClass::Host, &[], &[])
        .unwrap_or_else(|error| panic!("{name} does not bind: {error}"));

    // The readout, resident once and pointed at once — the shape the runtime's
    // attachment will take, driven here so the seam is exercised rather than
    // merely present.
    let readout = logits(rows, name);
    let _resident = if rows == 0 {
        None
    } else {
        let mut buffer = Buffer::zeroed(readout.len() * size_of::<f32>())
            .unwrap_or_else(|error| panic!("{name}: the readout does not fit: {error}"));
        let bytes: Vec<u8> = readout
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        buffer
            .write(0, &bytes)
            .unwrap_or_else(|error| panic!("{name}: staging the readout: {error}"));
        plane
            .bind_intrinsic(
                instance,
                IntrinsicId::Logits,
                buffer.ptr(),
                engine_cuda::program::launch::INTRINSIC_STORAGE_F32,
                VOCAB,
                VOCAB,
                0,
            )
            .unwrap_or_else(|error| panic!("{name}: binding the readout: {error}"));
        Some(buffer)
    };
    // THE DRAFT COLUMN, RESIDENT ONCE AND POINTED AT ONCE — the same shape,
    // the same staging, a different rectangle and a different side-table slot.
    // Salted apart from the readout so a shell that bound the trunk's base
    // under the draft's name reads recognisably wrong numbers rather than
    // plausible ones.
    let drafts = logits(draft_rows, &format!("{name}/mtp"));
    let _draft_resident = if draft_rows == 0 {
        None
    } else {
        let mut buffer = Buffer::zeroed(drafts.len() * size_of::<f32>())
            .unwrap_or_else(|error| panic!("{name}: the draft column does not fit: {error}"));
        let bytes: Vec<u8> = drafts.iter().flat_map(|value| value.to_le_bytes()).collect();
        buffer
            .write(0, &bytes)
            .unwrap_or_else(|error| panic!("{name}: staging the draft column: {error}"));
        plane
            .bind_intrinsic(
                instance,
                IntrinsicId::MtpLogits,
                buffer.ptr(),
                engine_cuda::program::launch::INTRINSIC_STORAGE_F32,
                VOCAB,
                VOCAB,
                0,
            )
            .unwrap_or_else(|error| panic!("{name}: binding the draft column: {error}"));
        Some(buffer)
    };
    let inputs = if rows == 0 {
        PassInputs::none()
    } else {
        PassInputs {
            logits: Some(&readout),
            mtp_logits: (draft_rows != 0).then_some(drafts.as_slice()),
            rows,
            vocab: VOCAB,
            mtp_draft_row: None,
        }
    };

    // The seeds themselves are the zeroth comparison: an instance that starts
    // from different rings can only agree by luck.
    compare_rings(plane, instance, &interp, &package, name, "seed");

    let mut committed = 0usize;
    let mut blocked = 0usize;
    for round in 0..FIRES {
        // ── The same inputs, to both halves, in the same order. ──
        //
        // Every channel a caller can write: the host-visible writers through
        // `host_put`, and the private seeded ones through the ring's own
        // `push`, which is the door `make_host_instance` seeds them by. A
        // program like `structured_masks` consumes its three inputs and puts
        // none of them back, so without the refill it would run once and block
        // forever — seven fires of nothing compared.
        //
        // A refill the ring has no room for is REFUSED, and refused on both
        // sides: `counter_pingpong`'s loop-carried channel is full of what the
        // last fire published, so every round after the first declines here,
        // and the two halves have to decline together.
        for (index, declared) in package.channels.iter().enumerate() {
            let writable = host_writes(declared) || seeded(declared);
            if !writable {
                continue;
            }
            let channel = index as u32;
            let dtype = concrete_dtype(declared.dtype);
            let lanes = numel(declared);
            let value = cell(dtype, lanes, channel, round + 1);
            let host = if host_writes(declared) {
                host_put(&interp, &plan, channel, &value) == HostOp::Ok
            } else {
                interp.channels[index].push(&value)
            };
            let device = plane
                .instance_mut(instance)
                .expect("bound")
                .publish(channel, &wire(&value, dtype, lanes))
                .unwrap_or_else(|error| panic!("{name}: publishing channel {channel}: {error}"));
            assert_eq!(
                host, device,
                "{name} fire {round}: the host {host} channel {channel} and the device \
                 {device} — back-pressure has to be the same answer on both sides"
            );
        }

        // ── One fire each. ──
        let host = step(&mut interp, &plan, &inputs);
        let device = plane
            .fire(context, instance)
            .unwrap_or_else(|error| panic!("{name} fire {round}: {error}"));
        assert!(
            same_outcome(&host, &device),
            "{name} fire {round}: the host said {host:?} and the device said {device:?}"
        );
        match &host {
            StepOutcome::Committed => committed += 1,
            StepOutcome::Blocked(_) => blocked += 1,
            StepOutcome::Faulted(why) => panic!("{name} fire {round} faulted host-side: {why}"),
        }

        compare_rings(
            plane,
            instance,
            &interp,
            &package,
            name,
            &format!("fire {round}"),
        );

        // ── Drain every reader, and compare what came out. ──
        //
        // EXCEPT ON ONE STATED ROUND, and that is the blocked-fire case: a
        // reader channel nobody drains is full at capacity one, its
        // `NEEDS_EMPTY` requirement fails, and the NEXT fire launches nothing.
        // Both halves have to block, and to block on the same channel — which
        // is the assertion, and it is one no program in the corpus produces on
        // its own.
        if round == HOLD_THE_DRAIN {
            continue;
        }
        for (index, declared) in package.channels.iter().enumerate() {
            if !host_reads(declared) {
                continue;
            }
            let channel = index as u32;
            let dtype = concrete_dtype(declared.dtype);
            let lanes = numel(declared);
            let (op, value) = host_take(&interp, &plan, channel);
            let device = plane
                .instance_mut(instance)
                .expect("bound")
                .take(channel)
                .unwrap_or_else(|error| panic!("{name}: taking channel {channel}: {error}"));
            match (op, value, device) {
                (HostOp::Ok, Some(value), Some(bytes)) => assert_eq!(
                    wire(&value, dtype, lanes),
                    bytes,
                    "{name} fire {round}: channel {channel} published different bytes"
                ),
                (HostOp::WouldBlock, None, None) => {}
                (op, value, device) => panic!(
                    "{name} fire {round}: channel {channel} — host {op:?}/{}, device {}",
                    value.is_some(),
                    device.is_some()
                ),
            }
        }
    }

    println!("  {name:<26} {committed} committed, {blocked} blocked over {FIRES} fires — {why}");
    assert!(
        committed > 0,
        "{name}: every fire blocked, so nothing was actually compared"
    );
    assert!(
        blocked > 0,
        "{name}: no fire blocked, so the readiness gate was never compared — \
         holding the drain at round {HOLD_THE_DRAIN} is supposed to fill an \
         output ring this program declares `NEEDS_EMPTY` on"
    );
    plane
        .close_instance(instance)
        .expect("the instance closes once");
}

/// Every ring, slot for slot, plus both cursors.
///
/// The wire encoding is the comparison, because it is the one both halves can
/// speak: the host's cells ARE wire bytes and the device's are native, and
/// bool is the dtype where the two differ.
fn compare_rings(
    plane: &Plane,
    instance: u64,
    interp: &InterpInstance,
    package: &LaunchPackage,
    name: &str,
    at: &str,
) {
    let session = plane.instance(instance).expect("bound");
    for (index, declared) in package.channels.iter().enumerate() {
        let channel = index as u32;
        let dtype = concrete_dtype(declared.dtype);
        let lanes = numel(declared);
        let ring = &interp.channels[index];
        let slots = ring.capacity() as u64 + 1;

        assert_eq!(
            ring.head(),
            session.cursor(channel).expect("carried").head,
            "{name} {at}: channel {channel}'s heads diverged"
        );
        assert_eq!(
            ring.tail(),
            session.cursor(channel).expect("carried").tail,
            "{name} {at}: channel {channel}'s tails diverged"
        );

        for slot in 0..slots {
            let host = wire(&ring.decode_sequence(slot), dtype, lanes);
            let device = session
                .peek(channel, slot)
                .unwrap_or_else(|error| panic!("{name}: peeking channel {channel}: {error}"));
            assert_eq!(
                host, device,
                "{name} {at}: channel {channel} slot {slot} — the host holds {host:02x?} \
                 and the device holds {device:02x?}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The tests
// ─────────────────────────────────────────────────────────────────────────────

/// Skip at RUN time, saying what was missing: an `#[ignore]`d test on the one
/// box that could run it is a test nobody runs.
fn device_or_skip(what: &str) -> Option<Context> {
    if !present() {
        println!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    match Context::bind(0) {
        Ok(context) => Some(context),
        Err(error) => {
            println!("skipping {what}: device 0 does not bind ({error})");
            None
        }
    }
}

/// A scratch cubin cache of this test's own, so the run is hermetic: a shared
/// one would make "compiled nothing" true for the wrong reason.
fn scratch(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("pie-program-parity-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

#[test]
fn the_device_half_and_the_host_half_agree_byte_for_byte() {
    let _guard = serialized();
    let Some(context) = device_or_skip("program parity") else {
        return;
    };
    let mut plane = Plane::new(Disk::at(scratch("parity")));
    println!(
        "program parity — {} programs x {FIRES} fires",
        SUBJECTS.len()
    );
    for subject in SUBJECTS {
        parity(&context, &mut plane, subject);
    }
    let stats = plane.stats();
    println!(
        "  compiled {} region(s); memory hits {}, disk hits {}",
        stats.compilations, stats.memory_hits, stats.persistent_hits
    );
    assert!(
        stats.compilations > 0,
        "every program answered from a cache, so NVRTC was never exercised"
    );
}

#[test]
fn a_source_nvrtc_refuses_is_a_named_deterministic_refusal_and_nothing_else() {
    let _guard = serialized();
    let Some(context) = device_or_skip("compile taxonomy") else {
        return;
    };
    let mut plane = Plane::new(Disk::at(scratch("taxonomy")));

    // A real registration with one region's source corrupted. NOT a
    // hand-written stub: the failure has to arrive through the same door a
    // real program takes, and what makes it deterministic is the source rather
    // than the shape of the table around it.
    // A DIFFERENT program from the good one below, and deliberately so: the
    // negative tier is keyed by `cache_identity` over the PLAN, not over the
    // emitted source (a source is a pure function of plan and emitter
    // version, so in production the two cannot differ). Corrupting a
    // program's source and then registering the same plan intact would be
    // answered from that tier — correctly, and uselessly for this test.
    let mut broken = registration("structured_masks");
    broken.program_hash ^= 0xdead_beef;
    broken.emitted_kernels[0]
        .source
        .push_str("\n// this line is not CUDA:\n???\n");

    let refusal = plane
        .register(&context, &broken)
        .expect_err("NVRTC cannot compile `???`");
    let text = format!("{refusal}");
    assert!(
        matches!(
            refusal,
            engine_cuda::Fault::Compile(engine::Failure::Deterministic { .. })
        ),
        "a source NVRTC rejects is rejected forever, so it must be Deterministic: {text}"
    );
    assert!(
        text.contains("deterministic"),
        "the refusal must name its own taxonomy: {text}"
    );

    // Remembered, and remembered as this exact refusal: the second attempt
    // must answer from the negative tier without touching NVRTC.
    let before = plane.stats();
    let again = plane
        .register(&context, &broken)
        .expect_err("still refused");
    let after = plane.stats();
    assert_eq!(
        format!("{again}"),
        text,
        "the remembered refusal must be the one that was made"
    );
    assert_eq!(
        after.compilations, before.compilations,
        "a deterministic refusal is remembered; re-running NVRTC on it is what \
         makes a hot-looping guest a compile server"
    );
    assert_eq!(
        after.negative_hits,
        before.negative_hits + 1,
        "and the tier that answered has to say so"
    );

    // NO POISONED PROCESS. The plane that just refused a program compiles and
    // runs the next one — which is the whole difference between a refusal and
    // a crash.
    let good = registration("counter_pingpong");
    let program = plane
        .register(&context, &good)
        .expect("a good program still registers after a bad one");
    let instance = plane
        .bind(program, &seeds(&good.launch).device, Extents::default(), GeometryClass::Host, &[], &[])
        .expect("and binds");
    assert_eq!(
        plane.fire(&context, instance).expect("and fires"),
        Fired::Committed,
        "and commits"
    );
}

#[test]
fn the_second_bind_of_a_program_compiles_nothing() {
    let _guard = serialized();
    let Some(context) = device_or_skip("module cache") else {
        return;
    };
    let disk = scratch("cache");
    // A program that reads no intrinsic: this test fires without binding one,
    // and a program that wanted the readout would be refused by name for
    // exactly that reason (`Session::fire`'s unbound-intrinsic check).
    let registration = registration("counter_pingpong");
    let device_seeds = seeds(&registration.launch).device;

    // ── Tier one: the same plane, the same hash. ──
    let mut plane = Plane::new(Disk::at(&disk));
    let first = plane
        .register(&context, &registration)
        .expect("the first registration compiles");
    let compiled = plane.stats().compilations;
    assert!(compiled > 0, "the first registration must reach NVRTC");

    let again = plane
        .register(&context, &registration)
        .expect("the second registration is the same program");
    assert_eq!(first, again, "one program, one id");
    assert_eq!(
        plane.stats().compilations,
        compiled,
        "a re-registration of a live program compiles nothing — and an absence \
         has no output, so this counter is the only way to say so"
    );

    // Two instances of one program share its modules; binding is not compiling.
    let a = plane
        .bind(first, &device_seeds, Extents::default(), GeometryClass::Host, &[], &[])
        .expect("first instance");
    let b = plane
        .bind(first, &device_seeds, Extents::default(), GeometryClass::Host, &[], &[])
        .expect("second instance");
    assert_ne!(a, b, "two instances, two ids");
    assert_eq!(
        plane.stats().compilations,
        compiled,
        "binding an instance compiles nothing at all"
    );
    assert_eq!(
        plane.fire(&context, a).expect("fires"),
        Fired::Committed,
        "and both instances run"
    );
    assert_eq!(plane.fire(&context, b).expect("fires"), Fired::Committed);

    // A program with instances still bound may not be closed: unloading a
    // `CUmodule` under a live launch is a jump into freed machine code.
    plane
        .close_program(first)
        .expect_err("two instances are still bound");
    plane.close_instance(a).expect("closes");
    plane.close_instance(b).expect("closes");
    plane.close_program(first).expect("now it closes");

    // ── Tier two: a NEW plane over the same disk. ──
    //
    // The memory tier is gone with the old plane, so anything that does not
    // reach NVRTC here came off disk, keyed by the identity plus the source's
    // own fingerprint.
    let mut second = Plane::new(Disk::at(&disk));
    second
        .register(&context, &registration)
        .expect("a fresh plane registers the same program");
    let stats = second.stats();
    assert_eq!(
        stats.compilations, 0,
        "a fresh plane over a warm disk cache must compile nothing"
    );
    assert_eq!(
        stats.persistent_hits, compiled,
        "and every region it did not compile has to have come off disk"
    );
}
