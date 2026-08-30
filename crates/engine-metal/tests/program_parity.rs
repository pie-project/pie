//! The guest-program plane, host half against device half, byte for byte —
//! the Metal side.
//!
//! **THIS IS THE POINT OF STEP 7.** Every other test in this crate can pass
//! while an ETA fire computes the wrong thing: the launch succeeded, the
//! commit slot survived, a channel holds *a* value. What cannot be faked is
//! agreement with `eta_exec` — a complete interpreter of the same
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
//! which is why the device compiles with `MTLMathMode::Safe` and precise
//! floating-point functions — the Metal compiler contracts multiplies into
//! fused multiply-adds by DEFAULT and the host interpreter has no FMA at all
//! — and why "close enough" is not an option anywhere below.
//!
//! **THE SUBJECTS ARE REAL GOLDEN TRACES**, not fixtures written here: the
//! `eta-compiler` corpus under `tests/golden/`, decoded from the
//! `container:` line, bound against the profile each was authored for, planned
//! by `compile_bound`, and emitted for Metal by `emit_program`. That is
//! exactly the path a registration takes in production, so what this test
//! exercises is the path and not a mock of it. The five chosen are the
//! corpus's programs whose two halves must agree EXACTLY rather than nearly:
//! every op integral or boolean, no RNG to seed, and where a `logits` row is
//! read it is a buffer this test resident-ises itself and points BOTH halves
//! at — which is also the only exercise the runtime's attachment seam gets
//! until the runtime builds it.
//!
//! **THERE IS NO DRAFT-COLUMN SUBJECT HERE, AND THE REASON IS THE EMITTED
//! ABI RATHER THAN THE CORPUS.** The CUDA sibling carries a sixth subject,
//! `mtp_two_columns`: one epilogue holding two `IntrinsicVal`s of different
//! heights — `Logits [K+1, V]` and `MtpLogits [K, V]` — each argmaxed and
//! published on its own channel, which is the shape a draft-reading guest
//! has. It is unreachable on this plane. The Metal M2 emitter
//! (`eta_compiler::codegen::metal::fused::emit_fused_region`) writes ONE
//! intrinsic parameter into the kernel signature —
//! `const device uchar* logits [[buffer(6)]]` — and sets it as the first
//! argument of EVERY `INTRINSIC_VAL` op it lowers. So the two intrinsics do
//! not merely share a table pitch, they share the argument: a stage cannot be
//! pointed at two rectangles at once, and `Prepared::bind_intrinsic` refuses
//! the second binding by name rather than letting the last one to arrive
//! silently move the first. A subject that reads both columns is therefore
//! NOT UNTESTED, IT IS UNREACHABLE, and writing it here would assert a
//! property of a kernel nobody emits.
//!
//! The shape that fixes it is the M3 grouped form, which is already emitted
//! and not yet driven: there the intrinsic is not a bound buffer at all but a
//! raw address per lane (`lane.logits_base`), and the draft column is that
//! same address plus a per-row displacement the side table states
//! (`M3RowMeta::mtp_offset`) — two rectangles, one binding, exactly the split
//! the CUDA side gets from its five side tables. Reaching it needs
//! `MTLBuffer.gpuAddress` plumbed through `device/alloc.rs`, because the
//! grouped kernels dereference raw `ulong`s rather than encoder bindings, and
//! that is not this wave's. Until it is, the draft column has no device half
//! on Metal and this file states so instead of pretending otherwise: every
//! subject below asserts `!plan.needs_mtp_logits`.
//!
//! **THE READOUT IS STAGED AS bf16, WHICH IS WHY THE HOST IS FED THE WIDENED
//! NUMBERS AND NOT THE ORIGINALS.** The M1 runtime reads the intrinsic buffer
//! as `bfloat` — `ptir_m1_runtime.metal`, the `0xA0` arm, whose first line
//! reinterprets `a0` as `const device bfloat*` — and there is no storage-mode
//! word anywhere on this plane to say otherwise; the CUDA engine's
//! `INTRINSIC_STORAGE_F32` has no counterpart here and must not be reached
//! for. So [`logits`] produces f32 values as it always did, each is staged as
//! its bf16 TRUNCATION (the top sixteen bits of the f32, little-endian), and
//! what the HOST interpreter is handed is those truncations WIDENED BACK
//! (`f32::from_bits((bits >> 16) << 16)`) rather than the values they came
//! from. **THE WIDENING IS EXACT — every bf16 is an f32 — SO THE TWO HALVES
//! SEE THE SAME NUMBERS, AND THAT IS THE WHOLE REASON THE DIFF MEANS
//! ANYTHING.** Handing the host the untruncated values instead would leave
//! every comparison one rounding apart from the device's and turn a real
//! disagreement into an argument about the eighth mantissa bit.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-metal --test program_parity -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine::program::ProgramRegistration;
use engine_metal::device::{Buffer, Context, present};
use engine_metal::program::{Fired, Plane};
use eta_compiler::codegen::launch::{LaunchChannel, LaunchPackage, ValueOrigin};
use eta_compiler::codegen::program::KernelKind;
use eta_exec::{
    ExecPlan, Extents, HostOp, InterpInstance, PassInputs, StepOutcome, Value,
    adopt_launch_package, concrete_dtype, encode_wire, host_put, host_take, make_host_instance,
    step, wire_cell_bytes,
};
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::container::TraceContainer;
use eta_ir::op::IntrinsicId;
use eta_ir::registry::{GeometryClass, KernelInfo, ModelProfile};

/// **ONE PLANE AT A TIME, PER PROCESS — AND NOT FOR THE CUDA SIBLING'S
/// REASON.** That file serializes because `kernels-cuda`'s scratch slabs are
/// process-global and because two contexts racing for device 0 is a flake
/// nobody can read. Neither is true here: `kernels-metal` allocates nothing
/// and keeps no process-global scratch, an `MTLDevice` is an object rather
/// than per-thread state, and two Metal planes firing at once would be
/// CORRECT. What is not correct is reading the measurements — the compile
/// counters below are per-plane but the Metal compiler, the resident bytes
/// and the one GPU are not, and a `compiled 0 region(s)` line printed while a
/// neighbour thread is holding the compiler is a sentence about the machine
/// rather than about this test.
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
    ///
    /// **THERE IS NO SECOND NUMBER HERE**, where the CUDA sibling carries a
    /// `draft_rows` beside it. A draft column is a second rectangle, and this
    /// plane's emitted kernels have one intrinsic buffer between them — see
    /// the module doc. The absence is asserted rather than assumed: every
    /// subject's plan must declare `needs_mtp_logits == false`.
    rows: u32,
    /// Why this program is in the list.
    why: &'static str,
}

/// The programs both halves run.
///
/// Every one is integral where it matters: the two halves have to agree on the
/// bit, and an `exp` in the middle would turn a real disagreement into an
/// argument about the last mantissa bit of somebody's libm. The three that
/// read `logits` are handed the same bf16 rectangle on both sides — exact
/// binary fractions, all distinct, so an argmax has no tie to break
/// differently.
const SUBJECTS: &[Subject] = &[
    // A loop-carried counter: one channel taken AND put in the same fire,
    // which is the shape every decode loop has and the one where a commit
    // that consumed before it published leaves the wrong value behind.
    Subject {
        name: "counter_pingpong",
        rows: 0,
        why: "take, add, put back into the same ring",
    },
    // Reads that peek without consuming beside a take that does, plus a
    // gather, a select and an argmax over a bound `logits` row — the case
    // where conflating `chan_read` with `chan_take` drops a cell per fire.
    Subject {
        name: "dfa_ingraph",
        rows: 1,
        why: "two peeked channels, one taken, two published",
    },
    // A bool matrix in, an i32 selection out: the bool cell is the one dtype
    // whose device and wire spellings differ on the CUDA plane — here the
    // runtime packs and unpacks bits on the device, so this is the case that
    // proves the two encodings meet in the same bytes anyway.
    Subject {
        name: "matrix_select_mask",
        rows: 4,
        why: "a packed bool matrix selects rows of the logits",
    },
    // Seven channels, four of them bool masks — causal, sliding-window,
    // sink-window and a packed apply. The widest single region in the corpus
    // that reads no intrinsic at all.
    Subject {
        name: "structured_masks",
        rows: 0,
        why: "three u32 inputs, four bool mask outputs",
    },
    // No channel inputs at all, one output at capacity one: the second fire
    // has nowhere to publish, so this is the blocked-fire case, and both
    // halves must block on the same channel.
    Subject {
        name: "matrix_mask_apply_packed",
        rows: 2,
        why: "no channel inputs; blocks on its own full output",
    },
];

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
        "/../eta-compiler/tests/golden"
    ))
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex"))
        .collect()
}

/// The bind-time profile each golden was authored against, transcribed from
/// `eta-compiler`'s own corpus helper. The goldens do not carry it, and
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

/// One subject's trace, out of the corpus.
///
/// Every subject on this plane is the corpus's; the CUDA sibling's one
/// hand-built container was the draft-column subject, and the module doc says
/// why it is not here.
fn container_of(name: &str) -> TraceContainer {
    let path = golden_dir().join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path:?} is missing"));
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("container: "))
        .unwrap_or_else(|| panic!("{name} has no container line"));
    eta_ir::container::decode(&unhex(line))
        .unwrap_or_else(|why| panic!("{name} does not decode: {why:?}"))
}

/// One trace, all the way to what `register_program` takes.
fn registration(name: &str) -> ProgramRegistration {
    let container = container_of(name);
    let bound = eta_ir::validate::bind(container, golden_profile(name))
        .unwrap_or_else(|why| panic!("{name} does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Metal;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);

    ProgramRegistration {
        // The bound trace's own hash, which is what the host uses: two
        // registrations of one program have to be recognised as one.
        program_hash: bound.hash,
        emitted_kernels: emitted,
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
fn cell(dtype: Dtype, lanes: usize, channel: u32, round: usize) -> Value {
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
        Dtype::U32 => Value::U32((0..lanes).map(|_| next() % 7).collect()),
        Dtype::I32 => Value::I32((0..lanes).map(|_| (next() % 7) as i32).collect()),
        Dtype::Bool => Value::Bool((0..lanes).map(|_| u8::from(next() % 2 == 1)).collect()),
        // Small exact binary fractions: representable to the bit, so an f32
        // channel would still be comparable byte for byte.
        Dtype::F32 => Value::F32((0..lanes).map(|_| (next() % 16) as f32 / 8.0).collect()),
        // A channel carries one of the four dtypes ETA computes in —
        // `eta_exec::Value`'s own arms, and no more — so anything else names
        // no lane on either half of this diff and there is no cell to hand
        // them. Stated as a refusal rather than as a zero: a subject whose
        // channel is a dtype the interpreter cannot hold would otherwise be
        // compared against a fabrication.
        other => panic!(
            "{other:?} is not a dtype a channel carries, so this diff has no cell for it"
        ),
    }
}

fn wire(value: &Value, dtype: Dtype, lanes: usize) -> Vec<u8> {
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

/// One value as the sixteen bits the device will actually hold.
///
/// **TRUNCATION, NOT ROUNDING.** The staging is this test's own — no shipping
/// path narrows an f32 readout here, the model fire produces `bfloat` rows
/// already — so the rule only has to be one both halves can be handed, and a
/// truncation is the one whose inverse is exact and stateless.
fn to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// The same value as the runtime will read it back: bf16 widened to f32.
///
/// **THIS IS WHAT THE HOST INTERPRETER IS FED**, and it is the load-bearing
/// line of the whole staging. The device's `0xA0` arm loads a `bfloat` and
/// promotes it to `float`; the promotion loses nothing, so a host handed this
/// number and a device handed those two bytes are working on the SAME value,
/// and every byte-for-byte assertion downstream is about the arithmetic rather
/// than about a rounding neither half agreed to.
fn widen(value: f32) -> f32 {
    f32::from_bits((value.to_bits() >> 16) << 16)
}

/// Run `subject` on both halves for [`FIRES`] fires and assert they never
/// differ.
fn parity(context: &Context, plane: &mut Plane, subject: &Subject) {
    let Subject { name, rows, why } = *subject;
    let registration = registration(name);
    let package = registration.launch.clone();
    // **THE DEFAULT BOUNDARIES, WHERE THE CUDA SIBLING OVERRIDES.** That file
    // calls `adopt_launch_package_with(package, Boundaries::CUDA)` because the
    // CUDA vocabulary — `envelope_dot`, `lora`, `attn_page_mask` — is not the
    // default. This plane's vocabulary IS the default: `adopt_launch_package`
    // applies `Boundaries::METAL` (`metal.identity`, `metal.discard`), which
    // is exactly what `Plane::register` adopts with. Spelling it out with
    // `_with` here would restate the default and then drift from it the day
    // the Metal vocabulary grows a name — and the test would be adopting a
    // different plan from the plane it is diffing.
    let plan: ExecPlan = adopt_launch_package(package.clone())
        .unwrap_or_else(|error| panic!("{name} does not adopt: {error}"));
    // Adoption answers `Ok` for a package this backend cannot run and says so
    // in the plan instead, because the runtime wants the sentence rather than a
    // refusal. A subject that reached that arm would fire zero regions and
    // agree with the host about nothing.
    assert!(
        plan.executable,
        "{name}: this backend declines the program — {}",
        plan.reject_reason.unwrap_or_default()
    );
    assert_eq!(
        plan.needs_logits,
        rows != 0,
        "{name}: the subject table and the package disagree about whether this \
         program reads the readout"
    );
    // THE DRAFT COLUMN HAS NO DEVICE HALF ON THIS PLANE, so a subject that
    // declared one would be pointed at the trunk's rectangle by the emitted
    // kernel's single `logits [[buffer(6)]]` and would diff a different
    // program from the one the host ran. See the module doc.
    assert!(
        !plan.needs_mtp_logits,
        "{name}: this program reads the draft column, and the M2 emitter binds \
         one intrinsic buffer for every `INTRINSIC_VAL` op — the subject is \
         unreachable here rather than merely untested"
    );

    // The stated row count, checked against the package's own concrete shape:
    // a plan's shapes are symbolic and a package's are not, and the number
    // that reconciles them cannot be guessed.
    for value in &package.values {
        if value.source != ValueOrigin::Intrinsic {
            continue;
        }
        assert_ne!(
            value.intrinsic,
            Some(IntrinsicId::MtpLogits),
            "{name}: a draft column reached the value table past the plan check"
        );
        assert_eq!(
            value.shape,
            vec![rows, VOCAB],
            "{name}: the readout this program declares is not {rows}x{VOCAB}"
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
        .bind(
            context,
            program,
            &seeds.device,
            extents,
            GeometryClass::Host,
        )
        .unwrap_or_else(|error| panic!("{name} does not bind: {error}"));

    // The readout, resident once and pointed at once — the shape the runtime's
    // attachment will take, driven here so the seam is exercised rather than
    // merely present.
    //
    // STAGED AS bf16 AND READ BACK WIDENED. `readout` is what the numbers are;
    // `seen` is what both halves actually work on. The device gets the two
    // bytes, the host gets the widening of those same two bytes, and the
    // widening is exact — so the diff below is about the emitted arithmetic
    // and nothing else.
    let readout = logits(rows, name);
    let seen: Vec<f32> = readout.iter().copied().map(widen).collect();
    let _resident = if rows == 0 {
        None
    } else {
        let bytes: Vec<u8> = readout
            .iter()
            .map(|value| to_bf16(*value))
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut buffer = Buffer::zeroed(context, bytes.len() as u64)
            .unwrap_or_else(|error| panic!("{name}: the readout does not fit: {error}"));
        buffer
            .write(0, &bytes)
            .unwrap_or_else(|error| panic!("{name}: staging the readout: {error}"));
        // A BUFFER AND A BYTE OFFSET, WHERE THE CUDA TWIN TAKES AN ADDRESS AND
        // FIVE WORDS. Metal binds an object: the row offset the CUDA side
        // writes into a side table is the encoder's own
        // `setBuffer:offset:atIndex:`, and the rows begin at the base, so it
        // is zero. The width and the dtype travel because the slot table
        // ARGUES with them — `Prepared::bind_intrinsic` holds them against the
        // row width the program's own shapes resolved to — and this staging
        // is `VOCAB`-wide `bfloat`, which is what those shapes say.
        plane
            .bind_intrinsic(instance, IntrinsicId::Logits, &buffer, 0, VOCAB, Dtype::Bf16)
            .unwrap_or_else(|error| panic!("{name}: binding the readout: {error}"));
        Some(buffer)
    };
    let inputs = if rows == 0 {
        PassInputs::none()
    } else {
        PassInputs {
            logits: Some(&seen),
            // No draft column: this plane binds ONE intrinsic buffer, so a
            // subject reading both is unreachable rather than untested (the
            // module doc argues it).
            mtp_logits: None,
            rows,
            vocab: VOCAB,
            mtp_draft_row: None,
            // No score rectangle either, and for the same reason one buffer
            // deep: the score plane is not logits-shaped, so it would need a
            // second rectangle this plane has no index for, and
            // `program::session` refuses `needs_attn_scores` outright. A
            // subject that observed attention could not be seated here at
            // all, so this is unreachable rather than untested.
            attn_score: None,
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
/// speak. On the CUDA plane that is a conversion — its bool rings hold one
/// byte per lane where the wire packs eight — and here it is an identity: the
/// Metal runtime packs and unpacks bits on the device, so a channel cell IS a
/// wire cell for every dtype. The comparison is spelled the same way anyway,
/// because it is the same claim and the two files have to be readable against
/// each other.
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

/// Skip at RUN time, saying which precondition was missing: an `#[ignore]`d
/// test on the one box that could run it is a test nobody runs.
///
/// Two doors rather than the CUDA sibling's device ordinal. An Apple TARGET is
/// not a machine with a GPU — a headless build box publishes no device — and a
/// machine that publishes one may still be refused at bind, because
/// [`Context::bind`] asserts unified memory (this shell writes its buffers
/// through `contents()` and a discrete device would hand back a stale
/// mapping).
fn device_or_skip(what: &str) -> Option<Context> {
    if !present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    match Context::bind() {
        Ok(context) => Some(context),
        Err(error) => {
            println!("SKIP {what}: the device does not bind ({error})");
            None
        }
    }
}

#[test]
fn the_device_half_and_the_host_half_agree_byte_for_byte() {
    let _guard = serialized();
    let Some(context) = device_or_skip("program parity") else {
        return;
    };
    let mut plane = Plane::new();
    println!(
        "program parity — {} programs x {FIRES} fires",
        SUBJECTS.len()
    );
    for subject in SUBJECTS {
        parity(&context, &mut plane, subject);
    }
    let stats = plane.stats();
    println!(
        "  compiled {} region(s); memory hits {}",
        stats.compilations, stats.memory_hits
    );
    assert!(
        stats.compilations > 0,
        "every program answered from a cache, so the Metal compiler was never exercised"
    );
}

#[test]
fn a_source_metal_refuses_is_a_named_deterministic_refusal_and_nothing_else() {
    let _guard = serialized();
    let Some(context) = device_or_skip("compile taxonomy") else {
        return;
    };
    let mut plane = Plane::new();

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
    //
    // THE CORRUPTION IS UNAMBIGUOUS ON PURPOSE. The CUDA sibling appends
    // `???`, which NVRTC certainly rejects; a stray `@@@` line is the same
    // idea for a Metal compiler that is clang — three at-signs at file scope
    // are not a declaration in any dialect of MSL, so there is no reading of
    // the source under which this compiles and the refusal is the source's
    // forever.
    let mut broken = registration("structured_masks");
    broken.program_hash ^= 0xdead_beef;
    // **CORRUPT THE FUSED KERNEL, NOT KERNEL ZERO**, and the difference is
    // this plane's emitter rather than a detail. `emit_metal_stage` writes
    // FIVE kinds into one table — singleton, fused, grouped, readiness,
    // commit — where `emit_cuda_stage` writes only fused, so index 0 is a
    // singleton kernel and this shell compiles exactly one of the five
    // (`compile::KERNEL_FUSED`). Breaking index 0 broke a kernel nobody
    // compiles, and the registration succeeded: measured, and the reason
    // this loop exists.
    let fused = broken
        .emitted_kernels
        .iter()
        .position(|kernel| kernel.kind == KernelKind::Fused)
        .expect("the metal emitter writes a fused kernel for this subject");
    broken.emitted_kernels[fused]
        .source
        .push_str("\n// this line is not MSL:\n@@@ not msl @@@\n");

    let refusal = plane
        .register(&context, &broken)
        .expect_err("the Metal compiler cannot compile `@@@ not msl @@@`");
    let text = format!("{refusal}");
    // The taxonomy is REACHABLE here and is asserted, not softened:
    // `engine_metal::Fault::Compile`'s own doc states that the split comes off
    // `MTLLibraryError` — a compile failure is the source's and anything else
    // is the moment's — so a source clang rejects is `Deterministic` by
    // construction.
    assert!(
        matches!(
            refusal,
            engine_metal::Fault::Compile(eta_exec::Failure::Deterministic { .. })
        ),
        "a source the Metal compiler rejects is rejected forever, so it must be \
         Deterministic: {text}"
    );
    // **THE TAXONOMY IS READ OFF THE TYPE HERE, WHERE THE CUDA SIBLING READS
    // IT OFF THE SENTENCE**, and that is a difference in the two `Display`s
    // rather than in the two claims. `engine_cuda::Fault::Compile` prints
    // "(deterministic, remembered)" and its test asserts the word; this
    // plane's prints "the guest program does not compile: {reason}" and leaves
    // the class to the variant. The `matches!` above IS the stronger of the
    // two checks — it reads the thing the cache keys on rather than a
    // substring of a formatter — so what the text is asked for is the other
    // half of the contract: that the compiler's OWN words survive into it,
    // because a refusal nobody can read is a refusal nobody can fix.
    assert!(
        text.contains("does not compile"),
        "the refusal must say what it is: {text}"
    );
    assert!(
        text.len() > "the guest program does not compile: ".len(),
        "the Metal compiler's own diagnostic has to reach the sentence: {text}"
    );

    // Remembered, and remembered as this exact refusal: the second attempt
    // must answer from the negative tier without opening a library.
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
        "a deterministic refusal is remembered; re-running the Metal compiler on it \
         is what makes a hot-looping guest a compile server"
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
        .bind(
            &context,
            program,
            &seeds(&good.launch).device,
            Extents::default(),
            GeometryClass::Host,
        )
        .expect("and binds");
    assert_eq!(
        plane
            .fire(&context, instance)
            .expect("and fires"),
        Fired::Committed,
        "and commits"
    );
}

/// The memory tier, and only the memory tier.
///
/// **THERE IS NO SECOND HALF TO THIS TEST ON THIS PLANE, AND THAT IS A
/// PLATFORM FACT RATHER THAN AN OMISSION.** The CUDA sibling runs the same
/// claim twice: once against a live plane (a re-registration compiles
/// nothing), and once against a NEW plane over the same directory (a fresh
/// process compiles nothing either, because the cubin was on disk). The
/// second half has no counterpart here. `newLibraryWithSource:options:error:`
/// answers a live `MTLLibrary` — an object, refcounted by ARC — and no
/// serializable image: there is no cubin to write, no file to key by source
/// fingerprint, and therefore no `Disk` on this plane at all
/// ([`Plane::new`] takes no cache directory where the CUDA one takes a
/// `Disk`).
///
/// What a persistent tier WOULD have to be is `MTLBinaryArchive`, and the
/// shape of that is why it is not this wave's. An archive stores compiled
/// PIPELINE STATES, not a reloadable library: it is keyed to the device and
/// the driver build that produced it, it is opened as a hint rather than as a
/// source of truth (a miss silently recompiles, so "compiled nothing" stops
/// being observable from the counter this test asserts on), and every
/// descriptor that goes into it has to be reconstructed before it can be
/// looked up. That is a design with its own refusal taxonomy and its own
/// staleness rules, and it earns those only once cold-start compile time is
/// something someone has measured and minded.
#[test]
fn the_second_bind_of_a_program_compiles_nothing() {
    let _guard = serialized();
    let Some(context) = device_or_skip("module cache") else {
        return;
    };
    // A program that reads no intrinsic: this test fires without binding one,
    // and a program that wanted the readout would be refused by name for
    // exactly that reason (`Session::fire`'s unbound-intrinsic check, which on
    // this plane stands between the caller and a command buffer that dies with
    // `MTLCommandBufferErrorPageFault` and takes the queue with it).
    let registration = registration("counter_pingpong");
    let device_seeds = seeds(&registration.launch).device;

    // ── The same plane, the same hash. ──
    let mut plane = Plane::new();
    let first = plane
        .register(&context, &registration)
        .expect("the first registration compiles");
    let compiled = plane.stats().compilations;
    assert!(
        compiled > 0,
        "the first registration must reach the Metal compiler"
    );

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

    // Two instances of one program share its pipelines; binding is not
    // compiling.
    let a = plane
        .bind(
            &context,
            first,
            &device_seeds,
            Extents::default(),
            GeometryClass::Host,
        )
        .expect("first instance");
    let b = plane
        .bind(
            &context,
            first,
            &device_seeds,
            Extents::default(),
            GeometryClass::Host,
        )
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
    assert_eq!(
        plane.fire(&context, b).expect("fires"),
        Fired::Committed
    );

    // A program with instances still bound may not be closed. The refusal is
    // ADVISORY here and load-bearing on the CUDA plane: closing a CUDA program
    // calls `cuModuleUnload`, so a live instance's next launch enters freed
    // machine code, while an `MTLLibrary` and its pipelines are refcounted and
    // a `Session` holding its own `Compiled` keeps firing correctly after its
    // program is forgotten. It is asserted anyway, and for the reason
    // `Plane::close_program` states: a caller that closes a program with
    // instances still bound has lost track of its own instances, and answering
    // `Ok` would trade a nameable refusal for a leak nobody is looking for.
    plane
        .close_program(first)
        .expect_err("two instances are still bound");
    plane.close_instance(a).expect("closes");
    plane.close_instance(b).expect("closes");
    plane.close_program(first).expect("now it closes");
}
