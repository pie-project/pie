//! **A GGUF K-QUANT WEIGHT SERVES AS STORED, END TO END** (QNF wave, alto
//! `next.md` §J5) — the connective tissue between a model text that declares
//! the stored super-block variants and `kernels_cuda::linear::kquant`, held against the
//! tree's own host decode of the same bytes.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_stored_k_quant_row_serves_as_stored -- --nocapture
//! ```
//!
//! # What is under test, and what already was
//!
//! The kernels are golden (`kernels-cuda/tests/kquant_matmul.rs`, all five
//! schemes against a transcribed block decode). What had never run is
//! everything between a declaration and a launch:
//!
//! ```text
//! Dtype::U4g32k   the text's word
//!   -> Weight::planes            one plane, [n, Dtype::row_bytes(k)] BYTES
//!   -> checkpoint_dsl::encoding  -> qnf::scheme_of_sig -> GgufQ4K
//!   -> the ladder's identity rung (stored == wanted: no cast, no decode)
//!   -> weights::plane_bytes      the rectangle IS the byte count
//!   -> WeightRow::Dense at the DECLARED dtype
//!   -> Run::maybe_stored         re-badged as the U8 byte rectangle
//!   -> linear::kquant::{matmul, lm_head}
//! ```
//!
//! # THE ORACLE IS THE SAME FILE, READ THE OTHER WAY
//!
//! One checkpoint, two texts. The stored text declares its projection `q4_k`
//! and its head `q6_k` and serves them as they lie; the reference text
//! declares both `Bf16` over the SAME tensors, and the checkpoint's ladder
//! takes its `Quant -> Raw` rung — the dequant door — decoding each block on
//! the host at load and handing cuBLAS a dense rectangle.
//!
//! That is what makes this a gate rather than a self-comparison. The
//! reference decode is `checkpoint::executor::walk`'s, which carries its own
//! bit-identity evidence against the `gguf` package; nothing in this file
//! decodes anything, so a wrong super-block width, a wrong plane rectangle or
//! a weight bound at the wrong address fails here and cannot be papered over
//! by a matching mistake in an oracle written beside it.
//!
//! **THE SAME NUMBERS, NOT THE SAME BITS.** The two arms round differently on
//! purpose: the host decode lands each weight in BF16 (`logical_dtype`) and
//! the fused point decodes in f32 inside the dot, so the honest comparison is
//! against the logit row's own spread — §J4a-1's ruling, where a 13.6%
//! element turned out to be a cancellation point measured with the wrong
//! ruler.
//!
//! # Gating
//!
//! Skipped at run time with no device, as `routed_experts_stream.rs` is, and
//! for its reason: an `#[ignore]` on the one box that could run it is a test
//! nobody runs. Nothing here reads a checkpoint off disk — the container is
//! written from the trace's own params into a scratch directory.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::Trace;
use ztensor::format::cbor;

// ─────────────────────────────────────────────────────────────────────────
// The text
// ─────────────────────────────────────────────────────────────────────────

/// Rows of the head, and of the embedding table.
const VOCAB: u32 = 1024;

/// The contracted axis of both projections — two whole ggml super-blocks, so
/// every scheme's row width is `2 x its block` and the `k % 256` refusal in
/// `linear::kquant` is not what this gate measures.
const HIDDEN: u64 = 512;

/// `q4_k`, spelled. The mangled form is the format's only name and `sig` is a
/// `const fn`, so a typo here does not compile.
const Q4_K: Dtype = Dtype::U4g32k;

/// `q6_k` — what a Q4_K_M mix stores `output.weight` at, which is why the
/// head is the scheme's busiest consumer and why it is the head's row here.
const Q6_K: Dtype = Dtype::I6g16k;

/// The fact vocabulary a three-op trace needs: none.
struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

/// Which way the two banks are declared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `q4_k` and `q6_k`: served as the file stores them.
    Stored,
    /// `Bf16` over the same tensors: the checkpoint's dequant door decodes
    /// every block at load and cuBLAS folds the rectangle.
    Decoded,
}

/// **A MODEL NOBODY SHIPS, AND DELIBERATELY NOT A CATALOG ROW.** Every shipped
/// family declares its embedding table in the weight representation `w`, and
/// there is no embed-gather point over a braided K-quant block — the gather
/// twins are the affine family's. So a text that quantizes its PROJECTIONS
/// and leaves its table alone is not a parameter any catalog row takes, and
/// writing one here is smaller than teaching one to a family for a gate.
///
/// Three ops, which is every op this wave touched: a gather to make a token
/// rectangle, the `linear.matmul` arm, and the `linear.lm_head` arm.
struct Micro {
    embed: Weight,
    proj: Weight,
    head: Weight,
}

impl Micro {
    fn new(arm: Arm) -> Micro {
        let (proj, head) = match arm {
            Arm::Stored => (Q4_K, Q6_K),
            Arm::Decoded => (Dtype::Bf16, Dtype::Bf16),
        };
        Micro {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
            // The logical rectangle, in both arms and at both widths. What a
            // stored block costs is `Dtype::row_bytes(k)` and `Weight::planes`
            // is the one place that folding happens.
            proj: Weight::sym("proj", [HIDDEN, HIDDEN], proj),
            head: Weight::sym("lm_head", [u64::from(VOCAB), HIDDEN], head),
        }
    }

    /// The load contract, stated through the same builder a family's `load`
    /// uses. `read_own` is "this plane is stored under its own name", which is
    /// what makes the ladder's rung the only thing that differs between arms.
    fn load(&self, src: &ztensor::Source) -> ModelContract {
        let mut b = checkpoint_dsl::Builder::new(src, 1);
        for w in [&self.embed, &self.proj, &self.head] {
            b.read_own(w)
                .unwrap_or_else(|why| panic!("`{}`: {why}", w.name));
        }
        b.build()
    }
}

impl ForwardHybrid for Micro {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = ops::layout::embed(&inputs.tokens(), &self.embed, VOCAB);
        let h = ops::linear::matmul(&x, &self.proj);
        ops::linear::lm_head(&h, &self.head)
    }
}

fn trace(arm: Arm) -> (Micro, Trace) {
    let m = Micro::new(arm);
    let trace = trace_hybrid("kquant-micro", &m, Platform::Cuda);
    (m, trace)
}

// ─────────────────────────────────────────────────────────────────────────
// The checkpoint
// ─────────────────────────────────────────────────────────────────────────

/// Elements in one K-quant super-block, all five schemes.
const SUPER: usize = 256;

/// `block_q4_K`: `d` f16, `dmin` f16, twelve packed six-bit scale/min fields,
/// then 128 bytes of nibbles.
const Q4K_BYTES: usize = 144;

/// `block_q6_K`: 128 low nibbles, 64 high pairs, sixteen i8 sub-block scales,
/// `d` f16.
const Q6K_BYTES: usize = 210;

/// A seeded stream, so both loads read the same bytes and a difference in
/// their logits is a difference in the path and nothing else.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A byte.
    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }

    /// Uniform on `[-0.5, 0.5)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    /// A positive f16 in `[2^exp, 2^(exp+1))` — the magnitude knob. Chosen so
    /// a decoded weight lands near 0.05 and a row of 512 sums to O(1): tame
    /// enough that the comparison measures the decode and not an overflow.
    fn f16_at(&mut self, exp: i32) -> u16 {
        let mantissa = (((self.unit() + 1.0) * 512.0) as u32) & 0x3ff;
        let field = (exp + 15) as u32;
        ((field << 10) | mantissa) as u16
    }
}

/// One `block_q4_K`, byte for byte in ggml's order. The payload is arbitrary
/// — what is under test is the addressing, and an arbitrary block exercises it
/// exactly as a trained one does.
fn q4k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    out.extend_from_slice(&rng.f16_at(-12).to_le_bytes());
    out.extend_from_slice(&rng.f16_at(-9).to_le_bytes());
    for _ in 0..12 {
        out.push(rng.byte());
    }
    for _ in 0..128 {
        out.push(rng.byte());
    }
}

/// One `block_q6_K`. The sub-block scales are held to `-40..=40` so
/// `d * scale * q` stays in the same tame band `q4k_block` lands in.
fn q6k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    for _ in 0..192 {
        out.push(rng.byte());
    }
    for _ in 0..16 {
        let s = i32::from(rng.byte() % 81) - 40;
        out.push(s as i8 as u8);
    }
    out.extend_from_slice(&rng.f16_at(-12).to_le_bytes());
}

/// f32 to bf16, round-to-nearest-even — the conversion the loader does.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// **THE ONE CONTAINER BOTH ARMS READ**, written with the same profile
/// `checkpoint::file::write` stamps on a block: the layout names the scheme,
/// and `elems_per_block` / `block_bytes` are the constants a reader checks
/// sizes against. `file/zt.rs` recovers `QuantScheme::GgufQ4K` from that name,
/// which is the same scheme `qnf::scheme_of_sig` answers for the term the
/// stored text declares — and their equality is what puts the ladder on its
/// identity rung.
fn write_checkpoint(path: &Path) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));

    // The embedding, bf16 and unquantized in both arms — see `Micro`.
    let mut rng = Lcg(0xe1b_e11);
    let mut embed = Vec::with_capacity(VOCAB as usize * HIDDEN as usize * 2);
    for _ in 0..(u64::from(VOCAB) * HIDDEN) {
        embed.extend_from_slice(&bf16_bits(0.08 * rng.unit()).to_le_bytes());
    }
    writer
        .add(
            "embed",
            vec![u64::from(VOCAB), HIDDEN],
            ztensor::DType::BF16,
            &embed,
        )
        .expect("the table lands");

    // Sorted insertion, which is what canonical `.zt` form requires:
    // `embed` above, then `lm_head`, then `proj`.
    block_tensor(
        &mut writer,
        "lm_head",
        u64::from(VOCAB),
        "gguf.q6_k/1",
        Q6K_BYTES,
        0x6b1,
    );
    block_tensor(&mut writer, "proj", HIDDEN, "gguf.q4_k/1", Q4K_BYTES, 0x4b1);

    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// One `[rows, HIDDEN]` weight, stored as `rows` runs of super-blocks.
///
/// The declared SHAPE is the logical rectangle and the part's LENGTH is the
/// container's, which is the split this whole wave is about: a reader sizes
/// the payload from `block_bytes x (elements / elems_per_block)` and never
/// from the shape times an element width.
fn block_tensor(
    writer: &mut ztensor::Writer,
    name: &str,
    rows: u64,
    layout: &'static str,
    block: usize,
    seed: u64,
) {
    let blocks = HIDDEN as usize / SUPER;
    let mut rng = Lcg(seed);
    let mut bytes = Vec::with_capacity(rows as usize * blocks * block);
    for _ in 0..(rows as usize * blocks) {
        match block {
            Q4K_BYTES => q4k_block(&mut rng, &mut bytes),
            Q6K_BYTES => q6k_block(&mut rng, &mut bytes),
            other => panic!("no filler writes a {other}-byte block"),
        }
    }
    assert_eq!(bytes.len(), rows as usize * blocks * block);
    writer
        .object(name, |o| {
            o.shape(vec![rows, HIDDEN])
                .layout(layout)
                .attributes(cbor::map([
                    ("elems_per_block", SUPER as u64),
                    ("block_bytes", block as u64),
                ]))
                .part("data", |p| p.dtype(ztensor::DType::U8).bytes(&bytes))
        })
        .unwrap_or_else(|why| panic!("`{name}`: {why}"));
}

/// A scratch directory of this process's own.
fn scratch(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-j5-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

// ─────────────────────────────────────────────────────────────────────────
// The fire
// ─────────────────────────────────────────────────────────────────────────

/// Arbitrary ids inside the micro text's vocabulary: the model is synthetic,
/// so a prompt is a vector of numbers and nothing else.
const PROMPT: [u32; 6] = [11, 233, 7, 900, 42, 1001];

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn load(trace: Trace, contract: &ModelContract, container: &Path) -> Shell {
    Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract,
        checkpoint: container,
        budget: Budget::new(2, 32),
        patches: None,
        profile: None,
        page_size: 16,
        context: 64,
        slots: 2,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot weight artifact cache is off for a gate: a test that
        // shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the micro text loads its own checkpoint")
}

/// One prefill over `PROMPT`, and the logit row it lands.
fn fire(shell: &mut Shell) -> Vec<f32> {
    shell.open(0).expect("slot 0 opens");
    let rows = shell
        .fire(&[Lane {
            slot: 0,
            word: 0,
            tokens: &PROMPT,
        }])
        .expect("the prefill fires");
    rows[0].clone()
}

fn spread(logits: &[f32]) -> f32 {
    logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min)
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    assert!(
        spread(logits) > 1e-3,
        "{what} logits span {}, which is a rectangle nothing wrote",
        spread(logits),
    );
}

fn ready(what: &str) -> bool {
    if engine_cuda::device::present() {
        return true;
    }
    eprintln!("skipping {what}: no CUDA device on this machine");
    false
}

// ─────────────────────────────────────────────────────────────────────────
// (0) the host half — no device, and it runs on every `cargo test`
// ─────────────────────────────────────────────────────────────────────────

/// **THE DECLARATION FOLDS INTO THE CONTAINER, AND THE NUMBERS ARE ggml's.**
///
/// The trace is the artifact the engine is handed, so what it says about a
/// stored block is the whole of what the shell can know: one plane, no
/// companions, and a rectangle whose width is the row's BYTES. 288 is two
/// `block_q4_K`s and 420 is two `block_q6_K`s, and neither number is written
/// anywhere in this tree — both fall out of the term.
#[test]
fn a_stored_declaration_interns_one_byte_rectangle() {
    let (_, stored) = trace(Arm::Stored);
    let plane = |name: &str| {
        stored
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the trace interns `{name}`"))
    };
    assert_eq!(plane("proj").shape, vec![HIDDEN, 288]);
    assert_eq!(plane("proj").dtype, Q4_K);
    assert_eq!(plane("lm_head").shape, vec![u64::from(VOCAB), 420]);
    assert_eq!(plane("lm_head").dtype, Q6_K);
    // No `.scales`, no `.biases`: a braided block has no companion to name.
    assert_eq!(stored.params.len(), 3, "three planes for three weights");

    // And the reference text says the logical rectangle over the same names,
    // which is what makes the two arms one checkpoint.
    let (_, decoded) = trace(Arm::Decoded);
    assert_eq!(plane_of(&decoded, "proj").shape, vec![HIDDEN, HIDDEN]);
    assert_eq!(plane_of(&decoded, "proj").dtype, Dtype::Bf16);
}

fn plane_of<'a>(trace: &'a Trace, name: &str) -> &'a model_ir::Param {
    trace
        .params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("the trace interns `{name}`"))
}

/// **THE LADDER'S RUNG IS THE IDENTITY**, which is the whole of "serve as
/// stored": the contract states no cast for the two block planes, so nothing
/// decodes at load and the bytes on the device are the bytes in the file.
///
/// The reference arm's contract over the SAME tensors states a cast, and the
/// pair is what makes this test a statement about the ladder rather than about
/// one contract.
#[test]
fn the_stored_arm_states_no_cast_and_the_reference_arm_states_one() {
    let dir = scratch("contract");
    let container = dir.join("micro.zt");
    write_checkpoint(&container);
    let src = ztensor::Source::open(&container).expect("the fixture opens");

    for (arm, want_cast) in [(Arm::Stored, false), (Arm::Decoded, true)] {
        let m = Micro::new(arm);
        let contract = m.load(&src);
        for name in ["proj", "lm_head"] {
            let tensor = contract
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{arm:?}: the contract claims `{name}`"));
            let text = format!("{:?}", tensor.expr);
            assert_eq!(
                text.contains("Cast"),
                want_cast,
                "{arm:?}: `{name}` reads {text}, and this arm wanted \
                 {}a cast",
                if want_cast { "" } else { "no " },
            );
        }
    }
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────
// (1) the device half
// ─────────────────────────────────────────────────────────────────────────

/// **THE GATE.** One checkpoint, two texts, one prompt — and the stored arm's
/// logits are the decoded arm's.
///
/// **AGREEMENT IS ALSO THE PROOF THAT THE kquant POINT RAN.** There is no
/// counter read here and none is needed: the only other thing the anchor's
/// `None` arm can do is fall through to `linear::gemm::matmul`, which would
/// read a `[512, 288]` rectangle of ggml super-block bytes as bf16 elements —
/// a different shape holding different numbers, which cannot land within a
/// fifth of a percent of a decode of the same file. The entry's own
/// `debug_assert` that the plane binds as `U8` is a second witness, and this
/// gate runs in the `test` profile where it is live.
#[test]
fn a_stored_k_quant_projection_says_what_its_decode_says() {
    let _one = serialized();
    if !ready("a_stored_k_quant_projection_says_what_its_decode_says") {
        return;
    }
    let dir = scratch("parity");
    let container = dir.join("micro.zt");
    write_checkpoint(&container);

    let mut rows = Vec::new();
    for arm in [Arm::Decoded, Arm::Stored] {
        let (m, t) = trace(arm);
        let src = ztensor::Source::open(&container).expect("the fixture opens");
        let contract = m.load(&src);
        drop(src);
        let mut shell = load(t, &contract, &container);
        let logits = fire(&mut shell);
        finite(&logits, &format!("{arm:?}"));
        drop(shell);
        rows.push(logits);
    }

    let (decoded, stored) = (&rows[0], &rows[1]);
    assert_eq!(decoded.len(), stored.len(), "two readouts of one vocabulary");

    // **JUDGED AGAINST THE ROW'S OWN SCALE** (§J4a-1). The two arms round the
    // same weights in two places — the host decode lands bf16, the fused
    // point decodes in f32 inside the dot — so a per-element relative error
    // would be measuring cancellation at the near-zero logits and nothing
    // else. The spread is the ruler the argmax is read with.
    let ruler = spread(decoded);
    let worst = decoded
        .iter()
        .zip(stored)
        .enumerate()
        .max_by(|a, b| {
            (a.1.0 - a.1.1)
                .abs()
                .total_cmp(&(b.1.0 - b.1.1).abs())
        })
        .expect("a non-empty row");
    let gap = (worst.1.0 - worst.1.1).abs();
    eprintln!(
        "kquant parity: spread {ruler:.4}, worst gap {gap:.5} at column {} ({:.3}%)",
        worst.0,
        100.0 * gap / ruler,
    );
    assert!(
        gap <= 0.02 * ruler,
        "the stored arm and its host decode disagree by {gap} at column {}, which is \
         {:.2}% of the row's {ruler} spread — the two arms fold the same weights and \
         differ only in where they round",
        worst.0,
        100.0 * gap / ruler,
    );

    let _ = std::fs::remove_dir_all(&dir);
}
