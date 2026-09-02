//! **THE TILED FLIP, END TO END** (§J4b) — one model text in two orders, one
//! prompt, and the same logits.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda \
//!   --test a_repacked_projection_serves_what_the_row_major_one_serves -- --nocapture
//! ```
//!
//! # What is under test, and what already was
//!
//! Both kernels are golden (`kernels-cuda/tests/tiled_matmul.rs`: the tiled
//! GEMM and the tiled decode point against a host fold, and the repack
//! against a host un-repack), and so is the import half
//! (`checkpoint/tests/tiled_repack.rs`: `Expr::Repack` compiled at the
//! CONVERT target and run by the host executor, against a second
//! transcription of the same layout). What had never run is the SERVING
//! chain between a declaration and a launch:
//!
//! ```text
//! Dtype::U4g64tiled                        the text's word
//!   -> Weight::planes         three planes, rows banded up to TILED_BAND
//!   -> checkpoint_dsl::claim  the same banded rectangle, so the load agrees
//!   -> weights::plane_bytes   -> WeightRow::Planes { repacked: true }
//!   -> Run::maybe_tiled_planes
//!   -> linear::tiled::{matmul, matmul_gemv, lm_head, lm_head_gemv}
//! ```
//!
//! # THE ORACLE IS THE SAME WEIGHTS, IN THE OTHER ORDER
//!
//! One seeded set of affine planes, two containers. The reference text
//! declares its projections `U4g64` and the container holds the codes
//! row-major, which takes `linear::quant`'s fused GEMV and its decoded twin —
//! the roads every affine SKU takes today. The tiled text declares them
//! `U4g64tiled` and its container holds the SAME codes under the SAME
//! factors, relabelled by [`repack`] below into m16n8k16 fragment order.
//!
//! That is what makes this a gate rather than a self-comparison. The repack
//! here is written from `linear/tiled.cuh`'s banner and not from any
//! executor's code, so it is a THIRD independent statement of the layout
//! beside the kernel's and the host executor's; a wrong witness, a wrong
//! plane rectangle, a weight bound at the wrong address or a dispatch arm
//! that took the row-major road on tiled bytes all fail here, and none of
//! them can be papered over by a matching mistake in an oracle written
//! beside it.
//!
//! **THE SAME NUMBERS, NOT THE SAME BITS.** The fused GEMV never
//! materialises a weight element and the tiled points materialise every one
//! of them as a bf16 register, so the two arms round in different places —
//! §J4a-1's ruling, and `tiled_matmul.rs`'s own cross-arm gate. The ruler is
//! the logit row's spread.
//!
//! # Gating
//!
//! Skipped at run time with no device, as `a_stored_k_quant_row_serves_as_
//! stored.rs` is, and for its reason. Nothing here reads a checkpoint off
//! disk — both containers are written from the trace's own params into a
//! scratch directory.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::{TILED_BAND, TILED_STEP, Trace};
use ztensor::format::cbor;

// ─────────────────────────────────────────────────────────────────────────
// The text
// ─────────────────────────────────────────────────────────────────────────

/// Rows of the head, and of the embedding table.
///
/// **NOT A MULTIPLE OF [`TILED_BAND`], ON PURPOSE.** The head's rows are the
/// axis the repack pads, so a thousand of them is sixty-two whole bands and a
/// sixty-third that is eight columns of weight and eight of zero. That tail
/// is the one geometric fact the layout adds, and a vocabulary that divided
/// would never exercise it.
const VOCAB: u32 = 1000;

/// The contraction both projections walk — eight groups of sixty-four, which
/// is also eight whole [`TILED_STEP`]-wide steps.
const HIDDEN: u64 = 512;

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

/// Which order the two projections are declared in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `U4g64`: the codes as the checkpoint lays them, low nibble first.
    RowMajor,
    /// `U4g64tiled`: the same codes in m16n8k16 fragment order.
    Tiled,
}

impl Arm {
    fn dtype(self) -> Dtype {
        match self {
            Arm::RowMajor => Dtype::U4g64,
            Arm::Tiled => Dtype::U4g64tiled,
        }
    }
}

/// **A MODEL NOBODY SHIPS**, for `a_stored_k_quant_row_serves_as_stored.rs`'s
/// reason: the embedding table stays bf16 because the affine gather is a
/// different point with a different gate, and what this file is about is the
/// two `linear` arms.
struct Micro {
    embed: Weight,
    proj: Weight,
    head: Weight,
}

impl Micro {
    fn new(arm: Arm) -> Micro {
        let w = arm.dtype();
        Micro {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
            proj: Weight::sym("proj", [HIDDEN, HIDDEN], w),
            head: Weight::sym("lm_head", [u64::from(VOCAB), HIDDEN], w),
        }
    }

    /// The load contract, stated through the same builder a family's `load`
    /// uses. `read_own` in both arms: each container holds its weights under
    /// their own names, and the ONLY difference between the two loads is the
    /// dtype the text declared.
    fn load(&self, src: &ztensor::Source) -> ModelContract {
        let mut b = checkpoint_dsl::Builder::new(src, 1, model_dsl::Platform::Cuda);
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
    let trace = trace_hybrid("tiled-micro", &m, Platform::Cuda);
    (m, trace)
}

// ─────────────────────────────────────────────────────────────────────────
// The planes, and the relabelling
// ─────────────────────────────────────────────────────────────────────────

/// The codes under one factor, which is `U4g64`'s group and `U4g64tiled`'s.
const GROUP: usize = 64;

/// A seeded stream, so both containers hold the same weights.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn code(&mut self) -> u8 {
        ((self.next() >> 33) & 0xF) as u8
    }

    /// Uniform on `[-0.5, 0.5)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

/// f32 to bf16, round-to-nearest-even — the conversion the loader does.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// One affine triplet: `[rows, HIDDEN]` codes low-nibble-first, and two
/// `[rows, HIDDEN / GROUP]` bf16 factor planes.
///
/// The scale is near `2^-5` and the bias near zero, which is the band an
/// affine weight's factors sit in and which keeps a row of 512 summing to
/// O(1) — tame enough that the comparison measures the layout and not an
/// overflow.
struct Planes {
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

fn planes(rows: usize, seed: u64) -> Planes {
    let mut rng = Lcg(seed);
    let mut codes = vec![0u8; rows * HIDDEN as usize / 2];
    for at in 0..(rows * HIDDEN as usize) {
        codes[at / 2] |= rng.code() << (4 * (at % 2));
    }
    let groups = HIDDEN as usize / GROUP;
    let mut scales = Vec::with_capacity(rows * groups * 2);
    let mut biases = Vec::with_capacity(rows * groups * 2);
    for _ in 0..(rows * groups) {
        scales.extend_from_slice(&bf16_bits(0.06 * (rng.unit() + 0.6)).to_le_bytes());
        biases.extend_from_slice(&bf16_bits(0.2 * rng.unit()).to_le_bytes());
    }
    Planes {
        codes,
        scales,
        biases,
    }
}

/// **THE RELABELLING, WRITTEN FROM THE BANNER.** Word `lane` of tile
/// `(band, k tile)` holds, at nibble `s + 4h`, the code at
/// `k = 16*kt + 2*(lane%4) + 8*(s&1) + h` and
/// `n = 16*band + lane/4 + 8*(s>=2)`; four k tiles are grouped as one lane's
/// `uint4`, so the word order is `[band][k quad][lane][4]`. Columns past
/// `rows` are a zero code beside a zero factor, which decodes to a zero
/// weight.
///
/// This is `linear/tiled.cuh`'s `repack_affine_tiled`, transcribed — a THIRD
/// statement of the layout beside the kernel's and
/// `checkpoint::executor::walk`'s, which is what makes agreement here
/// evidence rather than a coincidence.
fn repack(codes: &[u8], rows: usize) -> Vec<u8> {
    let band = TILED_BAND as usize;
    let quad = (TILED_STEP / TILED_BAND) as usize;
    let k = HIDDEN as usize;
    let row_bytes = k / 2;
    let bands = rows.div_ceil(band);
    let quads = (k / band) / quad;
    let mut out = vec![0u8; bands * band * row_bytes];
    let mut at = 0usize;
    for b in 0..bands {
        for kq in 0..quads {
            for lane in 0..32usize {
                for word in 0..quad {
                    let kt = kq * quad + word;
                    let col_of = lane / 4;
                    let k_base = kt * band + 2 * (lane % 4);
                    let mut res = 0u32;
                    for s in 0..4usize {
                        let col = b * band + col_of + if s >= 2 { 8 } else { 0 };
                        if col >= rows {
                            continue;
                        }
                        for h in 0..2usize {
                            let kk = k_base + if s % 2 == 1 { 8 } else { 0 } + h;
                            let flat = col * k + kk;
                            let byte = codes[flat / 2];
                            let code = if flat % 2 == 0 {
                                u32::from(byte & 0xF)
                            } else {
                                u32::from(byte >> 4)
                            };
                            res |= code << (4 * (s + 4 * h));
                        }
                    }
                    out[at * 4..at * 4 + 4].copy_from_slice(&res.to_le_bytes());
                    at += 1;
                }
            }
        }
    }
    out
}

/// **THE FACTOR HALF** — `[rows][group]` becomes `[band][group][16]`, a
/// transpose of the (column, group) rectangle inside each band, with a
/// band's tail written as a zero factor.
fn repack_factors(factors: &[u8], rows: usize) -> Vec<u8> {
    let band = TILED_BAND as usize;
    let groups = HIDDEN as usize / GROUP;
    let padded = rows.div_ceil(band) * band;
    let mut out = vec![0u8; padded * groups * 2];
    for (at, slot) in out.chunks_exact_mut(2).enumerate() {
        let j = at % band;
        let rest = at / band;
        let g = rest % groups;
        let row = rest / groups * band + j;
        if row < rows {
            let from = (row * groups + g) * 2;
            slot.copy_from_slice(&factors[from..from + 2]);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// The containers
// ─────────────────────────────────────────────────────────────────────────

/// **ONE SET OF WEIGHTS, TWO ORDERS.** The reference container holds the
/// planes as `planes` drew them; the tiled one holds `repack`'s answer under
/// the banded rectangle `model_dsl::Weight::planes` publishes. Everything
/// else — the names, the profile, the group, the attributes — is identical,
/// which is what leaves the ORDER as the only difference between two loads.
fn write_checkpoint(path: &Path, arm: Arm) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));

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

    // Sorted insertion, which is what canonical `.zt` form requires: `embed`,
    // then the head's triplet, then the projection's.
    affine_tensor(&mut writer, "lm_head", VOCAB as usize, arm, 0x4ead);
    affine_tensor(&mut writer, "proj", HIDDEN as usize, arm, 0x9401);

    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// One affine weight, as the three tensors a triplet is — in whichever order
/// `arm` asks for.
fn affine_tensor(writer: &mut ztensor::Writer, name: &str, rows: usize, arm: Arm, seed: u64) {
    let drawn = planes(rows, seed);
    let (codes, scales, biases, stated) = match arm {
        Arm::RowMajor => (drawn.codes, drawn.scales, drawn.biases, rows),
        Arm::Tiled => (
            repack(&drawn.codes, rows),
            repack_factors(&drawn.scales, rows),
            repack_factors(&drawn.biases, rows),
            rows.div_ceil(TILED_BAND as usize) * TILED_BAND as usize,
        ),
    };
    let groups = HIDDEN as usize / GROUP;
    assert_eq!(codes.len(), stated * HIDDEN as usize / 2);
    assert_eq!(scales.len(), stated * groups * 2);

    // The affine-group profile `checkpoint::file::write` stamps on an MLX
    // bank, which `file::zt::affine_group_scheme` reads back as
    // `QuantScheme::MlxAffineU4`: four bits, packed low code first, a
    // zero point stored plain beside the scale, and its factors stated as
    // `f16_factors`.
    //
    // **`scale_form` IS REQUIRED AND THIS FIXTURE HAD BEEN WRITTEN BEFORE IT
    // WAS.** The QNF wave made it the key that separates schemes agreeing on
    // every other field — `MlxAffineU4` and `Int8Asymmetric` both read
    // lsb_first tensor/plain, and only the factor width tells them apart — so
    // a profile without it names no scheme and the reader says so rather than
    // guessing. Every field here is the one `write.rs` stamps for this scheme,
    // `word` included, because a fixture that imitates the writer loosely is a
    // fixture that stops imitating it the next time the writer moves.
    let attrs = cbor::map([
        ("axis", cbor::Value::from(1u64)),
        ("bits", cbor::Value::from(4u64)),
        ("group_size", cbor::Value::from(GROUP as u64)),
        (
            "packing",
            cbor::map([
                ("order", cbor::Value::from("lsb_first")),
                ("per_word", cbor::Value::from(8u64)),
                ("word", cbor::Value::from("u32")),
            ]),
        ),
        ("scale_form", cbor::Value::from("f16_factors")),
        (
            "zero_point",
            cbor::map([
                ("form", cbor::Value::from("tensor")),
                ("packing", cbor::Value::from("plain")),
            ]),
        ),
    ]);
    writer
        .object(name, |o| {
            o.shape(vec![stated as u64, HIDDEN])
                .layout("zt.quant_group/1")
                .attributes(attrs)
                .part("data", |p| p.dtype(ztensor::DType::U8).bytes(&codes))
        })
        .unwrap_or_else(|why| panic!("`{name}`: {why}"));
    for (suffix, bytes) in [(".biases", &biases), (".scales", &scales)] {
        let full = format!("{name}{suffix}");
        writer
            .add(
                &full,
                vec![stated as u64, groups as u64],
                ztensor::DType::BF16,
                bytes,
            )
            .unwrap_or_else(|why| panic!("`{full}`: {why}"));
    }
}

/// A scratch directory of this process's own.
fn scratch(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-j4b-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

// ─────────────────────────────────────────────────────────────────────────
// The fire
// ─────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────
// (0) the host half — no device, and it runs on every `cargo test`
// ─────────────────────────────────────────────────────────────────────────

/// **THE DECLARATION BANDS THE RECTANGLE, AND BOTH SIDES BAND IT THE SAME.**
///
/// The trace is what the engine is handed and the contract is what the load
/// reads; a repacked plane is `rows` rounded up to a whole band in the first
/// and would be `rows` in the second if nobody had said so, which is a
/// tensor arriving shorter than the plane reserved for it. 1008 is 1000
/// rounded up to sixteen, and it is written nowhere — it falls out of
/// [`TILED_BAND`].
#[test]
fn a_tiled_declaration_bands_the_rows_on_both_sides() {
    let (_, row_major) = trace(Arm::RowMajor);
    let (_, tiled) = trace(Arm::Tiled);
    let plane = |t: &Trace, name: &str| {
        t.params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the trace interns `{name}`"))
            .shape
            .clone()
    };
    let padded = u64::from(VOCAB).div_ceil(u64::from(TILED_BAND)) * u64::from(TILED_BAND);
    assert_eq!(plane(&row_major, "lm_head"), vec![u64::from(VOCAB), HIDDEN]);
    assert_eq!(plane(&tiled, "lm_head"), vec![padded, HIDDEN]);
    assert_eq!(plane(&tiled, "lm_head.scales"), vec![padded, HIDDEN / 64]);
    // The projection's rows already divide, so the two agree there — which
    // is the point: the banding is a padding and not a reshape.
    assert_eq!(plane(&tiled, "proj"), vec![HIDDEN, HIDDEN]);

    // And the CONTRACT claims the same rectangle, which is the half a trace
    // cannot check by itself. Read off the FACTOR plane: a quantized codes
    // entry is `TensorContract::inferred` — its shape is the expression's,
    // stated once — while its companions carry the declared rectangle, and
    // `interned` derives theirs from the codes' by dividing the last axis.
    // So a factor plane that is 1008 rows deep is a codes plane that was
    // claimed at 1008 too.
    let dir = scratch("bands");
    let container = dir.join("micro.zt");
    write_checkpoint(&container, Arm::Tiled);
    let src = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = Micro::new(Arm::Tiled).load(&src);
    let claimed = contract
        .tensors
        .iter()
        .find(|t| t.name == "lm_head.scales")
        .expect("the contract claims the head's scales");
    assert_eq!(
        claimed.shape.as_deref(),
        Some(&[padded as i64, (HIDDEN / 64) as i64][..]),
        "the contract claims the rectangle the trace interned"
    );
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────
// (1) the device half
// ─────────────────────────────────────────────────────────────────────────

