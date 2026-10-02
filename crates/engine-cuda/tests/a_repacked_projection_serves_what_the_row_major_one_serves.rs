use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::{TILED_BAND, TILED_STEP, Trace};

const VOCAB: u32 = 1000;

const HIDDEN: u64 = 512;

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    RowMajor,
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

const GROUP: usize = 64;

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

    fn unit(&mut self) -> f32 {
        ((self.next() >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

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
            ztensor::Leaf::BF16,
            &embed,
        )
        .expect("the table lands");

    affine_tensor(&mut writer, "lm_head", VOCAB as usize, arm, 0x4ead);
    affine_tensor(&mut writer, "proj", HIDDEN as usize, arm, 0x9401);

    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

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

    let term =
        ztensor::Term::parse(&format!("g{GROUP}_u4_bf16_b_bf16")).expect("the affine term parses");
    let shape = vec![stated as u64, HIDDEN];
    let blob = canonical_blob(&term, &shape, [&codes, &scales, &biases]);
    writer
        .object(name, |o| {
            let o = o.shape(shape.clone()).term(term.clone());
            match arm {
                Arm::RowMajor => o.planes([codes.as_slice(), scales.as_slice(), biases.as_slice()]),
                Arm::Tiled => o
                    .layout(checkpoint::serving::MMA_TILED)
                    .attr("band", u64::from(TILED_BAND))
                    .attr("step", u64::from(TILED_STEP))
                    .bytes(&blob),
            }
        })
        .unwrap_or_else(|why| panic!("`{name}`: {why}"));
}

fn canonical_blob(term: &ztensor::Term, shape: &[u64], planes: [&Vec<u8>; 3]) -> Vec<u8> {
    let laid = term.planes(shape).expect("the term lays out this shape");
    assert_eq!(laid.len(), planes.len());
    let total = term
        .canonical_size(shape)
        .expect("the term sizes this shape");
    let mut blob = vec![0u8; total as usize];
    for (plane, bytes) in laid.iter().zip(planes) {
        assert_eq!(plane.len as usize, bytes.len(), "plane `{}`", plane.path);
        blob[plane.range()].copy_from_slice(bytes);
    }
    blob
}

fn scratch(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-j4b-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

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
    assert_eq!(plane(&tiled, "proj"), vec![HIDDEN, HIDDEN]);

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
