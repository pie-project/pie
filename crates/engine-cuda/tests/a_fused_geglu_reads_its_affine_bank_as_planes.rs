#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Diagnostics, Graphs, Knobs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::{Linear, Operation, Trace};

const VOCAB: u32 = 1000;
const HIDDEN: u64 = 512;
const INTER: u32 = 256;
const GROUP: usize = 64;
const PAGE: u32 = 16;
const TOKENS: u32 = 8;

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

fn classify(_: &Request) -> u64 {
    0
}

struct Micro {
    embed: Weight,
    gate_up: Weight,
    down: Weight,
    head: Weight,
}

impl Micro {
    fn new() -> Micro {
        Micro {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
            gate_up: Weight::sym("gate_up", [2 * u64::from(INTER), HIDDEN], Dtype::U4g64),
            down: Weight::sym("down", [HIDDEN, u64::from(INTER)], Dtype::U4g64),
            head: Weight::sym("lm_head", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
        }
    }
}

impl ForwardHybrid for Micro {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = ops::layout::embed(&inputs.tokens(), &self.embed, VOCAB);
        let act =
            ops::linear::mlp_geglu_tanh_packed(&ops::linear::matmul(&x, &self.gate_up), INTER);
        let h = ops::linear::matmul(&act, &self.down);
        ops::linear::lm_head(&h, &self.head)
    }
}

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

fn affine_tensor(writer: &mut ztensor::Writer, name: &str, rows: usize, k: usize, seed: u64) {
    let mut rng = Lcg(seed);
    let mut codes = vec![0u8; rows * k / 2];
    for at in 0..(rows * k) {
        codes[at / 2] |= rng.code() << (4 * (at % 2));
    }
    let groups = k / GROUP;
    let mut scales = Vec::with_capacity(rows * groups * 2);
    let mut biases = Vec::with_capacity(rows * groups * 2);
    for _ in 0..(rows * groups) {
        scales.extend_from_slice(&bf16_bits(0.06 * (rng.unit() + 0.6)).to_le_bytes());
        biases.extend_from_slice(&bf16_bits(0.2 * rng.unit()).to_le_bytes());
    }
    let term =
        ztensor::Term::parse(&format!("g{GROUP}_u4_bf16_b_bf16")).expect("the affine term parses");
    writer
        .object(name, |o| {
            o.shape(vec![rows as u64, k as u64])
                .term(term.clone())
                .planes([codes.as_slice(), scales.as_slice(), biases.as_slice()])
        })
        .unwrap_or_else(|why| panic!("`{name}`: {why}"));
}

fn dense_tensor(writer: &mut ztensor::Writer, name: &str, rows: usize, k: usize, seed: u64) {
    let mut rng = Lcg(seed);
    let mut bytes = Vec::with_capacity(rows * k * 2);
    for _ in 0..(rows * k) {
        bytes.extend_from_slice(&bf16_bits(0.08 * rng.unit()).to_le_bytes());
    }
    writer
        .add(
            name,
            vec![rows as u64, k as u64],
            ztensor::Leaf::BF16,
            &bytes,
        )
        .unwrap_or_else(|why| panic!("`{name}`: {why}"));
}

fn write_checkpoint(path: &Path) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    affine_tensor(&mut writer, "down", HIDDEN as usize, INTER as usize, 0xd0e4);
    dense_tensor(
        &mut writer,
        "embed",
        VOCAB as usize,
        HIDDEN as usize,
        0xe1b_e11,
    );
    affine_tensor(
        &mut writer,
        "gate_up",
        2 * INTER as usize,
        HIDDEN as usize,
        0x9a7e,
    );
    dense_tensor(
        &mut writer,
        "lm_head",
        VOCAB as usize,
        HIDDEN as usize,
        0x4ead,
    );
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture() -> Fixture {
    let trace = trace_hybrid("geglu-affine-micro", &Micro::new(), Platform::Cuda);
    let dir = std::env::temp_dir().join(format!("pie-geglu-affine-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let container = dir.join("micro.zt");
    write_checkpoint(&container);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        trace,
        contract,
        container,
        _dir: Scratch(dir),
    }
}

fn fire(fixture: &Fixture, knobs: Knobs) -> engine_cuda::Result<Vec<f32>> {
    let ceiling = TOKENS.next_multiple_of(PAGE);
    let mut shell = Shell::load(Boot {
        classify,
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, ceiling),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: ceiling,
        slots: 1,
        pages: ceiling / PAGE,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs,
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })?;
    shell.open(0).expect("slot 0 opens");
    let prompt: Vec<u32> = (0..TOKENS).map(|t| (t * 7 + 11) % VOCAB).collect();
    let mut rows = shell.fire(&[Lane {
        slot: 0,
        word: 0,
        tokens: &prompt,
    }])?;
    Ok(rows.remove(0))
}

#[test]
fn a_fused_geglu_reads_its_affine_bank_as_planes() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture();

    let fused = model_ir::fuse::gemm_epilogues(fixture.trace.clone());
    assert!(
        fused
            .nodes
            .iter()
            .any(|node| matches!(node.op, Operation::Linear(Linear::MatmulGeglu { .. }))),
        "the gate-up matmul and the geglu over it fold into one epilogue, or this fires \
         nothing the row-major arm does not"
    );

    let folded = fire(&fixture, Knobs::default()).expect("the folded epilogue serves");
    let apart = fire(
        &fixture,
        Knobs {
            diagnostics: Diagnostics {
                fuse_chains: false,
                ..Diagnostics::default()
            },
            ..Knobs::default()
        },
    )
    .expect("the unfolded pair serves");

    assert!(!folded.is_empty(), "the folded fire read out no logits");
    assert!(
        folded.iter().all(|value| value.is_finite()),
        "the folded fire read out a non-finite logit"
    );
    assert_eq!(
        folded, apart,
        "the folded epilogue reads the same planes as the pair it replaces, so it answers \
         the same logits"
    );
}
