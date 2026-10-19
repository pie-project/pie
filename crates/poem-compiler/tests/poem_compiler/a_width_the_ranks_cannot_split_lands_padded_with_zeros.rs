//! An artifact imported whole stores gpt-oss's experts 2880 wide, 90 MXFP4
//! blocks no four ranks split; the model pads them, and each rank lands its
//! share of the padded plane, the padding zeros.

use std::path::{Path, PathBuf};

use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile_streaming};
use checkpoint::types::BackendKind;
use poem::{Dtype, Param, ParamSource, Platform, Shard};

const RANKS: u32 = 4;
const EXPERTS: usize = 2;
const ROWS: usize = 3;
/// Three blocks stored, four held: one a rank.
const STORED: usize = 3;
const HELD: usize = 4;
const BLOCK: usize = 32;
/// A bias of two segments, each three columns stored and four held.
const LEG: usize = 3;
const LEG_HELD: usize = 4;

fn code(at: usize) -> u8 {
    (at % 255 + 1) as u8
}

fn scale(at: usize) -> u8 {
    (100 + at % 50) as u8
}

fn bias(at: usize) -> [u8; 2] {
    (((1.0 + at as f32).to_bits() >> 16) as u16).to_le_bytes()
}

fn write(dir: &Path) {
    let mut writer = ztensor::Writer::create(dir.join("model.zt")).expect("the container opens");
    let blocks = EXPERTS * ROWS * STORED;
    let codes: Vec<u8> = (0..blocks * BLOCK / 2).map(code).collect();
    let scales: Vec<u8> = (0..blocks).map(scale).collect();
    let term = ztensor::Term::parse("g32_e2m1_e8m0_n").expect("the mxfp4 term");
    writer
        .object("bank", |o| {
            o.shape(vec![EXPERTS as u64, ROWS as u64, (STORED * BLOCK) as u64])
                .term(term)
                .planes([codes.as_slice(), scales.as_slice()])
        })
        .expect("the bank lands");
    let biases: Vec<u8> = (0..EXPERTS * 2 * LEG).flat_map(bias).collect();
    writer
        .add(
            "bias",
            vec![EXPERTS as u64, (2 * LEG) as u64],
            ztensor::Leaf::BF16,
            &biases,
        )
        .expect("the bias lands");
    writer.finish().expect("the container closes");
}

/// The params as a rank of four holds them.
fn params() -> Vec<Param> {
    let param = |name: &str, shape: Vec<u64>, shard: Shard, dtype: Dtype| Param {
        name: name.into(),
        shape,
        shard,
        dtype,
        source: ParamSource::Checkpoint,
        layout: Default::default(),
        placement: Default::default(),
    };
    // A codes plane is traced in blocks of sixteen bytes.
    let share = HELD as u64 / u64::from(RANKS);
    let leg = LEG_HELD as u64 / u64::from(RANKS);
    vec![
        param(
            "bank",
            vec![EXPERTS as u64, ROWS as u64, share, 16],
            Shard::Cut {
                axis: 2,
                segments: vec![share],
                heads: Vec::new(),
            },
            Dtype::Mxfp4,
        ),
        param(
            "bias",
            vec![EXPERTS as u64, 2 * leg],
            Shard::Cut {
                axis: 1,
                segments: vec![leg, leg],
                heads: Vec::new(),
            },
            Dtype::Bf16,
        ),
    ]
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("padded_share_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

#[test]
fn a_width_the_ranks_cannot_split_lands_padded_with_zeros() {
    let dir = scratch();
    write(&dir);
    let src = ztensor::Source::open(dir.join("model.zt")).expect("the artifact opens");
    let contract = poem::import::own_contract(&src, &params(), RANKS, Platform::Cuda)
        .unwrap_or_else(|why| panic!("a rank reads a plane stored narrower than it holds: {why}"));
    let metadata = parse_metadata(&dir).expect("the artifact reads");

    for rank in 0..RANKS as usize {
        let target = StorageTarget::for_backend(BackendKind::Cuda, rank as u32, RANKS);
        let plan = compile_streaming(&metadata, &contract, target)
            .unwrap_or_else(|why| panic!("rank {rank}'s share does not compile: {why}"));
        let mut sink = MemorySink::default();
        Execution::new(&plan, &dir)
            .streaming()
            .sink(&mut sink)
            .run()
            .unwrap_or_else(|why| panic!("rank {rank}'s share does not land: {why}"));
        let mut landed = |name: &str| {
            sink.tensors
                .remove(name)
                .unwrap_or_else(|| panic!("rank {rank} lands no `{name}`"))
        };

        // One block a row: the stored block of its rank, or zeros past them.
        let block = |row: usize| (rank < STORED).then_some(row * STORED + rank);
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for row in 0..EXPERTS * ROWS {
            match block(row) {
                Some(at) => {
                    codes.extend((at * BLOCK / 2..(at + 1) * BLOCK / 2).map(code));
                    scales.push(scale(at));
                }
                None => {
                    codes.extend([0; BLOCK / 2]);
                    scales.push(0);
                }
            }
        }
        assert_eq!(landed("bank"), codes, "rank {rank}'s codes");
        assert_eq!(landed("bank.scales"), scales, "rank {rank}'s scales");

        // Each segment's column of this rank, or a zero past those stored.
        let mut biases = Vec::new();
        for expert in 0..EXPERTS {
            for leg in 0..2 {
                biases.extend(match rank < LEG {
                    true => bias(expert * 2 * LEG + leg * LEG + rank),
                    false => [0; 2],
                });
            }
        }
        assert_eq!(landed("bias"), biases, "rank {rank}'s bias");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
