//! DeepSeek-V4-Flash's sublayer seams formed as `fused.hc_fold_pre` answer
//! what the fold and the pre-mix apart answer, under the chip's own profile:
//! the same logits at a decode's row (the kernel declines), and at the rows
//! the profile seams the same argmax and every logit within 2% of the
//! spread. A miniature of random planes amplifies a bf16 step of a normed
//! row through its routes and index picks: turning `fuse.hc_pre` off moves
//! its logits as far (about 1% of the spread at 256 rows).

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Knobs, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{Fused, Linear, Operation, ParamSource, Trace};

const ROW: &str = "dsv4-flash-mini-bf16-kv-bf16";
const ROWS: [u32; 4] = [1, 64, 256, 1024];
const CEILING: u32 = 1024;
const PAGE: u32 = 16;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A plane's bf16 values, by its name.
fn values(name: &str, count: usize) -> Vec<u16> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            (value.to_bits() >> 16) as u16
        })
        .collect()
}

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    // A hashed route's table names experts a token, each one of the
    // fewest experts any route of the row picks over.
    let experts = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Linear(Linear::MoeRoute { experts, .. }) => Some(u64::from(*experts)),
            _ => None,
        })
        .min()
        .unwrap_or(1);
    for param in planes {
        if param.dtype == Dtype::I64 {
            let picks = param.shape.last().copied().unwrap_or(1);
            let count: u64 = param.shape.iter().product();
            let bytes: Vec<u8> = (0..count)
                .flat_map(|at| (((at / picks) * 3 + at % picks) % experts).to_le_bytes())
                .collect();
            writer
                .add(
                    param.name.as_str(),
                    param.shape.clone(),
                    ztensor::Leaf::I64,
                    &bytes,
                )
                .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
            continue;
        }
        let plane = values(&param.name, param.shape.iter().product::<u64>() as usize);
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match param.dtype {
            Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                plane.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            Dtype::F32 => (
                ztensor::Leaf::F32,
                plane
                    .iter()
                    .flat_map(|v| f32::from_bits(u32::from(*v) << 16).to_le_bytes())
                    .collect(),
            ),
            other => panic!(
                "`{}` is {other:?}, which this fixture does not write",
                param.name
            ),
        };
        writer
            .add(param.name.as_str(), param.shape.clone(), leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
}

/// The chip's own resolved profile, the seam rule on or off.
fn chip(dir: &Path, seamed: bool) -> profile::Profile {
    let (name, facts) = engine_cuda::chip::probe(0).expect("a device");
    let profiles = dir.join(format!("profiles-{seamed}"));
    let user = engine_cuda::chip::SHELF.user_path(&profiles, &name);
    std::fs::create_dir_all(user.parent().unwrap()).unwrap();
    std::fs::write(&user, format!("api = 1\n[fuse]\nhc_fold_pre = {seamed}\n")).unwrap();
    let (chip, _) = engine_cuda::chip::resolve(&name, &facts, Some(&profiles));
    assert_eq!(
        chip.flag("fuse.hc_fold_pre"),
        Some(seamed),
        "the user file states the seam"
    );
    chip
}

/// Each row count's last-row logits and median step time.
fn run(trace: &Trace, container: &Path, dir: &Path, seamed: bool) -> Vec<(Vec<f32>, f64)> {
    let source = ztensor::Source::open(container).expect("the fixture opens");
    let contract: ModelContract =
        poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
            .expect("a container of the plan's own planes is read by its own names");
    drop(source);
    let staged = Shell::stage(engine_cuda::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: container,
        budget: Budget::new(1, CEILING),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: 2 * CEILING,
        slots: 1,
        kv_budget: None,
        memory_share: 0.5,
        ordinal: 0,
        knobs: Knobs {
            chip: Some(chip(dir, seamed)),
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the row loads");
    let mut shell = staged.finish().expect("the row arms");
    let repeats: usize = std::env::var("PIE_HC_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    ROWS.iter()
        .map(|&rows| {
            let prompt: Vec<u32> = (0..rows).map(|t| (t * 7 + 11) % 4096).collect();
            let word = trace.facts.word(&Request::new(rows, false));
            let fire = |shell: &mut Shell| {
                shell.open(0).unwrap();
                let at = std::time::Instant::now();
                let mut out = shell
                    .fire(&[Lane {
                        slot: 0,
                        word,
                        tokens: &prompt,
                    }])
                    .unwrap();
                (out.remove(0), at.elapsed().as_secs_f64() * 1e6)
            };
            let (first, _) = fire(&mut shell);
            let mut times: Vec<f64> = (0..repeats)
                .map(|_| {
                    let (again, us) = fire(&mut shell);
                    assert!(
                        again
                            .iter()
                            .zip(&first)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "seamed {seamed} rows {rows}: a fire answers as the first"
                    );
                    us
                })
                .collect();
            times.sort_by(f64::total_cmp);
            (first, times.get(times.len() / 2).copied().unwrap_or(0.0))
        })
        .collect()
}

#[test]
#[ignore = "a 7 GB DeepSeek-V4-Flash miniature on a tcgen05 chip"]
fn a_seamed_fold_answers_its_fold_and_pre_mix() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let row = poem_compiler::catalog::deployment(ROW).unwrap_or_else(|| panic!("`{ROW}`"));
    let trace = row.trace(Platform::Cuda);
    let formed = poem_compiler::fuse::fuse(trace.clone(), engine_cuda::FUSIONS.iter())
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Fused(Fused::HcFoldPre { .. })))
        .count();
    eprintln!("{ROW}: {formed} seams form");
    assert!(formed > 0, "{ROW}: no seam forms");

    let dir = std::env::temp_dir().join(format!("pie-hc-seam-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let _scratch = Scratch(dir.clone());
    let container = dir.join("model.zt");
    write_checkpoint(&container, &trace);

    let seamed = run(&trace, &container, &dir, true);
    let apart = run(&trace, &container, &dir, false);
    let mut failed = Vec::new();
    for ((&rows, (fused, fused_us)), (want, apart_us)) in ROWS.iter().zip(&seamed).zip(&apart) {
        let spread = want.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - want.iter().copied().fold(f32::INFINITY, f32::min);
        let gaps: Vec<f32> = fused.iter().zip(want).map(|(a, b)| (a - b).abs()).collect();
        let worst = gaps.iter().copied().fold(0.0f32, f32::max);
        let mean = gaps.iter().sum::<f32>() / gaps.len().max(1) as f32;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
        };
        let same = argmax(fused) == argmax(want);
        let differ = gaps.iter().filter(|g| **g > 0.0).count();
        eprintln!(
            "{rows:>5} rows: spread {spread:.4} worst {worst:.3e} ({:.3e} of spread) mean {mean:.3e} \
             differ {differ}/{} argmax {} | step seamed {fused_us:.1} us apart {apart_us:.1} us",
            worst / spread,
            gaps.len(),
            if same { "agrees" } else { "DIFFERS" },
        );
        assert!(
            spread > 1e-3,
            "{rows} rows: the logits are a rectangle nothing wrote"
        );
        if !fused.iter().all(|v| v.is_finite()) || !same || worst > 0.02 * spread {
            failed.push(rows);
        }
    }
    assert!(
        failed.is_empty(),
        "the seam answers apart at {failed:?} rows"
    );
}
