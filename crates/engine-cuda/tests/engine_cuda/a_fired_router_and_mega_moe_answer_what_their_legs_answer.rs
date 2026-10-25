//! On an sm_100 chip under its own profile, a routed row's fused router
//! (`fused.matmul_moe_route`) and mega-MoE (`fused.moe_mega`) take their
//! fires rather than declining to their primitives (the mega-MoE where the
//! kernel takes the row's experts and its routes reach
//! `kernel.moe.mega_bf16_from`), and the row answers what it answers with
//! both fusions off.
//!
//! DeepSeek-V4's biased sqrt-softplus router answers its pair to the bit.
//! Qwen3-MoE's and GLM-4.5-Air's fused legs move no logit past 1% of the
//! logits' spread, nor the top token. DeepSeek-V4's miniature, five layers
//! of hyper-connections over random planes, moves 5 to 17% (RMS, of the
//! logits' deviation) under nothing but another tuning of its apart kernels,
//! the registry's defaults; its mega-MoE is held to that: no further from
//! the apart row than the defaults' apart row is, and its top token one
//! that tuning's reach could also pick.
//!
//! A mega-MoE the kernel declines loses nothing: its row fires what the row
//! with the mega rule off fires, op for op, and answers it to the bit (GLM's
//! 1408-wide experts, a decode's few routes, Qwen3.5's gated shared expert
//! under the micro's narrow banks).
//!
//! The fusions are turned off as a user turns them off, by a `[fuse]` file
//! in the profile directory the load resolves; the walk is eager
//! (`engine.graphs = "off"`) so each fire's count is that fire's alone.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Knobs, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{Fused, Linear, Operation, ParamSource, Trace};

const CEILING: u32 = 256;
const PAGE: u32 = 16;

/// Routed rows of bf16 planes: Qwen3-MoE's softmax router, GLM-4.5-Air's
/// biased sigmoid, DeepSeek-V4's biased sqrt-softplus (its hashed layers
/// routed apart) and Qwen3.5's softmax under a gated shared expert, each MLP
/// of dense banks a mega-MoE; and whether the mega-MoE kernel takes the
/// row's experts. GLM-4.5-Air's are 1408 wide, not the multiple of 256 its
/// GLU tiles need, and the Qwen3.5 micro's 512-wide rows are under the
/// kernel's 1024-column chunk, so each of their fires is declined.
const ROWS: [(&str, bool); 4] = [
    ("qwen3-30b-a3b-mini-bf16-kv-bf16", true),
    ("glm45-air-mini-bf16-kv-bf16", false),
    ("dsv4-flash-mini-bf16-kv-bf16", true),
    ("qwen3-a3b-micro-bf16-kv-bf16", false),
];

/// The row whose fused router is its pair's to the bit, and whose mega-MoE
/// is held to its apart kernels' own reach under another tuning.
const CHAOTIC: &str = "dsv4-flash-mini-bf16-kv-bf16";

/// The share of the flags-off logits' spread a fused leg may move a logit:
/// the kernels' own bf16 bound, carried through the row.
const SPREAD_SHARE: f32 = 0.01;

/// Each op's fires over one fire.
type Fires = std::collections::BTreeMap<&'static str, u64>;

/// What one load answers: its logits, the fires each op counted over its one
/// fire, and that fire's time.
struct Answer {
    logits: Vec<f32>,
    fires: Fires,
    took: std::time::Duration,
}

impl Answer {
    fn fired(&self, op: &str) -> u64 {
        self.fires.get(op).copied().unwrap_or(0)
    }
}

/// The ops whose fires the fused kernels count under.
const GATE: &str = "linear.moe_gate";
const MEGA: &str = "fused.moe_mega";

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

/// Which fusions a load keeps.
#[derive(Clone, Copy, Debug)]
struct Flags {
    router: bool,
    mega: bool,
}

const OFF: Flags = Flags {
    router: false,
    mega: false,
};

/// What a load runs under: the chip's own profile with a user file stating
/// `Flags`, or the registry's defaults with the fusions off.
#[derive(Clone, Copy, Debug)]
enum Tuning {
    Chip(Flags),
    Defaults,
}

/// What `prompt` answers under `trace` and `tuning` (a chip profile read
/// through a user file in `profiles`).
fn logits(
    trace: &Trace,
    container: &Path,
    profiles: &Path,
    device: &str,
    tuning: Tuning,
    prompt: &[u32],
) -> Answer {
    let knobs = match tuning {
        Tuning::Chip(flags) => {
            let user = engine_cuda::chip::SHELF.user_path(profiles, device);
            std::fs::create_dir_all(user.parent().expect("a user file sits in a directory"))
                .expect("the user profile directory");
            std::fs::write(
                &user,
                format!(
                    "api = 1\n\n[engine]\ngraphs = \"off\"\n\n[fuse]\nmatmul_moe_route = {}\n\
                     moe_mega = {}\n",
                    flags.router, flags.mega
                ),
            )
            .expect("the user profile");
            Knobs {
                profiles: Some(profiles.to_path_buf()),
                ..Knobs::default()
            }
        }
        Tuning::Defaults => Knobs {
            chip: Some(engine_cuda::chip::profile(
                "[engine]\ngraphs = \"off\"\n[fuse]\nmatmul_moe_route = false\nmoe_mega = false\n",
            )),
            ..Knobs::default()
        },
    };
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
        context: CEILING,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs,
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
    shell.open(0).unwrap();
    let word = trace.facts.word(&Request::new(prompt.len() as u32, false));
    let before = kernels_cuda::jit::fire_counts();
    let began = std::time::Instant::now();
    let mut out = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens: prompt,
        }])
        .unwrap();
    let took = began.elapsed();
    let fires = kernels_cuda::jit::fire_counts()
        .into_iter()
        .map(|(op, n)| (op, n - before.get(op).copied().unwrap_or(0)))
        .filter(|(_, n)| *n > 0)
        .collect();
    Answer {
        logits: out.remove(0),
        fires,
        took,
    }
}

/// The largest |a - b|.
fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The RMS of `got - want`, over the RMS of `want`'s deviation from its
/// mean.
fn relative_rms(got: &[f32], want: &[f32]) -> f32 {
    let n = want.len() as f32;
    let mean = want.iter().sum::<f32>() / n;
    let deviation = (want.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n).sqrt();
    let rms = (got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    rms / deviation
}

fn top(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("a logit")
}

fn check(name: &str, mega_takes: bool, device: &str) {
    // The micro is served from its fixture's own spelling, not the list.
    let spelled;
    let row = match poem_compiler::catalog::deployment(name) {
        Some(row) => row,
        None => {
            spelled = poem_compiler::catalog::repository()
                .parse(name)
                .unwrap_or_else(|| panic!("`{name}` spells a deployment"));
            &spelled
        }
    };
    let trace = row.trace(Platform::Cuda);
    let fused = poem_compiler::fuse::fuse(trace.clone(), engine_cuda::FUSIONS.iter());
    let formed = |which: fn(&Fused) -> bool| {
        fused
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Fused(op) if which(op)))
            .count() as u64
    };
    let routers = formed(|op| matches!(op, Fused::MatmulMoeRoute { .. }));
    let megas = formed(|op| matches!(op, Fused::MoeMega { .. }));
    assert!(
        routers > 0 && megas > 0,
        "{name}: {routers} routers and {megas} mega-MoEs fold"
    );

    let dir = std::env::temp_dir().join(format!(
        "pie-fired-moe-{}-{}",
        name.replace('.', "_"),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let _scratch = Scratch(dir.clone());
    let container = dir.join("model.zt");
    write_checkpoint(&container, &trace);
    let profiles = dir.join("profiles");

    let vocab = row.model.vocab.max(1);
    // A decode's one row, its routes under `kernel.moe.mega_bf16_from` so
    // the mega-MoE declines; and prefills whose routes it takes, under
    // `kernel.moe.gate_tc5_until` rows so the router takes them too.
    for count in [1u32, 41, 200] {
        let prompt: Vec<u32> = (0..count).map(|t| (t * 7 + 11) % vocab.min(4096)).collect();
        let run = |tuning| logits(&trace, &container, &profiles, device, tuning, &prompt);
        // A shape's first load times its tactics (each trial a fire), so the
        // rows a declined mega-MoE is held to are each a second load's.
        let warm = |tuning| {
            run(tuning);
            run(tuning)
        };
        let off = warm(Tuning::Chip(OFF));
        assert_eq!(
            (off.fired(GATE), off.fired(MEGA)),
            (0, 0),
            "{name} over {count} rows: the fusions off fire neither"
        );
        let apart_row = off.logits.clone();
        let apart = &apart_row;
        let spread = apart.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - apart.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread > 1e-3,
            "{name}: the logits are a rectangle nothing wrote"
        );
        // The apart row's own reach under another tuning, where a mega-MoE
        // is held to it.
        let retuned = (name == CHAOTIC && count > 1).then(|| run(Tuning::Defaults).logits);
        // The row with the mega rule off, under each router flag: what a
        // declined mega-MoE answers.
        let mut unmega = vec![(false, off)];
        for flags in [
            Flags {
                router: true,
                mega: false,
            },
            Flags {
                router: false,
                mega: true,
            },
            Flags {
                router: true,
                mega: true,
            },
        ] {
            let answer = if flags.mega {
                run(Tuning::Chip(flags))
            } else {
                warm(Tuning::Chip(flags))
            };
            let (got, gate, mega) = (&answer.logits, answer.fired(GATE), answer.fired(MEGA));
            let what = format!("{name} over {count} rows, {flags:?}");
            assert_eq!(
                gate,
                if flags.router { routers } else { 0 },
                "{what}: each folded router fires the fused gate"
            );
            let takes = flags.mega && mega_takes && count > 1;
            assert_eq!(
                mega,
                if takes { megas } else { 0 },
                "{what}: each mega-MoE fires where the kernel takes its experts and its \
                 routes reach the bf16 floor"
            );
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{what}: a non-finite logit"
            );
            let (far, rms) = (worst(got, apart), relative_rms(got, apart));
            eprintln!(
                "{what}: {gate} gate and {mega} mega fires in {:?}; max |d| {far} of spread \
                 {spread}, RMS {rms:.4} of the logits' deviation",
                answer.took
            );
            if flags.mega && !takes {
                let (_, without) = unmega
                    .iter()
                    .find(|(router, _)| *router == flags.router)
                    .expect("the row without the mega rule ran first");
                eprintln!(
                    "{what}: declined in {:?}, the rule off in {:?}",
                    answer.took, without.took
                );
                assert_eq!(
                    answer.fires, without.fires,
                    "{what}: a declined mega-MoE fires what the row without the rule fires"
                );
                assert!(
                    got.iter()
                        .zip(&without.logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: a declined mega-MoE answers the row without the rule to the bit"
                );
            }
            if name == CHAOTIC && !flags.mega {
                assert!(
                    got.iter()
                        .zip(apart)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: the fused sqrt-softplus router answers its pair to the bit"
                );
            }
            match &retuned {
                Some(retuned) if takes => {
                    let (reach, reach_rms) = (worst(retuned, apart), relative_rms(retuned, apart));
                    eprintln!(
                        "{what}: the defaults' apart row is {reach} away, RMS {reach_rms:.4}"
                    );
                    assert!(
                        rms <= reach_rms,
                        "{what}: RMS {rms} past the apart row's own {reach_rms} under the defaults"
                    );
                    let (want, picked) = (top(apart), top(got));
                    assert!(
                        picked == want || apart[want] - apart[picked] <= reach,
                        "{what}: top token {picked} for {want}, {} below it, past a tie the \
                         defaults' reach of {reach} could break",
                        apart[want] - apart[picked]
                    );
                }
                _ => {
                    let bound = SPREAD_SHARE * spread;
                    assert!(
                        far <= bound,
                        "{what}: max |d| {far} past {bound} (1% of {spread})"
                    );
                    assert_eq!(top(got), top(apart), "{what}: the top token moves");
                }
            }
            if !flags.mega {
                unmega.push((flags.router, answer));
            }
        }
    }
}

#[test]
fn a_fired_router_and_mega_moe_answer_what_their_legs_answer() {
    let Some((device, facts)) = engine_cuda::chip::probe(0) else {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    };
    if facts.get(engine_cuda::chip::ARCH) != Some(100) {
        eprintln!(
            "skipping: {device} is not sm_100, whose profile fires the fused router and \
             the mega-MoE"
        );
        return;
    }
    kernels_cuda::jit::count_fires();
    let failed: Vec<String> = ROWS
        .iter()
        .filter_map(|&(name, mega_takes)| {
            std::panic::catch_unwind(|| check(name, mega_takes, &device))
                .err()
                .map(|why| {
                    let why = why
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| why.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    format!("{name}: {why}")
                })
        })
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
