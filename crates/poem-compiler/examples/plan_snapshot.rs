//! A fingerprint of everything the compiler decides for the catalog: every
//! listed deployment and its splits, on every platform, as traced and
//! lowered, its weight cuts inferred, and its plan under the profiles the
//! engines compile with (and with every fusion rule applied). One line an
//! item, `name<TAB>hash<TAB>bytes` of its `Debug` text (or its refusal), so two
//! trees' outputs compare with `diff`; `--dump <dir>` also writes each text.
//!
//! `cargo run --release -p pie-poem-compiler --example plan_snapshot -- [--dump DIR] > manifest.tsv`

use std::fmt::Debug;
use std::io::Write;

use poem_compiler::catalog::{deployments, splits};
use poem_compiler::{Budget, Budgets, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_compiler::{VoxelLadder, compile_axes, fuse, shard};
use poem_ir::{ParamSource, Platform, RowAxis, Trace, Ty};

const PLATFORMS: [Platform; 6] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
    Platform::Xla,
    Platform::Cerebras,
];

/// The profiles the engines and tests compile under: the default, one that
/// forces conditionals, the DAG scheduler with side streams, and that with
/// generated regions and anchors.
fn profiles() -> Vec<(&'static str, DeviceProfile)> {
    let dag = DeviceProfile {
        dag_streams: true,
        side_streams: 2,
        ..DeviceProfile::default()
    };
    vec![
        ("default", DeviceProfile::default()),
        (
            "forced",
            DeviceProfile {
                fat_region_us: 0.0,
                cond_fixed_us: 0.5,
                cond_per_arm_us: 0.0,
                ..DeviceProfile::default()
            },
        ),
        (
            "synth",
            DeviceProfile {
                synth_regions: true,
                synth_anchors: true,
                ..dag.clone()
            },
        ),
        ("dag", dag),
    ]
}

fn states(trace: &Trace, axis: RowAxis) -> bool {
    trace.values.iter().any(|decl| {
        matches!(&decl.ty, Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(axis))
    })
}

/// The budgets a serving shell derives for `trace` (the tests' `budgets_of`).
fn budgets_of(trace: &Trace) -> Budgets {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    let budget = Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
        readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
    };
    let mut budgets = Budgets::of(budget.clone());
    if states(trace, RowAxis::Patches) {
        let max_patches = budget.max_tokens.clamp(PATCH_LATTICE_FLOOR, 4096);
        let mut buckets = Vec::new();
        let mut rung = PATCH_LATTICE_FLOOR;
        while rung < max_patches {
            buckets.push(rung);
            rung = rung.saturating_mul(2);
        }
        buckets.push(max_patches);
        budgets = budgets.with_patches(PatchLadder {
            max_images: (max_patches / PATCH_LATTICE_FLOOR).max(1),
            max_patches,
            buckets,
        });
    }
    if states(trace, RowAxis::Voxels) {
        budgets = budgets.with_voxels(VoxelLadder::new(4096, 4));
    }
    budgets
}

struct Out {
    lines: Vec<String>,
    dump: Option<std::path::PathBuf>,
}

impl Out {
    fn item(&mut self, name: &str, text: &str) {
        // FNV-1a over the text, 64 bits: stable across runs and toolchains.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        self.lines.push(format!("{name}\t{h:016x}\t{}", text.len()));
        if let Some(dir) = &self.dump {
            let file = dir.join(name.replace(['/', ' '], "_"));
            std::fs::write(file, text).expect("dump");
        }
    }

    fn debug<T: Debug, E: std::fmt::Display>(&mut self, name: &str, r: &Result<T, E>) {
        let text = match r {
            Ok(v) => format!("{v:#?}"),
            Err(e) => format!("refused: {e}"),
        };
        self.item(name, &text);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dump = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from);
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("dump dir");
    }
    let rows: Vec<_> = deployments().chain(splits()).collect();
    let threads = std::env::var("PLAN_SNAPSHOT_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, Vec<String>)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(row) = rows.get(i) else { break };
                        let mut out = Out {
                            lines: Vec::new(),
                            dump: dump.clone(),
                        };
                        snapshot(row, &mut out);
                        mine.push((i, out.lines));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("worker"))
            .collect()
    });
    done.sort_by_key(|(i, _)| *i);

    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    for line in done.iter().flat_map(|(_, lines)| lines) {
        writeln!(w, "{line}").expect("stdout");
    }
}

/// Every item of one catalog row, on every platform.
fn snapshot(row: &poem_compiler::catalog::Deployment, out: &mut Out) {
    let rules = fuse::rules::all();
    for platform in PLATFORMS {
        let at = format!("{}@{}", row.name, platform.backend());
        let trace = match row.try_trace(platform) {
            Ok(trace) => trace,
            Err(why) => {
                out.item(&format!("{at}/trace"), &format!("refused: {why}"));
                continue;
            }
        };
        out.item(&format!("{at}/trace"), &format!("{trace:#?}"));
        out.item(
            &format!("{at}/cuts"),
            &format!("{:#?}", shard::infer::infer(&trace)),
        );
        let budgets = budgets_of(&trace);
        for (p, profile) in &profiles() {
            out.debug(
                &format!("{at}/plan.{p}"),
                &compile_axes(&trace, &budgets, profile),
            );
        }
        let fused = fuse::fuse(trace, rules.iter());
        out.debug(
            &format!("{at}/plan.fused"),
            &compile_axes(&fused, &budgets_of(&fused), &DeviceProfile::default()),
        );
    }
}
