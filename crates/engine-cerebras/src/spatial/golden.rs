//! Host-protocol goldens: every design's host side run on a recording
//! [`Device`](super::device::Device), each phase's calls pinned.
//!
//! A scenario is a design with representative parameters (the build
//! presets, the parameters the module tests use, the hardware Qwen3-8B
//! builds), a small deterministic synthetic checkpoint, and a script:
//!
//! - `load`: the routes and their `rbits` read, `init`, every weight's
//!   placement, the VM's ring and `gmode`, the warm-up launch;
//! - `prefill` and `decode`: a launch of every lane forcing its prompt, then
//!   one pass with the logits read;
//! - `copy`, `page_out`, `page_in`: page copies and the host tier (or their
//!   refusal);
//! - `vm`: a program loaded on the VM (one program, or per lane on each set)
//!   and a VM launch sampling one lane of each set; `vm_all`, a launch
//!   sampling every lane (DEFER's bit 16 where the build defers).
//!
//! Some scenarios leave a symbol out of the program (no `init`, no
//! `yscale`); the recorder refuses calls on it and logs them. One test
//! loads two scenarios through the driver's own reader (`read_design`) from
//! a program directory (manifest and `out.json`, no PE images) to the same
//! pins.
//!
//! Each recorded call is its kind, symbol, rectangle, elements a PE, width,
//! word count and a sha256 of the words; a phase pins its call count and the
//! sha256 of its lines. The generated wafer programs are fixed (validated on
//! hardware), so the host's side must not drift from what they expect: a
//! route table built by another design's rule, a table at the wrong PEs, a
//! missing flag word all change a pin. Pins change only with the programs.
//!
//! `PIE_GOLDEN_DUMP=<dir>` writes every phase's call lines there (for a
//! diff); `PIE_GOLDEN_PRINT=1` prints the pin table.

use std::collections::BTreeSet;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::build::{self, ServeConfig, VmImage, presets};
use super::ckpt::Ckpt;
use super::codegen::Design;
use super::device::Recorder;
use super::guest4::lane::Program;
use super::guest4::slots::LaneProgram;
use super::guest4::wafer::Seed;
use super::guest4::{Home, Table};
use super::plan::{SpatialPlan, WaferDims};
use super::program::ProgramSpec;
use super::serve2d::BatchLane;
use super::wafer::{Logits, WaferModel, Weights};

type Params = Vec<(String, String)>;

/// `K:V,...` as params.
fn kv(s: &str) -> Params {
    s.split(',')
        .filter_map(|p| p.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

fn with(mut p: Params, extra: &[(&str, usize)]) -> Params {
    for &(k, v) in extra {
        match p.iter_mut().find(|e| e.0 == k) {
            Some(e) => e.1 = v.to_string(),
            None => p.push((k.to_string(), v.to_string())),
        }
    }
    p
}

fn num(p: &Params, k: &str) -> Option<usize> {
    p.iter()
        .find(|(n, _)| n == k)
        .and_then(|(_, v)| v.parse().ok())
}

/// The serve2d (`GRID` 1) build of the tiny preset model.
fn serve2d_params(extra: &[(&'static str, usize)]) -> Params {
    let (plan, mut cfg) = presets::pipe_config();
    cfg.goals.pipeline = false;
    cfg.vm = None;
    cfg.mmax = None;
    cfg.hb = None;
    cfg.hlm = None;
    for &(k, v) in extra {
        cfg.pins.set(k, v).expect("a pin");
    }
    let plan = SpatialPlan::derive(&plan.spec, WaferDims::CS3, cfg.goals).expect("a tiny plan");
    build::params(&plan, &cfg).expect("serve2d params").1
}

/// The tiny serve2d_pipe preset, and with a second (per-lane-program) VM
/// set B.
fn pipe_params(set_b: bool) -> Params {
    let (plan, mut cfg) = presets::pipe_config();
    if set_b {
        if let Some(vm) = cfg.vm.as_mut() {
            vm.lanes = Some(2);
        }
        let img = VmImage {
            max_ops: 64,
            nreg: 8,
            pool_words: 128,
            feat: 7,
            lanes: Some(2),
            lane_programs: Some(900),
            ..cfg.vm.expect("the preset's VM")
        };
        cfg.vm_b = Some((img, 1));
    }
    let cfg = ServeConfig { ..cfg };
    build::params(&plan, &cfg).expect("serve2d_pipe params").1
}

/// serve8b's simulator build (the module tests'): B 2, page-copy kernels.
const S8B_SIM: &str = "PX:8,PY:16,Hc:5,HR:76,Dc:8,Gs:8,Qh:4,Ic:24,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,\
    KV16:1,KVCP:1,PROF:0,TIME:0,GW:2,GH:3,GS:2,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,PASSES:3,MODE:0,\
    OVL:0,VMC:1,CAP:16,iters:1";

/// A small serve8b_region: 15 layers in 11 units, host routes (`HROUTE`).
const S8BR_SIM: &str = "PX:8,PY:16,Hc:3,Dc:8,Gs:8,Ic:24,RPX:16,RPY:24,RHc:2,RDc:4,RGs:16,RIc:12,RNL:2,RGY:8,\
    RC:2,RR:2,NCOL:2,RTR:5,HR:40,Qh:4,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,KV16:1,KVCP:0,PROF:0,TIME:0,HROUTE:1,\
    ROT8:1,NOLAYER:0,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,PASSES:3,MODE:0,OVL:0,VMC:1,CAP:16,iters:1";

/// The hardware Qwen3-8B builds (routes and launches only).
const S8B_HW: &str = "PX:128,PY:144,Hc:29,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,W16:1,KV16:1,\
    GW:5,GH:8,GS:4,HEADMODE:2,HH:1152,Vr:132,Gx:8,VMC:1,vocab:151936,CAP:64,B:1,NB:1,KVCP:1";
const S8BR_HW: &str = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,\
    RNL:3,RGY:16,RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
    HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,OVL:0,VMC:1,\
    CAP:64,iters:1,B:1,NP:3";

/// The model a program's params describe, as checkpoint shapes.
fn shapes(g: &ProgramSpec, untied: bool) -> Vec<(String, Vec<usize>)> {
    let (hid, d, qh, vocab) = (g.hidden(), g.head_dim(), g.qh(), g.vocab());
    let nkv = g.px() / g.gs();
    let (qdim, inter) = (nkv * qh * d, g.px() * g.ic());
    let mut out = vec![
        ("embed".to_string(), vec![vocab, hid]),
        ("final_norm".to_string(), vec![hid]),
    ];
    if untied {
        out.push(("head".to_string(), vec![vocab, hid]));
    }
    for l in 0..g.layers() {
        for (name, shape) in [
            ("q", vec![qdim, hid]),
            ("k", vec![nkv * d, hid]),
            ("v", vec![nkv * d, hid]),
            ("o", vec![hid, qdim]),
            ("gate", vec![inter, hid]),
            ("up", vec![inter, hid]),
            ("down", vec![hid, inter]),
            ("q_norm", vec![d]),
            ("k_norm", vec![d]),
            ("attn_norm", vec![hid]),
            ("mlp_norm", vec![hid]),
        ] {
            out.push((format!("layers.{l}.{name}"), shape));
        }
    }
    out
}

/// A small VM program: a pool home holding the token, one output word.
fn vm_table() -> (Table, Vec<Seed>) {
    let home = Home {
        chan: 0,
        kind: "pool",
        at: 0,
        n: 2,
        dtype: eta_ir::Dtype::U32,
    };
    let seeds = vec![Seed::of(&home, vec![5, 9])];
    let table = Table {
        ins: vec![[1, 0, 0, 0, 0, 0], [2, 0, 1, 0, 1, 0]],
        pre_len: 0,
        homes: vec![home],
        outputs: vec![(0, 0, 1, eta_ir::Dtype::U32)],
        wide_outputs: Vec::new(),
        per_step: Vec::new(),
        tok_home: Some(0),
        nreg: 2,
        pool_words: 4,
        feat: 0,
        remat: false,
        bank_regs: Vec::new(),
        bank_pool: vec![(0, 2)],
    };
    (table, seeds)
}

/// What a scenario runs past the load.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Script {
    /// A one-sequence program: forward, greedy and the head VM's step.
    Sequence,
    /// A batched program: launches, page copies, the host tier, the VM.
    Batched,
    /// A hardware shape: the routes and launches only (no weights).
    Launches,
}

struct Scenario {
    name: &'static str,
    built: Design,
    params: Params,
    script: Script,
    /// Symbols the program does not export (the recorder refuses them).
    missing: &'static [&'static str],
}

fn scenarios() -> Vec<Scenario> {
    let s = |name, built, params, script| Scenario {
        name,
        built,
        params,
        script,
        missing: &[],
    };
    // serve2d's one-sequence grids (GRID 0, compiled outside PIE) with the
    // head's VM (guest4hm).
    let seq = with(
        serve2d_params(&[]),
        &[
            ("GRID", 0),
            ("VM", 1),
            ("max_ops", 48),
            ("nreg", 6),
            ("pool_words", 64),
            ("feat", 99),
            ("ring_w", 1),
            ("ring_d", 32),
            ("Tc", 8),
        ],
    );
    let pipe_ab = pipe_params(true);
    let conc = with(pipe_ab.clone(), &[("VMCONC", 1), ("DEFER", 1)]);
    vec![
        s(
            "serve2d",
            Design::Serve2d,
            serve2d_params(&[]),
            Script::Batched,
        ),
        s(
            "serve2d_kvst",
            Design::Serve2d,
            serve2d_params(&[("KVST", 2)]),
            Script::Batched,
        ),
        s(
            "serve2d_seq_vm",
            Design::Serve2d,
            seq.clone(),
            Script::Sequence,
        ),
        s(
            "serve2d_pipe",
            Design::Serve2dPipe,
            pipe_params(false),
            Script::Batched,
        ),
        s(
            "serve2d_pipe_ab",
            Design::Serve2dPipe,
            pipe_ab,
            Script::Batched,
        ),
        s(
            "serve2d_pipe_conc",
            Design::Serve2dPipe,
            conc,
            Script::Batched,
        ),
        s("serve8b", Design::Serve8b, kv(S8B_SIM), Script::Batched),
        s(
            "serve8b_kvst",
            Design::Serve8b,
            with(kv(S8B_SIM), &[("KVCP", 0), ("KVST", 1)]),
            Script::Batched,
        ),
        s(
            "serve8b_region",
            Design::Serve8bRegion,
            kv(S8BR_SIM),
            Script::Batched,
        ),
        s(
            "serve8b_region_kvst",
            Design::Serve8bRegion,
            with(kv(S8BR_SIM), &[("KVST", 1)]),
            Script::Batched,
        ),
        s(
            "serve8b_region_pipe",
            Design::Serve8bRegionPipe,
            with(kv(S8BR_SIM), &[("PIPE", 1), ("MMAX", 2), ("NP", 1)]),
            Script::Batched,
        ),
        // serve8b_region with the VM block east of the strip (VMX): the
        // host loads and fires it as any build (its VM's own path is not
        // the driver's).
        s(
            "serve8b_region_vmx",
            Design::Serve8bRegion,
            with(
                kv(S8BR_SIM),
                &[
                    ("Vr", 160),
                    ("VMC", 0),
                    ("VMX", 1),
                    ("VMREAL", 1),
                    ("max_ops", 104),
                    ("nreg", 12),
                    ("pool_words", 211),
                    ("bank_words", 768),
                    ("vm_lanes", 2),
                ],
            ),
            Script::Batched,
        ),
        // A program without `init` (nothing applies the routes), and one
        // without `yscale` (its `y` is not the logits: a logits request is
        // refused).
        Scenario {
            missing: &["init"],
            ..s(
                "serve2d_no_init",
                Design::Serve2d,
                serve2d_params(&[]),
                Script::Batched,
            )
        },
        Scenario {
            missing: &["yscale"],
            ..s(
                "serve2d_seq_no_yscale",
                Design::Serve2d,
                seq,
                Script::Sequence,
            )
        },
        s("hw_serve8b", Design::Serve8b, kv(S8B_HW), Script::Launches),
        s(
            "hw_serve8b_region",
            Design::Serve8bRegion,
            kv(S8BR_HW),
            Script::Launches,
        ),
        s(
            "hw_serve8b_region_pipe",
            Design::Serve8bRegionPipe,
            with(
                kv(S8BR_HW),
                &[("CAP", 32), ("NP", 1), ("PIPE", 1), ("MMAX", 12)],
            ),
            Script::Launches,
        ),
    ]
}

/// Every scenario's design and params (the program layer's round-trip
/// check reads them too).
pub(super) fn scenario_params() -> Vec<(&'static str, Design, Params)> {
    scenarios()
        .into_iter()
        .map(|sc| (sc.name, sc.built, sc.params))
        .collect()
}

/// A phase's calls: (phase, lines).
type Phases = Vec<(&'static str, Vec<String>)>;

/// Lane `b` of `n`: its prompt and its pages (consecutive, so on distinct
/// rows).
fn lanes(
    n: usize,
    pool: u32,
    pos: usize,
    npg: u32,
    forced: impl Fn(usize) -> Vec<u32>,
) -> Vec<BatchLane> {
    (0..n)
        .map(|b| BatchLane {
            pos: pos as u32,
            slot: pos as u32,
            pages: (0..npg).map(|i| (b as u32 * npg + i) % pool).collect(),
            visible: None,
            forced: forced(b),
        })
        .collect()
}

fn run(sc: &Scenario) -> Phases {
    run_spec(
        sc,
        ProgramSpec::from_params(sc.built, &sc.params).expect("the scenario's design"),
    )
}

/// `sc`'s script on the program `design`.
fn run_spec(sc: &Scenario, design: ProgramSpec) -> Phases {
    let rec = if sc.script == Script::Launches {
        Recorder::default().forgetful()
    } else {
        Recorder::default()
    }
    .without(sc.missing);
    let grid = &design;
    // The VM's ring as the image holds it: [depth, width(, lanes)].
    let ring = grid.vm_abi().map_or(0, |abi| abi.ring_words);
    let launch = grid.launch_lanes();
    for (vm, lanes_key) in [(grid.vm(), "vm_lanes"), (grid.vm_b(), "b_vm_lanes")] {
        if let Some(vm) = vm {
            let (x, y) = vm.hub();
            let lanes = num(&sc.params, lanes_key).unwrap_or(launch) as u32;
            let depth = num(&sc.params, "ring_d").unwrap_or(32) as u32;
            let width = num(&sc.params, "ring_w").unwrap_or(1) as u32;
            rec.preset("gring", x as i32, y as i32, &[depth, width, lanes][..ring]);
        }
    }
    let untied = grid.head.kind == super::program::HeadKind::Untied;
    let ckpt = Ckpt::synthetic(shapes(&design, untied));
    let weights = if sc.script == Script::Launches {
        Weights::Skip
    } else {
        Weights::Place(&ckpt)
    };
    let paged = grid.paged();
    let logit_lanes = grid.logit_lanes();
    let mut phases: Phases = Vec::new();
    let mut m =
        match WaferModel::on_device(Box::new(rec.clone()), design, weights, Some(1.0e6), ring) {
            Ok(m) => m,
            Err(e) => panic!("{}: the load fails: {e}", sc.name),
        };
    phases.push(("load", rec.take()));
    let mut note = |phases: &mut Phases, phase: &'static str, ok: bool| {
        let mut calls = rec.take();
        if !ok {
            calls.push("refused".into());
        }
        phases.push((phase, calls));
    };
    match sc.script {
        Script::Sequence => {
            let ok = m.forward(&[3, 1, 4], 0, Logits::Last).is_ok();
            note(&mut phases, "forward", ok);
            let ok = m.greedy(&[5, 9], 3, 3).is_ok();
            note(&mut phases, "greedy", ok);
            let (table, seeds) = vm_table();
            let ok = m.vm_load(&table, &seeds).is_ok() && m.vm_step(7, 8).is_ok();
            note(&mut phases, "vm", ok);
        }
        Script::Batched | Script::Launches => {
            let shape = paged.expect("a batched build");
            let pool = shape.pages() as u32;
            let base: [&[u32]; 4] = [&[3, 1, 4], &[7, 2], &[11, 8, 5], &[6, 6]];
            let npg = 2u32.min(pool / launch as u32).max(1);
            let one = lanes(launch, pool, 0, npg, |b| base[b % 4].to_vec());
            let ok = m.fire_batch(&one, 5, false).is_ok();
            note(&mut phases, "prefill", ok);
            let two = lanes(launch, pool, 5, npg, |b| vec![b as u32 + 1]);
            let ok = m.fire_batch(&two, 1, logit_lanes > 0).is_ok();
            note(&mut phases, "decode", ok);
            if sc.script == Script::Batched {
                let ok = m.copy_pages(&[(0, pool - 1), (1, pool - 2)]).is_ok();
                note(&mut phases, "copy", ok);
                let out = m.page_out(&[1, 2]);
                note(&mut phases, "page_out", out.is_ok());
                let back: Vec<(u32, Vec<u32>)> = match out {
                    Ok(pages) => pages
                        .into_iter()
                        .zip([pool - 3, pool - 4])
                        .map(|(w, p)| (p, w))
                        .collect(),
                    Err(_) => Vec::new(),
                };
                let ok = !back.is_empty() && m.page_in(&back).is_ok();
                note(&mut phases, "page_in", ok);
                vm_script(&mut m, &two, &mut phases, &mut note);
            }
        }
    }
    phases
}

/// The VM phase of a batched build with one: set A's program for every
/// lane (or per lane), set B's per lane, a launch sampling lane 0 of each.
fn vm_script(
    m: &mut WaferModel,
    lanes: &[BatchLane],
    phases: &mut Phases,
    note: &mut impl FnMut(&mut Phases, &'static str, bool),
) {
    let (na, nb) = m.vm_lanes();
    if na == 0 {
        return;
    }
    let (table, seeds) = vm_table();
    let program = Arc::new(Program {
        table,
        epilogue: 0,
        homes: BTreeSet::new(),
        logits: "logits".into(),
    });
    let mut ok = true;
    for (set, n) in [(0, na), (1, nb)] {
        if n == 0 {
            continue;
        }
        let lprog = if set == 0 {
            m.vm_shape().is_some_and(|(l, _)| l.lprog)
        } else {
            m.vm_shape_b().is_some_and(|(l, _)| l.lprog)
        };
        if lprog {
            let mut programs = vec![None; n];
            programs[0] = Some(LaneProgram {
                key: 1,
                program: program.clone(),
            });
            let mut lane_seeds = vec![None; n];
            lane_seeds[0] = Some(seeds.clone());
            ok &= m
                .vm_place_lanes(set, &programs, &lane_seeds, &vec![None; n])
                .is_ok();
        } else {
            ok &= m.vm_load_lanes(&program.table).is_ok();
            ok &= m.vm_seed_bank(0, &seeds).is_ok();
        }
    }
    let mut sampled = vec![false; na + nb];
    sampled[0] = true;
    if nb > 0 {
        sampled[na] = true;
    }
    ok &= m.vm_fire(lanes, &sampled, &[]).is_ok();
    note(phases, "vm", ok);
    // Every lane on the VM (with DEFER, the head defers their tokens: the
    // compute argument's bit 16).
    let all = vec![true; na + nb];
    let ok = m.vm_fire(lanes, &all, &[]).is_ok();
    note(phases, "vm_all", ok);
}

fn digest(lines: &[String]) -> String {
    let mut h = Sha256::new();
    for l in lines {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    h.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// (scenario/phase, calls, sha256 of the call lines).
const PINS: &[(&str, usize, &str)] = &[
    ("serve2d/load", 116, "6a7bea12ad749aa0c45ee7a5a165e7ad"),
    ("serve2d/prefill", 5, "a70e5a97f91aec231163c76f451cd693"),
    ("serve2d/decode", 8, "4c0d76fb471e4995e03211dd907ed984"),
    ("serve2d/copy", 4, "694493c8e05e1e00482941d8444891b3"),
    ("serve2d/page_out", 32, "a2615d7ab285682f5c7f491427d995d3"),
    ("serve2d/page_in", 64, "e8ef2c7183a3784f4eb4dbcefac75c42"),
    ("serve2d_kvst/load", 116, "6a7bea12ad749aa0c45ee7a5a165e7ad"),
    (
        "serve2d_kvst/prefill",
        5,
        "a70e5a97f91aec231163c76f451cd693",
    ),
    ("serve2d_kvst/decode", 8, "4c0d76fb471e4995e03211dd907ed984"),
    ("serve2d_kvst/copy", 4, "694493c8e05e1e00482941d8444891b3"),
    (
        "serve2d_kvst/page_out",
        18,
        "e9b508ab61e61649d133f403a8f9abff",
    ),
    (
        "serve2d_kvst/page_in",
        18,
        "1a54df2acd93b54523aa4cc1d40b5d60",
    ),
    (
        "serve2d_seq_vm/load",
        93,
        "d9f5b82b31fceb111428ea7c6e152e5e",
    ),
    (
        "serve2d_seq_vm/forward",
        7,
        "79f1551d89c73aabd4839620424eb5d3",
    ),
    (
        "serve2d_seq_vm/greedy",
        4,
        "dd58a313fd810e66ee4adc595e32eac9",
    ),
    ("serve2d_seq_vm/vm", 13, "9e80ec944c6ad29e73bb41c1ff7574cf"),
    ("serve2d_pipe/load", 120, "a36ded227e2af97b500343f3dc364689"),
    (
        "serve2d_pipe/prefill",
        7,
        "88ed3e245117d69f9a11acf89b3815ca",
    ),
    (
        "serve2d_pipe/decode",
        10,
        "116a7260ce6fe23eb8fb79c2cf129b8a",
    ),
    ("serve2d_pipe/copy", 4, "694493c8e05e1e00482941d8444891b3"),
    (
        "serve2d_pipe/page_out",
        32,
        "a2615d7ab285682f5c7f491427d995d3",
    ),
    (
        "serve2d_pipe/page_in",
        64,
        "e8ef2c7183a3784f4eb4dbcefac75c42",
    ),
    ("serve2d_pipe/vm", 20, "36f4e8752703c644db6c936b7622e4ee"),
    (
        "serve2d_pipe/vm_all",
        10,
        "fdd83420b44711881d1aa276472baa09",
    ),
    (
        "serve2d_pipe_ab/load",
        121,
        "39bc376e63853d7e5ee5260032c6816e",
    ),
    (
        "serve2d_pipe_ab/prefill",
        7,
        "b12b56b2c9a61ecc4bef1275a6815c66",
    ),
    (
        "serve2d_pipe_ab/decode",
        10,
        "fde95467b7147140a35954c7f97b7d9e",
    ),
    (
        "serve2d_pipe_ab/copy",
        4,
        "694493c8e05e1e00482941d8444891b3",
    ),
    (
        "serve2d_pipe_ab/page_out",
        32,
        "a2615d7ab285682f5c7f491427d995d3",
    ),
    (
        "serve2d_pipe_ab/page_in",
        64,
        "e8ef2c7183a3784f4eb4dbcefac75c42",
    ),
    ("serve2d_pipe_ab/vm", 31, "bbeccbec0bf7dd52c6c0b3dd6f3265f3"),
    (
        "serve2d_pipe_ab/vm_all",
        13,
        "e520c0ceb932a59c9b11aeea9fea7a3e",
    ),
    (
        "serve2d_pipe_conc/load",
        121,
        "39bc376e63853d7e5ee5260032c6816e",
    ),
    (
        "serve2d_pipe_conc/prefill",
        7,
        "b12b56b2c9a61ecc4bef1275a6815c66",
    ),
    (
        "serve2d_pipe_conc/decode",
        10,
        "fde95467b7147140a35954c7f97b7d9e",
    ),
    (
        "serve2d_pipe_conc/copy",
        4,
        "694493c8e05e1e00482941d8444891b3",
    ),
    (
        "serve2d_pipe_conc/page_out",
        32,
        "a2615d7ab285682f5c7f491427d995d3",
    ),
    (
        "serve2d_pipe_conc/page_in",
        64,
        "e8ef2c7183a3784f4eb4dbcefac75c42",
    ),
    (
        "serve2d_pipe_conc/vm",
        33,
        "d611b6dbad4f464ee740be90aedda436",
    ),
    (
        "serve2d_pipe_conc/vm_all",
        15,
        "6dd717a21952f43b6fba60410da0c936",
    ),
    ("serve8b/load", 82, "c7eb09b48feeff7ec0965e5f680af1dd"),
    ("serve8b/prefill", 6, "cc9215d8feac6b0f1291c2b4ff5ec3e0"),
    ("serve8b/decode", 9, "ded43a816c8edda228a3f79587af54e0"),
    ("serve8b/copy", 4, "5eb4b2c6d9206bcaca17c8dbe383fcb4"),
    ("serve8b/page_out", 20, "69b007050d80e7b78b7ea6248961820c"),
    ("serve8b/page_in", 40, "da2f079f662b1627f4f878e0adc341ba"),
    ("serve8b_kvst/load", 82, "c7eb09b48feeff7ec0965e5f680af1dd"),
    (
        "serve8b_kvst/prefill",
        6,
        "cc9215d8feac6b0f1291c2b4ff5ec3e0",
    ),
    ("serve8b_kvst/decode", 9, "ded43a816c8edda228a3f79587af54e0"),
    ("serve8b_kvst/copy", 60, "7fb6e058ee54a1f8686a21298bd00b33"),
    (
        "serve8b_kvst/page_out",
        12,
        "b8b79be4763b48c08fd4ffaedb4a10f7",
    ),
    (
        "serve8b_kvst/page_in",
        12,
        "23a9c4c9e9f478e27145837859b8bd46",
    ),
    (
        "serve8b_region/load",
        187,
        "5dbee20649af09ef45da4750703bb26e",
    ),
    (
        "serve8b_region/prefill",
        15,
        "3f0e0079fd009aea029da7bace79b591",
    ),
    (
        "serve8b_region/decode",
        18,
        "3a4f07ba3d2adae7ac5178c0af4b3dd3",
    ),
    (
        "serve8b_region/copy",
        132,
        "42854006f4beb33d6a2d6447c0ad0158",
    ),
    (
        "serve8b_region/page_out",
        44,
        "50a8103f5c7ca0ad8b91a41f830b41cf",
    ),
    (
        "serve8b_region/page_in",
        88,
        "6fbe225e35dcbd9fe6a4839aa3f3f858",
    ),
    (
        "serve8b_region_kvst/load",
        187,
        "5dbee20649af09ef45da4750703bb26e",
    ),
    (
        "serve8b_region_kvst/prefill",
        15,
        "3f0e0079fd009aea029da7bace79b591",
    ),
    (
        "serve8b_region_kvst/decode",
        18,
        "3a4f07ba3d2adae7ac5178c0af4b3dd3",
    ),
    (
        "serve8b_region_kvst/copy",
        132,
        "42854006f4beb33d6a2d6447c0ad0158",
    ),
    (
        "serve8b_region_kvst/page_out",
        24,
        "80266a747c0fd2922b97463301e6d2f0",
    ),
    (
        "serve8b_region_kvst/page_in",
        24,
        "826a4b1df17653296e72f9565847f44a",
    ),
    (
        "serve8b_region_pipe/load",
        188,
        "e571010ca4e7d00582c28514388558ca",
    ),
    (
        "serve8b_region_pipe/prefill",
        16,
        "6f107c50146ab58d3e5f4e9de5cea29f",
    ),
    (
        "serve8b_region_pipe/decode",
        19,
        "7c06198ad62eb59fc936559fb432674b",
    ),
    (
        "serve8b_region_pipe/copy",
        132,
        "42854006f4beb33d6a2d6447c0ad0158",
    ),
    (
        "serve8b_region_pipe/page_out",
        44,
        "50a8103f5c7ca0ad8b91a41f830b41cf",
    ),
    (
        "serve8b_region_pipe/page_in",
        88,
        "6fbe225e35dcbd9fe6a4839aa3f3f858",
    ),
    (
        "serve8b_region_vmx/load",
        187,
        "339dd112427cc9217e0493a7321a9289",
    ),
    (
        "serve8b_region_vmx/prefill",
        15,
        "3f0e0079fd009aea029da7bace79b591",
    ),
    (
        "serve8b_region_vmx/decode",
        18,
        "c476d149816fff1259323440066dbe9a",
    ),
    (
        "serve8b_region_vmx/copy",
        132,
        "42854006f4beb33d6a2d6447c0ad0158",
    ),
    (
        "serve8b_region_vmx/page_out",
        44,
        "50a8103f5c7ca0ad8b91a41f830b41cf",
    ),
    (
        "serve8b_region_vmx/page_in",
        88,
        "6fbe225e35dcbd9fe6a4839aa3f3f858",
    ),
    (
        "serve2d_no_init/load",
        115,
        "dcea3685a7dcc3cacbe717e6f50c9552",
    ),
    (
        "serve2d_no_init/prefill",
        5,
        "a70e5a97f91aec231163c76f451cd693",
    ),
    (
        "serve2d_no_init/decode",
        8,
        "4c0d76fb471e4995e03211dd907ed984",
    ),
    (
        "serve2d_no_init/copy",
        4,
        "694493c8e05e1e00482941d8444891b3",
    ),
    (
        "serve2d_no_init/page_out",
        32,
        "a2615d7ab285682f5c7f491427d995d3",
    ),
    (
        "serve2d_no_init/page_in",
        64,
        "e8ef2c7183a3784f4eb4dbcefac75c42",
    ),
    (
        "serve2d_seq_no_yscale/load",
        93,
        "d9f5b82b31fceb111428ea7c6e152e5e",
    ),
    (
        "serve2d_seq_no_yscale/forward",
        5,
        "2baf0c49ceed5da8e6713916d7880b1c",
    ),
    (
        "serve2d_seq_no_yscale/greedy",
        4,
        "dd58a313fd810e66ee4adc595e32eac9",
    ),
    (
        "serve2d_seq_no_yscale/vm",
        13,
        "9e80ec944c6ad29e73bb41c1ff7574cf",
    ),
    ("hw_serve8b/load", 43, "35a9778bb7c6102b059c039650c8170e"),
    ("hw_serve8b/prefill", 6, "6a4936e14d25e26f9c7983f60e4ba3f5"),
    ("hw_serve8b/decode", 9, "9af8a5d5ee15f795d4fd2593ca2b2c75"),
    (
        "hw_serve8b_region/load",
        66,
        "06678dd5597fa19f72ce53a422522088",
    ),
    (
        "hw_serve8b_region/prefill",
        24,
        "dc192d106172b6fd17f9719e4e142508",
    ),
    (
        "hw_serve8b_region/decode",
        27,
        "03d083fe4f71073f5be856e0b6d01956",
    ),
    (
        "hw_serve8b_region_pipe/load",
        67,
        "1c5f205228d4d2009f2b97479e6f10be",
    ),
    (
        "hw_serve8b_region_pipe/prefill",
        25,
        "89d9dbd4788dd6f4828c281ce2aea8d4",
    ),
    (
        "hw_serve8b_region_pipe/decode",
        28,
        "9f0e0622dbd94e780120b475379e9705",
    ),
];

/// Every scenario run once: (scenario, its phases).
fn runs() -> &'static [(Scenario, Phases)] {
    static RUNS: std::sync::OnceLock<Vec<(Scenario, Phases)>> = std::sync::OnceLock::new();
    RUNS.get_or_init(|| {
        scenarios()
            .into_iter()
            .map(|sc| {
                let phases = run(&sc);
                (sc, phases)
            })
            .collect()
    })
}

/// The host↔wafer ABI is declared once and agrees with both sides: every
/// call a scenario makes names a symbol its spec's components declare (a
/// variable at the width copied, a function), and every declared symbol is
/// one the design's generated layout exports.
/// The vmlane words (four a virtual lane) a VM launch of scenario `name`
/// writes when it samples lane 0 of each set, each set's lane 0 placed;
/// and which lanes' turns came back with a ring row.
fn vm_launch_control(name: &str) -> (Vec<u32>, Vec<bool>, Vec<bool>) {
    let sc = scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .expect("the scenario");
    let design = ProgramSpec::from_params(sc.built, &sc.params).expect("the scenario's design");
    let rec = Recorder::default();
    let ring = design.vm_abi().map_or(0, |abi| abi.ring_words);
    let launch = design.launch_lanes();
    for (vm, lanes_key) in [(design.vm(), "vm_lanes"), (design.vm_b(), "b_vm_lanes")] {
        if let Some(vm) = vm {
            let (x, y) = vm.hub();
            let lanes = num(&sc.params, lanes_key).unwrap_or(launch) as u32;
            let depth = num(&sc.params, "ring_d").unwrap_or(32) as u32;
            let width = num(&sc.params, "ring_w").unwrap_or(1) as u32;
            rec.preset("gring", x as i32, y as i32, &[depth, width, lanes][..ring]);
        }
    }
    let ckpt = Ckpt::synthetic(shapes(
        &design,
        design.head.kind == super::program::HeadKind::Untied,
    ));
    let col = (design.head_x() + design.r2x()) as i32;
    let pool = design.paged().expect("a batched build").pages() as u32;
    let mut m = WaferModel::on_device(
        Box::new(rec.clone()),
        design,
        Weights::Place(&ckpt),
        Some(1.0e6),
        ring,
    )
    .expect("the load");
    let (na, nb) = m.vm_lanes();
    let (table, seeds) = vm_table();
    let program = Arc::new(Program {
        table,
        epilogue: 0,
        homes: BTreeSet::new(),
        logits: "logits".into(),
    });
    for (set, n) in [(0, na), (1, nb)] {
        let shape = if set == 0 {
            m.vm_shape()
        } else {
            m.vm_shape_b()
        };
        if shape.is_some_and(|(l, _)| l.lprog) {
            let mut programs = vec![None; n];
            programs[0] = Some(LaneProgram {
                key: 1,
                program: program.clone(),
            });
            let mut lane_seeds = vec![None; n];
            lane_seeds[0] = Some(seeds.clone());
            m.vm_place_lanes(set, &programs, &lane_seeds, &vec![None; n])
                .expect("placed");
        } else {
            m.vm_load_lanes(&program.table).expect("loaded");
            m.vm_seed_bank(0, &seeds).expect("seeded");
        }
    }
    let mut sampled = vec![false; na + nb];
    sampled[0] = true;
    sampled[na] = true;
    let two = lanes(launch, pool, 5, 1, |b| vec![b as u32 + 1]);
    let steps = m.vm_fire(&two, &sampled, &[]).expect("the VM launch");
    let words = rec.memory("vmlane", col, 0).expect("vmlane written");
    (
        words,
        sampled,
        steps.iter().map(|(_, row, _)| !row.is_empty()).collect(),
    )
}

#[test]
fn a_vmconc_launch_keeps_every_lane_on_the_vm() {
    // VMCONC rebinds the head's argmax chain to set B's rows for a VM launch
    // (0196: a lane left on the head's argmax hung the wafer). Every lane is
    // a VM lane; one the launch does not sample sits its turn out (its
    // forced prefix past the launch's one lap), and only sampled lanes come
    // back with a turn.
    let (words, sampled, answered) = vm_launch_control("serve2d_pipe_conc");
    for (v, w) in words.chunks(4).enumerate() {
        assert_eq!(w[0], 1, "lane {v} on the head's argmax under VMCONC");
        if sampled[v] {
            assert_eq!(w[1], 1, "sampled lane {v} skips its turn");
        } else {
            assert_eq!(w[1], 2, "lane {v} is not a skipped turn");
        }
        assert_eq!(
            answered.get(v).copied().unwrap_or(false),
            sampled[v],
            "lane {v}'s turn read back"
        );
    }
    // Without VMCONC the chain stays the head's: an unsampled lane takes
    // its argmax as before.
    let (words, sampled, _) = vm_launch_control("serve2d_pipe_ab");
    for (v, w) in words.chunks(4).enumerate() {
        assert_eq!(
            w[0],
            u32::from(sampled[v]),
            "lane {v}'s gmode_l without VMCONC"
        );
        assert_eq!(w[1], 1, "lane {v}'s forced prefix without VMCONC");
    }
}

#[test]
fn the_abi_covers_every_call_and_the_sources_export_it() {
    use super::component::abi::Kind;
    for (sc, phases) in runs() {
        let spec = ProgramSpec::from_params(sc.built, &sc.params).unwrap();
        let abi = spec.abi();
        let files = sc.built.files().unwrap();
        let exports = super::codegen::layout_exports(&files[0].1);
        for (owner, sym) in &abi {
            assert!(
                exports.iter().any(|e| e == sym.name),
                "{}: {owner}'s `{}` is not exported by {}'s layout",
                sc.name,
                sym.name,
                sc.built.name()
            );
        }
        for line in phases.iter().flat_map(|(_, lines)| lines) {
            let mut words = line.split(' ');
            let (op, name) = (words.next().unwrap(), words.next().unwrap_or(""));
            let kind = match op {
                "launch" | "launch_nb" => Kind::Func {
                    args: line.matches(',').count() + usize::from(!line.ends_with("[]")),
                },
                "h2d" | "h2d_nb" | "d2h" | "rowbcast" => Kind::Var {
                    bits: if line.contains(" b16 ") { 16 } else { 32 },
                },
                _ => continue,
            };
            assert!(
                abi.iter().any(|(_, s)| s.name == name && s.kind == kind),
                "{}: `{line}` is not in the declared ABI ({kind:?})",
                sc.name
            );
        }
    }
}

/// A program directory PIE built (its manifest and out.json, no PE
/// images) loads through the driver's own reader (`read_design`) as the
/// scenario's spec, and runs to the scenario's pins.
#[test]
fn a_program_directory_loads_to_the_same_pins() {
    for name in ["serve2d_pipe_ab", "serve8b_region_pipe"] {
        let sc = scenarios()
            .into_iter()
            .find(|s| s.name == name)
            .expect("the scenario");
        let spec = ProgramSpec::from_params(sc.built, &sc.params).unwrap();
        let dir = tempfile::tempdir().unwrap();
        super::manifest::Manifest {
            program: spec.clone(),
            key: "k".into(),
            fabric_dims: (1, 1),
            fabric_offsets: (4, 1),
            system: false,
        }
        .write(dir.path())
        .unwrap();
        let out: serde_json::Map<String, serde_json::Value> = sc
            .params
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        std::fs::write(
            dir.path().join("out.json"),
            serde_json::json!({ "params": out }).to_string(),
        )
        .unwrap();
        let read = super::wafer::read_design(dir.path()).unwrap();
        assert_eq!(read, spec, "{name}");
        for (phase, lines) in run_spec(&sc, read) {
            let key = format!("{name}/{phase}");
            let pin = PINS.iter().find(|p| p.0 == key).expect("a pin");
            assert_eq!(
                (lines.len(), digest(&lines).as_str()),
                (pin.1, pin.2),
                "{key}"
            );
        }
    }
}

#[test]
fn host_protocol_is_pinned() {
    let dump = std::env::var_os("PIE_GOLDEN_DUMP").map(std::path::PathBuf::from);
    let mut got: Vec<(String, usize, String)> = Vec::new();
    for (sc, phases) in runs() {
        for (phase, lines) in phases {
            let key = format!("{}/{phase}", sc.name);
            if let Some(dir) = &dump {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(
                    dir.join(key.replace('/', "-") + ".txt"),
                    lines.join("\n") + "\n",
                )
                .unwrap();
            }
            got.push((key, lines.len(), digest(lines)));
        }
    }
    if std::env::var_os("PIE_GOLDEN_PRINT").is_some() {
        for (k, n, d) in &got {
            println!("    ({k:?}, {n}, {d:?}),");
        }
    }
    let want: Vec<(String, usize, String)> = PINS
        .iter()
        .map(|&(k, n, d)| (k.to_string(), n, d.to_string()))
        .collect();
    let differ: Vec<String> = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g != w)
        .map(|(g, w)| {
            format!(
                "{} got {} calls {}, pinned {} {} {}",
                g.0, g.1, g.2, w.0, w.1, w.2
            )
        })
        .collect();
    assert!(
        differ.is_empty() && got.len() == want.len(),
        "the host protocol drifted ({} phases, {} pinned):\n{}",
        got.len(),
        want.len(),
        differ.join("\n")
    );
}
