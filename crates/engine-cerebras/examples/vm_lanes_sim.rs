//! serve2d_pipe with the multi-lane eta VM (`VM` 1, `gring` lanes L =
//! MMAX·B) driven through the engine's own wafer path on the fabric
//! simulator: `vm_lanes_sim <program dir> <ckpt dir> <case json>[,<case
//! json>..] <steps> <out dir>`. Every lane loads the first case's program;
//! lane v takes case v's seeds (one bank each; the last case repeats), then
//! `steps` one-lap launches, lane v forced its last token at position s.
//! Writes `<out dir>/report.json` (each lane's tokens and out-ring rows) and
//! `<out dir>/planes_<v>.f32` (the logits rows lane v's VM turns sampled
//! from, a lap after another; the first HLB lanes). On an image with
//! per-lane programs (`vm_lprog`), lane v runs case v's own program; a
//! case's `per_step_words[s]` is its step input at step s.

use std::path::Path;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::guest4::{slots, wafer};
use engine_cerebras::spatial::serve2d::BatchLane;
use engine_cerebras::spatial::wafer::{Design, StepInput, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (program, ckpt, case_path) = (Path::new(&args[1]), Path::new(&args[2]), &args[3]);
    let steps: usize = args[4].parse().expect("steps");
    let out = Path::new(&args[5]);
    let jsons: Vec<serde_json::Value> = case_path
        .split(',')
        .map(|p| {
            serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
        })
        .collect::<Result<_, String>>()?;
    let cases: Vec<(engine_cerebras::spatial::guest4::Table, Vec<wafer::Seed>)> =
        jsons.iter().map(wafer::case_table).collect::<Result<_, String>>()?;
    let mut m = WaferModel::open(
        program,
        ckpt,
        Platform::Simulator(Simulator {
            num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
            ..Simulator::new(Target::Wse3)
        }),
        None,
    )?;
    let Design::Grid(grid) = m.design() else { return Err("not a serve2d build".into()) };
    let shape = grid.paged.ok_or("not a batched build")?;
    let n = grid.launch_lanes();
    let ps = shape.page_size as u32;
    let (lay, _) = m.vm_shape().ok_or("the build has no VM")?;
    let case_of = |lane: usize| lane.min(cases.len() - 1);
    // IDLE_LANES=1,2: lanes without a program (a per-lane-program image's
    // idle banks, as the engine leaves them).
    let idle: Vec<usize> = std::env::var("IDLE_LANES")
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let copy = |seeds: &[wafer::Seed]| -> Vec<wafer::Seed> {
        seeds.iter().map(|s| wafer::Seed { words: s.words.clone(), ..*s }).collect()
    };
    // REPLACE=<step>:<case json>: at that step lane 0 takes that case's
    // program, its own seeds, the other lanes kept (a request coming into a
    // bank another left): lane 0's steps from then on are that case's.
    let replace: Option<(usize, serde_json::Value)> = match std::env::var("REPLACE") {
        Ok(v) => {
            let (at, path) = v.split_once(':').ok_or("REPLACE=<step>:<case>")?;
            let json = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            Some((at.parse().map_err(|_| "REPLACE step")?, json))
        }
        Err(_) => None,
    };
    let mut lane0: (serde_json::Value, usize) = (jsons[0].clone(), 0);
    let mut placed: Vec<Option<slots::LaneProgram>> = Vec::new();
    if lay.lprog {
        // Each lane its case's program (a program per distinct case file).
        let mut programs = Vec::new();
        let mut seeds = Vec::new();
        for lane in 0..n {
            if idle.contains(&lane) {
                programs.push(None);
                seeds.push(None);
                continue;
            }
            let c = case_of(lane);
            let (table, s) = wafer::case_table(&jsons[c])?;
            let program = engine_cerebras::spatial::guest4::lane::Program {
                table,
                epilogue: 0,
                homes: Default::default(),
                logits: String::new(),
            };
            programs.push(Some(slots::LaneProgram { key: c as u64, program: std::sync::Arc::new(program) }));
            seeds.push(Some(copy(&s)));
        }
        m.vm_place_lanes(&programs, &seeds)?;
        placed = programs;
    } else {
        m.vm_load_lanes(&cases[0].0)?;
        for lane in 0..n {
            m.vm_seed_bank(lane, &copy(&cases[case_of(lane)].1))?;
        }
    }
    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); n];
    eprintln!("banks seeded: {n} lanes");
    let first = [3u32, 7, 11, 6, 5, 2, 9, 4];
    let mut toks: Vec<Vec<u32>> = (0..n).map(|v| vec![first[v % first.len()]]).collect();
    let mut rows: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n];
    let npg = (steps as u32).div_ceil(ps) + 1;
    for s in 0..steps {
        if let Some((at, _)) = replace.as_ref().filter(|(at, _)| *at == s) {
            // ARGMAX_BEFORE=n: n head-argmax launches (gmode 0) first, as a
            // new request's prefill runs before it takes a VM lane.
            let n_argmax: usize = std::env::var("ARGMAX_BEFORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            for k in 0..n_argmax {
                let lanes: Vec<BatchLane> = (0..n)
                    .map(|v| BatchLane {
                        pos: (s + k) as u32,
                        slot: (s + k) as u32,
                        pages: (0..npg).map(|i| v as u32 * npg + i).collect(),
                        visible: None,
                        forced: vec![3, 7],
                    })
                    .collect();
                m.fire_batch(&lanes, 2, false)?;
            }
            eprintln!("{n_argmax} argmax launches before the re-placement at step {at}");
        }
        if let Some((at, json)) = replace.as_ref().filter(|(at, _)| *at == s) {
            let (table, seeds) = wafer::case_table(json)?;
            let program = engine_cerebras::spatial::guest4::lane::Program {
                table,
                epilogue: 0,
                homes: Default::default(),
                logits: String::new(),
            };
            placed[0] = Some(slots::LaneProgram { key: 1000, program: std::sync::Arc::new(program) });
            let mut fresh: Vec<Option<Vec<wafer::Seed>>> = (0..n).map(|_| None).collect();
            fresh[0] = Some(seeds);
            m.vm_place_lanes(&placed, &fresh)?;
            lane0 = (json.clone(), *at);
            eprintln!("lane 0 re-placed at step {at}");
        }
        let lanes: Vec<BatchLane> = (0..n)
            .map(|v| BatchLane {
                pos: s as u32,
                slot: s as u32,
                pages: (0..npg).map(|i| v as u32 * npg + i).collect(),
                visible: None,
                forced: vec![*toks[v].last().unwrap()],
            })
            .collect();
        // Each lane's step input at step s (its case's `per_step_words`).
        let inputs: Vec<StepInput> = (0..n)
            .map(|v| {
                let (json, step) = if v == 0 { (&lane0.0, s - lane0.1) } else { (&jsons[case_of(v)], s) };
                let (table, _) = wafer::case_table(json)?;
                if table.per_step.is_empty() || idle.contains(&v) {
                    return Ok(None);
                }
                let entries = slots::step_entries(&lay, &table)?;
                let words = &json["per_step_words"][step];
                let values = entries
                    .iter()
                    .map(|e| {
                        words[e.chan.to_string()]
                            .as_array()
                            .map(|w| w.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect())
                            .unwrap_or_default()
                    })
                    .collect();
                Ok(Some((entries, values)))
            })
            .collect::<Result<_, String>>()?;
        // FIX_IDLE=0 runs the idle lanes' turns too (the bug the engine had).
        let skip_idle = std::env::var("FIX_IDLE").map_or(true, |v| v != "0");
        let sampled: Vec<bool> = (0..n).map(|v| !(skip_idle && idle.contains(&v))).collect();
        let answers = m.vm_fire(&lanes, &sampled, &inputs)?;
        for (v, plane) in m.lane_planes()?.into_iter().enumerate() {
            planes[v].extend(plane);
        }
        for (v, (tok, row, _)) in answers.into_iter().enumerate() {
            // A program without a token home answers no token the head
            // could embed: the next forced token stays in the vocabulary.
            toks[v].push(tok % lay.vocab as u32);
            rows[v].push(row);
        }
        eprintln!("step {s}: {:?}", toks.iter().map(|t| *t.last().unwrap()).collect::<Vec<_>>());
    }
    let report = serde_json::json!({"tokens": toks, "ring_rows": rows});
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    for (v, plane) in planes.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        let bytes: Vec<u8> = plane.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("planes_{v}.f32")), bytes).map_err(|e| e.to_string())?;
    }
    std::fs::write(out.join("report.json"), report.to_string()).map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
