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
    let copy = |seeds: &[wafer::Seed]| -> Vec<wafer::Seed> {
        seeds.iter().map(|s| wafer::Seed { words: s.words.clone(), ..*s }).collect()
    };
    if lay.lprog {
        // Each lane its case's program (a program per distinct case file).
        let mut programs = Vec::new();
        let mut seeds = Vec::new();
        for lane in 0..n {
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
                let (table, _) = &cases[case_of(v)];
                if table.per_step.is_empty() {
                    return Ok(None);
                }
                let entries = slots::step_entries(&lay, table)?;
                let words = &jsons[case_of(v)]["per_step_words"][s];
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
        let answers = m.vm_fire(&lanes, &[], &inputs)?;
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
