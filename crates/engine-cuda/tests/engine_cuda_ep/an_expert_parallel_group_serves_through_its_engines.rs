#![cfg(feature = "cuda")]

//! An expert-parallel group as a worker drives it: `open_lockstep` hands
//! back one engine a rank, each loaded and submitted to on a thread of its
//! own, every rank reading its slots of each bank from the one checkpoint.
//! A step both ranks announced fires both their prompts at once; a
//! step only one rank brings fires the other idle, serving the routes the
//! first sends it. Every answer is the one a single engine holding every
//! expert gives.

use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};
use poem_ir::Trace;

use crate::micro::{MICRO, RANKS, Scratch, close, prompt, write_checkpoint};

fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

fn request(trace: &Trace, checkpoint: &Path) -> LoadRequest {
    LoadRequest {
        trace: trace.clone(),
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets: Budgets {
            max_lanes: 2,
            max_tokens: 256,
            buckets: vec![16, 64, 256],
            max_adapters: 0,
            page_size: 16,
            max_context: 256,
            context_declared: false,
            slots: 2,
            max_patches: None,
            max_images: None,
            max_voxels: None,
            max_clips: None,
        },
        // Ranks sharing one device: each its share of it, and a small kv.
        residency: Residency {
            device_memory_share: 0.3,
            device_kv_budget: Some(64 << 20),
            host_kv_budget: Some(64 << 20),
            disk_kv_budget: Some(0),
            ..Residency::default()
        },
        ordinal: -1,
        frames_in_flight: 1,
    }
}

/// `whole`'s checkpoint, every expert, in a directory of its own.
fn checkpoint(whole: &Trace, tag: &str) -> (Scratch, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pie-ep-engine-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let path = dir.join("micro.zt");
    write_checkpoint(&path, whole);
    (Scratch(dir), path)
}

fn step(slot: u32, word: u64, tokens: &[u32]) -> Step {
    Step {
        lanes: vec![Lane {
            slot,
            word,
            tokens: tokens.to_vec(),
            readout: Readout::Last,
            ..Lane::default()
        }],
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    }
}

/// One step's last-row logits, submitted and settled.
fn answer(engine: &mut dyn Engine, step: &Step) -> Vec<f32> {
    let mut ticket = engine
        .submit(&FrameSubmission::of(step.clone()))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let readout = &ticket.steps[0].readouts[0];
    readout.values.clone()
}

#[test]
fn an_expert_parallel_group_serves_through_its_engines() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), RANKS, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let (a, b) = (prompt(37, 1), prompt(52, 2));
    let boot = || DeviceBoot {
        ordinal: 0,
        ..DeviceBoot::default()
    };

    // One engine holding every expert.
    let (want_a, want_b) = {
        let (_dir, path) = checkpoint(&whole, "one");
        let mut one = engine_cuda::open(boot(), contract_for).expect("the engine opens");
        one.load(request(&whole, &path)).expect("the micro loads");
        let want_a = answer(&mut one, &step(0, word(a.len()), &a));
        let want_b = answer(&mut one, &step(1, word(b.len()), &b));
        (want_a, want_b)
    };

    // A rank per boot, every rank loading at once: their loads meet at arming.
    let mut ranks = engine_cuda::lockstep::open_lockstep(vec![boot(), boot()], contract_for)
        .expect("the group opens");
    // Every rank reads its own slots of each bank from the one checkpoint.
    let (dir, path) = checkpoint(&whole, "group");
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });

    // Both ranks announce their prompts, so the step waits for both.
    let (first, second) = (step(0, word(a.len()), &a), step(0, word(b.len()), &b));
    ranks[0].expect_fire(&first);
    ranks[1].expect_fire(&second);
    let [r0, r1] = &mut ranks[..] else {
        unreachable!()
    };
    let (got_a, got_b) = std::thread::scope(|scope| {
        let one = scope.spawn(|| answer(r0, &first));
        let two = scope.spawn(|| answer(r1, &second));
        (one.join().unwrap(), two.join().unwrap())
    });
    close(&got_a, &want_a, "rank 0's prompt beside rank 1's");
    close(&got_b, &want_b, "rank 1's prompt beside rank 0's");

    // Rank 1 brings nothing: it idles through rank 0's step.
    let got = answer(r0, &step(1, word(a.len()), &a));
    close(&got, &want_a, "rank 0's prompt with rank 1 idle");
    drop(dir);
}
