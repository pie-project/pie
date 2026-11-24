#![cfg(target_vendor = "apple")]

//! A guest program's registration compiles off the engine's thread
//! (`engine::Engine::prepare_program` / `finish_program`): the plan on the
//! shell's thread, the libraries and pipelines on another, the program
//! taken in on the shell's thread again. A cold registration (every kernel
//! source carries a nonce, so neither the shell's cache nor the system's
//! shader cache holds it) compiles while the shell fires decodes: fires land
//! during the compile, none waits it out, and the program it registers binds
//! and reads its scores.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_metal::serve::Seated;
use engine_metal::Lane;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};
use super::a_score_reader_holds_its_rows_through_a_park::{at_lane, bind_reader, reader};

const CONTEXT: u32 = 256;

#[test]
fn a_cold_registration_compiles_while_lanes_fire() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    let prompt: Vec<u32> = (0..12u32).map(|t| (t * 7 + 11) % 256).collect();
    shell.open(0).expect("the slot opens");
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, prompt.len() as u32),
                tokens: &prompt,
            })
        }])
        .expect("the prompt prefills");
    let mut held = prompt.len() as u32;
    let decode = |shell: &mut engine_metal::Shell, held: &mut u32| {
        let began = Instant::now();
        shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(*held),
                ..Seated::of(Lane {
                    slot: 0,
                    word: word(&fixture.trace, 1),
                    tokens: &[5],
                })
            }])
            .expect("a decode fires");
        *held += 1;
        began.elapsed()
    };
    // A decode alone, for scale.
    let alone = (0..4).map(|_| decode(&mut shell, &mut held)).max().unwrap_or_default();

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut cold = reader(CONTEXT);
    for kernel in &mut cold.emitted_kernels {
        kernel.source.push_str(&format!("\n// registration nonce {nonce}\n"));
    }
    let registering = shell.prepare_program(&cold).expect("the registration plans");
    let done = Arc::new(AtomicBool::new(false));
    let compile = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let began = Instant::now();
            let registered = registering.compile();
            let took = began.elapsed();
            done.store(true, Ordering::Release);
            (registered, took)
        })
    };
    let mut during = Vec::new();
    while !done.load(Ordering::Acquire) {
        during.push(decode(&mut shell, &mut held));
    }
    let (registered, compile_took) = compile.join().expect("the compile runs");
    let program = shell.finish_program(registered).expect("the program is taken in");
    let worst = during.iter().max().copied().unwrap_or_default();
    assert!(
        during.len() >= 3,
        "the shell fired {} decodes in a {:?} compile",
        during.len(),
        compile_took
    );
    assert!(
        worst < compile_took / 2 && worst < alone * 4 + Duration::from_millis(20),
        "a decode waited on the compile: the slowest {worst:?}, the compile {compile_took:?}"
    );

    // The program it registered binds and reads its lane's scores.
    let (id, ..) = bind_reader(&mut shell, program, 1, 1);
    let scored = fixture
        .trace
        .facts
        .word(&poem::Request::new(1, false).capturing_scores(true));
    shell
        .fire_attached(
            &[Seated {
                replay_stated: None,
                held: Some(held),
                ..Seated::of(Lane {
                    slot: 0,
                    word: scored,
                    tokens: &[7],
                })
            }],
            &[at_lane(0, id)],
        )
        .expect("a scored decode fires with the program read in");
    let rows = shell.score_rows().expect("the rows read back");
    assert_eq!(rows.len(), 1, "the reader holds its lane's rows");
}
