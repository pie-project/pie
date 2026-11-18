#![cfg(target_vendor = "apple")]

//! A program the load prewarmed (`Shell::prewarm_programs`) registers
//! through the off-thread split (`prepare_program`, the compile on another
//! thread, `finish_program`) without compiling a region again: the split's
//! compile finds the prewarm's regions in the program cache. A second
//! registration of the same program compiles nothing either.

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with};
use super::a_score_reader_holds_its_rows_through_a_park::reader;

const CONTEXT: u32 = 256;

#[test]
fn a_prewarmed_program_registers_off_thread_without_compiling() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    let registration = reader(CONTEXT);
    shell
        .prewarm_programs(std::slice::from_ref(&registration))
        .expect("the prewarm compiles the program's regions");
    let prewarmed = shell.program_compilations();
    assert!(prewarmed > 0, "the prewarm compiled the program's regions");

    let register = |shell: &mut engine_metal::Shell| {
        let registering = shell.prepare_program(&registration).expect("the plan");
        let compiled = std::thread::spawn(move || registering.compile())
            .join()
            .expect("the compile runs");
        shell.finish_program(compiled).expect("the program is taken in")
    };
    let first = register(&mut shell);
    assert_eq!(
        shell.program_compilations(),
        prewarmed,
        "the off-thread registration found the prewarmed regions"
    );
    let second = register(&mut shell);
    assert_eq!(
        shell.program_compilations(),
        prewarmed,
        "a second registration compiled nothing"
    );
    let _ = (first, second);
}
