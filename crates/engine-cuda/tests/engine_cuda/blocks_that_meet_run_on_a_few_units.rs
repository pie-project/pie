//! Launches whose blocks meet complete where the device's units are hidden
//! from them. A region whose rows split across blocks never waits on another
//! block (a launch a phase); a sampler lane's blocks wait at its barrier, so
//! the launcher lays them over no more units than this process's queues run
//! on. The epilogue's agreement cases (a split row's among them) and the lane
//! sampler's run again in a process whose queues `HSA_CU_MASK` keeps to four
//! units: a block that waited on one not yet dispatched would hang there.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Far past the cases' time on four units.
const PATIENCE: Duration = Duration::from_secs(600);

/// The cases, by their full test names, run in a child kept to four units.
fn on_four_units(cases: &[&str]) {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args(cases)
        .args(["--exact", "--test-threads", "1"])
        .env("HSA_CU_MASK", "0:0-3")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the cases run in a child");
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the child is waited on") {
            break status;
        }
        if started.elapsed() > PATIENCE {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "{cases:?} did not finish on four units in {}s: a launch's blocks wait on \
                 blocks that are not resident",
                PATIENCE.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(status.success(), "{cases:?} on four units: {status}");
}

#[test]
fn a_split_row_runs_on_a_few_units() {
    on_four_units(&[
        "an_epilogue_agrees_with_the_host_interpreter::an_epilogue_agrees_with_the_host_interpreter_every_case",
    ]);
}

#[test]
fn a_lane_sampler_runs_on_a_few_units() {
    on_four_units(&[
        "a_lane_sampler_draws_what_the_interpreter_draws::a_lane_sampler_draws_what_the_interpreter_draws",
        "a_lane_sampler_draws_what_the_interpreter_draws::a_wide_fire_s_lanes_draw_what_the_interpreter_draws",
    ]);
}
