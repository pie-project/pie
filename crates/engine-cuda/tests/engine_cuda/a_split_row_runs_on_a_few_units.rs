//! A region whose rows split across blocks completes where the device's
//! units are hidden from its launch: its blocks never wait on one another,
//! so its grid need not be resident at once. The epilogue's agreement cases
//! (a split row's among them) run again in a process whose queues
//! `HSA_CU_MASK` keeps to four units, which a launch sized from the device's
//! count does not see; a block that waited on one not yet dispatched would
//! hang there.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Far past the cases' time on four units (seconds).
const PATIENCE: Duration = Duration::from_secs(300);

#[test]
fn a_split_row_runs_on_a_few_units() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "an_epilogue_agrees_with_the_host_interpreter::an_epilogue_agrees_with_the_host_interpreter_every_case",
            "--exact",
            "--test-threads",
            "1",
        ])
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
                "the epilogue's cases did not finish on four units in {}s: a region's blocks \
                 wait on one another",
                PATIENCE.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(status.success(), "the epilogue's cases on four units: {status}");
}
