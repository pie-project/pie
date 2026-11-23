//! Loading the PJRT plugin and making its client leave the process's own
//! handling of SIGTERM and SIGINT in place: libtpu installs handlers for
//! them as it first loads in a process (each dumps every thread's stack and
//! re-raises), which took pie's shutdown from it and, landing while
//! programs deserialized, hung the process. A process whose signals take
//! their default action loads the plugin: each still takes it. The plugin
//! loads once a process, so the probe runs in a process of its own (this
//! test binary, run again on the ignored probe), and takes no chip (the
//! suite's process holds it). Skipped without a PJRT plugin.

#![allow(unsafe_code)]

const SIGNALS: [libc::c_int; 2] = [libc::SIGTERM, libc::SIGINT];

fn handler_of(sig: libc::c_int) -> libc::sighandler_t {
    // SAFETY: `sigaction` is plain data, zeroed a valid value; the call only
    // reads `sig`'s action into it.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { libc::sigaction(sig, std::ptr::null(), &mut action) };
    action.sa_sigaction
}

#[test]
#[ignore = "the probe, run in a process of its own by the test below"]
fn signal_probe() {
    if std::env::var_os("PIE_XLA_SIGNAL_PROBE").is_none() {
        return;
    }
    // as a process starts: each signal's default action (pie installs its
    // own handling once it serves, chaining to what was there)
    for sig in SIGNALS {
        // SAFETY: as in `handler_of`.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = libc::SIG_DFL;
        // SAFETY: sets the default action.
        unsafe { libc::sigaction(sig, &action, std::ptr::null_mut()) };
    }
    // what each signal reaches now (a function's address may differ by cast)
    let mine: Vec<_> = SIGNALS.iter().map(|&sig| handler_of(sig)).collect();
    // the plugin loaded and initialized: no chip taken
    if engine_xla::pjrt::Api::discover(None).is_err() {
        println!("PROBE no device");
        return;
    }
    for (&sig, mine) in SIGNALS.iter().zip(mine) {
        if handler_of(sig) == mine {
            println!("PROBE kept {sig}");
        } else {
            println!("PROBE replaced {sig}");
        }
    }
}

#[test]
fn a_device_open_keeps_the_process_signal_handlers() {
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args([
            "a_device_open_keeps_the_process_signal_handlers::signal_probe",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("PIE_XLA_SIGNAL_PROBE", "1")
        .output()
        .expect("the probe runs");
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the probe: {said}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    if said.contains("PROBE no device") || !said.contains("PROBE ") {
        return;
    }
    for sig in SIGNALS {
        assert!(
            said.contains(&format!("PROBE kept {sig}")),
            "signal {sig}: the action set before the load is the one it takes: {said}"
        );
    }
}
