//! The `fabric-run` processes a phase program runs in: one per run, or a
//! resident server per program that runs spec after spec.

use super::*;

/// The resident servers, at most `most` at a time: a simulator that sits
/// idle still spins, so many of them starve the one at work (on the
/// simulator; the `servers` switch sets the cap, 4 by default). The least
/// recently used goes when one more is needed.
#[derive(Default)]
pub struct Servers {
    live: HashMap<PathBuf, Served>,
    recent: Vec<PathBuf>,
}

impl Servers {
    fn most() -> usize {
        crate::switches::parse::<usize>("servers")
            .filter(|v| *v > 0)
            .unwrap_or(4)
    }

    /// The server of `dir`, spawned when none is live (the least recently
    /// used dropped first when the cap is reached).
    pub(super) fn get(&mut self, runner: &Path, dir: &Path, target: &str) -> Result<&mut Served> {
        if !self.live.contains_key(dir) {
            while self.live.len() >= Self::most() {
                let Some(old) = self.recent.first().cloned() else {
                    break;
                };
                self.recent.remove(0);
                self.live.remove(&old);
            }
            self.live
                .insert(dir.to_path_buf(), Served::spawn(runner, dir, target)?);
        }
        self.recent.retain(|d| d != dir);
        self.recent.push(dir.to_path_buf());
        Ok(self.live.get_mut(dir).expect("just inserted"))
    }

    pub(super) fn drop_server(&mut self, dir: &Path) {
        self.live.remove(dir);
        self.recent.retain(|d| d != dir);
    }
}

/// A `fabric-run --serve` child: specs go in on its stdin, `ok`/`err`
/// lines come back on a channel a reader thread feeds.
pub struct Served {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    answers: std::sync::mpsc::Receiver<String>,
    /// The simulator's log in the server's directory.
    log: PathBuf,
}

impl Served {
    fn spawn(runner: &Path, dir: &Path, target: &str) -> Result<Served> {
        let mut child = runner_command(runner)
            .arg("--serve")
            .arg(dir)
            .arg(target)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| Fault::Compile {
                why: format!("spawning fabric-run --serve: {e}"),
            })?;
        let stdin = child.stdin.take().ok_or(Fault::Compile {
            why: "fabric-run --serve has no stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or(Fault::Compile {
            why: "fabric-run --serve has no stdout".into(),
        })?;
        let (tx, answers) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Served {
            child,
            stdin,
            answers,
            log: dir.join("serve").join("sim.log"),
        })
    }

    /// Runs `spec` on the loaded program, within `timeout`.
    pub(super) fn run(
        &mut self,
        spec: &Path,
        timeout: std::time::Duration,
        at: usize,
    ) -> Result<()> {
        use std::io::Write;
        writeln!(self.stdin, "{}", spec.display()).map_err(|e| Fault::Compile {
            why: format!("fabric-run --serve went away: {e}"),
        })?;
        let _ = self.stdin.flush();
        match self.answers.recv_timeout(timeout) {
            Ok(line) if line == "ok" => Ok(()),
            Ok(line) => Err(Fault::Compile {
                why: format!("phase {at} on the resident program: {line}"),
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(Fault::Compile {
                why: match simulator_fatal(&self.log) {
                    Some(fatal) => {
                        format!("the resident simulator faulted on phase {at}: {fatal}")
                    }
                    None => format!("phase {at} on the resident program ran past {timeout:?}"),
                },
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Fault::Compile {
                why: format!("fabric-run --serve died on phase {at}"),
            }),
        }
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Whether compiled programs stay resident between runs
/// (the `serve` switch at 0 turns it off).
pub(super) fn serving() -> bool {
    crate::switches::get("serve").is_none_or(|v| v != "0" && v != "false")
}

/// One `fabric-run` process for one phase: spawned, bounded by the phase
/// timeout, its log watched for a fault.
pub(super) fn run_once(
    runner: &Path,
    dir: &Path,
    spec_path: &Path,
    scratch: &Path,
    at: usize,
) -> Result<()> {
    // A simulator that faults can hang instead of exiting: bound the wait,
    // and watch its log for a fatal error (a PE's bad memory access, say).
    // A run the simulator's own startup kills (a signal with no fault
    // logged; seen when two simulators start at once) gets one more try.
    let sim_log = scratch.join("sim.log");
    let mut attempt = 0;
    loop {
        attempt += 1;
        let _ = std::fs::remove_file(&sim_log);
        let mut child = runner_command(runner)
            .arg(dir)
            .arg(spec_path)
            .stdout(std::process::Stdio::null())
            .spawn()
            .map_err(|e| Fault::Compile {
                why: format!("spawning fabric-run: {e}"),
            })?;
        let deadline = std::time::Instant::now() + phase_timeout();
        let mut polls: u32 = 0;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    polls += 1;
                    if polls.is_multiple_of(40)
                        && let Some(fatal) = simulator_fatal(&sim_log)
                    {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(Fault::Compile {
                            why: format!("the simulator faulted on phase {at}: {fatal}"),
                        });
                    }
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Fault::Compile {
                        why: format!(
                            "fabric-run on phase {at} ran past {:?} and was killed",
                            phase_timeout()
                        ),
                    });
                }
                Err(e) => {
                    return Err(Fault::Compile {
                        why: format!("waiting for fabric-run: {e}"),
                    });
                }
            }
        };
        if status.success() {
            break;
        }
        let crashed = status.code().is_none() && simulator_fatal(&sim_log).is_none();
        if crashed && attempt < 2 {
            tracing::warn!(phase = at, %status, "fabric-run died starting up; trying once more");
            continue;
        }
        return Err(Fault::Compile {
            why: format!("fabric-run failed on phase {at}: {status}"),
        });
    }
    Ok(())
}

/// The simulator's fatal error, if its log reports one: the first `FATAL`
/// line (the faulting PE, instruction and message).
fn simulator_fatal(log: &Path) -> Option<String> {
    let text = std::fs::read_to_string(log).ok()?;
    text.lines()
        .find(|l| l.contains("FATAL"))
        .map(|l| l.trim().to_string())
}

/// How long one phase program may run (the `phase_timeout` switch, seconds;
/// default 1800).
pub(super) fn phase_timeout() -> std::time::Duration {
    let secs = crate::switches::parse::<u64>("phase_timeout").unwrap_or(1800);
    std::time::Duration::from_secs(secs.max(1))
}

/// The fabric runner's command: through the `loader` switch's program when
/// set (the SDK's glibc loader wrapper), else the runner itself.
fn runner_command(runner: &Path) -> std::process::Command {
    match crate::switches::get("loader") {
        Some(loader) => {
            let mut cmd = std::process::Command::new(loader);
            cmd.arg(runner);
            cmd
        }
        None => std::process::Command::new(runner),
    }
}

/// The `fabric-run` binary: the `fabric_run` switch, or the one next to
/// the current executable.
pub fn fabric_run() -> Result<PathBuf> {
    if let Some(p) = crate::switches::get("fabric_run") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|e| Fault::Unbound {
        what: format!("the current executable: {e}"),
    })?;
    let mut dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    if dir
        .file_name()
        .is_some_and(|n| n == "deps" || n == "examples")
    {
        dir.pop();
    }
    let p = dir.join("fabric-run");
    if p.is_file() {
        Ok(p)
    } else {
        Err(Fault::NoDevice {
            detail: format!(
                "fabric-run not found at {} (build it with `cargo build -p engine-cerebras`)",
                p.display()
            ),
        })
    }
}
