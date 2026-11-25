//! Guest regions compiled in processes of their own, on ROCm.
//!
//! HIP's runtime compiler holds one lock a process (and comgr under it), so
//! a load's prewarm compiles one region at a time however many threads ask.
//! A host program that answers [`main_if_asked`] first thing in its `main`
//! and calls [`register`] lets [`compile_all`] run itself once a region
//! instead, as many at a time as the process has cores and memory for. A
//! host that does not (a test, an embedder) compiles in process, serially.
//!
//! A child takes one region on its stdin and writes its code object on its
//! stdout; it reads no argument past [`ARG`], no configuration and no file
//! of the home, and the parent alone writes the disk cache. It is killed
//! with the thread that started it, and where [`compile_all`] returns
//! before it is done.
//!
//! Both ends are memory files, not pipes: a child that dies unread leaves
//! the parent no broken pipe (pie takes `SIGPIPE`'s default), and one that
//! writes more than a pipe holds waits on nobody.

#![allow(unsafe_code)]

use std::ffi::CStr;
use std::fs::File;
use std::io::{Read as _, Seek as _, Write as _};
use std::os::fd::FromRawFd as _;
use std::os::unix::fs::FileExt as _;
use std::os::unix::process::{CommandExt as _, ExitStatusExt as _};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use eta_exec::Failure;

/// The one argument that asks a host program for a region.
pub const ARG: &str = "compile-guest-region";

const ASK: &[u8; 8] = b"PIERGN1\n";
const ANSWER: &[u8; 8] = b"PIEOBJ1\n";

static HOSTED: AtomicBool = AtomicBool::new(false);

/// States that this program's `main` answers [`main_if_asked`].
pub fn register() {
    HOSTED.store(true, Ordering::Relaxed);
}

pub(super) fn registered() -> bool {
    HOSTED.load(Ordering::Relaxed)
}

/// A host program's first act: where it was started as a region's compile
/// (its one argument is [`ARG`]) that compile, and its exit code.
#[must_use]
pub fn main_if_asked() -> Option<ExitCode> {
    let mut args = std::env::args_os().skip(1);
    if args.next()? != ARG || args.next().is_some() {
        return None;
    }
    Some(match answer() {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("{ARG}: {why}");
            ExitCode::FAILURE
        }
    })
}

fn answer() -> Result<(), String> {
    // SAFETY: sets this process's own death signal.
    unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) };
    let mut asked = Vec::new();
    std::io::stdin()
        .read_to_end(&mut asked)
        .map_err(|e| format!("stdin: {e}"))?;
    let (parent, architecture, source) = parse(&asked).ok_or("stdin holds no region")?;
    // A parent that died before the signal was set has left this one behind.
    if std::os::unix::process::parent_id() != parent {
        return Err("the process that asked is gone".into());
    }
    // What the runtime prints goes to stderr; the answer alone to stdout.
    // SAFETY: duplicates this process's own descriptors.
    let mut out = unsafe {
        let out = libc::dup(1);
        if out < 0 || libc::dup2(2, 1) < 0 {
            return Err(format!("stdout: {}", std::io::Error::last_os_error()));
        }
        File::from_raw_fd(out)
    };
    let (kind, bytes) = match super::compile::compile(source, architecture) {
        Ok(code) => (0u8, code),
        Err(Failure::Deterministic { reason }) => (1, reason.into_bytes()),
        Err(Failure::Retryable { reason }) => (2, reason.into_bytes()),
    };
    out.write_all(ANSWER)
        .and_then(|()| out.write_all(&[kind]))
        .and_then(|()| out.write_all(&(bytes.len() as u64).to_le_bytes()))
        .and_then(|()| out.write_all(&bytes))
        .map_err(|e| format!("stdout: {e}"))
}

fn ask(architecture: &str, source: &str) -> Vec<u8> {
    let mut bytes = ASK.to_vec();
    bytes.extend(std::process::id().to_le_bytes());
    for text in [architecture, source] {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    }
    bytes
}

/// `(parent, architecture, source)` of what [`ask`] wrote, all of it.
fn parse(bytes: &[u8]) -> Option<(u32, &str, &str)> {
    let rest = bytes.strip_prefix(ASK)?;
    let (parent, mut rest) = rest.split_first_chunk::<4>()?;
    let mut text = || {
        let (len, after) = rest.split_first_chunk::<8>()?;
        let len = usize::try_from(u64::from_le_bytes(*len)).ok()?;
        let (text, after) = after.split_at_checked(len)?;
        rest = after;
        std::str::from_utf8(text).ok()
    };
    let (architecture, source) = (text()?, text()?);
    rest.is_empty()
        .then_some((u32::from_le_bytes(*parent), architecture, source))
}

fn parse_answer(bytes: &[u8]) -> Option<Result<Vec<u8>, Failure>> {
    let rest = bytes.strip_prefix(ANSWER)?;
    let (&kind, rest) = rest.split_first()?;
    let (len, body) = rest.split_first_chunk::<8>()?;
    if u64::from_le_bytes(*len) != body.len() as u64 {
        return None;
    }
    let reason = || String::from_utf8_lossy(body).into_owned();
    match kind {
        0 => Some(Ok(body.to_vec())),
        1 => Some(Err(Failure::Deterministic { reason: reason() })),
        2 => Some(Err(Failure::Retryable { reason: reason() })),
        _ => None,
    }
}

fn memory_file(name: &CStr) -> std::io::Result<File> {
    // SAFETY: `name` is NUL-terminated; the descriptor returned is owned here.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a descriptor just made, owned by nothing else.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn whole(file: &File) -> std::io::Result<Vec<u8>> {
    let len = usize::try_from(file.metadata()?.len()).unwrap_or(0);
    let mut bytes = vec![0u8; len];
    file.read_exact_at(&mut bytes, 0)?;
    Ok(bytes)
}

/// A region's compile, running. Dropped, it is killed and reaped.
struct Running {
    at: usize,
    child: Child,
    out: File,
    err: File,
    /// Its resident bytes when last read.
    resident: u64,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Running {
    fn start(at: usize, architecture: &str, source: &str) -> std::io::Result<Running> {
        let mut input = memory_file(c"pie-region-in")?;
        input.write_all(&ask(architecture, source))?;
        input.rewind()?;
        let out = memory_file(c"pie-region-out")?;
        let err = memory_file(c"pie-region-err")?;
        // This very program, though its file was replaced since it started.
        let child = Command::new("/proc/self/exe")
            .arg0("pie")
            .arg(ARG)
            .stdin(Stdio::from(input))
            .stdout(Stdio::from(out.try_clone()?))
            .stderr(Stdio::from(err.try_clone()?))
            .spawn()?;
        Ok(Running {
            at,
            child,
            out,
            err,
            resident: 0,
        })
    }

    /// Reads what it holds; its own peak so far.
    fn sample(&mut self) -> u64 {
        let Ok(text) = std::fs::read_to_string(format!("/proc/{}/status", self.child.id())) else {
            return 0;
        };
        let field = |name: &str| {
            text.lines()
                .find_map(|l| {
                    l.strip_prefix(name)?
                        .trim()
                        .strip_suffix("kB")?
                        .trim()
                        .parse()
                        .ok()
                })
                .map_or(0, |kib: u64| kib * 1024)
        };
        self.resident = field("VmRSS:");
        field("VmHWM:")
    }

    /// Its answer, once it exited.
    fn done(&mut self) -> Option<Result<Vec<u8>, Failure>> {
        let status = match self.child.try_wait() {
            Ok(None) => return None,
            Ok(Some(status)) => status,
            Err(e) => {
                return Some(Err(Failure::Retryable {
                    reason: format!("its compile process could not be waited for: {e}"),
                }));
            }
        };
        if status.success()
            && let Some(answer) = whole(&self.out).ok().and_then(|bytes| parse_answer(&bytes))
        {
            return Some(answer);
        }
        let said = whole(&self.err).unwrap_or_default();
        let said = String::from_utf8_lossy(&said);
        let tail: Vec<&str> = said.trim().lines().rev().take(5).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join("; ");
        let how = status.signal().map_or_else(
            || status.to_string(),
            |signal| format!("killed by signal {signal}"),
        );
        Some(Err(Failure::Retryable {
            reason: format!(
                "its compile process {} ended with no code object ({how}); its stderr: {}",
                self.child.id(),
                if tail.is_empty() { "nothing" } else { &tail }
            ),
        }))
    }
}

/// What the host and the process's cgroups have left.
fn room() -> Result<u64, String> {
    let v2 = engine::host_memory::available()?.bytes();
    Ok(cgroup_v1_room().map_or(v2, |v1| v1.min(v2)))
}

/// What a cgroup v1 memory controller leaves (a v2 one is
/// [`engine::host_memory`]'s); `None` where there is none.
fn cgroup_v1_room() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = own.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':').skip(1);
        let controllers = fields.next()?;
        controllers
            .split(',')
            .any(|c| c == "memory")
            .then(|| fields.next())?
    })?;
    // In a container the path is the host's; its own group is the mount's root.
    let read = |name: &str| {
        [
            format!("/sys/fs/cgroup/memory{path}/{name}"),
            format!("/sys/fs/cgroup/memory/{name}"),
        ]
        .iter()
        .find_map(|file| {
            std::fs::read_to_string(file)
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()
        })
    };
    Some(read("memory.limit_in_bytes")?.saturating_sub(read("memory.usage_in_bytes")?))
}

/// Each of `jobs`' regions `(index, source)` compiled for `architecture`, a
/// process each, in `jobs`' order. The first that fails ends the rest.
///
/// At most a process a core this one may run on, the longest source first.
/// One more starts only while the memory left holds it, and what those
/// running may still grow, each at the largest peak any has reached; where
/// that peak rises past what is left, the newest is ended and its region
/// queued again. One always runs.
pub(super) fn compile_all(
    jobs: &[(u32, &str)],
    architecture: &str,
) -> Result<Vec<Vec<u8>>, Failure> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let width = cores.min(jobs.len());
    let mut queue: Vec<usize> = (0..jobs.len()).collect();
    queue.sort_by_key(|&at| (jobs[at].1.len(), std::cmp::Reverse(at)));
    let mut made: Vec<Option<Vec<u8>>> = jobs.iter().map(|_| None).collect();
    let mut running: Vec<Running> = Vec::new();
    let named = |at: usize, reason: &str| format!("region {}: {reason}", jobs[at].0);
    let (mut peak, mut most, mut held, mut again) = (0u64, 0usize, 0u64, 0usize);
    let mut short = None;
    while !(queue.is_empty() && running.is_empty()) {
        let mut at = 0;
        while at < running.len() {
            let Some(answer) = running[at].done() else {
                at += 1;
                continue;
            };
            let job = running.remove(at).at;
            made[job] = Some(answer.map_err(|failure| match failure {
                Failure::Deterministic { reason } => Failure::Deterministic {
                    reason: named(job, &reason),
                },
                Failure::Retryable { reason } => Failure::Retryable {
                    reason: named(job, &reason),
                },
            })?);
        }
        for one in &mut running {
            peak = peak.max(one.sample());
        }
        held = held.max(running.iter().map(|one| one.resident).sum());
        let mut wanted: u64 = running
            .iter()
            .map(|one| peak.saturating_sub(one.resident))
            .sum();
        let left = room();
        let fits = |wanted: u64| left.as_ref().is_ok_and(|&left| left >= wanted);
        if running.len() > 1
            && !fits(wanted)
            && let Some(newest) = running.pop()
        {
            queue.push(newest.at);
            again += 1;
        } else {
            while running.len() < width
                && let Some(&next) = queue.last()
            {
                if !running.is_empty() && (peak == 0 || !fits(wanted + peak)) {
                    if peak != 0 {
                        short = Some(match &left {
                            Ok(left) => format!("{} MiB left", left >> 20),
                            Err(why) => format!("what is left could not be read: {why}"),
                        });
                    }
                    break;
                }
                queue.pop();
                let one = Running::start(next, architecture, jobs[next].1).map_err(|e| {
                    Failure::Retryable {
                        reason: named(next, &format!("its compile process did not start: {e}")),
                    }
                })?;
                running.push(one);
                wanted += peak;
            }
        }
        most = most.max(running.len());
        std::thread::sleep(Duration::from_millis(5));
    }
    tracing::info!(
        "guest regions: {} compiled, at most {most} processes at once (limited by {}); the \
         largest held {} MiB, all at once {} MiB; {again} ended for memory and compiled again",
        jobs.len(),
        match &short {
            Some(left) => format!("memory: {left}"),
            None if cores < jobs.len() => format!("the {cores} cores"),
            None => format!("the {} regions", jobs.len()),
        },
        peak >> 20,
        held >> 20,
    );
    Ok(made.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::{ask, parse};

    #[test]
    fn only_a_whole_region_is_taken() {
        let asked = ask("gfx942", "source");
        assert_eq!(
            parse(&asked),
            Some((std::process::id(), "gfx942", "source"))
        );
        for cut in 0..asked.len() {
            assert_eq!(parse(&asked[..cut]), None, "{cut} bytes of it");
        }
        assert_eq!(parse(&[asked.as_slice(), b"x"].concat()), None);
    }
}
