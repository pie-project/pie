//! Subprocess driver: out-of-process supervisor for the Python drivers
//! (`dev`, `vllm`, `sglang`).
//!
//! Parallel to [`crate::embedded_driver::EmbeddedDriver`] in shape:
//!
//! | Lifecycle step  | Embedded (C++/Rust thread)        | Subprocess (Python child)         |
//! |-----------------|-----------------------------------|-----------------------------------|
//! | Launch          | `driver_ffi::run` on a thread     | `python -m pie_driver_<flavor>`   |
//! | Startup config  | Per-flavor TOML via `embedded_driver::write_*_startup_toml` | A flavor-neutral TOML this module writes (`write_subprocess_startup_toml`) |
//! | Handshake       | `ready_cb(caps_json)` callback    | One JSON line per group on a pipe (fd 3) terminated by a `{"ready":true}` sentinel |
//! | Cold-path RPC   | Standalone hosts via [`pie::device::RpcServer`] + [`crate::rpc_loop`] | Python launcher hosts its own `RpcServer` (via the `pie-rpc` wheel) inside `worker.py::_leader_loop`; standalone connects as the client |
//! | Shmem fast path | Driver allocates `/pie_shmem_g{N}` | Same — Python `pie_driver_dev.shmem_ipc.ShmemServer` |
//! | Stop signal     | `driver_ffi::request_stop`        | `SIGTERM` to the child            |
//! | Watchdog        | `JoinHandle::is_finished()`       | `Child::try_wait()`               |
//!
//! The handshake JSON shape is the contract between this module and each
//! launcher's `__main__.py`. **If you change `Handshake` here, update
//! the JSON shape in all three of `driver/{dev,vllm,sglang}/src/.../__main__.py`
//! to match.** The duplication is intentional — see the Phase 1.5 design
//! decision (option (b)) recorded in the `__main__.py` docstrings.

use std::io::{BufRead, BufReader};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::config::ModelConfig;
use crate::embedded_driver::DriverCapabilities;

/// Which Python driver flavor the subprocess hosts.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SubprocessFlavor {
    /// Reference Python driver (`pie_driver_dev`).
    Dev,
    /// vLLM-backed driver (`pie_driver_vllm`).
    Vllm,
    /// SGLang-backed driver (`pie_driver_sglang`).
    Sglang,
}

impl SubprocessFlavor {
    /// Lowercase string used in error messages, RPC `query` responses,
    /// and the launcher's `argv[0]`. Matches the TOML `driver.type`
    /// discriminator.
    pub fn as_str(self) -> &'static str {
        match self {
            SubprocessFlavor::Dev => "dev",
            SubprocessFlavor::Vllm => "vllm",
            SubprocessFlavor::Sglang => "sglang",
        }
    }

    /// Python module name to invoke as `python -m <name>`.
    pub fn module_name(self) -> &'static str {
        match self {
            SubprocessFlavor::Dev => "pie_driver_dev",
            SubprocessFlavor::Vllm => "pie_driver_vllm",
            SubprocessFlavor::Sglang => "pie_driver_sglang",
        }
    }
}

/// Per-launch handshake line. The launcher emits one of these per
/// DP-replica leader on the handshake pipe, then a `Sentinel` line.
#[derive(Debug, Deserialize)]
struct GroupLine {
    group_id: usize,
    server_name: String,
    /// Echoed back from the launcher for cross-checking against the
    /// standalone's expectation (`/pie_shmem_g{group_id}`). Mismatch is
    /// a hard error.
    shmem_name: String,
    caps: DriverCapabilities,
}

/// Decoded handshake — what `start_one_group` returns to the caller.
#[derive(Debug)]
pub struct Handshake {
    pub server_name: String,
    pub shmem_name: String,
    pub caps: DriverCapabilities,
}

/// Per-launch state directory: `$PIE_HOME/standalone/<pid>/sub-<flavor>-<group_id>/`.
/// Mirrors [`crate::embedded_driver::launch_state_dir`].
fn subprocess_state_dir(flavor: SubprocessFlavor, group_id: usize) -> PathBuf {
    pie::path::get_pie_home()
        .join("standalone")
        .join(std::process::id().to_string())
        .join(format!("sub-{}-g{}", flavor.as_str(), group_id))
}

/// Write the launcher's startup TOML.
///
/// Schema is the launcher's own (see `driver/{dev,vllm,sglang}/.../__main__.py`):
///
/// ```toml
/// [model]
/// name = "..."
/// hf_repo = "..."
///
/// [driver]
/// device = ["cuda:0"]
/// tensor_parallel_size = 1
/// activation_dtype = "bfloat16"
/// random_seed = 42
/// master_port = 29500
/// ready_timeout_s = 1200
///
/// [driver.options]
/// # raw passthrough of `[model.driver.options]` from the user's config
///
/// [telemetry]
/// enabled = false
/// endpoint = ""
/// service_name = ""
/// ```
pub fn write_subprocess_startup_toml(
    out_path: &Path,
    model: &ModelConfig,
    snapshot_dir: &Path,
    master_port: u16,
    ready_timeout_s: f64,
) -> Result<()> {
    let mut doc = toml::Table::new();

    // [model]
    let mut model_section = toml::Table::new();
    model_section.insert("name".into(), toml::Value::String(model.name.clone()));
    model_section.insert("hf_repo".into(), toml::Value::String(model.hf_repo.clone()));
    model_section.insert(
        "snapshot_dir".into(),
        toml::Value::String(snapshot_dir.display().to_string()),
    );
    doc.insert("model".into(), toml::Value::Table(model_section));

    // [driver]
    let mut driver_section = toml::Table::new();
    driver_section.insert(
        "device".into(),
        toml::Value::Array(
            model
                .driver
                .device
                .iter()
                .map(|d| toml::Value::String(d.clone()))
                .collect(),
        ),
    );
    driver_section.insert(
        "tensor_parallel_size".into(),
        toml::Value::Integer(model.driver.tensor_parallel_size as i64),
    );
    driver_section.insert(
        "activation_dtype".into(),
        toml::Value::String(model.driver.activation_dtype.clone()),
    );
    driver_section.insert(
        "random_seed".into(),
        toml::Value::Integer(model.driver.random_seed as i64),
    );
    driver_section.insert(
        "master_port".into(),
        toml::Value::Integer(master_port as i64),
    );
    driver_section.insert(
        "ready_timeout_s".into(),
        toml::Value::Float(ready_timeout_s),
    );
    // [driver.options] — passthrough, minus the standalone-side
    // `venv` / `python` keys (which `crate::python_resolve` consumed
    // before this function was called). The launcher's typed
    // `<DriverName>DriverConfig` would error on those unknown keys.
    let mut options = model.driver.options.clone();
    crate::python_resolve::strip_python_keys(&mut options);
    driver_section.insert("options".into(), toml::Value::Table(options));
    doc.insert("driver".into(), toml::Value::Table(driver_section));

    let serialized =
        toml::to_string(&doc).map_err(|e| anyhow!("serialize launcher TOML: {e}"))?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create launcher state dir {parent:?}: {e}"))?;
    }
    std::fs::write(out_path, serialized)
        .map_err(|e| anyhow!("write launcher TOML {out_path:?}: {e}"))?;
    Ok(())
}

/// Owns the Python child process for one DP replica. Same lifecycle
/// shape as [`EmbeddedDriver`] — caller calls `start_one_group` for
/// each group and gets back a `SubprocessDriver` per group.
pub struct SubprocessDriver {
    pub flavor: SubprocessFlavor,
    pub caps: DriverCapabilities,
    /// Server name from the Python launcher's `RpcServer` (cold-path
    /// channel). Goes straight into [`crate::bootstrap_translate::GroupHandshake`].
    pub server_name: String,
    /// `/pie_shmem_g{group_id}` — same convention as embedded; the
    /// launcher computes it from `group_id` and echoes it back so we
    /// can cross-check.
    pub shmem_name: String,
    child: Mutex<Option<Child>>,
    _state_dir: PathBuf,
}

impl SubprocessDriver {
    /// Spawn one Python launcher subprocess for one DP group.
    ///
    /// `python_exe` is the interpreter path. Phase 4 wires the
    /// resolution chain (`model.driver.options.venv` →
    /// `$PIE_PYTHON` → `~/.pie/config.toml [driver.<type>].venv` →
    /// `$VIRTUAL_ENV/bin/python` → `which python3`); for now the
    /// caller passes whatever they have.
    ///
    /// Currently launches one subprocess per group. The Python
    /// launcher's `mp.spawn(world_size=…)` then forks per-rank workers
    /// inside that subprocess. This matches `_spawn_model_workers` in
    /// the legacy Python server, except each model now gets its own
    /// process tree (one `python -m pie_driver_<flavor>` per group).
    ///
    /// **Note:** the current launcher TOML doesn't carry per-group
    /// info — `mp.spawn(world_size=N)` is invoked once per group with
    /// the same `[driver]` section. With DP > 1, the standalone calls
    /// `start` per group, so each call gets its own subprocess that
    /// thinks it's alone. M4.5 may collapse to one Python subprocess
    /// for all groups when TP rendezvous lands.
    pub fn start(
        flavor: SubprocessFlavor,
        python_exe: &Path,
        model: &ModelConfig,
        snapshot_dir: &Path,
        group_id: usize,
        master_port: u16,
    ) -> Result<Self> {
        if !snapshot_dir.is_dir() {
            bail!(
                "snapshot_dir {snapshot_dir:?} does not exist or is not a directory"
            );
        }

        let state_dir = subprocess_state_dir(flavor, group_id);
        std::fs::create_dir_all(&state_dir)
            .with_context(|| format!("create state dir {state_dir:?}"))?;
        let toml_path = state_dir.join("driver.toml");

        // Conservative default — vllm + sglang both load big weights;
        // 20 minutes covers a cold start. M4 surfaces this as a knob.
        let ready_timeout_s = 1200.0;

        write_subprocess_startup_toml(
            &toml_path,
            model,
            snapshot_dir,
            master_port,
            ready_timeout_s,
        )?;

        // Build the handshake pipe. Parent reads, child writes (dup'd
        // to fd 3 in the child via `pre_exec`).
        let (parent_read_fd, child_write_fd) = make_pipe()
            .context("creating handshake pipe")?;

        // Move ownership of the child end into the pre_exec closure
        // by raw fd (CommandExt::pre_exec captures by Fn so the OwnedFd
        // would have to be Copy; raw fd works).
        let child_write_raw = child_write_fd.into_raw_fd();

        let mut cmd = Command::new(python_exe);
        cmd.arg("-m")
            .arg(flavor.module_name())
            .arg("--config")
            .arg(&toml_path)
            .arg("--handshake-fd")
            .arg("3")
            // Inherit stderr so failures print directly. stdin closed
            // (subprocess shouldn't read). stdout inherited too —
            // launchers print boot diagnostics there.
            .stdin(Stdio::null());

        // SAFETY: `pre_exec` runs in the forked child between fork()
        // and exec(). The libc calls below are async-signal-safe (per
        // POSIX). `child_write_raw` is a fd we own; nothing else points
        // at it.
        unsafe {
            cmd.pre_exec(move || {
                if libc::dup2(child_write_raw, 3) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // dup2 already closed any prior fd 3. Close the source
                // if dup2 didn't alias it (dup2(x, x) is a no-op).
                if child_write_raw != 3 {
                    libc::close(child_write_raw);
                }
                // Linux: ask the kernel to deliver SIGTERM to this
                // child if its parent (the standalone process —
                // potentially a Python interpreter that embedded the
                // pyo3 wheel) dies hard (e.g. SIGKILL). Without this,
                // a hard-killed parent leaves the launcher orphaned
                // and the worker pool keeps holding the GPU.
                // `prctl` is async-signal-safe on Linux (`signal-safety(7)`).
                // No-op on macOS — best we can do there is rely on the
                // process group / `Drop` impl in the parent.
                #[cfg(target_os = "linux")]
                {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }

        let child = cmd
            .spawn()
            .with_context(|| format!(
                "spawn `{} -m {} --config {} --handshake-fd 3`",
                python_exe.display(),
                flavor.module_name(),
                toml_path.display(),
            ))?;

        // Close the parent's copy of the pipe's write end. The child
        // forked + dup2'd it to fd 3 in `pre_exec`; we have no use for
        // it here. Critical: without this close, the kernel pipe has
        // two write-side openers (parent + child) and the read end
        // never sees EOF when the child dies, so a launcher that
        // crashes before sending the handshake makes us block the
        // full `ready_timeout_s` instead of failing fast.
        // SAFETY: `child_write_raw` is the fd we created via `pipe2`;
        // we transferred ownership via `into_raw_fd` so it's our
        // responsibility to close. The pre_exec closure consumed a
        // *copy* in the forked child — closing here only affects the
        // parent's view.
        unsafe {
            libc::close(child_write_raw);
        }

        // Read handshake JSON lines until sentinel or timeout.
        // SAFETY: `parent_read_fd` is exclusively owned by us; the
        // OwnedFd takes ownership and is dropped at end of scope.
        let parent_read = parent_read_fd;
        let reader = BufReader::new(unsafe {
            std::fs::File::from_raw_fd(parent_read.into_raw_fd())
        });

        let handshake = read_handshake_for_group(reader, group_id, ready_timeout_s)
            .with_context(|| format!(
                "reading handshake for {} group {group_id}",
                flavor.as_str(),
            ))?;

        let expected_shmem = format!("/pie_shmem_g{group_id}");
        if handshake.shmem_name != expected_shmem {
            bail!(
                "{} group {group_id}: launcher reported shmem_name={:?}, \
                 expected {:?}",
                flavor.as_str(),
                handshake.shmem_name,
                expected_shmem,
            );
        }

        Ok(SubprocessDriver {
            flavor,
            caps: handshake.caps,
            server_name: handshake.server_name,
            shmem_name: handshake.shmem_name,
            child: Mutex::new(Some(child)),
            _state_dir: state_dir,
        })
    }

    /// Send SIGTERM to the launcher; idempotent. Looks up the pid
    /// through the child mutex — the lock is uncontended in practice
    /// (only `is_finished` / `join` also hold it, both fast paths).
    pub fn request_stop(&self) {
        let Some(pid) = self.child.lock().unwrap().as_ref().map(|c| c.id() as i32) else {
            return;
        };
        // SAFETY: `kill(pid, SIGTERM)` is async-signal-safe; an already-
        // exited or reaped pid returns `ESRCH` harmlessly.
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
    }

    /// True if the launcher has exited. Cheap; locks the child mutex
    /// for the duration of one `try_wait` call.
    pub fn is_finished(&self) -> bool {
        let mut guard = self.child.lock().unwrap();
        let Some(child) = guard.as_mut() else {
            return true;
        };
        match child.try_wait() {
            Ok(Some(_status)) => true,
            Ok(None) => false,
            Err(_) => true, // Give up watching; treat as finished.
        }
    }

    /// Wait for the launcher to exit and return its exit code (or -1
    /// if it died from a signal / we couldn't read it).
    pub fn join(self) -> i32 {
        let mut guard = self.child.lock().unwrap();
        let Some(mut child) = guard.take() else {
            return 0;
        };
        match child.wait() {
            Ok(status) => status.code().unwrap_or(-1),
            Err(_) => -1,
        }
    }
}

impl Drop for SubprocessDriver {
    /// Safety net for the "caller forgot to join" path. Don't leave
    /// orphan Python processes.
    fn drop(&mut self) {
        let mut guard = self.child.lock().unwrap();
        if let Some(mut child) = guard.take() {
            let pid = child.id() as i32;
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
            let _ = child.wait();
        }
    }
}

// -----------------------------------------------------------------------------
// helpers — pipe creation + handshake read loop.
// -----------------------------------------------------------------------------

/// Create a pipe with the read end set to close-on-exec so it doesn't
/// leak into the child. Returns `(parent_read, child_write)`.
fn make_pipe() -> Result<(OwnedFd, OwnedFd)> {
    let mut fds: [libc::c_int; 2] = [0; 2];
    // SAFETY: `pipe2` writes two ints into `fds`; we own that buffer.
    let rc = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
    if rc != 0 {
        return Err(anyhow!("pipe2: {}", std::io::Error::last_os_error()));
    }
    // Child end must NOT be O_CLOEXEC after the dup2 in pre_exec — but
    // since we dup it explicitly there, it loses the flag automatically
    // (dup2'd fds always have CLOEXEC=0).
    // SAFETY: each fd is fresh from pipe2; we own them.
    let parent_read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let child_write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((parent_read, child_write))
}

fn read_handshake_for_group(
    mut reader: impl BufRead,
    expected_group_id: usize,
    timeout_s: f64,
) -> Result<Handshake> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(1.0));
    let mut found: Option<GroupLine> = None;

    loop {
        if Instant::now() > deadline {
            bail!("launcher handshake timed out after {timeout_s:.1}s");
        }

        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| anyhow!("read handshake line: {e}"))?;
        if n == 0 {
            // EOF — launcher exited before sending sentinel.
            bail!(
                "launcher exited before handshake completed; check stderr for the \
                 launcher's last log line"
            );
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Probe for the sentinel first; group lines have no `ready` field.
        // The sentinel is just an end-of-stream marker — its `num_groups`
        // is informational, so we accept any JSON object with a `ready`
        // key without a typed parse.
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!("parse handshake line {trimmed:?}: {e}"))?;
        if value.get("ready").is_some() {
            return found.map(|g| Handshake {
                server_name: g.server_name,
                shmem_name: g.shmem_name,
                caps: g.caps,
            }).ok_or_else(|| anyhow!(
                "launcher emitted ready sentinel without a handshake line for \
                 group {expected_group_id}"
            ));
        }

        let group: GroupLine = serde_json::from_value(value)
            .map_err(|e| anyhow!("parse group handshake line: {e}"))?;
        if group.group_id == expected_group_id {
            found = Some(group);
        }
        // Other groups are noise from this caller's POV — the parent
        // calls `start` once per group and each call has its own pipe,
        // so we shouldn't see foreign group_ids; treat them as a
        // protocol bug.
        else if found.is_none() {
            bail!(
                "launcher reported group_id={} on the pipe for group {expected_group_id}",
                group.group_id,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_caps_json() -> &'static str {
        // Single-line — the handshake protocol is line-delimited JSON.
        r#"{"total_pages":1024,"kv_page_size":32,"swap_pool_size":0,"max_batch_tokens":10240,"max_batch_size":512,"arch_name":"qwen3","vocab_size":151936,"max_model_len":4096,"activation_dtype":"bfloat16","snapshot_dir":"/tmp/snap","shmem_name":"/pie_shmem_g0"}"#
    }

    #[test]
    fn flavor_strings_match_toml_discriminator() {
        assert_eq!(SubprocessFlavor::Dev.as_str(), "dev");
        assert_eq!(SubprocessFlavor::Vllm.as_str(), "vllm");
        assert_eq!(SubprocessFlavor::Sglang.as_str(), "sglang");
        assert_eq!(SubprocessFlavor::Dev.module_name(), "pie_driver_dev");
    }

    #[test]
    fn handshake_parses_group_then_sentinel() {
        let stream = format!(
            "{{\"group_id\":0,\"server_name\":\"/tmp/sock\",\
              \"shmem_name\":\"/pie_shmem_g0\",\"caps\":{}}}\n\
             {{\"ready\":true,\"num_groups\":1}}\n",
            fixture_caps_json(),
        );
        let h = read_handshake_for_group(stream.as_bytes(), 0, 5.0).unwrap();
        assert_eq!(h.server_name, "/tmp/sock");
        assert_eq!(h.shmem_name, "/pie_shmem_g0");
        assert_eq!(h.caps.total_pages, 1024);
        assert_eq!(h.caps.arch_name, "qwen3");
    }

    #[test]
    fn handshake_rejects_eof_before_sentinel() {
        let stream = format!(
            "{{\"group_id\":0,\"server_name\":\"/tmp/sock\",\
              \"shmem_name\":\"/pie_shmem_g0\",\"caps\":{}}}\n",
            fixture_caps_json(),
        );
        let err = read_handshake_for_group(stream.as_bytes(), 0, 5.0).unwrap_err();
        assert!(
            err.to_string().contains("launcher exited before handshake"),
            "got: {err}"
        );
    }

    #[test]
    fn handshake_rejects_wrong_group_id() {
        let stream = format!(
            "{{\"group_id\":7,\"server_name\":\"/tmp/sock\",\
              \"shmem_name\":\"/pie_shmem_g7\",\"caps\":{}}}\n\
             {{\"ready\":true,\"num_groups\":1}}\n",
            fixture_caps_json(),
        );
        let err = read_handshake_for_group(stream.as_bytes(), 0, 5.0).unwrap_err();
        assert!(err.to_string().contains("group_id=7"), "got: {err}");
    }

    #[test]
    fn handshake_rejects_sentinel_without_group() {
        let stream = "{\"ready\":true,\"num_groups\":0}\n";
        let err = read_handshake_for_group(stream.as_bytes(), 0, 5.0).unwrap_err();
        assert!(
            err.to_string().contains("without a handshake line"),
            "got: {err}"
        );
    }
}
