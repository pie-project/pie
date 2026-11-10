//! The Cerebras toolchain a build links with: the CSL compiler (`cslc`) and `cs-readelf`, run
//! as the SDK image lays them out.
//!
//! `cslc` is a Python front-end that runs `cslc-driver` (and, with `--memcpy`, the east / west
//! memcpy layouts) and writes ELFs plus `bin/out_rpc.json` into the output directory. It is
//! spawned with the SDK's bundled Python; no container is needed once the `/cb` and `/cbcore`
//! paths resolve. Where they live is the [`Toolchain`] config (every field defaults to the
//! image's layout).
//!
//! A memcpy program's fabric is [`memcpy_fabric`]'s halo around it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Which wafer generation to compile for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Wse2,
    Wse3,
}

impl Arch {
    fn flag(self) -> &'static str {
        match self {
            Arch::Wse2 => "wse2",
            Arch::Wse3 => "wse3",
        }
    }
}

/// Where the toolchain lives: the `[engine.toolchain]` table. Unset fields take the SDK image's
/// layout.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Toolchain {
    /// The `cslc` front-end (default: the newest `<cslang_root>/rel-*/*/bin/cslc`).
    pub cslc: Option<PathBuf>,
    /// Where the default `cslc` is looked for (default `/cb/toolchains/cslang`).
    pub cslang_root: Option<PathBuf>,
    /// The interpreter `cslc` runs under (default `/python/python-x86_64/bin/python3`).
    pub python: Option<PathBuf>,
    /// The SDK's libraries (default `/cbcore/lib`); its parent is the image's `/cbcore`.
    pub sdk_lib: Option<PathBuf>,
    /// Seconds a compile may run before it is killed (default 3600).
    pub compile_timeout: Option<u64>,
}

impl Toolchain {
    /// The SDK's library directory.
    pub fn sdk_lib(&self) -> PathBuf {
        self.sdk_lib
            .clone()
            .unwrap_or_else(|| PathBuf::from("/cbcore/lib"))
    }

    /// The image's `/cbcore` (the library directory's parent).
    fn cbcore(&self) -> PathBuf {
        self.sdk_lib()
            .parent()
            .map_or_else(|| PathBuf::from("/cbcore"), Path::to_path_buf)
    }
}

/// One compilation: a layout file, the fabric it targets and the values of
/// its comptime parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compile {
    pub arch: Arch,
    /// The top-level layout file; sibling files it imports resolve relative to it.
    pub layout: PathBuf,
    /// Simulated or reserved fabric size, columns by rows.
    pub fabric_dims: (u32, u32),
    /// Where the program rectangle sits inside the fabric, `(x, y)`.
    pub fabric_offsets: (u32, u32),
    /// `--params name:value` pairs handed to the layout.
    pub params: Vec<(String, String)>,
    /// Link the host memcpy infrastructure (`--memcpy`).
    pub memcpy: bool,
    /// Number of memcpy channels; must be at least 1 when `memcpy` is set.
    pub channels: u32,
    /// Extra `--import-path` directories.
    pub import_paths: Vec<PathBuf>,
    /// Output directory (`-o`), created if missing.
    pub out: PathBuf,
    /// The 16-bit float type `f16` names: IEEE (`false`; the wafer program's kernels assume it)
    /// or brain float (`--fp16-format=bf16`).
    pub bf16: bool,
}

impl Compile {
    /// A memcpy program on the smallest fabric that fits a `w` by `h`
    /// rectangle: the memcpy layouts need 7 extra columns and 2 extra rows,
    /// with the program placed at `(4, 1)`.
    pub fn memcpy(
        arch: Arch,
        layout: impl Into<PathBuf>,
        w: u32,
        h: u32,
        out: impl Into<PathBuf>,
    ) -> Self {
        let (fabric_dims, fabric_offsets) = memcpy_fabric(w, h);
        Compile {
            arch,
            layout: layout.into(),
            fabric_dims,
            fabric_offsets,
            params: Vec::new(),
            memcpy: true,
            // One channel a row: host copies stream in parallel down the rows.
            channels: h.clamp(1, 16),
            import_paths: Vec::new(),
            out: out.into(),
            bf16: false,
        }
    }

    pub fn param(mut self, name: &str, value: impl ToString) -> Self {
        self.params.push((name.to_string(), value.to_string()));
        self
    }
}

/// A failed compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The front-end or interpreter could not be located.
    Toolchain(String),
    /// The compiler could not be spawned.
    Spawn(String),
    /// The compiler exited with a failure; its stderr tail is included.
    Failed { status: Option<i32>, stderr: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Toolchain(e) => write!(f, "cslc toolchain not found: {e}"),
            Error::Spawn(e) => write!(f, "cannot spawn cslc: {e}"),
            Error::Failed { status, stderr } => {
                write!(f, "cslc failed (status {status:?}):\n{stderr}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// The located compiler.
#[derive(Debug, Clone)]
pub struct Cslc {
    python: PathBuf,
    frontend: PathBuf,
    cbcore: PathBuf,
    lib_dir: PathBuf,
    timeout: std::time::Duration,
}

impl Cslc {
    /// Finds the compiler where `tc` says (else the image's layout).
    pub fn find(tc: &Toolchain) -> Result<Self, Error> {
        let python = tc
            .python
            .clone()
            .unwrap_or_else(|| PathBuf::from("/python/python-x86_64/bin/python3"));
        let frontend = match &tc.cslc {
            Some(p) => p.clone(),
            None => default_frontend(tc.cslang_root.as_deref())?,
        };
        for (what, p) in [("python", &python), ("cslc", &frontend)] {
            if !p.is_file() {
                return Err(Error::Toolchain(format!(
                    "{what} not found at {}",
                    p.display()
                )));
            }
        }
        Ok(Cslc {
            python,
            frontend,
            cbcore: tc.cbcore(),
            lib_dir: tc.sdk_lib(),
            timeout: std::time::Duration::from_secs(tc.compile_timeout.unwrap_or(3600)),
        })
    }

    fn cslang_top(&self) -> Option<&Path> {
        // .../bin/cslc -> ...
        self.frontend.parent()?.parent()
    }

    /// Runs the compiler; on success the artifacts are in `compile.out`.
    pub fn compile(&self, compile: &Compile) -> Result<(), Error> {
        std::fs::create_dir_all(&compile.out).map_err(|e| Error::Spawn(e.to_string()))?;
        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.frontend)
            .arg(format!("--arch={}", compile.arch.flag()))
            .arg(&compile.layout)
            .arg(format!(
                "--fabric-dims={},{}",
                compile.fabric_dims.0, compile.fabric_dims.1
            ))
            .arg(format!(
                "--fabric-offsets={},{}",
                compile.fabric_offsets.0, compile.fabric_offsets.1
            ))
            .arg("-o")
            .arg(&compile.out);
        if compile.memcpy {
            cmd.arg("--memcpy")
                .arg("--channels")
                .arg(compile.channels.to_string());
        }
        if compile.bf16 {
            cmd.arg("--fp16-format=bf16");
        }
        for (k, v) in &compile.params {
            cmd.arg(format!("--params={k}:{v}"));
        }
        for p in &compile.import_paths {
            cmd.arg("--import-path").arg(p);
        }

        // The environment the SDK image's entry script would set up.
        let cbcore = &self.cbcore;
        let mut path = vec![cbcore.join("bin")];
        if let Some(top) = self.cslang_top() {
            path.push(top.join("bin"));
        }
        if let Some(py_bin) = self.python.parent() {
            path.push(py_bin.to_path_buf());
        }
        path.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let py_root = cbcore.join("py_root");
        cmd.env(
            "PATH",
            std::env::join_paths(path).map_err(|e| Error::Spawn(e.to_string()))?,
        )
        .env("LD_LIBRARY_PATH", &self.lib_dir)
        .env(
            "PYTHONPATH",
            format!(
                "{}:{}",
                py_root.display(),
                py_root.join("cerebras").display()
            ),
        )
        .env("CBCORE", cbcore)
        .env("LC_ALL", "C");
        if let Some(top) = self.cslang_top() {
            cmd.env("CSLANG_TOP", top);
        }
        cmd.env_remove("PYTHONHOME").env_remove("VIRTUAL_ENV");

        // The compiler is bounded: a hung front end or linker must not hold a load forever.
        let timeout = self.timeout;
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| Error::Spawn(e.to_string()))?;
        let started = std::time::Instant::now();
        // Drain stderr on a thread so a chatty compiler cannot block on a
        // full pipe while we wait.
        let stderr_pipe = child.stderr.take();
        let drain = std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(mut pipe) = stderr_pipe {
                use std::io::Read;
                let _ = pipe.read_to_string(&mut text);
            }
            text
        });
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() > timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::Failed {
                        status: None,
                        stderr: format!("cslc ran past {timeout:?} and was killed"),
                    });
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(200)),
                Err(e) => return Err(Error::Spawn(e.to_string())),
            }
        };
        let stderr_text = drain.join().unwrap_or_default();
        if !status.success() {
            let tail: Vec<&str> = stderr_text
                .lines()
                .rev()
                .take(40)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(Error::Failed {
                status: status.code(),
                stderr: tail.join("\n"),
            });
        }
        Ok(())
    }
}

impl Cslc {
    /// The `cslc` front-end this toolchain runs (its path names the SDK release).
    pub fn frontend(&self) -> &Path {
        &self.frontend
    }

    /// The toolchain's `cs-readelf` (the image's `/cbcore/bin`).
    fn readelf(&self) -> PathBuf {
        self.cbcore.join("bin/cs-readelf")
    }

    /// The least stack room of any PE image under a compile's `out` directory: the bytes between
    /// an image's used memory (`cs-readelf -m`: code, data and the task table) and `0xC000`, where
    /// the stack, which the linker does not count, grows down from. Every chunk of an image holds
    /// the same program, so one a image is read.
    pub fn stack_room(&self, out: &Path) -> Result<u32, Error> {
        let mut images: std::collections::BTreeMap<u32, PathBuf> = Default::default();
        let bin = out.join("bin");
        for entry in std::fs::read_dir(&bin).map_err(|e| Error::Spawn(format!("{}: {e}", bin.display())))? {
            let path = entry.map_err(|e| Error::Spawn(e.to_string()))?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let Some(rest) = name.strip_prefix("out_").and_then(|r| r.strip_suffix(".elf")) else {
                continue;
            };
            let Some((image, chunk)) = rest.split_once('_') else {
                continue;
            };
            let (Ok(image), true) = (image.parse(), chunk.bytes().all(|b| b.is_ascii_digit())) else {
                continue;
            };
            let slot = images.entry(image).or_insert_with(|| path.clone());
            if path < *slot {
                *slot = path;
            }
        }
        let mut least: Option<u32> = None;
        for path in images.values() {
            let text = Command::new(self.readelf())
                .arg("-m")
                .arg(path)
                .env("LD_LIBRARY_PATH", &self.lib_dir)
                .output()
                .map_err(|e| Error::Spawn(format!("cs-readelf: {e}")))?;
            let text = String::from_utf8_lossy(&text.stdout);
            let used = text
                .lines()
                .filter_map(|l| {
                    // "(4, 1): 16992 bytes (8496 words)"
                    let (_, tail) = l.split_once("):")?;
                    let mut words = tail.split_whitespace();
                    let n = words.next()?.parse::<u32>().ok()?;
                    (words.next()? == "bytes").then_some(n)
                })
                .max()
                .unwrap_or(0);
            let room = 0xC000u32.saturating_sub(used);
            least = Some(least.map_or(room, |l| l.min(room)));
        }
        least.ok_or_else(|| Error::Toolchain(format!("{}: no PE images", out.display())))
    }
}

/// The smallest fabric a memcpy program of `w` x `h` PEs fits, and its
/// offset: the memcpy layouts take 7 extra columns and 2 extra rows, the
/// program placed at (4, 1).
#[must_use]
pub fn memcpy_fabric(w: u32, h: u32) -> ((u32, u32), (u32, u32)) {
    ((w + 7, h + 2), (4, 1))
}

fn default_frontend(root: Option<&Path>) -> Result<PathBuf, Error> {
    let root = root.unwrap_or(Path::new("/cb/toolchains/cslang"));
    let mut found = Vec::new();
    for release in read_dirs(root) {
        for build in read_dirs(&release) {
            let candidate = build.join("bin/cslc");
            if candidate.is_file() {
                found.push(candidate);
            }
        }
    }
    found.sort();
    found
        .pop()
        .ok_or_else(|| Error::Toolchain(format!("no bin/cslc under {}", root.display())))
}

fn read_dirs(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default()
}
