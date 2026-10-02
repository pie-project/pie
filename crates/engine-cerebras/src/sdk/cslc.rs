//! Driving the CSL compiler (`cslc`) from Rust.
//!
//! `cslc` is a Python front-end that runs `cslc-driver` (and, with
//! `--memcpy`, the east/west memcpy layouts) and writes ELFs plus
//! `bin/out_rpc.json` into the output directory. It is spawned directly with
//! the SDK's bundled Python; no container is needed once the `/cb` and
//! `/cbcore` paths resolve (see the module docs of [`super`]).
//!
//! Locations come from the environment when set, otherwise from the
//! standard image layout:
//! - `PIE_CEREBRAS_CSLC`: the front-end script (default: the single
//!   `/cb/toolchains/cslang/rel-sdk-*/*/bin/cslc`);
//! - `PIE_CEREBRAS_PYTHON`: the interpreter (default
//!   `/python/python-x86_64/bin/python3`);
//! - `PIE_CEREBRAS_SDK_LIB`: shared with [`super::Sdk`] (default `/cbcore/lib`).

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

impl From<super::Target> for Arch {
    fn from(t: super::Target) -> Self {
        match t {
            super::Target::Wse2 => Arch::Wse2,
            super::Target::Wse3 => Arch::Wse3,
        }
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
        Compile {
            arch,
            layout: layout.into(),
            fabric_dims: (w + 7, h + 2),
            fabric_offsets: (4, 1),
            params: Vec::new(),
            memcpy: true,
            // One channel a row: host copies stream in parallel down the rows.
            channels: h.clamp(1, 16),
            import_paths: Vec::new(),
            out: out.into(),
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
    lib_dir: PathBuf,
}

impl Cslc {
    /// Finds the compiler from the environment or the standard image layout.
    pub fn find() -> Result<Self, Error> {
        let python = std::env::var_os("PIE_CEREBRAS_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/python/python-x86_64/bin/python3"));
        let frontend = match std::env::var_os("PIE_CEREBRAS_CSLC") {
            Some(p) => PathBuf::from(p),
            None => default_frontend()?,
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
            lib_dir: super::Sdk::default_lib_dir(),
        })
    }

    /// Whether the toolchain is present on this box.
    pub fn available() -> bool {
        Self::find().is_ok()
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
        // The 16-bit float type (`f16`) is brain float: the models' half
        // format, and what the host rounds bf16 handles to.
        cmd.arg("--fp16-format=bf16");
        for (k, v) in &compile.params {
            cmd.arg(format!("--params={k}:{v}"));
        }
        for p in &compile.import_paths {
            cmd.arg("--import-path").arg(p);
        }

        // The environment the SDK image's entry script would set up.
        let cbcore = self
            .lib_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("/cbcore"));
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
        .env("CBCORE", &cbcore)
        .env("LC_ALL", "C");
        if let Some(top) = self.cslang_top() {
            cmd.env("CSLANG_TOP", top);
        }
        cmd.env_remove("PYTHONHOME").env_remove("VIRTUAL_ENV");

        // The compiler is bounded (`PIE_CEREBRAS_COMPILE_TIMEOUT` seconds,
        // 1800 by default): a hung front end or linker must not hold a
        // fire forever.
        let timeout = std::time::Duration::from_secs(
            std::env::var("PIE_CEREBRAS_COMPILE_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1800),
        );
        cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
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

fn default_frontend() -> Result<PathBuf, Error> {
    let root = Path::new("/cb/toolchains/cslang");
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
