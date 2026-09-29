//! The device: a PJRT client on one addressable device, its compile cache,
//! and the one way a traced program runs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use dtype::Dtype;

use crate::error::{Fault, Result};
use crate::pjrt::{self, Api, Arg, Buffer, Client, ElementType, Executable};
use crate::trace::{Signature, storage};

/// A compiled program and the signature it was traced with.
pub struct Program {
    /// `None` on a dry device that traced the program without compiling it.
    pub exe: Option<Executable>,
    pub sig: Signature,
    pub text_bytes: usize,
}

/// Fields drop in order: the compiled programs go before the client that
/// owns them.
pub struct Device {
    cache: Mutex<HashMap<[u8; 32], Arc<Program>>>,
    ordinal: usize,
    kind: String,
    limit: Option<u64>,
    /// Where each compiled module is also written, for inspection
    /// (`PIE_XLA_DUMP`).
    dump: Option<PathBuf>,
    /// Where compiled executables persist across runs (`PIE_XLA_CACHE`,
    /// default `$XDG_CACHE_HOME/pie/xla` or `~/.cache/pie/xla`; `0` turns it
    /// off), keyed by the module text and this plugin and device.
    disk: Option<PathBuf>,
    /// `None` on a dry device that only traces (see [`Device::dry`]).
    client: Option<Client>,
    /// A dry device keeps every program text it was handed, and runs
    /// nothing: pools and weights hold no buffers (`crate::dry`).
    dry: Option<Mutex<Vec<String>>>,
    /// The device lock (`PIE_XLA_LOCK=1`), held while the device is open.
    _lock: Option<crate::bench::DeviceLock>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("kind", &self.kind)
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl Device {
    pub fn open(plugin: Option<&std::path::Path>, ordinal: usize) -> Result<Device> {
        // Opt-in: queue behind other processes that share the device through
        // the bench's lock file (tests, a second shell) instead of failing.
        let lock = std::env::var_os("PIE_XLA_LOCK")
            .is_some_and(|v| v != "0")
            .then(crate::bench::lock_device);
        let client = open_client(plugin).map_err(|e| Fault::NoDevice {
            detail: e.to_string(),
        })?;
        let device = *client.devices().get(ordinal).ok_or_else(|| Fault::NoDevice {
            detail: format!(
                "device {ordinal} asked of a client that addresses {}",
                client.devices().len()
            ),
        })?;
        let kind = client.device_kind(device).unwrap_or_default();
        let limit = client
            .memory_stats(device)
            .ok()
            .and_then(|stats| stats.bytes_limit)
            .and_then(|b| u64::try_from(b).ok());
        tracing::info!(kind, ?limit, "xla device open");
        Ok(Device {
            client: Some(client),
            dry: None,
            ordinal,
            kind,
            limit,
            cache: Mutex::new(HashMap::new()),
            dump: std::env::var_os("PIE_XLA_DUMP").map(PathBuf::from),
            disk: disk_cache(),
            _lock: lock,
        })
    }

    /// A device that traces and runs nothing: every program text it is
    /// handed is kept ([`Device::dry_texts`]); with `compile`, the plugin's
    /// client is opened (under the device lock) and each text is compiled
    /// too, but never run. Uploads land nothing; memory is unbounded.
    pub fn dry(compile: Option<(Option<&std::path::Path>, usize)>) -> Result<Device> {
        let (client, lock, ordinal) = match compile {
            None => (None, None, 0),
            Some((plugin, ordinal)) => {
                let lock = crate::bench::lock_device();
                let client = open_client(plugin).map_err(|e| Fault::NoDevice {
                    detail: e.to_string(),
                })?;
                (Some(client), Some(lock), ordinal)
            }
        };
        let kind = client
            .as_ref()
            .and_then(|c| c.devices().get(ordinal).and_then(|d| c.device_kind(*d).ok()))
            .unwrap_or_else(|| "dry".to_string());
        Ok(Device {
            client,
            dry: Some(Mutex::new(Vec::new())),
            ordinal,
            kind,
            limit: None,
            cache: Mutex::new(HashMap::new()),
            dump: std::env::var_os("PIE_XLA_DUMP").map(PathBuf::from),
            disk: None,
            _lock: lock,
        })
    }

    /// Whether this device only traces (see [`Device::dry`]).
    #[must_use]
    pub fn is_dry(&self) -> bool {
        self.dry.is_some()
    }

    /// The program texts a dry device was handed, oldest first.
    #[must_use]
    pub fn dry_texts(&self) -> Vec<String> {
        self.dry.as_ref().map_or_else(Vec::new, |texts| {
            texts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        })
    }

    fn client(&self) -> Result<&Client> {
        self.client.as_ref().ok_or(Fault::Deviceless)
    }

    #[must_use]
    pub fn api(&self) -> Option<&Arc<crate::pjrt::Api>> {
        self.client.as_ref().map(crate::pjrt::Client::api)
    }

    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    #[must_use]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Device memory the plugin reports, if it does.
    #[must_use]
    pub fn memory(&self) -> Option<u64> {
        self.limit
    }

    fn device(&self) -> Result<pjrt::Device> {
        Ok(self.client()?.devices()[self.ordinal])
    }

    /// Lands `bytes` as a `[rows, width]` array of `dtype` (packed dtypes as
    /// their raw row bytes).
    pub fn upload(&self, dtype: Dtype, rows: u32, width: u32, bytes: &[u8]) -> Result<Buffer> {
        let ty = storage(dtype, rows, width).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} plane, which has no storage form on this device"),
        })?;
        Ok(self
            .client()?
            .upload(self.device()?, bytes, element_type(ty.elem), &ty.dims)?)
    }

    /// Lands `bytes` as a rank-1 array of `len` elements of `ty`.
    pub fn upload_flat(&self, ty: ElementType, bytes: &[u8], len: i64) -> Result<Buffer> {
        Ok(self.client()?.upload(self.device()?, bytes, ty, &[len])?)
    }

    /// Zeros of `dtype` over `[rows, width]`.
    /// Zeros of `dtype` over `[rows, width]`, or nothing on a dry device.
    pub fn zeros_or_dry(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Option<Buffer>> {
        if self.is_dry() {
            return Ok(None);
        }
        self.zeros(dtype, rows, width).map(Some)
    }

    pub fn zeros(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Buffer> {
        let ty = storage(dtype, rows, width).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} plane, which has no storage form on this device"),
        })?;
        let bytes = element_type(ty.elem).bytes(usize::try_from(ty.elements()).unwrap_or(0));
        self.upload(dtype, rows, width, &vec![0u8; bytes])
    }

    /// The compiled form of `text`, from the cache or freshly built.
    pub fn program(&self, text: &str, sig: Signature) -> Result<Arc<Program>> {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(Arc::clone(hit));
        }
        if let Some(dir) = &self.dump {
            let name = format!("{}.mlir", blake3::hash(text.as_bytes()).to_hex());
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(name), text);
        }
        if let Some(texts) = &self.dry {
            texts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(text.to_string());
            let exe = match &self.client {
                None => None,
                Some(client) => Some(client.compile(text).map_err(|e| Fault::Xla {
                    what: "compile",
                    why: e.to_string(),
                })?),
            };
            let program = Arc::new(Program {
                exe,
                sig,
                text_bytes: text.len(),
            });
            self.cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key, Arc::clone(&program));
            return Ok(program);
        }
        let client = self.client()?;
        let started = std::time::Instant::now();
        let file = self.disk.as_ref().map(|dir| {
            let mut h = blake3::Hasher::new();
            h.update(text.as_bytes());
            h.update(client.api().path().to_string_lossy().as_bytes());
            let (major, minor) = client.api().version();
            h.update(format!("{major}.{minor}|{}", self.kind).as_bytes());
            dir.join(format!("{}.pjrt", h.finalize().to_hex()))
        });
        let cached = file
            .as_ref()
            .and_then(|f| std::fs::read(f).ok())
            .and_then(|bytes| client.deserialize(&bytes).ok());
        let exe = match cached {
            Some(exe) => {
                if let Some(file) = &file {
                    touch(file);
                }
                tracing::debug!(ms = started.elapsed().as_millis() as u64, "xla loaded a cached program");
                exe
            }
            None => {
                let exe = client.compile(text).map_err(|e| Fault::Xla {
                    what: "compile",
                    why: e.to_string(),
                })?;
                tracing::info!(
                    bytes = text.len(),
                    ms = started.elapsed().as_millis() as u64,
                    "xla compiled a program"
                );
                if let (Some(file), Ok(bytes)) = (&file, exe.serialize()) {
                    let _ = std::fs::create_dir_all(file.parent().unwrap_or(file));
                    let tmp = file.with_extension("tmp");
                    if std::fs::write(&tmp, bytes).is_ok() {
                        let _ = std::fs::rename(&tmp, file);
                    }
                    if let Some(dir) = file.parent() {
                        trim(dir, cache_cap());
                    }
                }
                exe
            }
        };
        let program = Arc::new(Program {
            exe: Some(exe),
            sig,
            text_bytes: text.len(),
        });
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, Arc::clone(&program));
        Ok(program)
    }

    #[must_use]
    pub fn compiled(&self) -> usize {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Runs `program` over `args` (in its parameter order); returns its
    /// results, and waits for the device when `wait`.
    pub fn run(&self, program: &Program, args: Vec<Arg<'_>>, wait: bool) -> Result<Vec<Buffer>> {
        if self.is_dry() {
            return Ok(Vec::new());
        }
        let exe = program.exe.as_ref().ok_or(Fault::Deviceless)?;
        let (outs, done) = exe.execute(self.device()?, args)?;
        if wait {
            done.wait()?;
        }
        Ok(outs)
    }
}

/// A client on `api`, waiting out a chip another process still holds: libtpu
/// admits one process per chip, and one that just let go of the bench's lock
/// file holds the chip a moment longer. `PIE_XLA_OPEN_WAIT` bounds the wait
/// in seconds (default 60; 0 fails at once).
pub(crate) fn open_client(plugin: Option<&std::path::Path>) -> Result<Client> {
    let wait = std::env::var("PIE_XLA_OPEN_WAIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(60);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(wait);
    let mut told = false;
    loop {
        let opened = Api::discover(plugin)
            .map_err(|e| e.to_string())
            .and_then(|api| Client::create(api).map_err(|e| e.to_string()));
        match opened {
            Ok(client) => return Ok(client),
            Err(why) => {
                let busy = why.contains("lockfile") || why.contains("already in use");
                if !busy || std::time::Instant::now() >= deadline {
                    return Err(Fault::NoDevice { detail: why });
                }
                if !told {
                    tracing::warn!(wait, "the device is held by another process; waiting for it");
                    told = true;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
}

/// The PJRT type an element lands as.
#[must_use]
pub fn element_type(elem: kernels_xla::hlo::Elem) -> ElementType {
    use kernels_xla::hlo::Elem;
    match elem {
        Elem::Pred => ElementType::Pred,
        Elem::I8 => ElementType::S8,
        Elem::I16 => ElementType::S16,
        Elem::I32 => ElementType::S32,
        Elem::I64 => ElementType::S64,
        Elem::U8 => ElementType::U8,
        Elem::U16 => ElementType::U16,
        Elem::U32 => ElementType::U32,
        Elem::U64 => ElementType::U64,
        Elem::F16 => ElementType::F16,
        Elem::Bf16 => ElementType::Bf16,
        Elem::F32 => ElementType::F32,
        Elem::F8E4m3fn => ElementType::F8E4m3fn,
        Elem::F8E5m2 => ElementType::F8E5m2,
        Elem::F8E8m0fnu => ElementType::F8E8m0fnu,
        Elem::F4E2m1fn => ElementType::F4E2m1fn,
        Elem::U4 => ElementType::U4,
    }
}

fn disk_cache() -> Option<PathBuf> {
    match std::env::var_os("PIE_XLA_CACHE") {
        Some(v) if v == "0" || v.is_empty() => None,
        Some(v) => Some(PathBuf::from(v)),
        None => std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .map(|base| base.join("pie").join("xla")),
    }
}

/// The most bytes of serialized programs the disk cache keeps:
/// `PIE_XLA_CACHE_GB`, default 4. A program is a few hundred MB at a large
/// model's shapes, so an unbounded cache fills a small disk in a day.
fn cache_cap() -> u64 {
    std::env::var("PIE_XLA_CACHE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(4 << 30, |gb| (gb * f64::from(1u32 << 30)) as u64)
}

/// Marks a cached program as just used, so `trim` keeps it longest.
fn touch(file: &std::path::Path) {
    if let Ok(f) = std::fs::File::options().append(true).open(file) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

/// Drops the least recently used programs until `dir` holds at most `cap`
/// bytes of them.
fn trim(dir: &std::path::Path, cap: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "pjrt"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort_by_key(|f| f.0);
    for (_, len, path) in files {
        if total <= cap {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
}
