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
    /// What the compiled program takes on the device (default on a dry
    /// device): its temp shares the runtime's one reservation, which the
    /// largest loaded program's sizes.
    pub memory: crate::pjrt::CompiledMemory,
}

/// Fields drop in order: the compiled programs go before the client that
/// owns them.
pub struct Device {
    cache: Mutex<HashMap<[u8; 32], Arc<Program>>>,
    /// Set once the warm ladder ran (`seal`): a program wanted after that is
    /// a serving fire missing the ladder.
    sealed: std::sync::atomic::AtomicBool,
    misses: std::sync::atomic::AtomicU64,
    /// Where a program a serving fire waits on is counted, from the end of
    /// the load on ([`Device::report_to`]).
    live: std::sync::OnceLock<engine::live::Live>,
    /// The largest temp of any program compiled or loaded, and of the
    /// load's own at its seal: the runtime's one temp reservation is the
    /// largest loaded program's, so a later program adds only its excess.
    temp_max: std::sync::atomic::AtomicU64,
    sealed_temp: std::sync::atomic::AtomicU64,
    ordinal: usize,
    kind: String,
    limit: Option<u64>,
    /// What the plugin reported in use when the device opened, before this
    /// load landed anything; zero when it reports nothing.
    held: u64,
    /// Where each compiled module is also written, for inspection
    /// (`PIE_XLA_DUMP`).
    dump: Option<PathBuf>,
    /// The cache files of every program this device compiled or loaded:
    /// the load's set, never evicted to make room.
    set: Mutex<std::collections::HashSet<PathBuf>>,
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
        let device = client.device(ordinal).ok_or_else(|| Fault::NoDevice {
            detail: format!(
                "device {ordinal} asked of a client that addresses {}",
                client.device_count()
            ),
        })?;
        let kind = client.device_kind(device).unwrap_or_default();
        let stats = client.memory_stats(device).ok();
        let limit = stats
            .as_ref()
            .and_then(|stats| stats.bytes_limit)
            .and_then(|b| u64::try_from(b).ok());
        let held = stats.map_or(0, |stats| u64::try_from(stats.bytes_in_use).unwrap_or(0));
        tracing::info!(kind, ?limit, held, "xla device open");
        Ok(Device {
            client: Some(client),
            dry: None,
            ordinal,
            kind,
            limit,
            held,
            cache: Mutex::new(HashMap::new()),
            sealed: std::sync::atomic::AtomicBool::new(false),
            misses: std::sync::atomic::AtomicU64::new(0),
            live: std::sync::OnceLock::new(),
            temp_max: std::sync::atomic::AtomicU64::new(0),
            sealed_temp: std::sync::atomic::AtomicU64::new(0),
            dump: std::env::var_os("PIE_XLA_DUMP").map(PathBuf::from),
            disk: disk_cache(),
            set: Mutex::new(std::collections::HashSet::new()),
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
            .and_then(|c| c.device(ordinal).and_then(|d| c.device_kind(d).ok()))
            // Without a client, the v6e the defaults were measured on.
            .unwrap_or_else(|| "TPU v6 lite".to_string());
        Ok(Device {
            client,
            dry: Some(Mutex::new(Vec::new())),
            ordinal,
            kind,
            limit: None,
            held: 0,
            cache: Mutex::new(HashMap::new()),
            sealed: std::sync::atomic::AtomicBool::new(false),
            misses: std::sync::atomic::AtomicU64::new(0),
            live: std::sync::OnceLock::new(),
            temp_max: std::sync::atomic::AtomicU64::new(0),
            sealed_temp: std::sync::atomic::AtomicU64::new(0),
            dump: std::env::var_os("PIE_XLA_DUMP").map(PathBuf::from),
            disk: None,
            set: Mutex::new(std::collections::HashSet::new()),
            _lock: lock,
        })
    }

    /// A dry device that keeps nothing it traces anywhere but in itself:
    /// no dump (`PIE_XLA_DUMP`), as a load's stage-only pass wants.
    pub fn quiet() -> Result<Device> {
        let mut device = Device::dry(None)?;
        device.dump = None;
        Ok(device)
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
            texts.lock().expect("texts lock poisoned").clone()
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

    /// Device memory in use when it opened, before this load landed
    /// anything (zero when the plugin reports none).
    #[must_use]
    pub fn held(&self) -> u64 {
        self.held
    }

    fn device(&self) -> Result<pjrt::Device<'_>> {
        self.client()?.device(self.ordinal).ok_or(Fault::Deviceless)
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

    /// [`Device::upload`] into the device's pinned host memory, which its
    /// programs read through a host parameter
    /// (`kernels_xla::hlo::Func::pin_host`).
    pub fn upload_host(&self, dtype: Dtype, rows: u32, width: u32, bytes: &[u8]) -> Result<Buffer> {
        let ty = storage(dtype, rows, width).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} plane, which has no storage form on this device"),
        })?;
        let client = self.client()?;
        let device = self.device()?;
        let host = client.memory(device, "pinned_host")?.ok_or_else(|| Fault::Unbound {
            what: "pinned host memory, which this device does not address".to_string(),
        })?;
        Ok(client.upload_to(device, Some(host), bytes, element_type(ty.elem), &ty.dims)?)
    }

    /// Lands `bytes` as a rank-1 array of `len` elements of `ty`.
    pub fn upload_flat(&self, ty: ElementType, bytes: &[u8], len: i64) -> Result<Buffer> {
        Ok(self.client()?.upload(self.device()?, bytes, ty, &[len])?)
    }

    /// Zeros of `dtype` over `[rows, width]`.
    /// Zeros of `dtype` over `[rows, width]`, or nothing on a dry device.
    pub fn zeros_or_dry(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Option<Buffer>> {
        if self.is_dry() {
            // Its program is the load's as on a device: traced (and compiled
            // where a dry device compiles), never run.
            self.zeros_program(dtype, rows, width)?;
            return Ok(None);
        }
        self.zeros(dtype, rows, width).map(Some)
    }

    /// The program that lays a `[rows, width]` plane of zeros.
    fn zeros_program(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Arc<Program>> {
        let ty = storage(dtype, rows, width).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} plane, which has no storage form on this device"),
        })?;
        let mut f = kernels_xla::hlo::Func::new("main");
        let z = if ty.elem.is_float() {
            f.const_f(ty.elem, 0.0, &ty.dims)
        } else {
            f.const_i(ty.elem, 0, &ty.dims)
        };
        f.pin_results_row_major();
        self.prepare(
            &f.module("zeros", &[z]),
            Signature {
                params: Vec::new(),
                results: vec![None],
            },
        )
    }

    /// A pool plane of zeros, row-major as every program that reads a pool
    /// lays it (`Func::pin_row_major`); an upload lands a narrow plane
    /// column-major.
    pub fn zeros(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Buffer> {
        let program = self.zeros_program(dtype, rows, width)?;
        let mut outs = self.run(&program, Vec::new(), true)?;
        outs.pop().ok_or_else(|| Fault::Unbound {
            what: "the zeros program's plane".into(),
        })
    }

    /// The compiled form of `text` a serving fire wants, from the cache or
    /// freshly built: one built after [`Device::seal`] is a fire the warm
    /// ladder missed.
    pub fn program(&self, text: &str, sig: Signature) -> Result<Arc<Program>> {
        self.compiled_form(text, sig, Some(&|| String::new()))
    }

    /// [`Device::program`] for a fire, `fire` naming its shape in the miss a
    /// build after [`Device::seal`] is.
    pub fn program_for(
        &self,
        text: &str,
        sig: Signature,
        fire: &dyn Fn() -> String,
    ) -> Result<Arc<Program>> {
        self.compiled_form(text, sig, Some(fire))
    }

    /// The compiled form of `text` a load or a program's registration
    /// prepares ahead of every fire that takes it, from the cache or
    /// freshly built: never a ladder miss, sealed or not.
    pub fn prepare(&self, text: &str, sig: Signature) -> Result<Arc<Program>> {
        self.compiled_form(text, sig, None)
    }

    /// `serving`: a fire's program (`Some`, naming the fire), built after the
    /// seal a ladder miss; `None` a load's or a registration's.
    fn compiled_form(
        &self,
        text: &str,
        sig: Signature,
        serving: Option<&dyn Fn() -> String>,
    ) -> Result<Arc<Program>> {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(hit) = self.cache.lock().expect("cache lock poisoned").get(&key) {
            return Ok(Arc::clone(hit));
        }
        if serving.is_some()
            && let Some(live) = self.live.get()
        {
            live.tally(engine::live::Tally::Compiled, 1);
        }
        if let Some(fire) = serving
            && self.sealed.load(std::sync::atomic::Ordering::Relaxed)
        {
            // Serving fires run programs the warm ladder prepared: one built
            // now is a fire the ladder missed, built on the fire's path.
            if let Some(live) = self.live.get() {
                live.tally(engine::live::Tally::BuiltInFire, 1);
            }
            let n = self
                .misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            tracing::warn!(
                bytes = text.len(),
                misses = n,
                fire = %fire(),
                "xla program outside the warm ladder"
            );
            assert!(
                std::env::var("PIE_XLA_WARM_STRICT").as_deref() != Ok("1"),
                "xla program outside the warm ladder ({} bytes)",
                text.len()
            );
        }
        if let Some(dir) = &self.dump {
            let name = format!("{}.mlir", blake3::hash(text.as_bytes()).to_hex());
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(name), text);
        }
        if let Some(texts) = &self.dry {
            texts
                .lock()
                .expect("texts lock poisoned")
                .push(text.to_string());
            let exe = match &self.client {
                None => None,
                Some(client) => Some(client.compile(text).map_err(|e| Fault::Xla {
                    what: "compile",
                    why: e.to_string(),
                })?),
            };
            let memory = exe
                .as_ref()
                .and_then(|e| e.compiled_memory().ok())
                .unwrap_or_default();
            let program = Arc::new(Program {
                exe,
                sig,
                text_bytes: text.len(),
                memory,
            });
            self.cache
                .lock()
                .expect("cache lock poisoned")
                .insert(key, Arc::clone(&program));
            return Ok(program);
        }
        let client = self.client()?;
        let started = std::time::Instant::now();
        let file = self.cache_file(client, text);
        let cached = file
            .as_ref()
            .and_then(|f| std::fs::read(f).ok())
            .and_then(unpacked)
            .and_then(|bytes| client.deserialize(&bytes).ok());
        let exe = match cached {
            Some(exe) => {
                if let Some(file) = &file {
                    touch(file);
                    self.keep(file);
                }
                tracing::debug!(
                    ms = started.elapsed().as_millis() as u64,
                    "xla loaded a cached program"
                );
                exe
            }
            None => {
                let exe = client.compile(text).map_err(|e| Fault::Xla {
                    what: "compile",
                    why: e.to_string(),
                })?;
                // A registration's program (prepared after the seal) built
                // rather than loaded: allowed, counted.
                if serving.is_none()
                    && self.sealed.load(std::sync::atomic::Ordering::Relaxed)
                    && let Some(live) = self.live.get()
                {
                    live.tally(engine::live::Tally::BuiltAtRegister, 1);
                }
                tracing::info!(
                    bytes = text.len(),
                    ms = started.elapsed().as_millis() as u64,
                    "xla compiled a program"
                );
                if let (Some(file), Ok(bytes)) = (&file, exe.serialize()) {
                    let bytes = packed(&bytes);
                    let dir = file.parent().unwrap_or(file);
                    let _ = std::fs::create_dir_all(dir);
                    // Room for it, made only of programs this load holds no
                    // claim on; the load's own set is never evicted.
                    self.make_room(dir, bytes.len() as u64)?;
                    // A name of this write's own: two registrations of one
                    // program compile it side by side and each lands its
                    // copy, the second rename replacing the first.
                    static WRITES: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let tmp = file.with_extension(format!(
                        "tmp.{}.{}",
                        std::process::id(),
                        WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    ));
                    std::fs::write(&tmp, &bytes)
                        .and_then(|()| std::fs::rename(&tmp, file))
                        .map_err(|why| Fault::Unbound {
                            what: format!(
                                "the program cache at {}, which took no {}-byte program: {why}",
                                dir.display(),
                                bytes.len()
                            ),
                        })?;
                    self.keep(file);
                }
                exe
            }
        };
        let memory = exe.compiled_memory().unwrap_or_default();
        self.temp_max
            .fetch_max(memory.temp, std::sync::atomic::Ordering::Relaxed);
        let program = Arc::new(Program {
            exe: Some(exe),
            sig,
            text_bytes: text.len(),
            memory,
        });
        self.cache
            .lock()
            .expect("cache lock poisoned")
            .insert(key, Arc::clone(&program));
        Ok(program)
    }

    /// Where the disk cache keeps the program compiled from `text` for this
    /// client's PJRT plugin and chip, if it keeps one.
    fn cache_file(&self, client: &Client, text: &str) -> Option<PathBuf> {
        let dir = self.disk.as_ref()?;
        let mut h = blake3::Hasher::new();
        h.update(text.as_bytes());
        h.update(client.api().path().to_string_lossy().as_bytes());
        let (major, minor) = client.api().version();
        h.update(format!("{major}.{minor}|{}", self.kind).as_bytes());
        // The compiler's own flags shape the program as much as its text.
        for var in COMPILER_ENV {
            h.update(b"|");
            h.update(std::env::var_os(var).unwrap_or_default().as_encoded_bytes());
        }
        h.update(format!("|{:?}", crate::pjrt::options::TPU_FLAGS).as_bytes());
        Some(dir.join(format!("{}.pjrt", h.finalize().to_hex())))
    }

    /// Holds `file` in the load's set.
    fn keep(&self, file: &std::path::Path) {
        self.set
            .lock()
            .expect("cache set lock poisoned")
            .insert(file.to_path_buf());
    }

    /// The bytes the load's set takes in the program cache.
    #[must_use]
    pub fn set_bytes(&self) -> u64 {
        let set = self.set.lock().expect("cache set lock poisoned");
        set.iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum()
    }

    /// Frees `need` bytes on the cache's filesystem by dropping the least
    /// recently used programs outside the load's set; refuses by name when
    /// those do not free enough (the set itself is never evicted).
    fn make_room(&self, dir: &std::path::Path, need: u64) -> Result<()> {
        let Some(mut free) = free_bytes(dir) else {
            return Ok(());
        };
        if free >= need {
            return Ok(());
        }
        let set = self.set.lock().expect("cache set lock poisoned").clone();
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(std::result::Result::ok)
                    .filter(|e| e.path().extension().is_some_and(|x| x == "pjrt"))
                    .filter(|e| !set.contains(&e.path()))
                    .filter_map(|e| {
                        let meta = e.metadata().ok()?;
                        Some((meta.modified().ok()?, meta.len(), e.path()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        files.sort_by_key(|f| f.0);
        for (_, len, path) in files {
            if free >= need {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                free += len;
            }
        }
        if free < need {
            return Err(Fault::Ceiling {
                what: "disk bytes the load's compiled programs take (the program cache)",
                need: self.set_bytes() + need,
                have: self.set_bytes() + free,
            });
        }
        Ok(())
    }

    /// Whether compiled programs go to a disk cache.
    #[must_use]
    pub fn parks(&self) -> bool {
        self.disk.is_some()
    }

    /// Whether `text`'s program is had without a compile: held
    /// ([`Self::holds`]), or in the disk cache.
    #[must_use]
    pub fn has_without_compile(&self, text: &str) -> bool {
        self.holds(text) || self.on_disk(text)
    }

    /// Whether the disk cache holds the program compiled from `text`.
    #[must_use]
    pub fn on_disk(&self, text: &str) -> bool {
        self.client()
            .ok()
            .and_then(|client| self.cache_file(client, text))
            .is_some_and(|file| file.exists())
    }

    /// Drops the program compiled from `text` from the cache: held nowhere
    /// else, it frees its host memory, and the next `program` of the text
    /// loads it from the disk cache.
    pub fn forget(&self, text: &str) {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        self.cache.lock().expect("cache lock poisoned").remove(&key);
    }

    /// [`Device::forget`] by the cache key of a program's text (the blake3
    /// of it): a stage the guest stage cache evicted.
    pub fn forget_key(&self, key: &[u8; 32]) {
        self.cache.lock().expect("cache lock poisoned").remove(key);
    }

    /// Marks the warm ladder done: every program wanted from now on is
    /// counted (and logged) as a miss.
    /// Whether the load sealed its programs: a serving fire's program built
    /// since is a ladder miss.
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn seal(&self) {
        if let Some(dir) = &self.disk {
            let held = self.set.lock().expect("cache set lock poisoned").len();
            tracing::info!(
                programs = held,
                bytes = self.set_bytes(),
                free = free_bytes(dir).unwrap_or(0),
                dir = %dir.display(),
                "xla program cache: the load's set, never evicted"
            );
        }
        self.sealed_temp.store(
            self.temp_max.load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.sealed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// The largest temp of the load's own programs (those compiled before
    /// the seal; every one so far where it has not sealed).
    #[must_use]
    pub fn load_temp(&self) -> u64 {
        if self.sealed.load(std::sync::atomic::Ordering::Relaxed) {
            self.sealed_temp.load(std::sync::atomic::Ordering::Relaxed)
        } else {
            self.temp_max.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// From now on every program a serving fire waits on (built, or loaded
    /// from the disk cache) is counted in `live`, prewarmed or not.
    pub fn report_to(&self, live: engine::live::Live) {
        let _ = self.live.set(live);
    }

    /// Programs wanted after [`Device::seal`] that the ladder had not built.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the program of `text` is held compiled.
    #[must_use]
    pub fn holds(&self, text: &str) -> bool {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        self.cache
            .lock()
            .expect("cache lock poisoned")
            .contains_key(&key)
    }

    #[must_use]
    pub fn compiled(&self) -> usize {
        self.cache.lock().expect("cache lock poisoned").len()
    }

    /// Runs `program` over `args` (in its parameter order); returns its
    /// results, and waits for the device when `wait`.
    pub fn run(&self, program: &Program, args: Vec<Arg<'_>>, wait: bool) -> Result<Vec<Buffer>> {
        if self.is_dry() {
            return Ok(Vec::new());
        }
        let exe = program.exe.as_ref().ok_or(Fault::Deviceless)?;
        let at = profile::before(self.api());
        let (outs, done) = exe.execute(self.device()?, args)?;
        if wait || at.is_some() {
            done.wait()?;
        }
        if let Some(at) = at {
            profile::after(at, program.text_bytes);
        }
        Ok(outs)
    }
}

/// `PIE_XLA_PROFILE=<skip>:<count>[,<skip>:<count>...]`: a device trace of
/// runs `skip..skip+count` (every program run counts), written as an
/// `XSpace` beside `PIE_BENCH_SERVER_LOG` (or under `PIE_XLA_PROFILE_DIR`).
/// Each traced run waits for the device.
mod profile {
    use std::sync::Mutex;

    struct State {
        runs: u64,
        windows: Vec<(u64, u64)>,
        live: Option<(crate::pjrt::profiler::Profiler, u64, std::time::Instant)>,
        sizes: Vec<usize>,
        seq: u32,
    }

    static STATE: Mutex<Option<State>> = Mutex::new(None);

    fn windows() -> Vec<(u64, u64)> {
        std::env::var("PIE_XLA_PROFILE")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|w| {
                        let (a, b) = w.split_once(':')?;
                        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn dir() -> std::path::PathBuf {
        if let Some(d) = std::env::var_os("PIE_XLA_PROFILE_DIR") {
            return d.into();
        }
        if let Some(log) = std::env::var_os("PIE_BENCH_SERVER_LOG") {
            let p = std::path::PathBuf::from(log);
            if let Some(parent) = p.parent() {
                return parent.to_path_buf();
            }
        }
        std::env::temp_dir()
    }

    /// The run's index when it is traced.
    pub(super) fn before(api: Option<&std::sync::Arc<crate::pjrt::Api>>) -> Option<u64> {
        let mut guard = STATE.lock().ok()?;
        let state = guard.get_or_insert_with(|| State {
            runs: 0,
            windows: windows(),
            live: None,
            sizes: Vec::new(),
            seq: 0,
        });
        if state.windows.is_empty() {
            return None;
        }
        let at = state.runs;
        state.runs += 1;
        if state.live.is_none()
            && let Some(&(skip, count)) = state.windows.first()
            && at == skip
            && let Some(api) = api
        {
            match crate::pjrt::profiler::Profiler::start(api) {
                Ok(p) => {
                    eprintln!("PIE_XLA_PROFILE: tracing runs {skip}..{}", skip + count);
                    state.live = Some((p, skip + count, std::time::Instant::now()));
                    state.sizes.clear();
                }
                Err(e) => {
                    eprintln!("PIE_XLA_PROFILE: the profiler did not start: {e}");
                    state.windows.remove(0);
                }
            }
        }
        state.live.as_ref().map(|_| at)
    }

    pub(super) fn after(at: u64, text_bytes: usize) {
        let Ok(mut guard) = STATE.lock() else {
            return;
        };
        let Some(state) = guard.as_mut() else {
            return;
        };
        state.sizes.push(text_bytes);
        let end = state.live.as_ref().map(|l| l.1);
        if end != Some(at + 1) {
            return;
        }
        let Some((p, _, t0)) = state.live.take() else {
            return;
        };
        state.windows.remove(0);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let seq = state.seq;
        state.seq += 1;
        match p.finish() {
            Ok(bytes) => {
                let d = dir();
                let _ = std::fs::create_dir_all(&d);
                let path = d.join(format!(
                    "xla-profile-{}-{seq}.xplane.pb",
                    std::process::id()
                ));
                let wrote = std::fs::write(&path, &bytes);
                eprintln!(
                    "PIE_XLA_PROFILE: {} runs in {ms:.1}ms (program texts {:?} B); {} bytes -> {} ({wrote:?})",
                    state.sizes.len(),
                    state.sizes,
                    bytes.len(),
                    path.display()
                );
            }
            Err(e) => eprintln!("PIE_XLA_PROFILE: collect failed: {e}"),
        }
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
                    tracing::warn!(
                        wait,
                        "the device is held by another process; waiting for it"
                    );
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

/// The environment the plugin's compiler reads its flags from: part of a
/// cached program's key, so a program compiled under other flags is not
/// loaded in its place.
const COMPILER_ENV: [&str; 2] = ["LIBTPU_INIT_ARGS", "XLA_FLAGS"];

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

/// A serialized program as the disk cache keeps it: zstd at level 1. A
/// program is a few hundred MB and about two fifths zeros (v6e Qwen3-0.6B:
/// 289 MB, ~80 MB packed), so the cache holds a model's whole warm ladder
/// rather than evicting half of it on every load.
fn packed(bytes: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(bytes, 1).unwrap_or_else(|_| bytes.to_vec())
}

/// A cached program's bytes, unpacked as `packed` wrote them (a zstd
/// frame); a file that is no such frame is no cached program (`None`, a
/// miss).
fn unpacked(bytes: Vec<u8>) -> Option<Vec<u8>> {
    zstd::stream::decode_all(bytes.as_slice()).ok()
}

/// Marks a cached program as just used, so `make_room` keeps it longest.
fn touch(file: &std::path::Path) {
    if let Ok(f) = std::fs::File::options().append(true).open(file) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

/// The bytes free on the filesystem holding `dir`, where it says.
#[allow(unsafe_code)]
fn free_bytes(dir: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a valid C string and `stat` a writable statvfs.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// The host memory free for new allocations, where the host says: the least
/// of `MemAvailable` and what each capped cgroup ancestor of the process
/// leaves ([`engine::host_memory`]). A cap the host's `MemAvailable` does
/// not see OOM-kills a load or a registration whose compiles fit the host
/// but not the cap.
pub(crate) fn available_memory() -> Option<u64> {
    engine::host_memory::available()
        .ok()
        .map(|available| available.bytes())
}

/// The host memory a compiled program holds: ~64 bytes a byte of its text,
/// less its Mosaic kernels' bodies (`tpu_custom_call` configs, one HLO
/// instruction each however long its base64), counted at their own size.
pub(crate) fn hold_bytes(text: &str) -> u64 {
    const BODY: &str = "\\22body\\22: \\22";
    let (mut kernels, mut rest) = (0, text);
    while let Some(at) = rest.find(BODY) {
        let tail = &rest[at + BODY.len()..];
        let len = tail.find("\\22").unwrap_or(tail.len());
        kernels += len;
        rest = &tail[len..];
    }
    (((text.len() - kernels) as u64) << 6) + kernels as u64
}

/// The host memory compiling `text` takes while it runs: ~0.55 GiB a MiB
/// of its text (v5e, Qwen3.6-35B-A3B's 9 MiB pf256 programs: 5 GB each),
/// counted at 0.75 GiB a MiB.
pub(crate) fn compile_need(text: &str) -> u64 {
    ((hold_bytes(text) >> 6) * 3) << 8
}

/// Admits compiles into a host memory room fixed when it is made: one
/// starts once its own need and every running one's fit (or none runs),
/// woken by one that ends, never by a poll of the free memory. A compile
/// in flight may not have taken its memory yet, so each counts until it
/// ends.
pub(crate) struct Admission {
    room: u64,
    running: Mutex<u64>,
    ended: std::sync::Condvar,
}

/// A compile [`Admission`] let in; its need leaves the room when dropped.
pub(crate) struct Admitted<'a> {
    by: &'a Admission,
    need: u64,
}

impl Admission {
    pub(crate) fn new(room: u64) -> Admission {
        Admission {
            room,
            running: Mutex::new(0),
            ended: std::sync::Condvar::new(),
        }
    }

    /// Waits until a compile of `text` fits the room.
    pub(crate) fn admit(&self, text: &str) -> Admitted<'_> {
        let need = compile_need(text);
        let mut busy = self.running.lock().expect("the compile admission lock poisoned");
        while *busy > 0 && *busy + need > self.room {
            busy = self.ended.wait(busy).expect("the compile admission lock poisoned");
        }
        *busy += need;
        Admitted { by: self, need }
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        *self.by.running.lock().expect("the compile admission lock poisoned") -= self.need;
        self.by.ended.notify_all();
    }
}
