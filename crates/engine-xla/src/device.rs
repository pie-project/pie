//! The device: a PJRT client on one addressable device, its compile cache,
//! and the one way a traced program runs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use dtype::Dtype;

use crate::error::{Fault, Result};
use crate::pjrt::{self, Api, Arg, Buffer, Client, ElementType, Event, Executable};
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
    /// The host memory compiling a text took, measured with the compile
    /// alone (`guest::run`): by the text's hash, persisted beside its
    /// compiled form in the disk cache ([`Device::keep_compile_peak`]).
    peaks: Mutex<HashMap<[u8; 32], u64>>,
    /// A gate a program's disk load holds ([`Device::load_named`]).
    loading: Mutex<HashMap<[u8; 32], Arc<Mutex<()>>>>,
    /// Records dropped because their program did not load ([`Device::mend_record`]).
    mended: std::sync::atomic::AtomicU64,
    /// The loads a fire waits on, under way or waiting ([`Device::load_named`]):
    /// loading behind serving starts none while any is ([`Device::wait_demand`]).
    demand: Mutex<usize>,
    quiet: std::sync::Condvar,
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
    /// The PJRT plugin's build (`Client::platform_version`): what the
    /// device's stored measurements are keyed by beside its kind.
    platform: String,
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
        let platform = client.platform_version().unwrap_or_default();
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
            platform,
            kind,
            limit,
            held,
            cache: Mutex::new(HashMap::new()),
            peaks: Mutex::new(HashMap::new()),
            loading: Mutex::new(HashMap::new()),
            mended: std::sync::atomic::AtomicU64::new(0),
            demand: Mutex::new(0),
            quiet: std::sync::Condvar::new(),
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
        let platform = client
            .as_ref()
            .and_then(|c| c.platform_version().ok())
            .unwrap_or_default();
        Ok(Device {
            client,
            dry: Some(Mutex::new(Vec::new())),
            ordinal,
            platform,
            kind,
            limit: None,
            held: 0,
            cache: Mutex::new(HashMap::new()),
            peaks: Mutex::new(HashMap::new()),
            loading: Mutex::new(HashMap::new()),
            mended: std::sync::atomic::AtomicU64::new(0),
            demand: Mutex::new(0),
            quiet: std::sync::Condvar::new(),
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

    /// The PJRT plugin's build (`Client::platform_version`).
    #[must_use]
    pub fn platform(&self) -> &str {
        &self.platform
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

    /// The disk cache's name for the program compiled from `text` (its
    /// file under the cache dir), if the device keeps a disk cache.
    pub fn cached_name(&self, text: &str) -> Option<String> {
        let client = self.client().ok()?;
        let file = self.cache_file(client, text)?;
        file.file_name()?.to_str().map(str::to_string)
    }

    /// Whether the disk cache holds the program `code` names: for one a
    /// record names, its file as long as the record kept it.
    pub fn code_on_disk(&self, code: &Code) -> bool {
        match code {
            Code::Text(text) => self.on_disk(text),
            Code::Cached { name, file_len, .. } => self
                .disk
                .as_ref()
                .and_then(|dir| std::fs::metadata(dir.join(name)).ok())
                .is_some_and(|meta| meta.len() == *file_len),
        }
    }

    /// [`Device::prepare`] of `code`: a cached one loads by its name, never
    /// compiled; `None` where its file is gone.
    pub fn prepare_code(&self, code: &Code, sig: Signature) -> Result<Option<Arc<Program>>> {
        match code {
            Code::Text(text) => self.prepare(text, sig).map(Some),
            Code::Cached {
                hash, name, len, ..
            } => self.load_named(hash, name, *len, sig, false),
        }
    }

    /// [`Device::load_parked`] of `code`: what the first fire of its shape
    /// waits on (the loader's load of it under way, or its own), said with
    /// the wait.
    pub fn load_parked_code(&self, code: &Code, sig: Signature) -> Result<Option<Arc<Program>>> {
        let started = std::time::Instant::now();
        let program = self.load_parked_inner(code, sig);
        tracing::info!(
            waited_ms = started.elapsed().as_secs_f64() * 1e3,
            bytes = code.len(),
            "xla a fire took its program from the disk cache"
        );
        program
    }

    fn load_parked_inner(&self, code: &Code, sig: Signature) -> Result<Option<Arc<Program>>> {
        match code {
            Code::Text(text) => self.load_parked(text, sig),
            Code::Cached {
                hash, name, len, ..
            } => self.load_named(hash, name, *len, sig, true),
        }
    }

    /// `code`'s program loaded from the disk cache behind serving (not a
    /// fire's load, so not counted as one); `None` where its file is gone
    /// or does not load.
    pub fn load_behind(&self, code: &Code, sig: Signature) -> Result<Option<Arc<Program>>> {
        match code {
            Code::Cached {
                hash, name, len, ..
            } => self.load_named(hash, name, *len, sig, false),
            Code::Text(text) => match self.cached_name(text) {
                Some(name) => self.load_named(&code.hash(), &name, text.len(), sig, false),
                None => Ok(None),
            },
        }
    }

    /// [`Device::forget`] of `code`.
    pub fn forget_code(&self, code: &Code) {
        self.forget_key(&code.hash());
    }

    /// The program the disk cache holds under `name` (its text `hash`,
    /// `len` bytes long), loaded and held in memory; `None` where the file
    /// is gone or unreadable. `parked`: the first fire of its shape loads it.
    fn load_named(
        &self,
        hash: &[u8; 32],
        name: &str,
        len: usize,
        sig: Signature,
        parked: bool,
    ) -> Result<Option<Arc<Program>>> {
        if let Some(hit) = self.cache.lock().expect("cache lock poisoned").get(hash) {
            return Ok(Some(Arc::clone(hit)));
        }
        // A fire waits on this load: loading behind serving starts no other
        // until it is done.
        let _demand = parked.then(|| Demand::on(self));
        // One load a program: a fire's load of one loading behind waits for
        // it and takes it from the cache rather than loading it twice.
        let gate = Arc::clone(
            self.loading
                .lock()
                .expect("loading lock poisoned")
                .entry(*hash)
                .or_default(),
        );
        let _one = gate.lock().expect("a program's load gate poisoned");
        if let Some(hit) = self.cache.lock().expect("cache lock poisoned").get(hash) {
            return Ok(Some(Arc::clone(hit)));
        }
        let Some(dir) = self.disk.as_ref() else {
            return Ok(None);
        };
        let client = self.client()?;
        let started = std::time::Instant::now();
        let file = dir.join(name);
        let Some(exe) = std::fs::read(&file)
            .ok()
            .and_then(unpacked)
            .and_then(|bytes| client.deserialize(&bytes).ok())
        else {
            return Ok(None);
        };
        touch(&file);
        self.keep(&file);
        if parked && let Some(live) = self.live.get() {
            live.tally(engine::live::Tally::LoadedInFire, 1);
        }
        tracing::debug!(
            bytes = len,
            ms = started.elapsed().as_millis() as u64,
            "xla loaded a cached program by its traced key"
        );
        let memory = exe.compiled_memory().unwrap_or_default();
        self.temp_max
            .fetch_max(memory.temp, std::sync::atomic::Ordering::Relaxed);
        let program = Arc::new(Program {
            exe: Some(exe),
            sig,
            text_bytes: len,
            memory,
        });
        self.cache
            .lock()
            .expect("cache lock poisoned")
            .insert(*hash, Arc::clone(&program));
        Ok(Some(program))
    }

    /// Where the traced fires' records of a load whose tracing inputs hash
    /// to `scope` lie: under the disk cache, a folder a scope.
    fn traces_dir(&self, scope: &str) -> Option<PathBuf> {
        Some(self.disk.as_ref()?.join("traces").join(scope))
    }

    /// The folder `scope`'s records lie in, if the device keeps a disk cache.
    #[must_use]
    pub fn trace_records_dir(&self, scope: &str) -> Option<PathBuf> {
        self.traces_dir(scope)
    }

    /// The record kept under fire `key` in `scope`, if one was kept.
    pub fn trace_record(&self, scope: &str, key: &[u8; 32]) -> Option<Vec<u8>> {
        let dir = self.traces_dir(scope)?;
        std::fs::read(dir.join(hex(key))).ok()
    }

    /// [`Device::drop_trace_record`] of a record whose program would not
    /// load, counted where the record was still kept, with a WARN naming it.
    pub fn mend_record(&self, scope: &str, key: &[u8; 32], name: &str) {
        let kept = self
            .traces_dir(scope)
            .is_some_and(|dir| dir.join(hex(key)).exists());
        if !kept {
            return;
        }
        tracing::warn!(
            record = %hex(key),
            scope,
            file = %name,
            "xla traced-fire record names a program that does not load from the disk cache: \
             dropped with the file; its fire builds the program and the next load records it"
        );
        self.drop_trace_record(scope, key, name);
        self.mended
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Waits while a fire waits on a load ([`Device::load_named`]): what
    /// loads behind serving starts a load only when no fire waits on one.
    pub fn wait_demand(&self) {
        let mut demand = self.demand.lock().expect("demand lock poisoned");
        while *demand > 0 {
            demand = self.quiet.wait(demand).expect("demand lock poisoned");
        }
    }

    /// Counts a record the ladder found its file gone or changed for.
    pub fn note_mended(&self) {
        self.mended
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The length of the disk cache's program file `name`, if it is there.
    #[must_use]
    pub fn cached_len(&self, name: &str) -> Option<u64> {
        std::fs::metadata(self.disk.as_ref()?.join(name))
            .ok()
            .map(|meta| meta.len())
    }

    /// The records [`Device::mend_record`] dropped.
    #[must_use]
    pub fn records_mended(&self) -> u64 {
        self.mended.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drops the record under fire `key` in `scope`, and the disk cache's
    /// program file `name` it named: a program that would not load, which
    /// the fire's trace builds and keeps again.
    pub fn drop_trace_record(&self, scope: &str, key: &[u8; 32], name: &str) {
        if let Some(dir) = self.traces_dir(scope) {
            let _ = std::fs::remove_file(dir.join(hex(key)));
        }
        if let Some(dir) = self.disk.as_ref() {
            let _ = std::fs::remove_file(dir.join(name));
        }
    }

    /// Keeps `record` under fire `key` in `scope` for later loads.
    pub fn keep_trace_record(&self, scope: &str, key: &[u8; 32], record: &[u8]) {
        let Some(dir) = self.traces_dir(scope) else {
            return;
        };
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let file = dir.join(hex(key));
        let tmp = file.with_extension(format!("tmp.{}", std::process::id()));
        if std::fs::write(&tmp, record).is_ok() {
            let _ = std::fs::rename(&tmp, &file);
        }
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

    /// The host memory compiling `text` took, measured alone
    /// ([`Self::keep_compile_peak`]): this process's, else the disk cache's
    /// beside its compiled form. `None`: never measured.
    #[must_use]
    pub fn compile_peak(&self, text: &str) -> Option<u64> {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(&peak) = self.peaks.lock().expect("peaks lock poisoned").get(&key) {
            return Some(peak);
        }
        let file = self
            .cache_file(self.client().ok()?, text)?
            .with_extension("peak");
        let peak = std::fs::read_to_string(file).ok()?.trim().parse().ok()?;
        self.peaks
            .lock()
            .expect("peaks lock poisoned")
            .insert(key, peak);
        crate::guest::run::note_peak(peak);
        Some(peak)
    }

    /// Keeps the host memory compiling `text` took, measured alone: for this
    /// process and beside its compiled form in the disk cache.
    pub fn keep_compile_peak(&self, text: &str, peak: u64) {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        self.peaks
            .lock()
            .expect("peaks lock poisoned")
            .insert(key, peak);
        crate::guest::run::note_peak(peak);
        if let Ok(client) = self.client()
            && let Some(file) = self.cache_file(client, text)
        {
            let _ = std::fs::write(file.with_extension("peak"), peak.to_string());
        }
    }

    /// Whether the disk cache holds the program compiled from `text`.
    #[must_use]
    /// The program the load compiled from `text` and parked in the disk
    /// cache (`serve::warm`), loaded for the first fire of its shape: from
    /// memory where it is held, else from the disk, never built. `None`:
    /// the disk holds it no more (the cache was cleared under the load).
    pub fn load_parked(&self, text: &str, sig: Signature) -> Result<Option<Arc<Program>>> {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(hit) = self.cache.lock().expect("cache lock poisoned").get(&key) {
            return Ok(Some(Arc::clone(hit)));
        }
        let client = self.client()?;
        let started = std::time::Instant::now();
        let Some(file) = self.cache_file(client, text) else {
            return Ok(None);
        };
        let Some(exe) = std::fs::read(&file)
            .ok()
            .and_then(unpacked)
            .and_then(|bytes| client.deserialize(&bytes).ok())
        else {
            return Ok(None);
        };
        touch(&file);
        self.keep(&file);
        if let Some(live) = self.live.get() {
            live.tally(engine::live::Tally::LoadedInFire, 1);
        }
        tracing::info!(
            bytes = text.len(),
            ms = started.elapsed().as_millis() as u64,
            "xla loaded a parked program for the first fire of its shape"
        );
        let memory = exe.compiled_memory().unwrap_or_default();
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
        Ok(Some(program))
    }

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
    /// The text bytes of the programs held in memory, and whether `key`'s
    /// is one of them.
    pub fn held_text_bytes(&self) -> u64 {
        let cache = self.cache.lock().expect("cache lock poisoned");
        cache.values().map(|p| p.text_bytes as u64).sum()
    }

    #[must_use]
    pub fn holds_key(&self, key: &[u8; 32]) -> bool {
        self.cache.lock().expect("cache lock poisoned").contains_key(key)
    }

    /// Takes `key`'s program out of the cache, the caller its last holder
    /// where nothing else holds it.
    pub fn take_key(&self, key: &[u8; 32]) -> Option<Arc<Program>> {
        self.cache.lock().expect("cache lock poisoned").remove(key)
    }

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
        self.run_fenced(program, args, wait).map(|(outs, _)| outs)
    }

    /// [`Device::run`], and the event that fires once the program has run on
    /// the device (whatever it outputs, none included): what a buffer it
    /// reads, a staging slot, is free again by. `None` where the run was
    /// already waited for, or ran nothing (a dry device).
    pub fn run_fenced(
        &self,
        program: &Program,
        args: Vec<Arg<'_>>,
        wait: bool,
    ) -> Result<(Vec<Buffer>, Option<Event>)> {
        if self.is_dry() {
            return Ok((Vec::new(), None));
        }
        let exe = program.exe.as_ref().ok_or(Fault::Deviceless)?;
        let at = profile::before(self.api());
        let (outs, done) = exe.execute(self.device()?, args)?;
        let done = if wait || at.is_some() {
            done.wait()?;
            None
        } else {
            Some(done)
        };
        if let Some(at) = at {
            profile::after(at, program.text_bytes);
        }
        Ok((outs, done))
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
pub(crate) const COMPILER_ENV: [&str; 2] = ["LIBTPU_INIT_ARGS", "XLA_FLAGS"];

fn disk_cache() -> Option<PathBuf> {
    let dir = disk_dir();
    // the stage compiles measured in runs before: the largest is the room
    // the program loaders leave free from the start, before a registration
    // states its own
    if let Some(dir) = &dir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "peak")
                && let Some(peak) = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| t.trim().parse::<u64>().ok())
            {
                crate::guest::run::note_peak(peak);
            }
        }
    }
    dir
}

fn disk_dir() -> Option<PathBuf> {
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
/// The process's resident bytes (`/proc/self/statm`), where the host
/// states them.
pub(crate) fn resident_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: `sysconf` reads a constant of the running system.
    #[allow(unsafe_code)]
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).ok().map(|page| pages * page)
}

/// The host memory programs loaded and held take, measured, and the room
/// they are loaded into. A program's hold is the resident growth of a load
/// that ran alone, per byte of its text, the most any such load showed
/// (a per-text-byte guess undercounted a v6e Qwen3.5-0.8B fire of 64
/// lanes and 1024 rows sixfold: 0.5 GiB guessed, 3.0 GiB held). Until a
/// load has been measured, loads run one at a time. A load goes ahead
/// while the host's free memory now, less what every load still under way
/// was reserved, holds its own hold and `floor`; else it waits for one
/// under way to end, and is turned away only when none is and it still
/// does not fit. With a `stated` room, what the measured holds and those
/// under way take stays within it.
pub(crate) struct HostRoom {
    stated: Option<u64>,
    floor: u64,
    base: u64,
    state: std::sync::Mutex<RoomState>,
    ended: std::sync::Condvar,
}

#[derive(Default)]
struct RoomState {
    /// The bytes reserved for the loads under way, and how many there are.
    reserved: u64,
    loads: usize,
    /// The most resident bytes a load alone grew by, per KiB of its text.
    per_kib: u64,
    /// Whether the load under way, if one alone, has stayed alone.
    alone: bool,
}

/// A load admitted by [`HostRoom::take`]: what it reserved, where the
/// process's resident bytes stood, and its text's length.
pub(crate) struct Reserved {
    bytes: u64,
    resident: Option<u64>,
    text: u64,
}

impl HostRoom {
    pub(crate) fn new(stated: Option<u64>, floor: u64) -> Self {
        Self {
            stated,
            floor,
            base: resident_bytes().unwrap_or(0),
            state: std::sync::Mutex::new(RoomState::default()),
            ended: std::sync::Condvar::new(),
        }
    }

    /// The most resident bytes a program held a KiB of its text, as a
    /// load alone measured it (0: none measured).
    pub(crate) fn per_kib(&self) -> u64 {
        self.state.lock().map_or(0, |st| st.per_kib)
    }

    /// What the programs loaded since the room was made hold: the
    /// process's resident growth.
    pub(crate) fn held(&self) -> u64 {
        resident_bytes().map_or(0, |r| r.saturating_sub(self.base))
    }

    /// Admits a load of a program of `text_len` bytes of text, waiting
    /// while those under way leave it no room; `None` when it does not fit
    /// with none under way.
    pub(crate) fn take(&self, text_len: usize) -> Option<Reserved> {
        let len = text_len as u64;
        let mut st = self.state.lock().expect("the host room's lock poisoned");
        loop {
            let hold = (st.per_kib > 0).then(|| st.per_kib.saturating_mul(len) / 1024);
            let fits = match hold {
                // unmeasured: alone, where there is room at all
                None => st.loads == 0 && self.room(0, 0),
                Some(hold) => self.room(st.reserved, hold),
            };
            if fits {
                st.alone = st.loads == 0;
                st.loads += 1;
                let bytes = hold.unwrap_or(0);
                st.reserved += bytes;
                return Some(Reserved {
                    bytes,
                    resident: resident_bytes(),
                    text: len,
                });
            }
            if st.loads == 0 {
                return None;
            }
            st = self.ended.wait(st).expect("the host room's lock poisoned");
        }
    }

    /// Whether a hold of `hold` fits beside `reserved` under way.
    fn room(&self, reserved: u64, hold: u64) -> bool {
        // the stated room, where stated, and always the host's: the floor,
        // and the room the guest stage compiles running and waiting take (a
        // loader beside serving held the host to 14.5 GB free and a
        // registration's sampler stages were refused)
        self.stated.is_none_or(|stated| {
            self.held().saturating_add(reserved).saturating_add(hold) <= stated
        }) && available_memory().is_none_or(|free| {
            free.saturating_sub(reserved).saturating_sub(hold)
                >= self
                    .floor
                    .saturating_add(crate::guest::run::guest_reserve())
        })
    }

    /// A load admitted ends; `held`: its program stays in memory. A load
    /// that ran alone and stays is measured.
    pub(crate) fn done(&self, r: Reserved, held: bool) {
        let mut st = self.state.lock().expect("the host room's lock poisoned");
        let alone = st.loads == 1 && st.alone;
        if held
            && alone
            && let (Some(before), Some(after)) = (r.resident, resident_bytes())
            && r.text > 0
        {
            let per_kib = after.saturating_sub(before).saturating_mul(1024) / r.text;
            st.per_kib = st.per_kib.max(per_kib);
        }
        st.loads -= 1;
        st.reserved -= r.bytes;
        drop(st);
        self.ended.notify_all();
    }
}

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

/// Admits compiles by what those before them grew the process by: one starts
/// once none runs, or while the host's free memory, read as it starts, holds
/// the most a compile was measured to grow the process by (its resident
/// growth over its run, over the most that ran beside it) for every compile
/// running and itself, and no more than twice as many as have ended run (the
/// first ones measure what the next admit on). A compile that does not fit
/// waits for one to end. The growth read is what a compile still holds as
/// it returns, below its peak (v6e Qwen3.5, programs compiled alone: the
/// peak 1.0 to 2.5 times the growth held at the end); the margin is that the
/// free memory read at a start counts again what the running ones already
/// took. Qwen3.5's fresh boot, up to 40 at once: 132 GiB anon of 150.
#[derive(Default)]
pub(crate) struct CompileGate {
    state: Mutex<GateState>,
    ended: std::sync::Condvar,
}

#[derive(Default)]
struct GateState {
    running: u64,
    ended: u64,
    largest: u64,
}

/// A compile the gate let in: its resident start and the most that ran
/// beside it; its growth is measured as it ends.
pub(crate) struct Gated<'a> {
    by: &'a CompileGate,
    resident: u64,
}

impl CompileGate {
    /// Waits until a compile fits.
    pub(crate) fn admit(&self) -> Gated<'_> {
        let mut st = self.state.lock().expect("the compile gate's lock poisoned");
        loop {
            let fits = st.running == 0
                || (st.running < 2 * st.ended
                    && available_memory().is_none_or(|free| {
                        free >= st.largest.saturating_mul(st.running + 1)
                    }));
            if fits {
                break;
            }
            st = self.ended.wait(st).expect("the compile gate's lock poisoned");
        }
        st.running += 1;
        Gated {
            by: self,
            resident: resident_bytes().unwrap_or(0),
        }
    }
}

impl Drop for Gated<'_> {
    fn drop(&mut self) {
        let mut st = self.by.state.lock().expect("the compile gate's lock poisoned");
        let grew = resident_bytes().unwrap_or(0).saturating_sub(self.resident);
        st.largest = st.largest.max(grew / st.running.max(1));
        st.running -= 1;
        st.ended += 1;
        self.by.ended.notify_all();
    }
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

/// A program's source: its text, or the disk cache's copy of it named by
/// a traced fire's record (the text never built).
#[derive(Clone, Debug)]
pub enum Code {
    Text(String),
    Cached {
        /// blake3 of the text (the in-memory cache's key).
        hash: [u8; 32],
        /// The disk cache's file name.
        name: String,
        /// The text's length.
        len: usize,
        /// What compiling the text takes of the host (`compile_need`).
        need: u64,
        /// The program file's length as it was kept.
        file_len: u64,
    },
}

impl Code {
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        match self {
            Code::Text(text) => *blake3::hash(text.as_bytes()).as_bytes(),
            Code::Cached { hash, .. } => *hash,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Code::Text(text) => text.len(),
            Code::Cached { len, .. } => *len,
        }
    }

    /// What compiling the program's text takes of the host.
    #[must_use]
    pub fn need(&self) -> u64 {
        match self {
            Code::Text(text) => compile_need(text),
            Code::Cached { need, .. } => *need,
        }
    }

    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Code::Text(text) => Some(text),
            Code::Cached { .. } => None,
        }
    }
}

pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A load a fire waits on, counted while it lasts ([`Device::wait_demand`]).
struct Demand<'d> {
    device: &'d Device,
}

impl<'d> Demand<'d> {
    fn on(device: &'d Device) -> Demand<'d> {
        *device.demand.lock().expect("demand lock poisoned") += 1;
        Demand { device }
    }
}

impl Drop for Demand<'_> {
    fn drop(&mut self) {
        let mut demand = self.device.demand.lock().expect("demand lock poisoned");
        *demand -= 1;
        if *demand == 0 {
            self.device.quiet.notify_all();
        }
    }
}
