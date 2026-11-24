use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    pub golden_probe: bool,
    pub arm_trace: bool,
    pub ptr_trace: Option<String>,
    pub arm_bench: Option<String>,
    pub graph_dot: Option<PathBuf>,
    pub grid_trace: Option<String>,
    pub plan_trace: Option<String>,
    pub capture_serial: bool,
    pub boundary_trace: bool,
    pub reap_trace: bool,
    pub trace_census: bool,
    pub nan_check: bool,
    pub fuse_chains: bool,
    /// What a fire that waits for the device does (`crate::device::guard`).
    pub sync_guard: Option<SyncGuard>,
    /// The chunk size the state pool's recipe aims near, in place of the
    /// device profile's: a small one cuts a miniature's records into the
    /// pieces only a full model's take.
    pub chunk_bytes: Option<u64>,
    /// Copies between the device and the host tier ride the compute stream,
    /// ahead of every fire after them, not the swap stream beside those
    /// fires: to weigh the overlap.
    pub tier_inline: bool,
    /// Fills device memory no kernel has written yet with 0xFF bytes (a NaN
    /// in f32, bf16 and f16): every pool segment as it is backed (kv pages,
    /// state records; the null chunk and a fresh state's records are still
    /// zeroed after, as their readers are owed) and the activation span. A
    /// kernel that reads a cell nothing wrote, under a zero weight or a
    /// mask, then answers NaN where it read.
    pub unwritten_nan: bool,
}

/// The sync guard's answer to a fire that waited for the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncGuard {
    /// Print each call the first time a thread's fire makes it, as it is
    /// made: a fire that deadlocks on it has still named it.
    Report,
    /// Refuse the fire: a panic naming every call it made.
    Panic,
}

impl Default for Diagnostics {
    fn default() -> Diagnostics {
        Diagnostics {
            golden_probe: false,
            arm_trace: false,
            ptr_trace: None,
            arm_bench: None,
            graph_dot: None,
            grid_trace: None,
            plan_trace: None,
            capture_serial: false,
            boundary_trace: false,
            reap_trace: false,
            trace_census: false,
            nan_check: false,
            fuse_chains: true,
            sync_guard: None,
            chunk_bytes: None,
            tier_inline: false,
            unwritten_nan: false,
        }
    }
}

impl Diagnostics {
    #[must_use]
    pub fn any(&self) -> bool {
        *self != Diagnostics::default()
    }

    #[must_use]
    pub fn words(&self) -> Option<String> {
        let mut words: Vec<String> = Vec::new();
        for (word, on) in [
            ("golden-probe", self.golden_probe),
            ("arm-trace", self.arm_trace),
            ("capture-serial", self.capture_serial),
            ("boundary-trace", self.boundary_trace),
            ("reap-trace", self.reap_trace),
            ("trace-census", self.trace_census),
            ("nan-check", self.nan_check),
        ] {
            if on {
                words.push(word.to_string());
            }
        }
        for (word, value) in [
            ("ptr-trace", self.ptr_trace.clone()),
            ("arm-bench", self.arm_bench.clone()),
            ("grid-trace", self.grid_trace.clone()),
            ("plan-trace", self.plan_trace.clone()),
            (
                "graph-dot",
                self.graph_dot.as_ref().map(|dir| dir.display().to_string()),
            ),
        ] {
            if let Some(value) = value {
                words.push(format!("{word}={value}"));
            }
        }
        if !self.fuse_chains {
            words.push("fuse-chains=off".to_string());
        }
        match self.sync_guard {
            Some(SyncGuard::Report) => words.push("sync-guard".to_string()),
            Some(SyncGuard::Panic) => words.push("sync-guard=panic".to_string()),
            None => {}
        }
        if let Some(bytes) = self.chunk_bytes {
            words.push(format!("chunk-bytes={bytes}"));
        }
        if self.tier_inline {
            words.push("tier-inline".to_string());
        }
        if self.unwritten_nan {
            words.push("unwritten-nan".to_string());
        }
        (!words.is_empty()).then(|| words.join(","))
    }
}

fn switch(word: &str, value: &str) -> std::result::Result<bool, String> {
    match value {
        "" | "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "`{word}={other}` is not a switch; write `{word}` or `{word}=off`"
        )),
    }
}

const WORDS: &str = "`golden-probe`, `arm-trace`, \
     `ptr-trace=<key substring>`, `graph-dot=<dir>`, `grid-trace=<key substring>`, \
     `plan-trace=<rows|all>`, `arm-bench=<key>`, `capture-serial`, `boundary-trace`, `reap-trace`, \
     `trace-census`, `nan-check`, `fuse-chains=off`, \
     `sync-guard[=panic]`, `chunk-bytes=<bytes>`, \
     `tier-inline`, `unwritten-nan`";

impl std::str::FromStr for Diagnostics {
    type Err = String;

    fn from_str(list: &str) -> std::result::Result<Diagnostics, String> {
        let mut diag = Diagnostics::default();
        for term in list.split(',') {
            let term = term.trim();
            if term.is_empty() {
                continue;
            }
            let (word, value) = match term.split_once('=') {
                Some((word, value)) => (word.trim(), value.trim()),
                None => (term, ""),
            };
            match word {
                "golden-probe" => diag.golden_probe = switch(word, value)?,
                "arm-trace" => diag.arm_trace = switch(word, value)?,
                "capture-serial" => diag.capture_serial = switch(word, value)?,
                "boundary-trace" => diag.boundary_trace = switch(word, value)?,
                "reap-trace" => diag.reap_trace = switch(word, value)?,
                "trace-census" => diag.trace_census = switch(word, value)?,
                "nan-check" => diag.nan_check = switch(word, value)?,
                "tier-inline" => diag.tier_inline = switch(word, value)?,
                "unwritten-nan" => diag.unwritten_nan = switch(word, value)?,
                "fuse-chains" => diag.fuse_chains = switch(word, value)?,
                "sync-guard" => {
                    diag.sync_guard = match value {
                        "panic" => Some(SyncGuard::Panic),
                        _ => switch(word, value)?.then_some(SyncGuard::Report),
                    }
                }
                "chunk-bytes" => {
                    diag.chunk_bytes = Some(value.parse().map_err(|_| {
                        format!(
                            "`chunk-bytes` takes a byte count: `chunk-bytes=<bytes>`, not `{value}`"
                        )
                    })?);
                }
                "ptr-trace" | "grid-trace" | "plan-trace" | "graph-dot" | "arm-bench" => {
                    if value.is_empty() {
                        return Err(format!(
                            "`{word}` takes a value: `{word}=<{}>`",
                            match word {
                                "graph-dot" => "directory",
                                "plan-trace" => "rows|all",
                                _ => "substring of a body key",
                            }
                        ));
                    }
                    match word {
                        "ptr-trace" => diag.ptr_trace = Some(value.to_string()),
                        "arm-bench" => diag.arm_bench = Some(value.to_string()),
                        "grid-trace" => diag.grid_trace = Some(value.to_string()),
                        "plan-trace" => diag.plan_trace = Some(value.to_string()),
                        _ => diag.graph_dot = Some(PathBuf::from(value)),
                    }
                }
                other => {
                    return Err(format!(
                        "`{other}` does not name a diagnostic; the words are {WORDS}"
                    ));
                }
            }
        }
        Ok(diag)
    }
}

static PUBLISHED: OnceLock<Diagnostics> = OnceLock::new();

pub(crate) fn publish(stated: &Diagnostics) {
    let published = PUBLISHED.get_or_init(|| stated.clone());
    if published != stated {
        tracing::warn!(
            "this process already published a diagnostics record \
             ({}), so the one this boot states ({}) is ignored; every rank of a \
             group traces the same thing",
            published.words().unwrap_or_else(|| "none".to_string()),
            stated.words().unwrap_or_else(|| "none".to_string()),
        );
    }
}

#[must_use]
pub fn on() -> &'static Diagnostics {
    static OFF: OnceLock<Diagnostics> = OnceLock::new();
    PUBLISHED
        .get()
        .unwrap_or_else(|| OFF.get_or_init(Diagnostics::default))
}

/// Runs `fire` under the sync guard when it is on: each call it made that
/// waits for the device printed on its first sighting, or the fire refused.
pub(crate) fn guarded<T>(fire: impl FnOnce() -> T) -> T {
    let Some(guard) = on().sync_guard else {
        return fire();
    };
    let (fired, waits) = crate::device::guard::watched(fire);
    if waits.is_empty() {
        return fired;
    }
    match guard {
        // Printed inside each call as it was made (`guard::speak`).
        SyncGuard::Report => {}
        SyncGuard::Panic => {
            let waits: Vec<String> = waits.iter().map(ToString::to_string).collect();
            panic!(
                "[sync-guard] a fire waited for the device: {}",
                waits.join("; ")
            );
        }
    }
    fired
}
