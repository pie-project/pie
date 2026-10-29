//! The engine's switches: `[cerebras.switches]` in the boot config, by
//! name. A switch the config does not set falls back to its old environment
//! variable (`PIE_CEREBRAS_<NAME>`), with a warning the first time, so a
//! config always wins.
//!
//! ```toml
//! [cerebras.switches]
//! pe_words = 9000        # the fabric planner's words a PE
//! whole_fire = true
//! phase_timeout = 600
//! ```
//!
//! Debug dumps and traces are not switches: they stay environment-only and
//! change no result ([`DEBUG_ENV`]).

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// A switch: its config name, its environment fallback, what it does.
pub struct Switch {
    pub name: &'static str,
    pub env: &'static str,
    pub doc: &'static str,
}

const fn switch(name: &'static str, env: &'static str, doc: &'static str) -> Switch {
    Switch { name, env, doc }
}

/// Every switch the engine reads.
pub const SWITCHES: &[Switch] = &[
    // Placement (the fabric planner, fabric.rs).
    switch("pe_words", "PIE_CEREBRAS_PE_WORDS", "words a PE holds for the planner"),
    switch("shard_words", "PIE_CEREBRAS_SHARD_WORDS", "words of a weight shard a PE"),
    switch("max_pes", "PIE_CEREBRAS_MAX_PES", "the most PEs a plan may use"),
    switch("resident_pes", "PIE_CEREBRAS_RESIDENT_PES", "PEs kept for resident weights"),
    switch("fabric_sum", "PIE_CEREBRAS_FABRIC_SUM", "sum partial products on the fabric"),
    switch("half_weights", "PIE_CEREBRAS_HALF_WEIGHTS", "hold weights in 16 bits"),
    switch("half_pool", "PIE_CEREBRAS_HALF_POOL", "hold the KV pool in 16 bits"),
    switch("attn_rows", "PIE_CEREBRAS_ATTN_ROWS", "rows of the attention band"),
    switch("table", "PIE_CEREBRAS_TABLE", "run phases from the fused table"),
    switch("whole_fire", "PIE_CEREBRAS_WHOLE_FIRE", "one program a fire (`force`: even past the checks)"),
    switch("keep_pools", "PIE_CEREBRAS_KEEP_POOLS", "keep the pools resident across whole fires"),
    switch("fuse", "PIE_CEREBRAS_FUSE", "the phase fuser's mode"),
    // Serving (device.rs, api.rs, serve.rs).
    switch("servers", "PIE_CEREBRAS_SERVERS", "phase servers kept running"),
    switch("serve", "PIE_CEREBRAS_SERVE", "which phases a server takes"),
    switch("serve_all", "PIE_CEREBRAS_SERVE_ALL", "serve every phase"),
    switch("keep", "PIE_CEREBRAS_KEEP", "keep compiled phase directories"),
    switch("phase_timeout", "PIE_CEREBRAS_PHASE_TIMEOUT", "seconds a phase may run"),
    switch("sync", "PIE_CEREBRAS_SYNC", "wait for each fire before the next"),
    switch("shape_check", "PIE_CEREBRAS_SHAPE_CHECK", "check every landed shape (1)"),
    switch("timing", "PIE_CEREBRAS_TIMING", "log each fire's timing"),
    // Guests (guest/).
    switch("guests_on_host", "PIE_CEREBRAS_GUESTS_ON_HOST", "run every guest on the host"),
    switch("guest_lanes", "PIE_CEREBRAS_GUEST_LANES", "lanes a guest program runs"),
    switch("guest_words", "PIE_CEREBRAS_GUEST_WORDS", "words a guest PE holds"),
    // The toolchain (sdk/, device.rs).
    switch("sdk_lib", "PIE_CEREBRAS_SDK_LIB", "the SDK runtime's library directory (default /cbcore/lib)"),
    switch("cslc", "PIE_CEREBRAS_CSLC", "the cslc front-end script"),
    switch("python", "PIE_CEREBRAS_PYTHON", "the python cslc runs under"),
    switch("compile_timeout", "PIE_CEREBRAS_COMPILE_TIMEOUT", "seconds a cslc compile may take"),
    switch("fabric_run", "PIE_CEREBRAS_FABRIC_RUN", "a wrapper the fabric runner runs through (the SDK loader)"),
];

/// Environment-only debug output: dumps and traces, none of which changes a
/// result.
pub const DEBUG_ENV: &[(&str, &str)] = &[
    ("PIE_CEREBRAS_DUMP", "a directory each phase's program and buffers are dumped to"),
    ("PIE_CEREBRAS_DUMP_PACK", "a prefix the packed inputs are dumped to"),
    ("PIE_CEREBRAS_CAPTURE", "a directory each fire's inputs and outputs are captured to"),
    ("PIE_CEREBRAS_TRACE_PHASES", "print each phase as it runs"),
    ("PIE_CEREBRAS_TRACE_FUSION", "print the fuser's decisions"),
    ("PIE_CEREBRAS_TRACE_ARENA", "print the arena's layout"),
    ("PIE_CEREBRAS_GUEST_DEBUG", "a guest lowering's debug spec"),
    ("PIE_CEREBRAS_GUEST_BISECT", "lower guests op by op (no shortcuts)"),
    ("PIE_SPATIAL_DUMP_PTIR", "a directory the spatial engine dumps registered programs to"),
];

static CONFIG: OnceLock<BTreeMap<String, String>> = OnceLock::new();

/// `[cerebras.switches]`: unknown names refused. The first engine to open
/// sets them (as `cache_dir`).
pub fn init(table: Option<&toml::Table>) -> Result<(), String> {
    let mut set = BTreeMap::new();
    for (k, v) in table.into_iter().flatten() {
        if !SWITCHES.iter().any(|s| s.name == k) {
            return Err(format!("[cerebras.switches]: unknown switch `{k}`"));
        }
        let v = match v {
            toml::Value::String(s) => s.clone(),
            toml::Value::Integer(i) => i.to_string(),
            toml::Value::Float(f) => f.to_string(),
            toml::Value::Boolean(b) => if *b { "1" } else { "0" }.to_string(),
            other => return Err(format!("[cerebras.switches] `{k}` is {other}; a string, number or boolean")),
        };
        set.insert(k.clone(), v);
    }
    let _ = CONFIG.set(set);
    Ok(())
}

/// Switch `name`'s value: the config's, else its environment variable's.
///
/// # Panics
/// On a name that is not in [`SWITCHES`] (a programming error).
pub fn get(name: &str) -> Option<String> {
    let s = SWITCHES.iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no switch {name}"));
    if let Some(v) = CONFIG.get().and_then(|c| c.get(name)) {
        return Some(v.clone());
    }
    let v = std::env::var(s.env).ok()?;
    static WARNED: OnceLock<std::sync::Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut warned = WARNED.get_or_init(Default::default).lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if !warned.contains(&s.name) {
        warned.push(s.name);
        tracing::warn!(switch = s.name, env = s.env, "a switch from the environment; set it in [cerebras.switches]");
    }
    Some(v)
}

/// A switch that is set and not `0` / `false`.
pub fn on(name: &str) -> bool {
    get(name).is_some_and(|v| v != "0" && v != "false")
}

/// A switch parsed as a number (or other `FromStr`).
pub fn parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    get(name).and_then(|v| v.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_are_named_once_and_known() {
        let mut names: Vec<&str> = SWITCHES.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SWITCHES.len());
        let t: toml::Table = "bogus = 1".parse().unwrap();
        assert!(init(Some(&t)).unwrap_err().contains("bogus"));
    }
}
