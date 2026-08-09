//! Differential proof that [`driver_cuda::layout::profile_cache`] behaves
//! like `store/planner_profile_cache.cpp`.
//!
//! `tests/oracle/profile_cache/run.sh` compiles the **real** C++ source --
//! only its two external inputs are stubbed -- drives it over the same grid
//! this file drives the Rust over, and hashes the transcript. The golden below
//! is that hash.
//!
//! It is the **C++'s** hash, never the Rust's. A golden captured from the
//! implementation under test can be re-blessed by the next change to that
//! implementation.
//!
//! # The one place the transcripts deliberately differ
//!
//! `planner_profile_cache_lookup` documents itself as never throwing. It does:
//! `nlohmann::json::value()` raises `type_error.302` on eight of the inputs
//! swept here, and the call sites are outside any `try`. Those rows are
//! rendered as `THROWS|...` by the oracle and as `unusable|...` by this file,
//! and [`normalise`] maps both onto a single token so the rest of the
//! transcript can be compared byte for byte. Every other row must match
//! exactly.

use driver_cuda::layout::profile_cache::{
    Lookup, ProfileCache, ShapeSample, cache_path, planner_budget_bytes,
    set_planner_budget_bytes,
};
use driver_cuda::layout::profile_key::{ProfileKey, ProfileShape};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Mutex;

/// FNV-1a 64 of `tests/oracle/profile_cache/run.sh`'s output, after
/// [`normalise`].
const GOLDEN_FNV1A64: u64 = 0xda3b_c358_199c_7f43;
/// Row count of the same.
const GOLDEN_ROWS: usize = 408;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The line-for-line counterpart of the `normalise` in
/// `tests/oracle/profile_cache/run.sh`. Both erase differences that are *not*
/// properties of the logic under test; everything else is compared byte for
/// byte.
fn normalise(line: &str) -> String {
    let line = line.to_owned();
    for marker in ["|THROWS|", "|unusable|"] {
        if let Some((head, _)) = line.split_once(marker) {
            return format!("{head}|REFUSED");
        }
    }
    if let Some((head, rest)) = line.split_once("|miss|err=")
        && !rest.is_empty()
        && !rest.starts_with("schema version")
    {
        // A parse failure is reported through the same channel in both, but
        // the wording is nlohmann's versus serde_json's.
        return format!("{head}|miss|err=PARSE");
    }
    line
}

/// `measured_at` is a wall-clock stamp read inside the C++ function under test
/// and an injected parameter here, so the two can never agree. Zeroed on both
/// sides *before* the byte count is taken, so that `bytes=` stays a real check
/// on whitespace rather than a reflection of how many digits the clock had.
fn zero_measured_at(text: &str) -> String {
    const KEY: &str = "\"measured_at\":";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(KEY) {
        out.push_str(&rest[..at]);
        out.push_str(KEY);
        out.push('0');
        let after = &rest[at + KEY.len()..];
        rest = after
            .trim_start_matches([' ', '-'])
            .trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

fn key_a() -> ProfileKey {
    ProfileKey {
        gpu_name: "NVIDIA L40S".into(),
        compute_major: 8,
        compute_minor: 9,
        sm_count: 142,
        kv_cache_dtype: "bf16".into(),
        tp_size: 1,
        model_type: "llama".into(),
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
    }
}

/// The document `oracle.cpp`'s `doc_for` builds, with the key serialised the
/// way the C++ serialises it (alphabetical, because nlohmann sorts).
fn doc_for(k: &ProfileKey, plan: &str, version: &str) -> String {
    format!(
        "{{\"version\":{version},\"entries\":[{{\"key\":{},\"plan\":{plan}}}]}}",
        k.to_json()
    )
}

struct Scribe {
    out: String,
    cache: ProfileCache,
    dir: std::path::PathBuf,
}

impl Scribe {
    fn put(&self, text: &str) {
        std::fs::create_dir_all(&self.dir).expect("create cache dir");
        std::fs::write(self.cache.path(), text).expect("write cache");
    }

    fn clear(&self) {
        let _ = std::fs::remove_file(self.cache.path());
    }

    fn show_lookup(&mut self, label: &str, key: &ProfileKey) {
        let line = match self.cache.lookup(key) {
            Lookup::Hit(s) => format!(
                "{label}|hit|{}|{}|{}|{}|{}|err=",
                s.policy_profile,
                s.kv_page_size,
                s.max_forward_tokens,
                s.max_forward_requests,
                s.budget_bytes
            ),
            Lookup::Miss => format!("{label}|miss|err="),
            Lookup::Unusable(m) if m.starts_with("schema version") => {
                format!("{label}|miss|err={m}")
            }
            // Everything else the C++ either throws on or reports as a parse
            // error; `normalise` folds both spellings together.
            Lookup::Unusable(m) if m.contains("delete the file") => {
                format!("{label}|unusable|{m}")
            }
            Lookup::Unusable(m) => format!("{label}|miss|err={m}"),
        };
        let _ = writeln!(self.out, "{line}");
    }

    fn dump_file(&mut self, label: &str) {
        let all = zero_measured_at(&std::fs::read_to_string(self.cache.path()).unwrap_or_default());
        let _ = writeln!(self.out, "store|{label}|bytes={}\n{all}<<END>>", all.len());
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one statement per oracle case; splitting it would hide the correspondence"
)]
#[expect(clippy::excessive_precision, reason = "the extra digits are the subject")]
fn transcript(dir: &Path) -> String {
    let mut s = Scribe {
        out: String::with_capacity(16 << 10),
        cache: ProfileCache::at(dir.join("cuda_memory_profiles.json")),
        dir: dir.to_path_buf(),
    };

    // --- path derivation ---------------------------------------------------
    fn env(
        xdg: Option<&'static str>,
        home: Option<&'static str>,
    ) -> impl Fn(&str) -> Option<String> {
        move |k: &str| match k {
            "XDG_CACHE_HOME" => xdg.map(str::to_owned),
            "HOME" => home.map(str::to_owned),
            _ => None,
        }
    }
    let show_path = |out: &mut String, label: &str, p: Option<std::path::PathBuf>| {
        let _ = writeln!(
            out,
            "path|{label}|{}",
            p.map(|p| p.display().to_string()).unwrap_or_default()
        );
    };
    show_path(
        &mut s.out,
        "cfg",
        cache_path("/cfg", env(Some("/xdg"), Some("/home/u"))),
    );
    show_path(
        &mut s.out,
        "xdg",
        cache_path("", env(Some("/xdg"), Some("/home/u"))),
    );
    show_path(
        &mut s.out,
        "xdg_empty",
        cache_path("", env(Some(""), Some("/home/u"))),
    );
    show_path(&mut s.out, "home", cache_path("", env(None, Some("/home/u"))));
    show_path(&mut s.out, "home_empty", cache_path("", env(None, Some(""))));
    let none = cache_path("", env(None, None));
    let _ = writeln!(
        s.out,
        "path|none||empty={}",
        i32::from(none.is_none())
    );

    // --- make_planner_profile_key -----------------------------------------
    // Field-for-field copy in the C++; reproduced here so a reordering would
    // show up rather than being assumed harmless.
    for tp in [1, 2, 8] {
        let k = ProfileKey {
            gpu_name: "NVIDIA L40S".into(),
            compute_major: 8,
            compute_minor: 9,
            sm_count: 142,
            kv_cache_dtype: "fp8_e4m3".into(),
            tp_size: tp,
            model_type: "llama".into(),
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
        };
        let _ = writeln!(
            s.out,
            "key|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            k.gpu_name,
            k.compute_major,
            k.compute_minor,
            k.sm_count,
            k.kv_cache_dtype,
            k.tp_size,
            k.model_type,
            k.hidden_size,
            k.num_hidden_layers,
            k.num_attention_heads,
            k.num_key_value_heads,
            k.head_dim
        );
    }

    // --- the budget publish/read pair -------------------------------------
    let _ = writeln!(s.out, "budget|initial|{}", planner_budget_bytes());
    for b in [0u64, 1, 42 * 1024 * 1024 * 1024] {
        set_planner_budget_bytes(b);
        let _ = writeln!(
            s.out,
            "budget|set|{b}|{}",
            planner_budget_bytes()
        );
    }

    let k = key_a();

    // --- lookup over crafted documents ------------------------------------
    s.clear();
    s.show_lookup("lookup|absent", &k);

    s.put("");
    s.show_lookup("lookup|empty_file", &k);
    s.put("not json at all");
    s.show_lookup("lookup|garbage", &k);
    s.put("[1,2,3]");
    s.show_lookup("lookup|array_root", &k);
    s.put("{}");
    s.show_lookup("lookup|no_entries", &k);
    s.put("{\"entries\":{}}");
    s.show_lookup("lookup|entries_not_array", &k);
    s.put("{\"entries\":[]}");
    s.show_lookup("lookup|entries_empty_no_version", &k);
    s.put("{\"version\":1,\"entries\":[]}");
    s.show_lookup("lookup|entries_empty_wrong_version", &k);
    s.put(&doc_for(&k, "{}", "1"));
    s.show_lookup("lookup|wrong_version_with_entry", &k);
    s.put("{\"version\":\"2\",\"entries\":[]}");
    s.show_lookup("lookup|version_string", &k);
    s.put("{\"version\":null,\"entries\":[]}");
    s.show_lookup("lookup|version_null", &k);
    s.put(&doc_for(&k, "{}", "2.0"));
    s.show_lookup("lookup|version_float", &k);
    s.put(&doc_for(&k, "{}", "true"));
    s.show_lookup("lookup|version_bool", &k);
    s.put(&doc_for(&k, "{}", "2.9"));
    s.show_lookup("lookup|version_float_truncates", &k);

    let plans = [
        "{}",
        r#"{"policy_profile":"throughput","kv_page_size":16,"max_forward_tokens":8192,"max_forward_requests":256,"budget_bytes":42949672960}"#,
        r#"{"policy_profile":7}"#,
        r#"{"policy_profile":null}"#,
        r#"{"kv_page_size":16.9}"#,
        r#"{"kv_page_size":"16"}"#,
        r#"{"kv_page_size":true}"#,
        r#"{"kv_page_size":null}"#,
        r#"{"kv_page_size":-5}"#,
        r#"{"budget_bytes":-1}"#,
        r#"{"budget_bytes":1.5}"#,
        r#"{"budget_bytes":null}"#,
        r#"{"budget_bytes":true}"#,
        r#"{"policy_profile":true}"#,
        r#"{"max_forward_requests":false}"#,
        r#"{"budget_bytes":18446744073709551615}"#,
        r#"{"max_forward_tokens":2147483648}"#,
    ];
    for plan in plans {
        s.put(&doc_for(&k, plan, "2"));
        s.show_lookup(&format!("lookup|plan|{plan}"), &k);
    }

    let mutations = [
        r#""gpu_name":"NVIDIA L40""#,
        r#""compute_major":9"#,
        r#""compute_minor":0"#,
        r#""sm_count":141"#,
        r#""sm_count":142.0"#,
        r#""sm_count":"142""#,
        r#""kv_cache_dtype":"fp8""#,
        r#""tp_size":2"#,
        r#""model_type":"qwen3""#,
        r#""hidden_size":4097"#,
        r#""num_hidden_layers":33"#,
        r#""num_attention_heads":31"#,
        r#""num_key_value_heads":4"#,
        r#""head_dim":64"#,
    ];
    for m in mutations {
        let mut doc = doc_for(&k, r#"{"kv_page_size":16}"#, "2");
        let end = doc.find("},\"plan\"").expect("plan marker");
        doc.insert_str(end, &format!(",{m}"));
        s.put(&doc);
        s.show_lookup(&format!("lookup|key|{m}"), &k);
    }
    {
        let doc = doc_for(&k, r#"{"kv_page_size":16}"#, "2").replace(r#""sm_count":142,"#, "");
        s.put(&doc);
        s.show_lookup("lookup|key|sm_count_missing", &k);
    }

    s.put(r#"{"version":2,"entries":[1,2,"x",null,{},{"key":{}},{"plan":{}},{"key":[],"plan":{}}]}"#);
    s.show_lookup("lookup|entries_junk", &k);

    {
        let a = doc_for(&k, r#"{"kv_page_size":16}"#, "2");
        let b = doc_for(&k, r#"{"kv_page_size":32}"#, "2");
        let start = b.find("[{").expect("entry start") + 1;
        let entry_b = &b[start..b.rfind("}]").expect("entry end") + 1];
        let insert_at = a.rfind("}]").expect("entry end") + 1;
        let mut a = a;
        a.insert_str(insert_at, &format!(",{entry_b}"));
        s.put(&a);
        s.show_lookup("lookup|two_matches", &k);
    }

    // --- store -------------------------------------------------------------
    s.clear();
    {
        let shape = ProfileShape {
            policy_profile: "throughput".into(),
            kv_page_size: 16,
            max_forward_tokens: 8192,
            max_forward_requests: 256,
            budget_bytes: 42_949_672_960,
        };
        let samples: Vec<ShapeSample> = (0..3)
            .map(|i| ShapeSample {
                max_forward_tokens: 1024 << i,
                max_forward_requests: 32 * (i + 1),
                tokens_per_request: 7 + i,
                step_ms: if i == 0 {
                    // A value Grisu2 and Rust's formatter disagree about, so
                    // the written file proves which one produced it.
                    46_934.815_584_012_416
                } else {
                    1.0 / f64::from(i + 3)
                },
                step_ms_stddev: if i == 1 {
                    0.0
                } else {
                    1e-7 * f64::from(i + 1)
                },
                tokens_per_s: if i == 2 {
                    1e21
                } else {
                    12345.0 * f64::from(i + 1)
                },
            })
            .collect();
        let ok = s.cache.store(&k, &shape, &samples, 0);
        let _ = writeln!(
            s.out,
            "store|first|ok={}|err={}",
            i32::from(ok.is_ok()),
            ok.err().map(|e| e.to_string()).unwrap_or_default()
        );
        s.dump_file("first");
    }
    {
        let k2 = ProfileKey {
            tp_size: 2,
            ..key_a()
        };
        let shape = ProfileShape {
            policy_profile: "latency".into(),
            max_forward_tokens: 4096,
            ..ProfileShape::default()
        };
        let _ = s.cache.store(&k2, &shape, &[], 0);
        s.dump_file("second_key");
    }
    {
        let shape = ProfileShape {
            kv_page_size: 32,
            ..ProfileShape::default()
        };
        let _ = s.cache.store(&k, &shape, &[], 0);
        s.dump_file("replace_first");
    }
    s.clear();
    {
        let _ = s.cache.store(&k, &ProfileShape::default(), &[], 0);
        s.dump_file("all_defaults");
    }
    s.put("{ this is not json");
    {
        let shape = ProfileShape {
            kv_page_size: 8,
            ..ProfileShape::default()
        };
        let ok = s.cache.store(&k, &shape, &[], 0);
        let _ = writeln!(
            s.out,
            "store|over_corrupt|ok={}|err={}",
            i32::from(ok.is_ok()),
            ok.err().map(|e| e.to_string()).unwrap_or_default()
        );
        s.dump_file("over_corrupt");
    }
    s.put(
        r#"{"version":2,"note":"kept","entries":[{"key":{"gpu_name":"other"},"plan":{"kv_page_size":4},"extra":[1,2.5,"x"]}]}"#,
    );
    {
        let shape = ProfileShape {
            kv_page_size: 64,
            ..ProfileShape::default()
        };
        let _ = s.cache.store(&k, &shape, &[], 0);
        s.dump_file("merge_preserves");
    }
    {
        let k3 = ProfileKey {
            gpu_name: "A\"B\\C\tD\u{1}E/F\u{e9}".into(),
            ..key_a()
        };
        let shape = ProfileShape {
            policy_profile: "p".into(),
            ..ProfileShape::default()
        };
        let _ = s.cache.store(&k3, &shape, &[], 0);
        s.dump_file("escapes");
    }

    s.out
}

struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp_dir(tag: &str) -> TempDir {
    let p = std::env::temp_dir().join(format!("pie_pc_rs_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("create temp dir");
    TempDir(p)
}

/// Build the raw transcript, then normalise every line the way `run.sh` does.
///
/// Serialised because the sweep reads and writes `planner_budget_bytes`, which
/// is process-wide by design -- the C++ makes it a file-static for the same
/// reason. Two tests sweeping it at once would interleave, and the failure
/// would look like a parity bug rather than a harness bug.
fn normalised_transcript(dir: &Path) -> String {
    static LOCK: Mutex<()> = Mutex::new(());
    let _held = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    set_planner_budget_bytes(0);
    transcript(dir).lines().map(|l| normalise(l) + "\n").collect()
}

#[test]
fn the_rust_cache_matches_the_cpp_transcript() {
    let dir = temp_dir("parity");
    let text = normalised_transcript(&dir.0);
    let bytes = text.as_bytes();

    assert_eq!(
        bytes.iter().filter(|&&b| b == b'\n').count(),
        GOLDEN_ROWS,
        "transcript shape diverged before content could be compared"
    );
    assert_eq!(
        fnv1a64(bytes),
        GOLDEN_FNV1A64,
        "transcript differs from the C++; run tests/oracle/profile_cache/run.sh \
         with PC_ORACLE_OUT set and diff against it"
    );
}

#[test]
fn the_golden_actually_discriminates() {
    // A pin nothing can break is not a pin.
    let dir = temp_dir("mutate");
    let text = normalised_transcript(&dir.0);
    for (what, mutated) in [
        ("a plan field read leniently", text.replacen("|hit||1|0|0|0|", "|REFUSED", 1)),
        ("the wrapped negative budget", text.replacen("18446744073709551615", "0", 1)),
        ("key order in the written file", text.replacen("\"budget_bytes\"", "\"zudget_bytes\"", 1)),
        ("a float in the measured array", text.replacen("46934.815584012416", "46934.81558401242", 1)),
    ] {
        assert_ne!(
            fnv1a64(mutated.as_bytes()),
            GOLDEN_FNV1A64,
            "golden failed to notice a change to {what}"
        );
        assert_ne!(mutated, text, "mutation {what} did not apply");
    }
}

/// Dump the Rust transcript when `PC_RUST_OUT` is set, so it can be diffed
/// against `run.sh`'s output directly.
#[test]
fn emit_transcript_when_asked() {
    if let Ok(dest) = std::env::var("PC_RUST_OUT") {
        let dir = temp_dir("emit");
        std::fs::write(dest, normalised_transcript(&dir.0)).expect("write transcript");
    }
}
