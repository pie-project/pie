//! Every boot knob this driver reads, parsed once and answered from a value.
//!
//! A knob comes from the boot TOML, or the environment where the TOML is
//! silent — that order matters, so an inherited variable cannot overrule what a
//! deployment stated. One struct so the hand-rolled parsers cannot drift apart.

/// A boolean knob: `0`/`false`/`off`/`no`/empty are off, anything else on.
fn truthy(v: &str) -> bool {
    !matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no" | ""
    )
}

/// A positive integer knob, clamped to `max`. `None` — the default stands —
/// for absent, unparseable, non-positive or above `max`.
fn positive(v: &str, max: i64) -> Option<i64> {
    let t = v.trim();
    let n: i64 = t.strip_prefix('+').unwrap_or(t).parse().ok()?;
    (n > 0 && n <= max).then_some(n)
}

/// The KV page size, in tokens.
///
/// NOT A KNOB: the paged-attention kernels are compiled for 16, so this is a
/// fact about `kernels-cuda` that the shell must agree with rather than a
/// choice the shell makes. It is why `PlannerConfig::kv_page_size` is pinned
/// instead of swept — sweeping page sizes would answer a geometry the fire
/// never builds.
///
/// IT HAD ONE READER AND THE NUMBER HAD EIGHT SPELLINGS. `fire::launch`,
/// `serve::transfer` (twice), the planner config, and three `unwrap_or(16)`
/// fallbacks each wrote the literal, so the constant documented a coupling it
/// did not enforce: a build against 32-token pages would have had to be found
/// eight times, and the `unwrap_or` sites are exactly where a miss is
/// silent — a wrong page size does not fault, it reads the neighbour's
/// tokens.
pub const KV_PAGE_SIZE: i32 = 16;

/// The whole driver's configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Boot {
    /// Return from a launch before the fire retires. `PIE_CUDA_RUNAHEAD`.
    pub runahead: bool,
    // `supergraph` and `trace_supergraph` STOOD HERE (`PIE_CUDA_SUPERGRAPH`,
    // `PIE_CUDA_TRACE_SUPERGRAPH`). They selected between the LEGACY fire
    // path's two walks -- a unionized capture whose guards were resolved on
    // the device, and the eager list -- and both walks are gone with the
    // lowering that produced them.
    //
    // THE PERF DEBT IS REAL AND IS NOT HIDDEN HERE: capture bought a
    // Qwen3-0.6B decode 12 ms of replay against 535 launches issued by hand,
    // and the baker path is eager, so it pays the issue cost every fire. A
    // baker capture is a design (`fire::launch`'s note at the walk), not a
    // knob that was switched off.
    //
    // `PIE_CUDA_TRACE_SUPERGRAPH` still WORKS as a variable, because
    // `fire::launch::sg_trace` reads it directly and always did; what it
    // traces now is the fire's phase timings.
    /// Apply weight transforms on the device rather than the host.
    pub device_transforms: bool,
    /// Advertise and allocate KV envelopes — read by both the allocating pool
    /// and the advertising caps, which must agree.
    pub kv_envelopes: bool,
    /// Attention-score rows the sink keeps. Default 32, ceiling 4096.
    pub attn_score_window: u32,
    /// Cap on the recurrent verify stash, in tokens. `None` is uncapped.
    pub rs_stash_tokens: Option<i32>,
    /// Sweep the shape ladder at load and store the winner. `PIE_CUDA_CALIBRATE`.
    pub calibrating: bool,
    // `baker` STOOD HERE (`[driver] baker`, `PIE_BAKER`) and is RETIRED, in
    // the direction that deletes it rather than flips its default. It was
    // the A/B switch: off, the legacy lane fired; on, the baker lane was
    // built BESIDE it and fired instead. There is no legacy lane to sit
    // beside. The baker lane is built on every load, and a checkpoint whose
    // lane will not build is REFUSED at load with the reason named --
    // `serve::load::load_impl`'s last block -- rather than served by
    // something else.
    /// Which catalog row to trace, stated rather than matched.
    /// `[baker] sku`, else `PIE_BAKER_SKU`.
    ///
    /// A STRING KNOB THAT OUTRANKS THE MATCH. `baker::identify` reads the
    /// checkpoint's own tensors against the import tables and answers a SKU
    /// (`"qwen35-d0.8b-bf16-kv-bf16"`); a deployment that sets this is
    /// telling the driver something the tensors cannot say, which is how a
    /// new row is proven before its checkpoint is one the reader can tell
    /// apart. The `BRIDGE` table this knob used to stand in for is gone with
    /// the second catalog — see `baker::identify`.
    pub baker_sku: Option<String>,
}

impl Default for Boot {
    /// What this driver does when nothing says otherwise — run-ahead and the
    /// its knob there to switch it off while bisecting.
    fn default() -> Self {
        Self {
            runahead: true,
            device_transforms: true,
            kv_envelopes: false,
            attn_score_window: 32,
            rs_stash_tokens: None,
            calibrating: false,
            baker_sku: None,
        }
    }
}

impl Boot {
    /// Parse the boot TOML, then let the environment fill what it did not state.
    ///
    /// `env` is a parameter, not [`std::env::var`], so it is testable.
    pub fn parse(boot: Option<&toml::Table>, env: impl Fn(&str) -> Option<String>) -> Self {
        let mut cfg = Self::default();
        let table =
            |section: &str, key: &str| -> Option<&toml::Value> { boot?.get(section)?.get(key) };
        let flag = |slot: &mut bool, section: &str, key: &str, var: &str| {
            if let Some(v) = table(section, key) {
                if let Some(b) = v.as_bool() {
                    *slot = b;
                    return;
                }
                if let Some(s) = v.as_str() {
                    *slot = truthy(s);
                    return;
                }
            }
            if let Some(s) = env(var) {
                *slot = truthy(&s);
            }
        };
        flag(&mut cfg.runahead, "driver", "runahead", "PIE_CUDA_RUNAHEAD");
        flag(
            &mut cfg.device_transforms,
            "driver",
            "device_transforms",
            "PIE_LOADER_DEVICE_TRANSFORMS",
        );
        flag(
            &mut cfg.kv_envelopes,
            "driver",
            "kv_envelopes",
            "PIE_CUDA_KV_ENVELOPES",
        );
        flag(
            &mut cfg.calibrating,
            "batching",
            "calibrate_planner",
            "PIE_CUDA_CALIBRATE",
        );
        cfg.baker_sku = table("baker", "sku")
            .and_then(toml::Value::as_str)
            .map(str::to_owned)
            .or_else(|| env("PIE_BAKER_SKU"))
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());

        if let Some(n) = table("driver", "attn_score_window")
            .and_then(toml::Value::as_integer)
            .filter(|n| *n > 0 && *n <= 4096)
            .or_else(|| env("PIE_ATTN_SCORE_WINDOW").and_then(|s| positive(&s, 4096)))
        {
            cfg.attn_score_window = u32::try_from(n).unwrap_or(32);
        }
        cfg.rs_stash_tokens = table("driver", "rs_stash_tokens")
            .and_then(toml::Value::as_integer)
            .filter(|n| *n > 0)
            .or_else(|| env("PIE_RS_STASH_TOKENS").and_then(|s| positive(&s, i64::from(i32::MAX))))
            .and_then(|n| i32::try_from(n).ok());
        cfg
    }

    /// Read the real environment — the one [`std::env::var`] site for a knob.
    #[must_use]
    pub fn from_boot(boot: Option<&toml::Table>) -> Self {
        Self::parse(boot, |k| std::env::var(k).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// The defaults are the driver's ordinary behaviour, and not all `false`.
    #[test]
    fn the_defaults_are_the_drivers_ordinary_behaviour() {
        let cfg = Boot::parse(None, no_env);
        assert!(cfg.runahead, "run-ahead is the point of the fire path");
        assert!(cfg.device_transforms);
        assert!(
            !cfg.kv_envelopes,
            "envelopes are opt-in; nothing binds them"
        );
        assert_eq!(cfg.attn_score_window, 32);
        assert_eq!(cfg.rs_stash_tokens, None, "uncapped");
    }

    /// Every false spelling reads as off — the parser drift this struct ends.
    #[test]
    fn every_boolean_knob_reads_false_the_same_way() {
        for spelling in ["0", "false", "off", "no", "FALSE", " off "] {
            let cfg = Boot::parse(None, |k| {
                (k == "PIE_LOADER_DEVICE_TRANSFORMS").then(|| spelling.to_owned())
            });
            assert!(!cfg.device_transforms, "{spelling:?} reads as off");
        }
        for spelling in ["1", "true", "on", "yes"] {
            let cfg = Boot::parse(None, |k| {
                (k == "PIE_CUDA_KV_ENVELOPES").then(|| spelling.to_owned())
            });
            assert!(cfg.kv_envelopes, "{spelling:?} reads as on");
        }
    }

    /// The TOML outranks an inherited environment variable.
    #[test]
    fn the_boot_config_outranks_the_environment() {
        let boot: toml::Table = "[driver]\nrunahead = false\n".parse().expect("parses");
        let cfg = Boot::parse(Some(&boot), |_| Some("1".to_owned()));
        assert!(
            !cfg.runahead,
            "the file said off; the environment does not overrule it"
        );
    }

    /// An unreadable override leaves the default rather than refusing.
    #[test]
    fn an_unparseable_number_leaves_the_default() {
        for junk in ["", "abc", "-4", "0", "99999"] {
            let cfg = Boot::parse(None, |k| {
                (k == "PIE_ATTN_SCORE_WINDOW").then(|| junk.to_owned())
            });
            assert_eq!(cfg.attn_score_window, 32, "{junk:?} leaves the default");
        }
        let cfg = Boot::parse(None, |k| {
            (k == "PIE_ATTN_SCORE_WINDOW").then(|| "64".to_owned())
        });
        assert_eq!(cfg.attn_score_window, 64);
    }
}
