//! Every boot knob this driver reads, parsed once and answered from a
//! value.
//!
//! Named for the BOOT rather than for configuration, because the two
//! things called "config" here are unrelated: one is what a deployment
//! asked this driver to do, the other is what the weights on disk say
//! they are. The second no longer has a schema — a checkpoint's identity
//! is a [`model::catalog`] row now, matched by tensors rather than parsed
//! from a `config.json` — which makes the distinction easier to keep than
//! it was when a `model::config::schema` sat next to this file.
//!
//! # Why one struct rather than ten call sites
//!
//! The knobs used to be read where they were needed: `PIE_CUDA_RUNAHEAD`
//! in the fire path, `PIE_CUDA_KV_ENVELOPES` in the KV pool,
//! `PIE_ATTN_SCORE_WINDOW` behind a `OnceLock` in the score sink,
//! `PIE_RS_STASH_TOKENS` in the recurrent cache, `PIE_LOADER_DEVICE_TRANSFORMS`
//! in the weight arena. Each had its own parser, and they did not agree:
//! one accepted `1|true|on`, another rejected only `0`, a third
//! hand-rolled a signed integer scan over `OsStr` bytes.
//!
//! Three things follow from that, and one struct fixes all three:
//!
//! 1. **There was nowhere to ask what this driver is configured as.** A
//!    reader had to already know the names to find them.
//! 2. **A typo is silent.** `PIE_CUDA_RUNAHED=0` reads as absent, which
//!    means enabled, which is the opposite of what was asked for.
//! 3. **The parsers drift.** `PIE_LOADER_DEVICE_TRANSFORMS=false` turned
//!    the transforms ON, because that one rejected only the string `"0"`.
//!    That is not hypothetical; it is what the code did.
//!
//! # The environment is a fallback, not the source
//!
//! Every knob can come from the boot TOML the engine hands
//! `pie_cuda_create`. The environment is read only where the TOML says
//! nothing, and the order matters: the TOML is what a deployment
//! STATED, and an inherited variable should not silently overrule it.

/// A boolean knob's spelling, in one place.
///
/// `0`, `false`, `off`, `no` and empty are off; anything else present is
/// on. The three hand-rolled versions this replaces disagreed.
fn truthy(v: &str) -> bool {
    !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no" | "")
}

/// A positive integer knob, clamped to a stated ceiling.
///
/// `None` for absent, unparseable, non-positive or above `max` — all of
/// which mean "the default stands" rather than "refuse", because a knob
/// is an override and an override nobody can read is not one.
fn positive(v: &str, max: i64) -> Option<i64> {
    let t = v.trim();
    let n: i64 = t.strip_prefix('+').unwrap_or(t).parse().ok()?;
    (n > 0 && n <= max).then_some(n)
}

/// The KV page size, in tokens.
///
/// SIXTEEN, and it is here because it was written five times: in the
/// fire path, in the capabilities, in the planner's fallback and twice
/// more. It is not a preference — the paged-attention kernels are
/// compiled for it — so it is a constant rather than a knob, and the
/// point of stating it once is that a sixth restatement cannot disagree
/// with the other five.
pub const KV_PAGE_SIZE: i32 = 16;

/// The whole driver's configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Boot {
    /// Run ahead of the device: return from a launch before the fire
    /// retires. `[driver] runahead`, `PIE_CUDA_RUNAHEAD`.
    pub runahead: bool,
    /// Capture fires into a unionized supergraph. `PIE_CUDA_SUPERGRAPH`.
    pub supergraph: bool,
    /// Trace the supergraph's decisions to stderr.
    pub trace_supergraph: bool,
    /// Apply weight transforms on the device rather than the host.
    pub device_transforms: bool,
    /// Advertise and allocate KV envelopes.
    ///
    /// Read by BOTH the pool that would allocate them and the caps that
    /// advertise them, and the two MUST agree — which is the clearest
    /// case for this struct, because agreeing used to be a matter of
    /// two sites spelling the same parse by hand.
    pub kv_envelopes: bool,
    /// How many attention-score rows the sink keeps. Default 32,
    /// ceiling 4096.
    pub attn_score_window: u32,
    /// Cap on the recurrent verify stash, in tokens. `None` is uncapped.
    pub rs_stash_tokens: Option<i32>,
    /// Sweep the shape ladder at load and store the winner.
    /// `[batching] calibrate_planner`.
    pub calibrating: bool,
}

impl Default for Boot {
    /// What this driver does when nothing says otherwise.
    ///
    /// Run-ahead and the supergraph are ON: they are what this shell is
    /// FOR, and their knobs exist to switch them off while bisecting.
    fn default() -> Self {
        Self {
            runahead: true,
            supergraph: true,
            trace_supergraph: false,
            device_transforms: true,
            kv_envelopes: false,
            attn_score_window: 32,
            rs_stash_tokens: None,
            calibrating: false,
        }
    }
}

impl Boot {
    /// Parse the boot TOML, then let the environment fill what it did
    /// not state.
    ///
    /// `env` is a parameter rather than a call to [`std::env::var`] so
    /// this is testable without mutating process state — the same
    /// reason `layout::profile_cache::cache_path` takes one.
    pub fn parse(boot: Option<&toml::Table>, env: impl Fn(&str) -> Option<String>) -> Self {
        let mut cfg = Self::default();
        let table = |section: &str, key: &str| -> Option<&toml::Value> {
            boot?.get(section)?.get(key)
        };
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
        flag(&mut cfg.supergraph, "driver", "supergraph", "PIE_CUDA_SUPERGRAPH");
        flag(&mut cfg.trace_supergraph, "driver", "trace_supergraph",
             "PIE_CUDA_TRACE_SUPERGRAPH");
        flag(&mut cfg.device_transforms, "driver", "device_transforms",
             "PIE_LOADER_DEVICE_TRANSFORMS");
        flag(&mut cfg.kv_envelopes, "driver", "kv_envelopes", "PIE_CUDA_KV_ENVELOPES");
        flag(&mut cfg.calibrating, "batching", "calibrate_planner", "PIE_CUDA_CALIBRATE");

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

    /// Read the real environment. The one place [`std::env::var`] is
    /// called for a driver knob.
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

    /// The defaults are what the driver does with an empty boot file,
    /// and they are not all `false`.
    #[test]
    fn the_defaults_are_the_drivers_ordinary_behaviour() {
        let cfg = Boot::parse(None, no_env);
        assert!(cfg.runahead, "run-ahead is the point of the fire path");
        assert!(cfg.supergraph);
        assert!(cfg.device_transforms);
        assert!(!cfg.kv_envelopes, "envelopes are opt-in; nothing binds them");
        assert_eq!(cfg.attn_score_window, 32);
        assert_eq!(cfg.rs_stash_tokens, None, "uncapped");
    }

    /// The disagreement this struct exists to end. The weight arena
    /// rejected only the string `"0"`, so
    /// `PIE_LOADER_DEVICE_TRANSFORMS=false` turned the transforms ON —
    /// the opposite of what was asked for.
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

    /// The TOML wins, because it is what a deployment STATED and the
    /// environment is what a shell happened to inherit.
    #[test]
    fn the_boot_config_outranks_the_environment() {
        let boot: toml::Table = "[driver]\nrunahead = false\n".parse().expect("parses");
        let cfg = Boot::parse(Some(&boot), |_| Some("1".to_owned()));
        assert!(!cfg.runahead, "the file said off; the environment does not overrule it");
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
        let cfg = Boot::parse(None, |k| (k == "PIE_ATTN_SCORE_WINDOW").then(|| "64".to_owned()));
        assert_eq!(cfg.attn_score_window, 64);
    }
}
