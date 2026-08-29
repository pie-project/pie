//! Opening a CUDA device from a boot config.
//!
//! # What this file stopped being
//!
//! It was 415 lines of adapter: a `CudaEngineHandle` per rank, an
//! `impl Engine for CudaEngine` forwarding fourteen verbs to a leader shell,
//! a `status(Result<T, i32>, verb)` translating the shell's own `i32` ladder,
//! a `CompletionBroker` per handle, and a `load_model` that fanned a
//! `Vec<ModelLoadDesc>` across `std::thread::scope`.
//!
//! Every one of those was a shape the palo rewrite dissolved:
//!
//! * **The `Engine` impl is the shell's.** `engine_cuda::Cuda` implements the
//!   contract, in the crate that owns the device — which is what decision 13
//!   ("no `trait Backend`; shells are thin call-order crates") means when the
//!   trait in question is the contract itself. This module selects a device;
//!   it does not adapt one.
//! * **The `i32` ladder is gone** with `PIE_STATUS_*` (design §7). The shell
//!   answers a typed `Fault` and `engine_cuda::api` maps it to `Error`
//!   once.
//! * **The broker is the runtime's** (`crate::engine::completion`), so a
//!   per-handle one here was the wrong side of the boundary twice over.
//! * **A rank is not a load.** `load_model(Vec<ModelLoadDesc>)` shipped one
//!   descriptor per rank and cross-checked that they agreed;
//!   [`LoadRequest`](engine_api::LoadRequest) is one plan, `Shard::Cut` is in
//!   the plan, and which rank a shell is, is the shell's own.
//!
//! # What is left, and it is the whole file
//!
//! Read the boot TOML for what it says about the MACHINE — which device, how
//! much of a fire to record, where the warm-boot weight artifacts live, and
//! the shell's own knobs — and hand the shell the load door
//! (`crate::engine::load::contract_for`) it cannot state for itself.
//!
//! **THE KNOBS ARRIVE HERE BECAUSE ARTICLE 9 SAYS SHELLS READ NO
//! ENVIRONMENT** (alto design §1). Nine `PIE_CUDA_*` words were read inside
//! `engine_cuda::serve` at load; they are `[engine]` keys now, and this is
//! where a boot document turns into the typed [`Knobs`] the shell is given.
//! Every key is optional and every default is what the absent environment
//! variable meant, so a deployment that states nothing fires exactly what it
//! fired before.

use anyhow::{Result, anyhow};
use engine_cuda::{Cuda, DeviceBoot, Graphs, Knobs};

/// Open one device from a boot document.
///
/// # Errors
///
/// A boot document that is not TOML. Binding the device itself happens at
/// [`Engine::load`](engine_api::Engine::load), not here: `Shell::load` is one
/// call that binds, bakes and lands, and there is nothing to bind before a
/// plan says what to bake.
pub fn open(config_bytes: &[u8]) -> Result<Cuda> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| anyhow!("the cuda boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| anyhow!("the cuda boot config is not TOML: {error}"))?;
    Ok(Cuda::new(
        DeviceBoot {
            ordinal: ordinal(&doc),
            graphs: graphs(&doc),
            knobs: knobs(&doc),
            weight_cache_dir: weight_cache_dir(&doc),
            program_cache_dir: program_cache_dir(&doc),
        },
        crate::engine::load::contract_for,
    ))
}

/// How many descriptor-port envelopes the CUDA shell has resolved off guest
/// device rings in this process (`palo B3`).
///
/// **THE ONE OBSERVABLE OF A NEGATIVE.** Device-carried decode's whole claim
/// is that a chained fire's token did not travel to the host, and a round
/// trip that does not happen leaves no trace. What DOES happen is one
/// envelope resolved per attached device-carried lane per fire, so a serving
/// gate asserts on this: zero says every decode serialized through the host
/// plane, `>= decodes` says the shell read the token off the ring the
/// epilogue wrote.
///
/// Re-exported here rather than reached for directly because `engine-cuda` is
/// a private link of this crate — `_engine-cuda` is what gates it — and a
/// test that named the shell crate would be a test that could not build
/// without a GPU feature it does not select.
#[must_use]
pub fn envelopes_resolved() -> u64 {
    engine_cuda::Shell::envelopes_resolved()
}

/// The CUDA fold's motion counters —
/// `(folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins)` —
/// re-exported on `envelopes_resolved`'s argument exactly: the shell is a
/// private link of this crate, the instance lives behind `Box<dyn Engine>`
/// on a scheduler lane thread, and what a runtime-level fold gate diffs is
/// this process-global mirror before and after a serving loop. `prebinds`
/// moving is the one observable that the runtime's own next-fire hint
/// (`Engine::expect_fire`, stated from `scheduler::worker::fire_frame`)
/// reached the shell — nothing else in the runtime can say so, because a
/// hint that lands leaves no trace in any completion. The two micros
/// columns split the same binding work into its on-critical-path and
/// hidden halves, which is the number the `PIE_CUDA_PIPELINE` A/B moves.
#[must_use]
pub fn fold_observed() -> (u64, u64, u64, u64, u64, u64, u64) {
    engine_cuda::Shell::fold_observed()
}

/// Which device `[model] device` names.
///
/// `"cuda:1"`, `"1"` and an absent key all mean something, and the third
/// means zero — a single-GPU box is the deployment that writes the least.
fn ordinal(doc: &toml::Table) -> i32 {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("device"))
        .and_then(toml::Value::as_str)
        .and_then(|device| device.rsplit(':').next())
        .and_then(|ordinal| ordinal.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

/// How much of a fire `[engine] graphs` asks to record.
///
/// Nothing overrides it any more: `PIE_CUDA_GRAPHS` died with the other eight
/// words when article 9 landed, so this key IS the answer.
fn graphs(doc: &toml::Table) -> Graphs {
    match doc
        .get("engine")
        .and_then(toml::Value::as_table)
        .and_then(|engine| engine.get("graphs"))
        .and_then(toml::Value::as_str)
    {
        Some("off" | "eager") => Graphs::Off,
        Some("shaped") => Graphs::Shaped,
        Some("on" | "graph") => Graphs::On,
        _ => Graphs::default(),
    }
}

/// The shell's own words, off the `[engine]` table.
///
/// **NINE ENVIRONMENT READS, ONE TABLE** (alto wave P, article 9). Each key
/// below was a `PIE_CUDA_*` variable read inside the shell at load; two of the
/// nine are not here because they were never shell flags — `[engine] graphs`
/// is [`graphs`] above and the shape lattice is a compiler input that reaches
/// the bake through `LoadBudgets::buckets`.
///
/// Absent means the shell's own default, which is what the absent variable
/// meant, so this function states nothing it was not told. A key spelled with
/// a value the reader does not understand is absent too: the arms below
/// accept exactly the spellings the environment words accepted, and anything
/// else leaves the default rather than inventing a third answer.
fn knobs(doc: &toml::Table) -> Knobs {
    let table = doc.get("engine").and_then(toml::Value::as_table);
    let word = |key: &str| -> Option<&str> {
        table
            .and_then(|engine| engine.get(key))
            .and_then(toml::Value::as_str)
    };
    // A boolean key may be written as one (`pad = false`) or as the word the
    // environment variable took (`pad = "off"`). Both are the operator saying
    // the same thing, and refusing one of them would be a schema this document
    // has nowhere to publish.
    let flag = |key: &str, default: bool| -> bool {
        match table.and_then(|engine| engine.get(key)) {
            Some(toml::Value::Boolean(set)) => *set,
            Some(toml::Value::String(text)) => match text.trim() {
                "off" | "0" | "false" | "no" => false,
                "on" | "1" | "true" | "yes" => true,
                _ => default,
            },
            _ => default,
        }
    };
    let stock = Knobs::default();
    Knobs {
        pad: flag("pad", stock.pad),
        fold: flag("fold", stock.fold),
        pipeline: flag("pipeline", stock.pipeline),
        // `all` (the default) disables every absent-window node of a folded
        // exec; `library` keeps pie windowed nodes enabled at fitted zero rows
        // and disables only the library residue. Spelled as the policy's own
        // name rather than as a boolean, because that is what the measurement
        // in `.wiki/palo/cuda-abi.md` §6d calls the two arms.
        fold_disable_library: match word("fold_disable") {
            Some("library" | "lib") => true,
            Some(_) | None => stock.fold_disable_library,
        },
        copies: flag("fallback_copy", stock.copies),
        grouped: flag("grouped", stock.grouped),
        // `off` is P6's off arm and bakes an artifact with no fork group at
        // all; a number caps how many side streams the compiler may hand out;
        // absent leaves the device profile's own figure, which is why this is
        // an `Option` and not a number with a sentinel.
        side_streams: match table.and_then(|engine| engine.get("side_streams")) {
            Some(toml::Value::Integer(streams)) => Some(u32::try_from(*streams).unwrap_or(0)),
            Some(toml::Value::String(text)) => match text.trim() {
                "off" | "none" => Some(0),
                other => other.parse().ok(),
            },
            _ => stock.side_streams,
        },
    }
}

/// Where `[model] weight_cache_dir` says this deployment keeps its warm-boot
/// weight artifacts (alto design §7).
///
/// **THE KEY WAS ALREADY BEING WRITTEN AND NOBODY READ IT.** The worker has
/// resolved this directory since the palo rewrite — `$PIE_HOME/cache/weights`
/// unless `[model] weight_cache_dir` names another — written it into every
/// CUDA boot document, and dropped it on the floor at this seam, because the
/// shell had no cache to point at. Now it has one, and this is the read.
///
/// Absent or empty is `None`: the feature is off. That distinction is the
/// operator's and it is preserved exactly — an empty string is what a config
/// says when it means "you decide", and the worker has already decided by the
/// time these bytes are written, so what arrives here empty is a deployment
/// that turned the cache off.
///
/// Relative is refused by [`crate::config`] long before this point, so what
/// reaches here is absolute or nothing.
fn weight_cache_dir(doc: &toml::Table) -> Option<std::path::PathBuf> {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("weight_cache_dir"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
}

/// Where `[cache] dir` says this deployment keeps its caches, plus the guest
/// program plane's own subdirectory.
///
/// **THE SECOND KEY THE WORKER HAS BEEN WRITING ALL ALONG.** `[cache] dir` is
/// `$PIE_HOME/cache`, emitted into every boot document since the palo rewrite;
/// the shell resolved the same path for itself with three `env::var_os` calls
/// (`PIE_HOME`, then `XDG_CACHE_HOME`, then `HOME`), which was the last
/// environment read in `engine-cuda` and is what article 9 forbids. The
/// subdirectory name is the one those calls produced, so a deployment that
/// booted through the worker finds the cubins it already wrote.
///
/// Absent or empty is `None`: no cubin is stored and every program compiles
/// through NVRTC, which costs time and never an answer.
fn program_cache_dir(doc: &toml::Table) -> Option<std::path::PathBuf> {
    doc.get("cache")
        .and_then(toml::Value::as_table)
        .and_then(|cache| cache.get("dir"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(|dir| std::path::Path::new(dir).join("ptir-cuda"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key the worker has been writing all along, finally read.
    #[test]
    fn the_weight_cache_directory_is_read_and_an_empty_one_is_off() {
        let read = |text: &str| weight_cache_dir(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(
            read("[model]\nweight_cache_dir = \"/var/pie/cache/weights\""),
            Some(std::path::PathBuf::from("/var/pie/cache/weights"))
        );
        assert_eq!(read("[model]\nweight_cache_dir = \"\""), None, "empty is off");
        assert_eq!(read("[model]"), None, "and so is absent");
        assert_eq!(read(""), None);
    }

    #[test]
    fn the_device_key_is_read_in_every_spelling_a_boot_config_uses() {
        let read = |text: &str| ordinal(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read(r#"[model]
device = "cuda:3""#), 3);
        assert_eq!(read(r#"[model]
device = "2""#), 2);
        assert_eq!(read("[model]"), 0, "a config that says nothing means device 0");
        assert_eq!(read(""), 0);
    }

    #[test]
    fn the_graph_mode_defaults_to_the_shells_own() {
        let read = |text: &str| graphs(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read(r#"[engine]
graphs = "on""#), Graphs::On);
        assert_eq!(read(r#"[engine]
graphs = "shaped""#), Graphs::Shaped);
        assert_eq!(read(""), Graphs::default());
    }

    /// **THE ROUND TRIP ARTICLE 9 IS FOR** (alto wave P's gate): a word set in
    /// the boot document reaches the shell's own toggle, with no environment
    /// anywhere in the path. `pad` and `fold` are the two the gate names.
    #[test]
    fn a_knob_set_in_the_boot_document_reaches_the_shells_toggle() {
        let read = |text: &str| knobs(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read("").pad, true, "absent is the shell's own default");
        assert_eq!(read("").fold, false);
        assert_eq!(read("[engine]\npad = false").pad, false);
        assert_eq!(read("[engine]\npad = \"off\"").pad, false, "the word spelling too");
        assert_eq!(read("[engine]\nfold = true").fold, true);
        assert_eq!(read("[engine]\nfold = \"on\"").fold, true);
        // Everything a key does not mention keeps the default, so one stated
        // knob never moves another.
        let one = read("[engine]\nfold = true");
        assert_eq!(
            Knobs { fold: false, ..one },
            Knobs::default(),
            "stating `fold` moved something else"
        );
    }

    /// `[cache] dir` and the subdirectory the shell used to derive itself.
    #[test]
    fn the_program_cache_directory_is_read_and_an_empty_one_is_off() {
        let read =
            |text: &str| program_cache_dir(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(
            read("[cache]\ndir = \"/pie-home/cache\""),
            Some(std::path::PathBuf::from("/pie-home/cache/ptir-cuda"))
        );
        assert_eq!(read("[cache]\ndir = \"\""), None, "empty is off");
        assert_eq!(read(""), None, "and so is absent");
    }

    /// The other five keys, and the two that are not booleans.
    #[test]
    fn every_engine_knob_key_is_read_in_the_spellings_the_words_took() {
        let read = |text: &str| knobs(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read("[engine]\npipeline = \"off\"").pipeline, false);
        assert_eq!(read("[engine]\nfallback_copy = 0").copies, true, "not a bool: default");
        assert_eq!(read("[engine]\nfallback_copy = \"0\"").copies, false);
        assert_eq!(read("[engine]\ngrouped = \"none\"").grouped, true, "unknown word: default");
        assert_eq!(read("[engine]\ngrouped = false").grouped, false);
        assert_eq!(
            read("[engine]\nfold_disable = \"library\"").fold_disable_library,
            true
        );
        assert_eq!(read("[engine]\nfold_disable = \"all\"").fold_disable_library, false);
        assert_eq!(read("[engine]\nside_streams = 4").side_streams, Some(4));
        assert_eq!(read("[engine]\nside_streams = \"off\"").side_streams, Some(0));
        assert_eq!(read("").side_streams, None, "absent leaves the profile's figure");
    }
}
