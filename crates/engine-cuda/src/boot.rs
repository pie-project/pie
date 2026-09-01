//! Opening a CUDA device from a boot config: this crate's `[…]` tables, read
//! into this crate's [`DeviceBoot`].
//!
//! # Why the reader is in the shell
//!
//! **THIS FILE USED TO BE `runtime::engine::backend::cuda`**, and the
//! runtime's `backend.rs` said why: *"the boot TOML is the runtime's format on
//! purpose: an engine that parsed it would be the second thing entitled to an
//! opinion about the file's shape, and the two would drift."*
//!
//! That argument was about the FORMAT and it survives intact — the worker
//! writes the document, `runtime::config` decides what a key may say, and
//! nothing below invents one. What did not survive was its cost. Every
//! function here parses into a type this crate declares ([`DeviceBoot`],
//! [`Graphs`], [`Knobs`]), so the crate that could not name a CUDA device
//! still had to name the structs one is opened with — 347 lines of
//! `engine_cuda::` in a crate that is not this one, and adding any backend to
//! the workspace meant editing the runtime.
//!
//! So the direction inverted. A shell reads its own tables and answers its own
//! type; the runtime hands over the bytes and the one thing a shell cannot
//! state for itself, which is the load door ([`ContractFor`]). The drift the
//! old comment feared would need a SECOND reader of the same table, and that
//! is exactly what the runtime no longer has: `open::cuda` is three lines and
//! names no key.
//!
//! # What this file stopped being, before that
//!
//! It was 415 lines of adapter: a `CudaEngineHandle` per rank, an
//! `impl Engine for CudaEngine` forwarding fourteen verbs to a leader shell,
//! a `status(Result<T, i32>, verb)` translating the shell's own `i32` ladder,
//! a `CompletionBroker` per handle, and a `load_model` that fanned a
//! `Vec<ModelLoadDesc>` across `std::thread::scope`.
//!
//! Every one of those was a shape the palo rewrite dissolved:
//!
//! * **The `Engine` impl is this crate's.** [`Cuda`] implements the contract,
//!   in the crate that owns the device — which is what decision 13 ("no
//!   `trait Backend`; shells are thin call-order crates") means when the trait
//!   in question is the contract itself. This module selects a device; it does
//!   not adapt one.
//! * **The `i32` ladder is gone** with `PIE_STATUS_*` (design §7). The shell
//!   answers a typed `Fault` and [`crate::api`] maps it to `Error` once.
//! * **The broker is the runtime's** (`runtime::engine::completion`), so a
//!   per-handle one was the wrong side of the boundary twice over.
//! * **A rank is not a load.** `load_model(Vec<ModelLoadDesc>)` shipped one
//!   descriptor per rank and cross-checked that they agreed;
//!   [`LoadRequest`](engine::LoadRequest) is one plan, `Shard::Cut` is in the
//!   plan, and which rank a shell is, is the shell's own. What is left of that
//!   fan-out is arity policy over the runtime's registry and stayed there
//!   (`open::cuda_group`): this door opens ONE device from ONE document and
//!   has no way to know how many documents a launcher was handed.
//!
//! # What is left, and it is the whole file
//!
//! Read the boot TOML for what it says about the MACHINE — which device, how
//! much of a fire to record, where the warm-boot weight artifacts live, and
//! this shell's own knobs — and take the load door it cannot state for itself
//! as a parameter.
//!
//! **THE KNOBS ARRIVE HERE BECAUSE ARTICLE 9 SAYS SHELLS READ NO
//! ENVIRONMENT** (alto design §1). Nine `PIE_CUDA_*` words were read inside
//! [`crate::serve`] at load; they are `[engine]` keys now, and this is where a
//! boot document turns into the typed [`Knobs`] the shell is given. Every key
//! is optional and every default is what the absent environment variable
//! meant, so a deployment that states nothing fires exactly what it fired
//! before.
//!
//! Article 9 and this file are not in tension, and the difference is the whole
//! point of the article: a `getenv` is a channel the deployment cannot see,
//! and an `[engine]` table is one it wrote.

use crate::api::{ContractFor, Cuda, DeviceBoot};
use crate::serve::{Graphs, Knobs};

/// Open one device from a boot document.
///
/// The contract lookup is a PARAMETER and not something this crate could
/// find: how a checkpoint's tensors become a plan's params is the model's
/// declaration, resolved by the party that links the catalog. See
/// [`ContractFor`], and [`crate::api`]'s header for the diagram. It is also
/// why this function can live here at all — it is the one ingredient of an
/// open that points the wrong way up the dependency graph, and taking it as
/// an argument is what keeps `engine-cuda → runtime` from existing.
///
/// # Errors
///
/// A boot document that is not UTF-8 or not TOML, as a sentence. `String`
/// rather than [`Fault`](crate::Fault), and for the reason
/// [`ContractFor`] is spelled the same way in the other direction: this is a
/// seam between a crate whose errors are `anyhow` and one whose errors are
/// `Fault`, and neither should have to name the other's error crate to open a
/// device. Nothing here touches a device, so no variant of `Fault` describes
/// what can go wrong anyway.
///
/// Binding the device itself happens at
/// [`Engine::load`](engine::Engine::load), not here: `Shell::load` is one call
/// that binds, bakes and lands, and there is nothing to bind before a plan
/// says what to bake.
pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Cuda, String> {
    Ok(Cuda::new(read(config_bytes)?, contract_for))
}

/// **The boot document, as this shell's own type** — [`open`] without the
/// device it would be opened as.
///
/// The whole of [`open`]'s reading, split out so that WHAT A DOCUMENT SAYS can
/// be checked without a machine. Every key here has had the same failure at
/// least once — `[model] weight_cache_dir` was written by the worker and
/// parsed by nobody for a whole rewrite, and `[engine] gpu_mem_utilization`
/// for four waves after that — and a key nobody can read back is a key whose
/// arrival nothing tests.
///
/// # Errors
///
/// A boot document that is not UTF-8 or not TOML, or one whose keys are out of
/// the range they declare, as a sentence.
pub fn read(config_bytes: &[u8]) -> Result<DeviceBoot, String> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the cuda boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the cuda boot config is not TOML: {error}"))?;
    Ok(DeviceBoot {
        ordinal: ordinal(&doc),
        graphs: graphs(&doc),
        knobs: knobs(&doc)?,
        weight_cache_dir: weight_cache_dir(&doc),
        program_cache_dir: program_cache_dir(&doc),
        adapter_dir: adapter_dir(&doc),
    })
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
/// the bake through `LoadBudgets::buckets` — and three more are not here
/// because they named the FOLD, which the tier-2 campaign deleted along with
/// the keyed capture path (`fold`, `pipeline`, `fold_disable`). A boot
/// document that still states one of the three is read exactly as a document
/// that states any other unknown key: ignored.
///
/// **AND ONE KEY BELOW WAS NEVER A VARIABLE AT ALL.** `[engine] bodies` landed
/// after article 9 did, so it has no environment ancestor to be the round trip
/// OF — it is simply a knob, read here, the way every knob should have been.
/// It is stated in the same closure as the rest because "how a boolean is
/// spelled in this document" is one answer and not a per-key one.
///
/// Absent means the shell's own default, which is what the absent variable
/// meant, so this function states nothing it was not told. A key spelled with
/// a value the reader does not understand is absent too: the arms below
/// accept exactly the spellings the environment words accepted, and anything
/// else leaves the default rather than inventing a third answer.
fn knobs(doc: &toml::Table) -> Result<Knobs, String> {
    let table = doc.get("engine").and_then(toml::Value::as_table);
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
    Ok(Knobs {
        gpu_mem_utilization: gpu_mem_utilization(table, stock.gpu_mem_utilization)?,
        pad: flag("pad", stock.pad),
        // The bodies path (`record::BodyKey`) — one exec per composition,
        // with the row count on the staged live-rows seat. Never a
        // `PIE_CUDA_*` word: it landed after article 9, so it was born here.
        //
        // **ON UNLESS THIS DOCUMENT SAYS OTHERWISE, SINCE THE TIER-2
        // CAMPAIGN.** It shipped off while a keyed cache stood beside it as
        // the arm it was diffed against; that cache is gone, so bodies are the
        // only recorded path and `bodies = off` is now the DIAGNOSTIC arm —
        // `graphs` still on, schedules still graph-shaped, every fire walking
        // eagerly and nothing captured. `Shell::load` prints a line when a
        // document asks for it, exactly as it does for `graphs = off`.
        //
        // **AND IT IS THE ONE KNOB HERE THAT DOES WORK AT LOAD** (the bodies
        // design's chunk C): leaving it on makes `Shell::arm_bodies` fire a
        // synthetic composition at every lattice rung the deployment can seat
        // and then SEAL the map, so the steady state's first fire replays and
        // the serving path captures nothing. `set_bodies` between fires still
        // turns the path on and off; it cannot re-run the arming, because
        // arming is a load-time pass and this is the word that decides whether
        // the load takes it.
        //
        // A load that arms prints one line saying how many rungs it armed and
        // how many it lost, and to what — the arming's own faults, the
        // compositions the admissibility rule turned away, and the schedules
        // that would not fit their workspace grant. Nothing else reports a
        // partial arm.
        bodies: flag("bodies", stock.bodies),
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
    })
}

/// **What fraction of the card `[engine] gpu_mem_utilization` lets pie hold**,
/// weights included (alto streaming §3 item 5, `next.md` B1).
///
/// **THE KEY WAS ALREADY BEING WRITTEN AND NOBODY READ IT** — the same
/// sentence [`weight_cache_dir`] earns, one table over and for four waves
/// longer. `worker::config` has declared it, defaulted it to `0.90`, validated
/// it in `(0.0, 1.0]` and put it in the schema since before the palo rewrite,
/// and `grep` found no reader in any shell: the elastic pool took ~100% of
/// what the card had free. This is the read.
///
/// Absent is `default`, which is [`crate::DEFAULT_GPU_MEM_UTILIZATION`] —
/// `0.90`, the worker's own default for the key, so a boot document that says
/// nothing means what the config that says nothing means.
///
/// **OUT OF RANGE REFUSES AT BOOT, BY THE KEY'S NAME**, rather than being
/// clamped into something that runs. It is validated in `worker::config` too
/// and this is not a duplicate schema: a boot document may be written by hand
/// or by another launcher, this shell is the party that turns the number into
/// bytes, and a fraction of `0` or `1.7` is a deployment nobody meant. An
/// INTEGER is accepted beside a float for the one value that has both
/// spellings (`gpu_mem_utilization = 1`), because refusing that would be a
/// schema this document has nowhere to publish — the same ruling `flag` makes
/// about `pad = "off"`.
fn gpu_mem_utilization(table: Option<&toml::Table>, default: f64) -> Result<f64, String> {
    let stated = match table.and_then(|engine| engine.get("gpu_mem_utilization")) {
        None => return Ok(default),
        Some(toml::Value::Float(fraction)) => *fraction,
        #[expect(
            clippy::cast_precision_loss,
            reason = "the only integers in range are 0 and 1, both exact"
        )]
        Some(toml::Value::Integer(fraction)) => *fraction as f64,
        Some(other) => {
            return Err(format!(
                "[engine] gpu_mem_utilization is a fraction of this card in (0.0, 1.0],                  and this document spells it {other}"
            ));
        }
    };
    if !stated.is_finite() || stated <= 0.0 || stated > 1.0 {
        return Err(format!(
            "[engine] gpu_mem_utilization must be finite and in (0.0, 1.0]; this \
             document says {stated}. It is the fraction of the whole card this \
             deployment lets pie hold, weights included — 1.0 is the whole card, \
             which is what the elastic pool took before the key reached a shell \
             at all."
        ));
    }
    Ok(stated)
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

/// Where `[model] adapter_dir` says this deployment keeps its shared adapters
/// (alto adapter §3.3).
///
/// **THE MOUNT, AND IT IS A DIRECTORY AND NOT A REGISTRY.** What lives under
/// it is one subdirectory per adapter, each holding an `adapter.toml` and the
/// plane files that manifest names. Adding one is writing files; nothing here
/// is a catalog, and `Adapters::slots` bounds how many can be RESIDENT at
/// once, not how many may exist.
///
/// Absent or empty is `None`: the feature is off, a shared bind refuses by
/// name, and an adapter registered from the caller's own bytes still works —
/// which is what makes this key optional rather than a floor.
///
/// Read here for the same reason as the two directories above it: the shell
/// takes typed words off the boot document and reads no environment (article
/// 9).
fn adapter_dir(doc: &toml::Table) -> Option<std::path::PathBuf> {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("adapter_dir"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
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

    /// The shared-adapter mount, read the same way and off by the same
    /// absence (alto adapter §3.3).
    #[test]
    fn the_shared_adapter_directory_is_read_and_an_empty_one_is_off() {
        let read = |text: &str| adapter_dir(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(
            read("[model]\nadapter_dir = \"/srv/pie/shared\""),
            Some(std::path::PathBuf::from("/srv/pie/shared"))
        );
        assert_eq!(read("[model]\nadapter_dir = \"  \""), None, "empty is off");
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
    /// anywhere in the path. `pad` is the one the gate names, and `bodies` is
    /// the first knob that never had an environment word to fall back on — so
    /// this round trip is the ONLY way it can be set at all.
    #[test]
    fn a_knob_set_in_the_boot_document_reaches_the_shells_toggle() {
        let read = |text: &str| {
            knobs(&text.parse::<toml::Table>().expect("valid TOML")).expect("a legal document")
        };
        assert_eq!(read("").pad, true, "absent is the shell's own default");
        assert_eq!(read("[engine]\npad = false").pad, false);
        assert_eq!(read("[engine]\npad = \"off\"").pad, false, "the word spelling too");
        assert_eq!(
            read("").bodies,
            true,
            "on is the shipping arm since the keyed path died"
        );
        assert_eq!(read("[engine]\nbodies = false").bodies, false);
        assert_eq!(read("[engine]\nbodies = \"off\"").bodies, false);
        assert_eq!(read("[engine]\nbodies = \"on\"").bodies, true);
        assert_eq!(
            read("[engine]\nbodies = 0").bodies,
            true,
            "an integer is not a spelling this document accepts, so the default stands",
        );
        // Everything a key does not mention keeps the default, so one stated
        // knob never moves another.
        let one = read("[engine]\nbodies = false");
        assert!(
            Knobs { bodies: true, ..one } == Knobs::default(),
            "stating `bodies` moved something else"
        );
        let one = read("[engine]\npad = false");
        assert!(
            Knobs { pad: true, ..one } == Knobs::default(),
            "stating `pad` moved something else"
        );
        // **AND A RETIRED KEY IS AN UNKNOWN KEY.** `fold`, `pipeline` and
        // `fold_disable` named the graph fold, which died with the keyed
        // capture path; a document that still states one is read exactly as a
        // document that states nothing.
        assert!(
            read("[engine]\nfold = true\npipeline = false\nfold_disable = \"library\"")
                == Knobs::default(),
            "a retired key moved a live one"
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

    /// The other keys, and the one that is not a boolean.
    #[test]
    fn every_engine_knob_key_is_read_in_the_spellings_the_words_took() {
        let read = |text: &str| {
            knobs(&text.parse::<toml::Table>().expect("valid TOML")).expect("a legal document")
        };
        assert_eq!(read("[engine]\nfallback_copy = 0").copies, true, "not a bool: default");
        assert_eq!(read("[engine]\nfallback_copy = \"0\"").copies, false);
        assert_eq!(read("[engine]\ngrouped = \"none\"").grouped, true, "unknown word: default");
        assert_eq!(read("[engine]\ngrouped = false").grouped, false);
        assert_eq!(read("[engine]\nside_streams = 4").side_streams, Some(4));
        assert_eq!(read("[engine]\nside_streams = \"off\"").side_streams, Some(0));
        assert_eq!(read("").side_streams, None, "absent leaves the profile's figure");
    }

    /// **THE `[engine]` KEY THAT WAS NEVER A `PIE_CUDA_*` WORD** (alto
    /// streaming §3 item 5, `next.md` B1): the fraction of the
    /// card this deployment lets pie hold. Declared, defaulted, validated and
    /// schema'd in `worker::config` for four waves and read by no shell —
    /// which is the failure `[model] weight_cache_dir` had, one table over.
    #[test]
    fn the_memory_fraction_is_read_and_an_illegal_one_refuses_by_the_keys_name() {
        let read = |text: &str| {
            let doc = text.parse::<toml::Table>().expect("valid TOML");
            let table = doc.get("engine").and_then(toml::Value::as_table);
            gpu_mem_utilization(table, crate::DEFAULT_GPU_MEM_UTILIZATION)
        };
        assert_eq!(read("").expect("absent is legal"), 0.90, "the config's own default");
        assert_eq!(read("[engine]\ngraphs = \"on\"").expect("legal"), 0.90);
        assert_eq!(read("[engine]\ngpu_mem_utilization = 0.5").expect("legal"), 0.5);
        // The whole card, in both spellings an operator writes it in.
        assert_eq!(read("[engine]\ngpu_mem_utilization = 1.0").expect("legal"), 1.0);
        assert_eq!(read("[engine]\ngpu_mem_utilization = 1").expect("legal"), 1.0);
        // And out of range refuses rather than clamping: a fraction silently
        // rounded into range is a number the operator cannot see is not theirs.
        for bad in ["0.0", "-0.5", "1.5", "nan", "\"lots\""] {
            let refusal = read(&format!("[engine]\ngpu_mem_utilization = {bad}"))
                .expect_err("outside (0.0, 1.0] is not a deployment");
            assert!(refusal.contains("gpu_mem_utilization"), "got: {refusal}");
        }
    }
}
