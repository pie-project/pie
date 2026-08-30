//! Opening a Metal device from a boot config.
//!
//! # Why the reader is in the shell
//!
//! **THIS FILE USED TO BE `runtime::engine::backend::metal`**, and the
//! runtime's `backend.rs` said why: *"the boot TOML is the runtime's format on
//! purpose: an engine that parsed it would be the second thing entitled to an
//! opinion about the file's shape, and the two would drift."* That argument
//! was about the FORMAT and it survives — the worker writes the document, and
//! nothing here invents a key. What it did not survive was its own cost: the
//! reader parsed into [`DeviceBoot`], which is this crate's type, so a crate
//! that could not name a Metal device still had to name the struct one is
//! opened with. Adding a backend meant editing the runtime.
//!
//! So the direction inverted. A shell reads its own `[…]` tables out of the
//! shared document and answers its own type; the runtime hands over the bytes
//! and the one thing a shell cannot state for itself, which is the load door
//! ([`ContractFor`]). No key here is read by anyone else, and the second
//! opinion the old comment feared would have to be a second reader of the SAME
//! table — which is what the runtime no longer has.
//!
//! # What this file stopped being, before that
//!
//! It was a door onto a shell that did not exist: a `MetalEngine` the runtime
//! defined itself, one field (`[model] id`, carried so the refusals could
//! name it), and every verb of the contract answering `Error::Unsupported`.
//! Its own header said what was missing — *"a device to bind, a checkpoint to
//! land, pools to reserve, and a command buffer to encode onto"* — and named
//! `engine-cuda/serve.rs` as the shape it would take.
//!
//! It took that shape. This crate is the whole shell now: `device/`,
//! `weights.rs`, `store/`, the arena, the resident inputs, the windows,
//! `serve.rs`, and the guest-program plane beside them.
//!
//! # What is left, and it is less than the CUDA door's
//!
//! `engine_cuda::boot` reads two things out of the boot TOML: which device,
//! and how much of a fire to record. Neither exists here. Metal selects with
//! `MTLCreateSystemDefaultDevice` and a Mac has one GPU, so there is no
//! ordinal to parse; and design §6 puts no capture on this plane at all
//! (*"no record.rs: dispatch is encode-only, so `EagerSink` per fire IS
//! encoding"*), so there is no mode to choose. What is left is taking the
//! document anyway, because a seam that refused to be handed one would be
//! entitled to an opinion about which documents concern it.

use crate::api::{ContractFor, DeviceBoot, Metal};

/// Open the system's default Metal device from a boot document.
///
/// The contract lookup is a PARAMETER and not a thing this crate could find:
/// how a checkpoint's tensors become a plan's params is the model's
/// declaration, resolved by the party that links the catalog. See
/// [`ContractFor`], and [`crate::api`]'s header for the diagram.
///
/// # Errors
///
/// A boot document that is not UTF-8 or not TOML, as a sentence. `String`
/// rather than [`Fault`](crate::Fault) because this is the same seam
/// [`ContractFor`] crosses and it is spelled the same way in both directions:
/// the caller is the runtime, whose errors are `anyhow`, and neither side
/// should have to name the other's error crate to open a device.
///
/// Binding the device itself happens at [`Engine::load`](engine::Engine::load),
/// not here: `Shell::load` is one call that binds, bakes and lands, and there
/// is nothing to bind before a plan says what to bake.
pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Metal, String> {
    // Parsing is what makes a malformed boot file fail HERE, at the door,
    // rather than somewhere later that has nothing to do with it.
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the metal boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the metal boot config is not TOML: {error}"))?;
    tuning(&doc);
    Ok(Metal::new(DeviceBoot::default(), contract_for))
}

/// `[metal.tuning]`: the kernel-selection crossovers, swept.
///
/// **THIS TABLE IS THE ONLY WAY TO MOVE ONE, AND THAT IS THE POINT** (art. 9).
/// The reference driver gave every constant a `PIE_METAL_*` environment
/// override, for a reason that survives — measuring a crossover means running
/// the same binary twice with different answers, and a rebuild between arms is
/// a different binary — and a mechanism that does not: a shell here reads no
/// environment. The sweep property is preserved by the boot document, which
/// the runtime already hands over per fire-group and which nothing has to be
/// recompiled to change.
///
/// A key this table does not name is not defaulted here; it stays whatever the
/// device's own measured row said. And a key it names to ZERO means zero —
/// `moe_batch_min_per_expert = 0` is the meaningful setting "batch at any
/// width", and it is the only way to reach the routed GEMM below its
/// crossover, which is how a wrong-answer bug in that kernel gets bisected. A
/// reader that folded that in with "absent" cost the reference driver two
/// false conclusions that were both written down as fact.
///
/// **An unreadable key is not a refusal.** A boot document is shared across
/// roles and this table is advisory: a value of the wrong TYPE is dropped and
/// the measured constant stands, which is the same answer as not having
/// written the key. What a malformed document does refuse is the parse above.
fn tuning(doc: &toml::Table) {
    let Some(table) = doc
        .get("metal")
        .and_then(toml::Value::as_table)
        .and_then(|metal| metal.get("tuning"))
        .and_then(toml::Value::as_table)
    else {
        return;
    };
    let int = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
    };
    let flag = |key: &str| table.get(key).and_then(toml::Value::as_bool);

    // The device row this table lays over. The shell that binds the device is
    // the one that should state this — `MTLGPUFamilyApple<N>` probed
    // NEWEST-FIRST, since the families are cumulative — and until it does, the
    // document can say it, which is also how a machine's constants are tried
    // against another machine's family without owning one.
    let described = kernels_metal::DeviceInfo {
        apple_family: int("apple_family").unwrap_or_default(),
        gpu_core_count: int("gpu_core_count").unwrap_or_default(),
    };
    if described != kernels_metal::DeviceInfo::default() {
        kernels_metal::tuning::describe(described);
    }

    kernels_metal::tuning::override_with(kernels_metal::tuning::Overrides {
        qmm_min_batch: int("qmm_min_batch"),
        qmm_min_batch_moe: int("qmm_min_batch_moe"),
        qmm_min_batch_emulated: int("qmm_min_batch_emulated"),
        qmm_bn_crossover_tg: int("qmm_bn_crossover_tg"),
        moe_tile_mid_per: int("moe_tile_mid_per"),
        moe_tile_wide_per: int("moe_tile_wide_per"),
        fp16_qmm: flag("fp16_qmm"),
        sdpa_tile_min_rows_per_request: int("sdpa_tile_min_rows_per_request"),
        sdpa_mma: flag("sdpa_mma"),
        gdn_scan_lanes: int("gdn_scan_lanes"),
        gdn_scan_rows: int("gdn_scan_rows"),
        moe_batch_min_per_expert: int("moe_batch_min_per_expert"),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A contract lookup that is never called: every test here fails or
    /// succeeds at the DOOR, and the door does not load a model.
    fn nothing(
        _trace: &model_ir::Trace,
        _path: &std::path::Path,
    ) -> Result<checkpoint::contract::ModelContract, String> {
        Err("this door never loads".to_string())
    }

    #[test]
    fn a_boot_document_that_says_nothing_about_this_engine_still_opens() {
        // The ordinary case: a document about some other role, or an empty
        // one. There is no key this seam requires.
        assert!(open(b"", nothing).is_ok());
        assert!(open(b"[model]\nid = \"qwen35-d0.8b\"\n", nothing).is_ok());
    }

    #[test]
    fn a_boot_document_that_is_not_toml_is_refused_at_the_door() {
        assert!(open(b"this is not = = toml", nothing).is_err());
    }

    #[test]
    fn a_tuning_table_is_advisory_and_never_a_refusal() {
        // Read, and — since this table is swept rather than declared — a key
        // whose value the reader cannot use leaves the measured constant
        // standing instead of failing a boot that has nothing else wrong
        // with it.
        assert!(open(b"[metal.tuning]\nsdpa_mma = false\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nqmm_min_batch = \"eight\"\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nqmm_min_batch = -3\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nno_such_knob = 1\n", nothing).is_ok());
    }
}
