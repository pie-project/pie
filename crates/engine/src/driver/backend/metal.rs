//! The seam to `driver-metal` — a door onto a shell that is not built yet.
//!
//! # `palo B-metal`: the shell went with the string-plan stack
//!
//! What stood here was a 440-line adapter over `driver_metal::serve::Shell`:
//! `load_model` against a ported checkpoint stager, `launch` against a
//! `FrameSubmission`, `copy_kv`/`copy_state`/`resize_pool` against the ported
//! KV and recurrent pools, and a `CompletionBroker` per driver. That `Shell`
//! no longer exists. `driver-metal`'s own header says why:
//!
//! > This crate took `driver-metal`'s name when the string-plan shell it
//! > re-imagines was deleted with the rest of the old stack (design, porting
//! > order step 6); serving plumbing (bind, device, serve) rejoins it as the
//! > fabric is rewired.
//!
//! The crate today is the DISPATCH layer only — a [`Run`](driver_metal::Run)
//! that resolves plan ids to device handles and answers the fifteen
//! `Dispatch*` traits — and it deliberately names no Metal API, which is what
//! lets it build on any OS. What is missing is everything around a fire: a
//! device to bind, a checkpoint to land, pools to reserve, and a command
//! buffer to encode onto. `driver-cuda/serve.rs` is the shape it will take;
//! the CUDA one is porting-order step 4 and this is a later milestone.
//!
//! So this file is the door, and every verb behind it is
//! [`DriverError::Unsupported`]. The one thing it does answer honestly is
//! [`Driver::kind`] — a caller that selected `driver-metal` on a Mac gets a
//! registered driver that refuses by name, rather than a build error or a
//! plausible wrong answer.

use anyhow::Result;

use driver_api::error::{DriverError, Result as DriverResult};
use driver_api::fire::{FireSubmission, FireTicket};
use driver_api::load::{LoadRequest, Loaded};
use driver_api::Driver;

/// The Metal seam, with no shell behind it.
///
/// One field, and it is the boot config's one key. The `shell` and `broker`
/// pair this replaced is what a driver with a device holds; there is neither
/// until the shell is ported.
pub struct MetalDriver {
    /// `[model] id`, if the boot document stated one — the operator's answer
    /// to "which model is this", carried so the refusals can name it.
    model_id: Option<String>,
}

impl MetalDriver {
    /// Read the boot document.
    ///
    /// # Errors
    ///
    /// None today: there is no device to fail to open. It stays a `Result`
    /// because binding one is the first thing the ported shell will do here.
    pub fn create(config_bytes: &[u8]) -> Result<Self> {
        // THE ONE KEY THIS DRIVER WANTS, parsed HERE. A driver that read the
        // boot TOML would be the second thing entitled to an opinion about
        // its shape; the seam is engine code, so the seam reads it. A
        // document that does not parse states no id, which is the ordinary
        // case and not an error.
        let model_id = std::str::from_utf8(config_bytes)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|doc| {
                doc.get("model")?
                    .as_table()?
                    .get("id")?
                    .as_str()
                    .map(str::to_owned)
            });
        Ok(Self { model_id })
    }

    /// The refusal every verb answers.
    fn refuse(&self, verb: &'static str) -> DriverError {
        tracing::warn!(
            model_id = ?self.model_id,
            verb,
            "the metal seam has no shell: `driver-metal` is the dispatch layer \
             only, and its serving half is palo B-metal"
        );
        DriverError::unsupported("metal", verb)
    }
}

impl Driver for MetalDriver {
    fn kind(&self) -> &'static str {
        "metal"
    }

    // `device_facts` answers `None`, which the trait's default already does
    // and which is the truth: nothing here has bound a device. It is what
    // makes `register_driver_backend` stamp `MemoryDomain::HostPinned` on the
    // spec — no page of this driver's lives on a device, because it has none.

    fn load(&mut self, request: LoadRequest) -> DriverResult<Loaded> {
        // palo B-metal: the shell's `load` is `driver-cuda/serve.rs`'s in
        // call order — bind, `compile(plan, budgets, profile)`, read the kv
        // spaces off the plan, land the checkpoint through
        // `model_loader::plan::compile` at `BackendKind::Metal`, reserve the
        // arena and pools, find the `out` seam. Every one of those pieces
        // except the residency is backend-neutral and already written.
        let _ = request;
        Err(self.refuse("load"))
    }

    fn fire(&mut self, submission: &FireSubmission) -> DriverResult<FireTicket> {
        // palo B-metal: `driver::fire::walk` is generic over `Dispatch` and
        // `Sink` (decision 11) and `driver_metal::Run` already implements the
        // first. What a fire needs beyond it is the staging — the fire's own
        // vectors onto a buffer — and a command buffer to encode into.
        let _ = submission;
        Err(self.refuse("fire"))
    }
}
