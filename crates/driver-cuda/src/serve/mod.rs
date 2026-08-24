//! The shell's door: the verbs the engine calls, as methods on
//! [`state::Shell`]. This file is the entry points and the panic boundary
//! every one crosses (`guard`); the bodies live in the sibling modules
//! [`state`], [`load`], [`launch`] (the forward path `.wiki/driver/graph.md`
//! covers), [`encode`], and [`transfer`].

// Error paths print to stderr; a driver has no tracing subscriber to rely on.
#![allow(clippy::print_stderr)]
use driver_api::completion::CompletionTarget;
use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT,
};

pub(crate) mod encode;
pub(crate) mod load;
pub(crate) mod state;
pub(crate) mod transfer;

pub use crate::fire::launch::fire_class_of;
pub use state::Shell;

/// The GQA group sizes this build's decode instantiates.
///
/// FlashInfer's decode reports anything else by throwing, and a throw crossing
/// the C ABI is undefined behaviour, so `load_model` checks at the door rather
/// than discovering it at launch. `Deployment::servable_by` takes this as an
/// argument: `model` states the shape, the driver states what it instantiated.
pub const DECODE_GQA_GROUPS: &[u32] = &[1, 2, 3, 4, 8];

use crate::fire::launch::launch_impl;
use load::{adopt_and_compile, load_impl};
use state::{ChannelState, InstanceEntry, ProgramEntry, channel_dtype};

/// Run a driver entry point, turning a panic into a status rather than the
/// caller's problem, so the other requests in flight get to finish.
///
/// A caught panic leaves the shell's invariants in an unknown state, so it is
/// reported loudly and the driver should be considered untrustworthy afterward.
pub(crate) fn guard<T>(what: &str, on_panic: T, body: impl FnOnce() -> T) -> T {
    // `panic = "abort"` breaks this: with no unwinding there is nothing to
    // catch, so every entry point becomes the bare SIGABRT this avoids.
    #[cfg(panic = "abort")]
    compile_error!(
        "driver-cuda's C ABI seam converts a panic into a status code, which \
         requires unwinding; `panic = \"abort\"` makes `guard` dead code and \
         takes the process down with every other request it was serving"
    );
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let why = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_owned());
            eprintln!(
                "[driver-cuda] {what}: PANICKED: {why}. The request is failed \
                 rather than the process; this driver's state is no longer \
                 trustworthy and it should be recreated."
            );
            on_panic
        }
    }
}

/// Settle a control op: publish the terminal outcome, THEN notify.
///
/// # The two halves, and what each omission costs
///
/// A control completion -- `copy_kv`, `copy_state`, `resize_pool`, `encode` --
/// carries a TERMINAL CELL, unlike a frame launch, whose members each answer
/// with their own. The engine resolves such an op by reading that cell after
/// the wait slot publishes, so BOTH writes have to happen and in this order:
///
/// * Notify without publishing, and the engine sees a slot that moved with a
///   cell still `Pending` and fails the op with `driver callback published
///   before terminal outcome settled`. That is what `prefix-tree-kv-cache`
///   hit on a 4090 -- the copy itself had already run and synchronized
///   correctly, so the KV pages were right and the answer was an error
///   anyway.
/// * Publish without notifying, or neither, and nothing wakes: the op hangs
///   forever. `engine/src/driver/backend.rs`'s `settle_control` records a fork
///   that hung a real `pie run` for 850 seconds this way, with the scheduler's
///   watchdog naming it `in_flight_control: KV copy ... settled=false`.
///
/// This driver had one of each. `copy_kv` and `resize_pool` notified without
/// publishing; `copy_state` bound its target to `_completion` and did neither,
/// which is why nothing in the tree that copies recurrent state on CUDA had
/// ever returned. The host-side seams share
/// `engine::driver::backend::settle_control` for exactly this; the
/// asynchronous drivers cannot, because they settle from wherever the work
/// finishes rather than before the call returns, so they get this instead.
///
/// The release fence is the caller's: every site below already places one
/// after its stream synchronize, which is what makes the device writes
/// visible before either of these two.
pub(crate) fn settle_control(
    broker: &driver_api::CompletionBroker,
    completion: driver_api::completion::CompletionTarget,
) {
    if !completion.terminal_cell.is_null() {
        // SAFETY: the broker owns this cell for the life of the completion the
        // engine minted it with, and `publish` is a release store into an
        // `AtomicU32` that the engine only ever reads.
        unsafe {
            (*completion.terminal_cell).publish(driver_api::PIE_TERMINAL_OUTCOME_SUCCESS);
        }
    }
    broker.notify(completion.wait_id, completion.target_epoch);
}

impl Shell {
    /// What this device is, as it answered at create. Parsed once, here, from
    /// the JSON this crate authors.
    #[must_use]
    pub fn device_facts(&self) -> &driver_api::DeviceFacts {
        &self.facts
    }

    /// Open the driver: the boot bytes and the broker it finishes work through.
    ///
    /// # Errors
    ///
    /// A boot config this driver refuses, or a device that would not open.
    /// The reason is on stderr; the status is which class it was.
    pub fn open(config_bytes: &[u8], broker: driver_api::CompletionBroker) -> Result<Self, i32> {
        guard("create", Err(PIE_STATUS_DRIVER_ERROR), || {
            load::create_impl(config_bytes, broker)
        })
    }

    /// Device bytes this driver holds — a diagnostic, not one of the verbs.
    ///
    /// A claim about the shell, not the device: a process sharing a GPU cannot
    /// tell a leak from a second consumer's allocation. Zero before the fire
    /// allocator exists, which reads as "holds nothing" and is true.
    #[must_use]
    pub fn live_device_bytes(&self) -> usize {
        self.fire_alloc
            .as_ref()
            .map_or(0, crate::device::Allocator::live_bytes)
    }

    /// Whether a fire lane is built — a diagnostic, not one of the verbs.
    ///
    /// `true` after any `load_model` that returned `Ok`, because a load that
    /// could not build a lane REFUSES rather than serving through something
    /// else. It is published anyway, from outside the crate, so a test can
    /// assert the tautology without reaching into `Shell`'s `pub(crate)`
    /// fields — a `Shell` that answered `Ok` with nothing to fire would be
    /// the one bug this cannot otherwise be seen.
    #[must_use]
    pub fn baker_is_armed(&self) -> bool {
        self.baker.is_some()
    }

    /// Load the model: one parse of the snapshot through the Rust loader and
    /// every bf16 weight resident on the device, with the llama-like fused
    /// trace names built beside the checkpoint names. Quantized encodings are
    /// refused, not mis-loaded; the memory plan and KV materialization land
    /// with `launch`.
    ///
    /// # Errors
    ///
    /// A checkpoint that will not parse, or one this device cannot run.
    pub fn load_model(
        &mut self,
        desc: &driver_api::ModelLoadDesc,
    ) -> Result<driver_api::DriverCapabilities, i32> {
        guard(
            "load_model",
            Err(PIE_STATUS_DRIVER_ERROR),
            move || -> Result<driver_api::DriverCapabilities, i32> {
                if desc.snapshot_dir.as_os_str().is_empty() {
                    eprintln!("[driver-cuda] load_model: snapshot_dir is empty");
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                // Refused, not ignored: this driver does not scope loads, and
                // silently loading the whole checkpoint for a scoped `Encode`
                // request would OOM rather than say so.
                match desc.component {
                    driver_api::ModelComponent::Full => {}
                    scope => {
                        eprintln!(
                            "[driver-cuda] load_model: component {scope:?} is not \
                             implemented; this driver loads the whole checkpoint \
                             (ModelComponent::Full) or nothing"
                        );
                        return Err(PIE_STATUS_INVALID_ARGUMENT);
                    }
                }
                // Same seam, same silence: `runtime_quant` is unimplemented, so
                // a config asking for fp8 would be accepted and silently run
                // bf16. Empty means "none"; the rest is refused.
                if !desc.runtime_quant.is_empty() {
                    eprintln!(
                        "[driver-cuda] load_model: runtime_quant '{}' is not \
                         implemented; this driver runs the checkpoint's own \
                         weights (runtime_quant = \"\")",
                        desc.runtime_quant
                    );
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                // The explicit `Mxfp4MoeRequest` variants pin a lowering and
                // fail if the device cannot provide it: this driver has only
                // routed decode, so `NativeGemm` and `EagerBf16` are refused
                // while `Auto` and `RoutedDecode` are served.
                match desc.mxfp4_moe {
                    driver_api::Mxfp4MoeRequest::Auto
                    | driver_api::Mxfp4MoeRequest::RoutedDecode => {}
                    pinned => {
                        eprintln!(
                            "[driver-cuda] load_model: mxfp4_moe {pinned:?} cannot be \
                             provided by this device; it runs routed decode \
                             (mxfp4_moe = \"auto\" or \"routed_dequant\")"
                        );
                        return Err(PIE_STATUS_INVALID_ARGUMENT);
                    }
                }
                load_impl(self, &desc.snapshot_dir)?;
                let model = self.model.as_ref().expect("load_impl stored the model");
                serde_json::from_slice(&model.load_caps).map_err(|error| {
                    eprintln!("[driver-cuda] load_model: capabilities JSON: {error}");
                    PIE_STATUS_DRIVER_ERROR
                })
            },
        )
    }

    /// Register a program: adopt its launch package, compile its generated
    /// regions, and answer an id. The program hash is the dedup key —
    /// re-registering answers the existing id without recompiling.
    ///
    /// An empty launch package is a forward-only deployment, not an error: an
    /// id is issued and nothing is adopted. An unexecutable plan is recorded
    /// with its reason and refused at the launch that needs it, not here. A
    /// compile needs a device, so a shell with no model loaded defers it to
    /// the first launch rather than compiling for a made-up architecture.
    ///
    /// # Errors
    ///
    /// A package that will not adopt, or a region NVRTC rejects.
    pub fn register_program(
        &mut self,
        program: &driver_api::ProgramRegistration,
    ) -> Result<u64, i32> {
        guard(
            "register_program",
            Err(PIE_STATUS_DRIVER_ERROR),
            move || {
                let state = self;
                let desc = program;
                if let Some(id) = state
                    .programs
                    .iter()
                    .find(|(_, p)| p.program_hash == desc.program_hash)
                    .map(|(&id, _)| id)
                {
                    return Ok(id);
                }

                let package = desc.launch.clone();
                let kernels = desc.emitted_kernels.as_slice();

                let id = state.next_id;
                state.next_id += 1;

                // No stages is the absence of a program, not a malformed one: a
                // forward-only deployment. `adopt_launch_package` rejects an
                // empty stage list, so the "not an error" judgement is made here.
                if !package.stages.is_empty() {
                    adopt_and_compile(state, id, desc, package, kernels)?;
                }

                state.programs.insert(
                    id,
                    ProgramEntry {
                        program_hash: desc.program_hash,
                        emitter_version: desc.emitter_version,
                    },
                );
                Ok(id)
            },
        )
    }

    /// Register a channel endpoint: a pinned host mirror of `(capacity + 1)`
    /// wire cells and four pinned control words (head 0, tail 1, poison 2,
    /// closed 3), both zeroed. A bool bit-packs; everything else is four bytes
    /// per element; `capacity + 1 ≤ 64`. Device-side rings ride with launch.
    ///
    /// # Errors
    ///
    /// A shape this shell cannot place, or pinned memory it could not get.
    pub fn register_channel(
        &mut self,
        plan: &driver_api::ChannelRegistrationPlan,
    ) -> Result<ChannelBinding, i32> {
        guard(
            "register_channel",
            Err(PIE_STATUS_DRIVER_ERROR),
            move || {
                use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};

                const MAX_RING: u64 = 64;
                let state = self;
                let desc = plan;
                if state.channels.contains_key(&desc.channel_id)
                    || desc.dtype > driver_api::local::PIE_CHANNEL_DTYPE_ACT
                {
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                let mut numel: u64 = 1;
                for &d in &desc.shape {
                    let Some(next) = numel.checked_mul(u64::from(d)) else {
                        return Err(PIE_STATUS_INVALID_ARGUMENT);
                    };
                    numel = next;
                }
                let wire_bytes: u64 = if desc.dtype == driver_api::local::PIE_CHANNEL_DTYPE_BOOL {
                    numel.div_ceil(8)
                } else {
                    match numel.checked_mul(4) {
                        Some(b) => b,
                        None => return Err(PIE_STATUS_INVALID_ARGUMENT),
                    }
                };
                let ring = u64::from(desc.capacity) + 1;
                if wire_bytes == 0 || ring > MAX_RING {
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                let Some(mirror_bytes) = wire_bytes.checked_mul(ring) else {
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                };
                let Ok(mirror_bytes) = usize::try_from(mirror_bytes) else {
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                };

                let mut ops = LiveStagingOps;
                let Some(mirror) = ops.malloc_host(mirror_bytes) else {
                    return Err(PIE_STATUS_EXHAUSTED);
                };
                let word_bytes = 4 * std::mem::size_of::<u64>();
                let Some(words) = ops.malloc_host(word_bytes) else {
                    ops.free_host(mirror);
                    return Err(PIE_STATUS_EXHAUSTED);
                };
                unsafe {
                    std::ptr::write_bytes(mirror.cast::<u8>(), 0, mirror_bytes);
                    std::ptr::write_bytes(words.cast::<u8>(), 0, word_bytes);
                }
                state.channels.insert(
                    desc.channel_id,
                    ChannelState {
                        mirror,
                        words,
                        mirror_bytes,
                        cell_bytes: usize::try_from(wire_bytes).unwrap_or(usize::MAX),
                        ring: u32::try_from(ring).expect("ring fits u32"),
                        host_role: desc.host_role,
                        numel: usize::try_from(numel).unwrap_or(usize::MAX),
                        dtype: channel_dtype(desc.dtype),
                        // Recorded at registration, refused at bind: registering
                        // an extern channel is harmless; attaching two programs
                        // to it is what `bind_instance` turns away.
                        extern_dir: desc.extern_dir,
                    },
                );
                Ok(ChannelBinding {
                    channel_id: desc.channel_id,
                    mirror_base: mirror as u64,
                    word_base: words as u64,
                    mirror_bytes: mirror_bytes as u64,
                    word_bytes: word_bytes as u64,
                    cell_bytes: u32::try_from(wire_bytes).unwrap_or(u32::MAX),
                    capacity: desc.capacity,
                    head_word_index: 0,
                    tail_word_index: 1,
                    poison_word_index: 2,
                    closed_word_index: 3,
                })
            },
        )
    }

    /// Bind an instance to a registered program: the id lifecycle, honoring a
    /// nonzero `requested_instance_id` and echoing the geometry class.
    ///
    /// # Errors
    ///
    /// No such program, an identity already bound, or an extern channel.
    pub fn bind_instance(
        &mut self,
        plan: &driver_api::InstanceBindingPlan,
    ) -> Result<InstanceBinding, i32> {
        guard("bind_instance", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            let desc = plan;
            if !state.programs.contains_key(&desc.program_id) {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            // An extern channel is not servable here: it needs one ring shared
            // between exporter and importer, and this driver builds a ring per
            // session. Binding anyway gives the importer a private ring no one
            // fills — a block forever, or a zeroed cell read as a value.
            if let Some(&cid) = desc
                .channel_ids
                .iter()
                .find(|c| state.channels.get(c).is_some_and(ChannelState::is_extern))
            {
                eprintln!(
                    "[driver-cuda] bind_instance: channel {cid} is declared extern. \
                 The RING is shared now — `program::channel::Rings` registers a \
                 channel once and every instance that names it holds the same \
                 slot — but nothing here reads the import/export direction, so \
                 which program may publish and which may consume is unchecked. \
                 Refusing rather than guessing it."
                );
                return Err(driver_api::PIE_STATUS_UNSUPPORTED);
            }
            let id = if desc.requested_instance_id != 0 {
                desc.requested_instance_id
            } else {
                let id = state.next_id;
                state.next_id += 1;
                id
            };
            if state.instances.contains_key(&id) {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            // The entry and binding carry the `GeometryClass` discriminant,
            // which is what the fire path indexes with.
            let geometry_class = desc.geometry_class as u32;
            state.instances.insert(
                id,
                InstanceEntry {
                    program_id: desc.program_id,
                    geometry_class,
                    channel_ids: desc.channel_ids.clone(),
                    // Kept, not applied (see `InstanceEntry::seeds`): a cell's
                    // home is a device ring and there is no allocator here.
                    seeds: desc
                        .seed_values
                        .iter()
                        .map(|value| (value.channel, value.bytes.clone()))
                        .collect(),
                },
            );
            Ok(InstanceBinding {
                instance_id: id,
                geometry_class,
                reserved0: 0,
            })
        })
    }

    /// Launch a frame: the executor's fire assembly, promoted from the smokes
    /// into the shell.
    ///
    /// Today: single-sub-batch frames fired through the loaded checkpoint's
    /// own `Program`; the KV pools are driver-owned and persist across
    /// launches, and every member's terminal cell is published (release)
    /// before the runtime is notified. A fire whose class no built lane
    /// serves refuses with `UNSUPPORTED`, by name — so does a frame that
    /// supplies its own attention mask, which no statement reads.
    ///
    /// # Errors
    ///
    /// A malformed frame, or a device that failed after accepting it.
    pub fn launch(
        &mut self,
        frame: &driver_api::submission::FrameSubmission,
        completion: CompletionTarget,
    ) -> Result<(), i32> {
        guard("launch", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            // `FrameSubmission::validate` carries every structural rule: roster
            // bounds, distinctness of members and cells (within a step and
            // across them), one cell per member, the CSRs, the recurrent-state
            // parallelism and flag bits, the ticket cover.
            if let Err(why) = frame.validate() {
                eprintln!("[driver-cuda] launch: {why}");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            // An `Ok` here means enqueued, not done: publishing the terminal
            // cells and notifying moved into a stream-ordered host callback
            // (`retire_fire`) so enqueuing fire n+1 need not wait for fire n to
            // retire. An error still returns synchronously — a fire that could
            // not be built owes nothing.
            launch_impl(state, frame, completion)
        })
    }

    /// Close an instance, idempotently: closing what is not open is not an
    /// error.
    ///
    /// # Errors
    ///
    /// Never today.
    pub fn close_instance(&mut self, instance_id: u64) -> Result<(), i32> {
        guard("close_instance", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            state.instances.remove(&instance_id);
            // The rings go with it: holding them past close is a device
            // allocation nothing can reach.
            state.ptir_sessions.remove(&instance_id);
            Ok(())
        })
    }

    /// Close a channel — idempotently, freeing its pinned endpoint.
    ///
    /// # Errors
    ///
    /// Never today.
    pub fn close_channel(&mut self, channel_id: u64) -> Result<(), i32> {
        guard("close_channel", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            if let Some(ch) = state.channels.remove(&channel_id) {
                // Unregistered now, freed later (see `InFlight::closed_channels`):
                // a queued fire's debt may publish into this mirror from a stream
                // callback, so it cannot go back to the allocator until every
                // fire that could name it has retired. No fire in flight frees now.
                match state.in_flight.back_mut() {
                    Some(last) => last.closed_channels.push(ch),
                    None => ch.free(),
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::state::instance_ring_shapes;
    use super::{ChannelState, InstanceEntry};

    /// A registered channel becomes the ring shape the device wants; the bool
    /// case cannot be derived any other way. `cell_bytes` is wire bytes, and
    /// for bools that is `numel.div_ceil(8)` — one and eight lanes are both one
    /// byte, so a ring sized from the width is eight times too small. That is
    /// why `numel` and `dtype` are stored rather than recomputed.
    #[test]
    fn a_bool_channels_ring_shape_cannot_be_derived_from_its_wire_width() {
        use driver::tensor_ir::DType;

        let chan = |numel: usize, dtype: DType, capacity: u32| ChannelState {
            mirror: std::ptr::null_mut(),
            words: std::ptr::null_mut(),
            mirror_bytes: 0,
            cell_bytes: crate::program::channel::wire_cell_bytes(dtype, numel),
            extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
            ring: capacity + 1,
            host_role: 0,
            numel,
            dtype,
        };

        let one = chan(1, DType::Bool, 1);
        let eight = chan(8, DType::Bool, 1);
        assert_eq!(one.cell_bytes, eight.cell_bytes, "one wire byte either way");
        assert_eq!(one.shape().numel, 1);
        assert_eq!(
            eight.shape().numel,
            8,
            "the shape knows what the width cannot"
        );
        assert_eq!(eight.shape().cell_bytes(), 8, "native is a byte per lane");

        // And the capacity survives the `ring = capacity + 1` round trip,
        // which is the one place the two vocabularies disagree.
        assert_eq!(chan(4, DType::F32, 7).shape().capacity, 7);
    }

    /// An instance's ring shapes come back in the order the program indexes
    /// them; a channel it names but this shell lacks refuses the whole list
    /// rather than shortening it, since skipping would renumber every channel
    /// after the gap.
    #[test]
    fn a_channel_an_instance_names_and_this_shell_lacks_refuses_the_whole_list() {
        use driver::tensor_ir::DType;

        let mut channels = std::collections::BTreeMap::new();
        for (id, numel) in [(10u64, 4usize), (20, 9)] {
            channels.insert(
                id,
                ChannelState {
                    mirror: std::ptr::null_mut(),
                    words: std::ptr::null_mut(),
                    mirror_bytes: 0,
                    cell_bytes: numel * 4,
                    extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
                    ring: 2,
                    host_role: 0,
                    numel,
                    dtype: DType::F32,
                },
            );
        }
        let inst = |ids: Vec<u64>| InstanceEntry {
            program_id: 1,
            geometry_class: 0,
            channel_ids: ids,
            seeds: Vec::new(),
        };

        // The list is the instance's order, not the map's.
        let shapes = instance_ring_shapes(&inst(vec![20, 10]), &channels).expect("all present");
        assert_eq!(shapes.len(), 2);
        assert_eq!(
            shapes[0].numel, 9,
            "channel 20 is index 0 because it is listed first"
        );
        assert_eq!(shapes[1].numel, 4);

        assert!(
            instance_ring_shapes(&inst(vec![10, 99]), &channels).is_none(),
            "an unknown channel refuses rather than shortening the list"
        );
    }
    /// The boot-TOML extraction, isolated: the exact chain `create` runs.
    #[test]
    fn the_boot_model_id_extracts() {
        let boot = "[model]\nid = \"qwen35-d0.8b-bf16-kv-bf16\"\n";
        let v = boot.parse::<toml::Table>().expect("parses");
        let id = v
            .get("model")
            .and_then(|m| m.get("id"))
            .and_then(|d| d.as_str())
            .expect("extracts");
        assert_eq!(id, "qwen35-d0.8b-bf16-kv-bf16");
    }
}
