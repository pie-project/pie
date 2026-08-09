//! The shell's door: the verbs the engine calls, as methods.
//!
//! # What these stopped being
//!
//! Thirteen `pie_cuda_*` free functions, each taking a `*mut PieDriver` —
//! an opaque `c_void` — and casting it back to a `Shell` on entry. The
//! engine reached them through an `unsafe extern "C"` block that the linker
//! resolved against this same crate, in this same workspace, in Rust.
//!
//! The declaration existed because the driver on the far side used to be
//! C++, and it outlived the C++ by the length of the cutover. What it cost
//! while it lived: no type checking across the call, no lifetimes, a
//! `*mut PieDriver` where a `&mut Shell` would do, seven `#[repr(C)]`
//! descriptors built by the engine purely to be taken apart again here, and
//! a null check plus a validator call on every entry — 48 `unsafe` in this
//! module's subtree against the Metal shell's 4, for the same work.
//!
//! They are methods on [`state::Shell`] now. The receiver is the handle, the
//! descriptors are the owned `driver_api` types the engine already had, and
//! the `i32` status they still answer is this crate's own convention rather
//! than an ABI — `engine`'s seam turns it into the verb's name and a message.
//!
//! # What is in here, and what is next door
//!
//! This file is the DOOR: the entry points and the panic boundary every one
//! of them crosses (`guard`). The bodies live beside it, one module per
//! thing the driver does:
//!
//! * [`state`] — the shell's nouns. A leaf; everything else calls in.
//! * [`load`] — create, destroy, and the once-per-model work.
//! * [`launch`] — the forward path. `.wiki/driver/graph.md` is about it.
//! * [`encode`] — the multimodal towers, run outside a fire.
//! * [`transfer`] — KV/state copies and pool resizes.
//!
//! It was one 6,364-line file, of which the doors were 18%. The split is
//! by what a caller is asking for, which is also how the work divides:
//! `launch` is `graph.md`'s subject and `state` is nobody's, so they can
//! be edited without meeting.

// Error paths print to stderr with the C++ shell's own prefix — that IS
// the behaviour being replaced (`abi.cpp` writes `[pie-driver-cuda]` to
// cerr), and a driver has no tracing subscriber to rely on.
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

/// The GQA group sizes THIS BUILD's decode instantiates.
///
/// FlashInfer's decode reports anything else by THROWING, and a throw
/// crossing a C ABI is undefined behaviour, which is why `load_model` asks at
/// the door rather than discovering it at launch.
///
/// **It is stated here and not in `model`.** The set is a fact about what
/// this crate's kernels were built with, and
/// [`Deployment::servable_by`](model::deployment::Deployment::servable_by)
/// takes it as an ARGUMENT for exactly that reason — its own doc says
/// "`model` states the shape, the driver states what it instantiated, and
/// neither one has to know the other's answer."
///
/// The const nevertheless lived in `model::shared::llama_like::project`,
/// so this crate reached through a FAMILY's namespace to read a fact about
/// its own build. Its doc there already argued it "sat inside the llama
/// lineage's derivation as though it were a property of that lineage"; this
/// is that argument finished. The live proof it belongs to no lineage is
/// Qwen3.6-27B — 24 query heads over 4 kv heads is a group of six, reaching
/// the same dispatch from a different generation.
pub const DECODE_GQA_GROUPS: &[u32] = &[1, 2, 3, 4, 8];

use crate::fire::launch::launch_impl;
use load::{adopt_and_compile, load_impl};
use state::{ChannelState, InstanceEntry, ProgramEntry, Shell as ShellState, channel_dtype};

/// Run a driver entry point, turning a panic into a status rather than into
/// the caller's problem.
///
/// # Why this is here now and was not before
///
/// The thirteen entry points used to be `extern "C"`, where a Rust panic is
/// not a recoverable event: unwinding out of one was undefined behaviour and
/// then, from Rust 1.81, an abort. So the C++ shell's `try/catch` on every
/// export had no counterpart on this side and could not have one -- catching
/// would have to happen INSIDE the frame, which is what this does.
///
/// They are plain Rust functions now, so a panic unwinds normally into the
/// engine. That makes catching possible; this makes it the contract. Every one
/// of these already answers a status code, and "the driver hit a bug" is a
/// status, not a reason to take the process down with the other requests it
/// was serving.
///
/// **This does not make a panic safe to ignore.** A caught panic means the
/// shell's invariants are in an unknown state -- a half-built plan, a stream
/// with work queued, an arena mid-resize -- so it is reported loudly and the
/// driver should be considered untrustworthy afterwards. What it buys is that
/// the OTHER requests in flight get to finish, and that the operator gets a
/// message naming the entry point instead of a bare SIGABRT.
pub(crate) fn guard<T>(what: &str, on_panic: T, body: impl FnOnce() -> T) -> T {
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

impl Shell {
    /// What this device is, as it answered at create.
    ///
    /// Parsed once, here, from the JSON this crate authors. The engine used
    /// to parse it itself out of a `{ptr, len}` handed back through a
    /// `PieDriverCaps` out-parameter, which is two readers of one document.
    #[must_use]
    pub fn device_facts(&self) -> &driver_api::DeviceFacts {
        &self.facts
    }

    /// Open the driver.
    ///
    /// Takes the boot bytes and the broker it will finish work through. It
    /// used to take a `*const PieDriverCreateDesc` carrying an `abi_version`,
    /// a `{ptr, len}` view of these same bytes, and a `{notify, ctx}` C
    /// callback pair — and it answered a `*mut PieDriver` plus a JSON blob
    /// through an out-parameter, because a C function that fails cannot
    /// return two things.
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

    /// Device bytes this driver holds — a DIAGNOSTIC, not one of the verbs.
    ///
    /// What a leak test should read. `cudaMemGetInfo` answers a question about
    /// the DEVICE, and a process sharing a GPU cannot answer that for itself:
    /// a second consumer allocating during the measurement is
    /// indistinguishable from a leak, and the fifty-step soak failed exactly
    /// that way against a concurrent agent on the same card.
    ///
    /// This is a claim about the SHELL. Zero for one whose fire allocator has
    /// not been made, which reads as "holds nothing" and is true.
    ///
    /// Not on `Driver`: the engine has no reason to ask, and a fifteenth verb
    /// would have to be a contract. It took a `*mut PieDriver` and answered
    /// zero for null, which is the shape everything in this module had.
    #[must_use]
    pub fn live_device_bytes(&self) -> usize {
        self.fire_alloc
            .as_ref()
            .map_or(0, crate::device::Allocator::live_bytes)
    }

    /// Load the model: one parse of the snapshot through the Rust loader,
    /// the `pie.model/1` descriptor (embedded meta, else the boot TOML's
    /// path), and every bf16 weight resident on the device — with the
    /// llama-like fused trace names built beside the checkpoint names, so the
    /// executor's resolver asks and receives.
    ///
    /// Still awaited here: quantized encodings (refused, not mis-loaded),
    /// the memory plan, and KV materialization — those land with `launch`.
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
                // A `PathBuf` where the descriptor carried a `{ptr, len}` of
                // UTF-8 that this function turned back into one.
                if desc.snapshot_dir.as_os_str().is_empty() {
                    eprintln!("[driver-cuda] load_model: snapshot_dir is empty");
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                load_impl(self, &desc.snapshot_dir)?;
                let model = self.model.as_ref().expect("load_impl stored the model");
                // The JSON was handed back as `{ptr, len}` into shell-owned
                // memory and parsed by the engine. Parsed here instead: the
                // bytes never leave this crate, and the caller is handed the
                // type it was going to build anyway.
                serde_json::from_slice(&model.load_caps).map_err(|error| {
                    eprintln!("[driver-cuda] load_model: capabilities JSON: {error}");
                    PIE_STATUS_DRIVER_ERROR
                })
            },
        )
    }

    /// Register a program: adopt its launch package, compile its generated
    /// regions, and answer an id.
    ///
    /// The C3 hash is the dedup key — re-registering answers the existing id
    /// without recompiling — which is what makes a program that is bound a
    /// thousand times compiled once.
    ///
    /// # What a failure here means, and why it is not always one
    ///
    /// Four outcomes, and only two of them are errors:
    ///
    /// * The descriptor carries NO launch package — an empty stage list. That
    ///   is a forward-only deployment: the model runs and the logits come
    ///   back through the instance's reader channel, with no user program
    ///   around the fire. An id is issued and nothing is adopted. `OK`.
    /// * The package adopts and every generated region compiles. `OK`.
    /// * The package adopts and the plan is UNEXECUTABLE — a per-layer tap
    ///   stage, an op this driver does not implement. That is not a driver
    ///   failure and not a registration failure either: the plan is recorded
    ///   with its reason, and the refusal surfaces at the launch that needs
    ///   it, where the caller can see which fire it lost.
    /// * A region NVRTC rejects, or an emitted table with a hole in it.
    ///   `UNSUPPORTED`, and remembered: this driver carries no emitter, so a
    ///   generated region with no host source has no slower path to fall
    ///   back to.
    ///
    /// A compile needs a device — the architecture comes from the GPU that
    /// will run the code, never a guess — so a shell with no model loaded has
    /// not bound one yet and defers the compile to the first launch rather
    /// than compiling for an architecture it made up.
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
                // No validate call: `validate_program_desc` stated an
                // `abi_version` and two reserved words and nothing else, and a
                // `ProgramRegistration` has none of the three.
                let desc = program;
                if let Some(id) = state
                    .programs
                    .iter()
                    .find(|(_, p)| p.program_hash == desc.program_hash)
                    .map(|(&id, _)| id)
                {
                    return Ok(id);
                }

                // NO ADOPTION. The package and the kernel table arrive owned, so
                // the copy that used to turn `PieLaunchPackage` back into
                // `LaunchPackage` -- 1,557 lines of field-for-field mapping whose
                // whole job was to undo `engine`'s `launch_abi.rs` -- has nothing
                // left to do. `driver::adopt_launch_package` below is a different
                // thing and stays: it lowers the package into an ExecPlan.
                let package = desc.launch.clone();
                let kernels = desc.emitted_kernels.as_slice();

                let id = state.next_id;
                state.next_id += 1;

                // A package with NO STAGES is not a malformed program; it is the
                // absence of one. The engine registers such a descriptor for a
                // forward-only deployment — the model runs, the logits come back
                // through the instance's reader channel, and no user program sits
                // around the fire. `adopt_launch_package` refuses an empty stage list
                // because an ExecPlan with nothing to execute is not a plan, and it is
                // right to; the judgement that this is not an ERROR belongs here,
                // where the difference between "the host sent a broken program" and
                // "the host sent no program" is visible.
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

    /// Register a channel endpoint: the C++ registry's binding contract —
    /// a pinned host MIRROR of `(capacity + 1)` wire cells and four pinned
    /// control words (head 0, tail 1, poison 2, closed 3), both zeroed, with
    /// the wire-cell math reproduced exactly (bool bit-packs, everything
    /// else is four bytes per element; `capacity + 1 ≤ 64`). Device-side
    /// rings and fire delivery ride with the launch integration.
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
                // No `abi_version` and no `validate_channel_desc`: the plan is an
                // owned `ChannelRegistrationPlan`, so the forty `ptr/len mismatch`
                // rules that validator carried are rules about a representation
                // that is gone. What is left is what a `Vec` still cannot state.
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
                        // RECORDED AT REGISTRATION, refused at bind. Registering
                        // an extern channel is harmless — the endpoint binding
                        // below is the same either way, and the host mirror is
                        // real. What cannot be served is ATTACHING two programs
                        // to it, which is what `bind_instance` turns away.
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

    /// Bind an instance to a registered program: the id lifecycle, honoring
    /// a nonzero `requested_instance_id` and echoing the geometry class.
    /// KV-slot and adapter state ride in with the `launch` arm.
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
            // No `abi_version` and no `validate_instance_desc`: an
            // `InstanceBindingPlan` is owned, so the slice-walking that
            // validator did has nothing to walk.
            let desc = plan;
            if !state.programs.contains_key(&desc.program_id) {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            // AN EXTERN CHANNEL IS NOT SERVABLE HERE, and the reason is on
            // `ChannelState::is_extern`: it needs ONE ring shared between the
            // exporting and the importing program, and this driver builds a
            // ring per session. Binding anyway gives the importer a private
            // ring that no one ever fills — a program that blocks forever, or
            // reads a zeroed cell and treats it as a value. Refusing at the
            // attach is the only point where the driver can still say so.
            if let Some(&cid) = desc
                .channel_ids
                .iter()
                .find(|c| state.channels.get(c).is_some_and(ChannelState::is_extern))
            {
                eprintln!(
                    "[driver-cuda] bind_instance: channel {cid} is declared extern \
                 and this driver allocates one ring per session, so the two \
                 programs sharing it would not share cells or cursors. \
                 Refusing rather than binding a ring nobody fills."
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
            // The plan states the class as a `GeometryClass`; the entry and
            // the binding both carry the discriminant, which is what the
            // fire path indexes with.
            let geometry_class = desc.geometry_class as u32;
            state.instances.insert(
                id,
                InstanceEntry {
                    program_id: desc.program_id,
                    geometry_class,
                    channel_ids: desc.channel_ids.clone(),
                },
            );
            Ok(InstanceBinding {
                instance_id: id,
                geometry_class,
                reserved0: 0,
            })
        })
    }

    /// Launch a frame: the executor's fire assembly, promoted from the
    /// smokes into the shell.
    ///
    /// What runs today: SINGLE-step, single-sub-batch frames over the loaded
    /// llama-like model — the frame's own CSRs become the fire, the KV pools
    /// are driver-owned and persist across launches, write targets derive
    /// from the CSR tails, and every batch member's terminal cell is
    /// published (release) before the runtime is notified. Multi-step
    /// frames, device-geometry sub-batches and channel-delivered outputs
    /// refuse with UNSUPPORTED until their machinery lands — logits stay in
    /// the shell until channels exist to carry them out.
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
            // THE FRAME IS VALIDATED, and the comment that used to sit here said
            // it was not. That comment claimed `validate_frame_desc` was too
            // strict to adopt ("this shell serves frames today that state
            // neither"); the code under it had adopted it anyway, and
            // `entry_validation::no_validator_is_deferred` records what doing so
            // caught — fixtures whose `roster_rows` counted TOKENS, steps with no
            // terminal cell, two steps sharing one. The claim had inverted and
            // the note had not been rewritten.
            //
            // `FrameSubmission::validate` carries every rule that validator did:
            // roster bounds, the distinctness of members and of cells (within a
            // step and across them), one cell per member, the CSRs, the
            // recurrent-state parallelism and flag bits, the ticket cover.
            //
            // What it does NOT carry is the forty-odd `ptr/len mismatch` checks,
            // and those are not weakened rules -- a `Vec` cannot be a null
            // pointer with a nonzero length, so they are rules about a
            // representation that is gone.
            if let Err(why) = frame.validate() {
                eprintln!("[driver-cuda] launch: {why}");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            // THE CALL RETURNS WITH THE WORK STILL ON THE STREAM.
            //
            // Publishing the terminal cells and notifying used to happen here,
            // after `launch_impl` had synchronized — which made the driver
            // serialize the engine's `frame_dispatch_depth`, because the call
            // that would enqueue fire n+1 could not start until fire n had
            // retired on the GPU.
            //
            // Both debts moved into a stream-ordered host callback
            // (`retire_fire`), so an `Ok` here means ENQUEUED rather than DONE.
            // An error still returns synchronously: a fire that could not be
            // built owes nothing and the runtime must hear so on this thread.
            launch_impl(state, frame, completion)
        })
    }

    /// Close an instance — idempotently, the C++'s reading: closing what is
    /// not open is not an error.
    ///
    /// # Errors
    ///
    /// Never today.
    pub fn close_instance(&mut self, instance_id: u64) -> Result<(), i32> {
        guard("close_instance", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            state.instances.remove(&instance_id);
            // The rings go with it. They are the instance's, and holding them
            // past its close is a device allocation nothing can reach — the
            // channel ids that named it are gone with the entry above.
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
                // UNREGISTERED NOW, FREED LATER. See `InFlight::closed_channels`:
                // a queued fire's debt holds a copy of this state and will publish
                // into it from a stream callback, so the mirror cannot go back to
                // the allocator until every fire that could name it has retired.
                //
                // No fire in flight means nothing can be holding it, and the free
                // happens here as it always did.
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

    /// A registered channel becomes the ring shape the device wants, and
    /// the bool case is the one that could not be derived any other way.
    ///
    /// `cell_bytes` is WIRE bytes, and for bools that is
    /// `numel.div_ceil(8)` — so one lane and eight lanes are both one
    /// byte, and a device ring sized from the width would be eight times
    /// too small for the eight-lane case. That is why `numel` and `dtype`
    /// are stored rather than recomputed, and this is the test that says
    /// so out loud.
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

    /// An instance's ring shapes come back in the order the program
    /// indexes them, and a channel it names but this shell lacks is a
    /// refusal rather than a shorter list.
    ///
    /// Skipping would RENUMBER every channel after the gap, so a program
    /// that meant `chan 1` would read channel 2 — a wrong answer that
    /// looks like a working fire, which is the class this whole driver
    /// keeps choosing to refuse instead.
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
        };

        // The list is the INSTANCE's order, not the map's.
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
    fn the_boot_config_extracts() {
        let boot = "[model]\nconfig = \"/tmp/x.json\"\n";
        let v = boot.parse::<toml::Table>().expect("parses");
        let path = v
            .get("model")
            .and_then(|m| m.get("config"))
            .and_then(|d| d.as_str())
            .expect("extracts");
        assert_eq!(path, "/tmp/x.json");
    }
}
