//! What a driver promises, as one trait.
//!
//! # Why this is a trait and was an `enum`
//!
//! The C++ shells could only be reached through an opaque handle and a set of
//! free functions, so the Rust side that called them had to be a `match`: a
//! `*mut PieDriver` cannot implement anything. `engine`'s `DriverBackend` was
//! that `match`, and it had grown to fourteen verbs over five variants —
//! **seventy arms whose bodies were all the same word**. Nothing in it chose
//! anything; every arm named a backend and forwarded.
//!
//! The cost was not the line count. It was that the `enum` made a driver's
//! properties into the CALLER's knowledge:
//!
//! - `kind()` and `device_domain()` were `match`es in the engine, so a new
//!   backend's domain was declared by the crate that dispatches to it rather
//!   than by the crate that owns the memory. A test had to exist
//!   (`backend::tests`) whose whole job was to check that the engine had not
//!   answered one backend's domain for another. A driver that states its own
//!   domain cannot fail that way, and the test that guarded it is gone with
//!   the arm it guarded.
//! - Every variant paid the widest variant's width, which is why two of them
//!   were `Box`ed with `size_of` measurements in the comments and a third was
//!   documented as wanting the same treatment across "eighteen call sites in
//!   code no backend here owns". `Box<dyn Driver>` is one word for all of
//!   them and the question does not arise.
//! - Adding a backend meant editing fourteen `match`es in a crate that has no
//!   opinion about the new device.
//!
//! # Refusing by name
//!
//! Not every driver serves every verb, and the seams said so in prose: four
//! backend headers each carry a paragraph on which verbs they refuse and why,
//! and each implemented the refusal by hand. The discipline they were all
//! describing — *"It refuses by name rather than being absent. A backend that
//! cannot be selected teaches nothing; one that is selected and says exactly
//! which verb it cannot serve is a working seam with a stated hole"* — is a
//! DEFAULT METHOD here. A driver that does not override [`Driver::encode`]
//! refuses it, by name, with its own kind in the message, in one line that no
//! backend had to write.
//!
//! [`Unsupported`] is a type rather than a string because it is the one
//! failure the engine matches on: a driver that cannot encode is a deployment
//! fact to route around, not an error to show a user.

use anyhow::Result;

use crate::capabilities::{DeviceFacts, DriverCapabilities, ModelLoadDesc};
use crate::channel::RegisteredChannel;
use crate::completion::SubmissionCompletion;
use crate::instance::{BoundInstance, InstanceBindingPlan, ProgramId};
use crate::local::DeviceDomain;
use crate::plan::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::submission::FrameSubmission;
use crate::transfer::KvHandle;

/// A verb this driver does not serve.
///
/// Carries both halves a caller needs: which verb was asked for, and which
/// driver was asked. The engine matches on this — a backend that cannot
/// `encode` is a deployment fact to route around rather than a failure to
/// report — which is why it is a type and not a formatted string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported {
    /// The verb, as the trait spells it.
    pub verb: &'static str,
    /// The driver that does not serve it, as [`Driver::kind`] answers.
    pub driver: &'static str,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the {} driver does not serve `{}`",
            self.driver, self.verb
        )
    }
}

impl std::error::Error for Unsupported {}

/// Outcome of a frame launch post: admission is folded into the launch call
/// (ABI v14), so a post either enters the driver with one completion, or
/// reports why it cannot right now.
pub enum FrameLaunchOutcome {
    /// The frame was admitted and posted; one completion settles it.
    Launched(SubmissionCompletion),
    /// Admission is full right now; the engine re-posts later.
    Exhausted,
    /// The frame can never fit within the driver's physical budget ceiling.
    Impossible,
}

/// One execution device, behind the verbs the engine has for it.
///
/// `Send` because a driver is moved onto the scheduler thread that owns it,
/// and `Sync` because the registry that holds it between `register` and
/// `take` is a `static`. Neither is a new obligation: every mutating verb
/// takes `&mut self`, so a driver still serves one caller at a time, and the
/// `&self` verbs below answer facts.
pub trait Driver: Send + Sync {
    // -- what the driver is ------------------------------------------------
    /// This driver's short name, as configuration and metrics spell it.
    fn kind(&self) -> &'static str;

    /// What the device is, as it answered at create time.
    ///
    /// `Option`, and the `None` is the remote driver's. A `RemoteDriver` is
    /// handed an `ExecutorRpcClient` and a `DriverCapabilities`, and never a
    /// `DeviceFacts` — the far side measured its own card and did not send
    /// the measurement. That is the same hole [`Self::device_domain`]
    /// documents from the other end, and answering some other driver's facts
    /// to fill it is exactly what that field's history warns about.
    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// Which memory this driver's KV pages live in.
    ///
    /// Answered by the DRIVER, which is the only side that knows. It used to
    /// be a `match` in the engine, and `DriverSpec::device_domain` records
    /// what the hardcoded answer before that cost: a
    /// `PIE_MEMORY_DOMAIN_CUDA_DEVICE` stamped on every `KvCopyPlan` at nine
    /// sites regardless of the driver it was for, which any driver that
    /// checks the tag refuses on every prefix-cache hit and every swap.
    fn device_domain(&self) -> DeviceDomain;

    /// Which backend's kernels this driver wants the HOST to generate, or
    /// `None` when it generates its own (or needs none).
    ///
    /// A driver's own answer, and it has to be: the alternative is a caller
    /// deriving it from something adjacent. It was a `match` in `engine`
    /// (`Self::Cuda(_) => Some("cuda"), _ => None`), and when that `match`
    /// became `Driver::kind()` the derivation broke — `"metal"` parses as a
    /// codegen backend, so a Metal driver started being handed host-generated
    /// Metal kernels it has no path for, silently. `kind` answers WHICH
    /// DEVICE this is; that is not the same question.
    ///
    /// `DriverCapabilities::codegen_backend` is the same fact on the load
    /// handshake, and every driver currently leaves it empty — so this is the
    /// live one until a driver fills that in.
    fn codegen_backend(&self) -> Option<&'static str> {
        None
    }

    /// The KV pages this driver can hand to another node, if any.
    ///
    /// `None` is the honest answer for a device with no external-memory
    /// export, which is why it is the default.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    // -- loading -----------------------------------------------------------
    /// Make the model resident and answer what it can then do.
    ///
    /// # Errors
    ///
    /// The checkpoint could not be read, or names something this device
    /// cannot run.
    fn load_model(&mut self, descs: Vec<ModelLoadDesc>) -> Result<DriverCapabilities>;

    // -- the registry ------------------------------------------------------
    /// Adopt a launch program and answer its id.
    ///
    /// Re-registering an identical program answers the existing id: the
    /// dedup key is the program hash, which is what makes a program bound a
    /// thousand times compiled once.
    ///
    /// # Errors
    ///
    /// The package could not be adopted, or a generated region failed to
    /// compile.
    fn register_program(&mut self, plan: &ProgramRegistration) -> Result<ProgramId>;

    /// Place one channel's ring and answer where it went.
    ///
    /// # Errors
    ///
    /// The ring could not be allocated, or the descriptor names a shape this
    /// driver cannot place.
    fn register_channel(&mut self, plan: &ChannelRegistrationPlan) -> Result<RegisteredChannel>;

    /// Instantiate a registered program against a set of channels.
    ///
    /// # Errors
    ///
    /// No such program, or a channel the plan names is not registered.
    fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance>;

    /// Release an instance and everything it held.
    ///
    /// # Errors
    ///
    /// The driver is closed. An unknown id is a no-op, not an error: close is
    /// idempotent because the engine may race a teardown against a completion.
    fn close_instance(&mut self, id: u64) -> Result<()>;

    /// Release a channel's ring.
    ///
    /// # Errors
    ///
    /// The driver is closed. An unknown id is a no-op, for the reason
    /// [`Self::close_instance`] gives.
    fn close_channel(&mut self, id: u64) -> Result<()>;

    // -- the work ----------------------------------------------------------
    /// Post one sealed frame.
    ///
    /// Admission is folded into the call: the driver evaluates the
    /// frame-union demand and either admits — one completion settles the
    /// whole frame — or reports
    /// [`Exhausted`](FrameLaunchOutcome::Exhausted)/[`Impossible`](FrameLaunchOutcome::Impossible)
    /// **without side effects**. A refusal that had already allocated would
    /// leak on every re-post, which is why the no-side-effects half is part
    /// of the contract rather than an implementation note.
    ///
    /// # Errors
    ///
    /// The frame is malformed, or the device failed after accepting it.
    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome>;

    /// Encode media into the model's embedding space.
    ///
    /// # Errors
    ///
    /// [`Unsupported`] unless overridden.
    fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    /// Move KV pages, within this device or across the host boundary.
    ///
    /// # Errors
    ///
    /// [`Unsupported`] unless overridden, or a plan whose ends are not both
    /// in a domain this driver owns.
    fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("copy_kv"))
    }

    /// Move recurrent state.
    ///
    /// # Errors
    ///
    /// [`Unsupported`] unless overridden. A driver that serves no row with
    /// recurrent state refuses here, and that refusal is unreachable rather
    /// than unfinished.
    fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("copy_state"))
    }

    /// Commit or release KV pages without moving an address a fire has bound.
    ///
    /// # Errors
    ///
    /// [`Unsupported`] unless overridden, or the device refused the growth.
    fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("resize_pool"))
    }

    // -- lifecycle ---------------------------------------------------------
    /// Tell whoever is waiting that this driver is gone.
    ///
    /// Meaningful only where there is a connection to drop. An in-process
    /// driver has none — a device that has gone away surfaces as a failed
    /// submission, which the driver reports there — so the default is to do
    /// nothing, and that is the honest answer for four of the five backends
    /// rather than a stub.
    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    /// Name a verb this driver does not serve.
    ///
    /// A provided method rather than a free function so that the refusals
    /// above can reach [`Self::kind`] through `self`: a driver names itself
    /// in its own refusal, which is the half a caller cannot supply.
    ///
    /// Overriding this is not useful and not forbidden — it changes the
    /// message, never whether the verb is served, because what a driver
    /// serves is decided by which methods it overrides.
    fn unsupported(&self, verb: &'static str) -> anyhow::Error {
        Unsupported {
            verb,
            driver: self.kind(),
        }
        .into()
    }
}

const _: () = {
    // `Driver` must stay object-safe: the registry holds `Box<dyn Driver>`,
    // which is the whole point of the trait. A verb that took `self` by value
    // or named `Self` in argument position would break that at the registry,
    // with an error about `DriverBackend` rather than about the verb that
    // caused it. This puts the error here instead.
    #[allow(dead_code)]
    fn object_safe(driver: &dyn Driver) -> &'static str {
        driver.kind()
    }
};
