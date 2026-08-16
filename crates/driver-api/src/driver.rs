//! What a driver promises, as one trait.
//!
//! A driver that does not override a verb REFUSES IT BY NAME through the
//! default method, with its own kind in the message. [`Unsupported`] is a
//! type rather than a string because the engine matches on it.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported {
    /// The verb that was asked for.
    pub verb: &'static str,
    /// The driver that does not serve it.
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
/// (ABI v14), so a post either enters the driver with one completion or says
/// why it cannot.
pub enum FrameLaunchOutcome {
    /// The frame was admitted and posted; one completion settles it.
    Launched(SubmissionCompletion),
    /// Admission is full right now; the engine re-posts later.
    Exhausted,
    /// The frame can never fit within the driver's physical budget ceiling.
    Impossible,
}

/// One execution device, behind the verbs the engine has for it. `Send` for the
/// scheduler thread that owns it, `Sync` for the `static` registry that holds it.
pub trait Driver: Send + Sync {
    /// This driver's short name, as configuration and metrics spell it.
    fn kind(&self) -> &'static str;

    /// What the device is, as it answered at create time. `None` is the
    /// remote driver's: the far side did not send its measurement.
    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// Which memory this driver's KV pages live in.
    fn device_domain(&self) -> DeviceDomain;

    /// Which backend's kernels this driver wants the HOST to generate, or
    /// `None` when it generates its own. NOT derivable from [`Self::kind`],
    /// which answers which DEVICE this is.
    fn codegen_backend(&self) -> Option<&'static str> {
        None
    }

    /// The KV pages this driver can hand to another node, if any.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    /// Make the model resident and answer what it can then do.
    ///
    /// # Errors
    /// The checkpoint could not be read, or names something this device
    /// cannot run.
    fn load_model(&mut self, descs: Vec<ModelLoadDesc>) -> Result<DriverCapabilities>;

    /// Adopt a launch program and answer its id. Re-registering an identical
    /// program answers the existing id; the dedup key is the program hash.
    ///
    /// # Errors
    /// The package could not be adopted, or a generated region failed to
    /// compile.
    fn register_program(&mut self, plan: &ProgramRegistration) -> Result<ProgramId>;

    /// Place one channel's ring and answer where it went.
    ///
    /// # Errors
    /// The ring could not be allocated, or the descriptor names a shape this
    /// driver cannot place.
    fn register_channel(&mut self, plan: &ChannelRegistrationPlan) -> Result<RegisteredChannel>;

    /// Instantiate a registered program against a set of channels.
    ///
    /// # Errors
    /// No such program, or a channel the plan names is not registered.
    fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance>;

    /// Release an instance and everything it held.
    ///
    /// # Errors
    /// The driver is closed. An unknown id is a no-op, not an error: close is
    /// idempotent because the engine may race a teardown against a completion.
    fn close_instance(&mut self, id: u64) -> Result<()>;

    /// Release a channel's ring.
    ///
    /// # Errors
    /// The driver is closed. An unknown id is a no-op, for the reason
    /// [`Self::close_instance`] gives.
    fn close_channel(&mut self, id: u64) -> Result<()>;

    /// Post one sealed frame. Admission is folded into the call: the driver
    /// either admits — one completion settles the whole frame — or reports
    /// [`Exhausted`](FrameLaunchOutcome::Exhausted)/[`Impossible`](FrameLaunchOutcome::Impossible)
    /// **without side effects**, since a refusal that had allocated would
    /// leak on every re-post.
    ///
    /// # Errors
    /// The frame is malformed, or the device failed after accepting it.
    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome>;

    /// Encode media into the model's embedding space.
    ///
    /// # Errors
    /// [`Unsupported`] unless overridden.
    fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    /// Move KV pages, within this device or across the host boundary.
    ///
    /// # Errors
    /// [`Unsupported`] unless overridden, or a plan whose ends are not both
    /// in a domain this driver owns.
    fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("copy_kv"))
    }

    /// Move recurrent state.
    ///
    /// # Errors
    /// [`Unsupported`] unless overridden.
    fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("copy_state"))
    }

    /// Commit or release KV pages without moving an address a fire has bound.
    ///
    /// # Errors
    /// [`Unsupported`] unless overridden, or the device refused the growth.
    fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let _ = plan;
        Err(self.unsupported("resize_pool"))
    }

    /// Tell whoever is waiting that this driver is gone.
    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    /// Name a verb this driver does not serve.
    fn unsupported(&self, verb: &'static str) -> anyhow::Error {
        Unsupported {
            verb,
            driver: self.kind(),
        }
        .into()
    }
}

const _: () = {
    // `Driver` must stay object-safe: the registry holds `Box<dyn Driver>`. A
    // verb taking `self` by value or naming `Self` in argument position would
    // break that at the registry; this puts the error here instead.
    #[allow(dead_code)]
    fn object_safe(driver: &dyn Driver) -> &'static str {
        driver.kind()
    }
};
