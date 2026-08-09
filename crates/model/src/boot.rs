//! The load path every driver walks: descriptor in, plan out.
//!
//! A driver boots a checkpoint by authoring a load contract and compiling it
//! into a [`LoadPlan`]. Both halves of that live here already —
//! [`contract::author_with_policy`] is this crate's, `plan::compile` is
//! `model-loader`'s — and what a driver adds is the [`Policy`]: which
//! projection layout its GEMMs take, which tensor names its bind path reads.
//!
//! Those two answers are the whole of the difference. `driver-metal` and
//! `driver-cuda` each carried a `compile_load_plan` that differed in exactly
//! [`Policy::projections`] and [`Policy::naming`]; every other field, the
//! author call, the compile call, and the file check were identical, and each
//! file's own comment said so ("carried the same block, bit for bit"). Two
//! copies of a policy is not a spelling problem: a field added to [`Policy`]
//! gets a considered value on the copy its author was looking at and a
//! `Default` on the other, so **equal requests stop authoring equal
//! contracts** — silently, because both still compile and both still boot.
//!
//! So the policy is stated once, here, and a driver names only what it knows
//! that this module cannot: its [`Binding`].
//!
//! # What this module does not do
//!
//! It does not read the snapshot directory to find out what is in it. The
//! caller passes a parsed [`CheckpointMetadata`], because *which* files a
//! checkpoint has is a fact a driver may already have needed for its own
//! reasons — `driver-cuda` parses it before boot and uses it twice — and a
//! second parse inside here would be this module deciding to do I/O the
//! caller had already done.
//!
//! It does stat the files the finished plan declares, which is I/O of a
//! different kind: not "what is here" but "is what I just promised still
//! true". That check has to happen after the plan exists, so no caller can
//! do it instead.

use std::path::Path;

use model_loader::checkpoint::CheckpointMetadata;
use model_loader::plan::{self, LoadPlan, StorageTarget};

use crate::catalog::{self, Override, Unmatched, Variant};
use crate::encoding::Encoding;
use crate::policy::{
    Component, FamilyKnobs, Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, Projections,
    RuntimeQuant,
};

/// What a driver knows about its own forward path that this module cannot.
///
/// Two answers, and they are answers about kernels rather than preferences:
/// the shape the GEMMs want their operands in, and the names the bind path
/// looks up. Everything else in the authored [`Policy`] is the same for every
/// driver and is filled in by [`compile_load_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// Whether the dense attention and MLP GEMMs take one fused operand
    /// ([`Projections::Fused`]) or the separate tensors as stored
    /// ([`Projections::InPlace`]).
    ///
    /// Fusing is a claim that the joins happen once, at load, as
    /// `BulkExtentWrite`s into the arena — not a preference for wide
    /// matrices. A driver whose kernels read `q`, `k`, `v` separately and
    /// asks for `Fused` gets tensors its bind path cannot find.
    pub projections: Projections,
    /// Which tensor names the plan should state: the checkpoint's own
    /// ([`Naming::Hf`]) or the MLX spelling ([`Naming::Mlx`]).
    pub naming: Naming,
}

impl Binding {
    /// Checkpoint names, fused projections — what a driver that binds
    /// HuggingFace tensor names and runs single-operand GEMMs asks for.
    pub const HF_FUSED: Self = Self {
        projections: Projections::Fused,
        naming: Naming::Hf,
    };

    /// MLX names, projections left as stored — what a driver whose lowering
    /// reads the separate tensors asks for.
    pub const MLX_IN_PLACE: Self = Self {
        projections: Projections::InPlace,
        naming: Naming::Mlx,
    };
}

/// Why a load plan could not be compiled.
///
/// Two ways now, where there were three. `Descriptor(String)` is gone
/// with the document it named: there is no `pie.model/1` blob crossing
/// the process boundary to fail to parse, because what crosses is an id.
/// `UnknownFamily(String)` folded into [`Self::Unidentified`], which is
/// the same refusal made honest — it used to say "no author for
/// model_type 'x'" after a `find` over a table, and it now says which
/// rows were CLOSE and by how many tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadPlanError {
    /// This checkpoint is no model this build serves, or an override
    /// named a row that does not exist, or — the table defect — two rows
    /// matched it equally well.
    Unidentified(Unmatched),
    /// The author refused, the plan did not compile, or a file the plan
    /// declares is absent or the wrong size on disk.
    Compile(String),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unidentified(u) => write!(f, "{u}"),
            Self::Compile(m) => write!(f, "{m}"),
        }
    }
}

impl From<Unmatched> for LoadPlanError {
    fn from(u: Unmatched) -> Self {
        Self::Unidentified(u)
    }
}

impl std::error::Error for LoadPlanError {}

/// Author the contract this driver wants and compile it into a plan.
///
/// The caller sends what only it can know — the compiled descriptor it was
/// handed, the target its device presents, and its [`Binding`] — and gets
/// back the plan plus the author's resolved MXFP4 MoE answer.
///
/// The rest of the [`Policy`] is stated here rather than per driver, and each
/// value is a claim rather than a default:
///
/// - [`RuntimeQuant::None`], because a requantization is a decision about an
///   artifact — made once by `pie model build --quant` and written down — not
///   one to re-run over every weight on each boot. The MLX authors do read
///   this field, so leaving it at `None` is a choice being made, not skipped.
/// - [`Mxfp4MoeRequest::Auto`], so a device that binds MXFP4 expert banks as
///   stored says so through [`StorageTarget::native_mxfp4_moe`] and one that
///   does not gets them transcoded. The device already answers this; asking
///   the operator again would let the two disagree.
/// - [`Component::Full`], because a driver boots a whole model. A component
///   split is a serving decision, and no driver takes it at load.
/// - `stream_routed_experts: false` and [`FamilyKnobs::default`], because
///   these are operator knobs no driver has a surface for. Stated as zeros
///   here rather than defaulted at the author, so that "nobody asked for
///   this" is visible at the call rather than inferred from an absence.
///
/// The returned [`Mxfp4MoePolicy`] is the author's *resolved* answer — a
/// family may override the device rule — handed back rather than recomputed,
/// so a bind path cannot disagree with the contract it binds.
///
/// # The file check
///
/// The C++ loader called `verify_model` at this point: a re-author on the far
/// side of the C ABI, holding the marshalled plan to the request, with
/// marshalling and author determinism both in scope. In-process there is no
/// marshalling and a same-process re-author is a restatement rather than a
/// second opinion, so what survives is the part that still checks something
/// real — each file the plan declares is stat'ed against `snapshot_dir`. A
/// snapshot that moved under a plan compiled against it is a refusal this
/// module owes the caller.
///
/// # Errors
///
/// The descriptor does not parse, no author claims its `model_type`, the
/// contract does not compile, or a file the plan declares is missing or the
/// wrong size on disk.
pub fn compile_load_plan(
    snapshot_dir: &Path,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    chosen: &Override,
    encoding: &Encoding,
    binding: Binding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let row = catalog::identify(metadata, chosen)?;
    compile_load_plan_for(snapshot_dir, metadata, target, row, encoding, binding)
}

/// The same, for a caller that already knows its row.
///
/// Two entries and not a default argument, because the two callers are
/// genuinely different: a driver boot has a checkpoint and must find out
/// what it is, and `pie model build --as` has been told. Splitting them
/// keeps [`catalog::identify`] off the second path entirely rather than
/// having it run and be ignored.
///
/// # Errors
///
/// As [`compile_load_plan`], minus the identification.
pub fn compile_load_plan_for(
    snapshot_dir: &Path,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    row: &dyn Variant,
    encoding: &Encoding,
    binding: Binding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let policy = Policy {
        projections: binding.projections,
        naming: binding.naming,
        runtime_quant: RuntimeQuant::None,
        moe_request: Mxfp4MoeRequest::Auto,
        component: Component::Full,
        stream_routed_experts: false,
        knobs: FamilyKnobs::default(),
    };
    let (contract, resolved_moe) =
        crate::contract::author_with_policy(row, encoding, metadata, target, &policy)
            .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    let plan = plan::compile(metadata, &contract, target.clone())
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    model_loader::checkpoint::read::verify_declared_files(&plan, snapshot_dir)
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    Ok((plan, resolved_moe))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint with no files and no tensors: enough to reach the two
    /// refusals below, which both happen before the metadata is read.
    fn empty_checkpoint() -> CheckpointMetadata {
        CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        }
    }

    #[test]
    fn the_two_bindings_are_the_two_a_driver_actually_asks_for() {
        // Named constants rather than a free `Binding { .. }` at each call,
        // so a third combination is a visible addition here rather than a
        // quiet literal in a driver.
        assert_eq!(Binding::HF_FUSED.naming, Naming::Hf);
        assert_eq!(Binding::HF_FUSED.projections, Projections::Fused);
        assert_eq!(Binding::MLX_IN_PLACE.naming, Naming::Mlx);
        assert_eq!(Binding::MLX_IN_PLACE.projections, Projections::InPlace);
    }

    /// A checkpoint no row matches is refused before anything is
    /// authored, and the refusal says which rows were CLOSE.
    ///
    /// This replaces two tests: one for a descriptor that did not parse
    /// and one for a `model_type` no author claimed. Both were about a
    /// STRING failing to place a checkpoint, and the string is gone.
    /// What is left is the real question — are these tensors any model
    /// this build serves — and it has one answer.
    #[test]
    fn a_checkpoint_no_row_matches_is_refused_at_the_door() {
        let err = compile_load_plan(
            Path::new("/nonexistent"),
            &empty_checkpoint(),
            &StorageTarget::default(),
            &Override::None,
            &Encoding::dense(),
            Binding::MLX_IN_PLACE,
        )
        .expect_err("an empty checkpoint is no model");
        assert!(
            matches!(err, LoadPlanError::Unidentified(_)),
            "identification comes first, so it is the first thing that can \
             refuse: {err}"
        );
        assert!(
            err.to_string().contains("matches no model"),
            "the refusal names what it is: {err}"
        );
    }

    /// An override naming a row that does not exist is a typo, and is
    /// answered as one.
    #[test]
    fn an_override_naming_no_row_suggests_the_nearest() {
        let err = compile_load_plan(
            Path::new("/nonexistent"),
            &empty_checkpoint(),
            &StorageTarget::default(),
            &Override::Id("no-such-model-9000".into()),
            &Encoding::dense(),
            Binding::MLX_IN_PLACE,
        )
        .expect_err("no row carries this id");
        // The message has to name the id: an operator reading it is being
        // told what they asked for could not be found, and "unknown model"
        // alone does not say which.
        assert!(
            err.to_string().contains("no-such-model-9000"),
            "the refusal must name the id it could not place: {err}"
        );
        assert!(matches!(err, LoadPlanError::Unidentified(Unmatched::NoSuchId { .. })));
    }

    /// An override that names a REAL row still gets its manifest
    /// checked.
    ///
    /// The escape hatch is deliberately not a way to load a checkpoint
    /// as something it is not — that is the failure the whole
    /// arrangement exists to prevent. An empty checkpoint overridden to
    /// a real row is refused for the tensors it does not have, not
    /// accepted because someone asked.
    #[test]
    fn an_override_does_not_skip_the_check() {
        let Some(known) = catalog::ids().first().copied() else {
            return;
        };
        let err = compile_load_plan(
            Path::new("/nonexistent"),
            &empty_checkpoint(),
            &StorageTarget::default(),
            &Override::Id(known.to_string()),
            &Encoding::dense(),
            Binding::MLX_IN_PLACE,
        )
        .expect_err("an empty checkpoint is not that row either");
        assert!(matches!(err, LoadPlanError::Unidentified(Unmatched::NoRow { .. })));
        assert!(
            err.to_string().contains(known),
            "the diff names the row that was asked for: {err}"
        );
    }

    /// `From<Unmatched>` is what lets the identification refuse with
    /// `?`.
    #[test]
    fn an_unmatched_converts_without_losing_its_message() {
        let u = Unmatched::Ambiguous { ids: vec!["a", "b"] };
        let e: LoadPlanError = u.clone().into();
        assert_eq!(e, LoadPlanError::Unidentified(u.clone()));
        assert_eq!(e.to_string(), u.to_string());
    }
}
