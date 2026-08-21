//! The query-only launch context the derived binder reads a launch through.
//!
//! This wrapped ~50 fact queries when bind bodies could ask; the no-ask sweep
//! leaves the binder's own reads — the operand run, the statement scalars,
//! the named weights — and nothing a signature now carries. The fire-scoped
//! objects (KV views, workspaces, state slabs) answer through the runtime
//! operand channel instead: `super::views`, addressed by the resolver at
//! bind, never queried from a body.

use core::ffi::c_void;

use kernels::Refusal;

use super::facts::Fire;
use kernels_cuda::attn::Rows;

/// The query-only launch context the derived binder reads. Wraps the
/// concrete [`Fire`].
#[derive(Clone, Copy)]
pub struct Cx<'a> {
    fire: &'a Fire<'a>,
}

impl core::fmt::Debug for Cx<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rows = self.fire.rows();
        f.debug_struct("Cx")
            .field("layer", &self.fire.layer())
            .field("rows", &rows)
            .finish()
    }
}

impl<'a> Cx<'a> {
    /// Wrap one fire's facts.
    #[must_use]
    pub const fn new(fire: &'a Fire<'a>) -> Self {
        Self { fire }
    }

    /// The fire behind the queries, for the GENERATED binder only.
    /// `table::derived_arm` needs the operand list UNINDEXED and the
    /// input/result split.
    pub(crate) const fn fire(&self) -> &'a Fire<'a> {
        self.fire
    }

    /// The `i`th named weight's address. Refuses `Absent`, being indexed.
    ///
    /// # Errors
    pub fn weight_named(&self, i: usize) -> Result<*mut c_void, Refusal> {
        self.fire.weight_named(i).ok_or(Refusal::Absent {
            what: "a named weight",
        })
    }

    /// The `i`th statement parameter.
    ///
    /// # Errors
    pub fn param(&self, i: usize) -> Result<u32, Refusal> {
        self.fire.param(i).ok_or(Refusal::Absent {
            what: "a statement parameter",
        })
    }

    /// The `i`th statement parameter as a float.
    ///
    /// # Errors
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        self.param(i).map(f32::from_bits)
    }

    /// Which rows this fire launches. Always answerable.
    #[must_use]
    pub fn rows(&self) -> Rows {
        self.fire.rows()
    }

    /// Which layer this statement belongs to. Always answerable.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.fire.layer()
    }

    /// The engine's cuBLAS handle. Null IS the answer when none exists, and
    /// `Ctx::cublas()` already turns that into a refusal that names it.
    #[must_use]
    pub fn cublas(&self) -> *mut c_void {
        self.fire.cublas()
    }

    /// The op join this statement was bound under.
    #[must_use]
    pub const fn spec(&self) -> &'a super::LaunchSpec {
        self.fire.spec
    }
}
