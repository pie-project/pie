//! What an emitter's kernels take of a lane's temporary area beyond the
//! whole-value rule (`eta_exec::whole_value_temporaries`): a select's state,
//! a top-k's histograms and candidates. Each emitter publishes its own need
//! for a region it emits, as a shape-free term the shared rule
//! (`eta_exec::stage_temporaries`) evaluates against a fire's descriptors,
//! so no emitter's figure is another's constant. An emitter that publishes
//! none takes the whole-value rule alone.

/// One region's need of the temporary area, in bytes:
/// `fixed + rows(per_row.0) * per_row.1`, plus the whole-value rule's
/// bytes where `above_whole` (a region whose own state lies below what the
/// rule sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TemporaryNeed {
    /// Bytes whatever the values' shapes.
    pub fixed: u64,
    /// `(value, bytes)`: so many bytes a row of the value.
    pub per_row: Option<(u32, u64)>,
    /// The whole-value rule's bytes lie above `fixed`.
    pub above_whole: bool,
}
