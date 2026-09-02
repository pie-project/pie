//! The indirect-command-buffer plane's one shader, and the layout its
//! tables travel in. `icb/rebind.metal` rewrites an `MTLIndirectCommandBuffer`
//! in place from a fire descriptor, so a Metal fire is a descriptor write
//! and one `executeCommandsInBuffer:`. The `#[repr(C)]` rows here are the
//! host half of that layout — field order, widths and padding must match
//! the shader's structs line for line (checked at compile time below).

/// The shader file, as a [`Fire`](crate::Fire) names it.
pub const FILE: &str = "icb/rebind.metal";

/// The entrypoint.
pub const ENTRYPOINT: &str = "icb_rebind";

/// How many probe directions a lowered law table may carry. A basis wider
/// than this is a named refusal in the lowering, not a silent truncation.
pub const MAX_AXES: usize = 4;

/// How many distinct pipelines one artifact's arms may name.
pub const MAX_PIPELINES: usize = 256;

/// How many distinct reservations one artifact's slots may bind.
pub const MAX_SLABS: usize = 128;

/// Buffer index: the argument buffer holding the ICB's `gpuResourceID`.
pub const HANDLE: usize = 0;
/// Buffer index: the argument buffer of `compute_pipeline_state`s.
pub const PIPES: usize = 1;
/// Buffer index: the argument buffer of reservation addresses.
pub const SLABS: usize = 2;
/// Buffer index: the [`Plan`] header.
pub const PLAN: usize = 3;
/// Buffer index: the packed `driver::fire::descriptor` bytes.
pub const DESCRIPTOR: usize = 4;
/// Buffer index: one constant per direction of the coordinate recipe.
pub const KONST: usize = 5;
/// Buffer index: the recipe's coefficients, `axes × 2 × classes`.
pub const COEFF: usize = 6;
/// Buffer index: one [`SlotRow`] per ICB slot.
pub const SLOTS: usize = 7;
/// Buffer index: the [`ArmRow`]s, in slot order.
pub const ARMS: usize = 8;
/// Buffer index: the [`LawRow`]s.
pub const LAWS: usize = 9;
/// Buffer index: the [`BindRow`]s.
pub const BINDS: usize = 10;
/// Buffer index: one [`PipeRow`] per pipeline.
pub const PIPE_FACTS: usize = 11;
/// Buffer index: one word per slot saying which arm is encoded in it.
pub const LIVE: usize = 12;
/// Buffer index: the shader's one output — a refusal code, or zero.
pub const STATUS: usize = 13;
/// Buffer index: the staged-scalar arena.
pub const CELLS: usize = 14;

/// How many buffers the rebind kernel binds.
pub const BINDINGS: usize = 15;

/// Law kind: the same number in every composition.
pub const LAW_CONST: u32 = 0;
/// Law kind: `base + Σ slope·coord`.
pub const LAW_AFFINE: u32 = 1;
/// Law kind: `mul · ⌈(α·rows + β)/div⌉`.
pub const LAW_CEIL: u32 = 2;

/// Law place: a grid axis.
pub const AT_LANE: u32 = 0;
/// Law place: a threadgroup axis.
pub const AT_GROUP: u32 = 1;
/// Law place: an argument.
pub const AT_ARG: u32 = 2;

/// Argument kind: a buffer binding whose offset moves.
pub const ARG_OFFSET: u32 = 0;
/// Argument kind: a four-byte scalar in a staged cell.
pub const ARG_WORD: u32 = 1;
/// Argument kind: an eight-byte scalar in a staged cell.
pub const ARG_WIDE: u32 = 2;

/// Binding kind: a reservation at a byte offset.
pub const BIND_SLAB: u32 = 0;
/// Binding kind: a staged scalar cell.
pub const BIND_CELL: u32 = 1;
/// Binding kind: an index the shader does not dereference on this arm.
pub const BIND_ABSENT: u32 = 2;

/// Arm pick: one arm, always.
pub const PICK_ONLY: u32 = 0;
/// Arm pick: a threshold on the window's rows.
pub const PICK_ROWS: u32 = 1;

/// Status: the descriptor's magic was not the one this table was lowered for.
pub const STATUS_MAGIC: u32 = 1;
/// Status: the descriptor's ABI version was not this build's.
pub const STATUS_VERSION: u32 = 2;
/// Status: the descriptor carries a different number of classes.
pub const STATUS_CLASSES: u32 = 3;

/// The header: what the whole table is, in five numbers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// How many ICB slots — one thread each.
    pub slots: u32,
    /// How many probe directions the laws are written in.
    pub axes: u32,
    /// How many classes the coordinate recipe reads.
    pub classes: u32,
    /// The descriptor magic this table was lowered against.
    pub magic: u32,
    /// The descriptor ABI version this table was lowered against.
    pub version: u32,
    pad: [u32; 3],
}

impl Plan {
    /// A header.
    #[must_use]
    pub const fn new(slots: u32, axes: u32, classes: u32, magic: u32, version: u32) -> Plan {
        Plan {
            slots,
            axes,
            classes,
            magic,
            version,
            pad: [0; 3],
        }
    }
}

/// One component's law.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LawRow {
    /// `Const`'s value, or `Affine`'s value at the coordinate origin.
    pub base: i64,
    /// `Affine`'s slopes, one per direction.
    pub slope: [i64; MAX_AXES],
    /// `Ceil`'s scale outside the ceiling.
    pub mul: i64,
    /// `Ceil`'s numerator slope over the window's rows.
    pub alpha: i64,
    /// `Ceil`'s numerator offset.
    pub beta: i64,
    /// `Ceil`'s tile.
    pub div: i64,
    /// [`LAW_CONST`], [`LAW_AFFINE`] or [`LAW_CEIL`].
    pub kind: u32,
    /// [`AT_LANE`], [`AT_GROUP`] or [`AT_ARG`].
    pub at_kind: u32,
    /// Which axis, or which argument index.
    pub at_index: u32,
    /// For an argument law: [`ARG_OFFSET`], [`ARG_WORD`] or [`ARG_WIDE`].
    pub arg_kind: u32,
    /// For [`ARG_OFFSET`]: which reservation the offset is into.
    pub slab: u32,
    /// For a scalar: the byte offset of its cell in the staged arena.
    pub cell: u32,
    pad: [u32; 2],
}

/// One binding of one arm's whole argument list — what a slot that was reset,
/// or that switched arms, is encoded again from.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindRow {
    /// Bytes into the reservation, or into the scalar arena.
    pub offset: u64,
    /// The argument index.
    pub index: u32,
    /// [`BIND_SLAB`], [`BIND_CELL`] or [`BIND_ABSENT`].
    pub kind: u32,
    /// Which reservation, for [`BIND_SLAB`].
    pub slab: u32,
    pad: u32,
}

/// One arm of one slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmRow {
    /// Which pipeline, in the argument buffer's order.
    pub pipe: u32,
    /// Where this arm's laws start, and how many.
    pub law_at: u32,
    /// How many laws.
    pub law_count: u32,
    /// Where this arm's whole binding list starts.
    pub bind_at: u32,
    /// How many bindings.
    pub bind_count: u32,
    /// The skeleton grid; a law overrides the axes that move.
    pub lanes: [u32; 3],
    /// The stated threadgroup, `[0, 0, 0]` to ask the pipeline's occupancy.
    pub group: [u32; 3],
    pad: u32,
}

/// One ICB slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotRow {
    /// Where this slot's arms start.
    pub arm_at: u32,
    /// How many arms it has.
    pub arm_count: u32,
    /// [`PICK_ONLY`] or [`PICK_ROWS`].
    pub pick: u32,
    /// The first window row count that takes the second arm.
    pub threshold: u32,
    /// Which law is this slot's window row count.
    pub rows_law: u32,
    pad: [u32; 3],
}

/// The two occupancy numbers of one compiled pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipeRow {
    /// `threadExecutionWidth`.
    pub width: u32,
    /// `maxTotalThreadsPerThreadgroup`.
    pub total: u32,
}

impl LawRow {
    /// A row with everything zero but its place — the shape every constructor
    /// starts from.
    #[must_use]
    pub const fn at(kind: u32, at_kind: u32, at_index: u32) -> LawRow {
        LawRow {
            base: 0,
            slope: [0; MAX_AXES],
            mul: 0,
            alpha: 0,
            beta: 0,
            div: 1,
            kind,
            at_kind,
            at_index,
            arg_kind: ARG_OFFSET,
            slab: 0,
            cell: 0,
            pad: [0; 2],
        }
    }
}

impl BindRow {
    /// One binding.
    #[must_use]
    pub const fn new(index: u32, kind: u32, slab: u32, offset: u64) -> BindRow {
        BindRow {
            offset,
            index,
            kind,
            slab,
            pad: 0,
        }
    }
}

impl ArmRow {
    /// One arm.
    #[must_use]
    pub const fn new(
        pipe: u32,
        law_at: u32,
        law_count: u32,
        bind_at: u32,
        bind_count: u32,
        lanes: [u32; 3],
        group: [u32; 3],
    ) -> ArmRow {
        ArmRow {
            pipe,
            law_at,
            law_count,
            bind_at,
            bind_count,
            lanes,
            group,
            pad: 0,
        }
    }
}

impl SlotRow {
    /// One slot.
    #[must_use]
    pub const fn new(
        arm_at: u32,
        arm_count: u32,
        pick: u32,
        threshold: u32,
        rows_law: u32,
    ) -> SlotRow {
        SlotRow {
            arm_at,
            arm_count,
            pick,
            threshold,
            rows_law,
            pad: [0; 3],
        }
    }
}

/// The bytes of any `#[repr(C)]` table row, for a shell that writes them into
/// a reservation.
///
/// # Safety
///
/// The caller states that `T` is one of this module's `#[repr(C)]` rows: no
/// padding a reader could observe as uninitialised, no pointer, no
/// `Drop`.
#[must_use]
pub fn bytes_of<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: every row in this module is `#[repr(C)]` over integers with its
    // padding spelled out as fields, so the whole object is initialised and
    // `u8` has no alignment requirement.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) }
}

// The host rows are the shader's structs: a field added on one side and not
// the other moves a size and fails here, at compile time.
const _: () = {
    assert!(size_of::<Plan>() == 32);
    assert!(size_of::<LawRow>() == 104);
    assert!(align_of::<LawRow>() == 8);
    assert!(size_of::<BindRow>() == 24);
    assert!(align_of::<BindRow>() == 8);
    assert!(size_of::<ArmRow>() == 48);
    assert!(size_of::<SlotRow>() == 32);
    assert!(size_of::<PipeRow>() == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader ships, resolves, and its structs are the ones above — the
    /// sizes are checked at compile time, so what is left to check is that
    /// the text really is in the rlib and really names the entrypoint.
    #[test]
    fn the_rebind_shader_ships_and_names_the_entrypoint_the_shell_asks_for() {
        let text = crate::sources::source(FILE).expect("the rebind shader ships");
        assert!(
            text.contains(&format!("kernel void {ENTRYPOINT}")),
            "the shipped source does not declare `{ENTRYPOINT}`"
        );
        let flat = crate::sources::resolve(FILE).expect("it resolves");
        assert!(flat.contains("#include <metal_command_buffer>"));
        let define = |name: &str| -> Option<usize> {
            flat.lines()
                .filter_map(|line| line.trim().strip_prefix("#define "))
                .find_map(|rest| {
                    let mut parts = rest.split_whitespace();
                    (parts.next()? == name)
                        .then(|| parts.next()?.trim_end_matches('u').parse().ok())?
                })
        };
        for (name, value) in [
            ("ICB_MAX_AXES", MAX_AXES),
            ("ICB_MAX_PIPELINES", MAX_PIPELINES),
            ("ICB_MAX_SLABS", MAX_SLABS),
            ("ICB_DESC_CLASSES", 5),
            ("ICB_CLASS_WORDS", 4),
        ] {
            assert_eq!(
                define(name),
                Some(value),
                "the shader's `{name}` is not this module's {value}"
            );
        }
    }
}
