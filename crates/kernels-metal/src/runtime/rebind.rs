pub const FILE: &str = "runtime/rebind.metal";

pub const ENTRYPOINT: &str = "icb_rebind";

pub const MAX_AXES: usize = 4;

pub const MAX_PIPELINES: usize = 256;

pub const MAX_SLABS: usize = 128;

pub const HANDLE: usize = 0;
pub const PIPES: usize = 1;
pub const SLABS: usize = 2;
pub const PLAN: usize = 3;
pub const DESCRIPTOR: usize = 4;
pub const KONST: usize = 5;
pub const COEFF: usize = 6;
pub const SLOTS: usize = 7;
pub const ARMS: usize = 8;
pub const LAWS: usize = 9;
pub const BINDS: usize = 10;
pub const PIPE_FACTS: usize = 11;
pub const LIVE: usize = 12;
pub const STATUS: usize = 13;
pub const CELLS: usize = 14;

pub const BINDINGS: usize = 15;

pub const LAW_CONST: u32 = 0;
pub const LAW_AFFINE: u32 = 1;
pub const LAW_CEIL: u32 = 2;

pub const AT_LANE: u32 = 0;
pub const AT_GROUP: u32 = 1;
pub const AT_ARG: u32 = 2;

pub const ARG_OFFSET: u32 = 0;
pub const ARG_WORD: u32 = 1;
pub const ARG_WIDE: u32 = 2;

pub const BIND_SLAB: u32 = 0;
pub const BIND_CELL: u32 = 1;
pub const BIND_ABSENT: u32 = 2;

pub const PICK_ONLY: u32 = 0;
pub const PICK_ROWS: u32 = 1;

pub const STATUS_MAGIC: u32 = 1;
pub const STATUS_VERSION: u32 = 2;
pub const STATUS_CLASSES: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub slots: u32,
    pub axes: u32,
    pub classes: u32,
    pub magic: u32,
    pub version: u32,
    pad: [u32; 3],
}

impl Plan {
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LawRow {
    pub base: i64,
    pub slope: [i64; MAX_AXES],
    pub mul: i64,
    pub alpha: i64,
    pub beta: i64,
    pub div: i64,
    pub kind: u32,
    pub at_kind: u32,
    pub at_index: u32,
    pub arg_kind: u32,
    pub slab: u32,
    pub cell: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindRow {
    pub offset: u64,
    pub index: u32,
    pub kind: u32,
    pub slab: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmRow {
    pub pipe: u32,
    pub law_at: u32,
    pub law_count: u32,
    pub bind_at: u32,
    pub bind_count: u32,
    pub lanes: [u32; 3],
    pub group: [u32; 3],
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotRow {
    pub arm_at: u32,
    pub arm_count: u32,
    pub pick: u32,
    pub threshold: u32,
    pub rows_law: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipeRow {
    pub width: u32,
    pub total: u32,
}

impl LawRow {
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

#[must_use]
pub fn bytes_of<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: every row in this module is `#[repr(C)]` over integers with its
    // padding spelled out as fields, so the whole object is initialised and
    // `u8` has no alignment requirement.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) }
}

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
