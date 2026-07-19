//! Compatibility accessors for compiler-owned canonical stage signatures.

use crate::compiler::{StageSignature, compile_stage_at};
use crate::registry::Stage;
use crate::validate::BoundTrace;

/// Canonical compiler signature for `stage`.
pub fn stage_signature(bound: &BoundTrace, stage: Stage) -> Option<StageSignature> {
    let index = bound
        .container
        .stages
        .iter()
        .position(|program| program.stage == stage)?;
    Some(stage_signature_at(bound, index))
}

/// Canonical compiler signature for the stage at `stage_index`.
pub fn stage_signature_at(bound: &BoundTrace, stage_index: usize) -> StageSignature {
    compile_stage_at(bound, stage_index).signature
}

/// Hash accessor retained for scheduler call sites that only need the compact
/// grouping key. Collision-sensitive caches also compare canonical bytes.
pub fn stage_key(bound: &BoundTrace, stage: Stage) -> Option<u64> {
    stage_signature(bound, stage).map(|signature| signature.hash)
}

pub fn stage_key_at(bound: &BoundTrace, stage_index: usize) -> u64 {
    stage_signature_at(bound, stage_index).hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use crate::op::Op;
    use crate::registry::ModelProfile;
    use crate::types::{DType, Literal, Shape};
    use crate::validate::bind;
    use alloc::{vec, vec::Vec};

    fn ch(dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// A seeded self-loop on channel `c` of dtype `dt`: take, +1, put.
    fn self_loop(c: u32, dt: DType) -> Vec<Op> {
        let one = match dt {
            DType::I32 => Literal::I32(1),
            DType::U32 => Literal::U32(1),
            DType::F32 => Literal::F32(1.0),
            DType::Bool => Literal::Bool(true),
        };
        // ChanTake -> v0, Const -> v1, Add -> v2, ChanPut(v2).
        vec![
            Op::ChanTake(c),
            Op::Const(one),
            Op::Add(0, 1),
            Op::ChanPut { chan: c, value: 2 },
        ]
    }

    fn bound_of(container: TraceContainer) -> BoundTrace {
        bind(container, ModelProfile::dummy()).expect("binds")
    }

    // A stage that self-loops a u32 channel, then an i32 channel. The op indices
    // are stage-local; each self-loop's Add refers to ids 0,1 / 4,5 across the
    // two loops, so build them with an explicit id base.
    fn two_loop_epilogue(u32_chan: u32, i32_chan: u32) -> StageProgram {
        // u32 loop: v0=take, v1=const, v2=add, put(v2); i32 loop: v3=take,
        // v4=const, v5=add, put(v5).
        let ops = vec![
            Op::ChanTake(u32_chan),
            Op::Const(Literal::U32(1)),
            Op::Add(0, 1),
            Op::ChanPut {
                chan: u32_chan,
                value: 2,
            },
            Op::ChanTake(i32_chan),
            Op::Const(Literal::I32(1)),
            Op::Add(3, 4),
            Op::ChanPut {
                chan: i32_chan,
                value: 5,
            },
        ];
        StageProgram {
            stage: Stage::Epilogue,
            ops,
        }
    }

    #[test]
    fn same_epilogue_two_passes_same_key() {
        // Pass A: channels [u32, i32]; epilogue touches global 0 (u32) then 1 (i32).
        let a = bound_of(TraceContainer {
            names: vec![],
            channels: vec![ch(DType::U32), ch(DType::I32)],
            ports: vec![],
            stages: vec![two_loop_epilogue(0, 1)],
            externs: vec![],
        });

        // Pass B: an extra f32 channel first + a prologue that uses it, and the
        // u32/i32 channels declared in SWAPPED global order (i32=1, u32=2). The
        // epilogue is the SAME program: first-used channel u32 (global 2), then
        // i32 (global 1). Global numbering differs; the stage-local form matches.
        let b = bound_of(TraceContainer {
            names: vec![],
            channels: vec![ch(DType::F32), ch(DType::I32), ch(DType::U32)],
            ports: vec![],
            stages: vec![
                StageProgram {
                    stage: Stage::Prologue,
                    ops: self_loop(0, DType::F32),
                },
                two_loop_epilogue(2, 1),
            ],
            externs: vec![],
        });

        assert_ne!(a.hash, b.hash, "the two passes are different programs (C3)");
        let ka = stage_key(&a, Stage::Epilogue).unwrap();
        let kb = stage_key(&b, Stage::Epilogue).unwrap();
        assert_eq!(
            ka, kb,
            "same epilogue embedded in two passes shares a stage key"
        );

        // The prologue-only stage exists in B, not A.
        assert!(stage_key(&b, Stage::Prologue).is_some());
        assert!(stage_key(&a, Stage::Prologue).is_none());
    }

    #[test]
    fn one_op_difference_differs() {
        let base = bound_of(TraceContainer {
            names: vec![],
            channels: vec![ch(DType::U32), ch(DType::I32)],
            ports: vec![],
            stages: vec![two_loop_epilogue(0, 1)],
            externs: vec![],
        });

        // Change one op in the u32 loop: Add -> Sub.
        let mut ops = two_loop_epilogue(0, 1).ops;
        ops[2] = Op::Sub(0, 1);
        let changed = bound_of(TraceContainer {
            names: vec![],
            channels: vec![ch(DType::U32), ch(DType::I32)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
            externs: vec![],
        });

        assert_ne!(
            stage_key(&base, Stage::Epilogue).unwrap(),
            stage_key(&changed, Stage::Epilogue).unwrap(),
            "a one-op difference yields a different stage key"
        );
    }

    #[test]
    fn channel_signature_matters() {
        // Same op graph + same renumbering, but the two touched channels differ
        // in dtype ordering: [u32, i32] vs [i32, u32]. The referenced-channel
        // signatures (local order) must separate the keys.
        let uv_iv = bound_of(TraceContainer {
            names: vec![],
            channels: vec![ch(DType::U32), ch(DType::I32)],
            ports: vec![],
            stages: vec![two_loop_epilogue(0, 1)],
            externs: vec![],
        });
        // First-used channel i32, second u32 — swap dtypes AND the loops' consts
        // to keep the body well-typed.
        let iv_uv = {
            let ops = vec![
                Op::ChanTake(0),
                Op::Const(Literal::I32(1)),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanTake(1),
                Op::Const(Literal::U32(1)),
                Op::Add(3, 4),
                Op::ChanPut { chan: 1, value: 5 },
            ];
            bound_of(TraceContainer {
                names: vec![],
                channels: vec![ch(DType::I32), ch(DType::U32)],
                ports: vec![],
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops,
                }],
                externs: vec![],
            })
        };
        assert_ne!(
            stage_key(&uv_iv, Stage::Epilogue).unwrap(),
            stage_key(&iv_uv, Stage::Epilogue).unwrap(),
            "differently typed referenced channels yield different keys"
        );
    }
}
