//! Per-stage batch keys (D5) — the normalization that captures "same stage
//! program" across passes, post-bind and host-side.
//!
//! The pass-level identity is the C3 container hash ([`crate::container_hash`]).
//! It is the wrong grain for stage co-batching: two passes may share an
//! identical epilogue (or prologue, …) yet hash differently, and — because a
//! stage's ops reference the pass-global dense channel table — raw byte-equality
//! of one [`StageProgram`] section does NOT recognize "same epilogue" across
//! passes (the same op graph lands on different global channel indices, and the
//! decls it references sit at different positions).
//!
//! [`stage_key`] is that recognizer. For one stage it builds a canonical form
//! from three parts and hashes it with the same FNV-1a as the container:
//!
//! 1. the stage's ops, with channel references renumbered into stage-local
//!    first-use order (the first channel touched becomes local 0, the next new
//!    one local 1, …), so pass-global numbering drops out;
//! 2. each referenced channel's resolved signature (element type with `ACT`
//!    materialized, capacity, host-role, seeded) in that same local order — so
//!    two stages that renumber identically but read differently typed channels
//!    stay distinct;
//! 3. the stage body's resolved SSA value types (the sidecar's per-op types) —
//!    so a difference the op tags alone miss (a model specialization) still
//!    separates the keys.
//!
//! Contract: structurally identical stages (same ops up to channel renumbering,
//! same referenced-channel signatures, same resolved types) share a key; any
//! op, signature, or type difference yields a different key. Keys never cross
//! the wasm boundary (D4-orthogonal): they are a runtime-derived scheduler grain
//! over an already-bound trace.

use alloc::vec::Vec;

use crate::container::{encode_op, encode_shape, put_u32};
use crate::op::Op;
use crate::registry::Stage;
use crate::types::ValueType;
use crate::validate::BoundTrace;

/// Domain separator so a stage key can never collide with a container hash or a
/// sidecar (distinct byte prefix into the shared FNV-1a).
const PTSK_MAGIC: [u8; 4] = *b"PTSK";
const PTSK_VERSION: u16 = 1;

/// The D5 per-stage batch key for `stage`, or `None` if the bound trace has no
/// program for that stage. See the module docs for the normalization.
pub fn stage_key(bound: &BoundTrace, stage: Stage) -> Option<u64> {
    let idx = bound
        .container
        .stages
        .iter()
        .position(|s| s.stage == stage)?;
    Some(stage_key_at(bound, idx))
}

/// The D5 per-stage batch key for the stage at container position `stage_idx`.
///
/// Panics if `stage_idx` is out of range for the bound trace's stages.
pub fn stage_key_at(bound: &BoundTrace, stage_idx: usize) -> u64 {
    let sp = &bound.container.stages[stage_idx];
    let value_types: &[ValueType] = &bound.stage_types[stage_idx];

    // 1. Stage-local first-use renumbering of channel references. `order` is
    // local -> global; `local_id` assigns on first use.
    let mut order: Vec<u32> = Vec::new();

    let mut w = Vec::new();
    w.extend_from_slice(&PTSK_MAGIC);
    w.extend_from_slice(&PTSK_VERSION.to_le_bytes());
    w.push(sp.stage as u8);

    // 2. Ops, with channel indices renumbered to stage-local first-use order.
    put_u32(&mut w, sp.ops.len() as u32);
    for op in &sp.ops {
        match op {
            Op::ChanTake(c) => encode_op(&mut w, &Op::ChanTake(local_id(&mut order, *c))),
            Op::ChanRead(c) => encode_op(&mut w, &Op::ChanRead(local_id(&mut order, *c))),
            Op::ChanPut { chan, value } => encode_op(
                &mut w,
                &Op::ChanPut {
                    chan: local_id(&mut order, *chan),
                    value: *value,
                },
            ),
            other => encode_op(&mut w, other),
        }
    }

    // 3. Referenced-channel signatures in local order: resolved element type
    // (ACT materialized), capacity, host-role, seeded.
    put_u32(&mut w, order.len() as u32);
    for &g in &order {
        let vt = bound.channel_types[g as usize];
        let decl = &bound.container.channels[g as usize];
        encode_value_type(&mut w, vt);
        put_u32(&mut w, decl.capacity);
        w.push(decl.host_role as u8);
        w.push(decl.seeded as u8);
    }

    // 4. Resolved SSA value types (sidecar grain).
    put_u32(&mut w, value_types.len() as u32);
    for vt in value_types {
        encode_value_type(&mut w, *vt);
    }

    crate::container_hash(&w)
}

fn encode_value_type(w: &mut Vec<u8>, vt: ValueType) {
    w.push(vt.dtype as u8);
    encode_shape(w, vt.shape);
}

/// Local id for global channel `g` in stage-local first-use order (assigning on
/// first sight; `order` maps local -> global).
fn local_id(order: &mut Vec<u32>, g: u32) -> u32 {
    match order.iter().position(|&x| x == g) {
        Some(l) => l as u32,
        None => {
            order.push(g);
            (order.len() - 1) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use crate::registry::ModelProfile;
    use crate::types::{DType, Literal, Shape};
    use crate::validate::bind;
    use alloc::vec;

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
