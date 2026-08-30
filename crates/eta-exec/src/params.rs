use eta_compiler::codegen::launch::LaunchOp;
use eta_ir::op::IntrinsicId;
use eta_ir::op::tags;
use eta_ir::types::wire_dtype;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Runtime {
    pub vocab: u32,

    pub mtp_draft_row: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct OpParams {
    pub tag: u32,

    pub a0: u32,

    pub a1: u32,

    pub a2: u32,

    pub o0: u32,

    pub o1: u32,

    pub imm: u32,

    pub imm2: u32,

    pub imm3: u32,

    pub kind: u32,

    pub pred_tag: u32,

    pub lit_dtype: u32,

    pub lit_bits: u32,

    pub channel_slot: u32,

    pub intr: u32,

    pub sink_bytes: u32,
}

const _: () = assert!(size_of::<OpParams>() == 64);

const NO_CHANNEL: u32 = 0;

impl OpParams {
    #[must_use]
    pub fn of(op: &LaunchOp, result_base: u32, runtime: Runtime) -> OpParams {
        let tag = op.tag;
        let intrinsic = tag == tags::INTRINSIC_VAL;
        let a0 = op.args.first().copied().unwrap_or(0);

        let a1 = op
            .args
            .get(1)
            .copied()
            .unwrap_or(if tag == tags::PIVOT_THRESHOLD {
                op.pred_payload
            } else {
                0
            });

        let o0 = if op.result_count > 0 { result_base } else { a0 };
        OpParams {
            tag: u32::from(op.tag),
            a0,
            a1,
            a2: op.args.get(2).copied().unwrap_or(0),
            o0,

            o1: if op.result_count > 1 {
                o0.saturating_add(1)
            } else {
                o0
            },
            imm: if intrinsic { runtime.vocab } else { op.imm },
            imm2: match runtime.mtp_draft_row {
                Some(row) if intrinsic && op.intrinsic.is_some_and(is_mtp) => row,
                _ => op.imm2,
            },
            imm3: op.imm3,
            kind: op.rng_kind as u32,
            pred_tag: u32::from(op.pred_tag),
            lit_dtype: u32::from(wire_dtype(op.lit_dtype)),
            lit_bits: op.lit_bits,
            // The device word keeps its own sentinel — `NO_CHANNEL = 0`, not
            // `u32::MAX` — so the contract's `Option` unwraps into it here and
            // the kernel-side encoding is byte-identical.
            channel_slot: op.channel.unwrap_or(NO_CHANNEL),
            intr: op.intrinsic.map_or(0, |intrinsic| intrinsic as u32),
            sink_bytes: 0,
        }
    }

    #[must_use]
    pub fn binds_second_argument(op: &LaunchOp) -> bool {
        op.args.len() > 1 || op.tag == tags::PIVOT_THRESHOLD
    }
}

fn is_mtp(intrinsic: IntrinsicId) -> bool {
    matches!(intrinsic, IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
}
