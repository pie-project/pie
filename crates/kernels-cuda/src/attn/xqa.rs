//! The tensor-core paged decode (`attn/xqa.cuh`): a whole GQA group's rows
//! as one MMA column tile, read off the FA2 decode plan and landing what the
//! FA2 decode kernel lands, so the same fold follows it. Taken where the
//! profile says (`kernel.sdpa.decode.xqa`), for groups of at most eight
//! query heads over a plain bf16 NHD pool, at the head widths it is stamped
//! for.

use dtype::Dtype;

use crate::attn::fa2_abi::DecodeParams;
use crate::attn::plan::DecodePlan;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, symbol};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/xqa.cuh";

/// One head width's stamp: key tiles of `16 * mt` rows, `warps` warps a
/// block, `stages` tiles in flight a warp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub head_dim: u32,
    pub mt: u32,
    pub warps: u32,
    pub stages: u32,
}

impl Stamp {
    /// The widths measured on the B200, each at the smallest block that
    /// keeps one tile in flight a warp beside the one it reads, so several
    /// blocks share an SM: Gemma 4's 256-wide sliding layers (two query heads
    /// a KV head) and gpt-oss's 64-wide groups of eight (from
    /// `kernel.sdpa.decode.xqa_from_64` requests: narrower, the FA2 decode
    /// reads them faster). Gemma's 512-wide global layers read faster
    /// through the prefill kernel.
    #[must_use]
    pub const fn of(head_dim: u32) -> Option<Self> {
        let (mt, warps, stages) = match head_dim {
            64 => (1, 4, 2),
            256 => (1, 3, 2),
            _ => return None,
        };
        Some(Self {
            head_dim,
            mt,
            warps,
            stages,
        })
    }

    /// The blocks a split decode aims for: as many as the SMs hold at once.
    #[must_use]
    pub fn grid(self, device: &crate::attn::plan::Device) -> u32 {
        let per_sm = (device.max_smem_per_sm / (self.smem() + 1024)).max(1);
        device.num_sm.max(1) * per_sm
    }

    /// The dynamic shared memory a block holds (`Shape::SMEM_BYTES`).
    #[must_use]
    pub const fn smem(self) -> u32 {
        let head_dim = self.head_dim;
        let ld = head_dim + 8;
        let tile = 16 * self.mt;
        let staged = self.warps * self.stages * 2 * tile * ld * 2;
        let fold = self.warps * (8 * head_dim + 16) * 4;
        let region = if staged > fold { staged } else { fold };
        region + (8 * ld + self.warps * tile * 8) * 2
    }
}

/// The profile's reading of the tensor-core decode: whether it is on, and
/// the fewest requests its 64-wide stamp takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gate {
    pub on: bool,
    pub from_64: u32,
    /// The widest window (`window_left + 1` keys) read whole, unsplit.
    pub whole_window: u32,
    /// The most blocks a cluster splits an unwindowed row across (0: the
    /// plan splits it).
    pub cluster: u32,
}

impl Gate {
    #[must_use]
    pub const fn of(tuning: crate::tuning::Tuning) -> Self {
        Self {
            on: tuning.decode_xqa,
            from_64: tuning.decode_xqa_from_64,
            whole_window: tuning.decode_xqa_whole_window,
            cluster: tuning.decode_xqa_cluster,
        }
    }

    /// Whether a plan of this width and group may take the tensor-core
    /// decode at some batch (the pool's own checks aside).
    #[must_use]
    pub fn stamps(self, head_dim: u32, group: u32) -> bool {
        self.on && (1..=8).contains(&group) && Stamp::of(head_dim).is_some()
    }

    /// Whether a plan of this width, group and request count takes it.
    #[must_use]
    pub fn takes(self, head_dim: u32, group: u32, requests: u32) -> bool {
        self.stamps(head_dim, group) && (head_dim != 64 || requests >= self.from_64)
    }

    /// Whether a window of `window_left + 1` keys is read whole at any
    /// batch: one block a (request, kv head) lands its rows, with no split
    /// to fold (a short window's split folds dearer than its block reads).
    #[must_use]
    pub fn whole(self, head_dim: u32, group: u32, window_left: i32) -> bool {
        window_left >= 0
            && u32::try_from(window_left).is_ok_and(|left| left < self.whole_window)
            && self.stamps(head_dim, group)
    }

    /// The blocks a cluster reads a row with at any batch (each
    /// a slice of the row's keys, the cluster folding them), as many as fill
    /// `grid` blocks across the batch's (request, kv head) pairs, at most
    /// `cluster`; 0 where the plan splits the row instead.
    #[must_use]
    pub fn ranks(
        self,
        head_dim: u32,
        group: u32,
        requests: u32,
        kv_heads: u32,
        window_left: i32,
        grid: u32,
    ) -> u32 {
        // A windowed row only at 64-wide heads past the whole-read windows
        // (Muse Spark's 2048-key windows; Gemma 4's 256-wide sliding rows
        // measured slower so).
        let windowed =
            window_left >= 0 && (head_dim != 64 || self.whole(head_dim, group, window_left));
        if self.cluster == 0 || windowed || !self.stamps(head_dim, group) {
            return 0;
        }
        let pairs = requests.saturating_mul(kv_heads).max(1);
        let mut ranks = self.cluster.min(8);
        while ranks > 1 && pairs.saturating_mul(ranks) > grid {
            ranks /= 2;
        }
        ranks.max(1)
    }
}

/// Whether this decode takes the tensor-core kernel.
#[must_use]
pub fn takes(ctx: &Ctx, plan: &DecodePlan, pool: &KvPool, rel: bool) -> bool {
    let group = plan.shape.group_size();
    let head_dim = plan.shape.head_dim;
    (plan.info.whole || Gate::of(ctx.tuning()).takes(head_dim, group, plan.shape.num_requests))
        && !rel
        && plan.shape.num_kv_heads * group == plan.shape.num_q_heads
        && !plan.shape.hnd_layout
        && pool.keys.dtype == Dtype::Bf16
        && pool.values.dtype == Dtype::Bf16
        && Stamp::of(head_dim).is_some_and(|stamp| {
            ctx.smem_per_block_optin()
                .is_some_and(|cap| cap >= stamp.smem())
        })
}

/// Fires the tensor-core decode over `params` (the FA2 decode's, its output
/// already the split's partial plane where the plan splits). With `sinks`
/// and no split, the sinks scale the output in the same pass.
pub fn decode(
    ctx: &Ctx,
    op: &'static str,
    plan: &DecodePlan,
    params: &DecodeParams,
    sinks: Option<Tensor>,
) -> Result<(), Error> {
    let head_dim = plan.shape.head_dim;
    let stamp = Stamp::of(head_dim).ok_or_else(|| {
        crate::jit::refuse(
            op,
            format!("no tensor-core decode at head width {head_dim}"),
        )
    })?;
    let sink_ty = match sinks {
        Some(s) if s.dtype == Dtype::F32 => "float",
        _ => "::pie::bf16",
    };
    let kv = &params.paged_kv;
    let ranks = plan.info.ranks.max(1);
    let launch = Launch::grid(
        [plan.shape.num_kv_heads * ranks, params.padded_batch_size, 1],
        [32 * stamp.warps, 1, 1],
    );
    let launch = if ranks > 1 {
        launch
            .cluster(ranks)
            .smem(stamp.smem() + (8 * head_dim + 8) * 4)
    } else {
        launch.smem(stamp.smem())
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::xqa::decode<{head_dim}, {}, {}, {}, {sink_ty}>",
                stamp.mt, stamp.warps, stamp.stages
            )),
        )
        .apply(launch),
        &[
            ArgValue::Ptr(params.q),
            ArgValue::Ptr(kv.k_data),
            ArgValue::Ptr(kv.v_data),
            ArgValue::Ptr(kv.indices),
            ArgValue::Ptr(kv.indptr),
            ArgValue::Ptr(kv.last_page_len),
            plan.shape.page_size.arg(),
            kv.stride_page.arg(),
            kv.stride_n.arg(),
            kv.stride_h.arg(),
            ArgValue::Ptr(params.q_indptr),
            u32::try_from(params.q_stride_n).unwrap_or(0).arg(),
            ArgValue::Ptr(params.request_indices),
            ArgValue::Ptr(params.kv_tile_indices),
            ArgValue::Ptr(params.kv_chunk_size_ptr),
            ArgValue::Ptr(params.block_valid_mask),
            params.partition_kv.arg(),
            params.num_qo_heads.arg(),
            plan.shape.group_size().arg(),
            params.window_left.arg(),
            params.sm_scale.arg(),
            ArgValue::Ptr(params.o),
            ArgValue::Ptr(params.lse),
            ArgValue::Ptr(match sinks {
                Some(s) if !params.partition_kv => s.ptr,
                _ => 0,
            }),
            ranks.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each stamp fits a B200 block, the 64-wide an sm_86 one too, and only
    /// the stamped widths, groups of at most eight and, at 64 wide, batches of
    /// `xqa_from_64` requests take the kernel.
    #[test]
    fn each_stamp_states_what_its_block_holds() {
        assert!(Stamp::of(64).expect("stamped").smem() <= 101_376);
        assert!(Stamp::of(256).expect("stamped").smem() <= 232_448);
        assert_eq!(Stamp::of(512), None);
        let gate = Gate {
            on: true,
            from_64: 16,
            whole_window: 128,
            cluster: 8,
        };
        assert!(gate.takes(256, 2, 1) && gate.takes(64, 8, 16));
        assert!(!gate.takes(64, 8, 15) && !gate.takes(256, 16, 64) && !gate.takes(512, 8, 64));
        // A window of up to 128 keys reads whole at any batch; a wider one,
        // a full row or an unstamped head does not.
        assert!(gate.whole(64, 8, 127) && gate.whole(256, 2, 0));
        assert!(!gate.whole(64, 8, 128) && !gate.whole(64, 8, -1) && !gate.whole(512, 8, 0));
        // A row a cluster reads takes as many blocks as fill the grid, at
        // most eight; a 64-wide one under a long window too (Muse Spark's
        // 2048); a whole-read window, Gemma's 256-wide sliding row, or any
        // where it is off, none.
        assert_eq!(gate.ranks(64, 8, 8, 8, -1, 740), 8);
        assert_eq!(gate.ranks(64, 8, 16, 8, -1, 740), 4);
        assert_eq!(gate.ranks(64, 8, 64, 8, -1, 740), 1);
        assert_eq!(gate.ranks(64, 8, 8, 8, 127, 740), 0);
        assert_eq!(gate.ranks(64, 8, 8, 16, 2047, 740), 4);
        assert_eq!(gate.ranks(256, 2, 1, 16, 1023, 740), 0);
        assert_eq!(gate.ranks(64, 8, 1, 8, -1, 740), 8);
        assert_eq!(Gate { cluster: 0, ..gate }.ranks(64, 8, 8, 8, -1, 740), 0);
        assert!(gate.stamps(64, 8));
        assert!(!Gate::default().stamps(256, 2));
    }
}
