//! The tensor-core paged decode (`attn/decode_gqa_sm80.cu`): a whole GQA group's rows
//! as one MMA column tile, read off the FA2 decode plan and landing what the
//! FA2 decode kernel lands, so the same fold follows it. Taken where the
//! profile says (`kernel.sdpa.decode.gqa`), for groups of at most eight
//! query heads over a plain bf16 NHD pool, at the head widths it is stamped
//! for.

use dtype::Dtype;

use crate::attn::fa2::abi::DecodeParams;
use crate::attn::plan::DecodePlan;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/decode_gqa_sm80.cu";

/// One head width's stamp: key tiles of `16 * mt` rows, `warps` warps a
/// block, `stages` tiles in flight a warp, its key rows swizzled rather than
/// padded (`swizzle`, a bf16 pool's alone).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub head_dim: u32,
    pub mt: u32,
    pub warps: u32,
    pub stages: u32,
    pub swizzle: bool,
}

impl Stamp {
    /// The widths measured on the B200, each at the smallest block that
    /// keeps one tile in flight a warp beside the one it reads, so several
    /// blocks share an SM: Gemma 4's 256-wide sliding layers (two query heads
    /// a KV head) and gpt-oss's 64-wide groups of eight (from
    /// `kernel.sdpa.decode.gqa_from_64` requests: narrower, the FA2 decode
    /// reads them faster). Gemma's 512-wide global layers read faster
    /// through the prefill kernel.
    #[must_use]
    pub const fn of(head_dim: u32) -> Option<Self> {
        let (mt, warps, stages) = match head_dim {
            64 | 128 => (1, 4, 2),
            256 => (1, 3, 2),
            _ => return None,
        };
        Some(Self {
            head_dim,
            mt,
            warps,
            stages,
            swizzle: false,
        })
    }

    /// The 64-wide stamp three tiles deep a warp, its key rows swizzled so
    /// four blocks still share an SM: a decode's two or three tiles a warp
    /// all in flight at once (`kernel.sdpa.decode.gqa_deep_64`).
    pub const DEEP_64: Self = Self {
        head_dim: 64,
        mt: 1,
        warps: 4,
        stages: 3,
        swizzle: true,
    };

    /// The blocks a split decode aims for: as many as the SMs hold at once.
    #[must_use]
    pub fn grid(self, device: &crate::attn::plan::Device) -> u32 {
        let per_sm = (device.max_smem_per_sm / (self.smem() + 1024)).max(1);
        device.num_sm.max(1) * per_sm
    }

    /// The dynamic shared memory a block holds (`Shape::SMEM_BYTES`).
    #[must_use]
    pub const fn smem(self) -> u32 {
        self.smem_over(false)
    }

    /// [`Stamp::smem`] over an e4m3 pool's pages, staged raw.
    #[must_use]
    pub const fn smem_over(self, e4m3: bool) -> u32 {
        let head_dim = self.head_dim;
        let ld = head_dim + 8;
        let kv_ld = if self.swizzle { head_dim } else { ld };
        let tile = 16 * self.mt;
        let staged = if e4m3 {
            self.warps * self.stages * 2 * tile * head_dim
        } else {
            self.warps * self.stages * 2 * tile * kv_ld * 2
        };
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
    /// 64-wide heads over a bf16 pool take [`Stamp::DEEP_64`]
    /// (`kernel.sdpa.decode.gqa_deep_64`).
    pub deep_64: bool,
    /// The fewest query heads a KV head (`kernel.sdpa.decode.gqa_min_group`).
    pub min_group: u32,
}

impl Gate {
    #[must_use]
    pub fn of(tuning: crate::tuning::Tuning) -> Self {
        Self {
            on: tuning.decode_gqa(),
            from_64: tuning.decode_gqa_from_64(),
            whole_window: tuning.decode_gqa_whole_window(),
            cluster: tuning.decode_gqa_cluster(),
            deep_64: tuning.decode_gqa_deep_64(),
            min_group: tuning.decode_gqa_min_group(),
        }
    }

    /// The stamp a head width takes, over an e4m3 pool or a bf16 one.
    #[must_use]
    pub const fn stamp(self, head_dim: u32, e4m3: bool) -> Option<Stamp> {
        if self.deep_64 && head_dim == 64 && !e4m3 {
            Some(Stamp::DEEP_64)
        } else if e4m3 {
            match Stamp::of(head_dim) {
                Some(stamp) => Some(Stamp {
                    stages: if head_dim > 128 { 3 } else { 4 },
                    ..stamp
                }),
                None => None,
            }
        } else {
            Stamp::of(head_dim)
        }
    }

    /// Whether a plan of this width and group may take the tensor-core
    /// decode at some batch (the pool's own checks aside).
    #[must_use]
    pub fn stamps(self, head_dim: u32, group: u32) -> bool {
        self.on && (self.min_group.max(1)..=8).contains(&group) && Stamp::of(head_dim).is_some()
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
        // Past 64 blocks, eight slices a row fold slower than four.
        while ranks > 4 && pairs.saturating_mul(ranks) > 64 {
            ranks /= 2;
        }
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
        && (pool.keys.dtype == Dtype::Bf16 && pool.values.dtype == Dtype::Bf16
            || crate::attn::kv::native_e4m3(pool))
        && Gate::of(ctx.tuning())
            .stamp(head_dim, crate::attn::kv::native_e4m3(pool))
            .is_some_and(|stamp| {
                ctx.smem_per_block_optin()
                    .is_some_and(|cap| cap >= stamp.smem_over(crate::attn::kv::native_e4m3(pool)))
            })
}

/// Fires the tensor-core decode over `params` (the FA2 decode's, its output
/// already the split's partial plane where the plan splits; its masks, where
/// it carries the fire's, read per request). With `sinks`
/// and no split, the sinks scale the output in the same pass. `e4m3` reads
/// the pages as an e4m3 pool's.
pub fn decode(
    ctx: &Ctx,
    op: &'static str,
    plan: &DecodePlan,
    params: &DecodeParams,
    sinks: Option<Tensor>,
    e4m3: bool,
) -> Result<(), Error> {
    let head_dim = plan.shape.head_dim;
    let gate = Gate::of(ctx.tuning());
    let stamp = gate.stamp(head_dim, e4m3).ok_or_else(|| {
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
    )
    .programmatic(ctx.tuning().pdl());
    let smem = stamp.smem_over(e4m3);
    let launch = if ranks > 1 {
        launch.cluster(ranks).smem(smem + (8 * head_dim + 8) * 4)
    } else {
        launch.smem(smem)
    };
    // The defaulted parameters spelled out up to the last one set.
    let pages = if e4m3 { ", ::pie::u8" } else { "" };
    // The swizzled stamp (a bf16 pool's) has its own entry, its registers
    // held to the four blocks its shared memory lets share an SM.
    let entry = if stamp.swizzle {
        format!(
            "::pie::attn::decode_deep<{head_dim}, {}, {}, {}, 4, {sink_ty}>",
            stamp.mt, stamp.warps, stamp.stages
        )
    } else {
        format!(
            "::pie::attn::decode<{head_dim}, {}, {}, {}, {sink_ty}{pages}>",
            stamp.mt, stamp.warps, stamp.stages
        )
    };
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &entry)).apply(launch),
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
            ArgValue::Ptr(params.maybe_custom_mask),
            ArgValue::Ptr(params.maybe_mask_indptr),
            ArgValue::Ptr(params.maybe_mask_origin),
            ranks.arg(),
        ],
    )
}
