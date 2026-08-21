use core::ffi::c_void;
use std::fmt;

use crate::jit::{ArgValue, Ctx, Launch};
use kernels::Fire;
use kernels::Refusal;

pub const CAN_LAUNCH: bool = true;

pub const VEC_SIZE: i32 = 8;

pub const CLUSTER_SIZE: i32 = 1;

pub const MAX_BLOCK_THREADS: i32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum FusionPattern {
    AllReduce = 0,
    ARResidualRMSNorm = 1,
    ARResidualRMSNormFp8Quant = 2,
    ARResidualRMSNormFp4Quant = 3,
    ARResidualRMSNormOutFp8Quant = 4,
    ARResidualRMSNormOutFp4Quant = 5,
    ARResidualRMSNormPerTokenGroupFp8PackedQuant = 8,
    ARResidualRMSNormOutPerTokenGroupFp8PackedQuant = 9,
}

impl FusionPattern {
    pub const ALL: &'static [Self] = &[
        Self::AllReduce,
        Self::ARResidualRMSNorm,
        Self::ARResidualRMSNormFp8Quant,
        Self::ARResidualRMSNormFp4Quant,
        Self::ARResidualRMSNormOutFp8Quant,
        Self::ARResidualRMSNormOutFp4Quant,
        Self::ARResidualRMSNormPerTokenGroupFp8PackedQuant,
        Self::ARResidualRMSNormOutPerTokenGroupFp8PackedQuant,
    ];

    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AllReduce => "kAllReduce",
            Self::ARResidualRMSNorm => "kARResidualRMSNorm",
            Self::ARResidualRMSNormFp8Quant => "kARResidualRMSNormFp8Quant",
            Self::ARResidualRMSNormFp4Quant => "kARResidualRMSNormFP4Quant",
            Self::ARResidualRMSNormOutFp8Quant => "kARResidualRMSNormOutFP8Quant",
            Self::ARResidualRMSNormOutFp4Quant => "kARResidualRMSNormOutFP4Quant",
            Self::ARResidualRMSNormPerTokenGroupFp8PackedQuant => {
                "kARResidualRMSNormPerTokenGroupFP8PackedQuant"
            }
            Self::ARResidualRMSNormOutPerTokenGroupFp8PackedQuant => {
                "kARResidualRMSNormOutPerTokenGroupFP8PackedQuant"
            }
        }
    }

    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.iter().copied().find(|p| p.code() == code)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SfLayout {
    Swizzled128x4 = 0,
    Swizzled8x4 = 1,
    Linear = 2,
}

pub static INSTANTIATED: &[FusionPattern] = &[FusionPattern::ARResidualRMSNorm];

pub static NRANKS: &[i32] = &[2, 4, 8, 16];

pub static PLAIN_NRANKS: &[i32] = &[2, 4, 6, 8];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leaf {
    OneShot,
    OneShotTriggerAtEnd,
    TwoShot,
}

impl Leaf {
    #[must_use]
    pub const fn of(use_oneshot: bool, trigger_completion_at_end: bool) -> Self {
        if !use_oneshot {
            Self::TwoShot
        } else if trigger_completion_at_end {
            Self::OneShotTriggerAtEnd
        } else {
            Self::OneShot
        }
    }

    #[must_use]
    pub const fn oneshot(self) -> bool {
        !matches!(self, Self::TwoShot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    OneStage,
    TwoStage,
}

pub const LEAVES: usize = 3;

pub const FP32_ACC_VALUES: usize = 2;

pub const UPSTREAM_POINTS: usize = 240;

pub const AOT_TU_SECONDS: usize = 111;

pub const AOT_CICC_SECONDS: usize = 40;

pub const AOT_PTXAS_SECONDS: usize = 44;

pub const AOT_POINTS_AFTER_PRUNING: usize =
    NRANKS.len() * INSTANTIATED.len() * FP32_ACC_VALUES * LEAVES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instantiation {
    pub nranks: i32,
    pub pattern: FusionPattern,
    pub fp32_acc: bool,
    pub leaf: Leaf,
}

impl Instantiation {
    #[must_use]
    pub fn name_expression(&self) -> Option<&'static str> {
        match (self.nranks, self.fp32_acc, self.leaf) {
            (2, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true, false>",
            ),
            (2, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true, true>",
            ),
            (2, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true>",
            ),
            (2, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false, false>",
            ),
            (2, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false, true>",
            ),
            (2, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false>",
            ),
            (4, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true, false>",
            ),
            (4, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true, true>",
            ),
            (4, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true>",
            ),
            (4, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false, false>",
            ),
            (4, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false, true>",
            ),
            (4, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false>",
            ),
            (8, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true, false>",
            ),
            (8, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true, true>",
            ),
            (8, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true>",
            ),
            (8, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false, false>",
            ),
            (8, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false, true>",
            ),
            (8, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false>",
            ),
            (16, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true, false>",
            ),
            (16, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true, true>",
            ),
            (16, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true>",
            ),
            (16, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false, false>",
            ),
            (16, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false, true>",
            ),
            (16, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false>",
            ),
            _ => None,
        }
    }
}

pub const REACHED: Instantiation = Instantiation {
    nranks: 2,
    pattern: FusionPattern::ARResidualRMSNorm,
    fp32_acc: true,
    leaf: Leaf::OneShot,
};

pub fn resolve(
    nranks: i32,
    pattern: FusionPattern,
    fp32_acc: bool,
    use_oneshot: bool,
    trigger_completion_at_end: bool,
) -> std::result::Result<Instantiation, Decline> {
    if !NRANKS.contains(&nranks) {
        return Err(Decline::WorldSizeUnsupported { nranks });
    }
    if !INSTANTIATED.contains(&pattern) {
        return Err(Decline::PatternNotInstantiated {
            code: pattern.code(),
        });
    }
    Ok(Instantiation {
        nranks,
        pattern,
        fp32_acc,
        leaf: Leaf::of(use_oneshot, trigger_completion_at_end),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    NoInstance,
    NotInitialised,
    NullInput,
    Bytes {
        bytes: usize,
        max_bytes: usize,
    },
    Vector {
        count: usize,
        width: i32,
    },
    NotFullyConnected {
        world_size: i32,
    },
    CaptureUnknown,
    Unregistered,
    AboveCrossover {
        bytes: usize,
        crossover: usize,
        world_size: i32,
    },
    NoFusionWorkspace,
    FusionTokens {
        tokens: i32,
        max_tokens: i32,
    },
    FusionHidden {
        hidden: i32,
        want: i32,
    },
    FusionWorldSize {
        world_size: i32,
    },
    FusionHiddenNotOctet {
        hidden: i32,
    },
    FusionBlockWidth {
        hidden: i32,
        threads: i32,
        max: i32,
    },
    FusionBlockNarrow {
        threads: i32,
        nranks: i32,
    },
    PatternNotInstantiated {
        code: i32,
    },
    WorldSizeUnsupported {
        nranks: i32,
    },
    NoTemplateId {
        nranks: i32,
    },
    DeviceQuery {
        what: &'static str,
    },
    Launch(Refusal),
    FellBack(Refusal),
}

impl fmt::Display for Decline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoInstance => write!(
                f,
                "this deployment configured no custom all-reduce, and the P2P \
                 reduction is stated; there is no other way to spell it"
            ),
            Self::NotInitialised => write!(f, "the custom all-reduce is not initialised"),
            Self::NullInput => write!(f, "`input` is null"),
            Self::Bytes { bytes, max_bytes } => write!(
                f,
                "{bytes} bytes is zero, above the {max_bytes}-byte ceiling, or not a multiple of 16"
            ),
            Self::Vector { count, width } => write!(
                f,
                "{count} elements is zero or not a multiple of {width}, the kernel's 16-byte \
                 vector width in bf16"
            ),
            Self::NotFullyConnected { world_size } => write!(
                f,
                "world size {world_size} needs peer access between every ordered pair and does not \
                 have it"
            ),
            Self::CaptureUnknown => write!(f, "`cudaStreamIsCapturing` failed on this stream"),
            Self::Unregistered => {
                write!(
                    f,
                    "the input's base allocation was never passed to `register_buffer`"
                )
            }
            Self::AboveCrossover {
                bytes,
                crossover,
                world_size,
            } => write!(
                f,
                "{bytes} bytes is at or above the {crossover}-byte crossover for world size \
                 {world_size}; NCCL wins on bandwidth here"
            ),
            Self::NoFusionWorkspace => write!(
                f,
                "no fusion workspace was built (world size 2 with a positive `fusion_max_tokens` \
                 and `fusion_hidden` is what builds one)"
            ),
            Self::FusionTokens { tokens, max_tokens } => {
                write!(
                    f,
                    "{tokens} tokens against a workspace sized for {max_tokens}"
                )
            }
            Self::FusionHidden { hidden, want } => {
                write!(
                    f,
                    "hidden {hidden} against a workspace sized for exactly {want}"
                )
            }
            Self::FusionWorldSize { world_size } => {
                write!(
                    f,
                    "the fused landing is world size 2 only; this group is {world_size}"
                )
            }
            Self::FusionHiddenNotOctet { hidden } => {
                write!(f, "hidden {hidden} is not a multiple of 8")
            }
            Self::FusionBlockWidth {
                hidden,
                threads,
                max,
            } => write!(
                f,
                "hidden {hidden} needs {threads} threads in one block and a block holds {max}; \
                 upstream would have spread this token over a cluster, and \
                 `comm::CLUSTER_SIZE` is pinned to 1"
            ),
            Self::FusionBlockNarrow { threads, nranks } => write!(
                f,
                "the two-shot fused kernel needs at least one thread per rank: {threads} threads \
                 for a world size of {nranks}"
            ),
            Self::PatternNotInstantiated { code } => write!(
                f,
                "`AllReduceFusionPattern` {code} is not in `comm::INSTANTIATED`; \
                 adding a pattern to a call site requires adding it there and to `comm::inst`"
            ),
            Self::WorldSizeUnsupported { nranks } => write!(
                f,
                "TP world size {nranks} is not instantiated (flashinfer's fused landing takes \
                 2, 4, 8, 16; vllm's plain reduction takes 2, 4, 6, 8)"
            ),
            Self::NoTemplateId { nranks } => write!(
                f,
                "`comm::inst` carries no template-id for world size {nranks}; the table and \
                 `comm::resolve` disagree"
            ),
            Self::DeviceQuery { what } => write!(f, "the driver would not say {what}"),
            Self::Launch(why) => write!(f, "the launch refused: {why}"),
            Self::FellBack(why) => {
                write!(f, "this message was NCCL's and NCCL refused it: {why}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum AllReduce {
    Launched,
    Declined(Decline),
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FusionParams {
    pub nranks: i32,
    pub rank: i32,
    pub size: i32,
    pub hidden_dim: i32,
    pub workspace: *mut *mut c_void,
    pub allreduce_in: *mut c_void,
    pub allreduce_out: *mut c_void,
    pub residual_in: *mut c_void,
    pub residual_out: *mut c_void,
    pub norm_out: *mut c_void,
    pub quant_out: *mut c_void,
    pub scale_out: *mut c_void,
    pub rms_gamma: *mut c_void,
    pub rms_eps: f32,
    pub weight_bias: f32,
    pub scale_factor: *mut f32,
    pub use_oneshot: bool,
    pub layout: SfLayout,
    pub stream: *mut c_void,
    pub pattern: FusionPattern,
    pub trigger_completion_at_end: bool,
    pub block_quant_group_size: i32,
    pub tma_aligned_mn: i32,
}

impl FusionParams {
    pub fn instantiation(&self, use_fp32_acc: bool) -> std::result::Result<Instantiation, Decline> {
        resolve(
            self.nranks,
            self.pattern,
            use_fp32_acc,
            self.use_oneshot,
            self.trigger_completion_at_end,
        )
    }

    fn arg(&self) -> ArgValue {
        ArgValue::Bytes {
            ptr: std::ptr::from_ref(self).cast::<u8>(),
            len: core::mem::size_of::<Self>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub grid: u32,
    pub block: u32,
}

pub fn fusion_geometry(
    tokens: i32,
    hidden: i32,
    nranks: i32,
    leaf: Leaf,
    multiprocessors: u32,
) -> std::result::Result<Geometry, Decline> {
    if tokens <= 0 {
        return Err(Decline::FusionTokens {
            tokens,
            max_tokens: 0,
        });
    }
    if hidden <= 0 || hidden % VEC_SIZE != 0 {
        return Err(Decline::FusionHiddenNotOctet { hidden });
    }

    let threads_per_token = hidden / VEC_SIZE;
    let threads_per_block = threads_per_token / CLUSTER_SIZE;
    if threads_per_block > MAX_BLOCK_THREADS {
        return Err(Decline::FusionBlockWidth {
            hidden,
            threads: threads_per_block,
            max: MAX_BLOCK_THREADS,
        });
    }

    if !leaf.oneshot() && threads_per_block < nranks {
        return Err(Decline::FusionBlockNarrow {
            threads: threads_per_block,
            nranks,
        });
    }

    let cluster_num = if leaf.oneshot() {
        tokens
    } else {
        let per_rank = tokens / nranks;
        per_rank + i32::from(tokens % nranks != 0)
    };

    let sm_count = i32::try_from(multiprocessors).unwrap_or(i32::MAX);
    let grid = cluster_num.min(sm_count).max(1);
    Ok(Geometry {
        grid: u32::try_from(grid).unwrap_or(1),
        block: u32::try_from(threads_per_block).unwrap_or(1),
    })
}

#[must_use]
pub fn twoshot_split(tokens: i32, nranks: i32) -> ([i32; 16], [i32; 16]) {
    let mut begin = [0i32; 16];
    let mut count = [0i32; 16];
    let per_rank = tokens / nranks;
    let remaining = tokens % nranks;
    for r in 0..nranks.clamp(0, 16) {
        let at = r as usize;
        begin[at] = r * per_rank + remaining.min(r);
        count[at] = per_rank + i32::from(remaining > r);
    }
    (begin, count)
}

pub fn plain_geometry(
    count: usize,
    world_size: i32,
    fully_connected: bool,
) -> std::result::Result<(Geometry, Stage, i32), Decline> {
    pub const ALL_REDUCE_THREADS: i32 = 512;

    pub const MAX_BLOCKS: i32 = 36;

    let width = usize::try_from(VEC_SIZE).unwrap_or(8);
    if count == 0 || !count.is_multiple_of(width) {
        return Err(Decline::Vector {
            count,
            width: VEC_SIZE,
        });
    }
    if !PLAIN_NRANKS.contains(&world_size) {
        return Err(Decline::WorldSizeUnsupported { nranks: world_size });
    }

    let vectors = count / width;
    let bytes = vectors * 16;
    let size = i32::try_from(vectors).unwrap_or(i32::MAX);

    let stage = if world_size == 2 {
        Stage::OneStage
    } else if !fully_connected {
        return Err(Decline::NotFullyConnected { world_size });
    } else if (world_size <= 4 && bytes < 512 * 1024) || (world_size <= 8 && bytes < 256 * 1024) {
        Stage::OneStage
    } else {
        Stage::TwoStage
    };

    let threads = ALL_REDUCE_THREADS;
    let blocks = MAX_BLOCKS
        .min(size.div_euclid(threads) + i32::from(size % threads != 0))
        .max(1);
    Ok((
        Geometry {
            grid: u32::try_from(blocks).unwrap_or(1),
            block: u32::try_from(threads).unwrap_or(512),
        },
        stage,
        size,
    ))
}

#[must_use]
pub fn plain_name_expression(world_size: i32, stage: Stage) -> Option<&'static str> {
    match (world_size, stage) {
        (2, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>"),
        (2, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 2>"),
        (4, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 4>"),
        (4, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 4>"),
        (6, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 6>"),
        (6, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 6>"),
        (8, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>"),
        (8, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 8>"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FusionPlane {
    pub workspace: *mut c_void,
    pub max_tokens: i32,
    pub hidden: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PeerPlane {
    pub signals: [*mut c_void; 8],
    pub self_signal: *mut c_void,
    pub rank_data: *mut c_void,
    pub fully_connected: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub world_size: i32,
    pub rank: i32,
    pub fusion: Option<FusionPlane>,
    pub peers: PeerPlane,
}

impl Plane {
    pub fn can_fuse_residual_rmsnorm(
        &self,
        tokens: i32,
        hidden: i32,
    ) -> std::result::Result<(), Decline> {
        let Some(fusion) = self.fusion.as_ref() else {
            return Err(Decline::NoFusionWorkspace);
        };

        if tokens <= 0 || tokens > fusion.max_tokens {
            return Err(Decline::FusionTokens {
                tokens,
                max_tokens: fusion.max_tokens,
            });
        }

        if hidden != fusion.hidden {
            return Err(Decline::FusionHidden {
                hidden,
                want: fusion.hidden,
            });
        }

        if self.world_size != 2 {
            return Err(Decline::FusionWorldSize {
                world_size: self.world_size,
            });
        }

        if hidden % VEC_SIZE != 0 {
            return Err(Decline::FusionHiddenNotOctet { hidden });
        }
        Ok(())
    }
}

pub fn all_reduce_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
) -> AllReduce {
    match plain_all_reduce_bf16(ctx, input, output, count) {
        AllReduce::Launched => AllReduce::Launched,
        AllReduce::Declined(why) => fall_back_out_of_place(ctx, input, output, count, why),
    }
}

pub fn fall_back_out_of_place(
    ctx: &Ctx<'_>,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    why: Decline,
) -> AllReduce {
    let elems = i64::try_from(count).unwrap_or(i64::MAX);
    match crate::dist::all_reduce_out_of_place(ctx, input, output, elems) {
        Ok(()) => AllReduce::Launched,
        Err(nccl) => AllReduce::Declined(match why {
            Decline::AboveCrossover { .. }
            | Decline::Bytes { .. }
            | Decline::NotFullyConnected { .. } => Decline::FellBack(nccl),
            structural => structural,
        }),
    }
}

fn plain_all_reduce_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
) -> AllReduce {
    if input.is_null() {
        return AllReduce::Declined(Decline::NullInput);
    }

    let Ok(plane) = ctx.comm() else {
        return AllReduce::Declined(Decline::NoInstance);
    };

    if plane.peers.self_signal.is_null() {
        return AllReduce::Declined(Decline::NotInitialised);
    }

    if plane.peers.rank_data.is_null() {
        return AllReduce::Declined(Decline::Unregistered);
    }

    let (geometry, stage, size) =
        match plain_geometry(count, plane.world_size, plane.peers.fully_connected) {
            Ok(what) => what,
            Err(decline) => return AllReduce::Declined(decline),
        };
    let Some(instantiation) = plain_name_expression(plane.world_size, stage) else {
        return AllReduce::Declined(Decline::NoTemplateId {
            nranks: plane.world_size,
        });
    };

    let signals = plane.peers.signals;
    let signals_arg = ArgValue::Bytes {
        ptr: std::ptr::from_ref(&signals).cast::<u8>(),
        len: core::mem::size_of::<[*mut c_void; 8]>(),
    };

    let fired = ctx.fire(
        Fire::at("comm/all_reduce.cuh", instantiation)
            .apply(Launch::grid([geometry.grid, 1, 1], [geometry.block, 1, 1])),
        &[
            ArgValue::Ptr(plane.peers.rank_data),
            signals_arg,
            ArgValue::Ptr(plane.peers.self_signal),
            ArgValue::Ptr(output),
            ArgValue::I32(plane.rank),
            ArgValue::I32(size),
        ],
    );
    match fired {
        Ok(()) => AllReduce::Launched,
        Err(why) => AllReduce::Declined(Decline::Launch(why)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn all_reduce_residual_rmsnorm_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: i32,
    hidden: i32,
    eps: f32,
) -> AllReduce {
    let Ok(plane) = ctx.comm() else {
        return AllReduce::Declined(Decline::NoInstance);
    };
    if let Err(decline) = plane.can_fuse_residual_rmsnorm(tokens, hidden) {
        return AllReduce::Declined(decline);
    }
    let Some(fusion) = plane.fusion.as_ref() else {
        return AllReduce::Declined(Decline::NoFusionWorkspace);
    };

    let use_fp32_acc = true;

    let params = FusionParams {
        nranks: plane.world_size,
        rank: plane.rank,
        size: tokens.saturating_mul(hidden),
        hidden_dim: hidden,
        workspace: fusion.workspace.cast::<*mut c_void>(),
        allreduce_in: input.cast_mut(),
        allreduce_out: core::ptr::null_mut(),
        residual_in: residual_inout,
        residual_out: residual_inout,
        norm_out,
        quant_out: core::ptr::null_mut(),
        scale_out: core::ptr::null_mut(),
        rms_gamma: rms_gamma.cast_mut(),
        rms_eps: eps,
        weight_bias: 0.0,
        scale_factor: core::ptr::null_mut(),
        use_oneshot: true,
        layout: SfLayout::Swizzled128x4,
        stream: ctx.stream(),
        pattern: FusionPattern::ARResidualRMSNorm,
        trigger_completion_at_end: false,
        block_quant_group_size: 0,
        tma_aligned_mn: 0,
    };
    let point = match params.instantiation(use_fp32_acc) {
        Ok(point) => point,
        Err(decline) => return AllReduce::Declined(decline),
    };
    let Some(instantiation) = point.name_expression() else {
        return AllReduce::Declined(Decline::NoTemplateId {
            nranks: point.nranks,
        });
    };
    let multiprocessors = match ctx.multiprocessors() {
        Ok(count) => count,
        Err(_) => {
            return AllReduce::Declined(Decline::DeviceQuery {
                what: "how many multiprocessors this device has",
            });
        }
    };
    let geometry = match fusion_geometry(tokens, hidden, point.nranks, point.leaf, multiprocessors)
    {
        Ok(geometry) => geometry,
        Err(decline) => return AllReduce::Declined(decline),
    };
    let launch = Launch::grid([geometry.grid, 1, 1], [geometry.block, 1, 1]);

    let fired = if point.leaf.oneshot() {
        ctx.fire(
            Fire::at("comm/all_reduce.cuh", instantiation).apply(launch),
            &[params.arg()],
        )
    } else {
        let (begin, per_rank) = twoshot_split(tokens, point.nranks);
        let bytes = core::mem::size_of::<i32>()
            * usize::try_from(point.nranks).unwrap_or(0).min(begin.len());
        ctx.fire(
            Fire::at("comm/all_reduce.cuh", instantiation).apply(launch),
            &[
                params.arg(),
                ArgValue::Bytes {
                    ptr: begin.as_ptr().cast::<u8>(),
                    len: bytes,
                },
                ArgValue::Bytes {
                    ptr: per_rank.as_ptr().cast::<u8>(),
                    len: bytes,
                },
            ],
        )
    };
    match fired {
        Ok(()) => AllReduce::Launched,
        Err(why) => AllReduce::Declined(Decline::Launch(why)),
    }
}

const ALL_REDUCE_BF16_ROW: ::kernels::routine::Routine<crate::Plane> = ::kernels::untraced!(
    crate::Plane,
    "all_reduce_bf16",
    all_reduce_bf16,
    namespace = "comm",
    whole,
    driver
);

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice(crate::ROUTINES)]
static ALL_REDUCE_BF16_ROUTINE: ::kernels::routine::Routine<crate::Plane> = ALL_REDUCE_BF16_ROW;

#[cfg(target_family = "wasm")]
::inventory::submit! { crate::Registered(ALL_REDUCE_BF16_ROW) }

const ALL_REDUCE_RESIDUAL_RMSNORM_BF16_ROW: ::kernels::routine::Routine<crate::Plane> =
    ::kernels::untraced!(
        crate::Plane,
        "all_reduce_residual_rmsnorm_bf16",
        all_reduce_residual_rmsnorm_bf16,
        namespace = "comm",
        whole,
        driver
    )
    .stating(&[Some(::kernels::Source::Alias(1, 0))]);

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice(crate::ROUTINES)]
static ALL_REDUCE_RESIDUAL_RMSNORM_BF16_ROUTINE: ::kernels::routine::Routine<crate::Plane> =
    ALL_REDUCE_RESIDUAL_RMSNORM_BF16_ROW;

#[cfg(target_family = "wasm")]
::inventory::submit! { crate::Registered(ALL_REDUCE_RESIDUAL_RMSNORM_BF16_ROW) }
