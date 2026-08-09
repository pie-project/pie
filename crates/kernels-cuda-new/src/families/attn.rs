use kernels::KernelSig;
use kernels::Lit;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The attention-sink pair and K3's residual blend CROSSED INTO FN-WORLD.

/// flashinfer's supported head widths CROSSED INTO FN-WORLD, and closed a

/// The units the small half of `attn` compiles.
pub const UNITS_SMALL: &[Unit] = &[
];

/// The units `attn` compiles: the small half's, then the heavy half's.
pub static UNITS: &[Unit] = &concat_halves();

const fn concat_halves() -> [Unit; UNITS_SMALL.len() + UNITS_HEAVY.len()] {
    let mut out = [EMPTY; UNITS_SMALL.len() + UNITS_HEAVY.len()];
    let mut w = 0;
    let mut i = 0;
    while i < UNITS_SMALL.len() {
        out[w] = UNITS_SMALL[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < UNITS_HEAVY.len() {
        out[w] = UNITS_HEAVY[j];
        w += 1;
        j += 1;
    }
    out
}

/// A slot to fill and never a unit anything fires: it names no source and
const EMPTY: Unit = Unit { name: "", root: "", rows: &[], options: &[] };

/// The one `__global__` `attn/attention_xqa.cuh` holds — and the LAST one the
pub static ATTN_XQA_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &ATTN_XQA_SIGS[0],
    template_path: "attn::device::build_xqa_metadata",
    elem: DeviceKernel::PLAIN,
}];

#[rustfmt::skip]
static ATTN_XQA_SIGS: [KernelSig; 1] = [
    kernel!(build_xqa_metadata "attn::build_xqa_metadata",
        file = Some("attn/attention_xqa.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            kv_page_indices: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_table: I32sMut,
            seq_lens: U32sMut,
            num_requests: I32,
            max_pages_per_seq: I32,
            page_size: I32,
        ]),
];

/// XQA's fire-wide prepare, as a JIT unit.
pub const ATTN_XQA: Unit = Unit {
    name: "attn/attention_xqa",
    root: include_str!("../../csrc/src/attn/attention_xqa.cuh"),
    rows: ATTN_XQA_ROWS,
    options: &[],
};

/// The three `__global__`s of `attn/attention_score_post.cuh`.
pub static ATTN_SCORE_POST_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[0],
        template_path: "attn::device::attn_score_normalize",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[1],
        template_path: "attn::device::attn_prefill_score_normalize",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[2],
        template_path: "attn::device::attn_prefill_score_fold",
        elem: DeviceKernel::PLAIN,
    },
];

#[rustfmt::skip]
static ATTN_SCORE_POST_SIGS: [KernelSig; 3] = [
    kernel!(attn_score_normalize "attn::attn_score_normalize",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: BufMut,
            score_indptr: I32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
        ]),

    kernel!(attn_prefill_score_normalize "attn::attn_prefill_score_normalize",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: BufMut,
            score_indptr: I32s,
            qo_indptr: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
            window: I32,
        ]),

    kernel!(attn_prefill_score_fold "attn::attn_prefill_score_fold",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: Buf,
            folded: BufMut,
            score_indptr: I32s,
            qo_indptr: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
            num_q_heads: I32,
            window: I32,
        ]),
];

/// The capture post-kernels, as a JIT unit.
pub const ATTN_SCORE_POST: Unit = Unit {
    name: "attn/attention_score_post",
    root: include_str!("../../csrc/src/attn/attention_score_post.cuh"),
    rows: ATTN_SCORE_POST_ROWS,
    options: &[],
};

/// The units the heavy half of `attn` compiles.
pub const UNITS_HEAVY: &[Unit] = &[
    ATTN_SCORE_POST,
    ATTN_XQA,
];

/// One member of the XQA lattice: a `-D` set, and what it is for.
pub struct XqaVariant {
    /// The `Unit::name` this member gets.
    pub unit: &'static str,
    /// The `-D` set, verbatim, as `Unit::options` would carry it.
    pub options: &'static [&'static str],
    /// The `extern "C"` device entry point this member exports, after the
    pub entry: &'static str,
    /// The archive file this member's `#define` block came from, with the
    pub from: &'static str,
    /// Why this member exists — the measurement its `.cu` carried.
    pub because: &'static str,
}

/// The twelve `-D`s every member of the lattice shares, plus the one that is
pub const XQA_COMMON_OPTIONS: &[&str] = &[
    "-DGENERATE_CUBIN=1",
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DINPUT_FP16=0",
    "-DDTYPE=device::bf16",
    "-DCACHE_ELEM_ENUM=0",
    "-DHEAD_ELEMS=128",
    "-DSLIDING_WINDOW=0",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
    "-DMLA_WRAPPER=0",
];

/// The root all six members compile, carried so a moved file is a compile
pub const XQA_ROOT: &str = include_str!("../../csrc/src/attn/attention_xqa_mha.cuh");

/// `sizeof(SharedMem)` (`xqa/mha.cu:409`), measured out of the PTX.
pub const XQA_SMEM_BYTES: u32 = 79_488;

/// The six units, as option sets.
pub const XQA_LATTICE: [XqaVariant; 6] = [
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p32",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p32_h128",
        from: "attn/attention_xqa_gqa2.cu:24-33",
        because: "head_group_size=2, used by small Qwen GQA models such as \
                  Qwen3-0.6B and Qwen3-1.7B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p16",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=16",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p16_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p16_h128",
        from: "attn/attention_xqa_gqa2_p16.cu:24-33",
        because: "the same head group at a 16-token page. Dead code TODAY and \
                  kept anyway: `xqa_decode_page_bucket` never returns 16 \
                  because `xqa_gqa2_page16_enabled()` returns false, so the \
                  only way to reach this member is to flip that — which is \
                  what it is for. Deleting it would make flipping the flag a \
                  port rather than a flag.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa4_p32",
        options: &[
            "-DHEAD_GRP_SIZE=4",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa4_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa4_bf16_p32_h128",
        from: "attn/attention_xqa_gqa4.cu:24-33",
        because: "head_group_size=4, used by medium Qwen GQA models such as \
                  Qwen3-4B and Qwen3-8B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa5_p32",
        options: &[
            "-DHEAD_GRP_SIZE=5",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa5_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa5_bf16_p32_h128",
        from: "attn/attention_xqa.cu:65-74",
        because: "head_group_size=5, the ratio Llama-3.1-8B-shaped models use \
                  (32 query heads over 8 KV heads is 4; 40 over 8 is 5). It \
                  lives in the family's dispatch head rather than a sibling \
                  file because that file is also where the host program was.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8.cu:24-33",
        because: "head_group_size=8, used by common large GQA models such as \
                  Qwen3-32B and Llama-70B-style shapes. Its launcher \
                  FORWARDS to the sm90 member when `current_device_major() \
                  >= 9`, which is why the two exist as a pair rather than as \
                  alternatives.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32_sm90",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=1",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8_sm90.cu:26-35",
        because: "Hopper GMMA/TMA, kept in a separate translation unit \
                  because FlashInfer's `xqa/mha.cu` and `xqa/mha_sm90.cu` \
                  intentionally define the same static kernel symbols. It \
                  also passes `enable_pdl = true` unconditionally where the \
                  other five pass `current_device_major() >= 9` — which is \
                  the same predicate, already known true here. NOT READY: \
                  measured at compute_90a it stops on `std::pair` in DEVICE \
                  code (`xqa/mha_sm90.cu:1980`, 12 diagnostics cascading \
                  from that one line), the header set has no `<utility>`, \
                  `csrc/shim/cuda.h` has no `CUtensorMap` or \
                  `CUtensorMapDataType_enum` for `xqa/tensorMap.h` to \
                  declare against, and the archive unit compiles \
                  `<xqa/tensorMap.cpp>` first — HOST code building tensor \
                  maps through `cuTensorMapEncodeTiled`, which is a second \
                  and larger host-to-Rust port than `launchMHA` was.",
    },
];
