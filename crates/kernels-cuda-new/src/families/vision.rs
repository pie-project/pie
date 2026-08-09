use kernels::{KernelSig, LaunchRule, kernel, operands};

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The six naive kernels more than one tower launches.
pub const TOWER_NAIVE_KERNELS: Unit = Unit {
    name: "vision/tower_naive_kernels",
    root: include_str!("../../csrc/src/vision/tower_naive_kernels.cuh"),
    rows: TOWER_NAIVE_ROWS,
    options: &[],
};

/// The one kernel the two gemma-4 towers share.
pub const GEMMA4_NAIVE_KERNELS: Unit = Unit {
    name: "vision/gemma4_naive_kernels",
    root: include_str!("../../csrc/src/vision/gemma4_naive_kernels.cuh"),
    rows: GEMMA4_NAIVE_ROWS,
    options: &[],
};

/// The Gemma-4 vision encoder's nine.
pub const GEMMA4_VISION: Unit = Unit {
    name: "vision/gemma4_vision",
    root: include_str!("../../csrc/src/vision/gemma4_vision.cuh"),
    rows: GEMMA4_VISION_ROWS,
    options: &[],
};

/// The Gemma-4 audio encoder's twelve.
pub const GEMMA4_AUDIO: Unit = Unit {
    name: "vision/gemma4_audio",
    root: include_str!("../../csrc/src/vision/gemma4_audio.cuh"),
    rows: GEMMA4_AUDIO_ROWS,
    options: &[],
};

/// The Qwen3-VL vision encoder: six rows of the header's eleven
pub const QWEN3_VL_TOWER: Unit = Unit {
    name: "vision/qwen3_vl_tower",
    root: include_str!("../../csrc/src/vision/qwen3_vl_tower.cuh"),
    rows: QWEN3_VL_ROWS,
    options: &[],
};

/// The units `vision` compiles.
pub static UNITS: &[Unit] = &[
    TOWER_NAIVE_KERNELS,
    GEMMA4_NAIVE_KERNELS,
    GEMMA4_VISION,
    GEMMA4_AUDIO,
    QWEN3_VL_TOWER,
];

/// [`TOWER_NAIVE_KERNELS`]' instantiations — all six.
static TOWER_NAIVE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[0],
        template_path: "vision::device::k_rms",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[1],
        template_path: "vision::device::k_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[2],
        template_path: "vision::device::k_f32_to_bf16",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[3],
        template_path: "vision::device::k_gelu_erf",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[4],
        template_path: "vision::device::k_layernorm",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[5],
        template_path: "vision::device::k_matmul",
        elem: "device::bf16",
    },
];

/// The contracts, in [`TOWER_NAIVE_ROWS`]' order.
#[rustfmt::skip]
static TOWER_NAIVE_SIGS: [KernelSig; 6] = [
    kernel!(k_rms "vision::k_rms_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            x: Buf,
            w: Buf | null,
            o: BufMut,
            rows: I32,
            width: I32,
            eps: F32,
        ]),
    kernel!(k_add "vision::k_add_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            a: BufMut,
            b: Buf,
            n: Usize,
        ]),
    kernel!(k_f32_to_bf16 "vision::k_f32_to_bf16_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            a: F32s,
            o: BufMut,
            n: Usize,
        ]),
    kernel!(k_gelu_erf "vision::k_gelu_erf_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            n: Usize,
        ]),
    kernel!(k_layernorm "vision::k_layernorm_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            x: Buf,
            g: Buf | null,
            beta: Buf | null,
            o: BufMut,
            rows: I32,
            width: I32,
            eps: F32,
        ]),
    kernel!(k_matmul "vision::k_matmul_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            w: Buf,
            y: BufMut,
            n: I32,
            k: I32,
            o: I32,
        ]),
];

/// [`GEMMA4_NAIVE_KERNELS`]' one instantiation.
static GEMMA4_NAIVE_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &GEMMA4_NAIVE_SIGS[0],
    template_path: "vision::device::k_clamp",
    elem: "device::bf16",
}];

/// The contract, and the one thing about it worth stating twice.
#[rustfmt::skip]
static GEMMA4_NAIVE_SIGS: [KernelSig; 1] = [
    kernel!(k_clamp "vision::k_clamp_bf16",
        file = Some("vision/gemma4_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            lo: Buf | null,
            hi: Buf | null,
            t: Usize,
        ]),
];

/// [`GEMMA4_VISION`]'s instantiations — eight of its nine.
static GEMMA4_VISION_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[0],
        template_path: "vision::device::k_scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[1],
        template_path: "vision::device::k_softmax",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[2],
        template_path: "vision::device::k_pool_finish",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[3],
        template_path: "vision::device::k_addpos_grid2d",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[4],
        template_path: "vision::device::k_rope_axial2d",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[5],
        template_path: "vision::device::k_qk",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[6],
        template_path: "vision::device::k_av",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[7],
        template_path: "vision::device::k_pool",
        elem: "device::bf16",
    },
];

/// The contracts, in [`GEMMA4_VISION_ROWS`]' order.
#[rustfmt::skip]
static GEMMA4_VISION_SIGS: [KernelSig; 8] = [
    kernel!(k_scale "vision::k_scale_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            p: Buf,
            o: BufMut,
            t: Usize,
        ]),
    kernel!(k_softmax "vision::k_softmax_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::PerRow,
        in_place = &[(0, 0)],
        operands = operands![
            s: F32sMut,
            n: I32,
        ]),
    kernel!(k_pool_finish "vision::k_pool_finish_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            input: F32s,
            o: BufMut,
            s: F32,
            t: Usize,
        ]),
    kernel!(k_addpos_grid2d "vision::k_addpos_grid2d_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut,
            tb: Buf,
            pos: F32s,
            n: I32,
            o: I32,
            p: I32,
        ]),
    kernel!(k_rope_axial2d "vision::k_rope_axial2d_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::AxialRope,
        in_place = &[(0, 0)],
        operands = operands![
            q: BufMut,
            pos: F32s,
            n: I32,
            h: I32,
            theta: F32,
        ]),
    kernel!(k_qk "vision::k_qk_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            q: Buf,
            k: Buf,
            s: F32sMut,
            n: I32,
            h: I32,
            head: I32,
            scale: F32,
        ]),
    kernel!(k_av "vision::k_av_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            s: F32s,
            v: Buf,
            o: BufMut,
            n: I32,
            h: I32,
            head: I32,
        ]),
    kernel!(k_pool "vision::k_pool_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            h: Buf,
            grp: I32s,
            o: F32sMut,
            n: I32,
            d: I32,
            k2: F32,
        ]),
];

/// [`GEMMA4_AUDIO`]'s instantiations — eight of its twelve.
static GEMMA4_AUDIO_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[0],
        template_path: "vision::device::k_silu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[1],
        template_path: "vision::device::k_axpy",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[2],
        template_path: "vision::device::k_matmul_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[3],
        template_path: "vision::device::k_glu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[4],
        template_path: "vision::device::k_layernorm_relu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[5],
        template_path: "vision::device::k_sscp_flatten",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[6],
        template_path: "vision::device::k_qkv_scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[7],
        template_path: "vision::device::k_rel_pos_enc",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[8],
        template_path: "vision::device::k_conv2d_s2",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[9],
        template_path: "vision::device::k_chlast",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[10],
        template_path: "vision::device::k_chfirst",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[11],
        template_path: "vision::device::k_local_attn",
        elem: "device::bf16",
    },
];

/// The contracts, in [`GEMMA4_AUDIO_ROWS`]' order.
#[rustfmt::skip]
static GEMMA4_AUDIO_SIGS: [KernelSig; 12] = [
    kernel!(k_silu "vision::k_silu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            t: Usize,
        ]),
    kernel!(k_axpy "vision::k_axpy_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            a: BufMut,
            b: Buf,
            scale: F32,
            t: Usize,
        ]),
    kernel!(k_matmul_bias "vision::k_matmul_bias_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            w: Buf,
            b: Buf | null,
            y: BufMut,
            n: I32,
            k: I32,
            o: I32,
        ]),
    kernel!(k_glu "vision::k_glu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            o: BufMut,
            n: I32,
            d: I32,
        ]),
    kernel!(k_layernorm_relu "vision::k_layernorm_relu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::PerRowNarrow,
        in_place = &[(0, 0)],
        operands = operands![
            x: Buf,
            w: Buf | null,
            o: BufMut,
            r: I32,
            c: I32,
            eps: F32,
        ]),
    kernel!(k_sscp_flatten "vision::k_sscp_flatten_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            input: Buf,
            out: BufMut,
            oc: I32,
            t_out: I32,
            f_out: I32,
        ]),
    kernel!(k_qkv_scale "vision::k_qkv_scale_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut,
            k: BufMut,
            pds: Buf,
            n: I32,
            h: I32,
            hd: I32,
            q_scale: F32,
            k_scale: F32,
        ]),
    kernel!(k_rel_pos_enc "vision::k_rel_pos_enc_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            pe: BufMut,
            p: I32,
            hidden: I32,
        ]),
    kernel!(k_conv2d_s2 "vision::k_conv2d_s2_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            input: Buf,
            w: Buf,
            out: BufMut,
            ic: I32,
            t_in: I32,
            f_in: I32,
            oc: I32,
            t_out: I32,
            f_out: I32,
        ]),
    kernel!(k_chlast "vision::k_chlast_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            input: Buf,
            out: BufMut,
            oc: I32,
            t_out: I32,
            f_out: I32,
        ]),
    kernel!(k_chfirst "vision::k_chfirst_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            input: Buf,
            out: BufMut,
            oc: I32,
            t_out: I32,
            f_out: I32,
        ]),
    kernel!(k_local_attn "vision::k_local_attn_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            q: Buf,
            k: Buf,
            v: Buf,
            relk: Buf,
            out: BufMut,
            n: I32,
            h: I32,
            hd: I32,
            p: I32,
            cap: F32,
        ]),
];

/// [`QWEN3_VL_TOWER`]'s instantiations — six of its eleven.
static QWEN3_VL_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[0],
        template_path: "vision::device::k_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[1],
        template_path: "vision::device::k_add_pe",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[2],
        template_path: "vision::device::k_gelu_tanh",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[3],
        template_path: "vision::device::k_gelu_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[4],
        template_path: "vision::device::k_merge_gather",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[5],
        template_path: "vision::device::k_split_rope_qkv",
        elem: "device::bf16",
    },
];

/// The contracts, in [`QWEN3_VL_ROWS`]' order.
#[rustfmt::skip]
static QWEN3_VL_SIGS: [KernelSig; 6] = [
    kernel!(k_bias "vision::k_bias_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut,
            b: Buf,
            m: Usize,
            n: I32,
        ]),
    kernel!(k_add_pe "vision::k_add_pe_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            h: BufMut,
            pe: Buf,
            t: Usize,
        ]),
    kernel!(k_gelu_tanh "vision::k_gelu_tanh_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            t: Usize,
        ]),
    kernel!(k_gelu_bias "vision::k_gelu_bias_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut,
            b: Buf | null,
            n: I32,
            d: I32,
        ]),
    kernel!(k_merge_gather "vision::k_merge_gather_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            h: Buf,
            g: BufMut,
            n_token: I32,
            u: I32,
            c: I32,
        ]),

    kernel!(k_split_rope_qkv "vision::k_split_rope_qkv_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            qkv: Buf,
            b: Buf | null,
            q: BufMut,
            k: BufMut,
            v: BufMut,
            pos: F32s,
            n: I32,
            nh: I32,
            head: I32,
            theta: F32,
        ]),
];
