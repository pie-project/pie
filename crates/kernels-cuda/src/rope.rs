
#![allow(clippy::too_many_arguments)]

use kernels::{Bind, Fire};
use kernels::routine::{Const, In, InOut, Out};
use kernels_macros::routine;
use crate::jit::Abi;
use crate::jit::abi::{MaybeConst, bf16, f16};
use crate::jit::abi::Tensor;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::raises::Struct;
use crate::views::KvCache;

use core::ptr::NonNull;

pub const ROTATE_BLOCK: i32 = 256;

pub const FUSED_BLOCK: u32 = 128;

pub const MAX_CACHED_PAIRS: i32 = 4096;

#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK { 1 } else { ROTATE_BLOCK / half }
}

#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

#[must_use]
const fn fused_launch(rows: i32, total_heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), total_heads.unsigned_abs(), 1], [FUSED_BLOCK, 1, 1])
}

#[must_use]
const fn rotate_launch(num_tokens: i32, total_heads: i32, per_block: i32, smem: u32) -> Launch {

    #[must_use]
    const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
    [
    num_tokens.unsigned_abs(),
    (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
    1,
    ]
    }

    Launch::grid(
        rotate_grid(num_tokens, total_heads, per_block),
        [ROTATE_BLOCK.unsigned_abs(), 1, 1],
    )
    .smem(smem)
}

#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {

    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    #[allow(clippy::cast_precision_loss)]
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

fn heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

fn q_heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "the q region's width" });
    }
    heads(width, head_dim)
}

fn k_heads<T>(q: *mut T, k: *mut T, width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 && !k.is_null() && !core::ptr::eq(k.cast_const(), q.cast_const()) {
        return Err(Refusal::Empty { what: "the k region's width" });
    }
    heads(width, head_dim)
}

#[routine]
pub fn rope_standard_table(
    ctx: &Ctx<'_>,
    table: Out<Tensor<f32>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;

    let positions = positions.ptr;
    if head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::standard_table<::pie::i32>").apply(Launch::per_row(table.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[positions.arg(), table.arg(), head_dim.arg(), theta.arg()])
}

#[routine(canon = rope)]
pub fn rope_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    num_q_heads: Const<i32>,
    num_kv_heads: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    interleaved: Const<bool>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let num_q_heads = *num_q_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;

    let theta = *theta;
    let interleaved = *interleaved;
    let positions = positions.ptr;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate<::pie::false_type::value, false>").apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                interleaved.arg(),
                pairs.arg(),
                per_block.arg(),
                MaybeConst::<bf16>::none().arg(),
                None::<NonNull<bf16>>.arg(),
                None::<NonNull<bf16>>.arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u8>::none().arg(),
                0_i32.arg(),
                0_i32.arg(),
            ])
}

#[routine(whole)]
pub fn rope_write_kv_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: In<Tensor<bf16>>,
    v: In<Tensor<bf16>>,
    interleaved: Const<bool>,
    kvc: In<Struct<KvCache>>,
    num_q_heads: Const<i32>,
    num_kv_heads: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    qo_indptr: In<Tensor<i32>>,
    row_valid: In<Tensor<i32>>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let interleaved = *interleaved;
    let page_size = kvc.page_size;
    let num_q_heads = *num_q_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let theta = *theta;
    let hnd_layout = kvc.layout != 0;

    let k_pages = kvc.keys as *mut bf16;
    // The request count is the CSR operand's own row count.
    let num_requests = qo_indptr.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let row_valid = row_valid.ptr as *const u8;
    let positions = positions.ptr;
    let v_pages = kvc.values as *mut bf16;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    let instantiation =
        if hnd_layout { "::pie::rope::rotate<\
                             ::pie::true_type::value, true>" } else { "::pie::rope::rotate<::pie::true_type::value, false>" };
    ctx.fire(Fire::at("rope/rope.cuh", instantiation).apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                interleaved.arg(),
                pairs.arg(),
                per_block.arg(),
                MaybeConst::new(v.ptr).arg(),
                NonNull::new(k_pages).arg(),
                NonNull::new(v_pages).arg(),
                MaybeConst::new(qo_indptr).arg(),
                MaybeConst::new(kv_page_indices).arg(),
                MaybeConst::new(kv_page_indptr).arg(),
                MaybeConst::new(kv_last_page_lens).arg(),
                MaybeConst::new(row_valid).arg(),
                num_requests.arg(),
                page_size.arg(),
            ])
}

#[routine]
pub fn qk_rmsnorm_rope_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let eps = *eps;

    let positions = positions.ptr;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

#[routine(whole)]
pub fn qk_rmsnorm_rope_bf16_devwin(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    n_max: Const<i32>,
    positions: In<Tensor<i32>>,
    win_start: Const<i32>,
    win_len: Const<i32>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let eps = *eps;

    let win = crate::stage_peel_window(ctx, "rope::qk_devwin", *win_start, *win_len)?;
    let n_max = *n_max;
    let positions = positions.ptr;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_devwin<::pie::i32(128)>").apply(fused_launch(n_max, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                win.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

#[routine]
pub fn qk_rmsnorm_mrope_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>,
    mrope_section_t: Const<i32>,
    mrope_section_h: Const<i32>,
    mrope_section_w: Const<i32>,
    num_q_heads: Const<i32>,
    num_kv_heads: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let mrope_section_t = *mrope_section_t;
    let mrope_section_h = *mrope_section_h;
    let mrope_section_w = *mrope_section_w;
    let num_q_heads = *num_q_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let theta = *theta;
    let eps = *eps;

    let positions = positions.ptr;
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_mrope<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
                mrope_section_t.arg(),
                mrope_section_h.arg(),
                mrope_section_w.arg(),
            ])
}

#[routine]
pub fn qk_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let eps = *eps;

    let positions = positions.ptr;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_rounded<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

#[routine]
pub fn q_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    qk_rmsnorm_rope_bf16_rounded(
        ctx,
        q,
        InOut {
            ptr: core::ptr::null_mut(),
            rows: q.rows,
            width: 0,
        },
        q_weight,
        Const { v: core::ptr::null() },
        head_dim,
        theta,
        eps,
        positions,
    )
}

#[routine]
pub fn rope_yarn_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    factor: Const<f32>,
    low_freq_factor: Const<f32>,
    high_freq_factor: Const<f32>,
    original_max_position: Const<i32>,
    num_q_heads: Const<i32>,
    num_kv_heads: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    // The YaRN block, which `Cx::yarn` has answered all along: the four
    // numbers were parameters only because no routine asked for them.
    let factor = *factor;
    let low_freq_factor = *low_freq_factor;
    let high_freq_factor = *high_freq_factor;
    let original_max_position = *original_max_position;
    let num_q_heads = *num_q_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let theta = *theta;

    let positions = positions.ptr;
    let half = head_dim / 2;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_yarn").apply(rotate_launch(q.rows, total_heads, per_block, 0)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                factor.arg(),
                low_freq_factor.arg(),
                high_freq_factor.arg(),
                orig_max_pos.arg(),
                per_block.arg(),
            ])
}

#[routine]
pub fn rope_yarn_original_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    factor: Const<f32>,
    beta_fast: Const<f32>,
    beta_slow: Const<f32>,
    attention_factor: Const<f32>,
    original_max_position: Const<i32>,
    interleaved: Const<bool>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let factor = *factor;
    let beta_fast = *beta_fast;
    let beta_slow = *beta_slow;
    let attention_factor = *attention_factor;
    let original_max_position = *original_max_position;
    let interleaved = *interleaved;

    let positions = positions.ptr;

    if original_max_position <= 0 {
        return Err(Refusal::Unstated { what: "the checkpoint's YaRN block" });
    }
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let (low_dim, high_dim) = ramp_bounds(
        head_dim,
        theta,
        beta_fast,
        beta_slow,
        original_max_position,
    );
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_yarn_original").apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                attention_factor.arg(),
                interleaved.arg(),
                per_block.arg(),
                pairs.arg(),
            ])
}

fn rope_partial<T>(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    q: *mut T,
    k: *mut T,
    positions: *const i32,
    num_tokens: i32,
    q_width: i32,
    k_width: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32) -> Result<(), Refusal>
where
    T: kernels::Elem,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Write: Abi,
{
    let (num_q_heads, num_kv_heads) =
        (q_heads(q_width, head_dim)?, k_heads(q, k, k_width, head_dim)?);
    ctx.fire(Fire::at("rope/rope.cuh", instantiation).apply(Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                0i32.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
            ])
}

#[routine(canon = "rope.partial")]
pub fn rope_partial_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    rotary_dim: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let positions = positions.ptr;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        k.ptr,
        positions,
        q.rows,
        q.width,
        k.width,
        head_dim,
        *rotary_dim,
        theta,
    )
}

#[routine]
pub fn rope_partial_q_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    rotary_dim: Const<i32>,
    head_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let theta = *theta;
    let positions = positions.ptr;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        q.ptr,
        positions,
        q.rows,
        q.width,
        0,
        head_dim,
        *rotary_dim,
        theta,
    )
}

#[routine(internal)]
pub fn rope_partial_f16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<f16>>,
    k: InOut<Tensor<f16>>,
    head_dim: Const<i32>,
    rotary_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let rotary_dim = *rotary_dim;
    let theta = *theta;

    let positions = positions.ptr;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::f16>",
        q.ptr,
        k.ptr,
        positions,
        q.rows,
        q.width,
        k.width,
        head_dim,
        rotary_dim,
        theta,
    )
}

#[routine]
pub fn rope_partial_last_bf16(
    ctx: &Ctx<'_>,
    q: Out<Tensor<bf16>>,
    k: Out<Tensor<bf16>>,
    head_dim: Const<i32>,
    rotary_dim: Const<i32>,
    theta: Const<f32>,
    interleaved: Const<bool>,
    yarn_factor: Const<f32>,
    yarn_beta_fast: Const<f32>,
    yarn_beta_slow: Const<f32>,
    yarn_original_max_position: Const<i32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let rotary_dim = *rotary_dim;
    let theta = *theta;
    let interleaved = *interleaved;
    let yarn_factor = *yarn_factor;
    let yarn_beta_fast = *yarn_beta_fast;
    let yarn_beta_slow = *yarn_beta_slow;
    let yarn_original_max_position = *yarn_original_max_position;

    let positions = positions.ptr;
    let (num_q_heads, num_kv_heads) = (q_heads(q.width, head_dim)?, heads(k.width, head_dim)?);
    let (low_dim, high_dim) = if yarn_factor > 1.0 && yarn_original_max_position > 0 {
        ramp_bounds(
            rotary_dim,
            theta,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_max_position,
        )
    } else {
        (0.0, 0.0)
    };
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_partial_last").apply(Launch::per_row(q.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
                false.arg(),
                interleaved.arg(),
                yarn_factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
            ])
}

#[expect(clippy::too_many_arguments, reason = "D1: a routine takes fields, never a struct")]
#[routine]
pub fn rope_partial_last_q_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    head_dim: Const<i32>,
    rotary_dim: Const<i32>,
    theta: Const<f32>,
    interleaved: Const<bool>,
    yarn_factor: Const<f32>,
    yarn_beta_fast: Const<f32>,
    yarn_beta_slow: Const<f32>,
    yarn_original_max_position: Const<i32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {

    rope_partial_last_bf16(
        ctx,
        Out { ptr: q.ptr, rows: q.rows, width: q.width },
        Out {
            ptr: q.ptr,
            rows: q.rows,
            width: 0,
        },
        head_dim,
        rotary_dim,
        theta,
        interleaved,
        yarn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_original_max_position,
        positions,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {

    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
    pub original_max_position: i32,
}
impl Yarn {

    pub const NONE: Self = Self {
        factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        attention_factor: 1.0,
        original_max_position: 0,
    };
}
