use crate::jit::{Ctx, Launch, aligned16};
use kernels::{Bind, Fire};
use kernels_macros::routine;

use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::Region;

use kernels::routine::{Const, In, Out};

const BLOCK: u32 = 256;

const WARP: u32 = 32;

#[must_use]
fn route_rows(rows: i32, width: i32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows.unsigned_abs(),
        width
            .unsigned_abs()
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

pub(crate) fn stated<P>(view: Result<Region<P>, Refusal>) -> Result<Region<P>, Refusal> {
    match view {
        Err(refusal) => Err(this_family(refusal)),
        ok => ok,
    }
}

#[must_use]
const fn this_family(refusal: Refusal) -> Refusal {
    match refusal {
        Refusal::Absent { what } => Refusal::Empty { what },
        said => said,
    }
}

#[routine(canon = split_rows)]
pub fn split_bf16_rows(
    ctx: &Ctx<'_>,
    src: In<Tensor<bf16>>,
    left: Out<Tensor<bf16>>,
    right: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let left_half = stated(left.all("left_dim or right_dim"))?;
    let right_half = stated(right.all("left_dim or right_dim"))?;
    let n = left_half.rows;

    let left_dim = left_half.stride;
    let right_dim = right_half.stride;
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            "::pie::layout::split_rows<::pie::bf16>",
        )
        .apply(route_rows(n, left_dim.0)),
        &[
            src.arg(),
            left.arg(),
            right.arg(),
            left_dim.arg(),
            right_dim.arg(),
        ],
    )
}

#[routine(bf16, out(b_out = rows(ba) x half(ba)), out(a_out = rows(ba) x half(ba)))]
pub fn split_qwen_gdn_ba<T>(
    ctx: &Ctx<'_>,
    ba: In<Tensor<T>>,
    b_out: Out<Tensor<T>>,
    a_out: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let half = stated(b_out.all("v_h"))?;
    let n = half.rows;
    let v_h = half.stride;
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            crate::jit::symbol(&format!("::pie::layout::split_qwen_gdn_ba<{}>", T::CPP)),
        )
        .apply(route_rows(n, v_h.0)),
        &[ba.arg(), b_out.arg(), a_out.arg(), v_h.arg()],
    )
}

#[routine(bf16, internal)]
pub fn deinterleave_rows<T>(
    ctx: &Ctx<'_>,
    fused: In<Tensor<T>>,
    gate_out: Out<Tensor<T>>,
    up_out: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let gate = stated(gate_out.all("h"))?;
    let rows = gate.rows;
    let h = gate.stride;
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            crate::jit::symbol(&format!("::pie::layout::deinterleave_rows<{}>", T::CPP)),
        )
        .apply(route_rows(rows, h.0)),
        &[fused.arg(), gate_out.arg(), up_out.arg(), h.arg()],
    )
}

#[routine(bf16, internal)]
pub fn deinterleave_vec<T>(
    ctx: &Ctx<'_>,
    fused: In<Tensor<T>>,
    gate_out: Out<Tensor<T>>,
    up_out: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let gate = stated(gate_out.all("i"))?;
    let i = gate.elements();
    if i <= 0 {
        return Err(Refusal::Empty { what: "i" });
    }
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            crate::jit::symbol(&format!("::pie::layout::deinterleave_vec<{}>", T::CPP)),
        )
        .apply(elementwise(i.unsigned_abs())),
        &[fused.arg(), gate_out.arg(), up_out.arg(), i.arg()],
    )
}

#[routine(internal)]
pub fn concat_bf16_rows(
    ctx: &Ctx<'_>,
    left: In<Tensor<bf16>>,
    right: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let left_half = stated(left.all("left_dim or right_dim"))?;
    let right_half = stated(right.all("left_dim or right_dim"))?;
    let rows = out.rows;
    let left_dim = left_half.stride;
    let right_dim = right_half.stride;
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            "::pie::layout::concat_rows<::pie::bf16>",
        )
        .apply(route_rows(rows, left_dim.0)),
        &[
            left.arg(),
            right.arg(),
            out.arg(),
            left_dim.arg(),
            right_dim.arg(),
        ],
    )
}

#[routine]
pub fn gather_bf16_rows(
    ctx: &Ctx<'_>,
    src: In<Tensor<u16>>,
    dst: Out<Tensor<u16>>,
    sampling_indices: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let row_indices = sampling_indices.ptr;

    let dense = stated(dst.all("width"))?;
    let num_dst_rows = dense.rows;
    let width = dense.stride;
    ctx.fire(
        Fire::at(
            "layout/gather_rows.cuh",
            "::pie::layout::gather_rows<::pie::u16>",
        )
        .apply(route_rows(num_dst_rows, width.0)),
        &[src.arg(), row_indices.arg(), dst.arg(), width.arg()],
    )
}

#[routine]
pub fn transpose_bf16_nld_to_lnd(
    ctx: &Ctx<'_>,
    src: In<Tensor<u16>>,
    dst: Out<Tensor<u16>>,
    dim: Const<i32>,
) -> Result<(), Refusal> {
    let dim = *dim;

    let source = stated(src.all("width"))?;
    let n = source.rows;
    let width = source.width;

    if dim <= 0 {
        return Err(Refusal::Empty { what: "ple_dim" });
    }
    if width % dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of PLE planes",
            at: i64::from(width),
        });
    }
    let layers = width / dim;
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    ctx.fire(
        Fire::at(
            "layout/gather_rows.cuh",
            "::pie::layout::transpose_nld_to_lnd<::pie::u16>",
        )
        .apply(elementwise(u32::try_from(total).unwrap_or(u32::MAX))),
        &[
            src.arg(),
            dst.arg(),
            n.arg(),
            layers.arg(),
            dim.arg(),
            total.arg(),
        ],
    )
}

#[routine(internal)]
pub fn copy_if_valid_slot(
    ctx: &Ctx<'_>,
    src: In<Tensor<u8>>,
    dst: Out<Tensor<u8>>,
    bytes: Const<usize>,
    slot_ids: In<Tensor<i32>>,
    request: Const<usize>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at("layout/slot_ops.cuh", "::pie::layout::copy_if_valid_slot")
            .apply(Launch::grid([1, 1, 1], [256, 1, 1])),
        &[
            src.arg(),
            dst.arg(),
            bytes.arg(),
            slot_ids.arg(),
            request.arg(),
        ],
    )
}

const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 {
        head_dim.unsigned_abs()
    } else {
        256
    }
}

#[routine(untraced, internal)]
pub fn envelope_merge_written(
    ctx: &Ctx<'_>,
    k_curr: In<Tensor<bf16>>,
    w_page: In<Tensor<u32>>,
    w_off: In<Tensor<u32>>,
    row_valid: crate::jit::abi::MaybeConst<u8>,
    env_min: Out<Tensor<bf16>>,
    env_max: Out<Tensor<bf16>>,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    const FUSE_MAX_TOKENS: i32 = 128;

    let launch = Launch::grid(
        [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        [threads_for(head_dim), 1, 1],
    );

    if num_tokens <= FUSE_MAX_TOKENS {
        return ctx.fire(
            Fire::at(
                "layout/envelope.cuh",
                "::pie::layout::merge_written_fused<::pie::i32(0)>",
            )
            .apply(launch),
            &[
                k_curr.arg(),
                w_page.arg(),
                w_off.arg(),
                row_valid.arg(),
                env_min.arg(),
                env_max.arg(),
                num_tokens.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        );
    }

    ctx.fire(
        Fire::at(
            "layout/envelope.cuh",
            "::pie::layout::reset_started_pages<::pie::i32(0)>",
        )
        .apply(launch),
        &[
            w_page.arg(),
            w_off.arg(),
            row_valid.arg(),
            env_min.arg(),
            env_max.arg(),
            num_tokens.arg(),
            num_kv_heads.arg(),
            head_dim.arg(),
        ],
    )?;
    ctx.fire(
        Fire::at(
            "layout/envelope.cuh",
            "::pie::layout::merge_written<::pie::i32(0)>",
        )
        .apply(launch),
        &[
            k_curr.arg(),
            w_page.arg(),
            row_valid.arg(),
            env_min.arg(),
            env_max.arg(),
            num_tokens.arg(),
            num_kv_heads.arg(),
            head_dim.arg(),
        ],
    )
}

#[routine(untraced, internal)]
pub fn envelope_seed_empty(
    ctx: &Ctx<'_>,
    env_min: Out<Tensor<bf16>>,
    env_max: Out<Tensor<bf16>>,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    const SEED_BLOCK: u32 = 256;

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    ctx.fire(
        Fire::at(
            "layout/envelope.cuh",
            "::pie::layout::seed_empty<::pie::i32(0)>",
        )
        .apply(Launch::grid(
            [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
            [SEED_BLOCK, 1, 1],
        )),
        &[env_min.arg(), env_max.arg(), n.arg()],
    )
}

#[routine(untraced, internal)]
pub fn envelope_update_appended(
    ctx: &Ctx<'_>,
    k_pages: In<Tensor<bf16>>,
    qo_indptr: In<Tensor<u32>>,
    kv_page_indices: In<Tensor<u32>>,
    kv_page_indptr: In<Tensor<u32>>,
    kv_last_page_lens: In<Tensor<u32>>,
    env_min: Out<Tensor<bf16>>,
    env_max: Out<Tensor<bf16>>,
    num_requests: i32,
    max_touched: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "layout/envelope.cuh",
            "::pie::layout::update_appended<::pie::bf16>",
        )
        .apply(Launch::grid(
            [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
            [threads_for(head_dim), 1, 1],
        )),
        &[
            k_pages.arg(),
            qo_indptr.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            env_min.arg(),
            env_max.arg(),
            num_requests.arg(),
            page_size.arg(),
            num_kv_heads.arg(),
            head_dim.arg(),
        ],
    )
}

const VEC_WIDTH: i32 = 8;

#[must_use]
pub fn vectorisable(hidden: i32, weight: *const bf16, y: *const bf16) -> bool {
    hidden % VEC_WIDTH == 0 && aligned16(weight.cast()) && aligned16(y.cast())
}

#[routine(canon = embed)]
pub fn embed_bf16(
    ctx: &Ctx<'_>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    token_ids: In<Tensor<i32>>,
    vocab: Const<i32>,
) -> Result<(), Refusal> {
    let token_ids = token_ids.ptr;

    let vocab = *vocab;
    const EMBED_BLOCK: u32 = 256;

    let dst = stated(y.all("hidden"))?;
    let num_tokens = dst.rows;

    let hidden = dst.stride;

    let vec = vectorisable(hidden.0, weight.v, dst.ptr.cast_const());
    let per_row = if vec { hidden.0 / VEC_WIDTH } else { hidden.0 };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let instantiation = if vec {
        "::pie::layout::embed<\
                                      ::pie::true_type::value>"
    } else {
        "::pie::layout::embed<::pie::false_type::value>"
    };

    ctx.fire(
        Fire::at("layout/embed.cuh", instantiation)
            .apply(Launch::grid([blocks, 1, 1], [EMBED_BLOCK, 1, 1])),
        &[
            token_ids.arg(),
            weight.arg(),
            dst.ptr.arg(),
            hidden.arg(),
            vocab.arg(),
            num_tokens.arg(),
            per_row.arg(),
        ],
    )
}

// `derived_name_is` STOOD HERE: a `const fn` byte-comparison of two `&str`,
// written because `str::eq` is not const and the only callers were
// `const _: () = { assert!(derived_name_is(..)) }` blocks pinning a derived
// symbol to its spelling. Those blocks went in the CUDA comment sweep -- a
// block that binds and asserts nothing proves only that the routine exists
// -- and this went unused with them. `str::eq` is const on this toolchain
// now, so the next reader who wants the claim should write it directly
// rather than restore this.
