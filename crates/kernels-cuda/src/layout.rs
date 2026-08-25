use crate::jit::{Ctx, Launch, aligned16};
use kernels::{Bind, Fire};

use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::Region;

use kernels::plane::{Const, In, Out};

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

fn stated_width(width: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(width).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(width),
        max: i64::from(i32::MAX),
    })
}

fn cuts_bf16<T: kernels::points::Scalar>() -> bool {
    T::KIND == kernels::points::ScalarKind::Bf16
}

fn as_bf16_in<T: kernels::points::Scalar>(x: In<Tensor<T>>) -> In<Tensor<bf16>> {
    In {
        ptr: x.ptr.cast::<bf16>(),
        rows: x.rows,
        width: x.width,
    }
}

fn as_bf16_out<T: kernels::points::Scalar>(y: Out<Tensor<T>>) -> Out<Tensor<bf16>> {
    Out {
        ptr: y.ptr.cast::<bf16>(),
        rows: y.rows,
        width: y.width,
    }
}

#[kernels_macros::claims]
impl kernels::points::Layout for Ctx<'_> {
    fn embed<T: kernels::points::Scalar>(
        &self,
        ids: In<Tensor<i32>>,
        table: Const<Tensor<T>>,
        vocab: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        const EMBED_BLOCK: u32 = 256;

        if !cuts_bf16::<T>() {
            return Err(Refusal::Absent {
                what: "layout.embed at an element other than bf16",
            });
        }
        let vocab = stated_width(vocab, "the embedding table's row count")?;
        if vocab <= 0 {
            return Err(Refusal::Empty {
                what: "the embedding table's row count",
            });
        }
        let dst = stated(y.all("the embedded row's width"))?;

        if ids.rows != dst.rows {
            return Err(Refusal::Narrow {
                what: "the gathered rows against the token ids handed over",
                at: i64::from(dst.rows),
            });
        }
        let hidden = dst.stride;
        let vec = vectorisable(hidden.0, table.v.cast(), dst.ptr.cast_const().cast());
        let per_row = if vec { hidden.0 / VEC_WIDTH } else { hidden.0 };
        let total = i64::from(dst.rows) * i64::from(per_row);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
        let instantiation = if vec {
            "::pie::layout::embed<::pie::true_type::value>"
        } else {
            "::pie::layout::embed<::pie::false_type::value>"
        };
        self.fire(
            Fire::at("layout/embed.cuh", instantiation)
                .apply(Launch::grid([blocks, 1, 1], [EMBED_BLOCK, 1, 1])),
            &[
                ids.ptr.arg(),
                table.arg(),
                dst.ptr.arg(),
                hidden.arg(),
                vocab.arg(),
                dst.rows.arg(),
                per_row.arg(),
            ],
        )
    }

    fn split_qkv<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<Tensor<T>>,
        k: Out<Tensor<T>>,
        v: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q_width, kv_width);
        if !cuts_bf16::<T>() {
            return Err(Refusal::Absent {
                what: "layout.split_qkv",
            });
        }
        let (packed, q, k, v) = (
            as_bf16_in(packed),
            as_bf16_out(q),
            as_bf16_out(k),
            as_bf16_out(v),
        );
        let q_rect = q.all("the q half")?;
        let k_rect = k.all("the k half")?;
        let (q_dim, kv_dim) = (q_rect.width, k_rect.width);
        if q_dim <= 0 && kv_dim <= 0 {
            return Err(Refusal::Empty {
                what: "q_dim and kv_dim",
            });
        }
        let width = q_dim.max(kv_dim).unsigned_abs();
        self.fire(
            Fire::at(
                "attn/split_packed.cuh",
                "::pie::attn::split_qkv<::pie::bf16>",
            )
            .apply(Launch::grid(
                [width.div_ceil(BLOCK), q_rect.rows.unsigned_abs(), 1],
                [BLOCK, 1, 1],
            )),
            &[
                packed.arg(),
                q.arg(),
                k.arg(),
                v.arg(),
                q_dim.arg(),
                kv_dim.arg(),
            ],
        )
    }

    fn split_q_gate<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        head_dim: u32,
        q: Out<Tensor<T>>,
        gate: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let head_dim = stated_width(head_dim, "the head pitch this cut states")?;
        if !cuts_bf16::<T>() {
            return Err(Refusal::Absent {
                what: "layout.split_q_gate",
            });
        }
        let (packed, q, gate) = (as_bf16_in(packed), as_bf16_out(q), as_bf16_out(gate));
        let q_rect = q.all("the query half")?;
        if head_dim <= 0 {
            return Err(Refusal::Unstated {
                what: "the head pitch a q/gate split grids by",
            });
        }
        if q_rect.width % head_dim != 0 {
            return Err(Refusal::Unstated {
                what: "a q/gate half whose width is not whole heads",
            });
        }
        let (n, num_heads) = (q_rect.rows, q_rect.width / head_dim);
        let block = if head_dim < 128 { 64 } else { 128 };
        self.fire(
            Fire::at(
                "layout/deinterleave.cuh",
                "::pie::layout::split_q_gate<::pie::bf16>",
            )
            .apply(Launch::grid(
                [n.unsigned_abs(), num_heads.unsigned_abs(), 1],
                [block, 1, 1],
            )),
            &[
                packed.arg(),
                q.arg(),
                gate.arg(),
                n.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }

    fn split_rows<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        width: u32,
        left: Out<Tensor<T>>,
        right: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = width;
        if !cuts_bf16::<T>() {
            return Err(Refusal::Absent {
                what: "layout.split_rows",
            });
        }
        let (src, left, right) = (as_bf16_in(x), as_bf16_out(left), as_bf16_out(right));
        let left_half = stated(left.all("left_dim or right_dim"))?;
        let right_half = stated(right.all("left_dim or right_dim"))?;
        let (left_dim, right_dim) = (left_half.stride, right_half.stride);
        self.fire(
            Fire::at(
                "layout/deinterleave.cuh",
                "::pie::layout::split_rows<::pie::bf16>",
            )
            .apply(route_rows(left_half.rows, left_dim.0)),
            &[
                src.arg(),
                left.arg(),
                right.arg(),
                left_dim.arg(),
                right_dim.arg(),
            ],
        )
    }

    fn select<T: kernels::points::Scalar>(
        &self,
        table: In<Tensor<T>>,
        layer: u32,
        width: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let width = stated_width(width, "the slice width this select states")?;
        let dst = stated(y.all("the selected slice's width"))?;
        if dst.width != width {
            return Err(Refusal::Narrow {
                what: "the selected slice is not the width the statement states",
                at: i64::from(dst.width),
            });
        }
        let src = stated(table.over(dst.rows, "the relayed table's row"))?;
        let stride = *src.stride;
        let offset = i32::try_from(layer)
            .ok()
            .and_then(|l| l.checked_mul(width))
            .ok_or(Refusal::Wide {
                what: "the column this layer's slice starts at",
                at: i64::from(layer) * i64::from(width),
                max: i64::from(i32::MAX),
            })?;
        if offset.checked_add(width).is_none_or(|end| end > stride) {
            return Err(Refusal::Narrow {
                what: "the relayed row does not reach this layer's slice",
                at: i64::from(stride),
            });
        }
        self.fire(
            Fire::at(
                "layout/deinterleave.cuh",
                crate::jit::symbol(&format!("::pie::layout::select<{}>", T::CPP)),
            )
            .apply(route_rows(dst.rows, width)),
            &[
                table.arg(),
                y.arg(),
                stride.arg(),
                offset.arg(),
                width.arg(),
            ],
        )
    }
}

const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 {
        head_dim.unsigned_abs()
    } else {
        256
    }
}

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn envelope_update_appended(
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
