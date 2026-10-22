use super::*;

const KDA: &str = "ssm/kda.cu";

const KDA_CHUNK_SM80: &str = "ssm/kda_chunk_sm80.cu";

const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

#[derive(Clone, Copy)]
struct Kda {
    n: u32,
    heads: u32,
    head_dim: u32,
    width: u32,
}

#[derive(Clone, Copy)]
struct KdaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    gate: u64,
    beta: u64,
}

impl Kda {
    fn of(
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        y: &Tensor,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the KDA heads this statement states", heads)?;
        nonzero(op, "the KDA head width this statement states", head_dim)?;
        let width = heads.checked_mul(head_dim).ok_or_else(|| {
            refuse(
                op,
                format!("the KDA plane will not size: {heads} heads x {head_dim} wide"),
            )
        })?;
        debug_assert_eq!(
            u64::from(mixed.width),
            3 * u64::from(width),
            "the post-convolution `[q | k | v]` row is three head planes"
        );
        debug_assert!(
            f.rows == mixed.rows && f.width == width,
            "the forget projection's row is one head plane"
        );
        debug_assert!(
            b.rows == mixed.rows && b.width == heads,
            "the beta projection's row is one entry per head"
        );
        debug_assert!(
            y.rows == mixed.rows && y.width == width,
            "the recurrence lands one head plane per row"
        );
        nonzero(op, "rows", mixed.rows)?;
        Ok(Self {
            n: mixed.rows,
            heads,
            head_dim,
            width,
        })
    }

    /// The scan walks each head's `D x D` state in place, so a fire in which
    /// some row may not fold its boundary (no fold asked, or a predicate
    /// naming the lanes that do) gets a working copy per row and head; a
    /// fire that folds everywhere walks the slots themselves.
    fn work(
        self,
        ctx: &Ctx,
        op: &'static str,
        state: &RecurrentPool,
        rows: u32,
    ) -> Result<u64, Error> {
        if state.write_state && state.write_state_mask.is_absent() {
            return Ok(0);
        }
        let cells = u64::from(self.head_dim) * u64::from(self.head_dim);
        plane(
            ctx,
            op,
            "attn.ssm_kda_work",
            u64::from(rows) * u64::from(self.heads) * cells,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        norm_eps: f32,
        gate_floor: f32,
    ) -> Result<KdaStaged, Error> {
        const PLANES: u32 = 3;

        let wide = u64::from(self.n) * u64::from(self.width);
        let decay = u64::from(self.n) * u64::from(self.heads);

        let qkv = plane(ctx, op, "attn.ssm_kda_qkv", 3 * wide)?;
        let gb = plane(ctx, op, "attn.ssm_kda_gates", wide + decay)?;
        let staged = KdaStaged {
            q_norm: qkv,
            k_norm: qkv + wide * u64::from(FLOAT),
            v: qkv + 2 * wide * u64::from(FLOAT),
            gate: gb,
            beta: gb + wide * u64::from(FLOAT),
        };

        ctx.fire(
            op,
            // A warp a head where the row is whole heads (a decode's row over
            // the card), else a block a plane.
            if self.head_dim > 0 && self.width.is_multiple_of(self.head_dim) {
                Fire::at(KDA, "::pie::ssm::kda_qkv_prep_heads<::pie::bf16, 128>").apply(
                    Launch::grid(
                        [
                            self.n,
                            PLANES,
                            (self.width / self.head_dim).div_ceil(PREP_BLOCK / WARP),
                        ],
                        [PREP_BLOCK, 1, 1],
                    )
                    .programmatic(ctx.tuning().pdl),
                )
            } else {
                Fire::at(KDA, "::pie::ssm::kda_qkv_prep<::pie::bf16, 128>")
                    .apply(Launch::grid([self.n, PLANES, 1], [PREP_BLOCK, 1, 1]))
            },
            &[
                mixed.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                stated(op, self.width)?.arg(),
                stated(op, self.head_dim)?.arg(),
                norm_eps.arg(),
                ctx.stage(),
            ],
        )?;
        // A row a block, four channels a thread.
        let gates = if self.head_dim.is_multiple_of(4) {
            Fire::at(KDA, "::pie::ssm::kda_gate_beta_rows<::pie::bf16>").apply(
                Launch::grid([self.n, self.width.div_ceil(1024), 1], [256, 1, 1])
                    .programmatic(ctx.tuning().pdl),
            )
        } else {
            Fire::at(KDA, "::pie::ssm::kda_gate_beta<::pie::bf16>").apply(Launch::grid(
                [self.n, self.heads, 1],
                [self.head_dim.clamp(WARP, PREP_BLOCK), 1, 1],
            ))
        };
        ctx.fire(
            op,
            gates,
            &[
                f.arg(),
                b.arg(),
                a_log.arg(),
                dt_bias.arg(),
                ArgValue::Ptr(staged.gate),
                ArgValue::Ptr(staged.beta),
                stated(op, self.n)?.arg(),
                stated(op, self.heads)?.arg(),
                stated(op, self.head_dim)?.arg(),
                gate_floor.arg(), // the decay's lower bound; zero leaves it unbounded
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kda_step(
    ctx: &Ctx,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_step";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "kda_step", true, false, false)?;
    let work = shape.work(ctx, OP, state, shape.n)?;
    if ctx.tuning().ssm_kda_step_fused && matches!(shape.head_dim, 32 | 64 | 128) {
        // Enough blocks to fill the device twice: each (row, head) split
        // over up to eight slices of its value rows.
        let sms = ctx.multiprocessors().unwrap_or(148);
        let blocks = shape.n.saturating_mul(shape.heads).max(1);
        let splits = (2 * sms)
            .div_ceil(blocks)
            .clamp(1, 8)
            .min(shape.head_dim.max(1));
        return ctx.fire(
            OP,
            Fire::at(
                KDA,
                symbol(&format!(
                    "::pie::ssm::kda_step_fused<::pie::bf16, {}>",
                    shape.head_dim / WARP
                )),
            )
            .apply(
                Launch::grid([shape.n, shape.heads, splits], [STEP_BLOCK, 1, 1])
                    .smem(kda_shmem(shape.head_dim) + shape.head_dim * FLOAT),
            ),
            &[
                mixed.arg(),
                f.arg(),
                b.arg(),
                a_log.arg(),
                dt_bias.arg(),
                gate_floor.arg(),
                norm_eps.arg(),
                state.slab.arg(),
                state.slot_ids.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, shape.heads)?.arg(),
                stated(OP, shape.head_dim)?.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ArgValue::Ptr(work),
                ctx.stage(),
            ],
        );
    }
    let staged = shape.stage(ctx, OP, mixed, f, b, dt_bias, a_log, norm_eps, gate_floor)?;
    ctx.fire(
        OP,
        Fire::at(KDA, "::pie::ssm::kda_step").apply(
            Launch::grid(
                [
                    shape.n,
                    shape.heads,
                    kda_step_splits(ctx, shape.n, shape.heads),
                ],
                [STEP_BLOCK, 1, 1],
            )
            .smem(kda_shmem(shape.head_dim))
            .programmatic(ctx.tuning().pdl),
        ),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.gate),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ArgValue::Ptr(work),
            ctx.stage(),
        ],
    )
}

/// The blocks a decode step's head splits its value rows over: as many as
/// keep two blocks a multiprocessor where the rows and heads alone do not, at
/// most eight.
fn kda_step_splits(ctx: &Ctx, n: u32, heads: u32) -> u32 {
    let sms = ctx.multiprocessors().unwrap_or(132);
    (2 * sms / (n * heads).max(1)).clamp(1, 8)
}

/// The raw decay and beta rows the tensor-core prep forms its gates from.
struct KdaGates {
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    floor: f32,
}

/// [`kda_chunked`] on the tensor cores in sixteen-token chunks: two warps a
/// (chunk, head) land the chunk's decayed queries and fold keys, its decay,
/// its update directions `w` and `u` and its decayed causal scores; then a
/// block a (request, head, 64 value columns) walks the chunks with the state
/// in f32. 128-wide heads.
#[allow(clippy::too_many_arguments)]
fn kda_chunked_mma(
    ctx: &Ctx,
    op: &'static str,
    shape: Kda,
    gates: KdaGates,
    mixed: RaggedTensor,
    norm_eps: f32,
    state: &RecurrentPool,
    lanes: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 16;
    const D: u32 = 128;
    const TASKS: u32 = 2;
    // `KDA16_TASK_SMEM`, and the scan's three `Kda16Scan<2>::STAGE`s and
    // their barriers.
    const LD: u32 = 136;
    const LDT: u32 = 24;
    const LDV: u32 = 136;
    const TASK_SMEM: u32 = 3 * CH * LD * 2 + CH * (CH + 1) * FLOAT + 3 * D * FLOAT;
    const STAGE: u32 = 3 * CH * LD * 2 + CH * LDV * 2 + 2 * CH * LDT * 2 + D * FLOAT;
    let heads = shape.heads;
    let slots = shape.n.div_ceil(CH) + lanes;
    let rows = u64::from(shape.n + CH) * u64::from(heads) * u64::from(D);
    let qg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_qg", rows)?;
    let kg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_kg", rows)?;
    let ke = plane_bf16(ctx, op, "attn.ssm_kda_chunk_ke", rows)?;
    let squares = u64::from(slots) * u64::from(heads) * u64::from(CH * CH);
    let tb = plane_bf16(ctx, op, "attn.ssm_kda_chunk_t", squares)?;
    let pm = plane_bf16(ctx, op, "attn.ssm_kda_chunk_p", squares)?;
    let gl = plane(
        ctx,
        op,
        "attn.ssm_kda_chunk_gl",
        u64::from(slots) * u64::from(heads) * u64::from(D),
    )?;
    ctx.fire(
        op,
        Fire::at(KDA_CHUNK_SM80, "::pie::ssm::kda_chunk_prep").apply(
            Launch::grid([heads.div_ceil(TASKS), slots, 1], [TASKS * 2 * WARP, 1, 1])
                .smem(TASKS * TASK_SMEM),
        ),
        &[
            mixed.data.arg(),
            norm_eps.arg(),
            gates.f.arg(),
            gates.b.arg(),
            gates.a_log.arg(),
            gates.dt_bias.arg(),
            gates.floor.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(ke),
            ArgValue::Ptr(tb),
            ArgValue::Ptr(pm),
            ArgValue::Ptr(gl),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(KDA_CHUNK_SM80, "::pie::ssm::kda_chunk_scan")
            .apply(Launch::grid([lanes * heads * 2, 1, 1], [128, 1, 1]).smem(3 * STAGE + 64)),
        &[
            ArgValue::Ptr(gl),
            mixed.data.arg(),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(ke),
            ArgValue::Ptr(tb),
            ArgValue::Ptr(pm),
            state.slab.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            stated(op, ctx.multiprocessors().unwrap_or(1).max(1))?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn kda_chunked(
    ctx: &Ctx,
    mixed: RaggedTensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_chunked";

    const PREFILL_MAX_WARPS: u32 = 32;

    dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    f32_state(OP, state)?;
    seated(OP, state, "kda_chunked", true, true, true)?;
    if ctx.tuning().ssm_kda_chunked && shape.head_dim == 128 {
        // The tensor-core prep reads the raw rows and forms the norms and
        // gates itself.
        let gates = KdaGates {
            f,
            b,
            dt_bias,
            a_log,
            floor: gate_floor,
        };
        return kda_chunked_mma(ctx, OP, shape, gates, mixed, norm_eps, state, lanes, y);
    }
    let staged = shape.stage(
        ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps, gate_floor,
    )?;
    let work = shape.work(ctx, OP, state, lanes)?;
    ctx.fire(
        OP,
        Fire::at(KDA, "::pie::ssm::kda_chunked").apply(
            Launch::grid(
                [lanes, shape.heads, 1],
                [shape.head_dim.min(PREFILL_MAX_WARPS) * WARP, 1, 1],
            )
            .smem(kda_shmem(shape.head_dim)),
        ),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.gate),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            state.write_state.arg(),
            state.commit_len.arg(),
            state.write_state_mask.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(work),
            ctx.stage(),
        ],
    )
}
