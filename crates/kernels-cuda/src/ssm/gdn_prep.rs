use super::*;
use crate::jit::kernel;

pub fn gdn_gates(
    ctx: &Ctx,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gdn_gates";
    dtype_dispatch!(OP, ba.dtype, { Bf16 => () });
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` lands an f32 decay row");
    if ba.width == 0 || !ba.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection does not halve into value heads",
                ba.width
            ),
        ));
    }
    let v_heads = ba.width / 2;
    debug_assert!(
        gates.rows == ba.rows && gates.width == ba.width,
        "the fused `[g_log | beta]` row rides the projection it is derived from"
    );
    let rows = nonzero(OP, "rows", ba.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            GDN_PREP,
            "::pie::ssm::gdn_prep_ba_gates<::pie::bf16>",
        ))
        .apply(Launch::grid(
            [rows, v_heads.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

impl Delta {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn of(
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        y: &Tensor,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the key heads this statement states", k_heads)?;
        nonzero(op, "the value heads this statement states", v_heads)?;
        nonzero(op, "the key head width this statement states", k_dim)?;
        nonzero(op, "the value head width this statement states", v_dim)?;
        if !v_heads.is_multiple_of(k_heads) {
            return Err(refuse(
                op,
                format!(
                    "the {v_heads} value heads are not a whole number of the {k_heads} key heads"
                ),
            ));
        }
        debug_assert_eq!(
            u64::from(qkv.width),
            2 * u64::from(k_heads) * u64::from(k_dim) + u64::from(v_heads) * u64::from(v_dim),
            "the post-convolution qkv's row is the four stated head numbers"
        );
        debug_assert!(
            gates.rows == qkv.rows && gates.width == 2 * v_heads,
            "the fused `[g_log | beta]` row is two entries per value head"
        );
        debug_assert!(
            y.rows == qkv.rows && u64::from(y.width) == u64::from(v_heads) * u64::from(v_dim),
            "the recurrence lands one value plane per row"
        );
        nonzero(op, "rows", qkv.rows)?;
        Ok(Self {
            n: qkv.rows,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            conv_dim: qkv.width,
        })
    }

    pub(super) const fn elems(self, heads: u32, width: u32) -> u64 {
        self.n as u64 * heads as u64 * width as u64
    }

    pub(super) fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        planes: StagePlanes,
    ) -> Result<DeltaStaged, Error> {
        let key = self.elems(self.k_heads, self.k_dim);
        let val = self.elems(self.v_heads, self.v_dim);
        let decay = self.elems(self.v_heads, 1);

        let (qk_name, v_name, gates_name) = planes.names();
        let qk = plane(ctx, op, qk_name, 2 * key)?;
        let v = plane(ctx, op, v_name, val)?;
        let gb = plane(ctx, op, gates_name, 2 * decay)?;
        let staged = DeltaStaged {
            q_norm: qk,
            k_norm: qk + key * u64::from(FLOAT),
            v,
            g_log: gb,
            beta: gb + decay * u64::from(FLOAT),
        };

        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        ctx.fire(
            op,
            Fire::at(kernel(
                GDN_PREP,
                "::pie::ssm::gdn_prep_qk_norm<::pie::bf16, 128>",
            ))
            .apply(Launch::grid([self.n, self.k_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                ctx.stage(),
            ],
        )?;
        ctx.fire(
            op,
            Fire::at(kernel(
                GDN_PREP,
                "::pie::ssm::gdn_prep_v_gates<::pie::bf16, 128>",
            ))
            .apply(Launch::grid([self.n, self.v_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.v_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}
