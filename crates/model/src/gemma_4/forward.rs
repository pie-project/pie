use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{
    Classify, Facts, Forward, Input, KvSpec, Norm, Pages, Predicate, Request, Tensor, Value,
    kernels, merge, seam,
};

use super::model::{Attn, AttnBanks, AttnKind, Model};

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
    pub masked: bool,
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self {
            qo_one: r.query_len() == 1,
            masked: r.has_custom_mask(),
        }
    }
}

impl<W1: Dtype, K: KvDtype, const TP: usize> Forward for Model<W1, K, TP> {
    type Facts = Facts;

    fn caches(&self) -> KvSpec {
        let mut c = KvSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            if let AttnBanks::Owned { .. } = &w.attn.banks {
                let ki = &w.attn.kind;
                c.kv(
                    format!("kv.{l}"),
                    [2, ki.kv_heads() as u64 * ki.head_dim() as u64],
                );
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        // THE RELAY IS A STACK OF `dim`-WIDE BLOCKS, ONE PER LAYER, and both
        // of the two things this line used to get wrong come from reading it
        // as one flat row.
        //
        // * The projection's norm is stated over ONE BLOCK — the checkpoint
        //   ships `per_layer_projection_norm` at `[dim]` and the reference
        //   reshapes to `[.., layers, dim]` before applying it, so the RMS is
        //   a layer's own 256 numbers. Normalising the whole `layers * dim`
        //   row divides every block by one figure and reads the `[dim]`
        //   weight `layers` times past its end. `rmsnorm_per_head` IS this
        //   reading — the point's own words, "normalise each `head_dim`-wide
        //   slice of a row independently; the weight is one head wide" — and
        //   `layout::select` downstream already cuts the same blocks.
        // * The two halves are averaged in QUADRATURE, not summed: the
        //   reference scales the sum by `2^-0.5`, which is what keeps a sum
        //   of two unit-variance planes at unit variance. Dropping it made
        //   every layer's relay √2 too loud.
        //
        // Measured against a transformers 5.15.1 forward on the cached
        // checkpoint, which is the third party that decides.
        // **THE TABLE IS GATHERED PER LAYER NOW, AND THE `select` MOVED RATHER
        // THAN WENT AWAY.**
        //
        // This built the whole relay here — one `layout.embed` of the whole
        // `[vocab, layers * ple_dim]` row, added to the projection, and then a
        // `layout.select` of the RELAY inside the loop. The gather is the one
        // thing that could not stay: `PleLayer::table` says why a 5.25 GiB bank
        // is a bank no shader plane can bind, and the fix is one param per
        // layer.
        //
        // So the projection is what survives to the loop, and the loop selects
        // IT. Elementwise nothing changes: the same slice of the same
        // normalised projection meets the same 256 embedded columns, scaled by
        // the same two constants. What changes is the statement graph — one
        // wide embed and one wide add become forty-two narrow ones — and
        // `driver-cuda`'s banked 785 at 7.5938 is what says the arithmetic did
        // not move with it.
        let relay = m.ple.as_ref().map(|ple| {
            let proj =
                kernels::gemm::matmul(&y, &ple.model_proj) * (m.hidden as f32).sqrt().recip();
            let n = &ple.model_norm;
            (
                ple,
                kernels::norm::rmsnorm_per_head(&proj, &n.weight, ple.dim, n.eps),
            )
        });

        for (l, w) in inputs.layers(&m.layers) {
            let an = &w.attn_norm;
            let normed = kernels::norm::rmsnorm(&y, &an.weight, an.eps);
            let at = &w.attn;
            let d = at.kind.head_dim();
            let pages = inputs.kv(&at.kv);

            let q = match &at.banks {
                AttnBanks::Shared { q_proj } => q_only(&normed, &inputs.positions(), at, q_proj),
                AttnBanks::Owned { qkv, k_norm } => {
                    if inputs.cuda() && K::NATIVE_BF16 && at.kind.sliding() {
                        let fused = Facts::qo_one() & !Facts::masked();
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = inputs.positions().split(&fused);
                        let qf = kernels::cuda::qkv_fused_qknorm_rope_vnorm_write(
                            &kernels::gemm::matmul(&fast_x, qkv),
                            &at.q_norm,
                            k_norm,
                            at.kind.kv_heads(),
                            d,
                            &pages,
                            at.kind.theta(),
                            &fast_pos,
                        );
                        let qr = qkv_unfused(&rest_x, &rest_pos, at, qkv, k_norm, &pages);
                        merge![qf, qr]
                    } else {
                        qkv_unfused(&normed, &inputs.positions(), at, qkv, k_norm, &pages)
                    }
                }
            };

            seam::at(seam::ATTN_Q, (&q,));

            let win = at.kind.window();
            let [mq, dq, p] = q.split([Facts::masked(), Facts::qo_one(), Predicate::rest()]);
            let a = merge![
                kernels::attention::masked(
                    &kernels::query_windows(&mq),
                    &pages,
                    win,
                    d,
                    at.sm_scale
                ),
                kernels::attention::decode(&dq, &pages, win, d, at.sm_scale),
                kernels::attention::prefill(
                    &kernels::query_windows(&p),
                    &pages,
                    win,
                    d,
                    at.kind.kv_heads(),
                    at.sm_scale
                ),
            ];
            seam::at(seam::ATTN_OUT, (&a,));
            let o = kernels::gemm::attention_landing(&a, &w.o_proj);
            let o = kernels::dist::reduce::<TP>(o);

            let pan = &w.post_attn_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&o, &pan.weight, pan.eps), &y);
            let pff = &w.pre_ffw_norm;
            let mlp_in = kernels::norm::rmsnorm(&y, &pff.weight, pff.eps);

            let act = kernels::mlp::geglu_tanh_packed(
                &kernels::gemm::matmul(&mlp_in, &w.gate_up),
                w.inter,
            );
            let f = kernels::gemm::matmul(&act, &w.down);
            let f = kernels::dist::reduce::<TP>(f);
            let pfn = &w.post_ffw_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&f, &pfn.weight, pfn.eps), &y);

            if let Some((ple, proj)) = &relay {
                let lp = &ple.per_layer[l as usize];
                let table =
                    kernels::layout::embed(&ids, &lp.table, m.vocab) * (ple.dim as f32).sqrt();
                let relay =
                    kernels::norm::residual_add(&table, &kernels::layout::select(proj, l, ple.dim))
                        * std::f32::consts::FRAC_1_SQRT_2;
                let gated = kernels::mlp::geglu_tanh(&kernels::gemm::matmul(&y, &lp.gate), &relay);
                let out = kernels::gemm::matmul(&gated, &lp.proj);
                let out = kernels::norm::rmsnorm(&out, &lp.norm.weight, lp.norm.eps);
                // THE SCALAR RIDES THE WHOLE LAYER, NOT THE RELAY'S BRANCH.
                // It scaled `out` alone here, and the checkpoint says
                // otherwise: `layer_scalar` is a `[1]` buffer of its own,
                // trained (0.06 at layer 0 to 0.89 at layer 37 on e4b, never
                // 1.0), and the reference multiplies the layer's RESULT by
                // it after the relay's residual lands. Scaling the branch
                // instead leaves the residual stream un-damped and the error
                // compounds 42 times — measured against a transformers 5.15.1
                // forward on the cached checkpoint, which is the third party
                // that decides.
                y = kernels::norm::scale(&lp.scalar, &kernels::norm::residual_add(&out, &y));
            }
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        let logits = kernels::gemm::lm_head(&x, &m.embed);
        if let Some(cap) = m.softcap {
            kernels::attention::logit_softcap(&logits, cap)
        } else {
            logits
        }
    }
}

fn qkv_unfused<W1: Dtype>(
    x: &Value,
    pos: &Value,
    at: &Attn<W1>,
    qkv: &Tensor<W1>,
    k_norm: &Norm<W1>,
    pages: &Pages,
) -> Value {
    let d = at.kind.head_dim();
    let (q, k, v) = kernels::layout::split_qkv(
        &kernels::gemm::matmul(x, qkv),
        at.q_heads * d,
        at.kind.kv_heads() * d,
    );
    seam::at(seam::ATTN_QV, (&q, &v));
    let v = kernels::norm::rmsnorm_no_scale(&v, d, at.q_norm.eps);
    let q = kernels::norm::rmsnorm_per_head(&q, &at.q_norm.weight, d, at.q_norm.eps);
    let k = kernels::norm::rmsnorm_per_head(&k, &k_norm.weight, d, k_norm.eps);
    let (q, k) = match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::rope::partial(&q, &k, pos, *rotary_dim, d, *theta),
        // NeoX pairing: gemma rotates `d` against `d + d/2`.
        AttnKind::Sliding { theta, .. } => kernels::rope::full(&q, &k, pos, d, *theta, false),
    };
    kernels::attention::kv_append(&k, &v, pages);
    q
}

fn q_only<W1: Dtype>(x: &Value, pos: &Value, at: &Attn<W1>, q_proj: &Tensor<W1>) -> Value {
    let d = at.kind.head_dim();
    let q = kernels::norm::rmsnorm_per_head(
        &kernels::gemm::matmul(x, q_proj),
        &at.q_norm.weight,
        d,
        at.q_norm.eps,
    );
    match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::rope::partial_q(&q, pos, *rotary_dim, d, *theta),
        AttnKind::Sliding { theta, .. } => kernels::rope::partial_q(&q, pos, d, d, *theta),
    }
}
