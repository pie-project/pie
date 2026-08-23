use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, Forward, Input, KvSpec, Norm, Pages, Predicate, Request, Tensor, Value};

use super::model::{Attn, AttnBanks, AttnKind, GateUp, Head, Model, Qkv};

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
                c.kv(format!("kv.{l}"), [2, ki.kv_heads() as u64 * ki.head_dim() as u64]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::layout::embed(&ids, &m.embed) * (m.hidden as f32).sqrt();

        let relay = m.ple.as_ref().map(|ple| {
            let table = kernels::layout::embed(&ids, &ple.table) * (ple.dim as f32).sqrt();
            let proj = kernels::gemm::matmul(&y, &ple.model_proj) * (m.hidden as f32).sqrt().recip();
            let n = &ple.model_norm;
            (ple, kernels::norm::residual_add(&table, &kernels::norm::rmsnorm(&proj, &n.weight, n.eps)))
        });

        let last = m.layers.len() as u32 - 1;
        let first = &m.layers[0].attn_norm;
        let mut normed = kernels::norm::rmsnorm(&y, &first.weight, first.eps);

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let at = &w.attn;
            let d = at.kind.head_dim();
            let pages = inputs.kv(&at.kv);

            let q = match &at.banks {
                AttnBanks::Shared { q_proj } => {
                    q_only(&normed, &inputs.positions(), at, q_proj)
                }
                AttnBanks::Owned { qkv, k_norm } => {
                    if let (Qkv::Packed(bank), true) =
                        (qkv, inputs.cuda() && K::NATIVE_BF16 && at.kind.sliding())
                    {
                        let fused = Facts::qo_one() & !Facts::masked();
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = inputs.positions().split(&fused);
                        let qf = kernels::cuda::qkv_fused_qknorm_rope_vnorm_write(
                            &kernels::gemm::matmul(&fast_x, bank),
                            &at.q_norm,
                            k_norm,
                            at.kind.kv_heads(),
                            d,
                            &pages,
                            at.kind.theta(),
                            &fast_pos,
                        );
                        let qr = qkv_unfused(&rest_x, &rest_pos, at, qkv, k_norm, &pages, l);
                        merge![qf, qr]
                    } else {
                        qkv_unfused(&normed, &inputs.positions(), at, qkv, k_norm, &pages, l)
                    }
                }
            };

            seam::at(seam::ATTN_Q, (&q,), l);

            let win = at.kind.window();
            let [mq, dq, p] = q.split([Facts::masked(), Facts::qo_one(), Predicate::rest()]);
            let a = merge![
                kernels::attention::masked(&kernels::query_windows(&mq), &pages, win, d, at.sm_scale),
                kernels::attention::decode(&dq, &pages, win, d, at.sm_scale),
                kernels::attention::prefill(&kernels::query_windows(&p), &pages, win, d, at.kind.kv_heads(), at.sm_scale),
            ];
            seam::at(seam::ATTN_OUT, (&a,), l);
            let o = kernels::gemm::attention_landing(&a, &w.o_proj, l);
            let o = if TP > 1 { kernels::dist::all_reduce(&o) } else { o };

            let pan = &w.post_attn_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&o, &pan.weight, pan.eps), &y);
            let pff = &w.pre_ffw_norm;
            let mlp_in = kernels::norm::rmsnorm(&y, &pff.weight, pff.eps);

            let act = match &w.gate_up {
                GateUp::Packed { bank, inter } => {
                    kernels::mlp::geglu_tanh_packed(&kernels::gemm::matmul(&mlp_in, bank), *inter)
                }
                GateUp::Split { gate, up } => {
                    kernels::mlp::geglu_tanh(&kernels::gemm::matmul(&mlp_in, gate), &kernels::gemm::matmul(&mlp_in, up))
                }
            };
            let f = kernels::gemm::matmul(&act, &w.down);
            let f = if TP > 1 { kernels::dist::all_reduce(&f) } else { f };
            let pfn = &w.post_ffw_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&f, &pfn.weight, pfn.eps), &y);

            if let Some((ple, relay)) = &relay {
                let lp = &ple.per_layer[l as usize];
                let gated = kernels::mlp::geglu_tanh(&kernels::gemm::matmul(&y, &lp.gate), &kernels::layout::select(relay, l));
                let out = kernels::gemm::matmul(&gated, &lp.proj);
                let out = kernels::norm::rmsnorm(&out, &lp.norm.weight, lp.norm.eps);
                y = kernels::norm::residual_add(&kernels::norm::scale(&lp.scalar, &out), &y);
            }
            if l < last {
                let next = &m.layers[l as usize + 1].attn_norm;
                normed = kernels::norm::rmsnorm(&y, &next.weight, next.eps);
            }
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        let logits = match &m.head {
            Head::Tied => kernels::gemm::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::gemm::lm_head(&x, bank),
        };
        let logits = if let Some(cap) = m.softcap {
            kernels::attention::logit_softcap(&logits, cap)
        } else {
            logits
        };

        logits
    }
}

fn qkv_unfused<W1: Dtype>(
    x: &Value,
    pos: &Value,
    at: &Attn<W1>,
    qkv: &Qkv<W1>,
    k_norm: &Norm<W1>,
    pages: &Pages,
    l: u32,
) -> Value {
    let d = at.kind.head_dim();
    let (q, k, v) = match qkv {
        Qkv::Packed(bank) => {
            kernels::layout::split_qkv(&kernels::gemm::matmul(x, bank), at.q_heads * d, at.kind.kv_heads() * d)
        }
        Qkv::Split { q, k, v } => {
            (kernels::gemm::matmul(x, q), kernels::gemm::matmul(x, k), kernels::gemm::matmul(x, v))
        }
    };
    seam::at(seam::ATTN_QV, (&q, &v), l);
    let v = kernels::norm::rmsnorm_no_scale(&v, d, at.q_norm.eps);
    let q = kernels::norm::rmsnorm_per_head(&q, &at.q_norm.weight, d, at.q_norm.eps);
    let k = kernels::norm::rmsnorm_per_head(&k, &k_norm.weight, d, k_norm.eps);
    let (q, k) = match &at.kind {
        AttnKind::Full { rotary_dim, theta, .. } => kernels::rope::partial(&q, &k, pos, *rotary_dim, d, *theta),
        // NeoX pairing: gemma rotates `d` against `d + d/2`.
        AttnKind::Sliding { theta, .. } => kernels::rope::full(&q, &k, pos, d, *theta, false),
    };
    kernels::attention::kv_append(&k, &v, pages);
    q
}

fn q_only<W1: Dtype>(x: &Value, pos: &Value, at: &Attn<W1>, q_proj: &Tensor<W1>) -> Value {
    let d = at.kind.head_dim();
    let q = kernels::norm::rmsnorm_per_head(&kernels::gemm::matmul(x, q_proj), &at.q_norm.weight, d, at.q_norm.eps);
    match &at.kind {
        AttnKind::Full { rotary_dim, theta, .. } => kernels::rope::partial_q(&q, pos, *rotary_dim, d, *theta),
        AttnKind::Sliding { theta, .. } => kernels::rope::partial_q(&q, pos, d, d, *theta),
    }
}
