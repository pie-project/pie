//! Gemma 4 dense (E2B, E4B) as the spatial kernels see it, read off its
//! trace by the catalog's names (`models/gemma_4.poem`'s layout).
//!
//! A Gemma 4 layer differs from a Qwen3 layer in ways the paged kernels
//! don't absorb, so the family reads apart from [`ModelSpec`]:
//!
//! - Sliding-window layers (`window`, `head_dim`, RoPE over every pair) and
//!   global layers (`global_head_dim`, RoPE over `global_rotary` dims: pairs
//!   `(i, i + d/2)` for `i < global_rotary / 2`) in the trace's order.
//! - The shared tail: its layers have a q projection only and attend the kv
//!   of the last owning layer of their kind (`kv_source`).
//! - Sandwich norms (`post_attn_norm`, `pre_ffw_norm`, `post_ffw_norm`), v
//!   RMS-normed without a weight, a softmax scale of 1.
//! - A GELU-tanh GLU, twice as wide in the shared tail for E2B (`inter`
//!   a layer).
//! - Per-layer embeddings: each layer adds `proj(gelu(gate(x)) * ple_l)`,
//!   normed, then scales the residual by its scalar; `ple_l` mixes the
//!   token's row of the layer's table with a projection of the embedding.
//! - A tied lm_head, `sqrt(hidden)`-scaled embedding, softcapped logits.
//!
//! [`ModelSpec`]: super::model::ModelSpec

use poem_ir::{Attn, Dtype, Elemwise, Operation, Trace, Turn};

use super::model::{Refused, refuse};

/// What a layer attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// The last `window` positions, `head_dim` heads.
    Sliding,
    /// Every position, `global_head_dim` heads, a partial RoPE.
    Global,
}

/// A Gemma 4 dense model's facts.
#[derive(Debug, Clone, PartialEq)]
pub struct G4Spec {
    pub name: String,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub global_kv_heads: usize,
    pub global_head_dim: usize,
    /// Dims a global head turns (the first `global_rotary / 2` pairs).
    pub global_rotary: usize,
    pub theta: f32,
    pub global_theta: f32,
    pub window: usize,
    pub vocab: usize,
    /// Per-layer embedding width (`None`: no PLE).
    pub ple_dim: Option<usize>,
    pub softcap: Option<f32>,
    pub eps: f32,
    pub weights: Dtype,
    /// Each layer's reading, its MLP width, and the layer whose kv it reads
    /// (itself unless it is in the shared tail).
    pub readings: Vec<Reading>,
    pub inter: Vec<usize>,
    pub kv_source: Vec<usize>,
}

impl G4Spec {
    pub fn layers(&self) -> usize {
        self.readings.len()
    }

    /// The layers that write kv.
    pub fn owners(&self) -> usize {
        (0..self.layers())
            .filter(|&l| self.kv_source[l] == l)
            .count()
    }

    /// Reads a Gemma 4 dense trace, or refuses what the units don't run.
    pub fn read(trace: &Trace) -> Result<G4Spec, Refused> {
        let name = trace.name.clone();
        let shape = |p: &str| -> Option<&[u64]> {
            trace
                .params
                .iter()
                .find(|q| q.name == p)
                .map(|q| q.shape.as_slice())
        };
        let need = |p: &str| -> Result<&[u64], Refused> {
            shape(p).ok_or_else(|| Refused(format!("{name}: no param {p}")))
        };
        if trace
            .params
            .iter()
            .any(|p| p.name.contains("experts") || p.name.contains("router"))
        {
            return refuse(format!("{name}: mixture of experts"));
        }
        if trace
            .params
            .iter()
            .any(|p| p.name.starts_with("tower.") || p.name.contains("draft"))
        {
            return refuse(format!("{name}: a vision tower or draft head"));
        }
        let embed = need("embed")?;
        let (vocab, hidden) = (embed[0] as usize, embed[1] as usize);
        if shape("lm_head").is_some() {
            return refuse(format!("{name}: an untied lm_head"));
        }
        let weights = trace
            .params
            .iter()
            .find(|p| p.name == "layers.0.o_proj")
            .map(|p| p.dtype);
        let Some(weights) = weights else {
            return refuse(format!("{name}: no layers.0.o_proj"));
        };

        // The attention ops state the geometry: sliding plans carry a window.
        let (mut sliding, mut global) = (None, None);
        let (mut ropes, mut softcap) = (Vec::new(), None);
        let mut eps = None;
        for n in &trace.nodes {
            match &n.op {
                Operation::Attn(Attn::PlanDecode {
                    q_heads,
                    kv_heads,
                    head_dim,
                    window,
                    ..
                }) => {
                    let g = (*q_heads as usize, *kv_heads as usize, *head_dim as usize);
                    match window {
                        Some(w) => sliding = Some((g, *w as usize)),
                        None => global = Some(g),
                    }
                }
                Operation::Elemwise(Elemwise::Rope {
                    head_dim,
                    theta: t,
                    turn,
                    ..
                }) => {
                    let (rot, span) = match turn {
                        Turn::Partial { rotary_dim, span } => {
                            (*rotary_dim as usize, *span as usize)
                        }
                        Turn::Full { interleaved: false } => {
                            (*head_dim as usize, *head_dim as usize / 2)
                        }
                        other => return refuse(format!("{name}: RoPE turn {other:?}")),
                    };
                    ropes.push((*head_dim as usize, *t, rot, span));
                }
                Operation::Elemwise(Elemwise::Softcap { cap, .. }) => softcap = Some(*cap),
                Operation::Elemwise(Elemwise::Rmsnorm { eps: e, .. }) => {
                    if eps.is_some_and(|x| x != *e) {
                        return refuse(format!("{name}: RMSNorm epsilons differ"));
                    }
                    eps = Some(*e);
                }
                _ => {}
            }
        }
        let (
            Some(((q_heads, kv_heads, head_dim), window)),
            Some((gq, global_kv_heads, global_head_dim)),
        ) = (sliding, global)
        else {
            return refuse(format!("{name}: no sliding and global attention plans"));
        };
        if gq != q_heads {
            return refuse(format!(
                "{name}: {gq} global query heads, {q_heads} sliding"
            ));
        }
        // RoPE by head width: a sliding head turns every pair, a global one
        // the first pairs at the proportional spacing (span d/2).
        let at = |d: usize| ropes.iter().find(|r| r.0 == d).copied();
        let (Some(local), Some(glob)) = (at(head_dim), at(global_head_dim)) else {
            return refuse(format!("{name}: RoPE of both readings not found"));
        };
        if local.2 != head_dim || glob.3 != global_head_dim / 2 {
            return refuse(format!("{name}: RoPE turns {local:?} / {glob:?}"));
        }
        let (theta, global_theta, global_rotary) = (local.1, glob.1, glob.2);

        let mut readings = Vec::new();
        let mut inter = Vec::new();
        let mut kv_source = Vec::new();
        let mut last = [None, None];
        for l in 0.. {
            let Some(qn) = shape(&format!("layers.{l}.attn.q_norm")) else {
                break;
            };
            let r = if qn[0] as usize == global_head_dim {
                Reading::Global
            } else {
                Reading::Sliding
            };
            let gu = need(&format!("layers.{l}.gate_up"))?;
            inter.push(gu[0] as usize / 2);
            let owns = shape(&format!("layers.{l}.attn.banks.qkv")).is_some();
            let k = r as usize;
            kv_source.push(if owns {
                last[k] = Some(l);
                l
            } else {
                last[k]
                    .ok_or_else(|| Refused(format!("{name}: layer {l} reads kv no layer wrote")))?
            });
            readings.push(r);
        }
        if readings.is_empty() {
            return refuse(format!("{name}: no decoder layers"));
        }
        let ple_dim = shape("ple.model_proj").map(|s| s[0] as usize / readings.len());
        if ple_dim.is_some_and(|d| {
            d * readings.len() != need("ple.model_proj").map(|s| s[0] as usize).unwrap_or(0)
        }) {
            return refuse(format!(
                "{name}: a PLE projection that does not split by layer"
            ));
        }
        Ok(G4Spec {
            name,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            global_kv_heads,
            global_head_dim,
            global_rotary,
            theta,
            global_theta,
            window,
            vocab,
            ple_dim,
            softcap,
            eps: eps.unwrap_or(1e-6),
            weights,
            readings,
            inter,
            kv_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn spec(name: &str) -> Result<G4Spec, Refused> {
        let catalog = poem_compiler::catalog::repository();
        let d = catalog
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"));
        G4Spec::read(&d.trace(poem_ir::Platform::Cerebras))
    }

    #[test]
    fn e2b_reads() {
        let s = spec("gemma4-e2b-bf16-kv-bf16").expect("e2b reads");
        assert_eq!(
            (s.hidden, s.q_heads, s.kv_heads, s.head_dim, s.vocab),
            (1536, 8, 1, 256, 262_144)
        );
        assert_eq!(
            (s.global_kv_heads, s.global_head_dim, s.global_rotary),
            (1, 512, 128)
        );
        assert_eq!(
            (s.theta, s.global_theta, s.window),
            (10_000.0, 1_000_000.0, 512)
        );
        assert_eq!((s.ple_dim, s.softcap, s.eps), (Some(256), Some(30.0), 1e-6));
        assert_eq!(s.layers(), 35);
        let globals: Vec<usize> = (0..35)
            .filter(|&l| s.readings[l] == Reading::Global)
            .collect();
        assert_eq!(globals, vec![4, 9, 14, 19, 24, 29, 34]);
        // The shared tail (15..) reads layer 13's sliding kv or layer 14's global.
        assert_eq!(s.owners(), 15);
        assert!((15..35).all(|l| s.kv_source[l] == if globals.contains(&l) { 14 } else { 13 }));
        assert!((0..15).all(|l| s.inter[l] == 6144) && (15..35).all(|l| s.inter[l] == 12_288));
    }

    #[test]
    fn e4b_reads() {
        let s = spec("gemma4-e4b-bf16-kv-bf16").expect("e4b reads");
        assert_eq!(
            (s.hidden, s.q_heads, s.kv_heads, s.head_dim),
            (2560, 8, 2, 256)
        );
        assert_eq!((s.layers(), s.owners(), s.ple_dim), (42, 24, Some(256)));
        assert!(s.inter.iter().all(|&i| i == 10_240));
    }

    #[test]
    fn others_are_refused() {
        assert!(
            spec("gemma4-26b-a4b-bf16-kv-bf16")
                .expect_err("moe")
                .0
                .contains("experts")
        );
    }
}
