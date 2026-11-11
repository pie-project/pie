//! The plan's layer program: what the composer and the schedule read. A program is a function of
//! its plan (the plan is what `plan.json` records and what a build's key digests), so the layer
//! program the build reads is stated from the plan's [`Shape`] in the layer program's own terms
//! ([`of`]); the shape is read off the trace, and the tests pin what this states to what the
//! trace's lowering ([`poem_compiler::spatial::layer::lower`]) reads, for every shipping model.

use poem_compiler::spatial::layer::{Axis, Head, LayerBody, LayerProgram, Mixer, Op, Phase};
use poem_ir::GluAct;

use crate::plan::{Act, LayerKind, Shape};

fn op(op: &'static str, width: u32) -> Op {
    Op {
        op,
        width,
        params: Vec::new(),
    }
}

/// `op` reading the layer's params `names` (a layer's number written `*`).
fn with(op: &'static str, width: u32, names: &[&str]) -> Op {
    Op {
        op,
        width,
        params: names.iter().map(|n| format!("layers.*.{n}")).collect(),
    }
}

/// The residual add after a mixer or the MLP; the output's bias (`bias`), then a sandwich norm
/// on it first.
fn residual(s: &Shape, width: u32, bias: Option<&str>) -> Phase {
    let mut ops = Vec::new();
    if let Some(b) = bias {
        ops.push(with("elemwise.add_bias", width, &[b]));
    }
    if s.norm.sandwich {
        ops.push(op("elemwise.rmsnorm", width));
    }
    ops.push(op("elemwise.residual_add", width));
    Phase::Root {
        axis: Axis::Row,
        ops,
    }
}

/// Layer `l`'s mixer.
fn mixer(s: &Shape, l: usize) -> Mixer {
    let a = &s.attn;
    let global = (
        a.global_kv_heads.unwrap_or(a.kv_heads),
        a.global_head_dim.unwrap_or(a.head_dim),
    );
    let local = (a.kv_heads, a.head_dim);
    let attn = |(kv_heads, head_dim), window, shared_from| Mixer::Attn {
        q_heads: a.q_heads,
        kv_heads,
        head_dim,
        window,
        shared_from,
    };
    match s.layers[l] {
        LayerKind::Gdn => {
            let g = s.gdn.unwrap_or(crate::plan::GdnSpec {
                key_heads: 0,
                value_heads: 0,
                key_dim: 0,
                value_dim: 0,
            });
            Mixer::Gdn {
                k_heads: g.key_heads,
                v_heads: g.value_heads,
                k_dim: g.key_dim,
                v_dim: g.value_dim,
            }
        }
        LayerKind::Global => attn(global, None, None),
        LayerKind::Sliding { window } => attn(local, Some(window), None),
        LayerKind::SharedKv { from, sliding } => {
            let window = match s.layers.get(from as usize) {
                Some(LayerKind::Sliding { window }) if sliding => Some(*window),
                _ => None,
            };
            attn(if sliding { local } else { global }, window, Some(from))
        }
    }
}

/// The MLP's gate activation as the trace states it.
fn glu(act: Act) -> GluAct {
    match act {
        Act::Silu => GluAct::Swiglu,
        Act::GeluTanh => GluAct::GegluTanh,
        // the clamped swiglu is gpt-oss's, its alpha 1.702
        Act::SwigluOai => GluAct::SwigluAlpha { alpha: 1.702 },
    }
}

/// Layer `l`'s body: the input norm, the mixer between its matvecs, the residual; the MLP (its
/// experts routed where the shape has them); a per-layer input after it.
fn body(s: &Shape, l: usize) -> LayerBody {
    let h = s.hidden;
    let norm = Phase::Root {
        axis: Axis::Row,
        ops: vec![op("elemwise.rmsnorm", h)],
    };
    let (width, experts) = match s.moe {
        Some(m) => (m.expert_intermediate, Some((m.experts, m.top_k))),
        None => (s.mlp.width(l as u32), None),
    };
    let a = &s.attn;
    // the q / k / v projections' biases and head norms at the column roots
    let mut qk = Vec::new();
    if a.bias {
        qk.push(with(
            "elemwise.add_bias",
            0,
            &["q_bias", "k_bias", "v_bias"],
        ));
    }
    if a.qk_norm && !matches!(s.layers[l], LayerKind::Gdn) {
        qk.push(with("elemwise.rmsnorm", 0, &["q_norm"]));
        qk.push(with("elemwise.rmsnorm", 0, &["k_norm"]));
    }
    let mixed = if a.sinks && !matches!(s.layers[l], LayerKind::Gdn) {
        vec![op("attn.sink", 0)]
    } else {
        Vec::new()
    };
    let leg = |name: &str| {
        let bias = format!("{name}_bias");
        match s.moe {
            Some(m) if m.expert_bias => with("linear.matmul", width, &[name, &bias]),
            _ => op("linear.matmul", width),
        }
    };
    let mut phases = vec![
        norm.clone(),
        Phase::Matvec {
            from: Axis::Row,
            to: Axis::Col,
            input: h,
            weights: vec![op("linear.matmul", 0)],
            experts: None,
        },
    ];
    if !qk.is_empty() {
        phases.push(Phase::Root {
            axis: Axis::Col,
            ops: qk,
        });
    }
    phases.extend([
        Phase::Mixer {
            mixer: mixer(s, l),
            ops: mixed,
        },
        Phase::Matvec {
            from: Axis::Col,
            to: Axis::Row,
            input: 0,
            weights: vec![op("linear.matmul", h)],
            experts: None,
        },
        residual(s, h, a.o_bias.then_some("o_bias")),
        norm,
    ]);
    if s.moe.is_some_and(|m| m.router_bias) {
        phases.push(Phase::Root {
            axis: Axis::Col,
            ops: vec![with("elemwise.add_bias", 0, &["router_bias"])],
        });
    }
    phases.extend([
        Phase::Matvec {
            from: Axis::Row,
            to: Axis::Col,
            input: h,
            weights: vec![leg("gate")],
            experts,
        },
        Phase::Root {
            axis: Axis::Col,
            ops: vec![op("linear.glu", width)],
        },
        Phase::Matvec {
            from: Axis::Col,
            to: Axis::Row,
            input: width,
            weights: vec![leg("down")],
            experts,
        },
        residual(s, h, None),
    ]);
    if let Some(p) = s.ple {
        phases.push(Phase::SideIn {
            ops: vec![op("gather.embed", p.width)],
        });
    }
    LayerBody {
        phases,
        glu: Some(glu(s.mlp.act)),
    }
}

/// The layer program a plan's shape states: each layer's body, layers whose bodies are the
/// same sharing one.
#[must_use]
pub fn of(s: &Shape) -> LayerProgram {
    let (mut bodies, mut layers) = (Vec::<LayerBody>::new(), Vec::new());
    for l in 0..s.layers.len() {
        let b = body(s, l);
        let at = match bodies.iter().position(|o| *o == b) {
            Some(i) => i,
            None => {
                bodies.push(b);
                bodies.len() - 1
            }
        };
        layers.push(at);
    }
    LayerProgram {
        hidden: s.hidden,
        prologue: Vec::new(),
        bodies,
        layers,
        epilogue: Vec::new(),
        head: Head {
            vocab: s.vocab,
            tied: s.head.tied,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::Needs;
    use crate::target::shape;

    /// What the composer and the schedule read of a layer: its mixers, its routed experts, its
    /// side input, its deferred norm and its MLP's activation.
    fn read(p: &LayerProgram, l: usize) -> (Vec<Mixer>, Needs, Option<GluAct>) {
        let b = p.body(l);
        let one = LayerProgram {
            bodies: vec![b.clone()],
            layers: vec![0],
            ..p.clone()
        };
        let mut n = Needs::of(&one);
        // own-cache attention vs a cache read across layers is the mixer's own
        n.attention = false;
        (b.mixers().cloned().collect(), n, b.glu)
    }

    /// The layer program a shipping model's shape states reads, layer by layer, as its trace's
    /// lowering does.
    #[test]
    fn a_shape_states_its_traces_layer_program() {
        for name in [
            "qwen35-d9b-bf16-kv-bf16",
            "qwen35-d0.8b-bf16-kv-bf16",
            "qwen3-8b-bf16-kv-bf16",
            "llama31-8b-bf16-kv-bf16",
            "gemma4-e2b-bf16-kv-bf16",
            "gemma4-e4b-bf16-kv-bf16",
            "gemma4-12b-bf16-kv-bf16",
            "gptoss-20b-bf16-mxfp4-kv-bf16",
            "qwen3-30b-a3b-mini-bf16-kv-bf16",
        ] {
            let t = poem_compiler::catalog::repository()
                .parse(name)
                .unwrap_or_else(|| panic!("{name} parses"))
                .trace(poem_ir::Platform::Cerebras);
            let a = poem_compiler::analysis::analyze(&t).unwrap();
            let lowered = poem_compiler::spatial::layer::lower(&t, &a, &|g| {
                poem_compiler::spatial::layer::single_token(&t, g)
            })
            .unwrap();
            let stated = of(&shape(&t).unwrap());
            assert_eq!(stated.layers.len(), lowered.layers.len(), "{name}");
            assert_eq!(stated.bodies.len(), lowered.bodies.len(), "{name}: bodies");
            assert_eq!(stated.head, lowered.head, "{name}: head");
            for l in 0..lowered.layers.len() {
                assert_eq!(read(&stated, l), read(&lowered, l), "{name}: layer {l}");
            }
        }
    }
}
