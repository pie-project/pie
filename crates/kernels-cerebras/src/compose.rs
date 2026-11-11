//! The composer: a program's blocks from what its layers need, not from a family. Each phase a
//! layer program has ([`poem_compiler::spatial::layer`]) asks for the blocks that run it: the
//! units for the matvecs and the roots, a scores segment for attention over a layer's own cache,
//! the delta rule's (k, q) multicast and o-norm segments for GDN; the head and the token loop
//! close every program. The partition says how big each is.
//!
//! The composer places the unit snake's phases today: attention over a layer's own cache, GDN,
//! the MLP. Side inputs, caches read across layers, sliding windows and routed experts are
//! placed by their families' blocks still ([`Program::build`](crate::program::Program::build)
//! falls back to them) until their blocks join it.

use poem_compiler::spatial::layer::{LayerProgram, Mixer, Phase};

use crate::block::Block;
use crate::block::gdn::Gdn;
use crate::block::head::{HeadCopies, HeadEast, Hemb};
use crate::block::segments::{Onorm, Scores};
use crate::block::token_loop::TokenLoop;
use crate::block::units::{UnitCopies, Units};
use crate::floor::{UnitGrid, UnitLayout};
use crate::plan::{Act, LayerKind, Plan, Shape};
use crate::weights::{From, Slice, Transform, from};

type R<T> = Result<T, String>;

/// What a model's layers ask the program to place, read off its layer program ([`Needs::of`])
/// or, for a plan stated as a shape, off the shape ([`Needs::of_shape`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Attention over a layer's own cache.
    pub attention: bool,
    /// The gated delta rule.
    pub gdn: bool,
    /// Attention over the last positions only.
    pub sliding: bool,
    /// Attention over a cache another layer appends.
    pub shared_cache: bool,
    /// A per-token side input.
    pub side_in: bool,
    /// A routed bank of experts.
    pub experts: bool,
    /// A norm on a mixer's or the MLP's output before its residual add (sandwich norms): the
    /// next input norm waits on its scalar.
    pub deferred_norm: bool,
}

impl Needs {
    /// The needs of a layer program's phases.
    #[must_use]
    pub fn of(program: &LayerProgram) -> Needs {
        let mut n = Needs::default();
        for p in program.bodies.iter().flat_map(|b| &b.phases) {
            match p {
                Phase::Mixer {
                    mixer:
                        Mixer::Attn {
                            window,
                            shared_from,
                            ..
                        },
                    ..
                } => {
                    n.attention |= shared_from.is_none();
                    n.shared_cache |= shared_from.is_some();
                    n.sliding |= window.is_some();
                }
                Phase::Mixer {
                    mixer: Mixer::Gdn { .. },
                    ..
                } => n.gdn = true,
                Phase::SideIn { .. } => n.side_in = true,
                Phase::Matvec {
                    experts: Some(_), ..
                } => n.experts = true,
                Phase::Root { ops, .. } => {
                    let at = |op: &str| ops.iter().position(|o| o.op == op);
                    if let (Some(norm), Some(add)) =
                        (at("elemwise.rmsnorm"), at("elemwise.residual_add"))
                    {
                        n.deferred_norm |= norm < add;
                    }
                }
                _ => {}
            }
        }
        n
    }

    /// The needs of a shape's layers (a plan stated as a shape has no layer program).
    #[must_use]
    pub fn of_shape(s: &Shape) -> Needs {
        let any = |f: fn(&LayerKind) -> bool| s.layers.iter().any(f);
        Needs {
            attention: any(|k| matches!(k, LayerKind::Global | LayerKind::Sliding { .. })),
            gdn: s.gdn.is_some() && any(|k| matches!(k, LayerKind::Gdn)),
            sliding: any(|k| {
                matches!(
                    k,
                    LayerKind::Sliding { .. } | LayerKind::SharedKv { sliding: true, .. }
                )
            }),
            shared_cache: any(|k| matches!(k, LayerKind::SharedKv { .. })),
            side_in: s.ple.is_some(),
            experts: s.moe.is_some(),
            deferred_norm: s.norm.sandwich,
        }
    }

    /// The first need the composer has no block for, if any.
    fn unplaced(&self) -> Option<&'static str> {
        [
            (self.sliding, "a sliding window"),
            (self.shared_cache, "a cache read across layers"),
            (self.side_in, "a per-token side input"),
            (self.experts, "routed experts"),
        ]
        .into_iter()
        .find_map(|(n, what)| n.then_some(what))
    }
}

/// The role kernels' layer phases (weight, CX and CY's comptime `NRM`, `MIXK`, `ACT`, `PLE`, `KVS`),
/// from what the layers need and the MLP's activation: one image a role over every layer program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    /// The input phases' norm: 0 direct, 1 deferred (sandwich norms), 2 scaled (attention / MoE
    /// halves).
    pub nrm: u32,
    /// The mixer kind: 0 own-cache attention / GDN, 1 sliding / global and shared caches, 2 an
    /// attention half's (its MoE half's is [`Schedule::moe`]).
    pub mixk: u32,
    /// The MLP's activation: 0 SiLU, 1 GELU-tanh (a MoE half's is its own).
    pub act: u32,
    /// 1: a per-layer embedding phase after the MLP.
    pub ple: u32,
    /// 1: the shared caches' KVS line.
    pub kvs: u32,
}

impl Schedule {
    /// The phases of layers that need `needs`, their MLP's activation `act`.
    #[must_use]
    pub fn of(needs: &Needs, act: Act) -> Schedule {
        Schedule {
            nrm: if needs.experts {
                2
            } else {
                u32::from(needs.deferred_norm)
            },
            mixk: if needs.experts {
                2
            } else {
                u32::from(needs.sliding || needs.shared_cache)
            },
            act: u32::from(act == Act::GeluTanh),
            ple: u32::from(needs.side_in),
            kvs: u32::from(needs.shared_cache),
        }
    }

    /// A MoE half's mixer kind.
    #[must_use]
    pub fn moe(&self) -> u32 {
        self.mixk + 1
    }
}

/// The unit slots' gap: none when the units fill the grid; one slot short, the last column's
/// row-1 slot stays out of the snake (the last unit still ends on row 0, the token loop's
/// return), on an even number of unit columns below the turn row.
pub fn snake_gap(units: u32, (unx, uny): (u32, u32)) -> R<Option<(u32, u32)>> {
    match (unx * uny).checked_sub(units) {
        Some(0) => Ok(None),
        Some(1) if unx % 2 == 0 && uny >= 3 => Ok(Some((unx - 1, 1))),
        _ => Err(format!(
            "dense: {units} units on {unx} x {uny} slots (at most one gap slot, on an even number of unit columns, below the turn row)"
        )),
    }
}

/// The blocks of `plan` for layers that need `needs`, in order (the units first, then what is
/// placed against them), or `None` where a need has no block of the composer's yet.
pub fn blocks(plan: &Plan, needs: &Needs) -> R<Option<Vec<Box<dyn Block>>>> {
    if needs.unplaced().is_some() {
        return Ok(None);
    }
    let (s, p) = (&plan.shape, &plan.partition);
    let ug = p
        .unit_grid
        .ok_or("dense: the plan has no unit grid (UNX, UNY)")?;
    let (cols, rows) = (p.cols, p.rows);
    // a segment within a half of the unit's columns (none spans the control column)
    for (n, sw) in [
        ("SWA", p.segments.scores),
        ("SWG", p.segments.gdn),
        ("SWV", p.segments.onorm),
    ] {
        if sw == 0 || (cols / 2) % sw != 0 {
            return Err(format!(
                "dense: {n} {sw} does not divide XC {} (a segment would span the control column)",
                cols / 2
            ));
        }
    }
    if p.head.cols * 2 != rows {
        return Err("dense: head columns = CX rows / 2".into());
    }
    // a hidden that the row split does not divide is zero-padded to it
    let placed = p.hidden_placed();
    let pad = |t: &str| -> From {
        let tr = if placed != s.hidden {
            vec![Transform::PadZero { dim: 1, to: placed }]
        } else {
            vec![]
        };
        vec![(t.to_string(), tr)]
    };
    let mut out: Vec<Box<dyn Block>> = vec![Box::new(Units {
        grid: UnitGrid {
            cols,
            rows,
            cx: cols / 2,
            cy: rows / 2,
            layout: UnitLayout::Snake {
                unx: ug.0,
                uny: ug.1,
                up_first: false,
                gaps: snake_gap(p.units, ug)?.into_iter().collect(),
            },
            layers_per_unit: p.layers_per_unit,
        },
        copies: UnitCopies {
            cols,
            rows,
            w_from: pad("layers.*.{q,k,v,o,gate,up,down}_proj, linear_attn.*"),
            gam_b16: true,
            gam_from: pad("layers.*.{input,post_attention}_layernorm, norm"),
            cy: &[
                ("cw", true),
                ("gpar", true),
                ("gnw", true),
                ("qkw", true),
                ("rot", true),
            ],
            cy_optional: true,
        },
    })];
    if needs.attention {
        out.push(Box::new(Scores {
            width: p.segments.scores,
        }));
    }
    if needs.gdn
        && let Some(g) = Gdn::of(plan)
    {
        out.push(Box::new(g));
        out.push(Box::new(Onorm {
            width: p.segments.onorm,
        }));
    }
    let token_loop = plan.contract.token_loop;
    out.push(Box::new(HeadEast {
        cols: p.head.cols,
        rows: p.head.rows,
        token_loop,
        copies: HeadCopies {
            cols: p.head.cols,
            rows: p.head.rows,
            hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
            hw_from: vec![(
                "lm_head".into(),
                vec![Transform::PadLastRow {
                    to: p.head.rows * p.head.vocab_rows,
                }],
            )],
            hg_from: from("norm"),
            hemb: if token_loop {
                Hemb::After(from(&format!("embed_tokens ({} rows)", s.vocab)))
            } else {
                Hemb::None
            },
        },
    }));
    if token_loop {
        out.push(Box::new(TokenLoop));
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit;
    use crate::family;
    use crate::program::Program;
    use crate::target::{Ask, plan, plan_shape, shape};

    fn trace(name: &str) -> poem_ir::Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    fn program(t: &poem_ir::Trace) -> LayerProgram {
        let a = poem_compiler::analysis::analyze(t).unwrap();
        poem_compiler::spatial::layer::lower(t, &a, &|g| {
            poem_compiler::spatial::layer::single_token(t, g)
        })
        .unwrap()
    }

    /// A shape's needs are its layer program's, for every shipping model and the Qwen3 MoE
    /// miniature.
    #[test]
    fn a_shape_needs_what_its_layer_program_needs() {
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
            let t = trace(name);
            assert_eq!(
                Needs::of(&program(&t)),
                Needs::of_shape(&shape(&t).unwrap()),
                "{name}"
            );
        }
    }

    /// The role kernels' phases read off each shipping model's layer program: what its family's
    /// kernels were built with.
    #[test]
    fn a_layer_program_schedules_its_family_kernels() {
        for (name, nrm, mixk, act, ple, kvs) in [
            ("qwen35-d9b-bf16-kv-bf16", 0, 0, 0, 0, 0),
            ("llama31-8b-bf16-kv-bf16", 0, 0, 0, 0, 0),
            ("gemma4-e4b-bf16-kv-bf16", 1, 1, 1, 1, 1),
            ("gemma4-12b-bf16-kv-bf16", 1, 1, 1, 0, 0),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", 2, 2, 0, 0, 0),
        ] {
            let t = trace(name);
            let sc = Schedule::of(&Needs::of(&program(&t)), shape(&t).unwrap().mlp.act);
            assert_eq!(
                sc,
                Schedule {
                    nrm,
                    mixk,
                    act,
                    ple,
                    kvs
                },
                "{name}"
            );
        }
    }

    /// What a program's blocks make: the floor, every PE's rcfg words, the layout and the copies.
    fn made(prog: &Program) -> (String, String, String, String) {
        (
            format!("{:?}", prog.floor),
            emit::rcfg::dump(prog).unwrap(),
            emit::layout::layout_text(prog, &prog.kernels).unwrap(),
            serde_json::to_string(&prog.bindings().unwrap()).unwrap(),
        )
    }

    /// The composer's blocks make what the unit snake family's made, for every one of its presets.
    #[test]
    fn the_composer_reproduces_the_unit_snake() {
        let dense = family::of(crate::plan::FamilyId::Dense);
        let mut composed = 0;
        for (name, preset) in [
            ("qwen35-d9b-bf16-kv-bf16", "9b"),
            ("qwen35-d4b-bf16-kv-bf16", "4b"),
            ("qwen35-d0.8b-bf16-kv-bf16", "0.8b"),
            ("qwen3-8b-bf16-kv-bf16", "q3-8b"),
            ("llama31-8b-bf16-kv-bf16", "l3-8b"),
        ] {
            let t = trace(name);
            let p = plan(&t, &Ask::preset(preset)).unwrap().plan;
            let fam =
                Program::build_with(dense.blocks(&p).unwrap(), family::of(p.family), p.clone())
                    .unwrap();
            let blocks = blocks(&p, &Needs::of(&program(&t)))
                .unwrap()
                .expect("composed");
            let comp = Program::build_with(blocks, family::of(p.family), p).unwrap();
            assert_eq!(made(&fam), made(&comp), "{name}");
            composed += 1;
        }
        // the tiny ones, from their shapes
        for (file, preset) in [
            ("tiny_q35_t08", "tiny-t08"),
            ("tiny_q35_t4", "tiny-t4"),
            ("tiny_q3", "tiny-q3"),
            ("tiny_q3_gap", "tiny-q3-gap"),
            ("tiny_llama", "tiny-q3"),
        ] {
            let path = format!(
                "{}/tools/wafer/baseline/shapes/{file}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let s: Shape = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let p = plan_shape(s, &Ask::preset(preset)).unwrap().plan;
            let fam =
                Program::build_with(dense.blocks(&p).unwrap(), family::of(p.family), p.clone())
                    .unwrap();
            let blocks = blocks(&p, &Needs::of_shape(&p.shape)).unwrap().unwrap();
            let comp = Program::build_with(blocks, family::of(p.family), p).unwrap();
            assert_eq!(made(&fam), made(&comp), "{file}");
        }
        assert_eq!(composed, 5);
    }

    /// Needs the composer has no block for leave the program to its family.
    #[test]
    fn what_the_composer_cannot_place_is_its_familys() {
        let t = trace("gemma4-e2b-bf16-kv-bf16");
        let p = plan(&t, &Ask::default()).unwrap().plan;
        assert!(blocks(&p, &Needs::of(&program(&t))).unwrap().is_none());
    }
}
