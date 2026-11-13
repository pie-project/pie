//! The composer: a program's blocks from what its layers need, not from the model's name. Each phase a
//! layer program has ([`poem_compiler::spatial::layer`]) asks for the blocks that run it, the
//! partition says how big each is:
//!
//! - the units hold the matvecs and the roots: a snake of units; stacked unit columns with a
//!   side input's storage above them ([`stacked`]) where the layers read a per-token input or a
//!   cache across layers; two runs of rows (attention rows, plain rows) in a U of two unit
//!   columns where the layers attend over windows; attention | MoE halves in a snake of two
//!   unit columns from the band where they route experts;
//! - the mixers' segments: the scores for attention, the delta rule's (k, q) multicast and its
//!   o-norm for GDN, the router's for experts; the side input's storage and bus; the shared
//!   caches' line;
//! - the head (east of the units, by row groups of the runs, or a band below a MoE snake) and the
//!   token loop close the program.
//!
//! What the role kernels read off each block's PEs (their ABI with the host) follows their set
//! ([`copies`], [`crate::kernels::KernelSet`]); what the kernels do not run is refused by name
//! ([`refuse`]).

pub mod copies;
pub mod stacked;

use poem_compiler::spatial::layer::{LayerProgram, Mixer, Phase};
use poem_ir::GluAct;

use crate::block::Block;
use crate::block::band::{Band, BandLoop};
use crate::block::gdn::Gdn;
use crate::block::head::{HeadCopies, HeadEast};
use crate::block::head_groups::HeadGroups;
use crate::block::moe::{MoeCopies, Router};
use crate::block::ple::Ple;
use crate::block::row_runs::RunCopies;
use crate::block::segments::{Onorm, Scores};
use crate::block::shared_kv::SharedKv;
use crate::block::token_loop::TokenLoop;
use crate::block::units::{UnitCopies, Units, UnitsCopies};
use crate::floorplan::{Aux, Floorplan, HeadAt, Segment};
use crate::kernels::KernelSet;
use crate::plan::{Act, Plan, Shape};

type R<T> = Result<T, String>;

/// What a model's layers ask the program to place, read off its layer program ([`Needs::of`]).
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
    /// Attention sinks: a per-head logit in every softmax's sum.
    pub sinks: bool,
    /// RMSNorms over each q / k head.
    pub qk_norm: bool,
    /// Biases on the q / k / v projections.
    pub attn_bias: bool,
    /// A bias on the output projection.
    pub o_bias: bool,
    /// A bias on the router's logits.
    pub router_bias: bool,
    /// Biases on the experts' legs.
    pub expert_bias: bool,
}

/// What a bias (`elemwise.add_bias`) of the param named `name` is on, by the param's last name:
/// the q / k / v projections, the output projection, the router; `None`: none the kernels know.
#[must_use]
pub fn bias_on(name: &str) -> Option<BiasOn> {
    match name.rsplit('.').next().unwrap_or(name) {
        "q_bias" | "k_bias" | "v_bias" | "qkv_bias" => Some(BiasOn::Qkv),
        "o_bias" => Some(BiasOn::O),
        "router_bias" => Some(BiasOn::Router),
        _ => None,
    }
}

/// The projections a bias is on ([`bias_on`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasOn {
    Qkv,
    O,
    Router,
}

/// Whether the param named `name` is a q / k head norm's weight, by its last name.
#[must_use]
pub fn head_norm(name: &str) -> bool {
    matches!(name.rsplit('.').next().unwrap_or(name), "q_norm" | "k_norm")
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
                    ops,
                } => {
                    n.sinks |= ops.iter().any(|o| o.op == "attn.sink");
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
                    experts: Some(_),
                    weights,
                    ..
                } => {
                    n.experts = true;
                    // a routed leg's bias rides in its select
                    n.expert_bias |= weights
                        .iter()
                        .any(|w| w.params.iter().any(|p| p.ends_with("bias")));
                }
                Phase::Root { ops, .. } => {
                    for o in ops {
                        match o.op {
                            "elemwise.add_bias" => {
                                for b in o.params.iter().filter_map(|p| bias_on(p)) {
                                    match b {
                                        BiasOn::Qkv => n.attn_bias = true,
                                        BiasOn::O => n.o_bias = true,
                                        BiasOn::Router => n.router_bias = true,
                                    }
                                }
                            }
                            "elemwise.rmsnorm" => {
                                n.qk_norm |= o.params.iter().any(|p| head_norm(p));
                            }
                            "linear.moe_bias_sum" => n.expert_bias = true,
                            _ => {}
                        }
                    }
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
}

/// Refuses, by name, what `shape` needs that the role kernels of its set do not run (else the
/// program would run the model without it). The own, shared and windowed caches' kernels run no
/// projection biases or sinks; the experts' attention / MoE halves run full and sliding-window
/// attention with or without q / k / v and output biases, sinks or q / k head norms (not head
/// norms and biases at once: the biases' words hold the norms' weights), a router with or without
/// its bias, MXFP4 or 16-bit experts with or without biases, and the clamped SwiGLU or SiLU's GLU.
pub fn refuse(shape: &Shape) -> R<()> {
    let set = KernelSet::of_shape(shape);
    let a = &shape.attn;
    let (unmet, kernels): (Vec<(bool, &str)>, &str) = if set == KernelSet::Experts {
        (
            vec![
                (
                    a.bias && a.qk_norm,
                    "q / k / v biases together with q / k head norms",
                ),
                (a.output_gate, "an attention output gate"),
                (
                    !matches!(shape.mlp.act, Act::SwigluOai | Act::Silu),
                    "an expert activation other than the clamped SwiGLU or SiLU's",
                ),
                (shape.ple.is_some(), "per-layer inputs"),
                (shape.norm.sandwich, "sandwich norms"),
                (shape.head.final_softcap.is_some(), "a final logit softcap"),
            ],
            "MoE",
        )
    } else {
        (
            vec![
                (a.bias, "q / k / v projection biases"),
                (a.o_bias, "an output projection bias"),
                (a.sinks, "attention sinks"),
            ],
            set.name(),
        )
    };
    match unmet.iter().find(|(on, _)| *on) {
        Some((_, what)) => Err(format!("the {kernels} kernels run no {what}")),
        None => Ok(()),
    }
}

/// The role kernels' layer phases (weight, CX and CY's comptime `NRM`, `MIXK`, `ACT`, `PLE`, `KVS`),
/// from the layer program: what its layers need and its MLP's activation; one image a role over
/// every layer program.
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
    /// 1: x leaves the last unit for the head by row groups ([`HeadAt::Groups`]).
    pub hg: u32,
    /// (attention halves) 1: the q / k / v biases; 1: the sinks; 1: the q / k head norms.
    pub qb: u32,
    pub snk: u32,
    pub qkn: u32,
    /// (MoE halves) 1: the output projection's bias; 1: the router's bias; 1: the experts' biases.
    pub ob: u32,
    pub rb: u32,
    pub xb: u32,
    /// (MoE halves) the experts' GLU: 0 the clamped SwiGLU, 1 SiLU's.
    pub glu: u32,
    /// (windowed rows) 1: a global layer projects q | raw k and takes v as k's unscaled norm (the
    /// windowed rows' cut leaves a global layer's v rows out: Gemma 4's k = v checkpoints, which
    /// the trace's packed bank does not tell apart).
    pub kev: u32,
}

impl Schedule {
    /// The phases of `program`'s layers laid as `fp`.
    #[must_use]
    pub fn of(program: &LayerProgram, fp: &Floorplan) -> Schedule {
        let needs = &Needs::of(program);
        let set = KernelSet::of(needs);
        let gelu = program
            .bodies
            .iter()
            .any(|b| b.glu == Some(GluAct::GegluTanh));
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
            act: u32::from(gelu),
            ple: u32::from(needs.side_in),
            kvs: u32::from(needs.shared_cache),
            hg: u32::from(fp.head.at == HeadAt::Groups),
            qb: u32::from(needs.attn_bias),
            snk: u32::from(needs.sinks),
            qkn: u32::from(needs.qk_norm),
            ob: u32::from(needs.o_bias),
            rb: u32::from(needs.router_bias),
            xb: u32::from(needs.expert_bias),
            glu: u32::from(program.bodies.iter().any(|b| b.glu == Some(GluAct::Swiglu))),
            kev: u32::from(set == KernelSet::Windows),
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
            "snake: {units} units on {unx} x {uny} slots (at most one gap slot, on an even number of unit columns, below the turn row)"
        )),
    }
}

/// What the role kernels read off the PEs of the blocks the composer places: the copies onto
/// each kind (the kernels' ABI with the host, by kernel set: [`copies::of`]). A kind the kernels have
/// no copies for is a kind they do not run.
#[derive(Default)]
pub struct KernelCopies {
    pub units: Option<UnitCopies>,
    pub runs: Option<RunCopies>,
    pub moe: Option<MoeCopies>,
    pub head: Option<HeadCopies>,
}

fn take<T>(c: Option<T>, what: &str) -> R<T> {
    c.ok_or_else(|| format!("compose: the kernels place no {what}"))
}

/// The blocks of `plan`'s floorplan, in order: the units first, then the segments laid over
/// them, the blocks placed beside them for a feature (a side input's storage, the shared caches'
/// line), the head, the token loop; each block's copies the role kernels' ([`copies::of`]).
pub fn blocks(plan: &Plan, set: KernelSet, fp: &Floorplan) -> R<Vec<Box<dyn Block>>> {
    let copies = copies::of(plan, set)?;
    let shape = &fp.units.shape;
    let mut out: Vec<Box<dyn Block>> = Vec::new();
    let unit_copies = match (copies.units, copies.runs, copies.moe) {
        (_, _, Some(c)) => UnitsCopies::Halves(c),
        (_, Some(c), None) => UnitsCopies::Runs(c),
        (c, None, None) => UnitsCopies::Plain(take(c, "units")?),
    };
    out.push(Box::new(Units {
        grid: fp.units.grid(),
        halves: shape.halves(),
        copies: unit_copies,
    }));
    for s in &fp.segments {
        match *s {
            Segment::Scores { width } => out.push(Box::new(Scores {
                width,
                rows: shape.attn_rows().collect(),
                cols: shape.attn_cols(),
            })),
            Segment::Router { width } => out.push(Box::new(Router { width })),
            Segment::Gdn => out.push(Box::new(
                Gdn::of(plan).ok_or("compose: GDN segments with no GDN layer")?,
            )),
            Segment::Onorm { width } => out.push(Box::new(Onorm { width })),
        }
    }
    for a in &fp.aux {
        match a {
            Aux::Store(columns) => out.push(Box::new(Ple {
                columns: columns.clone(),
            })),
            Aux::KvLine => out.push(Box::new(SharedKv)),
        }
    }
    let h = fp.head;
    match h.at {
        HeadAt::East => {
            out.push(Box::new(HeadEast {
                cols: h.cols,
                rows: h.rows,
                token_loop: fp.token_loop,
                copies: take(copies.head, "head")?,
            }));
            if fp.token_loop {
                out.push(Box::new(TokenLoop));
            }
        }
        HeadAt::Groups => out.push(Box::new(HeadGroups {
            cols: h.cols,
            rows: h.rows,
            copies: take(copies.head, "head")?,
        })),
        HeadAt::Band => {
            out.push(Box::new(Band {
                cols: h.cols,
                rows: h.rows,
                vocab_rows: h.vocab_rows,
            }));
            if fp.token_loop {
                out.push(Box::new(BandLoop));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Program;
    use crate::target::{Ask, plan, plan_shape};

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

    /// The schedule of `name`'s plan (the pinned plan `preset`, else the search's).
    fn schedule(name: &str, preset: Option<&str>) -> Schedule {
        let ask = preset.map_or_else(Ask::default, Ask::preset);
        let p = plan(&trace(name), &ask).unwrap().plan;
        let sc = Program::of(p.clone()).unwrap().schedule;
        let fp = Floorplan::of(
            &p,
            &crate::floorplan::Requests::of(&Needs::of(&program(&trace(name)))),
        );
        assert_eq!(
            Schedule::of(&program(&trace(name)), &fp.unwrap()),
            sc,
            "{name}"
        );
        sc
    }

    /// The role kernels' phases read off each shipping model's layer program and floorplan: what
    /// its kernels were built with.
    #[test]
    fn a_layer_program_schedules_its_kernels() {
        for (name, preset, nrm, mixk, act, ple, kvs, hg) in [
            ("qwen35-d9b-bf16-kv-bf16", "9b", 0, 0, 0, 0, 0, 0),
            ("llama31-8b-bf16-kv-bf16", "l3-8b", 0, 0, 0, 0, 0, 0),
            ("gemma4-e4b-bf16-kv-bf16", "e4b", 1, 1, 1, 1, 1, 0),
            ("gemma4-12b-bf16-kv-bf16", "g12", 1, 1, 1, 0, 0, 1),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", "full", 2, 2, 0, 0, 0, 0),
        ] {
            let sc = schedule(name, Some(preset));
            assert_eq!(
                (sc.nrm, sc.mixk, sc.act, sc.ple, sc.kvs, sc.hg),
                (nrm, mixk, act, ple, kvs, hg),
                "{name}"
            );
        }
        // the MoE halves' variants: gpt-oss's biases and sinks, the mini's head norms and SiLU
        for (name, preset, want) in [
            (
                "gptoss-20b-bf16-mxfp4-kv-bf16",
                Some("full"),
                (1, 1, 0, 1, 1, 1, 0),
            ),
            (
                "qwen3-30b-a3b-mini-bf16-kv-bf16",
                None,
                (0, 0, 1, 0, 0, 0, 1),
            ),
        ] {
            let sc = schedule(name, preset);
            assert_eq!(
                (sc.qb, sc.snk, sc.qkn, sc.ob, sc.rb, sc.xb, sc.glu),
                want,
                "{name}"
            );
        }
    }

    /// Every shipping model's preset and every tiny one composes and builds (the capture pins
    /// what each makes).
    #[test]
    fn every_preset_composes() {
        let check = |p: crate::plan::Plan, what: &str| {
            Program::of(p).unwrap_or_else(|e| panic!("{what}: {e}"));
        };
        for (name, preset) in [
            ("qwen35-d9b-bf16-kv-bf16", "9b"),
            ("qwen35-d4b-bf16-kv-bf16", "4b"),
            ("qwen35-d0.8b-bf16-kv-bf16", "0.8b"),
            ("qwen3-8b-bf16-kv-bf16", "q3-8b"),
            ("llama31-8b-bf16-kv-bf16", "l3-8b"),
            ("gemma4-e2b-bf16-kv-bf16", "e2b"),
            ("gemma4-e4b-bf16-kv-bf16", "e4b"),
            ("gemma4-e4b-bf16-kv-bf16", "e4b-meta"),
            ("gemma4-12b-bf16-kv-bf16", "g12"),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", "full"),
        ] {
            check(plan(&trace(name), &Ask::preset(preset)).unwrap().plan, name);
        }
        for (file, preset) in [
            ("tiny_q35_t08", "tiny-t08"),
            ("tiny_q35_t4", "tiny-t4"),
            ("tiny_q3", "tiny-q3"),
            ("tiny_q3_gap", "tiny-q3-gap"),
            ("tiny_llama", "tiny-q3"),
            ("tiny_gemma", "tiny-gemma"),
            ("tiny_g12", "tiny-g12"),
            ("tiny_gptoss", "tiny-gptoss"),
        ] {
            let path = format!(
                "{}/tools/wafer/baseline/shapes/{file}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let s: crate::plan::Shape =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            check(plan_shape(s, &Ask::preset(preset)).unwrap().plan, file);
        }
    }

    /// The token loop's three-level max-reduce (`G3W`) composes on every layout with a token
    /// loop, the shipping heads' control lines included, and lays as its tree: each member's
    /// send colour leaves its ramp and reaches its parent's ramp, where the parent takes it on a
    /// queue (a member's level-1 / level-2 input, the root's half).
    #[test]
    fn a_three_level_max_reduce_lays_its_tree() {
        use crate::fabric::{Endpoint, RouteWord};
        use crate::wiring::Link;
        let tiny = |file: &str| -> crate::plan::Shape {
            let path = format!(
                "{}/tools/wafer/baseline/shapes/{file}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
        };
        let mut plans = Vec::new();
        for (name, preset, set) in [
            ("qwen35-d0.8b-bf16-kv-bf16", "0.8b", "G2W=8,G3W=8"),
            ("qwen35-d9b-bf16-kv-bf16", "9b", "G2W=8,G3W=8"),
            ("gemma4-12b-bf16-kv-bf16", "g12", "G2W=8,G3W=8"),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", "full", "G2W=7,G3W=7"),
        ] {
            let ask = Ask {
                set: set.into(),
                ..Ask::preset(preset)
            };
            plans.push((name, plan(&trace(name), &ask).unwrap().plan));
        }
        for (file, preset, set) in [
            ("tiny_q35_t08", "tiny-t08", "G2W=2,G3W=2"),
            ("tiny_q35_t08", "tiny-t08", "HR=48,G2W=3,G3W=2"),
            ("tiny_gemma", "tiny-gemma", "G2W=1,G3W=1"),
            ("tiny_g12", "tiny-g12", "G2W=2,G3W=2"),
            ("tiny_gptoss", "tiny-gptoss", "G2W=2,G3W=3"),
        ] {
            let ask = Ask {
                set: set.into(),
                ..Ask::preset(preset)
            };
            plans.push((file, plan_shape(tiny(file), &ask).unwrap().plan));
        }
        for (what, p) in plans {
            let prog = Program::of(p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let trees: Vec<_> = prog
                .wiring
                .links
                .iter()
                .filter_map(|l| match l {
                    Link::Tree(t) => Some(t),
                    _ => None,
                })
                .collect();
            assert_eq!(trees.len(), 1, "{what}: one max-reduce tree");
            let t = trees[0];
            let f = &prog.fabric;
            let takes = |at, c| match f.io(at).slots.get(&t.slot) {
                Some(Endpoint::Member(m)) => m.recv1 == Some(c) || m.recv2 == Some(c),
                Some(Endpoint::Root(r)) => r.lo == Some(c) || r.hi == Some(c),
                _ => false,
            };
            for (i, &at) in t.pts.iter().enumerate() {
                let Some(p) = t.parent[i] else { continue };
                let Some(Endpoint::Member(m)) = f.io(at).slots.get(&t.slot) else {
                    panic!("{what}: point {i} is no member");
                };
                let c = m.send;
                assert!(f.route(at, c).unwrap().from_ramp(), "{what}: {i} sends");
                let mut j = i;
                loop {
                    let w = f.route(t.pts[j], c).unwrap();
                    if w.to_ramp() {
                        break;
                    }
                    let next = [j.wrapping_sub(1), j + 1]
                        .into_iter()
                        .filter(|&k| k < t.pts.len())
                        .find(|&k| {
                            t.pts[j]
                                .dir_to(t.pts[k])
                                .is_some_and(|d| w.0 & RouteWord::tx(d) != 0)
                        })
                        .unwrap_or_else(|| panic!("{what}: {i}'s sum stops at {j}"));
                    j = next;
                }
                assert_eq!(j, p, "{what}: {i}'s sum lands at {j}, not its parent {p}");
                assert!(
                    takes(t.pts[p], c),
                    "{what}: {p} takes no queue of {i}'s sum"
                );
            }
        }
    }

    /// A reduction group need not divide its chain: the last group of each half is short. Tiny
    /// q35 with G2 / G2Y 3 over 8 and tiny gpt-oss with G2 3 run the golden in the simulator;
    /// the chip runs that went wrong at G2Y 10 over 128 and G2 24 over 160 were greedy runs
    /// leaving the golden at a near-tie, not wrong sums.
    #[test]
    fn a_short_last_reduction_group_composes() {
        for (file, preset, set) in [
            ("tiny_q35_t08", "tiny-t08", "G2=3"),
            ("tiny_q35_t08", "tiny-t08", "G2Y=3"),
            ("tiny_gptoss", "tiny-gptoss", "G2=3"),
        ] {
            let path = format!(
                "{}/tools/wafer/baseline/shapes/{file}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let s: crate::plan::Shape =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let ask = Ask {
                set: set.into(),
                ..Ask::preset(preset)
            };
            Program::of(plan_shape(s, &ask).unwrap().plan)
                .unwrap_or_else(|e| panic!("{file} {set}: {e}"));
        }
    }
}
