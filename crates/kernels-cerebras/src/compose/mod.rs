//! The composer: a program's blocks from what its layers need, not from a family. Each phase a
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
//! What the role kernels read off each block's PEs (their ABI with the host) follows the layout
//! ([`copies`]); what the kernels do not run is refused by name ([`refuse`]).

pub mod copies;
pub mod stacked;

use poem_compiler::spatial::layer::{LayerProgram, Mixer, Phase};
use poem_ir::GluAct;

use crate::block::Block;
use crate::block::band::{Band, BandLoop};
use crate::block::gdn::Gdn;
use crate::block::head::{HeadCopies, HeadEast};
use crate::block::head_groups::HeadGroups;
use crate::block::moe::{AttnScores, MoeCopies, MoeUnits, Router};
use crate::block::ple::{Ple, PleColumn};
use crate::block::row_runs::{RowRuns, RunCopies};
use crate::block::segments::{Onorm, Scores};
use crate::block::shared_kv::SharedKv;
use crate::block::token_loop::TokenLoop;
use crate::block::units::{UnitCopies, Units};
use crate::floor::{UnitGrid, UnitLayout};
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

/// How a program's units lie, from what its layers need: the one choice the blocks and the role
/// kernels' phases both follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// A snake of units.
    Snake,
    /// Stacked unit columns, a side input's storage above them (a side input or a cache read
    /// across layers).
    Stacked,
    /// Attention / plain row runs in a U of two unit columns, the head fed by row groups
    /// (sliding windows).
    Runs,
    /// Attention | MoE halves in a snake of two unit columns from the band (routed experts).
    Split,
}

impl Layout {
    /// The layout's name (the plan's record, error messages).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Layout::Snake => "snake",
            Layout::Stacked => "stacked",
            Layout::Runs => "runs",
            Layout::Split => "split",
        }
    }

    /// The layout layers that need `needs` take.
    #[must_use]
    pub fn of(needs: &Needs) -> Layout {
        if needs.experts {
            Layout::Split
        } else if needs.side_in || needs.shared_cache {
            Layout::Stacked
        } else if needs.sliding {
            Layout::Runs
        } else {
            Layout::Snake
        }
    }
}

/// Refuses, by name, what `shape` needs that the role kernels of its layout do not run (else the
/// program would run the model without it). The snake, the stacked columns and the row runs run
/// no projection biases or sinks; the attention / MoE halves run full and sliding-window attention
/// with or without q / k / v and output biases, sinks or q / k head norms (not head norms and
/// biases at once: the biases' words hold the norms' weights), a router with or without its bias,
/// MXFP4 or 16-bit experts with or without biases, and the clamped SwiGLU or SiLU's GLU.
pub fn refuse(shape: &Shape) -> R<()> {
    let layout = Layout::of(&Needs::of(&crate::layers::of(shape)));
    let a = &shape.attn;
    let (unmet, kernels): (Vec<(bool, &str)>, &str) = if layout == Layout::Split {
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
            layout.name(),
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
    /// 1: x leaves the last unit for the head by row groups ([`Layout::Runs`]).
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
    /// (row runs) 1: a global layer projects q | raw k and takes v as k's unscaled norm (the
    /// row runs' cut leaves a global layer's v rows out: Gemma 4's k = v checkpoints, which the
    /// trace's packed bank does not tell apart).
    pub kev: u32,
}

impl Schedule {
    /// The phases of `program`'s layers.
    #[must_use]
    pub fn of(program: &LayerProgram) -> Schedule {
        let needs = &Needs::of(program);
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
            hg: u32::from(Layout::of(needs) == Layout::Runs),
            qb: u32::from(needs.attn_bias),
            snk: u32::from(needs.sinks),
            qkn: u32::from(needs.qk_norm),
            ob: u32::from(needs.o_bias),
            rb: u32::from(needs.router_bias),
            xb: u32::from(needs.expert_bias),
            glu: u32::from(program.bodies.iter().any(|b| b.glu == Some(GluAct::Swiglu))),
            kev: u32::from(Layout::of(needs) == Layout::Runs),
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

/// What the role kernels read off the PEs of the blocks the composer places: the copies onto
/// each kind (the kernels' ABI with the host, by layout: [`copies::of`]). A kind the kernels have
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

/// The unit grid of `plan` laid as `layout`, its control column and row through the middle.
fn grid(plan: &Plan, layout: UnitLayout) -> UnitGrid {
    let p = &plan.partition;
    UnitGrid {
        cols: p.cols,
        rows: p.rows,
        cx: p.cols / 2,
        cy: p.rows / 2,
        layout,
        layers_per_unit: p.layers_per_unit,
    }
}

/// A unit snake: the units in the plan's slots (UNX x UNY), every segment within a half of a
/// unit's columns, the head's columns half the rows.
fn snake(plan: &Plan) -> R<UnitGrid> {
    let p = &plan.partition;
    let ug = p
        .unit_grid
        .ok_or("dense: the plan has no unit grid (UNX, UNY)")?;
    // a segment within a half of the unit's columns (none spans the control column)
    for (n, sw) in [
        ("SWA", p.segments.scores),
        ("SWG", p.segments.gdn),
        ("SWV", p.segments.onorm),
    ] {
        if sw == 0 || !(p.cols / 2).is_multiple_of(sw) {
            return Err(format!(
                "dense: {n} {sw} does not divide XC {} (a segment would span the control column)",
                p.cols / 2
            ));
        }
    }
    if p.head.cols * 2 != p.rows {
        return Err("dense: head columns = CX rows / 2".into());
    }
    Ok(grid(
        plan,
        UnitLayout::Snake {
            unx: ug.0,
            uny: ug.1,
            up_first: false,
            gaps: snake_gap(p.units, ug)?.into_iter().collect(),
        },
    ))
}

/// Two unit columns, the first visited down, the second up: unit 0 and the last side by side on
/// row 0 (the token loop's return).
fn u_columns(plan: &Plan) -> R<UnitGrid> {
    let p = &plan.partition;
    let n = p.units;
    if n == 0 || !n.is_multiple_of(2) {
        return Err(format!(
            "gemma12: {n} units (two unit columns, unit 0 and the last on row 0)"
        ));
    }
    let (uw, uh, uny) = (p.cols + 1, p.rows + 1, n / 2);
    let origins = (0..n)
        .map(|u| {
            if u < uny {
                (0, uh * u)
            } else {
                (uw, uh * (n - 1 - u))
            }
        })
        .collect();
    Ok(grid(plan, UnitLayout::Explicit { origins }))
}

/// Attention | MoE halves (CX after the attention columns) in a snake of two unit columns that
/// starts upward, so the first and the last unit sit on the band.
fn split_snake(plan: &Plan) -> R<UnitGrid> {
    let p = &plan.partition;
    let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")?;
    if ax >= p.cols {
        return Err("gpt-oss plan: no MoE columns".into());
    }
    if !p.units.is_multiple_of(2) {
        return Err("gpt-oss: an even unit count (the last unit next to the band)".into());
    }
    Ok(UnitGrid {
        cx: ax,
        ..grid(
            plan,
            UnitLayout::Snake {
                unx: 2,
                uny: p.units / 2,
                up_first: true,
                gaps: Vec::new(),
            },
        )
    })
}

/// The blocks of `plan` for layers that need `needs`, in order (the units first, then what is
/// placed against them), each block's copies the layout's kernels' ([`copies::of`]).
pub fn blocks(plan: &Plan, needs: &Needs) -> R<Vec<Box<dyn Block>>> {
    let p = &plan.partition;
    let token_loop = plan.contract.token_loop;
    let layout = Layout::of(needs);
    let copies = copies::of(plan, layout)?;
    let mut out: Vec<Box<dyn Block>> = Vec::new();
    if layout == Layout::Split {
        out.push(Box::new(MoeUnits {
            grid: split_snake(plan)?,
            copies: take(copies.moe, "MoE units")?,
        }));
        out.push(Box::new(AttnScores {
            width: p.segments.scores,
        }));
        out.push(Box::new(Router {
            width: p
                .segments
                .router
                .ok_or("gpt-oss plan: no router segments")?,
        }));
        out.push(Box::new(Band {
            cols: p.head.cols,
            rows: p.head.rows,
            vocab_rows: p.head.vocab_rows,
        }));
        if token_loop {
            out.push(Box::new(BandLoop));
        }
        return Ok(out);
    }
    if layout == Layout::Stacked {
        let arr = crate::plan::Arrangement::of(plan)?;
        out.push(Box::new(Units {
            grid: grid(
                plan,
                UnitLayout::Explicit {
                    origins: arr.origins(p.cols + 1, p.rows + 1)?,
                },
            ),
            copies: take(copies.units, "units")?,
        }));
        out.push(Box::new(Scores {
            width: p.segments.scores,
        }));
        if needs.side_in {
            out.push(Box::new(Ple {
                columns: arr
                    .columns(p.cols + 1)
                    .into_iter()
                    .map(|(_, units, st)| PleColumn {
                        units,
                        rows: st.rows,
                        tokens: st.tokens,
                        bus: st.bus,
                    })
                    .collect(),
            }));
        }
        if needs.shared_cache {
            out.push(Box::new(SharedKv));
        }
    } else if layout == Layout::Runs {
        out.push(Box::new(RowRuns {
            grid: u_columns(plan)?,
            copies: take(copies.runs, "row runs")?,
        }));
        out.push(Box::new(HeadGroups {
            cols: p.head.cols,
            rows: p.head.rows,
            copies: take(copies.head, "head")?,
        }));
        return Ok(out);
    } else {
        out.push(Box::new(Units {
            grid: snake(plan)?,
            copies: take(copies.units, "units")?,
        }));
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
    }
    out.push(Box::new(HeadEast {
        cols: p.head.cols,
        rows: p.head.rows,
        token_loop,
        copies: take(copies.head, "head")?,
    }));
    if token_loop {
        out.push(Box::new(TokenLoop));
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

    /// The role kernels' phases read off each shipping model's layer program: what its family's
    /// kernels were built with.
    #[test]
    fn a_layer_program_schedules_its_family_kernels() {
        for (name, nrm, mixk, act, ple, kvs, hg) in [
            ("qwen35-d9b-bf16-kv-bf16", 0, 0, 0, 0, 0, 0),
            ("llama31-8b-bf16-kv-bf16", 0, 0, 0, 0, 0, 0),
            ("gemma4-e4b-bf16-kv-bf16", 1, 1, 1, 1, 1, 0),
            ("gemma4-12b-bf16-kv-bf16", 1, 1, 1, 0, 0, 1),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", 2, 2, 0, 0, 0, 0),
        ] {
            let t = trace(name);
            let sc = Schedule::of(&program(&t));
            assert_eq!(
                (sc.nrm, sc.mixk, sc.act, sc.ple, sc.kvs, sc.hg),
                (nrm, mixk, act, ple, kvs, hg),
                "{name}"
            );
        }
        // the MoE halves' variants: gpt-oss's biases and sinks, the mini's head norms and SiLU
        for (name, want) in [
            ("gptoss-20b-bf16-mxfp4-kv-bf16", (1, 1, 0, 1, 1, 1, 0)),
            ("qwen3-30b-a3b-mini-bf16-kv-bf16", (0, 0, 1, 0, 0, 0, 1)),
        ] {
            let sc = Schedule::of(&program(&trace(name)));
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
            ("tiny_gemma", "tiny"),
            ("tiny_g12", "tiny"),
            ("tiny_gptoss", "tiny"),
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
}
