//! What each op takes and yields when ranks split its values. An op without a
//! rule here gets every value whole and stays as it is; it may not read a
//! weight or a cache the ranks split.

use poem_ir::{Operands, Operation, ValueId};

use super::{Dist, Pass};
use crate::tree::{self, Tree};

/// How an op meets the ranks.
enum Rule {
    /// Every value whole.
    Whole,
    /// The `follow` inputs are split by head or channel, every output with
    /// them, and the `scale` counts are each rank's share. Other activations
    /// are whole; a split cache is required when `cache` says so.
    Heads {
        follow: &'static [&'static str],
        scale: &'static [&'static str],
        cache: bool,
    },
    /// `act · w`: a weight split along its `out` axis splits the output, one
    /// split along its `in` axis leaves each rank a partial sum.
    Contract {
        act: &'static str,
        w: &'static str,
        out: u32,
        input: u32,
    },
    /// Linear in `follow` jointly, so partial sums stay one.
    Linear { follow: &'static [&'static str] },
    /// A row lookup in a table the ranks split by row: each rank holds the
    /// rows it has, zeros elsewhere.
    Lookup { table: &'static str },
    /// An attention plan: its head counts are each rank's share.
    Plan { scale: &'static [&'static str] },
    /// A row norm, which needs the whole row unless its weight is split
    /// with it: then each rank norms its own share.
    Norm,
    /// Group indices for a grouped matmul, which reads no value: they are
    /// a rank's share of the groups when the row it groups is split.
    Groups,
}

const HEAD_OPS: &[(&str, &[&str], &[&str], bool)] = &[
    ("elementwise.silu", &["x"], &[], false),
    ("elementwise.gelu", &["x"], &[], false),
    ("elementwise.tanh", &["x"], &[], false),
    ("elementwise.mul", &["x", "y"], &[], false),
    ("elementwise.mul_scalar", &["x"], &[], false),
    ("elementwise.silu_scaled", &["x"], &[], false),
    ("elementwise.scale", &["x"], &[], false),
    ("elementwise.gate_sigmoid_mul", &["x", "gate"], &[], false),
    (
        "elementwise.gate_sigmoid_mul_heads",
        &["x", "gate"],
        &[],
        false,
    ),
    ("elementwise.rmsnorm_per_head", &["x"], &[], false),
    ("elementwise.rmsnorm_per_head_plus_one", &["x"], &[], false),
    ("elementwise.rmsnorm_no_scale", &["x"], &[], false),
    ("elementwise.rmsnorm_gated", &["x", "gate"], &[], false),
    (
        "elementwise.rmsnorm_gated_by",
        &["x", "gate"],
        &["heads"],
        false,
    ),
    ("elementwise.rope_full", &["q", "k"], &[], false),
    ("elementwise.rope_partial", &["q", "k"], &[], false),
    ("elementwise.rope_partial_q", &["q"], &[], false),
    ("elementwise.rope_partial_last", &["q"], &[], false),
    ("elementwise.rope_yarn", &["q", "k"], &[], false),
    ("elementwise.rope_mrope", &["q", "k"], &[], false),
    ("elementwise.rope_axes", &["x"], &[], false),
    ("elementwise.add_bias", &["out"], &[], false),
    (
        "layout.split_qkv",
        &["packed"],
        &["q_width", "kv_width"],
        false,
    ),
    ("layout.split_q_gate", &["packed"], &[], false),
    ("layout.split_rows", &["x"], &["width"], false),
    ("layout.gather_rows", &["x"], &[], false),
    ("layout.pack_rows", &["x"], &[], false),
    ("layout.unpack_rows", &["x"], &[], false),
    ("linear.mlp_swiglu", &["packed"], &["intermediate"], false),
    (
        "linear.mlp_swiglu_clamp",
        &["packed"],
        &["intermediate"],
        false,
    ),
    (
        "linear.mlp_swiglu_clamp_alpha",
        &["packed"],
        &["intermediate"],
        false,
    ),
    (
        "linear.mlp_geglu_tanh_packed",
        &["packed"],
        &["intermediate"],
        false,
    ),
    ("linear.mlp_situ", &["packed"], &["intermediate"], false),
    ("linear.mlp_swiglu_clamp_split", &["gate", "up"], &[], false),
    ("linear.mlp_geglu_tanh", &["gate", "up"], &[], false),
    ("linear.mlp_gelu_tanh", &["x"], &[], false),
    (
        "linear.matmul_grouped",
        &["x", "routes"],
        &["groups"],
        false,
    ),
    ("attention.decode", &["q"], &[], true),
    ("attention.prefill", &["q"], &["kv_heads"], true),
    ("attention.decode_lse", &["q"], &[], true),
    ("attention.prefill_lse", &["q"], &["kv_heads"], true),
    ("attention.masked", &["q"], &["kv_heads"], true),
    ("attention.masked_lse", &["q"], &["kv_heads"], true),
    ("attention.decode_selected", &["q"], &[], true),
    ("attention.prefill_selected", &["q"], &["kv_heads"], true),
    ("attention.ragged", &["q", "k", "v"], &["kv_heads"], false),
    ("attention.sink", &["o", "lse"], &[], false),
    (
        "attention.merge_lse",
        &["o1", "lse1", "o2", "lse2"],
        &["heads"],
        false,
    ),
    ("attention.pool_lse", &["q"], &["heads"], false),
    ("attention.pool_lse_selected", &["q"], &["heads"], false),
    ("attention.kv_append", &["k", "v"], &[], true),
    ("attention.kv_append_shared", &["plane"], &[], true),
    ("attention.mla_split_q_b", &["q_b"], &["heads"], false),
    ("attention.mla_absorb_q", &["q_nope"], &["heads"], false),
    ("attention.mla_absorb_out", &["latent"], &["heads"], false),
    ("attention.mla_decode", &["q", "q_pe"], &["heads"], false),
    ("attention.mla_prefill", &["q", "q_pe"], &["heads"], false),
    (
        "attention.mla_decode_selected",
        &["q", "q_pe"],
        &["heads"],
        false,
    ),
    (
        "attention.mla_prefill_selected",
        &["q", "q_pe"],
        &["heads"],
        false,
    ),
    ("attention.ssm_causal_conv1d", &["x"], &[], true),
    ("attention.ssm_causal_conv1d_chunked", &["x"], &[], true),
    ("attention.ssm_gdn_prep", &["ba"], &[], false),
    (
        "attention.ssm_gated_delta",
        &["qkv", "z", "gates"],
        &["k_heads", "v_heads"],
        true,
    ),
    (
        "attention.ssm_gated_delta_chunked",
        &["qkv", "z", "gates"],
        &["k_heads", "v_heads"],
        true,
    ),
    (
        "attention.ssm_kda_step",
        &["mixed", "f", "b"],
        &["heads"],
        true,
    ),
    (
        "attention.ssm_kda_chunked",
        &["mixed", "f", "b"],
        &["heads"],
        true,
    ),
];

/// Attention that reads kv a model may keep whole on every rank while it
/// splits the query heads (one kv head shared by all of them): its kv head
/// count is a rank's share only when the cache is split.
const KV_READERS: &[&str] = &[
    "attention.decode",
    "attention.prefill",
    "attention.decode_lse",
    "attention.prefill_lse",
    "attention.masked",
    "attention.masked_lse",
    "attention.decode_selected",
    "attention.prefill_selected",
];

fn rule(name: &str) -> Rule {
    match name {
        "linear.matmul" | "linear.lm_head" => Rule::Contract {
            act: "act",
            w: "w",
            out: 0,
            input: 1,
        },
        "linear.moe_matmul_select"
        | "linear.moe_matmul_select_bias"
        | "linear.moe_matmul_select_quant" => Rule::Contract {
            act: "x",
            w: "bank",
            out: 1,
            input: 2,
        },
        "linear.moe_weighted_sum" => Rule::Linear {
            follow: &["routed"],
        },
        "elementwise.residual_add" | "elementwise.add" => Rule::Linear {
            follow: &["x", "y"],
        },
        "linear.moe_sigmoid_gate_add" => Rule::Linear {
            follow: &["routed", "shared"],
        },
        "layout.embed" => Rule::Lookup { table: "table" },
        "attention.plan_prefill" | "attention.plan_decode" => Rule::Plan {
            scale: &["q_heads", "kv_heads"],
        },
        "attention.mla_plan" => Rule::Plan { scale: &["heads"] },
        "elementwise.rmsnorm" | "elementwise.rmsnorm_plus_one" => Rule::Norm,
        "linear.group_routes" => Rule::Groups,
        _ => HEAD_OPS.iter().find(|(n, ..)| *n == name).map_or(
            Rule::Whole,
            |(_, follow, scale, cache)| Rule::Heads {
                follow,
                scale,
                cache: *cache,
            },
        ),
    }
}

/// `op` as one rank runs it, and how the ranks hold each of its outputs.
pub(super) fn apply(pass: &mut Pass, op: &Operation) -> Result<(Operation, Vec<Dist>), String> {
    let mut at = At {
        tree: tree::of(op),
        name: op.name(),
    };
    let mut outs = Vec::new();
    op.outputs(&mut outs);
    let dists = match rule(at.name) {
        Rule::Whole => {
            at.whole_except(pass, &[])?;
            at.check_state(pass, false, false)?;
            vec![Dist::Whole; outs.len()]
        }
        Rule::Plan { scale } => {
            at.whole_except(pass, &[])?;
            at.check_state(pass, false, false)?;
            let split_kv = outs.iter().any(|plan| pass.planned_kv_split(*plan));
            let scale: Vec<&str> = scale
                .iter()
                .copied()
                .filter(|f| *f != "kv_heads" || split_kv)
                .collect();
            at.scale(pass, &scale)?;
            vec![Dist::Whole; outs.len()]
        }
        Rule::Lookup { table } => {
            at.whole_except(pass, &[table])?;
            let cut = match at.one(table) {
                Some(t) => pass.weight_cut(t)?,
                None => None,
            };
            match cut {
                Some((0, _)) => vec![Dist::Partial; outs.len()],
                Some((axis, _)) => {
                    return Err(format!(
                        "`{}` reads a table split along axis {axis}",
                        at.name
                    ));
                }
                None => vec![Dist::Whole; outs.len()],
            }
        }
        Rule::Norm => {
            at.whole_except(pass, &["x"])?;
            let x = at.one("x").expect("a norm reads a row");
            let cut = match at.one("weight") {
                Some(w) => pass.weight_cut(w)?,
                None => None,
            };
            match (pass.dist(x)?, cut) {
                (Dist::Split(blocks), Some((0, _))) => vec![Dist::Split(blocks); outs.len()],
                (_, None) => {
                    at.make_whole(pass, x)?;
                    vec![Dist::Whole; outs.len()]
                }
                (held, Some(_)) => {
                    return Err(format!(
                        "`{}` norms a row held {held:?} with a weight the ranks split",
                        at.name
                    ));
                }
            }
        }
        Rule::Groups => {
            let grouped = outs.iter().any(|routes| pass.grouped_row_split(*routes));
            if grouped {
                at.scale(pass, &["groups"])?;
                outs.iter()
                    .map(|o| Dist::even(&[pass.width(*o).unwrap_or(0)], pass.world))
                    .collect()
            } else {
                vec![Dist::Whole; outs.len()]
            }
        }
        Rule::Linear { follow } => {
            at.whole_except(pass, follow)?;
            let followed = at.all_of(follow);
            let held: Vec<Dist> = followed
                .iter()
                .map(|v| pass.dist(*v))
                .collect::<Result<_, _>>()?;
            if held.iter().all(|d| *d == held[0]) {
                vec![held[0].clone(); outs.len()]
            } else {
                for v in followed {
                    at.make_whole(pass, v)?;
                }
                vec![Dist::Whole; outs.len()]
            }
        }
        Rule::Contract { act, w, out, input } => {
            at.whole_except(pass, &[act, w])?;
            let a = at.one(act).expect("a contraction reads an activation");
            let weight = at.one(w).expect("a contraction reads a weight");
            match pass.weight_cut(weight)? {
                None => {
                    at.make_whole(pass, a)?;
                    vec![Dist::Whole; outs.len()]
                }
                Some((axis, blocks)) if axis == out => {
                    at.make_whole(pass, a)?;
                    vec![Dist::Split(blocks); outs.len()]
                }
                Some((axis, blocks)) if axis == input => match pass.dist(a)? {
                    held @ Dist::Split(_)
                        if !held.is_even(pass.world)
                            || blocks.iter().any(|b| b.parts != u64::from(pass.world)) =>
                    {
                        return Err(format!(
                            "`{}` contracts heads a group of ranks holds alike, whose \
                             partial sums would count each head once per rank",
                            at.name
                        ));
                    }
                    Dist::Split(_) => vec![Dist::Partial; outs.len()],
                    held => {
                        return Err(format!(
                            "`{}` contracts a weight split along its input with an \
                             activation held {held:?}, made by `{}`",
                            at.name,
                            pass.maker(a)
                        ));
                    }
                },
                Some((axis, _)) => {
                    return Err(format!(
                        "`{}` reads a weight split along axis {axis}",
                        at.name
                    ));
                }
            }
        }
        Rule::Heads {
            follow,
            scale,
            cache,
        } => {
            at.whole_except(pass, follow)?;
            let followed: Vec<ValueId> = follow.iter().flat_map(|f| at.all(f)).collect();
            for v in &followed {
                if pass.dist(*v)? == Dist::Partial {
                    at.make_whole(pass, *v)?;
                }
            }
            let held: Vec<Dist> = at
                .all_of(follow)
                .iter()
                .map(|v| pass.dist(*v))
                .collect::<Result<_, _>>()?;
            let split = held.iter().any(|d| matches!(d, Dist::Split(_)));
            if split && held.contains(&Dist::Whole) {
                return Err(format!("`{}` mixes split and whole values", at.name));
            }
            at.check_state(pass, split, cache)?;
            if split {
                let whole_kv = KV_READERS.contains(&at.name) && !at.reads_split_cache(pass);
                let scale: Vec<&str> = scale
                    .iter()
                    .copied()
                    .filter(|f| *f != "kv_heads" || !whole_kv)
                    .collect();
                at.check_kv_parts(pass)?;
                at.scale(pass, &scale)?;
                let followed = at.all_of(follow);
                outs.iter()
                    .enumerate()
                    .map(|(i, o)| held_like(pass, *o, i, &followed, &held, outs.len()))
                    .collect::<Result<_, String>>()?
            } else {
                vec![Dist::Whole; outs.len()]
            }
        }
    };
    Ok((tree::op(at.tree), dists))
}

/// An op being rewritten for one rank.
struct At {
    tree: Tree,
    name: &'static str,
}

impl At {
    fn find(&self, name: &str) -> Option<&Tree> {
        fn walk<'t>(tree: &'t Tree, name: &str) -> Option<&'t Tree> {
            match tree {
                Tree::Variant(_, inner) => walk(inner, name),
                Tree::Struct(fields) => fields.iter().find(|(n, _)| *n == name).map(|(_, t)| t),
                _ => None,
            }
        }
        walk(&self.tree, name)
    }

    fn all(&self, name: &str) -> Vec<ValueId> {
        self.find(name).map(tree::ids).unwrap_or_default()
    }

    fn all_of(&self, names: &[&str]) -> Vec<ValueId> {
        names.iter().flat_map(|n| self.all(n)).collect()
    }

    fn one(&self, name: &str) -> Option<ValueId> {
        self.all(name).first().copied()
    }

    /// Every activation outside `keep` made whole.
    fn whole_except(&mut self, pass: &mut Pass, keep: &[&str]) -> Result<(), String> {
        let kept = self.all_of(keep);
        let mut ins = Vec::new();
        tree::op(self.tree.clone()).inputs(&mut ins);
        for v in ins {
            if kept.contains(&v) {
                continue;
            }
            if !pass.is_weight(v) && pass.cache_split(v).is_none() {
                self.make_whole(pass, v)?;
            }
        }
        Ok(())
    }

    fn make_whole(&mut self, pass: &mut Pass, v: ValueId) -> Result<(), String> {
        let whole = pass.whole(v)?;
        if whole != v {
            fn swap(tree: &mut Tree, from: ValueId, to: ValueId) {
                match tree {
                    Tree::Id(id) if *id == from.0 => *id = to.0,
                    Tree::Some(inner) | Tree::Variant(_, inner) => swap(inner, from, to),
                    Tree::Seq(items) => items.iter_mut().for_each(|t| swap(t, from, to)),
                    Tree::Struct(fields) => fields.iter_mut().for_each(|(_, t)| swap(t, from, to)),
                    _ => {}
                }
            }
            swap(&mut self.tree, v, whole);
        }
        Ok(())
    }

    /// An op on whole values reads no weight the ranks split, and a cache it
    /// reads is split exactly when its values are and it `wants` one, but
    /// for attention whose split query heads share a kv head every rank keeps
    /// whole.
    fn check_state(&self, pass: &Pass, split: bool, wants: bool) -> Result<(), String> {
        let mut ins = Vec::new();
        tree::op(self.tree.clone()).inputs(&mut ins);
        for v in ins {
            if !split && pass.weight_cut(v)?.is_some() {
                return Err(format!("`{}` reads a weight the ranks split", self.name));
            }
            let reader = KV_READERS.contains(&self.name);
            if let Some(cut) = pass.cache_split(v)
                && cut != (split && wants)
                && !(split && !cut && reader)
            {
                return Err(format!(
                    "`{}` reads a cache {} while its values are {}",
                    self.name,
                    if cut {
                        "the ranks split"
                    } else {
                        "every rank holds whole"
                    },
                    if split { "split" } else { "whole" },
                ));
            }
        }
        Ok(())
    }

    fn reads_split_cache(&self, pass: &Pass) -> bool {
        let mut ins = Vec::new();
        tree::op(self.tree.clone()).inputs(&mut ins);
        ins.into_iter().any(|v| pass.cache_split(v) == Some(true))
    }

    fn scale(&mut self, pass: &Pass, names: &[&str]) -> Result<(), String> {
        let world = i128::from(pass.world);
        let op = self.name;
        for (field, value) in tree::fields_mut(&mut self.tree) {
            if names.contains(field) {
                let Tree::Int(n) = value else {
                    return Err(format!("`{op}`.{field} is not a count"));
                };
                // A group of ranks holds each kv head when there are fewer
                // of them than ranks.
                let parts = if *field == "kv_heads" && *n > 0 && *n < world {
                    *n
                } else {
                    world
                };
                if *n % parts != 0 || world % parts != 0 {
                    return Err(format!("`{op}`.{field} of {n} does not split {world} ways"));
                }
                *n /= parts;
            }
        }
        Ok(())
    }
}

/// How the ranks hold output `out`, the `index`-th of `count`, of an op
/// whose `followed` inputs they hold as `held`: as the input in its place
/// or the first one as wide, else as the block of a lone packed input it
/// unpacks, else cut once per rank.
fn held_like(
    pass: &Pass,
    out: ValueId,
    index: usize,
    followed: &[ValueId],
    held: &[Dist],
    count: usize,
) -> Result<Dist, String> {
    let width = pass.width(out);
    if width.is_none() {
        return Ok(Dist::Whole);
    }
    let same = |i: usize| pass.width(followed[i]) == width;
    if index < followed.len() && same(index) {
        return Ok(held[index].clone());
    }
    if let Some(i) = (0..followed.len()).find(|i| same(*i)) {
        return Ok(held[i].clone());
    }
    if let [Dist::Split(blocks)] = held
        && blocks.len() == count
        && Some(blocks[index].width) == width
    {
        return Ok(Dist::Split(vec![blocks[index]]));
    }
    if held.iter().all(|d| d.is_even(pass.world)) {
        return Ok(Dist::even(&[width.unwrap_or(0)], pass.world));
    }
    Err(format!(
        "an output of `{}` is {} wide, which no input it follows is, and they \
         are held by groups of ranks alike",
        pass.maker(out),
        width.unwrap_or(0),
    ))
}

impl At {
    /// Kv an op appends to a cache, or attends to without one, is cut as
    /// the cache is and as its kv head count is.
    fn check_kv_parts(&self, pass: &Pass) -> Result<(), String> {
        let mut ins = Vec::new();
        tree::op(self.tree.clone()).inputs(&mut ins);
        let mut want = None;
        for v in ins {
            if let Some(parts) = pass.cache_parts(v)? {
                want = Some(parts);
            }
        }
        let want = match (self.name, want) {
            ("attention.kv_append" | "attention.kv_append_shared", Some(parts)) => parts,
            ("attention.ragged", _) => match self.find("kv_heads") {
                Some(Tree::Int(n)) => {
                    let n = u64::try_from(*n).map_err(|_| "a negative head count")?;
                    super::Shard::parts(Some(n), u64::from(pass.world))
                        .map_err(|why| format!("`{}`: {why}", self.name))?
                }
                _ => return Ok(()),
            },
            _ => return Ok(()),
        };
        for name in ["k", "v", "plane"] {
            for v in self.all(name) {
                if let Dist::Split(blocks) = pass.dist(v)?
                    && blocks.iter().any(|b| b.parts != want)
                {
                    return Err(format!(
                        "`{}` writes or reads kv `{name}` cut {:?} ways where its heads \
                         are cut {want} ways",
                        self.name,
                        blocks.iter().map(|b| b.parts).collect::<Vec<_>>(),
                    ));
                }
            }
        }
        Ok(())
    }
}
